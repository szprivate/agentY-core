# Runs inside Maya (commandPort session, or mayapy). Y up.
# build: exec the script, optionally save a copy and render a still. check: raw camera + native projections.

_DEFAULT_CAMS = ("persp", "top", "front", "side", "left", "back", "bottom")


def _cmds():
    import maya.cmds as cmds
    return cmds


def _objects_summary():
    cmds = _cmds()
    skip = set(_DEFAULT_CAMS)
    transforms = [t for t in (cmds.ls(type="transform") or []) if t not in skip]
    top = sorted(t for t in (cmds.ls(assemblies=True) or []) if t not in skip)
    return {"count": len(transforms), "top_level": top[:60], "top_level_total": len(top), "collections": []}


def _shape_of(node):
    cmds = _cmds()
    if cmds.nodeType(node) == "camera":
        return node
    shapes = cmds.listRelatives(node, shapes=True, type="camera", fullPath=True) or []
    return shapes[0] if shapes else None


def _find_camera(name):
    cmds = _cmds()
    if name:
        if not cmds.objExists(name):
            return None
        return _shape_of(name)
    cams = cmds.ls(type="camera", long=True) or []
    user = [c for c in cams if cmds.listRelatives(c, parent=True)[0].split("|")[-1] not in _DEFAULT_CAMS]
    renderable = [c for c in user if cmds.getAttr(c + ".renderable")]
    pick = renderable or user
    if pick:
        return pick[0]
    persp = [c for c in cams if cmds.listRelatives(c, parent=True)[0].split("|")[-1] == "persp"]
    return persp[0] if persp else None


def _save_copy(path):
    cmds = _cmds()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kind = "mayaAscii" if path.lower().endswith(".ma") else "mayaBinary"
    old = cmds.file(q=True, sceneName=True)
    modified = cmds.file(q=True, modified=True)
    cmds.file(rename=path)
    try:
        cmds.file(save=True, type=kind, force=True)
    finally:
        if old:
            cmds.file(rename=old)
        cmds.file(modified=modified)


def _collect(outdir, stem, path, written):
    """Maya files a picture under <images>/<camera>/<prefix>.png (or <prefix>.png): find the newest one and
    move it to ``path``. True when ``path`` exists afterwards."""
    found = [f for f in written if isinstance(f, str) and os.path.isfile(f)]
    if not found:
        for root, _d, files in os.walk(outdir):
            found += [os.path.join(root, f) for f in files if f.startswith(stem) and f.lower().endswith(".png")
                      and os.path.abspath(os.path.join(root, f)) != os.path.abspath(path)]
    if found:
        newest = max(found, key=os.path.getmtime)
        if os.path.abspath(newest) != os.path.abspath(path):
            import shutil
            shutil.move(newest, path)
            try:
                os.rmdir(os.path.dirname(newest))
            except OSError:
                pass
    return os.path.exists(path)


def _render(job, live):
    cmds = _cmds()
    cam = _find_camera(job.get("camera"))
    path = job["render_to"]
    if cam is None:
        return {"error": "the scene has no camera to render through"}
    outdir = os.path.dirname(os.path.abspath(path))
    os.makedirs(outdir, exist_ok=True)
    w, h = parse_resolution(job.get("resolution")) or (
        int(cmds.getAttr("defaultResolution.width")), int(cmds.getAttr("defaultResolution.height")))
    final = job.get("render_engine") == "final"
    out = {"path": path, "engine": "arnold" if final else ("playblast" if live else "mayaHardware2"),
           "resolution": [w, h]}
    xform = cmds.listRelatives(cam, parent=True, fullPath=True)[0]
    if os.path.exists(path):
        os.remove(path)   # a picture left by an earlier run must not pass for this one
    try:
        if live and not final:
            panel = cmds.getPanel(withFocus=True)
            if not panel or cmds.getPanel(typeOf=panel) != "modelPanel":
                panel = (cmds.getPanel(type="modelPanel") or [None])[0]
            if panel:
                cmds.modelEditor(panel, edit=True, camera=xform)
            got = cmds.playblast(frame=[cmds.currentTime(q=True)], format="image", compression="png",
                                 completeFilename=path, viewer=False, offScreen=True, forceOverwrite=True,
                                 framePadding=0, showOrnaments=False, widthHeight=[w, h], percent=100)
            out["exists"] = os.path.exists(path)
            if not out["exists"]:
                out["error"] = "playblast wrote nothing (%s)" % (got,)
            return out
        cmds.setAttr("defaultResolution.width", w)
        cmds.setAttr("defaultResolution.height", h)
        cmds.setAttr("defaultRenderGlobals.imageFormat", 32)   # png
        stem = os.path.splitext(os.path.basename(path))[0]
        cmds.setAttr("defaultRenderGlobals.imageFilePrefix", stem, type="string")
        cmds.workspace(fileRule=["images", outdir])
        for c in cmds.ls(type="camera", long=True) or []:
            cmds.setAttr(c + ".renderable", c == cam)
        def hardware():
            cmds.setAttr("defaultRenderGlobals.currentRenderer", "mayaHardware2", type="string")
            for attr, val in (("gammaCorrectionEnable", 1), ("gammaValue", 2.2), ("lightingMode", 1), ("useMaximumHardwareLights", 1), ("multiSampleEnable", 1), ("multiSampleCount", 16), ("ssaoEnable", 1),
                              ("ssaoAmount", 1.5), ("ssaoRadius", 8), ("lineAAEnable", 1), ("shadowEnable", 1), ("hwFogEnable", 0)):
                try:
                    cmds.setAttr("hardwareRenderingGlobals." + attr, val)
                except Exception:  # noqa: BLE001
                    pass
            return cmds.render(xform, x=w, y=h) or []

        written = []
        if final:
            cmds.loadPlugin("mtoa", quiet=True)
            cmds.setAttr("defaultRenderGlobals.currentRenderer", "arnold", type="string")
            try:
                import mtoa.core as core
                core.createOptions()
            except Exception:  # noqa: BLE001
                pass
            cmds.setAttr("defaultArnoldRenderOptions.AASamples", 3)
            # arnoldRender, not cmds.render: in a standalone session cmds.render does not run Arnold at all
            # (it writes a flat viewport-style picture that ignores the lights) yet reports success.
            try:
                cmds.arnoldRender(cam=xform, width=w, height=h, batch=True)
            except Exception as e:  # noqa: BLE001
                out["arnold_error"] = "%s: %s" % (type(e).__name__, e)
            if not _collect(outdir, stem, path, []):
                # No Arnold picture: in batch it aborts without a licence (no RLM / Autodesk entitlement).
                out["engine"] = "mayaHardware2"
                out["fallback"] = ("Arnold wrote no picture (usually no licence: set ARNOLD_LICENSE_ORDER / "
                                   "solidangle_LICENSE or sign in); rendered with Maya Hardware 2.0 instead")
                written = hardware()
        else:
            written = hardware()
        if isinstance(written, str):
            written = [written]
        _collect(outdir, stem, path, written)
        out["exists"] = os.path.exists(path)
        if not out["exists"]:
            out["error"] = "the renderer wrote no picture"
    except Exception as e:  # noqa: BLE001
        out["error"] = "%s render failed: %s: %s" % (out["engine"], type(e).__name__, e)
    return out


def build(job):
    cmds = _cmds()
    live = job["live"]
    if not live:
        if job.get("scene_in") and os.path.exists(job["scene_in"]):
            cmds.file(job["scene_in"], open=True, force=True)
        else:
            cmds.file(new=True, force=True)
    ok, out, err, tb = exec_build_script(job["script_path"], job.get("stage", ""))
    res = {"ok": ok, "stdout": out, "error": err, "traceback": tb}
    res["objects"] = _objects_summary()
    if ok and job.get("save_to"):
        _save_copy(job["save_to"])
        res["saved"] = job["save_to"]
    if ok and job.get("render_to"):
        res["render"] = _render(job, live)
    return res


def check(job):
    cmds = _cmds()
    import maya.api.OpenMaya as om
    if not job["live"]:
        cmds.file(job["scene_in"], open=True, force=True)
    shape = _find_camera(job.get("camera"))
    if shape is None:
        have = [c for c in (cmds.ls(type="camera") or [])]
        return {"ok": False, "error": ("camera %r not found" % job.get("camera") if job.get("camera")
                                       else "the scene has no camera") + "; cameras in the scene: %s" % (have or "none")}
    xform = cmds.listRelatives(shape, parent=True, fullPath=True)[0]
    sel = om.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    fn = om.MFnCamera(dag)
    inv = dag.inclusiveMatrixInverse()
    width = int(cmds.getAttr("defaultResolution.width"))
    height = int(cmds.getAttr("defaultResolution.height"))
    # deviceAspectRatio is NOT kept in step with width/height when a script sets them (it stays at the old value
    # while lockDeviceAspectRatio is on), and the renderer uses width x height x pixelAspect: so compute it.
    aspect = float(width) / float(height) * float(cmds.getAttr("defaultResolution.pixelAspect"))
    left, right, bottom, top = fn.getRenderingFrustum(aspect)[:4]
    near = fn.nearClippingPlane
    native = []
    for p in job["points"]:
        c = om.MPoint(p[0], p[1], p[2]) * inv
        depth = -c.z
        if abs(depth) < 1e-12:
            native.append([0.5, 0.5, False])
            continue
        x = (c.x / depth * near - left) / (right - left)
        yb = (c.y / depth * near - bottom) / (top - bottom)
        native.append([x, 1.0 - yb, depth > 0.0])
    m = cmds.xform(xform, q=True, ws=True, matrix=True)
    ax = lambda r: [m[r * 4], m[r * 4 + 1], m[r * 4 + 2]]  # noqa: E731
    def unit(v):
        n = sum(a * a for a in v) ** 0.5 or 1.0
        return [a / n for a in v]
    return {"ok": True, "native": native, "camera": {
        "name": xform.split("|")[-1], "kind": "ORTHO" if cmds.getAttr(shape + ".orthographic") else "PERSP",
        "position": [m[12], m[13], m[14]], "right": unit(ax(0)), "up": unit(ax(1)), "back": unit(ax(2)),
        "focal_mm": cmds.getAttr(shape + ".focalLength"),
        "film_w_in": cmds.getAttr(shape + ".horizontalFilmAperture"),
        "film_h_in": cmds.getAttr(shape + ".verticalFilmAperture"),
        "film_offset_x_in": cmds.getAttr(shape + ".horizontalFilmOffset"),
        "film_offset_y_in": cmds.getAttr(shape + ".verticalFilmOffset"),
        "film_fit": cmds.getAttr(shape + ".filmFit"), "overscan": cmds.getAttr(shape + ".overscan"),
        "device_aspect": aspect, "res_x": width, "res_y": height}}


def run(job):
    return dispatch(job, build, check)
