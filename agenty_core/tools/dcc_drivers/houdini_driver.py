# Runs inside Houdini (hython). Y up. Batch only: Houdini has no live bridge here (see dcc.py).
import glob
import re
# build: exec the script, save the .hip, optionally render a still. check: raw camera + projections.

_UNIT_MM = {"mm": 1.0, "m": 1000.0, "in": 25.4, "ft": 304.8, "cm": 10.0, "km": 1e6}


def _objects_summary():
    import hou
    kids = hou.node("/obj").children()
    return {"count": len(kids), "top_level": sorted(k.name() for k in kids)[:60], "top_level_total": len(kids),
            "collections": []}


def _all_cameras():
    import hou
    out = []
    stack = list(hou.node("/obj").children())
    while stack:
        n = stack.pop(0)
        if n.type().name() in ("cam", "stereocam"):
            out.append(n)
        if n.type().category().name() == "Object":
            stack.extend(n.children())
    return sorted(out, key=lambda n: n.path())


def _find_camera(name):
    import hou
    if name:
        n = hou.node(name) or hou.node("/obj/" + name.lstrip("/"))
        if n is not None:
            return n
        for c in _all_cameras():
            if c.name() == name:
                return c
        return None
    cams = _all_cameras()
    return cams[0] if cams else None


def _parm(node, name, default):
    p = node.parm(name)
    return default if p is None else p.eval()


def _render(job):
    import hou
    cam = _find_camera(job.get("camera"))
    path = job["render_to"]
    if cam is None:
        return {"error": "the scene has no camera to render through"}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w, h = parse_resolution(job.get("resolution")) or (int(cam.evalParm("resx")), int(cam.evalParm("resy")))
    final = job.get("render_engine") == "final"
    out = {"path": path, "engine": "karma" if final else "opengl", "resolution": [w, h]}
    try:
        if not final:
            rop = hou.node("/out").createNode("opengl", "dcc_opengl")
            rop.parm("camera").set(cam.path())
            rop.parm("tres").set(1) if rop.parm("tres") else None
            for a, b in (("res1", w), ("res2", h)):
                if rop.parm(a):
                    rop.parm(a).set(b)
            rop.parm("picture").set(path)
            fr = str(job.get("frames") or "").replace(" ", "")
            if fr:
                a, _, b = fr.partition("-")
                rop.parm("trange").set(1)
                rop.parm("f1").set(int(a))
                rop.parm("f2").set(int(b or a))
                rop.parm("f3").set(1)
                out["frames"] = fr
            else:
                rop.parm("trange").set(0)
            rop.render()
        else:
            _karma(cam, path, w, h)
        out["exists"] = os.path.exists(path) or bool(job.get("frames") and glob.glob(re.sub(r"\$F\d*", "*", path)))
        if not out["exists"]:
            out["error"] = "the renderer wrote no picture"
    except Exception as e:  # noqa: BLE001
        out["error"] = "%s render failed: %s: %s" % (out["engine"], type(e).__name__, e)
    return out


def _karma(cam, path, w, h):
    """A throw-away LOP chain: import /obj into USD, Karma CPU at low samples. Karma wants a light:
    when the scene has none, a dome light keeps the picture from being black."""
    import hou
    stage = hou.node("/stage")
    imp = stage.createNode("sceneimport", "dcc_import")
    imp.parm("objects").set("/obj/*")
    last = imp
    usd = imp.stage()
    prims = list(usd.Traverse())
    if not any(pr.GetTypeName().endswith("Light") for pr in prims):
        dome = stage.createNode("domelight", "dcc_dome")
        dome.setInput(0, imp)
        last = dome
    cam_prims = [pr for pr in prims if pr.GetTypeName() == "Camera"]
    cam_path = next((str(pr.GetPath()) for pr in cam_prims if pr.GetName() == cam.name()),
                    str(cam_prims[0].GetPath()) if cam_prims else cam.path())
    rop = stage.createNode("karma", "dcc_karma")
    rop.setInput(0, last)
    rop.parm("camera").set(cam_path)
    rop.parm("picture").set(path)
    rop.parm("trange").set(0)
    if rop.parm("resolutionMenu"):
        rop.parm("resolutionMenu").set("custom") if "custom" in rop.parm("resolutionMenu").menuItems() else None
    for a, b in (("resolutionx", w), ("resolutiony", h), ("samplesperpixel", 16)):
        if rop.parm(a):
            rop.parm(a).set(b)
    if rop.parm("engine"):
        rop.parm("engine").set("cpu")
    rop.parm("execute").pressButton()


def build(job):
    import hou
    if job["live"]:
        return {"ok": False, "error": "Houdini has no live mode here; use mode='batch'"}
    if job.get("scene_in") and os.path.exists(job["scene_in"]):
        hou.hipFile.load(job["scene_in"], suppress_save_prompt=True, ignore_load_warnings=True)
    else:
        hou.hipFile.clear(suppress_save_prompt=True)
    ok, out, err, tb = exec_build_script(job["script_path"], job.get("stage", ""))
    res = {"ok": ok, "stdout": out, "error": err, "traceback": tb}
    res["objects"] = _objects_summary()
    if ok and job.get("save_to"):
        os.makedirs(os.path.dirname(os.path.abspath(job["save_to"])), exist_ok=True)
        hou.hipFile.save(job["save_to"])
        res["saved"] = job["save_to"]
    if ok and job.get("render_to"):
        res["render"] = _render(job)
    return res


def check(job):
    import hou
    hou.hipFile.load(job["scene_in"], suppress_save_prompt=True, ignore_load_warnings=True)
    cam = _find_camera(job.get("camera"))
    if cam is None:
        return {"ok": False, "error": ("camera %r not found" % job.get("camera") if job.get("camera")
                                       else "the scene has no camera")
                + "; cameras in the scene: %s" % ([c.path() for c in _all_cameras()] or "none")}
    wt = cam.worldTransform()
    rows = [list(r) for r in wt.asTupleOfTuples()]

    def unit(v):
        n = sum(a * a for a in v) ** 0.5 or 1.0
        return [a / n for a in v]
    right, up, back = unit(rows[0][:3]), unit(rows[1][:3]), unit(rows[2][:3])
    pos = rows[3][:3]
    # Houdini ships no point-to-NDC call: go through the camera's inverse world matrix, then the lens.
    inv = wt.inverted()
    units = _UNIT_MM.get(str(_parm(cam, "focalunits", "mm")), 1.0)
    focal = float(_parm(cam, "focal", 50.0)) * units
    aperture = float(_parm(cam, "aperture", 41.4214)) * units
    resx, resy = int(_parm(cam, "resx", 1280)), int(_parm(cam, "resy", 720))
    aspect = float(_parm(cam, "aspect", 1.0))
    winx, winy = float(_parm(cam, "winx", 0.0)), float(_parm(cam, "winy", 0.0))
    wsx, wsy = float(_parm(cam, "winsizex", 1.0)), float(_parm(cam, "winsizey", 1.0))
    film_w = aperture
    film_h = aperture * resy / (resx * aspect)
    native = []
    for p in job["points"]:
        c = hou.Vector3(p) * inv
        depth = -c[2]
        if abs(depth) < 1e-12:
            native.append([0.5, 0.5, False])
            continue
        x = 0.5 + (focal / film_w) * (c[0] / depth) / wsx - winx / wsx
        yb = 0.5 + (focal / film_h) * (c[1] / depth) / wsy - winy / wsy
        native.append([x, 1.0 - yb, depth > 0.0])
    return {"ok": True, "native": native, "camera": {
        "name": cam.path(), "kind": "ORTHO" if int(_parm(cam, "projection", 0)) == 1 else "PERSP",
        "position": pos, "right": right, "up": up, "back": back,
        "focal_mm": focal, "aperture_mm": aperture, "res_x": resx, "res_y": resy, "pixel_aspect": aspect,
        "win_x": winx, "win_y": winy, "win_size_x": wsx, "win_size_y": wsy}}


def run(job):
    return dispatch(job, build, check)
