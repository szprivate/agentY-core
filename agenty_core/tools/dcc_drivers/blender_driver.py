# Runs inside Blender (live socket session, or `blender -b --python`). Z up.
# build: exec the script, optionally save a copy and render a still. check: raw camera + native projections.


def _scene():
    import bpy
    return bpy.context.scene


def _objects_summary():
    import bpy
    objs = list(bpy.data.objects)
    top = sorted(o.name for o in objs if o.parent is None)
    return {"count": len(objs), "top_level": top[:60], "top_level_total": len(top),
            "collections": sorted(c.name for c in bpy.data.collections)[:40]}


def _open_or_new(path):
    import bpy
    if path and os.path.exists(path):
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)


def _engine_ids(scene):
    return [e.identifier for e in scene.render.bl_rna.properties["engine"].enum_items]


def _enable_gpu(scene):
    import bpy
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
            except Exception:  # noqa: BLE001
                continue
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                scene.cycles.device = "GPU"
                return
    except Exception:  # noqa: BLE001
        pass


def _render(job, scene, live):
    import bpy
    r = scene.render
    path = job["render_to"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if scene.camera is None:
        return {"error": "the scene has no active camera to render through"}
    saved = (r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage, r.filepath,
             r.image_settings.file_format)
    res = parse_resolution(job.get("resolution"))
    if res:
        r.resolution_x, r.resolution_y = res
        r.resolution_percentage = 100
    r.image_settings.file_format = "PNG"
    r.filepath = path
    ids = _engine_ids(scene)
    final = job.get("render_engine") == "final"
    # fast = Workbench first: a blockout has no lights, and Workbench shows grey shapes without any
    tries = ["CYCLES"] if final else ["BLENDER_WORKBENCH"] + [e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE") if e in ids]
    out = {"path": path, "engine": None}
    err = None
    try:
        for eng in tries:
            r.engine = eng
            if eng == "CYCLES":
                scene.cycles.samples = 64
                scene.cycles.use_denoising = True
                if not live:
                    _enable_gpu(scene)
            try:
                bpy.ops.render.render(write_still=True)
                out["engine"] = eng
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = "%s render failed: %s" % (eng, e)
        if err:
            out["error"] = err
        out["resolution"] = [r.resolution_x, r.resolution_y]
        out["exists"] = os.path.exists(path)
    finally:
        r.engine, r.resolution_x, r.resolution_y, r.resolution_percentage, r.filepath = saved[:5]
        r.image_settings.file_format = saved[5]
    return out


def build(job):
    import bpy
    live = job["live"]
    if not live:
        _open_or_new(job.get("scene_in"))
    ok, out, err, tb = exec_build_script(job["script_path"])
    res = {"ok": ok, "stdout": out, "error": err, "traceback": tb}
    try:
        bpy.context.view_layer.update()
    except Exception:  # noqa: BLE001
        pass
    res["objects"] = _objects_summary()
    if ok and job.get("save_to"):
        p = job["save_to"]
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        # copy=True: the open session keeps pointing at the user's own file
        bpy.ops.wm.save_as_mainfile(filepath=p, copy=True, check_existing=False)
        res["saved"] = p
    if ok and job.get("render_to"):
        res["render"] = _render(job, _scene(), live)
    return res


def check(job):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector
    if not job["live"]:
        bpy.ops.wm.open_mainfile(filepath=job["scene_in"])
    scene = _scene()
    bpy.context.view_layer.update()
    name = job.get("camera")
    cam = bpy.data.objects.get(name) if name else scene.camera
    if cam is None or cam.type != "CAMERA":
        have = sorted(o.name for o in bpy.data.objects if o.type == "CAMERA")
        return {"ok": False, "error": ("camera %r not found" % name if name else "the scene has no active camera")
                + "; cameras in the scene: %s" % (have or "none")}
    cd, r = cam.data, scene.render
    m = cam.matrix_world
    rot = m.to_3x3().normalized()
    native = []
    for p in job["points"]:
        co = world_to_camera_view(scene, cam, Vector(p))
        native.append([co.x, 1.0 - co.y, co.z > 0.0])
    return {"ok": True, "native": native, "camera": {
        "name": cam.name, "kind": cd.type,
        "position": list(m.translation), "right": list(rot.col[0]), "up": list(rot.col[1]), "back": list(rot.col[2]),
        "focal_mm": cd.lens, "sensor_width_mm": cd.sensor_width, "sensor_height_mm": cd.sensor_height,
        "sensor_fit": cd.sensor_fit, "shift_x": cd.shift_x, "shift_y": cd.shift_y,
        "res_x": r.resolution_x, "res_y": r.resolution_y,
        "pixel_aspect_x": r.pixel_aspect_x, "pixel_aspect_y": r.pixel_aspect_y}}


def run(job):
    return dispatch(job, build, check)
