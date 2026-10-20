"""Render ONE new view of a World Labs Atlas world at a chosen camera (atlasGenerate), optionally upscaled.

A world made by ``atlas_create_world`` is a set of posed pictures with depth plus one mesh,
``world.glb``. This asks Atlas to paint a NEW picture of the same place from any camera, so a
background or environment can be shot again and again from different angles and still be the same
place: first frames / references for LTX 2.5 i2v, Layout-To-Render or MiniMax H3 r2v.

Cameras (the *mesh frame* is ``world.glb``'s own: the first picture's camera is the origin, looking
down -Z, Y up, metres as Atlas scales them; this is also what ``render_camera_move`` and the bEpic
viewer's 3D view use):

* relative move: ``{"forward": m, "right": m, "up": m, "yaw": deg, "pitch": deg, "roll": deg}`` from
  the first picture's camera;
* absolute: ``{"position": [x, y, z], "look_at": [x, y, z]}`` in the mesh frame;
* absolute matrix: ``{"matrix": 4x4 camera-to-world}`` (row-major, camera looks down its -Z, +Y up);
* the viewer's camera: ``{"viewer": {"camera_world": [16], "model_world": [16], "fov", "aspect"}}``,
  the raw three.js ``matrixWorld.elements`` (column-major) of the camera and of the world model's
  node. The conversion into the world's frame happens here, once (:func:`viewer_pose_to_mesh`).

Atlas ``atlasGenerate`` takes target cameras of exactly 1280 x 720; the render is that size. With
``upscale`` 2 or 4 an installed ComfyUI upscale model (``/models/upscale_models``) makes it larger.
Output: ``<world>/renders/render_NNN.png`` (+ ``_x2`` / ``_x4``) with ``.agenty.json`` sidecars, published
like any generated picture. Key ``WLT_API_KEY`` / ``WORLDLABS_API_KEY`` (never logged).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools.worldlabs_atlas import AtlasError

logger = logging.getLogger("agenty_core.atlas_render")

MIN_FOV, MAX_FOV = 10.0, 120.0
_ASPECT = wl._GRID[0] / wl._GRID[1]                     # 16:9: what Atlas renders


# -- small 3x3 / 4x4 helpers ---------------------------------------------------


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _unit(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return [x / n for x in a]


def _col(m, j):
    return [m[0][j], m[1][j], m[2][j]]


def _from_cols(c0, c1, c2):
    return [[c0[i], c1[i], c2[i]] for i in range(3)]


def _inv3(m):
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise AtlasError("the world model's transform is not invertible (a zero scale?)", "args")
    return [[(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
            [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
            [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det]]


def orthonormal(m):
    """The rotation nearest a matrix that may carry scale: Gram-Schmidt on its columns."""
    x = _unit(_col(m, 0))
    y0 = _col(m, 1)
    y = _unit([y0[i] - _dot(x, y0) * x[i] for i in range(3)])
    z = _cross(x, y)                                   # right-handed, whatever the input's handedness
    return _from_cols(x, y, z)


def _split_col_major(e):
    """three.js ``matrix.elements`` (column-major, 16) -> (3x3 rows incl. scale, translation)."""
    if not isinstance(e, (list, tuple)) or len(e) != 16:
        raise AtlasError("a matrix needs 16 numbers (three.js matrix.elements, column-major)", "args")
    v = [float(x) for x in e]
    return [[v[0], v[4], v[8]], [v[1], v[5], v[9]], [v[2], v[6], v[10]]], [v[12], v[13], v[14]]


# -- the viewer's camera -> the world's frame (the one place this is written) --


def viewer_pose_to_mesh(camera_world, model_world=None):
    """The viewer camera's pose in the mesh frame, as ``(R, t)`` (R: camera axes as columns).

    Both arguments are three.js ``matrix.elements`` (16 numbers, column-major): the camera's
    ``matrixWorld`` and the ``matrixWorld`` of the node the world's GLB was loaded under. three.js and
    glTF are both right-handed, Y up, and a three.js camera looks down its -Z with +X right and +Y up,
    which is exactly the "RUB" convention the mesh frame and the Atlas cameras use, so there is NO axis
    flip: the only thing to undo is the viewer's transform of the model (``camera_in_model = inverse(model) * camera``).
    A scaled model (a uniform "metres per unit") is divided out of the position, not the rotation.
    """
    Ac, tc = _split_col_major(camera_world)
    if model_world is None:
        return orthonormal(Ac), tc
    Am, tm = _split_col_major(model_world)
    Ai = _inv3(Am)
    return orthonormal(wl._mm(Ai, Ac)), wl._mv(Ai, [tc[i] - tm[i] for i in range(3)])


def viewer_fov(fov: float, aspect: float, gated: bool = False) -> float:
    """The vertical field of view (degrees) of the 16:9 render for a viewer camera.

    ``gated``: the user looks through a scene camera whose picture IS 16:9 (the gate): its own fov.
    Otherwise the free camera shows the whole viewport (``aspect`` = width / height) and the 16:9 render
    must contain what is on screen: the viewport's own vertical field when it is narrower than 16:9,
    and a wider one (same horizontal extent) when the viewport is wider than 16:9.
    """
    fov = float(fov)
    if not gated and aspect and aspect > 0:
        half_h = math.tan(math.radians(fov) / 2.0) * float(aspect)            # horizontal half-extent at z=1
        fov = max(fov, math.degrees(2.0 * math.atan(half_h / _ASPECT)))
    return max(MIN_FOV, min(MAX_FOV, fov))


def _matrix4(m):
    rows = [list(map(float, r)) for r in m] if m and isinstance(m[0], (list, tuple)) else None
    if rows is None:
        flat = [float(x) for x in m]
        if len(flat) != 16:
            raise AtlasError("matrix needs a 4x4 (16 numbers, row-major)", "args")
        rows = [flat[i * 4:(i + 1) * 4] for i in range(4)]
    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        raise AtlasError("matrix needs a 4x4 (16 numbers, row-major)", "args")
    return [r[:3] for r in rows[:3]], [rows[0][3], rows[1][3], rows[2][3]]


def resolve_camera(world: dict, spec: dict, fov: float = 0.0):
    """``(api_camera, (R, t) in the mesh frame, fov used)`` for a camera spec (see the module doc)."""
    from agenty_core.tools.previz_shot import camera_pose_keys, euler_to_matrix, look_at_matrix
    if not isinstance(spec, dict) or not spec:
        raise AtlasError("give a camera: a relative move (forward / right / up / yaw / pitch / roll), "
                         "position + look_at, a matrix, or the viewer's camera", "args")
    base = wl._first_posed(world)
    use_fov = float(fov or 0.0)
    if isinstance(spec.get("viewer"), dict):
        v = spec["viewer"]
        R, t = viewer_pose_to_mesh(v.get("camera_world"), v.get("model_world"))
        if not use_fov:
            if spec.get("fov"):
                use_fov = float(spec["fov"])
            elif v.get("fov"):
                use_fov = viewer_fov(v["fov"], v.get("aspect") or 0.0, bool(v.get("gated")))
    elif spec.get("matrix") is not None:
        A, t = _matrix4(spec["matrix"])
        R = orthonormal(A)
    elif spec.get("position") is not None:
        pos = [float(x) for x in spec["position"]]
        if spec.get("look_at") is None:
            raise AtlasError("an absolute camera needs look_at next to position", "args")
        tgt = [float(x) for x in spec["look_at"]]
        if _dot([a - b for a, b in zip(pos, tgt)], [a - b for a, b in zip(pos, tgt)]) < 1e-12:
            raise AtlasError("look_at is the camera's own position", "args")
        R, t = look_at_matrix(pos, tgt), pos
        if spec.get("roll"):
            from agenty_core.tools.previz_shot import _mul, _rz
            R = _mul(R, _rz(float(spec["roll"])))
    else:
        keys = camera_pose_keys([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [{**spec, "t": 0.0}], 24.0, None)
        _f, t, rot, _fv = keys[0]
        R = euler_to_matrix(rot)
    if not use_fov and spec.get("fov"):
        use_fov = float(spec["fov"])
    use_fov = max(MIN_FOV, min(MAX_FOV, use_fov)) if use_fov else 0.0
    K = wl.grid_intrinsics(base, use_fov)
    if not use_fov:
        use_fov = math.degrees(2.0 * math.atan((K["height"] / 2.0) / K["fy"]))
    return wl.mesh_pose_to_camera(base, K, R, t), (R, [float(x) for x in t]), use_fov


# -- which views to show Atlas as context --------------------------------------


def _pick_context(world: dict, cam: dict, n: int = 4) -> list:
    """The posed views with depth nearest the target camera (position, then where they look)."""
    tgt = wl.normalize_camera(cam)
    tpos, tfwd = tgt["t"], [-v for v in _col(tgt["R"], 2)]
    scored = []
    for f in world["frames"]:
        if not (f.get("depth") and f.get("camera")):
            continue
        c = wl.normalize_camera(f["camera"])
        fwd = [-v for v in _col(c["R"], 2)]
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(c["t"], tpos)))
        ang = math.acos(max(-1.0, min(1.0, _dot(fwd, tfwd))))
        scored.append((dist + 2.0 * ang, f["id"], f))
    scored.sort(key=lambda s: (s[0], s[1]))
    return [f for _s, _i, f in scored[:n]]


# -- the upscale (a small ComfyUI workflow through the core client) ------------

_PREFERRED = ("ultrasharp", "nmkd", "superscale", "realesrgan", "siax", "esrgan", "lsdir", "swinir")
_ILLUSTRATION = ("anime", "manga", "cartoon", "lineart")


def native_scale(name: str) -> int:
    m = re.search(r"(?<![0-9a-z])(\d)x(?![0-9a-z])|(?<![0-9a-z])x(\d)(?![0-9])|(\d)x[_\-.]|[_\-.](\d)x", name.lower())
    for g in (m.groups() if m else ()):
        if g and int(g) in (2, 3, 4, 8):
            return int(g)
    return 4


def pick_upscaler(names: list, scale: int, requested: str = "") -> str:
    """An installed upscale model: the one asked for (substring), else the best photo-style one,
    preferring a native factor equal to ``scale`` (a 4x model is scaled down for 2x)."""
    names = [n for n in names if isinstance(n, str)]
    if not names:
        raise AtlasError("ComfyUI has no upscale models installed (models/upscale_models); "
                         "install one (e.g. 4x-UltraSharp or RealESRGAN_x4plus) or ask for upscale 0.", "dependency")
    if requested:
        hit = [n for n in names if requested.lower() in n.lower()]
        if not hit:
            raise AtlasError(f"no upscale model matching '{requested}'; installed: {', '.join(names)}", "args")
        return hit[0]

    def rank(n: str):
        low = n.lower()
        return (any(k in low for k in _ILLUSTRATION), native_scale(n) != scale,
                min([i for i, k in enumerate(_PREFERRED) if k in low] or [len(_PREFERRED)]), low)
    return sorted(names, key=rank)[0]


def build_upscale_workflow(image_name: str, model: str, scale: int, native: int) -> dict:
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
          "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model}},
          "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}}}
    last = "3"
    if native != scale:
        wf["4"] = {"class_type": "ImageScaleBy",
                   "inputs": {"image": ["3", 0], "upscale_method": "lanczos", "scale_by": scale / float(native)}}
        last = "4"
    wf["5"] = {"class_type": "SaveImage", "inputs": {"images": [last, 0], "filename_prefix": "agenty_atlas_upscale"}}
    return wf


def upscale_image(src: Path, dest: Path, scale: int, model: str = "", timeout_s: float = 600.0) -> dict:
    """Upscale ``src`` x``scale`` with a ComfyUI upscale model into ``dest``. Raises AtlasError."""
    from agenty_core.tools.image_io import stage_image
    from agenty_core.utils.comfyui_client import get_client
    client = get_client()
    try:
        names = client.get("/models/upscale_models")
    except Exception as exc:  # noqa: BLE001
        raise AtlasError(f"could not ask ComfyUI for its upscale models: {exc}", "dependency") from exc
    chosen = pick_upscaler(names if isinstance(names, list) else [], scale, model)
    staged = stage_image(str(src), overwrite=True)
    if not isinstance(staged, dict) or staged.get("error") or not staged.get("name"):
        raise AtlasError(f"could not stage the render in ComfyUI: {(staged or {}).get('error', staged)}", "dependency")
    image_name = (staged.get("subfolder") + "/" if staged.get("subfolder") else "") + staged["name"]
    wf = build_upscale_workflow(image_name, chosen, scale, native_scale(chosen))
    try:
        sub = client.post("/prompt", json_data={"prompt": wf, "client_id": uuid.uuid4().hex})
    except Exception as exc:  # noqa: BLE001
        raise AtlasError(f"ComfyUI refused the upscale workflow: {exc}", "dependency") from exc
    pid = sub.get("prompt_id") if isinstance(sub, dict) else None
    if not pid:
        raise AtlasError(f"ComfyUI did not queue the upscale: {sub!r}"[:300], "dependency")
    deadline = time.monotonic() + float(timeout_s)
    out = None
    while time.monotonic() < deadline:
        hist = client.get(f"/history/{pid}")
        rec = hist.get(pid) if isinstance(hist, dict) else None
        if rec:
            st = (rec.get("status") or {})
            if st.get("status_str") == "error":
                raise AtlasError("the ComfyUI upscale workflow failed", "execution")
            imgs = [i for o in (rec.get("outputs") or {}).values() for i in (o.get("images") or [])]
            if imgs:
                out = imgs[0]
                break
        wl._sleep(1.5)
    if out is None:
        raise AtlasError(f"the upscale did not finish in {int(timeout_s)} s (prompt {pid})", "timeout")
    resp = client.get("/view", params={"filename": out["filename"], "subfolder": out.get("subfolder", ""),
                                       "type": out.get("type", "output")}, raw=True)
    dest.write_bytes(resp.content)
    return {"model": chosen, "native_scale": native_scale(chosen), "scale": scale, "prompt_id": pid}


# -- the render ----------------------------------------------------------------


def resolve_world_dir(world: str, output_dir: str = "") -> Path:
    """A world by name (under ComfyUI's output/3d_scenes/atlas), or by the path of its folder / world.glb / world.json."""
    ref = str(world or "").strip().strip('"')
    if not ref:
        raise AtlasError("name the world", "args")
    p = Path(ref).expanduser()
    if p.is_file():
        p = p.parent
    if p.is_dir() and (p / "world.json").exists():
        return p
    return wl._world_dir(ref, output_dir)


def _next_index(rdir: Path) -> int:
    best = -1
    for f in rdir.glob("render_*.png"):
        m = re.match(r"render_(\d+)", f.stem)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def finish_render(job: dict, op: dict) -> str:
    """Download the new view, optionally add it to the world, upscale, publish. Called by worldlabs_atlas._finish."""
    wd, op_id = Path(job["world_dir"]), op.get("id", "")
    frames = (op.get("response") or {}).get("frames") or []
    if not frames:
        raise AtlasError("the operation succeeded but returned no frame", "api", operation_id=op_id)
    fr = frames[0]
    if not fr.get("camera"):
        fr = {**fr, "camera": job["camera"]}
    world = wl._load_world(wd)
    rdir = wd / "renders"
    rdir.mkdir(parents=True, exist_ok=True)
    idx = _next_index(rdir)
    raw = rdir / f"render_{idx:03d}.png"
    added = None
    if job.get("add_to_world"):
        entry = wl._frame_entry(world, wd, max([f["id"] for f in world["frames"]] + [-1]) + 1, fr, "render", op_id)
        shutil.copyfile(wd / entry["image"], raw)
        world["frames"].append(entry)
        added = entry["id"]
        wl.build_world_mesh(wd, world, **job.get("mesh_opts", {}))
    else:
        wl._download(fr["imageAsset"], raw)
        d = fr.get("depth") or {}
        if job.get("depth") and (d.get("depthAsset") or d.get("logDepthAsset")):
            asset = d.get("depthAsset") or d["logDepthAsset"]
            wl._download(asset, rdir / (raw.stem + ("_depth.exr" if d.get("depthAsset") else "_depth.png")))
    base = wl._first_posed(world)
    mesh_cam = wl.camera_to_mesh_pose(base, fr["camera"])
    meta = {"world": world["name"], "operation": op_id, "camera_world": fr["camera"],
            "camera_mesh_frame": {"position": [round(v, 5) for v in mesh_cam[1]],
                                  "rotation_matrix": [[round(v, 6) for v in r] for r in mesh_cam[0]]},
            "fov_deg": job.get("fov"), "prompt": job.get("prompt"), "source_spec": job.get("spec"),
            "added_to_world_as_frame": added}
    wl._sidecar(raw, "Atlas render (atlasGenerate, 1280x720)", meta)
    final, up = raw, None
    upscale_error = ""
    if job.get("upscale"):
        dest = rdir / f"{raw.stem}_x{int(job['upscale'])}.png"
        try:
            up = upscale_image(raw, dest, int(job["upscale"]), job.get("upscale_model", ""))
            wl._sidecar(dest, f"Atlas render upscaled x{job['upscale']}", {**meta, "upscale": up, "from": raw.name})
            final = dest
        except AtlasError as exc:
            upscale_error = str(exc)
            logger.warning("atlas render upscale failed: %s", exc)
    world.setdefault("renders", []).append({"file": f"renders/{final.name}", "original": f"renders/{raw.name}",
                                            "operation": op_id, "camera": fr["camera"], "added_frame": added,
                                            "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
    wl._save_world(wd, world)
    wl._pending_path(wd, op_id).unlink(missing_ok=True)
    wl._publish(final)
    out = {"status": "ok", "world": world["name"], "path": str(final), "original": str(raw),
           "size": list(wl._image_size(final)), "camera_world": fr["camera"], "camera_mesh_frame": meta["camera_mesh_frame"],
           "fov_deg": job.get("fov"), "operation_id": op_id, "added_to_world": added is not None,
           "upscale": up, "usage": wl._usage_of(op) or None,
           "usage_note": "Atlas reports no per-task cost; credits (1,000 per US$) show on the platform's Usage page."}
    if added is not None:
        out["glb"] = str(wd / world["glb"])
    if upscale_error:
        out["upscale_error"] = upscale_error
    return json.dumps(out)


def render_view(world: str, camera: dict, upscale: int = 0, upscale_model: str = "", fov: float = 0.0,
                prompt: str = "", add_to_world: bool = False, depth: bool = False, mesh_detail: int = 4,
                timeout_s: int = 600, output_dir: str = "") -> str:
    """Plain-function form of :func:`atlas_render_view` (the host's viewer route calls this)."""
    return _render(world, camera, upscale, upscale_model, fov, prompt, add_to_world, depth, mesh_detail,
                   timeout_s, output_dir)


@wl._guard
def _render(world, camera, upscale, upscale_model, fov, prompt, add_to_world, depth, mesh_detail, timeout_s,
            output_dir):
    scale = int(upscale or 0)
    if scale in (0, 1):
        scale = 0
    elif scale not in (2, 4):
        raise AtlasError("upscale is 0 (none), 2 or 4", "args")
    wd = resolve_world_dir(world, output_dir)
    w = wl._load_world(wd)
    cam, (R, t), fov_used = resolve_camera(w, camera, fov)
    ctx = _pick_context(w, cam)
    if not ctx:
        raise AtlasError("the world has no view with depth to use as context", "args")
    use_prompt = prompt or w.get("prompt_used") or w.get("prompt") or None
    body = {"contextFrames": wl.context_frames(wd, ctx), "targetCameras": [cam],
            "returnDepth": bool(depth or add_to_world),
            "enhancePrompt": not bool(use_prompt and not prompt)}
    if use_prompt:
        body["prompt"] = use_prompt
    op = wl._submit(wl._TASK_GEN, body)
    spec = {k: v for k, v in camera.items() if k != "viewer"}
    if "viewer" in camera:
        spec["viewer"] = {k: v for k, v in camera["viewer"].items() if k in ("fov", "aspect", "gated")}
    job = {"kind": "render", "task": wl._TASK_GEN, "name": w["name"], "world_dir": str(wd), "camera": cam,
           "fov": round(fov_used, 3), "spec": spec, "prompt": use_prompt, "upscale": scale,
           "upscale_model": upscale_model, "add_to_world": bool(add_to_world), "depth": bool(depth),
           "mesh_opts": {"stride": max(1, int(mesh_detail))}}
    return wl._run(job, op, timeout_s)


@tool
def atlas_render_view(world: str, camera: dict, upscale: int = 0, upscale_model: str = "", fov: float = 0.0,
                      prompt: str = "", add_to_world: bool = False, depth: bool = False, mesh_detail: int = 4,
                      timeout_s: int = 600, output_dir: str = "") -> str:
    """Render ONE new picture of an Atlas world from a camera you choose (World Labs atlasGenerate,
    1280x720), optionally upscaled 2x / 4x with a local ComfyUI upscale model.

    The picture is a new view of the SAME place, so use it for consistent backgrounds and
    environments: a first frame or reference for an image-to-video model (LTX 2.5 i2v,
    Layout-To-Render, MiniMax H3 r2v) from a different angle than the original picture. One call is
    one billed Atlas task (the user's World Labs credits); do not loop it. Atlas invents what the
    picture never showed, so keep the camera a believable step (a few metres, under ~60 degrees of
    turn) from an existing view; with add_to_world the new view joins the mesh so later renders
    stay consistent with it.

    Args:
        world: name of a world made by atlas_create_world (or the path of its folder / world.glb).
        camera: where to look from, in the world's mesh frame (the first picture's camera is the
            origin looking down -Z, Y up, metres). One of:
            {"forward": m, "right": m, "up": m, "yaw": deg, "pitch": deg, "roll": deg} - relative to
            the first picture's camera (omitted = 0; yaw > 0 turns left);
            {"position": [x, y, z], "look_at": [x, y, z]} - absolute;
            {"matrix": [[...4x4...]]} - absolute camera-to-world (camera looks down -Z, +Y up);
            plus an optional "fov" (vertical degrees) in any of them.
        upscale: 0 (default, 1280x720), 2 or 4: upscale with an installed ComfyUI upscale model
            (a photographic one is picked; needs ComfyUI running).
        upscale_model: force an upscale model by (part of) its file name.
        fov: vertical field of view in degrees; 0 = the camera's "fov", else the first picture's.
        prompt: scene description; default the one Atlas used for the world (cheaper, consistent).
        add_to_world: also add the view (with depth) to the world and rebuild world.glb.
        depth: also save the view's depth map (renders/render_NNN_depth.exr); implied by add_to_world.
        mesh_detail: depth-grid stride for the rebuilt mesh (add_to_world only).
        timeout_s: how long to wait; on timeout the error carries operation_id for atlas_fetch,
            which finishes the job (download, upscale, publish) when it is done.
        output_dir: where the world lives, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON: path (the final picture, also published to the canvas and the viewer), original (the
        1280x720 one), camera_world / camera_mesh_frame, fov_deg, operation_id, upscale details - or
        {"error", "error_kind"} like the other Atlas tools (timeout -> atlas_fetch with operation_id).
    """
    return _render(world, camera, upscale, upscale_model, fov, prompt, add_to_world, depth, mesh_detail,
                   timeout_s, output_dir)


# -- open a world in the viewer's 3D view ---------------------------------------


def world_scene(wd: Path, world: dict) -> dict:
    """A previz scene (viewer format) holding the world's mesh and a 16:9 camera at the first picture's
    viewpoint, set as the active camera: the user looks through the very frame Atlas will render."""
    from agenty_core.tools.previz_shot import camera_item, model_item
    if not world.get("glb"):
        raise AtlasError("the world has no mesh (world.glb); atlas_rebuild_world_mesh makes one", "args")
    base = wl._first_posed(world)
    K = wl.grid_intrinsics(base)
    fov = math.degrees(2.0 * math.atan((K["height"] / 2.0) / K["fy"]))
    cam = camera_item([(0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], fov)], fov, wl._GRID)
    cam["id"], cam["name"] = "atlascam", "Atlas camera"
    cam["tracks"] = {}
    scene = {"version": 1, "items": [model_item(str(wd / world["glb"]), f"Atlas: {world['name']}"), cam],
             "activeCamera": "atlascam", "fps": 24.0, "length": 1, "lengthSet": True}
    return scene


@tool
def atlas_open_in_viewer(world: str, focus: bool = True, output_dir: str = "") -> str:
    """Open an Atlas world in the bEpic viewer's 3D view, looking through a 16:9 camera at the first
    picture's viewpoint, so the user can navigate (Alt + mouse) and press "Render with Atlas" in the
    Previz panel. Use when the user wants to pick a camera themselves.

    Args:
        world: name of a world made by atlas_create_world (or the path of its folder / world.glb).
        focus: bring the viewer up on that tab.
        output_dir: where the world lives, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON {"status": "ok", "tab", "scene_file"} or {"error", "error_kind"}.
    """
    return _open_in_viewer(world, focus, output_dir)


def _open_in_viewer(world, focus, output_dir):
    try:
        wd = resolve_world_dir(world, output_dir)
        w = wl._load_world(wd)
        scene = world_scene(wd, w)
        from agenty_core.utils.comfyui_client import get_client
        try:
            job = get_client().post("/bepic/previz_shot", json_data={
                "scene": scene, "name": f"atlas_{w['name']}", "focus": bool(focus), "add_node": False})
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc):
                return wl._err("the bEpic viewer is not installed in ComfyUI, or is too old for previz shots "
                               "(update ComfyUI-ImageViewer and restart ComfyUI)", "api")
            return wl._err(f"could not reach the viewer through ComfyUI: {exc}", "api")
        return json.dumps({"status": "ok", "tab": (job or {}).get("key"), "scene_file": (job or {}).get("scene_file"),
                           "world_dir": str(wd), "next": "the user navigates and presses 'Render with Atlas' in the "
                                                         "Previz panel; the picture lands in the viewer and on the canvas"})
    except AtlasError as exc:
        return exc.render()
    except Exception as exc:  # noqa: BLE001
        logger.exception("atlas_open_in_viewer failed")
        return wl._err(f"{type(exc).__name__}: {exc}", "api")
