"""Animate a camera through a 3D scene in the bEpic viewer and render the move.

The step after "picture -> 3D scene" (a bEpic world, or a mesh from MoGe /
TripoSplat / SHARP): author a keyframed camera, have the viewer play it and write
the clip. That clip is a camera-accurate layout of the shot, which a video model
with a control input (Wan Fun Control, LTX IC-LoRA) turns into the finished shot.

The scene is plain JSON in the viewer's previz format (see the viewer's
``bEpicViewer_scene3d.js``): items with ``tracks`` of ``{f, v, ease}`` keys, rotations
in degrees, XYZ order, a camera looking down -Z. One call to the viewer's
``POST /bepic/previz_shot`` (through ComfyUI's HTTP API) opens it and renders; the
page does the drawing, so the browser with the viewer has to be open.

Camera moves are written relative to where the picture's own camera stood (the
world's ``refcam``, or the origin looking down -Z for a bare mesh): ``forward`` is
metres along the horizontal view direction, ``up`` metres of height, ``right``
sideways, ``yaw`` / ``pitch`` / ``roll`` degrees turned.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Optional

from agenty_core._compat import tool

logger = logging.getLogger("agenty_core.previz_shot")

CAMERA_ID = "shotcam"
_EASES = ("linear", "smooth", "hold")

# -- small vector / matrix helpers (3x3 row-major lists) ----------------------


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    n = math.sqrt(sum(x * x for x in a)) or 1.0
    return [x / n for x in a]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _ry(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def euler_to_matrix(rot):
    """three.js Euler 'XYZ' (degrees) -> rotation matrix: R = Rx * Ry * Rz."""
    return _mul(_mul(_rx(rot[0]), _ry(rot[1])), _rz(rot[2]))


def matrix_to_euler(m, near=None):
    """Inverse of euler_to_matrix, degrees; angles unwrapped toward `near` (the previous key)."""
    y = math.asin(max(-1.0, min(1.0, m[0][2])))
    if abs(m[0][2]) < 0.9999999:
        x = math.atan2(-m[1][2], m[2][2])
        z = math.atan2(-m[0][1], m[0][0])
    else:
        x = math.atan2(m[2][1], m[1][1])
        z = 0.0
    out = [math.degrees(x), math.degrees(y), math.degrees(z)]
    if near is not None:
        for i in range(3):
            while out[i] - near[i] > 180:
                out[i] -= 360
            while out[i] - near[i] < -180:
                out[i] += 360
    return out


def look_at_matrix(position, target, up=(0.0, 1.0, 0.0)):
    """A camera at `position` looking at `target` (camera looks down its own -Z)."""
    back = _norm(_sub(position, target))
    right = _norm(_cross(list(up), back))
    true_up = _cross(back, right)
    return [[right[0], true_up[0], back[0]], [right[1], true_up[1], back[1]], [right[2], true_up[2], back[2]]]


def _col(m, j):
    return [m[0][j], m[1][j], m[2][j]]


def _r(v, n=4):
    return [round(float(x), n) for x in v]


# -- the camera path ----------------------------------------------------------


def camera_pose_keys(base_pos, base_rot, moves, fps, base_fov=None):
    """[(frame, position, rotation_deg, fov|None)] for `moves`, relative to the base pose.

    Each move: ``t`` (seconds) and any of ``forward``/``right``/``up`` (metres),
    ``yaw``/``pitch``/``roll`` (degrees), ``fov`` (vertical degrees), or an absolute
    ``position`` [x, y, z] with ``look_at`` [x, y, z] instead of the relative fields.
    """
    r0 = euler_to_matrix(base_rot)
    fwd_cam = [-x for x in _col(r0, 2)]                      # where the camera looks
    flat = [fwd_cam[0], 0.0, fwd_cam[2]]
    if abs(flat[0]) + abs(flat[2]) < 1e-6:                   # looking straight down/up: use its up vector
        flat = [_col(r0, 1)[0], 0.0, _col(r0, 1)[2]]
    flat = _norm(flat)
    side = _norm(_cross(flat, [0.0, 1.0, 0.0]))              # horizontal right
    keys, prev = [], None
    for mv in moves:
        f = int(round(float(mv.get("t", 0.0)) * fps))
        if mv.get("position") is not None:
            pos = [float(x) for x in mv["position"]]
            if mv.get("look_at") is not None:
                rm = look_at_matrix(pos, [float(x) for x in mv["look_at"]])
            else:
                rm = r0
        else:
            fw, rt, up = float(mv.get("forward", 0)), float(mv.get("right", 0)), float(mv.get("up", 0))
            pos = [base_pos[i] + flat[i] * fw + side[i] * rt + (up if i == 1 else 0.0) for i in range(3)]
            rm = _mul(_mul(_mul(_ry(float(mv.get("yaw", 0))), r0), _rx(float(mv.get("pitch", 0)))),
                      _rz(float(mv.get("roll", 0))))
        rot = matrix_to_euler(rm, prev)
        prev = rot
        keys.append((f, pos, rot, float(mv["fov"]) if mv.get("fov") is not None else base_fov))
    return keys


def camera_item(keys, fov, resolution, ease="linear"):
    """The scene's camera item: first key as its static pose, the keys as tracks."""
    ease = ease if ease in _EASES else "linear"
    _f0, p0, r0, _ = keys[0]
    tracks = {"position": [{"f": f, "v": _r(p), "ease": ease} for f, p, _rot, _fv in keys],
              "rotation": [{"f": f, "v": _r(r, 3), "ease": ease} for f, _p, r, _fv in keys]}
    fovs = [(f, v) for f, _p, _rot, v in keys if v is not None]
    if len({round(v, 4) for _f, v in fovs}) > 1:
        tracks["fov"] = [{"f": f, "v": round(v, 3), "ease": ease} for f, v in fovs]
    return {"id": CAMERA_ID, "kind": "camera", "name": "Shot camera", "position": _r(p0),
            "rotation": _r(r0, 3), "scale": [1, 1, 1], "fov": round(float(keys[0][3] or fov), 3),
            "resolution": [int(resolution[0]), int(resolution[1])], "pivot": [0, 0, 0], "visible": True,
            "tracks": tracks, "parent": None}


def model_item(path, name="Scene model", position=None, rotation=None, scale=1.0):
    ext = os.path.splitext(path)[1][1:].lower()
    s = [float(scale)] * 3 if not isinstance(scale, (list, tuple)) else [float(x) for x in scale]
    return {"id": "scene", "kind": "model", "name": name,
            "src": {"kind": "model", "format": ext, "path": path, "name": os.path.basename(path), "external": True},
            "position": _r(position or [0, 0, 0]), "rotation": _r(rotation or [0, 0, 0], 3), "scale": s,
            "pivot": [0, 0, 0], "visible": True, "tracks": {}, "parent": None}


# Pieces of a world that are not the picture's own place: dropped from a control render.
_WORLD_DROP = {"terrain", "scatter", "light"}


def scene_from_world(world: dict, keep_all: bool = False) -> tuple[dict, dict]:
    """(scene, refcam) from a world's scene JSON. Unless `keep_all`, the terrain, scatter and
    lamps go, and the picture's depth view goes too when a scene model stands in for it."""
    scene = json.loads(json.dumps(world))
    items = scene.get("items") or []
    ref = next((i for i in items if i.get("id") == "refcam"), None) or {}
    if not keep_all:
        has_model = any(i.get("id") == "scene" for i in items)
        scene["items"] = [i for i in items if i.get("kind") not in _WORLD_DROP
                          and not (has_model and i.get("kind") == "depthmesh")]
    scene.pop("walk", None)
    return scene, ref


def ground_item(y: float, color: str = "#4a5564", size: float = 4000.0) -> dict:
    """A big flat plane at height `y`: the ground a single-picture model does not hold
    (it only has what the picture shows, so a camera moving forward soon films nothing below)."""
    return {"id": "ground", "kind": "primitive", "name": "Ground", "primitive": {"type": "plane"},
            "color": color, "position": [0, round(float(y), 4), 0], "rotation": [0, 0, 0], "scale": [size, 1, size],
            "pivot": [0, 0, 0], "visible": True, "tracks": {}, "parent": None}


def build_scene(*, world: Optional[dict] = None, model: str = "", moves: list, seconds: float, fps: float,
                fov: Optional[float], resolution, ease: str, keep_world: bool = False,
                model_scale=1.0, model_position=None, model_rotation=None, render: Optional[dict] = None,
                ground=True, ground_color: str = "#4a5564"):
    """The previz scene for a camera move. Returns (scene, summary)."""
    if world is not None:
        scene, ref = scene_from_world(world, keep_world)
        base_pos = [float(x) for x in ref.get("position", [0, 0, 0])]
        base_rot = [float(x) for x in ref.get("rotation", [0, 0, 0])]
        base_fov = float(fov if fov else ref.get("fov", 50.0))
    elif model:
        scene = {"version": 1, "items": [model_item(model, "Scene model", model_position, model_rotation, model_scale)]}
        base_pos, base_rot, base_fov = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], float(fov or 50.0)
    else:
        raise ValueError("give a world (name) or a model (path to a mesh)")
    if not moves:
        raise ValueError("no camera moves given")
    moves = sorted(moves, key=lambda m: float(m.get("t", 0)))
    keys = camera_pose_keys(base_pos, base_rot, moves, fps, base_fov)
    if len(keys) < 2 or keys[-1][0] <= keys[0][0]:
        raise ValueError("the camera needs at least two moves at different times")
    shift = keys[0][0]
    keys = [(f - shift, p, r, v) for f, p, r, v in keys]
    if ground and not keep_world:
        sm = next((i for i in scene["items"] if i.get("id") == "scene"), None)
        gy = ((sm or {}).get("scene_model") or {}).get("ground_y")
        if gy is not None and not any(i.get("id") == "ground" for i in scene["items"]):
            scene["items"].append(ground_item(gy, ground_color))
    cam = camera_item(keys, base_fov, resolution, ease)
    scene["items"] = [i for i in scene["items"] if i.get("id") != CAMERA_ID] + [cam]
    length = keys[-1][0] + 1
    scene.update({"fps": float(fps), "length": length, "lengthSet": True, "activeCamera": CAMERA_ID})
    if render:
        scene["render"] = render
    return scene, {"frames": length, "fps": fps,
                   "keys": [{"frame": f, "position": _r(p, 3), "rotation": _r(r, 2)} for f, p, r, _v in keys]}


# -- the tool -----------------------------------------------------------------


def _client():
    from agenty_core.utils.comfyui_client import get_client
    return get_client()


def _error(msg: str) -> str:
    return json.dumps({"status": "error", "error": msg})


def _resolve_model(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    try:
        from agenty_core.utils.comfyui_client import comfyui_output_dir
        out = comfyui_output_dir()
        if out and os.path.exists(os.path.join(str(out), path)):
            return os.path.join(str(out), path)
    except Exception:  # noqa: BLE001
        pass
    return os.path.abspath(path)


@tool
def render_camera_move(
    name: str,
    moves: list[dict],
    world: str = "",
    model: str = "",
    seconds: float = 0.0,
    fps: float = 24.0,
    fov: float = 0.0,
    width: int = 864,
    height: int = 480,
    ease: str = "linear",
    shading: str = "original",
    background: str = "viewer",
    color: str = "#7d8794",
    keep_world: bool = False,
    ground: bool = True,
    ground_color: str = "#4a5564",
    model_scale: float = 1.0,
    render: bool = True,
    add_to_canvas: bool = True,
    timeout_seconds: float = 900.0,
) -> str:
    """Fly a camera through a 3D scene in the bEpic viewer and render the move to a clip.

    Use it after a picture became a 3D scene (a bEpic world made with world_create and
    world_scene_model, or a mesh file). The clip is a camera-accurate layout of the new
    shot: feed it to a video model with a control input (Wan Fun Control, an LTX IC-LoRA)
    to make a finished shot with that camera move. The viewer (with the browser open)
    does the drawing; the scene opens in a "Previz: <name>" tab, so the user sees the 3D
    scene, the camera path and the render.

    Args:
        name: Name of the shot (a-z, 0-9, _ -); the clip is output/previz/<name>.mp4.
        moves: Camera keyframes relative to where the picture's own camera stood.
            Each: {"t": seconds, "forward": m, "right": m, "up": m, "yaw": deg, "pitch": deg,
            "roll": deg, "fov": deg}. Omitted fields are 0. forward is metres along the
            horizontal view direction. Needs at least two, e.g. a low push-in with a slight
            rise: [{"t":0}, {"t":5, "forward":40, "up":1.5}]. For absolute control use
            {"t", "position":[x,y,z], "look_at":[x,y,z]} in scene coordinates.
        world: A bEpic world's name; its scene model (or depth view) is the set. The terrain,
            scatter and lamps are left out unless keep_world.
        model: Instead of a world: absolute path of a mesh (.glb/.ply/.obj), built in the
            picture's camera (MoGe, SHARP) so the camera starts at the origin looking -Z.
        seconds: Informational; the length comes from the last move's "t".
        fps: Frame rate of the render (24 matches most clips).
        fov: Vertical field of view in degrees; 0 = the picture's own (the world's reference camera).
        width: Render width in px (video encoders want even numbers).
        height: Render height in px.
        ease: "linear" (constant speed, default), "smooth" (eases in and out) or "hold".
        shading: "original" | "clay" (grey layout look) | "normal" | "wireframe".
        background: "viewer" | "color" (then `color`, e.g. the sky's colour).
        color: Background colour for background="color".
        keep_world: Keep a world's terrain / scatter / lamps in the render.
        ground: With a world's scene model, stand a flat ground plane at the model's ground height
            (a single picture holds no ground under or just ahead of the camera).
        ground_color: Colour of that plane (match the picture's ground).
        model_scale: Scale of a bare mesh (metres per model unit).
        render: False opens the scene and the camera in the viewer without rendering.
        add_to_canvas: A finished render also lands on the ComfyUI canvas as a loader node.
        timeout_seconds: How long to wait for the render.

    Returns:
        JSON: status, path (the rendered clip), frames, fps, scene_file, keys (the camera path).
    """
    t0 = time.time()
    try:
        world_scene = None
        if world:
            world_scene = _client().get("/bepic_worlds/world", params={"name": world})
            if not isinstance(world_scene, dict) or "items" not in world_scene:
                return _error(f"no world '{world}'")
        render_cfg = {"format": "mp4", "quality": "high", "shading": shading, "background": background,
                      "color": color, "aa": 2, "fps": fps}
        scene, summary = build_scene(world=world_scene, model=_resolve_model(model), moves=moves or [],
                                     seconds=seconds, fps=fps, fov=fov or None, resolution=(width, height),
                                     ease=ease, keep_world=keep_world, model_scale=model_scale,
                                     render=render_cfg, ground=ground, ground_color=ground_color)
        body = {"scene": scene, "name": name, "focus": True, "add_node": bool(add_to_canvas)}
        if render:
            body["render"] = {**render_cfg, "camera": CAMERA_ID, "useCameraRes": True}
        try:
            job = _client().post("/bepic/previz_shot", json_data=body)
        except Exception as exc:  # noqa: BLE001
            if "404" in str(exc):
                return _error("the bEpic viewer is not installed in ComfyUI, or is too old for previz shots "
                              "(update ComfyUI-ImageViewer and restart ComfyUI)")
            return _error(f"could not reach the viewer through ComfyUI: {exc}")
        if not isinstance(job, dict) or "id" not in job:
            return _error(f"unexpected answer from the viewer: {job!r}")
        out = {"status": "ok", "scene_file": job.get("scene_file"), "tab": job.get("key"), **summary}
        if not render:
            return json.dumps({**out, "rendered": False})
        deadline = t0 + float(timeout_seconds)
        state = {}
        while time.time() < deadline:
            time.sleep(2.0)
            try:
                state = _client().get("/bepic/previz_shot", params={"id": job["id"]})
            except Exception as exc:  # noqa: BLE001
                logger.warning("previz shot poll failed: %s", exc)
                continue
            if state.get("state") in ("done", "error"):
                break
        if state.get("state") == "done":
            return json.dumps({**out, "rendered": True, "path": state.get("path"), "where": state.get("where"),
                               "message": state.get("message"), "seconds_taken": round(time.time() - t0, 1)})
        if state.get("state") == "error":
            return _error(state.get("message") or "the viewer could not render the shot")
        return _error("timed out waiting for the viewer to render (is the bEpic viewer open in a browser "
                      f"tab? last state: {state.get('state') or 'unknown'}, "
                      f"{state.get('done', 0)} frames done)")
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("render_camera_move failed")
        return _error(str(exc))
