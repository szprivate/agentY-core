"""Capture an Atlas world all round: views in every direction from the middle of the place, for a splat.

``atlas_expand_world`` adds a handful of views where the caller says. A splat of a whole room needs dozens,
turning all the way round, and Atlas invents what no view showed yet, so the order matters: this grows the
world outwards from the picture in yaw steps (0, +-30, +-60 ... 180), each batch seeing the existing views
nearest to it as context, so the back of the room is invented next to what is already there. Ceiling and
floor views come last.

Cameras are built in the world's mesh frame (the first picture's camera at the origin looking down -Z,
Y up): a ring around ``center``, each camera standing ``radius`` behind the centre and looking outwards
(so neighbouring views see the same walls from different places: parallax for the splat), plus an inner
camera every other step. Rotation = yaw about Y, then pitch about the camera's own X.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools.worldlabs_atlas import AtlasError

_BATCH = 8            # atlasGenerate targets per request


def _ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def look_dir(yaw: float, pitch: float = 0.0):
    """Where a camera with this yaw (left-positive) and pitch (up-positive) looks, in the mesh frame."""
    R = wl._mm(_ry(yaw), _rx(pitch))
    return [-R[0][2], -R[1][2], -R[2][2]]


def plan_ring(center, radius: float = 1.0, step_deg: float = 30.0, ceiling_floor: bool = True,
              inner: bool = True) -> list[dict]:
    """The capture, in the order it is generated: dicts with yaw, pitch, position (mesh frame)."""
    step = max(10.0, min(90.0, float(step_deg)))
    n = int(round(360.0 / step))
    step = 360.0 / n
    yaws = [0.0]
    for k in range(1, n // 2 + 1):
        a = k * step
        yaws += [a] if abs(a - 180.0) < 1e-6 else [a, -a]
    views = []
    for i, yaw in enumerate(yaws):
        d = look_dir(yaw)
        views.append({"yaw": yaw, "pitch": 0.0, "position": [center[j] - radius * d[j] for j in range(3)]})
        if inner and i % 2 == 0:
            views.append({"yaw": yaw, "pitch": 0.0, "position": [center[j] - 0.2 * radius * d[j] for j in range(3)]})
    if ceiling_floor:
        for yaw in (0.0, 90.0, 180.0, -90.0):
            d = look_dir(yaw)
            views.append({"yaw": yaw, "pitch": 40.0, "position": [center[j] - 0.5 * radius * d[j] for j in range(3)]})
        for yaw in (0.0, 180.0):
            d = look_dir(yaw)
            up = [center[0], center[1] + 0.4, center[2]]
            views.append({"yaw": yaw, "pitch": -35.0, "position": [up[j] - 0.8 * radius * d[j] for j in range(3)]})
    return views


def batches(views: list, size: int = _BATCH) -> list[list]:
    return [views[i:i + size] for i in range(0, len(views), size)]


def auto_center(world: dict, wd: Path) -> list:
    """The middle of the place: half-way to the far side straight ahead of the first picture."""
    import numpy as np
    f0 = next(f for f in world["frames"] if f.get("camera") and f.get("depth"))
    depth = wl._decode_depth(wd / f0["depth"], f0.get("depth_log") or {})
    h, w = depth.shape
    mid = depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    mid = mid[np.isfinite(mid) & (mid > 0)]
    dist = float(np.median(mid)) if mid.size else 4.0
    return [0.0, 0.0, -max(1.0, min(8.0, 0.5 * dist))]


def _center(world: dict, wd: Path, center) -> list:
    if center is None or center == {} or center == "":
        return auto_center(world, wd)
    if isinstance(center, dict):
        return [float(center.get("right", 0.0)), float(center.get("up", 0.0)), -float(center.get("forward", 0.0))]
    c = [float(x) for x in center]
    if len(c) != 3:
        raise AtlasError("center is {forward, right, up} metres from the first picture's camera, or [x, y, z]", "args")
    return c


def _context(world: dict, targets: list, base: dict, n: int = 4) -> list:
    """The n views with depth nearest any of the targets (position, then looking direction)."""
    scored = []
    for f in world["frames"]:
        if not (f.get("depth") and f.get("camera")):
            continue
        R, t = wl.camera_to_mesh_pose(base, f["camera"])
        fwd = [-R[0][2], -R[1][2], -R[2][2]]
        best = math.inf
        for v in targets:
            d = look_dir(v["yaw"], v["pitch"])
            dist = math.dist(t, v["position"])
            ang = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(fwd, d)))))
            best = min(best, dist + 2.0 * ang)
        scored.append((best, f["id"], f))
    scored.sort(key=lambda s: (s[0], s[1]))
    return [f for _s, _i, f in scored[:n]]


@wl._guard
def _capture(world, center, radius, step_deg, ceiling_floor, prompt, fov, mesh_detail, timeout_s, output_dir):
    from agenty_core.tools.atlas_render import resolve_world_dir
    wd = resolve_world_dir(world, output_dir)
    w = wl._load_world(wd)
    c = _center(w, wd, center)
    views = plan_ring(c, float(radius), float(step_deg), bool(ceiling_floor))
    base = wl._first_posed(w)
    K = wl.grid_intrinsics(base, fov)
    use_prompt = prompt or w.get("prompt_used") or w.get("prompt") or None
    before = len(w["frames"])
    done_batches, t0 = 0, time.monotonic()
    for b in batches(views):
        w = wl._load_world(wd)
        cams = [wl.mesh_pose_to_camera(base, K, wl._mm(_ry(v["yaw"]), _rx(v["pitch"])), v["position"]) for v in b]
        ctx = _context(w, b, base)
        if not ctx:
            raise AtlasError("the world has no view with depth to use as context", "args")
        body = {"contextFrames": wl.context_frames(wd, ctx), "targetCameras": cams, "returnDepth": True,
                "enhancePrompt": not bool(use_prompt and not prompt)}
        if use_prompt:
            body["prompt"] = use_prompt
        op = wl._submit(wl._TASK_GEN, body)
        job = {"kind": "expand", "task": wl._TASK_GEN, "name": w["name"], "world_dir": str(wd), "cameras": cams,
               "build_mesh": False}
        res = json.loads(wl._run(job, op, timeout_s))
        if res.get("error"):
            raise AtlasError(f"batch {done_batches + 1} of {len(batches(views))} failed: {res['error']}",
                             res.get("error_kind", "api"), views_added=len(wl._load_world(wd)["frames"]) - before)
        done_batches += 1
    w = wl._load_world(wd)
    stats = wl.build_world_mesh(wd, w, stride=max(1, int(mesh_detail)))
    wl._save_world(wd, w)
    return json.dumps({"status": "ok", "world": w["name"], "world_dir": str(wd), "views_added": len(w["frames"]) - before,
                       "views_total": len(w["frames"]), "center_mesh_frame": [round(x, 3) for x in c],
                       "batches": done_batches, "seconds": int(time.monotonic() - t0), "glb": str(wd / w["glb"]),
                       "mesh": stats,
                       "usage_note": "One Atlas task per batch of 8 views (World Labs credits).",
                       "next": f"train_splat('{w['name']}') turns these views into a Gaussian splat."})


@tool
def atlas_capture_around(world: str, center: Optional[dict] = None, radius: float = 1.0, step_deg: float = 30.0,
                         ceiling_floor: bool = True, prompt: str = "", fov: float = 0.0, mesh_detail: int = 4,
                         timeout_s: int = 900, output_dir: str = "") -> str:
    """Have Atlas paint an Atlas world ALL ROUND: views in every direction (plus ceiling and floor) from the
    middle of the place, grown outwards from the first picture so the unseen sides stay consistent with it.
    Use it before ``train_splat`` when the user wants a full 360 environment / an explorable scene.

    About 40 new views with the defaults: 5-6 billed Atlas tasks (World Labs credits), a few minutes.
    Atlas keeps what the prompt says: to keep people out (or anything else) say so in ``prompt``, and
    remove them from the starting picture first (an image edit), or they will be repeated all round.

    Args:
        world: name of a world made by atlas_create_world (or its folder).
        center: the middle of the place, {"forward": m, "right": m, "up": m} from the first picture's
            camera; default: half-way to what the first picture shows straight ahead.
        radius: how far behind the centre each camera stands, looking outwards (metres); 1.0 suits a room,
            more for a hall or a square, less for a small space.
        step_deg: turn between neighbouring views (30 = 12 directions; 20 = finer, more views and cost).
        ceiling_floor: also look up at the ceiling and down at the floor.
        prompt: scene description for every view; default the one the world was made with. Say "no people"
            here for an empty place.
        fov: vertical field of view in degrees; 0 = the first picture's.
        mesh_detail: depth-grid stride for the rebuilt world.glb.
        timeout_s: how long to wait for each batch.
        output_dir: where the world lives, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON: views_added, views_total, center_mesh_frame, glb, next - or {"error", "error_kind"} (a failed
        batch says how many views were already added; calling again adds a full new ring).
    """
    return _capture(world, center, radius, step_deg, ceiling_floor, prompt, fov, mesh_detail, timeout_s, output_dir)
