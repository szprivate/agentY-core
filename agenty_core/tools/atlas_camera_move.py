"""Render a camera MOVE through an Atlas world with Atlas-painted pictures (not a splat, not a bare mesh).

What the API offers (docs: https://atlas-beta.worldlabs.ai/docs/api-reference/atlas-generate, /docs/atlas-generate):
there is NO video, trajectory or sequence mode in the Marble 2 Atlas beta Developer API. ``atlasGenerate``
takes posed RGBD context frames and a LIST of target cameras (1280 x 720 each) and returns one still per
camera; no maximum is published (this layer uses at most 8 per task). Video along a camera path exists only
in World Labs' closed early-access "Atlas" omni model, which has no public API.

So the move is made on our side, in two steps:

1. **Paint.** Keyframe cameras are taken along the path (every 1.0 / 0.5 / 0.25 s by ``quality``) and Atlas
   paints them, a few per task, each task conditioned on the nearest views of the world INCLUDING the ones
   painted by the previous tasks (each batch is added to the world with its depth), so the keyframes agree.
2. **Fill.** Every video frame is rendered through the grown world's depth mesh (textured with the painted
   keyframes) in the bEpic viewer, along the exact same camera path. At a keyframe the picture is Atlas's;
   in between it is those pictures reprojected - temporally stable by construction (no per-frame generation).

Cost: World Labs publishes no per-task price for Atlas ("see the rate card in the platform"; 1,000 credits per
US$). The estimate is therefore a RANGE built from two settings (``WLT_ATLAS_CREDITS_PER_TASK``, default 100 -
an assumption - and ``WLT_CREDITS_PER_USD``, default 1000): the low end bills per task, the high end per target
camera. There is no balance API; a cap (``max_usd`` / ``WLT_ATLAS_MOVE_MAX_USD``, default 5) refuses bigger
plans, and anything at or above ``WLT_ATLAS_CONFIRM_ABOVE_USD`` (default 0.25, never above the partner-credit
threshold) needs ``confirm_cost=True``. ``dry_run=True`` prints the plan and spends nothing.

A re-run with the same arguments resumes (``<work world>/moves/<name>/plan.json`` remembers which tasks were
paid for), so a failure or a timeout never pays a task twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from agenty_core._compat import tool
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools.worldlabs_atlas import AtlasError

logger = logging.getLogger("agenty_core.atlas_camera_move")

# quality -> (seconds between painted keyframes, cameras per Atlas task)
QUALITY = {"draft": (1.0, 8), "standard": (0.5, 4), "high": (0.25, 4)}
MAX_PER_TASK = 8
MAX_SECONDS = 30.0
MAX_TASKS = 40
DEFAULT_MAX_USD = 5.0
DEFAULT_CONFIRM_USD = 0.25
DEFAULT_CREDITS_PER_TASK = 100.0
DEFAULT_CREDITS_PER_USD = 1000.0


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


# -- the camera path ----------------------------------------------------------


def _smooth(u: float) -> float:
    return u * u * (3.0 - 2.0 * u)


def _to_mesh_point(base: dict, p, frame: str):
    """An absolute point -> the mesh frame. 'picture' (and the mesh itself) is already it; 'world' is the Atlas
    world's own coordinates, where the first picture's camera stands at base['t']."""
    if frame == "world":
        return wl._mv(wl._tr(base["R"]), [float(a) - b for a, b in zip(p, base["t"])])
    return [float(x) for x in p]


def waypoints(world: dict, moves=None, keys=None, fov: float = 0.0, frame: str = "picture", fps: float = 24.0):
    """``[{"t", "R", "p", "fov"}]`` in the mesh frame (first picture's camera = origin, looking down -Z)."""
    from agenty_core.tools.previz_shot import camera_pose_keys, euler_to_matrix
    from agenty_core.tools import atlas_render as ar
    if frame not in ("picture", "world", ""):
        raise AtlasError("frame must be 'picture' or 'world'", "args")
    base = wl._first_posed(world)
    K = wl.grid_intrinsics(base, fov)
    default_fov = math.degrees(2.0 * math.atan((K["height"] / 2.0) / K["fy"]))
    out = []
    if keys:
        for i, k in enumerate(keys):
            if not isinstance(k, dict) or k.get("t") is None:
                raise AtlasError(f"key {i}: give a time 't' (seconds) and a camera", "args")
            spec = {a: b for a, b in k.items() if a != "t"}
            if frame == "world":
                for a in ("position", "look_at"):
                    if spec.get(a) is not None:
                        spec[a] = _to_mesh_point(base, spec[a], "world")
            _api, (R, p), f = ar.resolve_camera(world, spec, fov)
            out.append({"t": float(k["t"]), "R": R, "p": p, "fov": f})
    elif moves:
        mv = []
        for m in moves:
            m = dict(m)
            if frame == "world":
                for a in ("position", "look_at"):
                    if m.get(a) is not None:
                        m[a] = _to_mesh_point(base, m[a], "world")
            mv.append(m)
        mv.sort(key=lambda m: float(m.get("t", 0)))
        for (_f, pos, rot, kfov), m in zip(camera_pose_keys([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], mv, fps, None), mv):
            use = float(fov) if fov else (float(kfov) if kfov else default_fov)
            out.append({"t": float(m.get("t", 0.0)), "R": euler_to_matrix(rot), "p": [float(x) for x in pos],
                        "fov": max(ar.MIN_FOV, min(ar.MAX_FOV, use))})
    else:
        raise AtlasError("give the camera path: moves=[{t, forward/right/up/yaw/pitch/roll | position+look_at}, ...] "
                         "or keys=[{t, camera}, ...]", "args")
    out.sort(key=lambda w: w["t"])
    if len(out) < 2 or out[-1]["t"] - out[0]["t"] <= 1e-6:
        raise AtlasError("the camera path needs at least two keys at different times", "args")
    t0 = out[0]["t"]
    for w in out:
        w["t"] -= t0
    return out


def _slerp(qa, qb, u):
    d = sum(a * b for a, b in zip(qa, qb))
    if d < 0:
        qb, d = [-x for x in qb], -d
    if d > 0.9995:
        q = [a + (b - a) * u for a, b in zip(qa, qb)]
    else:
        th = math.acos(max(-1.0, min(1.0, d)))
        s = math.sin(th)
        wa, wb = math.sin((1 - u) * th) / s, math.sin(u * th) / s
        q = [wa * a + wb * b for a, b in zip(qa, qb)]
    n = math.sqrt(sum(x * x for x in q)) or 1.0
    return [x / n for x in q]


def pose_at(wps: list, t: float, ease: str = "linear"):
    """``(R, p, fov)`` of the path at time ``t`` (position and lens linear, rotation by slerp)."""
    total = wps[-1]["t"]
    t = max(0.0, min(total, t))
    if ease == "smooth":
        t = total * _smooth(t / total)
    i = 0
    while i < len(wps) - 2 and wps[i + 1]["t"] <= t:
        i += 1
    a, b = wps[i], wps[i + 1]
    span = b["t"] - a["t"]
    u = 0.0 if ease == "hold" else (t - a["t"]) / span if span > 0 else 0.0
    if ease == "hold" and t >= b["t"]:
        u = 1.0
    p = [x + (y - x) * u for x, y in zip(a["p"], b["p"])]
    q = _slerp(wl._mat_to_quat(a["R"]), wl._mat_to_quat(b["R"]), u)
    return wl._quat_to_mat(q), p, a["fov"] + (b["fov"] - a["fov"]) * u


def camera_for(world: dict, R, p, fov: float) -> dict:
    base = wl._first_posed(world)
    return wl.mesh_pose_to_camera(base, wl.grid_intrinsics(base, fov), R, p)


# -- the plan and its price -----------------------------------------------------


def keyframe_times(total: float, interval: float) -> list:
    n = max(1, int(math.ceil(total / interval - 1e-9)))
    return [round(total * i / n, 4) for i in range(n + 1)]


def split_batches(times: list, per_task: int) -> list:
    k = len(times)
    nb = max(1, int(math.ceil(k / per_task)))
    base, extra = divmod(k, nb)
    out, i = [], 0
    for b in range(nb):
        size = base + (1 if b < extra else 0)
        out.append(times[i:i + size])
        i += size
    return out


def price(tasks: int, cameras: int) -> dict:
    per_task, per_usd = _envf("WLT_ATLAS_CREDITS_PER_TASK", DEFAULT_CREDITS_PER_TASK), \
        _envf("WLT_CREDITS_PER_USD", DEFAULT_CREDITS_PER_USD)
    lo, hi = tasks * per_task, cameras * per_task
    return {"credits_per_task_assumed": per_task, "credits_per_usd": per_usd,
            "min_credits": lo, "max_credits": hi, "min_usd": round(lo / per_usd, 3), "max_usd": round(hi / per_usd, 3),
            "basis": "World Labs publishes no per-task price for Atlas (billing docs: 'check the rate card in the "
                     "platform'). min = one charge per task, max = one per painted camera; the per-task credits are an "
                     "ASSUMPTION (WLT_ATLAS_CREDITS_PER_TASK). The platform's Usage page shows the real spend."}


def make_plan(world: dict, wps: list, fps: float, quality: str, ease: str, seconds: float = 0.0) -> dict:
    if quality not in QUALITY:
        raise AtlasError(f"quality is one of {', '.join(QUALITY)}", "args")
    total = wps[-1]["t"]
    if total > MAX_SECONDS:
        raise AtlasError(f"a move is at most {int(MAX_SECONDS)} s (this one is {total:.1f} s)", "args")
    interval, per_task = QUALITY[quality]
    times = keyframe_times(total, interval)
    batches = split_batches(times, min(per_task, MAX_PER_TASK))
    centres = [pose_at(wps, t, ease)[1] for t in times]
    spread = max(math.dist(c, centres[0]) for c in centres)
    if spread < 0.02:
        raise AtlasError("the camera only turns on the spot: Atlas needs the cameras to be at different positions "
                         "to return depth (a pure rotation is rejected with HTTP 422). Add some forward/right/up "
                         "movement, or use render_camera_move through the mesh for a pan.", "args")
    nframes = int(round(total * fps)) + 1
    est = price(len(batches), len(times))
    return {"approach": "Atlas-painted keyframes along the path (a few per atlasGenerate task, each task conditioned "
                        "on the previous ones) + every frame rendered through the grown world's depth mesh in the viewer",
            "seconds": round(total, 3), "fps": fps, "frames": nframes, "quality": quality,
            "keyframes": len(times), "keyframe_times": times, "tasks": len(batches),
            "batches": [{"task": i + 1, "times": b} for i, b in enumerate(batches)],
            "path_extent_m": round(spread, 3), "estimate": est}


def _plan_hash(world_name, wps, fps, quality, ease, fov, prompt, grow) -> str:
    blob = json.dumps([world_name, [[round(w["t"], 4), [[round(v, 4) for v in r] for r in w["R"]],
                                     [round(v, 4) for v in w["p"]], round(w["fov"], 3)] for w in wps],
                       fps, quality, ease, round(fov, 3), prompt, bool(grow)], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def gate(plan: dict, max_usd: float, confirm: bool) -> None:
    """Refuse a plan above the cap, or above the confirmation level without the user's go-ahead."""
    from agenty_core import partner_credits as pc
    est = plan["estimate"]
    cap = float(max_usd) if max_usd and max_usd > 0 else _envf("WLT_ATLAS_MOVE_MAX_USD", DEFAULT_MAX_USD)
    if est["max_usd"] > cap + 1e-9:
        raise AtlasError(f"Not run: the plan could cost up to ${est['max_usd']:.2f} ({plan['tasks']} tasks, "
                         f"{plan['keyframes']} painted views), above the cap of ${cap:.2f}. Lower quality, shorten the "
                         "move, or raise max_usd with the user's go-ahead.", "over_cap",
                         plan=plan, cap_usd=cap)
    thr = min(pc.confirm_threshold(), _envf("WLT_ATLAS_CONFIRM_ABOVE_USD", DEFAULT_CONFIRM_USD))
    given = confirm or pc.confirm_given()
    if est["max_usd"] >= thr and est["max_usd"] > 0 and not given:
        raise pc.CostGateError(
            "confirm_cost",
            f"Not run: this paints {plan['keyframes']} views in {plan['tasks']} Atlas tasks, about "
            f"${est['min_usd']:.2f}-${est['max_usd']:.2f} of World Labs credits (price assumed, see estimate.basis). "
            "Tell the user the plan and price and ask whether to go ahead; only when they say yes, run it again with "
            "confirm_cost=True.", {"min_usd": est["min_usd"], "max_usd": est["max_usd"], "nodes": []}, None)


# -- running it -----------------------------------------------------------------


def _copy_world(src: Path, dst: Path, name: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copyfile(f, dst / f.name)
    w = json.loads((dst / "world.json").read_text(encoding="utf-8"))
    w["name"] = name
    w["copied_from"] = src.name
    (dst / "world.json").write_text(json.dumps(w, indent=1), encoding="utf-8")


def _state_path(wd: Path, name: str) -> Path:
    return wd / "moves" / wl._slug(name) / "plan.json"


def _context_for(world: dict, cams: list) -> list:
    from agenty_core.tools import atlas_render as ar
    seen, out = set(), []
    picks = [cams[0], cams[-1]] + ([cams[len(cams) // 2]] if len(cams) > 2 else [])
    for c in picks:
        for f in ar._pick_context(world, c, 3):
            if f["id"] not in seen:
                seen.add(f["id"])
                out.append(f)
    return out[:8]


def _paint_batch(wd: Path, world: dict, cams: list, prompt: str, mesh_detail: int, timeout_s: float, state_b: dict,
                 save: Callable[[], None]) -> dict:
    """One atlasGenerate task for ``cams``; adds the views (with depth) to the world and rebuilds its mesh."""
    if state_b.get("op_id"):
        op = wl._request("GET", f"/operations/{state_b['op_id']}")
    else:
        ctx = _context_for(world, cams)
        if not ctx:
            raise AtlasError("the world has no view with depth to use as context", "args")
        use_prompt = prompt or world.get("prompt_used") or world.get("prompt") or None
        body = {"contextFrames": wl.context_frames(wd, ctx), "targetCameras": cams, "returnDepth": True,
                "enhancePrompt": not bool(use_prompt and not prompt)}
        if use_prompt:
            body["prompt"] = use_prompt
        op = wl._submit(wl._TASK_GEN, body)
        state_b.update(op_id=op.get("id"), status="submitted", context=[f["id"] for f in ctx])
        save()
    op = wl._wait(op, timeout_s)
    frames = (op.get("response") or {}).get("frames") or []
    if len(frames) != len(cams):
        raise AtlasError(f"Atlas returned {len(frames)} views for {len(cams)} cameras", "api", operation_id=op.get("id"))
    ids = []
    for j, fr in enumerate(frames):
        if not fr.get("camera"):
            fr = {**fr, "camera": cams[j]}
        if not (fr.get("depth") or {}):
            raise AtlasError("Atlas returned a view without depth, so it cannot join the world", "api",
                             operation_id=op.get("id"))
        nid = max([f["id"] for f in world["frames"]] + [-1]) + 1
        world["frames"].append(wl._frame_entry(world, wd, nid, fr, "move", op.get("id", "")))
        ids.append(nid)
    world["operations"].append({"id": op.get("id"), "task": wl._TASK_GEN, "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
    wl.build_world_mesh(wd, world, stride=max(1, int(mesh_detail)))
    wl._save_world(wd, world)
    state_b.update(status="done", frames=ids)
    save()
    return op


def _viewer_render(scene: dict, name: str, cfg: dict, add_to_canvas: bool, timeout_s: float) -> dict:
    from agenty_core.tools import previz_shot as ps
    t0 = time.time()
    body = {"scene": scene, "name": name, "focus": True, "add_node": bool(add_to_canvas),
            "render": {**cfg, "camera": ps.CAMERA_ID, "useCameraRes": True}}
    try:
        job = ps._client().post("/bepic/previz_shot", json_data=body)
    except Exception as exc:  # noqa: BLE001
        if "404" in str(exc):
            raise AtlasError("the bEpic viewer is not installed in ComfyUI, or too old for previz shots", "dependency")
        raise AtlasError(f"could not reach the viewer through ComfyUI: {exc}", "dependency")
    if not isinstance(job, dict) or "id" not in job:
        raise AtlasError(f"unexpected answer from the viewer: {job!r}", "dependency")
    state = {}
    while time.time() < t0 + float(timeout_s):
        wl._sleep(2.0)
        try:
            state = ps._client().get("/bepic/previz_shot", params={"id": job["id"]})
        except Exception as exc:  # noqa: BLE001
            logger.warning("previz shot poll failed: %s", exc)
            continue
        if state.get("state") in ("done", "error"):
            break
    if state.get("state") == "error":
        raise AtlasError(state.get("message") or "the viewer could not render the shot", "dependency")
    if state.get("state") != "done":
        raise AtlasError("timed out waiting for the viewer to render (is the bEpic viewer open in a browser tab?); "
                         "the painted views are saved, run the same call again to render only the video",
                         "timeout")
    return state


def _mesh_scene(wd: Path, world: dict, wps: list, fps: float, ease: str, size) -> dict:
    from agenty_core.tools import previz_shot as ps
    total = wps[-1]["t"]
    n = int(round(total * fps)) + 1
    keys, prev = [], None
    for i in range(n):
        R, p, fov = pose_at(wps, i / fps, ease)
        rot = ps.matrix_to_euler(R, prev)
        prev = rot
        keys.append((i, p, rot, fov))
    cam = ps.camera_item(keys, keys[0][3], size, "linear")
    scene = {"version": 1, "items": [ps.model_item(str(wd / world["glb"]), f"Atlas: {world['name']}"), cam],
             "fps": float(fps), "length": n, "lengthSet": True, "activeCamera": ps.CAMERA_ID}
    return scene


def render_move(name: str, world: str, moves=None, keys=None, fps: float = 24.0, seconds: float = 0.0,
                quality: str = "standard", fov: float = 0.0, frame: str = "picture", ease: str = "linear",
                prompt: str = "", width: int = 1280, height: int = 720, grow_world: bool = False,
                mesh_detail: int = 4, max_usd: float = 0.0, confirm_cost: bool = False, dry_run: bool = False,
                add_to_canvas: bool = True, timeout_s: int = 600, output_dir: str = "",
                on_progress: Optional[Callable[[dict], None]] = None) -> str:
    """Plain-function form of :func:`atlas_render_camera_move` (the host's viewer route calls this)."""
    try:
        return _render_move(name, world, moves, keys, fps, seconds, quality, fov, frame, ease, prompt, width, height,
                            grow_world, mesh_detail, max_usd, confirm_cost, dry_run, add_to_canvas, timeout_s,
                            output_dir, on_progress)
    except AtlasError as exc:
        return exc.render()
    except Exception as exc:  # noqa: BLE001
        from agenty_core import partner_credits as pc
        if isinstance(exc, pc.CostGateError):
            return json.dumps(exc.payload() | getattr(exc, "extra", {}))
        logger.exception("atlas_render_camera_move failed")
        return wl._err(f"{type(exc).__name__}: {exc}", "api")


def _render_move(name, world, moves, keys, fps, seconds, quality, fov, frame, ease, prompt, width, height, grow_world,
                 mesh_detail, max_usd, confirm_cost, dry_run, add_to_canvas, timeout_s, output_dir, on_progress):
    from agenty_core import partner_credits as pc
    from agenty_core.tools import atlas_render as ar
    shot = wl._slug(name)
    if not shot:
        raise AtlasError("name the shot (a-z, 0-9, _ -)", "args")
    fps = float(fps or 24.0)
    if not 1 <= fps <= 60:
        raise AtlasError("fps is 1 to 60", "args")
    src = ar.resolve_world_dir(world, output_dir)
    w0 = wl._load_world(src)
    if not w0.get("glb"):
        raise AtlasError("the world has no mesh (world.glb); atlas_rebuild_world_mesh makes one", "args")
    wps = waypoints(w0, moves, keys, float(fov or 0.0), frame, fps)
    if seconds and seconds > 0 and abs(wps[-1]["t"] - seconds) > 1e-6:       # scale the path to the length asked for
        k = float(seconds) / wps[-1]["t"]
        for p in wps:
            p["t"] *= k
    plan = make_plan(w0, wps, fps, quality, ease)
    plan.update(world=src.name, shot=shot, size=[int(width), int(height)], works_on="the world itself (grown)" if grow_world
                else f"a copy, {src.name}__{shot}; the original is not changed")
    use_prompt = prompt or w0.get("prompt_used") or w0.get("prompt") or ""
    plan["prompt"] = use_prompt[:200]
    if dry_run:
        return json.dumps({"status": "dry_run", "spent": 0, "plan": plan,
                           "note": "Nothing was sent. Run again without dry_run (and with confirm_cost=True once the "
                                   "user agrees to the price) to start."})
    try:
        gate(plan, max_usd, confirm_cost)
    except pc.CostGateError as exc:
        exc.extra = {"plan": plan}
        raise
    if not wl._api_key():
        return wl._err("WLT_API_KEY is not set (put the World Labs key in .env).", "auth")
    if plan["tasks"] > MAX_TASKS:
        raise AtlasError(f"too many tasks ({plan['tasks']})", "args")
    # -- the work world
    wd = src if grow_world else src.with_name(f"{src.name}__{shot}")
    h = _plan_hash(src.name, wps, fps, quality, ease, float(fov or 0.0), use_prompt, grow_world)
    sp = _state_path(wd, shot)
    state = None
    if sp.exists():
        state = json.loads(sp.read_text(encoding="utf-8"))
        if state.get("hash") != h:
            raise AtlasError(f"'{shot}' was already started with a different path or settings (state in {sp.parent}); "
                             "use another shot name", "args")
    if not grow_world and not (wd / "world.json").exists():
        _copy_world(src, wd, wd.name)
    world = wl._load_world(wd)
    if state is None:
        state = {"hash": h, "batches": [{"task": b["task"], "status": "todo"} for b in plan["batches"]],
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    sp.parent.mkdir(parents=True, exist_ok=True)

    def save():
        sp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    save()

    def report(phase, done):
        if on_progress:
            on_progress({"phase": phase, "tasks_done": done, "tasks_total": plan["tasks"]})

    paid_before = sum(1 for b in state["batches"] if b.get("op_id"))
    for i, b in enumerate(plan["batches"]):
        sb = state["batches"][i]
        done = sum(1 for x in state["batches"] if x["status"] == "done")
        if sb["status"] == "done":
            continue
        report(f"Atlas is painting views {i + 1} of {plan['tasks']}", done)
        cams = []
        for t in b["times"]:
            R, p, f = pose_at(wps, t, ease)
            cams.append(camera_for(world, R, p, f))
        try:
            _paint_batch(wd, world, cams, prompt, mesh_detail, timeout_s, sb, save)
        except AtlasError as exc:
            exc.extra.setdefault("tasks_done", done)
            exc.extra.setdefault("tasks_total", plan["tasks"])
            exc.extra.setdefault("resume", "run the same call again: finished tasks are not paid twice")
            raise
    report("Rendering the move through the painted world", plan["tasks"])
    cfg = {"format": "mp4", "quality": "high", "shading": "original", "background": "viewer", "color": "#7d8794",
           "aa": 2, "fps": fps}
    world = wl._load_world(wd)
    scene = _mesh_scene(wd, world, wps, fps, ease, (int(width), int(height)))
    out_name = f"{wd.name}_{shot}"
    res = _viewer_render(scene, out_name, cfg, add_to_canvas, 900.0)
    path = res.get("path")
    from agenty_core.tools import previz_shot as ps
    measured = ps._measure_render(path, expect_motion=True)
    tasks_used = sum(1 for x in state["batches"] if x.get("op_id"))
    out = {"status": "ok", "path": path, "where": res.get("where"), "frames": plan["frames"], "fps": fps,
           "seconds": plan["seconds"], "world": wd.name, "world_dir": str(wd), "glb": str(wd / world["glb"]),
           "approach": plan["approach"], "quality": quality, "tasks_used": tasks_used,
           "tasks_paid_this_run": tasks_used - paid_before, "keyframes_painted": plan["keyframes"],
           "operations": [x.get("op_id") for x in state["batches"]], "estimate": plan["estimate"],
           "usage_note": "Atlas reports no per-task cost; the real spend is on the World Labs platform's Usage page "
                         f"({tasks_used} atlasGenerate tasks)."}
    out.update(measured)
    if path and os.path.isfile(str(path)):
        wl._sidecar(Path(path), "Atlas camera move (painted keyframes + mesh render)",
                    {"world": wd.name, "plan": {k: plan[k] for k in ("seconds", "fps", "frames", "quality", "keyframes",
                                                                       "tasks", "keyframe_times")},
                     "operations": out["operations"], "path_hash": h})
    state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["video"] = path
    save()
    return json.dumps(out)


@tool
def atlas_render_camera_move(name: str, world: str, moves: Optional[list[dict]] = None, keys: Optional[list] = None,
                             fps: float = 24.0, seconds: float = 0.0, quality: str = "standard", fov: float = 0.0,
                             frame: str = "picture", ease: str = "linear", prompt: str = "", width: int = 1280,
                             height: int = 720, grow_world: bool = False, mesh_detail: int = 4, max_usd: float = 0.0,
                             confirm_cost: bool = False, dry_run: bool = False, add_to_canvas: bool = True,
                             timeout_s: int = 600, output_dir: str = "") -> str:
    """Render a camera MOVE through an Atlas world with Atlas-painted pictures: the high-quality moving shot.
    PAID (World Labs credits): always run it with dry_run=true first, tell the user the plan and the price,
    and run it for real only after they say yes (confirm_cost=true).

    Atlas has no video mode, so this paints keyframes along the path with atlasGenerate (a few per task, each
    task conditioned on the earlier ones and added to the world) and then renders every frame through the
    grown world in the bEpic viewer (the browser tab must be open). Use render_camera_move through the mesh
    or a splat as the free preview first; use this when the shot is the deliverable.

    Args:
        name: name of the shot (a-z, 0-9, _ -); the clip is output/previz/<world>_<name>.mp4.
        world: an Atlas world made by atlas_create_world (name, or the path of its folder / world.glb).
        moves: camera keys like render_camera_move: [{"t": s, "forward": m, "right": m, "up": m, "yaw": deg,
            "pitch": deg, "roll": deg, "fov": deg}] relative to the first picture's camera, or
            {"t", "position": [x,y,z], "look_at": [x,y,z]} absolute in `frame`. At least two, e.g.
            [{"t":0},{"t":3,"forward":1.2,"yaw":10}]. The camera must TRANSLATE (a pure turn gives no depth).
        keys: instead of moves: [{"t": s, "viewer": {"camera_world": [16], "model_world": [16], "fov", "aspect"}}]
            (the viewer's timeline camera) or [{"t", "matrix": 4x4}] - sampled densely from the timeline it
            renders exactly as shown.
        fps: frame rate of the clip.
        seconds: scale the path to this length (0 = as the keys say). At most 30.
        quality: "draft" (a painted view every 1 s, 8 per task), "standard" (every 0.5 s, 4 per task), "high"
            (every 0.25 s, 4 per task). More views = more tasks = more cost = more faithful.
        fov: vertical field of view in degrees (0 = the first picture's, or per key).
        frame: how absolute position / look_at are read: "picture" (metres from the picture camera = the mesh
            frame) or "world" (the Atlas world's own coordinates).
        ease: "linear", "smooth" (eases in and out) or "hold".
        prompt: scene description; default the world's own.
        width, height: size of the clip (the painted views are 1280x720).
        grow_world: false (default) works on a copy `<world>__<name>` and leaves the original untouched; true
            adds the painted views to the world itself.
        mesh_detail: depth-grid stride for the rebuilt mesh.
        max_usd: refuse a plan whose high estimate is above this (0 = WLT_ATLAS_MOVE_MAX_USD, default 5).
        confirm_cost: the user has seen the price and agreed.
        dry_run: spend nothing: return the plan (tasks, views, estimated credits / USD).
        add_to_canvas: the finished clip lands on the canvas as a loader node.
        timeout_s: how long to wait for each Atlas task (a timeout keeps the paid task; run again to resume).
        output_dir: where the world lives, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON. dry_run: {"status": "dry_run", "plan": {...tasks, keyframes, estimate...}}. Done: path (the clip),
        frames, fps, tasks_used, operations, estimate, world_dir, `measured` (numbers computed from the clip's
        pixels). Errors: confirm_cost (price shown, ask the user), over_cap, timeout (run again), balance.
    """
    return render_move(name, world, moves, keys, fps, seconds, quality, fov, frame, ease, prompt, width, height,
                       grow_world, mesh_detail, max_usd, confirm_cost, dry_run, add_to_canvas, timeout_s, output_dir)
