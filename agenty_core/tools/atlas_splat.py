"""A Gaussian splat of an Atlas world, built by World Labs from the world's posed views (``posedImages2Splats``).

Until this task existed agentY trained splats locally (``train_splat``: Brush, 30k steps, the quality stood or fell
with how well the generated views agreed). ``posedImages2Splats`` takes posed RGBD frames - picture, camera and a
linear-depth EXR, exactly what an Atlas world keeps for every view - reconstructs a draft splat, then refines it
by rendering new views, cleaning them up and training on them. The result is an SPZ.

Docs (2026-10-30): https://atlas-beta.worldlabs.ai/docs/posed-images-to-splats and .../splat-best-practices.

* ``POST /tasks:posedImages2Splats`` ``{posedFrames: [{imageAsset, camera, depth: {depthAsset}, maskAsset?}],
  prompt?, enhancePrompt, refine, targetSplatCounts?}``. At least 2 frames, all one image size; 8-16
  well-spread overlapping views suit a room. A refined run costs 5,000 credits (US$ 5), a draft (``refine:
  false``) 500 (US$ 0.50); failed runs are free. 10-20 minutes: submitted without ``?wait``, polled.
* Done: ``response.refinedSplats`` {splat count: TaskAsset} (refined) or ``response.draftSplat`` (draft).
* Cameras are the world's own: camera-to-world, RUB (x right, y up, the camera looks down -Z), the same frame
  the world.json frames are stored in, so a world frame goes in as it is. A pose from somewhere else (the
  viewer, the mesh frame) is turned into this frame by ``atlas_render`` / ``mesh_pose_to_camera`` first.
* The splat comes back in the same world frame (measured on the promo's interior2 world, 2026-10-30: SPZ v2,
  2,000,000 splats, SH degree 0, Y up, it lines up with the input views rendered through their cameras; ~17 min). The bEpic viewer (Spark) turns every splat 180 degrees about X
  (it expects the 3DGS / OpenCV Y-down convention), so the PLY written next to the SPZ is turned that way
  (``splat_train.flip_x``) and the viewer shows it upright - as it does a ``train_splat`` PLY.

Generated views do not always agree (a table moves, a coat appears on one view only); the splat then ghosts.
``view_consistency`` checks every view against the views kept before it (in world order: the source picture
first) by reprojecting depth: where one view puts a surface IN FRONT of what the other sees, the two disagree.
Views that disagree with a kept view over more than ``max_conflict`` of their shared pixels are left out before
anything is uploaded.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools.worldlabs_atlas import AtlasError

logger = logging.getLogger("agenty_core.atlas_splat")

TASK = "posedImages2Splats"
CREDITS = {"refined": 5000, "draft": 500}      # per successful run (docs, 2026-10-30); failed runs are free
USD_PER_CREDIT = 0.001                          # 1,000 credits per US$


# -- which views agree --------------------------------------------------------------------------------------------


def _view_samples(wd: Path, fr: dict, stride: int, conf_min: float) -> dict:
    """World-space points (one per ``stride`` pixels) of a frame's depth, plus what is needed to project into it."""
    import numpy as np
    cam = wl.normalize_camera(fr["camera"])
    depth = wl._decode_depth(wd / fr["depth"], fr.get("depth_log") or {})
    conf = wl._read_png_gray(wd / fr["confidence"]) if fr.get("confidence") else None
    if conf is not None and conf.shape != depth.shape:
        conf = None
    H, W = depth.shape
    K = cam["K"]
    ys, xs = np.mgrid[stride // 2:H:stride, stride // 2:W:stride]
    z = depth[ys, xs].astype("float64")
    ok = np.isfinite(z) & (z > 1e-6)
    if conf is not None:
        ok &= conf[ys, xs] >= conf_min
    u, v = (xs + 0.5) * K["width"] / W, (ys + 0.5) * K["height"] / H
    p_cam = np.stack([(u - K["cx"]) / K["fx"] * z, -(v - K["cy"]) / K["fy"] * z, -z], -1)[ok]
    R, t = np.array(cam["R"]), np.array(cam["t"])
    return {"id": fr["id"], "R": R, "t": t, "K": K, "depth": depth, "conf": conf, "points": p_cam @ R.T + t}


def _project(a: dict, b: dict, tol: float, conf_min: float) -> tuple[int, int]:
    """Project view a's surface into view b: (pixels both see, pixels where a's surface lies in front of b's)."""
    import numpy as np
    pc = (a["points"] - b["t"]) @ b["R"]                       # world -> b's camera (RUB)
    z = -pc[:, 2]
    front = z > 0.05
    zs = np.where(front, z, 1.0)
    K = b["K"]
    H, W = b["depth"].shape
    x = ((pc[:, 0] / zs * K["fx"] + K["cx"]) * W / K["width"]).astype(int)
    y = ((-pc[:, 1] / zs * K["fy"] + K["cy"]) * H / K["height"]).astype(int)
    m = front & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not m.any():
        return 0, 0
    db = b["depth"][y[m], x[m]]
    ok = np.isfinite(db) & (db > 1e-6)
    if b["conf"] is not None:
        ok &= b["conf"][y[m], x[m]] >= conf_min
    rel = (z[m][ok] - db[ok]) / db[ok]
    return int(ok.sum()), int((rel < -tol).sum())


def view_consistency(wd: Path, frames: list, max_conflict: float = 0.05, tol: float = 0.15, stride: int = 8,
                     conf_min: float = 0.2, min_shared: float = 0.1) -> dict:
    """Which of ``frames`` (world.json records with camera + depth) agree with each other.

    Frames are taken in order; each is compared with every frame kept before it with which it shares at least
    ``min_shared`` of their sampled pixels (both directions together). The conflict of a pair is the share of those
    pixels where one view puts a surface more than ``tol`` (relative depth) in front of what the other sees.
    A frame whose worst conflict with a kept frame is above ``max_conflict`` is excluded.

    Returns ``{"kept": [ids], "excluded": [{"id", "conflict", "with"}], "views": {id: {"worst", "with", "compared"}}}``.
    """
    samples = [_view_samples(wd, f, stride, conf_min) for f in frames]
    kept, excluded, views = [], [], {}
    for s in samples:
        worst, worst_with, compared = 0.0, None, 0
        for k in kept:
            n1, c1 = _project(s, k, tol, conf_min)
            n2, c2 = _project(k, s, tol, conf_min)
            if n1 + n2 < min_shared * (len(s["points"]) + len(k["points"])) or n1 + n2 == 0:
                continue
            compared += 1
            c = (c1 + c2) / (n1 + n2)
            if c > worst:
                worst, worst_with = c, k["id"]
        views[s["id"]] = {"worst": round(worst, 4), "with": worst_with, "compared": compared}
        if worst > max_conflict:
            excluded.append({"id": s["id"], "conflict": round(worst, 4), "with": worst_with})
        else:
            kept.append(s)
    return {"kept": [s["id"] for s in kept], "excluded": excluded, "views": views,
            "rule": f"left out: a surface more than {int(tol * 100)}% in front of what a kept view sees, on more than "
                    f"{int(max_conflict * 100)}% of the pixels they share"}


# -- the request ----------------------------------------------------------------------------------------------------


def _ids(spec) -> list[int]:
    """'0-24,35,42-45' / [0, 1, 2] -> sorted ids."""
    if spec in (None, "", []):
        return []
    if isinstance(spec, (list, tuple)):
        return sorted({int(v) for v in spec})
    out = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)


def _depth_asset(wd: Path, fr: dict) -> dict:
    """The frame's depth as the API wants it: a linear-depth EXR (stored asset id, else the local file uploaded)."""
    if fr.get("depth_log"):
        raise AtlasError(f"frame {fr['id']} has a log-depth PNG; posedImages2Splats takes linear-depth EXRs only", "args")
    if fr.get("depth_asset"):
        return {"depthAsset": {"assetId": fr["depth_asset"]}}
    path = wd / fr["depth"]
    return {"depthAsset": {"assetId": wl._upload_file(str(path), "image/x-exr")}}


def _image_asset(wd: Path, fr: dict) -> dict:
    if fr.get("image_asset"):
        return {"assetId": fr["image_asset"]}
    return {"assetId": wl._upload_file(str(wd / fr["image"]), "image/png")}


def _mask_for(masks: str, wd: Path, fid: int) -> Optional[Path]:
    if not masks:
        return None
    folder = Path(masks) if Path(masks).is_absolute() else wd / masks
    for name in (f"mask_{fid:03d}.png", f"mask_{fid}.png", f"{fid:03d}.png"):
        if (folder / name).exists():
            return folder / name
    return None


def posed_frames(wd: Path, frames: list, masks: str = "") -> list:
    """``posedFrames`` for world frames: stored asset ids where the world has them, local files uploaded otherwise.
    The camera is the world.json camera unchanged (camera-to-world, RUB, the world's frame)."""
    out = []
    for fr in frames:
        if not fr.get("camera"):
            raise AtlasError(f"frame {fr['id']} has no camera", "args")
        if not fr.get("depth"):
            raise AtlasError(f"frame {fr['id']} has no depth map (posedImages2Splats needs depth on every frame)", "args")
        item = {"imageAsset": _image_asset(wd, fr), "camera": wl._canon_camera(fr["camera"]),
                "depth": _depth_asset(wd, fr)}
        mask = _mask_for(masks, wd, int(fr["id"]))
        if mask:
            item["maskAsset"] = {"assetId": wl._upload_file(str(mask), "image/png")}
        out.append(item)
    return out


def estimate(refine: bool) -> dict:
    credits = CREDITS["refined" if refine else "draft"]
    return {"credits": credits, "usd": round(credits * USD_PER_CREDIT, 2), "min_usd": credits * USD_PER_CREDIT,
            "max_usd": credits * USD_PER_CREDIT,
            "basis": "World Labs docs 2026-10-30: refined 5,000 credits, draft 500; failed runs are free"}


def _check_sizes(wd: Path, frames: list) -> None:
    sizes = {}
    for fr in frames:
        sizes.setdefault(tuple(wl._image_size(wd / fr["image"])), []).append(fr["id"])
    if len(sizes) > 1:
        detail = "; ".join(f"{w}x{h}: {ids[:8]}" for (w, h), ids in sizes.items())
        raise AtlasError(f"all frames must share one image size for posedImages2Splats ({detail}); leave the odd "
                         "ones out with exclude", "args")


# -- finishing ------------------------------------------------------------------------------------------------------


def _splat_asset(resp: dict) -> tuple[Optional[str], Optional[dict]]:
    """(splat count, TaskAsset) of the largest refined level, else the draft."""
    refined = resp.get("refinedSplats") or {}
    if refined:
        def _n(k):
            try:
                return int(k)
            except (TypeError, ValueError):
                return -1
        key = max(refined, key=_n)
        return str(key), refined[key]
    if resp.get("draftSplat"):
        return "draft", resp["draftSplat"]
    return None, None


def first_camera_viewer(cam: dict) -> list:
    """A world camera as the 4x4 camera-to-world the viewer's splat camera takes. The viewer turns the (Y-down)
    PLY back upright, so the splat shows in the world frame and the camera stays as the world has it."""
    n = wl.normalize_camera(cam)
    R, t = n["R"], n["t"]
    return [R[0] + [t[0]], R[1] + [t[1]], R[2] + [t[2]], [0, 0, 0, 1]]


def finish_splat(job: dict, op: dict) -> str:
    """Download the SPZ, write a viewer-axes PLY next to it, record it in world.json, open it. Called by
    worldlabs_atlas._finish (so atlas_fetch collects a job that outlived the call)."""
    from agenty_core.tools import splat_formats as sf
    from agenty_core.tools import splat_train as st
    wd, op_id = Path(job["world_dir"]), op.get("id", "")
    resp = op.get("response") or {}
    level, asset = _splat_asset(resp)
    if not asset:
        raise AtlasError("the operation succeeded but named no splat", "api", operation_id=op_id)
    sd = wd / "splats"
    sd.mkdir(parents=True, exist_ok=True)
    name = job["name"]
    spz = wl._download(asset, sd / f"{name}.spz")
    meta = {"world": job.get("world"), "operation": op_id, "task": TASK, "refined": job.get("refine", True),
            "splat_level": level, "frames": job.get("frames"), "excluded": job.get("excluded"),
            "estimate": job.get("estimate"), "frame": "Atlas world frame, RUB (Y up)"}
    wl._sidecar(spz, f"Gaussian splat (World Labs {TASK})", meta)
    out = {"status": "ok", "world": job.get("world"), "spz": str(spz), "splat_level": level, "operation_id": op_id,
           "frames_used": job.get("frames"), "excluded": job.get("excluded"), "estimate": job.get("estimate"),
           "usage": wl._usage_of(op) or None}
    ply = lite = None
    try:
        props, a = sf.read_spz(spz)
        out["splats"] = int(len(a))
        flipped = st.flip_x(props, a)
        ply = sd / f"{name}.ply"
        sf.write_ply(ply, props, flipped, (f"World Labs {TASK} via agentY", "Axes: Y down (3DGS convention)"))
        if any(p.startswith("f_rest_") for p in props):
            lp, la = st.color_only(props, flipped)
            lite = sd / f"{name}_lite.ply"
            sf.write_ply(lite, lp, la, (f"World Labs {TASK} via agentY", "Colour only (SH degree 0)"))
        else:                                        # World Labs' splats are colour only already (SH degree 0)
            lite = ply
        wl._sidecar(ply, f"Gaussian splat (World Labs {TASK}), viewer axes", {**meta, "frame": "Y down (3DGS)"})
        out.update(ply=str(ply), ply_lite=str(lite))
    except Exception as exc:  # noqa: BLE001  (an SPZ the reader does not know - v4 - still is the result)
        out["ply_error"] = f"{type(exc).__name__}: {exc}"
    try:
        world = wl._load_world(wd)
        world.setdefault("splats", []).append({"spz": f"splats/{spz.name}", "ply": f"splats/{ply.name}" if ply else None,
                                               "operation": op_id, "level": level, "frames": job.get("frames"),
                                               "excluded": job.get("excluded"), "refined": job.get("refine", True),
                                               "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
        world.setdefault("operations", []).append({"id": op_id, "task": TASK, "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
        wl._save_world(wd, world)
    except AtlasError:
        pass
    wl._pending_path(wd, op_id).unlink(missing_ok=True)
    wl._publish(ply or spz)
    if job.get("open_in_viewer", True) and (lite or ply):
        out["viewer"] = st.open_splat_in_viewer(lite or ply, {"first_camera": job.get("first_camera")})
    out["next"] = ("Fly through it in the viewer or render_camera_move(model=<ply>); the PLY is in viewer axes (Y "
                   "down), the SPZ in the Atlas world frame (Y up). splats2Mesh / atlas_splat_to_mesh makes a mesh.")
    return json.dumps(out)


# -- the tool -------------------------------------------------------------------------------------------------------


def _views_to_splat(world, frames, exclude, check_consistency, max_conflict, refine, prompt, masks, name, dry_run,
                    confirm_cost, open_in_viewer, timeout_s, output_dir):
    from agenty_core import partner_credits as pc
    from agenty_core.tools.atlas_render import resolve_world_dir
    wd = resolve_world_dir(world, output_dir)
    w = wl._load_world(wd)
    want, drop = _ids(frames), set(_ids(exclude))
    chosen = [f for f in w["frames"] if (not want or int(f["id"]) in want) and int(f["id"]) not in drop]
    missing = sorted(set(want) - {int(f["id"]) for f in w["frames"]})
    if missing:
        raise AtlasError(f"the world has no frames {missing}", "args")
    chosen = [f for f in chosen if f.get("camera") and f.get("depth")]
    if len(chosen) < 2:
        raise AtlasError("posedImages2Splats needs at least 2 frames with a camera and a depth map", "args")
    _check_sizes(wd, chosen)
    check = None
    if check_consistency:
        check = view_consistency(wd, chosen, max_conflict=float(max_conflict))
        keep = set(check["kept"])
        chosen = [f for f in chosen if int(f["id"]) in keep]
        if len(chosen) < 2:
            raise AtlasError("fewer than 2 views agree with each other; nothing to splat", "args", consistency=check)
    est = estimate(bool(refine))
    ids = [int(f["id"]) for f in chosen]
    plan = {"world": w["name"], "frames": ids, "frame_count": len(ids), "refine": bool(refine), "estimate": est,
            "consistency": check, "masks": masks or None,
            "advice": ("8-16 well-spread, overlapping views suit a single room (World Labs splat best practices); "
                       f"this sends {len(ids)}.")}
    if dry_run:
        return json.dumps({"status": "dry_run", "spent": 0, "plan": plan,
                           "note": "Nothing was sent. Run again without dry_run, with confirm_cost=True once the user "
                                   f"agrees to about US$ {est['usd']:.2f}."})
    if not (confirm_cost or pc.confirm_given()):
        raise pc.CostGateError(
            "confirm_cost",
            f"Not run: a {'refined' if refine else 'draft'} splat from {len(ids)} views costs {est['credits']:,} World "
            f"Labs credits (about US$ {est['usd']:.2f}; failed runs are free). Tell the user and ask; only when they "
            "say yes, run again with confirm_cost=True.", {"min_usd": est["usd"], "max_usd": est["usd"], "nodes": []},
            None)
    if not wl._api_key():
        raise AtlasError("WLT_API_KEY is not set (put the World Labs key in .env).", "auth")
    stem = wl._slug(name) or f"{w['name']}_posed_splat"
    body = {"posedFrames": posed_frames(wd, chosen, masks), "refine": bool(refine)}
    if str(prompt or "").strip():
        body["prompt"] = str(prompt).strip()
    op = wl._request("POST", f"/tasks:{TASK}", json_body=body, idem=f"agenty-{uuid.uuid4().hex}")
    job = {"kind": "posed_splat", "task": TASK, "world": w["name"], "world_dir": str(wd), "name": stem,
           "frames": ids, "excluded": (check or {}).get("excluded"), "refine": bool(refine), "estimate": est,
           "open_in_viewer": bool(open_in_viewer), "first_camera": first_camera_viewer(chosen[0]["camera"]),
           "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return wl._run(job, op, timeout_s)


@tool
def atlas_views_to_splat(world: str, frames: str = "", exclude: str = "", check_consistency: bool = True,
                         max_conflict: float = 0.05, refine: bool = True, prompt: str = "", masks: str = "",
                         name: str = "", dry_run: bool = False, confirm_cost: bool = False,
                         open_in_viewer: bool = True, timeout_s: int = 1800, output_dir: str = "") -> str:
    """Have World Labs build a high-quality Gaussian splat from an Atlas world's posed views (posedImages2Splats).

    Prefer this over ``train_splat`` (local Brush): World Labs reconstructs the splat from the views' depth and
    refines it by rendering and cleaning up new views. Paid: a refined splat is 5,000 credits (US$ 5), a draft
    (``refine=false``) 500 (US$ 0.50); failed runs are free. 10-20 minutes. Run ``atlas_capture_around`` first for
    a whole room (8-16 well-spread, overlapping views is World Labs' starting point for one room).

    Views that disagree with each other (a generated view that moved a table or invented a coat) make the
    splat ghost, so each view is checked against the ones kept before it by reprojecting depth, and the ones
    that disagree are left out before upload (``check_consistency``). The result reports which and why.

    Args:
        world: an Atlas world (name, or its folder / world.glb).
        frames: which views to use, e.g. "0-24,35-37" (default: all with a depth map).
        exclude: views to leave out, e.g. "33,34".
        check_consistency: leave out views that disagree with the views kept before them.
        max_conflict: share of shared pixels on which two views may disagree before the later one is left out.
        refine: true = refined splat (US$ 5); false = draft (US$ 0.50, a fraction of the time).
        prompt: optional scene description (World Labs captions the images when empty).
        masks: optional folder (absolute, or relative to the world) of mask_NNN.png alpha masks per frame id
            (white = keep, black = leave out, e.g. the view through a window).
        name: file name of the splat (default <world>_posed_splat).
        dry_run: spend nothing: return the views it would send, the consistency check and the price.
        confirm_cost: the user has seen the price and agreed.
        open_in_viewer: open the result in the bEpic viewer's 3D view.
        timeout_s: how long to wait; on timeout the error carries operation_id for atlas_fetch.
        output_dir: where the world lives, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON: spz (Atlas world frame, Y up), ply + ply_lite (viewer axes, Y down), splats, frames_used,
        excluded, estimate, operation_id - or {"status": "dry_run", "plan"} - or {"error", "error_kind"}
        (confirm_cost: ask the user; timeout: atlas_fetch operation_id=...).
    """
    from agenty_core import partner_credits as pc
    try:
        return _views_to_splat(world, frames, exclude, check_consistency, max_conflict, refine, prompt, masks, name,
                               dry_run, confirm_cost, open_in_viewer, timeout_s, output_dir)
    except AtlasError as exc:
        return exc.render()
    except pc.CostGateError as exc:
        return json.dumps(exc.payload())
    except Exception as exc:  # noqa: BLE001
        logger.exception("atlas_views_to_splat failed")
        return wl._err(f"{type(exc).__name__}: {exc}", "api")
