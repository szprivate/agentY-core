"""Edit or paint ONE picture with World Labs ``atlasChisel`` (masked edits and depth-guided generation).

``atlasChisel`` makes one 1280 x 720 picture per *target frame*. Each target frame is a pinhole camera
plus what is already known at that camera (all optional, but the same kinds on every frame):

* ``imageAsset`` + ``maskAsset``: a picture and a mask (Atlas: white = keep, black = generate) -> a masked
  edit, e.g. remove an object and continue the ground through its place;
* ``depth``: a depth map (linear-depth EXR as ``depthAsset``, or a log-depth PNG as ``logDepthAsset`` with
  ``zMin`` / ``zMax``) -> a picture built on that geometry, e.g. a Blender / Houdini blockout's depth pass.
  ``voxelSize`` (0 - 0.4) simplifies that geometry first, for a looser read of a rough blockout.

Request shape (the API changed in 2026-11; this module follows the current one)::

    POST /tasks:atlasChisel
    {targetFrames: [{camera, imageAsset?, maskAsset?, depth?}],   # was targetCameras + contextFrames
     contextFrames: [{imageAsset, camera}],                       # posed photos; was sourceFrames
     styleReferenceImageAsset?, prompt (required), enhancePrompt, voxelSize?, voxelGridOrientation?,
     modelParameters: {seed?, numSteps?}, returnDepth}

``returnDepth`` needs 2 - 256 targets, so this one-picture tool never asks for depth. Pictures and masks
are cover-resized and centre-cropped to 1280 x 720 (the model's only target size) before they are sent.

With ``world`` (an ``atlas_create_world`` world) the camera is given like ``atlas_render_view``'s, a frame of
the world can be edited by id, and the world's nearest views go along as ``contextFrames`` so the picture
stays the same place. Output: ``<world>/chisel/chisel_NNN.png`` (no world: ``<atlas root>/chisel/``) with an
``.agenty.json`` sidecar, published like any generated picture. Key ``WLT_API_KEY`` (never logged).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools.worldlabs_atlas import AtlasError

logger = logging.getLogger("agenty_core.atlas_chisel")

TASK = "atlasChisel"
MAX_CONTEXT = 4
MAX_VOXEL = 0.4
_DEFAULT_FOV = 50.0          # vertical degrees, for a picture with no world and no camera


# -- inputs -------------------------------------------------------------------


def _png_b64(img) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"base64": base64.b64encode(buf.getvalue()).decode("ascii")}


def _cover(img, size=wl._GRID):
    """Cover-resize and centre-crop to ``size`` (what Atlas does with photos), so picture and mask stay aligned."""
    from PIL import Image
    w, h = img.size
    tw, th = size
    if (w, h) == (tw, th):
        return img
    s = max(tw / w, th / h)
    nw, nh = max(tw, round(w * s)), max(th, round(h * s))
    img = img.resize((nw, nh), Image.LANCZOS if img.mode != "L" else Image.BILINEAR)
    x0, y0 = (nw - tw) // 2, (nh - th) // 2
    return img.crop((x0, y0, x0 + tw, y0 + th))


def _open_image(ref: str):
    from PIL import Image
    from agenty_core.tools.image_io import resolve_local_image
    path = resolve_local_image(str(ref).strip())
    if not path:
        raise AtlasError(f"image not found: {ref}", "args")
    return Image.open(path)


def picture_asset(ref: str) -> dict:
    """A TaskAsset for a picture on the 1280 x 720 grid (a URL is sent as given)."""
    ref = str(ref).strip()
    if ref.startswith(("http://", "https://")):
        return {"url": ref}
    img = _cover(_open_image(ref).convert("RGB"))
    return _png_b64(img)


def mask_asset(ref: str, white_is: str = "change") -> dict:
    """An 8-bit grey mask on the grid in Atlas's convention (white = keep). ``white_is="change"`` (the
    usual inpainting convention: paint what should change) is inverted here."""
    from PIL import ImageOps
    if white_is not in ("change", "keep"):
        raise AtlasError("mask_white_is is 'change' (white = make new) or 'keep' (white = keep)", "args")
    img = _open_image(ref)
    if img.mode in ("RGBA", "LA") and img.getextrema()[-1][0] < 255:
        img = img.getchannel("A")          # a painted alpha mask
    img = _cover(img.convert("L"))
    img = img.point(lambda v: 255 if v >= 128 else 0)
    if white_is == "change":
        img = ImageOps.invert(img)
    return _png_b64(img)


def depth_input(path: str, z_min: float = 0.0, z_max: float = 0.0) -> dict:
    """A depth buffer: a linear-depth EXR (``depthAsset``, planar depth in world units) or a log-depth PNG
    (``logDepthAsset`` + ``zMin`` / ``zMax``; white nearest, 0 = no surface)."""
    from agenty_core.tools.image_io import resolve_local_image
    p = resolve_local_image(str(path).strip()) or (Path(str(path).strip()) if Path(str(path).strip()).is_file() else None)
    if not p:
        raise AtlasError(f"depth map not found: {path}", "args")
    p = Path(p)
    ext = p.suffix.lower()
    if ext == ".exr":
        return {"depthAsset": {"assetId": wl._upload_file(str(p), "image/x-exr")}}
    if ext == ".png":
        if not (z_max and z_max > z_min > 0):
            raise AtlasError("a log-depth PNG needs z_min and z_max (the nearest and farthest depth, z_max > "
                             "z_min > 0); or give a linear-depth EXR", "args")
        return {"logDepthAsset": wl._image_input(str(p)), "zMin": float(z_min), "zMax": float(z_max)}
    raise AtlasError("the depth map must be a linear-depth .exr or a log-depth .png", "args")


def _is_api_camera(c) -> bool:
    return isinstance(c, dict) and isinstance(c.get("extrinsics"), dict) and isinstance(c.get("intrinsics"), dict)


def _on_grid(cam: dict) -> dict:
    """Atlas targets must be 1280 x 720: rescale another size's intrinsics (same lens, same pose)."""
    K = dict(cam["intrinsics"])
    gw, gh = wl._GRID
    if (int(K["width"]), int(K["height"])) != (gw, gh):
        sx, sy = gw / float(K["width"]), gh / float(K["height"])
        K = {"fx": K["fx"] * sx, "fy": K["fy"] * sy, "cx": K["cx"] * sx, "cy": K["cy"] * sy,
             "width": gw, "height": gh}
    return {"extrinsics": cam["extrinsics"], "intrinsics": K}


def default_camera(fov: float = 0.0) -> dict:
    """A camera at the origin looking down -Z (RUB), on the grid, with a vertical ``fov``."""
    gw, gh = wl._GRID
    fy = (gh / 2.0) / math.tan(math.radians(float(fov or _DEFAULT_FOV)) / 2.0)
    K = {"fx": fy, "fy": fy, "cx": gw / 2.0, "cy": gh / 2.0, "width": gw, "height": gh}
    return wl.camera_to_api([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0], K)


def context_photos(wd: Path, frames: list) -> list:
    """``contextFrames`` (posed RGB) for world frames: picture + camera, no depth."""
    out = []
    for f in frames:
        img = {"assetId": f["image_asset"]} if f.get("image_asset") else wl._image_input(str(wd / f["image"]))
        out.append({"imageAsset": img, "camera": f["camera"]})
    return out


def _next_index(d: Path) -> int:
    best = -1
    for f in d.glob("chisel_*.png"):
        m = re.match(r"chisel_(\d+)", f.stem)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


# -- the request --------------------------------------------------------------


def build_request(prompt: str, camera: dict, image: Optional[dict] = None, mask: Optional[dict] = None,
                  depth: Optional[dict] = None, context: Optional[list] = None, style: Optional[dict] = None,
                  voxel_size: float = 0.0, voxel_grid: str = "", seed: int = -1, num_steps: int = 0,
                  enhance_prompt: bool = True) -> dict:
    """The ``atlasChisel`` body for ONE target frame (validated the way the API does, before it is billed)."""
    if not str(prompt or "").strip():
        raise AtlasError("atlasChisel needs a prompt (describe the scene, or the edit)", "args")
    if mask is not None and image is None:
        raise AtlasError("a mask needs the picture it masks (image_path, or a world frame)", "args")
    if image is None and depth is None:
        raise AtlasError("give something to build on: a picture + mask to edit, or a depth map", "args")
    if voxel_size and image is not None and depth is None:
        raise AtlasError("voxel_size simplifies depth geometry; it needs a depth map", "args")
    tf = {"camera": _on_grid(camera)}
    if image is not None:
        tf["imageAsset"] = image
    if mask is not None:
        tf["maskAsset"] = mask
    if depth is not None:
        tf["depth"] = depth
    body = {"targetFrames": [tf], "contextFrames": list(context or []), "prompt": str(prompt).strip(),
            "enhancePrompt": bool(enhance_prompt), "returnDepth": False}
    if style is not None:
        body["styleReferenceImageAsset"] = style
    if voxel_size and float(voxel_size) > 0:
        v = float(voxel_size)
        if v > MAX_VOXEL:
            raise AtlasError(f"voxel_size {v} is above the recommended 0 - {MAX_VOXEL} (try 0.05 - 0.15 for a blockout)",
                             "args")
        body["voxelSize"] = v
        if voxel_grid:
            if voxel_grid not in ("world", "gauge"):
                raise AtlasError("voxel_grid is 'world' or 'gauge'", "args")
            body["voxelGridOrientation"] = voxel_grid
    mp = {}
    if seed is not None and int(seed) >= 0:
        mp["seed"] = int(seed)
    if num_steps and int(num_steps) > 0:
        if int(num_steps) > 100:
            raise AtlasError("num_steps is 1 - 100", "args")
        mp["numSteps"] = int(num_steps)
    if mp:
        body["modelParameters"] = mp
    return body


def finish_chisel(job: dict, op: dict) -> str:
    """Download the picture, write the sidecar, publish. Called by worldlabs_atlas._finish."""
    wd, op_id = Path(job["world_dir"]), op.get("id", "")
    resp = op.get("response") or {}
    frames = resp.get("frames") or []
    if not frames:
        raise AtlasError("the operation succeeded but returned no frame", "api", operation_id=op_id)
    fr = frames[0]
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"chisel_{_next_index(out_dir):03d}.png"
    wl._download(fr["imageAsset"], dest)
    cam = fr.get("camera") or job.get("camera")
    meta = {"task": TASK, "operation": op_id, "mode": job.get("mode"), "world": job.get("world"),
            "frame": job.get("frame"), "camera_world": cam, "prompt": job.get("prompt"),
            "prompt_used": resp.get("promptUsed"), "inputs": job.get("inputs"), "voxel_size": job.get("voxel_size"),
            "seed": job.get("seed")}
    wl._sidecar(dest, f"Atlas Chisel {job.get('mode', 'picture')} (atlasChisel, 1280x720)", meta)
    if job.get("world"):
        try:
            world = wl._load_world(wd)
            world.setdefault("chisel", []).append({"file": f"chisel/{dest.name}", "operation": op_id,
                                                   "mode": job.get("mode"), "frame": job.get("frame"),
                                                   "camera": cam, "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
            world.setdefault("operations", []).append({"id": op_id, "task": TASK,
                                                       "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
            wl._save_world(wd, world)
        except Exception:  # noqa: BLE001
            logger.warning("could not note the chisel picture in world.json", exc_info=True)
    wl._pending_path(wd, op_id).unlink(missing_ok=True)
    wl._publish(dest)
    return json.dumps({"status": "ok", "path": str(dest), "size": list(wl._image_size(dest)), "mode": job.get("mode"),
                       "world": job.get("world"), "camera_world": cam, "prompt_used": resp.get("promptUsed"),
                       "operation_id": op_id, "usage": wl._usage_of(op) or None,
                       "usage_note": "Atlas reports no per-task cost; credits (1,000 per US$) show on the platform's "
                                     "Usage page."})


@wl._guard
def _chisel(prompt, world, frame, camera, image_path, mask_path, mask_white_is, depth_path, z_min, z_max, fov,
            style_image, context, voxel_size, voxel_grid, seed, num_steps, enhance_prompt, timeout_s, output_dir):
    from agenty_core.tools import atlas_render as ar
    w, wd, fr = None, None, None
    if str(world or "").strip():
        wd = ar.resolve_world_dir(world, output_dir)
        w = wl._load_world(wd)
    if frame is not None and int(frame) >= 0:
        if w is None:
            raise AtlasError("frame picks a picture of a world; name the world too", "args")
        fr = next((f for f in w["frames"] if int(f["id"]) == int(frame)), None)
        if fr is None or not fr.get("camera"):
            raise AtlasError(f"the world has no posed frame {frame}", "args")
    # the camera
    if _is_api_camera(camera):
        cam = camera
    elif fr is not None and not camera:
        cam = fr["camera"]
    elif isinstance(camera, dict) and camera:
        if w is None:
            raise AtlasError("a relative / look-at camera is placed in a world; name the world, or give an Atlas "
                             "camera ({extrinsics, intrinsics})", "args")
        cam, _pose, _fov = ar.resolve_camera(w, camera, fov)
    elif w is not None:
        base = wl._first_posed(w)               # the world's first view, on the grid
        cam = wl.mesh_pose_to_camera(base, wl.grid_intrinsics(base, fov),
                                     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0, 0.0, 0.0])
    else:
        cam = default_camera(fov)
    # what is known at the target
    image = mask = depth = None
    inputs = {}
    if str(image_path or "").strip():
        image, inputs["image"] = picture_asset(image_path), str(image_path)
    elif fr is not None and str(mask_path or "").strip():
        image, inputs["image"] = picture_asset(str(wd / fr["image"])), f"frame {fr['id']}"
    if str(mask_path or "").strip():
        mask, inputs["mask"] = mask_asset(mask_path, mask_white_is), str(mask_path)
    if str(depth_path or "").strip():
        depth, inputs["depth"] = depth_input(depth_path, z_min, z_max), str(depth_path)
    style = None
    if str(style_image or "").strip():
        style, inputs["style"] = wl._image_input(style_image), str(style_image)
    ctx = []
    if w is not None and context is not False:
        if isinstance(context, list) and context:
            ids = {int(c) for c in context}
            chosen = [f for f in w["frames"] if f["id"] in ids and f.get("camera")]
        else:
            chosen = [f for f in ar._pick_context(w, cam, MAX_CONTEXT + 1) if fr is None or f["id"] != fr["id"]]
        ctx = context_photos(wd, chosen[:MAX_CONTEXT])
        inputs["context_frames"] = [f["id"] for f in chosen[:MAX_CONTEXT]]
    body = build_request(prompt, cam, image, mask, depth, ctx, style, voxel_size, voxel_grid, seed, num_steps,
                         enhance_prompt)
    mode = "edit" if mask is not None else ("depth" if depth is not None else "restyle")
    if wd is None:
        wd = wl._root_dir(output_dir) / "chisel"
        out_dir = wd
    else:
        out_dir = wd / "chisel"
    op = wl._request("POST", f"/tasks:{TASK}", json_body=body, idem=f"agenty-{uuid.uuid4().hex}")
    job = {"kind": "chisel", "task": TASK, "world": (w or {}).get("name"), "world_dir": str(wd), "out_dir": str(out_dir),
           "mode": mode, "frame": None if fr is None else fr["id"], "camera": body["targetFrames"][0]["camera"],
           "prompt": prompt, "inputs": inputs, "voxel_size": body.get("voxelSize"),
           "seed": (body.get("modelParameters") or {}).get("seed")}
    return wl._run(job, op, timeout_s)


@tool
def atlas_chisel(prompt: str, world: str = "", frame: int = -1, camera: Optional[dict] = None,
                 image_path: str = "", mask_path: str = "", mask_white_is: str = "change", depth_path: str = "",
                 z_min: float = 0.0, z_max: float = 0.0, fov: float = 0.0, style_image: str = "",
                 context: Optional[list] = None, voxel_size: float = 0.0, voxel_grid: str = "", seed: int = -1,
                 num_steps: int = 0, enhance_prompt: bool = True, timeout_s: int = 600, output_dir: str = "") -> str:
    """Make ONE 1280x720 picture with World Labs Atlas Chisel: a masked edit of a picture (remove or
    replace something, continue the scene through it), or a picture built on a depth map (a 3D
    blockout's depth pass from Blender / Houdini / the previz scene), optionally in the look of a style
    image. One call is one billed Atlas task (the user's World Labs credits); do not loop it.

    Modes (pick by what you give):
      * edit: image_path (or world + frame) + mask_path + a prompt saying what to do there;
      * depth-guided: depth_path (+ camera / fov matching the render that made it) + a scene prompt;
        voxel_size 0.05 - 0.15 loosens a rough blockout;
      * both: a picture + mask + depth.

    Args:
        prompt: required. The scene, or the edit ("remove the fence, continue the meadow").
        world: an atlas_create_world world (name or folder); then its nearest views go along as
            context so the picture stays the same place, and the result is noted in world.json.
        frame: edit this frame id of the world (its picture and camera); needs mask_path.
        camera: where the picture is taken from. In a world: like atlas_render_view (relative move,
            position + look_at, matrix, viewer camera). Anywhere: an Atlas camera
            {"extrinsics": {"position", "quaternion", "coordinateSystem"}, "intrinsics": {fx, fy, cx,
            cy, width, height}} (another size is rescaled to 1280x720). Default: the world's first view,
            else the origin looking down -Z.
        image_path: the picture to edit (cover-resized / centre-cropped to 1280x720).
        mask_path: the mask (same crop). By default white = change it, black = keep
            (mask_white_is="keep" for Atlas's own convention). An alpha mask works too.
        depth_path: planar depth at the camera: a linear-depth .exr (world units), or a log-depth .png
            (white nearest, 0 = sky / no surface) with z_min and z_max.
        fov: vertical field of view in degrees for a default or relative camera.
        style_image: a picture whose look to match (no camera or geometry taken from it).
        context: world frame ids to show as context (default: the nearest few; [] none with world).
        voxel_size: 0 - 0.4, simplify the depth geometry first (0 = keep it as is).
        voxel_grid: "world" (default) or "gauge" voxel grid orientation (with voxel_size).
        seed: >= 0 for a fixed seed (Atlas default 42). num_steps: 1 - 100 (0 = model default).
        enhance_prompt: let Atlas expand the prompt (default true); false uses your words unchanged.
        timeout_s: how long to wait; on timeout the error carries operation_id for atlas_fetch.
        output_dir: where Atlas worlds live, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON {"status": "ok", "path", "mode", "camera_world", "prompt_used", "operation_id"} (the picture
        is published to the canvas and the viewer) or {"error", "error_kind"} like the other Atlas tools.
    """
    ctx = False if (isinstance(context, list) and not context) else context   # [] = no context
    return _chisel(prompt, world, frame, camera, image_path, mask_path, mask_white_is, depth_path, z_min, z_max,
                   fov, style_image, ctx, voxel_size, voxel_grid, seed, num_steps, enhance_prompt, timeout_s,
                   output_dir)
