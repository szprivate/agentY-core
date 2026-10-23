"""Measure a reference photo: metric 3D positions, camera height, field of view (MoGe through ComfyUI).

When the agent models a building (or any object) from ONE photo it has to guess dimensions and the camera.
MoGe-2 estimates, from a single image, a metric point map (x, y, z in metres per pixel), a validity mask and
the camera intrinsics. This tool runs it once through the local ComfyUI (nodes ``LoadMoGeModel``,
``MoGeInference``, ``MoGePointMapToMesh`` -> ``SaveGLB`` for the per-pixel points, ``MoGeRender`` (mask) ->
``SaveImage`` for the validity mask, ``MoGeGeometryToFOV`` -> ``PreviewAny`` for the field of view), caches
the result next to the picture, and answers point and pair queries from the cache.

Frame of every number returned: camera at the origin, X right, Y up, Z away from the camera (into the
scene), levelled so the fitted ground plane is horizontal at ``y = -camera_height``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import tempfile
import time
import uuid
from typing import Optional

from agenty_core._compat import tool

logger = logging.getLogger("agenty_core.reference_measure")

CACHE_VERSION = 1
DECIMATION = 2            # the point map is stored at every 2nd pixel: plenty for picking points, 4x lighter
_MODEL = "moge_2_vitl_normal_fp16.safetensors"


# -- pure maths (tested on synthetic arrays) ----------------------------------


def parse_region(hint: str, default=(0.0, 0.75, 1.0, 1.0)) -> tuple:
    """A ground hint -> (x0, y0, x1, y1) fractions of the image, y from the top.
    "bottom 25%", "bottom quarter", "lower third", "rect 0.1,0.7,0.9,1.0"; anything else -> the bottom quarter."""
    h = (hint or "").strip().lower()
    if not h:
        return default
    m = re.search(r"rect\w*\s*[:=]?\s*\(?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", h)
    if m:
        x0, y0, x1, y1 = (float(g) for g in m.groups())
        if x1 > x0 and y1 > y0:
            return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))
    frac = None
    m = re.search(r"([\d.]+)\s*%", h)
    if m:
        frac = float(m.group(1)) / 100.0
    elif "quarter" in h:
        frac = 0.25
    elif "third" in h:
        frac = 1 / 3
    elif "half" in h:
        frac = 0.5
    if frac is not None and 0 < frac <= 1:
        if "top" in h or "upper" in h:
            return (0.0, 0.0, 1.0, frac)
        return (0.0, 1.0 - frac, 1.0, 1.0)
    return default


def fit_plane(points, tol: float = 0.08, iters: int = 300, min_up: float = 0.6, seed: int = 1):
    """RANSAC + least-squares plane through (N, 3) points in a Y-up frame. Returns (normal, d, inlier_share)
    with the normal pointing up and the plane ``normal . p + d = 0``; None when no near-horizontal plane
    (normal.y >= min_up) holds at least 15 % of the points."""
    import numpy as np
    pts = np.asarray(points, dtype="float64").reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    if len(pts) > 20000:
        pts = pts[rng.choice(len(pts), 20000, replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n = n / ln
        if n[1] < 0:
            n = -n
        if n[1] < min_up:
            continue
        cnt = int((np.abs(pts @ n - a @ n) < tol).sum())
        if cnt > best_n:
            best, best_n = (n, -float(a @ n)), cnt
    if best is None or best_n < 0.15 * len(pts):
        return None
    n, d = best
    for _ in range(3):
        inl = pts[np.abs(pts @ n + d) < tol]
        if len(inl) < 30:
            break
        ctr = inl.mean(axis=0)
        n = np.linalg.eigh(np.cov((inl - ctr).T))[1][:, 0]
        if n[1] < 0:
            n = -n
        d = -float(ctr @ n)
    share = float((np.abs(pts @ n + d) < tol).mean())
    return n, d, share


def level_matrix(normal):
    """Rotation matrix taking `normal` to +Y (smallest rotation)."""
    import numpy as np
    n = np.asarray(normal, dtype="float64")
    n = n / np.linalg.norm(n)
    y = np.array([0.0, 1.0, 0.0])
    axis = np.cross(n, y)
    sn, cs = float(np.linalg.norm(axis)), float(n @ y)
    if sn < 1e-9:
        return np.eye(3)
    axis = axis / sn
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sn * k + (1 - cs) * (k @ k)


def sample_point(points, valid, u: float, v: float, radius_px: int = 3):
    """Robust sample of an (H, W, 3) point grid at fractions u (from the left) and v (from the top):
    the per-axis median over the (2r+1)^2 window, skipping invalid pixels.
    Returns (xyz ndarray or None, number of valid pixels used)."""
    import numpy as np
    H, W = points.shape[:2]
    cx = min(max(int(round(u * (W - 1))), 0), W - 1)
    cy = min(max(int(round(v * (H - 1))), 0), H - 1)
    y0, y1 = max(cy - radius_px, 0), min(cy + radius_px + 1, H)
    x0, x1 = max(cx - radius_px, 0), min(cx + radius_px + 1, W)
    win = points[y0:y1, x0:x1].reshape(-1, 3)
    ok = valid[y0:y1, x0:x1].reshape(-1) & np.isfinite(win).all(axis=1)
    win = win[ok]
    if len(win) == 0:
        return None, 0
    return np.median(win, axis=0), int(len(win))


def pair_metrics(a, b) -> dict:
    """Metric distance, vertical difference (b - a, along Y) and ground-plane distance between two Y-up points."""
    dx, dy, dz = float(b[0] - a[0]), float(b[1] - a[1]), float(b[2] - a[2])
    return {"distance_m": math.sqrt(dx * dx + dy * dy + dz * dz),
            "vertical_diff_m": dy, "horizontal_m": math.hypot(dx, dz)}


def pair_yaw_deg(a, b) -> float:
    """Yaw of the segment a-b in the ground plane relative to the camera's image plane, in a levelled frame
    (X right, Z away): atan2(dz, dx) with the left end first. Positive: the RIGHT end is farther away."""
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    return math.degrees(math.atan2(float(right[2] - left[2]), float(right[0] - left[0])))


SCALE_RANGE = (0.5, 2.0)


def apply_anchor(result: dict, anchor: Optional[dict]) -> dict:
    """Rescale a finished ``analyse`` result so that the anchor pair measures ``anchor["meters"]``.
    Every length in the result (point positions, distances, heights, pair numbers, camera height) is replaced by
    its scaled value; the unscaled numbers stay under ``raw``. Without an anchor nothing changes except a
    top-level warning that the metres are unscaled."""
    warnings = result.setdefault("warnings", [])
    if not anchor:
        warnings.append("absolute metres are UNSCALED: MoGe is often 10-40 % off at 20-30 m. Pass anchor={'pair': "
                        "[nameA, nameB], 'meters': known_size} (a railing ~1.0 m, a door ~2.0 m) before they become "
                        "build constants. Ratios, angles and yaw are unaffected.")
        result["scale"] = None
        return result
    try:
        pair = [str(n) for n in anchor["pair"]]
        meters = float(anchor["meters"])
        if len(pair) != 2 or meters <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        warnings.append("anchor ignored: it must be {'pair': [nameA, nameB], 'meters': positive number}; metres are unscaled")
        result["scale"] = None
        return result
    ref = next((q for q in result["pairs"] if [q["a"], q["b"]] == pair or [q["b"], q["a"]] == pair), None)
    if ref is None:
        # the anchor pair was not asked for as a pair: measure it from the points
        pts = {q["name"]: q for q in result["points"] if q.get("valid")}
        if pair[0] in pts and pair[1] in pts:
            ref = {"distance_m": pair_metrics(pts[pair[0]]["position_m"], pts[pair[1]]["position_m"])["distance_m"]}
    if ref is None or not ref.get("distance_m"):
        warnings.append("anchor ignored: no valid measurement for the pair %s; metres are unscaled" % pair)
        result["scale"] = None
        return result
    k = meters / ref["distance_m"]
    result["scale"] = round(k, 4)
    result["anchor"] = {"pair": pair, "meters": meters, "measured_raw_m": round(ref["distance_m"], 3)}
    if not SCALE_RANGE[0] <= k <= SCALE_RANGE[1]:
        warnings.append("scale factor %.2f is outside %.1f-%.1f: the anchor points are probably on the wrong spots "
                        "(background behind a thin railing?) or the anchor size is wrong; do not trust the scaled metres"
                        % (k, *SCALE_RANGE))
    raw: dict = {}
    if result.get("camera_height_m") is not None:
        raw["camera_height_m"] = result["camera_height_m"]
        result["camera_height_m"] = round(result["camera_height_m"] * k, 2)
    raw_pts = []
    for q in result["points"]:
        if not q.get("valid"):
            continue
        raw_pts.append({"name": q["name"], "position_m": q["position_m"], "distance_m": q["distance_m"],
                        **({"height_above_ground_m": q["height_above_ground_m"]} if "height_above_ground_m" in q else {})})
        q["position_m"] = [round(c * k, 3) for c in q["position_m"]]
        q["distance_m"] = round(q["distance_m"] * k, 2)
        if "height_above_ground_m" in q:
            q["height_above_ground_m"] = round(q["height_above_ground_m"] * k, 3)
    raw_pairs = []
    for q in result["pairs"]:
        if "error" in q:
            continue
        raw_pairs.append({kk: q[kk] for kk in ("a", "b", "distance_m", "vertical_diff_m", "horizontal_m")})
        for kk in ("distance_m", "vertical_diff_m", "horizontal_m"):
            q[kk] = round(q[kk] * k, 3)
    raw["points"], raw["pairs"] = raw_pts, raw_pairs
    result["raw"] = raw
    result["frame"] += "; lengths scaled by the anchor (x%.3f), unscaled numbers are under 'raw'" % k
    return result


def horizon_from_normal(normal_cam, f_px: float, width: int, height: int) -> dict:
    """Where the ground plane's horizon falls in the image. `normal_cam`: the ground normal in the camera's
    own Y-up frame (X right, Y up, Z forward). Returns y_from_top at the image centre (fraction and px) and
    the roll of the line in degrees; y is None when the plane is (nearly) edge-on to the view."""
    nx, ny, nz = (float(c) for c in normal_cam)
    # ray through pixel (u, v): ((u - W/2)/f, -(v - H/2)/f, 1); the horizon is n . ray = 0
    if abs(ny) < 1e-6:
        return {"y_from_top": None}

    def y_at(u):
        return height / 2 + (nx * (u - width / 2) + nz * f_px) / ny

    yc = y_at(width / 2)
    roll = math.degrees(math.atan2(y_at(width) - y_at(0), width))
    return {"y_from_top": yc / height, "y_px": yc, "roll_deg": roll}


# -- ComfyUI side -------------------------------------------------------------


def _client():
    from agenty_core.utils.comfyui_client import get_client
    return get_client()


def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _cache_path(image_path: str, digest: str) -> str:
    folder = os.path.dirname(os.path.abspath(image_path))
    stem = os.path.splitext(os.path.basename(image_path))[0]
    name = f"{stem}.moge-{digest}.npz"
    if os.access(folder, os.W_OK):
        return os.path.join(folder, name)
    fallback = os.path.join(tempfile.gettempdir(), "agenty_moge")
    os.makedirs(fallback, exist_ok=True)
    return os.path.join(fallback, name)


def build_graph(input_name: str, prefix: str) -> dict:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {"class_type": "LoadMoGeModel", "inputs": {"model_name": _MODEL}},
        "3": {"class_type": "MoGeInference", "inputs": {
            "moge_model": ["2", 0], "image": ["1", 0], "resolution_level": 9, "fov_x_degrees": 0.0,
            "batch_size": 1, "force_projection": True, "apply_mask": False, "refine_steps": 3}},
        "4": {"class_type": "MoGePointMapToMesh", "inputs": {
            "moge_geometry": ["3", 0], "batch_index": 0, "decimation": DECIMATION,
            "discontinuity_threshold": 0.0, "texture": False}},
        "5": {"class_type": "SaveGLB", "inputs": {"mesh": ["4", 0], "filename_prefix": prefix + "_pts"}},
        "6": {"class_type": "MoGeRender", "inputs": {"moge_geometry": ["3", 0], "output": "mask"}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": prefix + "_mask"}},
        "8": {"class_type": "MoGeGeometryToFOV", "inputs": {"moge_geometry": ["3", 0], "axis": "vertical", "unit": "degrees"}},
        "9": {"class_type": "MoGeGeometryToFOV", "inputs": {"moge_geometry": ["3", 0], "axis": "horizontal", "unit": "degrees"}},
        "10": {"class_type": "PreviewAny", "inputs": {"source": ["8", 0]}},
        "11": {"class_type": "PreviewAny", "inputs": {"source": ["9", 0]}},
    }


def _run_graph(graph: dict, timeout_s: int = 900) -> dict:
    client = _client()
    res = client.post("/prompt", json_data={"prompt": graph, "client_id": uuid.uuid4().hex})
    pid = res.get("prompt_id") if isinstance(res, dict) else None
    if not pid:
        raise RuntimeError(f"ComfyUI did not accept the MoGe graph: {res!r}"[:600])
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hist = client.get(f"/history/{pid}")
        entry = hist.get(pid) if isinstance(hist, dict) else None
        if entry:
            st = (entry.get("status") or {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                detail = (msgs[-1][1].get("exception_message") if msgs else "") or "execution error"
                raise RuntimeError(f"MoGe run failed in ComfyUI: {detail}"[:600])
            if entry.get("outputs") is not None and st.get("completed", True):
                return entry["outputs"]
        time.sleep(1.0)
    raise RuntimeError("MoGe run timed out")


def _fetch(file_info: dict) -> bytes:
    resp = _client().get("/view", params={"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""),
                                          "type": file_info.get("type", "output")}, raw=True)
    return resp.content


def _first_file(outputs: dict, node: str) -> dict:
    out = outputs.get(node) or {}
    for key in ("images", "3d", "files", "result"):
        items = out.get(key)
        if items:
            it = items[0]
            if isinstance(it, dict) and "filename" in it:
                return it
    raise RuntimeError(f"ComfyUI produced no file from node {node}: {out!r}"[:400])


def _text(outputs: dict, node: str) -> float:
    txt = (outputs.get(node) or {}).get("text") or []
    return float(str(txt[0]).strip())


def run_moge(image_path: str) -> dict:
    """Run MoGe on the picture; returns the arrays converted to X right, Y up, Z forward."""
    import io
    import numpy as np
    from PIL import Image
    from agenty_core.tools.previz_shot import glb_vertices

    digest = _file_hash(image_path)
    with Image.open(image_path) as im:
        W, H = im.size
    ext = os.path.splitext(image_path)[1] or ".png"
    input_name = f"agenty_moge_{digest}{ext}"
    with open(image_path, "rb") as fh:
        up = _client().post("/upload/image", data={"overwrite": "true", "type": "input"},
                            files={"image": (input_name, fh.read())})
    if isinstance(up, dict) and up.get("name"):
        input_name = up["name"] if not up.get("subfolder") else f"{up['subfolder']}/{up['name']}"
    outputs = _run_graph(build_graph(input_name, f"agenty/moge/{digest}"))

    glb = _first_file(outputs, "5")
    tmp = os.path.join(tempfile.gettempdir(), f"agenty_moge_{digest}.glb")
    with open(tmp, "wb") as fh:
        fh.write(_fetch(glb))
    try:
        verts = glb_vertices(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    gh, gw = -(-H // DECIMATION), -(-W // DECIMATION)
    if verts is None or len(verts) != gh * gw:
        raise RuntimeError(f"MoGe point map has {0 if verts is None else len(verts)} vertices, expected "
                           f"{gh * gw} ({gw}x{gh}): some pixels were dropped, cannot map them back")
    points = verts.reshape(gh, gw, 3)
    points = np.stack([points[..., 0], points[..., 1], -points[..., 2]], axis=-1).astype("float32")  # glTF -> Z forward

    mask_img = Image.open(io.BytesIO(_fetch(_first_file(outputs, "7")))).convert("L")
    mask = (np.asarray(mask_img.resize((W, H), Image.NEAREST))[::DECIMATION, ::DECIMATION] > 127)[:gh, :gw]
    return {"points": points, "mask": mask, "width": W, "height": H,
            "fov_y": _text(outputs, "10"), "fov_x": _text(outputs, "11")}


def load_or_run(image_path: str) -> tuple:
    import numpy as np
    digest = _file_hash(image_path)
    cache = _cache_path(image_path, digest)
    if os.path.exists(cache):
        try:
            z = np.load(cache)
            if int(z["version"]) == CACHE_VERSION:
                return ({"points": z["points"], "mask": z["mask"], "width": int(z["width"]), "height": int(z["height"]),
                         "fov_x": float(z["fov_x"]), "fov_y": float(z["fov_y"])}, True)
        except Exception:  # noqa: BLE001
            logger.warning("bad MoGe cache %s, recomputing", cache)
    data = run_moge(image_path)
    try:
        np.savez_compressed(cache, version=CACHE_VERSION, **data)
    except OSError:
        logger.warning("could not write MoGe cache %s", cache)
    return data, False


def analyse(data: dict, points: Optional[list] = None, pairs: Optional[list] = None, ground_hint: str = "",
            anchor: Optional[dict] = None) -> dict:
    """The tool's maths on a loaded MoGe result (also what the tests drive with synthetic arrays)."""
    import numpy as np
    pts = np.asarray(data["points"], dtype="float64")
    gh, gw = pts.shape[:2]
    W, H = int(data["width"]), int(data["height"])
    fov_x, fov_y = float(data["fov_x"]), float(data["fov_y"])
    valid = np.asarray(data["mask"], dtype=bool) & np.isfinite(pts).all(axis=-1) & (pts[..., 2] > 0)
    f_px = (H / 2) / math.tan(math.radians(fov_y) / 2)

    result: dict = {"image_size": {"width": W, "height": H}, "fov_vertical_deg": round(fov_y, 2),
                    "fov_horizontal_deg": round(fov_x, 2), "focal_px": round(f_px, 1),
                    "frame": "camera at origin, X right, Y up, Z away from camera; metres"}

    # ground plane: lowest valid region (or the hinted one)
    x0, y0, x1, y1 = parse_region(ground_hint)
    ys = (np.arange(gh) + 0.5) / gh
    xs = (np.arange(gw) + 0.5) / gw
    region = (ys[:, None] >= y0) & (ys[:, None] <= y1) & (xs[None, :] >= x0) & (xs[None, :] <= x1) & valid
    fit = fit_plane(pts[region]) if region.sum() >= 50 else None
    rot = np.eye(3)
    if fit is not None:
        normal, d, share = fit
        rot = level_matrix(normal)
        cam_h = abs(d)
        result["camera_height_m"] = round(cam_h, 2)
        result["ground"] = {"region_used": [round(v, 3) for v in (x0, y0, x1, y1)], "inlier_share": round(share, 2),
                            "camera_tilt_deg": round(math.degrees(math.acos(min(1.0, float(normal[1])))), 1)}
        hor = horizon_from_normal(normal, f_px, W, H)
        result["horizon"] = {k: (round(v, 3) if v is not None else None) for k, v in hor.items()}
    else:
        result["camera_height_m"] = None
        result["ground"] = {"error": "no near-horizontal ground plane found in the region; pass ground_hint "
                                     "(e.g. 'bottom 30%' or 'rect 0.1,0.7,0.9,1.0') over visible ground"}
        result["horizon"] = {"y_from_top": None}

    radius = max(2, int(round(3 / DECIMATION)) + 1)
    out_points, by_name = [], {}
    for p in points or []:
        name = str(p.get("name", f"p{len(out_points)}"))
        px = p.get("px") or p.get("xy")
        entry: dict = {"name": name, "px": px}
        if not px or len(px) != 2:
            entry.update(valid=False, reason="px must be [x_frac, y_from_top_frac]")
        else:
            xyz, n = sample_point(pts, valid, float(px[0]), float(px[1]), radius)
            if xyz is None:
                entry.update(valid=False, reason="no valid depth near this pixel (sky, mask or edge)")
            else:
                q = rot @ xyz
                entry.update(valid=True, position_m=[round(float(c), 3) for c in q],
                             distance_m=round(float(np.linalg.norm(xyz)), 2), valid_pixels=n)
                if result["camera_height_m"] is not None:
                    entry["height_above_ground_m"] = round(float(q[1] + result["camera_height_m"]), 3)
                by_name[name] = q
        out_points.append(entry)
    result["points"] = out_points

    out_pairs = []
    for pair in pairs or []:
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        if a in by_name and b in by_name:
            m = pair_metrics(by_name[a], by_name[b])
            out_pairs.append({"a": a, "b": b, **{k: round(v, 3) for k, v in m.items()},
                              "yaw_deg": round(pair_yaw_deg(by_name[a], by_name[b]), 1),
                              "nearer": a if by_name[a][2] <= by_name[b][2] else b})
        else:
            missing = [n for n in (a, b) if n not in by_name]
            out_pairs.append({"a": a, "b": b, "error": f"no valid measurement for {', '.join(missing)}"})
    result["pairs"] = out_pairs
    return apply_anchor(result, anchor)


@tool
def measure_reference(image_path: str, points: Optional[list] = None, pairs: Optional[list] = None,
                      ground_hint: str = "", anchor: Optional[dict] = None) -> str:
    """Measure a reference photo with MoGe (monocular metric geometry): camera field of view, camera height
    above the ground, the horizon, and the 3D position of points you pick in the picture.

    Use it before modelling a building or object from ONE photo, instead of guessing sizes. Workflow: make a
    gridded copy of the photo (contact sheet / grid overlay), read the points you need off the grid as
    fractions, call this with them, then use the returned numbers as the CONSTANTS of the build script
    (deck height, house width, roof ridge height, and the camera's fov / height for the matching render
    camera). MoGe runs once per picture (a few seconds on the GPU, needs the MoGe model in ComfyUI) and is
    cached next to the image, so later calls with other points are instant.

    Metric scale is approximate (MoGe is often 10-40 % off at 20-30 m, it ran 1.3x too large for a house 29 m
    away): pass ``anchor`` (a pair of points whose real size you know: railing ~1.0 m, door ~2.0 m, a step
    ~0.18 m) and every length comes back scaled; without it a top-level warning says the metres are unscaled.
    Ratios, angles and yaw are far more reliable than absolute metres. Points on thin
    structures (railings, stilts) can land on the background: pick the thickest part and check
    ``valid_pixels`` and the distance. If ``camera_height_m`` is null, name visible ground in ``ground_hint``.

    Args:
        image_path: the photo (any size; portrait or landscape).
        points: [{"name": "stilt_base", "px": [x_frac, y_from_top_frac]}, ...], both fractions 0..1 of the
            image (x from the left, y from the TOP).
        pairs: [["stilt_base", "deck_edge"], ...] names from ``points``; per pair the metric distance,
            the vertical difference (second minus first, along up) and the horizontal distance. A deck height
            is the pair [ground_contact, deck_line].
        anchor: {"pair": [nameA, nameB], "meters": 1.0}: the pair must be points you gave. scale = meters /
            measured distance of the pair. Every point position, distance, height, pair length and the camera
            height are then SCALED in the result (the unscaled numbers stay under ``raw``), with ``scale`` and
            ``anchor`` reported. A scale outside 0.5-2.0 raises a warning (anchor on the wrong spot).
        ground_hint: where visible ground is, to fit the ground plane: "bottom 25%" (the default), "bottom
            40%", or "rect x0,y0,x1,y1" in fractions. Use it when something other than ground fills the bottom.

    Returns:
        JSON: image_size, fov_vertical_deg / fov_horizontal_deg, focal_px, camera_height_m, horizon
        (y_from_top fraction at the image centre, roll_deg), per point position_m [x, y, z] in a
        ground-aligned frame (camera at the origin, Y up, ground at y = -camera_height_m, Z away from the
        camera), distance_m, height_above_ground_m or valid=false with a reason, and per pair the numbers:
        distance_m, vertical_diff_m, horizontal_m, ``yaw_deg`` and ``nearer``. warnings, scale, anchor, raw.

    Yaw (per pair): the angle of the segment a-b in the ground plane relative to the camera's image plane,
    atan2(dz, dx) in the levelled frame (X right, Z away) with the left end first. POSITIVE yaw: the RIGHT end is
    farther from the camera (a facade seen from the front-left recedes to the right); NEGATIVE: the left end is
    farther; 0: parallel to the image plane (seen straight on). ``nearer`` is the name of the end closer to the
    camera. Yaw does not depend on scale. Reproduce it in the model by rotating the model or moving the camera,
    and pass the same number as ``yaw_pairs`` to dcc_silhouette_check.
    """
    try:
        if not image_path or not os.path.isfile(image_path):
            return json.dumps({"status": "error", "error": f"image not found: {image_path}"})
        data, cached = load_or_run(image_path)
        result = analyse(data, points, pairs, ground_hint, anchor)
        result.update(status="ok", cached=cached)
        return json.dumps(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("measure_reference failed")
        msg = str(exc)
        if "moge" in msg.lower() or "400" in msg:
            msg += " (needs the MoGe nodes and models/geometry_estimation/" + _MODEL + " in ComfyUI)"
        return json.dumps({"status": "error", "error": msg[:800]})
