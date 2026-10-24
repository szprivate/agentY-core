"""A cheap sanity check of a camera path against the scene it flies through, before a whole move is rendered.

A camera that starts under the floor, outside the room or looking at nothing renders a few hundred
frames of garbage - and the vision check afterwards may or may not notice. This reads the scene's own
geometry (a Gaussian-splat .ply or a .glb: a subsample of the points), takes a handful of camera samples
along the path, and measures: is the camera inside the scene's bounds, above the floor, not buried in
the geometry, and does it see a meaningful share of the scene. The numbers come back with the verdict.

Splat .ply files are in the 3DGS convention (Y down); the viewer shows them flipped about X, so scene
coordinates are (x, -y, -z) of the file.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agenty_core.camera_check")

MAX_POINTS = 200_000
SPLAT_EXT = {".ply", ".splat", ".spz", ".ksplat"}
_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
              "char": "i1", "int8": "i1", "ushort": "<u2", "uint16": "<u2", "short": "<i2", "int16": "<i2",
              "uint": "<u4", "uint32": "<u4", "int": "<i4", "int32": "<i4"}
_CACHE: dict = {}


def read_ply_header(path) -> Optional[dict]:
    """{"count", "dtype", "offset", "props", "comments"} of a binary little-endian PLY, else None."""
    import numpy as np
    with open(path, "rb") as fh:
        head = fh.read(65536)
    end = head.find(b"end_header")
    if end < 0:
        return None
    nl = head.find(b"\n", end)
    lines = head[:end].decode("ascii", "replace").splitlines()
    if not any(l.startswith("format binary_little_endian") for l in lines):
        return None
    props, count, comments, in_vertex = [], 0, [], False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "comment":
            comments.append(line[8:])
        elif parts[:2] == ["element", "vertex"]:
            count, in_vertex = int(parts[2]), True
        elif parts[0] == "element":
            in_vertex = False
        elif parts[0] == "property" and in_vertex:
            if parts[1] not in _PLY_TYPES:
                return None
            props.append((parts[-1], _PLY_TYPES[parts[1]]))
    if not {"x", "y", "z"} <= {p for p, _ in props}:
        return None
    return {"count": count, "dtype": np.dtype(props), "offset": nl + 1, "props": [p for p, _ in props],
            "comments": comments}


def is_splat_ply(path) -> bool:
    try:
        h = read_ply_header(path)
    except Exception:  # noqa: BLE001
        return False
    return bool(h) and {"f_dc_0", "opacity"} <= set(h["props"])


def is_splat_path(path: str) -> bool:
    ext = os.path.splitext(str(path))[1].lower()
    if ext == ".ply":
        return is_splat_ply(path)
    return ext in SPLAT_EXT


def splat_points(path, max_points: int = MAX_POINTS):
    """(points in scene coordinates, total gaussians) - an even subsample, read through a memmap."""
    import numpy as np
    h = read_ply_header(path)
    if h is None:
        raise ValueError(f"{path}: not a binary little-endian PLY with x/y/z")
    mm = np.memmap(path, dtype=h["dtype"], mode="r", offset=h["offset"], shape=(h["count"],))
    n = h["count"]
    idx = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype("int64")
    sub = mm[idx]
    pts = np.stack([sub["x"], sub["y"], sub["z"]], -1).astype("float64")
    pts[:, 1] *= -1.0                                       # 3DGS Y-down file -> the viewer's Y-up scene
    pts[:, 2] *= -1.0
    return pts, n


def mesh_points(path, max_points: int = MAX_POINTS):
    from agenty_core.tools.previz_shot import glb_vertices
    import numpy as np
    v = glb_vertices(path)
    if v is None:
        raise ValueError(f"cannot read the vertices of {path}")
    n = len(v)
    if n > max_points:
        v = v[np.linspace(0, n - 1, max_points).astype("int64")]
    return v, n


def _item_matrix_apply(pts, item: dict):
    """Apply a scene item's scale, rotation (Euler XYZ degrees) and position to model-space points."""
    import numpy as np
    from agenty_core.tools.previz_shot import euler_to_matrix
    s = np.array(item.get("scale") or [1, 1, 1], dtype="float64")
    r = np.array(euler_to_matrix(item.get("rotation") or [0, 0, 0]), dtype="float64")
    t = np.array(item.get("position") or [0, 0, 0], dtype="float64")
    return (pts * s) @ r.T + t


def scene_points(scene: dict):
    """(points in scene coordinates, total count, kind) of the scene's main model item, or None."""
    for it in scene.get("items") or []:
        if it.get("id") != "scene" or it.get("kind") != "model":
            continue
        path = ((it.get("src") or {}).get("path")) or ""
        if not path or not os.path.isfile(path):
            return None
        key = (path, os.path.getmtime(path), repr(it.get("position")), repr(it.get("rotation")), repr(it.get("scale")))
        if key in _CACHE:
            return _CACHE[key]
        try:
            if is_splat_path(path):
                pts, n = splat_points(path)
                kind = "splat"
            elif path.lower().endswith(".glb"):
                pts, n = mesh_points(path)
                kind = "mesh"
            else:
                return None
            res = (_item_matrix_apply(pts, it), n, kind)
        except Exception:  # noqa: BLE001
            logger.exception("could not read the geometry of %s", path)
            return None
        _CACHE.clear()
        _CACHE[key] = res
        return res
    return None


def scene_bounds(pts, kind: str = "splat", total: int = 0, ground_y: Optional[float] = None) -> dict:
    """Robust bounds of a point cloud: the 1st..99th percentile box, the full box, a floor and a ceiling."""
    import numpy as np
    lo, hi = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    ext = hi - lo
    ys = pts[:, 1]
    floor = _floor_y(ys)
    if ground_y is not None:
        floor = float(ground_y)
    return {"kind": kind, "points_sampled": int(len(pts)), "points_total": int(total or len(pts)),
            "min": _r3(lo), "max": _r3(hi), "size": _r3(ext),
            "full_min": _r3(pts.min(axis=0)), "full_max": _r3(pts.max(axis=0)),
            "floor_y": round(float(floor), 3), "top_y": round(float(np.percentile(ys, 99.5)), 3),
            "note": "min/max are the 1st-99th percentile box of the geometry (scene coordinates, Y up)"}


def _r3(v):
    return [round(float(x), 3) for x in v]


def _floor_y(ys) -> float:
    """The height of the lowest dense slab (the floor): the first 4 cm bin holding >= 15% of the fullest bin."""
    import numpy as np
    lo, hi = float(np.percentile(ys, 0.5)), float(np.percentile(ys, 99.5))
    if hi - lo < 0.2:
        return lo
    bins = max(8, int((hi - lo) / 0.04))
    counts, edges = np.histogram(ys, bins=bins, range=(lo, hi))
    ok = np.nonzero(counts >= 0.15 * counts.max())[0]
    return float(edges[ok[0]]) if len(ok) else lo


# -- the camera samples ------------------------------------------------------------------------


def path_samples(keys, n: int = 9) -> list:
    """[{frame, position, forward}] - every key plus `n` points spread along the path (linear between keys)."""
    from agenty_core.tools.previz_shot import _col, euler_to_matrix
    pts = []
    for f, pos, rot in keys:
        fw = [-x for x in _col(euler_to_matrix(rot), 2)]
        pts.append((float(f), [float(x) for x in pos], fw))
    out = [{"frame": int(f), "position": p, "forward": fw, "key": True} for f, p, fw in pts]
    if len(pts) >= 2 and pts[-1][0] > pts[0][0]:
        f0, f1 = pts[0][0], pts[-1][0]
        for i in range(1, n + 1):
            f = f0 + (f1 - f0) * i / (n + 1)
            for (fa, pa, wa), (fb, pb, wb) in zip(pts, pts[1:]):
                if fa <= f <= fb and fb > fa:
                    u = (f - fa) / (fb - fa)
                    p = [pa[k] + (pb[k] - pa[k]) * u for k in range(3)]
                    w = [wa[k] + (wb[k] - wa[k]) * u for k in range(3)]
                    ln = math.sqrt(sum(x * x for x in w)) or 1.0
                    out.append({"frame": int(round(f)), "position": p, "forward": [x / ln for x in w], "key": False})
                    break
    return sorted(out, key=lambda s: s["frame"])


def frustum_share(pts, pos, fwd, fov_v: float, aspect: float, near: float = 0.05, far: float = 60.0) -> dict:
    """How many of the points a camera at `pos` looking along `fwd` sees (inside its frustum, near..far)."""
    import numpy as np
    d = pts - np.asarray(pos)
    f = np.asarray(fwd, dtype="float64")
    f /= np.linalg.norm(f) or 1.0
    up0 = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.99 else np.array([0.0, 0.0, -1.0])
    right = np.cross(f, up0)
    right /= np.linalg.norm(right) or 1.0
    up = np.cross(right, f)
    z = d @ f
    th = math.tan(math.radians(fov_v) / 2.0)
    ok = (z > near) & (z < far) & (np.abs(d @ up) <= z * th) & (np.abs(d @ right) <= z * th * aspect)
    dist = np.linalg.norm(d, axis=1)
    return {"in_view": int(ok.sum()), "share": float(ok.mean()), "within_12cm": int((dist < 0.12).sum()),
            "median_distance": round(float(np.median(dist[ok])), 2) if ok.any() else None}


def check_path(samples, pts, bounds: dict, fov_v: float, aspect: float, strict: bool = True) -> dict:
    """Test camera samples against the scene. Returns {"problems": [...], "warnings": [...], "measured": [...]}.

    strict (splats, Atlas rooms): outside the bounds or looking at nothing is an error. Otherwise (a single-picture
    mesh that ends at the picture's depth) those are warnings; below the floor is an error either way.
    """
    lo, hi = bounds["min"], bounds["max"]
    ext = [h - l for l, h in zip(lo, hi)]
    margin = [max(0.35, 0.12 * e) if strict else max(1.0, 0.5 * e) for e in ext]
    floor = bounds["floor_y"]
    problems, warnings, measured = [], [], []
    for s in samples:
        p, fw = s["position"], s["forward"]
        tag = f"frame {s['frame']}" + (" (key)" if s.get("key") else "")
        fs = frustum_share(pts, p, fw, fov_v, aspect)
        measured.append({"frame": s["frame"], "position": _r3(p), "forward": _r3(fw), "gaussians_in_view": fs["in_view"],
                         "share_in_view": round(fs["share"], 4), "median_distance_in_view_m": fs["median_distance"],
                         "points_within_12cm": fs["within_12cm"]})
        if p[1] < floor - 0.05:
            problems.append(f"{tag}: camera at y={p[1]:.2f} is below the floor (scene floor is about y={floor:.2f})")
        out_axes = [a for a, i in (("x", 0), ("y", 1), ("z", 2))
                    if p[i] < lo[i] - margin[i] or p[i] > hi[i] + margin[i]]
        if out_axes:
            msg = (f"{tag}: camera at [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}] is outside the scene "
                   f"(bounds {_r3(lo)} .. {_r3(hi)}, off on {'/'.join(out_axes)})")
            (problems if strict else warnings).append(msg)
        if fs["share"] < 0.005 and fs["in_view"] < 600:
            msg = (f"{tag}: the camera sees almost nothing ({fs['in_view']} of {len(pts)} sampled points in view, "
                   f"{fs['share'] * 100:.2f}%)")
            (problems if strict else warnings).append(msg)
        elif fs["share"] < 0.02:
            warnings.append(f"{tag}: little in view ({fs['share'] * 100:.1f}% of the scene)")
        if fs["within_12cm"] > max(40, 0.002 * len(pts)):
            warnings.append(f"{tag}: the camera is inside / touching dense geometry ({fs['within_12cm']} points within 12 cm)")
    return {"problems": problems, "warnings": warnings, "measured": measured}


def apply_frame(point, picture_pose: Optional[dict], frame: str):
    """A point given in `frame` -> scene coordinates. 'picture': the picture camera is the origin, -Z forward
    (R * p + t of its pose in the scene); 'world': already scene coordinates."""
    if frame == "picture" and picture_pose:
        R, t = picture_pose["R"], picture_pose["t"]
        return [sum(R[i][k] * point[k] for k in range(3)) + t[i] for i in range(3)]
    return [float(x) for x in point]


def invert_frame(point, picture_pose: dict):
    """Scene coordinates -> the picture camera's frame."""
    R, t = picture_pose["R"], picture_pose["t"]
    d = [point[i] - t[i] for i in range(3)]
    return [sum(R[k][i] * d[k] for k in range(3)) for i in range(3)]


def read_first_camera(model_path: str) -> Optional[dict]:
    """The picture camera of a trained splat, from the job.json Brush training wrote next to it:
    {"R", "t", "fov"|None, "source"} in the splat's own (Atlas) frame, else None."""
    import json
    folder = Path(model_path).parent
    for cand in (folder / "job.json",):
        try:
            job = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        m = job.get("first_camera")
        if m:
            return {"R": [row[:3] for row in m[:3]], "t": [m[0][3], m[1][3], m[2][3]], "fov": None,
                    "source": f"{cand} (first_camera)"}
    return None
