"""Gaussian-splat file formats: SPZ <-> PLY, in pure numpy (no extra dependency).

* **SPZ** (Niantic, https://github.com/nianticlabs/spz) is a gzip stream: a 16-byte header, then one array per
  attribute - positions (24-bit fixed point), alphas, colours, log-scales, rotations, SH coefficients. Versions
  2 and 3 are read and version 3 is written. Version 4 (zstd streams) is refused with a pointer to the PLY export.
  The reader/writer here do NOT change the axes: what is in the file comes out in the PLY. Marble's SPZ are in
  its "marble_raw_opencv" frame (x right, y DOWN, z forward) - the same Y-down frame as a 3DGS PLY and as the
  splats ``train_splat`` saves, which the bEpic viewer (Spark) turns upright with its 180 degree turn about X.
  (Niantic's own CLI flips y and z when it converts to or from PLY because it keeps SPZ in RUB; Marble's files
  are documented as raw OpenCV, so nothing is flipped here.)
* **PLY**: the usual 3DGS vertex layout (x y z, f_dc_*, f_rest_*, opacity, scale_*, rot_*), binary little-endian,
  any scalar property type on the way in, float32 on the way out.

``apply_metric`` applies Marble's ``semantics_metadata`` (docs.worldlabs.ai/api/rendering-spz): centres times
``metric_scale_factor`` with ``ground_plane_offset`` taken off Y, log-scales plus ``ln(metric_scale_factor)``.
"""

from __future__ import annotations

import gzip
import math
import struct
from pathlib import Path
from typing import Optional

SPZ_MAGIC = 0x5053474E            # "NGSP"
_COLOR_SCALE = 0.15
_SQRT_HALF = math.sqrt(0.5)
_SH_DIM = {0: 0, 1: 3, 2: 8, 3: 15, 4: 24}
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "<i2", "int16": "<i2",
              "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
              "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


class SplatFormatError(Exception):
    pass


# -- SPZ -----------------------------------------------------------------------------


def spz_header(path) -> dict:
    """{version, count, sh_degree, fractional_bits, flags} from the first 16 bytes (no full decode)."""
    with gzip.open(path, "rb") as fh:
        head = fh.read(16)
    return _parse_header(head, path)


def _parse_header(head: bytes, path="") -> dict:
    if len(head) < 16:
        raise SplatFormatError(f"{path}: too short to be an SPZ file")
    magic, version, count, sh_deg, frac, flags, _ = struct.unpack("<IIIBBBB", head[:16])
    if magic != SPZ_MAGIC:
        raise SplatFormatError(f"{path}: not an SPZ file (bad magic)")
    return {"version": version, "count": count, "sh_degree": sh_deg, "fractional_bits": frac, "flags": flags}


def read_spz(path):
    """-> (props, float32 [N, P]) in 3DGS PLY terms: x y z, f_dc_0..2, f_rest_*, opacity, scale_0..2, rot_0..3 (w x y z)."""
    import numpy as np
    p = Path(path)
    raw = p.read_bytes()
    if raw[:4] == struct.pack("<I", SPZ_MAGIC):
        v = struct.unpack("<I", raw[4:8])[0]
        raise SplatFormatError(f"{p.name} is SPZ version {v} (compressed streams); this reader takes versions 2-3. "
                               "Ask marble_export for the PLY instead.")
    try:
        data = gzip.decompress(raw)
    except OSError as exc:
        raise SplatFormatError(f"{p.name}: not a gzip-compressed SPZ file ({exc})") from exc
    h = _parse_header(data[:16], p)
    ver, n, sh_deg, frac = h["version"], h["count"], h["sh_degree"], h["fractional_bits"]
    if ver not in (1, 2, 3):
        raise SplatFormatError(f"{p.name}: SPZ version {ver} is not supported (2 and 3 are)")
    if sh_deg not in _SH_DIM:
        raise SplatFormatError(f"{p.name}: unsupported SH degree {sh_deg}")
    dim = _SH_DIM[sh_deg]
    rot_bytes = 4 if ver >= 3 else 3
    sizes = [n * 9, n, n * 3, n * 3, n * rot_bytes, n * dim * 3]
    if len(data) < 16 + sum(sizes):
        raise SplatFormatError(f"{p.name}: truncated ({len(data)} bytes, need {16 + sum(sizes)})")
    off = 16
    parts = []
    for s in sizes:
        parts.append(np.frombuffer(data, dtype=np.uint8, count=s, offset=off))
        off += s
    pos_b, alpha_b, col_b, scale_b, rot_b, sh_b = parts

    pb = pos_b.reshape(n, 3, 3).astype(np.int64)
    fixed = pb[..., 0] | (pb[..., 1] << 8) | (pb[..., 2] << 16)
    fixed = np.where(fixed & 0x800000, fixed - (1 << 24), fixed)
    xyz = (fixed / float(1 << frac)).astype(np.float32)

    a = alpha_b.astype(np.float64) / 255.0
    a = np.clip(a, 1e-6, 1 - 1e-6)
    opacity = np.log(a / (1 - a)).astype(np.float32)
    f_dc = ((col_b.reshape(n, 3).astype(np.float32) / 255.0 - 0.5) / _COLOR_SCALE).astype(np.float32)
    scales = (scale_b.reshape(n, 3).astype(np.float32) / 16.0 - 10.0).astype(np.float32)

    if ver >= 3:
        comp = rot_b.reshape(n, 4).astype(np.uint32)
        comp = comp[:, 0] | (comp[:, 1] << 8) | (comp[:, 2] << 16) | (comp[:, 3] << 24)
        largest = (comp >> 30).astype(np.int64)
        q = np.zeros((n, 4), dtype=np.float64)
        rem = comp.copy()
        sumsq = np.zeros(n)
        for i in (3, 2, 1, 0):
            mask = largest != i
            mag = (rem & 511).astype(np.float64)
            neg = ((rem >> 9) & 1).astype(bool)
            val = _SQRT_HALF * mag / 511.0
            val = np.where(neg, -val, val)
            q[:, i] = np.where(mask, val, 0.0)
            sumsq += np.where(mask, val * val, 0.0)
            rem = np.where(mask, rem >> 10, rem)
        big = np.sqrt(np.maximum(0.0, 1.0 - sumsq))
        q[np.arange(n), largest] = big
    else:
        r = rot_b.reshape(n, 3).astype(np.float64) / 127.5 - 1.0
        w = np.sqrt(np.maximum(0.0, 1.0 - (r * r).sum(1)))
        q = np.concatenate([r, w[:, None]], axis=1)          # x y z w
    rot = np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], axis=1).astype(np.float32)   # w x y z

    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    cols = [xyz, f_dc]
    if dim:
        sh = (sh_b.reshape(n, dim, 3).astype(np.float32) - 128.0) / 128.0     # file: per coefficient r,g,b
        cols.append(sh.transpose(0, 2, 1).reshape(n, 3 * dim))                # PLY: all R, then all G, then all B
        props += [f"f_rest_{i}" for i in range(3 * dim)]
    props += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    cols += [opacity[:, None], scales, rot]
    return props, np.concatenate(cols, axis=1).astype(np.float32)


def write_spz(path, props: list, a, fractional_bits: int = 12) -> dict:
    """A 3DGS-PLY-shaped array -> an SPZ v3 file. Returns its header. Needs x y z, f_dc_*, opacity, scale_*, rot_*."""
    import numpy as np
    ix = {p: i for i, p in enumerate(props)}
    need = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3"]
    miss = [k for k in need if k not in ix]
    if miss:
        raise SplatFormatError(f"not a 3DGS splat: missing {', '.join(miss)}")
    a = np.asarray(a, dtype=np.float64)
    n = len(a)
    k = sum(1 for p in props if p.startswith("f_rest_")) // 3
    sh_deg = {0: 0, 3: 1, 8: 2, 15: 3}.get(k)
    if sh_deg is None:
        raise SplatFormatError(f"unsupported number of SH coefficients per channel: {k}")

    fixed = np.clip(np.rint(a[:, [ix["x"], ix["y"], ix["z"]]] * (1 << fractional_bits)), -(1 << 23), (1 << 23) - 1)
    fixed = fixed.astype(np.int64) & 0xFFFFFF
    pos = np.stack([fixed & 255, (fixed >> 8) & 255, (fixed >> 16) & 255], axis=-1).astype(np.uint8)

    op = 1.0 / (1.0 + np.exp(-a[:, ix["opacity"]]))
    alpha = np.clip(np.rint(op * 255.0), 0, 255).astype(np.uint8)
    dc = a[:, [ix["f_dc_0"], ix["f_dc_1"], ix["f_dc_2"]]]
    col = np.clip(np.rint((dc * _COLOR_SCALE + 0.5) * 255.0), 0, 255).astype(np.uint8)
    sc = np.clip(np.rint((a[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]] + 10.0) * 16.0), 0, 255).astype(np.uint8)

    q = a[:, [ix["rot_1"], ix["rot_2"], ix["rot_3"], ix["rot_0"]]]            # x y z w
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    largest = np.abs(q).argmax(1)
    sign = np.where(q[np.arange(n), largest] < 0, -1.0, 1.0)
    q = q * sign[:, None]
    comp = largest.astype(np.uint32)
    for i in range(4):
        use = largest != i
        mag = np.clip(np.rint(np.abs(q[:, i]) / _SQRT_HALF * 511.0), 0, 511).astype(np.uint32)
        neg = (q[:, i] < 0).astype(np.uint32)
        comp = np.where(use, (comp << np.uint32(10)) | (neg << np.uint32(9)) | mag, comp).astype(np.uint32)
    rot = np.stack([comp & 255, (comp >> 8) & 255, (comp >> 16) & 255, (comp >> 24) & 255], axis=-1).astype(np.uint8)

    chunks = [pos.tobytes(), alpha.tobytes(), col.tobytes(), sc.tobytes(), rot.tobytes()]
    if k:
        rest = np.stack([a[:, ix[f"f_rest_{c * k + j}"]] for c in range(3) for j in range(k)], axis=1).reshape(n, 3, k)
        sh = np.clip(np.rint(rest.transpose(0, 2, 1) * 128.0 + 128.0), 0, 255).astype(np.uint8)    # [n, coef, rgb]
        chunks.append(sh.tobytes())
    head = struct.pack("<IIIBBBB", SPZ_MAGIC, 3, n, sh_deg, fractional_bits, 0, 0)
    with gzip.GzipFile(str(path), "wb", compresslevel=6, mtime=0) as fh:
        fh.write(head)
        for c in chunks:
            fh.write(c)
    return {"version": 3, "count": n, "sh_degree": sh_deg, "fractional_bits": fractional_bits, "flags": 0}


# -- PLY -----------------------------------------------------------------------------


def read_ply_any(path):
    """A binary little-endian vertex PLY with scalar properties of any type -> (names, float32 [N, P])."""
    import numpy as np
    b = Path(path).read_bytes()
    marker = b"end_header\n"
    k = b.find(marker)
    if k < 0:
        marker = b"end_header\r\n"
        k = b.find(marker)
    if k < 0:
        raise SplatFormatError(f"{path}: no PLY header")
    end = k + len(marker)
    head = b[:end].decode("ascii", "replace").splitlines()
    if not any(l.startswith("format binary_little_endian") for l in head):
        raise SplatFormatError(f"{path} is not a binary little-endian PLY")
    names, dts, n, in_vertex = [], [], 0, False
    for line in head:
        t = line.split()
        if t[:1] == ["element"]:
            in_vertex = t[1] == "vertex"
            if in_vertex:
                n = int(t[2])
        elif t[:1] == ["property"] and in_vertex:
            if t[1] == "list" or t[1] not in _PLY_TYPES:
                raise SplatFormatError(f"{path}: unsupported PLY property type '{t[1]}'")
            names.append(t[-1])
            dts.append(_PLY_TYPES[t[1]])
    dt = np.dtype([(nm, d) for nm, d in zip(names, dts)])
    rec = np.frombuffer(b, dtype=dt, count=n, offset=end)
    return names, np.stack([rec[nm].astype(np.float32) for nm in names], axis=1)


def write_ply(path, props: list, a, comments: tuple = ()) -> None:
    import numpy as np
    head = ["ply", "format binary_little_endian 1.0", *[f"comment {c}" for c in comments],
            f"element vertex {len(a)}", *[f"property float {p}" for p in props], "end_header", ""]
    with open(path, "wb") as fh:
        fh.write("\n".join(head).encode("ascii"))
        fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def apply_metric(props: list, a, metric_scale_factor: Optional[float], ground_plane_offset: Optional[float]):
    """Marble's metric scale and ground alignment (see module docstring). Returns a new array."""
    import numpy as np
    out = np.array(a, copy=True)
    ix = {p: i for i, p in enumerate(props)}
    f = float(metric_scale_factor) if metric_scale_factor else 1.0
    out[:, [ix["x"], ix["y"], ix["z"]]] *= f
    out[:, ix["y"]] -= float(ground_plane_offset or 0.0)
    if f > 0 and f != 1.0:
        for j in range(3):
            if f"scale_{j}" in ix:
                out[:, ix[f"scale_{j}"]] += math.log(f)
    return out


# -- conversions -----------------------------------------------------------------------


def spz_to_ply(spz_path, ply_path, metric: Optional[dict] = None, comments: tuple = ()) -> dict:
    props, a = read_spz(spz_path)
    if metric:
        a = apply_metric(props, a, metric.get("metric_scale_factor"), metric.get("ground_plane_offset"))
    write_ply(ply_path, props, a, comments or ("Converted from Marble SPZ by agentY", "Axes: raw OpenCV (Y down)"))
    return {"count": int(len(a)), "sh_degree": sum(1 for p in props if p.startswith("f_rest_")) // 3}


def ply_to_spz(ply_path, spz_path) -> dict:
    props, a = read_ply_any(ply_path)
    return write_spz(spz_path, props, a)
