"""Measured facts about a clip, so a QA verdict does not rest on a vision model's opinion alone.

A vision-language model reading frames reports smear, void or drift that is not there, and calls a
panning camera "locked off". The numbers here are computed from the pixels - per-frame emptiness,
sharpness and brightness, frame-to-frame motion (phase correlation: translation and zoom), duplicate
and frozen frames - and carry a plain verdict. When a measurement and a description disagree, the
measurement wins.

Pure numpy + Pillow; the frames are small (256 px wide) so a clip takes about a second.

Conventions
  * ``content_shift`` is how the PICTURE moved between two frames: content moving left means the
    camera panned right.
  * ``empty_fraction`` is the share of 8x8 blocks with no detail (a flat colour): background, a void,
    or a blank wall. A furnished room is below ~0.3, a render of nothing above ~0.85.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("agenty_core.video_facts")

WIDTH = 256                 # frames are measured at this width
BLOCK = 8
FLAT_STD = 2.5              # luma std of a block below which it counts as flat
VOID_FRACTION = 0.85        # a frame this empty is a void
EMPTY_FRACTION = 0.6        # "mostly empty"
BLACK_MEAN = 8.0
FROZEN_DIFF = 0.15          # mean abs luma difference (0-255) below which two frames are the same picture
STATIC_DIFF = 1.0           # mean abs difference per step under which the picture does not change
STATIC_SHIFT = 0.5          # ... and a drift under this many % of the width per second
MIN_RESPONSE = 0.08         # phase-correlation peak below this = no reliable translation
MAX_FRAMES = 240


def _gray(rgb):
    import numpy as np
    a = rgb.astype("float32")
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def _laplacian_var(g) -> float:
    import numpy as np
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(np.var(lap))


def _empty_fraction(g) -> float:
    import numpy as np
    h, w = g.shape
    h8, w8 = h // BLOCK * BLOCK, w // BLOCK * BLOCK
    if h8 == 0 or w8 == 0:
        return 0.0
    blocks = g[:h8, :w8].reshape(h8 // BLOCK, BLOCK, w8 // BLOCK, BLOCK)
    return float((blocks.std(axis=(1, 3)) < FLAT_STD).mean())


def phase_shift(a, b):
    """(dx, dy, response): how far the picture in `b` moved against `a` (pixels; +x = to the right,
    +y = down), from the phase-correlation peak (sub-pixel), and the peak's height (0-1)."""
    import numpy as np
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype("float32")
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    corr = np.fft.ifft2(r).real
    py, px = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[py, px])

    def refine(c0, c1, c2):
        d = c0 - 2 * c1 + c2
        return 0.0 if abs(d) < 1e-9 else 0.5 * (c0 - c2) / d

    sx = refine(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    sy = refine(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dx, dy = px + sx, py + sy
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    return float(dx), float(dy), peak


def _pair_motion(ga, gb) -> dict:
    """Translation and zoom between two gray frames."""
    h, w = ga.shape
    dx, dy, resp = phase_shift(ga, gb)
    out = {"dx": dx, "dy": dy, "response": resp, "zoom": 0.0, "zoom_ok": False}
    hh, hw = h // 2, w // 2
    quads = {}
    for name, (ys, xs) in {"tl": (slice(0, hh), slice(0, hw)), "tr": (slice(0, hh), slice(hw, w)),
                           "bl": (slice(hh, h), slice(0, hw)), "br": (slice(hh, h), slice(hw, w))}.items():
        qx, qy, qr = phase_shift(ga[ys, xs], gb[ys, xs])
        quads[name] = (qx, qy, qr)
    if all(q[2] >= MIN_RESPONSE for q in quads.values()):
        dxr = (quads["tr"][0] + quads["br"][0]) / 2
        dxl = (quads["tl"][0] + quads["bl"][0]) / 2
        dyb = (quads["bl"][1] + quads["br"][1]) / 2
        dyt = (quads["tl"][1] + quads["tr"][1]) / 2
        # content expanding outward from the centre: the right half moves right, the left half left
        out["zoom"] = ((dxr - dxl) / (w / 2) + (dyb - dyt) / (h / 2)) / 2
        out["zoom_ok"] = True
    return out


def _to_small_rgb(frame):
    """Any HxWx3 (or HxW) uint8 array -> RGB uint8 at WIDTH px wide."""
    import numpy as np
    from PIL import Image
    a = np.asarray(frame)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype("uint8")
    h, w = a.shape[:2]
    if w != WIDTH:
        a = np.asarray(Image.fromarray(a).resize((WIDTH, max(8, round(h * WIDTH / w))), Image.BILINEAR))
    return a


def measure_frames(frames, fps: float = 24.0, step: int = 1, labels: Optional[list] = None) -> dict:
    """Measure a list of frames (HxWx3 uint8 arrays, in order). `step` is how many source frames lie
    between consecutive entries (1 = every frame), so speeds stay per second of the real clip."""
    import numpy as np
    n = len(frames)
    if n == 0:
        return {"frames": 0, "error": "no frames"}
    grays, per = [], []
    for i, f in enumerate(frames):
        rgb = _to_small_rgb(f)
        g = _gray(rgb)
        grays.append(g)
        per.append({"index": i if labels is None else labels[i],
                    "mean_luma": round(float(g.mean()), 1),
                    "sharpness": round(_laplacian_var(g), 1),
                    "empty_fraction": round(_empty_fraction(g), 3)})
    h, w = grays[0].shape
    pairs = []
    for a, b in zip(grays, grays[1:]):
        mad = float(np.abs(a - b).mean())
        m = _pair_motion(a, b) if mad >= FROZEN_DIFF else {"dx": 0.0, "dy": 0.0, "response": 1.0, "zoom": 0.0,
                                                           "zoom_ok": True}
        m["mad"] = mad
        pairs.append(m)
    secs_per_pair = max(1, int(step)) / max(1e-6, float(fps))
    shifts = [(p["dx"] / w * 100.0 / secs_per_pair, p["dy"] / w * 100.0 / secs_per_pair)
              for p in pairs if p["response"] >= MIN_RESPONSE]
    out: dict = {"frames": n, "fps": fps, "sampled_every": int(step), "size": [w, h]}
    # --- picture quality per frame
    lum = [p["mean_luma"] for p in per]
    sharp = [p["sharpness"] for p in per]
    empt = [p["empty_fraction"] for p in per]
    med_sharp = float(np.median(sharp))
    out["empty_fraction"] = {"mean": round(float(np.mean(empt)), 3), "max": round(float(np.max(empt)), 3),
                             "void_frames": [per[i]["index"] for i, e in enumerate(empt) if e >= VOID_FRACTION],
                             "mostly_empty_frames": [per[i]["index"] for i, e in enumerate(empt) if e >= EMPTY_FRACTION]}
    out["sharpness"] = {"median": round(med_sharp, 1), "min": round(float(np.min(sharp)), 1),
                        "max": round(float(np.max(sharp)), 1),
                        "soft_frames": [per[i]["index"] for i, s in enumerate(sharp)
                                        if med_sharp > 0 and s < 0.35 * med_sharp],
                        "note": "variance of the Laplacian at 256 px wide: compare frames of ONE clip, not clips"}
    out["brightness"] = {"mean": round(float(np.mean(lum)), 1), "min": round(float(np.min(lum)), 1),
                         "max": round(float(np.max(lum)), 1),
                         "black_frames": [per[i]["index"] for i, v in enumerate(lum) if v < BLACK_MEAN],
                         "jumps": [per[i + 1]["index"] for i in range(n - 1) if abs(lum[i + 1] - lum[i]) > 40.0]}
    # --- motion
    if pairs:
        mads = [p["mad"] for p in pairs]
        frozen = [i for i, m in enumerate(mads) if m < FROZEN_DIFF]
        run = best = 0
        for i, m in enumerate(mads):
            run = run + 1 if m < FROZEN_DIFF else 0
            best = max(best, run)
        mean_dx = float(np.mean([s[0] for s in shifts])) if shifts else 0.0     # % of width per second
        mean_dy = float(np.mean([s[1] for s in shifts])) if shifts else 0.0
        speed = float(np.mean([np.hypot(*s) for s in shifts])) if shifts else 0.0
        zooms = [p["zoom"] for p in pairs if p["zoom_ok"]]
        zoom_per_s = float(np.mean(zooms)) * 100.0 / secs_per_pair if zooms else 0.0   # % scale change / s
        mean_mad = float(np.mean(mads))
        out["motion"] = {
            "mean_abs_diff": round(mean_mad, 2),
            "content_shift_pct_width_per_s": {"x": round(mean_dx, 2), "y": round(mean_dy, 2), "speed": round(speed, 2)},
            "zoom_pct_per_s": round(zoom_per_s, 2),
            "frozen_pairs": len(frozen), "longest_frozen_run": best,
            "duplicate_frame_indices": [per[i + 1]["index"] for i in frozen][:40],
        }
        out["camera"] = _classify_camera(mean_mad, mean_dx, mean_dy, speed, zoom_per_s, shifts, zooms)
    out["verdict"] = _verdict(out)
    out["rule"] = ("These numbers come from the pixels. If a vision model's description disagrees (calls a "
                   "moving camera static, a furnished frame a void, a sharp one smeared), trust the numbers "
                   "and say so.")
    return out


def _classify_camera(mad, dx, dy, speed, zoom, shifts, zooms) -> dict:
    moving = mad >= STATIC_DIFF or speed >= STATIC_SHIFT or abs(zoom) >= 0.5
    if not moving:
        return {"state": "static", "direction": "none"}
    parts = []
    if speed >= STATIC_SHIFT:
        if abs(dx) >= abs(dy):
            parts.append("pan right" if dx < 0 else "pan left")      # content left = camera right
        else:
            parts.append("tilt up" if dy > 0 else "tilt down")        # content down = camera up
    if abs(zoom) >= 0.5:
        parts.append("push in / move forward" if zoom > 0 else "pull out / move back")
    return {"state": "moving", "direction": " + ".join(parts) or "changing picture (parallax / other)",
            "steadiness": _steadiness(shifts)}


def _steadiness(shifts) -> str:
    import numpy as np
    if len(shifts) < 4:
        return "n/a"
    sp = np.array([np.hypot(*s) for s in shifts])
    if sp.mean() < 1e-6:
        return "n/a"
    cv = float(sp.std() / sp.mean())
    return "steady" if cv < 0.5 else "uneven (speed varies a lot)"


def _verdict(m: dict) -> list:
    v = []
    cam = m.get("camera")
    if cam:
        v.append(f"camera: {cam['state']}" + (f" ({cam['direction']})" if cam["direction"] != "none" else ""))
    e = m["empty_fraction"]
    if e["void_frames"]:
        v.append(f"VOID: {len(e['void_frames'])} frame(s) are >= {int(VOID_FRACTION * 100)}% empty (first: {e['void_frames'][0]})")
    elif e["mostly_empty_frames"]:
        v.append(f"mostly empty: {len(e['mostly_empty_frames'])} frame(s) >= {int(EMPTY_FRACTION * 100)}% flat colour")
    else:
        v.append(f"not empty: flat-colour share mean {e['mean']}, max {e['max']}")
    if m["brightness"]["black_frames"]:
        v.append(f"black frames: {m['brightness']['black_frames'][:8]}")
    if m["brightness"]["jumps"]:
        v.append(f"brightness jumps at {m['brightness']['jumps'][:8]}")
    if m["sharpness"]["soft_frames"]:
        v.append(f"soft frames (< 35% of the median sharpness): {m['sharpness']['soft_frames'][:8]}")
    mo = m.get("motion")
    if mo and mo["longest_frozen_run"] >= 2:
        v.append(f"frozen: {mo['longest_frozen_run']} identical frames in a row ({mo['frozen_pairs']} duplicate pairs)")
    return v


# -- clips ------------------------------------------------------------------------------------


def _decode(path: Path, max_frames: int):
    """(frames, fps, step, total) - up to `max_frames` RGB frames, every k-th when the clip is longer."""
    try:
        import imageio.v3 as iio
        meta = {}
        try:
            meta = iio.immeta(path, plugin="pyav") or {}
        except Exception:  # noqa: BLE001
            try:
                meta = iio.immeta(path) or {}
            except Exception:  # noqa: BLE001
                meta = {}
        fps = float(meta.get("fps") or 24.0)
        allf = [f for f in iio.imiter(path)]
        total = len(allf)
        step = max(1, -(-total // max_frames))
        return allf[::step], fps, step, total
    except Exception as exc:  # noqa: BLE001
        logger.info("imageio could not read %s (%s); trying OpenCV", path, exc)
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, -(-total // max_frames)) if total else 1
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return frames, fps, step, total or i


def measure_video(path, max_frames: int = MAX_FRAMES) -> dict:
    """Measured facts about a video file (see measure_frames). Never raises: {"error": ...}."""
    p = Path(path)
    try:
        frames, fps, step, total = _decode(p, max(8, int(max_frames)))
        if not frames:
            return {"error": f"no frames could be read from {p}"}
        out = measure_frames(frames, fps=fps, step=step, labels=[i * step for i in range(len(frames))])
        out["total_frames"] = total
        return out
    except Exception as exc:  # noqa: BLE001
        logger.exception("measuring %s failed", p)
        return {"error": f"{type(exc).__name__}: {exc}"}


def measure_images(paths, fps: float = 24.0) -> dict:
    """Measure an ordered list of image files as a sequence of frames."""
    from PIL import Image
    import numpy as np
    frames = []
    for pth in paths:
        with Image.open(pth) as im:
            frames.append(np.asarray(im.convert("RGB")))
    return measure_frames(frames, fps=fps, step=1, labels=[Path(x).stem for x in paths])


def summary_text(m: dict) -> str:
    """Short text for a tool result / prompt: the verdict lines and the rule."""
    if not m or m.get("error"):
        return ""
    lines = ["MEASURED from the pixels (trust these over a description):"]
    lines += [f"- {v}" for v in m.get("verdict", [])]
    mo = m.get("motion") or {}
    if mo:
        s = mo["content_shift_pct_width_per_s"]
        lines.append(f"- motion: mean frame difference {mo['mean_abs_diff']}/255, picture drift {s['speed']}% of width "
                     f"per second (x {s['x']}, y {s['y']}), zoom {mo['zoom_pct_per_s']}%/s")
    lines.append(f"- sharpness median {m['sharpness']['median']} (min {m['sharpness']['min']}), "
                 f"brightness {m['brightness']['mean']}")
    return "\n".join(lines)
