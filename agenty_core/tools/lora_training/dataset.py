"""Build, caption and review a LoRA dataset (the logic behind ``lora_dataset_prepare`` / ``lora_dataset_review``).

Pure Pillow + numpy for pictures, the ffmpeg that ``imageio-ffmpeg`` ships for clips. Captioning is a seam:
the host registers ``set_captioner(fn)`` (agentY: its vision agent); without one the items come back with
empty captions, ``status: needs_captions``, and the agent writes them with ``lora_dataset_set_captions``.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from . import formats as F
from .presets import get_preset

logger = logging.getLogger("agenty_core.lora.dataset")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

# aspect ratios (w / h) a picture is cropped to, so the trainer's own bucketing has little left to do
RATIOS = [(1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16)]

MIN_ITEMS = {"character": 12, "style": 20, "product": 12, "motion": 8, "concept": 12}
BLUR_VAR = 12.0            # variance of the Laplacian (512 px, 0-255) under which a picture is "blurry"
DUP_BITS = 5               # dHash Hamming distance at or under which two pictures are the same shot

# captioner(path, instruction) -> caption text. path is an image, or a video for kind video items.
Captioner = Callable[[str, str], str]
_captioner: Optional[Captioner] = None


def set_captioner(fn: Optional[Captioner]) -> None:
    global _captioner
    _captioner = fn


def get_captioner() -> Optional[Captioner]:
    return _captioner


# -- source collection ---------------------------------------------------------


def collect_sources(sources) -> list[tuple[str, str]]:
    """[(path, source kind)] from files, folders, globs, URLs (downloaded) or a comma / newline separated string."""
    import glob
    if isinstance(sources, (str, os.PathLike)):
        sources = [s for s in re.split(r"[\n;]|,(?=\s*(?:[A-Za-z]:|/|\\|http))", str(sources)) if s.strip()]
    out, seen = [], set()
    for raw in sources or []:
        s = str(raw).strip().strip('"')
        if not s:
            continue
        found: list[tuple[str, str]] = []
        if s.lower().startswith(("http://", "https://")):
            p = _download(s)
            if p:
                found.append((p, "web"))
        else:
            p = Path(os.path.expanduser(s))
            if p.is_dir():
                for f in sorted(p.rglob("*")):
                    if f.suffix.lower() in IMAGE_EXT | VIDEO_EXT and f.is_file():
                        found.append((str(f), "folder"))
            elif p.is_file():
                found.append((str(p), "file"))
            else:
                for g in sorted(glob.glob(os.path.expanduser(s))):
                    if Path(g).suffix.lower() in IMAGE_EXT | VIDEO_EXT:
                        found.append((g, "file"))
        for path, kind in found:
            key = os.path.normcase(os.path.abspath(path))
            if key not in seen:
                seen.add(key)
                out.append((os.path.abspath(path), kind))
    return out


def _download(url: str) -> Optional[str]:
    try:
        import requests
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        ext = Path(url.split("?")[0]).suffix.lower()
        if ext not in IMAGE_EXT | VIDEO_EXT:
            ext = ".jpg"
        from agenty_core.paths import project_root
        d = project_root() / "output" / "agent" / "lora_downloads"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"dl_{int(time.time() * 1000)}{ext}"
        f.write_bytes(r.content)
        return str(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not download %s: %s", url, exc)
        return None


# -- measuring -----------------------------------------------------------------


def dhash(img, size: int = 8) -> int:
    from PIL import Image
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(g.getdata())
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (1 if px[r * (size + 1) + c] > px[r * (size + 1) + c + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sharpness(img) -> float:
    """Variance of the Laplacian on a 512 px grey copy (higher = sharper)."""
    import numpy as np
    from PIL import Image
    g = img.convert("L")
    g.thumbnail((512, 512), Image.LANCZOS)
    a = np.asarray(g, dtype="float32")
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = -4.0 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())


def nearest_ratio(w: int, h: int) -> tuple[int, int]:
    a = w / max(1, h)
    return min(RATIOS, key=lambda r: abs(r[0] / r[1] - a))


def bucket_size(ratio: tuple[int, int], res: int, mult: int = 16) -> tuple[int, int]:
    """(w, h) with about res*res pixels at *ratio*, a multiple of *mult*."""
    rw, rh = ratio
    k = (res * res / (rw * rh)) ** 0.5
    return max(mult, round(rw * k / mult) * mult), max(mult, round(rh * k / mult) * mult)


def auto_crop(w: int, h: int, ratio: tuple[int, int]) -> list[float]:
    """Normalised centre crop [x, y, w, h] of a w x h picture to *ratio*."""
    target = ratio[0] / ratio[1]
    if w / h > target:
        cw = h * target / w
        return [round((1 - cw) / 2, 5), 0.0, round(cw, 5), 1.0]
    ch = w / target / h
    return [0.0, round((1 - ch) / 2, 5), 1.0, round(ch, 5)]


def _fit(w: int, h: int, longest: int) -> tuple[int, int]:
    k = min(1.0, longest / max(w, h))
    return max(1, round(w * k)), max(1, round(h * k))


def render_crop(path: str, crop, size: tuple[int, int]):
    """The picture at *path* cropped to normalised *crop* and resized to *size* (PIL image, RGB)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        W, H = im.size
        if crop:
            x, y, cw, ch = (float(v) for v in crop)
            box = (round(max(0.0, x) * W), round(max(0.0, y) * H),
                   round(min(1.0, x + cw) * W), round(min(1.0, y + ch) * H))
            if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                im = im.crop(box)
        return im.resize(size, Image.LANCZOS)


# -- video ---------------------------------------------------------------------


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg") or "ffmpeg"


def probe_video(path: str) -> dict:
    """{duration, fps, width, height, has_audio} read off ``ffmpeg -i`` (no ffprobe needed)."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True, errors="replace")
    t = p.stderr
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", t)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    fps, w, h = 0.0, 0, 0
    v = re.search(r"Stream #\S+: Video:.*?(\d{2,5})x(\d{2,5}).*?([\d.]+) fps", t, re.S)
    if v:
        w, h, fps = int(v.group(1)), int(v.group(2)), float(v.group(3))
    return {"duration": dur, "fps": fps, "width": w, "height": h, "has_audio": "Audio:" in t}


def cut_clip(src: str, dst: str, start: float, frames: int, fps: float, size: tuple[int, int], crop=None,
             audio: bool = False) -> bool:
    """Cut *frames* frames at *fps* from *start* s, cropped (normalised) and scaled to *size*."""
    vf = []
    if crop:
        x, y, cw, ch = crop
        vf.append(f"crop=iw*{cw}:ih*{ch}:iw*{x}:ih*{y}")
    vf.append(f"fps={fps}")
    vf.append(f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,crop={size[0]}:{size[1]}")
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", src,
           "-frames:v", str(frames), "-vf", ",".join(vf), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p"]
    cmd += ["-c:a", "aac", "-t", f"{frames / fps:.3f}"] if audio else ["-an"]
    cmd.append(dst)
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return r.returncode == 0 and os.path.exists(dst)


def video_thumb(path: str, dst: str, at: float = 0.0) -> bool:
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{at:.3f}", "-i", path,
                        "-frames:v", "1", dst], capture_output=True)
    return r.returncode == 0 and os.path.exists(dst)


# -- captioning ----------------------------------------------------------------


def caption_instruction(kind: str, subject: str, style: str, trigger: str, video: bool) -> str:
    what = "video clip" if video else "picture"
    base = (f"Write the training caption for this {what} of a LoRA that teaches: {subject or 'the subject'} "
            f"(kind: {kind}). Caption style: {style}. Describe what is visible - setting, pose, clothing, light, "
            "camera, composition")
    if video:
        base += ", and the movement and camera motion over time"
    base += ". Leave out what the LoRA should learn as its identity: do not describe the "
    base += {"character": "face and body features that stay the same in every picture",
             "style": "style itself (medium, palette, brushwork)",
             "product": "product's fixed shape and branding details",
             "motion": "motion pattern itself"}.get(kind, "learned subject") + ". "
    if trigger:
        base += f"Do not write the trigger word '{trigger}'; it is added for you. "
    base += ("If the picture has a visible watermark, logo or caption text overlay, start your answer with "
             "WATERMARK: . Answer with the caption only, no preface, no quotes.")
    return base


def clean_caption(text: str, trigger: str) -> str:
    t = " ".join(str(text or "").replace("\r", " ").split()).strip().strip('"').strip()
    t = re.sub(r"^(caption|description)\s*:\s*", "", t, flags=re.I)
    return t


def with_trigger(caption: str, trigger: str) -> str:
    """The caption as the trainer sees it: the trigger word first, once."""
    c = (caption or "").strip()
    if not trigger or trigger.lower() in c.lower():
        return c
    return f"{trigger}, {c}" if c else trigger


# -- prepare -------------------------------------------------------------------


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def prepare_dataset(sources, subject: str, kind: str, model: str, name: str = "", trigger_word: str = "",
                    caption_style: str = "", max_items: int = 200, resolution: int = 0, output_dir: str = "",
                    clip_seconds: float = 0.0, max_clips_per_video: int = 6) -> dict:
    """Collect, check, crop and caption. Returns {dataset, manifest_path, summary, removed, needs_captions}."""
    preset = get_preset(model)
    kind = (kind or "character").lower()
    video_model = preset["kind"] == "video"
    res = int(resolution) or max(preset.get("resolutions") or [1024])
    name = F.slug(name) or F.slug(subject) or f"lora_{int(time.time())}"
    ds = F.datasets_root(output_dir) / name
    if (ds / "manifest.json").exists():
        old = F.read_json(ds / "manifest.json", {})
        if old.get("status") in ("training",):
            raise ValueError(f"dataset '{name}' is in use by a running training")
        for sub in ("items",):
            shutil.rmtree(ds / sub, ignore_errors=True)
    items_dir = ds / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    trigger = (trigger_word or "").strip()
    if not trigger:
        trigger = re.sub(r"[^a-z0-9]", "", name.lower())[:14] + "_lora"
    style = caption_style or preset.get("caption_style", "natural sentences")
    files = collect_sources(sources)
    if not files:
        raise ValueError("no pictures or clips found in the sources")

    items: list[dict] = []
    hashes: list[tuple[int, str]] = []
    buckets: dict[tuple[int, int], int] = {}
    n = 0

    def add(entry: dict) -> dict:
        items.append(entry)
        return entry

    for path, src_kind in files:
        if len([i for i in items if i["included"]]) >= int(max_items):
            break
        ext = Path(path).suffix.lower()
        n += 1
        iid = f"i{n:03d}"
        if ext in IMAGE_EXT:
            e = _prepare_image(path, src_kind, iid, ds, res, hashes, buckets, video_model)
            add(e)
        elif ext in VIDEO_EXT:
            clips = _prepare_clips(path, src_kind, iid, ds, preset, res, buckets, clip_seconds, max_clips_per_video,
                                   video_model)
            n += max(0, len(clips) - 1)
            for e in clips:
                add(e)
    inc = [i for i in items if i["included"]]
    if not inc:
        raise ValueError("every item was removed (too small, blurry or duplicate); check the sources")

    captioner = get_captioner()
    needs = []
    for it in items:
        if not it["included"]:
            continue
        src = str(ds / it["file"]) if it["kind"] == "video" else str(ds / it["file"])
        if captioner is None:
            needs.append(it["id"])
            continue
        try:
            cap = clean_caption(captioner(src, caption_instruction(kind, subject, style, trigger,
                                                                    it["kind"] == "video")), trigger)
        except Exception as exc:  # noqa: BLE001
            logger.warning("captioning %s failed: %s", it["id"], exc)
            needs.append(it["id"])
            continue
        if cap.upper().startswith("WATERMARK"):
            it["included"], it["removed_by"] = False, "agent"
            it["note"] = "watermark / overlay text visible"
            cap = ""
        it["caption"] = cap
        if not cap and it["included"]:
            needs.append(it["id"])

    warnings = _warnings(items, kind, preset, needs)
    manifest = {
        "version": 1, "name": name, "subject": subject, "kind": kind, "model": preset["key"],
        "trigger_word": trigger, "caption_style": style, "status": "draft", "approved_at": None,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "resolution": res,
        "buckets": [list(b) for b in sorted(buckets)], "items": items,
        "summary": {"warnings": warnings}, "needs_captions": needs,
    }
    F.save_manifest(ds, manifest)
    (ds / "review.json").unlink(missing_ok=True)
    try:
        make_contact_sheet(ds, manifest)
    except Exception as exc:  # noqa: BLE001
        logger.info("contact sheet failed: %s", exc)
    return manifest


def _prepare_image(path, src_kind, iid, ds: Path, res, hashes, buckets, video_model) -> dict:
    from PIL import Image, ImageOps
    e = {"id": iid, "file": "", "kind": "image", "caption": "", "included": True, "crop": None, "original": path,
         "size": None, "frames": None, "fps": None, "source": src_kind, "note": "", "removed_by": None}
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = im.size
            e["size"] = [W, H]
            reason = None
            if min(W, H) < 0.5 * min(res, 1024):
                reason = f"too small ({W}x{H}) for {res}px training"
            else:
                sh = sharpness(im)
                if sh < BLUR_VAR:
                    reason = f"blurry (sharpness {sh:.1f})"
            h = dhash(im)
            if not reason:
                for hh, other in hashes:
                    if hamming(h, hh) <= DUP_BITS:
                        reason = f"duplicate of {other}"
                        break
            ratio = nearest_ratio(W, H)
            crop = auto_crop(W, H, ratio)
            size = bucket_size(ratio, res)
            e["crop"] = crop
            out = ds / "items" / f"{iid}.jpg"
            # the item file is the WHOLE picture (long side <= 1280): the crop box is normalised on it, and on the
            # original, the same way; training renders the crop from the original (write_trainer_dataset)
            render_crop(path, None, _fit(W, H, 1280)).save(out, quality=92)
            e["file"] = _rel(out, ds)
            if reason:
                e["included"], e["removed_by"], e["note"] = False, "agent", reason
            else:
                hashes.append((h, iid))
                buckets[size] = buckets.get(size, 0) + 1
                e["note"] = f"kept; crop {ratio[0]}:{ratio[1]} -> trained at {size[0]}x{size[1]}"
    except Exception as exc:  # noqa: BLE001
        e["included"], e["removed_by"], e["note"] = False, "agent", f"unreadable ({type(exc).__name__})"
    return e


def _prepare_clips(path, src_kind, iid0, ds: Path, preset, res, buckets, clip_seconds, max_clips, video_model) -> list[dict]:
    info = probe_video(path)
    base = {"kind": "video", "caption": "", "included": True, "crop": None, "original": path, "source": src_kind,
            "removed_by": None}
    if not video_model:
        return [{**base, "id": iid0, "file": "", "size": [info["width"], info["height"]], "frames": None, "fps": None,
                 "included": False, "removed_by": "agent",
                 "note": f"a clip, but {preset['label']} trains on pictures; pull frames first"}]
    fps = float(preset.get("fps", 24))
    frames = int(preset.get("frames", 49))
    if clip_seconds:
        frames = max(9, int(clip_seconds * fps) // 4 * 4 + 1)
    seg = frames / fps
    ratio = nearest_ratio(info["width"] or 16, info["height"] or 9)
    size = bucket_size(ratio, res, 32)
    out: list[dict] = []
    if info["duration"] < seg * 0.9:
        return [{**base, "id": iid0, "file": "", "size": [info["width"], info["height"]], "frames": None, "fps": info["fps"],
                 "included": False, "removed_by": "agent",
                 "note": f"too short ({info['duration']:.1f}s) for {frames} frames at {fps:g} fps"}]
    count = min(int(max_clips), int(info["duration"] // seg))
    # spread the cuts over the source instead of taking only the first seconds
    step = (info["duration"] - seg) / max(1, count - 1) if count > 1 else 0.0
    for k in range(count):
        iid = f"{iid0[0]}{int(iid0[1:]) + k:03d}"
        dst = ds / "items" / f"{iid}.mp4"
        ok = cut_clip(path, str(dst), k * step, frames, fps, size, None, bool(preset.get("audio")) and info["has_audio"])
        out.append({**base, "id": iid, "file": _rel(dst, ds) if ok else "", "size": list(size), "frames": frames,
                    "fps": fps, "start": round(k * step, 3), "included": ok, "removed_by": None if ok else "agent",
                    "note": f"clip {k + 1}/{count} at {k * step:.1f}s, {frames} frames @ {fps:g} fps -> {size[0]}x{size[1]}"
                            if ok else "ffmpeg could not cut this clip"})
        if ok:
            buckets[size] = buckets.get(size, 0) + 1
    return out


def _warnings(items, kind, preset, needs) -> list[str]:
    inc = [i for i in items if i["included"]]
    w = []
    need = MIN_ITEMS.get(kind, 12)
    if len(inc) < need:
        w.append(f"only {len(inc)} usable items; a {kind} LoRA wants about {need}+")
    dup = sum(1 for i in items if not i["included"] and str(i.get("note", "")).startswith("duplicate"))
    if dup >= max(3, len(items) // 4):
        w.append(f"{dup} near-duplicates were removed: the set is very repetitive")
    small = sum(1 for i in items if "too small" in str(i.get("note", "")))
    if small:
        w.append(f"{small} items were too small")
    blurry = sum(1 for i in items if "blurry" in str(i.get("note", "")))
    if blurry:
        w.append(f"{blurry} items were blurry")
    wm = sum(1 for i in items if "watermark" in str(i.get("note", "")))
    if wm:
        w.append(f"{wm} items had a watermark or overlay text and were removed")
    if needs:
        w.append(f"{len(needs)} items have no caption yet")
    sizes = {tuple(i["size"]) for i in inc if i.get("size")}
    if len(sizes) > 6:
        w.append("many different bucket sizes; training will be slower")
    return w


# -- captions, review ----------------------------------------------------------


def set_captions(ds: Path, captions: dict, trigger_word: str = "") -> dict:
    m = F.load_manifest(ds)
    if trigger_word:
        m["trigger_word"] = trigger_word.strip()
    by = {i["id"]: i for i in m["items"]}
    unknown = []
    for k, v in (captions or {}).items():
        if k in by:
            by[k]["caption"] = clean_caption(v, m.get("trigger_word", ""))
        else:
            unknown.append(k)
    m["needs_captions"] = [i["id"] for i in m["items"] if i["included"] and not i.get("caption")]
    s = m.setdefault("summary", {})
    s["warnings"] = [w for w in s.get("warnings", []) if "no caption yet" not in w]
    if m["needs_captions"]:
        s["warnings"].append(f"{len(m['needs_captions'])} items have no caption yet")
    F.save_manifest(ds, m)
    # an edit after approval takes the approval away
    r = F.load_review(ds)
    if r.get("state") == "approved":
        F.write_json(ds / "review.json", {**r, "state": "pending", "updated": time.time()})
    return {"missing": m["needs_captions"], "unknown_ids": unknown}


def make_contact_sheet(ds: Path, manifest: dict, cell: int = 320) -> str:
    """One JPEG of every included item with its id and caption underneath (for Slack / MCP review)."""
    from PIL import Image, ImageDraw, ImageFont
    inc = [i for i in manifest["items"] if i.get("included", True)][:60]
    cols = min(5, max(1, len(inc)))
    rows = (len(inc) + cols - 1) // cols
    pad, textw, lines = 8, cell, 5
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    th = lines * 16 + 4
    sheet = Image.new("RGB", (cols * (cell + pad) + pad, max(1, rows) * (cell + th + pad) + pad + 30), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 6), f"{manifest.get('name')}  |  {manifest.get('model')}  |  trigger: {manifest.get('trigger_word')}  |  "
                     f"{len(inc)} items", fill=(255, 140, 0), font=font)
    for idx, it in enumerate(inc):
        x, y = pad + (idx % cols) * (cell + pad), 30 + pad + (idx // cols) * (cell + th + pad)
        f = ds / it["file"]
        thumb = None
        try:
            if it["kind"] == "video":
                tmp = ds / f".thumb_{it['id']}.jpg"
                if video_thumb(str(f), str(tmp)):
                    thumb = Image.open(tmp).convert("RGB")
                    tmp.unlink(missing_ok=True)
            else:
                thumb = Image.open(f).convert("RGB")
                if it.get("crop"):
                    cx, cy, cw, ch = it["crop"]
                    W, H = thumb.size
                    thumb = thumb.crop((round(cx * W), round(cy * H), round((cx + cw) * W), round((cy + ch) * H)))
        except Exception:  # noqa: BLE001
            thumb = None
        if thumb is not None:
            thumb.thumbnail((cell, cell))
            sheet.paste(thumb, (x + (cell - thumb.width) // 2, y))
        d.rectangle([x, y, x + 34, y + 16], fill=(0, 0, 0))
        d.text((x + 3, y + 1), it["id"], fill=(255, 255, 255), font=font)
        cap = it.get("caption") or "(no caption)"
        words, line, ln = cap.split(), "", 0
        for wd in words:
            if len(line) + len(wd) + 1 > 46:
                d.text((x, y + cell + 2 + ln * 16), line, fill=(220, 220, 220), font=font)
                line, ln = wd, ln + 1
                if ln >= lines:
                    break
            else:
                line = (line + " " + wd).strip()
        if ln < lines:
            d.text((x, y + cell + 2 + ln * 16), line, fill=(220, 220, 220), font=font)
    out = ds / "contact_sheet.jpg"
    sheet.convert("RGB").save(out, quality=88)
    return str(out)


def request_review(ds: Path, by_note: str = "") -> dict:
    m = F.load_manifest(ds)
    if m.get("needs_captions"):
        raise ValueError(f"{len(m['needs_captions'])} items have no caption yet "
                         f"({', '.join(m['needs_captions'][:8])}); write them with lora_dataset_set_captions first")
    sheet = make_contact_sheet(ds, m)
    now = time.time()
    F.write_json(ds / "review.json", {"state": "pending", "requested": now, "updated": now, "note": by_note,
                                      "manifest_sha": "", "by": ""})
    m["status"] = "review"
    F.save_manifest(ds, m)
    return {"contact_sheet": sheet, "manifest": str(ds / "manifest.json"), "summary": m["summary"],
            "trigger_word": m.get("trigger_word"), "items": m["summary"].get("included")}


# -- trainer files -------------------------------------------------------------


def write_trainer_dataset(ds: Path, manifest: dict, bucket_mult: int = 16) -> dict:
    """Render every included item (crop applied) into ``trainer/`` as file + .txt caption. Returns counts + folder."""
    out = ds / "trainer"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    trigger = manifest.get("trigger_word", "")
    res = int(manifest.get("resolution") or 1024)
    count = 0
    for it in manifest["items"]:
        if not it.get("included", True):
            continue
        stem = it["id"]
        if it["kind"] == "image":
            orig = it.get("original", "")
            src = orig if orig and os.path.exists(orig) else str(ds / it["file"])
            from PIL import Image
            with Image.open(src) as im0:
                W, H = im0.size
            crop = it.get("crop")
            cw, ch = (crop[2] * W, crop[3] * H) if crop else (W, H)
            ratio = nearest_ratio(max(1, int(cw)), max(1, int(ch)))
            render_crop(src, crop, bucket_size(ratio, res, bucket_mult)).save(out / f"{stem}.png")
        else:
            shutil.copyfile(ds / it["file"], out / f"{stem}.mp4")
        (out / f"{stem}.txt").write_text(with_trigger(it.get("caption", ""), trigger), encoding="utf-8")
        count += 1
    return {"folder": str(out), "count": count}
