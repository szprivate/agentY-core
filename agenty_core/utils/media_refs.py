"""Find the media files a tool's result talks about.

A tool that makes a picture, film, sound or 3D file outside a ComfyUI save node (an
MCP server's render, a script, a sub-conversation) says so only in its result text:
"saved to D:/renders/final.mp4", or ``{"url": "https://cdn.../clip.mp4"}``. A host
that wants to show those files has nothing to go on but that text. This reads it,
conservatively:

* a **local path** counts only when the file exists, has a media extension, and (when
  *min_mtime* is given) was written since the call began - so a tool that merely
  echoes an input file back is not mistaken for having made it;
* a **URL** counts only when its path ends in a media extension;
* nothing is downloaded here unless :func:`download` is called.

Pure standard library apart from ``requests`` inside :func:`download`, so the MCP
server can use it as the agentY host does.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".avif", ".exr")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi")
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
MODEL_EXTS = (".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz")
MEDIA_EXTS = IMAGE_EXTS + VIDEO_EXTS + AUDIO_EXTS + MODEL_EXTS

_EXT_ALT = "|".join(re.escape(e) for e in sorted(MEDIA_EXTS, key=len, reverse=True))
_WIN_START = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
_WIN_TAIL = re.compile(r"[^\n\"'<>|*?]*?(?:%s)(?![A-Za-z0-9_])" % _EXT_ALT, re.IGNORECASE)
_POSIX = re.compile(r"(?<![\w:/.\\])/(?:[\w.\-]+/)+[\w.\-]*?(?:%s)(?![A-Za-z0-9_])" % _EXT_ALT,
                    re.IGNORECASE)
_FILE_URL = re.compile(r"file:///?([^\s\"'<>]+)", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)

MAX_STRINGS = 4000


def kind_of(ref: str) -> str:
    """``image`` / ``video`` / ``audio`` / ``model`` by extension, else ``""``."""
    ext = os.path.splitext(urlparse(ref).path if "://" in ref else ref)[1].lower()
    for kind, exts in (("image", IMAGE_EXTS), ("video", VIDEO_EXTS),
                       ("audio", AUDIO_EXTS), ("model", MODEL_EXTS)):
        if ext in exts:
            return kind
    return ""


def strings_of(result: Any) -> Iterator[str]:
    """Every string inside a tool result (text blocks, JSON values, nested lists).

    A string that is itself JSON is opened, since an MCP server often returns its
    structured answer as one text block."""
    seen = 0
    stack = [result]
    while stack and seen < MAX_STRINGS:
        item = stack.pop()
        if isinstance(item, str):
            seen += 1
            yield item
            s = item.strip()
            if s[:1] in "{[" and s[-1:] in "}]":
                try:
                    stack.append(json.loads(s))
                except ValueError:
                    pass
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _local_candidates(text: str) -> Iterator[str]:
    text = text.replace("\\\\", "\\")
    stripped = text.strip().strip("\"'")
    if "\n" not in stripped and len(stripped) < 1024 and kind_of(stripped):
        yield stripped                       # the whole string is the path (spaces and all)
    for m in _FILE_URL.finditer(text):
        yield unquote(m.group(1)).lstrip("/") if re.match(r"/?[A-Za-z]:", unquote(m.group(1))) \
            else "/" + unquote(m.group(1))
    for line in text.splitlines():
        for start in _WIN_START.finditer(line):
            tail = _WIN_TAIL.match(line, start.start() + 3)
            if tail:
                yield line[start.start():tail.end()]
        for m in _POSIX.finditer(line):
            yield m.group(0)


def find_local(result: Any, *, min_mtime: float | None = None, limit: int = 24) -> list[str]:
    """Existing media files named in *result*, as absolute paths, in the order found.

    With *min_mtime* (a ``time.time()`` value), only files modified at or after it."""
    found: dict[str, str] = {}
    for text in strings_of(result):
        for cand in _local_candidates(text):
            cand = cand.strip().rstrip(".,;:)")
            if not cand or not kind_of(cand):
                continue
            try:
                if not os.path.isabs(cand) or not os.path.isfile(cand):
                    continue
                if min_mtime is not None and os.path.getmtime(cand) < min_mtime:
                    continue
            except OSError:
                continue
            found.setdefault(_norm(cand), os.path.abspath(cand))
            if len(found) >= limit:
                return list(found.values())
    return list(found.values())


def find_urls(result: Any, *, limit: int = 12) -> list[str]:
    """http(s) URLs in *result* whose path ends in a media extension."""
    out: list[str] = []
    for text in strings_of(result):
        for m in _URL.finditer(text):
            url = m.group(0).rstrip(".,;:")
            if kind_of(url) and url not in out:
                out.append(url)
                if len(out) >= limit:
                    return out
    return out


_CT_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif",
           "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov",
           "audio/mpeg": ".mp3", "audio/wav": ".wav", "model/gltf-binary": ".glb"}


def download(url: str, dest_dir: str | os.PathLike, stem: str = "media",
             timeout: float = 180.0) -> str | None:
    """Stream *url* into *dest_dir*; the saved path, or None on any failure."""
    import requests  # noqa: PLC0415
    try:
        d = Path(dest_dir)
        d.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            ext = os.path.splitext(urlparse(url).path)[1].lower()
            if ext not in MEDIA_EXTS:
                ext = _CT_EXT.get((r.headers.get("content-type") or "").split(";")[0].strip().lower(), "")
            if not ext:
                return None
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:40] or "media"
            dest = d / f"{safe}_{uuid.uuid4().hex[:8]}{ext}"
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return str(dest)
    except Exception:  # noqa: BLE001
        return None


def now() -> float:
    """Where a caller takes its *min_mtime* from, with a little slack for clocks."""
    return time.time() - 2.0
