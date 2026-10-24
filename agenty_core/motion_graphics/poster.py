"""The poster PNG that travels with a rendered video.

``render_scene`` writes ``<stem>.png`` next to the video. The PNG carries the whole scene in an
iTXt chunk ``mgscene`` (compressed) and the source scene's absolute path in a tEXt chunk
``mgscene_path``, so the video stays editable even when only the PNG is passed around.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

SCENE_KEY = "mgscene"
PATH_KEY = "mgscene_path"


def poster_frame(scene: dict) -> int:
    """``comp.poster`` when it is a frame inside the composition, else the middle frame."""
    comp = scene.get("comp") or {}
    duration = int(comp.get("duration") or 1)
    p = comp.get("poster")
    if isinstance(p, int) and not isinstance(p, bool) and 0 <= p < duration:
        return p
    return max(0, duration // 2)


def write_poster(png_path: str, scene: dict, scene_path: str) -> None:
    """Re-save the PNG at ``png_path`` with the scene embedded (atomic replace)."""
    info = PngInfo()
    info.add_itxt(SCENE_KEY, json.dumps(scene, ensure_ascii=False), zip=True)
    info.add_text(PATH_KEY, os.path.abspath(scene_path))
    tmp = png_path + ".tmp"
    with Image.open(png_path) as im:
        im.load()
        im.save(tmp, format="PNG", pnginfo=info)
    os.replace(tmp, png_path)


def read_scene_from_png(path: str) -> Optional[tuple]:
    """``(scene, scene_path)`` from a poster PNG, or None when it carries no scene."""
    try:
        with Image.open(path) as im:
            text = dict(getattr(im, "text", None) or im.info)
    except (OSError, ValueError):
        return None
    raw = text.get(SCENE_KEY)
    if not raw:
        return None
    try:
        scene = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(scene, dict):
        return None
    return scene, str(text.get(PATH_KEY) or "")
