"""Editable motion graphics: write a scene (data), check it, look at it, render it.

The agent does NOT write React for motion graphics any more. It writes a scene
description (``*.mgscene.json``, format in ``agenty_core/motion_graphics`` and the
``motion-graphics-scene`` skill); one generic Remotion composition renders any
scene, and the bEpic viewer edits the same file (layers, timeline, keyframes,
curves) so the user and the agent change one thing. The viewer's preview and the
render share one module, so what the editor shows is what comes out.

Tools: ``mg_scene_validate`` (instant, no Node), ``mg_scene_preview_frame`` (PNG
stills / a contact sheet to look at), ``mg_render`` (the video). The first render
installs the renderer's npm packages once (Node 18+ on PATH).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from agenty_core._compat import tool
from agenty_core.motion_graphics import render as _render
from agenty_core.motion_graphics import schema as _schema
from agenty_core.utils.progress_signal import push as _push_progress

logger = logging.getLogger("agenty_core.motion_graphics")


def _log(msg: str) -> None:
    if msg:
        logger.info("mg: %s", msg)


def _say(msg: str) -> None:
    """A line in the chat while a long step runs; never fails the tool."""
    try:
        _push_progress(msg)
    except Exception:  # noqa: BLE001
        pass


def _summary(scene: dict) -> dict:
    comp = scene.get("comp") or {}
    layers = scene.get("layers") or []
    fps = comp.get("fps") or 30
    return {
        "size": [comp.get("width"), comp.get("height")],
        "fps": fps,
        "frames": comp.get("duration"),
        "seconds": round((comp.get("duration") or 0) / fps, 2),
        "layers": [f"{l.get('id')} ({l.get('type')})" for l in layers if isinstance(l, dict)],
    }


@tool
def mg_scene_validate(scene_path: str) -> str:
    """Check a motion-graphics scene file (``*.mgscene.json``) without rendering it.

    Instant and needs nothing installed. Run it after writing or editing a scene,
    before ``mg_scene_preview_frame`` / ``mg_render``.

    Args:
        scene_path: Absolute path of the scene file.

    Returns:
        JSON: ``ok``; ``errors`` (each ``code``, ``path`` into the file, ``message``
        - fix these, the scene cannot render); ``warnings`` (probably not what was
        meant); ``summary`` (size, fps, frames, seconds, layer ids).
    """
    path = os.path.abspath(os.path.expanduser(str(scene_path or "").strip().strip('"')))
    try:
        scene = _schema.load_scene(path)
    except ValueError as exc:
        return json.dumps({"ok": False, "errors": [{"code": "E_FILE", "path": "", "message": str(exc)}], "warnings": []})
    res = _schema.validate_scene(scene)
    if isinstance(scene, dict):
        res["summary"] = _summary(scene)
    res["path"] = path
    return json.dumps(res)


@tool
def mg_scene_preview_frame(scene_path: str, frames: Optional[list[int]] = None, scale: float = 0.5) -> str:
    """Render still frames of a scene so you can LOOK at them (then ``view_image`` the result).

    Several frames come back as one numbered contact sheet - the cheapest way to
    check a whole animation (e.g. the start, the middle of each move, the end).

    Args:
        scene_path: Absolute path of the ``*.mgscene.json`` file.
        frames: Comp frames to render (default: first, middle, last). At most 16.
        scale: 0.1-1.0 of the composition size (default 0.5; stills are PNG).

    Returns:
        JSON: ``sheet`` (path of the contact sheet, when several frames) and
        ``frames`` (each ``frame`` and ``path``); ``error`` when it failed.
    """
    path = os.path.abspath(os.path.expanduser(str(scene_path or "").strip().strip('"')))
    try:
        scene = _schema.load_scene(path)
        res = _schema.validate_scene(scene)
        if not res["ok"]:
            return json.dumps({"error": "the scene is not valid", "errors": res["errors"][:8]})
        dur = scene["comp"]["duration"]
        want = [int(f) for f in (frames or [0, dur // 2, dur - 1])][:16]
        want = [max(0, min(dur - 1, f)) for f in want]
        scale = max(0.1, min(1.0, float(scale or 0.5)))
        _say("🎞️ rendering preview frame(s) of " + os.path.basename(path))
        stills = _render.render_stills(path, want, scale=scale, log=_log)
        out: dict = {"frames": [{"frame": s["frame"], "path": s["out"]} for s in stills]}
        if len(stills) > 1:
            from agenty_core.utils.contact_sheet import build_contact_sheet
            sheet = build_contact_sheet([s["out"] for s in stills], [f"f{s['frame']}" for s in stills], cell=640,
                                        out_path=os.path.join(os.path.dirname(stills[0]["out"]),
                                                              _render._stem(path) + "_sheet.jpg"))
            out["sheet"] = sheet["path"]
        return json.dumps(out)
    except (_render.RenderError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("mg_scene_preview_frame failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@tool
def mg_render(scene_path: str, out_path: str = "", format: str = "mp4", frame_start: Optional[int] = None,
              frame_end: Optional[int] = None, scale: float = 1.0, crf: int = 18, poster: bool = True) -> str:
    """Render a motion-graphics scene (``*.mgscene.json``) to a video file.

    Validates first (a scene with errors is not rendered). Blocking: a few
    seconds of 1080p take roughly a minute; the first call after install or after
    the renderer changed takes longer (npm install, bundling, browser download).

    Args:
        scene_path: Absolute path of the scene file.
        out_path: Output file. Default: next to the scene, ``<name>.<format>``.
        format: ``mp4`` (h264), ``webm`` (vp9), ``mov`` (ProRes 4444, keeps
            transparency when the scene's ``comp.background`` is null) or ``gif``.
        frame_start: First frame to render (default 0).
        frame_end: Last frame to render, inclusive (default: the last).
        scale: Render scale, 1.0 = the composition size.
        crf: Quality for mp4/webm, lower is better (default 18).
        poster: Also write ``<name>.png`` next to the video: a poster still (``comp.poster`` frame, else the
            middle) that carries the whole scene, so the PNG alone reopens in the Motion panel (default True).

    Returns:
        JSON: ``path`` of the video, ``frames``, ``width``, ``height``, ``fps``,
        ``bytes``, ``seconds`` (the render took), ``poster`` (the PNG path; or ``poster_error`` if only the
        still failed, the video is fine); or ``error``.
    """
    try:
        rng = None
        if frame_start is not None or frame_end is not None:
            rng = (frame_start or 0, frame_end if frame_end is not None else 10 ** 9)
            if frame_end is None:
                scene = _schema.load_scene(os.path.abspath(scene_path))
                rng = (frame_start or 0, int(scene["comp"]["duration"]) - 1)
        _say("🎬 rendering " + os.path.basename(str(scene_path)) + " (Remotion)")
        res = _render.render_scene(scene_path, out_path, format=format, frame_range=rng, scale=float(scale or 1.0),
                                   crf=int(crf), log=_log, poster=bool(poster))
        res.pop("event", None)
        return json.dumps(res)
    except (_render.RenderError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("mg_render failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
