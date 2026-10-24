"""Load files into the bEpic viewer (ComfyUI-ImageViewer) on the user's request.

"Show me clip 3 in the viewer", "open these frames side by side", "open the edit".
One call to the viewer's ``POST /bepic/push`` (reached through ComfyUI's HTTP API,
like every other core tool), which files the media in a tab, can bring the viewer
up, select or play the first item and open the Edit panel on a cut.

Paths must lie in folders the viewer may open (ComfyUI's input / output / temp and
the roots the user allowed); the viewer refuses the rest and says so in ``refused``.
A path may be an image, a video, a folder, an image sequence
(``frame_%04d.png`` / ``frame_####.png`` / ``frame_*.png``) or an ``.otio`` (which
stands for the film rendered beside it).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional

from agenty_core._compat import tool

logger = logging.getLogger("agenty_core.show_in_viewer")

DEFAULT_TAB = "agentY"

_tab_provider: Optional[Callable[[], str]] = None


def set_default_tab_provider(fn: Optional[Callable[[], str]]) -> None:
    """A host that lets the user rename the agent's viewer tab says how."""
    global _tab_provider
    _tab_provider = fn


def default_tab() -> str:
    if _tab_provider is not None:
        try:
            name = _tab_provider()
            if name:
                return name
        except Exception:  # noqa: BLE001
            pass
    return os.environ.get("AGENTY_VIEWER_TAB", "").strip() or DEFAULT_TAB


def open_in_viewer(
    paths,
    tab_name: str = "",
    group: str = "",
    label: str = "",
    focus: bool = True,
    play: bool = False,
    panel: str = "",
) -> dict:
    """Push *paths* to the viewer; returns the viewer's answer (``shown``,
    ``refused``, ``opened``, ``edit``, ``notes``) or ``{"error": ...}``. Never raises."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    clean = [os.path.abspath(str(p)) for p in (paths or []) if str(p or "").strip()]
    if not clean:
        return {"error": "no paths given"}
    body = {
        "paths": clean,
        "tab_name": " ".join(str(tab_name or "").split()) or default_tab(),
        "group": group or "",
        "label": label or "",
        "focus": bool(focus),
        "play": bool(play),
        "panel": str(panel or "").strip().lower(),
    }
    try:
        from agenty_core.utils.comfyui_client import get_client
        res = get_client().post("/bepic/push", json_data=body)
    except Exception as exc:  # noqa: BLE001
        if "404" in str(exc):
            return {"error": "the bEpic viewer is not installed in ComfyUI, or is too old for "
                             "show_in_viewer (update ComfyUI-ImageViewer)."}
        return {"error": f"could not reach the viewer through ComfyUI: {exc}"}
    return res if isinstance(res, dict) else {"error": f"unexpected answer from the viewer: {res!r}"}


@tool
def show_in_viewer(
    paths: list[str],
    tab_name: str = "",
    group: str = "",
    label: str = "",
    focus: bool = True,
    panel: str = "",
    play: bool = False,
) -> str:
    """Show files in the bEpic viewer, when the user asks to see them there.

    Use it for "show me clip 3 in the viewer", "open these frames side by side",
    "open the edit". Not for files the user did not ask to look at: generated
    pictures already reach the viewer by themselves.

    Args:
        paths: Files to load: images, videos, folders, image sequences
            (``frame_%04d.png`` or ``frame_*.png``), or an ``.otio`` cut (stands for
            the film beside it). Absolute paths inside ComfyUI's input/output/temp
            folders or a folder the user allowed in the viewer.
        tab_name: The viewer tab to put them in. Empty uses the agentY tab; a new
            name makes a new tab (e.g. ``"side by side"``).
        group: Optional heading the files sit under in the tab's history strip.
        label: Optional caption on the thumbnail.
        focus: Bring the viewer up (even when closed or docked) and select the
            first item. Default True.
        panel: ``"edit"`` opens the Edit panel on the cut (the ``.otio`` beside the
            film), so the user sees the edit shot by shot. Needs a film with an
            ``.otio`` beside it. Empty for none.
        play: Also start playing the first item (ignored with panel ``"edit"``).

    Returns:
        JSON: ``shown`` (how many items were loaded), ``motion`` (scenes opened in the Motion panel), ``refused`` (paths the viewer
        may not open or that are not media), ``opened`` (viewer brought up),
        ``edit`` (Edit panel opened), ``notes`` and ``error`` when it failed.
    """
    res = open_in_viewer(paths, tab_name, group, label, focus, play, panel)
    if "error" not in res and not res.get("shown") and not res.get("motion"):
        res["error"] = "nothing was shown: none of the paths is a media file the viewer may open"
    return json.dumps(res)
