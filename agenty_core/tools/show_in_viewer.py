"""Load files into the ComfyUI Image Viewer (ComfyUI-ImageViewer) on the user's request.

"Show me clip 3 in the viewer", "open these frames side by side", "open the edit".
One call to the viewer's ``POST /imageviewer/push`` (reached through ComfyUI's HTTP API,
like every other core tool), which files the media in a tab, can bring the viewer
up, select or play the first item and open the Edit panel on a cut.

Paths must lie in folders the viewer may open (ComfyUI's input / output / temp and
the roots the user allowed); the viewer refuses the rest and says so in ``refused``.
A path may be an image, a video, a folder, an image sequence
(``frame_%04d.png`` / ``frame_####.png`` / ``frame_*.png``) or an ``.otio`` (which
stands for the film rendered beside it; an ``.otio`` with no film beside it is opened
as a cut in the Edit panel).
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
        # The group of the viewer's Viewer tab; tab_name is its old name.
        "group": " ".join(str(group or tab_name or "").split()) or default_tab(),
        "label": label or "",
        "focus": bool(focus),
        "play": bool(play),
        "panel": str(panel or "").strip().lower(),
    }
    try:
        from agenty_core.utils.comfyui_client import get_client
        res = get_client().post("/imageviewer/push", json_data=body)
    except Exception as exc:  # noqa: BLE001
        if "404" in str(exc):
            return {"error": "the ComfyUI Image Viewer is not installed in ComfyUI, or is too old for "
                             "show_in_viewer (update ComfyUI-ImageViewer)."}
        return {"error": f"could not reach the viewer through ComfyUI: {exc}"}
    return res if isinstance(res, dict) else {"error": f"unexpected answer from the viewer: {res!r}"}


@tool
def show_in_viewer(
    paths: list[str],
    group: str = "",
    label: str = "",
    focus: bool = True,
    panel: str = "",
    play: bool = False,
    tab_name: str = "",
) -> str:
    """Show files in the ComfyUI Image Viewer, when the user asks to see them there.

    Use it for "show me clip 3 in the viewer", "open these frames side by side",
    "open the edit". Not for files the user did not ask to look at: generated
    pictures already reach the viewer by themselves.

    Args:
        paths: Files to load: images, videos, folders, image sequences
            (``frame_%04d.png`` or ``frame_*.png``), or an ``.otio`` cut (stands for
            the film beside it; with no film beside it, the cut itself is opened in the
            Edit panel). Absolute paths inside ComfyUI's input/output/temp
            folders or a folder the user allowed in the viewer.
        group: The group of the viewer's Viewer tab to put them in (a heading in
            its History strip). Empty uses the agentY group; a new name makes a new
            group (e.g. ``"side by side"``).
        label: Optional caption on the thumbnail.
        focus: Bring the viewer up (even when closed or docked) and select the
            first item. Default True.
        panel: ``"edit"`` opens the Edit panel on the cut (the ``.otio`` beside the
            film), so the user sees the edit shot by shot. Needs a film with an
            ``.otio`` beside it. ``"paint"`` opens the pictures in Paint (layers, a brush,
            masks - change them with edit_image); an ``.ora`` / ``.psd`` always opens there.
            ``"layer"`` puts each picture as a NEW LAYER on top of the picture open in Paint,
            scaled to its canvas (what a generation made for that picture should do; the
            picture's unsaved changes are kept and it is saved). Empty for none.
        play: Also start playing the first item (ignored with panel ``"edit"``).
        tab_name: Old name of ``group``; used when ``group`` is empty.

    Returns:
        JSON: ``shown`` (how many items were loaded), ``motion`` (scenes opened in the Motion panel), ``cuts`` (``.otio`` cuts opened in the Edit panel), ``paint`` (pictures opened in Paint), ``paint_layers`` (pictures put as layers, panel ``"layer"``), ``refused`` (paths the viewer
        may not open or that are not media), ``opened`` (viewer brought up),
        ``edit`` (Edit panel opened), ``notes`` and ``error`` when it failed.
    """
    if str(panel or "").strip().lower() in ("layer", "layers", "paint_layer"):
        return _as_layers(paths, label)
    res = open_in_viewer(paths, tab_name, group, label, focus, play, panel)
    if "error" not in res and not res.get("shown") and not res.get("motion") and not res.get("cuts") and not res.get("paint"):
        res["error"] = "nothing was shown: none of the paths is a media file the viewer may open"
    return json.dumps(res)


def _as_layers(paths, label: str = "") -> str:
    """show_in_viewer(panel="layer"): the pictures as new layers of the picture open in Paint (edit_image)."""
    from agenty_core.tools.edit_image import PICTURE_EXTS, edit_image
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files = [os.path.abspath(str(p)) for p in (paths or []) if str(p or "").strip()]
    pics = [f for f in files if os.path.splitext(f)[1].lower() in PICTURE_EXTS]
    refused = [f for f in files if f not in pics]
    if not pics:
        return json.dumps({"error": "panel='layer' takes pictures (png, jpg, webp, exr ...)", "refused": refused})
    ops = [{"op": "add_layer", "file": f, "fit": "canvas",
            "name": (label if label and len(pics) == 1 else os.path.splitext(os.path.basename(f))[0])[:60]}
           for f in pics]
    got = json.loads(edit_image(ops=ops))
    if got.get("status") != "ok":
        return json.dumps({"error": got.get("error") or "the pictures were not put into Paint", "refused": refused})
    return json.dumps({"paint_layers": len(pics), "doc": got.get("doc"), "said": got.get("said", []),
                       "refused": refused, "layers": len((got.get("outline") or {}).get("layers") or [])})
