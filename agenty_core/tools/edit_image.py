"""Read and change a picture in layers in the ComfyUI Image Viewer's Paint tab: ``edit_image``.

"Put the logo in the top right corner", "make the sky warmer", "add a title", "mask the
background out", "put the inpainted result back as a layer". The picture is the one open in
Paint (the viewer's state names it) or any other: an ``.ora`` (Paint's own file, layers kept),
a ``.psd`` or a plain picture (which becomes a new ``.ora`` the first time it is changed).

The viewer does the work (``POST /imageviewer/paint/doc``, reached through ComfyUI's HTTP API like
every other core tool): it changes the document the way the Paint tab would, saves it, and tells
the open tab, which takes the change in as ONE undo step. Never write the .ora by hand
(run_script): the tab would not see the change and its next save would drop it.

Layers are numbered from 1 at the TOP, as the Layers panel shows them; a layer can also be
named by its name, or "active" (the one selected in the panel).
"""

from __future__ import annotations

import json
import os

from agenty_core._compat import tool

PAINT_EXTS = (".ora", ".psd")
PICTURE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".tga", ".gif", ".exr", ".hdr", ".dpx")
#: Ops that name a file the viewer reads.
_FILE_KEYS = ("file", "mask_file")


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _guard(path: str) -> str:
    from agenty_core.utils.path_guard import refuse_read
    return refuse_read(path, "edit_image")


def _open_doc() -> tuple[str, str]:
    """``(the picture open in Paint, why not)``. Refuses while it has unsaved changes."""
    from agenty_core.tools.viewer_state import read_state
    try:
        got = json.loads(read_state())
    except Exception as e:  # noqa: BLE001
        return "", f"the viewer's state could not be read ({e})"
    if got.get("status") != "ok":
        return "", got.get("error") or "the viewer's state could not be read"
    p = (got.get("state") or {}).get("paint") or {}
    if not p.get("open"):
        return "", "no picture is open in the viewer's Paint tab - pass doc= (an .ora, .psd or picture path)"
    if p.get("empty"):
        return "", "the Paint tab has no picture in it yet - pass doc=, or show_in_viewer(paths=[...], panel='paint') first"
    if p.get("dirty"):
        return "", ("the picture in Paint has changes the user has not saved - ask the user to save it "
                    "(Ctrl+S in Paint) first, so that nothing of theirs is lost")
    doc = p.get("path") or p.get("source") or ""
    if not doc:
        return "", "the picture in Paint has no file yet - ask the user to save it (Ctrl+S in Paint), or pass doc="
    return str(doc), ""


def _clean_ops(ops):
    if ops is None:
        return [], ""
    if isinstance(ops, str):
        s = ops.strip()
        if not s:
            return [], ""
        try:
            ops = json.loads(s)
        except ValueError:
            return None, "ops is a list of changes (JSON)"
    if isinstance(ops, dict):
        ops = [ops]
    if not isinstance(ops, list):
        return None, "ops is a list of changes"
    return ops, ""


@tool
def edit_image(doc: str = "", ops: list | None = None) -> str:
    """Read the picture open in the ComfyUI Image Viewer's Paint tab (or any .ora / .psd / picture), and
    change it in LAYERS: add a picture or a colour as a layer, add an adjustment layer, move,
    hide, blend, mask, select, fill, heal, blur, crop, resize, add text.

    Use it to put generated pixels back where the user is painting (generative fill, remove an
    object, background removal, outpaint, upscale - the Paint tab asks you with the paths), and
    for "put the logo top right", "warm up the sky", "add a title". The viewer saves the change
    and the open Paint tab shows it at once (one Ctrl+Z undoes it). Never write the .ora yourself.

    Args:
        doc: The .ora / .psd / picture path; "" for the picture open in Paint. A plain picture or
            a .psd is never overwritten: the first change makes a new .ora (its path is returned).
        ops: Leave empty to READ the document (its size, layers from the top, selection box). To
            change it, a list (all done, or none). Layers are named by number from 1 at the TOP,
            by name, or "active". Colours are "#rrggbb" or [r, g, b]. Pixels, top-left origin.
            ``{"op": "add_layer", "file": <picture>, "name": "", "fit": "none|canvas|contain|cover",
              "x": .., "y": .., "w": .., "h": .., "opacity": 100, "blend": "normal",
              "mask": "selection", "mask_file": <white = shown>, "above"/"below": <layer>}`` -
              a picture as a layer (centred unless x/y); no file: an empty / colour layer (``color``).
            ``{"op": "add_adjustment", "type": "levels|curves|brightness_contrast|exposure|hue_saturation|grade|invert|threshold", "params": {..}}``
            ``{"op": "set_layer", "layer": 2, "name", "visible", "opacity" (0-100), "blend", "x", "y", "locked"}``
              blend: normal multiply screen overlay darken lighten difference add color_dodge color_burn soft_light hard_light.
            ``{"op": "remove_layer" | "duplicate_layer" | "merge_down", "layer": ..}``, ``{"op": "flatten"}``,
            ``{"op": "move_layer", "layer": .., "to": "top|bottom|up|down|<number>"}``
            ``{"op": "select", "mode": "rect|ellipse|polygon|all|none|invert|layer|file", "x", "y", "w", "h",
              "points": [[x, y], ..], "file": <mask>, "combine": "replace|add|subtract|intersect", "feather": px}``
            ``{"op": "mask", "layer": .., "from": "selection|file|all|invert|on|off|remove", "file": <mask>}``
            ``{"op": "fill", "layer": .., "color": ..}`` / ``{"op": "clear", "layer": ..}`` (inside the selection, or all)
            ``{"op": "heal", "layer": ..}`` - fill the selection from its surroundings (no AI)
            ``{"op": "filter", "layer": .., "type": "blur|sharpen|<adjustment type>", "radius", "amount", "params"}`` (baked into the pixels)
            ``{"op": "crop", "x", "y", "w", "h"}`` or ``{"op": "crop", "to": "selection"}``;
            ``{"op": "canvas_size", "w", "h", "anchor": "center|top_left|..."}``; ``{"op": "resize", "scale": 2}`` or w / h;
            ``{"op": "transform", "layer": .., "flip": "h|v", "rotate": 90|180|270, "scale": 0.5}``
            ``{"op": "add_text", "text": "..", "font": "Arial", "size": 64, "color": "#ffffff", "bold", "italic",
              "align": "left|center|right", "x", "y"}``

    Returns JSON ``{status, doc, written, outline: {w, h, layer_count, layers: [{layer, name, kind,
    visible, opacity, blend?, x?, y?, w?, h?, type?, params?, mask?, active?, locked?}], selection},
    said, pushed}`` - ``doc`` is where the document now is (a new .ora for a picture / .psd).
    """
    from agenty_core.utils import canvas_scope as _cs
    ops, err = _clean_ops(ops)
    if err:
        return _j({"status": "error", "error": err})
    doc = str(doc or "").strip()
    if not doc:
        if _cs.detached():
            return _j({"status": "error", "error": _cs.NO_CANVAS + " Pass doc= (the .ora / .psd / picture path)."})
        doc, why = _open_doc()
        if not doc:
            return _j({"status": "error", "error": why})
    ext = os.path.splitext(doc)[1].lower()
    if ext not in PAINT_EXTS + PICTURE_EXTS:
        return _j({"status": "error", "error": f"{os.path.basename(doc)} is not an .ora, .psd or picture"})
    for path in [doc] + [str(o.get(k)) for o in ops if isinstance(o, dict) for k in _FILE_KEYS if o.get(k)]:
        why = _guard(path)
        if why:
            return _j({"status": "error", "error": why})
    from urllib.parse import urlencode
    from agenty_core.utils.comfyui_client import get_client
    cid = _cs.client_id()
    route = "/imageviewer/paint/doc" + (("?" + urlencode({"client_id": cid})) if cid else "")
    try:
        r = get_client().post(route, json_data={"doc": doc, "ops": ops})
    except Exception as e:  # noqa: BLE001
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
        said = ""
        if resp is not None:
            try:
                said = resp.json().get("error") or ""
            except Exception:  # noqa: BLE001
                said = ""
        if said:
            return _j({"status": "error", "error": str(said)[:400]})
        if status == 404 or (status is None and "404" in str(e)):
            return _j({"status": "error", "error": (
                "the viewer in this ComfyUI predates Paint (no /imageviewer/paint/doc route) - update "
                "ComfyUI-ImageViewer under <ComfyUI>/custom_nodes/ and restart ComfyUI.")})
        return _j({"status": "error", "error": f"could not reach ComfyUI ({str(e)[:160]}). Is it running?"})
    if not isinstance(r, dict) or r.get("error"):
        return _j({"status": "error", "error": str((r or {}).get("error") if isinstance(r, dict) else r)[:400]})
    out = {"status": "ok", "doc": r.get("path") or doc, "outline": r.get("outline") or {}}
    if ops:
        out["written"] = bool(r.get("written"))
        out["said"] = [str(s)[:200] for s in (r.get("said") or [])][:60]
        out["pushed"] = bool(r.get("token"))
    if r.get("notes"):
        out["notes"] = [str(n)[:200] for n in r["notes"]][:10]
    return _j(out)
