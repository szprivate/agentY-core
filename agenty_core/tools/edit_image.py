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
import re

from agenty_core._compat import tool

PAINT_EXTS = (".ora", ".psd")
PICTURE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".tga", ".gif", ".exr", ".hdr", ".dpx")
#: Ops that name a file the viewer reads.
_FILE_KEYS = ("file", "mask_file")
#: A Paint session's token, as the viewer's state names it.
_TOKEN = re.compile(r"[0-9a-f]{32}")


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _guard(path: str) -> str:
    from agenty_core.utils.path_guard import refuse_read
    return refuse_read(path, "edit_image")


def _open_doc() -> tuple[str, str, bool]:
    """``(the picture open in Paint, why not, live)``.

    A viewer that names the tab's session (``token``) is worked on LIVE: the viewer asks the page for what it
    has not saved before the change and saves the result, so an unsaved picture - a new canvas too - needs no
    saving first. An older viewer (no token) is refused while the picture has unsaved changes.
    """
    from agenty_core.tools.viewer_state import read_state
    try:
        got = json.loads(read_state())
    except Exception as e:  # noqa: BLE001
        return "", f"the viewer's state could not be read ({e})", False
    if got.get("status") != "ok":
        return "", got.get("error") or "the viewer's state could not be read", False
    p = (got.get("state") or {}).get("paint") or {}
    if not p.get("open"):
        return "", "no picture is open in the viewer's Paint tab - pass doc= (an .ora, .psd or picture path)", False
    if p.get("empty"):
        return "", "the Paint tab has no picture in it yet - pass doc=, or show_in_viewer(paths=[...], panel='paint') first", False
    tok = str(p.get("token") or "")
    if _TOKEN.fullmatch(tok):
        return tok, "", True
    if p.get("dirty"):
        return "", ("the picture in Paint has changes the user has not saved, and this viewer is too old to hand "
                    "them over - ask the user to save it (Ctrl+S in Paint) first, or update ComfyUI-ImageViewer"), False
    doc = p.get("path") or p.get("source") or ""
    if not doc:
        return "", "the picture in Paint has no file yet - ask the user to save it (Ctrl+S in Paint), or pass doc=", False
    return str(doc), "", False


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
def edit_image(doc: str = "", ops: list | None = None, snapshot: bool = False) -> str:
    """Read the picture open in the ComfyUI Image Viewer's Paint tab (or any .ora / .psd / picture), and
    change it in LAYERS: add a picture or a colour as a layer, add an adjustment layer, move,
    hide, blend, mask, select, fill, heal, blur, crop, resize, add text; group layers, clip them,
    give them styles and smart filters; pen paths, shapes, gradients, perspective, guides.

    Use it to put generated pixels back where the user is painting (generative fill, remove an
    object, background removal, outpaint, upscale - the Paint tab asks you with the paths), and
    for "put the logo top right", "warm up the sky", "add a title". The viewer saves the change
    and the open Paint tab shows it at once (one Ctrl+Z undoes it). Never write the .ora yourself.

    The picture open in Paint (doc="") is worked on as it is ON SCREEN, unsaved changes included,
    and saved with the change - never ask the user to save first. A picture you GENERATE for it
    (an image edit of the whole picture, a fill) goes back as a NEW LAYER: ``add_layer`` with
    ``file`` and ``fit: "canvas"`` (or x / y / w / h of the selection's box, ``mask: "selection"``).
    Generate at the canvas's size (the viewer line says ``paint=<name> <w>x<h>``).

    Args:
        doc: The .ora / .psd / picture path; "" for the picture open in Paint (live, as above). A
            plain picture or a .psd is never overwritten: the first change makes a new .ora (its
            path is returned).
        ops: Leave empty to READ the document (its size, layers from the top, selection box). To
            change it, a list (all done, or none). Layers are named by number from 1 at the TOP,
            by name, or "active". Colours are "#rrggbb" or [r, g, b]. Pixels, top-left origin.
            ``{"op": "add_layer", "file": <picture>, "name": "", "fit": "none|canvas|contain|cover",
              "x": .., "y": .., "w": .., "h": .., "opacity": 100, "blend": "normal",
              "mask": "selection", "mask_file": <white = shown>, "above"/"below": <layer>, "into": <group>}`` -
              a picture as a layer (centred unless x/y); no file: an empty / colour layer (``color``).
              ``into`` puts a new layer at the top of that group (also on add_adjustment, add_text, shape).
              A picture from a file comes in as a SMART layer: it keeps the whole original, so a later
              transform / distort / resize redraws it from that at full resolution. ``"smart": false`` for a
              plain one. Painting into a smart layer (fill, clear, filter, heal, gradient) makes it plain (``said``
              says so); ``{"op": "rasterize", "layer": ..}`` does that on purpose.
            ``{"op": "add_adjustment", "type": "levels|curves|brightness_contrast|exposure|hue_saturation|grade|invert|threshold", "params": {..}}``
            ``{"op": "set_layer", "layer": 2, "name", "visible", "opacity" (0-100), "blend", "x", "y", "locked",
              "clip": true|false, "lock_alpha": true|false, "collapsed" (a group), "style": {..}|null, "filters": [..]|null}``
              blend: normal multiply screen overlay darken lighten difference add color_dodge color_burn soft_light hard_light
              (a group also pass_through). ``clip``: clipped to the nearest unclipped layer under it.
              ``style`` (pixel layer; merged into what it has, an effect given as null is taken off, null takes all off):
              ``{"shadow": {"color", "opacity" 0-1, "angle", "distance", "size"}, "glow": {"color", "opacity", "size"},
              "stroke": {"color", "opacity", "size"}, "overlay": {"color", "opacity"}}`` (each also ``"on"``).
              ``filters`` (pixel layer): smart filters kept as settings, the whole list in order -
              ``[{"type": "blur", "params": {"radius"}}, {"type": "sharpen", "params": {"radius", "amount"}},
              {"type": <adjustment type>, "params": {..}}, ..]`` (each may have ``"on": false``); null takes them off.
            ``{"op": "remove_layer" | "duplicate_layer" | "merge_down", "layer": ..}``, ``{"op": "flatten"}``,
            ``{"op": "move_layer", "layer": .., "to": "top|bottom|up|down|<number>", "into": <group>|"none"}``
            ``{"op": "group", "layers": [..] (or "layer"), "name", "blend": "pass_through", "opacity"}`` -
              the layers into a new group; ``{"op": "ungroup" | "merge_group", "layer": <group>}``
            ``{"op": "select", "mode": "rect|ellipse|polygon|all|none|invert|layer|file", "x", "y", "w", "h",
              "points": [[x, y], ..], "file": <mask>, "combine": "replace|add|subtract|intersect", "feather": px}``
              Also changing the selection: ``"mode": "expand|contract|smooth|border", "by": px``, and
              ``"mode": "color_range", "color": .., "fuzziness": 1-255, "sample": "picture|layer", "layer"``.
            ``{"op": "mask", "layer": .., "from": "selection|file|layer|all|invert|on|off|remove|settings", "file": <mask>}``
              ``settings``: ``"density"`` (0-100) and / or ``"feather"`` (px), kept on the mask, not baked in.
              ``layer``: ``"source": <pixel layer>`` becomes the mask (its brightness where it is opaque, as
              dropping a layer on another in Layers does); the source layer goes, unless ``"keep": true``.
            ``{"op": "fill", "layer": .., "color": ..}`` / ``{"op": "clear", "layer": ..}`` (inside the selection, or all)
            ``{"op": "heal", "layer": ..}`` - fill the selection from its surroundings (no AI)
            ``{"op": "filter", "layer": .., "type": "blur|sharpen|<adjustment type>", "radius", "amount", "params"}`` (baked into the pixels)
            ``{"op": "crop", "x", "y", "w", "h"}`` or ``{"op": "crop", "to": "selection"}``;
            ``{"op": "canvas_size", "w", "h", "anchor": "center|top_left|..."}``; ``{"op": "resize", "scale": 2}`` or w / h;
            ``{"op": "transform", "layer": .., "flip": "h|v", "rotate": 90|180|270, "scale": 0.5}``
            ``{"op": "add_text", "text": "..", "font": "Arial", "size": 64, "color": "#ffffff", "bold", "italic",
              "align": "left|center|right", "x", "y"}``
            ``{"op": "path", "points": [[x, y] | [x, y, in_x, in_y, out_x, out_y], ..], "name", "closed": true,
              "do": "keep|select|fill|stroke|mask|remove", "layer", "color", "width", "combine", "feather"}`` -
              a pen path (bezier handles absolute), kept in the .ora by name; or ``"path": <name>`` to use one kept.
            ``{"op": "shape", "kind": "rect|rounded|ellipse|polygon|line", "box": [x, y, w, h] (a line: "from", "to"),
              "fill": .., "stroke": .., "width", "radius" (rounded), "sides" (polygon), "name", "as_path": false}`` -
              a new layer (or a kept path with as_path)
            ``{"op": "gradient", "layer": .., "type": "linear|radial|angle|reflected|diamond", "from": [x, y], "to": [x, y],
              "colors": [.., ..] or "stops": [{"t": 0-1, "color"}], "reverse", "opacity"}`` (inside the selection, or all)
            ``{"op": "distort", "layer": .., "corners": [[x, y] x4]}`` - the layer's corners (top left, top right,
              bottom right, bottom left) put there: perspective / distort; convex only
            ``{"op": "guides", "add": [{"axis": "x|y", "pos"}], "remove": [..], "clear": true}``

        snapshot: With no ops: also write the picture AS IT LOOKS and the selection (white = selected) as
            PNGs in ComfyUI's input folder - ``snapshot: {picture, selection (null: nothing selected), box, w, h}``.
            Use it when the user asks in chat for a generative fill, a removal, an extend of the picture
            open in Paint ("fill the selection with a cat"): run the inpaint / outpaint workflow on those
            files, then put the result back with ``add_layer`` ``mask: "selection"`` - as when they press
            Generate.

    Returns JSON ``{status, doc, written, outline: {w, h, layer_count, layers: [{layer, name, kind
    (pixel|adjustment|text|group), visible, opacity, blend?, x?, y?, w?, h?, type?, params?, mask?, active?,
    locked?, in_group?, layers_inside?, clipped?, mask_density?, mask_feather?, style?, lock_alpha?, filters?,
    smart? ({source: [w, h], scale: [x %, y %], corners})}], selection, paths?, guides?},
    said, pushed}`` - ``doc`` is where the document now is (a new .ora for a picture / .psd).
    """
    from agenty_core.utils import canvas_scope as _cs
    ops, err = _clean_ops(ops)
    if err:
        return _j({"status": "error", "error": err})
    doc = str(doc or "").strip()
    live = False
    if not doc:
        if _cs.detached():
            return _j({"status": "error", "error": _cs.NO_CANVAS + " Pass doc= (the .ora / .psd / picture path)."})
        doc, why, live = _open_doc()
        if not doc:
            return _j({"status": "error", "error": why})
    if not live:
        ext = os.path.splitext(doc)[1].lower()
        if ext not in PAINT_EXTS + PICTURE_EXTS:
            return _j({"status": "error", "error": f"{os.path.basename(doc)} is not an .ora, .psd or picture"})
    for path in ([] if live else [doc]) + [str(o.get(k)) for o in ops if isinstance(o, dict) for k in _FILE_KEYS if o.get(k)]:
        why = _guard(path)
        if why:
            return _j({"status": "error", "error": why})
    from urllib.parse import urlencode
    from agenty_core.utils.comfyui_client import get_client
    cid = _cs.client_id()
    route = "/imageviewer/paint/doc" + (("?" + urlencode({"client_id": cid})) if cid else "")
    try:
        r = get_client().post(route, json_data={"doc": doc, "ops": ops, **({"sync": True} if live else {}),
                                                **({"snapshot": True} if snapshot and not ops else {})})
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
    out = {"status": "ok", "doc": r.get("path") or ("" if live else doc), "outline": r.get("outline") or {}}
    if live:
        # The tab's own picture, as it is on screen (unsaved changes included); saved when changed.
        out["live"] = True
    if ops:
        out["written"] = bool(r.get("written"))
        out["said"] = [str(s)[:200] for s in (r.get("said") or [])][:60]
        out["pushed"] = bool(r.get("token"))
    if snapshot and not ops:
        snap = r.get("snapshot")
        if isinstance(snap, dict) and snap.get("picture"):
            out["snapshot"] = {k: snap.get(k) for k in ("picture", "selection", "box", "w", "h")}
        else:
            out["snapshot_error"] = ("the viewer in this ComfyUI does not hand over a snapshot yet - update "
                                     "ComfyUI-ImageViewer under <ComfyUI>/custom_nodes/ and restart ComfyUI")
    elif snapshot:
        out["snapshot_error"] = "snapshot is for a read (no ops); read again after the change"
    if r.get("notes"):
        out["notes"] = [str(n)[:200] for n in r["notes"]][:10]
    return _j(out)
