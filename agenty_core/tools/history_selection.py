"""What the user picked in a history panel, as files the agent can use.

Two panels keep a history the user can pick from, and the agentY-comfyuiConnect
page code reads both (``web/agent_history_selection.js``):

* the **bEpic viewer's history strip** - snapshots Ctrl-selected, the two pinned
  in a compare, or one older snapshot clicked open. Each comes with its file (or
  ComfyUI's filename/subfolder/type for it), its kind, the label and group its
  sender gave it, the run (``prompt_id``) that made it and its agentY ``#tag``.
* **ComfyUI's Media Assets / queue sidebar** - only asset ids. In ComfyUI's
  history mode an id is the run's ``prompt_id`` (the newest output of that run),
  ``<prompt_id>-["<node>","<subfolder>","<file>"]`` for one output of an
  expanded run, or ``input-<n>-<file>`` for a file in the input folder. Those
  are resolved here; any other id (the asset service's own) is passed on as is.

The page answers with ``{"viewer": {...}, "assets": {...}}`` (either may be
missing); :func:`resolve` turns that into one flat list of items with an
absolute ``path`` wherever one can be found, and the role/title from the file's
``.agenty.json`` sidecar. A path is what makes a pick usable: it goes into a
LoadImage the way a ``#tag``'s file does.

Two callers: the ``get_history_selection`` tool (a live read through ComfyUI's
``/agent/history_selection`` route), and the agentY host, which receives the
same data with every chat message and lists it at the top of the turn
(:func:`describe_lines`).
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from agenty_core._compat import tool

SIDECAR_SUFFIX = ".agenty.json"
# The most items one answer carries; the count says how many there were.
MAX_ITEMS = 50
# Most items listed in the turn's context block; the rest are counted.
BLOCK_ITEMS = 12

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_RUN_ID = re.compile(rf"^({_UUID})$")
_RUN_OUTPUT_ID = re.compile(rf"^({_UUID})-(\[.*\])$")
_INPUT_ID = re.compile(r"^input-\d+-(.+)$")

_VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
_MODEL_EXT = {".glb", ".gltf", ".obj", ".fbx", ".ply", ".spz", ".splat", ".usd", ".usda",
              ".usdc", ".usdz", ".stl"}
_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# ComfyUI's folders by type, cached briefly: asked once per pick, every turn.
_DIRS_TTL = 60.0
_dirs_lock = threading.Lock()
_dirs_cache: tuple[float, dict] = (0.0, {})


def _kind_of(name: str, given: str = "") -> str:
    g = str(given or "").lower()
    if g in ("image", "video", "model", "audio"):
        return g
    ext = Path(str(name or "")).suffix.lower()
    if ext in _MODEL_EXT:
        return "model"
    if ext in _VIDEO_EXT:
        return "video"
    if ext in _AUDIO_EXT:
        return "audio"
    return "image"


def _type_dirs() -> dict:
    """``{"input": dir, "output": dir, "temp": dir}`` for this ComfyUI ({} when unknown)."""
    global _dirs_cache
    now = time.time()
    with _dirs_lock:
        if _dirs_cache[1] and now - _dirs_cache[0] < _DIRS_TTL:
            return dict(_dirs_cache[1])
    out: dict = {}
    try:
        from agenty_core.tools.comfyui import get_comfyui_dirs
        d = json.loads(get_comfyui_dirs())
        if isinstance(d, dict):
            if d.get("input_dir"):
                out["input"] = str(d["input_dir"])
            if d.get("output_dir"):
                out["output"] = str(d["output_dir"])
    except Exception:  # noqa: BLE001
        pass
    try:
        from agenty_core.utils.comfyui_client import get_client, parse_argv_dir_flag
        stats = get_client().get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        tmp = parse_argv_dir_flag(argv, "--temp-directory")
        if tmp:
            out["temp"] = str(Path(tmp) / "temp")
        else:
            from agenty_core.tools.comfyui import _comfyui_root
            root = _comfyui_root()
            if root:
                out["temp"] = str(Path(root) / "temp")
    except Exception:  # noqa: BLE001
        pass
    if out:
        with _dirs_lock:
            _dirs_cache = (now, dict(out))
    return out


def _join(dirs: dict, ftype: str, subfolder: str, filename: str) -> str:
    """The absolute path of a ComfyUI file reference, or "" when it cannot be named."""
    base = dirs.get(str(ftype or "output").lower())
    if not base or not filename:
        return ""
    try:
        p = Path(base)
        for part in str(subfolder or "").replace("\\", "/").split("/"):
            if part and part not in (".", ".."):
                p = p / part
        name = Path(str(filename)).name
        return str(p / name) if name else ""
    except Exception:  # noqa: BLE001
        return ""


def read_sidecar(path: str) -> dict:
    """The ``.agenty.json`` beside *path*, or {}."""
    if not path:
        return {}
    try:
        f = Path(str(path) + SIDECAR_SUFFIX)
        if not f.is_file():
            return {}
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _run_outputs(prompt_id: str, lookup: Callable[[str], object]) -> list[dict]:
    """The files a run produced (``[{filename, subfolder, type}]``), newest node last."""
    try:
        h = lookup(prompt_id)
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(h, dict):
        return []
    entry = h.get(prompt_id) if prompt_id in h else h
    outputs = entry.get("outputs") if isinstance(entry, dict) else None
    files: list[dict] = []
    for node_out in (outputs or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("images", "gifs", "videos", "video", "3d", "audio"):
            for f in node_out.get(key) or []:
                if isinstance(f, dict) and f.get("filename"):
                    files.append({"filename": str(f["filename"]),
                                  "subfolder": str(f.get("subfolder") or ""),
                                  "type": str(f.get("type") or "output")})
    return files


def _default_lookup(prompt_id: str):
    from agenty_core.utils.comfyui_client import get_client
    return get_client().get(f"/history/{prompt_id}")


def _finish(item: dict) -> dict:
    """Existence, kind and the sidecar's role for an item that has a path."""
    p = item.get("path") or ""
    if p:
        try:
            item["exists"] = Path(p).is_file()
        except Exception:  # noqa: BLE001
            item["exists"] = False
        side = read_sidecar(p)
        if side.get("role"):
            item["role"] = " ".join(str(side["role"]).split())[:160]
        for k in ("title", "workflow", "hook"):
            v = side.get(k)
            if isinstance(v, str) and v.strip() and k not in item:
                item[k] = v.strip()[:200]
    item["kind"] = _kind_of(p or item.get("filename", ""), item.get("kind", ""))
    return item


def _viewer_items(viewer: dict, dirs: dict) -> list[dict]:
    out: list[dict] = []
    for raw in viewer.get("items") or []:
        if not isinstance(raw, dict):
            continue
        it: dict = {"source": "viewer"}
        for k in ("path", "filename", "subfolder", "type", "kind", "label", "group",
                  "prompt_id", "agent_tag", "frames", "frame_paths", "index"):
            v = raw.get(k)
            if v not in (None, "", [], {}):
                it[k] = v
        if not it.get("path") and it.get("filename"):
            joined = _join(dirs, it.get("type", "output"), it.get("subfolder", ""), it["filename"])
            if joined:
                it["path"] = joined
        out.append(_finish(it))
    return out


def _asset_items(assets: dict, dirs: dict, lookup) -> list[dict]:
    out: list[dict] = []
    for aid in assets.get("ids") or []:
        aid = str(aid or "")
        if not aid:
            continue
        it: dict = {"source": "comfy_assets", "asset_id": aid}
        m_in = _INPUT_ID.match(aid)
        m_out = _RUN_OUTPUT_ID.match(aid)
        m_run = _RUN_ID.match(aid)
        if m_in:
            it.update(filename=m_in.group(1), type="input")
        elif m_out:
            it["prompt_id"] = m_out.group(1)
            try:
                node, sub, fname = json.loads(m_out.group(2))[:3]
                it.update(filename=str(fname), subfolder=str(sub or ""), node_id=str(node))
            except Exception:  # noqa: BLE001
                pass
        elif m_run:
            it["prompt_id"] = m_run.group(1)
            files = _run_outputs(it["prompt_id"], lookup)
            if files:
                # The sidebar shows a run by its newest output; the rest are its siblings.
                first = files[-1]
                it.update(filename=first["filename"], subfolder=first["subfolder"], type=first["type"])
                if len(files) > 1:
                    paths = [_join(dirs, f["type"], f["subfolder"], f["filename"]) for f in files]
                    it["run_outputs"] = [p for p in paths if p][:16]
        if it.get("filename"):
            ftype = it.get("type") or "output"
            path = _join(dirs, ftype, it.get("subfolder", ""), it["filename"])
            if path and not Path(path).is_file() and not it.get("type"):
                alt = _join(dirs, "temp", it.get("subfolder", ""), it["filename"])
                if alt and Path(alt).is_file():
                    path, ftype = alt, "temp"
            it["type"] = ftype
            if path:
                it["path"] = path
        else:
            it["unresolved"] = True
        out.append(_finish(it))
    return out


def resolve(selection: Optional[dict], *, dirs: Optional[dict] = None,
            history_lookup: Optional[Callable[[str], object]] = None) -> dict:
    """The page's ``{viewer, assets}`` answer as ``{count, items, panels, truncated?}``.

    *dirs* (``{"input"|"output"|"temp": folder}``) and *history_lookup*
    (``prompt_id -> /history/<id> answer``) are asked of ComfyUI when not given,
    and only when an item needs them. Never raises.
    """
    sel = selection if isinstance(selection, dict) else {}
    viewer = sel.get("viewer") if isinstance(sel.get("viewer"), dict) else None
    assets = sel.get("assets") if isinstance(sel.get("assets"), dict) else None
    items: list[dict] = []
    panels: list[dict] = []
    needs_dirs = bool(assets) or any(
        isinstance(r, dict) and not r.get("path") and r.get("filename")
        for r in ((viewer or {}).get("items") or []))
    d = dirs if dirs is not None else (_type_dirs() if needs_dirs else {})
    try:
        if viewer and viewer.get("items"):
            got = _viewer_items(viewer, d)
            items += got
            panels.append({"panel": "bEpic viewer history",
                           "tab": str(viewer.get("tab_label") or viewer.get("tab") or ""),
                           "how": str(viewer.get("mode") or ""),
                           "count": int(viewer.get("count") or len(got))})
        if assets and assets.get("ids"):
            got = _asset_items(assets, d, history_lookup or _default_lookup)
            items += got
            panels.append({"panel": "ComfyUI assets sidebar",
                           "count": int(assets.get("count") or len(got))})
    except Exception as exc:  # noqa: BLE001
        return {"count": len(items), "items": items[:MAX_ITEMS], "panels": panels,
                "error": f"could not read every item: {exc}"}
    total = sum(p["count"] for p in panels) if panels else len(items)
    out = {"count": max(total, len(items)), "items": items[:MAX_ITEMS], "panels": panels}
    if out["count"] > len(out["items"]):
        out["truncated"] = True
    return out


_HOW = {"multi": "selected", "compare": "pinned in a comparison", "open": "opened"}


def describe_lines(resolved: dict, cap: int = BLOCK_ITEMS) -> tuple[str, str]:
    """``(where, list)`` for the turn's context block, or ``("", "")`` when nothing
    is picked. *where* names the panel(s); *list* is one line per item, capped,
    with the rest counted."""
    items = [i for i in (resolved or {}).get("items") or [] if isinstance(i, dict)]
    if not items:
        return "", ""
    where_parts = []
    for p in (resolved or {}).get("panels") or []:
        bit = p.get("panel", "")
        if p.get("tab"):
            bit += f', tab "{p["tab"]}"'
        how = _HOW.get(p.get("how", ""), "")
        if how:
            bit += f" ({how})"
        where_parts.append(bit)
    lines: list[str] = []
    for n, it in enumerate(items[:max(1, cap)], 1):
        target = it.get("path") or (
            f'{it.get("filename")} [{it.get("type", "output")}]' if it.get("filename")
            else f'asset id {it.get("asset_id", "?")} (not resolved to a file)')
        line = f"{n}. {it.get('kind', 'image')}: {target}"
        extras = []
        if it.get("role"):
            extras.append(f'is: "{it["role"]}"')
        if it.get("label"):
            extras.append(f'label "{it["label"]}"')
        if it.get("agent_tag"):
            extras.append(f'#{it["agent_tag"]}')
        if it.get("frames"):
            extras.append(f'{it["frames"]} frames')
        if it.get("run_outputs") and len(it["run_outputs"]) > 1:
            extras.append(f'run made {len(it["run_outputs"])} files')
        if it.get("prompt_id"):
            extras.append(f'run {it["prompt_id"]}')
        if it.get("workflow"):
            extras.append(f'workflow {it["workflow"]}')
        if it.get("exists") is False:
            extras.append("FILE MISSING")
        if extras:
            line += " - " + ", ".join(extras)
        lines.append(line)
    total = int((resolved or {}).get("count") or len(items))
    if total > len(lines):
        lines.append(f"... and {total - len(lines)} more (get_history_selection lists them all).")
    return "; ".join(where_parts), "\n".join(lines)


@tool
def get_history_selection() -> str:
    """Read what the user has picked in a history panel right now: the bEpic
    viewer's history strip (Ctrl-selected snapshots, two pinned in a compare, or
    one older snapshot clicked open) and ComfyUI's Media Assets / queue sidebar.

    Use when the user points at renders rather than nodes - "the selected
    images", "these three", "the one I have open in the viewer", "use the picked
    ones as references" - and the pick may have changed since the turn began.
    Each item has an absolute ``path`` you can use as an input exactly like a
    #tag's file (upload_image, then the LoadImage value; view_image to look),
    its ``kind`` (image / video / model), the ``prompt_id`` of the run that made
    it when known, and ``role`` from its ``.agenty.json`` sidecar.

    Requires ComfyUI open in a browser with agentY-comfyuiConnect installed.

    Returns JSON: ``{status, count, items: [{source, path, kind, label, role,
    prompt_id, agent_tag, ...}], panels}``; ``count`` 0 when nothing is picked.
    """
    from agenty_core.utils import canvas_scope as _cs
    if _cs.detached():
        return json.dumps({"status": "error", "error": _cs.NO_CANVAS})
    try:
        from agenty_core.utils.comfyui_client import get_client
        cid = _cs.client_id()
        r = get_client().get("/agent/history_selection" + (f"?client_id={cid}" if cid else ""))
    except Exception as e:  # noqa: BLE001
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404 or "404" in str(e):
            msg = ("the agentY-comfyuiConnect extension in this ComfyUI predates the "
                   "history-selection route - update it under <ComfyUI>/custom_nodes/ and "
                   "restart ComfyUI.")
        else:
            msg = f"could not reach ComfyUI ({e}). Is it running?"
        return json.dumps({"status": "error", "error": msg})
    if not isinstance(r, dict):
        return json.dumps({"status": "error", "error": f"unexpected response: {r!r}"})
    if not r.get("ok"):
        return json.dumps({"status": "error", "error": r.get("error", "unknown error")})
    res = resolve(r.get("selection"))
    out = {"status": "ok", **res}
    if not res.get("items"):
        out["hint"] = ("Nothing is picked in a history panel. In the bEpic viewer: Ctrl+click "
                       "thumbnails, or click one to open it; in ComfyUI's assets sidebar: select "
                       "the outputs.")
    return json.dumps(out)
