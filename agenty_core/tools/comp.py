"""Build, read, render and farm out comps of the ComfyUI Image Viewer's Comp space (``*.comp.json``).

``edit_comp`` reads a comp or changes it (``POST /imageviewer/comp_ops``: the viewer's ``comp_api.Comp.apply``,
all or nothing, saved, and an open Comp tab takes it in); ``render_comp`` renders its Write nodes in ComfyUI
(``/imageviewer/comp_write``) or sends them to the Deadline farm (the viewer plans the job, ``comp_farm`` with
``dry_run``, and ``deadline_farm.submit_comp`` submits it from here). Never write the .comp.json by hand: the
open tab would not see it.
"""

from __future__ import annotations

import json
import time

from agenty_core._compat import tool

_OPS_HELP = "ops is a list of changes (JSON)"


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _said(e) -> str:
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            return str(resp.json().get("error") or "")[:400]
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _status(e):
    return getattr(getattr(e, "response", None), "status_code", None)


def _fail(e, what: str) -> str:
    said = _said(e)
    if said:
        return _j({"status": "error", "error": said})
    if _status(e) == 404 or (_status(e) is None and "404" in str(e)):
        return _j({"status": "error", "error": (
            f"the viewer in this ComfyUI cannot {what} yet (no such route) - update ComfyUI-ImageViewer under "
            "<ComfyUI>/custom_nodes/ and restart ComfyUI.")})
    return _j({"status": "error", "error": f"could not reach ComfyUI ({str(e)[:160]}). Is it running?"})


def _open_comp() -> str:
    """The comp open in the viewer's Comp tab ("" when none)."""
    try:
        from agenty_core.tools.viewer_state import read_state
        got = json.loads(read_state())
    except Exception:  # noqa: BLE001
        return ""
    key = str(((got.get("state") or {}).get("tab") or {}).get("key") or "")
    return key[5:] if key.startswith("comp:") and key.endswith(".comp.json") else ""


def _path(path: str) -> tuple[str, str]:
    path = (path or "").strip() or _open_comp()
    if not path:
        return "", "no comp is open in the viewer - pass path= (a .comp.json)"
    if not path.lower().endswith(".comp.json"):
        return "", "path must be a *.comp.json"
    from agenty_core.utils.path_guard import refuse_read
    why = refuse_read(path, "edit_comp")
    return ("", why) if why else (path, "")


def _clean_ops(ops):
    if ops is None:
        return [], ""
    if isinstance(ops, str):
        if not ops.strip():
            return [], ""
        try:
            ops = json.loads(ops)
        except ValueError:
            return None, _OPS_HELP
    if isinstance(ops, dict):
        ops = [ops]
    return (ops, "") if isinstance(ops, list) else (None, _OPS_HELP)


@tool
def edit_comp(path: str = "", ops: list | None = None, create: bool = False, width: int = 1920, height: int = 1080,
              fps: float = 24, frames: int = 100) -> str:
    """Read or change a comp of the ComfyUI Image Viewer's Comp space (a node graph of pixel operations in a
    ``*.comp.json``): add nodes, wire them, set and key their parameters, add Write nodes. The viewer saves the
    change and an open Comp tab shows it at once.

    Args:
        path: The .comp.json; "" for the comp open in the viewer's Comp tab.
        ops: Leave empty to READ it (format, output, nodes with their inputs and the parameters that differ from
            the defaults, keyed ones). To change it, a list, all done or none:
            ``{"op": "add", "type": "read|constant|merge|grade|blur|transform|write|metadata|...", "name", "inputs": [..],
              "x", "y", <param>: <value>}`` - its id comes back in ``ids``; later ops can name it "$1", "$2" (the
              first, second node added by this call).
            ``{"op": "connect", "node": .., "from": <node>, "port": <input name or index>}``,
            ``{"op": "disconnect", "node": .., "port": ..}``,
            ``{"op": "set", "node": .., "params": {..}}`` (also "name", "disable"),
            ``{"op": "key", "node": .., "param": .., "frame": .., "value": .., "ease": "smooth|linear|hold|..."}``,
            ``{"op": "unkey", "node": .., "param": .., "frame": ..}``, ``{"op": "remove", "node": ..}``,
            ``{"op": "output", "node": ..}`` (what the Output node shows), ``{"op": "format", "width", "height", "fps", "frames"}``.
            Nodes are named by id or by their (unique) name. A Read takes stills, films, sequences
            (shot.####.exr) and other comps (a precomp). A Write: file (#### = the frame number), file_type
            exr|png|tiff|jpg|mov|mp4|webm, channels rgba|rgb|all (all: every extra channel as EXR layers),
            depth, compression, first / last (comp frames, from 0), handles, first_number (1001 = comp frame 0).
        create: Make the comp when the file is not there (with width / height / fps / frames).

    Returns:
        JSON: {"status": "ok", "path", "comp": {format, output, nodes: [...]}, "ids"?: [new node ids]} or
        {"status": "error", "error"}.
    """
    path, why = _path(path)
    if why:
        return _j({"status": "error", "error": why})
    ops, why = _clean_ops(ops)
    if why:
        return _j({"status": "error", "error": why})
    from agenty_core.utils.comfyui_client import get_client
    try:
        if not ops and not create:
            r = get_client().get("/imageviewer/comp_summary", params={"path": path})
            return _j({"status": "ok", "path": path, "comp": r})
        body = {"path": path, "ops": ops}
        if create:
            body["create"] = {"width": int(width), "height": int(height), "fps": float(fps), "frames": int(frames)}
        r = get_client().post("/imageviewer/comp_ops", json_data=body)
    except Exception as e:  # noqa: BLE001
        return _fail(e, "edit comps")
    if not isinstance(r, dict) or r.get("error"):
        return _j({"status": "error", "error": str((r or {}).get("error") if isinstance(r, dict) else r)[:400]})
    return _j({"status": "ok", "path": r.get("path") or path, "ids": r.get("ids") or [], "comp": r.get("comp")})


@tool
def render_comp(path: str = "", writes: str = "", first: int = -1, last: int = -1, handles: int = -1,
                farm: bool = False, chunk: int = 10, pool: str = "", group: str = "", priority: int = -1,
                wait_s: int = 600, dry_run: bool = False) -> str:
    """Render a comp's Write nodes - each writes its own files (EXR sequences with layers, PNG, films) - in
    ComfyUI, or send them to the Deadline render farm as one job in chunks of frames.

    Args:
        path: The .comp.json; "" for the comp open in the viewer. It is rendered as SAVED (edit_comp saves).
        writes: Write nodes by id or name, comma separated; "" = all that are switched on.
        first, last: comp frames (from 0); -1 = each Write's own range.
        handles: extra frames each side; -1 = each Write's own.
        farm: send to Deadline instead of rendering here (needs Deadline set up for agentY, and the comp, its
            footage and its outputs on paths the farm machines can reach - the answer warns about any that are not).
        chunk, pool, group, priority: the farm job's frames per task, pool, group, priority (0-100).
        wait_s: rendering here, wait this long for the files (max 3600); the job goes on if it takes longer.
        dry_run: say what would be written (and, with farm, the job) without rendering.

    Returns:
        JSON: here: {"status": "ok", "writes": [{name, frames, pattern, files}], "notes"} (or "running" with a job id);
        farm: {"status": "submitted", "job_id", "tasks", "warnings"}; or {"status": "error", "error"}.
    """
    path, why = _path(path)
    if why:
        return _j({"status": "error", "error": why})
    body = {"path": path, "writes": [w.strip() for w in (writes or "").split(",") if w.strip()] or None}
    for k, v in (("first", first), ("last", last), ("handles", handles)):
        if int(v) >= 0:
            body[k] = int(v)
    from agenty_core.utils.comfyui_client import get_client
    c = get_client()
    if farm:
        try:
            r = c.post("/imageviewer/comp_farm", json_data={**body, "dry_run": True, "chunk": int(chunk), "pool": pool,
                                                             "group": group, "priority": int(priority)})
        except Exception as e:  # noqa: BLE001
            return _fail(e, "send comps to the farm")
        if dry_run:
            return _j({"status": "ok", "dry_run": True, "request": r.get("request"), "warnings": r.get("warnings") or [],
                       "writes": r.get("writes") or []})
        from agenty_core.tools import deadline_farm as DF
        try:
            out = DF.submit_comp(r.get("request") or {})
        except DF.FarmError as e:
            return e.render()
        return _j({**out, "warnings": r.get("warnings") or []})
    try:
        if dry_run:
            r = c.post("/imageviewer/comp_write_plan", json_data=body)
            return _j({"status": "ok", "dry_run": True, "writes": r.get("writes") or [], "refused": r.get("refused")})
        r = c.post("/imageviewer/comp_write", json_data=body)
        jid = r.get("job")
        end = time.time() + max(0, min(3600, int(wait_s)))
        while True:
            s = c.get("/imageviewer/comp_write", params={"job": jid})
            if s.get("state") == "done":
                res = s.get("result") or {}
                return _j({"status": "ok", "writes": res.get("writes") or [], "notes": res.get("notes") or []})
            if s.get("state") != "running":
                return _j({"status": "error", "error": s.get("error") or f"the render {s.get('state')}"})
            if time.time() >= end:
                return _j({"status": "running", "job": jid, "done": s.get("done"), "total": s.get("total"),
                           "hint": "it renders on; call again later to see the files"})
            time.sleep(1.0)
    except Exception as e:  # noqa: BLE001
        return _fail(e, "render Write nodes")
