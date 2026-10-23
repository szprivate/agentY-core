"""Ayon pipeline actions for a ComfyUI session that was launched THROUGH Ayon.

The studio's Ayon ComfyUI addon (a fork of ynput/ayon-comfyui) is not touched: everything here uses
what it already exposes. Read ``docs/ayon.md`` in agentY for the full picture; the short version:

* ComfyUI is shown inside an <iframe> on a small static page the Ayon host serves (default
  ``http://127.0.0.1:5454``). A script on that page connects to the Ayon host's WSRPC server
  (default ``ws://127.0.0.1:55056/ws/``). That server has ONE namespace, ``ayonComfyUI``, with three
  methods: ``pingAyonMenu``, ``requestToolByName`` (opens a Qt tool: loader, publisher, ...) and
  ``requestSaveByName``. Everything else the Ayon host does to the canvas goes the other way, host ->
  browser tab, and only the browser tab's own socket can receive it. This module is a second client of the
  server; it cannot receive those calls, so it never pretends to.
* What a loader or a creator leaves on the canvas is plain node data: ``AYON Load Image`` carries
  ``ayon_container_info`` JSON, ``AYON Image Save`` carries ``ayon_info`` JSON, and one ``AYON Context``
  node carries ``{"context", "instances", "containers"}`` JSON. That data can be written without the Ayon
  host, through the canvas bridge agentY already has.
* Publishing runs pyblish inside the Ayon host process with a Qt UI. There is no RPC for it, so
  :func:`ayon_publish` opens the Publisher and says so; it does not fake a publish.

Nothing here reads a login token or talks to the Ayon server: projects, products, versions, representations and
comments are the official ayon-mcp server's job. These tools act on the live canvas session only.
"""

from __future__ import annotations

import glob
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

import requests

from agenty_core._compat import tool
from agenty_core.utils.comfyui_client import ComfyUIClient, get_client

logger = logging.getLogger("agenty_core.ayon")

# ── What the addon uses (read from ayon_comfyui 0.0.28+bpc.1) ──────────────────────────────────────────────
DEFAULT_STATIC_URL = "http://127.0.0.1:5454"     # settings: http_server_port
RPC_NAMESPACE = "ayonComfyUI"                    # rpc_server.py: WebSocketAsync.add_route("ayonComfyUI", ...)
CONTEXT_NODE = "AYON Context"
CONTEXT_WIDGET = "ayon_context_info"
AYON_CONTAINER_ID = "ayon.load.container"        # ayon_core.pipeline.constants
AYON_INSTANCE_ID = "ayon.create.instance"
TOOLS = ("loader", "create", "publisher", "workfiles", "sceneinventory")   # ayon_menu.js AYON_TOOL_ACTIONS

KINDS: dict[str, dict] = {
    "image": {"load_node": "AYON Load Image", "load_widget": "ayon_container_info", "loader": "ImageLoader",
              "save_node": "AYON Image Save", "save_input": "images_in", "creator": "io.ayon.creators.comfyui.image",
              "product_type": "image", "extensions": {"png", "jpg", "jpeg", "exr", "tif", "tiff", "webp", "bmp", "tga", "hdr"}},
    "video": {"load_node": "AYON Load Video", "load_widget": "ayon_container_info", "loader": "VideoLoader",
              "save_node": "AYON Video Save", "save_input": "video_in", "creator": "io.ayon.creators.comfyui.render",
              "product_type": "render", "extensions": {"mp4", "mov", "webm", "mkv", "avi"}},
    "model": {"load_node": "AYON Load 3D Model", "load_widget": "ayon_container_info", "loader": "ModelLoader",
              "save_node": "AYON 3D Model Save", "save_input": "mesh", "creator": "io.ayon.creators.comfyui.model",
              "product_type": "model", "extensions": {"glb", "gltf", "obj", "fbx", "ply", "stl", "spz", "splat", "ksplat"}},
}
_DEFAULT_NAME = "ayon"          # create_*.py default_*_name: the prefix / unique name the Creator UI proposes

# The ONLY environment variables read from the ComfyUI process. A launched process can carry more (the
# service-user key included); an allow-list keeps everything else unread.
_LAUNCH_ENV = {"AYON_PROJECT_NAME": "project", "AYON_FOLDER_PATH": "folder_path", "AYON_TASK_NAME": "task",
               "AYON_WORKDIR": "workdir", "AYON_APP_NAME": "app", "AYON_SERVER_URL": "server_url",
               "AYON_LAST_WORKFILE": "last_workfile"}

_REPLY_WAIT = 5.0


class AyonError(Exception):
    """Something this module can say in one sentence."""


class AyonRpcError(AyonError):
    """The WSRPC server could not be reached or answered with an error."""


# ═══════════════════════════════════════════════════════════════════════════════
# Host adapters (set by the app that embeds the tools; both optional)
# ═══════════════════════════════════════════════════════════════════════════════

_canvas_provider: Optional[Callable[[], Any]] = None
_canvas_placer: Optional[Callable[[list, str], bool]] = None


def set_canvas_provider(fn: Optional[Callable[[], Any]]) -> None:
    """Register a function returning the live canvas as an API-format prompt ``{node_id: {class_type, inputs}}``.

    agentY's host passes the graph its panel posted. Without one, the tools read only the nodes the user has
    selected (``/agent/canvas_selection``)."""
    global _canvas_provider
    _canvas_provider = fn


def set_canvas_placer(fn: Optional[Callable[[list, str], bool]]) -> None:
    """Register a function that applies ``edit_graph`` ops (add / connect) to the live canvas in place.

    ``/agent/load_workflow`` is not a substitute: it reloads the tab under another name, and Ayon finds its
    workfile by the tab name. Without a placer the tools hand back the node data to place by hand."""
    global _canvas_placer
    _canvas_placer = fn


# ═══════════════════════════════════════════════════════════════════════════════
# Session discovery and the WSRPC client
# ═══════════════════════════════════════════════════════════════════════════════

_IFRAME_RE = re.compile(r'<iframe[^>]*\bsrc="([^"]+)"', re.I)
_PORT_RE = re.compile(r"window\.rpc_port\s*=\s*(\d+)")


def parse_static_page(html: str) -> tuple[Optional[str], Optional[int]]:
    """The ComfyUI URL and the RPC port out of the page ``page_templater.template_html`` renders."""
    m1, m2 = _IFRAME_RE.search(html or ""), _PORT_RE.search(html or "")
    return (m1.group(1) if m1 else None), (int(m2.group(1)) if m2 else None)


def _hostport(url: str) -> str:
    p = urlparse(url if "//" in url else "//" + url)
    host = "127.0.0.1" if (p.hostname or "") in ("localhost", "127.0.0.1", "::1", "0.0.0.0") else (p.hostname or "")
    port = p.port or (443 if p.scheme == "https" else 80)
    return f"{host}:{port}"


def discover_session(static_url: str = "", timeout: float = 1.5) -> Optional[dict]:
    """Find the Ayon-launched session, or None. The static page names the ComfyUI URL and the RPC port."""
    base = static_url or os.environ.get("AYON_COMFYUI_STATIC_URL") or DEFAULT_STATIC_URL
    try:
        r = requests.get(base, timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    comfy, port = parse_static_page(r.text)
    if not comfy or not port:
        return None
    return {"static_url": base.rstrip("/"), "comfy_url": comfy.rstrip("/"), "rpc_port": port}


def _ws_connect(url: str, timeout: float):
    from websockets.sync.client import connect
    return connect(url, open_timeout=timeout, max_size=2 ** 22)


def rpc_call(port: int, method: str, params: Optional[dict] = None, *, timeout: float = _REPLY_WAIT,
             host: str = "127.0.0.1") -> Any:
    """Call one method on the Ayon host's WSRPC server and return its result.

    Wire format (wsrpc_aiohttp): a call is ``{"id": n, "method": "ayonComfyUI.x", "params": {..}}``; the
    answer is ``{"id": n, "result": ..}`` or ``{"id": n, "error": {"type", "message"}}``. The server also
    pings every client (``{"id", "method": "ping", "params": {"seq"}}``) and drops one that does not echo
    ``params`` back as the result, so the loop answers those. No ``Origin`` header is sent: the host hands its
    own calls to the first client whose Origin is the static page, and this client must never be that one.
    """
    url = f"ws://{host}:{port}/ws/"
    try:
        ws = _ws_connect(url, timeout)
    except Exception as e:  # noqa: BLE001
        raise AyonRpcError(f"cannot reach the Ayon host's RPC server at {url}: {e}") from e
    call_id = 3     # the browser client counts 3, 5, 7 ... (wsrpc.js); the server's own calls count apart
    deadline = time.monotonic() + timeout
    with ws:
        ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise AyonRpcError(f"{method} did not answer within {timeout:g} s")
            try:
                raw = ws.recv(timeout=left)
            except TimeoutError as e:
                raise AyonRpcError(f"{method} did not answer within {timeout:g} s") from e
            except Exception as e:  # noqa: BLE001
                raise AyonRpcError(f"the RPC connection closed: {e}") from e
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("method") == "ping":
                ws.send(json.dumps({"id": msg.get("id"), "result": msg.get("params") or {}}))
                continue
            if msg.get("method"):
                ws.send(json.dumps({"id": msg.get("id"), "error": {
                    "type": "NotImplementedError", "message": "agentY registers no routes"}}))
                continue
            if msg.get("id") != call_id:
                continue
            if msg.get("error") is not None:
                err = msg["error"]
                raise AyonRpcError(f"{method}: {err.get('type', 'error')}: {err.get('message', err)}"
                                   if isinstance(err, dict) else f"{method}: {err}")
            return msg.get("result")


def launch_context_of(comfy_url: str) -> Optional[dict]:
    """The Ayon context the ComfyUI process was launched with, from an allow-list of its environment.

    The Ayon launch hook starts ComfyUI with ``AYON_PROJECT_NAME`` / ``AYON_FOLDER_PATH`` / ``AYON_TASK_NAME`` /
    ``AYON_WORKDIR`` in its environment (launch_logic.py passes ``os.environ`` on). Needs ``psutil``; None
    when it is missing, the process is not found or its environment cannot be read."""
    try:
        import psutil
    except ImportError:
        return None
    port = int(_hostport(comfy_url).rsplit(":", 1)[1])
    try:
        pid = next((c.pid for c in psutil.net_connections(kind="tcp")
                    if c.status == "LISTEN" and c.laddr and c.laddr.port == port and c.pid), None)
        if pid is None:
            return None
        env = psutil.Process(pid).environ()
    except Exception:  # noqa: BLE001  (access denied, process gone)
        return None
    out = {name: env[key] for key, name in _LAUNCH_ENV.items() if env.get(key)}
    return out or None


# ═══════════════════════════════════════════════════════════════════════════════
# Files
# (The Ayon SERVER - projects, products, versions, representations, comments - is the job of the official
#  ayon-mcp server, not of these tools: they take what it returns as arguments.)
# ═══════════════════════════════════════════════════════════════════════════════

def expand_files(path: str) -> list[str]:
    """One file, or the frames of a ``%04d`` / ``####`` / ``<UDIM>`` style sequence that exist on disk."""
    pat = re.sub(r"%0?(\d*)d", lambda m: "?" * int(m.group(1) or 1), path)
    pat = re.sub(r"#+", lambda m: "?" * len(m.group(0)), pat)
    if pat != path:
        return sorted(glob.glob(pat))
    return [path] if os.path.isfile(path) else []


# ═══════════════════════════════════════════════════════════════════════════════
# The canvas (bridge routes of agentY-comfyuiConnect, plus the host adapters)
# ═══════════════════════════════════════════════════════════════════════════════

def _client_for(session: dict) -> ComfyUIClient:
    """The ComfyUI the Ayon session runs: agentY's own client when it is the same server, else a client for that URL."""
    default = get_client()
    if _hostport(default.base_url) == _hostport(session["comfy_url"]):
        return default
    return ComfyUIClient(base_url=session["comfy_url"])


def read_canvas(client: ComfyUIClient) -> dict:
    """Nodes of the live canvas: ``{"source": canvas|selection|none, "nodes": [{id, type, values}]}``."""
    if _canvas_provider is not None:
        try:
            prompt = _canvas_provider()
        except Exception as e:  # noqa: BLE001
            logger.warning("canvas provider failed: %s", e)
            prompt = None
        if isinstance(prompt, dict) and prompt:
            nodes = [{"id": str(nid), "type": n["class_type"], "values": dict(n.get("inputs") or {})}
                     for nid, n in prompt.items() if isinstance(n, dict) and n.get("class_type")]
            if nodes:
                return {"source": "canvas", "nodes": nodes}
    try:
        r = client.get("/agent/canvas_selection")
    except Exception as e:  # noqa: BLE001
        return {"source": "none", "nodes": [], "error": str(e)}
    if isinstance(r, dict) and r.get("ok") and r.get("nodes"):
        return {"source": "selection", "nodes": [{"id": str(n.get("id")), "type": n.get("type", ""),
                                                  "values": dict(n.get("widgets") or {})} for n in r["nodes"]]}
    return {"source": "none", "nodes": [], "error": (r or {}).get("error", "") if isinstance(r, dict) else ""}


def parse_context_doc(text: Any) -> dict:
    """The AYON Context node's JSON, with the three keys the addon's ``_ensure_wellformed_context`` guarantees."""
    try:
        doc = json.loads(text) if isinstance(text, str) and text.strip() else {}
    except ValueError:
        doc = {}
    if not isinstance(doc, dict):
        doc = {}
    if not isinstance(doc.get("context"), dict):
        doc["context"] = {}
    for key in ("instances", "containers"):
        if not isinstance(doc.get(key), list):
            doc[key] = []
    return doc


def _merge_unique(old: list, new: list, key: str) -> list:
    """``new`` replaces an item of ``old`` with the same ``key`` and is appended otherwise."""
    out = {str(i.get(key)): i for i in old if isinstance(i, dict)}
    for item in new:
        out[str(item.get(key))] = item
    return list(out.values())


_last_written: dict[tuple, dict] = {}     # (comfy_url, node id) -> document this module last wrote


def _context_state(canvas: dict, comfy_url: str) -> tuple[Optional[str], dict]:
    """(id of the AYON Context node or None, its parsed document)."""
    node = next((n for n in canvas["nodes"] if n["type"] == CONTEXT_NODE), None)
    if node is None:
        return None, parse_context_doc("")
    doc = parse_context_doc(node["values"].get(CONTEXT_WIDGET))
    if canvas["source"] == "canvas":     # the host's snapshot can be older than our last write in this same turn
        mine = _last_written.get((comfy_url, node["id"]))
        if mine:
            doc["instances"] = _merge_unique(mine.get("instances", []), doc["instances"], "instance_id")
            doc["containers"] = _merge_unique(mine.get("containers", []), doc["containers"], "container_uuid")
    return node["id"], doc


def _next_ids(canvas: dict, count: int) -> list[str]:
    ids = [int(n["id"]) for n in canvas["nodes"] if str(n["id"]).isdigit()]
    start = (max(ids) if canvas["source"] == "canvas" and ids else 900000) + 1
    return [str(start + i) for i in range(count)]


def _write_context(client: ComfyUIClient, session: dict, canvas: dict, ctx_id: Optional[str], doc: dict,
                   launch: dict, reason: str) -> dict:
    """Put the document on the AYON Context node: set the widget, or add the node when the canvas has none."""
    text = json.dumps(doc)
    if ctx_id:
        r = client.post("/agent/set_node_params", json_data={"node_id": ctx_id, "params": {CONTEXT_WIDGET: text}})
        ok = isinstance(r, dict) and CONTEXT_WIDGET in (r.get("applied") or [])
        if ok:
            _last_written[(session["comfy_url"], ctx_id)] = doc
            return {"registered": True, "context_node": ctx_id}
        return {"registered": False, "context_node": ctx_id,
                "problem": (r or {}).get("error") if isinstance(r, dict) else str(r)}
    if _canvas_placer is not None:
        new_id = _next_ids(canvas, 1)[0]
        if not doc["context"]:
            doc["context"] = {k: v for k, v in (("project_name", launch.get("project")),
                                                ("folder_path", launch.get("folder_path")),
                                                ("task_name", launch.get("task"))) if v}
            text = json.dumps(doc)
        if _canvas_placer([{"op": "add", "class_type": CONTEXT_NODE, "node_id": new_id,
                            "params": {CONTEXT_WIDGET: text}}], reason):
            _last_written[(session["comfy_url"], new_id)] = doc
            return {"registered": True, "context_node": new_id, "added_context_node": True}
    return {"registered": False, "context_node": None,
            "problem": "there is no AYON Context node I can write to (select it, or open the Ayon Creator once)"}


# ═══════════════════════════════════════════════════════════════════════════════
# Session bundle used by every tool
# ═══════════════════════════════════════════════════════════════════════════════

def _session() -> tuple[Optional[dict], str]:
    """(session, '') or (None, why not)."""
    s = discover_session()
    if s is None:
        return None, ("not launched through Ayon: nothing answers on the Ayon static page "
                      f"({os.environ.get('AYON_COMFYUI_STATIC_URL') or DEFAULT_STATIC_URL}). Start ComfyUI from the "
                      "Ayon launcher (ComfyUI app, with the task you want) and keep that browser tab open.")
    try:
        got = rpc_call(s["rpc_port"], f"{RPC_NAMESPACE}.pingAyonMenu", {"message": "agentY"}, timeout=3.0)
        s["rpc_ok"], s["rpc_error"] = got == "agentY", ""
    except AyonRpcError as e:
        s["rpc_ok"], s["rpc_error"] = False, str(e)
    s["same_comfyui_as_agenty"] = _hostport(get_client().base_url) == _hostport(s["comfy_url"])
    return s, ""


def _fail(status: str, message: str, **extra) -> str:
    return json.dumps({"status": status, "message": message, **extra})


def _need_launch(session: dict, launch: Optional[dict]) -> Optional[str]:
    if not launch or not all(launch.get(k) for k in ("project", "folder_path", "task")):
        return _fail("no_context", "could not read project / folder / task for this session (the ComfyUI process "
                     "environment was not readable, and psutil may be missing). Run ayon_context to see what is known.",
                     session=session)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def ayon_context() -> str:
    """The Ayon context of the running ComfyUI session: project, folder, task, workdir, and the folder's
    frame range, fps and resolution.

    Use first whenever the user talks about their Ayon / pipeline setup ("what shot am I on", "publish this to
    the shot", "which frame range"). Answers ``not_launched_through_ayon`` when ComfyUI was not started from the
    Ayon launcher. The launch context comes from the ComfyUI process; the canvas side (instances and containers
    already registered) comes from the AYON Context node when the canvas can be read. Frame range, fps and
    resolution are server data: ask the ayon-mcp server.

    Returns JSON: ``{status, launched_through_ayon, session, launch_context, canvas, folder, notes}``.
    """
    session, why = _session()
    if session is None:
        return _fail("not_launched_through_ayon", why, launched_through_ayon=False)
    notes: list[str] = []
    if not session["rpc_ok"]:
        notes.append(f"The Ayon host's RPC server did not answer ping: {session['rpc_error']}")
    if not session["same_comfyui_as_agenty"]:
        notes.append(f"agentY talks to {get_client().base_url}, but the Ayon session runs ComfyUI at "
                     f"{session['comfy_url']}. The Ayon tools use the session's ComfyUI; agentY's own tools do not. "
                     "Point COMFYUI_URL at the session's URL to work on it.")
    launch = launch_context_of(session["comfy_url"])
    if launch is None:
        notes.append("Could not read the launch context from the ComfyUI process (psutil missing or access denied).")
    client = _client_for(session)
    canvas = read_canvas(client)
    ctx_id, doc = _context_state(canvas, session["comfy_url"])
    canvas_out: dict[str, Any] = {"source": canvas["source"], "context_node": ctx_id}
    if ctx_id:
        canvas_out.update(context=doc["context"], instances=[
            {k: i.get(k) for k in ("instance_id", "productType", "productName", "variant")} for i in doc["instances"]],
            containers=[{k: c.get(k) for k in ("container_uuid", "name", "namespace", "representation")}
                        for c in doc["containers"]])
        c = doc["context"]
        if launch and c and (c.get("project_name"), c.get("folder_path"), c.get("task_name")) != (
                launch.get("project"), launch.get("folder_path"), launch.get("task")):
            notes.append("The AYON Context node names another context than the one ComfyUI was launched with "
                         "(the context was switched in the Ayon tools, or the workfile came from elsewhere).")
    else:
        notes.append("No AYON Context node was visible on the canvas (select it for a read, or open the Ayon "
                     "Creator / Loader once: the addon adds it).")
    out: dict[str, Any] = {"status": "ok", "launched_through_ayon": True, "session": session,
                           "launch_context": launch, "canvas": canvas_out}
    notes.append("Frame range, fps and resolution of the folder live on the Ayon server: ask the ayon-mcp server.")
    out["notes"] = notes
    return json.dumps(out)


def _kind_of(product_type: str, path: str, kind: str) -> str:
    if kind in KINDS:
        return kind
    ext = Path(path).suffix.lstrip(".").lower()
    for k, spec in KINDS.items():
        if ext in spec["extensions"]:
            return k
    return {"model": "model", "render": "video"}.get(product_type, "image")


def _upload(client: ComfyUIClient, path: str, subfolder: str) -> dict:
    """Same call as the addon's ``upload_image``: POST /upload/image into input/AYON/<subfolder>."""
    sub = f"AYON/{subfolder}" if subfolder else "AYON"
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        r = client.post("/upload/image", data={"type": "input", "subfolder": sub},
                        files={"image": (os.path.basename(path), fh, mime)})
    if not isinstance(r, dict) or "name" not in r:
        raise AyonError(f"ComfyUI did not accept the upload of {os.path.basename(path)}: {str(r)[:200]}")
    r["subfolder"] = str(r.get("subfolder", "")).replace("\\", "/")
    return r


def build_container(name: str, namespace: str, representation_id: str, uploads: list[dict], loader: str) -> dict:
    """What ``api.pipeline.containerise`` returns, which the addon puts on the load node and in the context."""
    return {"schema": "ayon:container-3.0", "id": AYON_CONTAINER_ID, "name": name, "namespace": namespace,
            "loader": loader, "image_upload_info": uploads, "representation": representation_id,
            "container_uuid": str(uuid.uuid4()), "container_name": name + "_CON"}


@tool
def ayon_load(path: str, product: str, representation_id: str = "", folder_path: str = "", version: int = 0,
              kind: str = "", load_node_id: str = "", dry_run: bool = False) -> str:
    """Load a published Ayon file into the Ayon-launched ComfyUI canvas, the way the Ayon Loader does.

    Look the product up with the ayon-mcp server first (its representation's file path and id), then pass them
    here. This uploads the file (or the frames of a ``%04d`` / ``####`` sequence) to ComfyUI's input/AYON folder,
    fills an AYON Load node (``ayon_container_info``) and registers the container on the AYON Context node so
    the Scene Inventory lists it. The node is added in place when the host can edit the canvas; otherwise the
    result carries the exact node data to paste, or pass ``load_node_id`` of an AYON Load node you placed.

    Args:
        path: Local path of the representation's file, as ayon-mcp reports it (roots resolved for this machine).
        product: Product name, e.g. ``imageMain``; names the container.
        representation_id: The representation's id (lets the Ayon Scene Inventory update / switch it).
        folder_path: Folder path (``agentRND/agentRND_0010``); default: this session's folder.
        version: Version number, for the report only.
        kind: ``image`` | ``video`` | ``model``; default: from the file extension.
        load_node_id: An existing AYON Load node to fill instead of adding one.
        dry_run: Check the file and the session and report what would happen; upload nothing, touch nothing.

    Returns JSON: ``{status, resolved, container, placed, registered, ...}``.
    """
    session, why = _session()
    if session is None:
        return _fail("not_launched_through_ayon", why)
    launch = launch_context_of(session["comfy_url"])
    bad = _need_launch(session, launch)
    if bad:
        return bad
    folder_path = (folder_path or launch["folder_path"]).strip("/")
    files = expand_files(path)
    if not files:
        return _fail("error", f"the file does not exist on this machine: {path}")
    k = _kind_of("", files[0], kind)
    spec = KINDS[k]
    namespace, name = folder_path.split("/")[-1], product
    resolved = {"project": launch["project"], "folder": folder_path, "product": product, "version": int(version or 0),
                "representation_id": representation_id, "kind": k, "files": files[:5], "file_count": len(files)}
    if dry_run:
        return json.dumps({"status": "dry_run", "resolved": resolved, "would_upload_to": f"input/AYON/{namespace}_{name}",
                           "would_place": spec["load_node"], "comfy_url": session["comfy_url"],
                           "message": "Nothing was uploaded or changed."})
    client = _client_for(session)
    try:
        uploads = [_upload(client, f, f"{namespace}_{name}") for f in files]
    except (AyonError, OSError, requests.RequestException) as e:
        return _fail("error", f"upload to ComfyUI failed: {e}", resolved=resolved)
    container = build_container(name, namespace, representation_id, uploads, spec["loader"])
    container_json = json.dumps(container)
    canvas = read_canvas(client)
    placed: dict[str, Any] = {"how": "none"}
    if load_node_id:
        r = client.post("/agent/set_node_params", json_data={"node_id": str(load_node_id),
                                                             "params": {spec["load_widget"]: container_json}})
        if isinstance(r, dict) and spec["load_widget"] in (r.get("applied") or []):
            placed = {"how": "filled_existing", "node_id": str(load_node_id)}
        else:
            placed = {"how": "none", "problem": (r or {}).get("error") if isinstance(r, dict) else str(r)}
    elif _canvas_placer is not None:
        nid = _next_ids(canvas, 1)[0]
        label = f"v{int(version):03d}" if version else ""
        if _canvas_placer([{"op": "add", "class_type": spec["load_node"], "node_id": nid,
                            "params": {spec["load_widget"]: container_json}}], f"Ayon load {product} {label}".strip()):
            placed = {"how": "added", "node_id": nid, "class_type": spec["load_node"]}
    if placed["how"] == "none":
        placed["paste"] = {"class_type": spec["load_node"], "widget": spec["load_widget"], "value": container_json}
    ctx_id, doc = _context_state(canvas, session["comfy_url"])
    doc["containers"] = _merge_unique(doc["containers"], [container], "container_uuid")
    registered = _write_context(client, session, canvas, ctx_id, doc, launch, "Ayon: register container")
    return json.dumps({"status": "loaded" if placed["how"] != "none" else "prepared", "resolved": resolved,
                       "container": {k2: container[k2] for k2 in ("name", "namespace", "container_uuid", "loader")},
                       "uploaded": uploads, "placed": placed, "registered": registered})


def build_instance(kind: str, product_type: str, variant: str, launch: dict, prefix: str = _DEFAULT_NAME,
                   unique_name: str = _DEFAULT_NAME, use_unique_name: bool = True, product_name: str = "",
                   force_recook: bool = False, extra: Optional[dict] = None) -> dict:
    """The dict ``CreatedInstance.data_to_store()`` holds for a ComfyUI creator (create_image.py / _video / _model)."""
    spec = KINDS[kind]
    variant = variant or "Main"
    unique = unique_name if use_unique_name else ""
    base = product_name or f"{product_type}{variant[:1].upper()}{variant[1:]}"
    full = "_".join(p for p in (prefix, unique, base) if p) if not product_name else product_name
    attrs: dict[str, Any] = {"keep_metadata": True, "use_unique_name": use_unique_name, "prefix": prefix,
                             "unique_name": unique, "force_recook_on_publish": force_recook}
    if kind == "image":
        attrs["compression_level"] = 4
    elif kind == "video":
        attrs.update(formatcodec="mp4/h264", webm_noaudio=False, webm_crf=32)
    else:
        attrs["fallback_format"] = "glb"
    attrs.update(extra or {})
    return {"id": AYON_INSTANCE_ID, "productType": product_type, "productBaseType": product_type,
            "productName": full, "active": True, "creator_identifier": spec["creator"], "variant": variant,
            "folderPath": launch["folder_path"], "task": launch["task"], "projectName": launch["project"],
            "instance_id": str(uuid.uuid4()), "creator_attributes": attrs, "publish_attributes": {}}


@tool
def ayon_mark_publish(node_id: str = "", kind: str = "image", variant: str = "Main", prefix: str = "ayon",
                      unique_name: str = "ayon", product_name: str = "", output: int = 0,
                      save_node_id: str = "", force_recook: bool = False) -> str:
    """Make an output of the canvas a publish instance, the way the Ayon Creator does.

    Adds an AYON Image / Video / 3D Model Save node (``ayon_info`` filled with the creator's instance data),
    wires the given node's output into it, and registers the instance on the AYON Context node, which is what
    the Publisher reads. Nothing is published: run ``ayon_publish`` for that.

    Args:
        node_id: The canvas node whose output should be published (a decode / video / mesh node).
        kind: ``image`` | ``video`` | ``model``.
        variant: Variant name (``Main``); it ends the product name (``ayon_ayon_imageMain``).
        prefix: File prefix the Creator proposes (default ``ayon``).
        unique_name: Unique name part of the product name (default ``ayon``); '' to leave it out.
        product_name: Full product name, to override the one built from the parts.
        output: Output slot of ``node_id`` to wire (0 = the first).
        save_node_id: An AYON Save node already on the canvas to fill instead of adding one.
        force_recook: Re-run the graph for this node at publish time.

    Returns JSON: ``{status, instance, placed, wired, registered, todo}``.
    """
    if kind not in KINDS:
        return _fail("error", f"kind must be one of {sorted(KINDS)}.")
    session, why = _session()
    if session is None:
        return _fail("not_launched_through_ayon", why)
    launch = launch_context_of(session["comfy_url"])
    bad = _need_launch(session, launch)
    if bad:
        return bad
    spec = KINDS[kind]
    client = _client_for(session)
    canvas = read_canvas(client)
    if node_id and canvas["source"] == "canvas" and not any(n["id"] == str(node_id) for n in canvas["nodes"]):
        return _fail("error", f"node {node_id} is not on the canvas.")
    inst = build_instance(kind, spec["product_type"], variant, launch, prefix, unique_name,
                          bool(unique_name), product_name, force_recook)
    info_json = json.dumps(inst)
    placed: dict[str, Any] = {"how": "none"}
    wired = False
    todo: list[str] = []
    if save_node_id:
        r = client.post("/agent/set_node_params", json_data={"node_id": str(save_node_id), "params": {
            "ayon_info": info_json, "recook": bool(force_recook)}})
        if isinstance(r, dict) and "ayon_info" in (r.get("applied") or []):
            placed = {"how": "filled_existing", "node_id": str(save_node_id)}
        else:
            placed = {"how": "none", "problem": (r or {}).get("error") if isinstance(r, dict) else str(r)}
    elif _canvas_placer is not None:
        nid = _next_ids(canvas, 1)[0]
        ops: list[dict] = [{"op": "add", "class_type": spec["save_node"], "node_id": nid,
                            "params": {"ayon_info": info_json, "recook": bool(force_recook)},
                            **({"near": str(node_id)} if node_id else {})}]
        if node_id:
            ops.append({"op": "connect", "from": str(node_id), "output": int(output), "to": nid,
                        "input": spec["save_input"]})
        if _canvas_placer(ops, f"Ayon publish instance {inst['productName']}"):
            placed = {"how": "added", "node_id": nid, "class_type": spec["save_node"]}
            wired = bool(node_id)
    if placed["how"] == "none":
        placed["paste"] = {"class_type": spec["save_node"], "widget": "ayon_info", "value": info_json}
        todo.append(f"Add an {spec['save_node']} node and paste the value above into its 'ayon_info' field.")
    if placed["how"] != "none" and not wired:
        todo.append(f"Wire the output you want published into the save node's '{spec['save_input']}' input"
                    + (f" (node {placed.get('node_id')})." if placed.get("node_id") else "."))
    ctx_id, doc = _context_state(canvas, session["comfy_url"])
    doc["instances"] = _merge_unique(doc["instances"], [inst], "instance_id")
    registered = _write_context(client, session, canvas, ctx_id, doc, launch, "Ayon: register publish instance")
    if not registered["registered"]:
        todo.append("The instance is not registered on the AYON Context node, so the Publisher will not list it: "
                    + str(registered.get("problem", "")))
    return json.dumps({"status": "marked" if not todo else "partly_marked", "instance": {
        k: inst[k] for k in ("instance_id", "productType", "productName", "variant", "folderPath", "task")},
        "placed": placed, "wired": wired, "registered": registered, "todo": todo})


@tool
def ayon_publish() -> str:
    """Start the Ayon publish for this session: opens the Ayon Publisher on the registered instances.

    The publish itself (pyblish: collect, validate, extract, integrate) runs inside the Ayon host's Qt
    process, and no RPC runs it without that window. So this does NOT publish by itself: it checks that
    instances are registered, opens the Publisher, and says what the user presses. The version it creates can be
    looked up (or commented on) afterwards through the ayon-mcp server, by the product names returned here.

    Returns JSON: ``{status, instances, needs_user}``.
    """
    session, why = _session()
    if session is None:
        return _fail("not_launched_through_ayon", why)
    if not session["rpc_ok"]:
        return _fail("error", f"the Ayon host's RPC server is not answering: {session['rpc_error']}")
    client = _client_for(session)
    _ctx_id, doc = _context_state(read_canvas(client), session["comfy_url"])
    names = [i.get("productName") for i in doc["instances"] if i.get("productName")]
    if not names:
        return _fail("nothing_to_publish", "no publish instance is registered on the canvas. Mark an output "
                     "with ayon_mark_publish first (or create one in the Ayon Creator).")
    try:
        rpc_call(session["rpc_port"], f"{RPC_NAMESPACE}.requestToolByName", {"tool_name": "publisher"})
    except AyonRpcError as e:
        return _fail("error", f"could not open the Publisher: {e}")
    return json.dumps({"status": "publisher_opened", "instances": names, "headless": False,
                       "needs_user": "Press Publish in the Ayon Publisher window (Qt, on the artist's machine)."})


@tool
def ayon_open_tool(tool_name: str) -> str:
    """Open one of the Ayon host's own tools (a window on the artist's machine) from the running session.

    The one thing the Ayon RPC server offers beyond a ping. Use it for what has no tool of its own: the Loader
    (pick a representation by hand), Creator, Publisher, Workfiles, Scene Inventory.

    Args:
        tool_name: ``loader`` | ``create`` | ``publisher`` | ``workfiles`` | ``sceneinventory``.
    """
    if tool_name not in TOOLS:
        return _fail("error", f"tool_name must be one of {list(TOOLS)}.")
    session, why = _session()
    if session is None:
        return _fail("not_launched_through_ayon", why)
    try:
        got = rpc_call(session["rpc_port"], f"{RPC_NAMESPACE}.requestToolByName", {"tool_name": tool_name})
    except AyonRpcError as e:
        return _fail("error", str(e))
    return json.dumps({"status": "requested", "tool": tool_name, "host_answer": got})
