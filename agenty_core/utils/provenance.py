"""Provenance: the exact graph that made a file, kept beside it (format v1).

A generated file carries (or can be traced to) the API-format graph ComfyUI ran.
That is what lets "recreate this shot with one change" re-run the original with
everything else identical.

On disk, next to ``<file>``:

* ``<file>.agenty.json`` - the existing sidecar; provenance is merged in under the
  key ``"provenance"`` (everything else in the sidecar is kept);
* ``<file>.prov.json.gz`` - gzip of ``{"v": 1, "prompt": <API graph>, "workflow":
  <UI workflow or null>}``.

Secrets are never stored: :func:`redact_graph` runs on everything written.
The graph is INTERNAL - it never goes into a tool result (see tools/regenerate.py).

Where :func:`provenance_of` looks, in order: the sidecar (exact), the PNG text
chunks / video container metadata ComfyUI and VHS embed (high), ComfyUI's
``/history`` (high/medium, in memory only), else nothing. Nothing in a file is ever
executed: metadata is parsed as JSON text.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import zlib
from pathlib import Path

VERSION = 1
SIDECAR_SUFFIX = ".agenty.json"
GRAPH_SUFFIX = ".prov.json.gz"
REDACTED = "<redacted>"

_SECRET_NAME = re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|auth|credential)")
_MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin", ".sft")
_MODEL_INPUT = re.compile(
    r"(?i)^(ckpt_name|unet_name|vae_name|clip_name\d*|lora_name|model_name|control_net_name|"
    r"controlnet_name|upscale_model_name|style_model_name|gligen_name|clip_vision_name|"
    r"diffusion_model|text_encoder|lora_name_\d+)$")
_SEED_INPUT = re.compile(r"(?i)^(seed|noise_seed|seed_\w+)$")
#: Class names of ComfyUI's partner (API) nodes. Offline: the real test is the node's
#: ``api_node`` flag in /object_info (partner_credits / partner_run use that).
_PARTNER_CLASS = re.compile(
    r"^(Kling|Veo|Runway|Luma|Pika|Pixverse|Vidu|Minimax|MiniMax|Moonvalley|Recraft|Ideogram|Stability|"
    r"BFL|Flux\w*(?:Pro|Ultra|Kontext)|OpenAI|GPTImage|Dalle|Gemini|NanoBanana|ByteDance|Bytedance|"
    r"Seedance|Seedream|Sora|Grok|Tripo|Rodin|Meshy|ElevenLabs|Magnific|Topaz|Hitpaw|Reve[A-Z_]|"
    r"Higgsfield|Wavespeed|Sonilo|Veed|Quiver|Hunyuan3D\w*Api)")
_PARTNER_API = re.compile(r"(?:[a-z0-9]Api|^Api)")      # WanImageToVideoApi, LtxvApiImageToVideo
_PARTNER_NOT = re.compile(r"(?i)^(VHS_|Load|Save|Preview|GetVideo|CreateVideo|SaveVideo|Primitive|Note|Reroute)")
_MAX_PARTNER_STR = 200

#: Held while a ``.agenty.json`` sidecar is read, changed and written back. A host that
#: keeps other keys in the same sidecar (agentY's output tags) takes it too, so two
#: writers on two threads never lose each other's key.
SIDECAR_LOCK = threading.RLock()


# ── redaction ────────────────────────────────────────────────────────────────

def _redact_value(v):
    from agenty_core.utils.redact import redact_obj
    return redact_obj(v)


def _redact_walk(obj, key_name: str = ""):
    if isinstance(obj, dict):
        return {k: _redact_walk(v, str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_walk(v, key_name) for v in obj]
    if isinstance(obj, str):
        if key_name and _SECRET_NAME.search(key_name) and obj:
            return REDACTED
        return _redact_value(obj)
    return obj


def _redact_ui_workflow(wf: dict) -> dict:
    """A UI workflow keeps widget values as a bare list: a node that has a secret-named
    widget gets every string widget value masked (the file is informational)."""
    wf = _redact_walk(copy.deepcopy(wf))
    for node in wf.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        names = [str(i.get("widget", {}).get("name", "") or i.get("name", ""))
                 for i in node.get("inputs") or [] if isinstance(i, dict)]
        if any(_SECRET_NAME.search(n) for n in names) and isinstance(node.get("widgets_values"), list):
            node["widgets_values"] = [REDACTED if isinstance(x, str) and x else x
                                      for x in node["widgets_values"]]
    return wf


def redact_graph(graph: dict) -> dict:
    """A deep copy of an API graph with every secret masked: string inputs whose NAME
    looks like a credential become ``"<redacted>"``, and every string passes
    through ``utils.redact.redact`` (key shapes, configured values)."""
    if not isinstance(graph, dict):
        return {}
    if "nodes" in graph and "links" in graph:       # a UI workflow
        return _redact_ui_workflow(graph)
    return _redact_walk(copy.deepcopy(graph))


def canonical_sha1(graph: dict) -> str:
    text = json.dumps(graph, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ── summary ──────────────────────────────────────────────────────────────────

def _is_link(v) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], int)
            and not isinstance(v[0], bool) and isinstance(v[0], (str, int)))


def is_partner_class(class_type: str) -> bool:
    ct = str(class_type or "")
    if not ct or _PARTNER_NOT.match(ct):
        return False
    return bool(_PARTNER_CLASS.search(ct) or _PARTNER_API.search(ct))


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool)) and not isinstance(v, type(None))


def _nodes(graph: dict):
    for nid, node in (graph or {}).items():
        if isinstance(node, dict) and node.get("class_type"):
            yield str(nid), node, (node.get("inputs") or {})


def summarize(graph: dict) -> dict:
    """``{"models", "seeds", "partner"}`` of an API graph (small, redacted values)."""
    models: list = []
    seeds: list = []
    partner: list = []
    for nid, node, inputs in _nodes(graph):
        for name, v in inputs.items():
            if _is_link(v):
                continue
            if isinstance(v, str) and (_MODEL_INPUT.match(name) or v.lower().endswith(_MODEL_EXT)):
                models.append({"node": nid, "input": name, "value": v})
            elif (_SEED_INPUT.match(name) and isinstance(v, int) and not isinstance(v, bool)):
                seeds.append({"node": nid, "input": name, "value": v})
        ct = node["class_type"]
        if is_partner_class(ct):
            params = {}
            for name, v in inputs.items():
                if _is_link(v) or not _scalar(v):
                    continue
                if _SECRET_NAME.search(name):
                    params[name] = REDACTED if v else v
                elif isinstance(v, str):
                    params[name] = _redact_value(v)[:_MAX_PARTNER_STR]
                else:
                    params[name] = v
            partner.append({"node": nid, "class": ct, "params": params})
    return {"models": models, "seeds": seeds, "partner": partner}


# ── write ────────────────────────────────────────────────────────────────────

def sidecar_path(path) -> Path:
    return Path(str(path) + SIDECAR_SUFFIX)


def graph_path(path) -> Path:
    return Path(str(path) + GRAPH_SUFFIX)


def _atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write(path, graph: dict, *, prompt_id: str = "", output_node: str = "", output_index: int = 0,
          workflow_file: str = "", workflow_ui: dict | None = None, comfyui_version: str = "",
          source: str = "agentY", derived_from: dict | None = None, created: str | None = None) -> bool:
    """Store *graph* as the provenance of the file *path* (best effort; never raises)."""
    try:
        if not path:
            return False
        p = Path(str(path))
        red = redact_graph(graph)
        if not red:
            return False
        ui = redact_graph(workflow_ui) if isinstance(workflow_ui, dict) and workflow_ui else None
        gfile = graph_path(p)
        blob = json.dumps({"v": VERSION, "prompt": red, "workflow": ui},
                          ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        _atomic_write(gfile, gzip.compress(blob, 6, mtime=0))
        prov = {
            "v": VERSION, "source": source, "prompt_id": str(prompt_id or ""),
            "output_node": str(output_node or ""), "output_index": int(output_index or 0),
            "workflow_file": str(workflow_file or ""), "comfyui_version": str(comfyui_version or ""),
            "created": created or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "graph_file": gfile.name, "graph_sha1": canonical_sha1(red),
            **summarize(red),
        }
        if derived_from:
            prov["derived_from"] = _redact_walk(copy.deepcopy(derived_from))
        sc = sidecar_path(p)
        with SIDECAR_LOCK:
            existing = _read_json(sc) if sc.is_file() else None
            doc = existing if isinstance(existing, dict) else {}
            doc["provenance"] = prov
            _atomic_write(sc, json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"))
        return True
    except Exception:  # noqa: BLE001 - provenance is never worth failing a run
        return False


# ── capture at the end of a run ──────────────────────────────────────────────

_pending: list = []
_pending_lock = threading.Lock()


def _capture_jobs(paths, history, prompt_id: str) -> tuple[dict, list]:
    entry = history.get(prompt_id) if isinstance(history, dict) else None
    if not isinstance(entry, dict) and isinstance(history, dict) and "prompt" in history:
        entry = history
    graph = graph_from_history(entry or {})
    if not _looks_api(graph):
        return {}, []
    return graph, [(str(p), *output_position(entry, os.path.basename(str(p)))) for p in paths or () if p]


def capture(paths, history, prompt_id: str, workflow_path: str = "", *, source: str = "agentY",
            background: bool = True):
    """Write the provenance of a finished run's output files *paths*, from the run's own
    ``/history`` entry (*history*: ``{prompt_id: entry}`` or the entry): the exact API
    prompt that ran - never its extra_data - with the output node and index of each file.

    On a background thread by default (an output folder on a network share is slow): it
    returns the thread, or with ``background=False`` how many were written. Never raises.
    :func:`wait_pending` waits for what was started."""
    try:
        graph, jobs = _capture_jobs(paths, history, str(prompt_id or ""))
    except Exception:  # noqa: BLE001
        graph, jobs = {}, []
    if not jobs:
        return None if background else 0

    def write_all() -> int:
        n = 0
        for path, node, index in jobs:
            if write(path, graph, prompt_id=str(prompt_id or ""), output_node=node, output_index=index,
                     workflow_file=str(workflow_path or ""), source=source):
                n += 1
        return n
    if not background:
        return write_all()

    def run():
        try:
            write_all()
        finally:
            with _pending_lock:
                if t in _pending:
                    _pending.remove(t)

    t = threading.Thread(target=run, name="provenance-capture", daemon=True)
    with _pending_lock:
        _pending.append(t)
    t.start()
    return t


def wait_pending(timeout: float = 60.0) -> None:
    """Wait for the provenance writes :func:`capture` started so far."""
    with _pending_lock:
        pending = list(_pending)
    for t in pending:
        t.join(timeout)


def recorded(path) -> dict:
    """The ``provenance`` key of *path*'s sidecar ({} when there is none)."""
    doc = _read_json(sidecar_path(path)) if sidecar_path(path).is_file() else None
    prov = doc.get("provenance") if isinstance(doc, dict) else None
    return prov if isinstance(prov, dict) else {}


def move_with_sidecars(src, dst) -> None:
    """Move *src* to *dst* and its ``.agenty.json`` / ``.prov.json.gz`` along; the
    graph file is renamed to match and the sidecar's ``graph_file`` follows."""
    src, dst = Path(str(src)), Path(str(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with SIDECAR_LOCK:
        shutil.move(str(src), str(dst))
        for suf in (GRAPH_SUFFIX, SIDECAR_SUFFIX):
            s = Path(str(src) + suf)
            if s.is_file():
                shutil.move(str(s), str(Path(str(dst) + suf)))
        sc = sidecar_path(dst)
        doc = _read_json(sc) if sc.is_file() else None
        if isinstance(doc, dict) and isinstance(doc.get("provenance"), dict):
            doc["provenance"]["graph_file"] = graph_path(dst).name
            try:
                _atomic_write(sc, json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8"))
            except OSError:
                pass


# ── history ──────────────────────────────────────────────────────────────────

def graph_from_history(history_entry) -> dict:
    """The API graph of a ``/history/<id>`` entry (never its extra_data)."""
    try:
        prompt = history_entry.get("prompt")
        if isinstance(prompt, (list, tuple)) and len(prompt) >= 3 and isinstance(prompt[2], dict):
            return prompt[2]
        if isinstance(prompt, dict):
            return prompt
    except AttributeError:
        pass
    return {}


def _history_prompt_id(history_entry) -> str:
    try:
        pr = history_entry.get("prompt")
        if isinstance(pr, (list, tuple)) and len(pr) >= 2:
            return str(pr[1])
    except AttributeError:
        pass
    return ""


_OUT_KEYS = ("images", "gifs", "videos", "audio", "video", "files", "3d", "meshes")


def output_position(history_entry, filename: str) -> tuple[str, int]:
    """``(node_id, index)`` of *filename* among the entry's outputs; ``("", 0)`` if absent."""
    name = os.path.basename(str(filename or ""))
    outs = (history_entry or {}).get("outputs") or {}
    for nid, nout in outs.items():
        if not isinstance(nout, dict):
            continue
        for key in _OUT_KEYS:
            items = nout.get(key)
            if not isinstance(items, list):
                continue
            for i, it in enumerate(items):
                fn = it.get("filename") if isinstance(it, dict) else (it if isinstance(it, str) else "")
                if fn and os.path.basename(str(fn)) == name:
                    return str(nid), i
    return "", 0


# ── PNG / video metadata (parsed, never executed) ────────────────────────────

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def png_text_chunks(path) -> dict:
    """``{keyword: text}`` of a PNG's tEXt / zTXt / iTXt chunks ({} when none / not a PNG)."""
    out: dict = {}
    try:
        with open(path, "rb") as f:
            if f.read(8) != _PNG_SIG:
                return {}
            while True:
                head = f.read(8)
                if len(head) < 8:
                    break
                length, ctype = struct.unpack(">I4s", head)
                if ctype in (b"tEXt", b"zTXt", b"iTXt"):
                    if length > 64 * 1024 * 1024:
                        break
                    data = f.read(length)
                    f.read(4)
                    try:
                        if ctype == b"tEXt":
                            k, _, v = data.partition(b"\x00")
                            out[k.decode("latin-1")] = v.decode("utf-8", "replace")
                        elif ctype == b"zTXt":
                            k, _, rest = data.partition(b"\x00")
                            out[k.decode("latin-1")] = zlib.decompress(rest[1:]).decode("utf-8", "replace")
                        else:
                            k, _, rest = data.partition(b"\x00")
                            comp, rest = rest[0], rest[2:]
                            _lang, _, rest = rest.partition(b"\x00")
                            _tr, _, text = rest.partition(b"\x00")
                            if comp:
                                text = zlib.decompress(text)
                            out[k.decode("utf-8", "replace")] = text.decode("utf-8", "replace")
                    except Exception:  # noqa: BLE001
                        continue
                elif ctype == b"IEND":
                    break
                else:
                    f.seek(length + 4, 1)
    except OSError:
        return {}
    return out


def _ffprobe() -> str:
    return shutil.which("ffprobe") or ""


def video_tags(path) -> dict:
    """The container / stream metadata tags of a video, lower-cased keys (ffprobe, else PyAV)."""
    tags: dict = {}
    exe = _ffprobe()
    if exe:
        try:
            r = subprocess.run([exe, "-v", "quiet", "-show_format", "-show_streams",
                                "-print_format", "json", str(path)],
                               capture_output=True, timeout=60)
            doc = json.loads(r.stdout.decode("utf-8", "replace") or "{}")
            for src in [doc.get("format") or {}] + list(doc.get("streams") or []):
                for k, v in (src.get("tags") or {}).items():
                    tags.setdefault(str(k).lower(), v)
        except Exception:  # noqa: BLE001
            pass
    if not tags:
        try:
            import av  # type: ignore
            with av.open(str(path)) as c:
                for k, v in c.metadata.items():
                    tags.setdefault(str(k).lower(), v)
        except Exception:  # noqa: BLE001
            pass
    return tags


def _looks_api(g) -> bool:
    return (isinstance(g, dict) and bool(g)
            and all(isinstance(v, dict) and "class_type" in v for v in g.values()))


def _json_or_none(text):
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _pull(meta: dict) -> tuple[dict, dict | None]:
    """(API graph, UI workflow) out of a metadata dict with ``prompt`` / ``workflow`` / ``comment``."""
    graph, wf = {}, None
    for key in ("prompt", "comment", "description"):
        doc = _json_or_none(meta.get(key))
        if _looks_api(doc):
            graph = doc
            break
        if isinstance(doc, dict):
            inner = _json_or_none(doc.get("prompt"))
            if _looks_api(inner):
                graph = inner
                w = _json_or_none(doc.get("workflow"))
                wf = w if isinstance(w, dict) else None
                break
    if wf is None:
        w = _json_or_none(meta.get("workflow"))
        wf = w if isinstance(w, dict) else None
    return graph, wf


_SAVE_CLASS = re.compile(r"(?i)(^Save|SaveImage|SaveVideo|VideoCombine|SendToViewer|SaveAnimated|SaveWEBP|SaveAudio)")


def _pick_output(graph: dict, filename: str) -> tuple[str, bool]:
    """(node id, certain): the save node a file came from, guessed from the graph."""
    saves = [(nid, inputs) for nid, node, inputs in _nodes(graph) if _SAVE_CLASS.search(node["class_type"])]
    if len(saves) == 1:
        return saves[0][0], True
    stem = Path(filename).name
    hit = []
    for nid, inputs in saves:
        pre = inputs.get("filename_prefix")
        if isinstance(pre, str) and pre:
            base = pre.replace("\\", "/").rsplit("/", 1)[-1]
            if base and stem.startswith(base):
                hit.append(nid)
    if len(hit) == 1:
        return hit[0], True
    return (hit[0] if hit else (saves[0][0] if saves else "")), False


# ── lookup ───────────────────────────────────────────────────────────────────

NOT_FOUND_NOTE = ("can't recreate exactly: no provenance in a sidecar, the file's metadata, "
                  "or ComfyUI's history")
_IMG_EXT = {".png"}
_VID_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".gif", ".webp"}


def _result(found, source, confidence, graph=None, **kw) -> dict:
    return {"found": found, "source": source, "confidence": confidence,
            "graph": graph if graph is not None else {},
            "prompt_id": kw.get("prompt_id", ""), "output_node": kw.get("output_node", ""),
            "output_index": kw.get("output_index", 0), "workflow_file": kw.get("workflow_file", ""),
            "note": kw.get("note", ""), "workflow_ui": kw.get("workflow_ui")}


def _from_sidecar(p: Path) -> dict | None:
    doc = _read_json(sidecar_path(p)) if sidecar_path(p).is_file() else None
    prov = doc.get("provenance") if isinstance(doc, dict) else None
    if not isinstance(prov, dict):
        return None
    gf = p.parent / str(prov.get("graph_file") or graph_path(p).name)
    if not gf.is_file():
        gf = graph_path(p)
    try:
        blob = json.loads(gzip.decompress(gf.read_bytes()).decode("utf-8"))
    except (OSError, ValueError, EOFError, zlib.error):
        return None
    g = blob.get("prompt") if isinstance(blob, dict) else None
    if not _looks_api(g):
        return None
    return _result(True, "sidecar", "exact", redact_graph(g), prompt_id=str(prov.get("prompt_id") or ""),
                   output_node=str(prov.get("output_node") or ""), output_index=int(prov.get("output_index") or 0),
                   workflow_file=str(prov.get("workflow_file") or ""),
                   workflow_ui=blob.get("workflow") if isinstance(blob.get("workflow"), dict) else None)


def _from_media(p: Path) -> dict | None:
    ext = p.suffix.lower()
    if ext in _IMG_EXT:
        meta, src = png_text_chunks(p), "png"
    elif ext in _VID_EXT:
        meta, src = video_tags(p), "video"
    else:
        return None
    graph, wf = _pull(meta)
    if not graph:
        return None
    nid, sure = _pick_output(graph, p.name)
    return _result(True, src, "high" if (sure or not nid) else "medium", redact_graph(graph),
                   output_node=nid, workflow_ui=redact_graph(wf) if wf else None,
                   note="" if sure else "output node guessed from the graph")


def _from_history(p: Path, prompt_id: str) -> dict | None:
    if not prompt_id:
        return None
    try:
        from urllib.parse import quote
        from agenty_core.utils.comfyui_client import get_client
        doc = get_client().get(f"/history/{quote(prompt_id, safe='')}")
    except Exception:  # noqa: BLE001
        return None
    entry = doc.get(prompt_id) if isinstance(doc, dict) else None
    if not isinstance(entry, dict) and isinstance(doc, dict) and "prompt" in doc:
        entry = doc
    graph = graph_from_history(entry or {})
    if not _looks_api(graph):
        return None
    nid, idx = output_position(entry, p.name)
    if nid:
        return _result(True, "history", "high", redact_graph(graph), prompt_id=prompt_id,
                       output_node=nid, output_index=idx)
    guess, _ = _pick_output(graph, p.name)
    return _result(True, "history", "medium", redact_graph(graph), prompt_id=prompt_id,
                   output_node=guess, note="the file is not among the history entry's outputs")


def provenance_of(path, *, prompt_id: str = "", use_history: bool = True) -> dict:
    """The provenance of a file: sidecar, then embedded metadata, then ComfyUI's history.

    ``{"found", "source", "confidence", "graph", "prompt_id", "output_node", "output_index",
    "workflow_file", "note"}``. ``graph`` is INTERNAL: never put it into a tool result."""
    p = Path(str(path))
    sc_prompt = ""
    try:
        got = _from_sidecar(p)
        if got:
            return got
        doc = _read_json(sidecar_path(p)) if sidecar_path(p).is_file() else None
        if isinstance(doc, dict):
            sc_prompt = str((doc.get("provenance") or {}).get("prompt_id") or doc.get("prompt_id") or "")
        if p.is_file():
            got = _from_media(p)
            if got:
                return got
    except Exception:  # noqa: BLE001
        pass
    if use_history:
        try:
            got = _from_history(p, str(prompt_id or sc_prompt or ""))
            if got:
                return got
        except Exception:  # noqa: BLE001
            pass
    return _result(False, "none", "none", note=NOT_FOUND_NOTE)
