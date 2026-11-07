"""Dependencies between files: what a file was made from, and what was made from it.

Provenance (:mod:`agenty_core.utils.provenance`) keeps, for each output, the graph
that made it. That answers "how was this made" but not the two questions a shot
needs answered when the same plate is worked on again:

* **reuse** - "does a depth pass of this clip, frames 0-96, already exist?" Then
  use it instead of spending minutes of GPU on it again, and keep variants built
  on the same mask lined up with each other;
* **staleness** - "the plate was re-graded: which masks, passes and renders were
  made from the old one?"

So each output also records, in its sidecar under ``provenance.deps``:

* ``inputs`` - the files its graph loaded (resolved paths, each with the
  :func:`fingerprint` it had when it was read);
* ``kind`` - what sort of output it is: ``render`` when a sampler / generator made
  it, else the pass its graph computed (``mask``, ``depth``, ``pose``, ...), else
  ``derived`` (a resize, a crop);
* ``params`` - what decides whether another run would make the same pass: the
  pass node's own settings and the loader's frame range.

And each input gets the reverse record, ``agent_metadata/<name>.derived.json``
beside it: the outputs made from it. Both are plain JSON next to the data, so they
travel with a folder.

A file is identified by its :func:`fingerprint` (size + hash of its first and last
MiB), not by its path: the copy agentY stages into ComfyUI's input dir is the same
file as the output it came from.

Everything is best effort: nothing in here raises into a run.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from pathlib import Path

VERSION = 1
DERIVED_SUFFIX = ".derived.json"
_CHUNK = 1 << 20                 # bytes hashed at each end of a file
_MAX_PARAM_STR = 200
_MAX_DERIVED = 500               # entries kept per input (newest win)

# ── what kind of output a graph made ─────────────────────────────────────────

#: Nodes that generate: anything downstream of one of these is a render.
_GENERATIVE = re.compile(
    r"(?i)(sampler|guider|ksampler|generate|text2image|texttoimage|t2i|i2v|t2v|img2vid|"
    r"imagetovideo|texttovideo|_api$|^api_|flux.*kontext.*pro|seedream|seedance|kling|veo|"
    r"runway|luma|minimax|pika|ideogram|recraft|openai|gemini|dashscope|elevenlabs|trellis|"
    r"hunyuan3d|tripo|rodin|stableaudio|acestep|ace_step)")

#: Pass nodes, first match wins. (kind, pattern)
PASS_KINDS: list[tuple[str, re.Pattern]] = [
    ("camera", re.compile(r"(?i)(colmap|glomap|cameratrack|camera_?solve|camerasolve|vggt|megasam|"
                          r"camera_?track)")),
    ("key", re.compile(r"(?i)(corridorkey|keyer|chroma_?key|chromakey)")),
    ("mask", re.compile(r"(?i)(sam(2|3|_?hq)?[^a-z]|^sam|segment|groundingdino|birefnet|rmbg|rembg|"
                        r"backgroundremov|remove_?background|matting|matte|vitmatte|clipseg|"
                        r"yolo.*(seg|mask)|mask_?from|personmask|facemask|roto)")),
    ("depth", re.compile(r"(?i)(depth|midas|zoe|marigold|leres)")),
    ("normals", re.compile(r"(?i)(normal_?map|normalmap|normals|dsine|bae_?normal|moge)")),
    ("pose", re.compile(r"(?i)(dwpose|openpose|dwpreprocessor|poseestimat|densepose|"
                        r"mediapipe.*(pose|face|hand)|animalpose|vitpose)")),
    ("edges", re.compile(r"(?i)(canny|lineart|line_?art|softedge|pidinet|mlsd|scribble|"
                         r"anyline|^hed|hed_?pre|^teed)")),
    ("flow", re.compile(r"(?i)(opticalflow|optical_?flow|^raft|raft_?flow|flowformer)")),
    ("upscale", re.compile(r"(?i)(upscalewithmodel|upscale_?model|esrgan|seedvr|supir|ultimatesd)")),
]

#: Loader settings that decide which frames / size a pass covers.
_FRAME_INPUT = re.compile(
    r"(?i)^(frame_load_cap|skip_first_frames|select_every_nth|start_frame|end_frame|frame_count|"
    r"frames?|force_rate|force_size|custom_width|custom_height|start_time|duration|fps)$")
_MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin", ".sft", ".onnx")
_MEDIA_EXT = {
    ".png", ".jpg", ".jpeg", ".webp", ".exr", ".tif", ".tiff", ".bmp", ".gif", ".dpx", ".hdr",
    ".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".mxf",
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
    ".glb", ".gltf", ".obj", ".fbx", ".ply", ".spz", ".splat", ".usd", ".usda", ".usdc", ".usdz",
    ".abc", ".json", ".npz", ".npy",
}
_ANNOT = re.compile(r"\s*\[(input|output|temp)\]\s*$")


def _is_link(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], int) and not isinstance(v[1], bool)


def _nodes(graph: dict):
    for nid, node in (graph or {}).items():
        if isinstance(node, dict) and node.get("class_type"):
            yield str(nid), node, (node.get("inputs") or {})


def upstream(graph: dict, start: str) -> list[str]:
    """Node ids feeding *start* (itself first), nearest first."""
    nodes = {nid: inputs for nid, _n, inputs in _nodes(graph)}
    if start not in nodes:
        return []
    seen, order, queue = {start}, [start], [start]
    while queue:
        nid = queue.pop(0)
        for v in nodes.get(nid, {}).values():
            if _is_link(v):
                src = str(v[0])
                if src in nodes and src not in seen:
                    seen.add(src)
                    order.append(src)
                    queue.append(src)
    return order


def _output_nodes(graph: dict) -> list[str]:
    """Nodes nothing reads from: where a graph's results come out."""
    read = {str(v[0]) for _nid, _n, inputs in _nodes(graph) for v in inputs.values() if _is_link(v)}
    return [nid for nid, _n, _i in _nodes(graph) if nid not in read]


def kind_of_class(class_type: str) -> str:
    """The pass a node class computes, or ``""``."""
    for kind, pat in PASS_KINDS:
        if pat.search(str(class_type or "")):
            return kind
    return ""


def classify(graph: dict, output_node: str = "") -> dict:
    """``{"kind", "pass_node", "pass_class", "nodes"}`` for the output of *output_node*
    (all output nodes when it is unknown). ``nodes`` are the ids that fed it."""
    starts = [output_node] if output_node and output_node in (graph or {}) else _output_nodes(graph)
    order: list[str] = []
    for s in starts:
        for nid in upstream(graph, s):
            if nid not in order:
                order.append(nid)
    by_id = {nid: node for nid, node, _i in _nodes(graph)}
    if any(_GENERATIVE.search(by_id[n]["class_type"]) for n in order if n in by_id):
        return {"kind": "render", "pass_node": "", "pass_class": "", "nodes": order}
    for nid in order:
        k = kind_of_class(by_id.get(nid, {}).get("class_type", ""))
        if k:
            return {"kind": k, "pass_node": nid, "pass_class": by_id[nid]["class_type"], "nodes": order}
    return {"kind": "derived", "pass_node": "", "pass_class": "", "nodes": order}


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _clip(v):
    if isinstance(v, str) and len(v) > _MAX_PARAM_STR:
        return v[: _MAX_PARAM_STR - 1] + "…"
    return v


# ── where a loader's value points ────────────────────────────────────────────

_roots_lock = threading.Lock()
_roots_cache: dict | None = None


def set_roots(roots: dict | None) -> None:
    """Pin ComfyUI's folders (``{"input", "output", "temp"}``; tests, hosts that know them).
    ``None`` asks ComfyUI again on next use."""
    global _roots_cache
    with _roots_lock:
        _roots_cache = dict(roots) if roots else None


def roots() -> dict:
    """ComfyUI's ``input`` / ``output`` / ``temp`` folders, as far as they are known."""
    global _roots_cache
    with _roots_lock:
        if _roots_cache is not None:
            return dict(_roots_cache)
    got: dict = {}
    try:
        from agenty_core.tools.comfyui import _dirs_from_comfyui
        d = _dirs_from_comfyui() or {}
        if d.get("input_dir"):
            got["input"] = str(d["input_dir"])
        if d.get("output_dir"):
            got["output"] = str(d["output_dir"])
    except Exception:  # noqa: BLE001
        pass
    if "output" not in got:
        try:
            from agenty_core.utils.comfyui_client import comfyui_output_dir
            out = comfyui_output_dir()
            if out:
                got["output"] = str(out)
        except Exception:  # noqa: BLE001
            pass
    if got.get("output"):
        base = Path(got["output"]).parent
        got.setdefault("input", str(base / "input"))
        got.setdefault("temp", str(base / "temp"))
    with _roots_lock:
        if got:
            _roots_cache = dict(got)
    return got


def resolve_value(value, near: str | Path | None = None, rts: dict | None = None) -> str:
    """The existing file a loader's string value names, or ``""``.

    ``hero.png [output]`` / ``agentY/hero.png`` / ``C:\\shots\\plate.mov`` all work;
    model files and anything that is not a media file are ignored."""
    if not isinstance(value, str):
        return ""
    raw = value.strip().strip('"')
    if not raw or "\n" in raw or len(raw) > 1024:
        return ""
    m = _ANNOT.search(raw)
    where = m.group(1) if m else ""
    name = _ANNOT.sub("", raw)
    low = name.lower()
    if low.endswith(_MODEL_EXT) or Path(low).suffix not in _MEDIA_EXT:
        return ""
    rts = rts if rts is not None else roots()
    cands: list[Path] = []
    p = Path(name.replace("\\", os.sep) if os.sep == "/" else name)
    if p.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", name):
        cands.append(Path(name))
    else:
        rel = name.replace("\\", "/")
        order = [where] if where else ["input", "output", "temp"]
        for key in order:
            if rts.get(key):
                cands.append(Path(rts[key]) / rel)
        if near:
            cands.append(Path(near).parent / rel)
    for c in cands:
        try:
            if c.is_file():
                return str(c.resolve())
        except OSError:
            continue
    return ""


# ── a file's identity ────────────────────────────────────────────────────────

_fp_lock = threading.Lock()
_fp_cache: dict = {}


def fingerprint(path) -> str:
    """``v1:<size>:<sha1 of the first and last MiB>`` of a file, or ``""``.

    Cheap on a 4 GB plate, and the same for a byte-identical copy anywhere."""
    try:
        p = Path(str(path))
        st = p.stat()
    except OSError:
        return ""
    key = (str(p), st.st_size, st.st_mtime_ns)
    with _fp_lock:
        hit = _fp_cache.get(key)
    if hit:
        return hit
    h = hashlib.sha1()
    try:
        with open(p, "rb") as f:
            h.update(f.read(_CHUNK))
            if st.st_size > 2 * _CHUNK:
                f.seek(-_CHUNK, os.SEEK_END)
                h.update(f.read(_CHUNK))
            elif st.st_size > _CHUNK:
                h.update(f.read())
    except OSError:
        return ""
    fp = f"v1:{st.st_size}:{h.hexdigest()[:20]}"
    with _fp_lock:
        if len(_fp_cache) > 4096:
            _fp_cache.clear()
        _fp_cache[key] = fp
    return fp


# ── what one output was made from ────────────────────────────────────────────

def analyze(graph: dict, output_node: str = "", *, out_path: str = "", rts: dict | None = None) -> dict:
    """``{"v", "kind", "pass_class", "params", "inputs": [{"path", "fp", "node", "input"}]}``
    of the output of *output_node* in *graph* (the API graph that made it)."""
    rts = rts if rts is not None else roots()
    cls = classify(graph, output_node)
    by_id = {nid: (node, inputs) for nid, node, inputs in _nodes(graph)}
    own = str(Path(out_path).resolve()) if out_path else ""
    inputs: list[dict] = []
    seen: set = set()
    params: dict = {}
    for nid in cls["nodes"]:
        node, ins = by_id.get(nid, ({}, {}))
        found_here = False
        for name, v in ins.items():
            if _is_link(v):
                continue
            path = resolve_value(v, near=out_path or None, rts=rts)
            if not path or path == own or path in seen:
                continue
            seen.add(path)
            found_here = True
            inputs.append({"path": path, "fp": fingerprint(path), "node": nid, "input": name})
        if found_here:     # a loader: its frame range decides which frames a pass covers
            for name, v in ins.items():
                if not _is_link(v) and _FRAME_INPUT.match(name) and _scalar(v):
                    params[name] = v
    if cls["pass_node"]:
        _node, ins = by_id.get(cls["pass_node"], ({}, {}))
        for name, v in ins.items():
            if not _is_link(v) and _scalar(v) and not re.search(r"(?i)seed", name):
                params.setdefault(name, _clip(v))
    return {"v": VERSION, "kind": cls["kind"], "pass_class": cls["pass_class"],
            "params": params, "inputs": inputs}


def _sidecar_doc(path) -> dict:
    from agenty_core.utils import provenance as pv
    return pv.read_sidecar(path)


def recorded_deps(path) -> dict:
    """``provenance.deps`` of *path*'s sidecar ({} when none)."""
    prov = _sidecar_doc(path).get("provenance")
    deps = prov.get("deps") if isinstance(prov, dict) else None
    return deps if isinstance(deps, dict) else {}


def deps_of(path, *, learn: bool = True) -> dict:
    """The dependency record of *path*: from its sidecar, else worked out from the
    graph stored with it or embedded in it (and, with *learn*, written back so the
    inputs learn about it too). ``{}`` when nothing says how the file was made."""
    have = recorded_deps(path)
    if have:
        return have
    try:
        from agenty_core.utils import provenance as pv
        got = pv.provenance_of(path, use_history=False)
    except Exception:  # noqa: BLE001
        return {}
    if not got.get("found") or not isinstance(got.get("graph"), dict):
        return {}
    deps = analyze(got["graph"], str(got.get("output_node") or ""), out_path=str(path))
    if learn:
        try:
            _store(path, deps, workflow=str(got.get("workflow_file") or ""))
        except Exception:  # noqa: BLE001
            pass
    return deps


def inputs_of(path) -> list[dict]:
    """The files *path* was made from: ``[{"path", "fp", "node", "input", "exists", "changed"}]``.

    ``changed`` is True when the file there now is not the one that was read."""
    out = []
    for i in deps_of(path).get("inputs") or []:
        p = str(i.get("path") or "")
        now = fingerprint(p) if p else ""
        out.append({**i, "exists": bool(now), "changed": bool(now) and bool(i.get("fp")) and now != i.get("fp")})
    return out


# ── the reverse record, beside each input ────────────────────────────────────

_index_lock = threading.Lock()


def derived_path(path) -> Path:
    from agenty_core.utils import provenance as pv
    p = Path(str(path))
    return p.parent / pv.META_DIR / (p.name + DERIVED_SUFFIX)


def _read_index(path) -> list[dict]:
    from agenty_core.utils import provenance as pv
    doc = pv._read_json(derived_path(path))  # noqa: SLF001
    items = doc.get("items") if isinstance(doc, dict) else None
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _add_to_index(src: str, entry: dict) -> None:
    from agenty_core.utils import provenance as pv
    with _index_lock:
        items = [i for i in _read_index(src) if i.get("path") != entry["path"]]
        items.append(entry)
        items = items[-_MAX_DERIVED:]
        pv.write_json(derived_path(src), {"v": VERSION, "items": items})


def index_inputs(path, deps: dict, *, workflow: str = "", created: str = "") -> None:
    """Add the output *path* to the ``.derived.json`` of each of its inputs."""
    out = str(Path(str(path)).resolve())
    entry = {"path": out, "fp": fingerprint(out), "kind": deps.get("kind", ""),
             "params": deps.get("params") or {}, "workflow": os.path.basename(workflow or ""),
             "created": created or time.strftime("%Y-%m-%dT%H:%M:%S")}
    for i in deps.get("inputs") or []:
        try:
            _add_to_index(str(i["path"]), {**entry, "input_fp": i.get("fp", "")})
        except Exception:  # noqa: BLE001 - a read-only source folder is fine
            pass


def _store(path, deps: dict, *, workflow: str = "", created: str = "") -> None:
    """Put *deps* into *path*'s sidecar and *path* into each input's index."""
    from agenty_core.utils import provenance as pv
    with pv.SIDECAR_LOCK:
        sc = pv.sidecar_path(path)
        doc = pv._read_json(sc) if sc.is_file() else None  # noqa: SLF001
        doc = doc if isinstance(doc, dict) else {}
        prov = doc.get("provenance") if isinstance(doc.get("provenance"), dict) else {}
        prov["deps"] = deps
        doc["provenance"] = prov
        pv.write_json(sc, doc)
    index_inputs(path, deps, workflow=workflow, created=created)


def record(path, graph: dict, output_node: str = "", *, workflow_file: str = "", created: str = "") -> dict:
    """Work out and store the dependency record of the output *path* (for writers other
    than :func:`agenty_core.utils.provenance.write`, which does this itself)."""
    try:
        deps = analyze(graph, output_node, out_path=str(path))
        _store(path, deps, workflow=workflow_file, created=created)
        return deps
    except Exception:  # noqa: BLE001
        return {}


def _copies(path) -> list[Path]:
    """*path* and the byte-identical copies agentY stages into ComfyUI's input dir."""
    p = Path(str(path))
    out = [p]
    fp = fingerprint(p)
    inp = roots().get("input")
    if not fp or not inp:
        return out
    for c in (Path(inp) / "agentY" / p.name, Path(inp) / p.name):
        try:
            if c.resolve() != p.resolve() and c.is_file() and fingerprint(c) == fp:
                out.append(c)
        except OSError:
            continue
    return out


def derived_of(path) -> list[dict]:
    """What was made from *path* (or from a staged copy of it), newest last:
    ``[{"path", "kind", "params", "workflow", "created", "exists", "input_fp"}]``."""
    seen: dict = {}
    for c in _copies(path):
        for i in _read_index(c):
            p = str(i.get("path") or "")
            if p:
                seen[p] = {**i, "exists": os.path.isfile(p)}
    return sorted(seen.values(), key=lambda i: str(i.get("created") or ""))


# ── questions ────────────────────────────────────────────────────────────────

def stale_reason(path, _memo: dict | None = None, _depth: int = 0) -> str:
    """Why *path* is out of date, or ``""``: an input changed or is gone since it was
    read, or an input is itself out of date."""
    memo = _memo if _memo is not None else {}
    key = str(path)
    if key in memo:
        return memo[key]
    memo[key] = ""
    why = ""
    if _depth < 12:
        for i in inputs_of(path):
            name = os.path.basename(str(i.get("path") or ""))
            if not i["exists"]:
                why = f"{name} is gone"
            elif i["changed"]:
                why = f"{name} changed since this was made"
            elif stale_reason(i["path"], memo, _depth + 1):
                why = f"{name} is out of date"
            if why:
                break
    memo[key] = why
    return why


def _params_match(have: dict, want: dict) -> bool:
    for k, v in (want or {}).items():
        if k in (have or {}) and str(have[k]).strip().lower() != str(v).strip().lower():
            return False
    return True


def find_existing(source, kind: str, params: dict | None = None) -> list[dict]:
    """Passes of *kind* already made from *source* whose recorded settings agree with
    *params* (keys missing on either side are not held against it), that still exist
    and were made from the file *source* is now. Newest first."""
    now = fingerprint(source)
    hits = []
    for i in derived_of(source):
        if i.get("kind") != kind or not i.get("exists"):
            continue
        if now and i.get("input_fp") and i["input_fp"] != now:
            continue
        if not _params_match(i.get("params") or {}, params or {}):
            continue
        hits.append(i)
    return list(reversed(hits))


def existing_for_graph(graph: dict, rts: dict | None = None) -> list[dict]:
    """For a graph about to run: each output whose pass already exists.
    ``[{"output_node", "kind", "source", "existing": [paths], "params"}]``. Renders
    and plain derivations are never reported (a new seed is a new render)."""
    out = []
    for nid in _output_nodes(graph):
        a = analyze(graph, nid, rts=rts)
        if a["kind"] in ("render", "derived") or not a["inputs"]:
            continue
        for i in a["inputs"]:
            hits = find_existing(i["path"], a["kind"], a["params"])
            if hits:
                out.append({"output_node": nid, "kind": a["kind"], "source": i["path"],
                            "params": a["params"], "existing": [h["path"] for h in hits[:3]],
                            "created": hits[0].get("created", "")})
                break
    return out


def graph(root, up: int = 4, down: int = 4, limit: int = 200) -> dict:
    """The dependency graph around *root*: ``{"root", "nodes", "edges", "truncated"}``.

    ``nodes``: ``{"id" (path), "file", "kind", "created", "exists", "stale", "root"}``;
    ``edges``: ``{"from" (input), "to" (output), "kind", "label"}``, pointing from what
    was read to what was made."""
    root = str(Path(str(root)).resolve()) if root else ""
    nodes: dict = {}
    edges: dict = {}
    memo: dict = {}
    truncated = False

    def add(p: str, **kw) -> bool:
        nonlocal truncated
        if p in nodes:
            nodes[p].update({k: v for k, v in kw.items() if v})
            return True
        if len(nodes) >= limit:
            truncated = True
            return False
        nodes[p] = {"id": p, "file": os.path.basename(p), "kind": kw.get("kind", ""),
                    "created": kw.get("created", ""), "exists": os.path.isfile(p)}
        return True

    if not root:
        return {"root": "", "nodes": [], "edges": [], "truncated": False}
    d = deps_of(root)
    add(root, kind=d.get("kind", ""))
    # Up: what root was made from, and what that was made from.
    frontier = [root]
    for _ in range(max(0, up)):
        nxt = []
        for p in frontier:
            dp = deps_of(p) if p != root else d
            for i in dp.get("inputs") or []:
                src = str(i.get("path") or "")
                if not src or not add(src, kind=deps_of(src).get("kind", "") or "source"):
                    continue
                edges[(src, p)] = {"from": src, "to": p, "kind": dp.get("kind", ""),
                                   "label": dp.get("pass_class") or dp.get("kind", "")}
                nxt.append(src)
        frontier = nxt
    # Down: what was made from root, and from that.
    frontier = [root]
    for _ in range(max(0, down)):
        nxt = []
        for p in frontier:
            for i in derived_of(p):
                q = str(i.get("path") or "")
                if not q or not add(q, kind=i.get("kind", ""), created=i.get("created", "")):
                    continue
                edges[(p, q)] = {"from": p, "to": q, "kind": i.get("kind", ""),
                                 "label": i.get("workflow") or i.get("kind", "")}
                nxt.append(q)
        frontier = nxt
    for p, n in nodes.items():
        n["root"] = p == root
        n["stale"] = stale_reason(p, memo) if n["exists"] else ""
    return {"root": root, "nodes": list(nodes.values()), "edges": list(edges.values()), "truncated": truncated}


def reindex(folder, recursive: bool = False) -> int:
    """Record the dependencies of every file in *folder* that has a sidecar or an
    embedded graph but no record yet (files made before this existed). Returns how
    many were recorded."""
    base = Path(str(folder))
    n = 0
    files = base.rglob("*") if recursive else base.glob("*")
    for p in files:
        try:
            if not p.is_file() or "agent_metadata" in p.parts or p.suffix.lower() not in _MEDIA_EXT:
                continue
            if recorded_deps(p):
                continue
            if deps_of(p, learn=True):
                n += 1
        except Exception:  # noqa: BLE001
            continue
    return n
