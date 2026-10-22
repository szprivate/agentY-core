"""Machine-local templates: jobs that were solved here, kept so the next request is a fast path.

A request that matches a job already solved on this machine should not go through
research, template search, assembly and repair again: the executed graph is
already a working workflow, and only its inputs change. This module is the store.

Where: ``<custom templates dir>/local/`` with its own ``index.json``. The folder is
git-ignored (it is a fact about one machine and one user's work), and the read
side needs nothing new - the catalog and the template loader already walk the
custom folder recursively (``load_descriptions``, ``_find_template_file``), so a
saved job shows up in ``get_workflow_catalog`` and loads with
``get_workflow_template`` like any other template.

What it deliberately does NOT do: it never touches the shared catalog
(``workflow_templates.json`` is retired anyway) and never feeds the recipe
database - the recipe generator skips the ``local`` folder - so a user's jobs
cannot leak into the shared, committed corpus.

Each entry carries beyond the usual name / models / io:

* ``description`` - what it does, in the catalog's shape, with the PARAMETERS
  (which loaders take new files, which text is the prompt, which settings matter);
* ``parameters`` - ``[{node_id, input, kind: image|video|audio|text|setting, label, value}]``;
* ``output_kind`` and ``nodes`` (the model / partner nodes it runs);
* ``request`` - the words it was solved for, and ``uses`` / ``last_used``.

:func:`match_local` finds saved jobs for a new request. It is conservative on
purpose - a wrong match costs more than the research it saves: the output kind
must be the one asked for, a model named in the request must be in the template,
the template must not be rejected by :mod:`agenty_core.utils.template_fit`, and
the wording must overlap.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from agenty_core._compat import tool

LOCAL_DIRNAME = "local"
MAX_LOCAL = 200
# share of the NEW request's words the saved job must cover; an automatically recorded job
# also needs a request of some substance (short vague requests never match it)
MIN_COVER = 0.5
AUTO_MIN_COVER = 0.7
AUTO_MIN_WORDS = 4
_SETTING_NAMES = ("resolution", "duration", "width", "height", "steps", "cfg", "aspect_ratio", "ratio",
                  "model", "generate_audio", "quality", "size", "n", "length", "fps", "denoise",
                  "sampler_name", "scheduler")
_PROMPT_NAMES = ("prompt", "text", "positive", "value")
_STOP = frozenset("""a an the and or of to in on for with from into by at as is are was be it its this that these those
make made create generate give please can you me my your our i we them then than but not no do does did have has had
using use used each every per all any some one two three new old
image images video videos audio picture pictures photo photos clip clips film""".split())


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def local_dir() -> Path:
    from agenty_core.tools import comfyui as C
    return C._custom_templates_dir() / LOCAL_DIRNAME


def local_index_path() -> Path:
    return local_dir() / "index.json"


def _read_entries() -> list[dict]:
    p = local_index_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8") or "[]")
    except (OSError, ValueError):
        return []
    return [t for g in data if isinstance(g, dict) for t in (g.get("templates") or []) if isinstance(t, dict)]


def _write_entries(entries: list[dict]) -> None:
    d = local_dir()
    d.mkdir(parents=True, exist_ok=True)
    group = [{"moduleName": "local", "title": "Local templates (this machine)", "type": "mixed",
              "templates": entries}]
    local_index_path().write_text(json.dumps(group, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def local_names() -> set:
    """Names of the saved jobs on this machine (they are offered by matching, not listed in bulk)."""
    return {str(e.get("name")) for e in _read_entries() if e.get("name")}


def _drop_caches() -> None:
    try:
        from agenty_core.tools import comfyui as C
        C.clear_tool_caches()
        C._index_cache = None
        C._template_cache.clear()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# reading a graph
# ---------------------------------------------------------------------------

def _is_wire(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)


def describe_parameters(workflow: dict) -> list[dict]:
    """The inputs of an API graph a new request would change: files, the prompt, key settings."""
    params: list[dict] = []
    for nid, node in sorted(workflow.items(), key=lambda kv: (len(str(kv[0])), str(kv[0]))):
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type") or "")
        title = str((node.get("_meta") or {}).get("title") or cls)
        ins = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        for key, val in ins.items():
            if _is_wire(val):
                continue
            tail = str(key).rsplit(".", 1)[-1]
            if cls in ("LoadImage", "LoadImageMask") and key == "image":
                params.append({"node_id": str(nid), "input": key, "kind": "image", "label": title, "value": val})
            elif cls in ("LoadVideo", "VHS_LoadVideo") and key in ("file", "video"):
                params.append({"node_id": str(nid), "input": key, "kind": "video", "label": title, "value": val})
            elif cls == "LoadAudio" and key == "audio":
                params.append({"node_id": str(nid), "input": key, "kind": "audio", "label": title, "value": val})
            elif isinstance(val, str) and tail in _PROMPT_NAMES and len(val) > 12 and "Save" not in cls:
                params.append({"node_id": str(nid), "input": key, "kind": "text", "label": title,
                               "value": val[:200]})
            elif tail in _SETTING_NAMES and isinstance(val, (str, int, float, bool)) and "Load" not in cls:
                params.append({"node_id": str(nid), "input": key, "kind": "setting", "label": title, "value": val})
    return params


def signature(workflow: dict) -> str:
    """A hash of the graph's STRUCTURE (classes, model names, wiring) without its values."""
    parts = []
    for nid in sorted(workflow, key=lambda k: (len(str(k)), str(k))):
        node = workflow[nid]
        if not isinstance(node, dict):
            continue
        ins = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        wired = sorted(f"{k}<-{v[0]}" for k, v in ins.items() if _is_wire(v))
        models = sorted(str(v) for k, v in ins.items()
                        if isinstance(v, str) and re.search(r"\.(safetensors|ckpt|gguf|pt|pth|bin)$", v, re.I))
        opts = [str(ins.get("model"))] if isinstance(ins.get("model"), str) else []
        parts.append(f"{nid}:{node.get('class_type')}|{','.join(wired)}|{','.join(models)}|{','.join(opts)}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


_PARTNER_PREFIXES = ("ByteDance", "Kling", "OpenAI", "Gemini", "Ideogram", "Meshy", "Veo", "Runway", "Minimax",
                     "Pixverse", "Flux", "Bfl", "Recraft", "Stability", "Luma", "Tripo", "Rodin", "Sora",
                     "Magnific", "Topaz", "Moonvalley", "Vidu", "Hunyuan3D", "Wan2", "ElevenLabs")


def uses_api(graph: dict) -> bool:
    """Does the graph run a partner (API) node? ComfyUI's own flag when it can be asked, else the names."""
    try:
        from agenty_core.tools import comfyui as C
        info = C._object_info_cache or {}        # only what is already loaded: never a network call here
    except Exception:  # noqa: BLE001
        C, info = None, {}
    for node in graph.values():
        cls = str((node or {}).get("class_type") or "")
        meta = info.get(cls) if isinstance(info, dict) else None
        if isinstance(meta, dict) and C is not None:
            if C._is_api_node(meta):
                return True
        elif cls.startswith(_PARTNER_PREFIXES):
            return True
    return False


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.]+", str(text or "").lower().replace("_", " "))
            if (len(w) >= 3 or any(c.isdigit() for c in w)) and w not in _STOP}


def _auto_description(workflow: dict, name: str, request: str, params: list[dict], kind: str | None) -> str:
    from agenty_core.utils import template_fit
    classes = [str(n.get("class_type")) for n in workflow.values() if isinstance(n, dict)]
    heavy = [c for c in classes if not c.startswith(("Load", "Save", "Preview", "Primitive", "Reroute", "Note"))]
    shown = ", ".join(dict.fromkeys(heavy))[:90] or "workflow"
    ins = sorted({p["kind"] for p in params if p["kind"] in ("image", "video", "audio")})
    inputs = (" + ".join(ins) + " + prompt") if ins else "prompt"
    api = any(c.startswith(("ByteDance", "Kling", "OpenAI", "Gemini", "Ideogram", "Meshy", "Veo", "Runway",
                            "Minimax", "Pixverse", "Flux", "Bfl", "Recraft", "Stability", "Luma", "Tripo",
                            "Rodin", "Sora")) for c in heavy)
    first = f"[{'API' if api else 'Local'}] Saved job ({kind or 'image'} output) via {shown}."
    note = f' Solved for: "{" ".join(str(request).split())[:160]}".' if request else ""
    sets = [f"{p['input']}={p['value']}" for p in params if p["kind"] == "setting"][:5]
    return (f"{first} {inputs} -> {kind or 'image'}."
            + (f" Key settings: {', '.join(sets)}." if sets else "") + note)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def save_local(workflow: dict, name: str = "", description: str = "", request: str = "",
               source: str = "manual", origin: str = "") -> dict:
    """Store an API-format *workflow* as a machine-local template.

    Same structure (classes / models / wiring) already saved: that entry is updated
    (uses, last_used, description if one is given) instead of a second copy added.
    Returns ``{"name", "path", "index_path", "created": bool, "entry"}``.
    """
    from agenty_core.utils import template_fit
    from agenty_core.utils.workflow_parser import parse_workflow
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError("nothing to save: the workflow is empty")
    graph = {k: v for k, v in workflow.items() if isinstance(v, dict) and v.get("class_type")}
    if not graph:
        raise ValueError("not an API-format workflow (no class_type nodes)")

    sig = signature(graph)
    entries = _read_entries()
    now = time.time()
    same = next((e for e in entries if e.get("signature") == sig), None)
    facts = template_fit.template_facts(graph, name)
    kind = facts.get("primary")
    params = describe_parameters(graph)

    if same is not None:
        # The same job again: keep ONE template, refresh what it remembers. The file is
        # replaced by the newest working graph (the inputs it carries are the latest).
        name = same["name"]
        same["uses"] = int(same.get("uses") or 1) + 1
        same["last_used"] = now
        if description.strip():
            same["description"] = description.strip()
            same["description_source"] = "agent"
        if source == "manual":
            same["source"] = "manual"
        if request and len(str(same.get("request") or "")) < 20:
            same["request"] = " ".join(request.split())[:400]
        entry, created = same, False
    else:
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name or "").strip("_")[:60]
        if not stem:
            base = _tokens(request) or {"job"}
            stem = "_".join(sorted(base, key=lambda w: (-len(w), w))[:3])[:40] + "_" + sig[:5]
        stem = f"local_{stem}" if not stem.startswith("local_") else stem
        taken = {e.get("name") for e in entries}
        unique, i = stem, 2
        while unique in taken:
            unique = f"{stem}_{i}"
            i += 1
        name = unique
        parsed = parse_workflow(graph, name=name, update_index=False)["templates"][0]
        desc = description.strip() or _auto_description(graph, name, request, params, kind)
        entry = {**parsed, "description": desc, "description_source": "agent" if description.strip() else "graph",
                 "local": True, "source": source, "signature": sig, "created": now, "last_used": now,
                 "uses": 1, "request": " ".join(str(request).split())[:400], "output_kind": kind,
                 "uses_api": uses_api(graph),
                 "parameters": params, "nodes": facts.get("classes", [])[:24], "origin": origin}
        entries.append(entry)
        created = True
        # least recently used goes first when the folder is full
        while len(entries) > MAX_LOCAL:
            entries.sort(key=lambda e: float(e.get("last_used") or 0))
            gone = entries.pop(0)
            try:
                (local_dir() / f"{gone['name']}.json").unlink()
            except OSError:
                pass

    d = local_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_entries(entries)
    _drop_caches()
    return {"name": name, "path": str(d / f"{name}.json"), "index_path": str(local_index_path()),
            "created": created, "entry": entry}


def forget_local(name: str) -> bool:
    entries = _read_entries()
    keep = [e for e in entries if e.get("name") != name]
    if len(keep) == len(entries):
        return False
    _write_entries(keep)
    try:
        (local_dir() / f"{name}.json").unlink()
    except OSError:
        pass
    _drop_caches()
    return True


def touch_local(name: str) -> None:
    """Note that saved job *name* was used again (keeps it out of the LRU drop)."""
    entries = _read_entries()
    hit = next((e for e in entries if e.get("name") == name), None)
    if hit is None:
        return
    hit["uses"] = int(hit.get("uses") or 1) + 1
    hit["last_used"] = time.time()
    _write_entries(entries)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def match_local(request: str, limit: int = 3, min_cover: float = MIN_COVER, local_only: bool = False) -> list[dict]:
    """Saved jobs that are the same job as *request*, best first.

    ``[{name, description, score, parameters, output_kind, verdict, adjust}]``.
    Conservative: see the module docstring. An empty list is the normal answer."""
    from agenty_core.utils import template_fit
    want = template_fit.request_wants(request)
    rtoks = _tokens(request)
    if not rtoks or not want.get("kind"):
        return []
    out = []
    for e in _read_entries():
        if e.get("output_kind") and e["output_kind"] != want["kind"]:
            continue
        if local_only and e.get("uses_api"):
            continue                  # "local" is a hard constraint: a partner-API job is never offered
        facts = template_fit.facts_for_template(e["name"], str(e.get("description") or ""))
        if facts is None:
            continue
        # a model the request names must be one the saved job runs
        fams = set(facts.get("families") or [])
        if want.get("families") and not set(want["families"]) <= fams:
            continue
        verdict = template_fit.check(want, facts)
        if verdict["verdict"] == "reject":
            continue
        etoks = _tokens(" ".join([str(e.get("request") or ""), str(e.get("description") or ""),
                                  str(e.get("name") or ""), " ".join(e.get("nodes") or [])]))
        cover = len(rtoks & etoks) / max(1, len(rtoks))
        if len(rtoks & etoks) < 2:
            continue                  # one shared word is a coincidence, not a match
        auto = e.get("source") == "auto"
        if cover < (max(min_cover, AUTO_MIN_COVER) if auto else min_cover):
            continue
        if auto and len(rtoks) < AUTO_MIN_WORDS:
            continue
        score = cover + (0.3 if want.get("families") else 0) + min(0.2, 0.02 * int(e.get("uses") or 1))
        out.append({"name": e["name"], "description": e.get("description", ""), "score": round(score, 3),
                    "output_kind": e.get("output_kind"), "parameters": e.get("parameters") or [],
                    "verdict": verdict["verdict"], "adjust": verdict["adjust"]})
    out.sort(key=lambda m: -m["score"])
    return out[:limit]


def catalog_block(request: str, limit: int = 3, local_only: bool = False) -> str:
    """A prompt section naming the saved jobs that match *request* ("" when none do)."""
    found = match_local(request, limit, local_only=local_only)
    if not found:
        return ""
    lines = []
    for m in found:
        files = [f"{p['kind']} '{p['input']}' (node {p['node_id']})" for p in m["parameters"]
                 if p["kind"] in ("image", "video", "audio")][:4]
        text = [f"prompt (node {p['node_id']}, '{p['input']}')" for p in m["parameters"] if p["kind"] == "text"][:2]
        sets = [f"{p['input']}={p['value']}" for p in m["parameters"] if p["kind"] == "setting"][:4]
        lines.append(f"  - {m['name']}: {m['description'][:200]}\n      new inputs: "
                     + "; ".join(files + text) + ("\n      settings: " + ", ".join(sets) if sets else "")
                     + (("\n      adjust: " + "; ".join(m["adjust"])) if m["adjust"] else ""))
    return ("YOUR SAVED JOBS - these LOCAL templates were solved on this machine and match the request. "
            "If the request is the same job with new inputs, PICK ONE (set template.name to its exact name): "
            "it is a known-working graph, so only its inputs change. If it is a different job, ignore them.\n"
            + "\n".join(lines) + "\n\n")


# ---------------------------------------------------------------------------
# the tool surface lives in register_workflow_template(local=True); this is its body
# ---------------------------------------------------------------------------

def register_local_file(workflow_file: str, name: str = "", description: str = "", request: str = "") -> dict:
    """``register_workflow_template(..., local=True)``: save a workflow FILE as a local template."""
    from agenty_core.paths import project_root
    from agenty_core.tools.comfyui import _convert_graph_to_api, _is_graph_format
    p = Path(workflow_file)
    if not p.is_absolute():
        p = Path(project_root()) / workflow_file
    if not p.is_file():
        return {"error": f"Workflow file not found: {workflow_file}"}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        return {"error": f"Invalid workflow JSON: {exc}"}
    converted = False
    if _is_graph_format(data):
        data = _convert_graph_to_api(data)
        converted = True
    if isinstance(data.get("prompt"), dict):
        data = data["prompt"]
    try:
        got = save_local(data, name or p.stem, description, request, source="manual", origin=str(p))
    except ValueError as exc:
        return {"error": str(exc)}
    return {"registered": got["name"], "local": True, "created": got["created"], "copied_to": got["path"],
            "index_path": got["index_path"], "converted_from_graph_format": converted, "entry": got["entry"],
            "message": (f"Saved '{got['name']}' as a LOCAL template (this machine only; the shared catalog and "
                        "recipe database are untouched). It is offered first for matching requests.")}
