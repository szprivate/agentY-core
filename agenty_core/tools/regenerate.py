"""Recreate generated shots with a few changes - everything else exactly as it was.

The stored provenance (``utils/provenance.py``) holds the API graph a file was made
from. These tools list the parameters a user would change, apply ONLY the
requested changes to that graph, run it, save the result as the next version beside
the original, and swap it into the viewer's cut as a take.

Token rule: the graph lives on disk and in memory only. A tool result carries compact
parameter lists, diffs and results - never the graph, never a node dict.

* :func:`recreatable_params` - the editable parameters of one file, compactly.
* :func:`regenerate_from_source` - dry run (diff + cost + next path) or the real run.
* :func:`swap_in_takes` - replace clips in the cut by the new versions (viewer route).

The hosts (agentY, agentY-mcp) install their own runner with :func:`set_runner`, or use
the split API (:func:`plan_regeneration`, :func:`describe_plans`,
:func:`finish_regeneration`) to run each shot through their own pipeline.
"""

from __future__ import annotations

import copy
import difflib
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Callable

from agenty_core._compat import tool
from agenty_core.utils import provenance as _prov

#: Most tokens (len // 4) one shot's answer may take.
SHOT_TOKEN_CAP = 400
_CAP_CHARS = SHOT_TOKEN_CAP * 4
_PROMPT_SHOWN = 120
RUN_TIMEOUT_S = 3600.0

_SEED_RE = re.compile(r"(?i)^(seed|noise_seed|seed_\w+)$")
_SAMPLER_NAMES = ("steps", "cfg", "sampler_name", "scheduler", "denoise", "guidance", "guidance_scale")
_SIZE_NAMES = ("width", "height", "length", "batch_size", "num_frames", "frame_count", "video_frames")
_FPS_NAMES = ("fps", "frame_rate")
_FRAME_NAMES = ("length", "num_frames", "frame_count", "video_frames")
_MODEL_NAMES = ("ckpt_name", "unet_name", "vae_name", "clip_name", "clip_name1", "clip_name2", "clip_name3",
                "clip_name4", "control_net_name", "controlnet_name", "upscale_model_name", "model_name",
                "diffusion_model", "style_model_name", "clip_vision_name")
_TEXT_INPUTS = ("text", "prompt", "text_g", "text_l", "positive_prompt", "caption")
_LOAD_INPUTS = ("image", "video", "file", "audio", "audio_file", "video_file", "path", "filename")
_ALIASES = {
    "positive": ("label", "positive"), "negative": ("label", "negative"), "prompt": ("label", "positive"),
    "sampler": ("name", "sampler_name"), "frames": ("names", _FRAME_NAMES), "length": ("names", _FRAME_NAMES),
    "fps": ("names", _FPS_NAMES), "lora": ("name", "lora_name"),
    "lora_strength": ("names", ("strength_model", "strength")), "guidance": ("names", ("guidance", "guidance_scale")),
}

_VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi", ".gif"}
_AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


# ── small helpers ────────────────────────────────────────────────────────────

def _tokens(text: str) -> int:
    return len(text) // 4


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _is_link(v) -> bool:
    return _prov._is_link(v)


def _scalar(v) -> bool:
    return isinstance(v, (str, int, float, bool))


def _fmt(v, cap: int = 60) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    s = " ".join(str(v).split())
    return s if len(s) <= cap else s[: cap - 1] + "…"


def _guard(path: str) -> str:
    """Why *path* may not be touched, or ''."""
    from agenty_core.utils.path_guard import refuse_read
    return refuse_read(path, "regenerate")


# ── the parameter index ──────────────────────────────────────────────────────

def _node_inputs(graph: dict, nid) -> dict:
    n = graph.get(str(nid))
    return (n.get("inputs") or {}) if isinstance(n, dict) else {}


def _literal_for(graph: dict, nid: str, name: str):
    """(node, input, value) of the literal behind ``graph[nid].inputs[name]``: the input
    itself, or - when it is wired to a Primitive* node - that node's ``value``."""
    v = _node_inputs(graph, nid).get(name)
    if v is None:
        return None
    if not _is_link(v):
        return (nid, name, v) if _scalar(v) else None
    tid = str(v[0])
    t = graph.get(tid)
    if not isinstance(t, dict):
        return None
    ti = t.get("inputs") or {}
    if str(t.get("class_type", "")).startswith("Primitive") or len(ti) == 1:
        for pn in ("value", "string", "text", "int", "float"):
            if pn in ti and not _is_link(ti[pn]) and _scalar(ti[pn]):
                return (tid, pn, ti[pn])
    return None


def _prompt_targets(graph: dict, nid: str, seen: set, depth: int = 0) -> list:
    """The literal text inputs behind a conditioning link: the text-encode node(s) reached
    from *nid* through conditioning pass-through nodes."""
    if nid in seen or depth > 8 or str(nid) not in graph:
        return []
    seen.add(nid)
    inputs = _node_inputs(graph, nid)
    found = []
    for tn in _TEXT_INPUTS:
        if tn in inputs:
            lit = _literal_for(graph, nid, tn)
            if lit and isinstance(lit[2], str):
                found.append(lit)
    if found:
        return found
    for name, v in inputs.items():
        if _is_link(v) and (name.startswith("conditioning") or name in ("positive", "negative", "cond")):
            found += _prompt_targets(graph, str(v[0]), seen, depth + 1)
    return found


def _consumers(graph: dict) -> dict:
    """target node id -> (consumer id, input name) of its first consumer."""
    out: dict = {}
    for nid, node, inputs in _prov._nodes(graph):
        for name, v in inputs.items():
            if _is_link(v):
                out.setdefault(str(v[0]), (nid, name))
    return out


def _entry(graph, nid, name, value, label, group, eff=None, extra_names=()):
    node = graph.get(str(nid)) or {}
    names = [name] + ([eff] if eff and eff != name else []) + [x for x in extra_names if x]
    return {"id": f"{nid}.{name}", "node": str(nid), "input": name, "label": label, "value": value,
            "class": str(node.get("class_type", "")), "group": group, "names": names}


def param_index(graph: dict) -> list[dict]:
    """The parameters of an API graph a user would change, each
    ``{"id": "<node>.<input>", "node", "input", "label", "value", "class"}`` (+ "group", "names").
    Linked inputs are not parameters; a literal behind a Primitive node is listed under the
    primitive's id."""
    entries: list[dict] = []
    taken: set = set()
    cons = _consumers(graph)

    def add(e):
        k = (e["node"], e["input"])
        if k not in taken:
            taken.add(k)
            entries.append(e)

    # 1. prompts, found through the samplers' positive / negative links
    claimed: dict = {}
    for nid, node, inputs in _prov._nodes(graph):
        for role in ("positive", "negative"):
            v = inputs.get(role)
            if _is_link(v):
                for tid, tname, tval in _prompt_targets(graph, str(v[0]), set()):
                    claimed.setdefault((tid, tname), role)
    for (tid, tname), role in claimed.items():
        add(_entry(graph, tid, tname, _node_inputs(graph, tid)[tname], role, "prompt",
                   extra_names=(role, "prompt")))
    # prompts typed straight into a partner / generator node
    roles = {"prompt": "positive", "negative_prompt": "negative", "positive_prompt": "positive"}
    for nid, node, inputs in _prov._nodes(graph):
        for name, v in inputs.items():       # also a dynamic combo's "model.prompt"
            role = roles.get(name.rsplit(".", 1)[-1])
            if role and isinstance(v, str) and (nid, name) not in taken:
                add(_entry(graph, nid, name, v, role, "prompt", extra_names=(role, "prompt")))
    # other text nodes nobody wired to a sampler
    for nid, node, inputs in _prov._nodes(graph):
        if str(node["class_type"]).startswith("CLIPTextEncode") and isinstance(inputs.get("text"), str):
            add(_entry(graph, nid, "text", inputs["text"], "text", "prompt"))

    # 2. name-based parameters (a Primitive's value takes the name of what it feeds)
    for nid, node, inputs in _prov._nodes(graph):
        for name, v in inputs.items():
            if _is_link(v) or not _scalar(v):
                continue
            eff = name.rsplit(".", 1)[-1]
            if name == "value" and str(node["class_type"]).startswith("Primitive") and nid in cons:
                eff = cons[nid][1].rsplit(".", 1)[-1]
            num = isinstance(v, (int, float)) and not isinstance(v, bool)
            if _SEED_RE.match(eff) and isinstance(v, int) and not isinstance(v, bool):
                add(_entry(graph, nid, name, v, eff, "seed", eff))
            elif eff in _SAMPLER_NAMES:
                add(_entry(graph, nid, name, v, eff, "sampler", eff))
            elif eff in _SIZE_NAMES and num:
                add(_entry(graph, nid, name, v, eff, "size", eff))
            elif eff in _FPS_NAMES and num:
                add(_entry(graph, nid, name, v, eff, "fps", eff))
            elif eff in ("strength_model", "strength_clip") or (eff == "strength" and "lora" in
                                                                  str(node["class_type"]).lower()):
                add(_entry(graph, nid, name, v, eff, "lora", eff))
            elif isinstance(v, str) and (name == "lora_name" or (
                    (name in _MODEL_NAMES or v.lower().endswith(_prov._MODEL_EXT)) and not name.startswith("lora"))):
                add(_entry(graph, nid, name, v, name, "lora" if name == "lora_name" else "model", name))
            elif (isinstance(v, str) and str(node["class_type"]).lower().find("load") >= 0
                  and name in _LOAD_INPUTS):
                add(_entry(graph, nid, name, v, name, "input", name))

    # 3. what is left on a partner node
    for nid, node, inputs in _prov._nodes(graph):
        if not _prov.is_partner_class(node["class_type"]):
            continue
        for name, v in inputs.items():
            if _is_link(v) or not _scalar(v) or _prov._SECRET_NAME.search(name):
                continue
            add(_entry(graph, nid, name, v, name, "partner", name))
    return entries


def _match(entries: list, token: str) -> list:
    t = token.strip()
    tl = t.lower()
    by_id = [e for e in entries if e["id"] == t]
    if by_id:
        return by_id
    if tl in _ALIASES:
        kind, arg = _ALIASES[tl]
        if kind == "label":
            return [e for e in entries if e["label"] == arg]
        if kind == "name":
            return [e for e in entries if e["input"] == arg or arg in e["names"]]
        return [e for e in entries if any(n in arg for n in e["names"])]
    if tl in ("model", "checkpoint"):
        got = [e for e in entries if e["input"] in ("ckpt_name", "unet_name", "diffusion_model")]
        return got or [e for e in entries if e["input"] == "model"]
    return [e for e in entries if tl in [n.lower() for n in e["names"]]]


# ── reading the parameters ───────────────────────────────────────────────────

def _prompt_line(e: dict, cap: int = _PROMPT_SHOWN) -> str:
    text = " ".join(str(e["value"]).split())
    shown = text if len(text) <= cap else text[:cap].rstrip() + "…"
    tail = f" [{len(text)} chars]" if len(text) > cap else ""
    lab = f" ({e['label']})" if e["label"] in ("positive", "negative") else ""
    return f'{e["id"]}{lab}: "{shown}"{tail}'


def _kv(e: dict) -> str:
    return f'{e["id"]} {_fmt(e["value"])}'


def _lines(entries: list, header: str, cap: int = _PROMPT_SHOWN) -> list[tuple[int, str]]:
    """(priority, line) pairs; a higher priority is dropped first when the answer is too long."""
    out: list[tuple[int, str]] = [(-1, header)]
    for e in entries:
        if e["group"] == "prompt":
            out.append((0, _prompt_line(e, cap)))
    row = [_kv(e) for e in entries if e["group"] in ("seed", "sampler")]
    if row:
        out.append((1, " · ".join(row)))
    row = [_kv(e) for e in entries if e["group"] in ("size", "fps")]
    if row:
        out.append((1, " · ".join(row)))
    models = [e for e in entries if e["group"] in ("model", "lora") and e["input"] not in ("strength_model", "strength_clip", "strength")]
    row = []
    for e in models:
        s = f'{e["id"]} {_fmt(e["value"], 48)}'
        if e["input"] == "lora_name":
            st = [f'{x["id"]} {_fmt(x["value"])}' for x in entries
                  if x["node"] == e["node"] and x["group"] == "lora" and x is not e]
            if st:
                s += " (" + " · ".join(st) + ")"
        row.append(s)
    if row:
        out.append((3, "models: " + " · ".join(row)))
    by_node: dict = {}
    for e in entries:
        if e["group"] == "partner":
            by_node.setdefault(e["node"], []).append(e)
    for nid, es in by_node.items():
        out.append((2, f'{es[0]["class"]} ({nid}): ' + " · ".join(_kv(e) for e in es)))
    row = [f'{e["id"]} {_fmt(os.path.basename(str(e["value"])), 48)}' for e in entries if e["group"] == "input"]
    if row:
        out.append((4, "inputs: " + " · ".join(row)))
    return out


def _compose(entries: list, header: str) -> str:
    cap = _PROMPT_SHOWN
    while True:
        lines = _lines(entries, header, cap)
        dropped: list[str] = []
        text = "\n".join(l for _, l in lines)
        while _tokens(text) > 320 and len(lines) > 1:
            worst = max(p for p, _ in lines)
            if worst <= 0:
                break
            idx = max(i for i, (p, _) in enumerate(lines) if p == worst)
            ln = lines.pop(idx)[1]
            dropped.append(ln.split(":", 1)[0] if ":" in ln else "params")
            text = "\n".join(l for _, l in lines)
        if dropped:
            text += f"\n+{len(dropped)} more: " + ", ".join(dict.fromkeys(dropped))
        if _tokens(text) <= 320 or cap <= 30:
            return text
        cap = max(30, cap // 2)


def _short_id(pid: str) -> str:
    return (pid[:6] + "…") if pid else ""


@tool
def recreatable_params(path: str, full: str = "") -> str:
    """List the parameters of a generated file that can be changed to recreate it, with
    their current values - compactly (prompt texts shortened, never the graph).

    Reads the provenance stored beside the file, or embedded in it, or in ComfyUI's
    history. Each parameter is addressed as ``<node id>.<input name>`` (for example
    ``6.text``, ``3.seed``); regenerate_from_source takes those ids, or aliases:
    positive, negative, seed, steps, cfg, sampler, scheduler, denoise, guidance, width,
    height, frames, fps, model, lora, lora_strength.

    Args:
        path: The generated image or video.
        full: A parameter id (``6.text``) or alias: return that ONE value in full
            (a long prompt is shortened in the list).

    Returns JSON ``{status, file, source, confidence, params, hint}``; ``params`` is a few
    lines of text, at most about 400 tokens.
    """
    why = _guard(path)
    if why:
        return _j({"status": "error", "error": why})
    name = os.path.basename(str(path))
    got = _prov.provenance_of(path)
    if not got["found"]:
        return _j({"status": "error", "file": name, "source": "none", "error": got["note"]})
    graph = got["graph"]
    entries = param_index(graph)
    if str(full or "").strip():
        hit = _match(entries, str(full))
        if not hit:
            hit = []
            lit = str(full).strip().split(".", 1)
            if len(lit) == 2 and not _is_link(_node_inputs(graph, lit[0]).get(lit[1])) \
                    and lit[1] in _node_inputs(graph, lit[0]):
                hit = [_entry(graph, lit[0], lit[1], _node_inputs(graph, lit[0])[lit[1]], lit[1], "other")]
        if len(hit) != 1:
            ids = ", ".join(e["id"] for e in hit[:8])
            return _j({"status": "error", "file": name,
                       "error": (f"{full!r} matches {len(hit)} parameters ({ids}) - give one id"
                                 if hit else f"no parameter {full!r} in this graph")})
        return _j({"status": "ok", "file": name, "param": hit[0]["id"], "value": hit[0]["value"]})
    head = (f'{name} — source: {got["source"]} ({got["confidence"]})'
            + (f' · prompt {_short_id(got["prompt_id"])}' if got["prompt_id"] else "")
            + f" · {len(graph)} nodes")
    while True:
        text = _compose(entries, head)
        out = {"status": "ok", "file": name, "source": got["source"], "confidence": got["confidence"],
               "params": text, "hint": "full text: recreatable_params(path, full='6.text')"}
        body = _j(out)
        if _tokens(body) <= SHOT_TOKEN_CAP:
            return body
        head = head[: max(40, len(head) - 20)]
        entries = entries[:-1] if len(entries) > 1 else entries
        if len(entries) <= 1:
            return body[:_CAP_CHARS]


# ── versioned naming ─────────────────────────────────────────────────────────

_VDIR = re.compile(r"^v(\d{1,4})$", re.I)
_VTOKEN = re.compile(r"(?<=[_.\-])v(\d{1,4})(?=[_.\-]|$)", re.I)


def next_version_path(original: str) -> tuple[str, int]:
    """The path of the next version of *original* and its number.

    ``.../<kind>/vNNN/<stem>_vNNN_<rest>.<ext>`` (the project layout) -> the same name in the
    next free ``vMMM`` folder; a name with a ``_vNNN`` / ``-vNNN`` / ``.vNNN`` token -> that
    token bumped to the first unused; otherwise ``<stem>_v002.<ext>`` (the original is v001).
    Nothing is created."""
    p = Path(str(original))
    ext, stem = p.suffix, p.stem
    dm = _VDIR.match(p.parent.name)
    if dm:
        n = int(dm.group(1))
        for m in re.finditer(r"_v(\d{1,4})_", stem + "_x", re.I):
            if int(m.group(1)) == n and len(m.group(1)) == len(dm.group(1)):
                pre, rest = stem[: m.start()], (stem + "_x")[m.end():-2]
                if pre and rest:
                    k = n + 1
                    while True:
                        tok = f"v{k:0{len(dm.group(1))}d}"
                        cand = p.parent.parent / tok / f"{pre}_{tok}_{rest}{ext}"
                        if not cand.exists():
                            return str(cand), k
                        k += 1
    toks = list(_VTOKEN.finditer(stem))
    if toks:
        m = toks[-1]
        n, width = int(m.group(1)), len(m.group(1))
        k = n + 1
        while True:
            cand = p.parent / f"{stem[: m.start(1)]}{k:0{width}d}{stem[m.end(1):]}{ext}"
            if not cand.exists():
                return str(cand), k
            k += 1
    base = stem.rstrip("_") or stem
    k = 2
    while True:
        cand = p.parent / f"{base}_v{k:03d}{ext}"
        if not cand.exists():
            return str(cand), k
        k += 1


# ── applying changes ─────────────────────────────────────────────────────────

class _ChangeError(ValueError):
    pass


def _coerce(new, old):
    if isinstance(old, bool):
        if isinstance(new, bool):
            return new
        s = str(new).strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
        raise _ChangeError(f"{new!r} is not a boolean")
    if isinstance(old, int):
        try:
            f = float(new)
        except (TypeError, ValueError):
            raise _ChangeError(f"{new!r} is not a number")
        if f != int(f):
            raise _ChangeError(f"{new!r} is not a whole number")
        return int(f)
    if isinstance(old, float):
        try:
            return float(new)
        except (TypeError, ValueError):
            raise _ChangeError(f"{new!r} is not a number")
    return new if isinstance(new, str) else _fmt(new, 100000)


def _resolve(graph: dict, entries: list, token: str) -> dict:
    t = str(token or "").strip()
    if not t:
        raise _ChangeError("a change without a param")
    hit = _match(entries, t)
    if len(hit) == 1:
        return hit[0]
    if len(hit) > 1:
        raise _ChangeError(f"{t!r} matches {len(hit)} parameters ({', '.join(e['id'] for e in hit[:10])}) - use one id")
    nid, _, name = t.partition(".")
    inputs = _node_inputs(graph, nid)
    if name in inputs:
        v = inputs[name]
        if _is_link(v):
            lit = _literal_for(graph, nid, name)
            hint = f" - set {lit[0]}.{lit[1]} instead" if lit else ""
            raise _ChangeError(f"{t} is wired to another node and cannot be set{hint}")
        if _scalar(v):
            return _entry(graph, nid, name, v, name, "other")
        raise _ChangeError(f"{t} is not a plain value")
    raise _ChangeError(f"no parameter {t!r} in this graph")


def _word_diff(old: str, new: str) -> str:
    split = lambda t: re.findall(r"\s+|\w+|[^\w\s]", t)      # words, punctuation and spaces apart
    a, b = split(old), split(new)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        o, n = "".join(a[i1:i2]).strip(), "".join(b[j1:j2]).strip()
        if not o and not n:
            continue
        o = o if len(o) <= 40 else o[:39] + "…"
        n = n if len(n) <= 40 else n[:39] + "…"
        if o and n:
            parts.append(f"…{o}… → …{n}…")
        elif n:
            parts.append(f"+ …{n}…")
        else:
            parts.append(f"− …{o}…")
    s = " | ".join(parts[:3]) + (f" | +{len(parts) - 3} more" if len(parts) > 3 else "")
    return s or "(whitespace only)"


def _diff_line(e: dict, old, new) -> str:
    lab = f' ({e["label"]})' if e["label"] in ("positive", "negative") else ""
    if isinstance(old, str) and isinstance(new, str) and (len(old) > 40 or len(new) > 40 or " " in old):
        return f'{e["id"]}{lab}: {_word_diff(old, new)}'
    return f'{e["id"]}{lab}: {_fmt(old)} → {_fmt(new)}'


def _apply_changes(graph: dict, changes: list, new_seed: bool) -> tuple[dict, list[str], set]:
    """(new graph, diff lines, ids changed). Raises _ChangeError on a bad change."""
    new = copy.deepcopy(graph)
    entries = param_index(new)
    explicit: dict = {}
    for ch in changes:
        e = _resolve(new, entries, ch.get("param"))
        old = _node_inputs(new, e["node"])[e["input"]]
        if "append" in ch:
            if not isinstance(old, str):
                raise _ChangeError(f"{e['id']} is not a text: cannot append")
            value = (old.rstrip() + " " + str(ch["append"]).strip()) if old.strip() else str(ch["append"])
        elif "replace" in ch:
            pair = ch["replace"]
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2) or not isinstance(old, str):
                raise _ChangeError(f"{e['id']}: replace needs [old, new] on a text")
            if str(pair[0]) not in old:
                raise _ChangeError(f"{e['id']}: {str(pair[0])[:40]!r} is not in the text")
            value = old.replace(str(pair[0]), str(pair[1]))
        elif "value" in ch:
            value = _coerce(ch["value"], old)
        else:
            raise _ChangeError(f"change for {ch.get('param')!r} has no value / append / replace")
        explicit[e["id"]] = (e, value)
    if new_seed:
        seeds = [e for e in entries if e["group"] == "seed"]
        if not seeds:
            raise _ChangeError("new_seed: this graph has no seed input")
        rnd = random.SystemRandom()
        for e in seeds:
            if e["id"] not in explicit:
                explicit[e["id"]] = (e, rnd.randrange(1, 2 ** 31 - 1))
    diff, touched = [], set()
    for pid, (e, value) in explicit.items():
        old = _node_inputs(new, e["node"])[e["input"]]
        if value == old and type(value) is type(old):
            continue
        new[e["node"]]["inputs"][e["input"]] = value
        diff.append(_diff_line(e, old, value))
        touched.add((e["node"], e["input"]))
    return new, diff, touched


def _applies(ch: dict, path: str) -> bool:
    want = ch.get("path")
    if not want:
        return True
    w = str(want).replace("\\", "/").lower()
    full = str(path).replace("\\", "/").lower()
    base = os.path.basename(full)
    return w == full or w == base or w == os.path.splitext(base)[0] or full.endswith("/" + w)


def _as_list(x, what: str):
    if x is None or x == "":
        return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") or s.startswith("{"):
            try:
                x = json.loads(s)
            except ValueError:
                pass
    if isinstance(x, (dict,)) and what == "changes":
        return [x]
    if isinstance(x, (str, os.PathLike)):
        return [str(x)]
    return list(x)


# ── cost ─────────────────────────────────────────────────────────────────────

def _cost_text(graph: dict) -> str:
    try:
        from agenty_core import partner_credits as pc
        est = pc.estimate_workflow(graph)
    except Exception:  # noqa: BLE001
        return "cost unknown"
    if not est.get("paid"):
        return "local: no credits"
    lo, hi = est.get("min_usd"), est.get("max_usd")
    if est.get("price_unknown") or hi is None:
        return f"partner: from ${lo:g}, price partly unknown"
    return f"partner: ${lo:g}" if lo == hi else f"partner: ${lo:g}-${hi:g}"


# ── planning ─────────────────────────────────────────────────────────────────

def _workflow_dir() -> Path:
    try:
        from agenty_core.tools.comfyui import _workflows_dir
        d = Path(_workflows_dir()) / "regenerate"
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:  # noqa: BLE001
        d = Path(tempfile.gettempdir()) / "agenty_regenerate"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _count_inputs(graph: dict) -> int:
    return sum(len(i) for _, _, i in _prov._nodes(graph))


def plan_regeneration(paths, changes=None, new_seed: bool = False, *, workflow_dir: str | None = None) -> list[dict]:
    """One plan per shot. Each holds the new graph INTERNALLY under ``_graph`` (and the
    original's provenance under ``_orig``), plus ``workflow_path`` (written to disk -
    in *workflow_dir* when given), ``target``, ``version``, ``diff``, ``file``, ``shot``,
    ``ok`` and ``error``. Nothing is run."""
    files = list(dict.fromkeys(_as_list(paths, "paths")))
    chs = [c for c in _as_list(changes, "changes") if isinstance(c, dict)]
    plans: list[dict] = []
    for ch in chs:
        if ch.get("path") and not any(_applies(ch, f) for f in files):
            return [{"shot": "", "file": "", "ok": False, "status": "refused",
                     "error": f"a change names the shot {ch['path']!r}, which is not among the paths"}]
    for f in files:
        plan = {"shot": Path(f).stem, "file": str(f), "ok": False, "status": "refused", "diff": []}
        plans.append(plan)
        why = _guard(f)
        if why:
            plan["error"] = why
            continue
        if not Path(f).is_file():
            plan["error"] = "file not found"
            continue
        got = _prov.provenance_of(f)
        plan.update(source=got["source"], confidence=got["confidence"])
        if not got["found"]:
            plan["error"] = got["note"]
            continue
        if any(v == _prov.REDACTED for _, _, i in _prov._nodes(got["graph"]) for v in i.values()):
            plan["error"] = "can't recreate exactly: the stored graph holds a redacted secret (an API key input)"
            continue
        mine = [c for c in chs if _applies(c, f)]
        if not mine and not new_seed:
            plan["error"] = "nothing to change - ComfyUI would return the cached result"
            continue
        try:
            graph, diff, touched = _apply_changes(got["graph"], mine, new_seed)
        except _ChangeError as e:
            plan["error"] = str(e)
            continue
        if not diff:
            plan["error"] = ("the requested values equal the original ones - nothing to change, "
                             "ComfyUI would return the cached result")
            continue
        target, version = next_version_path(f)
        nodes_touched = {n for n, _ in touched}
        plan.update(
            ok=True, status="planned", diff=diff, target=target, version=version,
            unchanged=(f"everything else unchanged ({len(graph) - len(nodes_touched)} nodes, "
                       f"{_count_inputs(graph) - len(touched)} inputs)"),
            cost=_cost_text(graph), prompt_id=got["prompt_id"],
            output_node=got["output_node"], output_index=got["output_index"],
            _graph=graph, _orig=got)
        wd = Path(workflow_dir) if workflow_dir else _workflow_dir()
        wd.mkdir(parents=True, exist_ok=True)
        wf = wd / f"regen_{Path(f).stem}_v{version:03d}_{int(time.time())}.json"
        wf.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
        plan["workflow_path"] = str(wf)
    return plans


def _fit_shot(d: dict) -> dict:
    """Trim a shot's answer to the per-shot token cap (diff lines first)."""
    diff = list(d.get("diff") or [])
    n = len(diff)
    out = d
    while n > 0 and _tokens(_j(out)) > SHOT_TOKEN_CAP:
        n -= 1
        out = {**d, "diff": diff[:n] + [f"+{len(diff) - n} more changes"]}
    if _tokens(_j(out)) > SHOT_TOKEN_CAP and out.get("error"):
        out = {**out, "error": str(out["error"])[:600]}
    return out


def describe_plans(plans: list[dict], dry_run: bool = True) -> dict:
    """The compact answer for plans: per shot the diff, cost and next path (no graph,
    no key starting with ``_``)."""
    shots = []
    for pl in plans:
        d: dict = {"shot": pl.get("shot", ""), "file": pl.get("file", "")}
        if not pl.get("ok"):
            d.update(status="refused", error=str(pl.get("error", "")))
            if pl.get("source"):
                d["source"] = pl["source"]
        else:
            d.update(source=pl.get("source", ""), confidence=pl.get("confidence", ""),
                     diff=pl["diff"][:20], unchanged=pl.get("unchanged", ""), cost=pl.get("cost", ""),
                     next=pl.get("target", ""))
        shots.append(_fit_shot(d))
    out = {"status": "ok" if any(p.get("ok") for p in plans) else "refused", "dry_run": bool(dry_run),
           "shots": shots}
    if dry_run and out["status"] == "ok":
        out["hint"] = "nothing was run; call again with dry_run=False to recreate these shots"
    return out


# ── running ──────────────────────────────────────────────────────────────────

_runner: Callable | None = None


def set_runner(fn: Callable | None) -> None:
    """Install the function that runs a workflow file: ``fn(workflow_path, *, confirm_cost=False)
    -> {"status", "outputs": [abs paths], "prompt_id", "error"?, "error_kind"?}`` (blocking)."""
    global _runner
    _runner = fn


def _default_runner(workflow_path: str, *, confirm_cost: bool = False) -> dict:
    """Submit through comfyui_exec and poll ``/history/<id>`` until done."""
    from agenty_core import comfyui_exec as ex
    from agenty_core import partner_credits as pc
    from agenty_core.utils import canvas_scope as cs
    from agenty_core.utils.comfyui_client import get_client
    try:
        with pc.confirmed(confirm_cost):
            pid = ex._submit_workflow(workflow_path, cs.client_id())
    except pc.CostGateError as e:
        return {"status": "error", "error_kind": e.error_kind, "error": str(e)[:400], "outputs": [], "prompt_id": ""}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": f"could not queue: {str(e)[:300]}", "outputs": [], "prompt_id": ""}
    deadline = time.time() + RUN_TIMEOUT_S
    entry = None
    while time.time() < deadline:
        try:
            doc = get_client().get(f"/history/{pid}")
        except Exception:  # noqa: BLE001
            doc = None
        e = doc.get(pid) if isinstance(doc, dict) else None
        if isinstance(e, dict) and (e.get("outputs") or (e.get("status") or {}).get("completed") is not None):
            st = e.get("status") or {}
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                err = (msgs[-1][1].get("exception_message") if msgs else "") or "execution failed"
                return {"status": "error", "error": str(err)[:300], "outputs": [], "prompt_id": pid}
            entry = e
            break
        time.sleep(1.5)
    if entry is None:
        return {"status": "error", "error": "timed out waiting for ComfyUI", "outputs": [], "prompt_id": pid}
    outs, by_node = [], {}
    for item in ex._extract_output_files({pid: entry}):
        if item.get("type") == "temp":
            continue
        try:
            fp = str(ex._resolve_output_path(item["filename"], item.get("subfolder", ""), item.get("type", "output")))
        except Exception:  # noqa: BLE001
            continue
        outs.append(fp)
        by_node.setdefault(item["node_id"], []).append(fp)
    return {"status": "done" if outs else "error", "outputs": outs, "prompt_id": pid, "outputs_by_node": by_node,
            **({} if outs else {"error": "the run produced no output files"})}


def _kind(path: str) -> str:
    e = Path(path).suffix.lower()
    return "video" if e in _VIDEO_EXT else "audio" if e in _AUDIO_EXT else "image"


def _pick_output(plan: dict, outputs: list[str], prompt_id: str, by_node: dict | None):
    """The produced file from the original's output node and index (else the first of the
    same media kind, else the first)."""
    node, idx = str(plan.get("output_node") or ""), int(plan.get("output_index") or 0)
    if not by_node and prompt_id and node:
        try:
            from agenty_core.utils.comfyui_client import get_client
            doc = get_client().get(f"/history/{prompt_id}")
            entry = doc.get(prompt_id) if isinstance(doc, dict) else None
            if isinstance(entry, dict):
                by_node = {}
                for o in outputs:
                    n, _ = _prov.output_position(entry, o)
                    if n:
                        by_node.setdefault(n, []).append(o)
        except Exception:  # noqa: BLE001
            by_node = None
    if by_node and node in by_node:
        lst = by_node[node]
        return lst[idx] if idx < len(lst) else lst[0], node, idx if idx < len(lst) else 0
    kind = _kind(plan.get("file", ""))
    same = [o for o in outputs if _kind(o) == kind] or list(outputs)
    return (same[0] if same else ""), node, idx


def finish_regeneration(plan: dict, outputs: list[str], prompt_id: str = "", *,
                        outputs_by_node: dict | None = None) -> dict:
    """Pick the produced file, move it to the plan's versioned path (with its sidecars),
    write its provenance and return the per-shot result."""
    res = {"shot": plan.get("shot", ""), "file": plan.get("file", ""), "new_file": "",
           "version": plan.get("version"), "prompt_id": prompt_id or "", "status": "error"}
    try:
        outs = [str(o) for o in (outputs or []) if o and Path(str(o)).is_file()]
        if not outs:
            res["error"] = "the run produced no output file"
            return res
        chosen, node, idx = _pick_output(plan, outs, prompt_id, outputs_by_node)
        target = plan["target"]
        if Path(target).exists():
            res["error"] = f"{target} already exists"
            return res
        _prov.move_with_sidecars(chosen, target)
        _prov.write(target, plan["_graph"], prompt_id=prompt_id, output_node=node, output_index=idx,
                    workflow_file=plan.get("workflow_path", ""), source="regenerate",
                    derived_from={"file": plan.get("file", ""), "prompt_id": plan.get("prompt_id", ""),
                                  "changes": list(plan.get("diff") or [])})
        res.update(new_file=target, status="done")
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return res


def _validation_error(workflow_path: str) -> str:
    try:
        from agenty_core.tools import comfyui as cf
        doc = json.loads(cf.validate_workflow(workflow_path))
    except Exception as e:  # noqa: BLE001
        return f"could not validate: {str(e)[:160]}"
    if doc.get("valid"):
        return ""
    errs = list(doc.get("local_errors") or []) + [str(v)[:120] for v in (doc.get("server_errors") or {}).values()][:3]
    return "invalid workflow: " + "; ".join(str(e)[:140] for e in errs[:3])


def _run_one(plan: dict, confirm_cost: bool) -> dict:
    res = {"shot": plan["shot"], "file": plan["file"], "new_file": "", "version": plan.get("version"),
           "prompt_id": "", "status": "error"}
    bad = _validation_error(plan["workflow_path"])
    if bad:
        res["error"] = bad
        return res
    runner = _runner or _default_runner
    try:
        out = runner(plan["workflow_path"], confirm_cost=confirm_cost) or {}
    except Exception as e:  # noqa: BLE001
        res["error"] = f"run failed: {str(e)[:200]}"
        return res
    res["prompt_id"] = str(out.get("prompt_id") or "")
    if out.get("error_kind") == "confirm_cost":
        res.update(error_kind="confirm_cost", error=str(out.get("error", "needs the user's go-ahead"))[:300])
        return res
    if out.get("error") or not out.get("outputs"):
        res["error"] = str(out.get("error") or "the run produced no output")[:300]
        return res
    fin = finish_regeneration(plan, out["outputs"], res["prompt_id"], outputs_by_node=out.get("outputs_by_node"))
    res.update({k: v for k, v in fin.items() if v not in (None, "")})
    return res


@tool
def regenerate_from_source(paths: list, changes: list | None = None, new_seed: bool = False,
                           dry_run: bool = True, confirm_cost: bool = False) -> str:
    """Recreate generated shots with specific changes; every other parameter stays EXACTLY
    as in the original run (the stored graph is re-run with only your changes applied).

    Call recreatable_params(path) first to see the parameters. Start with dry_run=True: it
    shows, per shot, what changes, what it costs and where the new version will land,
    and runs nothing. Then call again with dry_run=False. Shots run one after another.

    Args:
        paths: The generated file(s), one or several shots.
        changes: A list of ``{"param": "<node>.<input>" or an alias, "value": v}``. For a
            prompt text use ``{"param": "positive", "append": "text"}`` or
            ``{"param": "positive", "replace": ["old words", "new words"]}`` so the long
            text is never repeated. Aliases: positive, negative, seed, steps, cfg, sampler,
            scheduler, denoise, guidance, width, height, frames, fps, model, lora,
            lora_strength. Add ``"path": "shot_03"`` to restrict a change to one shot.
        new_seed: Draw a new random seed for every seed input (otherwise the seed is kept).
        dry_run: True (default) - only describe; False - run and save the new version.
        confirm_cost: True once the user agreed to the cost of paid partner nodes.

    A shot with no change is refused (ComfyUI would return the cached result). The result
    never contains the graph. A finished run lists the new files; swap them into the cut
    with swap_in_takes(takes=[...]).
    """
    try:
        if dry_run:
            with tempfile.TemporaryDirectory(prefix="agenty_regen_") as td:
                plans = plan_regeneration(paths, changes, new_seed, workflow_dir=td)
                return _j(describe_plans(plans, True))
        plans = plan_regeneration(paths, changes, new_seed)
        results = []
        for pl in plans:
            if not pl.get("ok"):
                results.append({"shot": pl.get("shot", ""), "file": pl.get("file", ""), "new_file": "",
                                "version": None, "prompt_id": "", "status": "refused",
                                "error": str(pl.get("error", ""))[:300]})
            else:
                results.append(_run_one(pl, bool(confirm_cost)))
        done = [r for r in results if r["status"] == "done"]
        out = {"status": "ok" if len(done) == len(results) and results else ("partial" if done else "error"),
               "dry_run": False, "results": [_fit_shot(r) for r in results]}
        if done:
            out["takes"] = [{"from_file": r["file"], "file": r["new_file"], "version": r["version"]} for r in done]
            out["next_step"] = ("swap_in_takes(takes=<takes>, otio=<the cut's .otio path>) puts the new "
                                "versions into the cut")
        if any(r.get("error_kind") == "confirm_cost" for r in results):
            out["hint"] = "paid nodes: ask the user, then repeat with confirm_cost=True"
        return _j(out)
    except Exception as e:  # noqa: BLE001
        return _j({"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"})


# ── the viewer ───────────────────────────────────────────────────────────────

@tool
def swap_in_takes(takes: list, otio: str = "", retime: bool = False) -> str:
    """Put new versions of shots into the bEpic viewer's cut as takes (the original stays
    one click away). Needs ComfyUI open with the viewer.

    Args:
        takes: ``[{"from_file": original, "file": new version, "version": int, "note": short text}]``
            (regenerate_from_source returns them as ``takes``).
        otio: The cut's .otio path (the viewer's Edit state shows it); "" for the open cut.
        retime: Fit the new clip to the old clip's length on the timeline.

    Returns JSON ``{status, saved, applied, pushed, notes}``.
    """
    items = _as_list(takes, "takes")
    clean = []
    for t in items:
        if not isinstance(t, dict) or not t.get("file") or not t.get("from_file"):
            return _j({"status": "error", "error": "each take needs from_file and file"})
        for k in ("file", "from_file"):
            why = _guard(str(t[k]))
            if why:
                return _j({"status": "error", "error": why})
        if not Path(str(t["file"])).is_file():
            return _j({"status": "error", "error": f"{t['file']} does not exist"})
        c = {"from_file": str(t["from_file"]), "file": str(t["file"])}
        if t.get("version") is not None:
            c["version"] = t["version"]
        if t.get("note"):
            c["note"] = str(t["note"])[:120]
        clean.append(c)
    if not clean:
        return _j({"status": "error", "error": "no takes"})
    if otio:
        why = _guard(str(otio))
        if why:
            return _j({"status": "error", "error": why})
    body = {"otio": str(otio or ""), "takes": clean, "retime": bool(retime)}
    from urllib.parse import urlencode
    from agenty_core.utils import canvas_scope as _cs
    from agenty_core.utils.comfyui_client import get_client
    cid = _cs.client_id()
    route = "/bepic/take" + (("?" + urlencode({"client_id": cid})) if cid else "")
    try:
        r = get_client().post(route, json_data=body)
    except Exception as e:  # noqa: BLE001
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status == 404 or "404" in str(e):
            return _j({"status": "error", "error": (
                "the viewer in this ComfyUI predates the take route - update ComfyUI-ImageViewer "
                "under <ComfyUI>/custom_nodes/ and restart ComfyUI.")})
        return _j({"status": "error", "error": f"could not reach ComfyUI ({str(e)[:160]}). Is it running?"})
    if not isinstance(r, dict):
        return _j({"status": "error", "error": f"unexpected response: {str(r)[:120]}"})
    out = {"status": "ok" if r.get("ok") else "error", "saved": bool(r.get("saved")),
           "applied": int(r.get("applied") or 0), "pushed": bool(r.get("pushed"))}
    if r.get("notes"):
        out["notes"] = [str(n)[:160] for n in list(r["notes"])[:5]]
    if r.get("error"):
        out["error"] = str(r["error"])[:300]
    return _j(out)
