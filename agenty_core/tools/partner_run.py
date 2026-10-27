"""``partner_run`` - build (and run) one partner/API node directly from its schema.

A request that names a partner model ("Seedance 2.5 first-frame", "GPT Image",
"Nano Banana", "Kling start-end", "Meshy") needs no template: the node IS the
workflow - loaders for its files, the node, a save node. Searching a template
catalog, assembling, validating and fixing one costs many LLM steps and picks the
wrong template now and then (a Seedance 2.0 storyboard board for a Seedance 2.5
first-frame job); this builds the graph from the node's own ``/object_info`` entry
in one call.

What it does, in order:

1. find the node class (exact class name, else display name / partial name among
   the partner nodes) - an unknown name returns candidates, never a guess;
2. resolve every ``settings`` name against the inputs IN FORCE for the chosen
   options (a dynamic combo brings dotted inputs, ``model.resolution``; an autogrow
   group brings slots, ``model.reference_images.image_1`` - the helpers in
   :mod:`agenty_core.tools.assembly_deterministic` expand them). A bare name that
   is unique by its last segment (``resolution``) is taken for the dotted one;
   values are checked against the schema (combo options, numeric range) and a bad
   one is an error that lists the options - not silently snapped;
3. give every file input a loader (LoadImage / LoadVideo / LoadAudio, file staged
   into ComfyUI's input folder) wired to the slot; a list fills an autogrow group
   in order;
4. add the save node for the output kind (agentY's host swaps in the user's chosen
   saver before it submits);
5. fill defaults, validate locally, write the workflow file;
6. run it through the host's execution path (:func:`set_run_hook`) - agentY and
   agentY-mcp register theirs so API key, output tagging and the viewer / canvas
   all happen as for any other workflow; with no host hook the graph is submitted
   straight to ComfyUI and the outputs are waited for.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path
import logging
import os
import random
import re
from typing import Callable, Optional

from agenty_core._compat import tool

logger = logging.getLogger("agentY.partner_run")

# Wire type of a file input -> (loader class, its file widget, output index).
_LOADERS = {
    "IMAGE": ("LoadImage", "image", 0),
    "VIDEO": ("LoadVideo", "file", 0),
    "AUDIO": ("LoadAudio", "audio", 0),
}
# Output kind -> (saver class, the wire types it takes).
_SAVERS = {
    "image": ("SaveImage", ("IMAGE",)),
    "video": ("SaveVideo", ("VIDEO",)),
    "audio": ("SaveAudio", ("AUDIO",)),
}
_KIND_OF_TYPE = {"IMAGE": "image", "VIDEO": "video", "AUDIO": "audio"}
_ALIASES = {
    "aspect_ratio": ("ratio", "aspect_ratio"),
    "ratio": ("ratio", "aspect_ratio"),
    "length": ("duration",),
    "seconds": ("duration",),
    "audio": ("generate_audio", "audio"),
    "with_audio": ("generate_audio",),
    "sound": ("generate_audio",),
    "text": ("prompt",),
    "res": ("resolution",),
    "size": ("size", "resolution", "size_preset"),
}
_WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "BOOL", "COMBO", "COMFY_DYNAMICCOMBO_V3"}

# The host's way to run a saved workflow: ``fn(workflow_path) -> JSON string / dict``.
_run_hook: Optional[Callable[[str], object]] = None


def set_run_hook(fn: Optional[Callable[[str], object]]) -> None:
    """Register the host's own execution path for ``partner_run(run=True)``."""
    global _run_hook
    _run_hook = fn


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------

def _is_partner(info: dict | None) -> bool:
    info = info or {}
    return bool(info.get("api_node")) or str(info.get("python_module") or "").startswith("comfy_api_nodes")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def find_partner_nodes(query: str, object_info: dict, limit: int = 8) -> list[dict]:
    """Partner nodes whose class name, display name or category match *query*,
    best first: ``[{class_type, display_name, category}]``."""
    q = _norm(query)
    toks = [t for t in re.findall(r"[a-z0-9.]+", str(query or "").lower()) if t]
    scored = []
    for cls, info in (object_info or {}).items():
        if not isinstance(info, dict) or not _is_partner(info):
            continue
        hay = _norm(cls + " " + str(info.get("display_name") or "") + " " + str(info.get("category") or ""))
        if _norm(cls) == q or _norm(info.get("display_name")) == q:
            score = 100
        elif q and q in hay:
            score = 50 + len(q)
        else:
            score = sum(10 for t in toks if _norm(t) and _norm(t) in hay)
            if len(toks) and score < 10 * len(toks):
                score = 0 if score < 10 * max(1, len(toks) - 1) else score
        if score:
            scored.append((score, cls, info))
    scored.sort(key=lambda s: (-s[0], len(s[1])))
    return [{"class_type": c, "display_name": i.get("display_name"), "category": i.get("category")}
            for _s, c, i in scored[:limit]]


def resolve_node_type(node_type: str, object_info: dict) -> tuple[str | None, list]:
    """``(class_type, candidates)``: the class when *node_type* names exactly one
    node (a class name, a display name, or a unique best partial match)."""
    if node_type in (object_info or {}):
        return node_type, []
    q = _norm(node_type)
    exact = [c for c, i in object_info.items()
             if isinstance(i, dict) and (_norm(c) == q or _norm(i.get("display_name")) == q)]
    if len(exact) == 1:
        return exact[0], []
    cands = find_partner_nodes(node_type, object_info)
    if len(cands) == 1:
        return cands[0]["class_type"], cands
    return None, cands


def _spec_type(spec) -> str:
    if isinstance(spec, (list, tuple)) and spec:
        first = spec[0]
        return "COMBO" if isinstance(first, (list, tuple)) else str(first)
    return str(spec)


def _spec_meta(spec) -> dict:
    return spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}


def _is_wire_spec(spec) -> bool:
    return _spec_type(spec) not in _WIDGET_TYPES and "AUTOGROW" not in _spec_type(spec).upper()


def _combo_choices(spec) -> list:
    from agenty_core.tools.assembly_deterministic import _dynamic_combo_option_keys, combo_options
    keys = _dynamic_combo_option_keys(spec)
    if keys:
        return keys
    return [str(o) for o in (combo_options(spec) or [])]


def _match_choice(value, choices: list):
    """*value* as one of *choices* (exact, then case/punctuation-insensitive), or None."""
    if str(value) in [str(c) for c in choices]:
        return next(c for c in choices if str(c) == str(value))
    n = _norm(value)
    hits = [c for c in choices if _norm(c) == n]
    return hits[0] if len(hits) == 1 else None


def _resolve_name(name: str, expanded: dict) -> tuple[str | None, list]:
    """The in-force input a caller's *name* means: ``(key, candidates)``."""
    if name in expanded:
        return name, []
    tail = {k: k.rsplit(".", 1)[-1] for k in expanded}
    hits = [k for k, t in tail.items() if t == name]
    if len(hits) == 1:
        return hits[0], []
    if len(hits) > 1:
        # a name used by a top-level input and an option's: the top-level one is the node's own
        top = [k for k in hits if "." not in k]
        if len(top) == 1:
            return top[0], []
        return None, hits
    for alias in _ALIASES.get(_norm(name), ()):
        hits = [k for k, t in tail.items() if t == alias]
        if len(hits) == 1:
            return hits[0], []
    close = difflib.get_close_matches(name, list(tail.values()), n=3, cutoff=0.6)
    return None, [k for k, t in tail.items() if t in close]


def _as_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "on", "y"):
        return True
    if s in ("false", "0", "no", "off", "n", "none", ""):
        return False
    raise ValueError(f"not a yes/no value: {value!r}")


def _coerce(key: str, spec, value, warnings: list):
    """*value* as the type *spec* wants, or ``ValueError`` saying what is wrong."""
    typ = _spec_type(spec)
    meta = _spec_meta(spec)
    if typ == "COMFY_DYNAMICCOMBO_V3" or typ == "COMBO":
        choices = _combo_choices(spec)
        got = _match_choice(value, choices)
        if got is None:
            shown = ", ".join(map(str, choices[:14])) + (" ..." if len(choices) > 14 else "")
            raise ValueError(f"'{key}'={value!r} is not an option. Options: {shown}")
        return got
    if typ in ("BOOLEAN", "BOOL"):
        return _as_bool(value)
    if typ in ("INT", "FLOAT"):
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{key}'={value!r} is not a number") from None
        lo, hi = meta.get("min"), meta.get("max")
        clamped = num
        if isinstance(lo, (int, float)) and clamped < lo:
            clamped = lo
        if isinstance(hi, (int, float)) and clamped > hi:
            clamped = hi
        if clamped != num:
            warnings.append(f"'{key}'={value!r} is outside {lo}..{hi}; used {clamped:g}")
        return int(round(clamped)) if typ == "INT" else float(clamped)
    if typ == "STRING":
        return str(value)
    raise ValueError(f"'{key}' is a {typ} input and cannot be set as a plain value")


def _slots_of_group(group_key: str, expanded: dict) -> list[str]:
    prefix = group_key + "."
    return [k for k in expanded if k.startswith(prefix) and "." not in k[len(prefix):]]


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------

def _loader_for(wire_type: str, object_info: dict):
    """``(class, widget, output_index)`` of the loader for *wire_type* on this ComfyUI."""
    cls, widget, idx = _LOADERS[wire_type]
    info = object_info.get(cls)
    if not isinstance(info, dict):
        raise ValueError(f"this ComfyUI has no {cls} node, so a {wire_type} file cannot be loaded.")
    required = ((info.get("input") or {}).get("required") or {})
    if widget not in required:
        widget = next((n for n, s in required.items() if _spec_type(s) == "COMBO"), widget)
    return cls, widget, idx


def _stage(path: str) -> str:
    """Put *path* into ComfyUI's input folder; return the name a loader uses."""
    from agenty_core.tools import image_io
    got = image_io.stage_image(path)
    if not isinstance(got, dict) or got.get("error") or not got.get("name"):
        raise ValueError(f"could not stage {path}: {(got or {}).get('error', 'no answer')}")
    sub = str(got.get("subfolder") or "").strip("/")
    return f"{sub}/{got['name']}" if sub else str(got["name"])


def _output_choice(info: dict, save: str) -> tuple[str | None, int, str, list]:
    """``(kind, output_index, wire_type, problems)`` for what the node makes."""
    outs = list(info.get("output") or [])
    if info.get("output_node"):
        return None, 0, "", []
    want = str(save or "auto").strip().lower()
    want = {"images": "image", "videos": "video", "gif": "video", "sound": "audio", "music": "audio"}.get(want, want)
    for i, t in enumerate(outs):
        kind = _KIND_OF_TYPE.get(str(t))
        if kind and (want in ("auto", "", kind)):
            return kind, i, str(t), []
    kinds = sorted({_KIND_OF_TYPE[t] for t in map(str, outs) if t in _KIND_OF_TYPE})
    if want not in ("auto", ""):
        return None, 0, "", [f"save='{save}' but this node's outputs are {outs or 'none'}"
                            + (f" (it can save: {', '.join(kinds)})" if kinds else "")]
    return None, 0, "", [f"this node's outputs {outs} have no saver partner_run knows (image, video, audio)"]


def _add_saver(workflow: dict, kind: str, src_id: str, out_idx: int, wire_type: str,
               stem: str, object_info: dict, next_id) -> str:
    from agenty_core.tools.assembly_deterministic import default_dynamic_combos
    from agenty_core.tools.comfyui import _agent_output_prefix
    cls, types = _SAVERS[kind]
    info = object_info.get(cls)
    if not isinstance(info, dict):
        raise ValueError(f"this ComfyUI has no {cls} node to save the {kind}.")
    required = ((info.get("input") or {}).get("required") or {})
    wire_in = next((n for n, s in required.items() if _spec_type(s) in types), None)
    if wire_in is None:
        raise ValueError(f"{cls} has no {types[0]} input here.")
    prefix_in = next((n for n in ("filename_prefix", "filename", "prefix") if n in required), "filename_prefix")
    nid = next_id()
    node = {"class_type": cls,
            "inputs": {wire_in: [src_id, out_idx], prefix_in: _agent_output_prefix(cls, stem)},
            "_meta": {"title": f"Save {kind.title()} (partner_run)"}}
    default_dynamic_combos(node, required)
    workflow[nid] = node
    return nid


def build_partner_workflow(node_type: str, inputs: dict | None = None, settings: dict | None = None,
                           save: str = "auto", name: str = "", object_info: dict | None = None,
                           stage: bool = True) -> dict:
    """Build the API graph for one partner node. Returns ``{"ok": True, "workflow": {...},
    "node_id", "loaders", "saver", "applied", "warnings"}`` or ``{"ok": False, "errors": [...]}``."""
    from agenty_core.tools import assembly_deterministic as AD

    if object_info is None:
        from agenty_core.tools.comfyui import _get_object_info
        object_info = _get_object_info()
    inputs = dict(inputs or {})
    settings = dict(settings or {})
    errors: list[str] = []
    warnings: list[str] = []

    cls, cands = resolve_node_type(node_type, object_info)
    if cls is None:
        return {"ok": False, "errors": [f"no partner node named '{node_type}'."],
                "candidates": cands,
                "hint": "pick one of the candidates (class_type) and call again" if cands else
                        "search_nodes('<model name>') lists the nodes this ComfyUI has"}
    info = object_info[cls]
    if not _is_partner(info):
        warnings.append(f"{cls} is not a partner (API) node; built the same way, but check it needs no model files.")
    required = ((info.get("input") or {}).get("required") or {})
    optional = ((info.get("input") or {}).get("optional") or {})

    # -- settings: option choices first (they decide which inputs exist) --------
    values: dict = {}
    pending = dict(settings)
    expanded = AD.expand_input_specs(required, optional, values)
    def _is_selector(spec) -> bool:
        return _spec_type(spec) == "COMFY_DYNAMICCOMBO_V3"

    for _ in range(6):
        progressed = False
        # The choices that decide which other inputs exist go first, and the inputs
        # in force are re-read after each one: "resolution" must be checked against
        # the options of the model that was CHOSEN, not of the node's default model.
        for key in list(pending):
            real, _cand = _resolve_name(key, expanded)
            if real is not None and _is_selector(expanded[real]):
                try:
                    values[real] = _coerce(real, expanded[real], pending.pop(key), warnings)
                except ValueError as exc:
                    errors.append(str(exc))
                expanded = AD.expand_input_specs(required, optional, values)
                progressed = True
        for key in list(pending):
            real, _cand = _resolve_name(key, expanded)
            if real is None:
                continue
            spec = expanded[real]
            if _is_wire_spec(spec) or "AUTOGROW" in _spec_type(spec).upper():
                errors.append(f"'{key}' is a file/link input ({_spec_type(spec)}): put it in `inputs`, not `settings`.")
                pending.pop(key)
                continue
            try:
                values[real] = _coerce(real, spec, pending.pop(key), warnings)
            except ValueError as exc:
                errors.append(str(exc))
            progressed = True
        expanded = AD.expand_input_specs(required, optional, values)
        if not progressed or not pending:
            break
    for key in pending:
        _real, cand = _resolve_name(key, expanded)
        errors.append(f"unknown setting '{key}' for {cls}."
                      + (f" Did you mean: {', '.join(cand)}?" if cand else "")
                      + f" Inputs in force: {', '.join(AD.describe_input_names(expanded, 30))}")

    # -- file inputs: loaders ---------------------------------------------------
    workflow: dict = {}
    counter = [0]

    def next_id() -> str:
        counter[0] += 1
        return str(counter[0])

    node_id = next_id()                     # the partner node is "1"
    node_inputs: dict = {}
    loaders: list[dict] = []
    staged: dict = {}

    def wire_file(slot_key: str, spec, path: str) -> None:
        wtype = _spec_type(spec)
        if wtype not in _LOADERS:
            errors.append(f"'{slot_key}' takes {wtype}, which partner_run cannot load from a file.")
            return
        p = str(path or "").strip()
        if re.match(r"^https?://", p):
            errors.append(f"'{slot_key}': {p} is a URL - download it first (download_image) and pass the file.")
            return
        try:
            cls_l, widget, idx = _loader_for(wtype, object_info)
            if p not in staged:
                staged[p] = _stage(p) if stage else os.path.basename(p)
        except ValueError as exc:
            errors.append(f"'{slot_key}': {exc}")
            return
        lid = next_id()
        workflow[lid] = {"class_type": cls_l, "inputs": {widget: staged[p]},
                         "_meta": {"title": f"{cls_l}: {os.path.basename(p)}"}}
        node_inputs[slot_key] = [lid, idx]
        loaders.append({"node_id": lid, "class_type": cls_l, "input": slot_key, "file": p})

    for key, given in inputs.items():
        real, cand = _resolve_name(key, expanded)
        if real is None:
            errors.append(f"unknown input '{key}' for {cls}."
                          + (f" Did you mean: {', '.join(cand)}?" if cand else "")
                          + f" Inputs in force: {', '.join(AD.describe_input_names(expanded, 30))}")
            continue
        spec = expanded[real]
        files = list(given) if isinstance(given, (list, tuple)) else [given]
        files = [f for f in files if str(f or "").strip()]
        if "AUTOGROW" in _spec_type(spec).upper():
            slots = _slots_of_group(real, expanded)
            if len(files) > len(slots):
                errors.append(f"'{key}' takes at most {len(slots)} files; got {len(files)}.")
                continue
            for slot, path in zip(slots, files):
                wire_file(slot, expanded[slot], path)
        elif _is_wire_spec(spec):
            if len(files) != 1:
                errors.append(f"'{real}' takes one file; got {len(files)}.")
                continue
            wire_file(real, spec, files[0])
        else:
            errors.append(f"'{real}' is a plain value, not a file input: put it in `settings`.")

    if errors:
        return {"ok": False, "errors": errors, "node": cls}

    # -- the node ---------------------------------------------------------------
    node_inputs.update(values)
    if "seed" in expanded and "seed" not in node_inputs:
        node_inputs["seed"] = random.randint(0, 2**31 - 1)
    node = {"class_type": cls, "inputs": node_inputs,
            "_meta": {"title": str(info.get("display_name") or cls)}}
    reshaped: list = []
    missing = AD.harden_node_inputs(node, required, None, optional, None, reshaped)
    warnings.extend(reshaped)
    # a required input of the node that is a wire and has no loader
    for m in missing:
        if m not in node["inputs"]:
            errors.append(f"{cls} needs '{m}' ({_spec_type(required.get(m))}) - pass it in `inputs` or `settings`.")
    if errors:
        return {"ok": False, "errors": errors, "node": cls}
    workflow[node_id] = node

    # -- the saver --------------------------------------------------------------
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", name or cls).strip("_")[:48] or cls
    kind, out_idx, wire_type, problems = _output_choice(info, save)
    if problems:
        return {"ok": False, "errors": problems, "node": cls}
    saver = None
    if kind:
        try:
            saver = _add_saver(workflow, kind, node_id, out_idx, wire_type, stem, object_info, next_id)
        except ValueError as exc:
            return {"ok": False, "errors": [str(exc)], "node": cls}
    return {"ok": True, "workflow": workflow, "node": cls, "node_id": node_id, "loaders": loaders,
            "saver": saver, "kind": kind or "node output", "stem": stem,
            "applied": {k: v for k, v in node["inputs"].items() if not isinstance(v, list)},
            "warnings": warnings}


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------

def prepare_partner_workflow(node_type: str, inputs: dict | None = None, settings: dict | None = None,
                             save: str = "auto", name: str = "") -> dict:
    """Build, save and locally validate. ``{"status": "ready", "workflow_path", ...}`` or an error dict."""
    from agenty_core.tools import comfyui as C

    inputs = _as_dict(inputs)
    settings = _as_dict(settings)
    try:
        built = build_partner_workflow(node_type, inputs, settings, save, name)
    except Exception as exc:  # noqa: BLE001 - object_info down, upload failed ...
        return {"status": "error", "errors": [f"{type(exc).__name__}: {exc}"]}
    if not built.get("ok"):
        return {"status": "error", **{k: v for k, v in built.items() if k != "ok"}}
    stem = f"partner_{built['stem']}_{random.randint(0, 0xffff):04x}"
    path = C._save_workflow(built["workflow"], stem)
    token = C.local_only().set(True)         # the run itself is the server-side check; never submit twice
    try:
        check = json.loads(C.validate_workflow(path))
    finally:
        C.local_only().reset(token)
    local_errors = list(check.get("local_errors") or [])
    if local_errors or check.get("missing_inputs"):
        return {"status": "error", "workflow_path": path, "errors": local_errors,
                "missing_inputs": check.get("missing_inputs") or [], "node": built["node"]}
    return {"status": "ready", "workflow_path": path, "node": built["node"], "node_id": built["node_id"],
            "loaders": built["loaders"], "save_node": built["saver"], "output_kind": built["kind"],
            "applied_settings": built["applied"], "warnings": built["warnings"]}


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            got = json.loads(value)
            return got if isinstance(got, dict) else {}
        except ValueError:
            return {}
    return {}


def _default_run(path: str, timeout: float = 1800.0) -> dict:
    """No host hook: submit to ComfyUI and wait for the outputs."""
    from agenty_core import comfyui_exec as X
    from agenty_core import partner_credits as _credits
    from agenty_core.batch_runner import _wait_for_completion
    from agenty_core.utils.comfyui_client import get_client
    try:
        pid = X._submit_workflow(path)
    except _credits.CostGateError as gate:
        return gate.payload()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}
    done = _wait_for_completion(get_client(), pid, timeout=timeout)
    spent = _credits.settle(pid)
    extra = {"partner_cost_usd": spent} if spent is not None else {}
    if done.get("error"):
        return {"status": "error", "prompt_id": pid, "error": done["error"], **extra}
    files = X._extract_output_files(done.get("history") or {})
    outs = []
    for f in files:
        try:
            outs.append(str(X._resolve_output_path(f["filename"], f.get("subfolder", ""), f.get("type", "output"))))
        except Exception as exc:  # noqa: BLE001
            outs.append(f"{f.get('filename')} (could not be located: {exc})")
    return {"status": "done", "prompt_id": pid, "outputs": outs, **extra}


@tool
def partner_run(node_type: str, inputs: dict | None = None, settings: dict | None = None,
                save: str = "auto", name: str = "", run: bool = True, confirm_cost: bool = False) -> str:
    """Build and run ONE partner (API) node directly - no template search.

    The preferred route when the request names a partner model or node: Seedance,
    Kling, GPT Image, Nano Banana / Gemini image, Ideogram, Meshy, Veo, Runway ...
    The node's own schema decides the inputs; files get loaders, the result gets a
    save node, the graph is validated and (``run=True``) executed like any other
    workflow. One call instead of search -> template -> assemble -> validate.

    ``node_type`` is the class name (``ByteDance2ReferenceNodeV2``) or the name you
    know it by ("Seedance 2.5 first last frame"); an ambiguous name returns the
    candidates and builds nothing.

    ``inputs`` maps a FILE slot to a path, or a list of paths for a group of slots,
    e.g. ``{"first_frame": "D:/a.png"}`` or
    ``{"model.reference_images": ["a.png", "b.png"], "model.reference_videos": "m.mp4"}``
    (a group fills ``image_1, image_2, ...`` in order; or name one slot:
    ``model.reference_images.image_3``).

    ``settings`` maps everything else - the prompt, ``model`` (a dynamic combo: the
    option decides which other settings exist, e.g. "Seedance 2.5"), ``resolution``,
    ``duration``, ``generate_audio``, ``seed``. A bare name works when it is
    unambiguous (``resolution`` for ``model.resolution``). A value the node does
    not offer is an ERROR that lists the options - nothing is silently changed.

    ``save``: "image", "video" or "auto" (by what the node outputs); ``name`` names
    the output files. ``run=False`` only builds and validates (returns
    ``workflow_path``).

    Costs real money on the user's partner credits when it runs: do not call it
    speculatively, and not twice for the same request. The price shows before it
    runs (``estimate_usd`` in the answer; ``run=False`` returns it without running).
    Above the confirmation level (default $1) the run is refused with
    ``error_kind: confirm_cost`` - tell the user the estimate, ask, and call again
    with ``confirm_cost=True`` only after a yes. A run the balance cannot cover is
    refused as ``insufficient_credits``. A finished run reports ``partner_cost_usd``.

    Args:
        node_type: Node class name or its display name.
        inputs: ``{slot: file path | [paths]}`` for image / video / audio inputs.
        settings: ``{name: value}`` for prompt and every widget.
        save: "image", "video" or "auto".
        name: Short name for the output files and the workflow file.
        run: False to build and validate only.
        confirm_cost: True only once the user has agreed to the estimated cost.
    """
    from agenty_core import partner_credits as _credits
    prepared = prepare_partner_workflow(node_type, inputs, settings, save, name)
    if prepared.get("status") != "ready":
        return json.dumps(prepared)
    wf: dict = {}
    try:
        wf = json.loads(Path(prepared["workflow_path"]).read_text(encoding="utf-8"))
        est = _credits.estimate_report(wf)
        if est["paid"]:
            prepared["estimate_usd"] = {"min": est["min_usd"], "max": est["max_usd"],
                                        "unknown_nodes": est["unknown"],
                                        **({"price_unknown": True, "warning": est["warning"]}
                                           if est.get("price_unknown") else {}),
                                        "balance": (est.get("balance") or {}).get("effective_usd")}
    except Exception:  # noqa: BLE001 - an estimate must never stop a build
        pass
    if not run:
        return json.dumps({**prepared, "status": "ready", "ran": False})
    if wf:
        try:
            _credits.preflight(wf, confirm=confirm_cost, workflow_path=prepared["workflow_path"])
        except _credits.CostGateError as gate:
            return json.dumps({**prepared, **gate.payload(), "ran": False})
        except Exception:  # noqa: BLE001
            pass
    if confirm_cost:
        # The host's run hook submits the file later; the go-ahead travels with it.
        _credits.confirm_path(prepared["workflow_path"], float("inf"))
    hook = _run_hook or _default_run
    try:
        ran = hook(prepared["workflow_path"])
    except Exception as exc:  # noqa: BLE001
        ran = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if isinstance(ran, str):
        try:
            ran = json.loads(ran)
        except ValueError:
            ran = {"result": ran}
    return json.dumps({**prepared, "ran": True, "run": ran})
