"""
ComfyUI tools - server communication, workflow management, and node inspection.

Consolidates all ComfyUI-related @tool functions into a single module:
  • Server: models, execution control, queue, history, prompt submission
  • Workflows: template loading, patching, validation
  • Nodes: schema inspection, keyword search
"""

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from agenty_core._compat import tool
from agenty_core.tools._batch import as_list as _as_list, one_or_many as _one_or_many

from agenty_core.utils.comfyui_client import get_client, parse_argv_dir_flag
# Deterministic-assembly hardening lives in its own module; apply_brainbriefing
# and update_workflow delegate to it (the LLM brain and the deterministic path
# share the same robustness).
from agenty_core.tools.assembly_deterministic import (
    LINK_ONLY_TYPES as _LINK_ONLY_TYPES_SHARED,
    _MODEL_EXTS as _MODEL_EXTS_SHARED,
    autowire_dangling_inputs as _autowire_dangling_inputs,
    coerce_dim as _coerce_dim,
    dead_node_warnings as _dead_node_warnings,
    ensure_output_node as _ensure_output_node,
    harden_node_inputs as _harden_node_inputs,
    link_only_scalars as _link_only_scalars,
    _is_wire as _is_wire_value,
    autogrow_slot_names as _autogrow_slot_names,
    declared_input_keys as _declared_input_keys,
    dynamic_sub_specs as _dynamic_sub_specs,
    is_autogrow_spec as _is_autogrow_group,
    normalize_autogrow_keys as _normalize_autogrow_keys,
    undeclared_wires as _undeclared_wires,
    rebind_placeholder_images as _rebind_placeholder_images,
    strip_annotation_nodes as _strip_annotation_nodes,
    strip_reroute_nodes as _strip_reroute_nodes,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level state
# ═══════════════════════════════════════════════════════════════════════════════

# Workflow files are saved to disk and referenced by path to avoid bloating
# the LLM's sliding-window context with full JSON.
# patch_workflow failure guard
_PATCH_FAIL_LIMIT: int = 3
_patch_fail_count: int = 0
_patch_last_workflow_path: str | None = None

# /object_info cache – the full node database doesn't change during a session.
_object_info_cache: dict | None = None

# In-memory caches for template data (reset on process restart).
_index_cache: list | None = None
_official_index_cache: list | None = None
_template_cache: dict[str, dict] = {}

# ── Session-level tool-response caches ──────────────────────────────────────
# These are reset at the start of each pipeline session via clear_tool_caches().
# Thread-safety: all access is protected by _tool_cache_lock.
_tool_cache_lock: threading.Lock = threading.Lock()
_CATALOG_TTL: int = 3600  # seconds – workflow catalog TTL

_tool_catalog_result: str | None = None         # cached get_workflow_catalog() return value
_tool_catalog_timestamp: float | None = None    # time.time() when catalog was last fetched
_tool_dirs_result: str | None = None             # cached get_comfyui_dirs() return value
_tool_template_results: dict[str, str] = {}     # cached get_workflow_template() results, keyed by name


# ═══════════════════════════════════════════════════════════════════════════════
# Public helpers (non-tool, used by other modules)
# ═══════════════════════════════════════════════════════════════════════════════

def reset_patch_workflow_guard() -> None:
    """Reset the patch_workflow failure counter.  Call once per orchestration session."""
    global _patch_fail_count, _patch_last_workflow_path
    _patch_fail_count = 0
    _patch_last_workflow_path = None


def _server_validate(workflow: dict) -> dict:
    """Server-side validate a workflow WITHOUT disturbing other ComfyUI jobs.

    ComfyUI validates a prompt only by *submitting* it to ``/prompt`` — which then
    QUEUES and RUNS it (there is no dry-run). For an API/partner node that means a
    real API call, and the only way to stop the run is a GLOBAL ``/interrupt``
    (ComfyUI has no per-prompt interrupt). Historically this validation submitted
    the graph and blanket-interrupted / cleared the queue, which:

      * executed the workflow's API nodes just to validate them (→ "Task cancelled"),
      * killed whatever ELSE was running — e.g. the executor's real render — which
        the executor then reported as a failure and tried to *heal*, re-validating
        and interrupting again: an endless submit→interrupt→heal loop.

    This helper makes validation side-effect-free for other jobs:

      * If ComfyUI is BUSY (anything running), SKIP server validation and return
        ``{}`` — never risk a live render; local validation still ran.
      * Otherwise submit, delete our own prompt by id, and ``/interrupt`` ONLY when
        we re-confirm OUR validation prompt is the sole running job. Never a blanket
        interrupt of someone else's render, never ``{"clear": True}``.

    Returns a ``server_errors`` dict (empty when valid or skipped).
    """
    server_errors: dict = {}
    try:
        _vc = get_client()
        # Don't touch ComfyUI while a real job is running — rely on local checks.
        try:
            if (_vc.get("/queue").get("queue_running") or []):
                return {}
        except Exception:
            pass
        _vpayload: dict = {"prompt": workflow}
        if _vc.api_key:  # API/partner nodes 401 without the key, even to validate
            _vpayload["extra_data"] = {"api_key_comfy_org": _vc.api_key}
        result = _vc.post("/prompt", json_data=_vpayload)
        if isinstance(result, dict):
            if "error" in result:
                server_errors = {
                    "error": result.get("error"),
                    "node_errors": result.get("node_errors", {}),
                }
            elif "prompt_id" in result:
                # A prompt_id is NOT a clean bill of health. ComfyUI validates
                # per output node: when some outputs are good and others are not
                # it drops the bad ones, keeps the rest, and answers 200 with a
                # populated `node_errors` and no top-level `error`
                # (execution.validate_prompt -> server.py). Reading that as valid
                # is how a graph whose entire video branch was rejected —
                # CreateVideo.bit_depth=16 against a max of 10 — reported
                # `valid: true, server_errors: {}` over and over, ran the image
                # half, and returned "success" with no video and no explanation
                # anywhere the agent could look.
                _vnode_errors = result.get("node_errors")
                if isinstance(_vnode_errors, dict) and _vnode_errors:
                    server_errors = {
                        "error": {
                            "type": "outputs_rejected",
                            "message": ("ComfyUI rejected some output branches and "
                                        "would run the rest — those outputs will "
                                        "silently not be produced."),
                        },
                        "node_errors": _vnode_errors,
                    }
                _vpid = result["prompt_id"]
                try:
                    _vc.post("/queue", json_data={"delete": [_vpid]})
                    _running = _vc.get("/queue").get("queue_running") or []
                    # Interrupt ONLY when our validation prompt is the SOLE running
                    # job — never blanket-kill a render that started meanwhile.
                    if (len(_running) == 1 and isinstance(_running[0], (list, tuple))
                            and len(_running[0]) > 1 and _running[0][1] == _vpid):
                        _vc.post("/interrupt", json_data={})
                except Exception:
                    pass
    except Exception as e:
        err_str = str(e)
        if hasattr(e, "response"):
            try:
                server_errors = e.response.json()
            except Exception:
                server_errors = {"error": err_str}
        else:
            server_errors = {"error": err_str}
    return server_errors


def clear_tool_caches() -> None:
    """Reset all session-level tool-response caches.

    Call this once at the start of each pipeline session so that every new
    session fetches fresh data from ComfyUI rather than reusing stale results
    from a previous session in the same process (e.g. in long-running servers).

    Affected caches:
      - get_workflow_catalog  (1-hour TTL; cleared so next call fetches fresh)
      - get_comfyui_dirs      (no TTL; cleared so next call fetches fresh)
      - get_workflow_template (per-name; cleared so templates re-fetched if changed)
    """
    global _tool_catalog_result, _tool_catalog_timestamp, _tool_dirs_result, _tool_template_results, _object_info_cache
    with _tool_cache_lock:
        _tool_catalog_result = None
        _tool_catalog_timestamp = None
        _tool_dirs_result = None
        _tool_template_results = {}
        # Drop /object_info too — its LoadImage combo lists the input dir, which
        # changes as images are uploaded, so a stale copy misroutes input images.
        _object_info_cache = None


def reset_object_info_cache() -> None:
    """Drop the cached ``/object_info`` so the next fetch reflects newly-staged
    input files.

    ``LoadImage``'s valid-file combo is part of ``/object_info``. When an image is
    staged into ComfyUI's input dir mid-session (e.g. ``upload_image`` during the
    researcher run), a stale cache omits it, so ``apply_brainbriefing``'s
    snap-to-valid step would replace the user's freshly-uploaded image with the
    template's default — i.e. the workflow "ignores" the input image. Call this
    after staging an input, or before assembling, to force a refresh.
    """
    global _object_info_cache
    _object_info_cache = None


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_object_info() -> dict:
    """Return the full /object_info dict, cached after first fetch."""
    global _object_info_cache
    if _object_info_cache is None:
        _object_info_cache = get_client().get("/object_info")
    return _object_info_cache


def registered_node_classes() -> set[str]:
    """Every node class THIS ComfyUI has, or an empty set if it cannot be asked.

    Which custom node packs are installed is a property of the machine, not of
    any workflow, so anything choosing between "use the node from pack X" and
    "fall back to core" has to ask rather than assume.

    An empty set means *unknown*, not *none*: ComfyUI being unreachable is
    indistinguishable here from an install with no nodes at all, and a caller
    that reads it as "the pack is absent" would commit to a fallback it has no
    way to verify either.
    """
    try:
        info = _get_object_info()
    except Exception:  # noqa: BLE001 — ComfyUI down is "cannot be asked"
        return set()
    return set(info) if isinstance(info, dict) else set()


def node_default_inputs(class_type: str) -> dict:
    """Every REQUIRED widget input of *class_type*, at its declared default.

    ComfyUI does not fill defaults in. A required input missing from an API
    prompt is rejected outright — ``"Required input is missing"``, whatever
    default the node declares — so anything BUILDING a node from scratch has to
    write them itself. A node assembled with only its interesting widget set
    looks complete and fails validation the moment it is queued:
    ``VHS_LoadImagePath`` needs ``custom_width`` and ``custom_height`` as much as
    it needs the path.

    Inputs that take a WIRE are left out. Those need a link, and a literal in one
    is not a default, it is a type error.
    """
    try:
        info = _get_object_info() or {}
    except Exception:  # noqa: BLE001
        return {}
    required = ((info.get(class_type) or {}).get("input") or {}).get("required") or {}
    out: dict = {}
    for name, spec in required.items():
        if not isinstance(spec, (list, tuple)) or not spec:
            continue
        kind = spec[0]
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if "default" in opts:
            out[name] = opts["default"]
        elif isinstance(kind, list) and kind:
            out[name] = kind[0]      # a combo declares its default by being first
    return out


from agenty_core.paths import project_root as _project_root
from agenty_core.paths import corpus_root as _corpus_root


def _load_config() -> dict:
    """Load the settings.json configuration."""
    config_path = _project_root() / "config" / "settings.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.loads("".join(ln for ln in f if not ln.lstrip().startswith("//")))
    return {}


# NOTE: `_user_workflows_dir` intentionally removed. Use `_custom_templates_dir()`
# instead — user ad-hoc workflows are stored with the custom templates.


def _custom_templates_dir() -> Path:
    """Return the path to the custom workflow templates directory (has its own index.json)."""
    cfg = _load_config()
    ct_dir = cfg.get(
        "comfyui_custom_templates_dir",
        "./comfyui_workflow_templates_custom/templates/",
    )
    # Canonical corpus lives in agenty_core (shared), not the per-app root.
    return (_corpus_root() / ct_dir).resolve()


def _official_templates_dir() -> Path:
    """Return the path to the official (Comfy-Org) workflow templates directory.

    Templates live here when present; get_workflow_template falls back to this
    directory so official templates are loadable, not just listed in the catalog.
    """
    cfg = _load_config()
    od = cfg.get("comfyui_official_templates_dir", "./comfyui_workflow_templates_official/")
    return (_corpus_root() / od).resolve()


def _workflows_dir() -> Path:
    """Return the directory where generated/patched workflow JSON files are saved.

    These belong to the *user's ComfyUI install*, not to an agent checkout: they
    are workflows they will want to open, and writing them inside the repo both
    dirties it (which used to block the startup updater) and hides them from
    ComfyUI's workflow browser. So the default target is
    ``<user-directory>/default/workflows/agentY/``.

    ``output_workflows_dir`` in settings still wins when set to an ABSOLUTE path,
    for anyone who wants them elsewhere. The relative default is used only as a
    fallback for when ComfyUI can't be reached (nothing can run then anyway, but
    assembly tools still need somewhere to write).
    """
    cfg = _load_config()
    wd = str(cfg.get("output_workflows_dir", "") or "").strip()
    if wd and Path(wd).is_absolute():
        return Path(wd).resolve()
    try:
        from agenty_core.utils.comfyui_client import comfyui_agent_workflows_dir
        target = comfyui_agent_workflows_dir()
        if target is not None:
            return target
    except Exception:  # noqa: BLE001 — fall through to the in-repo fallback
        pass
    return (_project_root() / (wd or "./output_workflows/")).resolve()


def _load_index() -> list:
    """Return the templates index as a flat list from the custom templates directory.

    Result is cached for the lifetime of the process.
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    flat: list[dict] = []
    index_path = _custom_templates_dir() / "index.json"
    if index_path.exists():
        try:
            with open(index_path, encoding="utf-8") as f:
                raw = json.load(f)
            for group in (raw or []):
                group_category = group.get("title", group.get("category", ""))
                group_media = group.get("type", "")
                for tpl in group.get("templates", []):
                    tpl["_group_category"] = group_category
                    tpl["_group_media"] = group_media
                    flat.append(tpl)
        except Exception:
            pass

    _index_cache = flat
    return flat


def _official_index() -> list:
    """Flat list of official (Comfy-Org) template descriptors from the official
    ``index.json`` files. Used by the fuzzy fallback so a near-miss / hallucinated
    name can resolve to an official template (which _fetch_template can load by
    name), not just the custom ones. Cached for the process lifetime.

    The official corpus is two mirrors — ``blueprints/`` and ``templates/`` — each
    carrying upstream's own index, so every index.json under the root is read.
    """
    global _official_index_cache
    if _official_index_cache is not None:
        return _official_index_cache
    flat: list[dict] = []

    def _walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("name"), str):
                flat.append(o)
            for v in o.values():
                _walk(v)
        elif isinstance(o, list):
            for x in o:
                _walk(x)

    root = _official_templates_dir()
    for index_path in sorted(root.glob("**/index.json")):
        try:
            with open(index_path, encoding="utf-8") as f:
                _walk(json.load(f))
        except Exception:
            continue
    _official_index_cache = flat
    return flat


def _find_template_file(base: Path, name: str) -> Path | None:
    """Resolve template *name* to a file under *base*, or None.

    The official corpus is split into ``blueprints/`` and ``templates/`` mirrors,
    so a flat lookup no longer finds everything. Flat is tried first — it is the
    common case and stays cheap — then a walk. Sorted, so the winner is
    deterministic if a name ever appears in both mirrors.
    """
    for candidate in (base / f"{name}.json", base / name):
        if candidate.exists():
            return candidate
    hits = sorted(base.glob(f"**/{name}.json"))
    return hits[0] if hits else None


def _fetch_template(name: str) -> dict | None:
    """Load a single template JSON by name from the custom templates directory.

    Returns the parsed dict, or None if not found.
    Results are cached.
    """
    if name in _template_cache:
        return _template_cache[name]

    data: dict | None = None
    # Custom templates take precedence; fall back to the official directory so
    # official templates load too (previously they were "not found").
    for base in (_custom_templates_dir(), _official_templates_dir()):
        found = _find_template_file(base, name)
        if found is not None:
            with open(found, encoding="utf-8") as f:
                data = json.load(f)
            break

    if data is not None:
        _template_cache[name] = data
    return data


def _save_workflow(workflow: dict, name: str = "") -> str:
    """Save *workflow* dict to a JSON file and return the absolute path."""
    wd = _workflows_dir()
    wd.mkdir(parents=True, exist_ok=True)
    stem = name or uuid.uuid4().hex[:8]
    path = wd / f"{stem}.json"
    path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return str(path.resolve())


def _edit_target(workflow_path) -> Path | None:
    """The file an edit to *workflow_path* is written back to, or None.

    Every editing tool loads the file it is handed, and they all used to save to
    ``<agentY workflows>/<stem>.json`` whatever that was. Handed a path anywhere
    else, each call therefore started over from the untouched original: an agent
    re-linked a prompt in one call, fixed another input in the next, and the
    second call - re-reading a file the first had never written to - discarded
    the first fix while both results listed their patches as applied. The
    executor, queued with the path the agent had named, ran that original as
    well, so a repaired graph was re-queued unrepaired three times over.

    So an edit goes back where it came from. None - keep a copy in the agentY
    folder, as before - for a raw JSON string, a canvas-format file (a workflow
    someone saved from ComfyUI: converted, never overwritten), and anything
    inside the template corpus.
    """
    try:
        p = Path(str(workflow_path or "").strip())
        if p.suffix.lower() != ".json" or not p.is_file():
            return None
        resolved = p.resolve()
        if resolved.is_relative_to(_corpus_root().resolve()):
            return None
        if _is_graph_format(json.loads(resolved.read_text(encoding="utf-8"))):
            return None
        return resolved
    except (OSError, ValueError):
        return None


def _save_edit(workflow: dict, workflow_path: str) -> str:
    """Save an edited *workflow* back to *workflow_path* - see :func:`_edit_target`."""
    target = _edit_target(workflow_path)
    if target is None:
        return _save_workflow(workflow, name=Path(str(workflow_path or "").strip()).stem)
    target.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    return str(target)


def _moved_note(workflow_path: str, saved_path: str) -> str:
    """A sentence when an edit landed somewhere other than the path passed in."""
    try:
        same = Path(str(workflow_path).strip()).resolve() == Path(saved_path).resolve()
    except (OSError, ValueError):
        same = False
    if same:
        return ""
    return (f"Saved as {saved_path}, not over the file you passed (a canvas workflow "
            f"or a template is never overwritten). Use this workflow_path from now "
            f"on - the original still holds the unedited graph.")


def _load_workflow(path_or_json: str) -> dict:
    """Load a workflow from a file path or raw JSON string.

    Auto-converts graph-format workflows to API format.
    """
    # Paths come from an LLM, sometimes with a newline on either side, and a
    # stray one made a file that exists look missing.
    raw = str(path_or_json).strip()
    p = Path(raw)
    if p.suffix == ".json" and p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(raw)
    if _is_graph_format(data):
        data = _convert_graph_to_api(data)
    return data


def _is_graph_format(workflow: dict) -> bool:
    """Return True if *workflow* is in ComfyUI graph/export format."""
    return isinstance(workflow.get("nodes"), list)


# Input types that ComfyUI always wires via links (never appear as widget values).
# Defined next to the repair that depends on it, so alignment and repair can never
# drift apart again — see assembly_deterministic.LINK_ONLY_TYPES.
_LINK_ONLY_TYPES = _LINK_ONLY_TYPES_SHARED

# The only input types the frontend renders as a widget (plus inline COMBOs,
# which arrive as a list of options, and the V3 combo/autogrow wrappers handled
# separately). _LINK_ONLY_TYPES was an allowlist of socket types, so anything a
# custom node invented — MODEL_TASK_ID, IMAGECOMPARE, WANVIDIMAGE_EMBEDS, and
# 200-odd others — slipped through and got reported as a missing *widget value*
# when it is a wire. Deciding by what IS a widget is the closed set.
_WIDGET_TYPES: frozenset[str] = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"})


def _is_widget_spec(spec) -> bool:
    """True when this input is a widget the user types/picks, not a socket."""
    typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    if isinstance(typ, list):          # inline COMBO: the options themselves
        return True
    if not isinstance(typ, str):
        return False
    return typ in _WIDGET_TYPES or typ == "COMFY_DYNAMICCOMBO_V3"


def _widget_options(spec) -> list:
    """The selectable values for a COMBO-ish widget, else [].

    Covers both shapes. A plain COMBO lists its options as strings. A V3 dynamic
    combo lists them as objects — ``{"key": "MiniMax H3", "inputs": {…}}`` — where
    the key is the value the node accepts and the rest describes what choosing it
    unfolds. Reading only the strings left every dynamic combo looking
    unconstrained, which is 114 inputs, 89 of them on partner-API nodes, and they
    are exactly the model selectors whose exact spelling cannot be guessed:
    ``Flux.2 [pro]``, ``Nano Banana 2 (Gemini 3.1 Flash Image)``.
    """
    typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    if isinstance(typ, list):
        return list(typ)
    meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
    if isinstance(meta, dict) and isinstance(meta.get("options"), list):
        return [o.get("key") if isinstance(o, dict) else o for o in meta["options"]]
    return []


# Widgets with no schema default where one value is the obvious intent. Kept
# deliberately tiny and node-specific: a general "pick the first option" rule
# would silently choose things like an export codec at random.
_CONVENTIONAL_DEFAULTS: dict = {
    ("VHS_VideoCombine", "format"): "video/h264-mp4",
}

_SEED_CONTROL_VALUES: frozenset[str] = frozenset({"fixed", "randomize", "increment", "decrement"})
_SEED_INPUT_NAMES: frozenset[str] = frozenset({"seed", "noise_seed"})


def _schema_widget_slots(schema: dict, linked_names: set[str]) -> list[tuple]:
    """Ordered ``(input_name, spec)`` widget slots for a node schema.

    Mirrors the logic ComfyUI's frontend uses to assign widget_values entries
    to named inputs.

    Note this excludes by ``_LINK_ONLY_TYPES`` rather than by ``_is_widget_spec``.
    The two answer different questions and measurably disagree: alignment has to
    mirror what the *frontend serialises*, which includes a value for some
    custom-typed widgets (socketless display widgets and the like), while
    ``_is_widget_spec`` answers "could a caller type a value here" for reporting
    and defaulting. Using the strict rule for alignment costs 38 templates.

    Whether a *wired-up* widget still occupies a slot depends on how the graph
    was written: a seed driven by a primitive node sometimes keeps its last UI
    value (and control string) in widgets_values and sometimes does not. Neither
    answer is right everywhere — dropping the slot when the value is present
    shifts everything after it (a KSampler came out with steps=561594583201063,
    cfg="fixed", denoise="euler"), and consuming it when it is absent shifts them
    the other way. So this returns the *excluding* variant and the caller tries
    both, keeping whichever assigns values that fit their slots.
    """
    slots: list[tuple] = []
    for section in ("required", "optional"):
        for name, spec in (schema.get(section) or {}).items():
            if name in linked_names:
                continue
            typ = spec[0] if isinstance(spec, (list, tuple)) and spec else ""
            if isinstance(typ, str) and typ in _LINK_ONLY_TYPES:
                continue
            # COMFY_MATCHTYPE_V3 takes whatever type is wired into it (switches,
            # concatenators): a socket wearing a V3 name, never a widget.
            if typ == "COMFY_MATCHTYPE_V3":
                continue
            # An autogrow group is a row of sockets, never a widget: the frontend
            # saves no value for it (video_minimax_h3_r2v declares four groups and
            # its widgets_values are prompt, width, height, length, ref_image_size).
            # A slot here wrote a null per group into every graph built from an API
            # workflow, and could swallow a real value reading one back.
            if isinstance(typ, str) and "AUTOGROW" in typ.upper():
                continue
            meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
            # forceInput promotes a widget-typed input to a socket-only one, so it
            # has no widget and claims no slot — 104 inputs where letting it claim
            # one would shift every value after it.
            if isinstance(meta, dict) and meta.get("forceInput"):
                continue
            slots.append((name, spec))
    return slots


def _schema_widget_names(schema: dict, linked_names: set[str]) -> list[str]:
    """Ordered widget input names for a node schema (names only)."""
    return [name for name, _spec in _schema_widget_slots(schema, linked_names)]


def _fill_required_defaults(api_inputs: dict, schema: dict, linked_names: set,
                            class_type: str = "") -> None:
    """Give every required widget input still absent a defensible value.

    A template need not serialise a widget the author never touched —
    api_kling_v3_flf2v ships five widget values and no ``seed`` — but /prompt
    requires it all the same. ComfyUI's frontend materialises one from the
    schema, so do the same rather than submitting a graph that fails validation
    on a value nobody chose.

    Three sources, in descending confidence: the schema's explicit ``default``,
    a COMBO with exactly one option (no choice to get wrong), and the small
    ``_CONVENTIONAL_DEFAULTS`` table. Anything else is left absent for the
    caller to decide — validate_workflow reports it with its options attached.
    """
    for name, spec in (schema.get("required") or {}).items():
        if name in linked_names or _is_autogrow_spec(spec):
            continue
        if not _is_widget_spec(spec):
            continue                    # a socket, not a widget: it wants a wire
        opts = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
        if name not in api_inputs:
            choices = _widget_options(spec)
            if isinstance(opts, dict) and "default" in opts:
                api_inputs[name] = opts["default"]
            elif len(choices) == 1:
                api_inputs[name] = choices[0]
            elif (class_type, name) in _CONVENTIONAL_DEFAULTS:
                conventional = _CONVENTIONAL_DEFAULTS[(class_type, name)]
                if not choices or conventional in choices:
                    api_inputs[name] = conventional
        # The selected option of a dynamic combo brings its own required inputs;
        # they are just as mandatory and just as often left unserialised.
        subs = _dynamic_combo_suboptions(spec)
        if subs is None or name not in api_inputs:
            continue
        chosen = next((o for o in (opts or {}).get("options") or []
                       if isinstance(o, dict) and str(o.get("key")) == str(api_inputs[name])), None)
        for sub, sub_spec in ((chosen or {}).get("inputs") or {}).get("required", {}).items():
            key = f"{name}.{sub}"
            sub_opts = sub_spec[1] if isinstance(sub_spec, (list, tuple)) and len(sub_spec) > 1 else None
            if key not in api_inputs and isinstance(sub_opts, dict) and "default" in sub_opts:
                api_inputs[key] = sub_opts["default"]


def _value_fits(spec, value) -> bool:
    """Is *value* a plausible occupant of this widget slot?

    Deliberately shallow — the point is to tell a *shifted* mapping from an
    aligned one, and a shift shows up immediately as a sampler name landing in a
    float or a control string landing in a combo.
    """
    if isinstance(value, list):            # a link, not a literal
        return True
    options = _widget_options(spec)
    if options:
        return str(value) in {str(o) for o in options}
    typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    if typ == "BOOLEAN":
        return isinstance(value, bool)
    if typ in ("INT", "FLOAT"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        # Honour the declared range. Type alone cannot separate two readings when
        # both put *a number* in a numeric slot, and that tie is what shipped
        # CreateVideo.bit_depth=16 into a slot whose max is 10: the frame rate
        # from the shifted reading looked exactly as plausible as the real 8, so
        # the tie-break picked the shifted one and ComfyUI silently dropped the
        # whole video branch. A value outside min/max is the clearest evidence a
        # mapping has slipped, and it is free.
        meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
        if isinstance(meta, dict):
            lo, hi = meta.get("min"), meta.get("max")
            if isinstance(lo, (int, float)) and value < lo:
                return False
            if isinstance(hi, (int, float)) and value > hi:
                return False
        return True
    if typ == "STRING":
        return isinstance(value, str)
    return True


def _mapping_score(mapped: dict, slots: list) -> int:
    """+1 per assignment that fits its slot, -1 per one that cannot.

    Used to choose between two readings of the same widget list. Counting
    leftovers cannot do it: the reading with more slots always consumes more
    values, so it wins on leftovers even when every value it assigned is wrong.
    """
    specs = {name: spec for name, spec in slots}
    score = 0
    for name, value in mapped.items():
        spec = specs.get(name)
        if spec is None:
            continue
        score += 1 if _value_fits(spec, value) else -1
    return score


def _map_widget_values(slots: list, widgets_values: list, linked_names: set) -> tuple:
    """Assign ``widgets_values`` positionally to *slots*, ``(mapped, leftover)``.

    *slots* is ``[(input_name, spec_or_None)]``. A seed widget swallows the
    control value that follows it ("randomize"/"fixed"), and a dynamic combo
    swallows the widgets its selected option contributes. ``leftover`` is the
    values no slot claimed — non-empty means the slot list did not describe this
    node, which is the caller's signal to try the other source.
    """
    mapped: dict = {}
    i = 0
    for name, spec in slots:
        if i >= len(widgets_values):
            break
        val = widgets_values[i]
        if name not in linked_names:
            mapped[name] = val
        i += 1
        # A dynamic combo's selected option contributes its own widgets right
        # after it. Without this every following value shifts by one per
        # sub-input: the prompt lands in `seed`, the resolution in `watermark`,
        # and the sub-inputs ComfyUI requires (model.prompt, model.resolution,
        # model.duration) are never set, so /prompt rejects the graph.
        subs = _dynamic_combo_suboptions(spec)
        if subs is not None:
            for sub in subs.get(str(val), []):
                if i >= len(widgets_values):
                    break
                mapped[f"{name}.{sub}"] = widgets_values[i]
                i += 1
            continue
        # A widget with control_after_generate is followed by its control value
        # ("fixed"/"randomize"/…), which is UI state, not an input. The schema
        # declares which widgets have one; the name list is the fallback for
        # specs that predate the flag.
        meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
        has_control = (isinstance(meta, dict) and meta.get("control_after_generate")) \
            or name in _SEED_INPUT_NAMES
        if (has_control
                and i < len(widgets_values)
                and widgets_values[i] in _SEED_CONTROL_VALUES):
            i += 1
    return mapped, list(widgets_values[i:])


def _dynamic_combo_suboptions(inp_spec) -> dict | None:
    """``{option_key: [sub_input_name, ...]}`` for a V3 dynamic-combo input, else None.

    A ``COMFY_DYNAMICCOMBO_V3`` input is one widget whose *selected option* pulls
    in further widgets: picking "MiniMax H3" on a Minimax node adds prompt,
    resolution and duration. The frontend serialises those as dotted inputs
    (``model.prompt``), and they occupy the widget_values slots straight after
    the combo's own value. Which sub-inputs appear depends on the value, so the
    expansion cannot be read from the schema alone.
    """
    if not (isinstance(inp_spec, (list, tuple)) and len(inp_spec) >= 2):
        return None
    if inp_spec[0] != "COMFY_DYNAMICCOMBO_V3" or not isinstance(inp_spec[1], dict):
        return None
    out: dict = {}
    for opt in inp_spec[1].get("options") or []:
        if not isinstance(opt, dict):
            continue
        inner = opt.get("inputs") or {}
        # Only the widgets. An option can bring sockets as well - Nano Banana 2
        # adds an ``images`` autogrow group and a ``files`` wire - and those take
        # no place in widgets_values: that template's nine values are prompt,
        # model, aspect_ratio, resolution, thinking_level, seed, its control,
        # response_modalities and system_prompt. Counting them put the seed into
        # model.images and shifted every value after it.
        names = [n for n, _spec in _schema_widget_slots(inner, set())]
        out[str(opt.get("key"))] = names
    return out


def _dynamic_widget_names(schema: dict, inputs: dict, linked: set[str]) -> list[str]:
    """Widget names in canvas order for an API node, expanding dynamic combos.

    The inverse of what :func:`_map_widget_values` does when reading a canvas node:
    each dynamic combo is followed by the dotted sub-inputs its *selected* option
    contributes, in schema order. Empty when the schema is unknown, which tells the
    caller to fall back rather than emit an order it cannot justify.
    """
    if not schema:
        return []
    names: list[str] = []
    for name, spec in _schema_widget_slots(schema, linked):
        names.append(name)
        subs = _dynamic_combo_suboptions(spec)
        if subs is not None:
            names.extend(f"{name}.{sub}" for sub in subs.get(str(inputs.get(name)), []))
    return names


def _flatten_subgraphs(workflow: dict) -> dict:
    """Inline ComfyUI subgraph nodes into concrete nodes.

    Newer official templates wrap their whole graph in a *subgraph* node whose
    ``type`` is a UUID declared under ``definitions.subgraphs`` (sometimes nested
    one inside another). ComfyUI's /prompt API rejects such nodes ("Node not
    found"), so before converting graph format to API format we expand every
    subgraph instance: its inner nodes are promoted with fresh ids, inner links
    are rewired, each boundary input is connected to whatever the instance's
    input slot was wired to (or dropped, so the inner node keeps its own widget
    value / stays free for the Brain to wire), and each boundary output is
    redirected to its real inner producer. Nested subgraphs expand recursively.

    Returns a graph-format workflow (``nodes`` + ``links``) with no subgraph
    references. If the workflow declares no subgraphs it is returned unchanged.
    """
    import itertools

    defs = (workflow.get("definitions") or {}).get("subgraphs") or []
    registry = {s.get("id"): s for s in defs if isinstance(s, dict) and s.get("id")}
    if not registry or not any(
        isinstance(n, dict) and n.get("type") in registry
        for n in workflow.get("nodes", [])
    ):
        return workflow

    uid_counter = itertools.count(1)
    lid_counter = itertools.count(1)
    out_nodes: list[dict] = []
    out_links: list[list] = []

    def add_link(src_uid: str, src_slot, dst_uid: str, dst_slot, typ) -> int:
        lid = next(lid_counter)
        out_links.append([lid, src_uid, int(src_slot), dst_uid, int(dst_slot), typ])
        return lid

    def index_links(links):
        """(by_link_id, by_target) for a scope's links (dict or array form)."""
        by_id: dict = {}
        by_target: dict = {}
        for lk in links or []:
            if isinstance(lk, dict):
                lid, o, os_ = lk.get("id"), lk.get("origin_id"), lk.get("origin_slot")
                t, ts, ty = lk.get("target_id"), lk.get("target_slot"), lk.get("type")
            elif isinstance(lk, (list, tuple)) and len(lk) >= 5:
                lid, o, os_, t, ts = lk[0], lk[1], lk[2], lk[3], lk[4]
                ty = lk[5] if len(lk) > 5 else None
            else:
                continue
            by_id[lid] = (o, os_, ty)
            by_target[(t, ts)] = (o, os_, ty)
        return by_id, by_target

    def process(sg_nodes, sg_links, in_bid, out_bid, resolve_input):
        """Emit concrete nodes for one scope; return its resolve_output(slot)."""
        by_id, by_target = index_links(sg_links)
        inner_by_id = {n["id"]: n for n in sg_nodes if isinstance(n, dict) and "id" in n}
        uid_map: dict = {}
        nested_out: dict = {}

        def uid_of(inner_id) -> str:
            if inner_id not in uid_map:
                uid_map[inner_id] = str(next(uid_counter))
            return uid_map[inner_id]

        def get_nested(inst_id):
            if inst_id in nested_out:
                return nested_out[inst_id]
            nsg = registry[inner_by_id[inst_id]["type"]]
            synth_cache: dict = {}   # boundary slot -> synthesized (uid, slot)

            def nested_resolve_input(j):
                src = by_target.get((inst_id, j))
                if src:
                    return resolve_endpoint(src[0], src[1])
                if j in synth_cache:
                    return ("link", synth_cache[j])
                # Unconnected boundary input: for a real media input (no widget
                # value backing it) synthesize a Load node so the Brain has a
                # concrete source to wire — classic templates always shipped one;
                # subgraph templates expose a bare boundary port instead.
                conns = inner_by_id[inst_id].get("inputs", [])
                conn = conns[j] if 0 <= j < len(conns) else {}
                # Any image-bearing boundary type counts (e.g. 'IMAGE,MASK' on a
                # resize/inpaint port) — a synthesized LoadImage satisfies it.
                if "IMAGE" in str(conn.get("type") or "").upper() and "widget" not in conn:
                    luid = str(next(uid_counter))
                    out_nodes.append({
                        "id": luid, "type": "LoadImage", "inputs": [],
                        "widgets_values": [""], "title": "Load Image",
                    })
                    synth_cache[j] = (luid, 0)
                    return ("link", (luid, 0))
                return ("drop",)

            rout = process(
                nsg.get("nodes", []), nsg.get("links", []),
                (nsg.get("inputNode") or {}).get("id", -10),
                (nsg.get("outputNode") or {}).get("id", -20),
                nested_resolve_input,
            )
            nested_out[inst_id] = rout
            return rout

        def resolve_endpoint(origin_id, origin_slot):
            if in_bid is not None and origin_id == in_bid:
                return resolve_input(origin_slot)
            node = inner_by_id.get(origin_id)
            if node is None:
                return ("drop",)
            if node.get("type") in registry:
                return get_nested(origin_id)(origin_slot)
            return ("link", (uid_of(origin_id), origin_slot))

        for n in sg_nodes:
            if not isinstance(n, dict) or "id" not in n:
                continue
            if n.get("type") in registry:
                rout = get_nested(n["id"])   # force-expand even if unconsumed
                # Top-level terminal IMAGE outputs have no external consumer in a
                # subgraph-wrapped template; realize them as SaveImage nodes so the
                # workflow actually writes a result (classic-template invariant).
                if in_bid is None:
                    for slot, oconn in enumerate(n.get("outputs", [])):
                        if not isinstance(oconn, dict) or oconn.get("links"):
                            continue
                        if str(oconn.get("type") or "").upper() != "IMAGE":
                            continue
                        res = rout(slot)
                        if res[0] != "link":
                            continue
                        suid, sslot = res[1]
                        svid = str(next(uid_counter))
                        lid = add_link(suid, sslot, svid, 0, "IMAGE")
                        out_nodes.append({
                            "id": svid, "type": "SaveImage",
                            "inputs": [{"name": "images", "type": "IMAGE", "link": lid}],
                            "widgets_values": ["ComfyUI"], "title": "Save Image",
                        })
                continue
            uid = uid_of(n["id"])
            new_inputs: list = []
            for conn in n.get("inputs", []):
                if not isinstance(conn, dict):
                    continue
                keep = dict(conn)
                keep["link"] = None
                link = conn.get("link")
                if link is not None and link in by_id:
                    o, os_, ty = by_id[link]
                    res = resolve_endpoint(o, os_)
                    if res[0] == "link":
                        suid, sslot = res[1]
                        keep["link"] = add_link(
                            suid, sslot, uid, len(new_inputs), ty or conn.get("type"))
                new_inputs.append(keep)
            out_nodes.append({
                "id": uid,
                "type": n.get("type", "unknown"),
                "inputs": new_inputs,
                "widgets_values": n.get("widgets_values", n.get("widget_values", [])),
                "title": n.get("title", ""),
            })

        out_src = {ts: (o, os_) for (t, ts), (o, os_, _ty) in by_target.items()
                   if out_bid is not None and t == out_bid}

        def resolve_output(slot):
            src = out_src.get(slot)
            return resolve_endpoint(src[0], src[1]) if src else ("drop",)

        return resolve_output

    process(workflow.get("nodes", []), workflow.get("links", []),
            None, None, lambda _slot: ("drop",))

    flat = dict(workflow)
    flat.pop("definitions", None)
    flat["nodes"] = out_nodes
    flat["links"] = out_links
    return flat


def _convert_graph_to_api(workflow: dict) -> dict:
    """Convert a ComfyUI graph-format workflow dict to API format."""
    workflow = _flatten_subgraphs(workflow)
    # Build link lookup: link_id → [str(src_node_id), src_slot]
    link_table: dict[int, list] = {}
    for link in workflow.get("links", []):
        if isinstance(link, (list, tuple)) and len(link) >= 3:
            link_id, src_node, src_slot = int(link[0]), link[1], link[2]
            link_table[link_id] = [str(src_node), int(src_slot)]

    try:
        object_info = _get_object_info()
    except Exception:
        object_info = {}

    api_workflow: dict = {}

    for node in workflow.get("nodes", []):
        if not isinstance(node, dict) or "id" not in node:
            continue

        nid = str(node["id"])
        class_type: str = node.get("type", "unknown")
        api_inputs: dict = {}

        # Map linked inputs
        linked_names: set[str] = set()
        for connector in node.get("inputs", []):
            if not isinstance(connector, dict):
                continue
            name = connector.get("name", "")
            link_id = connector.get("link")
            if name and link_id is not None and link_id in link_table:
                api_inputs[name] = link_table[link_id]
                linked_names.add(name)

        # Map widget values → named inputs.
        # Preferred source of truth: the graph node's OWN widget declarations
        # (inputs[] entries carrying widget.name, in declared order — including
        # V3 dotted names like 'resize_type.width'). Positional mapping from
        # object_info cannot know V3 dynamic-combo/nested widget order and used
        # to shift every value (input←option-key, resize_type←1920, junk
        # __extra_widget_*). A linked input keeps its link; its widgets_values
        # slot is still consumed (the widget value is just the UI fallback).
        widgets_values: list = node.get("widgets_values", node.get("widget_values", []))
        unmapped_widgets: list = []
        if isinstance(widgets_values, list) and widgets_values:
            declared = [(c.get("widget", {}).get("name") or c.get("name", ""), None)
                        for c in node.get("inputs", [])
                        if isinstance(c, dict) and isinstance(c.get("widget"), dict)]
            schema = object_info.get(class_type, {}).get("input", {}) if object_info else {}
            schema_slots = _schema_widget_slots(schema, linked_names) if schema else []
            # Prefer whichever slot list *explains every value*. Declarations are
            # the better source when complete (they carry V3 dotted names the
            # schema order can't reproduce), but a node inlined from a subgraph
            # declares only its promoted widget while still carrying the full
            # value list — trusting that dropped CLIPLoader.type, UNETLoader.
            # weight_dtype, KSampler.sampler_name/scheduler/denoise and so on.
            mapped, leftover = _map_widget_values(declared, widgets_values, linked_names)
            if leftover and schema_slots:
                # The declarations left values unexplained, so fall back to schema
                # order. Measured over the whole corpus this is the rule that
                # resolves the most nodes (against "switch only when the schema
                # names more widgets" and "switch only when it leaves fewer
                # leftovers", both of which scored ~80 templates worse).
                #
                # Wired-up widgets may or may not still hold a value, so try both
                # readings and keep the one whose assignments actually fit their
                # slots. Ties go to the excluding reading — fewer assumptions.
                # Only wired *widgets* keep a slot: a wired socket (MESH, MODEL, a
                # custom type) never had one. Counting the socket too gave
                # UnwrapMesh's "mesh" the segmenter's value and shifted the rest, so
                # this reading lost to the excluding one and shipped padding=2048
                # (clamped to 16) and weld_distance=1 - every vertex welded into
                # one, and ApplyTextureToMesh died on an empty mesh.
                # A wired widget says so on its connector ("widget": {...}), which
                # also covers multi-typed widgets such as Preview3D's model_file.
                all_specs = {**(schema.get("required") or {}), **(schema.get("optional") or {})}
                wired_widgets = {c.get("name") for c in node.get("inputs", [])
                                 if isinstance(c, dict) and isinstance(c.get("widget"), dict)}
                linked_sockets = {n for n in linked_names
                                  if n not in wired_widgets
                                  and not _is_widget_spec(all_specs.get(n))}
                incl_slots = _schema_widget_slots(schema, linked_sockets)
                without_linked = _map_widget_values(schema_slots, widgets_values, linked_names)
                with_linked = _map_widget_values(incl_slots, widgets_values, linked_names)
                incl_score = _mapping_score(with_linked[0], incl_slots)
                excl_score = _mapping_score(without_linked[0], schema_slots)
                # Fit decides. When the two readings fit equally well — every
                # value is *a* number in *a* numeric slot and nothing is out of
                # range — the one that explains the whole list wins. A wired
                # widget still occupies its slot in widgets_values, so a reading
                # that accounts for every value has almost certainly found the
                # right offset, while a leftover means one value had nowhere to
                # go. (Leftovers cannot be the primary rule: the longer slot list
                # always consumes more, so it would win even when every
                # assignment is wrong. As a tie-break, after fit, it is safe —
                # measured over the corpus it drops unexplained values 2035 ->
                # 1207 without adding a single schema violation.)
                mapped, leftover = (
                    with_linked
                    if (incl_score, -len(with_linked[1]))
                    > (excl_score, -len(without_linked[1]))
                    else without_linked)
            if not declared and not schema_slots:
                api_inputs["__widgets_values"] = list(widgets_values)
            else:
                api_inputs.update(mapped)
                # Values no slot claimed are a mapping failure, not node inputs.
                # They used to be POSTed as __extra_widget_N keys; park them in
                # _meta instead so they stay visible for debugging without being
                # sent to /prompt as inputs the node never declared.
                if leftover:
                    unmapped_widgets = leftover

        node_schema = object_info.get(class_type, {}).get("input", {}) if object_info else {}
        if node_schema:
            _fill_required_defaults(api_inputs, node_schema, linked_names, class_type)

        api_node: dict = {"class_type": class_type, "inputs": api_inputs}
        title = node.get("title", "")
        if title or unmapped_widgets:
            api_node["_meta"] = {}
            if title:
                api_node["_meta"]["title"] = title
            if unmapped_widgets:
                api_node["_meta"]["unmapped_widgets"] = unmapped_widgets
        api_workflow[nid] = api_node

    _inline_primitive_nodes(api_workflow)
    return api_workflow


def _inline_primitive_nodes(api_workflow: dict) -> list[str]:
    """Write each legacy ``PrimitiveNode``'s value into the inputs it feeds, and
    drop the node. Returns the ids removed.

    ``PrimitiveNode`` exists only in the frontend: it holds one widget value and
    the frontend copies that value into every input wired to it when it builds
    the API prompt. The server has no such class. Left in, a template's "Song
    Duration = 120" node made the graph invalid, a repair agent replaced it with
    a PrimitiveFloat on every build, and the length the user asked for was never
    applied — the inputs that carry it (``seconds``, ``duration``) held a link,
    not a value, so nothing knew to set them. Inlined, they are ordinary inputs.
    """
    removed: list[str] = []
    for nid, node in list(api_workflow.items()):
        if not isinstance(node, dict) or node.get("class_type") != "PrimitiveNode":
            continue
        values = (node.get("inputs") or {}).get("__widgets_values")
        if not isinstance(values, list) or not values:
            continue                # nothing to inline: leave it for the repair pass to report
        value = values[0]           # [value, control_after_generate]
        for other in api_workflow.values():
            inputs = other.get("inputs") if isinstance(other, dict) else None
            if not isinstance(inputs, dict):
                continue
            for name, link in list(inputs.items()):
                if isinstance(link, list) and len(link) == 2 and str(link[0]) == str(nid):
                    inputs[name] = value
        del api_workflow[nid]
        removed.append(str(nid))
    return removed


def _canvas_node_size(title: str, spec: dict, conns: list, inputs: dict,
                      widget_names: list, outputs: list) -> list:
    """The size the canvas opens this node at, estimated - see graph_groups.estimate_size."""
    from agenty_core.tools import graph_groups as _gg  # noqa: PLC0415
    allspec = {**(spec.get("optional") or {}), **(spec.get("required") or {})}
    slots: list = []
    for name, sp in allspec.items():
        meta = sp[1] if isinstance(sp, list) and len(sp) > 1 and isinstance(sp[1], dict) else {}
        if _is_autogrow_group(sp):
            # The wired slots, and the one empty slot the frontend adds after them.
            ag = _autogrow_info(meta, name)
            wired = sum(1 for key in ag["keys"] if isinstance(inputs.get(key), list))
            slots.extend(ag["slots"][:min(len(ag["slots"]), wired + 1)])
        elif not _is_widget_spec(sp) or meta.get("forceInput"):
            slots.append(name)
    slots.extend(n for n, _t in conns if n not in allspec and "." not in n)
    widgets = list(widget_names) + [n for n in widget_names if n in _SEED_INPUT_NAMES]
    wired = [n for n in widget_names if isinstance(inputs.get(n), list)]
    multiline = sum(1 for n in widget_names
                    if isinstance(allspec.get(n), list) and allspec[n] and allspec[n][0] == "STRING"
                    and len(allspec[n]) > 1 and isinstance(allspec[n][1], dict)
                    and allspec[n][1].get("multiline"))
    return _gg.estimate_size(title, slots, [n for n, _t in outputs], widgets, wired, multiline)


def _api_to_graph(workflow: dict) -> dict:
    """Build a ComfyUI *graph*-format workflow (nodes + links, auto-laid-out) from
    an API/prompt-format workflow, so the exact thing the agent ran can be opened
    on the canvas. Inverse of _convert_graph_to_api: reuses object_info for slot
    order, widget order and output types. Best-effort — unknown node classes still
    render (with no slots) so the graph is always openable."""
    api = workflow.get("prompt", workflow) if isinstance(workflow, dict) else {}
    api = {str(k): v for k, v in api.items()
           if isinstance(v, dict) and v.get("class_type")}
    try:
        oi = _get_object_info()
    except Exception:  # noqa: BLE001
        oi = {}

    def _is_link(v):
        return (isinstance(v, list) and len(v) == 2
                and isinstance(v[0], (str, int)) and not isinstance(v[0], bool)
                and isinstance(v[1], int))

    # Stable int id per (possibly colon-namespaced) api id.
    id_map: dict = {}
    used: set = set()
    for k in api:
        if k.isdigit():
            id_map[k] = int(k); used.add(int(k))
    nxt = 1
    for k in api:
        if k not in id_map:
            while nxt in used:
                nxt += 1
            id_map[k] = nxt; used.add(nxt); nxt += 1

    # Per node: ordered connection inputs, widget names, output (name, type).
    meta: dict = {}
    for k, nd in api.items():
        info = oi.get(nd["class_type"], {})
        spec = info.get("input", {})
        allspec = {**(spec.get("required") or {}), **(spec.get("optional") or {})}
        ins = nd.get("inputs", {})
        conns, seen, grown = [], set(), set()
        for name, sp in allspec.items():
            t = sp[0] if isinstance(sp, list) and sp else None
            if _is_autogrow_group(sp):
                # A group has no socket of its own. Each wired slot is one, named
                # as the prompt addresses it and typed by the group's template -
                # which is how the frontend saves it ('ref_images.ref_image_0',
                # IMAGE, in video_minimax_h3_r2v).
                ag = _autogrow_info(sp[1] if len(sp) > 1 and isinstance(sp[1], dict) else {}, name)
                for key in ag["keys"]:
                    if _is_link(ins.get(key)):
                        conns.append((key, ag["grown_type"])); seen.add(key); grown.add(key)
                continue
            if _is_link(ins.get(name)) or (isinstance(t, str) and t in _LINK_ONLY_TYPES):
                conns.append((name, t if isinstance(t, str) else "*")); seen.add(name)
        for name, v in ins.items():  # linked custom inputs absent from schema
            if _is_link(v) and name not in seen:
                conns.append((name, "*")); seen.add(name)
        # A widget driven by a wire keeps its place in widgets_values - official
        # templates hold "" there for a linked prompt - so only true sockets leave
        # the widget order. Dropping the linked prompt shifted every value of
        # MiniMaxH3ReferenceToVideo one place left: the canvas showed width 768 and
        # height 124, and the agent "repaired" it by turning the prompt into a
        # literal placeholder so the numbers lined up again.
        sockets = {n for n, _t in conns if not _is_widget_spec(allspec.get(n))}
        widgets = (_schema_widget_names(spec, sockets) if spec
                   else [n for n, v in ins.items() if not _is_link(v)])
        outs = info.get("output", []) or []
        onames = info.get("output_name", []) or outs
        meta[k] = {"conns": conns, "widgets": widgets, "spec": spec, "sockets": sockets,
                   "grown": grown, "outputs": list(zip(onames, outs))}

    # Links + slot bookkeeping.
    links: list = []
    lid = 0
    out_links: dict = {}
    in_link: dict = {}
    for k, nd in api.items():
        for i, (name, typ) in enumerate(meta[k]["conns"]):
            v = nd.get("inputs", {}).get(name)
            if not _is_link(v) or str(v[0]) not in id_map:
                continue
            lid += 1
            links.append([lid, id_map[str(v[0])], int(v[1]), id_map[k], i, typ])
            out_links.setdefault((str(v[0]), int(v[1])), []).append(lid)
            in_link[(k, name)] = lid

    # Longest-path depth → column; row within column → y.
    level: dict = {}

    def _depth(k, stack):
        if k in level:
            return level[k]
        if k in stack:
            return 0
        d = 0
        for name, _ in meta[k]["conns"]:
            v = api[k].get("inputs", {}).get(name)
            if _is_link(v) and str(v[0]) in api:
                d = max(d, _depth(str(v[0]), stack | {k}) + 1)
        level[k] = d
        return d

    for k in api:
        _depth(k, set())

    from agenty_core.tools import graph_groups as _gg  # noqa: PLC0415

    # Widget values first: they decide what each node shows, and so its size.
    built: dict = {}
    for k in api:
        m = meta[k]
        ins = api[k].get("inputs", {})
        # Widget order for the canvas node's positional widgets_values.
        # Normally object_info's schema order (m["widgets"]). Dynamic-combo
        # nodes (e.g. OpenAIGPTImageNodeV2, Wan2ImageToVideoApi) also expose
        # dotted sub-widgets (model.prompt, model.size, …) which m["widgets"]
        # omits — it carries only the top-level combo — so using it as-is drops
        # every dotted value and shifts the rest into the wrong slots.
        #
        # The sub-widgets ARE derivable: the selected option's own inputs, in
        # schema order, right after the combo — the same expansion
        # _map_widget_values uses to read them back. Deriving beats trusting
        # the API dict's key order, which only matches the canvas while nothing
        # has rewritten the node: dropping and re-adding an input (as splicing
        # a canvas hook off a widget-backed input does) appends it, and the
        # prompts then land in whichever widgets happen to sit at their index.
        _non_link = [n for n in ins if not _is_link(ins.get(n))]
        if any("." in str(n) for n in _non_link):
            derived = _dynamic_widget_names(m["spec"], ins, m["sockets"])
            # Only trust the schema when it accounts for every value present;
            # otherwise keep the authored order, which is all we have.
            _widget_names = (derived if derived and set(_non_link) <= set(derived)
                             else _non_link)
        else:
            _widget_names = m["widgets"]
        wv: list = []
        _specs = {**(m["spec"].get("optional") or {}), **(m["spec"].get("required") or {})}
        for wname in _widget_names:
            val = ins.get(wname)
            if wname in ins and not _is_link(val):
                wv.append(val)
            elif _is_link(val):
                # The wire decides the value; the slot still wants a
                # placeholder of the widget's own kind.
                wspec = _specs.get(wname)
                wmeta = (wspec[1] if isinstance(wspec, list) and len(wspec) > 1
                         and isinstance(wspec[1], dict) else {})
                is_text = isinstance(wspec, list) and bool(wspec) and wspec[0] == "STRING"
                wv.append(wmeta.get("default", "" if is_text else None))
            else:
                wv.append(None)
            if wname in _SEED_INPUT_NAMES:
                wv.append("fixed")  # frontend's control_after_generate widget
        built[k] = (_widget_names, wv)

    sizes: dict = {}
    for k in api:
        cls = api[k]["class_type"]
        title = ((api[k].get("_meta") or {}).get("title")
                 or (oi.get(cls) or {}).get("display_name") or cls)
        sizes[k] = _canvas_node_size(title, meta[k]["spec"], meta[k]["conns"],
                                     api[k].get("inputs", {}), built[k][0], meta[k]["outputs"])

    # Columns by depth, each node below the one above it at its own height, and -
    # when the graph splits into stages - a band and a box per group. See
    # agenty_core.tools.graph_groups.
    try:
        positions, group_boxes, hint = _gg.place(api, meta, level, oi, sizes)
    except Exception as exc:  # noqa: BLE001 — a graph must open, grouped or not
        _gg.logger.debug("_api_to_graph: could not lay out the graph — %s", exc)
        positions, group_boxes, hint = {}, [], None

    nodes: list = []
    ordered = sorted(api, key=lambda k: (level.get(k, 0), (positions.get(k) or [0, 0])[1], id_map[k]))
    for order, k in enumerate(ordered):
        m = meta[k]
        wv = built[k][1]
        nodes.append({
            "id": id_map[k], "type": api[k]["class_type"],
            "pos": positions.get(k) or [_gg.X0 + level.get(k, 0) * 420, _gg.Y0 + order * 60],
            "size": sizes[k],
            "flags": {}, "order": order, "mode": 0,
            # An autogrow slot carries its short label and the optional-socket
            # shape, as the frontend saves it; without the label the canvas shows
            # 'ref_images.ref_image_0' beside its own 'ref_image_1'.
            "inputs": [{"name": n, "type": t, "link": in_link.get((k, n)),
                        **({"label": n.rsplit(".", 1)[-1], "shape": 7} if n in m["grown"] else {}),
                        **({} if n in m["sockets"] else {"widget": {"name": n}})}
                       for n, t in m["conns"]],
            "outputs": [{"name": (n or t), "type": t, "slot_index": i,
                         "links": out_links.get((k, i)) or None}
                        for i, (n, t) in enumerate(m["outputs"])],
            "properties": {"Node name for S&R": api[k]["class_type"]},
            "widgets_values": wv,
            "title": (api[k].get("_meta") or {}).get("title"),
        })

    extra: dict = {}
    if hint:
        # The arrangement without the pixels, keyed by canvas node id, so the
        # sidebar extension can lay the graph out again with the sizes the canvas
        # itself measures.
        extra["agentY_layout"] = {
            **hint,
            "slots": {str(id_map[k]): v for k, v in hint["slots"].items()},
            "groups": [[id_map[k] for k in members] for members in hint["groups"]],
        }
    return {
        "last_node_id": max((n["id"] for n in nodes), default=0),
        "last_link_id": lid, "nodes": nodes, "links": links,
        "groups": group_boxes, "config": {}, "extra": extra, "version": 0.4,
    }


_MODEL_FILE_EXTS = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx",
)


def _resolve_model_names(api_workflow: dict) -> None:
    """Snap each node's model-file input to the exact filename ComfyUI expects.

    Template widget values drift from what the live server accepts: a bare
    ``ae.safetensors`` or a forward-slash ``FLUX1/ae.safetensors`` will not
    match the object_info combo entry ``FLUX1\\ae.safetensors`` on Windows, and
    the Brain then wrongly reports the model as missing. This rewrites such a
    value (in place) to the exact object_info string when the current value is
    not already valid but a case-insensitive basename match exists. Genuinely
    absent models are left untouched so they still surface as real blockers.
    """
    try:
        object_info = _get_object_info()
    except Exception:
        object_info = {}
    if not object_info:
        return

    def _base(s: str) -> str:
        return Path(str(s).replace("\\", "/")).name.lower()

    for node in api_workflow.values():
        if not isinstance(node, dict):
            continue
        schema = object_info.get(node.get("class_type"), {}).get("input", {})
        if not schema:
            continue
        specs = {**schema.get("required", {}), **schema.get("optional", {})}
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for iname, ival in list(inputs.items()):
            if not isinstance(ival, str) or not ival.lower().endswith(_MODEL_FILE_EXTS):
                continue
            spec = specs.get(iname)
            options = (spec[0] if isinstance(spec, (list, tuple)) and spec
                       and isinstance(spec[0], list) else None)
            if not options or ival in options:
                continue
            target = _base(ival)
            match = next((o for o in options if _base(o) == target), None)
            if match is None:
                # Precision-variant fallback: a template may reference a variant
                # that is not installed (e.g. ltx-2.3-22b-dev-fp8) while the plain
                # build is (ltx-2.3-22b-dev). Snap when exactly one option shares
                # the precision-reduced stem.
                q = _reduce_stem(target.rsplit(".", 1)[0])
                cands = {o for o in options
                         if _reduce_stem(_base(o).rsplit(".", 1)[0]) == q}
                if len(cands) == 1:
                    match = next(iter(cands))
            if match:
                inputs[iname] = match


def _strip_history(data: dict | list) -> dict | list:
    """Strip embedded workflow/prompt JSON from history entries to save tokens."""
    if isinstance(data, list):
        return [_strip_history(item) for item in data]
    if not isinstance(data, dict):
        return data

    stripped: dict = {}
    for prompt_id, entry in data.items():
        if not isinstance(entry, dict):
            stripped[prompt_id] = entry
            continue
        slim: dict = {}
        if "status" in entry:
            slim["status"] = entry["status"]
        if "outputs" in entry:
            outputs: dict = {}
            for node_id, node_out in entry.get("outputs", {}).items():
                if isinstance(node_out, dict):
                    slim_out: dict = {}
                    for key, val in node_out.items():
                        if isinstance(val, list):
                            slim_out[key] = [
                                {k: v for k, v in item.items() if k != "abs_path"}
                                if isinstance(item, dict) else item
                                for item in val
                            ]
                        else:
                            slim_out[key] = val
                    outputs[node_id] = slim_out
            slim["outputs"] = outputs
        stripped[prompt_id] = slim
    return stripped


def _autogrow_info(meta: dict, name: str = "") -> dict:
    """Parse a ``COMFY_AUTOGROW_V3`` metadata dict (2nd element of the input spec).

    Pass the group's own *name* to get *keys* as the prompt addresses them
    (``ref_images.ref_image_0``); *slots* always holds the bare slot names.

    V3 dynamic *autogrow* inputs (image batchers, multi-reference encoders, …)
    never appear under their umbrella name (e.g. ``images``): ComfyUI grows them
    into per-slot keys. Returns ``{grown_type, keys, min, max}`` where *keys* are
    the concrete per-slot input names — a 0-based ``prefix`` sequence
    (``image0, image1, …``, per ComfyUI source ``[f"{prefix}{i}" for i in
    range(max)]``) or an explicit ``names`` list (``image_1, …`` / ``a, b, …``).
    """
    tmpl = (meta or {}).get("template", {}) or {}
    grown_spec = next(iter((tmpl.get("input", {}).get("required", {}) or {}).values()), None)
    if isinstance(grown_spec, list) and grown_spec:
        grown_type = "COMBO" if isinstance(grown_spec[0], list) else str(grown_spec[0])
    else:
        grown_type = "*"
    if isinstance(tmpl.get("names"), list):
        keys = [str(k) for k in tmpl["names"]]
    else:
        prefix = str(tmpl.get("prefix", ""))
        try:
            mx = int(tmpl.get("max", 0))
        except (TypeError, ValueError):
            mx = 0
        keys = [f"{prefix}{i}" for i in range(mx)]  # 0-based, per ComfyUI source
    try:
        mn = int(tmpl.get("min", 0) or 0)
    except (TypeError, ValueError):
        mn = 0
    slots = keys
    if name:
        # ComfyUI binds a slot through its group - finalize_prefix joins the two
        # with a dot. The bare slot name is what this used to hand out, and a wire
        # written under it reaches the node as an argument it does not take.
        keys = [f"{name}.{k}" for k in keys]
    return {"grown_type": grown_type, "keys": keys, "slots": slots, "min": mn,
            "max": tmpl.get("max")}


def _is_autogrow_spec(spec) -> bool:
    """True when a raw object_info input spec is a V3 autogrow (``COMFY_AUTOGROW_V3``)."""
    return (isinstance(spec, list) and spec and isinstance(spec[0], str)
            and "AUTOGROW" in spec[0].upper())


def _parse_autogrow(name: str, opts: dict) -> dict:
    """Render a V3 autogrow input legibly so the agent can actually wire it."""
    ag = _autogrow_info(opts, name)
    keys, gtype = ag["keys"], ag["grown_type"]
    sample = keys[:6] + (["…"] if len(keys) > 6 else [])
    hint = ", ".join(keys[:2]) if keys else f"{name}.image0"
    bare = ag["slots"][0] if ag["slots"] else "image0"
    entry: dict = {
        "type": gtype,
        "dynamic": True,
        "connect_as": sample,
        "note": (
            f"Dynamic autogrow input: wire one {gtype} producer per slot, keyed by the "
            f"slot's full address ({hint}, …) in this node's 'inputs'. Neither "
            f"'{name}' alone nor the bare slot name '{bare}' works: ComfyUI hands a "
            f"wire under an undeclared key to the node, which fails at run time."
        ),
    }
    if ag.get("min") is not None:
        entry["min"] = ag["min"]
    if ag.get("max") is not None:
        entry["max"] = ag["max"]
    if isinstance(opts, dict) and "tooltip" in opts:
        entry["tooltip"] = opts["tooltip"]
    return entry


def _parse_inputs_schema(spec: dict) -> dict:
    """Turn ComfyUI's input spec into a friendlier format."""
    result = {}
    for name, definition in spec.items():
        entry: dict = {}
        if isinstance(definition, list) and len(definition) >= 1:
            type_info = definition[0]
            opts = definition[1] if len(definition) > 1 else {}
            if isinstance(type_info, str) and "AUTOGROW" in type_info.upper():
                # V3 dynamic autogrow input — surface the grown type + slot keys
                # instead of the opaque COMFY_AUTOGROW_V3 sentinel.
                result[name] = _parse_autogrow(name, opts if isinstance(opts, dict) else {})
                continue
            if isinstance(type_info, list):
                entry["type"] = "COMBO"
                entry["options"] = type_info
            else:
                entry["type"] = type_info
            if isinstance(opts, dict):
                for key in ("default", "min", "max", "step", "tooltip"):
                    if key in opts:
                        entry[key] = opts[key]
        else:
            entry["type"] = str(definition)
        result[name] = entry
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Models
# ═══════════════════════════════════════════════════════════════════════════════

# Precision / quantisation / format suffix tokens that distinguish variants of
# the SAME model (not different models). Stripped from a trailing position when
# fuzzily matching a generic name to an installed file.
_PRECISION_TOKENS: frozenset[str] = frozenset({
    "fp8", "fp16", "fp32", "fp4", "bf16", "nf4", "int4", "int8", "fp8mixed",
    "fp8scaled", "e4m3fn", "e5m2", "scaled", "mixed", "pruned", "ema", "gguf",
    "safetensors", "q2", "q3", "q4", "q5", "q6", "q8", "k", "s", "m", "l",
    "ks", "km", "kl",
})


def _reduce_stem(stem: str) -> str:
    """Drop trailing precision/format tokens so variants of one model collapse
    to a common key: ``qwen_image_fp8_e4m3fn`` -> ``qwen_image``. Meaningful
    variant words (``edit``, ``turbo``, ``vae``, version numbers) are kept."""
    parts = re.split(r"[_\-.]", stem.lower())
    while len(parts) > 1 and parts[-1] in _PRECISION_TOKENS:
        parts.pop()
    return "_".join(parts)


def _fuzzy_model_match(query_key: str, basename_index: dict[str, str]) -> str | None:
    """Resolve a generic filename to an installed one when unambiguous.

    Used only when an exact basename lookup fails. A candidate qualifies when
    its precision-reduced stem equals the query's reduced stem (so
    ``qwen_image.safetensors`` finds ``qwen_image_fp8_e4m3fn.safetensors`` but
    never the VAE or an ``_edit_`` variant). Returns the path only if exactly
    one distinct file qualifies; otherwise ``None`` (never guesses)."""
    q = _reduce_stem(query_key.rsplit(".", 1)[0])
    if not q:
        return None
    hits = {path for bn, path in basename_index.items()
            if _reduce_stem(bn.rsplit(".", 1)[0]) == q}
    return next(iter(hits)) if len(hits) == 1 else None


def _model_inventory_index() -> dict[str, str]:
    """lowercase file name -> the path ComfyUI lists it under, for every model
    in the cached inventory (config/models.json)."""
    models_path = _project_root() / "config" / "models.json"
    available: dict = {}
    if models_path.exists():
        raw = "".join(
            ln for ln in models_path.read_text(encoding="utf-8").splitlines(keepends=True)
            if not ln.lstrip().startswith("//")
        )
        data = json.loads(raw) if raw.strip() else {}
        available = data.get("available", {})
    index: dict[str, str] = {}
    for folder_entries in available.values():
        if not isinstance(folder_entries, list):
            continue
        for entry in folder_entries:
            index[Path(entry).name.lower()] = entry
    return index


def installed_model(name: str) -> str | None:
    """The inventory path of a model file with exactly this file name, or None."""
    try:
        return _model_inventory_index().get(Path(str(name)).name.lower())
    except Exception:  # noqa: BLE001
        return None


def node_managed_option(name: str) -> dict | None:
    """A node that offers this file name as one of its own menu options.

    Preprocessor and detector packs keep their weights in their own folders and
    fetch them on first use: ``DepthAnythingV2Preprocessor.ckpt_name`` lists
    ``depth_anything_v2_vitl.pth``, which is in no ComfyUI model folder and so
    not in the inventory. Asked about it, check_model said "False"; the agent
    took that for a missing model and went to download a copy from an unrelated
    repository. A name a node offers is a value that node accepts — nothing to
    install. Returns ``{"node_class", "input", "value", "offered_by"}``: the first
    node found, and every ``Class.input`` that offers the name.
    """
    want = Path(str(name)).name.lower()
    if not want:
        return None
    try:
        object_info = _get_object_info()
    except Exception:  # noqa: BLE001
        return None
    found: list[tuple] = []
    for cls, info in (object_info or {}).items():
        spec_groups = (info or {}).get("input") or {}
        for group in ("required", "optional"):
            for inp, spec in (spec_groups.get(group) or {}).items():
                options = None
                if isinstance(spec, list) and spec:
                    if isinstance(spec[0], list):
                        options = spec[0]
                    elif spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                        options = spec[1].get("options")
                for opt in options or ():
                    if isinstance(opt, str) and Path(opt).name.lower() == want:
                        found.append((cls, inp, opt))
                        break
    if not found:
        return None
    cls, inp, opt = found[0]
    return {"node_class": cls, "input": inp, "value": opt,
            "offered_by": [f"{c}.{i}" for c, i, _ in found]}


@tool
def check_model(model_names: list | str) -> str:
    """Check whether model files exist in the current ComfyUI installation.

    Searches the cached model inventory in config/models.json (refreshed at
    startup) across ALL model folders (checkpoints, loras, vae, unet, clip,
    etc.) using a permissive, case-insensitive filename match.

    The search is by **exact filename** (case-insensitive), ignoring the
    subfolder.  This means you can pass just the bare filename and the tool
    will find it even if it lives in an unexpected subfolder.

    Args:
        model_names: List of model filenames to look up, e.g.
            ["flux1-dev-fp8.safetensors", "detail_tweaker_xl.safetensors"].
            You may include or omit the subfolder prefix — only the filename
            part is matched.

    Returns a JSON object mapping each queried name to either:
    - The full relative path as it appears in ComfyUI (e.g.
      ``"FLUX1/flux1-dev-fp8.safetensors"``), ready to drop into a
      ``Load Checkpoint`` or ``Load LoRA`` node.
    - The string ``"False"`` when the model is not found in the inventory.

    A name that is in no model folder but IS one of a node's own menu options
    (a preprocessor's or detector's weights, which that node keeps and fetches
    itself) comes back as that option value, with a line under ``"_notes"``
    saying which node offers it. It needs no download.

    Example output::

        {
            "flux1-dev-fp8.safetensors": "FLUX1/flux1-dev-fp8.safetensors",
            "missing_model.safetensors": "False"
        }

    If models.json is missing or has no ``available`` key, all entries will
    return ``"False"``.
    """
    try:
        # Flat lookup: lowercase_basename -> full_relative_path
        basename_index = _model_inventory_index()

        result: dict = {}
        notes: dict[str, str] = {}
        for name in _as_list(model_names):
            name = str(name)
            key = Path(name).name.lower()
            hit = basename_index.get(key)
            if hit is None:
                hit = _fuzzy_model_match(key, basename_index)
            if not hit:
                managed = node_managed_option(name)
                if managed:
                    hit = managed["value"]
                    notes[name] = (f"not in a ComfyUI model folder: it is an option the node offers itself "
                                   f"({', '.join(managed['offered_by'][:6])}). Such a node keeps and "
                                   f"fetches its weights itself — use the value as is, do not download it.")
            result[name] = hit if hit else "False"
        if notes:
            result["_notes"] = notes

        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Execution control
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def interrupt_execution() -> str:
    """Immediately stop the currently running ComfyUI workflow execution."""
    try:
        return json.dumps(get_client().post("/interrupt", json_data={}))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def free_memory(unload_models: bool = True, free_memory_flag: bool = True) -> str:
    """Free GPU/system memory in ComfyUI by unloading models and clearing caches.

    Args:
        unload_models: Unload all loaded models from VRAM (default True).
        free_memory_flag: Free cached memory (default True).
    """
    try:
        payload = {
            "unload_models": unload_models,
            "free_memory": free_memory_flag,
        }
        return json.dumps(get_client().post("/free", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Queue
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def queue(action: str = "status") -> str:
    """Get or manage the ComfyUI execution queue.

    Args:
        action: 'status' (view queue), 'clear' (clear pending), or 'clear_running' (stop running items).
    """
    try:
        if action == "status":
            return json.dumps(get_client().get("/queue"))
        elif action in ("clear", "clear_running"):
            payload = {"clear": True} if action == "clear" else {"clear_running": True}
            return json.dumps(get_client().post("/queue", json_data=payload))
        else:
            return json.dumps({"error": f"Unknown action '{action}'. Use 'status', 'clear', or 'clear_running'"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: History
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_history(max_items: int = 3) -> str:
    """Get recent ComfyUI execution history (status and output filenames only).

    Args:
        max_items: Max entries to return (default 3; 0 = all).
    """
    try:
        params = {}
        if max_items > 0:
            params["max_items"] = max_items
        raw = get_client().get("/history", params=params or None)
        return json.dumps(_strip_history(raw))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_prompt_status_by_id(prompt_id: str) -> str:
    """Check execution status for a specific prompt ID. Returns status and output filenames only.

    Args:
        prompt_id: Prompt ID returned by submit_prompt.
    """
    try:
        raw = get_client().get(f"/history/{prompt_id}")
        return json.dumps(_strip_history(raw))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def clear_history(prompt_id: str = "") -> str:
    """Clear ComfyUI execution history. If prompt_id given, deletes that entry only.

    Args:
        prompt_id: Optional specific prompt ID to delete. If empty, clears all history.
    """
    try:
        if prompt_id:
            payload = {"delete": [prompt_id]}
        else:
            payload = {"clear": True}
        return json.dumps(get_client().post("/history", json_data=payload))
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_system_stats() -> str:
    """Return ComfyUI system info: GPU, VRAM, Python version, PyTorch version, OS."""
    try:
        return json.dumps(get_client().get("/system_stats"))
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def _dirs_from_comfyui() -> dict:
    """ComfyUI's own directories, via the agentY ComfyUI extension.

    ``/agent/comfy_dirs`` is served by agentY-comfyuiConnect from inside the
    ComfyUI process, where ``folder_paths`` can simply be asked. That is the only
    place the question can be answered: a directory ComfyUI resolved at startup,
    from flags this process never saw and defaults relative to a root this process
    cannot see, is not something an outsider can compute.

    Returns {} when the extension is absent or too old - an install without the
    sidebar is a supported one, and argv still answers whenever the flags were
    passed explicitly.
    """
    try:
        data = get_client().get("/agent/comfy_dirs")
    except Exception:  # noqa: BLE001 - no extension, older build, server down
        return {}
    if not isinstance(data, dict) or not data.get("ok"):
        return {}
    out = {}
    for key in ("input_dir", "output_dir", "user_dir"):
        value = str(data.get(key) or "").strip()
        if value:
            out[key] = value
    return out


# Model kinds get_comfyui_dirs reports. /internal/folder_paths lists some sixty
# categories, most of them custom-node folders (fonts, ultralytics, sam, …) that
# nobody downloads into by hand; all of them would be ~6000 characters per call.
_MODEL_DIR_CATEGORIES = (
    "checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "controlnet",
    "clip_vision", "upscale_models", "latent_upscale_models", "embeddings",
    "model_patches", "audio_encoders", "style_models",
)


def _model_dirs() -> dict:
    """``{category: [folder, ...]}`` ComfyUI loads each model kind from, or ``{}``.

    From the running server, so it includes extra model paths configured at startup
    — which on this kind of setup is where the models actually are. An agent asked
    to fetch a model once spent a whole turn hunting for that config file on disk.
    """
    try:
        raw = get_client().get("/internal/folder_paths")
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for category in _MODEL_DIR_CATEGORIES:
        paths = raw.get(category)
        if isinstance(paths, list) and paths:
            out[category] = [str(path) for path in paths]
    return out


def get_comfyui_dirs() -> str:
    """Return the authoritative ComfyUI server directory paths.

    Queries ``/system_stats`` and extracts the ``--input-directory``,
    ``--output-directory``, and ``--user-directory`` flags from the server's
    argv.  Falls back to ComfyUI's compiled-in defaults when a flag is absent.

    Use this tool to resolve:
    - Where uploaded images land after ``upload_image()`` (input_dir).
    - Where ComfyUI will save generated outputs (output_dir) — use this path
      when populating ``output_nodes[].output_path`` in the brainbriefing.
    - Where workflow JSON files are stored (user_dir/workflows/).
    - Where each kind of model is loaded from (model_dirs) — including extra
      model paths configured at startup, which may be on other drives.

    Returns a JSON object with keys ``input_dir``, ``output_dir``, ``user_dir``,
    ``source`` ("argv" when resolved from server flags, "default" otherwise), and
    ``model_dirs`` — ``{category: [absolute folder, ...]}`` for the model kinds
    (checkpoints, diffusion_models, text_encoders, vae, loras, …): every folder
    ComfyUI loads that kind from, including extra model paths on other drives.

    Results are cached for the lifetime of the session (``clear_tool_caches()``
    resets the cache at the start of every new pipeline session).
    """
    global _tool_dirs_result
    with _tool_cache_lock:
        if _tool_dirs_result is not None:
            return _tool_dirs_result
    try:
        stats = get_client().get("/system_stats")
        if not isinstance(stats, dict):
            return json.dumps({"error": "Unexpected /system_stats response format"})

        argv: list = stats.get("system", {}).get("argv", [])
        result: dict = {"source": "argv"}

        for key, flag in (
            ("input_dir", "--input-directory"),
            ("output_dir", "--output-directory"),
            ("user_dir", "--user-directory"),
        ):
            val = parse_argv_dir_flag(argv, flag)
            if val:
                result[key] = val

        # Fill in whatever argv did not name. Two sources, in order of how much
        # they actually know.
        if "input_dir" not in result or "output_dir" not in result or "user_dir" not in result:
            result["source"] = "partial_argv_with_defaults"

            # 1. Ask ComfyUI. folder_paths is the only thing that knows for
            #    certain: it accounts for the flags when they are given and for
            #    the compiled-in defaults when they are not, and it answers with
            #    absolute paths either way.
            asked = _dirs_from_comfyui()
            for key in ("input_dir", "output_dir", "user_dir"):
                if key not in result and asked.get(key):
                    result[key] = asked[key]
                    result["source"] = "argv+comfyui"

            # 2. Failing that, infer from argv[0] - but ONLY when it is absolute.
            #
            #    `python main.py` puts "./main.py" there, and resolving a relative
            #    path uses the CALLER's working directory. The caller is agentY,
            #    which lives somewhere else entirely, so the old
            #    `Path(argv[0]).parent.resolve()` did not fail or return nothing:
            #    it returned <agentY>/user, absolute and confident and wrong.
            #    Project memory was then written there while the ComfyUI nodes
            #    read <ComfyUI>/user, and the load-item node said "nothing stored
            #    yet" about a store that existed and had things in it.
            #
            #    "unknown" is the honest answer when nobody can say, and every
            #    caller already handles it. A wrong path is the one outcome that
            #    cannot be recovered from downstream.
            comfy_root: str | None = None
            if argv and argv[0]:
                first = Path(str(argv[0]))
                if first.is_absolute():
                    comfy_root = str(first.parent)

            if "input_dir" not in result:
                result["input_dir"] = str(Path(comfy_root) / "input") if comfy_root else "unknown"
            if "output_dir" not in result:
                result["output_dir"] = str(Path(comfy_root) / "output") if comfy_root else "unknown"
            if "user_dir" not in result:
                result["user_dir"] = str(Path(comfy_root) / "user") if comfy_root else "unknown"

        models = _model_dirs()
        if models:
            result["model_dirs"] = models

        result_json = json.dumps(result)
        with _tool_cache_lock:
            _tool_dirs_result = result_json
        return result_json
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def get_agent_output_dirs() -> str:
    """Return the canonical, absolute folders the agent must write into.

    Use this BEFORE writing any file yourself — especially when you produce media
    with a Python script (``run_script`` / ``write_text_file``) instead of a
    ComfyUI workflow.  All agent-generated media must land in the same buckets a
    ComfyUI workflow would use, so everything is in one predictable place.

    Returns a JSON object with absolute paths (folders are created if missing):
      - ``images``  — ``<comfyui_output>/agent/images``  (save any image here)
      - ``videos``  — ``<comfyui_output>/agent/videos``  (save any video here)
      - ``audio``   — ``<comfyui_output>/agent/audio``
      - ``models``  — ``<comfyui_output>/agent/models``  (3D assets)
      - ``scripts`` — ``<agent_repo>/output/scripts``     (write scripts here; git-ignored)
      - ``proposals`` — ``<agent_repo>/output/proposals`` (proposed code changes for human review)
      - ``comfyui_output`` — the resolved ComfyUI ``--output-directory``.

    Rule of thumb: images → ``images``, videos → ``videos``, the script that made
    them → ``scripts``.  Never scatter files in the ComfyUI output root or the CWD.
    """
    result: dict[str, str] = {}
    # Resolve ComfyUI's output dir via the same authoritative source as
    # get_comfyui_dirs (argv from /system_stats, with defaults).
    comfy_out: Path | None = None
    try:
        dirs = json.loads(get_comfyui_dirs())
        od = dirs.get("output_dir") if isinstance(dirs, dict) else None
        if od and od != "unknown":
            comfy_out = Path(od)
    except Exception:  # noqa: BLE001
        comfy_out = None

    if comfy_out is not None:
        agent_root = comfy_out / _AGENT_SUBFOLDER
        for bucket in _MEDIA_BUCKETS:
            d = agent_root / bucket
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001
                pass
            result[bucket] = str(d)
        result["comfyui_output"] = str(comfy_out)
    else:
        result["warning"] = (
            "Could not resolve ComfyUI --output-directory; media buckets omitted. "
            "Call get_comfyui_dirs() and write under <output_dir>/agent/<kind>."
        )

    # Scripts + code-change proposals live in the consuming app's own repo
    # (git-ignored ``output/`` tree), not in ComfyUI's output dir.
    try:
        from agenty_core.paths import project_root as _project_root
        out_root = _project_root() / "output"
        scripts_dir = out_root / "scripts"
        proposals_dir = out_root / "proposals"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        proposals_dir.mkdir(parents=True, exist_ok=True)
        result["scripts"] = str(scripts_dir)
        result["proposals"] = str(proposals_dir)
    except Exception as exc:  # noqa: BLE001
        result["scripts_error"] = str(exc)

    return json.dumps(result)


@tool
def get_logs(keyword: str = "", max_lines: int = 100) -> str:
    """Get the latest ComfyUI error / exception block from the runtime log.

    Scans /internal/logs/raw for lines matching known error markers
    (Error, FAILED, Cannot import, Exception, Failed to initialize,
    Error handling request) and returns ONLY the most recent error event
    with ±5 lines of context.  Adjacent error-marker lines (within 20 lines
    of each other) are treated as a single event so a multi-frame
    traceback comes back as one block.

    Interruptions/cancellations (``ProcessingInterrupted``/``Task cancelled``,
    ``Global interrupt``, ``execution_interrupted``) are NOT errors — they are a
    deliberate stop of the queue, so they are skipped here and the most recent
    *real* error is reported instead. This stops a benign stop from being read as
    a fault the agent then tries to repair.

    Returns the literal string "None" when no (real) errors are found.

    Args:
        keyword: Optional further filter — restrict to errors whose marker
                 line also contains this string (case-insensitive).
        max_lines: Max lines to return (default 100).
    """
    try:
        import re as _re

        raw = get_client().get("/internal/logs/raw")
        entries = raw.get("entries", []) if isinstance(raw, dict) else []

        _ANSI = _re.compile(r"\x1b\[[0-9;]*m")
        _TIME = _re.compile(r"(\d{2}:\d{2}:\d{2})")

        def _fmt_ts(t) -> str:
            s = str(t or "")
            m = _TIME.search(s)
            return f"[{m.group(1)}]" if m else (f"[{s}]" if s else "")

        # Flatten entries → list of (timestamp_prefix, ansi_stripped_line)
        items: list[tuple[str, str]] = []
        for entry in entries:
            ts = _fmt_ts(entry.get("t", ""))
            for sub in str(entry.get("m", "")).splitlines():
                if sub:
                    items.append((ts, _ANSI.sub("", sub)))

        # Matched case-INSENSITIVELY, which is the whole point: ComfyUI writes its
        # level as "[ERROR]", and "Error" is not a substring of "ERROR". Every
        # validation failure ComfyUI logs — "[ERROR] Failed to validate prompt for
        # output 4", "[ERROR]   - Value 16 bigger than max of 10: bit_depth" —
        # was therefore invisible here, and because *something* has to be
        # returned, this fell back to the newest line that did happen to contain
        # "Error" and served a four-hour-old AttributeError as the current
        # failure. The agent believed it and spent two turns repairing a bug that
        # had been fixed that morning.
        error_markers = (
            "error", "failed", "cannot import", "exception",
            "traceback", "output will be ignored",
        )
        matches = [
            i for i, (_, line) in enumerate(items)
            if any(m in line.lower() for m in error_markers)
        ]

        if keyword:
            kw = keyword.lower()
            matches = [i for i in matches if kw in items[i][1].lower()]

        if not matches:
            return "None"

        # Cluster matches into discrete error events (runs within 20 lines).
        CLUSTER_GAP = 20
        events: list[list[int]] = []
        for idx in matches:  # ascending
            if events and idx - events[-1][-1] <= CLUSTER_GAP:
                events[-1].append(idx)
            else:
                events.append([idx])

        # A cancellation / interruption is NOT a workflow error. Stopping the queue
        # while a node is running logs "ProcessingInterrupted: Task cancelled" (API
        # nodes) / "Global interrupt" / "!!! Exception during processing !!! Task
        # cancelled", which trips the generic Error/Exception markers above. Left
        # as-is, get_logs returns that as "the most recent error" and the agent
        # tries to REPAIR a workflow that was merely stopped. Skip interruption
        # events and report the most recent REAL error instead (or "None").
        _INTERRUPT_SIGNS = (
            "processinginterrupted", "interruptprocessingexception",
            "task cancelled", "task canceled", "global interrupt",
            "execution_interrupted", "keyboardinterrupt",
        )
        _EXC_RE = _re.compile(r"([A-Za-z_][\w.]*(?:Error|Exception|Interrupted))\s*:")
        # ComfyUI's prompt rejection, in its own words (execution.validate_prompt).
        # Plain text with no exception class anywhere in it.
        _VALIDATION_SIGNS = (
            "failed to validate prompt", "output will be ignored",
            "invalid prompt", "required input is missing",
            "bigger than max of", "smaller than min of", "value not in list",
        )

        def _block_lines(ev: list[int]) -> list[str]:
            first = max(0, ev[0] - 5)
            last = min(len(items) - 1, ev[-1] + 5)
            sel = list(range(first, last + 1))
            if len(sel) > max_lines:
                sel = sel[-max_lines:]
            return [
                f"{items[i][0]} {items[i][1]}".strip() if items[i][0] else items[i][1]
                for i in sel
            ]

        def _is_benign_interrupt(block: list[str]) -> bool:
            """A block is a benign interruption only if it carries an interrupt sign
            AND names no genuine (non-interrupt) exception class — so a real error
            that merely landed in the same 20-line cluster still surfaces."""
            joined = " ".join(block).lower()
            if not any(s in joined for s in _INTERRUPT_SIGNS):
                return False
            # A rejected prompt has no exception class to find — ComfyUI logs it
            # as plain text — so the class scan below cannot see it, and a
            # validation failure that happened to land near an interrupt was
            # being thrown away as benign. It is the opposite of benign: the
            # branch never ran, and the run reports success regardless.
            if any(s in joined for s in _VALIDATION_SIGNS):
                return False
            for line in block:
                for m in _EXC_RE.finditer(line):
                    cls = m.group(1).lower()
                    if "interrupt" in cls or "cancel" in line.lower():
                        continue  # the interrupt exception itself
                    return False  # a genuine exception (KeyError, RuntimeError, …)
            return True

        # Where the most recent submission starts. An error BEFORE it belongs to
        # an earlier run: the log keeps hours of history, so "the latest error"
        # and "the error from the run you just did" are different things, and
        # reading one as the other is how a fixed bug gets repaired twice.
        last_submit = max(
            (i for i, (_, line) in enumerate(items) if "got prompt" in line.lower()),
            default=-1,
        )

        for ev in reversed(events):  # newest → oldest
            block = _block_lines(ev)
            if _is_benign_interrupt(block):
                continue  # benign stop — not an error; keep looking for a real one
            out: dict = {"lines": block, "count": len(block)}
            when = items[ev[-1]][0].strip("[]")
            if when:
                out["at"] = when
            if last_submit >= 0 and ev[-1] < last_submit:
                out["from_earlier_run"] = True
                out["note"] = (
                    f"This error is from BEFORE the most recent submission"
                    f"{' at ' + items[last_submit][0].strip('[]') if items[last_submit][0] else ''}"
                    " — the run you are asking about logged no error at all. Do "
                    "not repair against it. A run that reports success but "
                    "produces no output usually had a whole output branch "
                    "REJECTED at submission: check the node_errors on the submit "
                    "response, or validate_workflow."
                )
            return json.dumps(out)

        # Every recent error-marker event was an interruption — no real error.
        return "None"
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Prompt submission
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def submit_prompt(workflow_path: str, client_id: str = "") -> str:
    """Submit a workflow to the ComfyUI execution queue. Returns prompt_id and client_id on success.

    A WebSocket-compatible client_id is auto-generated if one is not supplied,
    so progress events can be streamed for this prompt.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template or save_workflow).
        client_id: Optional client identifier for tracking; auto-generated when empty.
    """
    try:
        p = Path(workflow_path)
        if p.exists() and p.suffix == ".json":
            workflow = json.loads(p.read_text(encoding="utf-8"))
        else:
            # Legacy fallback: accept inline JSON string
            workflow = json.loads(workflow_path)

        client = get_client()
        if not client_id:
            client_id = uuid.uuid4().hex
        payload: dict = {"prompt": workflow, "client_id": client_id}
        # Forward the ComfyUI API key so API/partner nodes receive it.
        if client.api_key:
            payload["extra_data"] = {"api_key_comfy_org": client.api_key}
        result = client.post("/prompt", json_data=payload)
        if isinstance(result, dict):
            # Note it as ours, so a Stop can delete this prompt from the queue
            # without touching work the user queued themselves.
            try:
                from agenty_core.queue_ledger import remember as _remember_prompt
                _remember_prompt(result.get("prompt_id", ""))
            except Exception:  # noqa: BLE001
                pass
            # Echo the client_id back so the caller (and the interrupt hook) can
            # use it to subscribe to the matching WebSocket stream.
            result.setdefault("client_id", client_id)
        return json.dumps(result)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON in workflow: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def duplicate_workflow(source_path: str) -> str:
    """Create a copy of an existing workflow JSON with a fresh random seed.

    Use this during **batch runs** (``count_iter > 1``) to produce iteration
    copies from the base validated workflow.  Each duplicate gets a new random
    seed injected into every KSampler / RandomNoise / similar seed field so
    outputs are genuinely varied.

    Typical batch flow:
    1. Assemble + validate the base workflow normally — this is iteration 1.
    2. For iterations 2..N: call ``duplicate_workflow(base_path)`` and validate
       the returned path.
    3. Hand every validated workflow off for execution the same way you run a
       single workflow (one handoff per iteration).

    Args:
        source_path: Path to the already-validated base workflow JSON.

    Returns:
        JSON with ``new_path`` (the path of the newly created workflow copy)
        or ``error`` on failure.
    """
    import random
    import shutil

    p = Path(source_path)
    if not p.exists():
        return json.dumps({"error": f"Source workflow not found: {source_path}"})

    # Build a unique destination path: <stem>_iter_<N>.json in the same directory.
    parent = p.parent
    stem = p.stem
    # Strip any existing _iter_<N> suffix so duplicating a duplicate stays tidy.
    import re as _re
    stem = _re.sub(r"_iter_\d+$", "", stem)
    idx = 1
    while True:
        candidate = parent / f"{stem}_iter_{idx:03d}.json"
        if not candidate.exists():
            break
        idx += 1

    shutil.copy2(p, candidate)

    # Randomise seed fields in the copy so each iteration produces a unique result.
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        _SEED_KEYS = {"seed", "noise_seed"}
        changed = 0
        for node in data.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            for key in _SEED_KEYS:
                if key in inputs and isinstance(inputs[key], (int, float)):
                    inputs[key] = random.randint(0, 2**32 - 1)
                    changed += 1
        candidate.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        # If seed patching fails, the copy is still usable (same seed → same image,
        # but the pipeline will still queue it as requested).
        return json.dumps({
            "new_path": str(candidate),
            "warning": f"Seed randomisation failed — copy uses original seed: {exc}",
        })

    return json.dumps({
        "new_path": str(candidate),
        "seeds_randomised": changed,
        "message": f"Workflow duplicated to {candidate.name} with {changed} seed(s) randomised.",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Node inspection
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_node_schema(node_class: list | str) -> str:
    """Get the schema for one or more ComfyUI nodes: required/optional inputs with types and defaults, output types, and description.

    Ask for EVERY node you need in ONE call — the graph you are about to build or
    repair is already known, so there is nothing to learn from fetching them one
    at a time, and each extra call replays the whole conversation as input.

    Args:
        node_class: Exact node class names, e.g.
            ["KSampler", "CLIPTextEncode", "VAEDecode"]. A single name (as a
            plain string) returns that node's schema on its own; several return
            {"schemas": {name: schema}}, with an "error" entry for any name that
            is not installed — the other schemas still come back.
    """
    names = _as_list(node_class)
    return _one_or_many(names, _get_node_schema_one, "schemas")


def _get_node_schema_one(node_class: str) -> str:
    try:
        raw = get_client().get(f"/object_info/{node_class}")
        if not raw or node_class not in raw:
            return json.dumps({"error": f"Node class '{node_class}' not found."})

        info = raw[node_class]
        input_spec = info.get("input", {})
        schema = {
            "node_class": node_class,
            "display_name": info.get("display_name", node_class),
            "description": info.get("description", ""),
            "category": info.get("category", ""),
            "input_required": _parse_inputs_schema(input_spec.get("required", {})),
            "input_optional": _parse_inputs_schema(input_spec.get("optional", {})),
            "output_types": info.get("output", []),
            "output_names": info.get("output_name", []),
            "output_is_list": info.get("output_is_list", []),
            "is_output_node": info.get("output_node", False),
        }
        return json.dumps(schema)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_workflow_node_info(node_id: list | str, workflow_path: str) -> str:
    """Return full metadata for one or more nodes inside a saved workflow.

    Combines each node's current state (class_type, title, literal inputs,
    connected inputs, widget values) with the ComfyUI schema for its class.

    Diagnosing a failure usually means reading the failing node AND the nodes
    feeding it. Ask for all of them at once — they come from the same file, so
    a second call learns nothing the first could not have returned.

    Args:
        node_id: Node keys inside the workflow JSON, e.g. ["25", "5", "13"].
            A single id (as a plain string) returns that node's info on its own;
            several return {"nodes": {id: info}}, with an "error" entry for any
            id that is not in the workflow.
        workflow_path: File path to the workflow JSON.
    """
    ids = _as_list(node_id)
    return _one_or_many(ids, lambda one: _get_workflow_node_info_one(one, workflow_path),
                        "nodes")


def _get_workflow_node_info_one(node_id: str, workflow_path: str) -> str:
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    node_id = str(node_id)
    if node_id not in workflow:
        available = sorted(workflow.keys())
        return json.dumps({"error": f"Node '{node_id}' not found.", "available_node_ids": available})

    raw_node = workflow[node_id]
    cls = raw_node.get("class_type", "")
    title = raw_node.get("_meta", {}).get("title", cls)

    inputs_raw = raw_node.get("inputs", {})
    literal_inputs: dict = {}
    connected_inputs: dict = {}
    for name, val in inputs_raw.items():
        if isinstance(val, list) and len(val) == 2 and isinstance(val[1], int):
            connected_inputs[name] = {"from_node": str(val[0]), "from_slot": val[1]}
        else:
            literal_inputs[name] = val

    schema: dict = {}
    if cls:
        try:
            all_nodes = _get_object_info()
            if cls in all_nodes:
                info = all_nodes[cls]
                input_spec = info.get("input", {})
                schema = {
                    "display_name": info.get("display_name", cls),
                    "description": info.get("description", ""),
                    "category": info.get("category", ""),
                    "input_required": _parse_inputs_schema(input_spec.get("required", {})),
                    "input_optional": _parse_inputs_schema(input_spec.get("optional", {})),
                    "output_types": info.get("output", []),
                    "output_names": info.get("output_name", []),
                    "is_output_node": info.get("output_node", False),
                }
            else:
                schema = {"warning": f"Class '{cls}' not found in ComfyUI object_info."}
        except Exception as e:
            schema = {"warning": f"Could not fetch schema: {e}"}

    result = {
        "node_id": node_id,
        "class_type": cls,
        "title": title,
        "literal_inputs": literal_inputs,
        "connected_inputs": connected_inputs,
        "widget_values": raw_node.get("widgets_values", raw_node.get("widget_values")),
        "schema": schema,
    }
    return json.dumps(result)


@tool
def search_nodes(query: list | str, limit: int = 10) -> str:
    """Search ComfyUI nodes by keyword across names, descriptions, and categories.

    Args:
        query: Search terms, e.g. ["upscale", "mask", "vae decode"]. Pass every
            term you want to try in one call rather than guessing one at a time.
            A single term (as a plain string) returns its hits on their own;
            several return {"searches": {term: hits}}.
        limit: Max results per term (default 10).
    """
    terms = _as_list(query)
    return _one_or_many(terms, lambda one: _search_nodes_one(one, limit), "searches")


def _search_nodes_one(query: str, limit: int = 10) -> str:
    try:
        all_nodes = _get_object_info()
        if isinstance(all_nodes, dict) and "error" in all_nodes:
            return json.dumps(all_nodes)

        query_lower = query.lower()
        matches = []

        for class_name, info in all_nodes.items():
            display = info.get("display_name", class_name)
            category = info.get("category", "")
            desc = info.get("description", "")
            outputs = info.get("output", [])
            input_spec = info.get("input", {})

            input_types = set()
            for section in ("required", "optional"):
                for _name, defn in input_spec.get(section, {}).items():
                    if isinstance(defn, list) and defn:
                        t = defn[0]
                        if isinstance(t, str):
                            input_types.add(t)

            searchable = " ".join(filter(None, [
                class_name,
                display or "",
                category or "",
                desc or "",
                " ".join(str(o) for o in outputs if o is not None),
                " ".join(input_types),
            ])).lower()

            if query_lower in searchable:
                matches.append({
                    "node_class": class_name,
                    "display_name": display,
                    "category": category,
                    "description": desc[:120] if desc else "",
                })

        def sort_key(m):
            exact = 0 if query_lower in m["node_class"].lower() else 1
            return (exact, m["category"], m["node_class"])

        matches.sort(key=sort_key)
        matches = matches[:limit]

        return json.dumps({
            "query": query,
            "count": len(matches),
            "results": matches,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Custom node management (install + auto-heal a missing node)
# ═══════════════════════════════════════════════════════════════════════════════

def _comfyui_root() -> "Path | None":
    """Best-effort absolute path to the ComfyUI install root (the folder that
    contains ``custom_nodes/``).

    Resolved from ``/system_stats`` argv: an explicit ``--base-directory`` wins,
    otherwise the parent of ``argv[0]`` (``main.py``). Returns ``None`` when it
    cannot be determined (e.g. a remote ComfyUI that doesn't echo a usable argv).
    """
    try:
        stats = get_client().get("/system_stats")
    except Exception:
        return None
    argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
    base = parse_argv_dir_flag(argv, "--base-directory")
    if base:
        p = Path(base)
        if p.exists():
            return p.resolve()
    if argv and isinstance(argv[0], str) and argv[0]:
        root = Path(argv[0]).parent
        if root.exists():
            return root.resolve()
    return None


def _custom_nodes_dir() -> "Path | None":
    """The ComfyUI ``custom_nodes/`` directory, or ``None`` when unresolved."""
    root = _comfyui_root()
    if root is None:
        return None
    cn = root / "custom_nodes"
    return cn if cn.exists() else None


def _resolve_comfyui_python() -> "Path | None":
    """Best-effort path to the interpreter ComfyUI itself runs on, so a pack's
    requirements install into ComfyUI's environment (not the agent's venv).

    Checks the common portable / venv layouts around the ComfyUI root. Returns
    ``None`` when none is found — the caller then reports that pip was skipped so
    the user can install requirements themselves.
    """
    root = _comfyui_root()
    if root is None:
        return None
    if os.name == "nt":
        candidates = [
            root.parent / "python_embeded" / "python.exe",   # windows portable
            root / "python_embeded" / "python.exe",
            root / ".venv" / "Scripts" / "python.exe",
            root / "venv" / "Scripts" / "python.exe",
        ]
    else:
        candidates = [
            root / ".venv" / "bin" / "python",
            root / "venv" / "bin" / "python",
            root.parent / "venv" / "bin" / "python",
        ]
    return next((c for c in candidates if c.exists()), None)


def _manager_get_mappings() -> dict:
    """ComfyUI-Manager's node-class → repository mapping, or ``{}`` when Manager
    isn't installed / reachable.

    The payload maps a repo URL → ``[[node_class_names, …], {"title": …, …}]``,
    which is what lets us resolve which pack provides a given (missing) node.
    """
    for path in ("/customnode/getmappings?mode=nickname",
                 "/api/customnode/getmappings?mode=nickname",
                 "/customnode/getmappings"):
        try:
            data = get_client().get(path)
            if isinstance(data, dict) and data:
                return data
        except Exception:
            continue
    return {}


def _registry_search(query: str, limit: int = 5) -> list:
    """Query the public ComfyUI Registry (``api.comfy.org``) for node packs
    matching *query*. Best-effort → ``[]`` on any failure.
    """
    import requests
    try:
        resp = requests.get(
            "https://api.comfy.org/nodes",
            params={"search": query, "limit": limit},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    nodes = data.get("nodes", data) if isinstance(data, dict) else data
    out: list = []
    for n in (nodes or [])[:limit]:
        if not isinstance(n, dict):
            continue
        repo = n.get("repository") or n.get("repository_url") or ""
        out.append({
            "id": n.get("id") or n.get("name") or "",
            "name": n.get("name") or n.get("id") or "",
            "repository": repo,
            "description": (n.get("description") or "")[:160],
        })
    return out


def _manager_reboot() -> bool:
    """Ask ComfyUI-Manager to restart the ComfyUI server. Returns True when the
    reboot was (very likely) triggered.

    The reboot tears the server down, so the request connection usually drops
    mid-flight — that dropped connection is treated as success. Only a clean 404
    (endpoint absent → Manager not installed / old version) counts as failure.
    """
    import requests as _rq
    for path in ("/api/manager/reboot", "/manager/reboot"):
        try:
            get_client().get(path)
            return True
        except _rq.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                continue
            return True
        except Exception:
            # Connection dropped == the server is going down to reboot => success.
            return True
    return False


@tool
def find_custom_node_for(node_type: str) -> str:
    """Find which custom-node pack provides a given ComfyUI node class.

    Use this to auto-heal an "unknown/missing node type" failure: pass the class
    name ComfyUI reported as missing and get back candidate packs (each with a git
    URL) to hand to ``install_custom_node``.

    Resolution order: ComfyUI-Manager's node→repo mappings (when Manager is
    installed), then the public ComfyUI Registry (``api.comfy.org``).

    Args:
        node_type: The missing node's class name (e.g. 'UltralyticsDetectorProvider').

    Returns:
        JSON ``{node_type, candidates: [{source, name, node_count?, description?,
        via}], source}``. ``source`` of the winning resolver ("manager" |
        "registry" | ""); ``candidates`` is empty when nothing matched.
    """
    if not isinstance(node_type, str) or not node_type.strip():
        return json.dumps({"status": "error", "error": "node_type must be a non-empty string"})
    node_type = node_type.strip()
    nt_low = node_type.lower()
    candidates: list = []

    mappings = _manager_get_mappings()
    for url, meta in mappings.items():
        try:
            names = meta[0] if isinstance(meta, list) and meta else []
            title = meta[1].get("title", "") if len(meta) > 1 and isinstance(meta[1], dict) else ""
        except Exception:
            names, title = [], ""
        if any(isinstance(n, str) and n.lower() == nt_low for n in names):
            candidates.append({
                "source": url,
                "name": title or url.rstrip("/").split("/")[-1],
                "node_count": len(names),
                "via": "manager",
            })

    source = "manager" if candidates else ""
    if not candidates:
        for r in _registry_search(node_type):
            if r.get("repository"):
                candidates.append({
                    "source": r["repository"],
                    "name": r["name"],
                    "description": r["description"],
                    "via": "registry",
                })
        if candidates:
            source = "registry"

    return json.dumps({"node_type": node_type, "candidates": candidates[:5], "source": source})


@tool
def install_custom_node(source: str, run_pip: bool = True, restart: bool = False) -> str:
    """Install a ComfyUI custom-node pack from a git URL.

    Clones the pack into ComfyUI's ``custom_nodes/`` directory and (when
    ``run_pip`` is True) installs its ``requirements.txt`` into ComfyUI's *own*
    Python environment. New nodes only load after a ComfyUI restart — set
    ``restart=True`` to ask ComfyUI-Manager to reboot the server (works only when
    Manager is installed); otherwise restart ComfyUI manually.

    To resolve a git URL from a *missing node class*, call ``find_custom_node_for``
    first and pass one of its candidate ``source`` URLs here.

    Args:
        source: A git URL (``https://…`` or ``git@…``) to the pack's repository.
        run_pip: Install the pack's requirements.txt into ComfyUI's Python (default True).
        restart: Ask ComfyUI-Manager to reboot ComfyUI afterwards (default False).

    Returns:
        JSON ``{status, path?, cloned, pip, restarted, needs_restart, message}``.
    """
    import shutil
    import subprocess

    if not isinstance(source, str) or not source.strip():
        return json.dumps({"status": "error", "error": "source must be a git URL"})
    source = source.strip()
    if not (source.startswith("http://") or source.startswith("https://") or source.startswith("git@")):
        return json.dumps({
            "status": "error",
            "error": (f"source does not look like a git URL: {source!r}. Call "
                      "find_custom_node_for(node_type) to resolve a URL first."),
        })

    cn_dir = _custom_nodes_dir()
    if cn_dir is None:
        return json.dumps({
            "status": "error",
            "error": "Could not locate ComfyUI's custom_nodes directory from /system_stats.",
        })
    if shutil.which("git") is None:
        return json.dumps({"status": "error", "error": "'git' is not on PATH; cannot clone."})

    name = source.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    dest = cn_dir / name

    result = {
        "status": "ok", "path": str(dest), "cloned": False, "pip": "skipped",
        "restarted": False, "needs_restart": True, "message": "",
    }

    if dest.exists():
        result["message"] = f"'{name}' already present in custom_nodes; skipped clone."
    else:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", source, str(dest)],
                check=True, capture_output=True, text=True, timeout=300,
            )
            result["cloned"] = True
        except subprocess.CalledProcessError as e:
            return json.dumps({"status": "error",
                               "error": f"git clone failed: {(e.stderr or '').strip() or e}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"status": "error", "error": f"git clone failed: {e}"})

    req = dest / "requirements.txt"
    if run_pip and req.exists():
        py = _resolve_comfyui_python()
        if py is None:
            result["pip"] = "unresolved-python"
            result["message"] += (" Could not locate ComfyUI's Python — install the pack's "
                                  "requirements.txt into ComfyUI's environment manually.")
        else:
            try:
                subprocess.run(
                    [str(py), "-m", "pip", "install", "-r", str(req)],
                    check=True, capture_output=True, text=True, timeout=600,
                )
                result["pip"] = "installed"
            except subprocess.CalledProcessError as e:
                result["pip"] = "failed"
                result["message"] += f" pip install failed: {(e.stderr or '').strip()[:300]}"
            except Exception as e:  # noqa: BLE001
                result["pip"] = "failed"
                result["message"] += f" pip install failed: {e}"
    elif run_pip:
        result["pip"] = "no-requirements"

    if restart:
        if _manager_reboot():
            result["restarted"] = True
            result["needs_restart"] = False
        else:
            result["message"] += (" Restart requested, but the ComfyUI-Manager reboot endpoint "
                                  "was unavailable — restart ComfyUI manually.")

    if result["needs_restart"] and not result["restarted"]:
        result["message"] = (result["message"] + " Restart ComfyUI to load the new nodes.").strip()
    return json.dumps(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Workflow templates
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def _force_build() -> bool:
    """True when build-from-scratch mode is on (AGENTY_FORCE_BUILD set).

    In this mode template loading is disabled so the agent must assemble the
    workflow node-by-node from the recipe standard + node schemas, exercising
    (and exposing weaknesses in) its build capability rather than patching a
    ready-made template.
    """
    return bool(os.environ.get("AGENTY_FORCE_BUILD"))


def _synthesize_catalog_descriptions(names: list) -> dict:
    """Intent-synthesized one-line descriptions for corpus templates that their
    index.json does not describe, so get_workflow_catalog can be a complete
    inventory (every template listed *and* described) without persisting the
    synthesized text. Best-effort: a template whose file can't be found or parsed
    is skipped (left blank)."""
    out: dict = {}
    if not names:
        return out
    try:
        from agenty_core.workflow_recipes import parser as _wr_parser
        from agenty_core.workflow_recipes.intent import IntentClassifier
    except Exception:
        return out
    ic = IntentClassifier()
    dirs = [_custom_templates_dir(), _official_templates_dir()]
    for name in names:
        path = next((f for f in (_find_template_file(Path(d), name) for d in dirs)
                     if f is not None), None)
        if path is None:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            g = (_wr_parser.parse_ui(data, name, path.name, "custom")
                 if isinstance(data.get("nodes"), list)
                 else _wr_parser.parse_api(data, name, path.name, "custom"))
            it = ic.classify(g)
            desc = ic.when_to_use(it.media, it.task, it.model_families)
            if desc:
                out[name] = desc
        except Exception:
            continue
    return out


def get_workflow_catalog() -> str:
    """Return the workflow template catalog as a flat {name: description} dictionary.

    This is the cheapest way to discover available templates. Keys are the exact
    names to pass to get_workflow_template(), and cover **every** template in the
    corpus — both custom/API workflows and the official Comfy templates, each
    described in its own index.json. For the few templates the index does not
    describe, a one-line description is synthesized from the workflow's intent at
    read time, so the catalog is a complete, fully-described inventory that agrees
    with the recipe DB's template set.

    Results are cached for up to 1 hour per session (``clear_tool_caches()`` resets
    the cache at the start of every new pipeline session).
    """
    if _force_build():
        # Build mode: no templates to match, so the Researcher sets build_new.
        return json.dumps({})
    global _tool_catalog_result, _tool_catalog_timestamp
    now = time.time()
    with _tool_cache_lock:
        if (
            _tool_catalog_result is not None
            and _tool_catalog_timestamp is not None
            and (now - _tool_catalog_timestamp) < _CATALOG_TTL
        ):
            return _tool_catalog_result
    try:
        # Read every template's metadata from its index.json (official =
        # ComfyUI-curated; custom = agentY's, descriptions migrated in). Reuses
        # the generator's loader. The flat workflow_templates.json catalog is
        # retired — index.json is the sole description store.
        from agenty_core.workflow_recipes.parser import load_descriptions
        folders = {
            "custom": str(_custom_templates_dir()),
            "official": str(_official_templates_dir()),
        }
        meta = load_descriptions(folders, None, log=lambda *a, **k: None)
        catalog = {name: (m.get("description") or "") for name, m in sorted(meta.items())}
        # Fill any template the catalog sources leave blank with an intent-
        # synthesized description, so every listed template is also described.
        blank = [n for n, d in catalog.items() if not d.strip()]
        for name, desc in _synthesize_catalog_descriptions(blank).items():
            catalog[name] = desc
        data = json.dumps(catalog, ensure_ascii=False)
        with _tool_cache_lock:
            _tool_catalog_result = data
            _tool_catalog_timestamp = time.time()
        return data
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def get_workflow_template(template_name: list | str) -> str:
    """Load one or more workflow templates by name. Saves each full workflow to its own file and returns a compact summary with the file path.

    Each summary includes: node list (id, class, title, key literal inputs),
    model info, and io metadata. The full workflow JSON is at that entry's
    ``workflow_path`` — pass that path to validate_workflow / submit_prompt.

    Fetching two templates to fuse them (a text-to-image stage feeding a
    video stage, say) is ONE call with both names, not two calls.

    Results are cached per template name for the lifetime of the session
    (``clear_tool_caches()`` resets the cache at the start of every new pipeline
    session).  Error responses are never cached so transient failures are retried.

    Args:
        template_name: Template names (without .json) from get_workflow_catalog(),
            e.g. ["flux_dev_full_text_to_image", "video_wan2_2_14B_i2v"]. A single
            name (as a plain string) returns that template's summary on its own;
            several return {"templates": {name: summary}}, each with its own
            workflow_path.
    """
    names = _as_list(template_name)
    return _one_or_many(names, _get_workflow_template_one, "templates")


def _get_workflow_template_one(template_name: str) -> str:
    if _force_build():
        # Build mode: hand back an EMPTY canvas instead of a ready-made scaffold,
        # so the agent assembles every node itself (from the recipe standard) yet
        # still has a workflow file to add nodes to.
        empty_path = _save_workflow({}, name=f"build_{Path(template_name).stem or 'new'}")
        return json.dumps({
            "name": template_name,
            "source": "build-mode",
            "workflow_path": empty_path,
            "node_count": 0,
            "nodes": [],
            "io": {"inputs": [], "outputs": [], "nodes": []},
            "build_from_scratch": True,
            "hint": "This is an EMPTY canvas - there is no scaffold. You MUST first "
                    "call get_workflow_recipe(task, model) and build the workflow to "
                    "that recipe: create EVERY node in build_nodes (class + count - "
                    "it includes model-specific nodes like WanImageToVideo that "
                    "required_nodes omits; do NOT substitute a generic node such as "
                    "VAEEncode for it), WIRE THEM PER reference_wiring (the real "
                    "edges of the recipe's reference_member, as "
                    "'<class>#<instance>:<slot> -> <class>#<instance>.<input>'; "
                    "connection_patterns are role-level orientation and are often "
                    "empty), wire every load_bearing_inputs entry, expose "
                    "boundary_ports, and set each node's widget params from "
                    "node_defaults (do NOT guess weight_dtype/model variants). Create "
                    "nodes with add_workflow_node, wire/set inputs with "
                    "update_workflow, confirm input names/slots via get_node_schema, "
                    "and wire a Save/output node. These are STANDARD ComfyUI nodes "
                    "that support this model - never claim a custom node is needed, "
                    "never call the model unsupported, and never substitute a "
                    "different model. Build exactly the recipe.",
        })
    with _tool_cache_lock:
        if template_name in _tool_template_results:
            return _tool_template_results[template_name]
    try:
        lookup = template_name.removesuffix(".json")
        workflow = None
        source = ""
        metadata: dict = {}

        # Try user ad-hoc workflows first (stored alongside custom templates)
        tdir = _custom_templates_dir()
        for candidate in [tdir / f"{lookup}.json", tdir / template_name]:
            if candidate.exists():
                with open(candidate, encoding="utf-8") as f:
                    workflow = json.load(f)
                source = "user"
                break

        # Try custom + official templates (custom dir, then GitHub, then local fallback)
        if workflow is None:
            data = _fetch_template(lookup)
            if data is None and lookup != template_name:
                data = _fetch_template(template_name)
            if data is not None:
                workflow = data
                source = "templates"
                for tpl in _load_index():
                    if tpl.get("name") == lookup:
                        metadata = {
                            "models": tpl.get("models", []),
                            "io": tpl.get("io", {}),
                        }
                        break

        # Fuzzy fallback: the researcher may name a template that doesn't exist
        # verbatim (e.g. 'z_image_turbo' vs 'text_to_image_z_image_turbo'). Match
        # the closest catalog name by token overlap and load that instead.
        if workflow is None:
            import re as _re  # noqa: PLC0415
            # Noise tokens a researcher tends to pad a name with — dropping them
            # keeps the score focused on the distinctive tokens (model, task).
            _common = {"text", "to", "the", "a", "of", "and", "workflow", "dev", "v",
                       "template", "generation", "generate", "using", "model",
                       "custom", "new", "base", "simple", "basic", "default", "for"}
            _qt = set(_re.findall(r"[a-z0-9]+", lookup.lower())) - _common
            _best, _bscore = None, 0.0
            if _qt:
                _seen: set = set()
                # Search custom AND official templates (official are fetchable by
                # name but absent from the custom index).
                for _t in _load_index() + _official_index():
                    _nm = _t.get("name", "")
                    if not _nm or _nm in _seen:
                        continue
                    _seen.add(_nm)
                    _nt = set(_re.findall(r"[a-z0-9]+", _nm.lower())) - _common
                    if not _nt:
                        continue
                    # Overlap relative to the smaller token set — rewards a name
                    # that fully contains the query's distinctive tokens.
                    _sc = len(_qt & _nt) / min(len(_qt), len(_nt))
                    if _sc > _bscore:
                        _best, _bscore = _nm, _sc
            if _best and _bscore >= 0.6:
                data = _fetch_template(_best)
                if data is not None:
                    workflow = data
                    source = "templates"
                    lookup = _best
                    for tpl in _load_index():
                        if tpl.get("name") == _best:
                            metadata = {"models": tpl.get("models", []), "io": tpl.get("io", {})}
                            break

        if workflow is None:
            return json.dumps({
                "error": f"Template '{template_name}' not found.",
                "hint": "Use get_workflow_catalog() to see available templates.",
            })

        # Normalise to API format
        converted = False
        if _is_graph_format(workflow):
            workflow = _convert_graph_to_api(workflow)
            converted = True

        # Snap model-file inputs to the exact names the live server accepts so
        # separator/folder drift is not mistaken for a missing model.
        if isinstance(workflow, dict):
            _resolve_model_names(workflow)

        workflow_path = _save_workflow(workflow, name=lookup)

        # Build compact node summary
        node_summary = []
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            cls = node.get("class_type", "unknown")
            title = node.get("_meta", {}).get("title", cls)
            inputs = node.get("inputs", {})
            key_inputs = {
                k: v for k, v in inputs.items()
                if not isinstance(v, list) and v is not None and v != ""
            }
            entry: dict = {"id": nid, "class": cls, "title": title}
            if key_inputs:
                entry["inputs"] = key_inputs
            node_summary.append(entry)

        result: dict = {
            "name": lookup,
            "source": source,
            "workflow_path": workflow_path,
            "node_count": len(node_summary),
            "nodes": node_summary,
        }
        if converted:
            result["converted_from_graph_format"] = True
        if metadata.get("models"):
            result["models"] = metadata["models"]
        if metadata.get("io"):
            result["io"] = metadata["io"]

        result_json = json.dumps(result)
        with _tool_cache_lock:
            _tool_template_results[template_name] = result_json
        return result_json
    except Exception as e:
        return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Workflow recipes (task -> model -> node clusters knowledge base)
# ═══════════════════════════════════════════════════════════════════════════════
# The recipe database is produced by the workflow_recipes generator
# (``python -m agenty_core.workflow_recipes.cli``). It groups the template corpus
# into task -> model recipes, each describing the required node clusters,
# connection patterns, boundary ports, and the concrete member templates that
# implement it.
_RECIPE_DB_RELPATHS = (
    Path("config") / "workflow_recipes.json",
)

# Brainbriefing task-type shorthands -> canonical taxonomy task names.
_RECIPE_TASK_ALIASES = {
    "image_generation": "text to image", "txt2img": "text to image",
    "image_edit": "image edit", "img2img": "image edit",
    "controlnet": "image edit with controlnet",
    "inpaint": "inpaint outpaint", "outpaint": "inpaint outpaint",
    "video_i2v": "image to video", "i2v": "image to video",
    "video_t2v": "text to video", "t2v": "text to video",
    "video_flf": "first last frame to video", "flf": "first last frame to video",
    "video_v2v": "video to video", "v2v": "video to video",
    "upscale": "upscale", "audio": "audio", "3d": "3d",
}


def _load_recipe_db():
    """Return (db_dict, path_str) for the recipe database, or (None, None)."""
    root = _corpus_root()
    for rel in _RECIPE_DB_RELPATHS:
        p = root / rel
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")), str(p)
            except Exception:
                continue
    return None, None


def _recipe_tokens(text: str) -> list:
    return "".join(c if c.isalnum() else " " for c in (text or "").lower()).split()


# Widget keys that are request-specific and must NOT be copied from the template
# (the Brain sets them from the brainbriefing / per run).
_NODE_DEFAULT_SKIP = frozenset({
    "text", "filename_prefix", "seed", "noise_seed", "prompt", "negative_prompt",
})

# Only these non-model config keys are reliably extracted and worth carrying as
# node_defaults. Everything else (sampler_name, scheduler, steps, cfg, width,
# height, guidance, shift, ...) is either request-specific or prone to
# widget-position drift, so the Brain derives it via get_node_schema instead.
_NODE_DEFAULT_CONFIG_KEEP = frozenset({"weight_dtype", "type", "device"})


def _keep_node_default(class_type: str, key: str, value, object_info: dict) -> bool:
    """Keep a param in node_defaults only if it is reliable: a known config key,
    or a model-file value that is actually a valid installed option for this
    node input (drops widget-garbled numbers and uninstalled/bogus filenames)."""
    if key in _NODE_DEFAULT_CONFIG_KEEP:
        return True
    if isinstance(value, str) and value.lower().endswith(_MODEL_FILE_EXTS):
        schema = object_info.get(class_type, {}).get("input", {})
        specs = {**schema.get("required", {}), **schema.get("optional", {})}
        spec = specs.get(key)
        options = (spec[0] if isinstance(spec, (list, tuple)) and spec
                   and isinstance(spec[0], list) else None)
        return bool(options and value in options)
    return False


def _is_utility_class(class_type: str) -> bool:
    """Plumbing (primitive / math / switch / reroute), per the recipe vocabulary."""
    try:
        from agenty_core.workflow_recipes.roles import is_utility  # noqa: PLC0415
        return bool(is_utility(class_type))
    except Exception:  # noqa: BLE001
        return False


def _reference_wiring(workflow: dict, object_info: dict, cap: int = 160) -> tuple:
    """The member graph's own edges, addressed by class and instance.

    ``("CheckpointLoaderSimple#0:2 -> VAEDecode#0.vae", …)`` — source class,
    instance, output SLOT on the left; destination class, instance, input NAME on
    the right. Node ids are deliberately not used: the build is creating its own
    nodes, and what it needs is the shape, not someone else's numbering.

    This exists because the recipe's ``connection_patterns`` cannot address a wire.
    They are ``(from_role, to_role, data_type)`` triples agreed across every member,
    which loses three things a build needs: which instance (two samplers in a
    two-stage graph share one triple), which input (``CFGGuider`` takes `positive`
    and `negative`, both CONDITIONING) and which output slot (a checkpoint loader's
    VAE is slot 2; ``LTXVConditioning`` emits two CONDITIONING outputs where the
    index is the whole meaning). Measured on this corpus, a recipe with four or more
    members shows a median of ZERO invariant patterns — the intersection of a rich
    group is empty — so the agent was being handed a 35-node build with no wiring at
    all and inventing the graph.

    Second return value: the optional inputs this member wires. An input the schema
    marks optional is one nothing will complain about, and some of them carry the
    whole point of the workflow — a ``CreateVideo.audio`` left unwired gives a silent
    video out of an audio model, valid and green the whole way. They are listed apart
    from the edges because a flat list of forty wires does not say which ones are
    silently droppable.

    Both come from the ONE member the rest of the build spec comes from, not from a
    vote across members: a vote over a heterogeneous group is what leaves
    ``connection_patterns`` empty in the first place.
    """
    if not isinstance(workflow, dict):
        return (), ()
    nodes = {str(nid): n for nid, n in workflow.items()
             if isinstance(n, dict) and n.get("class_type")
             and n.get("class_type") not in ("Note", "MarkdownNote", "Reroute",
                                             "PrimitiveNode")}

    def _key(nid: str):
        return (0, int(nid)) if str(nid).isdigit() else (1, str(nid))

    # Instance index per class, in the member's own node order — stable, and the
    # same order build_nodes' counts are read in.
    label: dict = {}
    seen: dict = {}
    for nid in sorted(nodes, key=_key):
        cls = nodes[nid]["class_type"]
        label[nid] = f"{cls}#{seen.get(cls, 0)}"
        seen[cls] = seen.get(cls, 0) + 1

    edges: list = []
    optional_wired: list = []
    for nid in sorted(nodes, key=_key):
        cls = nodes[nid]["class_type"]
        optional = set(((object_info.get(cls) or {}).get("input") or {})
                       .get("optional") or {})
        for name, value in (nodes[nid].get("inputs") or {}).items():
            if not (isinstance(value, list) and len(value) == 2
                    and isinstance(value[1], (int, float))):
                continue
            src = str(value[0])
            if src not in label:
                continue
            edges.append(f"{label[src]}:{int(value[1])} -> {label[nid]}.{name}")
            # A switch's `on_true` or a math node's operand is optional and wired in
            # every graph that uses one; saying so teaches nothing. The ones worth
            # naming are on nodes that do the work.
            if name in optional and not _is_utility_class(cls):
                optional_wired.append(f"{label[nid]}.{name}")
    if len(edges) > cap:
        # Never let a truncated list look complete: a build that thinks it has the
        # whole shape stops looking for the rest.
        left = len(edges) - cap
        edges = edges[:cap] + [
            f"... {left} more edge(s) not listed - load 'reference_member' with "
            f"get_workflow_template and read them off it"]
    return tuple(edges), tuple(dict.fromkeys(optional_wired))


def _recipe_build_spec(member_files: list, model: str = "",
                       required_classes: set | None = None) -> dict:
    """Build spec from the recipe's BEST-matching member template.

    Returns ``{"node_defaults": {class: {input: value}}, "build_nodes":
    [{"node_class": cls, "count": n}], "reference_member": name,
    "reference_wiring": [...], "load_bearing_inputs": [...]}``.
    ``node_defaults`` gives correct node configs (weight_dtype, model-file variant,
    ...) so a from-scratch build does not guess. ``build_nodes`` is the COMPLETE
    node list of the member - unlike the recipe's invariant ``required_nodes`` (an
    intersection across members), it keeps model-specific nodes such as
    ``WanImageToVideo`` that the build otherwise omits. ``reference_wiring`` is that
    same member's edge list, which is the only wiring in the recipe precise enough
    to build from (see :func:`_reference_wiring`). The member is chosen to match the
    recipe's model (name-token overlap) then required-class coverage.
    """
    from collections import Counter
    required_classes = required_classes or set()
    mtoks = {t for t in re.split(r"[^a-z0-9]+", (model or "").lower()) if len(t) > 1}
    try:
        object_info = _get_object_info()
    except Exception:
        object_info = {}

    def _name_score(name: str) -> int:
        toks = set(re.split(r"[^a-z0-9]+", str(name).lower()))
        return len(toks & mtoks)

    def _as_nodes(counter: Counter) -> list:
        return [{"node_class": c, "count": n} for c, n in sorted(counter.items())]

    ordered = sorted(member_files or [], key=_name_score, reverse=True)
    best = {"node_defaults": {}, "build_nodes": [], "reference_member": "",
            "reference_wiring": [], "load_bearing_inputs": []}
    best_score = (-1, -1)
    for name in ordered:
        try:
            data = _fetch_template(str(name).removesuffix(".json"))
        except Exception:
            data = None
        if data is None:
            continue
        try:
            wf = _convert_graph_to_api(data) if _is_graph_format(data) else data
            if isinstance(wf, dict):
                _resolve_model_names(wf)
        except Exception:
            continue
        defaults: dict = {}
        counts: Counter = Counter()
        for node in (wf.values() if isinstance(wf, dict) else []):
            if not isinstance(node, dict):
                continue
            cls = node.get("class_type")
            inputs = node.get("inputs", {})
            if not cls or cls in ("Note", "MarkdownNote", "Reroute", "PrimitiveNode"):
                continue
            counts[cls] += 1
            if cls in defaults or not isinstance(inputs, dict):
                continue
            lit = {k: v for k, v in inputs.items()
                   if not isinstance(v, list) and not str(k).startswith("__")
                   and k not in _NODE_DEFAULT_SKIP
                   and _keep_node_default(cls, k, v, object_info)}
            if lit:
                defaults[cls] = lit
        if not counts:
            continue
        wiring, load_bearing = _reference_wiring(wf, object_info)
        spec = {"node_defaults": defaults, "build_nodes": _as_nodes(counts),
                "reference_member": str(name).removesuffix(".json"),
                "reference_wiring": list(wiring),
                "load_bearing_inputs": list(load_bearing)}
        score = (_name_score(name), len(set(counts) & required_classes))
        if score > best_score:
            best_score, best = score, spec
        # A model-name match with good required coverage is authoritative.
        if score[0] > 0 and score[1] >= len(required_classes) > 0:
            return spec
    return best


def _recipe_leaf_view(task: dict, model: dict) -> dict:
    """The build-oriented view of a (task, model) recipe leaf for the Brain."""
    required = [
        {"node_class": e["node_class"], "min_instances": e.get("min_instances", 1),
         "role": e.get("role")}
        for e in model.get("required_node_roles", []) if not e.get("utility")
    ]
    ui = model.get("user_intent", {})
    return {
        "id": model["id"],
        "task": task["task"],
        "model": model["model"],
        # Where it runs: "local" (local models), "api" (remote partner-node
        # generation - needs API credentials/credits, no local models), or
        # "hybrid" (local generation plus a remote helper node).
        "execution": model.get("execution", "local"),
        "uses_api_nodes": model.get("uses_api_nodes", False),
        "api_node_classes": model.get("api_node_classes", []),
        "when_to_use": ui.get("when_to_use"),
        "example_requests": ui.get("example_requests", []),
        "description": model.get("description"),
        "boundary_ports": model.get("boundary_ports", {}),
        "node_clusters": model.get("node_clusters", []),
        "connection_patterns": [
            p for p in model.get("connection_patterns", []) if p.get("invariant")
        ],
        "required_nodes": required,
        **_recipe_build_spec(
            model.get("member_files", []), model.get("model", ""),
            {e["node_class"] for e in required},
        ),
        "member_workflows": model.get("member_files", []),
        "unresolved_nodes": model.get("unresolved_nodes", []),
        "custom_nodes": model.get("custom_nodes", []),
        "how_to_build": (
            "Create the COMPLETE node set in 'build_nodes' (class + count - it "
            "includes model-specific nodes like WanImageToVideo that "
            "'required_nodes' omits); do not drop any and do not substitute a "
            "generic node (e.g. do NOT use VAEEncode where 'build_nodes' lists "
            "WanImageToVideo). "
            "WIRE THEM PER 'reference_wiring', which is the working graph of "
            "'reference_member' written as "
            "'<class>#<instance>:<output slot> -> <class>#<instance>.<input name>': "
            "map each '<class>#<n>' to the node you created for that class and "
            "instance, and reproduce every line. It is the only wiring here precise "
            "enough to build from - 'connection_patterns' are role-level triples "
            "that cannot say which instance, which input or which output slot, and "
            "are often empty. Deviate from it only where the request needs "
            "something the reference does not do, and then wire what you add. "
            "Every input named in 'load_bearing_inputs' MUST end up wired: the "
            "schema calls them optional, so nothing will complain, and the run will "
            "quietly be wrong (an unwired CreateVideo.audio is a silent video from "
            "an audio model). "
            "Expose 'boundary_ports', and set each node's widget params from "
            "'node_defaults' (weight_dtype, model-file variant, ...) - these are "
            "template-verified, so do NOT guess or 'match the filename' for "
            "weight_dtype. Confirm each node's input names/slots with "
            "get_node_schema. Override only prompt text, seed and output paths "
            "from the brainbriefing."
        ),
    }


@tool
def list_workflow_recipes() -> str:
    """List the available workflow recipes as a task -> model index.

    Cheapest way to discover what task+model recipes exist before fetching one
    with get_workflow_recipe(). Each entry has the recipe id, model family,
    member count, and a when_to_use line. Returns an error with a hint if the
    recipe database has not been generated yet.
    """
    db, path = _load_recipe_db()
    if not db:
        return json.dumps({
            "error": "recipe database not found",
            "hint": "generate it with: python -m agenty_core.workflow_recipes.cli",
        })
    tasks = []
    for t in db.get("tasks", []):
        tasks.append({
            "task": t["task"],
            "models": [
                {"id": m["id"], "model": m["model"],
                 "member_count": m["member_count"],
                 "when_to_use": m.get("user_intent", {}).get("when_to_use")}
                for m in t.get("models", [])
            ],
        })
    return json.dumps({"source": path, "tasks": tasks})


@tool
def get_workflow_recipe(task: str, model: str = "") -> str:
    """Return the build recipe for a task (and optionally a model family).

    Use this when building a new workflow from scratch (template.name ==
    "build_new"). The recipe is the standard to build to: it lists the required
    node clusters, the invariant connection patterns, the boundary ports, and
    the concrete member templates that implement this task+model (load the
    closest one via get_workflow_template() as a scaffold).

    Args:
        task: The task - a canonical name ("Image to Video", "Image Edit with
            ControlNet") or a brainbriefing shorthand ("video_i2v", "image_edit").
        model: Optional model family ("WAN 2.2", "LTX-2", "Flux", "Qwen Image").
            When omitted, the best-matching task is returned along with the list
            of models available under it so you can pick one.
    """
    db, path = _load_recipe_db()
    if not db:
        return json.dumps({
            "error": "recipe database not found",
            "hint": "generate it with: python -m agenty_core.workflow_recipes.cli",
        })

    tkey = task.strip().lower()
    task_q = set(_recipe_tokens(_RECIPE_TASK_ALIASES.get(tkey, task)))
    model_q = set(_recipe_tokens(model))

    best = None
    best_score = -1.0
    for t in db.get("tasks", []):
        tnorm = set(_recipe_tokens(t["task"]) + _recipe_tokens(t["id"]))
        tscore = len(task_q & tnorm)
        if task_q and tscore == 0:
            continue
        for m in t.get("models", []):
            mnorm = set(_recipe_tokens(m["model"]) + _recipe_tokens(m["id"]))
            mscore = len(model_q & mnorm) if model_q else 0
            # Prefer a local recipe over an api/hybrid one when they otherwise
            # tie (local OSS models build without partner credits/cost).
            exec_bonus = {"local": 2.0, "hybrid": 1.0}.get(m.get("execution", "local"), 0.0)
            score = tscore * 10 + mscore * 5 + exec_bonus + m.get("member_count", 0) * 0.01
            if score > best_score:
                best_score = score
                best = (t, m)

    if not best:
        return json.dumps({
            "error": "no recipe matched",
            "available_tasks": [t["task"] for t in db.get("tasks", [])],
        })

    t, m = best
    result = {"source": path, "recipe": _recipe_leaf_view(t, m)}
    if not model_q:
        result["models_in_task"] = [
            {"id": mm["id"], "model": mm["model"], "member_count": mm["member_count"]}
            for mm in t.get("models", [])
        ]
    return json.dumps(result)


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Live canvas selection
# ═══════════════════════════════════════════════════════════════════════════════
# The selection lives in the browser, so both tools below are a round trip
# through the agentY-comfyuiConnect extension: it relays the request to the open
# ComfyUI page over the websocket and answers from there (see that repo's
# /agent/canvas_selection and /agent/set_node_params routes). Nothing is cached
# anywhere — each call reads the graph as it stands at that moment.

def _canvas_bridge_error(exc: Exception) -> str:
    """Turn a failed /agent/* call into an actionable message.

    The two things that actually go wrong are ComfyUI not running and the
    extension not being installed (or being an older build without these
    routes); both surface as opaque HTTP/connection errors otherwise.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 404:
        return ("the agentY-comfyuiConnect extension is not installed in this ComfyUI, "
                "or predates the canvas-selection routes — install/update it under "
                "<ComfyUI>/custom_nodes/ and restart ComfyUI.")
    return f"could not reach ComfyUI ({exc}). Is it running?"


@tool
def get_canvas_selection() -> str:
    """Read the nodes the user has selected on the live ComfyUI canvas.

    Use whenever the user talks about their canvas deictically — "this node",
    "the prompt I have selected", "what did I just pick", "change these settings"
    — instead of guessing from a saved workflow file. Returns each selected node
    with its id, class type, title and current widget values, read from the open
    browser page at the moment of the call (nothing is cached, so it reflects
    edits the user made seconds ago). Pair it with ``set_canvas_node_params`` to
    write a value back.

    Requires ComfyUI open in a browser (a background tab is fine) with the
    agentY-comfyuiConnect extension installed.

    Returns JSON: ``{status, count, nodes: [{id, type, title, mode, widgets}],
    workflow}``. ``mode`` is 0 normally, 2 muted, 4 bypassed — a muted or
    bypassed node takes no part in the next run.
    """
    try:
        r = get_client().get("/agent/canvas_selection")
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "error": _canvas_bridge_error(e)})
    if not isinstance(r, dict):
        return json.dumps({"status": "error", "error": f"unexpected response: {r!r}"})
    if not r.get("ok"):
        return json.dumps({"status": "error", "error": r.get("error", "unknown error")})
    nodes = r.get("nodes") or []
    out = {"status": "ok", "count": len(nodes), "nodes": nodes}
    if r.get("workflow"):
        out["workflow"] = r["workflow"]
    if not nodes:
        out["hint"] = ("Nothing is selected on the canvas — ask the user to click the "
                       "node(s) they mean.")
    return json.dumps(out)


@tool
def set_canvas_node_params(node_id: str, params: dict) -> str:
    """Write widget values onto a node of the live ComfyUI canvas.

    Use when the user asks you to change something on a node they are pointing at
    — "rewrite this prompt", "set steps to 30", "bump the CFG". Call
    ``get_canvas_selection`` first to get the node id and see the current values.
    The edit lands on the open graph instantly: no file to save, no browser
    refresh, no re-queue. It does NOT run the graph — the user queues it when
    they are ready.

    Args:
        node_id: Node id as reported by ``get_canvas_selection``.
        params: Mapping of widget name -> new value, e.g.
            ``{"text": "a rainy neon street"}`` or ``{"steps": 30, "cfg": 6.5}``.
            Only include the widgets you are changing.

    Returns JSON: ``{status, node, applied, unknown}``. ``unknown`` lists widget
    names that node does not have — check them against ``get_canvas_selection``
    rather than retrying blindly.
    """
    if not isinstance(params, dict) or not params:
        return json.dumps({"status": "error",
                           "error": "params must be a non-empty {widget: value} mapping."})
    try:
        r = get_client().post("/agent/set_node_params",
                              json_data={"node_id": str(node_id), "params": params})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "error": _canvas_bridge_error(e)})
    if not isinstance(r, dict):
        return json.dumps({"status": "error", "error": f"unexpected response: {r!r}"})
    applied, unknown = r.get("applied") or [], r.get("unknown") or []
    if not r.get("ok") and not applied:
        return json.dumps({"status": "error",
                           "error": r.get("error") or "nothing was changed",
                           "unknown": unknown})
    out = {"status": "applied", "node_id": str(node_id),
           "node": r.get("node", ""), "applied": applied}
    if unknown:
        out["unknown"] = unknown
        out["warning"] = (f"widget(s) {unknown} are not on that node — they were ignored.")
    return json.dumps(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Workflow modification
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def open_workflow_in_canvas(workflow_path: str, name: str = "",
                            push_to_canvas: bool = True) -> str:
    """Load a workflow the agent built or ran into the ComfyUI canvas so the user
    can visually inspect it.

    Executed workflows are in API/prompt format, which the canvas can't render, so
    this converts them to graph format (auto-laid-out) and saves them into
    ComfyUI's user workflow folder via the /userdata API. The file then appears in
    the Workflows sidebar of the open ComfyUI web UI (http://127.0.0.1:8188) under
    an ``agent/`` folder — the user opens it there to see exactly what was built
    and executed (nodes, connections, and widget values).

    Call this after signalling/executing a workflow when the user wants to review
    it visually. A graph-format workflow is saved as-is; an API-format one is
    converted first.

    Args:
        workflow_path: Path to the workflow JSON (API or graph format).
        name: Optional file/display name (defaults to the workflow's file stem).
        push_to_canvas: Also swap it onto the canvas the user is looking at
            (default). Pass False to only file it under ``agent/`` in the
            Workflows sidebar — which is what you want when the graph they
            currently have open is the thing being worked on, and replacing it
            would take away what they are watching.

    Returns JSON: ``{status, saved_as, node_count, url, hint}``.
    """
    try:
        wf = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"status": "error", "error": f"Cannot load workflow: {e}"})

    graph = wf if _is_graph_format(wf) else _api_to_graph(wf)
    if not graph.get("nodes"):
        return json.dumps({"status": "error",
                           "error": "workflow has no nodes to display"})

    stem = re.sub(r"[^A-Za-z0-9_.-]", "_",
                  (name or Path(workflow_path).stem or "agent_workflow"))
    if stem.endswith(".json"):
        stem = stem[:-5]
    rel = f"workflows/agent/{stem}.json"

    try:
        import urllib.parse as _up  # noqa: PLC0415
        # /userdata writes into the user's workflow folder (creates subdirs).
        get_client().post("/userdata/" + _up.quote(rel, safe=""), json_data=graph)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "error": f"Could not save to ComfyUI: {e}"})

    # Also push it straight onto the open canvas via the AgentCanvas extension
    # (custom_nodes/comfyui-agent-canvas). This REPLACES the graph the user has
    # open, so a caller with a reason to leave it alone can opt out and keep the
    # sidebar copy above as the only delivery. Harmless 404 when the extension
    # isn't installed/loaded — the sidebar copy is always available as fallback.
    opened = False
    if push_to_canvas:
        try:
            _r = get_client().post("/agent/load_workflow", json_data={"workflow": graph})
            opened = isinstance(_r, dict) and bool(_r.get("ok"))
        except Exception:  # noqa: BLE001
            pass

    return json.dumps({
        "status": "ok",
        "saved_as": f"agent/{stem}",
        "opened_on_canvas": opened,
        "node_count": len(graph["nodes"]),
        "url": "http://127.0.0.1:8188",
        "hint": ("Opened on the ComfyUI canvas." if opened else
                 f"Saved — open the Workflows sidebar and select agent/{stem}"
                 + ("." if not push_to_canvas else
                    " (install/restart to enable auto-open on the canvas).")),
    })


@tool
def save_workflow(workflow_json: str, name: str = "") -> str:
    """Save a complete workflow JSON to a file and return the file path.

    Only use for building entirely new workflows from scratch.
    For editing existing workflows, use patch_workflow() instead.

    Args:
        workflow_json: The complete workflow JSON string in ComfyUI API format.
        name: Optional name for the file (default: auto-generated).
    """
    try:
        workflow = json.loads(workflow_json) if isinstance(workflow_json, str) else workflow_json
        path = _save_workflow(workflow, name=name)
        return json.dumps({"workflow_path": path, "node_count": len([k for k in workflow if isinstance(workflow.get(k), dict)])})
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def patch_workflow(workflow_path: str, patches: str) -> str:
    """Apply targeted edits to a saved workflow without re-outputting the full JSON.

    Much more token-efficient than save_workflow for modifying templates.
    Each patch targets a specific node input or widget value.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template).
        patches: JSON string — a list of patch objects. Each patch object has:
            - node_id (str): The node ID to modify (e.g. "6", "190").
            - input_name (str): The input field name to set (e.g. "text", "image", "filename").
            - value: The new value (string, number, bool, or list for links like [node_id, slot]).
            Optional fields:
            - widget_values_index (int): If set, patch widget_values[index] instead of inputs.
            - class_type (str): If set, change the node's class_type.
            Example: [{"node_id": "6", "input_name": "text", "value": "a photo of a chimp"},
                      {"node_id": "190", "input_name": "image", "value": "image.png"}]
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    try:
        patch_list = json.loads(patches) if isinstance(patches, str) else patches
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid patches JSON: {e}"})

    if not isinstance(patch_list, list):
        return json.dumps({"error": "patches must be a JSON array of patch objects."})

    applied: list[str] = []
    errors: list[str] = []

    for i, patch in enumerate(patch_list):
        nid = str(patch.get("node_id", ""))
        if nid not in workflow:
            errors.append(f"Patch {i}: node '{nid}' not found in workflow.")
            continue

        node = workflow[nid]

        # Optional: change class_type
        if "class_type" in patch:
            node["class_type"] = patch["class_type"]
            applied.append(f"Node {nid}: class_type → {patch['class_type']}")

        # Patch widget_values by index
        if "widget_values_index" in patch:
            idx = int(patch["widget_values_index"])
            wv = node.get("widget_values", node.get("widgets_values"))
            wv_key = "widget_values" if "widget_values" in node else "widgets_values"
            if isinstance(wv, list) and 0 <= idx < len(wv):
                wv[idx] = patch["value"]
                node[wv_key] = wv
                applied.append(f"Node {nid}: {wv_key}[{idx}] → {patch['value']!r}")
            else:
                errors.append(f"Patch {i}: node '{nid}' has no {wv_key}[{idx}].")
            continue

        # Patch inputs
        inp_name = patch.get("input_name")
        if inp_name:
            if "inputs" not in node:
                node["inputs"] = {}
            value = patch["value"]
            # A LoadImage reference must carry the 'agent/' input subfolder that
            # upload_image stages files into; a bare filename makes ComfyUI look
            # in the input root and reject it as "Invalid image file".
            if inp_name == "image" and node.get("class_type") == "LoadImage" and isinstance(value, str):
                value = _agent_input_ref(value)
            node["inputs"][inp_name] = value
            val_repr = repr(value)
            if len(val_repr) > 80:
                val_repr = val_repr[:77] + "..."
            applied.append(f"Node {nid}.inputs.{inp_name} → {val_repr}")

    # Save back
    path = _save_edit(workflow, workflow_path)

    # Failure guard
    global _patch_fail_count, _patch_last_workflow_path
    _patch_last_workflow_path = path

    if errors:
        _patch_fail_count += 1
        print(
            f"[patch_workflow] Failure {_patch_fail_count}/{_PATCH_FAIL_LIMIT}: "
            f"{len(errors)} patch error(s)."
        )

        if _patch_fail_count >= _PATCH_FAIL_LIMIT:
            debug_name = f"{Path(workflow_path).stem}_patch_debug"
            debug_path = _save_workflow(workflow, name=debug_name)
            print(
                f"[patch_workflow] LIMIT REACHED — debug snapshot saved to: {debug_path}"
            )
            return json.dumps({
                "workflow_path": path,
                "applied": applied,
                "errors": errors,
                "patch_count": len(applied),
                "patch_failure_limit_reached": True,
                "debug_workflow_path": debug_path,
                "message": (
                    f"patch_workflow has failed {_PATCH_FAIL_LIMIT} times. "
                    f"Current workflow snapshot saved to: {debug_path}. "
                    "STOP — do not call patch_workflow again. "
                    "Report the debug_workflow_path to the user and ask for guidance."
                ),
            })
    else:
        _patch_fail_count = 0

    return json.dumps({
        "workflow_path": path,
        "applied": applied,
        "errors": errors,
        "patch_count": len(applied),
    })


@tool
def update_workflow(
    workflow_path: str,
    patches: str = "[]",
    add_nodes: str = "[]",
    remove_nodes: str = "[]",
) -> str:
    """Add/remove nodes, patch inputs, and validate a workflow in one atomic step.

    Use this instead of calling add_workflow_node / remove_workflow_node,
    patch_workflow, and validate_workflow separately.  Returns status "ok" when
    the workflow is valid and ready to hand off, or "error" with a full list of
    problems to fix.

    Execution order: remove nodes → add nodes → apply patches → validate.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template).
        patches: JSON array of patch objects (same format as patch_workflow).
            Each: {"node_id": "6", "input_name": "text", "value": "..."}
            Optional per-patch fields: widget_values_index, class_type.
            Pass "[]" (default) to skip patching.
        add_nodes: JSON array of node-add specs.
            Each: {"node_id": "200", "class_type": "LoadImage",
                   "inputs": {"image": "file.png"}, "meta_title": "My Node"}
            Pass "[]" (default) to add nothing.
        remove_nodes: JSON array of node ID strings to remove.
            Example: ["42", "55"]
            Pass "[]" (default) to remove nothing.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"status": "error", "error": f"Cannot load workflow: {e}"})

    try:
        remove_list: list = json.loads(remove_nodes) if isinstance(remove_nodes, str) else remove_nodes
        add_list: list = json.loads(add_nodes) if isinstance(add_nodes, str) else add_nodes
        patch_list: list = json.loads(patches) if isinstance(patches, str) else patches
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"Invalid JSON argument: {e}"})

    removed: list[str] = []
    added: list[str] = []
    applied: list[str] = []
    node_errors: list[str] = []
    cleaned_links: int = 0

    # ── Remove nodes ──────────────────────────────────────────────────────────
    for nid in (str(x) for x in remove_list):
        if nid not in workflow:
            node_errors.append(f"remove: node '{nid}' not found.")
            continue
        del workflow[nid]
        removed.append(nid)

    # Clean dangling links pointing to removed nodes
    removed_set = set(removed)
    for _nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs_dict = node.get("inputs", {})
        for inp_name, inp_val in list(inputs_dict.items()):
            if isinstance(inp_val, list) and len(inp_val) == 2 and str(inp_val[0]) in removed_set:
                del inputs_dict[inp_name]
                cleaned_links += 1

    # ── Add nodes ─────────────────────────────────────────────────────────────
    for spec in add_list:
        nid = str(spec.get("node_id", ""))
        cls = spec.get("class_type", "")
        if not nid or not cls:
            node_errors.append(f"add: spec missing node_id or class_type — {spec}")
            continue
        if nid in workflow:
            node_errors.append(f"add: node '{nid}' already exists.")
            continue
        raw_inputs = spec.get("inputs", {})
        node_inputs = json.loads(raw_inputs) if isinstance(raw_inputs, str) else raw_inputs
        new_node: dict = {"class_type": cls, "inputs": node_inputs}
        if spec.get("meta_title"):
            new_node["_meta"] = {"title": spec["meta_title"]}
        workflow[nid] = new_node
        added.append(nid)

    # ── Patch ─────────────────────────────────────────────────────────────────
    if not isinstance(patch_list, list):
        return json.dumps({"status": "error", "error": "patches must be a JSON array of patch objects."})

    for i, patch in enumerate(patch_list):
        nid = str(patch.get("node_id", ""))
        if nid not in workflow:
            node_errors.append(f"patch {i}: node '{nid}' not found in workflow.")
            continue

        node = workflow[nid]

        if "class_type" in patch:
            node["class_type"] = patch["class_type"]
            applied.append(f"Node {nid}: class_type → {patch['class_type']}")

        if "widget_values_index" in patch:
            idx = int(patch["widget_values_index"])
            wv = node.get("widget_values", node.get("widgets_values"))
            wv_key = "widget_values" if "widget_values" in node else "widgets_values"
            if isinstance(wv, list) and 0 <= idx < len(wv):
                wv[idx] = patch["value"]
                node[wv_key] = wv
                applied.append(f"Node {nid}: {wv_key}[{idx}] → {patch['value']!r}")
            else:
                node_errors.append(f"patch {i}: node '{nid}' has no {wv_key}[{idx}].")
            continue

        inp_name = patch.get("input_name")
        if inp_name:
            if "inputs" not in node:
                node["inputs"] = {}
            value = patch["value"]
            # A LoadImage reference must carry the 'agent/' input subfolder that
            # upload_image stages files into; a bare filename makes ComfyUI look
            # in the input root and reject it as "Invalid image file".
            if inp_name == "image" and node.get("class_type") == "LoadImage" and isinstance(value, str):
                value = _agent_input_ref(value)
            node["inputs"][inp_name] = value
            val_repr = repr(value)
            if len(val_repr) > 80:
                val_repr = val_repr[:77] + "..."
            applied.append(f"Node {nid}.inputs.{inp_name} → {val_repr}")

    # ── Save ──────────────────────────────────────────────────────────────────
    path = _save_edit(workflow, workflow_path)

    # Update patch failure guard
    global _patch_fail_count, _patch_last_workflow_path
    _patch_last_workflow_path = path

    if node_errors:
        _patch_fail_count += 1
        print(
            f"[update_workflow] Failure {_patch_fail_count}/{_PATCH_FAIL_LIMIT}: "
            f"{len(node_errors)} error(s)."
        )
        if _patch_fail_count >= _PATCH_FAIL_LIMIT:
            debug_name = f"{Path(workflow_path).stem}_update_debug"
            debug_path = _save_workflow(workflow, name=debug_name)
            print(f"[update_workflow] LIMIT REACHED — debug snapshot: {debug_path}")
            return json.dumps({
                "status": "error",
                "workflow_path": path,
                "removed_nodes": removed,
                "added_nodes": added,
                "applied_patches": applied,
                "node_errors": node_errors,
                "valid": False,
                "local_errors": [],
                "server_errors": {},
                "patch_failure_limit_reached": True,
                "debug_workflow_path": debug_path,
                "message": (
                    f"update_workflow has failed {_PATCH_FAIL_LIMIT} times. "
                    f"Debug snapshot saved to: {debug_path}. "
                    "STOP — do not call update_workflow again. "
                    "Report the debug_workflow_path to the user and ask for guidance."
                ),
            })
    else:
        _patch_fail_count = 0

    # ── Validate ──────────────────────────────────────────────────────────────
    local_errors: list[str] = []
    server_errors: dict = {}
    missing_models: list[str] = []
    # Scalars found sitting in link-only sockets, deleted by the hardening
    # pass. Reported as repairs, never as errors: the value was a widget-
    # alignment leftover and removing it IS the correct graph.
    stripped_inputs: list[str] = []
    # Objects written into a dynamic-combo input, unpacked into the dotted keys
    # ComfyUI binds. Reported as repairs like `stripped_inputs`: the file on disk
    # is corrected and re-saved below, so "valid" stays a statement about what
    # the executor will actually submit.
    reshaped_inputs: list[str] = []
    # What the graph looked like before hardening, so the re-save below can tell
    # whether the pass actually changed it.
    _snapshot_before = json.dumps(workflow, sort_keys=True, default=str)

    try:
        all_nodes = _get_object_info()
    except Exception:
        all_nodes = {}

    node_ids = set(workflow.keys())

    for nid, node in workflow.items():
        cls = node.get("class_type", "")
        if not cls:
            local_errors.append(f"Node {nid}: missing 'class_type'.")
            continue
        if all_nodes and cls not in all_nodes:
            local_errors.append(f"Node {nid}: unknown class_type '{cls}'.")
            continue
        node_info = all_nodes.get(cls, {})
        required = node_info.get("input", {}).get("required", {})
        optional = node_info.get("input", {}).get("optional", {})
        node_inputs = node.get("inputs", {})
        # Inject widget/combo defaults + snap invalid combo values (collecting any
        # un-installed model into missing_models), and strip scalars left in
        # link-only sockets (collecting them into stripped_inputs); what remains
        # is a genuinely-missing connection input (needs real wiring).
        _reshaped_here: list[str] = []
        for _missing in _harden_node_inputs(node, required, missing_models, optional,
                                            stripped_inputs, _reshaped_here):
            local_errors.append(f"Node {nid} ({cls}): missing required input '{_missing}'.")
        reshaped_inputs.extend(f"Node {nid} ({cls}): {n}" for n in _reshaped_here)
        # A shape that cannot be corrected mechanically is a real error. It would
        # otherwise fail at execution with a message naming neither the input nor
        # the shape it wanted, which is the one failure worth catching here.
        from agenty_core.tools.assembly_deterministic import (  # noqa: PLC0415
            flatten_dynamic_combos as _flatten,
        )
        for _bad in _flatten(node, required, optional):
            local_errors.append(f"Node {nid} ({cls}): {_bad}.")
        for _key in _undeclared_wires(node, required, optional, node_info):
            local_errors.append(_undeclared_wire_error(nid, cls, _key, node, required, optional))
        node_inputs = node.get("inputs", {})
        for inp_name, inp_val in node_inputs.items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_id = str(inp_val[0])
                if src_id not in node_ids:
                    local_errors.append(
                        f"Node {nid} ({cls}): input '{inp_name}' references "
                        f"non-existent node '{src_id}'."
                    )

    # The save above happens BEFORE the hardening pass, so nothing that pass
    # fixed had been reaching disk: the validation result described a graph the
    # executor would never see. That gap shipped CreateVideo.bit_depth=16 to
    # ComfyUI — the clamp had already corrected it to 10 in memory and the
    # corrected value was discarded — and the whole video branch was rejected at
    # submission while the run reported success. Re-save whenever the pass
    # changed anything, so `valid: true` is a statement about the file.
    if _snapshot_before != json.dumps(workflow, sort_keys=True, default=str):
        path = _save_edit(workflow, workflow_path)
        _patch_last_workflow_path = path

    # Side-effect-free server validation (skips while ComfyUI is busy; never a
    # blanket interrupt / queue clear that would kill a real render). See
    # _server_validate.
    server_errors = _server_validate(workflow)

    is_valid = len(local_errors) == 0 and len(server_errors) == 0
    all_errors = node_errors + local_errors

    # Reported on every edit, because this is the moment the wiring was in the
    # caller's hands: a node added in one call and meant to be wired in the next
    # shows up here until it is, which is exactly the reminder that was missing.
    # Never an error — see the note in validate_workflow.
    _dead, _dead_lines = _dead_node_warnings(workflow, all_nodes)

    _note = _moved_note(workflow_path, path)
    return json.dumps({
        "status": "ok" if (is_valid and not node_errors) else "error",
        "workflow_path": path,
        "removed_nodes": removed,
        "cleaned_links": cleaned_links,
        "added_nodes": added,
        "applied_patches": applied + [f"stripped {s} (link-only socket)"
                                      for s in stripped_inputs],
        "node_errors": node_errors,
        "valid": is_valid,
        "local_errors": local_errors,
        "reshaped_inputs": reshaped_inputs,
        "server_errors": server_errors,
        **({"dead_nodes": _dead, "warnings": _dead_lines} if _dead else {}),
        **({"note": _note} if _note else {}),
    })


# ---------------------------------------------------------------------------
# Agent asset-folder routing — keep all agent-managed inputs/outputs under an
# "agent/" subfolder of ComfyUI's input/output dirs instead of cluttering the
# roots.  Applied deterministically when a brainbriefing is bound to a workflow.
# ---------------------------------------------------------------------------

_AGENT_SUBFOLDER = "agent"


def _agent_input_ref(filename: str) -> str:
    """Return the bare LoadImage filename (input dir root, no subfolder).

    ``upload_image`` / ``download_image`` now stage agent inputs flat in ComfyUI's
    input dir root, because LoadImage on some ComfyUI builds cannot load files
    from input subdirectories. Any leftover ``agent/`` (or ``agent/references/``)
    prefix from a briefing is flattened to the basename so the reference points at
    the root file. Absolute / drive-qualified paths are left untouched.
    """
    f = (filename or "").replace("\\", "/").strip()
    if not f:
        return filename
    if f.startswith("/") or (len(f) > 1 and f[1] == ":"):
        return filename  # absolute path — leave as-is
    return f.rsplit("/", 1)[-1]  # bare filename; agent inputs live in the input root


# Media-kind buckets under agent/.  Every agent-generated output is routed into
# one of a small, predictable set of folders keyed by the *kind* of media the
# saver node produces (not by task type), so files are easy to find and script
# tooling can rely on the layout:  agent/images, agent/videos, agent/audio,
# agent/models.
_MEDIA_BUCKETS = ("images", "videos", "audio", "models")
_VIDEO_SAVERS = {"VHS_VideoCombine", "SaveVideo", "SaveWEBM", "VHS_SaveVideo"}
_AUDIO_SAVERS = {"SaveAudio", "VHS_SaveAudio", "SaveAudioMP3", "SaveAudioOpus"}
_MODEL_SAVERS = {"SaveGLB", "SaveGLTF", "Save3DModel", "Save3D"}
_BUCKET_DEFAULT_STEM = {"images": "image", "videos": "video", "audio": "audio", "models": "model"}


def _agent_media_bucket(class_type: str) -> str:
    """Map an output-saver node's ``class_type`` to a media-kind bucket name.

    Defaults to ``images`` for the common ``SaveImage`` / ``PreviewImage`` case
    (and anything unrecognised — an image is the safe fallback).
    """
    ct = class_type or ""
    if ct in _AUDIO_SAVERS or "Audio" in ct:
        return "audio"
    if ct in _VIDEO_SAVERS or "Video" in ct:
        return "videos"
    if ct in _MODEL_SAVERS or ct.startswith("Save3D") or ct.startswith("TripoSG"):
        return "models"
    return "images"


def _agent_output_prefix(class_type: str = "", path_or_prefix: str = "") -> str:
    """Route a saver node's ``filename_prefix`` under ``agent/<media-kind>/``.

    Buckets by media kind derived from *class_type* (images/videos/audio/models),
    then keeps a descriptive filename stem so results stay recognisable:
    ``(SaveImage, "W:/.../output/image_generation")`` → ``"agent/images/image_generation"``.
    Files therefore land in exactly two folders for the common cases —
    ``agent/images/`` and ``agent/videos/`` — regardless of task type.
    """
    bucket = _agent_media_bucket(class_type)
    p = (path_or_prefix or "").replace("\\", "/").strip().strip("/")
    stem = p.rsplit("/", 1)[-1] if p else ""
    # Drop a stem that is itself a bucket / generic label so we don't produce
    # noise like agent/images/images; fall back to the media-kind default.
    if not stem or stem in _MEDIA_BUCKETS or stem in ("agent", "output"):
        stem = _BUCKET_DEFAULT_STEM.get(bucket, "output")
    return f"{_AGENT_SUBFOLDER}/{bucket}/{stem}"


@tool
def apply_brainbriefing(workflow_path: str, brainbriefing_json: str) -> str:
    """Apply a brainbriefing to a loaded workflow template in one atomic step.

    Performs all standard template patching programmatically without requiring
    the agent to construct individual patch objects:

    1. Replaces filenames in input nodes  (brainbriefing ``input_nodes``).
    2. Replaces positive / negative prompts (brainbriefing ``prompt`` +
       ``positive_prompt_node_id``).
    3. Updates output node ``filename_prefix`` values (brainbriefing
       ``output_nodes``).
    4. Sets resolution width / height where literal inputs exist (brainbriefing
       ``resolution_width`` / ``resolution_height``).
    5. Validates the result locally and via the ComfyUI server.

    Returns ``status: "ok"`` with the saved ``workflow_path`` on success, or
    ``status: "error"`` with a ``problems`` list describing every issue found.

    Args:
        workflow_path: File path to the workflow JSON (from
            ``get_workflow_template``).
        brainbriefing_json: The full brainbriefing JSON string (or dict).
    """
    # ── Load workflow ─────────────────────────────────────────────────────────
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"status": "error", "problems": [f"Cannot load workflow: {e}"]})

    # ── Parse brainbriefing ───────────────────────────────────────────────────
    try:
        bb = json.loads(brainbriefing_json) if isinstance(brainbriefing_json, str) else brainbriefing_json
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "problems": [f"Invalid brainbriefing JSON: {e}"]})

    applied: list[str] = []
    problems: list[str] = []

    # Refresh /object_info before validating input images: any image staged during
    # the researcher run (upload_image) is absent from the session-cached
    # object_info, so LoadImage's valid-file combo would be stale and the
    # snap-to-valid step below would replace the user's image with the template's
    # default ("reproduces the template"). Only needed when the briefing wires
    # input images.
    if bb.get("input_nodes"):
        reset_object_info_cache()

    # Strip pure-annotation nodes (Note / MarkdownNote) that ComfyUI rejects.
    for _nid in _strip_annotation_nodes(workflow):
        applied.append(f"Removed annotation node {_nid}")
    # Bypass + remove Reroute/Primitive passthrough nodes (rewire links through
    # them) — ComfyUI's /prompt API rejects them as unknown class_type.
    _rr = _strip_reroute_nodes(workflow)
    if _rr:
        applied.append(f"Bypassed {len(_rr)} reroute/primitive node(s): {', '.join(_rr)}")

    # ── 1. Input nodes: replace filenames ─────────────────────────────────────
    for inp in bb.get("input_nodes", []):
        nid = str(inp.get("node_id", ""))
        filename = inp.get("filename") or inp.get("path", "")
        slot = inp.get("slot", "image")
        # A briefing may over-specify or mis-reference input nodes; skip broken
        # entries (leaving the template's own default) rather than failing — the
        # ComfyUI server validation is the real backstop.
        if not nid:
            applied.append("input_nodes: skipped entry with no node_id")
            continue
        if nid not in workflow:
            applied.append(f"input_nodes: skipped '{nid}' (not in workflow)")
            continue
        if not filename:
            applied.append(f"input_nodes: skipped '{nid}' (no filename)")
            continue
        node = workflow[nid]
        if "inputs" not in node:
            node["inputs"] = {}
        # Route the LoadImage reference under the input dir's 'agent/' subfolder
        # (matches where upload_image / download_image stage inputs).
        ref = _agent_input_ref(filename)
        node["inputs"][slot] = ref
        applied.append(f"Node {nid}.inputs.{slot} → {ref!r}")

    # ── 1b. Rebind LoadImage placeholders ─────────────────────────────────────
    # Templates ship a sample image name and researchers often omit input_nodes,
    # so a LoadImage left on its placeholder 400s at submission. Bind any such
    # node to a real available input (preferring the staged agent/ images).
    try:
        applied.extend(_rebind_placeholder_images(workflow, _get_object_info()))
    except Exception:  # noqa: BLE001
        pass

    # ── 2. Prompts ────────────────────────────────────────────────────────────
    prompt_block = bb.get("prompt", {})
    positive_text = prompt_block.get("positive", "")
    negative_text = prompt_block.get("negative", "")

    # ── 2a. prompt_nodes: explicit per-node prompt injection (preferred path) ──
    prompt_nodes = bb.get("prompt_nodes", [])
    handled_nids: set[str] = set()
    for pn in prompt_nodes:
        pn_nid = str(pn.get("node_id", ""))
        role = pn.get("role", "positive")  # "positive" | "negative"
        slot = pn.get("slot", "text")
        if not pn_nid:
            problems.append("prompt_nodes entry missing node_id")
            continue
        if pn_nid not in workflow:
            problems.append(f"prompt_nodes: node '{pn_nid}' not found in workflow")
            continue
        text = positive_text if role == "positive" else negative_text
        if not text:
            applied.append(f"prompt_nodes: node '{pn_nid}' role='{role}' — no text provided (skipped)")
            continue
        node = workflow[pn_nid]
        if "inputs" not in node:
            node["inputs"] = {}
        node["inputs"][slot] = text
        applied.append(f"Node {pn_nid}.inputs.{slot} → ({role} prompt, {len(text)} chars)")
        handled_nids.add(pn_nid)

    # ── 2b. positive_prompt_node_id (explicit), then heuristic ───────────────
    # Only runs for nodes not already handled by prompt_nodes above.
    positive_injected = any(
        pn.get("role", "positive") == "positive" and str(pn.get("node_id", "")) in handled_nids
        for pn in prompt_nodes
    )
    # Treat null / "None" / "" as "not provided" (a common researcher omission)
    # rather than a literal node id to look up — that spurious lookup used to
    # fail every one-shot apply and force the LLM into manual update_workflow.
    pos_nid = str(bb.get("positive_prompt_node_id") or "").strip()
    if pos_nid.lower() == "none":
        pos_nid = ""
    if positive_text and not positive_injected and pos_nid:
        if pos_nid not in workflow:
            # Malformed / stale id (e.g. ':0') — fall through to the heuristic
            # below rather than failing.
            applied.append(f"positive_prompt_node_id '{pos_nid}' not in workflow — using heuristic")
        else:
            node = workflow[pos_nid]
            node.setdefault("inputs", {})["text"] = positive_text
            applied.append(f"Node {pos_nid}.inputs.text → (positive prompt, {len(positive_text)} chars)")
            handled_nids.add(pos_nid)
            positive_injected = True
    # Heuristic: no explicit target — inject into the unambiguous positive
    # text-conditioning node (a CLIPTextEncode-style node that is not negative).
    # Prefer one titled "positive"; else the sole candidate. If ambiguous/none,
    # record a problem so apply returns error and the caller can fall back.
    if positive_text and not positive_injected:
        preferred: str | None = None
        cands: list[str] = []
        for nid, node in workflow.items():
            if not isinstance(node, dict) or nid in handled_nids:
                continue
            cls = node.get("class_type", "").lower()
            title = (node.get("_meta") or {}).get("title", "").lower()
            if "negative" in title:
                continue
            is_text_cond = ("cliptextencode" in cls) or (
                ("text" in cls or "prompt" in cls)
                and ("encode" in cls or "clip" in cls or "condition" in cls)
            )
            if not is_text_cond:
                continue
            if "positive" in title:
                preferred = nid
            cands.append(nid)
        # Prefer a 'positive'-titled node; else the sole candidate; else the
        # lowest-id candidate (the positive text node is conventionally first).
        target = preferred or (cands[0] if len(cands) == 1 else
                               (min(cands, key=lambda n: (len(str(n)), str(n))) if cands else None))
        if target:
            node = workflow[target]
            node.setdefault("inputs", {})["text"] = positive_text
            applied.append(f"Node {target}.inputs.text → (positive prompt, heuristic, {len(positive_text)} chars)")
            handled_nids.add(target)
            positive_injected = True
        elif cands:
            # Candidates existed but none could be disambiguated — a real
            # ambiguity the LLM brain should resolve.
            problems.append(
                "positive prompt: no unambiguous target node found "
                "(provide prompt_nodes or positive_prompt_node_id)"
            )
        else:
            # No CLIPTextEncode-style node. Before treating this as prompt-less,
            # try API / unified-text nodes that carry a literal ``prompt`` (or
            # ``text``) STRING input — e.g. GeminiNanoBanana2, GeminiImage2Node,
            # IdeogramV3. The LLM brain frequently reconstructs the briefing and
            # drops its ``prompt_nodes`` / sets ``positive_prompt_node_id`` to null,
            # so the target node must be found here or the prompt is lost.
            try:
                _oi = _get_object_info()
            except Exception:  # noqa: BLE001
                _oi = {}
            _api_cands: list[tuple[str, str]] = []  # (node_id, input_name)
            for _nid, _node in workflow.items():
                if not isinstance(_node, dict) or _nid in handled_nids:
                    continue
                if "negative" in ((_node.get("_meta") or {}).get("title", "")).lower():
                    continue
                _in = (_oi.get(_node.get("class_type", ""), {}) or {}).get("input", {}) or {}
                _allin = {**(_in.get("optional") or {}), **(_in.get("required") or {})}
                for _pname in ("prompt", "text"):  # exact keys — never system_prompt
                    _pspec = _allin.get(_pname)
                    if isinstance(_pspec, list) and _pspec and _pspec[0] == "STRING":
                        _api_cands.append((_nid, _pname))
                        break
            if _api_cands:
                # Prefer a ``prompt`` input over ``text``, then the lowest node id.
                _pref = [c for c in _api_cands if c[1] == "prompt"] or _api_cands
                _tnid, _tname = min(_pref, key=lambda c: (len(str(c[0])), str(c[0])))
                workflow[_tnid].setdefault("inputs", {})[_tname] = positive_text
                applied.append(
                    f"Node {_tnid}.inputs.{_tname} → (positive prompt, api-node, "
                    f"{len(positive_text)} chars)"
                )
                handled_nids.add(_tnid)
                positive_injected = True
            else:
                # Genuinely prompt-less (background removal, upscale, depth, …).
                applied.append(
                    "positive prompt: workflow has no text-conditioning node — "
                    "prompt not applicable (skipped)"
                )

    # Negative prompt: find a node with "negative" in its title that has a text input
    if negative_text and not any(
        str(pn.get("node_id", "")) in handled_nids and pn.get("role") == "negative"
        for pn in prompt_nodes
    ):
        neg_nid: str | None = None
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            title = (node.get("_meta") or {}).get("title", "").lower()
            cls = node.get("class_type", "").lower()
            # Match nodes that look like text-conditioning nodes with "negative" in title
            if "negative" in title and (
                "clip" in cls or "text" in cls or "prompt" in cls or "condition" in cls
            ):
                neg_nid = nid
                break
        if neg_nid:
            node = workflow[neg_nid]
            if "inputs" not in node:
                node["inputs"] = {}
            node["inputs"]["text"] = negative_text
            applied.append(f"Node {neg_nid}.inputs.text → (negative prompt, {len(negative_text)} chars)")
        else:
            # Non-fatal: many modern pipelines have no negative prompt node
            applied.append("negative prompt: no matching node found (skipped)")

    # ── 3. Output nodes: route filename_prefix under the 'agent/' subfolder ────
    for out in bb.get("output_nodes", []):
        nid = str(out.get("node_id", ""))
        output_path = out.get("output_path", "")
        # Skip broken output-node references (the template keeps its own output
        # node + default prefix); server validation is the backstop.
        if not nid:
            applied.append("output_nodes: skipped entry with no node_id")
            continue
        if nid not in workflow:
            applied.append(f"output_nodes: skipped '{nid}' (not in workflow)")
            continue
        node = workflow[nid]
        if "inputs" not in node:
            node["inputs"] = {}
        # Keep all agent-generated outputs under <output_dir>/agent/<media-kind>/.
        # The bucket (images/videos/audio/models) is derived from the saver node's
        # class_type; a descriptive stem from output_path (or the node's existing
        # filename_prefix) is kept as the filename base (e.g. agent/images/image_edit).
        existing_prefix = node["inputs"].get("filename_prefix", "")
        prefix = _agent_output_prefix(node.get("class_type", ""), output_path or existing_prefix)
        node["inputs"]["filename_prefix"] = prefix
        applied.append(f"Node {nid}.inputs.filename_prefix → {prefix!r}")

    # ── 4. Resolution ─────────────────────────────────────────────────────────
    res_w = _coerce_dim(bb.get("resolution_width"))
    res_h = _coerce_dim(bb.get("resolution_height"))
    if res_w or res_h:
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            changed = False
            if res_w and "width" in inputs and not isinstance(inputs["width"], list):
                inputs["width"] = int(res_w)
                changed = True
            if res_h and "height" in inputs and not isinstance(inputs["height"], list):
                inputs["height"] = int(res_h)
                changed = True
            if changed:
                applied.append(
                    f"Node {nid}: resolution → {res_w or '(unchanged)'}×{res_h or '(unchanged)'}"
                )

    # ── 5. ModelSamplingFlux: auto-inject required inputs ─────────────────────
    # This node fails ComfyUI validation if any of the four inputs are missing.
    # Defaults match the ModelSamplingFlux requirements in the assemble-from-template
    # skill; width/height come from brainbriefing.
    _MSF_DEFAULTS = {"max_shift": 1.15, "base_shift": 0.5, "width": 1024, "height": 1024}
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "ModelSamplingFlux":
            continue
        if "inputs" not in node:
            node["inputs"] = {}
        msf_inputs = node["inputs"]
        injected: list[str] = []
        for key, default in _MSF_DEFAULTS.items():
            if key not in msf_inputs or isinstance(msf_inputs[key], list):
                # Use brainbriefing resolution if available, otherwise default
                if key == "width" and res_w:
                    msf_inputs[key] = int(res_w)
                elif key == "height" and res_h:
                    msf_inputs[key] = int(res_h)
                else:
                    msf_inputs[key] = default
                injected.append(key)
        if injected:
            applied.append(f"Node {nid} (ModelSamplingFlux): auto-injected {injected}")

    # ── 6. Test-mode latent clamp (AGENTY_MAX_DIM) ────────────────────────────
    # Reliability testing on power-limited hardware (an RTX 5090 that hard-crashed
    # under sustained load): hard-cap every literal width/height so the generation
    # latent can never exceed sub-HD dimensions, regardless of what the researcher
    # requested or the template defaults to. Purely a downward clamp; only active
    # when AGENTY_MAX_DIM is set (the reliability sweep sets it).
    _max_dim = os.environ.get("AGENTY_MAX_DIM")
    if _max_dim:
        try:
            cap = int(_max_dim)
        except ValueError:
            cap = 0
        if cap > 0:
            for nid, node in workflow.items():
                if not isinstance(node, dict):
                    continue
                inputs = node.get("inputs", {})
                if not isinstance(inputs, dict):
                    continue
                for dim in ("width", "height"):
                    v = inputs.get(dim)
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, (int, float)) and v > cap:
                        inputs[dim] = cap
                        applied.append(f"Node {nid}: {dim} clamped {int(v)}→{cap} (AGENTY_MAX_DIM)")

    # ── 7. Ensure a terminal output node ──────────────────────────────────────
    try:
        _oi_out = _get_object_info()
    except Exception:  # noqa: BLE001
        _oi_out = {}
    _sv = _ensure_output_node(workflow, _oi_out)
    if _sv:
        applied.append(f"Synthesized SaveVideo node {_sv} for terminal VIDEO output")

    # ── Validate (harden) ─────────────────────────────────────────────────────
    local_errors: list[str] = []
    server_errors: dict = {}
    missing_models: list[str] = []
    # Scalars found sitting in link-only sockets, deleted by the hardening
    # pass. Reported as repairs, never as errors: the value was a widget-
    # alignment leftover and removing it IS the correct graph.
    stripped_inputs: list[str] = []

    try:
        all_nodes = _get_object_info()
    except Exception:
        all_nodes = {}

    # Wire dangling required connection inputs (e.g. a VAE/mask boundary the
    # subgraph flattener dropped) to the graph's unique same-type producer,
    # before hardening would flag them as genuinely missing.
    for _w in _autowire_dangling_inputs(workflow, all_nodes):
        applied.append(_w)

    node_ids = set(workflow.keys())

    for nid, node in workflow.items():
        cls = node.get("class_type", "")
        if not cls:
            local_errors.append(f"Node {nid}: missing 'class_type'.")
            continue
        if all_nodes and cls not in all_nodes:
            local_errors.append(f"Node {nid}: unknown class_type '{cls}'.")
            continue
        node_info = all_nodes.get(cls, {})
        required = node_info.get("input", {}).get("required", {})
        optional = node_info.get("input", {}).get("optional", {})
        node_inputs = node.get("inputs", {})
        # Inject widget/combo defaults + snap invalid combo values (collecting any
        # un-installed model into missing_models), and strip scalars left in
        # link-only sockets (collecting them into stripped_inputs); what remains
        # is a genuinely-missing connection input (needs real wiring).
        for _missing in _harden_node_inputs(node, required, missing_models, optional,
                                            stripped_inputs):
            local_errors.append(f"Node {nid} ({cls}): missing required input '{_missing}'.")
        for _key in _undeclared_wires(node, required, optional, node_info):
            local_errors.append(_undeclared_wire_error(nid, cls, _key, node, required, optional))
        for inp_name, inp_val in node_inputs.items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_id = str(inp_val[0])
                if src_id not in node_ids:
                    local_errors.append(
                        f"Node {nid} ({cls}): input '{inp_name}' references "
                        f"non-existent node '{src_id}'."
                    )

    # Side-effect-free server validation (skips while ComfyUI is busy; never a
    # blanket interrupt / queue clear that would kill a real render). See
    # _server_validate.
    server_errors = _server_validate(workflow)

    # Save AFTER hardening so the file the executor submits reflects every
    # deterministic fix (reroutes bypassed, images rebound, combos snapped,
    # widget defaults injected) — not the raw pre-harden template.
    path = _save_edit(workflow, workflow_path)

    all_problems = problems + local_errors
    is_valid = len(all_problems) == 0 and len(server_errors) == 0

    return json.dumps({
        "status": "ok" if is_valid else "error",
        "workflow_path": path,
        "applied": applied + [f"stripped {s} (link-only socket)"
                              for s in stripped_inputs],
        "problems": all_problems,
        "valid": is_valid,
        "local_errors": local_errors,
        "server_errors": server_errors,
        "missing_models": missing_models,
    })


@tool
def _undeclared_wire_error(nid, cls: str, key: str, node: dict, required: dict,
                           optional: dict | None) -> str:
    """The error for a wire under a key *cls* does not declare, naming what it takes."""
    specs = {**(optional or {}), **(required or {})}
    keys, _open = _declared_input_keys(node, required, optional)
    groups = sorted(g for g in keys if _is_autogrow_group(specs.get(g)))
    shown = [k for k in sorted(keys)
             if k not in groups and not any(k.startswith(g + ".") for g in groups)]
    for g in groups:
        slots = _autogrow_slot_names(specs[g])
        if slots:
            shown.append(f"{g}.{slots[0]}..{slots[-1]}")
    more = f", ... ({len(shown) - 24} more)" if len(shown) > 24 else ""
    return (f"Node {nid} ({cls}): '{key}' is wired but is not an input of {cls}. "
            f"ComfyUI passes a wire on under any key, so the run would fail with "
            f"\"unexpected keyword argument '{key}'\". Inputs it takes: "
            f"{', '.join(shown[:24])}{more}.")


def normalize_autogrow_wires(workflow: dict) -> list[str]:
    """Move every wire written under a bare autogrow slot name to its dotted address.

    In place, on a prompt about to be submitted: the last line of defence for a
    graph that reached the executor without passing through update_workflow.
    Returns the notes, ``[]`` when ComfyUI's node schemas cannot be read.
    """
    try:
        all_nodes = _get_object_info()
    except Exception:  # noqa: BLE001
        return []
    notes: list[str] = []
    for nid, node in (workflow or {}).items():
        if not isinstance(node, dict):
            continue
        spec = (all_nodes.get(node.get("class_type", "")) or {}).get("input", {}) or {}
        for note in _normalize_autogrow_keys(node, spec.get("required") or {},
                                             spec.get("optional") or {}):
            notes.append(f"Node {nid} ({node.get('class_type')}): {note}")
    return notes


def _spec_type(spec) -> str:
    typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    if isinstance(typ, list):
        return "COMBO"
    return typ if isinstance(typ, str) else ""


def _types_compatible(given, wanted) -> bool:
    """Would a *given* output type satisfy a socket that *wanted* one? Unknown is yes."""
    if not given or not wanted:
        return True
    g = {t.strip() for t in str(given).split(",")}
    w = {t.strip() for t in str(wanted).split(",")}
    return "*" in g or "*" in w or bool(g & w)


def _connection_keys(node: dict, required: dict, optional: dict | None) -> list[tuple[str, str]]:
    """``[(key, type)]`` for every socket in force on *node*, autogrow slots expanded,
    required before optional."""
    req_sub, opt_sub = _dynamic_sub_specs(node, required, optional)
    out: list[tuple[str, str]] = []
    for name, spec in {**(required or {}), **req_sub, **(optional or {}), **opt_sub}.items():
        if _is_autogrow_group(spec):
            gtype = _autogrow_info(spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {})["grown_type"]
            out.extend((f"{name}.{slot}", gtype) for slot in _autogrow_slot_names(spec))
        elif not _is_widget_spec(spec) and _spec_type(spec):
            out.append((name, _spec_type(spec)))
    return out


def _input_wire_type(node: dict, name: str, all_nodes: dict) -> str:
    """The type *node*'s input *name* takes, or "" when the schema cannot say."""
    spec = (all_nodes.get(node.get("class_type", "")) or {}).get("input", {}) or {}
    for key, typ in _connection_keys(node, spec.get("required") or {}, spec.get("optional") or {}):
        if key == name:
            return typ
    return ""


def _output_type(workflow: dict, wire, all_nodes: dict) -> str:
    """The type of the output *wire* comes from, or "" when unknown."""
    producer = workflow.get(str(wire[0])) or {}
    outs = (all_nodes.get(producer.get("class_type", "")) or {}).get("output") or []
    slot = int(wire[1])
    return outs[slot] if 0 <= slot < len(outs) and isinstance(outs[slot], str) else ""


def _check_replacement(workflow: dict, old_id: str, new_node: dict, all_nodes: dict,
                       remap: bool) -> tuple[list, list, list]:
    """Can *new_node* stand in for node *old_id*? ``(problems, dropped, remapped)``.

    Changes *new_node*'s inputs only to carry out what it reports: literals the new
    class has no input for are removed, and with *remap* orphaned wires are moved.
    """
    cls = new_node["class_type"]
    info = all_nodes.get(cls)
    if not info:
        return [f"'{cls}' is not an installed node class."], [], []
    spec = info.get("input", {}) or {}
    required, optional = spec.get("required") or {}, spec.get("optional") or {}
    inputs = new_node["inputs"]
    _normalize_autogrow_keys(new_node, required, optional)
    problems: list[str] = []

    # Every wire leaving the old node must find its output slot, of its type.
    outs = info.get("output") or []
    for nid, node in workflow.items():
        if nid == old_id or not isinstance(node, dict):
            continue
        for name, val in (node.get("inputs") or {}).items():
            if not (_is_wire_value(val) and str(val[0]) == old_id):
                continue
            slot = int(val[1])
            wanted = _input_wire_type(node, name, all_nodes)
            if slot >= len(outs):
                problems.append(f"node {nid}.{name} takes output {slot} and {cls} has "
                                f"only {len(outs)} output(s).")
            elif not _types_compatible(outs[slot], wanted):
                problems.append(f"node {nid}.{name} takes a {wanted} from output {slot}, "
                                f"where {cls} gives {outs[slot]}.")

    # Every wire into the old node needs somewhere to go on the new one.
    keys, open_prefixes = _declared_input_keys(new_node, required, optional)
    homeless = [k for k in inputs if k not in keys and not k.startswith(open_prefixes)]
    dropped = [k for k in homeless if not _is_wire_value(inputs[k])]
    wires = [k for k in homeless if _is_wire_value(inputs[k])]
    for k in dropped:
        del inputs[k]
    remapped: list[str] = []
    if wires and remap:
        free = [(k, t) for k, t in _connection_keys(new_node, required, optional) if k not in inputs]
        for k in list(wires):
            given = _output_type(workflow, inputs[k], all_nodes)
            match = next((f for f in free if given and _types_compatible(given, f[1])), None)
            if match is None:
                continue
            free.remove(match)
            inputs[match[0]] = inputs.pop(k)
            remapped.append(f"{k} -> {match[0]}")
            wires.remove(k)
    if wires:
        problems.append(
            f"{cls} has no input for the wire(s) into {', '.join(repr(w) for w in wires)}"
            + ("" if remap else "; remap_inputs=true carries wires over by type, if the "
                                "class genuinely has to change")
            + ".")
    return problems, dropped, remapped


def replace_node(workflow_path: str, old_node_id: str, new_class_type: str,
                 new_node_id: str = "", meta_title: str = "",
                 remap_inputs: bool = False) -> str:
    """Replace a node with a different class_type, keeping its wiring.

    For a node whose class is missing, renamed or retired and needs an equivalent.
    It keeps the node's job, so it refuses a swap the wiring cannot survive -
    writing nothing and saying why:

    * a wire leaving the node from an output the new class lacks, or of another
      type than the consumer takes;
    * a wire into the node under an input the new class does not have - unless
      *remap_inputs* is true, which carries such wires to the new class's free
      inputs of the same type, in order (``BatchImagesNode`` -> ``ImageBatch``).

    Literal values the new class has no input for are dropped and listed. Never a
    way round an input that would not wire: fix that input with update_workflow.

    Args:
        workflow_path: File path to the workflow JSON.
        old_node_id: ID of the node to remove.
        new_class_type: class_type for the replacement node.
        new_node_id: ID for the new node.  Defaults to old_node_id (in-place swap).
        meta_title: Optional display title for the new node.
        remap_inputs: Carry wires with no same-named input over by type, in order.
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"status": "error", "error": f"Cannot load workflow: {e}"})

    old_id = str(old_node_id)
    if old_id not in workflow:
        return json.dumps({"status": "error", "error": f"Node '{old_id}' not found in workflow."})

    new_id = str(new_node_id) if new_node_id else old_id

    if new_id != old_id and new_id in workflow:
        return json.dumps({"status": "error", "error": f"Node ID '{new_id}' already exists."})

    old_node = workflow[old_id]

    # Build the replacement node, inheriting all inputs from the old node.
    import copy
    new_node: dict = {
        "class_type": new_class_type,
        "inputs": copy.deepcopy(old_node.get("inputs", {})),
    }
    if meta_title:
        new_node["_meta"] = {"title": meta_title}
    elif "_meta" in old_node:
        new_node["_meta"] = copy.deepcopy(old_node["_meta"])

    try:
        all_nodes = _get_object_info()
    except Exception:  # noqa: BLE001 — no schemas: swap as asked, as before
        all_nodes = {}
    problems, dropped, remapped = ([], [], [])
    if all_nodes:
        problems, dropped, remapped = _check_replacement(
            workflow, old_id, new_node, all_nodes, bool(remap_inputs))
    if problems:
        return json.dumps({
            "status": "error",
            "error": (f"Not replacing node {old_id} ({old_node.get('class_type')}) with "
                      f"{new_class_type}: " + " ".join(problems)),
            "hint": ("replace_node swaps a node for an equivalent one. If this node's "
                     "inputs are what fails, fix them with update_workflow instead of "
                     "changing what the node does."),
        })

    # Insert the new node (if new_id == old_id this temporarily overwrites it).
    if new_id != old_id:
        del workflow[old_id]
    workflow[new_id] = new_node

    # Rewrite outgoing links: any node whose input is [old_id, slot] → [new_id, slot].
    rewritten: list[str] = []
    if new_id != old_id:
        for nid, node in workflow.items():
            if not isinstance(node, dict):
                continue
            for inp_name, inp_val in node.get("inputs", {}).items():
                if isinstance(inp_val, list) and len(inp_val) == 2 and str(inp_val[0]) == old_id:
                    inp_val[0] = new_id
                    rewritten.append(f"{nid}.inputs.{inp_name}")

    path = _save_edit(workflow, workflow_path)
    result = {
        "status": "ok",
        "workflow_path": path,
        "old_node_id": old_id,
        "new_node_id": new_id,
        "new_class_type": new_class_type,
        "inherited_inputs": list(new_node["inputs"].keys()),
        "rewired_downstream": rewritten,
    }
    if dropped:
        result["dropped_inputs"] = dropped
    if remapped:
        result["remapped_inputs"] = remapped
    return json.dumps(result)


@tool
def add_workflow_node(workflow_path: str, node_id: str, class_type: str, inputs: str = "{}", meta_title: str = "") -> str:
    """Add a new node to an existing workflow file.

    Args:
        workflow_path: File path to the workflow JSON.
        node_id: The node ID string (e.g. "200"). Must not already exist.
        class_type: The ComfyUI node class (e.g. "LoadImage", "CLIPTextEncode").
        inputs: JSON string of the node's inputs dict.
        meta_title: Optional display title for the node.
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    if node_id in workflow:
        return json.dumps({"error": f"Node '{node_id}' already exists."})

    try:
        inputs_dict = json.loads(inputs) if isinstance(inputs, str) else inputs
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid inputs JSON: {e}"})

    node: dict = {"class_type": class_type, "inputs": inputs_dict}
    if meta_title:
        node["_meta"] = {"title": meta_title}

    workflow[node_id] = node
    path = _save_edit(workflow, workflow_path)

    return json.dumps({
        "workflow_path": path,
        "added_node": node_id,
        "class_type": class_type,
        "node_count": len([k for k in workflow if isinstance(workflow.get(k), dict)]),
    })


@tool
def remove_workflow_node(workflow_path: str, node_id: str) -> str:
    """Remove a node from an existing workflow and clean up any links pointing to it.

    Args:
        workflow_path: File path to the workflow JSON.
        node_id: The node ID to remove.
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"error": f"Cannot load workflow: {e}"})

    if node_id not in workflow:
        return json.dumps({"error": f"Node '{node_id}' not found."})

    del workflow[node_id]

    # Remove dangling links to the deleted node
    cleaned = 0
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for inp_name, inp_val in list(inputs.items()):
            if isinstance(inp_val, list) and len(inp_val) == 2 and str(inp_val[0]) == node_id:
                del inputs[inp_name]
                cleaned += 1

    path = _save_edit(workflow, workflow_path)

    return json.dumps({
        "workflow_path": path,
        "removed_node": node_id,
        "cleaned_links": cleaned,
        "node_count": len([k for k in workflow if isinstance(workflow.get(k), dict)]),
    })


# Lower-cased for the endswith() test in summarize_workflow_graph; the list of
# extensions itself is assembly_deterministic's, so there is one place to add to.
_MODEL_EXTS_LOWER = tuple(e.lower() for e in _MODEL_EXTS_SHARED)


def summarize_workflow_graph(workflow: dict, max_nodes: int = 60) -> dict:
    """A compact account of what a graph actually contains, for its builder to report.

    Handing back only ``{"status": "ready", "workflow_path": …}`` tells the caller
    nothing about what was built, and a caller who cannot see the work does not
    trust it: one orchestrator answered a ``ready`` by hunting for a template-load
    tool it does not have, waiting out a 120-second permission prompt, listing 42
    files and re-reading both workflow JSONs — six calls and ~297K input tokens —
    only to conclude that the graph had been assembled correctly all along.

    So say it up front. Small on purpose: the class list is what answers "did it
    really wire both stages together", and it costs a few hundred tokens against
    the thousands a re-read costs.

    Returns ``{node_count, nodes, models, resolution, outputs, prompt}``, omitting
    whatever the graph does not have.
    """
    nodes = {nid: n for nid, n in (workflow or {}).items() if isinstance(n, dict)}
    summary: dict = {"node_count": len(nodes)}

    listed = list(nodes.items())[:max_nodes]
    summary["nodes"] = {nid: n.get("class_type", "unknown") for nid, n in listed}
    if len(nodes) > max_nodes:
        summary["nodes_truncated"] = len(nodes) - max_nodes

    models, outputs, prompt, width, height = [], [], "", None, None
    for nid, node in nodes.items():
        cls = node.get("class_type", "")
        for key, val in (node.get("inputs") or {}).items():
            if not isinstance(val, str):
                if key == "width" and isinstance(val, int):
                    width = val
                elif key == "height" and isinstance(val, int):
                    height = val
                continue
            if val.lower().endswith(_MODEL_EXTS_LOWER) and val not in models:
                models.append(val)
            elif key in ("filename_prefix", "output_path", "filename") and val:
                outputs.append({"node": nid, "class": cls, "path": val})
            elif key in ("text", "prompt") and len(val) > len(prompt):
                prompt = val

    if models:
        summary["models"] = models
    if outputs:
        summary["outputs"] = outputs
    if width and height:
        summary["resolution"] = f"{width}x{height}"
    if prompt:
        summary["prompt"] = prompt[:200] + ("…" if len(prompt) > 200 else "")
    return summary



# ═══════════════════════════════════════════════════════════════════════════════
# Tools: Workflow validation
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def _spec_for_input(schema: dict, inputs: dict, name: str):
    """Schema spec for *name*, resolving V3 dotted sub-inputs.

    ``model.resolution`` is not in the node's own input map — it belongs to
    whichever option ``model`` currently holds, so the lookup has to go through
    the value to find the spec.
    """
    if "." not in name:
        return ((schema.get("required") or {}).get(name)
                or (schema.get("optional") or {}).get(name))
    head, sub = name.split(".", 1)
    parent = ((schema.get("required") or {}).get(head)
              or (schema.get("optional") or {}).get(head))
    meta = parent[1] if isinstance(parent, (list, tuple)) and len(parent) > 1 else None
    if not isinstance(meta, dict):
        return None
    chosen = str(inputs.get(head))
    for opt in meta.get("options") or []:
        if isinstance(opt, dict) and str(opt.get("key")) == chosen:
            inner = opt.get("inputs") or {}
            return ((inner.get("required") or {}).get(sub)
                    or (inner.get("optional") or {}).get(sub))
    return None


def _suggest_option(value, options: list):
    """Closest legal option to *value*, or None.

    Exact-but-for-case first, then filename basename (model paths drift by
    folder and separator), then fuzzy. "kling-v3-pro" → "kling-v3" is the shape
    of mistake this catches; a semantically sensible invention like
    "MiniMax-Hailuo-02" usually has no close match, and saying so is better than
    snapping it to something arbitrary.
    """
    text = str(value)
    lowered = {str(o).lower(): o for o in options}
    if text.lower() in lowered:
        return lowered[text.lower()]
    base = Path(text.replace("\\", "/")).name.lower()
    for opt in options:
        if Path(str(opt).replace("\\", "/")).name.lower() == base:
            return opt
    import difflib  # noqa: PLC0415 - only needed on the error path
    near = difflib.get_close_matches(text, [str(o) for o in options], n=1, cutoff=0.75)
    return near[0] if near else None


def _invalid_widget_values(workflow: dict, object_info: dict) -> list:
    """Literal inputs whose value is not one the node accepts.

    Presence was the only thing checked before, so a value that is merely wrong
    — ``sampler_name="euler_supreme"``, ``model="MiniMax-Hailuo-02"`` — passed
    local validation and failed at the server, or silently produced the wrong
    thing. Enums whose options are not knowable offline (lazily loaded or
    ``remote``) are skipped rather than guessed at.
    """
    out: list = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        schema = (object_info.get(node.get("class_type")) or {}).get("input") or {}
        if not schema:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for name, value in inputs.items():
            if isinstance(value, list):        # a link, not a literal
                continue
            spec = _spec_for_input(schema, inputs, name)
            if spec is None:
                continue
            meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
            if isinstance(meta, dict) and meta.get("remote"):
                continue                        # options come from an endpoint
            options = _widget_options(spec)
            if not options or str(value) in {str(o) for o in options}:
                continue
            entry = {
                "node_id": str(nid),
                "class_type": node.get("class_type"),
                "input": name,
                "value": value,
            }
            # Short enums (samplers, model makes, aspect ratios) are worth
            # quoting whole — the exact spelling is the whole point. Long ones
            # are asset lists: a node offering 400 checkpoints is telling you the
            # file is not installed, and pasting all 400 in helps nobody.
            if len(options) <= 12:
                entry["options"] = list(options)
            else:
                import difflib  # noqa: PLC0415 - error path only
                entry["options_count"] = len(options)
                entry["closest_options"] = difflib.get_close_matches(
                    str(value), [str(o) for o in options], n=5, cutoff=0.3)
            suggestion = _suggest_option(value, options)
            if suggestion is not None:
                entry["did_you_mean"] = suggestion
            out.append(entry)
    return out


def _missing_widget_facts(nid: str, cls: str, name: str, spec) -> dict:
    """Everything needed to choose a value for one missing widget, in one place.

    Without this the repair loop is: validate → read an opaque "missing required
    input 'format'" → call get_node_schema to learn what format even accepts →
    guess → patch → validate again, each step its own model call. The schema is
    already in hand here, so hand it over.
    """
    meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else {}
    meta = meta if isinstance(meta, dict) else {}
    typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
    options = _widget_options(spec)
    out = {
        "node_id": str(nid),
        "class_type": cls,
        "input": name,
        "type": "COMBO" if isinstance(typ, list) else typ,
    }
    if options:
        out["options"] = options[:40]
        if len(options) > 40:
            out["options_truncated"] = len(options)
    for key in ("min", "max", "step", "tooltip"):
        if key in meta:
            out[key] = meta[key]
    return out


def validate_workflow(workflow_path: str) -> str:
    """Validate a ComfyUI workflow (local + server-side) without executing it.

    Returns valid=true/false, local_errors, server_errors, plus — when something
    is missing — ``missing_inputs``: one entry per unset required widget carrying
    its type, selectable options and range, so the value can be chosen without a
    separate get_node_schema call. Unconnected socket inputs are reported as
    wiring problems instead, since no literal can satisfy them.

    Args:
        workflow_path: File path to the workflow JSON (from get_workflow_template or save_workflow).
    """
    try:
        workflow = _load_workflow(workflow_path)
    except (json.JSONDecodeError, FileNotFoundError, OSError) as e:
        return json.dumps({"valid": False, "local_errors": [f"Cannot load workflow: {e}"], "server_errors": {}})

    local_errors = []
    missing_inputs: list = []

    try:
        all_nodes = _get_object_info()
    except Exception:
        all_nodes = {}

    node_ids = set(workflow.keys())

    for nid, node in workflow.items():
        cls = node.get("class_type", "")
        if not cls:
            local_errors.append(f"Node {nid}: missing 'class_type'.")
            continue

        if all_nodes and cls not in all_nodes:
            local_errors.append(f"Node {nid}: unknown class_type '{cls}'.")
            continue

        node_info = all_nodes.get(cls, {})
        required = node_info.get("input", {}).get("required", {})
        inputs = node.get("inputs", {})

        for req_name, req_spec in required.items():
            # V3 autogrow umbrella (COMFY_AUTOGROW_V3): the name is never a literal
            # key — ComfyUI grows it into per-slot keys (image0, image1, … / a, b…).
            # Don't flag the umbrella as missing; verify enough grown slots are wired.
            if _is_autogrow_spec(req_spec):
                meta = req_spec[1] if len(req_spec) > 1 and isinstance(req_spec[1], dict) else {}
                ag = _autogrow_info(meta, req_name)
                slot_keys = set(ag["keys"])
                have = sum(1 for k in inputs if k in slot_keys)
                if ag["min"] and have < ag["min"]:
                    local_errors.append(
                        f"Node {nid} ({cls}): autogrow input '{req_name}' needs at "
                        f"least {ag['min']} connected slot(s) "
                        f"(e.g. {', '.join(ag['keys'][:2]) or 'image0, image1'}); found {have}."
                    )
                continue
            if req_name in inputs:
                continue
            if not _is_widget_spec(req_spec):
                # A socket type (IMAGE, MODEL_TASK_ID, a custom node's own type):
                # it wants a wire from another node, not a literal. Reported as a
                # wiring problem, never as a value the caller could type in.
                local_errors.append(
                    f"Node {nid} ({cls}): required input '{req_name}' "
                    f"is unconnected (expects a {req_spec[0]} link)."
                )
                continue
            facts = _missing_widget_facts(nid, cls, req_name, req_spec)
            missing_inputs.append(facts)
            local_errors.append(
                f"Node {nid} ({cls}): missing required input '{req_name}'."
            )

        for inp_name, inp_val in inputs.items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                src_id = str(inp_val[0])
                if src_id not in node_ids:
                    local_errors.append(
                        f"Node {nid} ({cls}): input '{inp_name}' references "
                        f"non-existent node '{src_id}'."
                    )

        # A scalar in a link-only socket. ComfyUI's own validation accepts this
        # and then dies inside the sampler reaching for an attribute the number
        # does not have — so it has to be caught here or not at all. Say the fix,
        # because the fix is always the same one.
        _specs = dict(node_info.get("input", {}).get("optional", {}))
        _specs.update(required)
        for _name, _val in _link_only_scalars(node, _specs):
            local_errors.append(
                f"Node {nid} ({cls}): input '{_name}' = {_val!r} is a value in a "
                f"link-only socket ({_specs[_name][0]}) — it accepts a wire or "
                f"nothing. Set it to null, or wire it to a node that outputs "
                f"{_specs[_name][0]}."
            )

        # A wire under a key the node does not declare. A bare autogrow slot name
        # is the usual one and has an exact fix; anything else is named together
        # with the inputs the node does take.
        _probe = {"class_type": cls, "inputs": dict(inputs)}
        _opt = node_info.get("input", {}).get("optional", {}) or {}
        for _note in _normalize_autogrow_keys(_probe, required, _opt):
            local_errors.append(f"Node {nid} ({cls}): {_note} - write that key "
                                f"(update_workflow renames it itself).")
        for _key in _undeclared_wires(_probe, required, _opt, node_info):
            local_errors.append(_undeclared_wire_error(nid, cls, _key, _probe, required, _opt))

    invalid_inputs = _invalid_widget_values(workflow, all_nodes)
    for bad in invalid_inputs:
        local_errors.append(
            f"Node {bad['node_id']} ({bad['class_type']}): input '{bad['input']}' "
            f"= {bad['value']!r} is not one of the accepted values."
        )

    # Server-side validation (side-effect-free — never interrupts/clears other
    # jobs; skips entirely while ComfyUI is busy). See _server_validate.
    server_errors: dict = _server_validate(workflow)

    is_valid = len(local_errors) == 0 and len(server_errors) == 0

    result = {
        "valid": is_valid,
        "local_errors": local_errors,
        "server_errors": server_errors,
    }
    # Nodes ComfyUI would never execute. A WARNING, never an error: this same
    # function validates graphs people have open on the canvas, where a parked
    # node is a normal way to work, and refusing to run one of those would be
    # worse than the defect. In a workflow being BUILT it is a defect, and the
    # builder is told so where it can act (see the build gate in agentY's
    # Pipeline._run_generate_new_workflow).
    _dead, _dead_lines = _dead_node_warnings(workflow, all_nodes)
    if _dead:
        result["dead_nodes"] = _dead
        result["warnings"] = _dead_lines
    if missing_inputs:
        result["missing_inputs"] = missing_inputs
    if invalid_inputs:
        result["invalid_inputs"] = invalid_inputs
    if missing_inputs or invalid_inputs:
        parts = []
        if missing_inputs:
            parts.append("set each entry in missing_inputs")
        if invalid_inputs:
            parts.append("replace each invalid_inputs value with one of its "
                         "'options' (never invent one — these are exact strings "
                         "the node matches literally)")
        result["hint"] = ("Use update_workflow to " + " and ".join(parts)
                          + ", then re-validate.")
    return json.dumps(result)
