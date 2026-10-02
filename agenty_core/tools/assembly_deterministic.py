"""Deterministic (no-LLM) workflow assembly.

Pure-Python helpers that harden a ComfyUI template into an executable workflow
without any LLM reasoning, plus the LLM/MCP-callable ``assemble_workflow_deterministic``
tool that runs the full mechanical happy-path (load template -> apply briefing ->
validate) in one call.

The helpers here are imported by ``apply_brainbriefing`` (agenty_core.tools.comfyui)
so both the LLM brain and the deterministic path share the same robustness. The
tool lazily imports comfyui to avoid a module-load import cycle.
"""
from __future__ import annotations

import json
import re

from agenty_core._compat import tool

# Pure-annotation node classes: they carry no executable inputs/outputs and
# ComfyUI validation rejects them as unknown class_type.
ANNOTATION_NODE_CLASSES = ("Note", "MarkdownNote")


def strip_annotation_nodes(workflow: dict) -> list[str]:
    """Remove Note / MarkdownNote nodes in-place; return the removed ids."""
    removed: list[str] = []
    for nid in [n for n, nd in list(workflow.items())
                if isinstance(nd, dict) and nd.get("class_type") in ANNOTATION_NODE_CLASSES]:
        del workflow[nid]
        removed.append(nid)
    return removed


# UI-only passthrough node classes: ComfyUI's /prompt API rejects them as
# unknown class_type, so they must be bypassed (links rewired through them) —
# not merely deleted like annotation nodes. NOTE: only true single-input
# passthroughs belong here. PrimitiveNode is deliberately excluded — it is a
# value *source* (widget → output), not a passthrough, so rewiring "through" it
# would drop the consumer's input; it needs value-inlining instead.
REROUTE_NODE_CLASSES = ("Reroute", "Reroute (rgthree)", "ReroutePrimitive",
                        "Reroute//nodes")

# Input types ComfyUI always wires via links — they never hold a widget value.
#
# This set does two jobs that MUST agree, which is why it lives here rather than
# beside either one: it excludes a socket from positional widget alignment
# (``_schema_widget_slots`` in comfyui.py) and it strips a scalar that landed in
# one anyway (:func:`strip_link_only_scalars`). While those two disagreed,
# ``WanImageToVideo.clip_vision_output`` took the leftover ``1`` from a shifted
# ``widgets_values`` list, passed every validation ComfyUI can do, and then died
# six minutes into the render with ``'int' object has no attribute
# 'penultimate_hidden_states'``.
#
# Deliberately curated, not derived. "Is this type some node's output?" looks
# like the general rule and isn't: COLOR and CURVE are outputs too, and are also
# genuine socketless widgets that legitimately carry a value. Adding a type here
# changes how every template aligns, so measure the corpus first —
# tools/tests/test_link_only_alignment.py does exactly that.
LINK_ONLY_TYPES: frozenset[str] = frozenset({
    "MODEL", "CLIP", "VAE", "LATENT", "IMAGE", "MASK", "CONDITIONING",
    "CONTROL_NET", "EMBEDS", "SAMPLER", "SIGMAS", "AUDIO", "VIDEO",
    "SEGS", "BBOX", "UPSCALE_MODEL", "CLIPREGION", "PHOTOMAKER",
    "GEMINI_INPUT_FILES", "CLIP_VISION_OUTPUT",
})

# Model-file extensions — used to tell a model combo (downloadable) apart from an
# ordinary enum combo (sampler_name, scheduler, …) when a value can't be snapped.
_MODEL_EXTS = (".safetensors", ".ckpt", ".pth", ".pt", ".gguf", ".bin", ".onnx", ".sft")

# LoadImage-family classes whose primary widget names an input image file.
_LOADIMAGE_CLASSES = ("LoadImage", "LoadImageMask", "LoadImageOutput")


def strip_reroute_nodes(workflow: dict) -> list[str]:
    """Bypass and remove Reroute/Primitive passthrough nodes (API format), in
    place. Every consumer input wired to a Reroute's output is rewired to the
    Reroute's own upstream source (following Reroute chains); a Reroute with no
    upstream has its consuming inputs dropped. Returns the removed ids."""
    reroutes = {nid: nd for nid, nd in workflow.items()
                if isinstance(nd, dict) and nd.get("class_type") in REROUTE_NODE_CLASSES}
    if not reroutes:
        return []

    def upstream(rid: str):
        """The [src_id, slot] a Reroute forwards (its first link-valued input)."""
        for v in (reroutes[rid].get("inputs") or {}).values():
            if isinstance(v, list) and len(v) == 2:
                return v
        return None

    def resolve(link):
        """Follow a link through any chain of Reroutes to the real producer."""
        seen: set = set()
        while isinstance(link, list) and len(link) == 2 and str(link[0]) in reroutes:
            if str(link[0]) in seen:
                return None
            seen.add(str(link[0]))
            link = upstream(str(link[0]))
            if link is None:
                return None
        return link

    for nid, node in workflow.items():
        if nid in reroutes or not isinstance(node, dict):
            continue
        for k, v in list((node.get("inputs") or {}).items()):
            if isinstance(v, list) and len(v) == 2 and str(v[0]) in reroutes:
                r = resolve(v)
                if r is None:
                    node["inputs"].pop(k, None)  # dangling reroute → drop input
                else:
                    node["inputs"][k] = r
    for rid in reroutes:
        del workflow[rid]
    return list(reroutes.keys())


def rebind_placeholder_images(workflow: dict, object_info: dict) -> list[str]:
    """Bind LoadImage nodes that still hold a *placeholder* filename (one the
    server doesn't actually have as an input) to a real available input image,
    preferring the harness-staged ``agent/`` inputs. Returns applied notes.

    Video/edit templates ship a sample image name (e.g. ``egyptian_queen.png``)
    and researchers frequently omit ``input_nodes``, so the graph would 400 with
    ``value not in list`` at submission. Deterministically rebinding to a real
    input rescues it."""
    if not object_info:
        return []
    li = (object_info.get("LoadImage", {}) or {}).get("input", {}).get("required", {})
    avail = [o for o in (combo_options(li.get("image")) or []) if isinstance(o, str)]
    if not avail:
        return []
    staged = [o for o in avail if o.replace("\\", "/").lower().startswith("agent/")] or avail
    notes: list[str] = []
    j = 0
    for nid, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") not in _LOADIMAGE_CLASSES:
            continue
        cur = (node.get("inputs") or {}).get("image")
        if isinstance(cur, str) and cur in avail:
            continue  # already a real input — leave it
        pick = staged[j % len(staged)]
        node.setdefault("inputs", {})["image"] = pick
        notes.append(f"Node {nid}.inputs.image → {pick!r} (rebound placeholder)")
        j += 1
    return notes


def coerce_dim(v):
    """Coerce a briefing resolution field to a positive int, or None (guards an
    ``int(dict)`` crash when the researcher emits junk)."""
    if isinstance(v, bool) or isinstance(v, (dict, list)):
        return None
    if isinstance(v, (int, float)):
        return int(v) if v > 0 else None
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        n = int(v.strip())
        return n if n > 0 else None
    return None


def combo_options(spec):
    """Return the option list for a ComfyUI combo input spec, handling both the
    ``['COMBO', {'options': [...]}]`` and legacy ``[[opt1, opt2, ...], {...}]`` formats."""
    if not isinstance(spec, list) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


_GENERIC_MODEL_TOKENS = frozenset({
    "model", "models", "diffusion", "pytorch", "vae", "clip", "text", "encoder", "encoders",
    "unet", "lora", "checkpoint", "comfy", "comfyui", "image", "video", "audio", "latent",
    "upscale", "upscaler", "spatial", "temporal", "the",
})


def _model_brand(stem: str) -> str | None:
    """The first distinctive token of a model file's name (``ltx``, ``flux1``),
    or None when it starts with a generic word or a token too short to trust."""
    for tok in re.findall(r"[a-z0-9]+", stem.lower()):
        if tok.isdigit():
            continue
        if tok in _GENERIC_MODEL_TOKENS or len(tok) < 3:
            return None
        return tok
    return None


def snap_combo(val: str, opts: list, fallback_first: bool = True):
    """Snap an invalid combo value (e.g. a model file the template references but
    that isn't installed) to the best same-family option. With ``fallback_first``
    (default) an unmatched value falls back to the first option; pass False to
    return None instead (so a model combo can be surfaced for download rather than
    snapped to an unrelated file)."""
    base = str(val).replace("\\", "/").rsplit("/", 1)[-1].lower()
    stem = base.rsplit(".", 1)[0]
    for o in opts:
        ob = str(o).replace("\\", "/").rsplit("/", 1)[-1].lower()
        if ob == base or ob.rsplit(".", 1)[0] == stem:
            return o
    vt = set(re.findall(r"[a-z0-9]+", stem))
    if not fallback_first:
        # A model file stays in its family. Word overlap alone tied an LTX-2.5
        # video VAE between "LTX2/ltx-2-3-22b-VAE" and "hunyuan_video_vae_bf16"
        # (video, vae, bf16) and took the Hunyuan one, which loads and then
        # breaks the graph. When any option carries the file's brand (its first
        # distinctive token: ltx, flux1, wan2, qwen, hunyuan), choose among those.
        brand = _model_brand(stem)
        if brand:
            same = [o for o in opts if brand in str(o).replace("\\", "/").rsplit("/", 1)[-1].lower()]
            if same:
                opts = same
    best, best_n = None, 0
    for o in opts:
        ot = set(re.findall(r"[a-z0-9]+",
                            str(o).replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()))
        n = len(vt & ot)
        if n > best_n:
            best, best_n = o, n
    if best_n >= 1:
        return best
    return opts[0] if (fallback_first and opts) else None


def _is_model_combo(cval, copts) -> bool:
    """True if a combo names model files (downloadable) rather than an enum."""
    def _isf(x):
        return isinstance(x, str) and x.lower().endswith(_MODEL_EXTS)
    return _isf(cval) or (bool(copts) and _isf(copts[0]))


def _dynamic_combo_option_keys(spec) -> list:
    """The legal values of a ``COMFY_DYNAMICCOMBO_V3`` input: its options' keys."""
    if not (isinstance(spec, list) and len(spec) >= 2
            and spec[0] == "COMFY_DYNAMICCOMBO_V3" and isinstance(spec[1], dict)):
        return []
    return [str(o.get("key")) for o in (spec[1].get("options") or [])
            if isinstance(o, dict) and o.get("key") is not None]


def _dynamic_combo_sub_inputs(spec, option_key: str) -> set:
    """The widget names one option of a dynamic combo contributes."""
    for opt in (spec[1].get("options") or []):
        if not isinstance(opt, dict) or str(opt.get("key")) != str(option_key):
            continue
        inner = opt.get("inputs") or {}
        return {n for section in ("required", "optional")
                for n in (inner.get(section) or {})}
    return set()


def dynamic_sub_specs(node: dict, required: dict, optional: dict | None = None) -> tuple:
    """``(required, optional)`` sub-input specs, dotted, for the options in force.

    A dynamic combo's widgets are declared inside the option they belong to, so
    they are absent from the class's own required/optional lists — and every
    repair in this module works off those lists. That is why a missing
    `model.prompt_optimization` was never defaulted and a `model.height` of 736
    was never clamped to its minimum of 1024: nothing knew they had specs. With
    the specs hoisted under their dotted names, defaulting, clamping and combo
    snapping all reach them exactly as they reach any other input.
    """
    node_inputs = node.get("inputs", {})
    specs = dict(optional or {})
    specs.update(required or {})
    req_out: dict = {}
    opt_out: dict = {}
    for name, spec in specs.items():
        if not _dynamic_combo_option_keys(spec):
            continue
        chosen = node_inputs.get(name)
        if not isinstance(chosen, str):
            continue
        for opt in (spec[1].get("options") or []):
            if not isinstance(opt, dict) or str(opt.get("key")) != chosen:
                continue
            inner = opt.get("inputs") or {}
            for sub, sub_spec in (inner.get("required") or {}).items():
                req_out[f"{name}.{sub}"] = sub_spec
            for sub, sub_spec in (inner.get("optional") or {}).items():
                opt_out[f"{name}.{sub}"] = sub_spec
    return req_out, opt_out


def default_dynamic_combos(node: dict, required: dict, reshaped: list | None = None) -> None:
    """Give a required dynamic combo that is absent its default option, and the
    same for the dynamic combos that option requires in turn.

    ComfyUI's own validation does not ask for these: a ``SaveVideo`` with no
    ``format`` is accepted, queued, renders the whole video and only then fails
    with ``execute() missing 1 required positional argument: 'format'``. They are
    not connections either, so reporting them as "missing" left nothing to wire.
    The first option is what the frontend's widget starts on.
    """
    inputs = node.setdefault("inputs", {})
    for _ in range(4):                      # an option's own dynamic combos, a few levels deep
        sub_req, _sub_opt = dynamic_sub_specs(node, required, None)
        changed = False
        for name, spec in {**(required or {}), **sub_req}.items():
            keys = _dynamic_combo_option_keys(spec)
            if not keys or name in inputs:
                continue
            default = spec[1].get("default") if len(spec) > 1 and isinstance(spec[1], dict) else None
            inputs[name] = default if default in keys else keys[0]
            changed = True
            if reshaped is not None:
                reshaped.append(f"input '{name}': was missing, set to {inputs[name]!r} (its default option)")
        if not changed:
            return


def is_autogrow_spec(spec) -> bool:
    """True for a V3 autogrow group (``COMFY_AUTOGROW_V3``)."""
    return (isinstance(spec, (list, tuple)) and bool(spec) and isinstance(spec[0], str)
            and "AUTOGROW" in spec[0].upper())


def autogrow_slot_names(spec) -> list[str]:
    """The slots one autogrow group grows into, as its template names them.

    A ``prefix`` template counts from 0 (``ref_image_0 … ref_image_8``), a
    ``names`` template lists them — the expansion ComfyUI's ``Autogrow`` performs.
    Undotted: the prompt addresses each one as ``<group>.<slot>``.
    """
    meta = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
    tmpl = (meta.get("template") or {}) if isinstance(meta, dict) else {}
    if isinstance(tmpl.get("names"), list):
        return [str(n) for n in tmpl["names"]]
    try:
        count = int(tmpl.get("max") or 0)
    except (TypeError, ValueError):
        count = 0
    return [f"{tmpl.get('prefix', '')}{i}" for i in range(count)]


def is_v3_schema(node_info) -> bool:
    """True when *node_info* (one ``/object_info`` entry) describes a V3 node.

    Only V3 schemas carry ``price_badge`` — ``None`` on a local node, but present;
    a V1 entry has no such key. It matters because the two treat an input they
    never declared differently: a V1 function often takes ``**kwargs`` on purpose
    (Impact's switches and KJNodes' batchers grow inputs that way), while a V3
    ``execute`` names every argument and raises on anything else.
    """
    return isinstance(node_info, dict) and "price_badge" in node_info


def _in_force_specs(node: dict, required: dict, optional: dict | None) -> dict:
    """Every input spec in force on *node*: the class's own, plus the dotted
    sub-inputs of each dynamic combo's selected option."""
    req_sub, opt_sub = dynamic_sub_specs(node, required, optional)
    return {**(optional or {}), **opt_sub, **(required or {}), **req_sub}


def _is_wire(value) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int)) and not isinstance(value[0], bool)
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def declared_input_keys(node: dict, required: dict,
                        optional: dict | None = None) -> tuple[set, tuple]:
    """``(keys, open_prefixes)`` — every input key ComfyUI binds for *node*.

    Spelled as the API prompt spells them: each slot of an autogrow group as
    ``<group>.<slot>`` (plus the group itself, which ComfyUI binds as an empty
    dict when nothing is wired), a dynamic combo's sub-inputs as
    ``<combo>.<input>``. A dynamic combo whose value is not a plain option key —
    wired, absent, or nested inside another option — cannot say which branch is
    in force, so everything under it is returned as an open prefix instead of
    being guessed at.
    """
    inputs = node.get("inputs") or {}
    keys: set = set()
    open_prefixes: list = []
    for name, spec in _in_force_specs(node, required, optional).items():
        keys.add(name)
        if is_autogrow_spec(spec):
            keys.update(f"{name}.{slot}" for slot in autogrow_slot_names(spec))
        elif _dynamic_combo_option_keys(spec) and (
                "." in name or not isinstance(inputs.get(name), str)):
            open_prefixes.append(name + ".")
    return keys, tuple(open_prefixes)


def normalize_autogrow_keys(node: dict, required: dict, optional: dict | None = None,
                            reshaped: list | None = None) -> list[str]:
    """Move a wire written under a bare autogrow slot name to its real address.

    ``MiniMaxH3ReferenceToVideo`` grows ``ref_images`` into slots, and the prompt
    addresses one as ``ref_images.ref_image_0``. Written bare — ``ref_image_0``,
    the name the group's own template uses — ComfyUI accepts the graph, validates
    it, loads every model upstream, and then hands the wire to the node under that
    name: ``execute() got an unexpected keyword argument 'ref_image_0'``.

    Mechanical when exactly one group in force owns that slot name and its dotted
    address is still free. A real input of the same name, or two groups sharing a
    slot name, is left alone for validation to report. Only wires move: a literal
    under an undeclared name is dropped by ComfyUI and harms nothing. Returns the
    notes, also appended to *reshaped* when given.
    """
    inputs = node.get("inputs") or {}
    if not inputs:
        return []
    specs = _in_force_specs(node, required, optional)
    owners: dict = {}
    for name, spec in specs.items():
        if is_autogrow_spec(spec):
            for slot in autogrow_slot_names(spec):
                owners.setdefault(slot, []).append(f"{name}.{slot}")
    notes: list[str] = []
    for key in list(inputs):
        if key in specs or "." in key or not _is_wire(inputs[key]):
            continue
        targets = owners.get(key) or []
        if len(targets) != 1 or targets[0] in inputs:
            continue
        inputs[targets[0]] = inputs.pop(key)
        group = targets[0].rsplit(".", 1)[0]
        notes.append(f"input '{key}' is a slot of autogrow input '{group}', "
                     f"addressed '{targets[0]}'")
    if reshaped is not None:
        reshaped.extend(notes)
    return notes


def undeclared_wires(node: dict, required: dict, optional: dict | None = None,
                     node_info: dict | None = None) -> list[str]:
    """Keys on a V3 node that carry a wire but name no input the node declares.

    ComfyUI binds a *value* only to an input the node declares and silently drops
    the rest, but it passes a *wire* on under whatever key it sits
    (``execution.get_input_data``). Each of these therefore reaches ``execute`` as
    an argument it does not take, and the run fails after everything upstream has
    already been computed. V1 nodes are not judged: many take ``**kwargs``.
    """
    if not is_v3_schema(node_info):
        return []
    keys, open_prefixes = declared_input_keys(node, required, optional)
    return [k for k, v in (node.get("inputs") or {}).items()
            if _is_wire(v) and k not in keys and not k.startswith(open_prefixes)]


def drop_overridden_dimensions(node: dict, required: dict, optional: dict | None = None,
                               dropped: list | None = None) -> None:
    """Remove width/height that a chosen size preset already decides.

    These nodes say it themselves: *"Pick a recommended size. Select Custom to
    use the width and height below."* So unless the preset IS ``Custom`` the
    node ignores width and height — but ComfyUI still range-checks them, and a
    1312x736 pair written beside a 16:9 preset failed validation for a height
    below the 1024 minimum. Clamping them would satisfy the check and quietly
    produce 1312x1024, which is not the shape anyone asked for. The preset is the
    answer; the leftovers are noise, so they go.

    Driven off the literal ``Custom`` option in the schema rather than off any
    node's name, so it applies wherever that convention is used and nowhere else.
    """
    node_inputs = node.get("inputs", {})
    req_sub, opt_sub = dynamic_sub_specs(node, required, optional)
    sub_specs = {**opt_sub, **req_sub}
    for name, spec in sub_specs.items():
        opts = combo_options(spec) or []
        if "Custom" not in opts:
            continue
        value = node_inputs.get(name)
        if not isinstance(value, str) or value == "Custom":
            continue
        prefix = name.rsplit(".", 1)[0] + "." if "." in name else ""
        for dim in ("width", "height"):
            key = f"{prefix}{dim}"
            if key in node_inputs:
                node_inputs.pop(key)
                if dropped is not None:
                    dropped.append(f"'{key}' (decided by '{name}'={value!r})")


def flatten_dynamic_combos(node: dict, required: dict, optional: dict | None = None,
                           reshaped: list | None = None) -> list[str]:
    """Unpack a dict written into a ``COMFY_DYNAMICCOMBO_V3`` input. Returns errors.

    The API format spells a dynamic combo's widgets as dotted siblings —
    ``model`` alongside ``model.size_preset`` — and ComfyUI binds them by those
    names. Handed an object instead::

        "model": {"model": "seedream 5.0 pro", "size_preset": "…", "seed": 47629}

    it binds nothing, and the node dies at execution with

        ByteDanceSeedreamNodeV3.execute() missing 1 required positional argument: 'model'

    which names neither the input that is wrong nor the shape it wanted. Local
    validation used to pass that graph — it checks class names, missing required
    inputs and link targets, never the shape of a value — so the agent was told
    "valid", submitted, met a TypeError it could not map back to anything, and
    tried to repair a graph that was never structurally wrong.

    Mechanical either way, which is why it belongs here beside the other
    repairs: the option key becomes the combo's value, a top-level input that got
    swept into the object is hoisted back out, and the option's own widgets
    become dotted. A key the selected option does not offer is dropped rather
    than renamed — moving it would only move the problem. An object naming no
    known option is NOT guessed at; it comes back as an error that lists the
    keys, because picking one would run a model nobody asked for.
    """
    node_inputs = node.get("inputs", {})
    if not node_inputs:
        return []
    tops = set(required or {}) | set(optional or {})
    specs = dict(optional or {})
    specs.update(required or {})
    errors: list[str] = []

    for name, spec in specs.items():
        value = node_inputs.get(name)
        if not isinstance(value, dict):
            continue
        keys = _dynamic_combo_option_keys(spec)
        if not keys:
            continue
        inner = dict(value)
        chosen = None
        if isinstance(inner.get(name), str) and inner[name] in keys:
            chosen = inner.pop(name)
        else:
            for k, v in list(inner.items()):
                if isinstance(v, str) and v in keys:
                    chosen = inner.pop(k)
                    break
        if chosen is None:
            errors.append(
                f"input '{name}' is an object naming no known option; set it to "
                f"one of {keys} and put its settings in '{name}.<widget>' keys")
            continue

        subs = _dynamic_combo_sub_inputs(spec, chosen)
        node_inputs[name] = chosen
        moved, dropped = [], []
        for k, v in inner.items():
            if v is None or v == {} or v == []:
                continue                    # a placeholder, not a value
            # The selected option's own input first. SaveVideo has both: a
            # top-level optional `codec` and a `codec` its format option requires.
            # Hoisting {"format": "auto", "codec": "auto"} to the top-level one
            # left `format.codec` missing - reported as an error the agent could
            # only clear by guessing, three calls later, at the dotted spelling.
            if k in subs:
                node_inputs[f"{name}.{k}"] = v
                moved.append(f"{name}.{k}")
            elif k in tops and k != name:
                node_inputs[k] = v          # a real input of the node itself
                moved.append(k)
            else:
                dropped.append(k)
        if reshaped is not None:
            note = (f"input '{name}': unpacked an object into '{name}'="
                    f"{chosen!r} plus {', '.join(sorted(moved)) or 'nothing'}")
            if dropped:
                note += f"; dropped {', '.join(sorted(dropped))} (not offered by {chosen!r})"
            reshaped.append(note)
    return errors


def harden_node_inputs(node: dict, required: dict, missing_models: list | None = None,
                       optional: dict | None = None,
                       stripped: list | None = None,
                       reshaped: list | None = None) -> list[str]:
    """Make one node's inputs valid where it can be done mechanically:

    * inject a widget/combo default for a missing required *widget* input
      (ComfyUI needs the value present in API format),
    * snap a present combo value that isn't a valid option to a same-family
      substitute,
    * for a model combo with no same-family match, append the value to
      *missing_models* (if given) so the caller can download it — instead of
      snapping to an unrelated model file (which would render garbage), and
    * delete a scalar left in a link-only socket, describing each into
      *stripped* (if given). See :func:`strip_link_only_scalars`.

    Returns the names of required inputs that remain genuinely missing — the
    *connection* inputs (a bare type, no default) that need real wiring.
    """
    node_inputs = node.get("inputs", {})
    missing: list[str] = []
    # Before anything else: a dict written into a dynamic combo is not a value
    # this pass can sanitize, it is the wrong shape entirely. Unpacked first so
    # the steps below see the dotted keys they expect.
    flatten_dynamic_combos(node, required, optional, reshaped)
    # Then a wire written under a bare autogrow slot name, moved to the dotted
    # address ComfyUI binds, so nothing below mistakes it for a stray input.
    normalize_autogrow_keys(node, required, optional, reshaped)
    # A preset that decides the size makes a stray width/height both redundant
    # and, when it falls outside the input's own range, fatal.
    drop_overridden_dimensions(node, required, optional, reshaped)
    # A required dynamic combo left out entirely gets its default option: ComfyUI
    # accepts the graph without it and fails only when the node runs.
    default_dynamic_combos(node, required, reshaped)
    # The widgets an option contributes have specs too — hoist them so the
    # repairs below default, clamp and snap them like anything else.
    _sub_req, _sub_opt = dynamic_sub_specs(node, required, optional)
    required = {**required, **_sub_req}
    optional = {**(optional or {}), **_sub_opt}
    node_inputs = node.get("inputs", {})
    for req_name, spec in required.items():
        # Variadic / autogrow inputs (COMFY_AUTOGROW_*) are grown dynamically by
        # ComfyUI into per-slot keys (a, b, c… / image1, image2…); the umbrella
        # name (e.g. 'values', 'images') is never a literal input key, so neither
        # inject a widget default nor report it as a missing connection.
        if isinstance(spec, list) and spec and isinstance(spec[0], str) \
                and "AUTOGROW" in spec[0].upper():
            continue
        if req_name not in node_inputs:
            default = None
            if isinstance(spec, list) and len(spec) >= 2 and isinstance(spec[1], dict) \
                    and spec[1].get("default") is not None:
                default = spec[1]["default"]
            else:
                _opts = combo_options(spec)
                default = _opts[0] if _opts else None
            if default is not None:
                node.setdefault("inputs", {})[req_name] = default
            else:
                missing.append(req_name)
    # Sanitize PRESENT values in required and optional inputs alike (a bad value
    # in an optional slot — e.g. CreateVideo.bit_depth=24, max 10 — is rejected
    # by the server just the same); defaults are only injected for required.
    _spec_by_name = dict(optional or {})
    _spec_by_name.update(required)
    for cinp, cspec in _spec_by_name.items():
        cval = node_inputs.get(cinp)
        if cval is None or isinstance(cval, list):
            continue  # absent, or a node connection
        copts = combo_options(cspec)
        if copts:
            if cval in copts:
                continue
            # A non-string value in a combo slot (e.g. an int written into
            # scale_method by a shifted widget mapping) can never be valid —
            # match/snap by string; enum combos fall back to the first option.
            sval = cval if isinstance(cval, str) else str(cval)
            if sval in copts:
                node.setdefault("inputs", {})[cinp] = sval
                continue
            is_model = _is_model_combo(sval, copts)
            snapped = snap_combo(sval, copts, fallback_first=not is_model)
            if snapped is not None and snapped != cval:
                node.setdefault("inputs", {})[cinp] = snapped
            elif snapped is None and is_model and missing_models is not None:
                # Un-installed model with no same-family substitute — surface it
                # for download rather than snapping to an unrelated file.
                if sval not in missing_models:
                    missing_models.append(sval)
            continue
        # Numeric range clamp: ComfyUI rejects INT/FLOAT widget values outside
        # the spec's min/max (e.g. bit_depth 24 > max 10).
        if (isinstance(cval, (int, float)) and not isinstance(cval, bool)
                and isinstance(cspec, list) and cspec
                and cspec[0] in ("INT", "FLOAT")
                and len(cspec) > 1 and isinstance(cspec[1], dict)):
            lo, hi = cspec[1].get("min"), cspec[1].get("max")
            clamped = cval
            if isinstance(lo, (int, float)) and clamped < lo:
                clamped = lo
            if isinstance(hi, (int, float)) and clamped > hi:
                clamped = hi
            if clamped != cval:
                node.setdefault("inputs", {})[cinp] = (
                    int(clamped) if cspec[0] == "INT" else float(clamped))
    # A scalar sitting in a link-only socket is never something the caller meant.
    strip_link_only_scalars(node, _spec_by_name, stripped)
    return missing


def link_only_scalars(node: dict, specs: dict) -> list[tuple[str, object]]:
    """``(input_name, value)`` for every LINK_ONLY_TYPES input holding a scalar.

    A socket of one of those types accepts a wire and nothing else, so a bare
    value there cannot have been intended and cannot work: ComfyUI takes it,
    validates it, schedules it, and only then reaches into it for an attribute a
    number does not have. The value is always a widget-alignment leftover.

    *specs* maps input name to its schema spec (required and optional merged).
    """
    found: list[tuple[str, object]] = []
    for name, spec in specs.items():
        val = (node.get("inputs") or {}).get(name)
        if val is None or isinstance(val, list):
            continue  # absent, or a real connection [node_id, slot]
        typ = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
        if isinstance(typ, str) and typ in LINK_ONLY_TYPES:
            found.append((name, val))
    return found


def strip_link_only_scalars(node: dict, specs: dict,
                            stripped: list | None = None) -> list[str]:
    """Delete link-only-socket scalars from *node*, in place. Returns the names.

    Removing the key is the whole repair: every such input is optional in
    practice (a required socket with no wire is already reported as missing), so
    an absent key is exactly the "not supplied" the node expects. Names are
    appended to *stripped* when given, so the caller can say what it did.
    """
    names: list[str] = []
    for name, val in link_only_scalars(node, specs):
        del node["inputs"][name]
        names.append(name)
        if stripped is not None:
            stripped.append(f"{node.get('class_type', '?')}.{name} = {val!r}")
    return names


# Widget/value input types — never a graph connection, so never auto-wired.
_NON_CONNECTION_TYPES = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")


def autowire_dangling_inputs(workflow: dict, object_info: dict) -> list[str]:
    """Wire a node's missing *required connection* input to the graph's unique
    producer of that type, in place. Returns applied notes.

    Subgraph flattening preserves IMAGE boundaries (it synthesises a LoadImage)
    but drops other boundary connections — e.g. the VAE feeding an LTX
    decoder/upsampler, or a mask into InpaintModelConditioning — leaving the
    consuming node with a dangling required input that 400s at submission. When
    exactly one node in the graph outputs that type, the wiring is unambiguous;
    ambiguous types (e.g. two CONDITIONING sources) are left for the LLM brain."""
    if not object_info:
        return []
    # type -> [(node_id, output_slot)]
    producers: dict[str, list[tuple[str, int]]] = {}
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        outs = (object_info.get(node.get("class_type", ""), {}) or {}).get("output") or []
        for slot, typ in enumerate(outs):
            if isinstance(typ, str):
                producers.setdefault(typ, []).append((nid, slot))
    def _is_audio(node_id: str) -> bool:
        return "audio" in str(workflow.get(node_id, {}).get("class_type", "")).lower()

    def _sig(node_id: str, slot: int):
        """Producer equivalence signature: same class + same literal inputs
        + same slot ⇒ identical output (e.g. two CheckpointLoaderSimple loading
        the same checkpoint yield the same VAE) — picking either is safe."""
        n = workflow.get(node_id, {})
        lits = tuple(sorted((k, str(v)) for k, v in (n.get("inputs") or {}).items()
                            if not isinstance(v, list)))
        return (n.get("class_type"), lits, slot)

    wired: list[str] = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        cls = node.get("class_type", "")
        info = object_info.get(cls, {}) or {}
        required = (info.get("input", {}) or {}).get("required", {}) or {}
        node_inputs = node.get("inputs", {})
        for rname, rspec in required.items():
            if rname in node_inputs:
                continue
            # A connection input's spec is [<TYPE-string>, ...]; a combo's spec[0]
            # is a list of options, a widget's is INT/FLOAT/STRING/BOOLEAN.
            if not (isinstance(rspec, list) and rspec and isinstance(rspec[0], str)):
                continue
            typ = rspec[0]
            if typ in _NON_CONNECTION_TYPES or "AUTOGROW" in typ.upper():
                continue
            cands = [p for p in producers.get(typ, []) if p[0] != nid]
            # VAE audio/video partition (verified against the official LTX-2
            # templates + docs): an audio VAE (producer class contains "Audio",
            # e.g. LTXVAudioVAELoader) feeds ONLY audio consumers (input named
            # like 'audio_vae' or an Audio* node class); every other VAE consumer
            # takes the video/image VAE. Filter candidates to the matching side.
            if len(cands) > 1 and typ == "VAE":
                want_audio = "audio" in rname.lower() or "audio" in cls.lower()
                side = [p for p in cands if _is_audio(p[0]) == want_audio]
                if side:
                    cands = side
            # Collapse equivalent producers (identical class + literal inputs +
            # slot produce the same object) — any one of them is correct.
            if len(cands) > 1 and len({_sig(i, s) for i, s in cands}) == 1:
                cands = cands[:1]
            if len(cands) == 1:
                src_id, src_slot = cands[0]
                node.setdefault("inputs", {})[rname] = [src_id, src_slot]
                wired.append(f"Node {nid}.{rname} <- {src_id}:{src_slot} ({typ})")
    return wired


# Names that mean "this is where a branch ends" for a class ComfyUI cannot be
# asked about. Consulted only when the schema is missing, and only ever to make a
# node count AS an output — i.e. to stay silent rather than accuse.
_OUTPUT_NAME_HINTS = ("save", "preview", "viewer", "sendto", "output", "display", "show")


def _is_output_node(class_type: str, object_info: dict) -> bool:
    """Whether ComfyUI treats *class_type* as an output — the end of a branch.

    Biased toward yes for anything it cannot look up, because this decides whether
    a node gets REPORTED as dead: a custom saver mistaken for an ordinary node
    would be named as a defect in a graph that is perfectly fine, and a check that
    cries wolf is a check people learn to skip.
    """
    cls = str(class_type or "")
    if not cls:
        return True
    info = (object_info or {}).get(cls)
    if not isinstance(info, dict):
        return True                       # unknown class / ComfyUI down
    if info.get("output_node"):
        return True
    return any(hint in cls.lower() for hint in _OUTPUT_NAME_HINTS)


def dead_nodes(workflow: dict, object_info: dict) -> list[dict]:
    """The nodes in *workflow* that ComfyUI would never execute.

    ComfyUI runs a graph BACKWARDS from its output nodes, so a node whose output
    nothing reads is not slow or misconfigured — it simply never happens. Nothing
    else in validation sees this: such a node can have every one of its own
    required inputs satisfied, which is how a graph carrying three of them
    validates clean, locally AND server-side, and then quietly does less than it
    says.

    Observed, all from real builds: a `VAEDecodeAudio` whose consumer's optional
    `audio` input was left unwired (a silent video out of an audio model), a
    `GetImageSize` added to derive the resolution while the latent kept hardcoded
    dimensions (the input's aspect ratio quietly ignored), and a second
    `VAEDecodeTiled` decoding the same latent as the decoder that *is* wired.

    Iterated to a fixed point, so a chain of dead nodes is reported whole; an entry
    names the dead consumer it fed, because fixing the root fixes the rest.

    Returns ``[]`` — not a guess — when *object_info* is empty: without schemas
    there is no reliable way to tell an output node from an ordinary one.
    """
    if not isinstance(workflow, dict) or not workflow or not object_info:
        return []
    live = {str(nid): node for nid, node in workflow.items() if isinstance(node, dict)}
    found: list[dict] = []
    while True:
        consumers: dict = {}
        for nid, node in live.items():
            for value in (node.get("inputs") or {}).values():
                if _is_wire(value):
                    consumers.setdefault(str(value[0]), set()).add(nid)
        dead = [nid for nid, node in live.items()
                if not consumers.get(nid)
                and not _is_output_node(node.get("class_type"), object_info)]
        if not dead:
            return found
        already = {d["node_id"] for d in found}
        for nid in sorted(dead, key=lambda n: (len(n), n)):
            node = live.pop(nid)
            cls = str(node.get("class_type") or "")
            # The already-dead nodes this one was feeding: those are the ones to
            # fix, since wiring this output into a dead branch changes nothing.
            fed = sorted(other for other in already
                         if any(_is_wire(v) and str(v[0]) == nid for v in
                                ((workflow.get(other) or {}).get("inputs") or {}).values()))
            because = (f" The node(s) it feeds ({', '.join(fed)}) are dead too — "
                       f"fix those first." if fed else "")
            found.append({
                "node_id": nid,
                "class_type": cls,
                "title": str((node.get("_meta") or {}).get("title") or ""),
                "problem": ("nothing reads this node's output, so ComfyUI will never "
                            "execute it (a graph runs backwards from its output "
                            f"nodes).{because} Wire its output into the branch that "
                            "reaches an output node, or remove the node."),
            })


def dead_node_warnings(workflow: dict, object_info: dict) -> tuple[list, list]:
    """``(dead_nodes, one_line_warnings)`` for a validation payload.

    The list is for a caller that will act on it; the lines are for a reader that
    only ever sees text, which includes every model this gets reported to.
    """
    found = dead_nodes(workflow, object_info)
    lines = [
        f"Node {d['node_id']} ({d['class_type']}"
        + (f" - {d['title']}" if d["title"] and d["title"] != d["class_type"] else "")
        + f"): {d['problem']}"
        for d in found
    ]
    return found, lines


def dead_nodes_in_file(workflow_path: str) -> tuple[list, list]:
    """:func:`dead_node_warnings` for a workflow ON DISK — ``(dead, lines)``.

    The cheap way to ask the question: it needs the graph and ``/object_info``
    (cached), not a server validation, so it can be asked on the way to every
    submission — including each member of a 25-variant batch — without adding a
    round trip per workflow.

    Answers ``([], [])`` for a file that cannot be read or parsed. Something that
    cannot load is not a graph with dead nodes, it is a different problem, and the
    caller already has a report for that one.
    """
    try:
        from agenty_core.tools.comfyui import (  # noqa: PLC0415 — avoids an import cycle
            _get_object_info, _load_workflow,
        )
        workflow = _load_workflow(workflow_path)
        object_info = _get_object_info()
    except Exception:  # noqa: BLE001
        return [], []
    return dead_node_warnings(workflow, object_info)


def ensure_output_node(workflow: dict, object_info: dict) -> str | None:
    """If the graph has no output node but a terminal VIDEO producer (e.g.
    CreateVideo without a SaveVideo), synthesize a SaveVideo wired to it so the
    workflow is executable. Returns the synthesized node id, or None."""
    if not object_info:
        return None
    has_output = any(
        isinstance(n, dict) and (object_info.get(n.get("class_type", "")) or {}).get("output_node")
        for n in workflow.values()
    )
    if has_output or "SaveVideo" not in object_info:
        return None
    video_src = next(
        (nid for nid, n in workflow.items()
         if isinstance(n, dict)
         and "VIDEO" in ((object_info.get(n.get("class_type", "")) or {}).get("output") or [])),
        None,
    )
    if video_src is None:
        return None
    new_id = str(max((int(k) for k in workflow if str(k).isdigit()), default=0) + 1)
    workflow[new_id] = {
        "class_type": "SaveVideo",
        "inputs": {"video": [video_src, 0], "filename_prefix": "agent/video"},
        "_meta": {"title": "Save Video (auto)"},
    }
    # Its `format` (and the codec that format requires): without them the node
    # validates and then fails after the whole video has rendered.
    default_dynamic_combos(workflow[new_id], ((object_info.get("SaveVideo") or {}).get("input") or {})
                           .get("required") or {})
    return new_id


@tool
def assemble_workflow_deterministic(brainbriefing_json: str) -> str:
    """Assemble a ComfyUI workflow from a brainbriefing with NO LLM reasoning.

    Runs the mechanical brain happy-path in code: load the template named in the
    briefing, apply the briefing (the deterministic patcher, which strips
    annotation nodes, injects widget defaults, snaps invalid model names,
    synthesizes a video output, etc.), and validate against ComfyUI.

    Use this for the common case where a validated brainbriefing names a standard
    template — it needs no agent reasoning. Complex jobs (unwired connections,
    inpaint masks, batch/annotation flows) return ``status: "error"`` with the
    remaining ``problems`` so the caller can fall back to the LLM brain.

    Args:
        brainbriefing_json: The full brainbriefing JSON (string or dict).

    Returns JSON: ``{status: "ready"|"error"|"blocked", workflow_path, problems, applied}``.
    """
    # Lazy import to avoid a module-load cycle (comfyui imports the helpers above).
    from agenty_core.tools.comfyui import get_workflow_template, apply_brainbriefing  # noqa: PLC0415

    try:
        bb = json.loads(brainbriefing_json) if isinstance(brainbriefing_json, str) else brainbriefing_json
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "problems": [f"invalid briefing JSON: {e}"]})
    if not isinstance(bb, dict):
        return json.dumps({"status": "error", "problems": ["briefing is not a JSON object"]})
    if bb.get("status") == "blocked":
        return json.dumps({"status": "blocked", "blockers": bb.get("blockers", [])})

    name = (bb.get("template") or {}).get("name") if isinstance(bb.get("template"), dict) else None
    if not name or name == "build_new":
        return json.dumps({"status": "error",
                           "problems": ["briefing has no standard template.name to assemble"]})

    try:
        tinfo = json.loads(get_workflow_template(name))
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "problems": [f"get_workflow_template failed: {e}"]})
    path = tinfo.get("workflow_path")
    if not path:
        return json.dumps({"status": "error", "problems": [tinfo.get("error") or f"template '{name}' not found"]})

    try:
        res = json.loads(apply_brainbriefing(path, json.dumps(bb)))
    except Exception as e:  # noqa: BLE001
        return json.dumps({"status": "error", "problems": [f"apply_brainbriefing exception: {e}"]})

    problems = list(res.get("problems") or [])
    # Surface ComfyUI server-side validation errors too — otherwise a decline
    # can look reason-less (problems=[]) when only the server rejected it.
    _srv = res.get("server_errors")
    if _srv and res.get("status") != "ok":
        problems.append(f"server: {json.dumps(_srv)[:500]}")

    return json.dumps({
        "status": "ready" if res.get("status") == "ok" else "error",
        "workflow_path": res.get("workflow_path", path),
        "problems": problems,
        "applied": res.get("applied", []),
        "missing_models": res.get("missing_models", []),
    })
