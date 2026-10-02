"""What a workflow does, read off its graph — the raw material of a description.

A template's description is what retrieval runs on: its words help file the
template under a task and a model, and its first sentence (cut to 60 characters)
is all the researcher sees beside the name. So it has to say the operation and
the distinctive model or nodes, first.

:func:`workflow_facts` collects what can be known without a model: which nodes
(and whose), which model files, what goes in and comes out, the notes the author
left in the graph. :func:`describe_from_facts` turns that into a line with no
model call; an LLM, where there is one, is given the same facts to write a
better one.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# Nodes every graph has: no help in telling one workflow from another.
_PLUMBING = {"Reroute", "Note", "MarkdownNote", "PrimitiveNode", "PreviewImage", "SaveImage",
             "LoadImage", "PreviewAny", "GetNode", "SetNode"}
_NOTE_TYPES = {"Note", "MarkdownNote"}


def _pack_of(python_module: str) -> str:
    """'custom_nodes.ComfyUI-SAM3' -> 'ComfyUI-SAM3'; '' for core and API nodes."""
    mod = str(python_module or "")
    return mod.split(".", 1)[1].split(".")[0] if mod.startswith("custom_nodes.") else ""


def _notes(data: dict, limit: int = 700) -> str:
    """Text of the Note / MarkdownNote nodes in a UI-format graph."""
    out = []
    for node in data.get("nodes") or []:
        if isinstance(node, dict) and node.get("type") in _NOTE_TYPES:
            for v in node.get("widgets_values") or []:
                if isinstance(v, str) and v.strip():
                    out.append(" ".join(v.split()))
    return " | ".join(out)[:limit]


def pack_about(pack: str) -> str:
    """What a node pack says it is: the ``description`` in its pyproject.toml.

    The one place a pack states its purpose in a sentence. Without it a model
    describing ComfyUI-Sharp's example guessed "predict sharpness"; the pack is a
    wrapper around Apple's SHARP image-to-3D-gaussians model. ``""`` when the pack's
    folder is not on this machine or says nothing.
    """
    if not pack:
        return ""
    try:
        import re  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        from agenty_core.tools.huggingface import _folder_paths  # noqa: PLC0415
        for base in _folder_paths().get("custom_nodes", []):
            f = Path(base) / pack / "pyproject.toml"
            if f.is_file():
                m = re.search(r'^description\s*=\s*"((?:[^"\\]|\\.)*)"', f.read_text(encoding="utf-8"),
                              re.M)
                if m:
                    return " ".join(m.group(1).split())[:400]
    except Exception:  # noqa: BLE001
        pass
    return ""


def workflow_facts(data: dict, name: str, *, object_info: dict | None = None) -> dict[str, Any]:
    """Deterministic facts about workflow *data* (UI graph or API prompt).

    ``{name, task, media, model_families, classes, pack_nodes, pack_about,
    node_descriptions, api_nodes, models, inputs, outputs, notes}`` — every field
    present, empty when unknown. Never raises: a graph a parser cannot read just
    yields fewer facts.
    """
    facts: dict[str, Any] = {"name": name, "task": "", "task_phrase": "", "media": "",
                             "model_families": [], "classes": [], "pack_nodes": {},
                             "pack_about": {}, "node_descriptions": {},
                             "api_nodes": [], "models": [], "inputs": {}, "outputs": {}, "notes": ""}
    if not isinstance(data, dict):
        return facts
    is_ui = isinstance(data.get("nodes"), list)
    if is_ui:
        facts["notes"] = _notes(data)

    if object_info is None:
        try:
            from agenty_core.tools.comfyui import _get_object_info  # noqa: PLC0415
            object_info = _get_object_info() or {}
        except Exception:  # noqa: BLE001
            object_info = {}

    # Intent: the same classifier the recipe database files templates with.
    try:
        from agenty_core.workflow_recipes import parser as _p  # noqa: PLC0415
        from agenty_core.workflow_recipes.intent import IntentClassifier  # noqa: PLC0415
        g = (_p.parse_ui if is_ui else _p.parse_api)(data, name, f"{name}.json", "custom")
        ic = IntentClassifier()
        it = ic.classify(g)
        facts["task"] = it.task or ""
        facts["media"] = it.media or ""
        facts["model_families"] = list(it.model_families or [])
        facts["task_phrase"] = ic.task_phrase(it.task, it.media)
    except Exception:  # noqa: BLE001
        pass

    api = data
    if is_ui:
        try:
            from agenty_core.tools.comfyui import _convert_graph_to_api  # noqa: PLC0415
            api = _convert_graph_to_api(data)
        except Exception:  # noqa: BLE001
            api = {str(n.get("id")): {"class_type": n.get("type"), "inputs": {}}
                   for n in data.get("nodes") or [] if isinstance(n, dict)}
    counts: Counter = Counter()
    for node in (api or {}).values():
        if isinstance(node, dict) and node.get("class_type"):
            counts[str(node["class_type"])] += 1
    facts["classes"] = [c for c, _ in counts.most_common() if c not in _PLUMBING][:40]
    packs: dict[str, list] = {}
    for cls, _n in counts.most_common():
        info = object_info.get(cls) or {}
        pack = _pack_of(info.get("python_module"))
        if pack:
            packs.setdefault(pack, []).append(cls)
        elif info.get("api_node") or str(info.get("python_module") or "").startswith("comfy_api_nodes"):
            facts["api_nodes"].append(cls)
    facts["pack_nodes"] = packs
    # The model an API node is set to: it is a widget value, not a file, so the
    # model-file list misses it — and a writer left to guess named "GPT-4o" for a
    # graph set to gpt-image.
    api_models = []
    for node in (api or {}).values():
        if isinstance(node, dict) and node.get("class_type") in facts["api_nodes"]:
            for key in ("model", "model_name"):
                val = (node.get("inputs") or {}).get(key)
                if isinstance(val, str) and val and val not in api_models:
                    api_models.append(val)
    facts["api_models"] = api_models[:6]
    # What the packs and their nodes say about themselves.
    for pack, nodes in packs.items():
        about = pack_about(pack)
        if about:
            facts["pack_about"][pack] = about
        for cls in nodes[:5]:
            desc = " ".join(str((object_info.get(cls) or {}).get("description") or "").split())
            if desc:
                facts["node_descriptions"][cls] = desc[:200]

    try:
        from agenty_core.utils.workflow_parser import parse_workflow  # noqa: PLC0415
        t = parse_workflow(api, name=name, update_index=False)["templates"][0]
        facts["models"] = list(t.get("models") or [])
        for side in ("inputs", "outputs"):
            facts[side] = dict(Counter(e.get("mediaType") or "?" for e in (t.get("io") or {}).get(side) or []))
    except Exception:  # noqa: BLE001
        pass
    return facts


def _io_phrase(counts: dict, none: str) -> str:
    if not counts:
        return none
    return " + ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))


def describe_from_facts(facts: dict, *, pack: str = "", workflow: str = "") -> str:
    """A description from *facts* alone, in the catalog's shape:
    ``[Local|API] <operation> using <what is distinctive>. <in> -> <out>. <where from>.``

    The first sentence carries the operation and the pack's own nodes (or the
    model family), because that sentence is the one retrieval shows.
    """
    where = "API" if facts.get("api_nodes") else "Local"
    op = str(facts.get("task") or "").strip().replace("_", " ") or (
        f"{facts.get('media') or 'image'} workflow")
    own = (facts.get("pack_nodes") or {}).get(pack) or []
    if pack and own:
        # Nodes before the pack's name: only 60 characters of this are shown, and
        # the pack is already in the template's name.
        using = f"{', '.join(own[:3])} ({pack})"
    elif facts.get("model_families"):
        using = ", ".join(facts["model_families"][:2])
    elif facts.get("api_nodes"):
        using = ", ".join(facts["api_nodes"][:2])
    else:
        others = [n for nodes in (facts.get("pack_nodes") or {}).values() for n in nodes]
        using = ", ".join(others[:3]) or ", ".join((facts.get("classes") or [])[:3])
    first = f"[{where}] {op[:1].upper()}{op[1:]}" + (f" with {using}" if using else "")
    io = (f"{_io_phrase(facts.get('inputs') or {}, 'text only')} in -> "
          f"{_io_phrase(facts.get('outputs') or {}, 'no saved output')} out")
    tail = (f'Example workflow "{workflow or facts.get("name")}" shipped with the {pack} custom '
            f"node pack." if pack else "")
    return ". ".join(p for p in (first, io) if p) + "." + (f" {tail}" if tail else "")
