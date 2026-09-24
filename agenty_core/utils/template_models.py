"""Where the official templates say each model file belongs.

Every official template states it, twice: loader nodes carry
``properties.models = [{"name", "url", "directory"}]``, and the MarkdownNote under
"Model Storage Location" draws the same thing as a folder tree::

    📂 ComfyUI/
    ├── 📂 models/
    │   ├── 📂 diffusion_models/
    │   │   └── trellis_2_int8_convrot.safetensors

That is the template author's word, so it outranks every guess download_hf_model
can make. Guessing is what put trellis_2_int8_convrot.safetensors into
``checkpoints``: the agent named the sampler stage, not the loader, as the node,
and the filename matches no naming rule.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_MODEL_EXT = r"(?:safetensors|sft|ckpt|pt|pth|bin|gguf|onnx)"
_TREE_ROW = re.compile(r"^(?P<indent>[\s│├└─|`+\-]*)(?:📂\s*)?(?P<name>[^\s/│├└─][^/│]*?)/?\s*$")


def _tree_models(text: str) -> dict[str, str]:
    """``{filename: directory}`` from a markdown folder tree below ``models/``."""
    out: dict[str, str] = {}
    stack: list[tuple[int, str]] = []          # (depth, folder name)
    for line in text.splitlines():
        m = _TREE_ROW.match(line.rstrip())
        if not m:
            continue
        depth = len(m.group("indent"))
        name = m.group("name").strip()
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if re.search(rf"\.{_MODEL_EXT}$", name, re.I):
            folders = [f for _d, f in stack]
            if "models" in folders:
                rel = folders[folders.index("models") + 1:]
                if rel:
                    out.setdefault(name, "/".join(rel))
        else:
            stack.append((depth, name.rstrip("/")))
    return out


def models_in_workflow(wf: dict) -> dict[str, dict]:
    """``{filename: {"directory", "url"?}}`` a UI-format template declares."""
    found: dict[str, dict] = {}
    notes: list[str] = []

    def walk(graph: dict) -> None:
        for n in graph.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            for m in (n.get("properties") or {}).get("models") or []:
                if isinstance(m, dict) and m.get("name") and m.get("directory"):
                    found.setdefault(m["name"], {"directory": m["directory"], "url": m.get("url", "")})
            if n.get("type") in ("MarkdownNote", "Note"):
                wv = n.get("widgets_values") or []
                if wv and isinstance(wv[0], str):
                    notes.append(wv[0])

    walk(wf)
    for sg in (wf.get("definitions") or {}).get("subgraphs") or []:
        walk(sg)
    for text in notes:
        if "models" not in text:
            continue
        for name, directory in _tree_models(text).items():
            found.setdefault(name, {"directory": directory, "url": ""})
    return found


@lru_cache(maxsize=1)
def template_model_index() -> dict[str, dict]:
    """Every model filename any template names, with its folder and download URL."""
    from agenty_core.paths import corpus_root  # noqa: PLC0415
    index: dict[str, dict] = {}
    root = corpus_root()
    for sub in ("comfyui_workflow_templates_official", "comfyui_workflow_templates_custom"):
        for p in sorted((root / sub / "templates").glob("*.json")):
            if p.name == "index.json":
                continue
            try:
                wf = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(wf, dict) or "nodes" not in wf:
                continue
            for name, info in models_in_workflow(wf).items():
                entry = index.setdefault(name, {"directory": info["directory"], "url": info["url"],
                                                "template": p.stem})
                if not entry["url"] and info["url"]:
                    entry["url"] = info["url"]
    return index


def template_model_dir(filename: str) -> str | None:
    """The folder under ``models/`` the templates put *filename* in, else None."""
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    hit = template_model_index().get(name)
    return hit["directory"] if hit else None
