"""Text-encoder files ComfyUI lists but its own loaders cannot read.

A file in ``models/text_encoders`` is offered by CLIPLoader whatever is in it.
Two kinds found on a real machine load without complaint and then fail the
first node that encodes, with ``NotImplementedError: Cannot copy out of meta
tensor; no data!`` — ComfyUI builds the encoder with placeholder ("meta")
weights and fills in what the file holds, so weights the file lacks stay
placeholders:

* one shard of a split checkpoint (``qwen_2511_text_encoder.safetensors``: 17
  tensors, layers 26-27 of 28), and
* a T5 saved in the WanVideoWrapper layout (``umt5-xxl-enc-bf16.safetensors``:
  ``blocks.N.attn.k.weight``), which only that pack's own loader reads.

The error names neither the file nor the reason, and the agent's own guesses
("memory pressure", "wrong loader type") cost attempts. The safetensors header
says which it is, and reading it costs a few kilobytes even over the network.
"""

import json
import logging
import os
import re
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# ComfyUI's own text-encoder loaders (the ones that want the layout core expects).
NATIVE_CLIP_LOADERS = {"CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader", "QuadrupleCLIPLoader"}

_LAYER = re.compile(r"(?:^|\.)(?:layers|layer|blocks|block|h)\.(\d+)\.")
_HEADER_MAX = 64 * 1024 * 1024

_CACHE: dict[tuple, str | None] = {}


def safetensors_keys(path) -> list[str] | None:
    """Tensor names from a .safetensors header, or None when it can't be read."""
    try:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            if not 0 < length <= _HEADER_MAX:
                return None
            header = json.loads(f.read(length))
        return [k for k in header if k != "__metadata__"]
    except Exception:  # noqa: BLE001 — unreadable is "don't know", never an error
        return None


def keys_problem(keys: list[str]) -> str | None:
    """Why a native loader cannot use a text encoder with these tensor names."""
    if not keys:
        return None
    layers = sorted({int(m.group(1)) for k in keys for m in [_LAYER.search(k)] if m})
    if layers and layers[0] > 0:
        return (f"it is one shard of a split checkpoint — it holds only layers {layers[0]}-{layers[-1]} "
                f"({len(keys)} tensors), not the whole encoder")
    if any(k.startswith("blocks.") and ".attn.k." in k for k in keys) and "token_embedding.weight" in keys:
        return ("it is a T5 encoder saved in the WanVideoWrapper layout (blocks.N.attn...), which only "
                "that pack's LoadWanVideoT5TextEncoder reads; ComfyUI's own loaders find no weights in it")
    return None


def _text_encoder_dirs() -> list[str]:
    try:
        from agenty_core.tools.huggingface import _folder_paths  # noqa: PLC0415
        fp = _folder_paths()
    except Exception:  # noqa: BLE001
        return []
    dirs: list[str] = []
    for cat in ("text_encoders", "clip"):
        for p in fp.get(cat, []):
            if p not in dirs:
                dirs.append(p)
    return dirs


def text_encoder_problem(name: str) -> str | None:
    """Why ComfyUI's own loaders cannot read the text encoder listed as *name*
    (``"UMT5\\\\file.safetensors"``), or None — also when the file can't be found
    or inspected. Answers are cached per file size and modification time."""
    name = str(name or "")
    if not name.lower().endswith(".safetensors"):
        return None
    for d in _text_encoder_dirs():
        path = Path(d) / name
        try:
            st = os.stat(path)
        except OSError:
            continue
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in _CACHE:
            keys = safetensors_keys(path)
            _CACHE[key] = keys_problem(keys) if keys else None
        return _CACHE[key]
    return None


def readable_siblings(name: str, inventory: list[str], limit: int = 4) -> list[str]:
    """Other text encoders in the same subfolder as *name* that pass the check."""
    parent = str(name).replace("/", "\\").rpartition("\\")[0].lower()
    out = []
    for entry in inventory or []:
        if entry == name or str(entry).replace("/", "\\").rpartition("\\")[0].lower() != parent:
            continue
        # The inventory spans every model folder (a LoRA can sit in a QWEN\ of
        # its own): only a file that is in a text-encoder folder counts.
        if not any((Path(d) / entry).is_file() for d in _text_encoder_dirs()):
            continue
        if not text_encoder_problem(entry):
            out.append(entry)
            if len(out) >= limit:
                break
    return out


def unloadable_encoder_errors(workflow: dict, inventory: list[str] | None = None) -> list[str]:
    """One line per native text-encoder loader in *workflow* pointed at a file it
    cannot read. The run would pass validation and fail at the first encode."""
    errors: list[str] = []
    for nid, node in (workflow or {}).items():
        if not isinstance(node, dict) or node.get("class_type") not in NATIVE_CLIP_LOADERS:
            continue
        for inp, val in (node.get("inputs") or {}).items():
            if not (inp.startswith("clip_name") and isinstance(val, str)):
                continue
            why = text_encoder_problem(val)
            if not why:
                continue
            alts = readable_siblings(val, inventory or [])
            hint = f" Use one of: {', '.join(alts)}." if alts else ""
            errors.append(
                f"Node {nid} ({node['class_type']}): {inp} '{val}' cannot be loaded by this node: {why}. "
                f"The run would fail at the first text encode with \"Cannot copy out of meta tensor; "
                f"no data!\".{hint}")
    return errors
