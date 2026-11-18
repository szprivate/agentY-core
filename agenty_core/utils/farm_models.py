"""Where a farm machine downloads a model from.

A workflow names its models by ComfyUI folder and file name (``diffusion_models/moss_image_sft_bf16.safetensors``).
A rented machine starts with no models, so before a render the farm asks it to download the ones it lacks
(the machine image's ``agenty_farm_helper`` node does the download). This module finds the download address,
in this order:

1. ``config/farm_models.json`` under the project root, written by the user::

       {"models": {"loras/my_style.safetensors": "https://huggingface.co/me/loras/resolve/main/my_style.safetensors"},
        "repos": ["me/my-private-models"]}

   ``models`` maps ``folder/name`` (or just ``name``) to a URL; ``repos`` are extra Hugging Face model repos to
   search like the Comfy-Org ones.
2. Addresses the workflow brought along (``model_urls``: ComfyUI's own templates carry one per loader node).
3. The Comfy-Org repos on Hugging Face (``Comfy-Org/MOSS-Image``, ``Comfy-Org/Z-Image-Turbo``, ...), whose folders
   mirror ComfyUI's model folders. Their file lists are read in one call and kept for a day in
   ``.cache/farm_model_index.json``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

HF = "https://huggingface.co"
COMFY_ORG = "Comfy-Org"
INDEX_MAX_AGE = 24 * 3600

_http: Optional[Callable] = None      # (url, params) -> list | dict ; tests replace it
_root: Optional[Path] = None
_lock = threading.Lock()
_index: Optional[dict] = None


def set_http(fn: Optional[Callable]) -> None:
    global _http, _index
    _http, _index = fn, None


def set_root(path: Optional[Path]) -> None:
    global _root, _index
    _root, _index = (Path(path) if path else None), None


def _project_root() -> Path:
    if _root:
        return _root
    from agenty_core.paths import project_root
    return project_root()


def _get_json(url: str, params: dict):
    if _http:
        return _http(url, params)
    import requests
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _user_file() -> dict:
    p = _project_root() / "config" / "farm_models.json"
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("farm_models: %s is not readable JSON: %s", p, exc)
        return {}


def _repo_files(repo: str) -> list:
    d = _get_json(f"{HF}/api/models/{repo}", {}) or {}
    return [s.get("rfilename") for s in d.get("siblings") or [] if s.get("rfilename")]


def _build_index(extra_repos: list) -> dict:
    """``{basename: [(repo, path), ...]}`` over the Comfy-Org repos and the user's extra repos."""
    files: dict = {}
    rows = _get_json(f"{HF}/api/models", {"author": COMFY_ORG, "full": "true", "limit": 1000}) or []
    for r in rows:
        repo = r.get("id") or r.get("modelId")
        if not repo:
            continue
        if r.get("siblings") is None:          # a listing without file names: ask the repo itself
            try:
                files[repo] = _repo_files(repo)
            except Exception as exc:  # noqa: BLE001
                logger.info("farm_models: cannot list %s: %s", repo, exc)
            continue
        files[repo] = [s.get("rfilename") for s in r["siblings"]]
    for repo in extra_repos:
        try:
            files[repo] = _repo_files(repo)
        except Exception as exc:  # noqa: BLE001
            logger.warning("farm_models: cannot list %s: %s", repo, exc)
    index: dict = {}
    for repo, paths in files.items():
        for p in paths or []:
            if p and p.endswith((".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".onnx")):
                index.setdefault(p.rsplit("/", 1)[-1], []).append([repo, p])
    return index


def _load_index(extra_repos: list) -> dict:
    global _index
    with _lock:
        if _index is not None and time.time() - _index["at"] < INDEX_MAX_AGE and _index["repos"] == extra_repos:
            return _index["files"]
        cache = _project_root() / ".cache" / "farm_model_index.json"
        try:
            c = json.loads(cache.read_text(encoding="utf-8"))
            if time.time() - float(c["at"]) < INDEX_MAX_AGE and c.get("repos") == extra_repos:
                _index = c
                return c["files"]
        except Exception:  # noqa: BLE001
            pass
        files = _build_index(extra_repos)
        _index = {"at": time.time(), "repos": extra_repos, "files": files}
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(_index), encoding="utf-8")
        except OSError:
            pass
        return files


def resolve(folder: str, name: str, model_urls: Optional[dict] = None) -> str:
    """The download URL for ``folder/name``, or ``""`` when no source knows it."""
    key = f"{folder}/{name}".replace("\\", "/")
    base = key.rsplit("/", 1)[-1]
    user = _user_file()
    own = user.get("models") or {}
    for k in (key, name, base):
        if isinstance(own.get(k), str) and own[k].startswith("https://"):
            return own[k]
    for k in (key, name, base):
        u = (model_urls or {}).get(k)
        if isinstance(u, str) and u.startswith("https://"):
            return u
    try:
        index = _load_index([str(r) for r in user.get("repos") or [] if isinstance(r, str)])
    except Exception as exc:  # noqa: BLE001
        logger.warning("farm_models: cannot read the Hugging Face index: %s", exc)
        return ""
    hits = index.get(base) or []
    if not hits:
        return ""
    # prefer a file in the same ComfyUI folder (diffusion_models/..., split_files/diffusion_models/...)
    hits = sorted(hits, key=lambda h: (f"/{folder}/" not in "/" + h[1], not h[0].startswith(COMFY_ORG + "/"),
                                       len(h[1])))
    repo, path = hits[0]
    return f"{HF}/{repo}/resolve/main/{quote(path)}"
