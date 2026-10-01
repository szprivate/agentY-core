"""
Hugging Face integration tools for agentY.

Provides @tool-decorated functions for discovering and downloading models
from the Hugging Face Hub via its HTTP API.

Environment variables:
    HF_TOKEN            – Hugging Face access token (required for gated models)
    COMFYUI_MODELS_DIR  – Base directory where ComfyUI stores models
                          (falls back to config/settings.json → comfyui_models_dir,
                           then to the sensible default D:/AI/ComfyUI/models)

Note on ``find_hf_file``:
    The HF search API tokenises on word boundaries — dots and full extensions
    break tokenisation entirely, so searching with a raw filename like
    ``gemma_3_12B_it_fp4_mixed.safetensors`` returns nothing.  The tool first
    does a single bulk scan of the Comfy-Org HF organisation (which repackages
    models specifically for ComfyUI and includes files like gemma text encoders
    and LTX checkpoints).  If that misses, it falls back to progressively
    broader stem-based queries (full stem → 4-token prefix → 3-token → 2-token
    → hints), verifying each candidate repo's sibling list for an exact match.
    When no exact match exists the tool returns close variants so the agent can
    pick the nearest available file.
"""

import io
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from agenty_core.utils.model_node_mapping import (
    NODE_TO_FOLDER,
    get_storage_path,
    guess_folder_from_filename,
)
from agenty_core.utils.progress_signal import push as _push_progress
from agenty_core.utils.secrets import get_secret
from agenty_core._compat import tool
from agenty_core.paths import project_root as _app_root

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api/models"



def _hf_headers() -> dict:
    """Return request headers including HF auth token if available."""
    headers = {"Accept": "application/json"}
    token = get_secret("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _models_base_dir() -> Path:
    """Resolve the ComfyUI models base directory.

    Priority:
    1. COMFYUI_MODELS_DIR env var
    2. comfyui_models_dir key in config/settings.json
    3. Default: D:/AI/ComfyUI/models
    """
    env_dir = get_secret("COMFYUI_MODELS_DIR")
    if env_dir:
        return Path(env_dir)

    config_path = _app_root() / "config" / "settings.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.loads("".join(ln for ln in f if not ln.lstrip().startswith("//")))
            d = config.get("comfyui_models_dir")
            if d:
                return Path(d)
            base = config.get("comfyui_base_dir")
            if base:
                return Path(base) / "models"
        except Exception:
            pass

    return Path("D:/AI/ComfyUI/models")


_FOLDER_PATHS_CACHE: dict | None = None


def _folder_paths() -> dict:
    """The running ComfyUI's actual model folder paths per category, from its
    ``/internal/folder_paths`` endpoint: ``{category: [abs_path, ...]}``.

    Authoritative source — it includes the *additional* model paths configured at
    server startup (extra_model_paths.yaml, e.g. an L: drive). Cached per session."""
    global _FOLDER_PATHS_CACHE
    if _FOLDER_PATHS_CACHE is not None:
        return _FOLDER_PATHS_CACHE
    fp: dict = {}
    try:
        from agenty_core.utils.comfyui_client import get_client
        r = get_client().get("/internal/folder_paths")
        if isinstance(r, dict):
            fp = {str(k).lower(): [str(p) for p in v]
                  for k, v in r.items() if isinstance(v, list)}
    except Exception:
        fp = {}
    _FOLDER_PATHS_CACHE = fp
    return fp


def _resolve_download_dir(
    node_class_type: str, destination_folder: str, filename: str = "", hf_subfolder: str = ""
) -> tuple[Path, str]:
    """Resolve where to download a model, preferring the *additional* ComfyUI model
    path (given at server startup) for the model's category so the file lands where
    ComfyUI actually loads from — not the default ``models/`` dir, which is often on
    a different drive. Returns (dir, source).

    Category resolution, most authoritative first:
      0. the templates' own word for this filename (properties.models /
         the "Model Storage Location" note — see utils.template_models),
      1. the loader node class (NODE_TO_FOLDER),
      2. an explicit destination_folder, else the HF repo subfolder when its
         leaf is a folder ComfyUI knows (Comfy-Org repos mirror the layout),
      3. the filename itself (guess_folder_from_filename — VAE / LoRA / CLIP /
         ControlNet / upscaler / diffusion-model conventions),
      4. else 'checkpoints' (always scanned; correct for a bare checkpoint)."""
    try:
        from agenty_core.utils.template_models import template_model_dir  # noqa: PLC0415
        declared = template_model_dir(filename) or ""
    except Exception:  # noqa: BLE001
        declared = ""
    category = declared.split("/")[0].lower() or None
    if category:
        # The template's folder, and any subfolder it names inside it: the loader
        # widget then holds "subfolder/file", so the file has to be there too.
        dl_dir, source = _category_dir(category)
        rest = declared.split("/")[1:]
        return (dl_dir.joinpath(*rest) if rest else dl_dir), f"template:{source}"
    if node_class_type and node_class_type in NODE_TO_FOLDER:
        category = NODE_TO_FOLDER[node_class_type].split("models/", 1)[-1].split("/")[0].lower()
    elif destination_folder:
        category = destination_folder.replace("\\", "/").strip("/").split("/")[0].lower()
    elif hf_subfolder:
        # A sampler stage named as the "node" (Trellis2ShapeStage) is in no map,
        # but the repo path said diffusion_models all along.
        leaf = hf_subfolder.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()
        if leaf in {k.lower() for k in _folder_paths()}:
            category = leaf

    # No loader/destination hint → classify by filename convention. A bare model
    # dropped in the models root would never be scanned (so never loaded or
    # verified), so this keeps peripheral types (vae/loras/clip/controlnet/…) in
    # their own folders instead of all defaulting to checkpoints.
    if not category and filename:
        guessed = guess_folder_from_filename(filename)
        if guessed:
            category = guessed.split("models/", 1)[-1].split("/")[0].lower()

    # Last resort: an uncategorised file → 'checkpoints' (always scanned, and the
    # right home for the common case of an uncategorised checkpoint).
    if not category:
        category = "checkpoints"
    return _category_dir(category)


def _category_dir(category: str) -> tuple[Path, str]:
    """The directory ComfyUI loads *category* (e.g. 'vae') from, and its source."""
    # Every ComfyUI folder-path whose leaf dir matches the category.
    cand = [p for paths in _folder_paths().values() for p in paths
            if p.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower() == category]
    if cand:
        extra_base = str(_models_base_dir()).replace("\\", "/").rstrip("/").lower()
        # 1) a path under the configured extra base (e.g. L:/.../Models)
        for p in cand:
            if p.replace("\\", "/").lower().startswith(extra_base):
                return Path(p), "comfyui_extra_path"
        # 2) else a path on a different drive than the (default) first candidate
        d0 = cand[0].split(":", 1)[0].lower()
        for p in cand[1:]:
            if p.split(":", 1)[0].lower() != d0:
                return Path(p), "comfyui_extra_path"
        return Path(cand[0]), "comfyui_folder_path"
    return _models_base_dir() / category, "models_base_dir"


# Never let a model download fill the C: system drive, and keep a safety margin
# of free space on whatever drive we do download to.
_MIN_FREE_BUFFER_GB = 5.0


def _drive_of(p) -> str:
    """Uppercase drive letter of a path (e.g. 'C'), or '' if it has none."""
    return os.path.splitdrive(str(p))[0].rstrip(":").upper()


def _free_gb(p) -> float:
    """Free space (GiB) on the volume holding *p*, walking up to an existing
    ancestor since the target dir may not exist yet. inf if it can't be probed."""
    probe = Path(p)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free / (1024 ** 3)
    except Exception:
        return float("inf")


def _ensure_not_c_drive(dl_dir: Path, dl_source: str) -> tuple[Path, str]:
    """Never download models onto the C: system drive. If the resolved dir is on
    C:, redirect to the first configured non-C: model location (preserving the
    category leaf dir); raise if none is configured."""
    if _drive_of(dl_dir) != "C":
        return dl_dir, dl_source
    leaf = Path(dl_dir).name
    # Prefer a models *root* (so the category leaf nests cleanly), then any
    # configured folder path, as long as it is not on C:.
    bases = [str(_models_base_dir())]
    bases += [p for paths in _folder_paths().values() for p in paths]
    for b in bases:
        if _drive_of(b) and _drive_of(b) != "C":
            bp = Path(b)
            target = bp if bp.name.lower() == leaf.lower() else bp / leaf
            logger.warning("download target was on C: — redirecting to %s", target)
            return target, "redirected_off_c"
    raise RuntimeError(
        "Refusing to download to the C: system drive and no non-C: model "
        "location is configured. Set COMFYUI_MODELS_DIR / comfyui_models_dir "
        "to a non-C: drive."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _refresh_model_cache() -> None:
    """Refresh config/models.json.

    Mirrors the logic in scripts/refresh_models.py so that the in-process
    tool call updates the cache immediately after a download, without needing
    to shell out to the script.
    """
    try:
        from agenty_core.utils.comfyui_retrieve_models_customnodes import fetch_available_models
    except Exception as exc:
        logger.warning("_refresh_model_cache: could not import refresh utils: %s", exc)
        return

    try:
        available = fetch_available_models()
    except Exception as exc:
        logger.warning("_refresh_model_cache: ComfyUI unreachable – skipping cache refresh: %s", exc)
        return

    models_path = _app_root() / "config" / "models.json"
    if models_path.exists():
        try:
            raw = "".join(ln for ln in models_path.read_text(encoding="utf-8").splitlines(keepends=True)
                          if not ln.lstrip().startswith("//"))
            models_data = json.loads(raw) if raw.strip() else {}
        except Exception:
            models_data = {}
    else:
        models_data = {}

    models_data["available"] = available
    with open(models_path, "w", encoding="utf-8") as f:
        json.dump(models_data, f, indent=2)

    total = sum(len(v) for v in available.values() if isinstance(v, list))
    logger.info("[refresh_models] models.json refreshed – %d folders, %d files", len(available), total)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_huggingface_models(
    query: str,
    filter_tag: str = "",
    limit: int = 10,
    full_text_search: bool = False,
) -> str:
    """Search the Hugging Face Hub for models by keyword.

    Args:
        query: Search string e.g. 'flux lora', 'wan2.1 video'.
        filter_tag: Optional pipeline/library tag e.g. 'diffusers', 'flux'.
        limit: Max results (default 10, max 50).
        full_text_search: When True, enables full-text search so the query also
            matches README / model card content — useful when a specific
            filename is referenced in a model card but not in the repo name
            (e.g. community quantization files).
    """
    try:
        params: dict = {
            "search": query,
            "limit": min(limit, 50),
            "sort": "downloads",
            "direction": "-1",
        }
        if filter_tag:
            params["filter"] = filter_tag
        if full_text_search:
            params["full_text_search"] = "true"

        resp = requests.get(
            HF_API_BASE,
            headers=_hf_headers(),
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        models = resp.json()

        results = []
        for m in models:
            results.append({
                "model_id": m.get("modelId") or m.get("id", ""),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "tags": m.get("tags", []),
                "last_modified": m.get("lastModified", ""),
            })

        return json.dumps({"ok": True, "count": len(results), "models": results})
    except requests.HTTPError as exc:
        logger.error("HF API HTTP error in search: %s", exc)
        return json.dumps({"ok": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"})
    except Exception as exc:
        logger.error("Error in search_huggingface_models: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


@tool
def find_hf_file(filename: str, hints: str = "") -> str:
    """Locate which Hugging Face repo(s) host a specific file.

    Useful for finding community quantizations or obscure model files whose
    repo name does not obviously match the filename.

    Strategy (fully automatic, no agent loop needed):

    Pass 0 – Comfy-Org bulk scan.  A single HF API call with ``author=Comfy-Org``
              and ``full=true`` returns all ~60 Comfy-Org repos with their full
              sibling lists inline.  Comfy-Org repos are curated specifically for
              ComfyUI and are checked first.  Stops here on an exact match.

    The HF search API tokenises on word boundaries — dots and full extensions
    break matching entirely, so passes 1-5 never search with the raw filename.
    Instead, progressively broader queries are built from the filename stem:

    Pass 1 – exact stem (e.g. ``ltx-2.3-22b-dev-fp8``).
    Pass 2 – first 4 hyphen/underscore tokens (``ltx 2 3 22b``).
    Pass 3 – first 3 tokens (``ltx 2 3``).
    Pass 4 – first 2 tokens (``ltx 2``).
    Pass 5 – ``hints`` string alone (if provided).

    Each pass fetches up to 10 candidate repos and checks every candidate's
    sibling list for the exact filename.  Searching stops as soon as at least
    one verified exact match is found.

    If no exact match is found across all passes, the tool returns the best
    *close matches* (files from the same repo whose stem shares a prefix with
    the requested filename) so the agent can make an informed decision.

    Args:
        filename: Exact filename to locate e.g.
            'gemma_3_12B_it_fp4_mixed.safetensors'.
        hints: Optional extra search keywords to narrow / broaden results
            e.g. 'gemma 12b quantized'.  Also used as a dedicated search pass
            when the stem-based passes return nothing.

    Returns:
        JSON ``{"ok": true, "count": N, "matches": [...]}`` where each match
        has ``repo_id``, ``filename``, ``subfolder``, ``url``, and
        ``exact`` (bool).  Feeds directly into ``download_hf_model``.
    """
    def _make_match(repo_id: str, rfilename: str, *, exact: bool) -> dict:
        if "/" in rfilename:
            subfolder, found_name = rfilename.rsplit("/", 1)
        else:
            subfolder, found_name = "", rfilename
        return {
            "repo_id": repo_id,
            "filename": found_name,
            "subfolder": subfolder,
            "url": f"https://huggingface.co/{repo_id}/resolve/main/{rfilename}",
            "exact": exact,
        }

    def _fetch_siblings(model_id: str) -> list[dict]:
        try:
            r = requests.get(
                f"{HF_API_BASE}/{model_id}",
                headers=_hf_headers(),
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("siblings", [])
        except Exception as exc:
            logger.warning("find_hf_file: could not fetch siblings for %s: %s", model_id, exc)
            return []

    def _search_candidates(query: str, limit: int = 10) -> list[str]:
        """Return list of model_ids from an HF API search."""
        try:
            r = requests.get(
                HF_API_BASE,
                headers=_hf_headers(),
                params={
                    "search": query,
                    "limit": limit,
                    "sort": "downloads",
                    "direction": "-1",
                    "full_text_search": "true",
                },
                timeout=30,
            )
            r.raise_for_status()
            return [m.get("modelId") or m.get("id", "") for m in r.json() if m.get("modelId") or m.get("id")]
        except Exception as exc:
            logger.warning("find_hf_file: search query %r failed: %s", query, exc)
            return []

    def _scan_org_repos(org: str, fname: str, ext: str, stem_prefix: str) -> tuple[list[dict], list[dict]]:
        """Bulk-scan all repos in *org* for *fname* using a single API call.

        Returns (exact_matches, close_matches).
        """
        exact: list[dict] = []
        close: list[dict] = []
        try:
            r = requests.get(
                HF_API_BASE,
                headers=_hf_headers(),
                params={"author": org, "limit": 200, "full": "true"},
                timeout=30,
            )
            r.raise_for_status()
            repos = r.json()
            logger.info("find_hf_file: Comfy-Org scan – %d repos fetched", len(repos))
            for repo in repos:
                mid = repo.get("modelId") or repo.get("id", "")
                if not mid:
                    continue
                found_exact = False
                for sibling in repo.get("siblings", []):
                    rf = sibling.get("rfilename", "")
                    tail = rf.rsplit("/", 1)[-1] if "/" in rf else rf
                    if tail == fname:
                        exact.append(_make_match(mid, rf, exact=True))
                        found_exact = True
                        break
                if not found_exact and ext:
                    for sibling in repo.get("siblings", []):
                        rf = sibling.get("rfilename", "")
                        tail = rf.rsplit("/", 1)[-1] if "/" in rf else rf
                        if tail.endswith("." + ext) and stem_prefix in tail.lower():
                            close.append(_make_match(mid, rf, exact=False))
                            break
        except Exception as exc:
            logger.warning("find_hf_file: Comfy-Org scan failed: %s", exc)
        return exact, close

    # Build stem-based query passes
    stem = filename.rsplit(".", 1)[0]          # strip extension
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    # Normalise to hyphen-separated tokens (underscores and dots become hyphens)
    tokens = [t for t in stem.replace("_", "-").replace(".", "-").split("-") if t]
    stem_prefix = stem[:min(8, len(stem))].replace("_", "-").replace(".", "-").lower()

    queries: list[str] = [stem]  # Pass 1: full stem
    for n in (4, 3, 2):
        if len(tokens) > n:
            q = " ".join(tokens[:n])
            if q not in queries:
                queries.append(q)
    if hints and hints not in queries:
        queries.append(hints)  # Pass 5: hints alone

    close_matches: list[dict] = []
    seen_repos: set[str] = set()
    exact_matches: list[dict] = []

    # ------------------------------------------------------------------
    # Pass -1: a template names this file — its link and folder are the
    # author's, so no search can do better.
    # ------------------------------------------------------------------
    try:
        from agenty_core.utils.template_models import template_model_index  # noqa: PLC0415
        declared = template_model_index().get(filename.replace("\\", "/").rsplit("/", 1)[-1])
    except Exception:  # noqa: BLE001
        declared = None
    hf = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$", (declared or {}).get("url", ""))
    if hf:
        match = _make_match(hf.group(1), hf.group(2), exact=True)
        match["directory"] = declared["directory"]
        match["source"] = f"template {declared['template']}"
        return json.dumps({"ok": True, "count": 1, "matches": [match]})

    # ------------------------------------------------------------------
    # Pass 0: Comfy-Org bulk scan (single request, all repos + siblings)
    # ------------------------------------------------------------------
    comfy_exact, comfy_close = _scan_org_repos("Comfy-Org", filename, ext, stem_prefix)
    for m in comfy_exact:
        seen_repos.add(m["repo_id"])
    for m in comfy_close:
        seen_repos.add(m["repo_id"])
    exact_matches.extend(comfy_exact)
    close_matches.extend(comfy_close)

    if exact_matches:
        logger.info("find_hf_file: Comfy-Org exact match found – skipping further passes")
        return json.dumps({"ok": True, "count": len(exact_matches[:5]), "matches": exact_matches[:5]})

    # ------------------------------------------------------------------
    # Passes 1-N: stem-based HF-wide search
    # ------------------------------------------------------------------
    for pass_num, query in enumerate(queries, 1):
        candidates = _search_candidates(query)
        logger.info(
            "find_hf_file: pass %d query=%r → %d candidates",
            pass_num, query, len(candidates),
        )

        for model_id in candidates:
            if model_id in seen_repos:
                continue
            seen_repos.add(model_id)

            siblings = _fetch_siblings(model_id)

            for sibling in siblings:
                rf = sibling.get("rfilename", "")
                tail = rf.rsplit("/", 1)[-1] if "/" in rf else rf

                if tail == filename:
                    exact_matches.append(_make_match(model_id, rf, exact=True))
                    break  # one exact hit per repo is enough

            else:
                # No exact match — look for close variant in same extension
                if ext:
                    for sibling in siblings:
                        rf = sibling.get("rfilename", "")
                        tail = rf.rsplit("/", 1)[-1] if "/" in rf else rf
                        if (
                            tail.endswith("." + ext)
                            and stem_prefix in tail.lower()
                        ):
                            close_matches.append(_make_match(model_id, rf, exact=False))
                            break  # one close match per repo

        if exact_matches:
            logger.info(
                "find_hf_file: found %d exact match(es) on pass %d – stopping",
                len(exact_matches), pass_num,
            )
            break

    # Return exact matches if found, otherwise best close matches (up to 5)
    if exact_matches:
        results = exact_matches[:5]
    else:
        results = close_matches[:5]
        if results:
            logger.info(
                "find_hf_file: no exact match for '%s' – returning %d close variant(s)",
                filename, len(results),
            )
        else:
            logger.info("find_hf_file: no match or close variant found for '%s'", filename)

    return json.dumps({"ok": True, "count": len(results), "matches": results})


@tool
def get_model_info(model_id: str) -> str:
    """Fetch metadata and file list for a specific Hugging Face model.

    Args:
        model_id: HF model identifier e.g. 'black-forest-labs/FLUX.1-dev'.
    """
    try:
        url = f"{HF_API_BASE}/{model_id}"
        resp = requests.get(url, headers=_hf_headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Extract the file listing from siblings
        files = []
        for s in data.get("siblings", []):
            files.append({
                "filename": s.get("rfilename", ""),
                "size": s.get("size"),
            })

        result = {
            "ok": True,
            "model_id": data.get("modelId") or data.get("id", model_id),
            "pipeline_tag": data.get("pipeline_tag", ""),
            "tags": data.get("tags", []),
            "license": data.get("cardData", {}).get("license", "unknown") if isinstance(data.get("cardData"), dict) else "unknown",
            "gated": data.get("gated", False),
            "downloads": data.get("downloads", 0),
            "likes": data.get("likes", 0),
            "last_modified": data.get("lastModified", ""),
            "files": files,
        }
        return json.dumps(result)
    except requests.HTTPError as exc:
        logger.error("HF API HTTP error in get_model_info: %s", exc)
        return json.dumps({"ok": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"})
    except Exception as exc:
        logger.error("Error in get_model_info: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})


import threading as _threading

# Stop for downloads. A download runs in a worker thread, which cancelling the
# turn cannot reach, so pressing Stop left a multi-GB fetch running and the turn
# waiting on it. The flag is checked between chunks, and the open responses are
# closed so a read that is waiting on the network ends at once. The partial
# (.downloading) file stays, and the next attempt resumes it.
_download_cancel = _threading.Event()
_active_downloads: set = set()
_active_lock = _threading.Lock()

_CANCELLED_MSG = ("Download stopped by the user. The partial file is kept; downloading the same "
                  "file again resumes where it stopped.")


class DownloadCancelled(Exception):
    """The user stopped the download."""


def cancel_downloads() -> int:
    """Stop every model download in progress. Returns how many were running."""
    _download_cancel.set()
    with _active_lock:
        resps = list(_active_downloads)
    for r in resps:
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    return len(resps)


def clear_download_cancel() -> None:
    """Allow downloads again — at the start of the next turn."""
    _download_cancel.clear()


@tool
def download_hf_model(
    model_id: str,
    filename: str,
    node_class_type: str = "",
    destination_folder: str = "",
    subfolder: str = "",
) -> str:
    """Download a file from a HuggingFace repo. Check model availability with check_model first.

    A file an official template names goes where that template says (its
    loader's model list / "Model Storage Location" note) — whatever else is
    passed. find_hf_file returns that folder as ``directory``.

    Otherwise prefer supplying *node_class_type* (the ComfyUI class name of the node that
    references the model, e.g. ``"UNETLoader"``).  The correct storage folder is
    then derived automatically via the NODE_TO_FOLDER mapping.  If
    *node_class_type* is unknown or omitted, fall back to *destination_folder*
    (relative path under the models base dir, e.g. ``"FLUX1"``).

    Args:
        model_id: HF model ID e.g. 'black-forest-labs/FLUX.1-dev'.
        filename: File to download e.g. 'flux1-dev.safetensors'.
        node_class_type: ComfyUI node class that loads this model
            e.g. 'UNETLoader', 'CheckpointLoaderSimple', 'LoraLoader'.
            Used to resolve the correct model sub-folder automatically.
        destination_folder: Fallback – target subfolder under models dir
            e.g. 'FLUX1'.  Ignored when *node_class_type* is provided.
        subfolder: Subfolder within the HF repo e.g. 'transformer'.
    """
    # Reliability/offline mode: when downloads are disabled, fail fast so the
    # caller treats the model as unavailable (a missing-model blocker) instead
    # of pulling multi-GB files. Set AGENTY_DISABLE_DOWNLOADS=1 to enable.
    if os.environ.get("AGENTY_DISABLE_DOWNLOADS"):
        return json.dumps({
            "ok": False,
            "skipped": True,
            "error": f"Model download is disabled in this environment; "
                     f"'{filename}' is not installed.",
            "hint": "Treat this model as unavailable: report it as a missing "
                    "model and do not retry the download.",
        })
    try:
        # Resolve destination: prefer the ComfyUI extra model path (the additional
        # model path given at server startup — often a different, larger drive)
        # for the model's category, so the file lands where ComfyUI actually loads
        # from. Falls back to the models base dir.
        dl_dir, dl_source = _resolve_download_dir(node_class_type, destination_folder, filename, subfolder)
        dl_dir, dl_source = _ensure_not_c_drive(dl_dir, dl_source)  # never fill C:
        dest_path = dl_dir / filename
        logger.info("download target: %s (%s)", dest_path, dl_source)

        dest_dir = dest_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Safety: don't re-download if already present
        if dest_path.exists():
            return json.dumps({
                "ok": True,
                "path": str(dest_path),
                "message": "File already exists — skipping download.",
                "size_mb": round(dest_path.stat().st_size / (1024 * 1024), 2),
            })

        # Build the download URL
        if subfolder:
            url = f"https://huggingface.co/{model_id}/resolve/main/{subfolder}/{filename}"
        else:
            url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"

        headers = {}
        token = get_secret("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        logger.info("Downloading %s from %s …", filename, model_id)
        _push_progress(f"⬇️ Starting download: **{filename}** from `{model_id}`")

        # Resume a prior partial (.downloading) via an HTTP Range request, so a
        # large model interrupted by a kill continues instead of restarting from
        # zero. HF's S3-backed storage supports ranged GETs (206).
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".downloading")
        resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"

        if _download_cancel.is_set():
            return json.dumps({"ok": False, "cancelled": True, "error": _CANCELLED_MSG})
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()

        resumed = resp.status_code == 206
        if resume_from and not resumed:
            resume_from = 0  # server ignored the Range header → full restart
        if resume_from:
            logger.info("resuming %s from %.1f GB", filename, resume_from / (1024 ** 3))
        remaining = int(resp.headers.get("content-length", 0))
        total_size = remaining + resume_from
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks

        need_gb = total_size / (1024 ** 3)

        # Size guard: an oversized model (e.g. a 43 GB diffusion checkpoint pulled
        # in as a template's "missing model") can blow a per-recipe timeout and
        # get the whole sweep killed mid-transfer. AGENTY_MAX_DOWNLOAD_GB caps the
        # inline download; over it we skip (treat as unavailable) so the caller
        # falls back instead of blocking. Unset = no cap.
        _cap = os.environ.get("AGENTY_MAX_DOWNLOAD_GB")
        if _cap:
            try:
                cap_gb = float(_cap)
            except ValueError:
                cap_gb = 0.0
            if cap_gb > 0 and need_gb > cap_gb:
                resp.close()
                return json.dumps({
                    "ok": False,
                    "skipped": True,
                    "error": f"'{filename}' is {need_gb:.1f} GB, over the "
                             f"AGENTY_MAX_DOWNLOAD_GB={cap_gb:.0f} GB inline cap.",
                    "hint": "Pre-download this large model out-of-band, or raise "
                            "AGENTY_MAX_DOWNLOAD_GB; treated as unavailable for now.",
                })

        # Space guard: don't start a download that would fill the destination
        # drive. Only the *remaining* bytes are written (a resume reuses the
        # partial already on disk). Treat "no room" as a missing-model blocker
        # rather than crashing mid-write or exhausting the disk.
        remaining_gb = remaining / (1024 ** 3)
        free_gb = _free_gb(dest_dir)
        if remaining and free_gb < remaining_gb + _MIN_FREE_BUFFER_GB:
            resp.close()
            return json.dumps({
                "ok": False,
                "skipped": True,
                "error": f"Not enough free space on {_drive_of(dest_dir)}: to "
                         f"download '{filename}': need {remaining_gb:.1f} GB (+"
                         f"{_MIN_FREE_BUFFER_GB:.0f} GB buffer), "
                         f"{free_gb:.1f} GB free.",
                "hint": "Free up space or configure a larger non-C: model drive; "
                        "treat this model as unavailable for now.",
            })

        # --- tqdm setup -------------------------------------------------
        # _TqdmSignalWriter intercepts tqdm's output, strips ANSI / carriage
        # returns, and pushes each refreshed bar line to the progress signal
        # (visible in Chainlit) as well as writing to the real stderr
        # (visible in the terminal).
        class _TqdmSignalWriter(io.RawIOBase):
            """File-like wrapper that tees tqdm output to progress_signal."""

            def __init__(self, real_file):
                self._real = real_file
                self._last_pushed: str = ""

            def write(self, s: str) -> int:  # tqdm always passes str
                self._real.write(s)
                # Strip carriage-returns / ANSI escapes to get a clean line.
                clean = s.replace("\r", "").replace("\n", "").strip()
                # Remove ANSI escape codes (colours)
                import re as _re
                clean = _re.sub(r"\x1b\[[0-9;]*m", "", clean)
                if clean and clean != self._last_pushed:
                    self._last_pushed = clean
                    _push_progress(f"⬇️ [{clean}]")
                return len(s)

            def flush(self) -> None:
                self._real.flush()

        _tqdm_writer = _TqdmSignalWriter(sys.stderr)

        # Write into the .downloading temp (append when resuming), rename on
        # completion. The partial is deliberately kept on error so the next
        # attempt resumes from where it stopped rather than restarting.
        with _active_lock:
            _active_downloads.add(resp)
        try:
            with open(tmp_path, "ab" if resume_from else "wb") as f, tqdm(
                total=total_size if total_size > 0 else None,
                initial=resume_from,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
                file=_tqdm_writer,
                dynamic_ncols=False,
                ncols=60,
                leave=True,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if _download_cancel.is_set():
                        raise DownloadCancelled(filename)
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Rename temp → final
            tmp_path.rename(dest_path)
        finally:
            with _active_lock:
                _active_downloads.discard(resp)

        size_mb = round(dest_path.stat().st_size / (1024 * 1024), 2)

        # Refresh the model cache so check_model sees the new file immediately.
        logger.info("[download_hf_model] Refreshing model cache after download…")
        _refresh_model_cache()

        return json.dumps({
            "ok": True,
            "path": str(dest_path),
            "size_mb": size_mb,
            "message": f"Downloaded {filename} ({size_mb} MB) to {dest_dir}/",
        })
    except requests.HTTPError as exc:
        status = exc.response.status_code
        body = exc.response.text[:400]
        logger.error("HF download HTTP error: %s %s", status, body)
        if status == 401:
            hint = " — Is HF_TOKEN set and authorised for this gated model?"
        elif status == 403:
            hint = " — Access denied. You may need to accept the model's license on HF."
        elif status == 404:
            hint = " — File not found. Check model_id, subfolder, and filename."
        else:
            hint = ""
        return json.dumps({"ok": False, "error": f"HTTP {status}{hint}: {body}"})
    except Exception as exc:
        # A closed response mid-read raises whatever urllib3 raises; the flag
        # says what actually happened.
        if _download_cancel.is_set():
            logger.info("download_hf_model: %s stopped by the user", filename)
            return json.dumps({"ok": False, "cancelled": True, "error": _CANCELLED_MSG})
        logger.error("Error in download_hf_model: %s", exc, exc_info=True)
        return json.dumps({"ok": False, "error": str(exc)})
