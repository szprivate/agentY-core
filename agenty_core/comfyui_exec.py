"""ComfyUI execution primitives shared by the headless batch worker.

These are the framework-agnostic pieces of the workflow executor that the
detached batch runner (:mod:`agenty_core.batch_runner`) needs: submit a
workflow, read back its output files, and resolve each output to a real on-disk
path.  They talk to ComfyUI over HTTP and spend no model tokens.

The full per-app executors (``src/executor.py`` in each repo) keep their own
copies of the streaming generators because they diverge — agentY weaves in an
Ollama Vision-QA pass, agentY-mcp returns image content to the host model.  Only
these three primitives plus their dir/config helpers are common, so they live
here once for the batch worker to import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agenty_core.paths import project_root as _project_root

logger = logging.getLogger("agentY.comfyui_exec")


def _load_config() -> dict:
    config_path = _project_root() / "config" / "settings.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.loads("".join(ln for ln in f if not ln.lstrip().startswith("//")))
    return {}


def _output_dir() -> Path:
    """Where a result is put when it is not already in ComfyUI's output folder.

    ``output_dir`` in settings is read against ComfyUI's ``--output-directory``:
    blank is that folder itself, a relative path a folder inside it, an absolute
    path is used as it is. Only when ComfyUI's folder is not on this machine (a
    remote ComfyUI) or cannot be asked does it fall back to this checkout.
    """
    cfg = _load_config()
    od = str(cfg.get("output_dir") or "").strip()
    if od and Path(od).is_absolute():
        return Path(od).resolve()
    comfy = _get_comfyui_output_dir()
    if comfy is not None and comfy.is_dir():
        return (comfy / od).resolve() if od else comfy
    return (_project_root() / (od or "./output_images/")).resolve()


# --- ComfyUI dir cache -----------------------------------------------------
# /system_stats returns ComfyUI's argv, which is constant for the lifetime of
# the server process. --output-directory is parsed once and memoised so
# per-output-file resolution doesn't trigger a new HTTP roundtrip every call.
_COMFYUI_DIR_CACHE_LOADED: bool = False
_COMFYUI_OUTPUT_DIR: Path | None = None


def _reset_comfyui_dir_cache() -> None:
    global _COMFYUI_DIR_CACHE_LOADED, _COMFYUI_OUTPUT_DIR
    _COMFYUI_DIR_CACHE_LOADED = False
    _COMFYUI_OUTPUT_DIR = None


def _load_comfyui_dirs() -> None:
    global _COMFYUI_DIR_CACHE_LOADED, _COMFYUI_OUTPUT_DIR
    if _COMFYUI_DIR_CACHE_LOADED:
        return
    try:
        from agenty_core.utils.comfyui_client import get_client, parse_argv_dir_flag

        stats = get_client().get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        out_dir = parse_argv_dir_flag(argv, "--output-directory")
        if out_dir:
            _COMFYUI_OUTPUT_DIR = Path(out_dir).resolve()
    except Exception as exc:
        logger.debug("comfyui_exec: could not query ComfyUI dirs — %s", exc)
    _COMFYUI_DIR_CACHE_LOADED = True


def _get_comfyui_output_dir() -> Path | None:
    """Return ComfyUI's --output-directory (cached for the process lifetime)."""
    _load_comfyui_dirs()
    return _COMFYUI_OUTPUT_DIR


def _submit_workflow(workflow_path: str, client_id: str = "") -> str:
    """Submit *workflow_path* to ComfyUI and return the ``prompt_id``.

    When *client_id* is provided it is forwarded so the matching WebSocket
    connection receives this prompt's progress events.

    Raises ``RuntimeError`` on failure.
    """
    from agenty_core.utils.comfyui_client import get_client

    p = Path(workflow_path)
    if not p.exists():
        raise RuntimeError(f"Workflow file not found: {workflow_path}")

    workflow = json.loads(p.read_text(encoding="utf-8"))
    # Paid partner nodes: refuse what the balance cannot cover and what needs the
    # user's go-ahead (raises partner_credits.CostGateError) before anything queues.
    from agenty_core import partner_credits as _credits
    _cost = _credits.gate_submission(workflow, workflow_path)
    client = get_client()
    payload: dict = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id
    if client.api_key:
        payload["extra_data"] = {"api_key_comfy_org": client.api_key}

    result = client.post("/prompt", json_data=payload)
    if isinstance(result, dict) and "prompt_id" in result:
        _credits.track(result["prompt_id"], _cost)
        # Ours, so Stop can remove it from the queue and leave the user's alone.
        try:
            from agenty_core.queue_ledger import remember as _remember_prompt
            _remember_prompt(result["prompt_id"])
        except Exception:  # noqa: BLE001
            pass
        return result["prompt_id"]
    raise RuntimeError(f"Unexpected response from ComfyUI /prompt: {result!r}")


def _extract_output_files(history: dict) -> list[dict]:
    """Return a flat list of ``{"filename", "subfolder", "type", "node_id"}`` dicts
    from a stripped history response.

    Handles the ``_strip_history`` output format where outputs are nested under
    ``{prompt_id: {"outputs": {node_id: {"images": [...], "gifs": [...], ...}}}}``.
    """
    files: list[dict] = []
    for _prompt_id, entry in history.items():
        if not isinstance(entry, dict):
            continue
        for node_id, node_out in entry.get("outputs", {}).items():
            if not isinstance(node_out, dict):
                continue
            # ComfyUI may use different keys depending on the output node type
            for key in ("images", "gifs", "videos", "audio"):
                for item in node_out.get(key, []):
                    if isinstance(item, dict) and "filename" in item:
                        files.append({**item, "node_id": str(node_id)})
    return files


def _resolve_output_path(
    filename: str,
    subfolder: str = "",
    image_type: str = "output",
    fallback_dir: "Path | None" = None,
) -> Path:
    """Return the authoritative on-disk path for a ComfyUI output file.

    Files are **never copied**.  Resolution order:

    1. ComfyUI's configured ``--output-directory`` (queried via ``/system_stats``).
       If the file exists there it is returned as-is.
    2. Falls back to downloading via ``/view`` into *fallback_dir* when supplied,
       or into the app's ``output_dir`` (from settings.json) as a last resort.
    """
    comfy_out = _get_comfyui_output_dir()
    if comfy_out is not None:
        src = comfy_out / subfolder / filename if subfolder else comfy_out / filename
        if src.exists():
            logger.info("comfyui_exec: output located at %s (%d bytes)", src, src.stat().st_size)
            return src
        logger.debug("comfyui_exec: %s not found in ComfyUI output dir, falling back to /view", src)

    from agenty_core.utils.comfyui_client import get_client

    if fallback_dir is None:
        fallback_dir = _output_dir()
    fallback_dir.mkdir(parents=True, exist_ok=True)
    dest = fallback_dir / filename

    params: dict = {"filename": filename, "type": image_type}
    if subfolder:
        params["subfolder"] = subfolder

    client = get_client()
    resp = client.get("/view", params=params, raw=True)
    image_bytes: bytes = resp.content  # type: ignore[attr-defined]
    dest.write_bytes(image_bytes)
    logger.info("comfyui_exec: downloaded output → %s (%d bytes)", dest, len(image_bytes))
    return dest
