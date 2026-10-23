"""
agentY – headless batch workflow runner.

This module is the *LLM-independent* worker behind the ``batch-process`` skill.
The MCP host (Claude) assembles and validates each stage workflow **once**, writes
a job spec, then launches this runner as a **detached subprocess** via the
``start_batch_job`` tool.  From that point on no model tokens are spent: the
runner drives ComfyUI directly over HTTP, iterating the whole input set on its
own, and reports progress purely by rewriting ``status.json`` — which the agent
polls with ``get_batch_status``.

Run with::

    python -m src.batch_runner <job_dir>

where ``<job_dir>`` contains ``spec.json``.  The runner writes ``status.json``
(progress + per-item results), tails its own log to ``runner.log`` (handled by
the launcher), and stops gracefully when a ``stop.flag`` file appears.

Job spec (``spec.json``)
------------------------
::

    {
      "mode": "pipeline",                       # only mode for now (stage chain per input)
      "inputs": ["D:/in/a.png", "D:/in/b.mp4"], # OR {"dir": "D:/in", "glob": "*.png"}
      "output_dir": "D:/out/batch_run",         # finals copied here
      "stages": [
        {
          "workflow_path": ".../stage1.json",   # a validated workflow JSON
          "input_node_id": "190",               # node to inject each item into (optional → auto)
          "input_field": "image",               # "image" | "video" | ... (optional → auto)
          "output_node_id": "9",                # node whose saved file feeds the next stage (optional → auto)
          "randomize_seed": false               # re-seed every node per item (optional, default false)
        },
        { "workflow_path": ".../stage2.json" }   # stage 1's output becomes stage 2's input
      ]
    }

Pipeline semantics: for every input item the stages run in order, each stage's
resolved output file becoming the next stage's input.  One final output (set)
per input item.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("agentY.batch_runner")

# Reuse the server's ComfyUI plumbing so the runner behaves identically to the
# blocking executor (same submit, same output-path resolution).
from agenty_core.utils.comfyui_client import get_client, parse_argv_dir_flag
from agenty_core.comfyui_exec import (
    _submit_workflow,
    _extract_output_files,
    _resolve_output_path,
)
from agenty_core.utils.workflow_parser import INPUT_NODE_TYPES, OUTPUT_NODE_TYPES

# Path-style loaders take an absolute on-disk path in their input field instead
# of a filename relative to ComfyUI's input dir — so we skip the upload step.
_PATH_LOADERS: frozenset[str] = frozenset({
    "VHS_LoadVideoPath", "LoadVideoPath", "VHS_LoadImagePath",
})

# Output nodes that expose a patchable ``filename_prefix`` we can make per-item
# unique so concurrent items never overwrite each other's files.
_PREFIXABLE_OUTPUTS: frozenset[str] = frozenset({
    "SaveImage", "VHS_VideoCombine", "SaveVideo", "SaveAudio",
})

_SEED_KEYS: frozenset[str] = frozenset({"seed", "noise_seed"})
_AGENT_SUBFOLDER = "agent"
_BATCH_OUTPUT_ROOT = "batch"  # under ComfyUI output dir: agent/batch/<job>/...

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
_DEFAULT_GLOBS = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.mp4", "*.mov", "*.webm"]

_STAGE_TIMEOUT_S = 60 * 60  # hard cap per single ComfyUI job


# ---------------------------------------------------------------------------
# Status file (the "status bar" the agent polls)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _bar(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "░" * width + "] 0/0"
    pct = int(done / total * 100)
    filled = int(width * done / total)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total} ({pct}%)"


class Status:
    """Accumulates job state and writes ``status.json`` atomically on every change."""

    def __init__(self, job_dir: Path, job_id: str, total_items: int, output_dir: str):
        self.path = job_dir / "status.json"
        self.tmp = job_dir / "status.json.tmp"
        self.data: dict = {
            "job_id": job_id,
            "state": "running",          # running | completed | failed | stopped
            "pid": os.getpid(),
            "mode": "pipeline",
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "total_items": total_items,
            "completed_items": 0,
            "failed_items": 0,
            "output_dir": output_dir,
            "progress_bar": _bar(0, total_items),
            "current": {},
            "items": [],
            "errors": [],
        }
        self.flush()

    def flush(self) -> None:
        self.data["updated_at"] = _now_iso()
        self.data["progress_bar"] = _bar(
            self.data["completed_items"], self.data["total_items"]
        )
        try:
            self.tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            os.replace(self.tmp, self.path)  # atomic on Windows + POSIX
        except Exception as exc:  # never let a status write kill the run
            logger.warning("batch_runner: status write failed — %s", exc)

    def set_current(self, **kwargs) -> None:
        self.data["current"] = {**self.data.get("current", {}), **kwargs}
        self.flush()

    def add_error(self, msg: str) -> None:
        self.data["errors"].append(msg)
        logger.error("batch_runner: %s", msg)
        self.flush()


# ---------------------------------------------------------------------------
# ComfyUI directory + input staging
# ---------------------------------------------------------------------------

def _comfyui_input_dir(client) -> Path | None:
    """Resolve ComfyUI's --input-directory from /system_stats (None if unknown)."""
    try:
        stats = client.get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        val = parse_argv_dir_flag(argv, "--input-directory")
        if val:
            return Path(val).resolve()
        # Fall back to ComfyUI's conventional ./input next to argv[0].
        if argv and argv[0]:
            cand = Path(argv[0]).parent / "input"
            if cand.exists():
                return cand.resolve()
    except Exception as exc:
        logger.debug("batch_runner: could not resolve input dir — %s", exc)
    return None


def _stage_input_into_comfyui(client, src: Path, dest_name: str) -> str:
    """Make *src* available to ComfyUI and return the reference for a load node.

    Prefers a direct filesystem copy into ``<input_dir>/agent/`` (works for both
    images and videos of any size); falls back to the HTTP ``/upload/image``
    endpoint when the input dir cannot be resolved locally.  Returns the
    ``agent/<name>`` reference a LoadImage / VHS_LoadVideo node expects.
    """
    input_dir = _comfyui_input_dir(client)
    if input_dir is not None:
        agent_dir = input_dir / _AGENT_SUBFOLDER
        try:
            agent_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, agent_dir / dest_name)
            return f"{_AGENT_SUBFOLDER}/{dest_name}"
        except Exception as exc:
            logger.warning("batch_runner: direct copy failed (%s); trying HTTP upload", exc)

    # HTTP fallback (image endpoint also accepts video files on modern ComfyUI).
    with open(src, "rb") as fh:
        files = {"image": (dest_name, fh, "application/octet-stream")}
        data = {"type": "input", "overwrite": "true", "subfolder": _AGENT_SUBFOLDER}
        resp = client.post("/upload/image", data=data, files=files)
    if isinstance(resp, dict) and resp.get("name"):
        sub = resp.get("subfolder", _AGENT_SUBFOLDER)
        return f"{sub}/{resp['name']}" if sub else resp["name"]
    return f"{_AGENT_SUBFOLDER}/{dest_name}"


# ---------------------------------------------------------------------------
# Workflow inspection / patching
# ---------------------------------------------------------------------------

def _auto_input_node(workflow: dict) -> tuple[str, str] | None:
    """Return (node_id, field) for the sole input node, or None if ambiguous."""
    candidates: list[tuple[str, str]] = []
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        media = INPUT_NODE_TYPES.get(ct)
        if media is None:
            continue
        inputs = node.get("inputs", {})
        # Pick the field this loader actually exposes.
        for field in ("image", "video", "filename", "audio"):
            if field in inputs:
                candidates.append((str(nid), field))
                break
        else:
            candidates.append((str(nid), "video" if media == "video" else "image"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _auto_output_node(workflow: dict) -> str | None:
    """Return the node_id of the sole output node, or None if ambiguous."""
    outs = [
        str(nid)
        for nid, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type", "") in OUTPUT_NODE_TYPES
    ]
    return outs[0] if len(outs) == 1 else None


def _randomize_seeds(workflow: dict) -> int:
    changed = 0
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for key in _SEED_KEYS:
            if key in inputs and isinstance(inputs[key], (int, float)):
                inputs[key] = random.randint(0, 2**32 - 1)
                changed += 1
    return changed


def _has_outputs(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    for node_out in (entry.get("outputs") or {}).values():
        if isinstance(node_out, dict) and any(
            node_out.get(k) for k in ("images", "gifs", "videos", "audio")
        ):
            return True
    return False


def _wait_for_completion(client, prompt_id: str, *, timeout: float, poll: float = 1.5) -> dict:
    """Block until *prompt_id* finishes; return {"history": raw} or {"error": ...}."""
    waited = 0.0
    while waited < timeout:
        try:
            raw = client.get(f"/history/{prompt_id}")
        except Exception as exc:
            logger.debug("batch_runner: history poll failed — %s", exc)
            raw = None
        if isinstance(raw, dict) and prompt_id in raw:
            entry = raw[prompt_id]
            status = entry.get("status", {}) if isinstance(entry, dict) else {}
            if status.get("status_str") == "error":
                return {"error": "ComfyUI reported an execution error", "history": raw}
            if status.get("completed") and _has_outputs(entry):
                return {"history": raw}
        time.sleep(poll)
        waited += poll
    return {"error": f"timed out after {int(timeout)}s waiting for prompt {prompt_id}"}


# ---------------------------------------------------------------------------
# Stage execution
# ---------------------------------------------------------------------------

def _run_stage(
    *,
    client,
    job_dir: Path,
    job_short: str,
    stage_idx: int,
    stage: dict,
    src_file: Path,
    item_stem: str,
) -> Path:
    """Run one stage on one input file; return the resolved primary output path.

    Raises ``RuntimeError`` on any failure so the caller can record it per item.
    """
    wf_path = stage["workflow_path"]
    workflow = json.loads(Path(wf_path).read_text(encoding="utf-8"))

    # ── Resolve the input node + field (explicit spec wins, else auto) ──────────
    input_node_id = str(stage.get("input_node_id", "")) or ""
    input_field = stage.get("input_field", "")
    if not input_node_id:
        auto = _auto_input_node(workflow)
        if auto is None:
            raise RuntimeError(
                f"stage {stage_idx + 1}: could not auto-detect a single input node — "
                "set 'input_node_id'/'input_field' in the spec."
            )
        input_node_id, auto_field = auto
        input_field = input_field or auto_field
    if not input_field:
        input_field = "image"
    if input_node_id not in workflow:
        raise RuntimeError(f"stage {stage_idx + 1}: input_node_id '{input_node_id}' not in workflow")

    node = workflow[input_node_id]
    node.setdefault("inputs", {})
    ct = node.get("class_type", "")
    if ct in _PATH_LOADERS:
        # Path loaders read an absolute on-disk path directly.
        node["inputs"][input_field] = str(src_file.resolve())
    else:
        dest_name = f"{job_short}_i{item_stem}_s{stage_idx + 1}{src_file.suffix.lower()}"
        ref = _stage_input_into_comfyui(client, src_file, dest_name)
        node["inputs"][input_field] = ref

    # ── Make the output filename per-item unique so items never collide ────────
    output_node_id = str(stage.get("output_node_id", "")) or (_auto_output_node(workflow) or "")
    prefix = f"{_AGENT_SUBFOLDER}/{_BATCH_OUTPUT_ROOT}/{job_short}/{item_stem}_s{stage_idx + 1}"
    if output_node_id and output_node_id in workflow:
        onode = workflow[output_node_id]
        if onode.get("class_type", "") in _PREFIXABLE_OUTPUTS:
            onode.setdefault("inputs", {})["filename_prefix"] = prefix
    else:
        # No declared output node — patch any prefixable output node we find.
        for onode in workflow.values():
            if isinstance(onode, dict) and onode.get("class_type", "") in _PREFIXABLE_OUTPUTS:
                onode.setdefault("inputs", {})["filename_prefix"] = prefix

    if stage.get("randomize_seed"):
        _randomize_seeds(workflow)

    # ── Persist the per-item workflow, submit, wait, resolve outputs ───────────
    runs_dir = job_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    item_wf = runs_dir / f"{item_stem}_s{stage_idx + 1}.json"
    item_wf.write_text(json.dumps(workflow, indent=2), encoding="utf-8")

    prompt_id = _submit_workflow(str(item_wf))
    result = _wait_for_completion(client, prompt_id, timeout=_STAGE_TIMEOUT_S)
    from agenty_core import partner_credits as _credits
    _credits.settle(prompt_id)
    if "error" in result:
        raise RuntimeError(f"stage {stage_idx + 1}: {result['error']}")

    files = _extract_output_files(result["history"])
    if not files:
        raise RuntimeError(f"stage {stage_idx + 1}: ComfyUI produced no output files")

    # Prefer the declared output node's file; else the first produced.
    chosen = None
    if output_node_id:
        chosen = next((f for f in files if f.get("node_id") == output_node_id), None)
    item = chosen or files[0]
    resolved = _resolve_output_path(
        item.get("filename", ""),
        item.get("subfolder", ""),
        item.get("type", "output"),
    )
    return Path(resolved)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------

def _resolve_inputs(spec: dict) -> list[Path]:
    """Turn the spec's ``inputs`` (list of paths OR {dir, glob}) into a file list."""
    raw = spec.get("inputs")
    files: list[Path] = []
    if isinstance(raw, dict):
        base = Path(raw.get("dir", "")).resolve()
        globs = raw.get("glob")
        if isinstance(globs, str):
            globs = [globs]
        if not globs:
            globs = _DEFAULT_GLOBS
        seen: set[str] = set()
        for pattern in globs:
            for p in sorted(base.glob(pattern)):
                if p.is_file() and str(p) not in seen:
                    seen.add(str(p))
                    files.append(p)
    elif isinstance(raw, list):
        for entry in raw:
            p = Path(str(entry))
            if p.is_dir():
                for pattern in _DEFAULT_GLOBS:
                    files.extend(sorted(q for q in p.glob(pattern) if q.is_file()))
            elif p.is_file():
                files.append(p)

    # De-duplicate while preserving order (e.g. a file and its parent dir both listed).
    deduped: list[Path] = []
    seen_paths: set[str] = set()
    for p in files:
        key = str(p.resolve())
        if key not in seen_paths:
            seen_paths.add(key)
            deduped.append(p)
    return deduped


def _safe_stem(p: Path, idx: int) -> str:
    """A filesystem-safe, collision-resistant stem for per-item output naming."""
    base = "".join(c if (c.isalnum() or c in "-_") else "_" for c in p.stem)[:40]
    return f"{idx:03d}_{base or 'item'}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(job_dir: Path) -> int:
    spec_path = job_dir / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    job_id = job_dir.name
    job_short = uuid.uuid4().hex[:6]
    stages: list[dict] = spec.get("stages", [])
    output_dir = Path(spec.get("output_dir", str(job_dir / "outputs"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _resolve_inputs(spec)
    status = Status(job_dir, job_id, len(inputs), str(output_dir))

    if not stages:
        status.add_error("spec has no stages")
        status.data["state"] = "failed"
        status.flush()
        return 1
    if not inputs:
        status.add_error("no input files resolved from spec.inputs")
        status.data["state"] = "failed"
        status.flush()
        return 1

    client = get_client()
    stop_flag = job_dir / "stop.flag"

    for idx, src in enumerate(inputs, 1):
        if stop_flag.exists():
            status.data["state"] = "stopped"
            status.flush()
            logger.info("batch_runner: stop.flag seen — halting before item %d", idx)
            return 0

        item_stem = _safe_stem(src, idx)
        item_rec: dict = {
            "index": idx, "input": str(src), "state": "running",
            "outputs": [], "error": None,
        }
        status.data["items"].append(item_rec)
        t0 = time.time()
        current_file = src
        try:
            for s_idx, stage in enumerate(stages):
                status.set_current(
                    item_index=idx, item=src.name,
                    stage_index=s_idx + 1, stage_total=len(stages),
                    elapsed_s=round(time.time() - t0, 1),
                    note=f"running stage {s_idx + 1}/{len(stages)}",
                )
                current_file = _run_stage(
                    client=client, job_dir=job_dir, job_short=job_short,
                    stage_idx=s_idx, stage=stage, src_file=current_file,
                    item_stem=item_stem,
                )

            # Copy the final output next to the user's requested output_dir.
            final_dest = output_dir / f"{item_stem}{current_file.suffix.lower()}"
            try:
                shutil.copy2(current_file, final_dest)
            except Exception as exc:
                logger.warning("batch_runner: final copy failed (%s); keeping source path", exc)
                final_dest = current_file

            item_rec["state"] = "done"
            item_rec["outputs"] = [str(final_dest)]
            status.data["completed_items"] += 1
        except Exception as exc:
            item_rec["state"] = "error"
            item_rec["error"] = str(exc)
            status.data["failed_items"] += 1
            status.add_error(f"item {idx} ({src.name}): {exc}")
        finally:
            status.flush()

    status.data["current"] = {}
    status.data["state"] = "completed" if status.data["failed_items"] == 0 else "failed"
    status.flush()
    logger.info(
        "batch_runner: finished — %d done, %d failed",
        status.data["completed_items"], status.data["failed_items"],
    )
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if len(sys.argv) < 2:
        print("usage: python -m src.batch_runner <job_dir>", file=sys.stderr)
        sys.exit(2)
    job_dir = Path(sys.argv[1]).resolve()
    if not (job_dir / "spec.json").exists():
        print(f"spec.json not found in {job_dir}", file=sys.stderr)
        sys.exit(2)
    try:
        sys.exit(run(job_dir))
    except Exception as exc:  # last-resort: mark the job failed in status.json
        logger.exception("batch_runner: fatal error")
        try:
            status_path = job_dir / "status.json"
            data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
            data["state"] = "failed"
            data.setdefault("errors", []).append(f"fatal: {exc}")
            data["updated_at"] = _now_iso()
            status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
