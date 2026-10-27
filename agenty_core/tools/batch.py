"""
Batch job-control tools — start / poll / stop a headless batch run.

The blocking ``execute_workflow`` / ``execute_workflows_batch`` tools tie up the
agent until ComfyUI finishes.  For a *large* batch (a folder of inputs run
through one or more chained workflows) that's the wrong shape: the user wants to
kick it off and check in on it.

These tools manage a **detached worker** (``src/batch_runner.py``) instead:

  • ``start_batch_job``  writes the job dir + spec and launches the runner as a
    background process, returning a ``job_id`` immediately (no waiting).
  • ``get_batch_status`` reads the runner's ``status.json`` — the live progress
    bar, per-item results, and errors.  Cheap; call it whenever you want.
  • ``stop_batch_job``   asks the runner to stop gracefully (and terminates it
    as a fallback).
  • ``list_batch_jobs``  lists recent jobs and their states.

The runner talks to ComfyUI directly, so no model tokens are spent while it
works — only the brief status reads cost anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agenty_core._compat import tool


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from agenty_core.paths import project_root as _project_root


def _load_config() -> dict:
    cfg = _project_root() / "config" / "settings.json"
    if cfg.exists():
        with open(cfg, encoding="utf-8") as f:
            return json.loads("".join(ln for ln in f if not ln.lstrip().startswith("//")))
    return {}


def _jobs_root() -> Path:
    """Where batch-job bookkeeping lives — always inside the app, never ComfyUI.

    Deliberately NOT derived from ``output_workflows_dir`` any more. That setting
    now points into ComfyUI's user directory so generated workflows appear in its
    workflow browser, and job state is not something anyone wants listed there.
    It is also blank by default, which would have resolved this to the app root.
    ``output_workflows/`` is gitignored, so job state still stays out of git.
    """
    root = (_project_root() / "output_workflows").resolve() / "batch_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _job_dir(job_id: str) -> Path:
    # Guard against path traversal — job_id is a flat directory name only.
    safe = Path(job_id).name
    return _jobs_root() / safe


def _read_status(job_id: str) -> dict | None:
    path = _job_dir(job_id) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return str(pid) in (out.stdout or "")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def start_batch_job(spec_json: str) -> str:
    """Launch a headless batch run over many inputs and return its job_id immediately.

    Use this (instead of ``execute_workflow``) when the user wants to run a
    folder of images/videos — or a set of attached files — through one workflow
    or a chain of workflows, without blocking on the result.  Assemble and
    VALIDATE every stage workflow first (one per stage); this tool only schedules
    the run.  A detached Python worker then drives ComfyUI on its own and reports
    progress via ``get_batch_status(job_id)``.

    Pipeline semantics: for each input item the stages run in order, each stage's
    output file feeding the next stage's input (so two stages = input → stage1 →
    stage2 → final). One final output per input.

    Args:
        spec_json: JSON string describing the job. Schema::

            {
              "inputs": ["D:/in/a.png", "D:/in/b.mp4"],   // OR {"dir": "...", "glob": "*.png"}
              "output_dir": "D:/out/run",                  // finals copied here
              "stages": [
                {
                  "workflow_path": "<validated stage1 workflow JSON path>",
                  "input_node_id": "190",     // optional — auto-detected if a single load node
                  "input_field": "image",     // optional — "image" | "video"
                  "output_node_id": "9",       // optional — auto-detected if a single output node
                  "randomize_seed": false      // optional — re-seed each item (default false)
                },
                { "workflow_path": "<validated stage2 workflow JSON path>" }
              ]
            }

    Returns:
        JSON ``{"job_id", "job_dir", "total_inputs", "stages", "message"}`` on
        success, or ``{"error": "..."}``. Poll ``get_batch_status(job_id)`` next.
    """
    try:
        spec = json.loads(spec_json) if isinstance(spec_json, str) else dict(spec_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid spec JSON: {e}"})

    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        return json.dumps({"error": "spec.stages must be a non-empty list"})
    for i, st in enumerate(stages):
        wf = st.get("workflow_path") if isinstance(st, dict) else None
        if not wf:
            return json.dumps({"error": f"stage {i + 1} is missing 'workflow_path'"})
        if not Path(wf).exists():
            return json.dumps({"error": f"stage {i + 1} workflow not found: {wf}"})
    if not spec.get("inputs"):
        return json.dumps({"error": "spec.inputs is required (a list of paths or {dir, glob})"})

    # ── Create the job dir + spec ──────────────────────────────────────────────
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    jdir = _job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    spec.setdefault("mode", "pipeline")
    (jdir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    # ── Launch the detached worker ─────────────────────────────────────────────
    log_path = jdir / "runner.log"
    creationflags = 0
    popen_kwargs: dict = {}
    if os.name == "nt":
        # DETACHED_PROCESS (0x8) + CREATE_NEW_PROCESS_GROUP (0x200): the worker
        # outlives this tool call and the MCP request that spawned it.
        creationflags = 0x00000008 | 0x00000200
    else:
        popen_kwargs["start_new_session"] = True

    # The detached worker is a fresh process: it can't inherit the in-process
    # set_project_root() binding, so pin the app root via the environment (it
    # also sets cwd to the app root, which is the paths.project_root() fallback).
    child_env = dict(os.environ)
    child_env["AGENTY_PROJECT_ROOT"] = str(_project_root())

    try:
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-m", "agenty_core.batch_runner", str(jdir)],
            cwd=str(_project_root()),
            env=child_env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
            **popen_kwargs,
        )
    except Exception as e:
        return json.dumps({"error": f"Failed to launch batch runner: {e}"})
    finally:
        # The detached child has its own inherited handle; release the parent's
        # copy so the server doesn't accumulate a file descriptor per job.
        try:
            log_fh.close()
        except Exception:
            pass

    # Seed an initial status so get_batch_status works before the worker writes one.
    n_inputs = spec["inputs"] if isinstance(spec["inputs"], list) else None
    initial = {
        "job_id": job_id,
        "state": "starting",
        "pid": proc.pid,
        "mode": "pipeline",
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_items": len(n_inputs) if n_inputs is not None else None,
        "stages": len(stages),
    }
    status_path = jdir / "status.json"
    if not status_path.exists():
        status_path.write_text(json.dumps(initial, indent=2), encoding="utf-8")

    return json.dumps({
        "job_id": job_id,
        "job_dir": str(jdir),
        "pid": proc.pid,
        "total_inputs": len(n_inputs) if n_inputs is not None else "resolved from dir/glob",
        "stages": len(stages),
        "message": (
            f"Batch job '{job_id}' started in the background. "
            f"Poll get_batch_status('{job_id}') to watch progress."
        ),
    })


@tool
def get_batch_status(job_id: str) -> str:
    """Return the live progress of a batch job started with ``start_batch_job``.

    Reads the worker's ``status.json`` — this does not touch ComfyUI and costs
    nothing to call repeatedly.  Use it to show the user the progress bar and to
    decide whether the run is done.

    Args:
        job_id: The id returned by ``start_batch_job``.

    Returns:
        JSON with ``state`` (starting/running/completed/failed/stopped),
        ``progress_bar``, ``completed_items``/``failed_items``/``total_items``,
        the ``current`` item/stage, recent ``errors``, and ``outputs`` (final
        files, once items finish). ``alive`` reports whether the worker process
        is still running — if it's false while ``state`` is still "running", the
        worker died unexpectedly (check ``runner.log`` in the job dir).
    """
    data = _read_status(job_id)
    if data is None:
        return json.dumps({"error": f"No status found for job '{job_id}'. Check the job_id."})

    items = data.get("items", []) or []
    outputs: list[str] = []
    for it in items:
        if it.get("state") == "done":
            outputs.extend(it.get("outputs", []))

    alive = _pid_alive(int(data.get("pid", 0) or 0))
    state = data.get("state", "unknown")

    summary = {
        "job_id": data.get("job_id", job_id),
        "state": state,
        "alive": alive,
        "progress_bar": data.get("progress_bar", ""),
        "total_items": data.get("total_items"),
        "completed_items": data.get("completed_items", 0),
        "failed_items": data.get("failed_items", 0),
        "current": data.get("current", {}),
        "errors": (data.get("errors", []) or [])[-5:],
        "output_dir": data.get("output_dir", ""),
        "outputs": outputs[:50],
        "output_count": len(outputs),
    }
    if state == "running" and not alive:
        summary["warning"] = (
            "Worker process is no longer running but state is still 'running' — "
            "it likely crashed. See runner.log in the job directory."
        )
    return json.dumps(summary)


@tool
def stop_batch_job(job_id: str) -> str:
    """Stop a running batch job.

    Drops a ``stop.flag`` so the worker halts gracefully before the next input
    item, and terminates the worker process as a fallback. Items already finished
    keep their outputs.

    Args:
        job_id: The id returned by ``start_batch_job``.
    """
    jdir = _job_dir(job_id)
    if not jdir.exists():
        return json.dumps({"error": f"No job directory for '{job_id}'."})

    (jdir / "stop.flag").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")

    data = _read_status(job_id) or {}
    pid = int(data.get("pid", 0) or 0)
    terminated = False
    if pid and _pid_alive(pid):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            else:
                os.kill(pid, 15)
            terminated = True
        except Exception:
            terminated = False

    return json.dumps({
        "job_id": job_id,
        "stop_requested": True,
        "process_terminated": terminated,
        "message": "Stop requested. Call get_batch_status to confirm the final state.",
    })


@tool
def list_batch_jobs(limit: int = 20) -> str:
    """List recent batch jobs (most recent first) with their state and progress.

    Args:
        limit: Max jobs to return (default 20).
    """
    root = _jobs_root()
    jobs: list[dict] = []
    for jdir in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not jdir.is_dir():
            continue
        data = _read_status(jdir.name) or {}
        jobs.append({
            "job_id": jdir.name,
            "state": data.get("state", "unknown"),
            "progress_bar": data.get("progress_bar", ""),
            "completed_items": data.get("completed_items", 0),
            "failed_items": data.get("failed_items", 0),
            "total_items": data.get("total_items"),
            "output_dir": data.get("output_dir", ""),
        })
        if len(jobs) >= limit:
            break
    return json.dumps({"count": len(jobs), "jobs": jobs})
