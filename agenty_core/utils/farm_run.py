"""Run a normal agentY workflow execution on the Deadline farm instead of the local ComfyUI.

The executors (agentY and agentY-mcp) submit a prompt to ComfyUI, stream its progress, and then collect the output
files from ComfyUI's ``/history``. This module lets them do the same against Deadline with three small hooks, so
everything around the run - QA, the output sink, healing, batches - behaves exactly as it does locally:

* :func:`enabled` - is routing on? Off by default (``DEADLINE_ROUTE_RUNS=1`` in the environment/.env, or the host's own
  setting passed in).
* :func:`submit` - turns the already-prepared prompt into a Deadline job (via :mod:`agenty_core.tools.deadline_farm`,
  so the dedicated-pool rule is the same) and returns a reference ``farm:<job id>`` where the executor expects a
  ComfyUI ``prompt_id``.
* :func:`stream` - the farm's counterpart of ``stream_comfyui_job``: progress lines, then ONE terminal dict, either
  ``{"history": ...}`` shaped like ComfyUI's history (the files the plugin wrote to the share are listed as images, by
  their full path, which the executors' path resolution takes as-is) or an error dict (``details`` parsed from the
  plugin's error text, so the heal loop sees a node id and exception), or ``{"interrupted": True, ...}`` when the job
  was suspended in Monitor. Cancelling the consumer (the user pressed Stop) suspends the job.

A run is one task (``batch=1``): the executor's own batch already submits one job per member and waits for them
together, so the farm sees them as separate jobs that can render on different machines at the same time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import AsyncGenerator, Optional

logger = logging.getLogger("agenty_core.farm_run")

PREFIX = "farm:"
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")
# Consecutive failed status polls (network, share, farm busy) before the stream gives up on a job.
_MAX_POLL_ERRORS = 8


def enabled(setting: Optional[bool] = None) -> bool:
    """Is routing runs to the farm on?

    The environment variable ``DEADLINE_ROUTE_RUNS`` (or the .env entry of the same name) decides when it is set to
    a yes/no word; otherwise *setting*, the host's own switch, decides; otherwise off.
    """
    from agenty_core.utils.secrets import get_secret
    raw = (get_secret("DEADLINE_ROUTE_RUNS", "") or "").strip().strip('"').lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return bool(setting)


def is_farm_ref(ref: str) -> bool:
    return str(ref or "").startswith(PREFIX)


def job_id_of(ref: str) -> str:
    return str(ref)[len(PREFIX):] if is_farm_ref(ref) else str(ref)


def submit(prompt: dict, *, name: str = "") -> str:
    """Queue *prompt* (an API-format dict) on the farm; return ``farm:<job id>``.

    Raises ``RuntimeError`` with the reason when the farm refuses (not configured, pool missing, no output folder):
    the executors already turn a ``RuntimeError`` from submission into a visible "submission failed" line.
    """
    import uuid
    from agenty_core.tools import deadline_farm as df
    # A short suffix keeps two runs of the same workflow in the same second from sharing an output folder.
    name = f"{name or 'agentY run'} {uuid.uuid4().hex[:6]}"
    try:
        res = df.submit(prompt, name=name, batch=1, randomize_seed=False,
                        comment="Submitted by agentY (routed run)")
    except df.FarmError as exc:
        raise RuntimeError(f"Deadline: {exc}") from exc
    return PREFIX + res["job_id"]


def cancel(ref: str) -> None:
    from agenty_core.tools import deadline_farm as df
    try:
        df.cancel_job(job_id_of(ref))
    except Exception:  # noqa: BLE001 - a stop must never fail because the farm did not answer
        logger.exception("could not suspend farm job %s", ref)


# The plugin's own wording: "ComfyUI execution_error in KSampler (node 3): RuntimeError: out of memory".
_EXEC = re.compile(r"ComfyUI execution_error in (?P<type>[^(]*?) \(node (?P<node>[^)]*)\): "
                   r"(?P<exc>[A-Za-z_.]*): ?(?P<msg>.*)", re.S)


def error_details(text: str, job_id: str = "") -> dict:
    """The error dict the executors' repair loop reads, from a failed job's error text."""
    d: dict = {"farm": True, "job_id": job_id, "traceback": []}
    m = _EXEC.search(text or "")
    if m:
        d.update(node_id=m.group("node").strip(), node_type=m.group("type").strip(),
                 exception_type=m.group("exc").strip(), exception_message=m.group("msg").strip()[:1500])
    else:
        d.update(node_id="", node_type="", exception_type="", exception_message=(text or "").strip()[:1500])
    return d


def to_history(ref: str, files: list) -> dict:
    """The farm's result in the shape of ComfyUI's ``/history`` (what ``_extract_output_files`` reads)."""
    images = [{"filename": str(f), "subfolder": "", "type": "output"} for f in files]
    return {ref: {"outputs": {"farm": {"images": images}} if images else {}, "status": {"completed": True}}}


def fetch_outputs(files: list, job_id: str = "") -> list:
    """Copy the results from the farm share into ``DEADLINE_FETCH_DIR`` when that is set, and return the paths the
    run should hand on. Unset (the default): the files stay where the workers wrote them and those paths are used.

    A file that cannot be copied keeps its share path, so a half-reachable share loses nothing.
    """
    import shutil
    from pathlib import Path
    from agenty_core.utils.secrets import get_secret
    dest = (get_secret("DEADLINE_FETCH_DIR", "") or "").strip().strip('"')
    if not dest or not files:
        return list(files)
    out = []
    folder = Path(dest) / (job_id or "farm")
    for f in files:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / Path(str(f)).name
            shutil.copy2(f, target)
            out.append(str(target))
        except OSError:
            logger.warning("could not fetch %s from the farm share; using it in place", f)
            out.append(str(f))
    return out


def _fmt_wait(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m} min" if m else f"{int(seconds)} s"


async def stream(ref: str, *, poll_s: float = 4.0, max_wait_s: float = 0.0) -> AsyncGenerator:
    """Progress lines for farm job *ref*, then one terminal dict (see the module doc).

    *max_wait_s* > 0 caps the total wait; the default waits as long as the job lives (a queued render can wait for
    a busy farm). The consumer going away (task cancelled) suspends the job.
    """
    from agenty_core.tools import deadline_farm as df
    job_id = job_id_of(ref)
    t0 = time.time()
    last_line = ""
    last_said = 0.0
    errors = 0
    finished = False        # the job reached an end of its own: closing the stream must not touch it
    try:
        yield f"🏭 Sent to the Deadline farm · job `{job_id}` — waiting for a worker…"
        while True:
            try:
                st = await asyncio.to_thread(df.job_status, job_id, 0)
                errors = 0
            except Exception as exc:  # noqa: BLE001
                errors += 1
                if errors >= _MAX_POLL_ERRORS:
                    finished = True      # not known to be over, but not ours to suspend blindly
                    yield {"error": f"lost contact with the Deadline job {job_id}: {exc}",
                           "details": error_details(str(exc), job_id)}
                    return
                await asyncio.sleep(poll_s)
                continue
            state = st.get("state", "Unknown")
            waited = time.time() - t0
            if state == "Completed":
                files = await asyncio.to_thread(fetch_outputs, list(st.get("outputs") or []), job_id)
                finished = True
                yield f"🏭 Farm job finished in {_fmt_wait(waited)} · {len(files)} file(s)"
                yield {"history": to_history(ref, files)}
                return
            if state == "Failed":
                text = st.get("errors") or "the job failed (no error report)"
                finished = True
                yield f"❌ Farm job {job_id} failed: {text[:300]}"
                yield {"error": "Deadline job failed", "details": error_details(text, job_id)}
                return
            if state in ("Suspended", "Archived", "Deleted"):
                finished = True
                yield f"🛑 Farm job {job_id} was {state.lower()} in Deadline"
                yield {"interrupted": True, "error": f"Deadline job {state.lower()}"}
                return
            tasks = st.get("tasks") or {}
            if state == "Rendering":
                pct = int(float(st.get("progress") or 0) * 100)
                line = f"🏭 Farm: rendering {tasks.get('completed', 0)}/{tasks.get('total', 1)}"
                line += f" · {pct}%" if pct else ""
            else:
                line = "🏭 Farm: queued — waiting for a free worker"
            now = time.time()
            if line != last_line or now - last_said > 60:
                suffix = f" ({_fmt_wait(waited)})" if waited >= 30 else ""
                yield line + suffix
                last_line, last_said = line, now
            if max_wait_s and waited > max_wait_s:
                cancel(ref)
                finished = True
                yield {"error": f"the Deadline job did not finish within {_fmt_wait(max_wait_s)} "
                                f"(suspended it); see Deadline Monitor job {job_id}"}
                return
            await asyncio.sleep(poll_s)
    except asyncio.CancelledError:
        # Stop pressed: do not leave the job rendering on a worker for nobody.
        if not finished:
            await asyncio.shield(asyncio.to_thread(cancel, ref))
        raise
    except GeneratorExit:
        if not finished:
            cancel(ref)
        raise


def max_wait_seconds() -> float:
    """``DEADLINE_MAX_WAIT_MINUTES`` (default 240; 0 = no limit) as seconds."""
    from agenty_core.utils.secrets import get_secret
    try:
        return max(0.0, float((get_secret("DEADLINE_MAX_WAIT_MINUTES", "240") or "240").strip())) * 60
    except ValueError:
        return 240 * 60.0
