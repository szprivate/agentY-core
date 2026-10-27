"""Run a normal agentY workflow execution on Comfy Cloud - the ``comfy_cloud`` backend beside local and Deadline.

Same three hooks as :mod:`agenty_core.utils.farm_run`, so the executors treat a cloud run like any other:

* :func:`submit` - gate the cost, upload the input files, queue the prompt; returns ``cloud:<prompt id>`` where the
  executor expects a ComfyUI ``prompt_id``.
* :func:`stream` - progress lines, then ONE terminal dict: ``{"history": ...}`` shaped like ComfyUI's history (the
  outputs are downloaded to a folder here and listed by full path, which the executors take as they are), or an
  error dict whose ``details`` carry the failing node and exception, so the heal loop treats a cloud failure like a
  local one.
* :func:`settle_line` - what the run cost (the balance before against after) and where it ran.

Cost: before every cloud run the balance is read, the run is refused when it is below ``execution.min_balance_usd``
(default $0.50), and a run at or above the confirmation level needs the user's go-ahead (the same ``confirm_cost``
as for partner nodes, :mod:`agenty_core.partner_credits`). Comfy Cloud bills GPU time by the second (standard
workflows on an RTX 6000 Pro: about 0.27 credits/s = $0.077/min) but quotes no run in advance, so the estimate is that
rate (``execution.cloud_gpu_usd_per_second``) times a runtime range: ``cloud_gpu_min_seconds`` to
``cloud_gpu_max_seconds_image`` / ``_video`` by what the workflow saves. It goes on top of the partner nodes' own
price badges; the REAL spend is measured from the balance afterwards and reported.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from agenty_core import partner_credits as pc
from agenty_core.utils import comfy_cloud, execution_mode, farm_run

logger = logging.getLogger("agenty_core.cloud_run")

PREFIX = "cloud:"
DEFAULT_GPU_USD_PER_SECOND = 0.077 / 60      # ~0.27 credits/s, standard GPU (RTX 6000 Pro)
DEFAULT_GPU_MIN_SECONDS = 20
DEFAULT_GPU_MAX_SECONDS_IMAGE = 120
DEFAULT_GPU_MAX_SECONDS_VIDEO = 600
_VIDEO_HINT = re.compile(r"video|vhs_|wan|ltx|hunyuanvideo|animatediff|framepack", re.I)
DEFAULT_MIN_BALANCE_USD = 0.50
_LOADER_KEYS = ("image", "video", "audio", "file", "mask")
_jobs: dict = {}


def is_cloud_ref(ref) -> bool:
    return str(ref or "").startswith(PREFIX)


def job_id_of(ref) -> str:
    return str(ref)[len(PREFIX):] if is_cloud_ref(ref) else str(ref)


def _setting(name: str, default: float) -> float:
    try:
        return max(0.0, float(execution_mode.settings().get(name, default)))
    except (TypeError, ValueError):
        return default


# ── the cost gate ────────────────────────────────────────────────────────────
def is_video_workflow(workflow: dict) -> bool:
    """A workflow that makes video (it runs much longer on the GPU than an image one)."""
    return any(isinstance(n, dict) and _VIDEO_HINT.search(str(n.get("class_type", "")))
               for n in (workflow or {}).values())


def gpu_time_usd(workflow: dict) -> tuple:
    """``(min, max)`` USD of Comfy Cloud GPU time for an open-weight run: the per-second rate times the expected
    runtime range (longer for video)."""
    rate = _setting("cloud_gpu_usd_per_second", DEFAULT_GPU_USD_PER_SECOND)
    lo_s = _setting("cloud_gpu_min_seconds", DEFAULT_GPU_MIN_SECONDS)
    hi_s = (_setting("cloud_gpu_max_seconds_video", DEFAULT_GPU_MAX_SECONDS_VIDEO) if is_video_workflow(workflow)
            else _setting("cloud_gpu_max_seconds_image", DEFAULT_GPU_MAX_SECONDS_IMAGE))
    return round(rate * lo_s, 4), round(rate * max(lo_s, hi_s), 4)


def gate(workflow: dict, workflow_path: str = "", *, open_weight: bool = True, confirm: bool = False) -> dict:
    """Refuse (``CostGateError``) a cloud run the balance cannot cover or the user has not confirmed. Returns the
    estimate (``paid`` True: the run costs credits)."""
    client = comfy_cloud.get_client()
    est = pc.estimate_workflow(workflow, info_for=client.node_info)
    lo, hi = gpu_time_usd(workflow) if open_weight else (0.0, 0.0)
    if hi > 0:
        est["nodes"].append({"node_id": "-", "class_type": "Comfy Cloud GPU time", "known": True,
                             "min_usd": lo, "max_usd": hi, "approximate": True,
                             "reason": "billed per second of GPU time; range from the expected runtime"})
        est["min_usd"] = round(est["min_usd"] + lo, 4)
        est["known_max_usd"] = round(est.get("known_max_usd", est["max_usd"] or 0.0) + hi, 4)
        est["max_usd"] = None if est.get("price_unknown") else est["known_max_usd"]
        est["paid"] = True
    if not est["paid"]:
        return est
    bal = pc.fetch_balance(force=True)
    floor = _setting("min_balance_usd", DEFAULT_MIN_BALANCE_USD)
    if bal is not None and bal["effective_usd"] < floor:
        raise pc.CostGateError(
            "insufficient_credits",
            f"Not run on Comfy Cloud: the credit balance is ${bal['effective_usd']:,.2f}, below the "
            f"${floor:,.2f} minimum for a cloud run. Nothing was submitted; the user has to top up "
            "(platform.comfy.org) or run it locally.", est, bal)
    return pc.gate_estimate(est, confirm=confirm, workflow_path=workflow_path)


# ── submit ───────────────────────────────────────────────────────────────────
def _input_dir() -> Optional[Path]:
    try:
        from agenty_core.tools.comfyui import _dirs_from_comfyui
        d = _dirs_from_comfyui().get("input_dir")
        if d:
            return Path(d)
    except Exception:  # noqa: BLE001
        pass
    try:
        from agenty_core.utils.comfyui_client import comfyui_output_dir
        out = comfyui_output_dir()
        return out.parent / "input" if out else None
    except Exception:  # noqa: BLE001
        return None


def stage_inputs(prompt: dict, client: comfy_cloud.CloudClient, find_input=None) -> dict:
    """Upload the files the loader nodes name (from ComfyUI's input folder, or by absolute path) to the cloud and
    point the nodes at the uploaded names. Returns ``{local name: cloud name}``."""
    find = find_input or (lambda name: (lambda p: p if p.is_file() else None)(
        Path(name) if Path(name).is_absolute() else (_input_dir() or Path(".")) / name))
    done: dict = {}
    for node in (prompt or {}).values():
        if not isinstance(node, dict) or not str(node.get("class_type", "")).startswith(("Load", "VHS_Load")):
            continue
        inputs = node.get("inputs") or {}
        for key in _LOADER_KEYS:
            val = inputs.get(key)
            if not isinstance(val, str) or not val:
                continue
            if val in done:
                inputs[key] = done[val]
                continue
            if val in done.values():          # already a cloud name
                continue
            path = find(val)
            if path is None:
                continue              # not a local file (an example, or already a cloud name)
            done[val] = client.upload(path)
            inputs[key] = done[val]
    return done


def submit(prompt: dict, *, name: str = "", dest_dir: Optional[Path] = None, workflow_path: str = "",
           open_weight: bool = True, client: Optional[comfy_cloud.CloudClient] = None) -> str:
    """Gate, stage and queue *prompt* on Comfy Cloud; returns ``cloud:<prompt id>``.

    Raises ``CostGateError`` (balance / confirmation) or ``RuntimeError`` (the reason the cloud refused); the
    executors turn both into the usual submission-failed / cost answer.
    """
    client = client or comfy_cloud.get_client()
    est = gate(prompt, workflow_path, open_weight=open_weight)
    try:
        prompt = {k: (dict(v, inputs=dict(v.get("inputs") or {})) if isinstance(v, dict) else v)
                  for k, v in prompt.items()}                    # staging rewrites file names: not on the caller's copy
        stage_inputs(prompt, client)
        pid = client.submit(prompt)
    except comfy_cloud.CloudError as exc:
        raise RuntimeError(f"Comfy Cloud: {exc}") from None
    ref = PREFIX + pid
    _jobs[ref] = {"name": name or "agentY run", "dest": Path(dest_dir) if dest_dir else None, "t0": time.time(),
                  "estimate": est}
    pc.track(ref, est)
    return ref


def _dest(ref: str) -> Path:
    d = (_jobs.get(ref) or {}).get("dest")
    if d:
        return Path(d)
    env = os.environ.get("COMFY_CLOUD_OUTPUT_DIR", "").strip()
    if env:
        return Path(env)
    from agenty_core.paths import project_root
    return project_root() / "output_images" / "comfy_cloud"


def _safe(name: str) -> str:
    return re.sub(r"[^\w.\- ]", "_", Path(str(name)).name) or "output"


def collect_outputs(ref: str, detail: dict, client: comfy_cloud.CloudClient) -> list:
    """Download every output file the finished job lists; returns their local paths."""
    out_dir = _dest(ref)
    paths: list = []
    for node_out in (detail.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for key in ("images", "gifs", "videos", "audio"):
            for item in node_out.get(key) or []:
                if not (isinstance(item, dict) and item.get("filename")) or item.get("type") == "temp":
                    continue
                target = out_dir / _safe(item["filename"])
                if target.exists():                                   # never overwrite an earlier result
                    target = out_dir / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
                client.download(item["filename"], target, subfolder=item.get("subfolder", ""),
                                kind=item.get("type", "output"))
                paths.append(str(target))
    return paths


def error_details(detail: dict, job_id: str = "", fallback: str = "") -> dict:
    """The error dict the repair loop reads, from the cloud's ``execution_error``."""
    ee = detail.get("execution_error") if isinstance(detail, dict) else None
    ee = ee if isinstance(ee, dict) else {}
    msg = ee.get("exception_message") or fallback or "the cloud job failed"
    return {"cloud": True, "job_id": job_id, "node_id": str(ee.get("node_id") or ""),
            "node_type": str(ee.get("node_type") or ""), "exception_type": str(ee.get("exception_type") or ""),
            "exception_message": str(msg)[:1500], "traceback": list(ee.get("traceback") or [])}


async def stream(ref: str, *, poll_s: float = 3.0, max_wait_s: float = 0.0,
                 client: Optional[comfy_cloud.CloudClient] = None) -> AsyncGenerator:
    """Progress lines for cloud job *ref*, then one terminal dict (see the module doc)."""
    client = client or comfy_cloud.get_client()
    jid = job_id_of(ref)
    t0 = time.time()
    last, last_said, errors = "", 0.0, 0
    finished = False
    try:
        yield f"☁️ Sent to Comfy Cloud · job `{jid}` - waiting for a GPU…"
        while True:
            try:
                st = await asyncio.to_thread(client.job_status, jid)
                errors = 0
            except comfy_cloud.CloudError as exc:
                errors += 1
                if exc.kind in ("auth", "credits", "subscription") or errors >= 8:
                    finished = True
                    yield {"error": f"lost contact with the Comfy Cloud job {jid}: {exc}",
                           "details": error_details({}, jid, str(exc))}
                    return
                await asyncio.sleep(poll_s)
                continue
            state, waited = st["state"], time.time() - t0
            if state == "done":
                detail = await asyncio.to_thread(client.job_detail, jid)
                try:
                    files = await asyncio.to_thread(collect_outputs, ref, detail, client)
                except comfy_cloud.CloudError as exc:
                    finished = True
                    yield {"error": f"the cloud job finished but its outputs could not be downloaded: {exc}",
                           "details": error_details({}, jid, str(exc))}
                    return
                finished = True
                yield f"☁️ Cloud job finished in {farm_run._fmt_wait(waited)} · {len(files)} file(s)"  # noqa: SLF001
                yield {"history": farm_run.to_history(ref, files)}
                return
            if state == "error":
                try:
                    detail = await asyncio.to_thread(client.job_detail, jid)
                except comfy_cloud.CloudError:
                    detail = {}
                det = error_details(detail, jid, st.get("error") or "")
                finished = True
                yield f"❌ Cloud job {jid} failed: {det['exception_message'][:300]}"
                yield {"error": "Comfy Cloud job failed", "details": det}
                return
            if state == "cancelled":
                finished = True
                yield f"🛑 Cloud job {jid} was cancelled"
                yield {"interrupted": True, "error": "Comfy Cloud job cancelled"}
                return
            line = "☁️ Cloud: running" if state == "running" else "☁️ Cloud: queued - waiting for a free GPU"
            now = time.time()
            if line != last or now - last_said > 60:
                yield line + (f" ({farm_run._fmt_wait(waited)})" if waited >= 30 else "")  # noqa: SLF001
                last, last_said = line, now
            if max_wait_s and waited > max_wait_s:
                await asyncio.to_thread(client.cancel, jid)
                finished = True
                yield {"error": f"the Comfy Cloud job did not finish within {farm_run._fmt_wait(max_wait_s)} "  # noqa: SLF001
                                "(cancelled it)"}
                return
            await asyncio.sleep(poll_s)
    except asyncio.CancelledError:
        if not finished:                  # Stop pressed: do not leave a paid job running for nobody
            await asyncio.shield(asyncio.to_thread(client.cancel, jid))
        raise
    except GeneratorExit:
        if not finished:
            client.cancel(jid)
        raise


def max_wait_seconds() -> float:
    return farm_run.max_wait_seconds()


# ── where it ran, and what it cost ───────────────────────────────────────────
def record_run(ran_on: str, cost_usd: Optional[float] = None, ref: str = "") -> dict:
    """Note where a run happened in this turn's scope (``ran_on``: local | comfy_cloud | deadline)."""
    from agenty_core.utils import turn_scope
    entry = {"ran_on": ran_on, "cost_usd": cost_usd, "ref": ref}
    sc = turn_scope.current()
    with sc.lock:
        sc.slot("ran_on", list).append(entry)
    return entry


def runs_this_turn() -> list:
    from agenty_core.utils import turn_scope
    return list(turn_scope.current().get("ran_on", []) or [])


def settle_line(ref: str) -> str:
    """After a cloud job ends: ``"📍 ran_on: comfy_cloud · cost $0.12"`` (cost measured from the balance; "cost
    unknown" when the balance could not be read). Blocking: call it from a thread."""
    usd = None
    try:
        usd = pc.settle(ref)
    except Exception:  # noqa: BLE001 - a measurement must never fail a run
        usd = None
    record_run("comfy_cloud", usd, ref)
    return "📍 ran_on: comfy_cloud · " + (f"cost ${usd:,.2f}" if usd is not None else "cost unknown")


def ran_on_line(ran_on: str) -> str:
    record_run(ran_on)
    return f"📍 ran_on: {ran_on}"
