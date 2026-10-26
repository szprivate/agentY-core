"""Send ComfyUI workflows to an on-prem Thinkbox Deadline 10 render farm, and follow them.

Three tools: ``submit_to_deadline`` (an API-format workflow becomes one Deadline job of the ``ComfyUI`` plugin,
one task per run), ``deadline_job_status`` (state, progress, errors, output files) and ``deadline_farm_info``
(pools, groups, workers, and the dedicated ComfyUI pool). The plugin itself lives in the agentY repo under
``deadline/plugins/ComfyUI`` and is installed into the Deadline repository by the user (see docs/setup/deadline.md).

ComfyUI is installed on only some farm machines, so jobs ALWAYS go to one dedicated pool (default ``comfyui``,
``DEADLINE_COMFYUI_POOL``) holding just those machines. A job is refused when that pool does not exist or has no
workers, instead of being queued where it can never start. The caller may name another pool explicitly.

Transport: the Deadline Web Service when ``DEADLINE_WEBSERVICE_URL`` is set and reachable, otherwise the
``deadlinecommand`` executable (``DEADLINE_PATH`` = its bin folder; the standard install folders and PATH are
tried too). Everything is optional; with neither, the tools answer "Deadline not configured".

Settings (environment or the app's .env): ``DEADLINE_REPOSITORY`` (informational / checked for reachability),
``DEADLINE_WEBSERVICE_URL``, ``DEADLINE_PATH``, ``DEADLINE_COMFYUI_PATH`` (ComfyUI install folder on the workers,
passed to the plugin as an override of the plugin's own per-OS setting), ``DEADLINE_COMFYUI_POOL``,
``DEADLINE_GROUP``, ``DEADLINE_OUTPUT_DIR`` (a folder the workers can write: a network path), ``DEADLINE_PRIORITY``.

Inputs the workflow reads (LoadImage files and so on) are copied next to the outputs into ``_inputs/`` and the
prompt is rewritten to those names; the plugin uploads them into the worker's ComfyUI before the run. No API key
is ever written into a job file: partner-node keys come from the worker's own environment (see the plugin).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool

logger = logging.getLogger("agenty_core.deadline_farm")

PLUGIN_NAME = "ComfyUI"
DEFAULT_POOL = "comfyui"
MANIFEST_PREFIX = ".agenty_farm_"          # written by the plugin: <prefix><jobid>_<task>.json -> {"files": [...]}
_MEDIA_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".exr", ".mp4", ".mov", ".webm",
              ".mkv", ".avi", ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".glb", ".obj", ".ply"}
_TERMINAL = {"Completed", "Failed", "Suspended", "Archived", "Deleted"}


class FarmError(Exception):
    def __init__(self, message: str, kind: str = "api", **extra):
        super().__init__(message)
        self.kind, self.extra = kind, extra

    def render(self) -> str:
        return json.dumps({"error": str(self), "error_kind": self.kind, **self.extra})


# -- configuration ------------------------------------------------------------------


def _cfg(key: str, default: str = "") -> str:
    from agenty_core.utils.secrets import get_secret
    return (get_secret(key, default) or default).strip().strip('"')


def comfyui_pool() -> str:
    return _cfg("DEADLINE_COMFYUI_POOL", DEFAULT_POOL) or DEFAULT_POOL


def _dc_exe() -> Optional[str]:
    names = ["deadlinecommand.exe", "deadlinecommand"] if os.name == "nt" else ["deadlinecommand"]
    folders = [_cfg("DEADLINE_PATH")]
    if os.name == "nt":
        folders.append(r"C:\Program Files\Thinkbox\Deadline10\bin")
    else:
        folders += ["/opt/Thinkbox/Deadline10/bin", "/Applications/Thinkbox/Deadline10/Resources"]
    for f in folders:
        for n in names:
            if f and (Path(f) / n).is_file():
                return str(Path(f) / n)
    return shutil.which("deadlinecommand")


def _ws_url() -> str:
    return _cfg("DEADLINE_WEBSERVICE_URL").rstrip("/")


def _require_configured() -> None:
    if not _ws_url() and not _dc_exe():
        raise FarmError(
            "Deadline not configured: neither DEADLINE_WEBSERVICE_URL nor a deadlinecommand executable "
            "(DEADLINE_PATH = the Deadline bin folder) was found. Set one of them in .env (see docs/setup/deadline.md).",
            "config")


# -- transports -----------------------------------------------------------------------


def _dc(args: list, timeout: float = 60) -> str:
    exe = _dc_exe()
    if not exe:
        raise FarmError("deadlinecommand not found (set DEADLINE_PATH)", "config")
    try:
        p = subprocess.run([exe, *[str(a) for a in args]], capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as exc:
        raise FarmError(f"deadlinecommand timed out after {timeout:.0f}s", "api") from exc
    out = (p.stdout or "").strip()
    if p.returncode != 0 or out.startswith("Error:") or out.startswith("Error "):
        raise FarmError(f"deadlinecommand failed: {(out or p.stderr or '').strip()[:800]}", "api")
    return out


def _kv(text: str) -> dict:
    d = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip()
    return d


def _ws(method: str, path: str, **kw):
    import requests
    url = _ws_url() + path
    try:
        r = requests.request(method, url, timeout=kw.pop("timeout", 30), **kw)
    except requests.RequestException as exc:
        raise FarmError(f"Deadline Web Service not reachable at {_ws_url()}: {exc}", "api", transport="webservice") from exc
    if r.status_code >= 400:
        raise FarmError(f"Deadline Web Service {r.status_code}: {r.text[:500]}", "api", transport="webservice")
    try:
        return r.json()
    except ValueError:
        return r.text


def _use_ws() -> bool:
    """The Web Service, when configured and answering; otherwise deadlinecommand."""
    if not _ws_url():
        return False
    try:
        _ws("GET", "/api/pools", timeout=5)
        return True
    except FarmError:
        if _dc_exe():
            logger.info("Deadline Web Service not reachable, using deadlinecommand")
            return False
        raise


# -- farm questions ---------------------------------------------------------------------


def _pools() -> list:
    if _use_ws():
        return [str(x) for x in _ws("GET", "/api/pools")]
    return [l.strip() for l in _dc(["-GetPoolNames"]).splitlines() if l.strip()]


def _groups() -> list:
    if _use_ws():
        return [str(x) for x in _ws("GET", "/api/groups")]
    return [l.strip() for l in _dc(["-GetGroupNames"]).splitlines() if l.strip()]


def _workers_state() -> dict:
    """{worker name: state} for every worker (Idle, Rendering, Offline, Stalled, ...)."""
    if _use_ws():
        data = _ws("GET", "/api/slaves?Data=infosettings")
        out = {}
        for s in data if isinstance(data, list) else []:
            info = s.get("Info", s) if isinstance(s, dict) else {}
            name = s.get("Name") or info.get("Name") or s.get("SlaveName", "")
            out[name] = str(info.get("Status") or info.get("SlaveState") or s.get("Stat", "Unknown"))
        return out
    text = _dc(["-GetSlaves"], timeout=120)
    out = {}
    for block in re.split(r"\r?\n\s*\r?\n", text):
        d = _kv(block)
        if d.get("Name") or d.get("SlaveName"):
            out[d.get("Name") or d["SlaveName"]] = d.get("SlaveState") or d.get("SlaveStatus") or "Unknown"
    return out


def _pool_workers(pool: str) -> list:
    """The workers assigned to one pool: [{"name", "state"}]."""
    if _use_ws():
        return [{"name": n, "state": st} for n, st in _workers_state().items()
                if pool in _worker_pools_ws(n)]
    names = [l.strip() for l in _dc(["-GetSlaveNamesInPool", pool]).splitlines() if l.strip()]
    out = []
    for n in names:
        try:
            st = _dc(["-GetSlaveInfo", n, "SlaveState"]).strip().splitlines()[-1].split("=")[-1].strip()
        except FarmError:
            st = "Unknown"
        out.append({"name": n, "state": st or "Unknown"})
    return out


def _worker_pools_ws(name: str) -> list:
    try:
        d = _ws("GET", f"/api/slaves?Data=settings&Name={name}")
        s = d[0] if isinstance(d, list) and d else d
        return list((s.get("Settings", s) if isinstance(s, dict) else {}).get("Pools", []) or [])
    except FarmError:
        return []


def _pool_status(pool: str) -> dict:
    exists = pool in _pools()
    workers = _pool_workers(pool) if exists else []
    usable = [w for w in workers if w["state"].lower() in ("idle", "rendering", "starting", "starting job")]
    return {"pool": pool, "exists": exists, "workers": workers, "worker_count": len(workers),
            "online_count": len(usable)}


def _setup_hint(pool: str) -> str:
    return (f"Create the pool and assign the machines that have ComfyUI installed (docs/setup/deadline.md): "
            f"Monitor > Tools > Manage Pools (add '{pool}'), then Super User mode > Manage Workers > assign the pool; or "
            f"`deadlinecommand -AddPool {pool}` and `deadlinecommand -SetPoolsForSlave <worker,worker> {pool}`. "
            f"A different pool name is set with DEADLINE_COMFYUI_POOL.")


def farm_info() -> dict:
    _require_configured()
    transport = "webservice" if _use_ws() else "deadlinecommand"
    pool = comfyui_pool()
    pools, groups = _pools(), _groups()
    states = _workers_state()
    counts: dict = {}
    for s in states.values():
        counts[s] = counts.get(s, 0) + 1
    ps = _pool_status(pool)
    issues = []
    if not ps["exists"]:
        issues.append(f"pool '{pool}' does not exist. " + _setup_hint(pool))
    elif not ps["workers"]:
        issues.append(f"pool '{pool}' has no workers. " + _setup_hint(pool))
    elif not ps["online_count"]:
        issues.append(f"no worker in pool '{pool}' is online right now (jobs will wait)")
    return {"transport": transport, "repository": _cfg("DEADLINE_REPOSITORY") or _repo_root(),
            "pools": pools, "groups": groups, "default_group": _cfg("DEADLINE_GROUP") or None,
            "workers_total": len(states), "workers_by_state": counts,
            "comfyui_pool": ps, "output_dir": _cfg("DEADLINE_OUTPUT_DIR") or None,
            "issues": issues, "ready": not issues}


def _repo_root() -> str:
    try:
        return _dc(["-GetRepositoryRoot"]).strip()
    except FarmError:
        return ""


# -- the workflow: API format, inputs ------------------------------------------------------


def _load_workflow(workflow) -> dict:
    if isinstance(workflow, (str, os.PathLike)):
        s = str(workflow)
        if not s.lstrip().startswith("{") and Path(s).is_file():
            s = Path(s).read_text(encoding="utf-8")
        try:
            workflow = json.loads(s)
        except ValueError as exc:
            raise FarmError(f"workflow is not valid JSON: {exc}", "args") from exc
    if not isinstance(workflow, dict):
        raise FarmError("workflow must be an API-format prompt (a JSON object of nodes)", "args")
    if isinstance(workflow.get("prompt"), dict):     # a wrapped /prompt payload: keep the prompt, drop extra_data (keys)
        workflow = workflow["prompt"]
    if "nodes" in workflow and "links" in workflow:
        raise FarmError("this is a UI-format workflow (nodes/links); the farm needs the API format "
                        "(the prompt that /prompt takes). Convert it first.", "args")
    bad = [k for k, v in workflow.items() if not (isinstance(v, dict) and "class_type" in v)]
    if not workflow or bad:
        raise FarmError("workflow is not an API-format prompt: every entry needs a class_type"
                        + (f" (offending: {bad[:3]})" if bad else ""), "args")
    return workflow


def _comfy_dirs() -> dict:
    try:
        from agenty_core.tools.comfyui import get_comfyui_dirs
        d = json.loads(get_comfyui_dirs())
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001 - ComfyUI may be off; absolute paths still work
        return {}


_ANNOT = re.compile(r"^(.*?)\s*\[(input|output|temp)\]\s*$")


def stage_inputs(prompt: dict, stage_dir: Path, input_dir: str = "", output_dir: str = "") -> tuple:
    """Copy the files the prompt reads into ``stage_dir`` and rewrite the prompt to their relative names.
    Returns (new prompt, [relative names])."""
    prompt = json.loads(json.dumps(prompt))
    staged: dict = {}          # source path -> relative name
    taken: set = set()
    for node in prompt.values():
        for key, val in list((node.get("inputs") or {}).items()):
            if not isinstance(val, str) or len(val) > 400 or Path(val.split(" [")[0]).suffix.lower() not in _MEDIA_EXT:
                continue
            name, base = val, input_dir
            m = _ANNOT.match(val)
            if m:
                name = m.group(1)
                base = {"input": input_dir, "output": output_dir, "temp": ""}[m.group(2)]
            src = Path(name) if os.path.isabs(name) else (Path(base) / name if base else None)
            if src is None or not src.is_file():
                continue
            key_src = str(src.resolve())
            if key_src not in staged:
                rel = name.replace("\\", "/") if not os.path.isabs(name) else src.name
                rel = rel.lstrip("/")
                i = 1
                while rel in taken:
                    rel = f"{i}_{Path(rel).name}"
                    i += 1
                taken.add(rel)
                dst = stage_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                staged[key_src] = rel
            node["inputs"][key] = staged[key_src]
    return prompt, list(staged.values())


# -- submit --------------------------------------------------------------------------------


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:60] or "comfyui"


def _write_info(path: Path, d: dict) -> None:
    path.write_text("".join(f"{k}={v}\n" for k, v in d.items() if v not in (None, "")), encoding="utf-8")


def submit(workflow, name: str = "", pool: str = "", group: str = "", priority: int = -1, output_dir: str = "",
           batch: int = 1, frames: str = "", randomize_seed: bool = True, comment: str = "") -> dict:
    _require_configured()
    prompt = _load_workflow(workflow)
    pool = (pool or "").strip() or comfyui_pool()
    ps = _pool_status(pool)
    if not ps["exists"]:
        raise FarmError(f"Deadline pool '{pool}' does not exist, so the job would never start. " + _setup_hint(pool),
                        "config", pool=pool)
    if not ps["workers"]:
        raise FarmError(f"Deadline pool '{pool}' has no workers, so the job would never start. " + _setup_hint(pool),
                        "config", pool=pool)

    base = (output_dir or "").strip() or _cfg("DEADLINE_OUTPUT_DIR")
    if not base:
        raise FarmError("no output folder: pass output_dir or set DEADLINE_OUTPUT_DIR to a network folder the "
                        "workers can write to", "config")
    name = (name or "").strip() or "agentY render"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(base) if output_dir else Path(base) / f"{_safe(name)}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    dirs = _comfy_dirs()
    prompt, inputs = stage_inputs(prompt, out / "_inputs", dirs.get("input_dir", ""), dirs.get("output_dir", ""))

    batch = max(1, int(batch or 1))
    frames = (frames or "").strip() or ("0" if batch == 1 else f"0-{batch - 1}")
    priority = int(priority) if int(priority) >= 0 else int(_cfg("DEADLINE_PRIORITY", "50") or 50)
    group = (group or "").strip() or _cfg("DEADLINE_GROUP")

    job = {"Plugin": PLUGIN_NAME, "Name": name, "BatchName": "agentY", "Pool": pool, "Group": group or "none",
           "Priority": priority, "Frames": frames, "ChunkSize": 1, "ConcurrentTasks": 1,
           "OutputDirectory0": str(out), "Comment": comment or "Submitted by agentY"}
    plug = {"PromptFile": "prompt.json", "OutputDirectory": str(out),
            "InputDirectory": str(out / "_inputs") if inputs else "",
            "SeedMode": "randomize" if (randomize_seed and (batch > 1 or "-" in frames or "," in frames)) else "keep",
            "ComfyUIPath": _cfg("DEADLINE_COMFYUI_PATH")}
    for i, rel in enumerate(inputs):
        plug[f"Input{i}"] = rel

    tmp = Path(tempfile.mkdtemp(prefix="agenty_deadline_"))
    try:
        pj = tmp / "prompt.json"
        pj.write_text(json.dumps(prompt), encoding="utf-8")
        job_id = _submit_files(job, plug, pj, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"status": "submitted", "job_id": job_id, "name": name, "pool": pool, "group": group or None,
            "tasks": _count_frames(frames), "output_dir": str(out), "inputs_staged": inputs,
            "workers_in_pool": ps["worker_count"], "workers_online": ps["online_count"],
            "hint": "follow it with deadline_job_status(job_id)"}


def _count_frames(frames: str) -> int:
    n = 0
    for part in frames.split(","):
        m = re.match(r"^\s*(-?\d+)\s*(?:-\s*(-?\d+))?", part)
        if m:
            n += (int(m.group(2)) - int(m.group(1)) + 1) if m.group(2) else 1
    return max(n, 1)


def _submit_files(job: dict, plug: dict, aux: Path, tmp: Path) -> str:
    if _use_ws():
        body = {"JobInfo": {k: str(v) for k, v in job.items() if v not in (None, "")},
                "PluginInfo": {k: str(v) for k, v in plug.items() if v not in (None, "")},
                "AuxFiles": [str(aux)], "IdOnly": True}
        r = _ws("POST", "/api/jobs", json=body)
        jid = r.get("_id") if isinstance(r, dict) else str(r).strip().strip('"')
        if not jid:
            raise FarmError(f"Deadline Web Service returned no job id: {r}", "api")
        return str(jid)
    ji, pi = tmp / "job_info.job", tmp / "plugin_info.job"
    _write_info(ji, job)
    _write_info(pi, plug)
    out = _dc([ji, pi, aux], timeout=180)
    jid = _kv(out).get("JobID")
    if not jid:
        raise FarmError(f"Deadline accepted no job: {out[:600]}", "api")
    return jid


# -- status ----------------------------------------------------------------------------------


def _job(job_id: str) -> dict:
    if _use_ws():
        r = _ws("GET", f"/api/jobs?JobID={job_id}")
        j = r[0] if isinstance(r, list) and r else None
        if not j:
            raise FarmError(f"no such job: {job_id}", "args")
        stat = {1: "Active", 2: "Suspended", 3: "Completed", 4: "Failed", 5: "Pending"}.get(j.get("Stat"), "Unknown")
        chunks = ("CompletedChunks", "RenderingChunks", "FailedChunks", "QueuedChunks", "SuspendedChunks", "PendingChunks")
        return {"JobStatus": stat, "TaskCount": sum(int(j.get(c, 0) or 0) for c in chunks),
                "CompletedChunks": j.get("CompletedChunks", 0), "RenderingChunks": j.get("RenderingChunks", 0),
                "FailedChunks": j.get("FailedChunks", 0), "QueuedChunks": j.get("QueuedChunks", 0),
                "SuspendedChunks": j.get("SuspendedChunks", 0), "ErrorReports": j.get("ErrorReports", 0),
                "OutputDirectories": ",".join(j.get("OutDir", []) or []), "Name": j.get("Props", {}).get("Name", ""),
                "Pool": j.get("Props", {}).get("Pool", "")}
    d = _kv(_dc(["-GetJob", job_id]))
    if not d.get("JobStatus") and not d.get("Status"):
        raise FarmError(f"no such job: {job_id}", "args")
    d["JobStatus"] = d.get("JobStatus") or d.get("Status")
    return d


def _error_text(job_id: str, limit: int = 1500) -> str:
    try:
        files = [l.strip() for l in _dc(["-GetJobErrorReportFilenames", job_id]).splitlines() if l.strip()]
        txt = Path(files[-1]).read_text(encoding="utf-8", errors="replace") if files else ""
    except (FarmError, OSError):
        return ""
    m = re.search(r"(?s)(Error Message[^\n]*\n.*)", txt)
    return (m.group(1) if m else txt).strip()[:limit]


def _outputs(job: dict, job_id: str) -> list:
    files: list = []
    for d in (job.get("OutputDirectories") or "").split(","):
        d = d.strip()
        if not d or not Path(d).is_dir():
            continue
        for mf in sorted(Path(d).glob(f"{MANIFEST_PREFIX}{job_id}_*.json")):
            try:
                files += json.loads(mf.read_text(encoding="utf-8")).get("files", [])
            except (OSError, ValueError):
                continue
    return list(dict.fromkeys(files))


def cancel_job(job_id: str) -> dict:
    """Stop a job started by ``submit``: suspend it (kept in Monitor for inspection, nothing deleted).

    Used when the person who asked stops the turn; the plugin interrupts ComfyUI on the worker when it sees the
    cancel. Never raises for "already finished".
    """
    _require_configured()
    job_id = (job_id or "").strip()
    if not job_id:
        raise FarmError("job_id is required", "args")
    try:
        if _use_ws():
            _ws("PUT", "/api/jobs", json={"Command": "suspend", "JobID": job_id})
        else:
            _dc(["-SuspendJob", job_id])
    except FarmError as exc:
        return {"job_id": job_id, "suspended": False, "note": str(exc)[:300]}
    return {"job_id": job_id, "suspended": True}


def job_status(job_id: str, wait_s: float = 0) -> dict:
    _require_configured()
    job_id = (job_id or "").strip()
    if not job_id:
        raise FarmError("job_id is required", "args")
    deadline = time.time() + max(0.0, min(float(wait_s or 0), 900.0))
    while True:
        j = _job(job_id)
        state = j.get("JobStatus", "Unknown")
        if state == "Active":
            state = "Rendering" if int(j.get("RenderingChunks") or 0) else "Queued"
        if state in _TERMINAL or time.time() >= deadline:
            break
        time.sleep(3)
    total = int(j.get("TaskCount") or 0)
    done = int(j.get("CompletedChunks") or 0)
    errors = int(j.get("ErrorReports") or 0)
    out = {"job_id": job_id, "name": j.get("Name", ""), "state": state, "pool": j.get("Pool", ""),
           "tasks": {"total": total, "completed": done, "rendering": int(j.get("RenderingChunks") or 0),
                     "queued": int(j.get("QueuedChunks") or 0), "failed": int(j.get("FailedChunks") or 0)},
           "progress": round(done / total, 3) if total else 0.0, "error_reports": errors}
    if errors or state == "Failed":
        out["errors"] = _error_text(job_id)
    if state == "Completed":
        out["outputs"] = _outputs(j, job_id)
        out["output_dir"] = (j.get("OutputDirectories") or "").split(",")[0]
    elif state in _TERMINAL:
        out["outputs"] = _outputs(j, job_id)
    if state == "Queued" and int(j.get("RenderingChunks") or 0) == 0:
        out["note"] = "waiting for a free worker in the pool"
    return out


# -- the tools ---------------------------------------------------------------------------------


def _guard(fn):
    def wrapper(*a, **kw):
        try:
            return json.dumps(fn(*a, **kw))
        except FarmError as exc:
            return exc.render()
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", fn.__name__)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "error_kind": "api"})
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


@tool
def submit_to_deadline(workflow: str, name: str = "", pool: str = "", group: str = "", priority: int = -1,
                       output_dir: str = "", batch: int = 1, frames: str = "", randomize_seed: bool = True) -> str:
    """Send a ComfyUI workflow to the on-prem Deadline render farm instead of the local ComfyUI.

    The workflow must be in API format (the prompt JSON ``/prompt`` takes), given as JSON text or a path to a
    .json file. It becomes one Deadline job; each task runs the workflow once on a farm machine that has ComfyUI.
    Jobs go to the dedicated ComfyUI pool (``DEADLINE_COMFYUI_POOL``, default "comfyui") unless ``pool`` is given;
    if that pool is missing or has no workers the call refuses and says how to set it up. Input images the
    workflow loads are copied to the farm share automatically. Returns at once with the job id: follow it with
    ``deadline_job_status``.

    Args:
        workflow: API-format workflow, JSON text or a path to a .json file.
        name: job name shown in Deadline Monitor.
        pool: override the ComfyUI pool (rarely wanted).
        group: Deadline group (default DEADLINE_GROUP, else none).
        priority: 0-100 (default DEADLINE_PRIORITY, else 50).
        output_dir: folder (reachable from the workers) for the results; default is a new folder under
            DEADLINE_OUTPUT_DIR.
        batch: run the workflow this many times (one task each) as variations.
        frames: Deadline frame list instead of batch, e.g. "0-9"; each number is one run.
        randomize_seed: with several runs, give every run its own random seeds (otherwise identical results).

    Returns:
        JSON: {"status": "submitted", "job_id", "pool", "output_dir", "tasks", ...} or {"error", "error_kind"}
        (config: Deadline or its pool not set up; args: not an API-format workflow; api: the farm refused).
    """
    return _submit_tool(workflow, name, pool, group, priority, output_dir, batch, frames, randomize_seed)


@tool
def deadline_job_status(job_id: str, wait_s: int = 0) -> str:
    """State of a Deadline job started with ``submit_to_deadline``: Queued, Rendering, Completed, Failed or
    Suspended, task progress, the error text of failed tasks, and the output files once it is done.

    Args:
        job_id: the id ``submit_to_deadline`` returned.
        wait_s: wait up to this many seconds (max 900) for the job to finish before answering.

    Returns:
        JSON: {"state", "progress" (0-1), "tasks": {...}, "errors"?, "outputs"? (file paths), "output_dir"?}.
    """
    return _status_tool(job_id, wait_s)


@tool
def deadline_farm_info() -> str:
    """A cheap look at the Deadline farm: pools, groups, worker counts by state, and the dedicated ComfyUI pool
    (does it exist, which workers are in it, are they online). Use it before submitting, or when a job does
    not start.

    Returns:
        JSON: {"transport", "pools", "groups", "workers_total", "workers_by_state", "comfyui_pool": {"pool",
        "exists", "workers": [{"name", "state"}], "online_count"}, "issues": [...], "ready": bool} or
        {"error": "Deadline not configured ..."}.
    """
    return _info_tool()


_submit_tool = _guard(submit)
_status_tool = _guard(job_status)
_info_tool = _guard(farm_info)
