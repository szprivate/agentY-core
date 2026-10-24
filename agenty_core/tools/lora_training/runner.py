"""The process that runs a training job: ``python -m agenty_core.tools.lora_training.runner <run_dir>``.

Detached from whoever started it, so a host restart does not stop a training, and the viewer can follow it from
``progress.json`` without agentY. It executes the job's stages one after the other (install, cache, train), tails
the trainer's output into ``train.log``, keeps ``progress.json`` / ``samples.json`` / ``events.jsonl`` current,
obeys ``control.json`` (stop), and on success hands the LoRA to ``jobs.finish``. For a hosted (fal) job it only
polls the queue.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import backends as B
from . import fal as FAL
from . import formats as F
from . import jobs as J

TICK = 2.0
SERIES_MAX = 400


class State:
    def __init__(self, run: Path, job: dict):
        self.run, self.job = run, job
        self.p = F.read_json(run / "progress.json", {}) or {}
        self.p.setdefault("loss_series", [])
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.seen_samples: set[str] = set()
        self.sample_log: list = F.read_json(run / "samples.json", []) or []
        self.seen_ckpt: set[str] = set()
        self.last_vram = 0.0

    def set(self, **kw):
        with self.lock:
            self.p.update(kw)

    def add_loss(self, step, loss):
        s = self.p["loss_series"]
        if loss is None or step is None:
            return
        if s and s[-1][0] == step:
            s[-1][1] = loss
            return
        s.append([step, loss])
        if len(s) > SERIES_MAX:
            del s[1::2]

    def write(self):
        with self.lock:
            tot = self.p.get("total") or 0
            self.p["pct"] = round(min(1.0, (self.p.get("step") or 0) / tot), 4) if tot else 0.0
            self.p["elapsed_s"] = int(time.time() - self.t0)
            self.p["updated"] = time.time()
            F.write_json(self.run / "progress.json", self.p)


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        kids = parent.children(recursive=True)
        for k in kids:
            try:
                k.terminate()
            except Exception:  # noqa: BLE001
                pass
        parent.terminate()
        _, alive = psutil.wait_procs([parent, *kids], timeout=10)
        for a in alive:
            try:
                a.kill()
            except Exception:  # noqa: BLE001
                pass
    except ImportError:
        proc.terminate()
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _vram() -> float:
    from .preflight import gpu_memory
    g = gpu_memory()
    return g["used_gb"] if g else 0.0


def _read_stream(proc, state: State, parse, logf):
    buf = b""
    n_prog = 0
    while True:
        chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for raw in parts:
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            is_prog = False
            if parse:
                snap = {}
                if parse(line, snap):
                    is_prog = True
                    with state.lock:
                        state.p.update({k: v for k, v in snap.items() if v is not None})
                        state.add_loss(snap.get("step"), snap.get("loss"))
            if not is_prog or n_prog % 25 == 0:
                logf.write(line + "\n")
                logf.flush()
            n_prog += 1 if is_prog else 0


def _scan(state: State, backend, final_check=False):
    """Index new samples and checkpoints, raise events."""
    col = backend.collect(state.run, state.job)
    step_now = state.p.get("step") or 0
    new_by_step: dict = {}
    for s in col["samples"]:
        key = s["path"]
        if key in state.seen_samples:
            continue
        try:
            if time.time() - os.path.getmtime(key) < 1.5:       # still being written
                continue
        except OSError:
            continue
        state.seen_samples.add(key)
        step = s["step"] if s["step"] is not None else step_now
        dst_dir = state.run / "samples" / f"{step:06d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(key).name
        try:
            shutil.copyfile(key, dst)
        except OSError:
            continue
        prompts = state.job.get("sample_prompts") or []
        idx = s.get("idx", 0)
        new_by_step.setdefault(step, []).append({"prompt": prompts[idx] if idx < len(prompts) else "", "prompt_idx": idx,
                                                 "file": dst.relative_to(state.run).as_posix()})
    for step, files in sorted(new_by_step.items()):
        existing = next((e for e in state.sample_log if e["step"] == step), None)
        if existing:
            existing["files"].extend(files)
        else:
            state.sample_log.append({"step": step, "files": files})
        F.event(state.run, "sample", f"samples at step {step} ({len(files)} files)")
    if new_by_step:
        state.sample_log.sort(key=lambda e: e["step"])
        F.write_json(state.run / "samples.json", state.sample_log)
    for c in col["checkpoints"]:
        if c["path"] not in state.seen_ckpt:
            state.seen_ckpt.add(c["path"])
            state.set(last_checkpoint=c["path"])
            F.event(state.run, "checkpoint", f"checkpoint at step {c['step']}")
    return col


def run_local(run: Path, job: dict) -> int:
    state = State(run, job)
    be = B.get(job["backend"])
    F.event(run, "started", f"training {job['name']} started ({job.get('model_label')}, {job['backend']})")
    state.set(status="running", message="starting")
    state.write()
    logf = open(run / "train.log", "a", encoding="utf-8")
    stop_req = None
    for st in job["stages"]:
        state.set(message=st["name"])
        state.write()
        env = {**os.environ, **{k: str(v) for k, v in (st.get("env") or {}).items()}}
        logf.write(f"\n=== {st['name']} ===\n{' '.join(map(str, st['cmd']))}\n")
        logf.flush()
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            proc = subprocess.Popen([str(c) for c in st["cmd"]], cwd=st.get("cwd") or str(run), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    creationflags=flags)
        except OSError as exc:
            logf.write(f"could not start: {exc}\n")
            return _fail(state, f"could not start '{st['name']}': {exc}")
        reader = threading.Thread(target=_read_stream, args=(proc, state, be.parse if st.get("parse") else None, logf), daemon=True)
        reader.start()
        tick = 0
        while proc.poll() is None:
            time.sleep(0.5)
            ctl = F.read_json(run / "control.json")
            if ctl and ctl.get("action") == "stop" and stop_req is None:
                stop_req = ctl
                state.set(message="stopping")
                _kill_tree(proc)
                break
            tick += 1
            if tick % int(TICK / 0.5) == 0:
                if st.get("kind") == "train":
                    _scan(state, be)
                    if tick % 20 == 0:
                        state.last_vram = _vram()
                        state.set(vram_gb=state.last_vram)
                    s = state.p
                    if s.get("step") and s.get("total") and not s.get("eta_s") and s.get("it_s"):
                        s["eta_s"] = int((s["total"] - s["step"]) / s["it_s"])
                state.write()
        reader.join(timeout=5)
        if stop_req is not None:
            break
        if proc.returncode != 0:
            tail = J.log_tail(run, 25)
            return _fail(state, f"'{st['name']}' failed (exit {proc.returncode}). Last log lines:\n{tail}")
    logf.close()
    time.sleep(2.0)
    col = _scan(state, be, True)
    if stop_req is not None:
        state.set(status="stopped", message="stopped by the user")
        res = J.finish(run, job, None, col["checkpoints"], partial=True) if stop_req.get("keep") and col["checkpoints"] else None
        F.event(run, "stopped", "training stopped" + (f"; kept the last checkpoint ({res['lora']})" if res and res.get("lora") else ""))
        state.write()
        return 0
    if not col["final"] and not col["checkpoints"]:
        return _fail(state, "the trainer finished but wrote no LoRA; see train.log")
    state.set(status="done", message="done", step=state.p.get("total") or state.p.get("step"), eta_s=0)
    res = J.finish(run, job, col["final"], col["checkpoints"])
    state.set(last_checkpoint=res.get("lora") or state.p.get("last_checkpoint"))
    F.event(run, "finished", f"training finished; LoRA: {res.get('lora')}")
    state.write()
    return 0


def run_fal(run: Path, job: dict) -> int:
    state = State(run, job)
    f = job["fal"]
    F.event(run, "started", f"hosted training {job['name']} queued on fal (about ${job.get('estimate_usd', 0):.2f})")
    state.set(status="running", message="queued on fal")
    state.write()
    last_log = ""
    while True:
        ctl = F.read_json(run / "control.json")
        if ctl and ctl.get("action") == "stop":
            FAL.cancel(f.get("cancel_url") or "")
            state.set(status="stopped", message="cancelled on fal")
            F.event(run, "stopped", "hosted training cancelled")
            state.write()
            return 0
        try:
            p = FAL.poll(f["status_url"])
        except Exception as exc:  # noqa: BLE001
            time.sleep(15)
            state.set(message=f"waiting for fal ({exc})")
            state.write()
            if time.time() - state.t0 > 6 * 3600:
                return _fail(state, f"fal did not answer: {exc}")
            continue
        if p["log"] != last_log:
            (run / "train.log").write_text(p["log"], encoding="utf-8")
            last_log = p["log"]
            snap = FAL.progress_from_log(p["log"], job["steps"])
            state.set(**snap)
            state.add_loss(snap.get("step"), snap.get("loss"))
        state.set(message=str(p["status"]).lower().replace("_", " "))
        state.write()
        if p["status"] == "COMPLETED":
            break
        time.sleep(10)
    try:
        res = FAL.result(f["response_url"])
        url = FAL.lora_url(res)
        if not url:
            return _fail(state, f"fal finished but returned no LoRA file: {str(res)[:300]}")
        out = run / "out"
        out.mkdir(exist_ok=True)
        local = FAL.download(url, out / f"{job['name']}.safetensors")
    except Exception as exc:  # noqa: BLE001
        return _fail(state, f"could not fetch the result from fal: {exc}")
    state.set(status="done", message="done", step=job["steps"], total=job["steps"], eta_s=0)
    r = J.finish(run, job, str(local), [])
    try:
        from agenty_core import partner_credits as PC
        PC._log({"ts": time.time(), "usd": job.get("estimate_usd"), "estimated": True, "service": "fal",
                 "job": job["name"], "endpoint": f.get("endpoint")})
    except Exception:  # noqa: BLE001
        pass
    F.event(run, "finished", f"hosted training finished; LoRA: {r.get('lora')}")
    state.write()
    return 0


def _fail(state: State, msg: str) -> int:
    state.set(status="failed", message=msg[:1500])
    state.write()
    F.event(state.run, "failed", msg[:600])
    return 1


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    run = Path(argv[0])
    job = F.read_json(run / "job.json")
    if not job:
        print("no job.json", file=sys.stderr)
        return 2
    try:
        return run_fal(run, job) if job.get("mode") == "fal" else run_local(run, job)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        st = State(run, job)
        return _fail(st, f"runner crashed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    sys.exit(main())
