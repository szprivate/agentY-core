"""Training jobs: start (gates, pre-flight, stages), status, stop, and the finish (copy + register the LoRA).

A job is a folder ``agent/lora_runs/<job>`` (see ``formats``). A local job is executed by ``runner`` - a detached
process, so it outlives the host that started it and the viewer can read its ``progress.json`` without agentY. A
hosted (fal) job is submitted here and the same runner only polls it.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import backends as B
from . import dataset as D
from . import fal as FAL
from . import formats as F
from . import preflight as PF
from .presets import get_preset, pick_backend

logger = logging.getLogger("agenty_core.lora.jobs")


class LoraError(Exception):
    def __init__(self, message: str, kind: str = "api", **extra):
        super().__init__(message)
        self.kind, self.extra = kind, extra

    def render(self) -> str:
        return json.dumps({"error": str(self), "error_kind": self.kind, **self.extra})


# test seams
_spawn_runner = None            # (run_dir) -> pid; tests replace it
_free_vram = None               # () -> bool


def free_comfyui_vram() -> bool:
    if _free_vram is not None:
        return _free_vram()
    try:
        from agenty_core.utils.comfyui_client import get_client
        get_client().post("/free", json_data={"unload_models": True, "free_memory": True})
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info("could not ask ComfyUI to free VRAM: %s", exc)
        return False


def default_sample_prompts(manifest: dict, n: int = 3) -> list[str]:
    inc = [i for i in manifest["items"] if i.get("included", True) and i.get("caption")]
    trig = manifest.get("trigger_word", "")
    out = []
    if inc:
        out.append(D.with_trigger(inc[0]["caption"], trig))
        if len(inc) > 2:
            out.append(inc[len(inc) // 2]["caption"])           # no trigger word: does the model still behave?
    kind = manifest.get("kind", "character")
    generic = {"character": f"{trig}, a candid photo in a different place, daylight",
               "style": f"a lighthouse on a cliff at dusk, in the style of {trig}",
               "product": f"{trig} on a wooden table in a bright kitchen",
               "motion": f"{trig}, a new scene"}.get(kind, f"{trig}, a new scene")
    out.append(generic)
    return out[: max(1, n)]


def _next_job_name(base: str, output_dir: str) -> str:
    root = F.runs_root(output_dir)
    k = 1
    while (root / f"{base}_v{k}").exists():
        k += 1
    return f"{base}_v{k}"


def _pid_alive(pid: Optional[int], started: float = 0.0) -> bool:
    if not pid:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        p = psutil.Process(pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        return not started or abs(p.create_time() - started) < 10
    except ImportError:
        return True
    except Exception:  # noqa: BLE001
        return False


def spawn_runner(run_dir: Path) -> tuple[int, float]:
    if _spawn_runner is not None:
        return _spawn_runner(run_dir)
    flags = 0
    if os.name == "nt":
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = open(Path(run_dir) / "runner.log", "wb")
    proc = subprocess.Popen([sys.executable, "-m", "agenty_core.tools.lora_training.runner", str(run_dir)],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(run_dir),
                            creationflags=flags, close_fds=True)
    started = time.time()
    try:
        import psutil
        started = psutil.Process(proc.pid).create_time()
    except Exception:  # noqa: BLE001
        pass
    return proc.pid, started


def start(dataset: str, model: str = "", backend: str = "auto", preset_overrides: Optional[dict] = None,
          name: str = "", free_vram: bool = True, allow_download: bool = False, confirm_cost: bool = False,
          output_dir: str = "", dry_run: bool = False, **overrides) -> dict:
    ds = F.resolve_dataset(dataset, output_dir)
    manifest = F.load_manifest(ds)
    ok, why = F.is_approved(ds)
    if not ok:
        raise LoraError(f"Not started: {why}. The user has to approve the dataset first: call lora_dataset_review "
                        "and wait for their approval (never approve on their behalf).", "not_approved", dataset=str(ds))
    preset = get_preset(model or manifest.get("model", ""))
    if manifest.get("model") and preset["key"] != manifest["model"] and not model:
        preset = get_preset(manifest["model"])
    try:
        be = pick_backend(preset, backend, allow_hosted=bool(confirm_cost) or backend == "fal")
    except ValueError as exc:
        raise LoraError(str(exc), "args") from exc
    cfg = dict(preset["backends"][be])
    ov = {k: v for k, v in {**(preset_overrides or {}), **overrides}.items() if v not in (None, "", 0)}
    steps = int(ov.get("steps") or preset.get("steps", 2000))
    job = {
        "dataset": str(ds), "dataset_name": manifest["name"], "model": preset["key"], "model_label": preset["label"],
        "backend": be, "steps": steps, "rank": int(ov.get("rank") or preset.get("rank", 32)),
        "learning_rate": float(ov.get("learning_rate") or preset.get("learning_rate", 1e-4)),
        "batch_size": int(ov.get("batch_size") or preset.get("batch_size", 1)),
        "optimizer": ov.get("optimizer") or preset.get("optimizer"), "seed": int(ov.get("seed", 42)),
        "resolutions": ov.get("resolutions") or preset.get("resolutions"),
        "save_every": int(ov.get("save_every") or max(100, steps // 8)),
        "sample_every": int(ov.get("sample_every") or max(50, steps // 8)),
        "frames": ov.get("frames") or preset.get("frames"), "fps": ov.get("fps") or preset.get("fps"),
        "low_vram": ov.get("low_vram"), "trigger_word": manifest.get("trigger_word", ""),
    }
    base = F.slug(name) or F.slug(manifest["name"])
    job["name"] = _next_job_name(base, output_dir) if not dry_run else f"{base}_dry"
    run = F.runs_root(output_dir) / job["name"]
    ag_out = F.agent_root(output_dir)
    report = PF.check(preset, be, run if not dry_run else ag_out, allow_download=allow_download)
    problems = list(report["problems"])
    if free_vram and be != "fal" and not dry_run:
        job["comfyui_vram_freed"] = free_comfyui_vram()
        if job["comfyui_vram_freed"]:
            time.sleep(1.0)
            report = PF.check(preset, be, run, allow_download=allow_download)
            problems = list(report["problems"])
    if problems:
        raise LoraError("Pre-flight failed: " + " | ".join(problems), "preflight", preflight=report)
    est = None
    if be == "fal":
        est = FAL.estimate_usd(cfg, steps)
        _cost_gate(est, preset, confirm_cost)
    if dry_run:
        return {"status": "dry_run", "job": job, "preflight": report, "estimate_usd": est,
                "stages": [] if be == "fal" else [s["name"] for s in _stages(job, preset, cfg, be, manifest, ds, run, report, plan_only=True)]}
    run.mkdir(parents=True, exist_ok=True)
    (run / "samples").mkdir(exist_ok=True)
    trainer = D.write_trainer_dataset(ds, manifest, 32 if preset["kind"] == "video" else 16)
    job["trainer_dataset"] = trainer["folder"]
    job["items"] = trainer["count"]
    job["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    job["sample_prompts"] = list(overrides.get("sample_prompts") or default_sample_prompts(manifest))
    job["preflight"] = {"warnings": report["warnings"], "gpu": report["gpu"], "base_model": report["base_model"]}
    if be == "fal":
        data_url = FAL.upload(FAL.zip_dataset(Path(trainer["folder"])), f"{job['name']}.zip")
        body = FAL.build_input(cfg, data_url, steps, job["learning_rate"] if ov.get("learning_rate") else None,
                               manifest.get("trigger_word", ""))
        sub = FAL.submit(cfg, body)
        job.update(mode="fal", fal=sub, estimate_usd=est, stages=[])
    else:
        ctx = {"preset": preset, "cfg": cfg, "job": job, "dataset_dir": trainer["folder"], "run_dir": str(run),
               "manifest": manifest, "base_model": report["base_model"], "sample_prompts": job["sample_prompts"],
               "hf_home": str(PF.hf_home())}
        job["mode"] = "local"
        job["stages"] = _stages(job, preset, cfg, be, manifest, ds, run, report, ctx=ctx)
    F.write_json(run / "job.json", job)
    F.write_json(run / "progress.json", {"job": job["name"], "status": "starting", "step": 0, "total": steps, "pct": 0.0,
                                         "eta_s": None, "it_s": None, "elapsed_s": 0, "loss": None, "lr": None,
                                         "vram_gb": None, "loss_series": [], "updated": time.time(),
                                         "message": "starting", "backend": be, "model": preset["key"],
                                         "dataset": manifest["name"], "last_checkpoint": None})
    pid, started = spawn_runner(run)
    job.update(runner_pid=pid, runner_started=started, started=time.time())
    F.write_json(run / "job.json", job)
    manifest["status"] = "training"
    F.save_manifest(ds, manifest)
    return {"status": "started", "job": job["name"], "run_dir": str(run), "backend": be, "steps": steps,
            "items": trainer["count"], "warnings": report["warnings"], "estimate_usd": est,
            "base_model": report["base_model"].get("where"), "sample_prompts": job["sample_prompts"]}


def _stages(job, preset, cfg, be, manifest, ds, run, report, ctx=None, plan_only=False):
    if be == "fal":
        return []
    if ctx is None:
        ctx = {"preset": preset, "cfg": cfg, "job": job, "dataset_dir": str(Path(ds) / "trainer"), "run_dir": str(run),
               "manifest": manifest, "base_model": report["base_model"], "sample_prompts": default_sample_prompts(manifest),
               "hf_home": str(PF.hf_home())}
    if plan_only:
        run = Path(run)
        saved = run.exists()
        run.mkdir(parents=True, exist_ok=True)
        try:
            return B.get(be).stages(ctx)
        finally:
            if not saved:
                shutil.rmtree(run, ignore_errors=True)
    return B.get(be).stages(ctx)


def _cost_gate(est: float, preset: dict, confirm: bool) -> None:
    from agenty_core import partner_credits as PC
    thr = PC.confirm_threshold()
    ok = confirm or PC._confirm_all.get()
    if est > 0 and est >= thr and not ok:
        raise LoraError(
            f"Not started: hosted training on fal costs about ${est:,.2f} ({preset['label']}; an estimate - check "
            f"fal.ai/pricing), at or above the ${thr:,.2f} confirmation level. Tell the user the estimate and ask; only "
            "when they say yes, call train_lora again with confirm_cost=True.", "confirm_cost", estimate_usd=est)


def load(job: str, output_dir: str = "") -> tuple[Path, dict]:
    p = Path(os.path.expanduser(str(job)))
    run = p if (p / "job.json").exists() else F.runs_root(output_dir) / F.slug(job)
    j = F.read_json(run / "job.json")
    if j is None:
        raise LoraError(f"no LoRA training called '{job}' (looked in {run})", "args")
    return run, j


def runs(output_dir: str = "") -> list[dict]:
    out = []
    for d in sorted(F.runs_root(output_dir).iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        pr = F.read_json(d / "progress.json")
        if pr:
            out.append({"job": d.name, "status": pr.get("status"), "step": pr.get("step"), "total": pr.get("total"),
                        "model": pr.get("model"), "dataset": pr.get("dataset")})
    return out


def status(job: str, wait_s: float = 0, output_dir: str = "", log_lines: int = 0) -> dict:
    run, j = load(job, output_dir)
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        pr = F.read_json(run / "progress.json", {}) or {}
        running = pr.get("status") in ("starting", "running")
        if running and not _pid_alive(j.get("runner_pid"), j.get("runner_started", 0)):
            pr["status"], pr["message"] = "failed", "the runner process is gone (host or machine restarted?); see train.log"
            F.write_json(run / "progress.json", pr)
            F.event(run, "failed", pr["message"])
            running = False
        if not running or time.monotonic() >= deadline:
            break
        time.sleep(2.0)
    out = {k: pr.get(k) for k in ("status", "step", "total", "pct", "eta_s", "it_s", "elapsed_s", "loss", "lr", "vram_gb",
                                  "message", "last_checkpoint")}
    out.update(job=j["name"], model=j.get("model_label"), backend=j.get("backend"), run_dir=str(run),
               samples=_latest_samples(run))
    if (run / "result.json").exists():
        out["result"] = F.read_json(run / "result.json")
    if log_lines:
        out["log_tail"] = log_tail(run, log_lines)
    out["new_events"] = F.read_events(run)[-10:]
    if pr.get("status") == "done":
        out["next"] = ("Training is done. The LoRA is registered; render the same prompts and seeds with and without it "
                       "(see result.test_plan), look at result.comparison_sheet, and give the user your verdict. "
                       "The user picks the checkpoint to keep.")
    elif pr.get("status") in ("starting", "running"):
        out["next"] = "Still training; do not poll in a loop. Check again when the user asks or after a sample/notification."
    return out


def _latest_samples(run: Path) -> list:
    s = F.read_json(run / "samples.json", []) or []
    return s[-1:] if s else []


def log_tail(run: Path, n: int = 40) -> str:
    try:
        return "\n".join((Path(run) / "train.log").read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def stop(job: str, keep: bool = True, output_dir: str = "") -> dict:
    run, j = load(job, output_dir)
    pr = F.read_json(run / "progress.json", {}) or {}
    if pr.get("status") not in ("starting", "running"):
        return {"status": pr.get("status", "unknown"), "job": j["name"], "note": "it is not running"}
    F.write_json(run / "control.json", {"action": "stop", "keep": bool(keep), "t": time.time()})
    if j.get("mode") == "fal" and j.get("fal", {}).get("cancel_url"):
        FAL.cancel(j["fal"]["cancel_url"])
    if not _pid_alive(j.get("runner_pid"), j.get("runner_started", 0)):
        pr["status"], pr["message"] = "stopped", "stopped (the runner was already gone)"
        F.write_json(run / "progress.json", pr)
    return {"status": "stopping", "job": j["name"], "keep_last_checkpoint": bool(keep),
            "note": "The runner stops the trainer within a few seconds" + (" and keeps the last checkpoint as the LoRA." if keep else ".")}


# -- finish --------------------------------------------------------------------


def loras_dir() -> Optional[Path]:
    """ComfyUI's loras folder (first one off C:); ``AGENTY_LORAS_DIR`` overrides (tests, smoke runs)."""
    env = os.environ.get("AGENTY_LORAS_DIR", "").strip()
    if env:
        return Path(env)
    try:
        from agenty_core.tools.model_research import _model_roots
        cands = [p for cat, p in _model_roots() if cat == "loras"]
    except Exception:  # noqa: BLE001
        cands = []
    for p in cands:
        if PF.drive_of(p) != "C":
            return p
    return cands[0] if cands else None


def finish(run: Path, job: dict, final: Optional[str], checkpoints: list, partial: bool = False) -> dict:
    """Copy the LoRA into ComfyUI's loras folder, register it, build the comparison sheet. Returns result.json."""
    name = job["name"]
    result = {"lora": None, "checkpoints": checkpoints, "registered": None, "partial": partial}
    src = final or (checkpoints[-1]["path"] if checkpoints else None)
    ld = loras_dir()
    if src and ld is not None:
        try:
            ld.mkdir(parents=True, exist_ok=True)
            sub = ld / "agentY"
            sub.mkdir(exist_ok=True)
            dst = sub / (f"{name}.safetensors" if not partial else f"{name}_step{(checkpoints[-1]['step'] if checkpoints else 0):06d}.safetensors")
            shutil.copyfile(src, dst)
            rel = f"agentY/{dst.name}"
            result["lora"] = str(dst)
            meta = {"name": name, "file": rel, "trigger_word": job.get("trigger_word", ""), "base_model": job.get("model_label"),
                    "model_key": job.get("model"), "recommended_strength": 1.0, "backend": job.get("backend"),
                    "steps": job.get("steps"), "dataset": job.get("dataset_name"), "trained": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "format_note": ("musubi-tuner LoRAs for some models need its convert_lora script before ComfyUI reads "
                                    "them; if the LoRA has no effect, check docs/<model>.md of musubi-tuner")
                    if job.get("backend") == "musubi" else ""}
            result["registered"] = meta
            reg = Path(run).parent.parent / "lora_registry.json"        # agent/lora_registry.json
            data = F.read_json(reg, {}) or {}
            data[name] = meta
            F.write_json(reg, data)
            try:
                from agenty_core.tools import worldlabs_atlas as wl
                wl._sidecar(dst, "LoRA", meta)
            except Exception:  # noqa: BLE001
                pass
        except OSError as exc:
            result["copy_error"] = str(exc)
    elif src:
        result["lora"] = src
        result["copy_error"] = "ComfyUI's loras folder was not found; the LoRA stays in the run folder"
    sheet = comparison_sheet(run)
    if sheet:
        result["comparison_sheet"] = sheet
    result["test_plan"] = {"prompts": job.get("sample_prompts", []), "seed": job.get("seed", 42),
                           "lora_name": (result.get("registered") or {}).get("file"),
                           "strengths": [0.0, 0.8, 1.0], "trigger_word": job.get("trigger_word", ""),
                           "note": "same prompts and seed with strength 0 (without the LoRA) and the others, then compare"}
    F.write_json(Path(run) / "result.json", result)
    try:
        m = F.read_json(Path(job["dataset"]) / "manifest.json")
        if m:
            m["status"] = "trained" if not partial else m.get("status", "approved")
            F.save_manifest(Path(job["dataset"]), m)
    except Exception:  # noqa: BLE001
        pass
    return result


def comparison_sheet(run: Path) -> Optional[str]:
    """One sheet: for each sample prompt, the first, a middle and the last step."""
    s = F.read_json(Path(run) / "samples.json", []) or []
    imgs = []
    for entry in s:
        for f in entry.get("files", []):
            if str(f["file"]).lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                imgs.append((entry["step"], f.get("prompt_idx", 0), Path(run) / f["file"]))
    if len(imgs) < 2:
        return None
    steps = sorted({i[0] for i in imgs if i[0] is not None})
    if not steps:
        return None
    pick = sorted({steps[0], steps[len(steps) // 2], steps[-1]})
    chosen = [i for i in imgs if i[0] in pick][:15]
    chosen.sort(key=lambda i: (i[1], i[0]))
    try:
        from agenty_core.utils.contact_sheet import build_contact_sheet
        out = Path(run) / "comparison.jpg"
        r = build_contact_sheet([str(i[2]) for i in chosen], [f"p{i[1] + 1} step {i[0]}" for i in chosen], cell=384,
                                columns=len(pick), out_path=str(out))
        return r["path"]
    except Exception as exc:  # noqa: BLE001
        logger.info("comparison sheet failed: %s", exc)
        return None
