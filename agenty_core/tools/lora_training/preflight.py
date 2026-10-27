"""Pre-flight checks before a LoRA training starts: GPU memory, disk (never C:), base model, backend installed.

Nothing here downloads or starts anything. ``check`` returns a report with ``ok`` and a list of ``problems``
(each says what to do); the tool refuses to start on any problem the user has not overridden.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import backends as B

logger = logging.getLogger("agenty_core.lora.preflight")

MIN_BUFFER_GB = 5.0     # same buffer the model downloader keeps


def gpu_memory() -> Optional[dict]:
    """{name, total_gb, free_gb, used_gb} of GPU 0 from nvidia-smi, or None (no NVIDIA GPU / no driver)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
        name, total, free, used = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "total_gb": round(float(total) / 1024, 1), "free_gb": round(float(free) / 1024, 1),
                "used_gb": round(float(used) / 1024, 1)}
    except Exception as exc:  # noqa: BLE001
        logger.info("nvidia-smi failed: %s", exc)
        return None


def drive_of(p) -> str:
    return os.path.splitdrive(str(p))[0].rstrip(":").upper()


def free_gb(p) -> float:
    probe = Path(p)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return float("inf")


def disk_check(path, need_gb: float, what: str) -> Optional[str]:
    """A problem string, or None. Refuses C:, and a volume that would be left under the buffer."""
    if drive_of(path) == "C":
        return (f"{what} would be on the C: system drive ({path}); LoRA training never writes there. "
                "Pass output_dir on another drive.")
    fr = free_gb(path)
    if fr - need_gb < MIN_BUFFER_GB:
        return f"{what} needs about {need_gb:.0f} GB on {drive_of(path) or path}: only {fr:.0f} GB free (keeping a {MIN_BUFFER_GB:.0f} GB buffer)"
    return None


def _yaml_roots() -> list[tuple[str, Path]]:
    """(category, folder) from ComfyUI's ``extra_model_paths.yaml`` read off disk: the fallback when ComfyUI does
    not answer ``/internal/folder_paths`` (the run itself finds files through the same extra paths)."""
    try:
        import yaml
        from agenty_core.tools.huggingface import _models_base_dir
        comfy = _models_base_dir().parent
        f = comfy / "extra_model_paths.yaml"
        if not f.is_file():
            return []
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        logger.info("extra_model_paths.yaml not read: %s", exc)
        return []
    out: list[tuple[str, Path]] = []
    for sect in doc.values() if isinstance(doc, dict) else []:
        if not isinstance(sect, dict):
            continue
        base = str(sect.get("base_path") or "")
        for cat, val in sect.items():
            if cat in ("base_path", "is_default") or not isinstance(val, str):
                continue
            for line in val.splitlines():
                line = line.strip()
                if line:
                    out.append((str(cat).lower(), Path(os.path.join(base, line)) if base else Path(line)))
    return out


def _local_model_files(pattern: str, authority: Optional[dict] = None) -> list[dict]:
    """Model files ComfyUI can load that match ``pattern``, by the same lookup the run uses (the running ComfyUI's
    folder list, extra model paths included). ``authority`` (optional dict) gets ``complete``: False when ComfyUI's
    folder list could not be read, so an empty answer means "do not know", not "not there"."""
    try:
        from agenty_core.tools import model_research as MR
        complete = bool(MR._folder_paths())
        if not complete:
            time.sleep(1.0)                              # ComfyUI may be busy, not down: one more try
            complete = bool(MR._folder_paths())
        hits = MR._local_files(pattern, limit=5)
        if not hits and not complete:                    # read the extra paths off disk instead
            index = []
            for cat, root in _yaml_roots():
                if root.is_dir():
                    for dp, _d, files in os.walk(root):
                        for f in files:
                            if f.lower().endswith(MR.MODEL_EXTENSIONS):
                                full = os.path.join(dp, f)
                                index.append((cat, os.path.relpath(full, root).replace("\\", "/"), full.replace("\\", "/")))
            hits = MR._local_files(pattern, limit=5, index=index)
            complete = bool(hits) or bool(index)
        if authority is not None:
            authority["complete"] = complete or bool(hits)
        return hits
    except Exception as exc:  # noqa: BLE001
        logger.info("local model lookup failed: %s", exc)
        if authority is not None:
            authority["complete"] = False
        return []


def hf_home() -> Path:
    """Where huggingface downloads go: ``HF_HOME`` / ``HF_HUB_CACHE``, else a folder beside the backend (never C:)."""
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if env:
        return Path(env)
    return B.tools_root() / "hf_cache"


def hf_cached(repo: str) -> bool:
    """Is a huggingface repo already in the cache (any of the usual cache places)?"""
    cands = [hf_home(), hf_home() / "hub", Path.home() / ".cache" / "huggingface" / "hub"]
    safe = "models--" + repo.replace("/", "--")
    return any((c / safe / "snapshots").is_dir() and any((c / safe / "snapshots").iterdir()) for c in cands if c.exists())


def base_model(preset: dict, backend: str) -> dict:
    """{present, where, download_gb, missing:[...]} for the backend's base weights."""
    cfg = preset["backends"].get(backend, {})
    if backend == "fal":
        return {"present": True, "where": "hosted", "download_gb": 0, "missing": []}
    if backend == "ai-toolkit":
        lf = cfg.get("local_comfy_file")
        if lf:
            auth: dict = {}
            hits = _local_model_files(lf.rsplit("/", 1)[-1], auth)
            if hits:
                return {"present": True, "where": hits[0]["path"], "download_gb": 0, "missing": [], "local_file": hits[0]["path"]}
        local = os.environ.get("AITOOLKIT_MODEL_" + preset["key"].upper())
        if local and Path(local).exists():
            return {"present": True, "where": local, "download_gb": 0, "missing": [], "local_dir": local}
        repo = cfg.get("name_or_path", "")
        if repo and hf_cached(repo):
            return {"present": True, "where": f"huggingface cache ({repo})", "download_gb": 0, "missing": []}
        if lf and not auth.get("complete", True):
            return {"present": False, "unverified": True, "where": repo, "download_gb": cfg.get("hf_gb", 0), "missing": [repo],
                    "note": (f"could not read ComfyUI's model folders (ComfyUI not reachable, no extra_model_paths.yaml found), "
                             f"so it is not known whether {lf.rsplit('/', 1)[-1]} is already on this machine")}
        return {"present": False, "where": repo, "download_gb": cfg.get("hf_gb", 0), "missing": [repo],
                "note": f"ai-toolkit downloads {repo} (~{cfg.get('hf_gb', '?')} GB) into {hf_home()} on first use"}
    # musubi: explicit files out of the ComfyUI model folders
    found, missing, unverified = {}, [], False
    for role, pattern in (cfg.get("files") or {}).items():
        auth: dict = {}
        hits = _local_model_files(pattern, auth)
        if hits:
            found[role] = hits[0]["path"]
        else:
            missing.append(f"{role} (a file matching '{pattern}')")
            unverified = unverified or not auth.get("complete", True)
    out = {"present": not missing, "where": found, "download_gb": 0, "missing": missing, "files": found}
    if missing and unverified:
        out["unverified"] = True
        out["note"] = ("could not read ComfyUI's model folders (ComfyUI not reachable, no extra_model_paths.yaml found), "
                       "so it is not known whether these files are already on this machine")
    return out


def check(preset: dict, backend: str, output_dir: Path, allow_download: bool = False, vram_override: float = 0.0) -> dict:
    """The full report. ``ok`` is False while ``problems`` is not empty."""
    problems: list[str] = []
    warnings: list[str] = []
    gpu = gpu_memory() if backend != "fal" else None
    need = float(vram_override or preset.get("vram_gb", 0))
    if backend != "fal":
        if gpu is None:
            problems.append("no NVIDIA GPU found (nvidia-smi); local training needs CUDA. Use backend='fal' for hosted training")
        elif gpu["free_gb"] < need:
            msg = (f"{preset['label']} needs about {need:.0f} GB of free GPU memory ({preset.get('vram_note') or 'preset minimum'}); "
                   f"{gpu['free_gb']:.1f} of {gpu['total_gb']:.0f} GB are free on {gpu['name']}.")
            if gpu["total_gb"] >= need:
                msg += " ComfyUI is probably holding models: train_lora frees its VRAM first (free_vram=True), then checks again."
                warnings.append(msg)
            else:
                problems.append(msg + " This card is too small for it; pick a smaller model or a hosted backend.")
        if preset.get("vram_note") and "disagree" in preset["vram_note"]:
            warnings.append(f"VRAM need is uncertain for {preset['label']}: {preset['vram_note']}")
    d = disk_check(output_dir, float(preset.get("disk_gb", 20)), "the training run")
    if d:
        problems.append(d)
    bm = base_model(preset, backend)
    if not bm["present"]:
        if backend == "ai-toolkit":
            dl = disk_check(hf_home(), float(bm.get("download_gb", 0)), "the base model download")
            if dl:
                problems.append(dl)
            if bm.get("unverified"):
                if not allow_download:
                    problems.append(f"cannot tell whether the base model is on this machine: {bm['note']}. Start ComfyUI and "
                                    "try again (no download is claimed or started on a guess), or call again with allow_download=True "
                                    "if the user accepts a download of up to ~" f"{bm.get('download_gb', '?')} GB")
            elif not allow_download:
                problems.append(f"the base model is not on this machine: {bm.get('note') or bm['missing']}. "
                                "Tell the user the size and, if they agree, call again with allow_download=True")
        elif bm.get("unverified"):
            problems.append("cannot tell whether the base model files are on this machine: " + "; ".join(bm["missing"]) +
                            f". {bm['note']}. Start ComfyUI and try again; download nothing on a guess")
        else:
            problems.append("base model files are missing: " + "; ".join(bm["missing"]) +
                            ". Download them with download_hf_model (see the musubi-tuner docs/<model>.md for the repositories)")
    be = B.get(backend) if backend != "fal" else None
    inst = be.locate() if be else {"installed": True}
    if backend != "fal" and not inst.get("installed"):
        need_disk = be.INSTALL_GB
        di = disk_check(inst["root"], need_disk, f"installing {backend}")
        if di:
            problems.append(di)
        warnings.append(f"{backend} is not installed; it will be installed into its own venv at {inst['root']} "
                        f"(~{need_disk:.0f} GB) as the first stage of this job")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "gpu": gpu, "backend_install": inst,
            "base_model": bm, "output_dir": str(output_dir)}
