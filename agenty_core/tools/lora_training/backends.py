"""The local training backends: ai-toolkit (ostris) and musubi-tuner (kohya-ss), each in its own venv.

A backend turns a job (dataset + preset + overrides) into *stages* - plain subprocess commands the runner
executes one after another (install on first use, cache latents / text encoder outputs, train) - and reads the
progress, samples and checkpoints back out of what the trainer wrote. Nothing here starts a process.

Where a backend lives (first hit wins): the environment variable (``AITOOLKIT_PATH`` / ``MUSUBI_PATH``), an
existing checkout beside the app (``<app>/../training/ai-toolkit``), else ``<app>/tools/lora/<name>`` - installed on
the job's first stage, in a venv of its own, like Brush. Never on C:.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from agenty_core.paths import project_root

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"     # Blackwell (RTX 50xx) needs CUDA 12.8 builds
TORCH_PIN = ["torch==2.9.1", "torchvision==0.24.1", "torchaudio==2.9.1"]

_TQDM = re.compile(r"(\d+)/(\d+)\s*\[([\d:]+)<([\d:?]+),\s*([\d.]+)\s*(it/s|s/it)(.*?)\]")
_LOSS = re.compile(r"(?:avr_loss|loss)[=:]\s*([0-9.]+(?:e[+-]?\d+)?)", re.I)
_LR = re.compile(r"\blr[=:]\s*([0-9.]+(?:e[+-]?\d+)?)", re.I)


def tools_root() -> Path:
    d = Path(os.environ.get("AGENTY_LORA_HOME", "") or (project_root() / "tools" / "lora"))
    return d


def _secs(t: str) -> Optional[int]:
    if not t or "?" in t:
        return None
    parts = [int(p) for p in t.split(":")]
    s = 0
    for p in parts:
        s = s * 60 + p
    return s


def parse_tqdm(line: str, state: dict) -> bool:
    """Update *state* (step, total, it_s, eta_s, loss, lr) from one log line; True when the line carried progress."""
    m = _TQDM.search(line)
    if not m:
        return False
    step, total = int(m.group(1)), int(m.group(2))
    if total <= 1:
        return False
    rate = float(m.group(5))
    it_s = rate if m.group(6) == "it/s" else (1.0 / rate if rate else 0.0)
    state.update(step=step, total=total, it_s=round(it_s, 4), eta_s=_secs(m.group(4)))
    rest = m.group(7) or ""
    lm, rm = _LOSS.search(rest), _LR.search(rest)
    if lm:
        state["loss"] = float(lm.group(1))
    if rm:
        state["lr"] = float(rm.group(1))
    return True


def _venv_python(root: Path) -> Path:
    for sub in (".venv", "venv"):
        for exe in ("Scripts/python.exe", "bin/python"):
            p = root / sub / exe
            if p.exists():
                return p
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _find_python() -> str:
    """A python 3.10-3.12 to make the venv with (ai-toolkit and musubi are not tested on 3.13)."""
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) <= (3, 12):
        return sys.executable
    for cand in ("python3.12", "python3.11", "python3.10"):
        p = shutil.which(cand)
        if p:
            return p
    return sys.executable


class Backend:
    NAME = ""
    INSTALL_GB = 10.0
    REPO = ""
    ENV = ""
    SIBLING = ""

    def root(self) -> Path:
        env = os.environ.get(self.ENV, "").strip().strip('"')
        if env:
            return Path(env)
        sib = project_root().parent / "training" / self.SIBLING
        if (sib / self.MARKER).exists():
            return sib
        return tools_root() / self.NAME

    MARKER = ""

    def locate(self) -> dict:
        r = self.root()
        py = _venv_python(r)
        return {"root": str(r), "python": str(py), "installed": (r / self.MARKER).exists() and py.exists()}

    def install_stages(self) -> list[dict]:
        r = self.root()
        py = _venv_python(r)
        stages = []
        if not (r / self.MARKER).exists():
            stages.append({"name": f"install {self.NAME}: clone", "kind": "setup",
                           "cmd": ["git", "clone", "--depth", "1", self.REPO, str(r)], "cwd": str(r.parent)})
        if not py.exists():
            stages.append({"name": f"install {self.NAME}: venv", "kind": "setup",
                           "cmd": [_find_python(), "-m", "venv", str(r / ".venv")], "cwd": str(r)})
            py = _venv_python(r)
        stages.append({"name": f"install {self.NAME}: torch", "kind": "setup",
                       "cmd": [str(py), "-m", "pip", "install", "--no-cache-dir", *TORCH_PIN, "--index-url", TORCH_INDEX],
                       "cwd": str(r)})
        stages.append(self._pip_stage(py, r))
        return stages

    def _pip_stage(self, py: Path, r: Path) -> dict:
        raise NotImplementedError

    def stages(self, ctx: dict) -> list[dict]:
        raise NotImplementedError

    def parse(self, line: str, state: dict) -> bool:
        return parse_tqdm(line, state)

    def collect(self, run_dir: Path, job: dict) -> dict:
        raise NotImplementedError


def _sample_step(name: str) -> Optional[int]:
    return None


class AIToolkit(Backend):
    NAME = "ai-toolkit"
    INSTALL_GB = 12.0
    REPO = "https://github.com/ostris/ai-toolkit"
    ENV = "AITOOLKIT_PATH"
    SIBLING = "ai-toolkit"
    MARKER = "run.py"

    def _pip_stage(self, py, r):
        return {"name": "install ai-toolkit: requirements", "kind": "setup",
                "cmd": [str(py), "-m", "pip", "install", "--no-cache-dir", "-r", str(r / "requirements.txt")],
                "cwd": str(r)}

    # -- config ------------------------------------------------------------
    def config(self, ctx: dict) -> dict:
        p, cfg, job = ctx["preset"], ctx["cfg"], ctx["job"]
        out = Path(ctx["run_dir"]) / "out"
        res = job.get("resolutions") or p.get("resolutions") or [1024]
        video = p["kind"] == "video"
        base = ctx.get("base_model") or {}
        name_or_path = base.get("local_file") or base.get("local_dir") or cfg["name_or_path"]
        model = {"name_or_path": name_or_path, "arch": cfg["arch"].split(":")[0]}
        if cfg.get("arch", "").startswith("wan22_14b"):
            model["arch"] = "wan22_14b:t2v" if ":t2v" in cfg["arch"] else "wan22_14b"
        if cfg.get("is_flux"):
            model["is_flux"] = True
        classic = cfg["arch"] in ("sd1", "sdxl", "flux")        # the older sd_trainer, no 'arch' key
        if classic:
            model.pop("arch", None)
            if cfg["arch"] == "sdxl":
                model["is_xl"] = True
        for k in ("quantize", "qtype", "low_vram", "quantize_te", "multistage"):
            if k in cfg:
                model[k] = cfg[k]
        if cfg.get("assistant_lora_path"):
            model["assistant_lora_path"] = cfg["assistant_lora_path"]
        if job.get("low_vram") is not None:
            model["low_vram"] = bool(job["low_vram"])
        ds = {"folder_path": ctx["dataset_dir"], "caption_ext": "txt", "caption_dropout_rate": 0.05,
              "cache_latents_to_disk": True, "resolution": list(res)}
        if video:
            ds.update(num_frames=int(job.get("frames") or p.get("frames", 33)), fps=int(job.get("fps") or p.get("fps", 16)))
            if cfg.get("do_audio") or p.get("audio"):
                ds.update(do_audio=True)
        sample = dict(p.get("sample") or {})
        sample_cfg = {"sampler": "flowmatch" if cfg.get("noise_scheduler") == "flowmatch" else "ddpm",
                      "sample_every": int(job["sample_every"]), "width": sample.get("width", 1024),
                      "height": sample.get("height", 1024), "prompts": list(ctx["sample_prompts"]), "neg": "",
                      "seed": int(job.get("seed", 42)), "walk_seed": False,
                      "guidance_scale": sample.get("guidance", 4.0), "sample_steps": sample.get("steps", 25)}
        if video:
            sample_cfg.update(num_frames=sample.get("frames", p.get("frames", 33)), fps=sample.get("fps", p.get("fps", 16)))
        train = {"batch_size": int(job.get("batch_size") or p.get("batch_size", 1)),
                 "steps": int(job["steps"]), "gradient_accumulation_steps": 1, "train_unet": True,
                 "train_text_encoder": False, "gradient_checkpointing": True,
                 "noise_scheduler": cfg.get("noise_scheduler", "flowmatch"),
                 "optimizer": job.get("optimizer") or p.get("optimizer", "adamw8bit"), "lr": float(job["learning_rate"]),
                 "ema_config": {"use_ema": True, "ema_decay": 0.99}, "dtype": p.get("dtype", "bf16"),
                 "skip_first_sample": bool(job.get("skip_first_sample", False))}
        if cfg.get("timestep_type"):
            train["timestep_type"] = cfg["timestep_type"]
        if job.get("disable_sampling"):
            train["disable_sampling"] = True
        proc = {"type": "sd_trainer" if classic else "diffusion_trainer",
                "training_folder": str(out), "device": "cuda:0",
                "trigger_word": ctx["manifest"].get("trigger_word", ""),
                "network": {"type": "lora", "linear": int(job["rank"]), "linear_alpha": int(job["rank"])},
                "save": {"dtype": "float16", "save_every": int(job["save_every"]), "max_step_saves_to_keep": 6,
                         "push_to_hub": False},
                "datasets": [ds], "train": train, "model": model, "sample": sample_cfg}
        return {"job": "extension", "config": {"name": job["name"], "process": [proc]},
                "meta": {"name": "[name]", "version": "1.0"}}

    def stages(self, ctx: dict) -> list[dict]:
        loc = self.locate()
        stages = [] if loc["installed"] else self.install_stages()
        run = Path(ctx["run_dir"])
        cfgp = run / "ai-toolkit.yaml"
        cfgp.write_text(json.dumps(self.config(ctx), indent=1), encoding="utf-8")   # JSON is YAML
        env = {"HF_HOME": str(ctx.get("hf_home") or ""), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "DISABLE_TELEMETRY": "YES", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
        env = {k: v for k, v in env.items() if v}
        py = _venv_python(Path(loc["root"]))
        stages.append({"name": "train (ai-toolkit)", "kind": "train", "cmd": [str(py), "run.py", str(cfgp)],
                       "cwd": loc["root"], "env": env, "parse": True})
        return stages

    def collect(self, run_dir: Path, job: dict) -> dict:
        d = Path(run_dir) / "out" / job["name"]
        samples = []
        for f in sorted((d / "samples").glob("*")) if (d / "samples").exists() else []:
            m = re.search(r"__(\d{6,9})_(\d+)\.(?:jpg|jpeg|png|webp|mp4|webm)$", f.name)
            if m:
                samples.append({"step": int(m.group(1)), "idx": int(m.group(2)), "path": str(f)})
            elif f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm"):
                samples.append({"step": None, "idx": 0, "path": str(f)})
        ck = sorted(p for p in d.glob("*.safetensors")) if d.exists() else []
        final = d / f"{job['name']}.safetensors"
        steps = [(int(re.search(r"_(\d{6,9})\.safetensors$", p.name).group(1)), p) for p in ck
                 if re.search(r"_(\d{6,9})\.safetensors$", p.name)]
        return {"samples": samples, "checkpoints": [{"step": s, "path": str(p)} for s, p in sorted(steps)],
                "final": str(final) if final.exists() else None}


class Musubi(Backend):
    NAME = "musubi-tuner"
    INSTALL_GB = 8.0
    REPO = "https://github.com/kohya-ss/musubi-tuner"
    ENV = "MUSUBI_PATH"
    SIBLING = "musubi-tuner"
    MARKER = "pyproject.toml"

    def _pip_stage(self, py, r):
        return {"name": "install musubi-tuner: package", "kind": "setup",
                "cmd": [str(py), "-m", "pip", "install", "--no-cache-dir", "-e", str(r)], "cwd": str(r)}

    def dataset_toml(self, ctx: dict) -> str:
        p, job = ctx["preset"], ctx["job"]
        res = int(max(job.get("resolutions") or p.get("resolutions") or [1024]))
        run = Path(ctx["run_dir"])
        lines = ["[general]", f"resolution = [{res}, {res}]", 'caption_extension = ".txt"',
                 f"batch_size = {int(job.get('batch_size') or p.get('batch_size', 1))}", "enable_bucket = true",
                 "bucket_no_upscale = false", "", "[[datasets]]"]
        cache = (run / "cache").as_posix()
        if p["kind"] == "video":
            nf = int(job.get("frames") or p.get("frames", 33))
            lines += [f'video_directory = "{Path(ctx["dataset_dir"]).as_posix()}"', f'cache_directory = "{cache}"',
                      f"target_frames = [{nf}]", 'frame_extraction = "head"']
        else:
            lines += [f'image_directory = "{Path(ctx["dataset_dir"]).as_posix()}"', f'cache_directory = "{cache}"']
        return "\n".join(lines) + "\n"

    def stages(self, ctx: dict) -> list[dict]:
        p, cfg, job = ctx["preset"], ctx["cfg"], ctx["job"]
        loc = self.locate()
        stages = [] if loc["installed"] else self.install_stages()
        run = Path(ctx["run_dir"])
        root = Path(loc["root"])
        py = str(_venv_python(root))
        files = (ctx.get("base_model") or {}).get("files") or {}
        toml = run / "dataset.toml"
        toml.write_text(self.dataset_toml(ctx), encoding="utf-8")
        prompts = run / "sample_prompts.txt"
        sw = (p.get("sample") or {})
        prompts.write_text("\n".join(
            f"{t} --w {sw.get('width', 1024)} --h {sw.get('height', 1024)} --d {int(job.get('seed', 42))} --s {sw.get('steps', 25)}"
            + (f" --f {sw.get('frames', 33)}" if p["kind"] == "video" else "")
            for t in ctx["sample_prompts"]) + "\n", encoding="utf-8")
        s = cfg["script"]
        sd = root / "src" / "musubi_tuner"
        env = {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "HF_HOME": str(ctx.get("hf_home") or "")}
        env = {k: v for k, v in env.items() if v}
        common_vae = ["--vae", files.get("vae", "")]
        cache_lat = [py, str(sd / f"{s}_cache_latents.py"), "--dataset_config", str(toml), *common_vae]
        cache_te = [py, str(sd / f"{s}_cache_text_encoder_outputs.py"), "--dataset_config", str(toml), "--batch_size", "4"]
        if s == "wan":
            cache_te += ["--t5", files.get("t5", "")]
        else:
            cache_te += ["--text_encoder", files.get("text_encoder", "")]
        stages.append({"name": "cache latents", "kind": "cache", "cmd": cache_lat, "cwd": str(root), "env": env})
        stages.append({"name": "cache text encoder outputs", "kind": "cache", "cmd": cache_te, "cwd": str(root), "env": env})
        out = run / "out"
        train = [py, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
                 str(sd / f"{s}_train_network.py")]
        if s == "wan":
            train += ["--task", cfg.get("task", "t2v-A14B"), "--dit", files.get("dit", ""), "--t5", files.get("t5", "")]
        else:
            train += ["--dit", files.get("dit", ""), "--text_encoder", files.get("text_encoder", "")]
        train += ["--vae", files.get("vae", ""), "--dataset_config", str(toml), "--sdpa", "--mixed_precision", "bf16",
                  "--optimizer_type", job.get("optimizer") or "adamw8bit", "--learning_rate", str(job["learning_rate"]),
                  "--gradient_checkpointing", "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
                  "--network_module", cfg["network_module"], "--network_dim", str(int(job["rank"])),
                  "--max_train_steps", str(int(job["steps"])), "--save_every_n_steps", str(int(job["save_every"])),
                  "--seed", str(int(job.get("seed", 42))), "--output_dir", str(out), "--output_name", job["name"],
                  "--sample_prompts", str(prompts), "--sample_every_n_steps", str(int(job["sample_every"])),
                  "--sample_at_first", *cfg.get("extra", [])]
        stages.append({"name": "train (musubi-tuner)", "kind": "train", "cmd": train, "cwd": str(root), "env": env,
                       "parse": True})
        return stages

    def collect(self, run_dir: Path, job: dict) -> dict:
        out = Path(run_dir) / "out"
        samples = []
        sd = out / "sample"
        for f in sorted(sd.glob("*")) if sd.exists() else []:
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".gif"):
                continue
            m = re.search(r"_(\d{5,7})_(\d{2})_\d{8,}", f.name) or re.search(r"_(\d{5,7})_(\d{2})\.", f.name)
            samples.append({"step": int(m.group(1)) if m else None, "idx": int(m.group(2)) if m else 0, "path": str(f)})
        ck = []
        for p in sorted(out.glob("*.safetensors")) if out.exists() else []:
            m = re.search(r"-(\d{6})\.safetensors$", p.name)
            if m:
                ck.append({"step": int(m.group(1)), "path": str(p)})
        final = out / f"{job['name']}.safetensors"
        return {"samples": samples, "checkpoints": ck, "final": str(final) if final.exists() else None}


_BACKENDS = {"ai-toolkit": AIToolkit(), "musubi": Musubi()}


def get(name: str) -> Backend:
    if name not in _BACKENDS:
        raise KeyError(f"no local backend '{name}'")
    return _BACKENDS[name]
