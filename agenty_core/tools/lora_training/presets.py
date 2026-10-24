"""Per-model training presets: the values the agent should rarely have to tune by hand.

Sources: the ai-toolkit UI defaults (``ui/src/app/jobs/new/options.ts``) and example configs, the musubi-tuner
docs (``docs/<model>.md``), the fal model pages. VRAM figures are what the guides state; they disagree for
the newest models, so ``vram_gb`` is a *minimum to try* and ``vram_note`` says how sure it is. ``preflight``
compares them with the free memory it measures.

A preset is a plain dict so the agent can read it (``list_lora_presets`` is part of ``lora_training_status``'s
sibling output) and a call can override single values (``steps=``, ``rank=``, ``learning_rate=``).
"""
from __future__ import annotations

from typing import Any, Optional

# Common shape of a preset (all keys optional except label / kind / backends):
#   label, kind: image|video, edit: bool (needs control pairs: not supported yet),
#   resolutions: [int,...]            training resolution buckets (long side / square size for ai-toolkit)
#   rank, learning_rate, steps, batch_size, optimizer, dtype, gradient_accumulation
#   vram_gb, vram_note, disk_gb       what to have free (GPU memory; disk for cache + checkpoints)
#   caption_style                     how captions should read for this model's text encoder
#   frames, fps                       video presets: clip length / rate to prepare
#   sample: {width, height, steps, guidance, frames, fps}
#   backends: {"ai-toolkit": {...}, "musubi": {...}, "fal": {...}}
PRESETS: dict[str, dict[str, Any]] = {
    "z_image": {
        "label": "Z-Image Turbo", "kind": "image", "aliases": ["z-image", "zimage", "z image turbo", "z_image_turbo"],
        "resolutions": [512, 768, 1024], "rank": 32, "learning_rate": 1e-4, "steps": 2500, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 16, "vram_note": "ai-toolkit qfloat8; ~24 GB without low_vram",
        "disk_gb": 30, "caption_style": "one natural sentence or two; subject first, then setting and style",
        "sample": {"width": 1024, "height": 1024, "steps": 8, "guidance": 1.0},
        "backends": {
            "ai-toolkit": {"arch": "zimage", "name_or_path": "Tongyi-MAI/Z-Image-Turbo", "quantize": True,
                           "qtype": "qfloat8", "low_vram": False, "noise_scheduler": "flowmatch",
                           "timestep_type": "weighted",
                           "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
                           "hf_gb": 33},
            "musubi": {"arch": "zimage", "script": "zimage", "network_module": "networks.lora_zimage",
                       "extra": ["--timestep_sampling", "shift", "--weighting_scheme", "none",
                                 "--discrete_flow_shift", "2.0"],
                       "files": {"dit": "z_image", "vae": "ae", "text_encoder": "qwen_3"}},
        },
    },
    "flux2_klein_4b": {
        "label": "FLUX.2 klein base 4B", "kind": "image", "aliases": ["flux2 klein", "flux.2 klein", "klein 4b", "flux-2-klein-4b"],
        "resolutions": [512, 768, 1024], "rank": 32, "learning_rate": 1e-4, "steps": 2500, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 16, "vram_note": "ai-toolkit qfloat8", "disk_gb": 30,
        "caption_style": "natural sentences",
        "sample": {"width": 1024, "height": 1024, "steps": 25, "guidance": 4.0},
        "backends": {
            "ai-toolkit": {"arch": "flux2_klein_4b", "name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
                           "quantize": True, "qtype": "qfloat8", "noise_scheduler": "flowmatch",
                           "timestep_type": "weighted", "hf_gb": 16},
            "musubi": {"arch": "flux2", "script": "flux_2", "network_module": "networks.lora_flux_2",
                       "extra": ["--model_version", "klein-base-4b", "--timestep_sampling", "flux2_shift",
                                 "--weighting_scheme", "none"],
                       "files": {"dit": "flux-2-klein", "vae": "flux2", "text_encoder": "qwen_3_4b"}},
        },
    },
    "flux2_klein_9b": {
        "label": "FLUX.2 klein base 9B", "kind": "image", "aliases": ["klein 9b"],
        "resolutions": [512, 768, 1024], "rank": 32, "learning_rate": 1e-4, "steps": 2500, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 24, "vram_note": "ai-toolkit qfloat8", "disk_gb": 45,
        "caption_style": "natural sentences",
        "sample": {"width": 1024, "height": 1024, "steps": 25, "guidance": 4.0},
        "backends": {
            "ai-toolkit": {"arch": "flux2_klein_9b", "name_or_path": "black-forest-labs/FLUX.2-klein-base-9B",
                           "quantize": True, "qtype": "qfloat8", "noise_scheduler": "flowmatch",
                           "timestep_type": "weighted", "hf_gb": 30},
            "musubi": {"arch": "flux2", "script": "flux_2", "network_module": "networks.lora_flux_2",
                       "extra": ["--model_version", "klein-base-9b", "--timestep_sampling", "flux2_shift",
                                 "--weighting_scheme", "none"],
                       "files": {"dit": "flux-2-klein-base-9b", "vae": "flux2", "text_encoder": "qwen_3_8b"}},
        },
    },
    "flux2_dev": {
        "label": "FLUX.2 dev", "kind": "image", "aliases": ["flux2", "flux.2", "flux 2 dev", "flux-2-dev"],
        "resolutions": [512, 768, 1024], "rank": 32, "learning_rate": 1e-4, "steps": 2000, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 48, "vram_note": "24 GB only with low_vram + qfloat8 and a lot of system RAM; a farm card is the comfortable route",
        "disk_gb": 120, "caption_style": "natural sentences",
        "sample": {"width": 1024, "height": 1024, "steps": 25, "guidance": 4.0},
        "backends": {
            "ai-toolkit": {"arch": "flux2", "name_or_path": "black-forest-labs/FLUX.2-dev", "quantize": True,
                           "qtype": "qfloat8", "low_vram": True, "noise_scheduler": "flowmatch",
                           "timestep_type": "weighted", "hf_gb": 110},
            "musubi": {"arch": "flux2", "script": "flux_2", "network_module": "networks.lora_flux_2",
                       "extra": ["--model_version", "dev", "--timestep_sampling", "flux2_shift",
                                 "--weighting_scheme", "none", "--fp8_base", "--fp8_scaled"],
                       "files": {"dit": "flux2_dev", "vae": "flux2", "text_encoder": "mistral"}},
            "fal": {"endpoint": "fal-ai/flux-2-trainer", "usd_per_step": 0.008, "min_images": 10,
                    "step_field": "steps", "lr_field": "learning_rate", "default_lr": 5e-5,
                    "extra": {"output_lora_format": "comfy"}},
        },
    },
    "flux1_dev": {
        "label": "FLUX.1 dev", "kind": "image", "aliases": ["flux", "flux.1", "flux1", "flux dev", "flux-dev"],
        "resolutions": [512, 768, 1024], "rank": 16, "learning_rate": 1e-4, "steps": 2000, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 24, "vram_note": "ai-toolkit quantized", "disk_gb": 60,
        "caption_style": "natural sentences",
        "sample": {"width": 1024, "height": 1024, "steps": 20, "guidance": 4.0},
        "backends": {
            "ai-toolkit": {"arch": "flux", "name_or_path": "black-forest-labs/FLUX.1-dev", "is_flux": True,
                           "quantize": True, "noise_scheduler": "flowmatch", "hf_gb": 34},
            "fal": {"endpoint": "fal-ai/flux-lora-fast-training", "usd_per_step": 0.0035, "min_images": 4,
                    "step_field": "steps", "lr_field": None, "extra": {"is_style": False}, "trigger_field": "trigger_word"},
        },
    },
    "qwen_image": {
        "label": "Qwen-Image", "kind": "image", "aliases": ["qwen image", "qwen-image", "qwen_image_2512", "qwen image 2512"],
        "resolutions": [512, 768, 1024], "rank": 32, "learning_rate": 1e-4, "steps": 3000, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 24, "vram_note": "ai-toolkit qfloat8 + low_vram (slow); 32 GB comfortable",
        "disk_gb": 70, "caption_style": "natural sentences, descriptive",
        "sample": {"width": 1024, "height": 1024, "steps": 25, "guidance": 4.0},
        "backends": {
            "ai-toolkit": {"arch": "qwen_image", "name_or_path": "Qwen/Qwen-Image-2512", "quantize": True,
                           "qtype": "qfloat8", "low_vram": True, "noise_scheduler": "flowmatch",
                           "timestep_type": "weighted", "hf_gb": 57},
            "musubi": {"arch": "qwen_image", "script": "qwen_image", "network_module": "networks.lora_qwen_image",
                       "extra": ["--timestep_sampling", "qwen_shift", "--weighting_scheme", "none",
                                 "--discrete_flow_shift", "2.2", "--fp8_base", "--fp8_scaled"],
                       "files": {"dit": "qwen_image_bf16", "vae": "qwen_image_vae", "text_encoder": "qwen_2.5_vl"}},
        },
    },
    "wan22_t2v_14b": {
        "label": "Wan 2.2 T2V A14B", "kind": "video", "aliases": ["wan 2.2", "wan2.2", "wan22", "wan 2.2 t2v", "wan22_14b"],
        "resolutions": [512], "rank": 32, "learning_rate": 2e-4, "steps": 2000, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 24, "vram_note": "quantized + low_vram + layer offload; 32 GB comfortable",
        "disk_gb": 90, "frames": 33, "fps": 16, "caption_style": "describe subject, action and camera in one paragraph",
        "sample": {"width": 768, "height": 768, "steps": 25, "guidance": 4.0, "frames": 33, "fps": 16},
        "backends": {
            "ai-toolkit": {"arch": "wan22_14b:t2v", "name_or_path": "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
                           "quantize": True, "qtype": "qfloat8", "low_vram": True, "noise_scheduler": "flowmatch",
                           "multistage": True, "hf_gb": 60},
            "musubi": {"arch": "wan", "script": "wan", "network_module": "networks.lora_wan", "task": "t2v-A14B",
                       "extra": ["--timestep_sampling", "shift", "--discrete_flow_shift", "3.0", "--fp8_base"],
                       "files": {"dit": "wan2.2_t2v_high_noise_14B", "dit_low": "wan2.2_t2v_low_noise_14B",
                                 "vae": "wan_2.1_vae", "t5": "umt5"}},
            "fal": {"endpoint": "fal-ai/wan-22-trainer/t2v-a14b", "usd_per_step": 0.01, "min_images": 5,
                    "step_field": "number_of_steps", "lr_field": "learning_rate", "default_lr": 7e-4,
                    "data_field": "training_data_url", "video": True},
        },
    },
    "wan22_5b": {
        "label": "Wan 2.2 TI2V 5B", "kind": "video", "aliases": ["wan 5b", "wan2.2 5b", "wan22_5b"],
        "resolutions": [512, 768], "rank": 32, "learning_rate": 2e-4, "steps": 2000, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 20, "vram_note": "guides: 24 GB", "disk_gb": 40,
        "frames": 33, "fps": 24, "caption_style": "describe subject, action and camera in one paragraph",
        "sample": {"width": 768, "height": 768, "steps": 25, "guidance": 4.0, "frames": 33, "fps": 24},
        "backends": {
            "ai-toolkit": {"arch": "wan22_5b", "name_or_path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "quantize": True,
                           "low_vram": True, "noise_scheduler": "flowmatch", "hf_gb": 30},
        },
    },
    "wan22_image": {
        "label": "Wan 2.2 (images, fal)", "kind": "image", "aliases": ["wan 2.2 image"], "hosted_only": True,
        "resolutions": [1024], "rank": 32, "learning_rate": 7e-4, "steps": 1000, "batch_size": 1, "vram_gb": 0,
        "disk_gb": 1, "caption_style": "natural sentences",
        "backends": {"fal": {"endpoint": "fal-ai/wan-22-image-trainer", "usd_per_step": 0.005, "min_images": 10,
                             "step_field": "steps", "lr_field": "learning_rate", "default_lr": 7e-4,
                             "data_field": "training_data_url"}},
    },
    "ltx2": {
        "label": "LTX-2 / LTX-2.5", "kind": "video", "aliases": ["ltx", "ltx-2", "ltx2.5", "ltx-2.5", "ltx 2.3", "ltx2.3"],
        "resolutions": [512, 768], "rank": 32, "learning_rate": 1e-4, "steps": 2000, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 32,
        "vram_note": "the guides disagree: ~32 GB with INT8 quantisation, 80 GB unquantised; MEASURE on this card",
        "disk_gb": 100, "frames": 49, "fps": 24, "audio": True,
        "caption_style": "describe the scene, the action and the sound in one paragraph",
        "sample": {"width": 768, "height": 768, "steps": 30, "guidance": 4.0, "frames": 121, "fps": 24},
        "backends": {
            "ai-toolkit": {"arch": "ltx2", "name_or_path": "Lightricks/LTX-2", "quantize": True, "quantize_te": True,
                           "low_vram": True, "noise_scheduler": "flowmatch", "timestep_type": "weighted",
                           "do_audio": True, "hf_gb": 80},
        },
        "docs_only": {"ltx-trainer": "Lightricks/LTX-2 packages/ltx-trainer: Linux / WSL2 / farm with CUDA + Triton. "
                                     "Not driven from here; run it on the farm (see the deadline-farm skill) or in WSL2 "
                                     "with the dataset this tool writes (items/ + captions)."},
    },
    "sdxl": {
        "label": "SDXL 1.0", "kind": "image", "aliases": ["sd xl", "stable diffusion xl"],
        "resolutions": [768, 1024], "rank": 32, "learning_rate": 1e-4, "steps": 2000, "batch_size": 1,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 12, "vram_note": "", "disk_gb": 20,
        "caption_style": "short descriptive phrases", "sample": {"width": 1024, "height": 1024, "steps": 25, "guidance": 6.0},
        "backends": {"ai-toolkit": {"arch": "sdxl", "name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
                                    "quantize": False, "noise_scheduler": "ddpm", "local_comfy_file": "sd_xl_base_1.0.safetensors",
                                    "hf_gb": 7}},
    },
    "sd15": {
        "label": "Stable Diffusion 1.5", "kind": "image", "aliases": ["sd 1.5", "sd1.5", "stable diffusion 1.5"],
        "resolutions": [512], "rank": 16, "learning_rate": 1e-4, "steps": 1500, "batch_size": 2,
        "optimizer": "adamw8bit", "dtype": "bf16", "vram_gb": 6, "vram_note": "", "disk_gb": 10,
        "caption_style": "short descriptive phrases", "sample": {"width": 512, "height": 512, "steps": 25, "guidance": 7.0},
        "backends": {"ai-toolkit": {"arch": "sd1", "name_or_path": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                                    "quantize": False, "noise_scheduler": "ddpm",
                                    "local_comfy_file": "SD15/realisticVisionV60B1_v60B1VAE.safetensors", "hf_gb": 4}},
    },
}

BACKENDS = ("ai-toolkit", "musubi", "fal")


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def resolve_model(name: str) -> Optional[str]:
    """A preset key from a key, label or alias ('Wan 2.2', 'z-image', 'flux.2 klein')."""
    n = _norm(name)
    if not n:
        return None
    for key, p in PRESETS.items():
        if n in {_norm(key), _norm(p["label"]), *(_norm(a) for a in p.get("aliases", []))}:
            return key
    for key, p in PRESETS.items():       # a looser pass: the alias is contained in what was said
        if any(_norm(a) and _norm(a) in n for a in (key, *p.get("aliases", []))):
            return key
    return None


def get_preset(model: str) -> dict:
    key = resolve_model(model)
    if key is None:
        raise KeyError(f"no LoRA preset for '{model}'. Known: {', '.join(sorted(PRESETS))}")
    p = dict(PRESETS[key])
    p["key"] = key
    return p


def pick_backend(preset: dict, backend: str = "auto", allow_hosted: bool = False) -> str:
    """The backend to use: the asked one if the preset has it, else the first local one."""
    have = list(preset.get("backends", {}))
    b = str(backend or "auto").lower().replace("_", "-").replace("aitoolkit", "ai-toolkit")
    if b in ("ai-toolkit", "musubi", "fal"):
        if b not in have:
            raise ValueError(f"{preset['label']} has no {b} preset; it has: {', '.join(have)}")
        return b
    for cand in ("ai-toolkit", "musubi"):
        if cand in have:
            return cand
    if "fal" in have and allow_hosted:
        return "fal"
    raise ValueError(f"{preset['label']} can only be trained on a hosted backend (fal); ask for backend='fal'")


def public_summary(preset: dict) -> dict:
    """What the agent sees of a preset."""
    keep = ("key", "label", "kind", "resolutions", "rank", "learning_rate", "steps", "vram_gb", "vram_note",
            "disk_gb", "frames", "fps", "caption_style", "hosted_only", "docs_only")
    out = {k: preset[k] for k in keep if k in preset}
    out["backends"] = list(preset.get("backends", {}))
    return out
