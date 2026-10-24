"""Hosted LoRA training on fal (queue REST API). Costs money: every start goes through the cost gate.

fal prices trainers per step. The numbers in ``presets.py`` (``usd_per_step``) are *estimates* and can be
overridden with ``FAL_USD_PER_STEP``; the gate shows the estimate, asks above the user's confirmation level
(``partner_credits.confirm_threshold``) and the real spend is written to ``.logs/partner_cost.jsonl`` when the
job reports done (fal does not return the bill, so the logged figure is the estimate and says so).

Not exercised against the live API (no credits were spent building it): the request shapes follow the fal docs
(queue.fal.run, status_url / response_url / cancel_url, rest.alpha.fal.ai storage upload); tests use a stub.
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("agenty_core.lora.fal")

QUEUE = "https://queue.fal.run"
STORAGE = "https://rest.alpha.fal.ai/storage/upload/initiate?storage_type=fal-cdn-v3"

# test seam: (method, url, **kw) -> object with .status_code/.json()/.text/.content
_http: Optional[Callable[..., Any]] = None


def api_key() -> str:
    try:
        from agenty_core.utils.secrets import get_secret
        k = get_secret("FAL_KEY") or get_secret("FAL_API_KEY")
    except Exception:  # noqa: BLE001
        k = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")
    return (k or "").strip()


def _req(method: str, url: str, **kw):
    if _http is not None:
        return _http(method, url, **kw)
    import requests
    return requests.request(method, url, timeout=kw.pop("timeout", 60), **kw)


def _headers() -> dict:
    k = api_key()
    if not k:
        raise RuntimeError("no fal key: set FAL_KEY in the environment or .env")
    return {"Authorization": f"Key {k}", "Content-Type": "application/json"}


def estimate_usd(cfg: dict, steps: int) -> float:
    per = float(os.environ.get("FAL_USD_PER_STEP", "") or cfg.get("usd_per_step", 0.01))
    return round(per * int(steps), 2)


def zip_dataset(trainer_dir: Path) -> bytes:
    """images / clips + their .txt captions, flat, as fal's trainers expect."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(Path(trainer_dir).iterdir()):
            if f.is_file():
                z.write(f, f.name)
    return buf.getvalue()


def upload(data: bytes, file_name: str, content_type: str = "application/zip") -> str:
    r = _req("POST", STORAGE, headers=_headers(), data=json.dumps({"content_type": content_type, "file_name": file_name}))
    if r.status_code >= 300:
        raise RuntimeError(f"fal upload init failed ({r.status_code}): {r.text[:200]}")
    j = r.json()
    p = _req("PUT", j["upload_url"], data=data, headers={"Content-Type": content_type}, timeout=600)
    if p.status_code >= 300:
        raise RuntimeError(f"fal upload failed ({p.status_code})")
    return j["file_url"]


def build_input(cfg: dict, data_url: str, steps: int, lr: Optional[float], trigger: str) -> dict:
    body = {cfg.get("data_field") or "image_data_url": data_url, cfg.get("step_field", "steps"): int(steps)}
    if cfg.get("lr_field"):
        body[cfg["lr_field"]] = float(lr if lr else cfg.get("default_lr", 1e-4))
    if cfg.get("trigger_field") and trigger:
        body[cfg["trigger_field"]] = trigger
    elif cfg.get("endpoint", "").endswith("flux-2-trainer") and trigger:
        body["default_caption"] = trigger
    body.update(cfg.get("extra") or {})
    return body


def submit(cfg: dict, body: dict) -> dict:
    r = _req("POST", f"{QUEUE}/{cfg['endpoint']}", headers=_headers(), data=json.dumps(body))
    if r.status_code >= 300:
        raise RuntimeError(f"fal rejected the job ({r.status_code}): {r.text[:300]}")
    j = r.json()
    return {"request_id": j["request_id"], "status_url": j["status_url"], "response_url": j["response_url"],
            "cancel_url": j.get("cancel_url")}


def poll(status_url: str) -> dict:
    """{status: IN_QUEUE|IN_PROGRESS|COMPLETED, log_tail, queue_position}"""
    r = _req("GET", status_url + ("&" if "?" in status_url else "?") + "logs=1", headers=_headers())
    if r.status_code >= 300:
        raise RuntimeError(f"fal status failed ({r.status_code}): {r.text[:200]}")
    j = r.json()
    logs = "\n".join(str(x.get("message", "")) for x in (j.get("logs") or []))
    return {"status": j.get("status"), "log": logs, "queue_position": j.get("queue_position")}


def result(response_url: str) -> dict:
    r = _req("GET", response_url, headers=_headers())
    if r.status_code >= 300:
        raise RuntimeError(f"fal result failed ({r.status_code}): {r.text[:300]}")
    return r.json()


def cancel(cancel_url: str) -> bool:
    try:
        return _req("PUT", cancel_url, headers=_headers()).status_code < 300
    except Exception:  # noqa: BLE001
        return False


def lora_url(res: dict) -> Optional[str]:
    """The LoRA weights file in a trainer's result (fields differ between trainers)."""
    for k in ("diffusers_lora_file", "lora_file", "high_noise_lora", "low_noise_lora"):
        f = res.get(k)
        if isinstance(f, dict) and f.get("url"):
            return f["url"]
    return None


def download(url: str, dest: Path) -> Path:
    r = _req("GET", url, timeout=600)
    if r.status_code >= 300:
        raise RuntimeError(f"download failed ({r.status_code})")
    dest.write_bytes(r.content)
    return dest


def progress_from_log(log: str, total: int) -> dict:
    """fal trainers log lines like 'step 120/1000 loss 0.12'; pick the last one."""
    import re
    best = {}
    for m in re.finditer(r"(?:step|steps)\D{0,6}(\d+)\s*/\s*(\d+)", log, re.I):
        best = {"step": int(m.group(1)), "total": int(m.group(2))}
    lm = list(re.finditer(r"loss\D{0,4}([0-9.]+)", log, re.I))
    if lm:
        best["loss"] = float(lm[-1].group(1))
    return best
