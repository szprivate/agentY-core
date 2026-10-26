"""Generate images and videos with Alibaba DashScope (Model Studio) directly.

Billed to the user's own DashScope key (``DASHSCOPE_API_KEY``) instead of ComfyUI's
partner nodes, which bill Comfy credits. One tool, two kinds:

* ``image`` — Qwen-Image 2.0 and the other messages-style image models answer
  synchronously; the older Wan text-to-image models are an async task.
* ``video`` — always an async task: create it (``X-DashScope-Async: enable``), poll
  ``/tasks/{id}`` until it is done, download the MP4 (its URL lives 24 hours).

Endpoints follow ``DASHSCOPE_BASE_URL`` (the same variable the LLM layer reads, so
a workspace host such as ``ws-….maas.aliyuncs.com`` keeps working); without it the
Singapore/international host is used. Request shapes are in
``docs/integrations/dashscope_generation.md`` of the agentY repo.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from agenty_core._compat import tool
from agenty_core.paths import project_root

_INTL_ROOT = "https://dashscope-intl.aliyuncs.com/api/v1"
_VIDEO_PATH = "/services/aigc/video-generation/video-synthesis"
_IMAGE_SYNC_PATH = "/services/aigc/multimodal-generation/generation"
_IMAGE_ASYNC_PATH = "/services/aigc/text2image/image-synthesis"

_MAX_INPUT_BYTES = 10 * 1024 * 1024  # the smallest per-image limit of the models below
_POLL_FIRST_S = 5.0
_POLL_MAX_S = 15.0
_sleep = time.sleep  # tests replace this

# Model-name families. Confirmed against the international endpoint (see the doc);
# a name outside these prefixes is refused before any request is made.
_VIDEO_PREFIXES = ("wan2.", "happyhorse-")
_IMAGE_SYNC_PREFIXES = ("qwen-image-2.0", "wan2.6-t2i", "wan2.7-image", "z-image")
_IMAGE_ASYNC_PREFIXES = ("wan2.2-t2i", "wan2.5-t2i", "wan2.1-t2i")

# Pixel sizes for the older Wan models, which take ``size`` ("W*H") not resolution+ratio.
_SIZES = {
    ("480P", "16:9"): "832*480", ("480P", "9:16"): "480*832", ("480P", "1:1"): "624*624",
    ("720P", "16:9"): "1280*720", ("720P", "9:16"): "720*1280", ("720P", "1:1"): "960*960",
    ("1080P", "16:9"): "1920*1080", ("1080P", "9:16"): "1080*1920", ("1080P", "1:1"): "1440*1440",
}

# DashScope error codes / phrases → what actually happened.
_REFUSAL = re.compile(r"DataInspectionFailed|IPInfringementSuspect|inappropriate content|"
                      r"data inspection failed", re.I)
_BALANCE = re.compile(r"Arrearage|overdue|good standing|insufficient|balance|"
                      r"FreeTierOnly|AllocationQuota|quota exceeded", re.I)


def _err(message: str, kind: str, **extra) -> str:
    out = {"error": message, "error_kind": kind}
    if kind == "content_refusal":
        out["content_refusal"] = True
    out.update({k: v for k, v in extra.items() if v not in (None, "")})
    return json.dumps(out)


def _api_root() -> str:
    """``<host>/api/v1`` for whichever host ``DASHSCOPE_BASE_URL`` points at."""
    base = (os.environ.get("DASHSCOPE_BASE_URL") or "").strip().rstrip("/")
    m = re.match(r"(https?://[^/]+)", base)
    return f"{m.group(1)}/api/v1" if m else _INTL_ROOT


def _api_key() -> str:
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("ALIBABA_API_KEY") or ""


def _classify(status: int, code: str, message: str) -> tuple[str, str]:
    """(error_kind, readable message) for a failed request or task."""
    text = f"{code} {message}"
    if _REFUSAL.search(text):
        return "content_refusal", f"DashScope refused the content ({code or status}): {message}"
    if status == 401 or code == "InvalidApiKey":
        return "auth", f"DashScope rejected the API key ({code or status}): {message}"
    if _BALANCE.search(text) or status == 402:
        return "balance", f"DashScope account balance or quota problem ({code or status}): {message}"
    if "Model not exist" in message or "model not found" in message.lower():
        return "unknown_model", f"DashScope does not offer this model here: {message}"
    if status == 429 or code.startswith("Throttling"):
        return "rate_limit", f"DashScope rate limit ({code or status}): {message}"
    return "api", f"DashScope error ({code or status}): {message}"


def _response_error(r: requests.Response) -> str:
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        body = {}
    kind, msg = _classify(r.status_code, str(body.get("code") or ""), str(body.get("message") or r.text[:300]))
    return _err(msg, kind)


def _encode_image(ref: str) -> str:
    """URL passes through; a local file becomes a ``data:`` URL (the API accepts both)."""
    ref = str(ref).strip()
    if ref.startswith(("http://", "https://", "data:", "oss://")):
        return ref
    from agenty_core.tools.image_io import resolve_local_image
    path = resolve_local_image(ref)
    if not path:
        raise FileNotFoundError(f"image not found: {ref}")
    size = os.path.getsize(path)
    if size > _MAX_INPUT_BYTES:
        raise ValueError(f"{path} is {size / 1e6:.1f} MB; DashScope takes at most 10 MB per image. "
                         "Downsize it first (image-downsize skill).")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode('ascii')}"


def _out_dir(kind: str, override: str) -> Path:
    if str(override or "").strip():
        d = Path(os.path.expanduser(str(override).strip().strip('"')))
    else:
        base = None
        try:
            from agenty_core.tools.comfyui import get_agent_output_dirs
            base = (json.loads(get_agent_output_dirs()) or {}).get("images" if kind == "image" else "videos")
        except Exception:  # noqa: BLE001
            base = None
        d = Path(base) if base else project_root() / "output" / "dashscope" / ("images" if kind == "image" else "videos")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(url: str, kind: str, model: str, out_dir: str, index: int, meta: dict) -> str:
    """Download *url*, write the ``.agenty.json`` sidecar, publish it to the host."""
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    ext = Path(url.split("?")[0]).suffix.lower() or (".mp4" if kind == "video" else ".png")
    name = f"ds_{re.sub(r'[^A-Za-z0-9.-]', '_', model)}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    path = _out_dir(kind, out_dir) / f"{name}{ext}"
    path.write_bytes(r.content)
    try:  # same shape as agentY's output_tags sidecar
        body = {"role": f"DashScope {model} {kind}", "made_by": "agentY",
                "when": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
        Path(str(path) + ".agenty.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from agenty_core.tools.image_io import _publish
    _publish(str(path))
    return str(path)


def _headers(async_: bool) -> dict:
    h = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    if async_:
        h["X-DashScope-Async"] = "enable"
    return h


def _poll(task_id: str, timeout_s: float) -> dict:
    """Block until the task ends. Returns the task body; raises TimeoutError."""
    deadline = time.monotonic() + timeout_s
    wait = _POLL_FIRST_S
    while True:
        r = requests.get(f"{_api_root()}/tasks/{task_id}", headers=_headers(False), timeout=60)
        if r.status_code == 429:
            pass  # throttled: keep waiting
        elif r.status_code >= 400:
            raise RuntimeError(_response_error(r))
        else:
            body = r.json()
            status = (body.get("output") or {}).get("task_status")
            if status in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"):
                return body
        if time.monotonic() + wait > deadline:
            raise TimeoutError(task_id)
        _sleep(wait)
        wait = min(wait * 1.5, _POLL_MAX_S)


def _task_failed(body: dict, task_id: str) -> str:
    out = body.get("output") or {}
    status = out.get("task_status")
    if status == "UNKNOWN":
        return _err("DashScope no longer knows this task (task ids last 24 hours).", "api", task_id=task_id)
    if status == "CANCELED":
        return _err("The DashScope task was canceled.", "api", task_id=task_id)
    kind, msg = _classify(200, str(out.get("code") or ""), str(out.get("message") or ""))
    return _err(msg, kind, task_id=task_id)


def _video_body(model: str, prompt: str, images: list[str], resolution: str, ratio: str,
                duration: int, negative_prompt: str, seed: int, watermark: bool,
                size: str) -> tuple[Optional[dict], str]:
    """``(body, "")`` or ``(None, error)``. ``-t2v/-i2v/-r2v`` in the name picks the task."""
    new = model.startswith(("wan2.7", "happyhorse-"))  # media[] + resolution/ratio
    inp: dict = {}
    if prompt:
        inp["prompt"] = prompt
    if negative_prompt and not model.startswith("happyhorse-"):
        inp["negative_prompt"] = negative_prompt
    par: dict = {"watermark": bool(watermark)}
    if resolution:
        par["resolution"] = resolution.upper()
    if duration:
        par["duration"] = int(duration)
    if seed >= 0:
        par["seed"] = int(seed)

    if "-t2v" in model:
        if not prompt:
            return None, "t2v needs a prompt"
        if new:
            if ratio:
                par["ratio"] = ratio
        else:
            par.pop("resolution", None)
            par["size"] = size or _SIZES.get(((resolution or "720P").upper(), ratio or "16:9"), "")
            if not par["size"]:
                par.pop("size")
    elif "-i2v" in model:
        if not images:
            return None, "i2v needs one input image (image_paths)"
        if new:
            inp["media"] = [{"type": "first_frame", "url": images[0]}]
        else:
            inp["img_url"] = images[0]
    elif "-r2v" in model:
        if not images or not prompt:
            return None, "r2v needs a prompt and at least one reference image (image_paths); the prompt names them character1, character2, ..."
        if new:
            inp["media"] = [{"type": "reference_image", "url": u} for u in images]
            if ratio:
                par["ratio"] = ratio
        else:
            inp["reference_urls"] = images
            par.pop("resolution", None)
            par["size"] = size or _SIZES.get(((resolution or "720P").upper(), ratio or "16:9"), "")
    else:
        return None, (f"cannot tell the task from {model!r}; use a model name ending -t2v, -i2v or -r2v "
                      "(e.g. wan2.7-t2v, wan2.2-i2v-flash, happyhorse-1.0-r2v)")
    return {"model": model, "input": inp, "parameters": par}, ""


@tool
def dashscope_generate(kind: str, model: str, prompt: str = "", image_paths: Optional[list] = None,
                       size: str = "", resolution: str = "", ratio: str = "", duration: int = 0,
                       negative_prompt: str = "", seed: int = -1, n: int = 1, watermark: bool = False,
                       timeout_s: int = 0, output_dir: str = "") -> str:
    """Generate an image or a video with an Alibaba DashScope (Model Studio) model,
    billed to the user's own DashScope key — not to ComfyUI partner credits.

    Use it when the user asks for a DashScope / Alibaba / Wan / HappyHorse / Qwen-Image
    model by name, or when a ComfyUI partner node fails with "Payment Required" (Comfy
    credits exhausted). The result is saved under ComfyUI's ``agent/images`` or
    ``agent/videos`` and, in agentY, shown in the chat and dropped on the canvas.
    A video takes 1-5 minutes; the call waits for it.

    Args:
        kind: ``"image"`` or ``"video"``.
        model: exact DashScope model id. Image: ``qwen-image-2.0`` (also edits when
            ``image_paths`` is given), ``qwen-image-2.0-pro``, ``wan2.6-t2i``,
            ``wan2.7-image``, ``z-image-turbo``, ``wan2.5-t2i-preview``. Video (the
            suffix picks the task): ``wan2.7-t2v`` / ``wan2.7-i2v`` / ``wan2.7-r2v``,
            ``wan2.6-t2v`` / ``wan2.6-i2v`` / ``wan2.6-i2v-flash`` / ``wan2.6-r2v``,
            ``wan2.2-i2v-flash`` (cheapest, 480P), ``wan2.2-t2v-plus``,
            ``happyhorse-1.0-t2v`` / ``-i2v`` / ``-r2v``.
        prompt: what to generate. For ``-r2v`` name the references character1, character2...
        image_paths: input images — local paths (sent base64) or http(s) URLs. i2v uses
            the first as the first frame; r2v uses all as references; image models
            use them as edit sources.
        size: image size ``"W*H"`` (e.g. ``"1024*1024"``) or, for older Wan video
            models, the video size; empty picks a default.
        resolution: video ``"480P"``, ``"720P"`` or ``"1080P"`` (what a model offers varies).
        ratio: video aspect ratio, e.g. ``"16:9"`` (not for i2v: it follows the image).
        duration: video seconds; 0 leaves it to the model (some models only do 5 s).
        negative_prompt: what to avoid (not HappyHorse).
        seed: -1 = random.
        n: images to make (1-4; the async Wan image models only).
        watermark: add the provider watermark (default off).
        timeout_s: give up waiting after this many seconds (0 = 900 for video, 180 for image).
        output_dir: save here instead of ComfyUI's agent folders.

    Returns:
        JSON ``{"files": [paths], "kind", "model", "task_id", "usage", "request_id"}``
        or ``{"error", "error_kind"}`` where error_kind is one of ``unknown_model``,
        ``balance`` (DashScope account out of money/quota), ``content_refusal``
        (reword or roll again), ``auth``, ``rate_limit``, ``timeout`` (the
        ``task_id`` can be fetched later with ``dashscope_fetch``), ``api``.
        ``usage`` is DashScope's own accounting (seconds / images / size) — what is billed.
    """
    kind = (kind or "").strip().lower()
    model = (model or "").strip()
    if kind not in ("image", "video"):
        return _err("kind must be 'image' or 'video'", "args")
    if not _api_key():
        return _err("DASHSCOPE_API_KEY is not set (add it to .env).", "auth")
    families = _VIDEO_PREFIXES if kind == "video" else _IMAGE_SYNC_PREFIXES + _IMAGE_ASYNC_PREFIXES
    if not model.startswith(families):
        return _err(f"unknown {kind} model {model!r}; expected a name starting with {', '.join(families)}.",
                    "unknown_model")
    try:
        images = [_encode_image(p) for p in (image_paths or [])]
    except (FileNotFoundError, ValueError) as exc:
        return _err(str(exc), "args")

    meta = {"prompt": prompt[:500], "model": model, "provider": "dashscope"}
    try:
        if kind == "video":
            body, bad = _video_body(model, prompt, images, resolution, ratio, duration,
                                    negative_prompt, seed, watermark, size)
            if body is None:
                return _err(bad, "args")
            return _run_task(_VIDEO_PATH, body, "video", model, timeout_s or 900, output_dir, meta)
        if model.startswith(_IMAGE_ASYNC_PREFIXES):
            inp = {"prompt": prompt}
            if negative_prompt:
                inp["negative_prompt"] = negative_prompt
            par = {"n": max(1, min(int(n), 4)), "watermark": bool(watermark)}
            if size:
                par["size"] = size
            if seed >= 0:
                par["seed"] = int(seed)
            return _run_task(_IMAGE_ASYNC_PATH, {"model": model, "input": inp, "parameters": par},
                             "image", model, timeout_s or 180, output_dir, meta)
        return _image_sync(model, prompt, images, size, negative_prompt, seed, n, watermark,
                           timeout_s or 180, output_dir, meta)
    except requests.Timeout:
        return _err("DashScope did not answer in time.", "timeout")
    except requests.RequestException as exc:
        return _err(f"could not reach DashScope: {exc}", "api")


def _run_task(path: str, body: dict, kind: str, model: str, timeout_s: int, out_dir: str, meta: dict) -> str:
    r = requests.post(_api_root() + path, headers=_headers(True), json=body, timeout=120)
    if r.status_code >= 400:
        return _response_error(r)
    task_id = (r.json().get("output") or {}).get("task_id")
    if not task_id:
        return _err(f"DashScope returned no task id: {r.text[:200]}", "api")
    return _finish_task(task_id, kind, model, timeout_s, out_dir, meta)


def _finish_task(task_id: str, kind: str, model: str, timeout_s: int, out_dir: str, meta: dict) -> str:
    try:
        done = _poll(task_id, timeout_s)
    except TimeoutError:
        return _err(f"still running after {timeout_s}s; fetch it later with dashscope_fetch "
                    f"(the task lives 24 h).", "timeout", task_id=task_id)
    except RuntimeError as exc:  # a polling error already rendered as JSON
        return str(exc)
    out = done.get("output") or {}
    if out.get("task_status") != "SUCCEEDED":
        return _task_failed(done, task_id)
    urls = [out["video_url"]] if out.get("video_url") else [
        x.get("url") for x in out.get("results") or [] if isinstance(x, dict) and x.get("url")]
    if not urls:
        return _err(f"task succeeded but returned no file: {json.dumps(out)[:300]}", "api", task_id=task_id)
    meta = {**meta, "task_id": task_id}
    files = [_save(u, kind, model, out_dir, i, meta) for i, u in enumerate(urls)]
    return json.dumps({"files": files, "kind": kind, "model": model, "task_id": task_id,
                       "usage": done.get("usage") or {}, "request_id": done.get("request_id", "")})


def _image_sync(model, prompt, images, size, negative_prompt, seed, n, watermark,
                timeout_s, out_dir, meta) -> str:
    content = [{"image": u} for u in images] + [{"text": prompt}]
    par: dict = {"watermark": bool(watermark), "n": max(1, int(n))}
    if size:
        par["size"] = size
    if negative_prompt:
        par["negative_prompt"] = negative_prompt
    if seed >= 0:
        par["seed"] = int(seed)
    r = requests.post(_api_root() + _IMAGE_SYNC_PATH, headers=_headers(False), timeout=timeout_s,
                      json={"model": model, "input": {"messages": [{"role": "user", "content": content}]},
                            "parameters": par})
    if r.status_code >= 400:
        return _response_error(r)
    body = r.json()
    urls = [c["image"] for ch in (body.get("output") or {}).get("choices") or []
            for c in (ch.get("message") or {}).get("content") or [] if isinstance(c, dict) and c.get("image")]
    if not urls:
        return _err(f"no image in the response: {r.text[:300]}", "api")
    files = [_save(u, "image", model, out_dir, i, meta) for i, u in enumerate(urls)]
    return json.dumps({"files": files, "kind": "image", "model": model, "task_id": "",
                       "usage": body.get("usage") or {}, "request_id": body.get("request_id", "")})


@tool
def dashscope_fetch(task_id: str, kind: str = "video", model: str = "dashscope", output_dir: str = "") -> str:
    """Collect a DashScope generation that ``dashscope_generate`` timed out on.

    Args:
        task_id: the ``task_id`` from the timeout error (valid 24 hours).
        kind: ``"video"`` or ``"image"``.
        model: the model id, only used to name the file.
        output_dir: save here instead of ComfyUI's agent folders.

    Returns:
        The same JSON as ``dashscope_generate``; if the task is still running,
        an ``error_kind: "timeout"`` error again.
    """
    if not _api_key():
        return _err("DASHSCOPE_API_KEY is not set (add it to .env).", "auth")
    try:
        return _finish_task(task_id.strip(), kind, model, 30, output_dir, {"provider": "dashscope", "model": model})
    except requests.RequestException as exc:
        return _err(f"could not reach DashScope: {exc}", "api")
