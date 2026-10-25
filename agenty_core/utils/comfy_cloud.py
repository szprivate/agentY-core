"""A small REST client for Comfy Cloud (https://cloud.comfy.org), authenticated with a Comfy API key.

Comfy Cloud serves the ComfyUI HTTP API (documented at docs.comfy.org/development/cloud/overview and the OpenAPI
file https://docs.comfy.org/openapi-cloud.yaml). Everything agentY needs is plain REST with the key in the
``X-API-Key`` header - no MCP, no websocket (polling is enough):

====================================  ==========================================================================
``POST /api/prompt``                  submit an API-format workflow (``{"prompt": ..., "extra_data": {...}}``)
``GET  /api/job/{id}/status``         state of a job (poll until terminal)
``GET  /api/jobs/{id}``               job detail: ``outputs`` (ComfyUI shape) and ``execution_error``
``GET  /api/view``                    an output file: a 302 to a signed URL (fetched WITHOUT the key)
``POST /api/upload/image``            upload an input image / video / audio (multipart)
``GET  /api/object_info``             every node class the cloud has (to know what it can run)
``GET  /api/experiment/models[/f]``   model folders and the model files in each (what the cloud has)
``GET  /api/user``                    account check (free)
``POST /api/queue``                   cancel a pending job
====================================  ==========================================================================

Every call in :class:`CloudClient` reads the key from ``COMFYUI_API_KEY`` (the same key partner nodes use) and never
puts it in a message, a log line or an exception text. The "has the models / the nodes" answers are cached in
memory and on disk (``.cache/comfy_cloud.json`` under the project root) because the lists are large and change
rarely.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("agenty_core.comfy_cloud")

BASE_URL = "https://cloud.comfy.org"
CACHE_TTL_S = 6 * 3600.0
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")

# Job states, normalised. The docs and the OpenAPI file name them differently; both are accepted.
_QUEUED = {"waiting_to_dispatch", "pending", "submitted", "queued_waiting", "queued"}
_RUNNING = {"in_progress", "preparing", "executing", "running"}
_DONE = {"completed", "success"}
_ERROR = {"error", "failed", "non_retryable_error", "lost"}
_CANCELLED = {"cancelled", "canceled", "cancel_requested"}


class CloudError(RuntimeError):
    """A Comfy Cloud call failed. ``kind``: auth | subscription | credits | invalid | http | network | no_key."""

    def __init__(self, kind: str, message: str, status: int = 0, details: Any = None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.details = details


def normalize_state(raw: str) -> str:
    """queued | running | done | error | cancelled | unknown."""
    s = str(raw or "").strip().lower()
    if s in _QUEUED:
        return "queued"
    if s in _RUNNING:
        return "running"
    if s in _DONE:
        return "done"
    if s in _ERROR:
        return "error"
    if s in _CANCELLED:
        return "cancelled"
    return "unknown"


def api_key() -> str:
    try:
        from agenty_core.utils.secrets import get_secret
        return (get_secret("COMFYUI_API_KEY") or "").strip().strip('"')
    except Exception:  # noqa: BLE001
        return ""


def has_key() -> bool:
    return bool(api_key())


def _cache_file() -> Path:
    from agenty_core.paths import project_root
    return project_root() / ".cache" / "comfy_cloud.json"


class CloudClient:
    """REST client. *http* is any object with ``request(method, url, **kw)`` (``requests`` by default) so tests
    can feed recorded responses."""

    def __init__(self, key: Optional[str] = None, base_url: str = "", http: Any = None,
                 timeout: float = 30.0, cache_path: Optional[Path] = None,
                 clock: Callable[[], float] = time.time):
        self._key = key
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self._http = http
        self.timeout = timeout
        self._cache_path = cache_path
        self._clock = clock
        self._lock = threading.Lock()
        self._mem: dict = {}

    # ── plumbing ────────────────────────────────────────────────────────
    @property
    def key(self) -> str:
        return self._key if self._key is not None else api_key()

    def _session(self):
        if self._http is not None:
            return self._http
        import requests
        return requests

    def _request(self, method: str, path: str, *, auth: bool = True, **kw):
        key = self.key
        if auth and not key:
            raise CloudError("no_key", "No Comfy API key: set COMFYUI_API_KEY (Settings > Authentication).")
        headers = dict(kw.pop("headers", {}) or {})
        if auth:
            headers["X-API-Key"] = key
        url = path if path.startswith("http") else self.base_url + path
        kw.setdefault("timeout", self.timeout)
        try:
            resp = self._session().request(method, url, headers=headers, **kw)
        except Exception as exc:  # noqa: BLE001 - never let the key ride along in a message
            raise CloudError("network", f"Comfy Cloud unreachable ({type(exc).__name__})") from None
        return resp

    @staticmethod
    def _body(resp) -> Any:
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return None

    def _check(self, resp, what: str) -> None:
        code = getattr(resp, "status_code", 200)
        if code < 400:
            return
        body = self._body(resp)
        text = ""
        if isinstance(body, dict):
            err = body.get("error")
            text = (err.get("message") if isinstance(err, dict) else err) or body.get("message") or ""
        text = str(text or getattr(resp, "text", "") or "")[:300]
        if code == 401:
            raise CloudError("auth", "Comfy Cloud refused the API key (401): check COMFYUI_API_KEY.", code, body)
        if code == 402:
            raise CloudError("credits", "Comfy Cloud: insufficient credits (402). Top up at platform.comfy.org.",
                             code, body)
        if code == 429:
            raise CloudError("subscription", "Comfy Cloud: no active subscription for API use (429).", code, body)
        if code in (400, 404, 422):
            raise CloudError("invalid", f"Comfy Cloud rejected {what} ({code}): {text}", code, body)
        raise CloudError("http", f"Comfy Cloud {what} failed ({code}): {text}", code, body)

    def _json(self, method: str, path: str, what: str, **kw) -> Any:
        resp = self._request(method, path, **kw)
        self._check(resp, what)
        return self._body(resp)

    # ── account ─────────────────────────────────────────────────────────
    def user(self) -> dict:
        """Free read-only check that the key works."""
        data = self._json("GET", "/api/user", "the account check")
        return data if isinstance(data, dict) else {}

    # ── running a workflow ──────────────────────────────────────────────
    def submit(self, prompt: dict, *, extra_data: Optional[dict] = None) -> str:
        """Queue an API-format workflow; returns the ``prompt_id``. Partner nodes get the key in ``extra_data``."""
        body = {"prompt": prompt, "extra_data": {**(extra_data or {}), "api_key_comfy_org": self.key}}
        data = self._json("POST", "/api/prompt", "the workflow", json=body)
        if not isinstance(data, dict):
            raise CloudError("http", "Comfy Cloud answered /api/prompt with something unexpected")
        node_errors = data.get("node_errors")
        if data.get("error") or (isinstance(node_errors, dict) and node_errors):
            err = data.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            raise CloudError("invalid", f"Comfy Cloud refused the workflow: {msg or 'node errors'}",
                             400, {"error": err, "node_errors": node_errors})
        pid = data.get("prompt_id")
        if not pid:
            raise CloudError("http", "Comfy Cloud did not return a prompt_id")
        return str(pid)

    def job_status(self, prompt_id: str) -> dict:
        """``{"state": queued|running|done|error|cancelled|unknown, "raw": ..., "error": str|None}``."""
        data = self._json("GET", f"/api/job/{prompt_id}/status", "the status check") or {}
        raw = data.get("status", "") if isinstance(data, dict) else ""
        return {"state": normalize_state(raw), "raw": raw,
                "error": (data.get("error_message") if isinstance(data, dict) else None)}

    def job_detail(self, prompt_id: str) -> dict:
        data = self._json("GET", f"/api/jobs/{prompt_id}", "the job details")
        return data if isinstance(data, dict) else {}

    def cancel(self, prompt_id: str) -> None:
        try:
            self._json("POST", "/api/queue", "the cancel", json={"delete": [prompt_id]})
            self._json("POST", "/api/interrupt", "the interrupt", json={})
        except CloudError:
            logger.debug("could not cancel cloud job %s", prompt_id)

    # ── files ───────────────────────────────────────────────────────────
    def download(self, filename: str, dest: Path, *, subfolder: str = "", kind: str = "output") -> Path:
        """Save one output. /api/view answers 302 to a signed storage URL; that is fetched without the key."""
        params = {"filename": filename, "subfolder": subfolder, "type": kind}
        resp = self._request("GET", "/api/view", params=params, allow_redirects=False)
        code = getattr(resp, "status_code", 200)
        if code in (301, 302, 303, 307, 308):
            loc = (getattr(resp, "headers", {}) or {}).get("Location") or (getattr(resp, "headers", {}) or {}).get("location")
            if not loc:
                raise CloudError("http", "Comfy Cloud redirected /api/view without a location")
            resp = self._request("GET", loc, auth=False)
        self._check(resp, "the download")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    def upload(self, path: Path, *, kind: str = "input") -> str:
        """Upload an input file (image, video, audio); returns the name the workflow must use."""
        path = Path(path)
        with path.open("rb") as fh:
            data = self._json("POST", "/api/upload/image", "the upload",
                              files={"image": (path.name, fh)}, data={"type": kind, "overwrite": "true"},
                              timeout=max(self.timeout, 300.0))
        name = (data or {}).get("name") if isinstance(data, dict) else None
        if not name:
            raise CloudError("http", "Comfy Cloud did not return a name for the upload")
        sub = (data or {}).get("subfolder") or ""
        return f"{sub}/{name}" if sub else str(name)

    # ── what the cloud has (cached) ─────────────────────────────────────
    def _cache_get(self, name: str) -> Any:
        with self._lock:
            hit = self._mem.get(name)
        if hit and self._clock() - hit["at"] < CACHE_TTL_S:
            return hit["value"]
        try:
            p = self._cache_path or _cache_file()
            disk = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
            hit = disk.get(name)
            if hit and self._clock() - float(hit.get("at", 0)) < CACHE_TTL_S:
                with self._lock:
                    self._mem[name] = hit
                return hit["value"]
        except Exception:  # noqa: BLE001
            pass
        return None

    def _cache_put(self, name: str, value: Any) -> None:
        entry = {"at": self._clock(), "value": value}
        with self._lock:
            self._mem[name] = entry
        try:
            p = self._cache_path or _cache_file()
            disk = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
            disk[name] = entry
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(disk), encoding="utf-8")
        except Exception:  # noqa: BLE001 - a cache that cannot be written is only slower
            pass

    def nodes(self, refresh: bool = False) -> dict:
        """``{class_type: {"api_node": bool, "display_name": str, "price_badge": dict|None}}`` of every node
        class the cloud has."""
        if not refresh:
            cached = self._cache_get("nodes")
            if cached is not None:
                return cached
        info = self._json("GET", "/api/object_info", "the node list", timeout=max(self.timeout, 120.0))
        out: dict = {}
        for cls, entry in (info or {}).items():
            if not isinstance(entry, dict):
                continue
            mod = str(entry.get("python_module") or "")
            out[cls] = {"api_node": bool(entry.get("api_node")) or mod.startswith("comfy_api_nodes"),
                        "display_name": entry.get("display_name") or cls,
                        "price_badge": entry.get("price_badge") if isinstance(entry.get("price_badge"), dict) else None,
                        "python_module": mod}
            if out[cls]["price_badge"]:
                out[cls]["input"] = entry.get("input")      # the estimator reads widget defaults from it
        self._cache_put("nodes", out)
        return out

    def node_info(self, class_type: str) -> Optional[dict]:
        """A schema-ish entry (``price_badge`` included) for partner_credits' estimator, or None."""
        n = self.nodes().get(class_type)
        return dict(n) if n else None

    def model_names(self, refresh: bool = False) -> dict:
        """``{lowercase basename: [folder, ...]}`` of every model file the cloud offers."""
        if not refresh:
            cached = self._cache_get("models")
            if cached is not None:
                return cached
        folders = self._json("GET", "/api/experiment/models", "the model folders") or []
        out: dict = {}
        for f in folders:
            name = f.get("name") if isinstance(f, dict) else str(f)
            if not name:
                continue
            try:
                files = self._json("GET", f"/api/experiment/models/{name}", f"the {name} models") or []
            except CloudError as exc:
                if exc.kind in ("auth", "credits", "subscription", "no_key", "network"):
                    raise
                continue
            for item in files:
                fn = item.get("name") if isinstance(item, dict) else str(item)
                if fn:
                    out.setdefault(Path(str(fn).replace("\\", "/")).name.lower(), []).append(name)
        self._cache_put("models", out)
        return out

    def has_model(self, filename: str) -> bool:
        return Path(str(filename).replace("\\", "/")).name.lower() in self.model_names()

    def has_node(self, class_type: str) -> bool:
        return class_type in self.nodes()


_client: Optional[CloudClient] = None


def get_client() -> CloudClient:
    global _client
    if _client is None:
        _client = CloudClient()
    return _client


def set_client(c: Optional[CloudClient]) -> None:
    """Tests and hosts swap the client."""
    global _client
    _client = c
