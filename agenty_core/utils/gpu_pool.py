"""A pool of rented GPU machines that render ComfyUI workflows like a render farm - the "Farm" of the viewer.

Machines come from a provider (vast.ai, RunPod) or are added by URL (any ComfyUI you can reach, a box in the
office, a pod rented by hand). Each one runs a plain ComfyUI; the pool talks to it with ComfyUI's own API
(``/object_info``, ``/upload/image``, ``/prompt``, ``/history``, ``/view``), so nothing has to be installed on it
beyond the machine image.

* **Machines**: :func:`offers` lists what a provider has for rent, :func:`rent` / :func:`start` / :func:`stop` /
  :func:`terminate` drive it, :func:`add_url` / :func:`remove` the by-URL ones. The provider keys are secrets
  (``VAST_API_KEY``, ``RUNPOD_API_KEY``, through :mod:`agenty_core.utils.secrets`) and are never part of
  :func:`state`.
* **Jobs**: :func:`submit` takes an API-format workflow and a batch size and makes one task per render, each with its
  own seed (``seed_mode`` fixed / increment / random). Tasks go to the least busy ready machine, one at a time per
  machine; a task that fails is tried once more on another machine. The outputs are downloaded to
  ``<dest>/<job id>/`` and handed to the :data:`listeners` (agentY puts them into the viewer's History).
* **Money**: every running provider machine adds its $/h to today's spend; at ``daily_cap_usd`` every provider
  machine is stopped and nothing new is rented or started that day. A ready machine with nothing to do for
  ``idle_stop_minutes`` is stopped. ``max_usd_per_hour`` refuses dearer offers.
* **Parity**: :func:`check_parity` compares a workflow's node types and model files against a machine's
  ``/object_info`` and ``/models/<folder>`` before anything is sent to it.

State (machines, jobs, settings, spend) is kept in ``.cache/gpu_pool.json`` under the project root. A background
thread (:func:`ensure_running`) calls :func:`tick` every few seconds; tests call :func:`tick` themselves with a fake
transport (:func:`set_transport`).

Not verified against live provider accounts: the vast.ai and RunPod request shapes follow their public API docs.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import logging
import random
import re
import threading
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote, urlparse

logger = logging.getLogger("agenty_core.gpu_pool")

VAST_API = "https://console.vast.ai/api/v0"
RUNPOD_REST = "https://rest.runpod.io/v1"
RUNPOD_GRAPHQL = "https://api.runpod.io/graphql"
PROVIDERS = ("vast", "runpod", "url")
KEY_NAMES = {"vast": "VAST_API_KEY", "runpod": "RUNPOD_API_KEY"}
# Optional .env overrides of the API addresses (agentY's Settings shows them, filled with the defaults above).
URL_NAMES = {"vast": ("VAST_API_URL", VAST_API), "runpod": ("RUNPOD_API_URL", RUNPOD_REST)}
SEED_MODES = ("increment", "random", "fixed")
SEED_KEYS = ("seed", "noise_seed")
LOADER_KEYS = ("image", "video", "audio", "file", "mask")
OUTPUT_KINDS = ("images", "gifs", "videos", "audio", "3d", "files")
MODEL_FOLDERS = {
    "ckpt_name": "checkpoints", "unet_name": "diffusion_models", "lora_name": "loras", "vae_name": "vae",
    "clip_name": "text_encoders", "clip_name1": "text_encoders", "clip_name2": "text_encoders",
    "clip_name3": "text_encoders", "control_net_name": "controlnet", "model_name": "upscale_models",
    "style_model_name": "style_models", "clip_vision_name": "clip_vision",
}
EU = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU",
      "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "NO", "CH", "IS", "GB", "UK"}
DEFAULT_SETTINGS = {
    "daily_cap_usd": 20.0,
    "idle_stop_minutes": 10,
    "eu_only": True,
    "max_usd_per_hour": 1.5,
    # the machine image: ComfyUI + the agentY farm helper (agentY deploy/farm-image, built by its Farm image workflow)
    "image": "ghcr.io/szprivate/agenty-farm:latest",
    "disk_gb": 80,
    "gpu": "RTX 5090",
    "retries": 1,
}
SETTING_TYPES = {"daily_cap_usd": float, "idle_stop_minutes": int, "eu_only": bool, "max_usd_per_hour": float,
                 "image": str, "disk_gb": int, "gpu": str, "retries": int}
TICK_SECONDS = 3.0
PROVIDER_POLL_SECONDS = 15.0
GONE_GRACE_SECONDS = 120.0
MAX_JOBS_KEPT = 50
MAX_BATCH = 256

listeners: list = []          # callables (job: dict, task: dict, files: list[str]) after a task's outputs are in

_lock = threading.RLock()
_state: Optional[dict] = None
_state_path: Optional[Path] = None
_transport: Optional[Callable] = None
_secret: Optional[Callable] = None
_thread: Optional[threading.Thread] = None
_last_provider_poll = 0.0


class PoolError(Exception):
    """A refusal worded for the user (no key, over the cap, offer too dear, a machine that does not answer)."""


# ── plumbing ─────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status: int, body: bytes = b"", headers: Optional[dict] = None):
        self.status_code, self.content, self.headers = status, body, headers or {}

    def json(self):
        return json.loads(self.content.decode("utf-8") or "null")


def _requests_transport(method, url, *, headers=None, json_body=None, data=None, files=None, timeout=30):
    import requests
    r = requests.request(method, url, headers=headers, json=json_body, data=data, files=files, timeout=timeout)
    return _Resp(r.status_code, r.content, dict(r.headers))


def set_transport(fn: Optional[Callable]) -> None:
    """Replace the HTTP transport (tests). ``fn(method, url, headers=, json_body=, data=, files=, timeout=)``
    returns an object with ``status_code``, ``content`` and ``json()``."""
    global _transport
    _transport = fn


def set_secret_reader(fn: Optional[Callable]) -> None:
    global _secret
    _secret = fn


def set_state_path(path: Optional[Path]) -> None:
    """Point the pool at another state file (tests); drops what is in memory."""
    global _state, _state_path
    with _lock:
        _state, _state_path = None, Path(path) if path else None


def _http(method, url, **kw):
    return (_transport or _requests_transport)(method, url, **kw)


def _api_url(provider: str) -> str:
    """The provider's API address: the .env override (``VAST_API_URL`` / ``RUNPOD_API_URL``) or the default."""
    name, default = URL_NAMES[provider]
    if _secret:
        v = _secret(name) or ""
    else:
        from agenty_core.utils.secrets import get_secret
        v = get_secret(name, "")
    return (v.strip() or default).rstrip("/")


def _key(provider: str) -> str:
    name = KEY_NAMES.get(provider)
    if not name:
        return ""
    if _secret:
        return _secret(name) or ""
    from agenty_core.utils.secrets import get_secret
    return get_secret(name, "")


def _path() -> Path:
    if _state_path:
        return _state_path
    from agenty_core.paths import project_root
    return project_root() / ".cache" / "gpu_pool.json"


def _load() -> dict:
    global _state
    if _state is None:
        try:
            _state = json.loads(_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _state = {}
        _state.setdefault("machines", {})
        _state.setdefault("jobs", {})
        _state.setdefault("spend", {})
        _state["settings"] = {**DEFAULT_SETTINGS, **(_state.get("settings") or {})}
    return _state


def _save() -> None:
    s = _load()
    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
        tmp.replace(p)
    except OSError as exc:      # the pool keeps working from memory
        logger.warning("gpu_pool: could not save state: %s", exc)


def _today() -> str:
    return date.today().isoformat()


def _now() -> float:
    return time.time()


# ── providers ────────────────────────────────────────────────────────────────
def _norm_gpu(name: str) -> str:
    return " ".join(str(name or "").lower().replace("_", " ").split())


def gpu_matches(pattern: str, name: str) -> bool:
    """Does a GPU name fit what was typed? Empty matches all; ``*`` and ``?`` are wildcards
    (``RTX 50*``, ``*5090*``, ``*PRO 6000*``); without one it is a case-insensitive part of the name."""
    p, n = _norm_gpu(pattern), _norm_gpu(name)
    if not p:
        return True
    if "*" in p or "?" in p:
        return fnmatch.fnmatchcase(n, p)
    return p in n or p.replace("rtx ", "") in n


def _country(geo: str) -> str:
    m = re.search(r"\b([A-Z]{2})\s*$", str(geo or ""))
    return m.group(1) if m else ""


class VastClient:
    """vast.ai's REST API (Bearer key)."""

    def __init__(self, key: str):
        self.h = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        self.base = _api_url("vast")

    def _call(self, method, path, body=None):
        r = _http(method, self.base + path, headers=self.h, json_body=body, timeout=30)
        if r.status_code >= 400:
            raise PoolError(f"vast.ai answered {r.status_code}: {r.content[:200].decode('utf-8', 'replace')}")
        return r.json() or {}

    def offers(self, gpu: str, eu_only: bool, limit: int = 30) -> list:
        # vast.ai only filters by an exact GPU name, so the name (with or without wildcards) is matched here,
        # over a wide page of offers; the EU filter is applied here too, so both come before the limit.
        q = {"rentable": {"eq": True}, "verified": {"eq": True}, "num_gpus": {"eq": 1},
             "order": [["dph_total", "asc"]], "type": "on-demand", "limit": 1000}
        out = []
        for o in self._call("POST", "/bundles/", q).get("offers") or []:
            cc = _country(o.get("geolocation"))
            if eu_only and cc not in EU:
                continue
            if not gpu_matches(gpu, o.get("gpu_name") or ""):
                continue
            out.append({"provider": "vast", "offer_id": str(o.get("id")), "gpu": o.get("gpu_name") or "",
                        "vram_gb": round((o.get("gpu_ram") or 0) / 1024, 1), "usd_per_hour": float(o.get("dph_total") or 0),
                        "region": o.get("geolocation") or "", "country": cc,
                        "reliability": round(float(o.get("reliability2") or o.get("reliability") or 0), 3),
                        "down_mbps": int(o.get("inet_down") or 0), "disk_gb": int(o.get("disk_space") or 0)})
            if len(out) >= limit:
                break
        return out

    def rent(self, offer_id: str, image: str, disk_gb: int, token: str) -> str:
        body = {"client_id": "me", "image": image, "disk": disk_gb, "runtype": "args",
                "env": {"-p 8188:8188": "1", "COMFY_TOKEN": token}, "label": "agentY farm"}
        r = self._call("PUT", f"/asks/{quote(str(offer_id))}/", body)
        if not r.get("success", True) or not r.get("new_contract"):
            raise PoolError(f"vast.ai did not rent the offer: {r.get('msg') or r}")
        return str(r["new_contract"])

    def instances(self) -> dict:
        out = {}
        for i in self._call("GET", "/instances/").get("instances") or []:
            port = ((i.get("ports") or {}).get("8188/tcp") or [{}])[0].get("HostPort")
            ip = i.get("public_ipaddr") or ""
            status = str(i.get("actual_status") or i.get("cur_state") or "")
            out[str(i.get("id"))] = {"status": {"running": "running", "exited": "stopped", "stopped": "stopped"}
                                     .get(status, "starting" if status in ("loading", "created", "") else status),
                                     "url": f"http://{ip}:{port}" if ip and port else "",
                                     "usd_per_hour": float(i.get("dph_total") or 0)}
        return out

    def set_running(self, pid: str, running: bool) -> None:
        self._call("PUT", f"/instances/{quote(pid)}/", {"state": "running" if running else "stopped"})

    def destroy(self, pid: str) -> None:
        self._call("DELETE", f"/instances/{quote(pid)}/")


class RunPodClient:
    """RunPod: prices from GraphQL, pods through the REST API."""

    def __init__(self, key: str):
        self.h = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
        self.base = _api_url("runpod")

    def _rest(self, method, path, body=None):
        r = _http(method, self.base + path, headers=self.h, json_body=body, timeout=30)
        if r.status_code >= 400:
            raise PoolError(f"RunPod answered {r.status_code}: {r.content[:200].decode('utf-8', 'replace')}")
        return r.json() if r.content else {}

    def offers(self, gpu: str, eu_only: bool, limit: int = 30) -> list:
        q = ("query { gpuTypes { id displayName memoryInGb securePrice communityPrice "
             "lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice } } }")
        r = _http("POST", RUNPOD_GRAPHQL, headers=self.h, json_body={"query": q}, timeout=30)
        if r.status_code >= 400:
            raise PoolError(f"RunPod answered {r.status_code}")
        out = []
        for g in ((r.json() or {}).get("data") or {}).get("gpuTypes") or []:
            name = g.get("displayName") or g.get("id") or ""
            if not gpu_matches(gpu, name):
                continue
            price = (g.get("lowestPrice") or {}).get("uninterruptablePrice") or g.get("securePrice") or 0
            if not price:
                continue
            out.append({"provider": "runpod", "offer_id": g.get("id"), "gpu": name, "vram_gb": g.get("memoryInGb") or 0,
                        "usd_per_hour": float(price), "region": "EU data centres" if eu_only else "any data centre",
                        "country": "EU" if eu_only else "", "reliability": 0.99, "down_mbps": 0, "disk_gb": 0})
        return sorted(out, key=lambda o: o["usd_per_hour"])[:limit]

    def rent(self, offer_id: str, image: str, disk_gb: int, token: str, eu_only: bool = True) -> str:
        body = {"name": "agentY farm", "imageName": image, "gpuTypeIds": [offer_id], "gpuCount": 1,
                "containerDiskInGb": disk_gb, "ports": ["8188/http"], "env": {"COMFY_TOKEN": token},
                "cloudType": "SECURE"}
        if eu_only:
            body["dataCenterIds"] = ["EU-RO-1", "EU-SE-1", "EU-NL-1", "EU-CZ-1", "EU-FR-1"]
        r = self._rest("POST", "/pods", body)
        if not r.get("id"):
            raise PoolError(f"RunPod did not create the pod: {r}")
        return str(r["id"])

    def instances(self) -> dict:
        r = self._rest("GET", "/pods")
        pods = r if isinstance(r, list) else r.get("pods") or []
        out = {}
        for p in pods:
            status = str(p.get("desiredStatus") or "").upper()
            out[str(p.get("id"))] = {"status": {"RUNNING": "running", "EXITED": "stopped", "TERMINATED": "gone"}
                                     .get(status, "starting"),
                                     "url": f"https://{p.get('id')}-8188.proxy.runpod.net",
                                     "usd_per_hour": float(p.get("costPerHr") or 0)}
        return out

    def set_running(self, pid: str, running: bool) -> None:
        self._rest("POST", f"/pods/{quote(pid)}/{'start' if running else 'stop'}")

    def destroy(self, pid: str) -> None:
        self._rest("DELETE", f"/pods/{quote(pid)}")


def _client(provider: str):
    key = _key(provider)
    if not key:
        raise PoolError(f"No {KEY_NAMES.get(provider, provider)} is set. Put it in agentY's .env "
                        "(Farm > Setup says how); the viewer never holds it.")
    return VastClient(key) if provider == "vast" else RunPodClient(key)


# ── ComfyUI on a machine ─────────────────────────────────────────────────────
def _mh(m: dict) -> dict:
    return {"Authorization": f"Bearer {m['token']}"} if m.get("token") else {}


def _comfy(m: dict, method: str, path: str, **kw):
    if not m.get("url"):
        raise PoolError(f"{m.get('name')} has no address yet")
    r = _http(method, m["url"].rstrip("/") + path, headers=_mh(m), timeout=kw.pop("timeout", 30), **kw)
    if r.status_code >= 400:
        raise PoolError(f"{m.get('name')}: ComfyUI answered {r.status_code} on {path}: "
                        f"{r.content[:300].decode('utf-8', 'replace')}")
    return r


def _probe(m: dict) -> bool:
    try:
        stats = _comfy(m, "GET", "/system_stats", timeout=8).json() or {}
        dev = ((stats.get("devices") or [{}])[0]) or {}
        if dev.get("name"):
            m["gpu_seen"] = str(dev["name"]).split(":")[-1].strip()[:60]
        m["comfy_version"] = (stats.get("system") or {}).get("comfyui_version") or ""
        return True
    except Exception:  # noqa: BLE001
        return False


def _queue_len(m: dict) -> int:
    try:
        q = _comfy(m, "GET", "/queue", timeout=8).json() or {}
        return len(q.get("queue_running") or []) + len(q.get("queue_pending") or [])
    except Exception:  # noqa: BLE001
        return 0


def check_parity(machine_id: str, workflow: dict) -> dict:
    """What *workflow* needs that the machine lacks: ``{ok, missing_nodes, missing_models}``."""
    with _lock:
        m = _machine(machine_id)
    info = _comfy(m, "GET", "/object_info", timeout=60).json() or {}
    types = {str(n.get("class_type")) for n in (workflow or {}).values() if isinstance(n, dict)}
    missing_nodes = sorted(t for t in types if t and t not in info)
    missing_models = [f"{f}/{n}" for f, n in _missing_models(m, workflow)]
    res = {"ok": not missing_nodes and not missing_models, "missing_nodes": missing_nodes,
           "missing_models": missing_models, "checked": _now()}
    with _lock:
        m["parity"] = res
        _save()
    return res


def _wanted_models(workflow: dict) -> dict:
    """``{folder: {name, ...}}`` of the model files a workflow's loaders name."""
    wanted: dict = {}
    for n in (workflow or {}).values():
        for k, v in ((n or {}).get("inputs") or {}).items() if isinstance(n, dict) else ():
            if k in MODEL_FOLDERS and isinstance(v, str) and v:
                wanted.setdefault(MODEL_FOLDERS[k], set()).add(v.replace("\\", "/"))
    return wanted


def _missing_models(m: dict, workflow: dict) -> list:
    """``[(folder, name), ...]`` the workflow needs and the machine does not list."""
    out = []
    for folder, names in sorted(_wanted_models(workflow).items()):
        try:
            have = {str(x).replace("\\", "/") for x in (_comfy(m, "GET", f"/models/{folder}").json() or [])}
        except PoolError:
            have = set()
        out += [(folder, n) for n in sorted(names) if n not in have]
    return out


# ── machines ─────────────────────────────────────────────────────────────────
def _machine(mid: str) -> dict:
    m = _load()["machines"].get(mid)
    if not m:
        raise PoolError(f"No machine {mid!r} in the farm")
    return m


def _new_machine(provider: str, name: str, **kw) -> dict:
    mid = "m" + uuid.uuid4().hex[:8]
    m = {"id": mid, "provider": provider, "name": name, "provider_id": "", "url": "", "token": "",
         "state": "starting", "usd_per_hour": 0.0, "gpu": "", "region": "", "created": _now(),
         "running_since": None, "idle_since": _now(), "cost_usd": 0.0, "task": None, "done": 0, "failed": 0,
         "error": "", "parity": None, "uploaded": {}}
    m.update(kw)
    _load()["machines"][mid] = m
    return m


def spent_today() -> float:
    return round(float(_load()["spend"].get(_today(), 0.0)), 4)


def _cap_left() -> float:
    return float(_load()["settings"]["daily_cap_usd"]) - spent_today()


def offers(provider: str, gpu: Optional[str] = None) -> list:
    """Offers for one GPU (default: the settings' ``gpu``), cheapest first, EU only unless switched off; offers dearer
    than ``max_usd_per_hour`` are marked ``too_dear``."""
    if provider not in ("vast", "runpod"):
        raise PoolError(f"Unknown provider {provider!r}")
    with _lock:
        st = dict(_load()["settings"])
    rows = _client(provider).offers(st["gpu"] if gpu is None else gpu, bool(st["eu_only"]))
    for o in rows:
        o["too_dear"] = o["usd_per_hour"] > float(st["max_usd_per_hour"])
    return rows


def rent(provider: str, offer_id: str, *, usd_per_hour: float = 0.0, gpu: str = "", region: str = "",
         name: str = "") -> dict:
    """Rent one offer. Refused without a machine image, over the daily cap or above ``max_usd_per_hour``."""
    with _lock:
        st = dict(_load()["settings"])
        if not st["image"]:
            raise PoolError("Set the machine image first (Farm > Setup): the Docker image with ComfyUI and your "
                            "custom nodes that every machine starts from.")
        if _cap_left() <= 0:
            raise PoolError(f"Today's spend has reached the ${st['daily_cap_usd']:.2f} cap; nothing new is rented "
                            "until tomorrow or until the cap is raised.")
        if usd_per_hour and usd_per_hour > float(st["max_usd_per_hour"]):
            raise PoolError(f"${usd_per_hour:.2f}/h is above the ${st['max_usd_per_hour']:.2f}/h limit in Setup.")
    token = uuid.uuid4().hex
    c = _client(provider)
    pid = c.rent(offer_id, st["image"], int(st["disk_gb"]), token, **({"eu_only": bool(st["eu_only"])}
                                                                       if provider == "runpod" else {}))
    with _lock:
        n = sum(1 for x in _load()["machines"].values() if x["provider"] == provider) + 1
        m = _new_machine(provider, name or f"{provider}-{n}", provider_id=pid, token=token,
                         usd_per_hour=float(usd_per_hour or 0), gpu=gpu, region=region, running_since=_now())
        _save()
        return _public_machine(m)


def add_url(url: str, name: str = "", token: str = "", usd_per_hour: float = 0.0) -> dict:
    """Add a ComfyUI that is already running somewhere (no provider: the pool cannot start or stop it)."""
    u = urlparse(str(url or "").strip())
    if u.scheme not in ("http", "https") or not u.netloc:
        raise PoolError("The address must be http(s)://host:port of a ComfyUI")
    with _lock:
        m = _new_machine("url", name or u.netloc, url=f"{u.scheme}://{u.netloc}{u.path.rstrip('/')}",
                         token=str(token or ""), usd_per_hour=float(usd_per_hour or 0), running_since=_now())
        _save()
        return _public_machine(m)


def start(mid: str) -> dict:
    with _lock:
        m = _machine(mid)
        if m["provider"] == "url":
            raise PoolError("A machine added by address is started where it runs")
        if _cap_left() <= 0:
            raise PoolError("Today's spend has reached the cap; raise it in Setup to start machines again.")
    _client(m["provider"]).set_running(m["provider_id"], True)
    with _lock:
        m.update(state="starting", error="", running_since=_now(), idle_since=_now())
        _save()
        return _public_machine(m)


def stop(mid: str, reason: str = "") -> dict:
    with _lock:
        m = _machine(mid)
    if m["provider"] != "url":
        _client(m["provider"]).set_running(m["provider_id"], False)
    with _lock:
        _requeue(m)
        m.update(state="stopped", running_since=None, error=reason or "")
        _save()
        return _public_machine(m)


def terminate(mid: str) -> None:
    """Destroy a rented machine (its disk too) and take it out of the farm."""
    with _lock:
        m = _machine(mid)
    if m["provider"] != "url" and m.get("provider_id"):
        _client(m["provider"]).destroy(m["provider_id"])
    with _lock:
        _requeue(m)
        _load()["machines"].pop(mid, None)
        _save()


def remove(mid: str) -> None:
    """Take a machine added by address out of the farm."""
    with _lock:
        m = _machine(mid)
        if m["provider"] != "url":
            raise PoolError("A rented machine is terminated, not removed (it would go on costing money)")
        _requeue(m)
        _load()["machines"].pop(mid, None)
        _save()


def set_settings(patch: dict) -> dict:
    with _lock:
        st = _load()["settings"]
        for k, v in (patch or {}).items():
            if k not in SETTING_TYPES:
                continue
            t = SETTING_TYPES[k]
            try:
                v = (v in (True, "true", "1", 1, "on")) if t is bool else t(v)
            except (TypeError, ValueError):
                raise PoolError(f"{k}: not a {t.__name__}") from None
            if t in (int, float) and v < 0:
                raise PoolError(f"{k} cannot be negative")
            st[k] = v.strip() if isinstance(v, str) else v
        _save()
        return dict(st)


# ── jobs ─────────────────────────────────────────────────────────────────────
def _seeded(workflow: dict, mode: str, i: int, base_rng: random.Random) -> tuple:
    wf = copy.deepcopy(workflow)
    seed = None
    for n in wf.values():
        inputs = (n or {}).get("inputs") if isinstance(n, dict) else None
        for k in SEED_KEYS:
            v = (inputs or {}).get(k)
            if isinstance(v, int) and not isinstance(v, bool):
                new = v if mode == "fixed" else (v + i if mode == "increment" else base_rng.randrange(2 ** 48))
                inputs[k] = new
                seed = new if seed is None else seed
    return wf, seed


def submit(workflow: dict, *, name: str = "", batch: int = 1, seed_mode: str = "increment",
           machines: Optional[list] = None, dest: Optional[str] = None,
           model_urls: Optional[dict] = None) -> dict:
    """Queue *workflow* (API format) ``batch`` times. ``machines``: ids to use (default all). ``model_urls``:
    ``{"folder/name": url}`` the workflow brought along (ComfyUI templates carry one per loader), used when a
    machine has to download a model."""
    if not isinstance(workflow, dict) or not workflow:
        raise PoolError("No workflow to render")
    if not any(isinstance(n, dict) and n.get("class_type") for n in workflow.values()):
        raise PoolError("That is not an API-format workflow (node id -> {class_type, inputs})")
    batch = max(1, min(MAX_BATCH, int(batch or 1)))
    if seed_mode not in SEED_MODES:
        seed_mode = "increment"
    rng = random.Random()
    tasks = []
    for i in range(batch):
        wf, seed = _seeded(workflow, seed_mode, i, rng)
        tasks.append({"i": i, "seed": seed, "state": "queued", "machine": None, "prompt_id": None, "tries": 0,
                      "tried": [], "outputs": [], "error": "", "started": None, "finished": None, "workflow": wf})
    with _lock:
        s = _load()
        jid = "j" + uuid.uuid4().hex[:8]
        s["jobs"][jid] = {"id": jid, "name": name or "Farm render", "created": _now(), "batch": batch,
                          "seed_mode": seed_mode, "machines": list(machines or []), "dest": dest or "",
                          "model_urls": {str(k): str(v) for k, v in (model_urls or {}).items()
                                         if isinstance(v, str) and v.startswith("https://")},
                          "state": "queued", "tasks": tasks}
        _trim_jobs()
        _save()
        return _public_job(s["jobs"][jid])


def cancel(jid: str) -> dict:
    with _lock:
        j = _load()["jobs"].get(jid)
        if not j:
            raise PoolError(f"No job {jid!r}")
        for t in j["tasks"]:
            if t["state"] in ("queued", "running", "uploading", "models"):
                m = _load()["machines"].get(t.get("machine") or "")
                if t["state"] == "running" and m and t.get("prompt_id"):
                    try:
                        _comfy(m, "POST", "/interrupt", timeout=8)
                        _comfy(m, "POST", "/queue", json_body={"delete": [t["prompt_id"]]}, timeout=8)
                    except Exception:  # noqa: BLE001
                        pass
                if m and m.get("task") == [jid, t["i"]]:
                    m["task"] = None
                    m["idle_since"] = _now()
                t["state"] = "cancelled"
        j["state"] = "cancelled"
        _save()
        return _public_job(j)


def retry(jid: str) -> dict:
    """Queue the failed and cancelled renders of a job again."""
    with _lock:
        j = _load()["jobs"].get(jid)
        if not j:
            raise PoolError(f"No job {jid!r}")
        for t in j["tasks"]:
            if t["state"] in ("failed", "cancelled"):
                t.update(state="queued", machine=None, prompt_id=None, tries=0, tried=[], error="")
        j["state"] = "queued"
        _save()
        return _public_job(j)


def remove_job(jid: str) -> None:
    with _lock:
        j = _load()["jobs"].get(jid)
        if j and j["state"] in ("queued", "running"):
            raise PoolError("Cancel the job first")
        _load()["jobs"].pop(jid, None)
        _save()


def _trim_jobs() -> None:
    jobs = _load()["jobs"]
    done = sorted((j for j in jobs.values() if j["state"] in ("done", "failed", "cancelled")),
                  key=lambda j: j["created"])
    while len(jobs) > MAX_JOBS_KEPT and done:
        jobs.pop(done.pop(0)["id"], None)


def _requeue(m: dict) -> None:
    """A machine goes away: its running task goes back to the queue."""
    if not m.get("task"):
        return
    jid, i = m["task"]
    j = _load()["jobs"].get(jid)
    if j and i < len(j["tasks"]) and j["tasks"][i]["state"] in ("running", "uploading", "models"):
        j["tasks"][i].update(state="queued", machine=None, prompt_id=None)
    m["task"] = None


def _input_dir() -> Optional[Path]:
    try:
        from agenty_core.utils.cloud_run import _input_dir as d
        return d()
    except Exception:  # noqa: BLE001
        return None


def _upload_inputs(m: dict, wf: dict, find_input=None) -> None:
    find = find_input or (lambda name: (lambda p: p if p.is_file() else None)(
        Path(name) if Path(name).is_absolute() else (_input_dir() or Path(".")) / name))
    for n in wf.values():
        if not isinstance(n, dict) or not str(n.get("class_type", "")).startswith(("Load", "VHS_Load")):
            continue
        inputs = n.get("inputs") or {}
        for k in LOADER_KEYS:
            v = inputs.get(k)
            if not isinstance(v, str) or not v:
                continue
            path = find(v)
            if path is None:
                continue
            sig = f"{path}|{path.stat().st_mtime_ns}|{path.stat().st_size}"
            if m["uploaded"].get(v) != sig:
                with open(path, "rb") as fh:
                    r = _comfy(m, "POST", "/upload/image", files={"image": (path.name, fh.read())},
                               data={"overwrite": "true", "subfolder": "farm"}, timeout=300)
                got = r.json() or {}
                m["uploaded"][v] = sig
                m["uploaded"]["@" + v] = "/".join(x for x in (got.get("subfolder"), got.get("name")) if x)
            inputs[k] = m["uploaded"].get("@" + v) or v


def _dest(j: dict) -> Path:
    if j.get("dest"):
        return Path(j["dest"]) / j["id"]
    from agenty_core.paths import project_root
    return project_root() / "output_images" / "farm" / j["id"]


def _download(m: dict, j: dict, t: dict, outputs: dict) -> list:
    d = _dest(j)
    d.mkdir(parents=True, exist_ok=True)
    files = []
    for node_out in (outputs or {}).values():
        for kind in OUTPUT_KINDS:
            for f in (node_out or {}).get(kind) or []:
                if not isinstance(f, dict) or f.get("type", "output") != "output" or not f.get("filename"):
                    continue
                q = (f"/view?filename={quote(f['filename'])}&subfolder={quote(f.get('subfolder') or '')}"
                     f"&type=output")
                body = _comfy(m, "GET", q, timeout=600).content
                safe = re.sub(r"[^\w.\-]+", "_", Path(f["filename"]).name)
                p = d / f"{t['i']:03d}_{safe}"
                p.write_bytes(body)
                files.append(str(p))
    return files


def _pick_machine(j: dict, t: dict) -> Optional[dict]:
    cands = [m for m in _load()["machines"].values()
             if m["state"] == "ready" and not m.get("task") and (not j["machines"] or m["id"] in j["machines"])
             and m["id"] not in t["tried"]]
    if not cands and t["tried"]:     # the retry may go back to one it tried when nothing else is there
        cands = [m for m in _load()["machines"].values()
                 if m["state"] == "ready" and not m.get("task") and (not j["machines"] or m["id"] in j["machines"])]
    return min(cands, key=lambda m: (m["done"] + m["failed"], m["usd_per_hour"])) if cands else None


def _start_task(m: dict, j: dict, t: dict) -> None:
    m["task"] = [j["id"], t["i"]]
    _live_wf[m["id"]] = {k: (v or {}).get("class_type", "") for k, v in t["workflow"].items() if isinstance(v, dict)}
    t.update(state="uploading", machine=m["id"], started=_now(), error="")
    t["tries"] += 1
    t["tried"].append(m["id"])
    m["state_note"] = f"{j['name']} #{t['i'] + 1}"
    try:
        need = _missing_models(m, t["workflow"])
        if need:
            _request_models(m, j, t, need)
            return
    except Exception as exc:  # noqa: BLE001
        _task_failed(m, j, t, str(exc))
        return
    _submit(m, j, t)


def _request_models(m: dict, j: dict, t: dict, need: list) -> None:
    """Ask the machine (its ``agenty_farm_helper`` node) to download the models it lacks; the task waits in
    state ``models`` until they are there (:func:`_poll_models`)."""
    from agenty_core.utils import farm_models
    items, unknown = [], []
    for folder, name in need:
        url = farm_models.resolve(folder, name, j.get("model_urls"))
        if url:
            items.append({"folder": folder, "name": name, "url": url})
        else:
            unknown.append(f"{folder}/{name}")
    if unknown:
        raise PoolError(f"{m['name']} lacks {', '.join(unknown)} and no download address is known for "
                        f"{'it' if len(unknown) == 1 else 'them'}: add one to config/farm_models.json")
    try:
        _comfy(m, "POST", "/farm/fetch", json_body={"items": items}, timeout=30)
    except PoolError as exc:
        if "answered 404" in str(exc) or "answered 405" in str(exc):
            raise PoolError(f"{m['name']} lacks {', '.join(f'{f}/{n}' for f, n in need)} and its machine image "
                            "cannot download models: use the agentY farm image (Setup > Machine image)") from None
        raise
    t.update(state="models", fetch=[f"{i['folder']}/{i['name']}" for i in items])
    logger.info("gpu_pool: %s downloads %s", m["name"], ", ".join(t["fetch"]))


def _poll_models(m: dict, j: dict, t: dict) -> None:
    try:
        st = _comfy(m, "GET", "/farm/fetch", timeout=20).json() or {}
    except Exception as exc:  # noqa: BLE001
        if not _probe(m):
            m["state"] = "unreachable"
            _task_failed(m, j, t, f"lost the machine: {exc}")
        return
    have = {f"{i.get('folder')}/{i.get('name')}": i for i in st.get("items") or [] if isinstance(i, dict)}
    mine = [(k, have.get(k)) for k in t.get("fetch") or []]
    for k, i in mine:
        if i is None:
            _task_failed(m, j, t, f"{m['name']} forgot the download of {k} (restarted?)")
            return
        if i.get("state") == "failed":
            _task_failed(m, j, t, f"could not download {k}: {i.get('error') or 'failed'}")
            return
    if all(i.get("state") == "done" for _, i in mine):
        t["fetch_note"] = ""
        _submit(m, j, t)
        return
    got = sum(int(i.get("bytes") or 0) for _, i in mine)
    total = sum(int(i.get("total") or 0) for _, i in mine)
    left = sum(1 for _, i in mine if i.get("state") != "done")
    pct = f"{100 * got / total:.0f}% · " if total else ""
    t["fetch_note"] = f"downloading {left} model{'s' if left != 1 else ''}: {pct}{got / 1e9:.1f}" + \
                      (f" of {total / 1e9:.1f} GB" if total else " GB")
    m["state_note"] = f"{j['name']} #{t['i'] + 1} ({t['fetch_note']})"


def _submit(m: dict, j: dict, t: dict) -> None:
    t["state"] = "uploading"
    m["state_note"] = f"{j['name']} #{t['i'] + 1}"
    try:
        wf = copy.deepcopy(t["workflow"])
        _upload_inputs(m, wf)
        _ensure_watcher(m)
        r = _comfy(m, "POST", "/prompt", json_body={"prompt": wf, "client_id": _client_id()}, timeout=60).json() or {}
        if r.get("node_errors"):
            raise PoolError(f"the workflow was refused: {json.dumps(r['node_errors'])[:300]}")
        t["prompt_id"] = r.get("prompt_id")
        t["state"] = "running"
    except Exception as exc:  # noqa: BLE001
        _task_failed(m, j, t, str(exc))


def _task_failed(m: dict, j: dict, t: dict, err: str) -> None:
    m["task"] = None
    m["failed"] += 1
    m["idle_since"] = _now()
    retries = int(_load()["settings"].get("retries", 1))
    if t["tries"] <= retries:
        t.update(state="queued", machine=None, prompt_id=None, error=err)
    else:
        t.update(state="failed", error=err, finished=_now())


def _poll_task(m: dict, j: dict, t: dict) -> None:
    try:
        h = _comfy(m, "GET", f"/history/{quote(str(t['prompt_id']))}", timeout=20).json() or {}
    except Exception as exc:  # noqa: BLE001
        if not _probe(m):
            m["state"] = "unreachable"
            _task_failed(m, j, t, f"lost the machine: {exc}")
        return
    entry = h.get(t["prompt_id"])
    if not entry:
        return
    status = entry.get("status") or {}
    if status.get("status_str") == "error":
        msg = next((d[1].get("exception_message") for d in status.get("messages") or []
                    if isinstance(d, list) and d and d[0] == "execution_error"), "") or "the render failed"
        _task_failed(m, j, t, str(msg)[:500])
        return
    if not status.get("completed", True) and status.get("status_str") != "success":
        return
    try:
        files = _download(m, j, t, entry.get("outputs") or {})
    except Exception as exc:  # noqa: BLE001
        _task_failed(m, j, t, f"could not fetch the outputs: {exc}")
        return
    t.update(state="done", outputs=files, finished=_now())
    m["task"] = None
    m["done"] += 1
    m["idle_since"] = _now()
    for fn in list(listeners):
        try:
            fn(_public_job(j), {k: v for k, v in t.items() if k != "workflow"}, files)
        except Exception:  # noqa: BLE001
            logger.exception("gpu_pool listener failed")


def _job_state(j: dict) -> str:
    states = {t["state"] for t in j["tasks"]}
    if j["state"] == "cancelled":
        return "cancelled"
    if states & {"queued", "running", "uploading", "models"}:
        return "running" if states & {"running", "uploading", "models", "done", "failed"} else "queued"
    return "failed" if "failed" in states and "done" not in states else "done"


# ── live progress: each busy machine's /ws feed ──────────────────────────────
# ComfyUI tells the client that queued a prompt how it goes: "executing" (which node), "progress" (step of steps in
# a sampler), "execution_cached" (nodes skipped), and binary preview frames while a sampler runs (only when that
# ComfyUI has previews on, e.g. --preview-method auto). The pool queues with its own client id and listens on
# /ws?clientId=<that id>, one thread per machine with a render, and keeps the latest of each per machine in _live
# (its own lock, so a slow feed never holds the pool). state() hands it out with the machine and the task.
PREVIEW_MAX_AGE = 30.0
PREVIEW_MAX_BYTES = 400_000
WS_IDLE_SECONDS = 60.0
_live: dict = {}
_live_lock = threading.Lock()
_watchers: dict = {}
_live_wf: dict = {}                          # machine id -> {node id: class_type} of the render it runs (not saved)
_ws_connect: Optional[Callable] = None        # tests: fn(url, headers) -> object with recv(timeout) and close()


def set_ws_connect(fn: Optional[Callable]) -> None:
    global _ws_connect
    _ws_connect = fn


def _client_id() -> str:
    s = _load()
    if not s.get("client_id"):
        s["client_id"] = "agentY-farm-" + uuid.uuid4().hex[:12]
    return s["client_id"]


def _ws_url(m: dict) -> str:
    u = urlparse(m["url"])
    scheme = "wss" if u.scheme == "https" else "ws"
    return f"{scheme}://{u.netloc}{u.path.rstrip('/')}/ws?clientId={quote(_client_id())}"


def _default_ws_connect(url: str, headers: dict):
    from websockets.sync.client import connect
    return connect(url, additional_headers=headers or None, open_timeout=10, max_size=2 ** 24)


_IMAGE_MAGIC = ((b"\xff\xd8\xff", "image/jpeg"), (b"\x89PNG", "image/png"), (b"RIFF", "image/webp"))


def preview_of(frame: bytes):
    """The picture in a binary /ws frame, as (bytes, mime), or None. ComfyUI puts a 4-byte event type in front, then
    a 4-byte format (or, in newer versions, a length and a JSON header); the picture is found by its magic bytes
    in the first part of the frame rather than by trusting one layout."""
    if not isinstance(frame, (bytes, bytearray)) or len(frame) < 12:
        return None
    head = bytes(frame[:4096])
    best = None
    for magic, mime in _IMAGE_MAGIC:
        i = head.find(magic, 4)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, mime)
    if best is None:
        return None
    data = bytes(frame[best[0]:])
    return (data, best[1]) if len(data) <= PREVIEW_MAX_BYTES else None


def _live_update(mid: str, **kw) -> None:
    with _live_lock:
        d = _live.setdefault(mid, {})
        d.update(kw, t=_now())


def live_of(mid: str) -> dict:
    with _live_lock:
        return dict(_live.get(mid) or {})


def handle_ws_message(m: dict, msg) -> None:
    """One message of a machine's feed (text JSON or a binary frame), for the render that machine runs now."""
    mid = m["id"]
    cur = live_of(mid)
    if isinstance(msg, (bytes, bytearray)):
        pic = preview_of(msg)
        if pic and cur.get("prompt_id"):
            _live_update(mid, preview=pic[0], preview_mime=pic[1], preview_t=_now())
        return
    try:
        ev = json.loads(msg)
    except (TypeError, ValueError):
        return
    kind, data = ev.get("type"), ev.get("data") or {}
    pid = data.get("prompt_id")
    if kind == "execution_start" and pid:
        _live_update(mid, prompt_id=pid, node=None, class_type="", value=0, max=0, nodes_done=0, done_nodes=[],
                     preview=None, preview_t=0)
    elif kind == "execution_cached" and pid and pid == cur.get("prompt_id"):
        nodes = [str(n) for n in data.get("nodes") or []]
        done = list(dict.fromkeys((cur.get("done_nodes") or []) + nodes))
        _live_update(mid, done_nodes=done, nodes_done=len(done))
    elif kind == "executing" and pid and pid == cur.get("prompt_id"):
        node = data.get("node")
        done = list(cur.get("done_nodes") or [])
        if cur.get("node") and cur["node"] not in done:
            done.append(cur["node"])
        if node is None:
            _live_update(mid, node=None, class_type="", value=0, max=0, done_nodes=done, nodes_done=len(done), finished=True)
        else:
            _live_update(mid, node=str(node), class_type=_class_of(m, str(node)), value=0, max=0, done_nodes=done,
                         nodes_done=len(done))
    elif kind == "progress" and pid and pid == cur.get("prompt_id"):
        _live_update(mid, value=int(data.get("value") or 0), max=int(data.get("max") or 0),
                     node=str(data.get("node") or cur.get("node") or "") or None)


def _class_of(m: dict, node: str) -> str:
    return str((_live_wf.get(m["id"]) or {}).get(node) or "")


def _watch(mid: str) -> None:
    """Listen to one machine's feed while it has a render; reconnect after a drop, end when it has none."""
    backoff = 2.0
    while True:
        with _lock:
            m = _load()["machines"].get(mid)
            if not m or not m.get("task") or not m.get("url") or m["state"] in ("stopped", "gone"):
                _watchers.pop(mid, None)
                return
            url, headers = _ws_url(m), _mh(m)
        conn = None
        try:
            conn = (_ws_connect or _default_ws_connect)(url, headers)
            backoff = 2.0
            last = _now()
            while True:
                try:
                    msg = conn.recv(timeout=5)
                except TimeoutError:
                    with _lock:
                        mm = _load()["machines"].get(mid)
                        if not mm or not mm.get("task"):
                            break
                    if _now() - last > WS_IDLE_SECONDS:
                        break
                    continue
                last = _now()
                handle_ws_message(m, msg)
        except Exception as exc:  # noqa: BLE001 - a feed that drops costs only the live view
            logger.debug("gpu_pool: %s feed: %s", mid, exc)
            time.sleep(backoff)
            backoff = min(30.0, backoff * 2)
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:  # noqa: BLE001
                pass


def _ensure_watcher(m: dict) -> None:
    t = _watchers.get(m["id"])
    if t and t.is_alive():
        return
    th = threading.Thread(target=_watch, args=(m["id"],), name=f"gpu-pool-ws-{m['id']}", daemon=True)
    _watchers[m["id"]] = th
    th.start()


def _live_public(m: dict, t: Optional[dict]) -> Optional[dict]:
    """What the page draws for the render a machine runs: node, steps, nodes done of all, the newest preview."""
    if not m.get("task") or not t or not t.get("prompt_id"):
        return None
    d = live_of(m["id"])
    if d.get("prompt_id") != t.get("prompt_id"):
        return {"started": False}
    import base64
    out = {"started": True, "node": d.get("node"), "class_type": d.get("class_type") or "",
           "value": d.get("value") or 0, "max": d.get("max") or 0, "nodes_done": d.get("nodes_done") or 0,
           "nodes_total": len(t.get("workflow") or {})}
    if d.get("preview") and _now() - float(d.get("preview_t") or 0) < PREVIEW_MAX_AGE:
        out["preview"] = f"data:{d.get('preview_mime') or 'image/jpeg'};base64," + base64.b64encode(d["preview"]).decode("ascii")
    return out


# ── the tick ─────────────────────────────────────────────────────────────────
def _poll_providers(force: bool = False) -> None:
    global _last_provider_poll
    if not force and _now() - _last_provider_poll < PROVIDER_POLL_SECONDS:
        return
    _last_provider_poll = _now()
    for provider in ("vast", "runpod"):
        ms = [m for m in _load()["machines"].values() if m["provider"] == provider]
        if not ms or not _key(provider):
            continue
        try:
            seen = _client(provider).instances()
        except Exception as exc:  # noqa: BLE001
            logger.info("gpu_pool: %s instances: %s", provider, exc)
            continue
        for m in ms:
            info = seen.get(m["provider_id"])
            if info is None or info["status"] == "gone":
                # Terminated on the provider's own site: take it out of the farm. A machine rented a moment ago
                # may not be listed yet, so it gets a grace period first.
                if _now() - float(m.get("created") or 0) < GONE_GRACE_SECONDS:
                    continue
                logger.info("gpu_pool: %s %s is no longer listed by %s; removed", m["name"], m["provider_id"], provider)
                _requeue(m)
                _load()["machines"].pop(m["id"], None)
                continue
            if info.get("url"):
                m["url"] = info["url"]
            if info.get("usd_per_hour"):
                m["usd_per_hour"] = info["usd_per_hour"]
            if info["status"] == "stopped" and m["state"] != "stopped":
                _requeue(m)
                m.update(state="stopped", running_since=None)
            elif info["status"] == "running" and m["state"] in ("stopped", "gone"):
                m.update(state="starting", running_since=_now())


def _bill(dt: float) -> None:
    s = _load()
    for m in s["machines"].values():
        if m["provider"] == "url" or m["state"] in ("stopped", "gone") or not m.get("running_since"):
            continue
        usd = float(m["usd_per_hour"] or 0) * dt / 3600.0
        m["cost_usd"] = round(m["cost_usd"] + usd, 6)
        s["spend"][_today()] = round(float(s["spend"].get(_today(), 0.0)) + usd, 6)
    for d in sorted(s["spend"])[:-31]:
        s["spend"].pop(d, None)


def tick(dt: Optional[float] = None) -> None:
    """One round: provider status, billing, cap, readiness, dispatch, polling, idle stop."""
    with _lock:
        s = _load()
        last = s.get("last_tick") or _now()
        dt = max(0.0, min(60.0, _now() - last)) if dt is None else dt
        s["last_tick"] = _now()
        _poll_providers()
        _bill(dt)
        st = s["settings"]
        over = _cap_left() <= 0
        for m in list(s["machines"].values()):
            if over and m["provider"] != "url" and m["state"] not in ("stopped", "gone"):
                try:
                    stop(m["id"], reason=f"stopped: the ${st['daily_cap_usd']:.2f} daily cap was reached")
                except Exception as exc:  # noqa: BLE001
                    m["error"] = f"could not stop at the cap: {exc}"
                continue
            if m["state"] in ("starting", "unreachable") and m.get("url"):
                if _probe(m):
                    m.update(state="ready", error="", idle_since=_now())
            if m["state"] == "ready" and not m.get("task") and m["provider"] != "url":
                idle = float(st["idle_stop_minutes"]) * 60
                busy_elsewhere = _queue_len(m) > 0
                if idle and not busy_elsewhere and _now() - float(m.get("idle_since") or _now()) > idle:
                    if not any(t["state"] == "queued" for j in s["jobs"].values() for t in j["tasks"]):
                        try:
                            stop(m["id"], reason=f"stopped after {st['idle_stop_minutes']} idle minutes")
                        except Exception as exc:  # noqa: BLE001
                            m["error"] = f"could not stop when idle: {exc}"
        for j in sorted(s["jobs"].values(), key=lambda j: j["created"]):
            if j["state"] in ("done", "failed", "cancelled"):
                continue
            for t in j["tasks"]:
                if t["state"] in ("running", "models"):
                    m = s["machines"].get(t["machine"] or "")
                    if not m:
                        t.update(state="queued", machine=None, prompt_id=None)
                        continue
                    (_poll_task if t["state"] == "running" else _poll_models)(m, j, t)
            for t in j["tasks"]:
                if t["state"] == "queued" and not over:
                    m = _pick_machine(j, t)
                    if not m:
                        break
                    _start_task(m, j, t)
            j["state"] = _job_state(j)
        _save()


def _loop() -> None:
    while True:
        try:
            tick()
        except Exception:  # noqa: BLE001
            logger.exception("gpu_pool tick")
        time.sleep(TICK_SECONDS)


def ensure_running() -> None:
    """Start the background tick (once per process)."""
    global _thread
    with _lock:
        if _thread and _thread.is_alive():
            return
        _thread = threading.Thread(target=_loop, name="gpu-pool", daemon=True)
        _thread.start()


# ── what the viewer sees ─────────────────────────────────────────────────────
def _public_machine(m: dict) -> dict:
    out = {k: v for k, v in m.items() if k not in ("token", "uploaded")}
    out["has_token"] = bool(m.get("token"))
    out["uptime_s"] = int(_now() - m["running_since"]) if m.get("running_since") else 0
    out["cost_usd"] = round(float(m.get("cost_usd") or 0), 4)
    if m.get("task"):
        out["task"] = {"job": m["task"][0], "i": m["task"][1]}
        j = _load()["jobs"].get(m["task"][0])
        t = j["tasks"][m["task"][1]] if j and m["task"][1] < len(j["tasks"]) else None
        live = _live_public(m, t)
        if live:
            out["live"] = live
    return out


def _public_job(j: dict) -> dict:
    tasks = [{k: v for k, v in t.items() if k != "workflow"} for t in j["tasks"]]
    counts: dict = {}
    for t in tasks:
        counts[t["state"]] = counts.get(t["state"], 0) + 1
    return {**{k: v for k, v in j.items() if k != "tasks"}, "tasks": tasks, "counts": counts}


_snapshot: dict = {}


def state() -> dict:
    """Everything the Farm space draws. Never a key or a machine token. While a tick is busy talking to a machine
    (a download can take minutes) the last snapshot is answered, so the page never waits on the network."""
    global _snapshot
    if not _lock.acquire(timeout=0.5):
        return dict(_snapshot, stale=True) if _snapshot else {"machines": [], "jobs": [], "stale": True}
    try:
        s = _load()
        _snapshot = {"machines": sorted((_public_machine(m) for m in s["machines"].values()), key=lambda m: m["created"]),
                "jobs": sorted((_public_job(j) for j in s["jobs"].values()), key=lambda j: -j["created"]),
                "settings": dict(s["settings"]),
                "spend": {"today": spent_today(), "cap": float(s["settings"]["daily_cap_usd"]),
                          "per_hour": round(sum(float(m["usd_per_hour"] or 0) for m in s["machines"].values()
                                                if m["provider"] != "url" and m["state"] not in ("stopped", "gone")), 4),
                          "days": dict(sorted(s["spend"].items())[-14:])},
                "keys": {p: bool(_key(p)) for p in KEY_NAMES},
                "providers": list(PROVIDERS)}
        return _snapshot
    finally:
        _lock.release()


def action(body: dict) -> dict:
    """One POST of the Farm space: ``{action, ...}`` -> a result dict (``state`` included)."""
    a = str((body or {}).get("action") or "")
    b = body or {}
    res: dict = {}
    if a == "offers":
        res["offers"] = offers(str(b.get("provider") or "vast"), b.get("gpu"))
    elif a == "rent":
        res["machine"] = rent(str(b.get("provider")), str(b.get("offer_id")), usd_per_hour=float(b.get("usd_per_hour") or 0),
                              gpu=str(b.get("gpu") or ""), region=str(b.get("region") or ""), name=str(b.get("name") or ""))
    elif a == "start":
        res["machine"] = start(str(b.get("id")))
    elif a == "stop":
        res["machine"] = stop(str(b.get("id")), "stopped by hand")
    elif a == "terminate":
        terminate(str(b.get("id")))
    elif a == "add_url":
        res["machine"] = add_url(str(b.get("url") or ""), str(b.get("name") or ""), str(b.get("token") or ""),
                                 float(b.get("usd_per_hour") or 0))
    elif a == "remove":
        remove(str(b.get("id")))
    elif a == "open_url":
        # The machine's ComfyUI for a browser: with its token once (the farm image keeps it as a cookie).
        with _lock:
            m = _machine(str(b.get("id")))
            if not m.get("url"):
                raise PoolError(f"{m['name']} has no address yet")
            res["url"] = m["url"].rstrip("/") + "/" + (f"?token={quote(m['token'])}" if m.get("token") else "")
    elif a == "parity":
        res["parity"] = check_parity(str(b.get("id")), b.get("workflow") or {})
    elif a == "submit":
        res["job"] = submit(b.get("workflow") or {}, name=str(b.get("name") or ""), batch=int(b.get("batch") or 1),
                            seed_mode=str(b.get("seed_mode") or "increment"), machines=b.get("machines") or None,
                            dest=b.get("dest") or None, model_urls=b.get("model_urls") or None)
    elif a == "cancel":
        res["job"] = cancel(str(b.get("id")))
    elif a == "retry":
        res["job"] = retry(str(b.get("id")))
    elif a == "remove_job":
        remove_job(str(b.get("id")))
    elif a == "settings":
        res["settings"] = set_settings(b.get("settings") or {})
    elif a == "refresh":
        with _lock:
            _poll_providers(force=True)
    else:
        raise PoolError(f"Unknown farm action {a!r}")
    if a in ("submit", "start", "add_url", "rent"):
        ensure_running()
    res["state"] = state()
    return res
