"""SpAItial Echo: a navigable Gaussian-splat world from one picture, a 360 panorama or a text prompt.

A second world generator beside World Labs (``worldlabs_atlas.py`` / ``worldlabs_marble.py``): SpAItial is a
European company (Munich) and its Developer API (``https://api.spaitial.ai``, docs https://docs.spaitial.ai/api)
is a plain REST API, written here from its published reference, not from any SDK or plugin.

* ``POST /v1/files``                         a local picture, multipart field ``file`` -> ``file_id`` (24 h)
* ``POST /v1/worlds``                        ``{input, model, title, output_format}`` -> 202 ``request_id``
* ``GET  /v1/worlds/requests/{id}/status``   ``status`` PENDING / PROCESSING / COMPLETED / FAILED / CANCELLED,
  ``progress`` 0..1 (or null), ``error`` {code, message}
* ``GET  /v1/worlds/requests/{id}``          the result: ``world`` {id, title, splat_url, splat_format,
  thumbnail_url, panorama_url, viewer_url, camera}
* ``GET  /v1/worlds/requests/{id}/splat``    302 to a fresh signed splat link (``/panorama`` likewise)
* ``POST / GET /v1/worlds/requests/{id}/exports/{type}``  ``mesh`` (full .ply), ``mesh-simplified``,
  ``floorplan`` (PNG); status PROCESSING / READY / FAILED and ``download_url`` when ready
* ``GET  /v1/models``                        the models this key may use, each with its price

Models: ``echo-2`` (about 8 minutes, 160 credits = US$ 1.60), ``echo-hq`` ("Echo 2 HQ", about an hour, 800
credits = US$ 8). One credit is US$ 0.01. ``echo-3`` (phone captures) is not wrapped. Text OR one picture
OR a panorama per world: the API takes no prompt beside a picture.

Output goes to ``<ComfyUI output>/3d_scenes/echo/<name>/`` with a ``world.json`` and ``.agenty.json``
sidecars. Key: ``SPAITIAL_API_KEY`` (``spt_live_...``, developers.spaitial.ai); ``SPAITIAL_BASE_URL``
overrides the address. The key is sent only to the API's own host, never to the signed download hosts,
and never logged.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests

from agenty_core._compat import tool
from agenty_core.paths import project_root
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools.worldlabs_atlas import AtlasError, AtlasTimeout, _err

logger = logging.getLogger("agenty_core.spaitial_echo")

_DEFAULT_BASE = "https://api.spaitial.ai"
_sleep = time.sleep                      # tests replace this
_POLL_S = 10.0
MODELS = ("echo-2", "echo-hq")
# docs.spaitial.ai/api/generation-models (credits, US$ 0.01 each); GET /v1/models is the authority
_PRICE = {"echo-2": 160, "echo-hq": 800}
_MINUTES = {"echo-2": 8, "echo-hq": 60}
FORMATS = ("spz", "ply", "sog")
EXPORTS = ("mesh", "mesh-simplified", "floorplan")
_EXPORT_FILE = {"mesh": "mesh.ply", "mesh-simplified": "mesh_simplified.ply", "floorplan": "floorplan.png"}
_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif")
MAX_IMAGE_BYTES = 25 * 1024 * 1024


# -- key, base, requests -----------------------------------------------------------------


def _key() -> str:
    from agenty_core.utils.secrets import get_secret
    return get_secret("SPAITIAL_API_KEY")


def _base() -> str:
    return (os.environ.get("SPAITIAL_BASE_URL") or _DEFAULT_BASE).strip().rstrip("/")


def _own_host(url: str) -> bool:
    """Whether `url` is on the API's own host (the only one that gets the key)."""
    try:
        return urlparse(url).netloc.lower() == urlparse(_base()).netloc.lower()
    except ValueError:
        return False


def _auth() -> dict:
    return {"Authorization": f"Bearer {_key()}"}


def _body_of(r) -> dict:
    try:
        b = r.json()
        return b if isinstance(b, dict) else {"data": b}
    except Exception:  # noqa: BLE001
        return {"message": (getattr(r, "text", "") or "")[:300]}


def _http_error(r) -> AtlasError:
    """Classify a non-2xx answer by SpAItial's error envelope ``{"error": {code, message, details}}``."""
    st, body = r.status_code, _body_of(r)
    e = body.get("error") if isinstance(body.get("error"), dict) else {}
    code = str(e.get("code") or "")
    msg = str(e.get("message") or body.get("message") or "")[:300]
    ra = (getattr(r, "headers", None) or {}).get("Retry-After")
    tail = f" ({code}: {msg})" if code or msg else ""
    if st == 401:
        return AtlasError("SpAItial rejected the API key (missing, unknown or revoked): set SPAITIAL_API_KEY "
                          f"(a key from developers.spaitial.ai) in agentY's Settings or .env.{tail}", "auth")
    if st == 402:
        return AtlasError(f"The SpAItial account has too few credits for this world (developers.spaitial.ai).{tail}",
                          "balance")
    if code == "MODERATION_REJECTED":
        return AtlasError(f"SpAItial declined the picture or prompt (moderation).{tail}", "content_refusal")
    if st == 403:
        return AtlasError(f"This SpAItial key may not do that (a scope, or a model it is not granted).{tail}", "scope")
    if st == 404:
        return AtlasError(f"SpAItial does not know it (a wrong id, or an upload older than 24 h).{tail}", "not_found")
    if st == 409:
        return AtlasError(f"SpAItial: not ready yet, or not possible now.{tail}", "not_ready")
    if st in (400, 422):
        return AtlasError(f"SpAItial refused the request.{tail}", "args")
    if st == 429:
        return AtlasError(f"SpAItial rate limit; retry after {ra or 'a minute'} s.{tail}", "rate_limit", retry_after=ra)
    return AtlasError(f"SpAItial error {st}.{tail}", "api")


def _request(method: str, path: str, *, json_body=None, files=None, params=None, idempotent: bool = False,
             timeout: float = 120.0) -> dict:
    """One API call. 429 / 5xx on a GET are retried a few times; a POST is sent once (with an
    Idempotency-Key when it creates something, so a resend would not be billed twice)."""
    url = f"{_base()}{path}"
    headers = _auth()
    if idempotent:
        headers["Idempotency-Key"] = uuid.uuid4().hex
    attempts = 3 if method == "GET" else 1
    for attempt in range(attempts):
        try:
            r = requests.request(method, url, headers=headers, json=json_body, files=files, params=params,
                                 timeout=timeout)
        except requests.Timeout as exc:
            raise AtlasError("SpAItial did not answer in time.", "timeout") from exc
        except requests.RequestException as exc:
            raise AtlasError(f"could not reach SpAItial: {exc}", "api") from exc
        if r.status_code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
            try:
                wait = min(float(r.headers.get("Retry-After")), 30.0)
            except (TypeError, ValueError):
                wait = 2.0 * (attempt + 1)
            _sleep(wait)
            continue
        if r.status_code >= 400:
            raise _http_error(r)
        return _body_of(r)
    raise AtlasError("SpAItial request failed", "api")  # pragma: no cover


def _fetch(url: str, dest: Path) -> Path:
    """Download `url` to `dest`. A link on the API's host gets the key and its redirect is followed by hand,
    so the key never travels to the storage host the signed link points at."""
    for _hop in range(5):
        own = _own_host(url)
        try:
            r = requests.get(url, headers=_auth() if own else {}, timeout=900, allow_redirects=not own)
        except requests.RequestException as exc:
            raise AtlasError(f"could not download from SpAItial: {exc}", "api") from exc
        if own and r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            url = urljoin(url, r.headers["Location"])
            continue
        if r.status_code >= 400:
            if own:
                raise _http_error(r)
            raise AtlasError(f"could not download a SpAItial file (HTTP {r.status_code}); the signed link may have "
                             "expired: echo_export(name, refresh=true) asks for a new one.", "api")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(r.content)
        os.replace(tmp, dest)
        return dest
    raise AtlasError("too many redirects downloading from SpAItial", "api")


# -- models and price ---------------------------------------------------------------------------


def _model_entries() -> list:
    body = _request("GET", "/v1/models")
    rows = body.get("models") or body.get("data") or []
    return [m for m in rows if isinstance(m, dict)] if isinstance(rows, list) else []


def _credits_in(pricing) -> Optional[float]:
    """A credit figure out of a model's ``pricing`` entry, whatever its exact shape."""
    if isinstance(pricing, (int, float)):
        return float(pricing)
    if isinstance(pricing, dict):
        for k in ("credits", "base_credits", "credits_per_world", "price_credits", "amount"):
            v = pricing.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def estimate_credits(model: str, models: Optional[list] = None) -> dict:
    """{"credits", "usd", "minutes", "source"} for one world: the model list's price when it gives one, else the
    documented one."""
    credits, source = float(_PRICE.get(model, _PRICE["echo-2"])), "docs"
    for m in models or []:
        if str(m.get("id") or m.get("slug") or m.get("model") or "") == model:
            c = _credits_in(m.get("pricing"))
            if c is not None:
                credits, source = c, "api"
    return {"credits": int(credits), "usd": round(credits / 100.0, 2), "minutes": _MINUTES.get(model, 10),
            "source": source,
            "note": "1 credit = US$ 0.01. Mesh exports are not priced in SpAItial's docs; check the balance at "
                    "developers.spaitial.ai after the first one."}


# -- files -------------------------------------------------------------------------------


def _root_dir(output_dir: str = "") -> Path:
    if str(output_dir or "").strip():
        d = Path(os.path.expanduser(str(output_dir).strip().strip('"')))
    else:
        base = None
        try:
            from agenty_core.utils.comfyui_client import comfyui_output_dir
            base = comfyui_output_dir()
        except Exception:  # noqa: BLE001
            base = None
        d = (Path(base) if base else project_root() / "output") / "3d_scenes" / "echo"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _world_dir(name: str, output_dir: str = "") -> Path:
    return _root_dir(output_dir) / wl._slug(name)


def _load(wd: Path) -> dict:
    f = wd / "world.json"
    if not f.exists():
        have = sorted(p.parent.name for p in wd.parent.glob("*/world.json"))
        raise AtlasError(f"no Echo world '{wd.name}' (looked in {wd.parent}); have: {', '.join(have) or 'none'}.", "args")
    return json.loads(f.read_text(encoding="utf-8"))


def _save(wd: Path, world: dict) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    tmp = wd / "world.json.part"
    tmp.write_text(json.dumps(world, indent=1), encoding="utf-8")
    os.replace(tmp, wd / "world.json")


def _sidecar(path: Path, role: str, meta: dict) -> None:
    wl._sidecar(path, role, {"provider": "spaitial-echo", **meta})


def _taken(wd: Path) -> list:
    """The record of earlier failed attempts of a world folder that may be made again ([] when new). A world
    that is ready or still running stays protected."""
    f = wd / "world.json"
    if not f.exists():
        return []
    try:
        old = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    if old.get("status") not in ("failed", "cancelled", None):
        raise AtlasError(f"an Echo world named '{wd.name}' already exists; pick another name, or echo_export to "
                         "fetch more of it", "args")
    keep = {k: old.get(k) for k in ("request_id", "created", "model", "error") if old.get(k)}
    return list(old.get("failed_attempts") or []) + ([keep] if keep else [])


# -- inputs -------------------------------------------------------------------------------


def _image_input(ref: str, is_pano: str) -> dict:
    """One picture as a world ``input``: an https URL as given, anything local uploaded to /v1/files."""
    ref = str(ref).strip()
    pano = {} if is_pano == "auto" else {"is_pano": is_pano == "true"}
    if ref.startswith("https://"):
        return {"type": "url", "image_url": ref, **pano}
    if ref.startswith("http://"):
        raise AtlasError("SpAItial takes https links only; download the picture first", "args")
    if ref.startswith("data:"):
        if len(ref) * 3 // 4 > MAX_IMAGE_BYTES:
            raise AtlasError("the picture is over SpAItial's 25 MB", "args")
        return {"type": "base64", "image_base64": ref, **pano}
    from agenty_core.tools.image_io import resolve_local_image
    path = resolve_local_image(ref)
    if not path:
        raise AtlasError(f"image not found: {ref}", "args")
    p = Path(path)
    data, name = p.read_bytes(), p.name
    if p.suffix.lower() not in _IMG_EXT:
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.open(p).convert("RGB").save(buf, "PNG")
        data, name = buf.getvalue(), f"{p.stem}.png"
    if len(data) > MAX_IMAGE_BYTES:
        raise AtlasError(f"{p.name} is over SpAItial's 25 MB; make it smaller first", "args")
    ctype = mimetypes.guess_type(name)[0] or "image/png"
    up = _request("POST", "/v1/files", files={"file": (name, data, ctype)})
    fid = up.get("file_id") or up.get("id") or (up.get("file") or {}).get("file_id")
    if not fid:
        raise AtlasError("the SpAItial upload answered without a file_id", "api")
    return {"type": "file_id", "file_id": str(fid), **pano}


# -- result handling ----------------------------------------------------------------------


def _status(body: dict) -> str:
    return str(body.get("status") or body.get("state") or "").upper()


def _poll(request_id: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + float(timeout_s)
    while True:
        st = _request("GET", f"/v1/worlds/requests/{request_id}/status")
        s = _status(st)
        if s == "COMPLETED":
            return st
        if s in ("FAILED", "CANCELLED"):
            err = st.get("error") if isinstance(st.get("error"), dict) else {}
            msg = err.get("message") or ("the job was cancelled" if s == "CANCELLED" else "the world could not be made")
            kind = "content_refusal" if re.search(r"moderat|polic|safety", str(err.get("code") or msg), re.I) else "execution"
            raise AtlasError(f"SpAItial could not make the world ({msg}). Calling echo_create_world again with the "
                             "same name is fine.", kind, request_id=request_id)
        if time.monotonic() >= deadline:
            raise AtlasTimeout(f"still running after {int(timeout_s)} s; collect it later with echo_export(name) "
                               "(it picks the job up where it is).", "timeout", request_id=request_id)
        _sleep(_POLL_S)


def _splat_name(fmt: str) -> str:
    return f"splat.{fmt}"


def _store(wd: Path, world: dict, refresh: bool = False) -> dict:
    """Fetch the finished world's result, splat, thumbnail and panorama into the folder; update world.json."""
    rid = world["request_id"]
    res = _request("GET", f"/v1/worlds/requests/{rid}")
    w = res.get("world") if isinstance(res.get("world"), dict) else {}
    fmt = str(w.get("splat_format") or world.get("format") or "spz").lower().lstrip(".")
    if fmt not in FORMATS:
        fmt = world.get("format") or "spz"
    files = world.setdefault("files", {})
    dest = wd / _splat_name(fmt)
    if refresh or not dest.exists():
        _fetch(f"{_base()}/v1/worlds/requests/{rid}/splat", dest)
        _sidecar(dest, f"Echo splat ({fmt.upper()})", {"world": world["name"], "request_id": rid,
                                                         "model": world.get("model")})
    files["splat"] = dest.name
    for key, url, stem in (("thumbnail", w.get("thumbnail_url"), "thumbnail"),
                           ("panorama", f"{_base()}/v1/worlds/requests/{rid}/panorama" if w.get("panorama_url") else "",
                            "panorama")):
        if not url or (files.get(key) and not refresh):
            continue
        try:
            ext = Path(urlparse(str(url)).path).suffix.lower()
            got = _fetch(str(url), wd / f"{stem}{ext if ext in _IMG_EXT else '.jpg'}")
            files[key] = got.name
        except AtlasError:
            pass                                 # a picture of it is nice to have, the splat is the world
    world.update({k: v for k, v in (("world_id", w.get("id")), ("viewer_url", w.get("viewer_url")),
                                    ("title", w.get("title")), ("camera", w.get("camera"))) if v})
    world["format"], world["status"] = fmt, "ready"
    _save(wd, world)
    return world


def _export(wd: Path, world: dict, kind: str, timeout_s: float, refresh: bool = False) -> Path:
    """A mesh / floorplan export of the world, made on SpAItial's side, then downloaded."""
    rid = world["request_id"]
    dest = wd / _EXPORT_FILE[kind]
    if dest.exists() and not refresh:
        return dest
    path = f"/v1/worlds/requests/{rid}/exports/{kind}"
    st: dict = {}
    if not refresh:
        try:
            st = _request("GET", path)
        except AtlasError as exc:
            if exc.kind != "not_found":
                raise
    if _status(st) not in ("READY", "PROCESSING"):
        try:
            st = _request("POST", path, idempotent=True)
        except AtlasError as exc:
            if exc.kind != "not_ready":
                raise
            st = _request("GET", path)
    deadline = time.monotonic() + float(timeout_s)
    while _status(st) != "READY":
        if _status(st) == "FAILED":
            raise AtlasError(f"SpAItial could not make the {kind} export", "execution", request_id=rid)
        if time.monotonic() >= deadline:
            raise AtlasTimeout(f"the {kind} export is still being made; echo_export(name, what='{kind}') later "
                               "collects it.", "timeout", request_id=rid)
        _sleep(_POLL_S)
        st = _request("GET", path)
    url = st.get("download_url") or f"{_base()}{path}?download=1"
    _fetch(url if str(url).startswith("http") else urljoin(_base() + "/", str(url)), dest)
    _sidecar(dest, f"Echo {kind} export", {"world": world["name"], "request_id": rid})
    world.setdefault("files", {})[kind] = dest.name
    _save(wd, world)
    return dest


def _open_viewer(path: Path) -> dict:
    try:
        from agenty_core.tools.splat_train import open_splat_in_viewer
        return open_splat_in_viewer(path, None)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not open it in the ComfyUI Image Viewer: {exc}"}


def _summary(wd: Path, world: dict, extra: Optional[dict] = None) -> str:
    files = {k: str(wd / v) for k, v in (world.get("files") or {}).items()}
    out = {"status": "ok", "name": world["name"], "provider": "spaitial-echo", "world_dir": str(wd),
           "request_id": world.get("request_id"), "world_id": world.get("world_id"), "model": world.get("model"),
           "files": files, "splat": files.get("splat", ""), "mesh": files.get("mesh") or files.get("mesh-simplified", ""),
           "viewer_url": world.get("viewer_url"), "estimate": world.get("estimate"),
           "next": "echo_export(name, what='mesh' | 'mesh-simplified' | 'floorplan') for more; show_in_viewer "
                   "for the 3D view; atlas_splat_to_mesh(splat_path=<the splat>) also makes a mesh (World Labs, paid)."}
    if extra:
        out.update(extra)
    return json.dumps({k: v for k, v in out.items() if v not in (None, "")})


def _guard(fn):
    def wrapper(*a, **kw):
        if not _key():
            return _err("No SpAItial API key: set SPAITIAL_API_KEY (a key from developers.spaitial.ai, spt_live_...) "
                        "in agentY's Settings > API keys or .env.", "auth")
        try:
            return fn(*a, **kw)
        except AtlasError as exc:
            return exc.render()
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", fn.__name__)
            return _err(f"{type(exc).__name__}: {exc}", "api")
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


# -- tools ------------------------------------------------------------------------------------


@tool
def echo_create_world(name: str, image: str = "", prompt: str = "", model: str = "echo-2", is_pano: str = "auto",
                      format: str = "spz", mesh: str = "", open_in_viewer: bool = True, dry_run: bool = False,
                      timeout_s: int = 1800, output_dir: str = "") -> str:
    """Make a navigable Gaussian-splat world with SpAItial Echo from ONE picture, a 360 panorama, or a prompt.

    The second world generator beside World Labs Atlas / Marble (SpAItial, a European company). PAID:
    echo-2 160 credits (US$ 1.60, about 8 minutes), echo-hq 800 credits (US$ 8, about an hour). Say the cost
    before calling; ``dry_run=true`` returns the request and the price without spending anything. One call =
    one world: do not loop.

    Give ``image`` OR ``prompt``: SpAItial takes no words beside a picture (a prompt given with a picture is
    left out, and the answer says so). The splat is saved as SpAItial writes it and opened in the ComfyUI
    Image Viewer's 3D view; ``mesh`` also asks SpAItial for a mesh export (.ply).

    Args:
        name: folder name, saved under output/3d_scenes/echo/<name>/.
        image: one picture of the place (a local path or an https URL), or a 360 equirectangular panorama.
        prompt: text describing the place (used when there is no picture), up to 2000 characters.
        model: echo-2 (default) or echo-hq (higher detail, about an hour).
        is_pano: "auto" (default), "true" (the picture is a 360 panorama) or "false".
        format: splat file: spz (default, small), ply or sog.
        mesh: "" (none), "mesh" (full-resolution .ply) or "mesh-simplified" (for real time). SpAItial's docs
            give no price for exports.
        open_in_viewer: open the splat in the viewer's 3D view when done.
        dry_run: no upload and nothing billed: return the request and the price.
        timeout_s: how long to wait; on timeout the job stays recorded and echo_export(name) collects it.
        output_dir: save here instead of ComfyUI's output/3d_scenes/echo.

    Returns:
        JSON: name, world_dir, splat (path), mesh (path, if asked), files, request_id, viewer_url, estimate,
        viewer - or {"error", "error_kind"} (auth, balance, content_refusal, args, timeout, ...).
    """
    return _create(name, image, prompt, model, is_pano, format, mesh, open_in_viewer, dry_run, timeout_s, output_dir)


@_guard
def _create(name, image, prompt, model, is_pano, fmt, mesh, open_in_viewer, dry_run, timeout_s, output_dir):
    model = str(model or "echo-2").strip().lower()
    if model not in MODELS:
        raise AtlasError(f"model must be one of {', '.join(MODELS)}", "args")
    fmt = str(fmt or "spz").strip().lower().lstrip(".")
    if fmt not in FORMATS:
        raise AtlasError("format must be spz, ply or sog", "args")
    pano = str(is_pano if not isinstance(is_pano, bool) else ("true" if is_pano else "false")).strip().lower()
    if pano not in ("auto", "true", "false"):
        raise AtlasError("is_pano must be auto, true or false", "args")
    mesh = str(mesh or "").strip().lower()
    if mesh in ("none", "false", "0"):
        mesh = ""
    if mesh and mesh not in ("mesh", "mesh-simplified"):
        raise AtlasError("mesh must be '', 'mesh' or 'mesh-simplified'", "args")
    image, prompt = str(image or "").strip(), str(prompt or "").strip()
    if not wl._slug(name):
        raise AtlasError("give the world a name", "args")
    if not image and not prompt:
        raise AtlasError("give a picture of the place or a prompt", "args")
    if len(prompt) > 2000:
        raise AtlasError("the prompt is over SpAItial's 2000 characters", "args")
    wd = _world_dir(name, output_dir)
    previous = _taken(wd)
    notes = []
    if image and prompt:
        notes.append("SpAItial takes a picture OR words; the prompt was left out (the picture decides).")
    try:
        models = _model_entries()
    except AtlasError as exc:
        if exc.kind in ("auth", "scope"):
            raise
        models = []
    if models and not any(str(m.get("id") or m.get("slug") or m.get("model") or "") == model for m in models):
        names = [str(m.get("id") or m.get("slug") or "") for m in models]
        raise AtlasError(f"this SpAItial key may not use {model}; it may use: {', '.join(n for n in names if n) or 'none'}",
                         "scope")
    est = estimate_credits(model, models)
    title = wl._slug(name)[:200]

    def body_with(inp):
        return {"input": inp, "model": model, "title": title, "output_format": fmt}

    if dry_run:
        shown = ({"type": "text", "prompt": prompt} if not image else
                 {"type": "file_id", "file_id": f"<uploaded from {image}>"} if not image.startswith(("https://", "data:"))
                 else {"type": "url" if image.startswith("https://") else "base64", "...": image[:60]})
        return json.dumps({"status": "dry_run", "spent": 0, "estimate": est, "request": body_with(shown),
                           "mesh": mesh or None, "world_dir": str(wd), "notes": notes or None,
                           "note": f"Nothing was sent. Call again without dry_run to spend about {est['credits']} "
                                   f"credits (US$ {est['usd']:.2f})."})
    inp = _image_input(image, pano) if image else {"type": "text", "prompt": prompt}
    sub = _request("POST", "/v1/worlds", json_body=body_with(inp), idempotent=True)
    rid = sub.get("request_id") or sub.get("id")
    if not rid:
        raise AtlasError("SpAItial accepted the world without a request_id", "api")
    world = {"version": 1, "name": title, "provider": "spaitial-echo", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "status": "pending", "model": model, "request_id": str(rid), "format": fmt,
             "input": {"type": inp["type"], "image": image or None, "is_pano": pano}, "prompt": prompt,
             "estimate": est, "request": {"mesh": mesh, "open_in_viewer": bool(open_in_viewer)}}
    if previous:
        world["failed_attempts"] = previous
    _save(wd, world)
    try:
        _poll(str(rid), timeout_s)
    except AtlasTimeout as exc:
        exc.extra.setdefault("name", world["name"])
        raise
    except AtlasError as exc:
        world.update(status="failed", error=str(exc)[:300])
        _save(wd, world)
        raise
    return _finish(wd, world, mesh, bool(open_in_viewer), timeout_s, notes)


def _finish(wd: Path, world: dict, mesh: str, open_in_viewer: bool, timeout_s: float, notes: list) -> str:
    world = _store(wd, world)
    extra: dict = {"notes": notes or None}
    if mesh:
        try:
            extra["mesh"] = str(_export(wd, world, mesh, timeout_s))
        except AtlasError as exc:             # the world is made and paid for either way
            extra["mesh_error"] = str(exc)
    if open_in_viewer:
        extra["viewer"] = _open_viewer(wd / world["files"]["splat"])
    return _summary(wd, world, extra)


@tool
def echo_export(name: str, what: str = "splat", refresh: bool = False, open_in_viewer: bool = False,
                timeout_s: int = 1800, output_dir: str = "") -> str:
    """Get files of a SpAItial Echo world made by echo_create_world; also collects one whose creation timed out.

    Args:
        name: the Echo world (folder name under output/3d_scenes/echo).
        what: "splat" (default; free), "mesh" (full-resolution .ply), "mesh-simplified" (for real time) or
            "floorplan" (a top-down PNG). Exports are made on SpAItial's side; its docs give them no price.
        refresh: download again (also renews an expired link).
        open_in_viewer: open the splat in the ComfyUI Image Viewer's 3D view.
        timeout_s: how long to wait for a still-running creation or export.
        output_dir: where the world lives, if not ComfyUI's output/3d_scenes/echo.

    Returns:
        JSON: files (all paths), file (the one asked for), splat, mesh - or {"error", "error_kind"}.
    """
    return _export_tool(name, what, refresh, open_in_viewer, timeout_s, output_dir)


@_guard
def _export_tool(name, what, refresh, open_in_viewer, timeout_s, output_dir):
    what = str(what or "splat").strip().lower()
    if what not in ("splat",) + EXPORTS:
        raise AtlasError("what must be splat, mesh, mesh-simplified or floorplan", "args")
    wd = _world_dir(name, output_dir)
    world = _load(wd)
    if not world.get("request_id"):
        raise AtlasError(f"the world '{world.get('name')}' has no SpAItial job", "args")
    if world.get("status") != "ready":
        try:
            _poll(world["request_id"], timeout_s)
        except AtlasTimeout as exc:
            exc.extra.setdefault("name", world["name"])
            raise
        except AtlasError:
            world["status"] = "failed"
            _save(wd, world)
            raise
        world = _store(wd, world)
    elif what == "splat" and (refresh or not (wd / (world.get("files") or {}).get("splat", "-")).exists()):
        world = _store(wd, world, refresh=True)
    path = wd / world["files"]["splat"] if what == "splat" else _export(wd, world, what, timeout_s, bool(refresh))
    extra = {"file": str(path)}
    if open_in_viewer:
        extra["viewer"] = _open_viewer(wd / world["files"]["splat"])
    return _summary(wd, world, extra)


def find_world(ref: str, output_dir: str = "") -> Optional[Path]:
    """The Echo world folder named by `ref` (a bare name), or None."""
    ref = str(ref or "").strip().strip('"')
    if not ref or any(c in ref for c in "/\\") or Path(ref).suffix:
        return None
    wd = _world_dir(ref, output_dir)
    return wd if (wd / "world.json").exists() else None


__all__ = ["echo_create_world", "echo_export", "estimate_credits", "find_world", "MODELS"]
