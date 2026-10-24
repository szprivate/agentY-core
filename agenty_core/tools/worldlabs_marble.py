"""World Labs Marble: a navigable Gaussian-splat world from pictures, with SPZ / PLY / collider-mesh export.

Marble is World Labs' world generator behind the *World API* (``https://api.worldlabs.ai/marble/v1``,
docs https://docs.worldlabs.ai/api). It is a different service from the Atlas developer API that
``worldlabs_atlas.py`` wraps (``api.atlas-beta.worldlabs.ai/api/v2``: posed RGBD views): same ``WLT-Api-Key``
header and the same kind of async *operation*, but its own key, its own billing (credits, 1,250 per US$) and a
different shape - one call makes a whole world and the result is a splat, so there is nothing to train.

* ``POST /worlds:generate``  text / image / multi-image / video prompt -> an operation (about 5 minutes)
* ``GET  /operations/{id}``  poll; ``response`` is the World when ``done``
* ``GET  /worlds/{id}``      the World: ``assets.splats.spz_urls`` (100k, 500k, full_res), ``semantics_metadata``,
  ``assets.mesh.collider_mesh_url``, ``assets.imagery.pano_url``
* ``POST /worlds/{id}:export``  splats as PLY (free, synchronous) / high-quality mesh (3,500 credits; not wrapped)
* ``POST /media-assets:prepare_upload``  local pictures are uploaded as media assets
* ``GET  /credits``          the balance (free)

Models (docs.worldlabs.ai/api/models): ``marble-1.1-plus`` (newest, largest worlds), ``marble-1.1``,
``marble-1.0``, ``marble-1.0-draft``. There is no "Marble 2" in this API: Atlas is what World Labs calls
"Marble 2 beta", and it has no world-generation task.

Output goes to ``<ComfyUI output>/3d_scenes/marble/<name>/`` with a ``world.json`` and ``.agenty.json`` sidecars.
Key: ``WLT_MARBLE_API_KEY`` (``WORLDLABS_MARBLE_API_KEY``), else the Atlas key. Never logged.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests

from agenty_core._compat import tool
from agenty_core.paths import project_root
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools.worldlabs_atlas import AtlasError, AtlasTimeout, _err

logger = logging.getLogger("agenty_core.worldlabs_marble")

_DEFAULT_BASE = "https://api.worldlabs.ai/marble/v1"
_sleep = time.sleep                      # tests replace this
_POLL_S = 5.0
MODELS = ("marble-1.1-plus", "marble-1.1", "marble-1.0", "marble-1.0-draft")
_RES = {"full": "full_res", "full_res": "full_res", "2m": "full_res", "500k": "500k", "150k": "150k", "100k": "100k"}
_SPZ_RES = ("full_res", "500k", "100k")      # the resolutions an SPZ comes in
CREDITS_PER_USD = 1250
# docs.worldlabs.ai/api/pricing (credits): (low, high) per input type; plus adds 0-1500 for world size
_PRICE = {"draft": {"pano": 150, "text": 230, "image": 230, "multi": 250, "video": 250},
          "standard": {"pano": 1500, "text": 1580, "image": 1580, "multi": 1600, "video": 1600}}
_IMG_EXT = {".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".webp": "webp"}


# -- key, base, requests -----------------------------------------------------------------


def _key() -> str:
    from agenty_core.utils.secrets import get_secret
    return (get_secret("WLT_MARBLE_API_KEY") or get_secret("WORLDLABS_MARBLE_API_KEY")
            or get_secret("WLT_API_KEY") or get_secret("WORLDLABS_API_KEY"))


def _own_key() -> bool:
    from agenty_core.utils.secrets import get_secret
    return bool(get_secret("WLT_MARBLE_API_KEY") or get_secret("WORLDLABS_MARBLE_API_KEY"))


def _base() -> str:
    return (os.environ.get("WLT_MARBLE_BASE_URL") or _DEFAULT_BASE).strip().rstrip("/")


def _http_error(r) -> AtlasError:
    st, body = r.status_code, wl._body_of(r)
    detail = body.get("detail") or body.get("message") or body.get("error")
    msg = detail if isinstance(detail, str) else json.dumps(detail)[:300] if detail else ""
    ra = (getattr(r, "headers", None) or {}).get("Retry-After")
    if st == 401:
        hint = ("" if _own_key() else " The Atlas key is not accepted by the Marble API: create an API key for "
                "the World API at platform.worldlabs.ai and put it in .env as WLT_MARBLE_API_KEY.")
        return AtlasError(f"World Labs rejected the Marble API key (missing, unknown, expired or revoked).{hint} {msg}".strip(),
                          "auth")
    if st == 402:
        return AtlasError("The World Labs API account is out of credits: buy credits at "
                          f"platform.worldlabs.ai/billing (API credits are separate from the Marble web app's). {msg}",
                          "balance")
    if st == 400:
        return AtlasError(f"World Labs refused the request (invalid, or the content breaks its policy): {msg}",
                          "content_refusal" if re.search(r"polic|content|moderat", msg, re.I) else "args")
    if st == 403:
        return AtlasError(f"The Marble API key lacks permission for this call. {msg}", "scope")
    if st == 404:
        return AtlasError(f"Not found (a wrong id, or this key is not enabled for the World API). {msg}", "not_found")
    if st == 422:
        return AtlasError(f"World Labs rejected the request shape: {msg}", "args")
    if st == 429:
        return AtlasError(f"World Labs rate limit; retry after {ra or 'a minute'} s. {msg}", "rate_limit", retry_after=ra)
    if st >= 500:
        return AtlasError(f"World Labs error {st}: {msg}", "api")
    return AtlasError(f"World Labs error {st}: {msg}", "api")


def _request(method: str, path: str, *, json_body=None, params=None, timeout: float = 120.0) -> dict:
    """One Marble API call. 429 / 5xx on a GET are retried a few times; a POST that creates something is not
    retried (a retry could be billed twice)."""
    url = f"{_base()}{path}"
    attempts = 3 if method == "GET" else 1
    for attempt in range(attempts):
        try:
            r = requests.request(method, url, headers={"WLT-Api-Key": _key(), "Content-Type": "application/json"},
                                 json=json_body, params=params, timeout=timeout)
        except requests.Timeout as exc:
            raise AtlasError("World Labs did not answer in time.", "timeout") from exc
        except requests.RequestException as exc:
            raise AtlasError(f"could not reach World Labs: {exc}", "api") from exc
        if r.status_code in (429, 500, 502, 503, 504) and attempt + 1 < attempts:
            try:
                wait = min(float(r.headers.get("Retry-After")), 30.0)
            except (TypeError, ValueError):
                wait = 2.0 * (attempt + 1)
            _sleep(wait)
            continue
        if r.status_code >= 400:
            raise _http_error(r)
        return wl._body_of(r)
    raise AtlasError("World Labs request failed", "api")  # pragma: no cover


def _op_error(op: dict) -> AtlasError:
    err = op.get("error") or {}
    msg = err.get("message") or "the world generation failed"
    kind = "content_refusal" if re.search(r"polic|moderat|safety|prohibit", msg, re.I) else "execution"
    return AtlasError(f"World Labs could not make the world ({msg}).", kind, operation_id=op.get("operation_id"))


def _poll(op: dict, timeout_s: float) -> dict:
    deadline = time.monotonic() + float(timeout_s)
    while not op.get("done"):
        if time.monotonic() >= deadline:
            raise AtlasTimeout(f"still running after {int(timeout_s)} s; collect it later with "
                               "marble_export(name) (it picks the job up where it is).", "timeout",
                               operation_id=op.get("operation_id"))
        _sleep(_POLL_S)
        op = _request("GET", f"/operations/{op['operation_id']}")
    if op.get("error"):
        raise _op_error(op)
    return op


def credits_remaining() -> Optional[float]:
    """The API credit balance (free GET); None if it cannot be read."""
    try:
        return float(_request("GET", "/credits").get("remaining_credits"))
    except Exception:  # noqa: BLE001
        return None


# -- cost -----------------------------------------------------------------------------


def estimate_credits(model: str, kind: str, pano: bool = False) -> dict:
    """{"min", "max", "usd_min", "usd_max", "note"} for one generation (docs.worldlabs.ai/api/pricing)."""
    tier = "draft" if model.endswith("draft") else "standard"
    key = "pano" if (kind == "image" and pano) else kind
    base = _PRICE[tier].get(key, _PRICE[tier]["image"])
    top = base + (1500 if model.endswith("plus") else 0)
    return {"min": base, "max": top, "usd_min": round(base / CREDITS_PER_USD, 2), "usd_max": round(top / CREDITS_PER_USD, 2),
            "note": ("1,250 credits per US$; marble-1.1-plus adds 0-1,500 credits for the world's size. "
                     "PLY and collider-mesh exports are free; the high-quality mesh export (3,500 credits) is not used here.")}


# -- files ----------------------------------------------------------------------------


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
        d = (Path(base) if base else project_root() / "output") / "3d_scenes" / "marble"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _world_dir(name: str, output_dir: str = "") -> Path:
    return _root_dir(output_dir) / wl._slug(name)


def _load(wd: Path) -> dict:
    f = wd / "world.json"
    if not f.exists():
        have = sorted(p.parent.name for p in wd.parent.glob("*/world.json"))
        raise AtlasError(f"no Marble world '{wd.name}' (looked in {wd.parent}); have: {', '.join(have) or 'none'}.", "args")
    return json.loads(f.read_text(encoding="utf-8"))


def _save(wd: Path, world: dict) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "world.json").write_text(json.dumps(world, indent=1), encoding="utf-8")


def _sidecar(path: Path, role: str, meta: dict) -> None:
    wl._sidecar(path, role, {"provider": "worldlabs-marble", **meta})


def _fetch_file(url: str, dest: Path) -> Path:
    try:
        r = requests.get(url, timeout=600)
    except requests.RequestException as exc:
        raise AtlasError(f"could not download a Marble file: {exc}", "api") from exc
    if r.status_code >= 400:
        raise AtlasError(f"could not download a Marble file (HTTP {r.status_code}); the link may have expired, "
                         "marble_export(name, refresh=true) asks for new ones.", "api")
    dest.write_bytes(r.content)
    return dest


# -- inputs ---------------------------------------------------------------------------


def _upload_image(ref: str) -> dict:
    """A picture as a Marble content reference: https URL as given, anything local uploaded as a media asset."""
    ref = str(ref).strip()
    if ref.startswith(("http://", "https://")):
        return {"source": "uri", "uri": ref}
    if ref.startswith("data:"):
        data, ext = base64.b64decode(ref.split(",", 1)[-1]), "png"
        name = "image.png"
    else:
        from agenty_core.tools.image_io import resolve_local_image
        path = resolve_local_image(ref)
        if not path:
            raise AtlasError(f"image not found: {ref}", "args")
        ext = _IMG_EXT.get(Path(path).suffix.lower())
        if ext:
            data = Path(path).read_bytes()
        else:
            from PIL import Image
            buf = io.BytesIO()
            Image.open(path).convert("RGB").save(buf, "PNG")
            data, ext = buf.getvalue(), "png"
        name = Path(path).name
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(name).stem)[:40] or "image"
    prep = _request("POST", "/media-assets:prepare_upload",
                    json_body={"file_name": f"{stem}.{ext}"[:64], "kind": "image", "extension": ext})
    mid = (prep.get("media_asset") or {}).get("media_asset_id")
    info = prep.get("upload_info") or {}
    if not mid or not info.get("upload_url"):
        raise AtlasError("the media-asset upload answered without an id / url", "api")
    hdrs = dict(info.get("required_headers") or {})
    try:
        r = requests.request(info.get("upload_method") or "PUT", info["upload_url"], data=data, headers=hdrs, timeout=600)
    except requests.RequestException as exc:
        raise AtlasError(f"upload to World Labs failed: {exc}", "api") from exc
    if r.status_code >= 400:
        raise AtlasError(f"upload to World Labs failed (HTTP {r.status_code})", "api")
    return {"source": "media_asset", "media_asset_id": mid}


def frame_azimuth(base_cam: dict, cam: dict) -> float:
    """Azimuth in degrees (0 front, 90 right, 180 back, 270 left) of an Atlas camera relative to the base one."""
    b, c = wl.normalize_camera(base_cam), wl.normalize_camera(cam)
    fwd_world = wl._mv(c["R"], [0.0, 0.0, -1.0])                  # RUB camera looks down -Z
    d = wl._mv(wl._tr(b["R"]), fwd_world)                          # in the base camera's frame
    return round(math.degrees(math.atan2(d[0], -d[2])) % 360.0, 1)


def pick_atlas_views(world: dict, wd: Path, limit: int) -> list[dict]:
    """Up to `limit` posed frames of an Atlas world, spread round the compass: the first picture, then each
    time the frame whose azimuth is farthest from all chosen ones. -> [{path, azimuth, id}]"""
    posed = [f for f in world.get("frames") or [] if f.get("camera") and f.get("image") and (wd / f["image"]).exists()]
    if not posed:
        raise AtlasError(f"the Atlas world '{world.get('name')}' has no posed pictures on disk", "args")
    base = posed[0]["camera"]
    rows = [{"id": f["id"], "path": str(wd / f["image"]), "azimuth": frame_azimuth(base, f["camera"])} for f in posed]
    chosen = [rows[0]]
    rest = rows[1:]
    while rest and len(chosen) < max(1, limit):
        def gap(r):
            return min(min(abs(r["azimuth"] - c["azimuth"]), 360 - abs(r["azimuth"] - c["azimuth"])) for c in chosen)
        best = max(rest, key=gap)
        if gap(best) < 5.0 and len(chosen) >= 2:
            break
        chosen.append(best)
        rest.remove(best)
    return chosen


def _find_atlas_world(name: str, output_dir: str) -> tuple[Path, dict]:
    slug = wl._slug(name)
    for root in ([Path(output_dir)] if str(output_dir or "").strip() else []) + [wl._root_dir("")]:
        wd = root / slug
        if (wd / "world.json").exists():
            return wd, json.loads((wd / "world.json").read_text(encoding="utf-8"))
    raise AtlasError(f"no Atlas world '{name}' (atlas_list shows what exists)", "args")


# -- result handling ---------------------------------------------------------------------


def _res_key(resolution: str) -> str:
    r = _RES.get(str(resolution or "full").strip().lower())
    if not r:
        raise AtlasError("resolution must be full, 500k or 100k (PLY also 150k)", "args")
    return r


def _spz_name(res: str) -> str:
    return f"splat_{res}.spz"


def _fresh_world(world: dict) -> dict:
    return _request("GET", f"/worlds/{world['world_id']}")


def _store_assets(wd: Path, world: dict, w: dict, res: str, collider: bool, refresh: bool = False) -> dict:
    """Download the SPZ at `res`, plus thumbnail and (if asked) the collider mesh; update world.json."""
    assets = w.get("assets") or {}
    spz_urls = ((assets.get("splats") or {}).get("spz_urls")) or {}
    files = world.setdefault("files", {})
    url = spz_urls.get(res)
    if res not in _SPZ_RES:
        raise AtlasError(f"no SPZ at {res}; SPZ comes as full, 500k or 100k", "args")
    if not url:
        raise AtlasError(f"this world has no SPZ at {res} (it offers: {', '.join(spz_urls) or 'none'})", "not_found")
    dest = wd / _spz_name(res)
    if refresh or not dest.exists():
        _fetch_file(url, dest)
        _sidecar(dest, f"Marble splat (SPZ, {res})", {"world": world["name"], "world_id": world.get("world_id"),
                                                        "model": world.get("model")})
    files[f"spz_{res}"] = dest.name
    if not files.get("thumbnail") and assets.get("thumbnail_url"):
        try:
            ext = Path(assets["thumbnail_url"].split("?")[0]).suffix or ".jpg"
            files["thumbnail"] = _fetch_file(assets["thumbnail_url"], wd / f"thumbnail{ext}").name
        except AtlasError:
            pass
    cm = (assets.get("mesh") or {}).get("collider_mesh_url")
    if collider:
        if not cm:
            raise AtlasError("this world has no collider mesh", "not_found")
        if refresh or not (wd / "collider.glb").exists():
            _fetch_file(cm, wd / "collider.glb")
            _sidecar(wd / "collider.glb", "Marble collider mesh (coarse, 100-200k triangles)", {"world": world["name"]})
        files["collider"] = "collider.glb"
    sem = (assets.get("splats") or {}).get("semantics_metadata") or {}
    if sem:
        world["semantics"] = {"metric_scale_factor": sem.get("metric_scale_factor"),
                              "ground_plane_offset": sem.get("ground_plane_offset")}
    for k, v in (("caption", assets.get("caption")), ("pano_available", bool((assets.get("imagery") or {}).get("pano_url")))):
        if v:
            world[k] = v
    world["available_spz"] = sorted(spz_urls)
    world["status"] = "ready"
    _save(wd, world)
    return world


def _splat_count(path: Path) -> Optional[int]:
    try:
        from agenty_core.tools.splat_formats import spz_header
        return spz_header(path)["count"]
    except Exception:  # noqa: BLE001
        return None


def _open_viewer(path: Path) -> dict:
    try:
        from agenty_core.tools.splat_train import open_splat_in_viewer
        return open_splat_in_viewer(path, None)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not open it in the bEpic viewer: {exc}"}


def _summary(wd: Path, world: dict, extra: Optional[dict] = None) -> str:
    files = {k: str(wd / v) for k, v in (world.get("files") or {}).items()}
    out = {"status": "ok", "name": world["name"], "world_dir": str(wd), "world_id": world.get("world_id"),
           "model": world.get("model"), "files": files,
           "splat": next((files[k] for k in ("spz_full_res", "spz_500k", "spz_100k") if k in files), ""),
           "semantics": world.get("semantics"), "caption": world.get("caption"),
           "marble_url": world.get("marble_url"), "operation_id": world.get("operation_id"),
           "cost": world.get("cost"),
           "axes": "raw OpenCV (x right, y down, z forward): the same Y-down frame as train_splat's .ply; the bEpic "
                   "viewer turns it upright. Scale is not metres: semantics has the factor (marble_export metric=true).",
           "next": "marble_export(name, format='ply') for a PLY; atlas_splat_to_mesh(splat_path=<name>) for a mesh; "
                   "show_in_viewer / open_in_viewer for the 3D view."}
    if extra:
        out.update(extra)
    return json.dumps({k: v for k, v in out.items() if v not in (None, "")})


def _guard(fn):
    def wrapper(*a, **kw):
        if not _key():
            return _err("No World Labs API key: set WLT_MARBLE_API_KEY in .env (a World API key from "
                        "platform.worldlabs.ai; the Atlas key is not accepted by the Marble API).", "auth")
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


def _finish(wd: Path, world: dict, op: dict, res: str, fmt: str, collider: bool, open_in_viewer: bool) -> str:
    w = op.get("response") or {}
    if not (w.get("assets") or {}).get("splats"):
        w = _fresh_world(world)
    world.update({"world_id": w.get("world_id") or world.get("world_id"), "marble_url": w.get("world_marble_url"),
                  "cost": op.get("cost"), "model": w.get("model") or world.get("model")})
    world = _store_assets(wd, world, w, res, collider)
    extra = {}
    if fmt == "ply":
        extra.update(json.loads(_export_ply(wd, world, res, False, w)))
    spz = wd / _spz_name(res)
    extra["splat_count"] = _splat_count(spz)
    if open_in_viewer:
        extra["viewer"] = _open_viewer(wd / world["files"].get("ply_" + res, spz.name) if fmt == "ply" else spz)
    return _summary(wd, world, extra)


# -- tools ----------------------------------------------------------------------------------


@tool
def marble_create_world(name: str, images: Optional[list] = None, atlas_world: str = "", prompt: str = "",
                        model: str = "marble-1.1-plus", azimuths: Optional[list] = None,
                        reconstruct_images: bool = False, max_images: int = 0, is_pano: str = "auto",
                        seed: int = -1, resolution: str = "full", format: str = "spz",
                        collider_mesh: bool = False, open_in_viewer: bool = True, dry_run: bool = False,
                        timeout_s: int = 900, output_dir: str = "") -> str:
    """Make a navigable Gaussian-splat world with World Labs Marble from one or more pictures (or a prompt).

    PAID, about 1,600 credits (US$ 1.3) for marble-1.1 / 3,100 (US$ 2.5) at most for marble-1.1-plus (1,250
    credits per US$; the World API has its own balance). Say the cost before calling; ``dry_run=true`` prints
    the request and the price without spending anything. One call = one world: do not loop. About 5 minutes.

    Inputs: ``images`` (local paths or https URLs) or ``atlas_world`` (the name of an Atlas world; the tool
    picks up to 4 posed views spread round the compass and gives each its azimuth). One picture = an image
    world; several = a multi-image world (4 pictures, 8 with ``reconstruct_images``). ``prompt`` is optional
    text that steers it (without one World Labs captions the picture). The result is a splat that needs no
    training: SPZ files (full resolution about 2 M splats, 500k, 100k), a collider mesh and a panorama.
    The SPZ is saved raw (OpenCV axes, Y down, like train_splat's .ply) and opened in the bEpic viewer.

    Args:
        name: folder name, saved under output/3d_scenes/marble/<name>/ (default <atlas_world>_marble).
        images: pictures of the same place (paths / urls). Unrelated pictures need separate worlds.
        atlas_world: use views of this Atlas world (output/3d_scenes/atlas/<name>) instead of ``images``.
        prompt: optional text prompt / description of the place.
        model: marble-1.1-plus (newest, biggest worlds), marble-1.1, marble-1.0, marble-1.0-draft (150-250
            credits, quick look). There is no "Marble 2" in this API.
        azimuths: with several ``images``: degrees round the compass for each (0 front, 90 right, 180 back,
            270 left); omit if unknown.
        reconstruct_images: multi-image only: stay close to the pictures (allows 8) instead of inventing.
        max_images: cap on pictures taken from ``atlas_world`` (default 4, or 8 with reconstruct_images).
        is_pano: single image only: "auto" (default), "true" or "false".
        seed: fixed seed (0 to 4294967295); -1 = random.
        resolution: SPZ to download now: full, 500k or 100k (others via marble_export).
        format: "spz" (default) or "ply" (adds the PLY export of the same resolution; free).
        collider_mesh: also download the collider mesh (collider.glb).
        open_in_viewer: open the splat in the viewer's 3D view when done.
        dry_run: no API call that costs anything and no upload: return the request and the price.
        timeout_s: how long to wait; on timeout the job stays recorded and marble_export(name) collects it.
        output_dir: save here instead of ComfyUI's output/3d_scenes/marble.

    Returns:
        JSON: name, world_dir, splat (path), files, world_id, semantics (metric scale), cost (credits charged),
        marble_url, viewer - or {"error", "error_kind"} (auth, balance, content_refusal, args, timeout, ...).
    """
    return _create(name, images, atlas_world, prompt, model, azimuths, reconstruct_images, max_images, is_pano, seed,
                   resolution, format, collider_mesh, open_in_viewer, dry_run, timeout_s, output_dir)


@_guard
def _create(name, images, atlas_world, prompt, model, azimuths, reconstruct_images, max_images, is_pano, seed,
            resolution, fmt, collider_mesh, open_in_viewer, dry_run, timeout_s, output_dir):
    model = str(model or "marble-1.1-plus").strip()
    if model not in MODELS:
        raise AtlasError(f"model must be one of {', '.join(MODELS)} (there is no 'Marble 2' in the World API)", "args")
    fmt = str(fmt or "spz").lower()
    if fmt not in ("spz", "ply"):
        raise AtlasError("format must be spz or ply", "args")
    res = _res_key(resolution)
    if res not in _SPZ_RES:
        raise AtlasError("resolution must be full, 500k or 100k", "args")
    if str(is_pano).lower() not in ("auto", "true", "false"):
        raise AtlasError("is_pano must be auto, true or false", "args")
    name = name or (f"{atlas_world}_marble" if atlas_world else "")
    if not wl._slug(name):
        raise AtlasError("give the world a name", "args")
    wd = _world_dir(name, output_dir)
    if (wd / "world.json").exists():
        raise AtlasError(f"a Marble world named '{wl._slug(name)}' already exists; pick another name or use "
                         "marble_export to fetch more of it", "args")
    imgs = [str(p).strip() for p in (images or []) if str(p).strip()]
    az = [None] * len(imgs)
    source = {"type": "images"}
    if atlas_world:
        if imgs:
            raise AtlasError("give images OR atlas_world, not both", "args")
        awd, aw = _find_atlas_world(atlas_world, output_dir)
        cap = int(max_images) if int(max_images or 0) > 0 else (8 if reconstruct_images else 4)
        picks = pick_atlas_views(aw, awd, min(cap, 8 if reconstruct_images else 4))
        imgs, az = [p["path"] for p in picks], [p["azimuth"] for p in picks]
        source = {"type": "atlas_world", "atlas_world": aw.get("name") or atlas_world,
                  "frames": [p["id"] for p in picks], "azimuths": az}
        prompt = prompt or aw.get("prompt_used") or aw.get("prompt") or ""
    elif azimuths:
        if len(azimuths) != len(imgs):
            raise AtlasError("azimuths must have one entry per image", "args")
        az = [float(a) for a in azimuths]
    if not imgs and not str(prompt or "").strip():
        raise AtlasError("give images, an atlas_world or a prompt", "args")
    limit = 8 if reconstruct_images else 4
    if len(imgs) > limit:
        raise AtlasError(f"{len(imgs)} pictures: Marble takes {limit} ({'8 with' if not reconstruct_images else '4 without'} "
                         "reconstruct_images)", "args")
    kind = "text" if not imgs else "image" if len(imgs) == 1 else "multi"
    est = estimate_credits(model, kind, str(is_pano).lower() == "true")

    def prompt_body(content_of):
        if kind == "text":
            wp = {"type": "text", "text_prompt": prompt}
        elif kind == "image":
            wp = {"type": "image", "image_prompt": content_of(imgs[0])}
            if str(is_pano).lower() != "auto":
                wp["is_pano"] = str(is_pano).lower() == "true"
        else:
            wp = {"type": "multi-image", "multi_image_prompt": [
                {**({"azimuth": a} if a is not None else {}), "content": content_of(p)} for p, a in zip(imgs, az)]}
            if reconstruct_images:
                wp["reconstruct_images"] = True
        if kind != "text" and str(prompt or "").strip():
            wp["text_prompt"] = prompt
        return wp

    def body_with(content_of):
        b = {"display_name": wl._slug(name)[:64], "model": model, "world_prompt": prompt_body(content_of)}
        if int(seed) >= 0:
            b["seed"] = int(seed)
        return b

    balance = credits_remaining()
    if dry_run:
        return json.dumps({"status": "dry_run", "spent": 0, "estimate": est, "credits_remaining": balance,
                           "request": body_with(lambda p: {"source": "uri", "uri": p} if str(p).startswith("http") else
                                                {"source": "media_asset", "media_asset_id": f"<uploaded from {p}>"}),
                           "input": source, "world_dir": str(wd),
                           "note": "Nothing was sent. Call again without dry_run to spend about "
                                   f"{est['min']}-{est['max']} credits."})
    if balance is not None and balance < est["min"]:
        raise AtlasError(f"The World API balance is {balance:.0f} credits; this world needs at least {est['min']} "
                         "(platform.worldlabs.ai/billing).", "balance")
    body = body_with(_upload_image)
    op = _request("POST", "/worlds:generate", json_body=body)
    world = {"version": 1, "name": wl._slug(name), "provider": "worldlabs-marble", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "status": "pending", "model": model, "operation_id": op.get("operation_id"),
             "world_id": (op.get("metadata") or {}).get("world_id"), "input": source, "prompt": prompt,
             "estimate": est, "request": {"resolution": res, "format": fmt, "collider_mesh": bool(collider_mesh),
                                           "open_in_viewer": bool(open_in_viewer)}}
    _save(wd, world)
    try:
        op = _poll(op, timeout_s)
    except AtlasTimeout as exc:
        exc.extra.setdefault("name", world["name"])
        raise
    except AtlasError:
        world["status"] = "failed"
        _save(wd, world)
        raise
    return _finish(wd, world, op, res, fmt, bool(collider_mesh), bool(open_in_viewer))


def _export_ply(wd: Path, world: dict, res: str, metric: bool, w: Optional[dict] = None, refresh: bool = False) -> str:
    """The PLY at `res`: the API's export (free, synchronous), else converted from the SPZ here."""
    from agenty_core.tools import splat_formats as sf
    sem = world.get("semantics") or {}
    out = wd / (f"splat_{res}_metric.ply" if metric else f"splat_{res}.ply")
    source = "cached"
    if refresh or not out.exists():
        raw = wd / f"splat_{res}_api.ply"
        source, note = "api", ""
        try:
            op = _request("POST", f"/worlds/{world['world_id']}:export",
                          json_body={"asset_type": "splats", "format": "ply", "resolution": res})
            op = _poll(op, 300)
            url = (op.get("response") or {}).get("url")
            if not url:
                raise AtlasError("the export answered without a url", "api")
            _fetch_file(url, raw)
        except AtlasError as exc:
            if exc.kind in ("auth", "balance", "rate_limit", "timeout"):
                raise
            source, note = "local", f"API PLY export unavailable ({exc}); converted the SPZ here"
            raw = None
        if raw is not None:
            if metric:
                props, a = sf.read_ply_any(raw)
                sf.write_ply(out, props, sf.apply_metric(props, a, sem.get("metric_scale_factor"), sem.get("ground_plane_offset")),
                             ("Marble PLY, metric scale + ground aligned by agentY",))
                raw.unlink(missing_ok=True)
            else:
                os.replace(raw, out)
        else:
            if res not in _SPZ_RES:
                raise AtlasError(f"no SPZ to convert at {res}", "args")
            spz = wd / _spz_name(res)
            if not spz.exists():
                _store_assets(wd, world, w or _fresh_world(world), res, False)
            sf.spz_to_ply(spz, out, sem if metric else None)
        _sidecar(out, f"Marble splat (PLY, {res}{', metric' if metric else ''})",
                 {"world": world["name"], "source": source, "note": note})
    world.setdefault("files", {})[("ply_" + res) + ("_metric" if metric else "")] = out.name
    _save(wd, world)
    return json.dumps({"ply": str(out), "ply_source": source})


@tool
def marble_export(name: str, format: str = "spz", resolution: str = "full", collider_mesh: bool = False,
                  metric: bool = False, refresh: bool = False, open_in_viewer: bool = False,
                  timeout_s: int = 900, output_dir: str = "") -> str:
    """Get the files of a Marble world made by marble_create_world: SPZ or PLY splats, the collider mesh.
    Free (no credits). Also collects a world whose creation timed out (it waits for the job first).

    SPZ comes at full (about 2 M splats), 500k and 100k. PLY (uncompressed, for other splat software) comes
    from World Labs' own export at full, 500k, 150k, 100k; if that is unavailable the SPZ is converted here.
    Files are raw: OpenCV axes (x right, y down, z forward), the same Y-down frame as train_splat's .ply, which
    the bEpic viewer turns upright. Other tools: a DCC that wants OpenGL axes negates Y and Z.

    Args:
        name: the Marble world (folder name under output/3d_scenes/marble).
        format: "spz" (default) or "ply".
        resolution: full, 500k, 100k (spz); also 150k (ply).
        collider_mesh: also download the coarse physics mesh (collider.glb, 100-200k triangles).
        metric: ply only: apply Marble's metric scale and ground alignment (metres, floor at y = 0).
        refresh: download again even if the file is already there (also renews expired links).
        open_in_viewer: open the result in the bEpic viewer's 3D view.
        timeout_s: how long to wait for a still-running creation.
        output_dir: where the world lives, if not ComfyUI's output/3d_scenes/marble.

    Returns:
        JSON: files (all paths), splat or ply path, ply_source (api / local), splat_count - or {"error", "error_kind"}.
    """
    return _export(name, format, resolution, collider_mesh, metric, refresh, open_in_viewer, timeout_s, output_dir)


@_guard
def _export(name, fmt, resolution, collider_mesh, metric, refresh, open_in_viewer, timeout_s, output_dir):
    fmt = str(fmt or "spz").lower()
    if fmt not in ("spz", "ply"):
        raise AtlasError("format must be spz or ply", "args")
    res = _res_key(resolution)
    if fmt == "spz" and res not in _SPZ_RES:
        raise AtlasError("SPZ comes as full, 500k or 100k", "args")
    wd = _world_dir(name, output_dir)
    world = _load(wd)
    if world.get("status") != "ready":
        if not world.get("operation_id"):
            raise AtlasError(f"the world '{world['name']}' has no job to collect", "args")
        op = _request("GET", f"/operations/{world['operation_id']}")
        try:
            op = _poll(op, timeout_s)
        except AtlasTimeout as exc:
            exc.extra.setdefault("name", world["name"])
            raise
        except AtlasError:
            world["status"] = "failed"
            _save(wd, world)
            raise
        w = op.get("response") or {}
        if not (w.get("assets") or {}).get("splats"):
            w = _request("GET", f"/worlds/{w.get('world_id') or world['world_id']}")
        world.update({"world_id": w.get("world_id") or world.get("world_id"), "marble_url": w.get("world_marble_url"),
                      "cost": op.get("cost")})
        world = _store_assets(wd, world, w, _res_key((world.get("request") or {}).get("resolution", "full")),
                              bool((world.get("request") or {}).get("collider_mesh")))
    w = _fresh_world(world)
    extra: dict = {}
    if fmt == "spz" or collider_mesh:
        world = _store_assets(wd, world, w, res if res in _SPZ_RES else "full_res", bool(collider_mesh), refresh)
    if fmt == "ply":
        if metric and not world.get("semantics"):
            world["semantics"] = {k: ((w.get("assets") or {}).get("splats") or {}).get("semantics_metadata", {}).get(k)
                                  for k in ("metric_scale_factor", "ground_plane_offset")}
        extra.update(json.loads(_export_ply(wd, world, res, bool(metric), w, refresh)))
        path = Path(extra["ply"])
    else:
        path = wd / _spz_name(res)
        extra["splat_count"] = _splat_count(path)
    if open_in_viewer:
        extra["viewer"] = _open_viewer(path)
    extra["file"] = str(path)
    return _summary(wd, world, extra)


# -- used by atlas_splat_to_mesh ---------------------------------------------------------------


def find_world(ref: str, output_dir: str = "") -> Optional[Path]:
    """The Marble world folder named by `ref` (a bare name), or None."""
    ref = str(ref or "").strip().strip('"')
    if not ref or any(c in ref for c in "/\\") or Path(ref).suffix:
        return None
    wd = _world_dir(ref, output_dir)
    return wd if (wd / "world.json").exists() else None


def _ensure_spz_for_mesh(wd: Path, output_dir: str = "") -> Path:
    _load(wd)
    for res in _SPZ_RES:
        p = wd / _spz_name(res)
        if p.exists():
            return p
    _export.__wrapped__(wd.name, "spz", "full", False, False, False, False, 900, output_dir)
    return wd / _spz_name("full_res")


def splat_for_mesh(splat_path: str, name: str, output_dir: str = "") -> tuple[Path, Path, str, str]:
    """-> (spz file to upload, folder for the result, file stem, what it was) for atlas_splat_to_mesh:
    a Marble world name, a .spz, or a .ply (converted to SPZ)."""
    wd = find_world(splat_path, output_dir)
    if wd is not None:
        if not _key():
            for res in _SPZ_RES:
                if (wd / _spz_name(res)).exists():
                    return wd / _spz_name(res), wd, "mesh", f"marble:{wd.name}"
            raise AtlasError("the Marble world has no SPZ on disk and no Marble API key to fetch it", "auth")
        return _ensure_spz_for_mesh(wd, output_dir), wd, "mesh", f"marble:{wd.name}"
    p = Path(os.path.expanduser(str(splat_path).strip().strip('"')))
    if not p.is_file():
        raise AtlasError(f"splat file not found (and no Marble world with that name): {splat_path}", "args")
    ext = p.suffix.lower()
    awd = wl._world_dir(name or p.stem, output_dir)
    if ext == ".spz":
        return p, awd, wl._slug(name or p.stem) or "mesh", str(p)
    if ext == ".ply":
        from agenty_core.tools import splat_formats as sf
        awd.mkdir(parents=True, exist_ok=True)
        out = awd / f"{wl._slug(p.stem) or 'splat'}.spz"
        try:
            sf.ply_to_spz(p, out)
        except sf.SplatFormatError as exc:
            raise AtlasError(f"cannot turn {p.name} into an SPZ: {exc}", "args") from exc
        return out, awd, wl._slug(name or p.stem) or "mesh", f"{p} (converted to SPZ)"
    raise AtlasError("splats2Mesh takes a .spz (a .ply is converted) or the name of a Marble world", "args")
