"""World Labs Atlas (Marble 2 beta Developer API): posed views, depth and a 3D scene from a picture or a prompt.

What the API is (docs: https://atlas-beta.worldlabs.ai/docs/api-reference). It is NOT a
one-call "make me a world" service. It is a set of view-synthesis tasks, each an async
*operation*:

* ``atlasTextToImage``   prompt -> image (+ optional metric depth and camera)
* ``images2PosedRGBD``   image(s) -> the same image(s) with an estimated camera and a depth map
* ``atlasGenerate``      posed RGBD context + new target cameras -> new views (+ optional depth)
* ``atlasMasked`` / ``atlasChisel``  edit objects / depth-guided generation (not wrapped here)
* ``splats2Mesh``        a Gaussian-splat ``.spz`` -> a GLB mesh

So "a world" here is built on our side: posed RGBD views become a textured depth mesh
(one GLB, in the first picture's camera frame: the camera at the origin looking down -Z,
exactly what ``render_camera_move(model=...)`` expects). ``atlas_expand_world`` asks Atlas to
invent more views at new cameras and grows the same mesh. Nothing here is a splat: Atlas
returns images, depth EXRs and (splats2Mesh only) a GLB, all of which the bEpic viewer
already loads.

Output goes to ``<ComfyUI output>/3d_scenes/atlas/<name>/`` (``world.json`` records the
cameras, assets and operations; every file has an ``.agenty.json`` sidecar).

Key: ``WLT_API_KEY`` (``WORLDLABS_API_KEY`` also read). Endpoint: ``WLT_API_BASE_URL``
(default ``https://api.atlas-beta.worldlabs.ai/api/v2``). Never logged.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import struct
import time
import uuid
from pathlib import Path
from typing import Optional

import requests

from agenty_core._compat import tool
from agenty_core.paths import project_root

logger = logging.getLogger("agenty_core.worldlabs_atlas")

_DEFAULT_BASE = "https://api.atlas-beta.worldlabs.ai/api/v2"
_MAX_INLINE_BYTES = 10 * 1024 * 1024 - 4096   # TaskAsset.base64 / asset base64: < 10 MiB decoded
_WAIT_SLICE_S = 30                            # the server caps :wait at 30 s
_sleep = time.sleep                           # tests replace this
_GRID = (1280, 720)                           # the model's camera grid

_TASK_I2P = "images2PosedRGBD"
_TASK_T2I = "atlasTextToImage"
_TASK_GEN = "atlasGenerate"
_TASK_S2M = "splats2Mesh"


# -- errors -------------------------------------------------------------------


def _err(message: str, kind: str, **extra) -> str:
    out = {"error": message, "error_kind": kind}
    if kind == "content_refusal":
        out["content_refusal"] = True
    out.update({k: v for k, v in extra.items() if v not in (None, "")})
    return json.dumps(out)


class AtlasError(Exception):
    """An API or operation failure, already classified."""

    def __init__(self, message: str, kind: str, **extra):
        super().__init__(message)
        self.kind, self.extra = kind, extra

    def render(self) -> str:
        return _err(str(self), self.kind, **self.extra)


class AtlasTimeout(AtlasError):
    pass


def _api_key() -> str:
    from agenty_core.utils.secrets import get_secret
    return get_secret("WLT_API_KEY") or get_secret("WORLDLABS_API_KEY")


def _base() -> str:
    return (os.environ.get("WLT_API_BASE_URL") or _DEFAULT_BASE).strip().rstrip("/")


def _headers(extra: Optional[dict] = None) -> dict:
    h = {"WLT-Api-Key": _api_key(), "Content-Type": "application/json"}
    h.update(extra or {})
    return h


def _body_of(r) -> dict:
    try:
        b = r.json()
        return b if isinstance(b, dict) else {"detail": b}
    except Exception:  # noqa: BLE001
        return {"detail": (getattr(r, "text", "") or "")[:300]}


def _http_error(r) -> AtlasError:
    """Classify a non-2xx answer (docs: 'Errors and retries')."""
    st, body = r.status_code, _body_of(r)
    code = str(body.get("code") or "")
    detail = body.get("detail")
    msg = detail if isinstance(detail, str) else json.dumps(detail)[:300] if detail else ""
    retry_after = (getattr(r, "headers", None) or {}).get("Retry-After")
    if st == 401:
        return AtlasError("World Labs rejected the API key (missing, unknown, expired or revoked). "
                          f"Set WLT_API_KEY in .env. {msg}", "auth")
    if st == 402:
        if code == "SPEND_LIMIT_EXCEEDED":
            return AtlasError("The World Labs monthly spend limit is reached: raise it or wait for the "
                              "reset (platform > Settings > Billing).", "spend_limit", code=code)
        return AtlasError("The World Labs account is out of credits ("
                          f"{code or '402'}): add credits (platform > Settings > Billing). {msg}", "balance", code=code)
    if st == 403:
        return AtlasError("The API key lacks a scope. It needs tasks.create, operations.read, "
                          f"assets.create and assets.read. {msg}", "scope")
    if st == 404:
        return AtlasError("Not found, or this Atlas task is not enabled for the account (it is a beta "
                          f"entitlement: ask World Labs for access). {msg}", "not_found")
    if st == 409:
        return AtlasError(f"Idempotency key / id collision. {msg}", "api")
    if st == 422:
        return AtlasError(f"World Labs rejected the request shape: {msg}", "args")
    if st == 429:
        return AtlasError(f"World Labs rate or concurrency limit ({code or '429'}); retry after "
                          f"{retry_after or 'a minute'} s. {msg}", "rate_limit", code=code, retry_after=retry_after)
    if st == 503:
        return AtlasError(f"World Labs capacity is temporarily unavailable. {msg}", "rate_limit", retry_after=retry_after)
    return AtlasError(f"World Labs error {st} {code}: {msg}", "api")


def _operation_error(op: dict) -> AtlasError:
    err = op.get("error") or {}
    reason = ""
    for d in err.get("details") or []:
        if isinstance(d, dict) and d.get("reason"):
            reason = str(d["reason"])
    msg = err.get("message") or "the task failed"
    if reason == "CONTENT_POLICY" or err.get("code") == 3:
        return AtlasError(f"World Labs declined the content under its policy ({msg}). Change the prompt or "
                          "the picture; a retry is declined again.", "content_refusal", operation_id=op.get("id"))
    if reason == "EXECUTION_TIMEOUT" or err.get("code") == 4:
        return AtlasError(f"The Atlas task exceeded its execution deadline ({msg}). A retry is a new task.",
                          "execution_timeout", operation_id=op.get("id"))
    return AtlasError(f"The Atlas task failed ({reason or err.get('code')}): {msg}", "execution",
                      operation_id=op.get("id"))


def _request(method: str, path: str, *, json_body=None, params=None, idem: str = "", timeout: float = 120.0) -> dict:
    """One API call with a few polite retries on 429 / 503. Returns the JSON body."""
    url = f"{_base()}{path}"
    extra = {"Idempotency-Key": idem} if idem else None
    for attempt in range(4):
        try:
            r = requests.request(method, url, headers=_headers(extra), json=json_body, params=params,
                                 timeout=timeout)
        except requests.Timeout as exc:
            raise AtlasError("World Labs did not answer in time.", "timeout") from exc
        except requests.RequestException as exc:
            raise AtlasError(f"could not reach World Labs: {exc}", "api") from exc
        if r.status_code in (429, 503) and attempt < 3:
            ra = (getattr(r, "headers", None) or {}).get("Retry-After")
            try:
                wait = min(float(ra), 60.0)
            except (TypeError, ValueError):
                wait = 2.0 * (attempt + 1)
            if _body_of(r).get("code") != "CONCURRENCY_LIMIT_EXCEEDED":
                _sleep(wait)
                continue
        if r.status_code >= 400:
            raise _http_error(r)
        return _body_of(r)
    raise AtlasError("World Labs kept refusing (rate limit).", "rate_limit")  # pragma: no cover


# -- operations ---------------------------------------------------------------


def _submit(task: str, body: dict) -> dict:
    return _request("POST", f"/tasks:{task}", json_body=body, idem=f"agenty-{uuid.uuid4().hex}")


def _wait(op: dict, timeout_s: float, on_progress=None) -> dict:
    """Block (server-side :wait slices, then GET) until ``done``. Raises AtlasTimeout / AtlasError."""
    deadline = time.monotonic() + float(timeout_s)
    delay = 2.0
    while not op.get("done"):
        if time.monotonic() >= deadline:
            raise AtlasTimeout(f"still running after {int(timeout_s)} s; collect it with atlas_fetch "
                               "(operations are kept 30 days).", "timeout", operation_id=op.get("id"))
        left = max(1, min(_WAIT_SLICE_S, int(deadline - time.monotonic())))
        try:
            op = _request("POST", f"/operations/{op['id']}:wait", json_body={"timeoutSecs": left})
        except AtlasError as exc:
            if exc.kind != "not_found":   # a fresh operation can 404 briefly
                raise
            _sleep(delay)
            delay = min(delay * 1.5, 10.0)
            continue
        if on_progress:
            on_progress(op)
        if not op.get("done"):
            _sleep(delay)
            delay = min(delay * 1.5, 10.0)
    if op.get("error"):
        raise _operation_error(op)
    return op


def _usage_of(op: dict) -> dict:
    """Whatever the operation says about cost. The API documents no per-task usage field (billing is on
    the platform's Usage page, 1,000 credits per US$), so this is only a pass-through of metadata keys
    that look like it."""
    md = op.get("metadata") or {}
    got = {k: v for k, v in md.items() if re.search(r"usage|credit|cost|price|bill", k, re.I)}
    return got


# -- files --------------------------------------------------------------------


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
        d = (Path(base) if base else project_root() / "output") / "3d_scenes" / "atlas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip()).strip("_")
    return s[:60]


def _world_dir(name: str, output_dir: str = "") -> Path:
    return _root_dir(output_dir) / _slug(name)


def _sidecar(path: Path, role: str, meta: dict) -> None:
    try:
        body = {"role": role, "made_by": "agentY", "provider": "worldlabs-atlas",
                "when": time.strftime("%Y-%m-%dT%H:%M:%S"), **meta}
        Path(str(path) + ".agenty.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _publish(path: Path) -> None:
    try:
        from agenty_core.tools.image_io import _publish as pub
        pub(str(path))
    except Exception:  # noqa: BLE001
        pass


def _download(asset: dict, dest: Path) -> Path:
    """Fetch a task output (`{assetId, url}`) to *dest*. The read URL is signed: no API key on it. An
    expired one is replaced through :createReadUrl."""
    if not isinstance(asset, dict):
        raise AtlasError("the response named no file where one was expected", "api")
    if asset.get("base64"):
        raw = asset["base64"].split(",", 1)[-1]
        dest.write_bytes(base64.b64decode(raw))
        return dest
    url = asset.get("url")
    for attempt in (0, 1):
        if not url or attempt == 1:
            if not asset.get("assetId"):
                raise AtlasError("the response named no downloadable file", "api")
            url = _request("POST", f"/assets/{asset['assetId']}:createReadUrl").get("readUrl")
        try:
            r = requests.get(url, timeout=300)
        except requests.RequestException as exc:
            raise AtlasError(f"could not download an Atlas output: {exc}", "api") from exc
        if r.status_code < 400:
            dest.write_bytes(r.content)
            return dest
        if attempt == 1:
            raise AtlasError(f"could not download an Atlas output (HTTP {r.status_code})", "api")
    raise AtlasError("download failed", "api")  # pragma: no cover


def _image_input(ref: str) -> dict:
    """A TaskAsset for a picture: a URL as given, a local file inline as base64 (< 10 MiB)."""
    ref = str(ref).strip()
    if ref.startswith(("http://", "https://")):
        return {"url": ref}
    if ref.startswith("data:"):
        return {"base64": ref}
    from agenty_core.tools.image_io import resolve_local_image
    path = resolve_local_image(ref)
    if not path:
        raise AtlasError(f"image not found: {ref}", "args")
    size = os.path.getsize(path)
    if size > _MAX_INLINE_BYTES:
        raise AtlasError(f"{path} is {size / 1e6:.1f} MB; Atlas takes inline images under 10 MiB. "
                         "Downsize it first (image-downsize skill).", "args")
    return {"base64": base64.b64encode(Path(path).read_bytes()).decode("ascii")}


def _upload_file(path: str, content_type: str) -> str:
    """Upload a (large) local file as a project asset; returns its assetId."""
    p = Path(path)
    data = p.read_bytes()
    created = _request("POST", "/assets", json_body={"asset": {"displayName": p.name[:250]}})
    aid = created.get("id")
    if not aid:
        raise AtlasError("asset creation returned no id", "api")
    grant = _request("POST", f"/assets/{aid}:createUploadUrl",
                     json_body={"contentType": content_type, "contentLength": len(data)})
    hdrs = dict(grant.get("headers") or {})
    hdrs.setdefault("Content-Type", content_type)
    try:
        r = requests.request(grant.get("method") or "PUT", grant["uploadUrl"], data=data, headers=hdrs, timeout=600)
    except requests.RequestException as exc:
        raise AtlasError(f"upload to World Labs failed: {exc}", "api") from exc
    if r.status_code >= 400:
        raise AtlasError(f"upload to World Labs failed (HTTP {r.status_code})", "api")
    _request("POST", f"/assets/{aid}:completeUpload", json_body={"sizeBytes": len(data)})
    return aid


# -- depth --------------------------------------------------------------------


def _read_exr(path: Path):
    """float32 [H, W] (first channel) of an EXR. Needs the ``OpenEXR`` wheel, or an OpenCV built with
    EXR support; row 0 is the top row in both."""
    import numpy as np
    try:
        import OpenEXR  # type: ignore
        f = OpenEXR.File(str(path))
        chans = f.channels()
        key = "Y" if "Y" in chans else "R" if "R" in chans else sorted(chans)[0]
        return np.asarray(chans[key].pixels, dtype=np.float32)
    except ImportError:
        pass
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2  # type: ignore
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is not None:
            return (a[..., 0] if a.ndim == 3 else a).astype("float32")
    except ImportError:
        pass
    raise AtlasError("cannot read the depth EXR: install the OpenEXR package "
                     "(pip install OpenEXR) into agentY's environment.", "dependency")


def _read_png_gray(path: Path):
    import numpy as np
    from PIL import Image
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim == 3:
        a = a[..., 0]
    top = 65535.0 if a.dtype == np.uint16 or im.mode.startswith("I;16") else 255.0
    return a.astype("float32") / top


def _decode_depth(depth_file: Optional[Path], depth_meta: dict):
    """Linear depth [H, W] float32 from a saved depth file (EXR linear, or inverted-log PNG)."""
    import numpy as np
    if depth_file is None:
        return None
    if str(depth_file).lower().endswith(".exr"):
        return _read_exr(depth_file)
    v = _read_png_gray(depth_file)           # white = zMin, black = zMax, log-spaced
    zmin, zmax = float(depth_meta["zMin"]), float(depth_meta["zMax"])
    return np.exp(math.log(zmax) + v * (math.log(zmin) - math.log(zmax))).astype("float32")


# -- camera math (RUB cameras, camera-to-world, quaternion XYZW) ---------------


def _quat_to_mat(q):
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _mat_to_quat(m):
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s]


def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def _tr(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def normalize_camera(cam: dict) -> dict:
    """An API PinholeCamera -> ``{"R": 3x3, "t": [3], "K": {...}}`` in RUB (the viewer's convention)."""
    ex, K = cam["extrinsics"], cam["intrinsics"]
    R = _quat_to_mat(ex["quaternion"])
    if str(ex.get("coordinateSystem", "rub")).lower() == "rdf":          # right-down-forward -> right-up-back
        R = _mm(R, [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    return {"R": R, "t": [float(v) for v in ex["position"]], "K": {k: K[k] for k in ("fx", "fy", "cx", "cy", "width", "height")}}


def camera_to_api(R, t, K: dict) -> dict:
    return {"extrinsics": {"position": [round(float(v), 6) for v in t],
                           "quaternion": [round(float(v), 8) for v in _mat_to_quat(R)],
                           "coordinateSystem": "rub"},
            "intrinsics": {"fx": K["fx"], "fy": K["fy"], "cx": K["cx"], "cy": K["cy"],
                           "width": int(K["width"]), "height": int(K["height"])}}


# -- depth grid -> textured mesh ----------------------------------------------


def frame_mesh(image_size, K: dict, depth, conf, cam: dict, base: dict, stride: int = 4,
               max_jump: float = 0.15, conf_min: float = 0.1, max_depth: float = 0.0):
    """Unproject one posed RGBD view into (positions[N,3], uvs[N,2], triangles[M,3]) in the *base* frame
    (the first picture's camera: origin, looking down -Z).

    ``depth`` is read as z-depth along the optical axis. Quads whose neighbouring depths jump by more
    than ``max_jump`` (relative) are cut, so the mesh does not stretch across silhouettes.
    """
    import numpy as np
    H, W = depth.shape
    s = max(1, int(stride))
    ys, xs = np.arange(s // 2, H, s), np.arange(s // 2, W, s)
    d = depth[np.ix_(ys, xs)].astype("float64")
    ok = np.isfinite(d) & (d > 1e-6)
    if max_depth and max_depth > 0:
        ok &= d <= max_depth
    if conf is not None:
        c = conf[np.ix_(np.minimum(ys, conf.shape[0] - 1), np.minimum(xs, conf.shape[1] - 1))]
        ok &= c >= conf_min
    d = np.where(ok, d, 1.0)
    sx, sy = K["width"] / W, K["height"] / H            # depth grid vs. camera grid
    u, v = np.meshgrid((xs + 0.5) * sx, (ys + 0.5) * sy)
    x = (u - K["cx"]) / K["fx"] * d
    y = -(v - K["cy"]) / K["fy"] * d
    p_cam = np.stack([x, y, -d], axis=-1)                # RUB: camera looks down -Z
    R, t = np.array(cam["R"]), np.array(cam["t"])
    R0, t0 = np.array(base["R"]), np.array(base["t"])
    local = (R0.T @ ((p_cam.reshape(-1, 3) @ R.T + t) - t0).T).T     # world -> the base camera's frame
    uv = np.stack([(xs[None, :] + 0.5) / W * np.ones((len(ys), 1)), (ys[:, None] + 0.5) / H * np.ones((1, len(xs)))], -1)
    gh, gw = d.shape
    idx = np.arange(gh * gw).reshape(gh, gw)
    a, b, c_, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    okq = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    zs = np.stack([d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]])
    okq &= (zs.max(0) - zs.min(0)) <= max_jump * zs.min(0)
    qa, qb, qc, qe = a[okq], b[okq], c_[okq], e[okq]
    tris = np.concatenate([np.stack([qa, qc, qb], -1), np.stack([qb, qc, qe], -1)])
    # keep only the vertices a triangle uses
    used = np.unique(tris)
    remap = -np.ones(gh * gw, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return (local[used].astype("float32"), uv.reshape(-1, 2)[used].astype("float32"),
            remap[tris].astype("uint32"))


def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path: Path, parts: list[dict]) -> dict:
    """A GLB of textured meshes. Each part: ``{"positions","uvs","indices","image": bytes, "mime", "name"}``.
    Materials are unlit (the picture's own colours); the viewer's clay / normal shading still applies."""
    import numpy as np
    blob = bytearray()
    views, accs, meshes, mats, texs, imgs, nodes = [], [], [], [], [], [], []

    def add_view(raw: bytes, target: Optional[int] = None) -> int:
        off = len(blob)
        blob.extend(_pad4(raw))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    for i, p in enumerate(parts):
        pos, uv, ind = (np.ascontiguousarray(p["positions"], dtype="<f4"), np.ascontiguousarray(p["uvs"], dtype="<f4"),
                        np.ascontiguousarray(p["indices"], dtype="<u4").reshape(-1))
        a_pos = len(accs)
        accs.append({"bufferView": add_view(pos.tobytes(), 34962), "componentType": 5126, "count": len(pos),
                     "type": "VEC3", "min": [float(v) for v in pos.min(0)], "max": [float(v) for v in pos.max(0)]})
        a_uv = len(accs)
        accs.append({"bufferView": add_view(uv.tobytes(), 34962), "componentType": 5126, "count": len(uv), "type": "VEC2"})
        a_ix = len(accs)
        accs.append({"bufferView": add_view(ind.tobytes(), 34963), "componentType": 5125, "count": len(ind), "type": "SCALAR"})
        imgs.append({"bufferView": add_view(p["image"]), "mimeType": p.get("mime", "image/png")})
        texs.append({"source": len(imgs) - 1, "sampler": 0})
        mats.append({"name": p.get("name", f"view{i}"), "doubleSided": True,
                     "pbrMetallicRoughness": {"baseColorTexture": {"index": len(texs) - 1}, "metallicFactor": 0.0,
                                              "roughnessFactor": 1.0},
                     "extensions": {"KHR_materials_unlit": {}}})
        meshes.append({"name": p.get("name", f"view{i}"),
                       "primitives": [{"attributes": {"POSITION": a_pos, "TEXCOORD_0": a_uv}, "indices": a_ix,
                                       "material": len(mats) - 1, "mode": 4}]})
        nodes.append({"mesh": len(meshes) - 1, "name": p.get("name", f"view{i}")})
    gltf = {"asset": {"version": "2.0", "generator": "agentY worldlabs_atlas"},
            "extensionsUsed": ["KHR_materials_unlit"], "scene": 0,
            "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "materials": mats,
            "textures": texs, "images": imgs, "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}],
            "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    js = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    binary = bytes(blob)
    total = 12 + 8 + len(js) + 8 + len(binary)
    out = bytearray(struct.pack("<III", 0x46546C67, 2, total))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(binary), 0x004E4942) + binary
    path.write_bytes(bytes(out))
    return {"vertices": int(sum(len(p["positions"]) for p in parts)),
            "triangles": int(sum(len(p["indices"]) for p in parts)), "bytes": total}


# -- the world on disk --------------------------------------------------------


def _load_world(wd: Path) -> dict:
    f = wd / "world.json"
    if not f.exists():
        raise AtlasError(f"no Atlas world '{wd.name}' (looked in {wd}); atlas_list shows what exists.", "args")
    return json.loads(f.read_text(encoding="utf-8"))


def _save_world(wd: Path, world: dict) -> None:
    (wd / "world.json").write_text(json.dumps(world, indent=1), encoding="utf-8")


def _canon_camera(cam: Optional[dict]) -> Optional[dict]:
    """A response camera, stored in RUB form (what normalize_camera / camera_to_api speak)."""
    if not cam:
        return None
    n = normalize_camera(cam)
    return camera_to_api(n["R"], n["t"], n["K"])


def _frame_entry(world: dict, wd: Path, i: int, fr: dict, src: str, op_id: str) -> dict:
    """Download one response frame into the world folder; returns its world.json record."""
    stem = f"{i:03d}"
    img = _download(fr["imageAsset"], wd / f"view_{stem}.png")
    entry = {"id": i, "image": img.name, "source": src, "operation": op_id,
             "camera": _canon_camera(fr.get("camera")),
             "image_asset": (fr["imageAsset"] or {}).get("assetId")}
    d = fr.get("depth")
    if d:
        if d.get("depthAsset"):
            entry["depth"] = _download(d["depthAsset"], wd / f"depth_{stem}.exr").name
            entry["depth_asset"] = d["depthAsset"].get("assetId")
        elif d.get("logDepthAsset"):
            entry["depth"] = _download(d["logDepthAsset"], wd / f"depth_{stem}.png").name
            entry["depth_log"] = {"zMin": d.get("zMin"), "zMax": d.get("zMax")}
            entry["depth_asset"] = d["logDepthAsset"].get("assetId")
        if d.get("confidenceAsset"):
            entry["confidence"] = _download(d["confidenceAsset"], wd / f"conf_{stem}.png").name
    for k in ("image", "depth", "confidence"):
        if entry.get(k):
            _sidecar(wd / entry[k], f"Atlas {src} {k}", {"world": world["name"], "operation": op_id})
    return entry


def build_world_mesh(wd: Path, world: dict, stride: int = 4, max_jump: float = 0.15,
                     conf_min: float = 0.1, max_depth: float = 0.0) -> dict:
    """Rebuild ``world.glb`` from every frame that has a depth map."""
    import numpy as np
    frames = [f for f in world["frames"] if f.get("depth") and f.get("camera")]
    if not frames:
        raise AtlasError("none of the world's frames has a depth map, so there is nothing to mesh", "args")
    base = normalize_camera(frames[0]["camera"])
    parts = []
    for fr in frames:
        cam = normalize_camera(fr["camera"])
        depth = _decode_depth(wd / fr["depth"], fr.get("depth_log") or {})
        conf = _read_png_gray(wd / fr["confidence"]) if fr.get("confidence") else None
        img_path = wd / fr["image"]
        w, h = _image_size(img_path)
        pos, uv, tri = frame_mesh((w, h), cam["K"], depth, conf, cam, base, stride, max_jump, conf_min, max_depth)
        if len(tri) == 0:
            continue
        parts.append({"positions": pos, "uvs": uv, "indices": tri, "image": img_path.read_bytes(),
                      "mime": "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png",
                      "name": f"view_{fr['id']:03d}"})
    if not parts:
        raise AtlasError("the depth maps gave no usable surface (all depth invalid or cut by the confidence / "
                         "jump limits); try a lower conf_min or a larger max_jump", "args")
    glb = wd / "world.glb"
    stats = write_glb(glb, parts)
    world["glb"] = glb.name
    world["mesh"] = {**stats, "stride": stride, "max_jump": max_jump, "conf_min": conf_min,
                     "frame": "first view's camera: origin, looking down -Z, Y up (metres as Atlas scales it)"}
    _sidecar(glb, "Atlas world mesh (textured depth mesh, first-camera frame)",
             {"world": world["name"], "frames": len(parts), **stats})
    _publish(glb)
    return stats


def _image_size(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _summary(wd: Path, world: dict, op_id: str, usage: dict, extra: Optional[dict] = None) -> str:
    glb = str(wd / world["glb"]) if world.get("glb") else ""
    out = {"status": "ok", "name": world["name"], "world_dir": str(wd), "glb": glb,
           "frames": [{"id": f["id"], "image": str(wd / f["image"]), "has_depth": bool(f.get("depth"))}
                      for f in world["frames"]],
           "mesh": world.get("mesh"), "prompt_used": world.get("prompt_used"), "operation_id": op_id,
           "usage": usage or None,
           "usage_note": "Atlas reports no per-task cost; credits (1,000 per US$) show on the platform's Usage page.",
           "next": ("render_camera_move(name, model=<glb>, moves=[...]) flies a camera through this mesh; "
                    "atlas_expand_world adds views the pictures never showed.") if glb else ""}
    if extra:
        out.update(extra)
    return json.dumps(out)


# -- pending jobs (resumable) -------------------------------------------------


def _pending_path(wd: Path, op_id: str) -> Path:
    return wd / "pending" / f"{op_id}.json"


def _save_pending(wd: Path, op: dict, job: dict) -> None:
    p = _pending_path(wd, op["id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(job, indent=1), encoding="utf-8")


def _find_pending(op_id: str, output_dir: str = "") -> tuple[Optional[Path], Optional[dict]]:
    for f in _root_dir(output_dir).glob(f"*/pending/{op_id}.json"):
        return f.parent.parent, json.loads(f.read_text(encoding="utf-8"))
    return None, None


# -- finishing a job ----------------------------------------------------------


def _finish(job: dict, op: dict, output_dir: str = "") -> str:
    kind, wd = job["kind"], Path(job["world_dir"])
    resp = op.get("response") or {}
    op_id = op.get("id", "")
    usage = _usage_of(op)
    if kind == "splat_mesh":
        glb = wd / f"{job['stem']}.glb"
        _download(resp.get("mesh") or {}, glb)
        _sidecar(glb, "Atlas splat mesh (splats2Mesh)", {"source_splat": job.get("source"), "operation": op_id})
        _publish(glb)
        _pending_path(wd, op_id).unlink(missing_ok=True)
        return json.dumps({"status": "ok", "glb": str(glb), "operation_id": op_id, "usage": usage or None,
                           "anchor_positions": (resp.get("anchorPositions") or [])[:16],
                           "next": "render_camera_move(name, model=<glb>, moves=[...]); the GLB keeps the splat's own "
                                   "axes, so rotate with model_scale/viewer if it stands on its side."})
    frames = resp.get("frames") or []
    if not frames:
        raise AtlasError("the operation succeeded but returned no frames", "api", operation_id=op_id)
    wd.mkdir(parents=True, exist_ok=True)
    if kind == "create":
        world = {"version": 1, "name": job["name"], "provider": "worldlabs-atlas", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "prompt": job.get("prompt", ""), "prompt_used": resp.get("promptUsed"), "frames": [], "operations": []}
        src = job["source"]
    else:
        world = _load_world(wd)
        src = "generated"
        if resp.get("promptUsed") and not world.get("prompt_used"):
            world["prompt_used"] = resp["promptUsed"]
    first = len(world["frames"])
    for j, fr in enumerate(frames):
        if not fr.get("camera") and job.get("cameras"):
            fr = {**fr, "camera": job["cameras"][j]}
        world["frames"].append(_frame_entry(world, wd, first + j, fr, src, op_id))
    world["operations"].append({"id": op_id, "task": job["task"], "when": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if kind == "create" and src == "text":
        for fr in world["frames"]:
            if not fr.get("depth"):
                raise AtlasError("Atlas returned the picture without depth; the world cannot be meshed.", "api",
                                 operation_id=op_id)
    if job.get("build_mesh", True):
        build_world_mesh(wd, world, **job.get("mesh_opts", {}))
    _save_world(wd, world)
    _pending_path(wd, op_id).unlink(missing_ok=True)
    return _summary(wd, world, op_id, usage)


def _run(job: dict, op: dict, timeout_s: float) -> str:
    """Persist the job (so a timeout is resumable), wait, finish."""
    wd = Path(job["world_dir"])
    wd.mkdir(parents=True, exist_ok=True)
    _save_pending(wd, op, job)
    try:
        op = _wait(op, timeout_s)
        return _finish(job, op)
    except AtlasTimeout as exc:
        exc.extra.setdefault("operation_id", op.get("id"))
        raise


def _guard(fn):
    def wrapper(*a, **kw):
        if not _api_key():
            return _err("WLT_API_KEY is not set (create a project API key on the World Labs platform: "
                        "scopes tasks.create, operations.read, assets.create, assets.read; put it in .env).", "auth")
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


# -- tools --------------------------------------------------------------------


@tool
def atlas_create_world(name: str, prompt: str = "", image_paths: Optional[list] = None,
                       aspect_ratio: str = "16:9", mesh_detail: int = 4, timeout_s: int = 600,
                       output_dir: str = "") -> str:
    """Start a 3D scene with World Labs Atlas, from a picture (or several) or from a text prompt.

    With ``image_paths`` Atlas estimates each picture's camera and a depth map (images2PosedRGBD);
    with only a ``prompt`` it first paints a world-like image with depth (atlasTextToImage). The
    posed views become a textured depth mesh, ``world.glb``, in the first picture's camera frame
    (camera at the origin looking down -Z) - ready for ``render_camera_move(model=<glb>)`` and the
    bEpic viewer's 3D tab. Several pictures given together must show the SAME place with
    overlapping views (Atlas reconstructs them together); unrelated pictures need separate worlds.
    Billed to the user's World Labs credits (``WLT_API_KEY``); a few seconds to minutes.

    Args:
        name: name of the scene (a-z, 0-9, _ -); saved under output/3d_scenes/atlas/<name>/.
        prompt: with no pictures: the environment to make (place, layout, materials, light, a wide
            eye-level view; no people, text or close-ups). With pictures: stored as the scene
            description for atlas_expand_world.
        image_paths: local paths or https URLs of the pictures (sharp, wide or medium views).
        aspect_ratio: for a prompt-only scene: 16:9, 9:16, 4:3, 3:4 or 1:1.
        mesh_detail: depth-grid stride in pixels for the mesh (4 = about 58k vertices per view;
            2 = finer and 4x heavier; 8 = light).
        timeout_s: how long to wait; on timeout the error carries operation_id for atlas_fetch.
        output_dir: save here instead of ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON: name, world_dir, glb, frames, mesh stats, prompt_used, operation_id, usage (Atlas
        reports no per-task cost) - or {"error", "error_kind"} with error_kind one of auth, scope,
        balance, spend_limit, not_found (task not enabled: beta access), rate_limit, content_refusal,
        execution, execution_timeout, timeout (use atlas_fetch with operation_id), args, dependency, api.
    """
    return _create(name, prompt, image_paths, aspect_ratio, mesh_detail, timeout_s, output_dir)


@_guard
def _create(name, prompt, image_paths, aspect_ratio, mesh_detail, timeout_s, output_dir):
    if not _slug(name):
        raise AtlasError("give the scene a name", "args")
    imgs = [p for p in (image_paths or []) if str(p).strip()]
    if not imgs and not str(prompt or "").strip():
        raise AtlasError("give pictures (image_paths) or a prompt", "args")
    wd = _world_dir(name, output_dir)
    if (wd / "world.json").exists():
        raise AtlasError(f"a world named '{_slug(name)}' already exists; pick another name, or use "
                         "atlas_expand_world to add to it", "args")
    mesh_opts = {"stride": max(1, int(mesh_detail))}
    if imgs:
        body = {"frames": [{"imageAsset": _image_input(p)} for p in imgs], "targetResolution": list(_GRID)}
        op = _submit(_TASK_I2P, body)
        job = {"kind": "create", "task": _TASK_I2P, "source": "posed", "name": _slug(name), "world_dir": str(wd),
               "prompt": prompt, "mesh_opts": mesh_opts}
    else:
        body = {"prompt": prompt, "aspectRatio": aspect_ratio if aspect_ratio in ("16:9", "9:16", "4:3", "3:4", "1:1") else "16:9",
                "returnDepth": True, "numSamples": 1}
        op = _submit(_TASK_T2I, body)
        job = {"kind": "create", "task": _TASK_T2I, "source": "text", "name": _slug(name), "world_dir": str(wd),
               "prompt": prompt, "mesh_opts": mesh_opts}
    return _run(job, op, timeout_s)


def _target_cameras(world: dict, moves: list, fov: float):
    """Atlas target cameras (world frame) for each move, relative to the first picture's camera."""
    from agenty_core.tools.previz_shot import camera_pose_keys, euler_to_matrix
    first = next((f for f in world["frames"] if f.get("camera")), None)
    if not first:
        raise AtlasError("the world has no posed frame to move from", "args")
    base = normalize_camera(first["camera"])
    k0 = base["K"]
    gw, gh = _GRID                      # targets must sit on the model's 1280 x 720 grid (a 1024 x 578
    sx, sy = gw / k0["width"], gh / k0["height"]   # text-to-image picture is rescaled to it)
    K = {"fx": k0["fx"] * sx, "fy": k0["fy"] * sy, "cx": gw / 2.0, "cy": gh / 2.0, "width": gw, "height": gh}
    if fov and fov > 0:
        fy = (gh / 2.0) / math.tan(math.radians(fov) / 2.0)
        K.update(fy=fy, fx=fy)
    keys = camera_pose_keys([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], moves, 24.0, None)
    cams = []
    for _f, pos, rot, _fv in keys:
        R = _mm(base["R"], euler_to_matrix(rot))
        t = [a + b for a, b in zip(_mv(base["R"], pos), base["t"])]
        cams.append(camera_to_api(R, t, K))
    return cams


@tool
def atlas_expand_world(name: str, moves: list[dict], prompt: str = "", fov: float = 0.0,
                       context: Optional[list] = None, mesh_detail: int = 4, timeout_s: int = 600,
                       output_dir: str = "") -> str:
    """Grow an Atlas scene: have Atlas invent new views at new cameras (atlasGenerate) and add them
    to the mesh, so a camera move can go further than the pictures show.

    Each move is ONE new view, placed like ``render_camera_move`` places keyframes - relative to the
    first picture's camera: ``{"forward": m, "right": m, "up": m, "yaw": deg, "pitch": deg, "roll": deg}``
    (omitted = 0; the "t" key is not needed). The views need different camera positions: a pure turn
    on the spot cannot give depth, so include some forward/right/up. Stay a believable step from
    the existing views (a few metres); chain calls to go further. Costs World Labs credits per view.

    Args:
        name: an existing scene made by atlas_create_world.
        moves: the new views, 1 to 8 (e.g. [{"forward":2,"yaw":25},{"forward":4,"right":-1.5,"yaw":-20}]).
        prompt: scene description; default: the one Atlas used before (faster).
        fov: vertical field of view in degrees for the new views; 0 = the first picture's.
        context: frame ids to give Atlas as context; default the first and the latest 3.
        mesh_detail: depth-grid stride for the rebuilt mesh.
        timeout_s: how long to wait; on timeout the error carries operation_id for atlas_fetch.
        output_dir: where the scene lives, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON like atlas_create_world (the GLB now holds the new views too), or {"error", "error_kind"}.
    """
    return _expand(name, moves, prompt, fov, context, mesh_detail, timeout_s, output_dir)


@_guard
def _expand(name, moves, prompt, fov, context, mesh_detail, timeout_s, output_dir):
    wd = _world_dir(name, output_dir)
    world = _load_world(wd)
    if not moves or len(moves) > 8:
        raise AtlasError("give 1 to 8 moves", "args")
    posed = [f for f in world["frames"] if f.get("depth") and f.get("camera")]
    if context:
        ctx = [f for f in posed if f["id"] in {int(c) for c in context}]
    else:
        ctx = posed[:1] + [f for f in posed[-3:] if f is not posed[0]]
    if not ctx:
        raise AtlasError("no context frames with depth to give Atlas", "args")
    cams = _target_cameras(world, [{**m, "t": 0.0} for m in moves], fov)
    cframes = []
    for f in ctx:
        d_asset = ({"assetId": f["depth_asset"]} if f.get("depth_asset") else None)
        if not d_asset:
            raise AtlasError(f"frame {f['id']} has no stored depth asset (made outside Atlas?)", "args")
        depth = ({"logDepthAsset": d_asset, **f["depth_log"]} if f.get("depth_log") else {"depthAsset": d_asset})
        cframes.append({"imageAsset": ({"assetId": f["image_asset"]} if f.get("image_asset") else _image_input(str(wd / f["image"]))),
                        "depth": depth, "camera": f["camera"]})
    use_prompt = prompt or world.get("prompt_used") or world.get("prompt") or None
    body = {"contextFrames": cframes, "targetCameras": cams, "returnDepth": True,
            "enhancePrompt": not bool(use_prompt and not prompt)}
    if use_prompt:
        body["prompt"] = use_prompt
    op = _submit(_TASK_GEN, body)
    job = {"kind": "expand", "task": _TASK_GEN, "name": world["name"], "world_dir": str(wd), "cameras": cams,
           "mesh_opts": {"stride": max(1, int(mesh_detail))}}
    return _run(job, op, timeout_s)


@tool
def atlas_rebuild_world_mesh(name: str, mesh_detail: int = 4, max_jump: float = 0.15, conf_min: float = 0.1,
                             max_depth: float = 0.0, output_dir: str = "") -> str:
    """Rebuild an Atlas scene's mesh from its saved depth maps (no API call, no cost): finer or lighter,
    or with the sky / far haze cut off.

    Args:
        name: an existing scene.
        mesh_detail: depth-grid stride in pixels (2 fine, 4 default, 8 light).
        max_jump: cut triangles where neighbouring depths differ by more than this fraction (smaller
            = more holes at silhouettes, larger = more stretched skins).
        conf_min: drop depth whose confidence (0-1) is below this.
        max_depth: drop everything farther than this many scene units (0 = keep all); use it to cut a sky.
        output_dir: where the scene lives, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON like atlas_create_world, or {"error", "error_kind"}.
    """
    return _rebuild(name, mesh_detail, max_jump, conf_min, max_depth, output_dir)


def _rebuild(name, mesh_detail, max_jump, conf_min, max_depth, output_dir):
    try:
        wd = _world_dir(name, output_dir)
        world = _load_world(wd)
        build_world_mesh(wd, world, stride=max(1, int(mesh_detail)), max_jump=float(max_jump),
                         conf_min=float(conf_min), max_depth=float(max_depth))
        _save_world(wd, world)
        return _summary(wd, world, "", {})
    except AtlasError as exc:
        return exc.render()
    except Exception as exc:  # noqa: BLE001
        logger.exception("atlas_rebuild_world_mesh failed")
        return _err(f"{type(exc).__name__}: {exc}", "api")


@tool
def atlas_splat_to_mesh(splat_path: str, name: str = "", texture_mode: str = "image_texture",
                        target_faces: int = 0, timeout_s: int = 900, output_dir: str = "") -> str:
    """Turn a Gaussian-splat scene (.spz, e.g. a World Labs Marble export) into a GLB mesh with
    Atlas's splats2Mesh, so it can be flown through with ``render_camera_move(model=...)``.

    Args:
        splat_path: local path of the .spz file (uploaded as a project asset).
        name: folder name under output/3d_scenes/atlas/ (default: the file's name).
        texture_mode: "image_texture" (UV atlas, best), "vertex_color" or "none".
        target_faces: decimate to at most this many faces (0 = keep all).
        timeout_s: how long to wait; on timeout the error carries operation_id for atlas_fetch.
        output_dir: save here instead of ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON {"glb", "operation_id", "anchor_positions", "usage"} or {"error", "error_kind"}.
    """
    return _splat_mesh(splat_path, name, texture_mode, target_faces, timeout_s, output_dir)


@_guard
def _splat_mesh(splat_path, name, texture_mode, target_faces, timeout_s, output_dir):
    p = Path(os.path.expanduser(str(splat_path).strip().strip('"')))
    if not p.is_file():
        raise AtlasError(f"splat file not found: {splat_path}", "args")
    if p.suffix.lower() != ".spz":
        raise AtlasError("splats2Mesh takes a .spz splat file", "args")
    if texture_mode not in ("image_texture", "vertex_color", "none"):
        raise AtlasError("texture_mode must be image_texture, vertex_color or none", "args")
    wd = _world_dir(name or p.stem, output_dir)
    aid = _upload_file(str(p), "application/octet-stream")
    body = {"splatAsset": {"assetId": aid}, "textureMode": texture_mode}
    if target_faces and int(target_faces) > 0:
        body["targetFaces"] = int(target_faces)
    op = _submit(_TASK_S2M, body)
    job = {"kind": "splat_mesh", "task": _TASK_S2M, "world_dir": str(wd), "stem": _slug(name or p.stem) or "mesh",
           "source": str(p)}
    return _run(job, op, timeout_s)


@tool
def atlas_fetch(operation_id: str, output_dir: str = "", timeout_s: int = 120) -> str:
    """Collect an Atlas job that atlas_create_world / atlas_expand_world / atlas_splat_to_mesh timed
    out on (operations are kept 30 days), or just look at one operation's state.

    Args:
        operation_id: the ``operation_id`` from the timeout error (op_...).
        output_dir: where the scene lives, if not ComfyUI's output/3d_scenes/atlas.
        timeout_s: how long to wait if it is still running.

    Returns:
        The same JSON as the tool that started it; if still running, a ``timeout`` error again; if the
        job was not started from this machine (no saved record), the operation's task and state.
    """
    return _fetch(operation_id, output_dir, timeout_s)


@_guard
def _fetch(operation_id, output_dir, timeout_s):
    op_id = str(operation_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", op_id):
        raise AtlasError("operation_id looks malformed", "args")
    op = _request("GET", f"/operations/{op_id}")
    wd, job = _find_pending(op_id, output_dir)
    if job is None:
        state = "failed" if op.get("error") else "done" if op.get("done") else "running"
        out = {"status": state, "operation_id": op_id, "task": (op.get("metadata") or {}).get("task"),
               "note": "no saved job record for this operation on this machine, so nothing was downloaded"}
        if op.get("error"):
            out["error"] = _operation_error(op).args[0]
        return json.dumps(out)
    op = _wait(op, timeout_s)
    return _finish(job, op)


@tool
def atlas_list(what: str = "worlds", task: str = "", status: str = "", limit: int = 20,
               output_dir: str = "") -> str:
    """List Atlas scenes on this machine, or the operations of the World Labs project.

    Args:
        what: "worlds" (scenes saved under output/3d_scenes/atlas) or "operations" (the project's
            recent API jobs, newest first - also those started elsewhere).
        task: operations only: filter by task, e.g. "atlasGenerate", "images2PosedRGBD".
        status: operations only: "RUNNING", "SUCCEEDED" or "FAILED".
        limit: how many operations (max 100).
        output_dir: where scenes live, if not ComfyUI's output/3d_scenes/atlas.

    Returns:
        JSON {"worlds": [...]} (name, glb, frame count, prompt) or {"operations": [...]} (id, task, done, error).
    """
    return _list(what, task, status, limit, output_dir)


def _list(what, task, status, limit, output_dir):
    what = (what or "worlds").strip().lower()
    if what == "worlds":
        rows = []
        for f in sorted(_root_dir(output_dir).glob("*/world.json")):
            try:
                w = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            rows.append({"name": w.get("name"), "glb": str(f.parent / w["glb"]) if w.get("glb") else "",
                         "frames": len(w.get("frames") or []), "prompt": (w.get("prompt") or "")[:120],
                         "created": w.get("created"), "dir": str(f.parent)})
        pend = [p.stem for p in _root_dir(output_dir).glob("*/pending/*.json")]
        return json.dumps({"worlds": rows, "pending_operations": pend})
    if what != "operations":
        return _err("what must be 'worlds' or 'operations'", "args")
    if not _api_key():
        return _err("WLT_API_KEY is not set (put it in .env).", "auth")
    try:
        params = {"pageSize": max(1, min(int(limit), 100))}
        if task:
            params["task"] = task
        if status:
            params["status"] = status.upper()
        body = _request("GET", "/operations", params=params)
    except AtlasError as exc:
        return exc.render()
    ops = [{"id": o.get("id"), "task": (o.get("metadata") or {}).get("task"), "done": o.get("done"),
            "error": (o.get("error") or {}).get("message"), "created": o.get("createdAtSecs")}
           for o in body.get("operations") or []]
    return json.dumps({"operations": ops})
