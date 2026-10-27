"""Train a Gaussian splat from posed pictures, locally, with Brush (github.com/ArthurBrussee/brush, Apache-2.0).

The usual source is a World Labs Atlas world (``atlas_create_world`` + ``atlas_expand_world`` /
``atlas_capture_around``): every view there carries its camera and a depth map, so no structure-from-motion
is needed. The world is written out as a Nerfstudio dataset (``transforms.json`` with the cameras,
``images/``, and ``points.ply`` unprojected from the depth maps as the starting splats) and Brush trains on
it. A folder that already is a Nerfstudio (``transforms.json``) or COLMAP (``sparse/``) dataset works too.

Training takes minutes to half an hour, so it runs as a background process: ``train_splat`` starts it and
waits up to ``wait_s``; ``splat_training_status`` checks on it later. When Brush is done the splat is cleaned
(huge and near-camera "floaters" removed), saved in the axes splat viewers expect (Y down, the bEpic viewer
turns it upright), with a lighter colour-only copy, a preview picture, and opened in the bEpic viewer.

Brush is downloaded once (pinned release, checksum verified) into ``<app>/tools/brush/`` unless
``BRUSH_PATH`` points at a ``brush_app`` binary.

Axes: Atlas worlds (and the Nerfstudio cameras) are right-handed, Y up, cameras looking down -Z. The
saved splat is turned 180 degrees about X (y and z negated), with its rotations and its view-dependent
colour (spherical harmonics, a sign per coefficient) turned the same way, so it is the same splat, just
stored the way 3DGS files usually are.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool
from agenty_core.paths import project_root

logger = logging.getLogger("agenty_core.splat_train")

BRUSH_VERSION = "v0.3.0"
_BRUSH_URL = "https://github.com/ArthurBrussee/brush/releases/download/{v}/{asset}"
# (system, machine) -> release asset, binary inside it, pinned sha256 (None = take the release's .sha256 file)
_BRUSH_ASSETS = {
    ("windows", "x86_64"): ("brush-app-x86_64-pc-windows-msvc.zip", "brush_app.exe",
                            "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6"),
    ("linux", "x86_64"): ("brush-app-x86_64-unknown-linux-gnu.tar.xz", "brush_app", None),
    ("darwin", "arm64"): ("brush-app-aarch64-apple-darwin.tar.xz", "brush_app", None),
}

# Sign of each spherical-harmonics coefficient (degrees 1-3, 3DGS order) under a 180-degree turn about X
# (x, y, z) -> (x, -y, -z): each real SH basis function is even or odd in (y, z).
SH_FLIP_X = [-1, -1, 1,                       # l=1: y, z, x
             -1, 1, 1, -1, 1,                 # l=2: xy, yz, 3z^2-1, xz, x^2-y^2
             -1, 1, -1, -1, 1, -1, 1]         # l=3: y(3x^2-y^2), xyz, y(5z^2-1), z(5z^2-3), x(5z^2-1), z(x^2-y^2), x(x^2-3y^2)

_PROCS: dict = {}                              # name -> Popen, for jobs started by this process


class SplatError(Exception):
    def __init__(self, message: str, kind: str = "api", **extra):
        super().__init__(message)
        self.kind, self.extra = kind, extra

    def render(self) -> str:
        return json.dumps({"error": str(self), "error_kind": self.kind, **self.extra})


# -- Brush --------------------------------------------------------------------


def _platform_key():
    system = platform.system().lower()
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return system, machine


def brush_dir() -> Path:
    return project_root() / "tools" / "brush" / BRUSH_VERSION


def _brush_search_dirs() -> list:
    """Other places a Brush install may live: the agentY app (agentY-mcp's own root has none), AGENTY_HOME, ~/.agenty."""
    rel = Path("tools") / "brush" / BRUSH_VERSION
    roots = []
    if os.environ.get("AGENTY_HOME", "").strip():
        roots.append(Path(os.environ["AGENTY_HOME"]).expanduser())
    roots.append(project_root().parent / "agentY")
    roots.append(Path.home() / ".agenty")
    return [r / rel for r in roots]


_DOWNLOAD_BUDGET_S = 120.0                     # a tool call must never hang on a stalled download


def brush_exe(download: bool = True) -> Path:
    """The Brush binary: ``BRUSH_PATH``, else the pinned release under ``<app>/tools/brush`` (fetched once)."""
    env = os.environ.get("BRUSH_PATH", "").strip().strip('"')
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise SplatError(f"BRUSH_PATH points at {p}, which is not a file", "dependency")
        return p
    key = _platform_key()
    if key not in _BRUSH_ASSETS:
        raise SplatError(f"no Brush build for {key[0]}/{key[1]}; build it from source and set BRUSH_PATH", "dependency")
    asset, binary, sha = _BRUSH_ASSETS[key]
    exe = brush_dir() / binary
    if exe.is_file():
        return exe
    for d in _brush_search_dirs():                # another checkout of the stack may already hold it
        if (d / binary).is_file():
            return d / binary
    if not download:
        raise SplatError("Brush is not installed yet", "dependency")
    _download_brush(asset, binary, sha)
    return exe


def _download_brush(asset: str, binary: str, sha: Optional[str]) -> None:
    import requests
    url = _BRUSH_URL.format(v=BRUSH_VERSION, asset=asset)
    logger.info("downloading Brush %s from %s", BRUSH_VERSION, url)
    try:
        deadline = time.monotonic() + _DOWNLOAD_BUDGET_S
        r = requests.get(url, timeout=(5, 20), stream=True)
        r.raise_for_status()
        chunks = []
        for chunk in r.iter_content(1 << 20):
            chunks.append(chunk)
            if time.monotonic() > deadline:
                r.close()
                raise TimeoutError(f"no complete download within {int(_DOWNLOAD_BUDGET_S)} s")
        data = b"".join(chunks)
        if not sha:
            s = requests.get(url + ".sha256", timeout=(5, 20))
            s.raise_for_status()
            sha = s.text.split()[0].strip().lower()
    except Exception as exc:  # noqa: BLE001
        raise SplatError(f"could not download Brush ({url}): {exc}; download it by hand and set BRUSH_PATH",
                         "dependency") from exc
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SplatError(f"Brush download checksum mismatch ({got} != {sha}); not installed", "dependency")
    d = brush_dir()
    d.mkdir(parents=True, exist_ok=True)
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members = [m for m in z.namelist() if Path(m).name == binary]
            if not members:
                raise SplatError(f"{binary} not found in {asset}", "dependency")
            (d / binary).write_bytes(z.read(members[0]))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as t:
            members = [m for m in t.getmembers() if m.isfile() and Path(m.name).name == binary]
            if not members:
                raise SplatError(f"{binary} not found in {asset}", "dependency")
            (d / binary).write_bytes(t.extractfile(members[0]).read())
    (d / binary).chmod(0o755)


# -- PLY in / out ---------------------------------------------------------------


def read_ply(path: Path):
    """A binary little-endian, float-only vertex PLY (what Brush writes) -> (property names, float32 [N, P])."""
    import numpy as np
    b = Path(path).read_bytes()
    end = b.index(b"end_header\n") + len(b"end_header\n")
    head = b[:end].decode("ascii", "replace").splitlines()
    if not any(l.startswith("format binary_little_endian") for l in head):
        raise SplatError(f"{path} is not a binary little-endian PLY", "args")
    props, n = [], 0
    for line in head:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            n = int(parts[2])
        elif parts and parts[0] == "property":
            if parts[1] != "float":
                raise SplatError(f"{path}: only float properties are supported (found {parts[1]})", "args")
            props.append(parts[-1])
    a = np.frombuffer(b, dtype="<f4", count=n * len(props), offset=end).reshape(n, len(props))
    return props, a


def write_ply(path: Path, props: list, a, comments: tuple = ()) -> None:
    import numpy as np
    head = ["ply", "format binary_little_endian 1.0", *[f"comment {c}" for c in comments],
            f"element vertex {len(a)}", *[f"property float {p}" for p in props], "end_header", ""]
    with open(path, "wb") as fh:
        fh.write("\n".join(head).encode("ascii"))
        fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes())


def sh_rest_count(props: list) -> int:
    """Higher-order SH coefficients per colour channel in a 3DGS PLY (0, 3, 8 or 15)."""
    return sum(1 for p in props if p.startswith("f_rest_")) // 3


def flip_x(props: list, a):
    """The splat turned 180 degrees about X: positions, rotations (w, x, y, z) and SH coefficients."""
    a = a.copy()
    ix = {p: i for i, p in enumerate(props)}
    a[:, ix["y"]] *= -1
    a[:, ix["z"]] *= -1
    if all(f"rot_{j}" in ix for j in range(4)):
        w, x, y, z = (a[:, ix[f"rot_{j}"]].copy() for j in range(4))
        # q' = (0, 1, 0, 0) * q
        a[:, ix["rot_0"]], a[:, ix["rot_1"]], a[:, ix["rot_2"]], a[:, ix["rot_3"]] = -x, w, -z, y
    k = sh_rest_count(props)
    for c in range(3):
        for j in range(k):
            col = ix.get(f"f_rest_{c * k + j}")
            if col is not None and SH_FLIP_X[j] < 0:
                a[:, col] *= -1
    return a


def prune(props: list, a, cameras=(), max_scale: float = 1.5, near_cam: float = 1.0, near_scale: float = 0.25,
          min_opacity: float = 0.02):
    """Drop floaters: huge splats, big splats right in front of a training camera, near-invisible ones."""
    import numpy as np
    ix = {p: i for i, p in enumerate(props)}
    P = a[:, [ix["x"], ix["y"], ix["z"]]]
    S = np.exp(a[:, [ix[f"scale_{j}"] for j in range(3)]]).max(1)
    op = 1.0 / (1.0 + np.exp(-a[:, ix["opacity"]]))
    keep = (S < max_scale) & (op > min_opacity)
    if len(cameras):
        d = np.full(len(P), np.inf, dtype="float32")
        for c in np.asarray(cameras, dtype="float32"):
            d = np.minimum(d, np.linalg.norm(P - c, axis=1))
        keep &= ~((d < near_cam) & (S > near_scale))
    return a[keep], int((~keep).sum())


def color_only(props: list, a):
    """The same splat without view-dependent colour (SH degree 0): about a quarter of the size."""
    keep = [p for p in props if not p.startswith("f_rest_")]
    return keep, a[:, [props.index(p) for p in keep]]


# -- Atlas world -> Nerfstudio dataset ---------------------------------------------


def export_atlas_dataset(world_dir: Path, out: Path, stride: int = 4, conf_min: float = 0.2,
                         eval_view: bool = True) -> dict:
    """Write an Atlas world as a Nerfstudio dataset Brush reads. Returns counts and the camera positions.

    ``eval_view``: put a copy of the first picture first, so Brush holds THAT copy out for evaluation
    (it holds out every n-th frame, starting at 0) and every real view still trains; its renders are
    the preview and the progress signal.
    """
    import numpy as np
    from PIL import Image
    from agenty_core.tools import worldlabs_atlas as wl
    world = wl._load_world(world_dir)
    frames = [f for f in world["frames"] if f.get("camera") and f.get("depth")]
    if len(frames) < 2:
        raise SplatError(f"the world has {len(frames)} view(s) with depth; a splat needs many views from different "
                         "places: grow it first with atlas_capture_around (all round) or atlas_expand_world", "args")
    (out / "images").mkdir(parents=True, exist_ok=True)
    entries, pts, cols, cams = [], [], [], []
    for f in frames:
        cam = wl.normalize_camera(f["camera"])
        K = cam["K"]
        W, H = int(K["width"]), int(K["height"])
        with Image.open(world_dir / f["image"]) as im:
            img = im.convert("RGB")
        if img.size != (W, H):
            img = img.resize((W, H), Image.LANCZOS)
        name = f"images/f{int(f['id']):03d}.png"
        img.save(out / name)
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(cam["R"])
        c2w[:3, 3] = cam["t"]
        entries.append({"file_path": name, "transform_matrix": c2w.tolist(), "fl_x": K["fx"], "fl_y": K["fy"],
                        "cx": K["cx"], "cy": K["cy"], "w": W, "h": H})
        cams.append(list(cam["t"]))
        depth = wl._decode_depth(world_dir / f["depth"], f.get("depth_log") or {})
        conf = wl._read_png_gray(world_dir / f["confidence"]) if f.get("confidence") else None
        dh, dw = depth.shape
        ys, xs = np.mgrid[stride // 2:dh:stride, stride // 2:dw:stride]
        z = depth[ys, xs]
        ok = np.isfinite(z) & (z > 1e-4)
        if conf is not None:
            ok &= conf[np.minimum(ys, conf.shape[0] - 1), np.minimum(xs, conf.shape[1] - 1)] >= conf_min
        u, v = (xs + 0.5) * W / dw, (ys + 0.5) * H / dh
        p = np.stack([(u - K["cx"]) / K["fx"] * z, -(v - K["cy"]) / K["fy"] * z, -z], -1)[ok]
        pts.append(p @ np.array(cam["R"]).T + np.array(cam["t"]))
        rgb = np.asarray(img)
        cols.append(rgb[np.minimum(v.astype(int), H - 1), np.minimum(u.astype(int), W - 1)][ok])
    if eval_view:
        shutil.copyfile(out / entries[0]["file_path"], out / "images" / "000_eval.png")
        entries.insert(0, {**entries[0], "file_path": "images/000_eval.png"})
    P = np.concatenate(pts).astype("<f4")
    C = np.concatenate(cols).astype("u1")
    rec = np.zeros(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = P.T
    rec["r"], rec["g"], rec["b"] = C.T
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(out / "points.ply", "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(rec.tobytes())
    (out / "transforms.json").write_text(json.dumps({"camera_model": "OPENCV", "ply_file_path": "points.ply",
                                                     "frames": entries}, indent=1), encoding="utf-8")
    return {"views": len(frames), "points": int(len(P)), "cameras": cams, "eval_view": bool(eval_view),
            "first_camera": entries[0]["transform_matrix"]}


# -- the job ------------------------------------------------------------------------


def _splat_root(output_dir: str = "") -> Path:
    if str(output_dir or "").strip():
        d = Path(os.path.expanduser(str(output_dir).strip().strip('"')))
    else:
        base = None
        try:
            from agenty_core.utils.comfyui_client import comfyui_output_dir
            base = comfyui_output_dir()
        except Exception:  # noqa: BLE001
            base = None
        d = (Path(base) if base else project_root() / "output") / "3d_scenes" / "splats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip()).strip("_")[:60]


def _resolve_source(source: str, output_dir: str = ""):
    """('atlas', world_dir) or ('dataset', folder)."""
    ref = str(source or "").strip().strip('"')
    if not ref:
        raise SplatError("name the source: an Atlas world, or a folder with transforms.json or a COLMAP sparse/ folder",
                         "args")
    p = Path(ref).expanduser()
    if p.is_file():
        p = p.parent
    if p.is_dir():
        if (p / "world.json").exists():
            return "atlas", p
        if (p / "transforms.json").exists() or (p / "sparse").is_dir():
            return "dataset", p
        raise SplatError(f"{p} is neither an Atlas world (world.json) nor a dataset (transforms.json / sparse/)",
                         "args")
    from agenty_core.tools.atlas_render import resolve_world_dir
    from agenty_core.tools.worldlabs_atlas import AtlasError
    try:
        wd = resolve_world_dir(ref)
    except AtlasError as exc:
        raise SplatError(str(exc), "args") from exc
    if not (wd / "world.json").exists():
        raise SplatError(f"no Atlas world or dataset called '{ref}' (looked in {wd})", "args")
    return "atlas", wd


def _job_path(jd: Path) -> Path:
    return jd / "job.json"


def _load_job(name: str, output_dir: str = "") -> tuple[Path, dict]:
    jd = _splat_root(output_dir) / _slug(name)
    f = _job_path(jd)
    if not f.exists():
        raise SplatError(f"no splat training called '{name}' (looked in {jd})", "args")
    return jd, json.loads(f.read_text(encoding="utf-8"))


def _save_job(jd: Path, job: dict) -> None:
    _job_path(jd).write_text(json.dumps(job, indent=1), encoding="utf-8")


def _alive(job: dict) -> bool:
    proc = _PROCS.get(job["name"])
    if proc is not None:
        return proc.poll() is None
    pid = job.get("pid")
    if not pid:
        return False
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE and abs(p.create_time() - job.get("pid_started", 0)) < 5
    except ImportError:
        return False                              # cannot tell: the result file decides
    except Exception:  # noqa: BLE001
        return False


def _progress(jd: Path, job: dict) -> int:
    best = 0
    for d in (jd / "brush").glob("eval_*"):
        try:
            best = max(best, int(d.name.split("_", 1)[1]))
        except ValueError:
            pass
    return best


def _result_ply(jd: Path, job: dict) -> Optional[Path]:
    p = jd / "brush" / f"export_{job['steps']}.ply"
    return p if p.exists() else None


def _log_tail(jd: Path, n: int = 1500) -> str:
    f = jd / "brush.log"
    try:
        return f.read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


def _free_comfyui_vram() -> bool:
    try:
        from agenty_core.utils.comfyui_client import get_client
        get_client().post("/free", json_data={"unload_models": True, "free_memory": True})
        return True
    except Exception as exc:  # noqa: BLE001
        logger.info("could not ask ComfyUI to free VRAM: %s", exc)
        return False


def start_training(source: str, name: str = "", steps: int = 30000, max_splats: int = 3000000,
                   sh_degree: int = 3, free_vram: bool = True, output_dir: str = "") -> tuple[Path, dict]:
    kind, src = _resolve_source(source, output_dir)
    name = _slug(name) or _slug(src.name)
    if not name:
        raise SplatError("give the splat a name", "args")
    steps = max(500, int(steps))
    exe = brush_exe()                              # first: a missing Brush fails now, not after the export
    jd = _splat_root(output_dir) / name
    if _job_path(jd).exists():
        old = json.loads(_job_path(jd).read_text(encoding="utf-8"))
        if old.get("status") == "running" and _alive(old):
            raise SplatError(f"'{name}' is already training; splat_training_status('{name}') checks on it", "busy")
    if jd.exists():
        for sub in ("dataset", "brush"):
            shutil.rmtree(jd / sub, ignore_errors=True)
    jd.mkdir(parents=True, exist_ok=True)
    job = {"name": name, "source": str(src), "source_kind": kind, "steps": steps, "max_splats": int(max_splats),
           "sh_degree": int(sh_degree), "status": "preparing", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if kind == "atlas":
        info = export_atlas_dataset(src, jd / "dataset")
        dataset = jd / "dataset"
        job.update(views=info["views"], init_points=info["points"], cameras=info["cameras"],
                   first_camera=info["first_camera"], eval_split=info["views"] + 2)
    else:
        dataset = src
        job.update(cameras=_dataset_cameras(src))
    args = [str(exe), str(dataset), "--total-steps", str(steps), "--max-splats", str(int(max_splats)),
            "--sh-degree", str(int(sh_degree)), "--export-every", str(steps), "--export-path", str(jd / "brush")]
    if job.get("eval_split"):
        args += ["--eval-split-every", str(job["eval_split"]), "--eval-every", str(max(500, steps // 10)),
                 "--eval-save-to-disk"]
    vram = _free_comfyui_vram() if free_vram else False
    (jd / "brush").mkdir(exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with open(jd / "brush.log", "wb") as log:     # the child keeps its own handle
        proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(jd), creationflags=flags)   # never the MCP server's stdio pipe
    _PROCS[name] = proc
    started = time.time()
    try:
        import psutil
        started = psutil.Process(proc.pid).create_time()
    except Exception:  # noqa: BLE001
        pass
    job.update(status="running", pid=proc.pid, pid_started=started, started=time.time(), command=args,
               comfyui_vram_freed=vram, brush=str(exe))
    _save_job(jd, job)
    return jd, job


def _dataset_cameras(folder: Path) -> list:
    """Camera positions of a Nerfstudio dataset (for the floater cleanup); none for COLMAP."""
    t = folder / "transforms.json"
    try:
        frames = json.loads(t.read_text(encoding="utf-8"))["frames"]
        return [[row[3] for row in f["transform_matrix"][:3]] for f in frames]
    except Exception:  # noqa: BLE001
        return []


def finish_training(jd: Path, job: dict, open_in_viewer: bool = True) -> dict:
    """Clean the trained splat, write it in viewer axes (+ a colour-only copy and a preview), open it."""
    raw = _result_ply(jd, job)
    if raw is None:
        raise SplatError("Brush finished without writing the splat", "execution", log_tail=_log_tail(jd))
    props, a = read_ply(raw)
    a, dropped = prune(props, a, job.get("cameras") or [])
    flipped = flip_x(props, a)
    name = job["name"]
    out = jd / f"{name}.ply"
    write_ply(out, props, flipped, ("Trained with Brush by agentY", "Axes: Y down (3DGS convention)"))
    lite_props, lite = color_only(props, flipped)
    lite_path = jd / f"{name}_lite.ply"
    write_ply(lite_path, lite_props, lite, ("Trained with Brush by agentY", "Colour only (SH degree 0)"))
    preview = None
    evals = sorted((jd / "brush").glob("eval_*"), key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else -1)
    if evals:
        pics = sorted(evals[-1].glob("*.png"))
        if pics:
            preview = jd / f"{name}_preview.png"
            shutil.copyfile(pics[0], preview)
    meta = {"source": job["source"], "steps": job["steps"], "splats": int(len(a)), "dropped_floaters": dropped,
            "trainer": f"Brush {BRUSH_VERSION}"}
    _sidecar(out, "Gaussian splat (Brush)", meta)
    if preview:
        _sidecar(preview, "Gaussian splat preview (held-out view)", meta)
        _publish(preview)
    shutil.rmtree(jd / "brush" / "tmp", ignore_errors=True)
    raw.unlink(missing_ok=True)                    # the cleaned copies replace it
    job.update(status="done", finished=time.time(), splat=str(out), splat_lite=str(lite_path),
               preview=str(preview) if preview else None, splats=int(len(a)), dropped_floaters=dropped)
    viewer = None
    if open_in_viewer:
        viewer = open_splat_in_viewer(lite_path, job)
    job["viewer"] = viewer
    _save_job(jd, job)
    return job


def _sidecar(path: Path, role: str, meta: dict) -> None:
    try:
        from agenty_core.tools import worldlabs_atlas as wl
        wl._sidecar(path, role, meta)
    except Exception:  # noqa: BLE001
        pass


def _publish(path: Path) -> None:
    try:
        from agenty_core.tools import worldlabs_atlas as wl
        wl._publish(path)
    except Exception:  # noqa: BLE001
        pass


def open_splat_in_viewer(splat: Path, job: Optional[dict] = None, focus: bool = True) -> dict:
    """Open a splat in the bEpic viewer's 3D view, looking through the first training camera."""
    from agenty_core.tools.previz_shot import camera_item, matrix_to_euler, model_item
    items = [model_item(str(splat), f"Splat: {Path(splat).stem}")]
    active = None
    m = (job or {}).get("first_camera")
    if m:
        R = [row[:3] for row in m[:3]]
        t = [m[0][3], m[1][3], m[2][3]]
        cam = camera_item([(0, t, matrix_to_euler(R), 50.0)], 50.0, (1280, 720))
        cam["id"], cam["name"], cam["tracks"] = "splatcam", "Training camera", {}
        items.append(cam)
        active = "splatcam"
    scene = {"version": 1, "items": items, "activeCamera": active, "fps": 24.0, "length": 1, "lengthSet": True}
    try:
        from agenty_core.utils.comfyui_client import get_client
        res = get_client().post("/bepic/previz_shot", json_data={
            "scene": scene, "name": f"splat_{Path(splat).stem}", "focus": bool(focus), "add_node": False})
        return {"tab": (res or {}).get("key"), "scene_file": (res or {}).get("scene_file")}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"could not open it in the bEpic viewer: {exc}"}


def _status(jd: Path, job: dict, wait_s: float, open_in_viewer: bool) -> str:
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        if job.get("status") == "done":
            return json.dumps(_report(jd, job))
        if not _alive(job):
            if _result_ply(jd, job) is not None:
                job = finish_training(jd, job, open_in_viewer)
                return json.dumps(_report(jd, job))
            job["status"] = "failed"
            _save_job(jd, job)
            raise SplatError("Brush stopped without a result (out of GPU memory? see log_tail)", "execution",
                             log_tail=_log_tail(jd), job=job["name"])
        if time.monotonic() >= deadline:
            done = _progress(jd, job)
            el = time.time() - job.get("started", time.time())
            eta = (el / done * (job["steps"] - done)) if done else None
            return json.dumps({"status": "running", "job": job["name"], "step": done, "steps": job["steps"],
                               "elapsed_s": int(el), "eta_s": int(eta) if eta else None,
                               "next": f"splat_training_status('{job['name']}', wait_s=...) when you want to check again; "
                                       "it finishes the splat and opens it in the viewer when Brush is done."})
        time.sleep(3.0)


def _report(jd: Path, job: dict) -> dict:
    return {"status": "ok", "job": job["name"], "splat": job.get("splat"), "splat_lite": job.get("splat_lite"),
            "preview": job.get("preview"), "splats": job.get("splats"), "dropped_floaters": job.get("dropped_floaters"),
            "steps": job["steps"], "train_s": int((job.get("finished") or time.time()) - job.get("started", time.time())),
            "viewer": job.get("viewer"), "folder": str(jd),
            "note": "The .ply is a standard 3DGS file (Y down): it opens in the bEpic viewer, SuperSplat, Spark, "
                    "Unreal/Unity splat plugins. _lite is colour-only and about 4x smaller."}


def _guard(fn):
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except SplatError as exc:
            return exc.render()
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", fn.__name__)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "error_kind": "api"})
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


@_guard
def _train(source, name, steps, max_splats, sh_degree, wait_s, free_vram, open_in_viewer, output_dir):
    jd, job = start_training(source, name, steps, max_splats, sh_degree, free_vram, output_dir)
    out = json.loads(_status(jd, job, wait_s, open_in_viewer))
    out.setdefault("dataset", {"views": job.get("views"), "init_points": job.get("init_points")})
    return json.dumps(out)


@_guard
def _check(name, wait_s, open_in_viewer, output_dir):
    jd, job = _load_job(name, output_dir)
    return _status(jd, job, wait_s, open_in_viewer)


@tool
def train_splat(source: str, name: str = "", steps: int = 30000, max_splats: int = 3000000, sh_degree: int = 3,
                wait_s: int = 600, free_vram: bool = True, open_in_viewer: bool = True, output_dir: str = "") -> str:
    """Train a Gaussian splat (an explorable 3D scene) from posed pictures, locally on the GPU with Brush.

    The source is usually an Atlas world: its views already carry cameras and depth. A splat needs MANY
    views from different places and directions: a world with only its first picture gives a flat shell.
    For a whole room or place, first run ``atlas_capture_around`` (views all round); for a corridor or a
    path, ``atlas_expand_world``. Free of API cost; takes ~10 min (25 views) to ~30 min (40+ views) on a
    big GPU and frees ComfyUI's VRAM first. Runs in the background: if it is not done within ``wait_s``
    you get status "running" and check later with ``splat_training_status``.

    Args:
        source: an Atlas world (name, or its folder), or a folder holding a Nerfstudio dataset
            (transforms.json) or a COLMAP one (sparse/).
        name: name of the splat (default: the source's name); saved under output/3d_scenes/splats/<name>/.
        steps: training steps (30000 = full quality; 7000 for a quick look).
        max_splats: upper limit of Gaussians (3,000,000 suits a room; more = finer and heavier files).
        sh_degree: view-dependent colour detail, 0-3 (3 = shiny surfaces look right; 0 = 4x smaller).
        wait_s: how long this call waits for the result before returning "running".
        free_vram: ask ComfyUI to unload its models first (training shares the GPU).
        open_in_viewer: open the result in the bEpic viewer's 3D view when done.
        output_dir: save under this folder instead of ComfyUI's output/3d_scenes/splats.

    Returns:
        JSON: status "ok" with splat (.ply, standard 3DGS), splat_lite (colour only), preview (a render of
        a held-out view, also shown in chat), splats, viewer tab - or status "running" with step / eta_s -
        or {"error", "error_kind"} (args, dependency: Brush could not be installed, execution, busy).
    """
    from agenty_core.utils import execution_mode
    refusal = execution_mode.local_only_refusal("train_splat")
    if refusal:                       # cloud mode: Brush needs this machine's GPU
        return json.dumps(refusal)
    return _train(source, name, steps, max_splats, sh_degree, wait_s, free_vram, open_in_viewer, output_dir)


@tool
def splat_training_status(name: str, wait_s: int = 0, open_in_viewer: bool = True, output_dir: str = "") -> str:
    """Check on (and finish) a splat started by ``train_splat``: progress while Brush runs; when it is done,
    cleans and saves the splat, makes the preview and opens it in the viewer.

    Args:
        name: the splat's name (as train_splat returned it in "job").
        wait_s: wait up to this many seconds for it to finish.
        open_in_viewer: open the result in the bEpic viewer when it finishes here.
        output_dir: where the splat lives, if not ComfyUI's output/3d_scenes/splats.

    Returns:
        JSON like train_splat.
    """
    return _check(name, wait_s, open_in_viewer, output_dir)
