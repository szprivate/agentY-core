"""Run the generic Remotion renderer (``renderer/``) on a scene file.

The renderer is a small Node project shipped inside this package. Its
``node_modules`` are NOT in git or in the wheel: :func:`ensure_installed` runs
``npm ci`` once (Node 18+ on PATH; Remotion then downloads its headless Chrome on
the first render). The project is used in place when the package folder is
writable (an editable / venv install), otherwise it is copied to
``~/.agenty/mg_renderer``; ``AGENTY_MG_HOME`` overrides both.

``render.mjs`` prints one JSON object per line (status / progress / done / error);
anything else on stdout (Remotion's download bars) is kept only as error context.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from agenty_core.motion_graphics import poster as _poster
from agenty_core.motion_graphics import schema

logger = logging.getLogger("agenty_core.motion_graphics")

PACKAGE_RENDERER = Path(__file__).resolve().parent / "renderer"
_COPY_IGNORE = shutil.ignore_patterns("node_modules", ".render-cache", "out", "test", "__pycache__")
_install_lock = threading.Lock()


class RenderError(Exception):
    """Something to tell the agent, in words that say what to do next."""


def _writable(folder: Path) -> bool:
    try:
        probe = folder / ".write-test"
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


def runtime_dir() -> Path:
    """The folder the renderer runs from (node_modules go here)."""
    env = os.environ.get("AGENTY_MG_HOME", "").strip()
    if env:
        home = Path(env).expanduser()
        _sync_copy(home)
        return home
    if _writable(PACKAGE_RENDERER):
        return PACKAGE_RENDERER
    home = Path.home() / ".agenty" / "mg_renderer"
    _sync_copy(home)
    return home


def _sync_copy(home: Path) -> None:
    """Keep a private copy of the renderer sources up to date (node_modules are left alone)."""
    home.mkdir(parents=True, exist_ok=True)
    for name in ("src", "render.mjs", "package.json", "package-lock.json"):
        src, dst = PACKAGE_RENDERER / name, home / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_COPY_IGNORE)
        elif src.exists():
            shutil.copy2(src, dst)


def node_exe() -> str:
    exe = shutil.which("node")
    if not exe:
        raise RenderError("Node.js is not installed (or not on PATH). The motion-graphics renderer needs Node 18 or "
                          "newer: install it from nodejs.org, then run the tool again.")
    return exe


def is_installed(home: Optional[Path] = None) -> bool:
    home = home or runtime_dir()
    return (home / "node_modules" / "remotion" / "package.json").is_file() \
        and (home / "node_modules" / "@remotion" / "renderer" / "package.json").is_file()


def ensure_installed(log: Callable[[str], None] = lambda _m: None) -> Path:
    """Install the renderer's npm packages if they are missing. Returns the runtime folder."""
    home = runtime_dir()
    if is_installed(home):
        return home
    with _install_lock:
        if is_installed(home):
            return home
        node_exe()
        npm = shutil.which("npm")
        if not npm:
            raise RenderError("npm was not found next to Node.js; reinstall Node.js from nodejs.org.")
        log("installing the motion-graphics renderer (npm, one time, a few minutes)")
        cmd = [npm, "ci" if (home / "package-lock.json").is_file() else "install", "--no-audit", "--no-fund"]
        try:
            done = subprocess.run(cmd, cwd=str(home), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
        except subprocess.TimeoutExpired:
            raise RenderError("npm install of the renderer timed out after 30 minutes (network?)") from None
        if done.returncode != 0 or not is_installed(home):
            tail = (done.stderr or done.stdout or "").strip()[-600:]
            raise RenderError(f"npm install of the renderer failed ({done.returncode}): {tail}")
    return home


def _run_job(job: dict, log: Callable[[str], None], timeout: float) -> dict:
    home = ensure_installed(log)
    node = node_exe()
    fd, job_path = tempfile.mkstemp(prefix="mgjob_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(job, fh)
    events: list[dict] = []
    noise: list[str] = []
    started = time.time()
    try:
        proc = subprocess.Popen([node, str(home / "render.mjs"), job_path], cwd=str(home),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                    if not isinstance(ev, dict):
                        raise ValueError
                except ValueError:
                    noise.append(line[-300:])
                    del noise[:-25]
                    continue
                events.append(ev)
                if ev.get("event") == "status":
                    log(str(ev.get("message", "")))
                elif ev.get("event") == "progress":
                    log(f"rendering {ev.get('pct', 0)}%")
            code = proc.wait()
            proc.stdout.close()
        finally:
            timer.cancel()
    finally:
        try:
            os.unlink(job_path)
        except OSError:
            pass
    done = next((e for e in reversed(events) if e.get("event") == "done"), None)
    err = next((e for e in reversed(events) if e.get("event") == "error"), None)
    if err:
        raise RenderError(str(err.get("error")))
    if code != 0 or not done:
        if time.time() - started >= timeout:
            raise RenderError(f"the render did not finish in {int(timeout)} s and was stopped")
        raise RenderError(f"the renderer exited with {code}: " + " | ".join(noise[-6:]))
    done["seconds"] = round(time.time() - started, 1)
    return done


def _stem(path: str) -> str:
    base = os.path.basename(path)
    return base[: -len(".mgscene.json")] if base.endswith(".mgscene.json") else os.path.splitext(base)[0]


def _checked(scene_path: str) -> tuple[str, dict]:
    path = os.path.abspath(os.path.expanduser(str(scene_path or "").strip().strip('"')))
    try:
        scene = schema.load_scene(path)
    except ValueError as exc:
        raise RenderError(str(exc)) from None
    result = schema.validate_scene(scene)
    if not result["ok"]:
        first = "; ".join(f"{e['path']}: {e['message']}" for e in result["errors"][:6])
        raise RenderError(f"the scene is not valid ({len(result['errors'])} error(s)): {first}")
    return path, scene


def render_scene(scene_path: str, out_path: str = "", *, format: str = "mp4", frame_range: Optional[tuple] = None,
                 scale: float = 1.0, crf: int = 18, lossless: bool = False,
                 log: Callable[[str], None] = lambda _m: None, timeout: float = 1800, poster: bool = True) -> dict:
    """Render the scene to a video. Returns the renderer's ``done`` event plus ``path``.

    With ``poster`` (default) a ``<stem>.png`` still next to the video carries the scene (see
    :mod:`poster`); the result gets ``poster``, or ``poster_error`` when only the still failed.
    """
    path, scene = _checked(scene_path)
    if format not in ("mp4", "webm", "mov", "gif"):
        raise RenderError("format must be mp4, webm, mov or gif")
    out = out_path or os.path.join(os.path.dirname(path), f"{_stem(path)}.{format}")
    job = {"mode": "video", "scene": path, "out": os.path.abspath(out), "format": format, "scale": scale,
           "crf": int(crf), "lossless": bool(lossless)}
    if frame_range:
        job["range"] = [int(frame_range[0]), int(frame_range[1])]
    res = _run_job(job, log, timeout)
    res["path"] = res.get("out", out)
    res["duration_frames"] = scene["comp"]["duration"]
    if poster:
        try:
            res["poster"] = _write_poster_still(path, scene, res["path"], log)
        except Exception as exc:  # noqa: BLE001 - the video is done; the still is a bonus
            logger.warning("poster still failed: %s", exc)
            res["poster_error"] = f"{type(exc).__name__}: {exc}"
    return res


def _write_poster_still(scene_path: str, scene: dict, video_path: str, log: Callable[[str], None]) -> str:
    target = os.path.splitext(os.path.abspath(video_path))[0] + ".png"
    tmp_dir = tempfile.mkdtemp(prefix="mgposter_")
    try:
        stills = render_stills(scene_path, [_poster.poster_frame(scene)], os.path.join(tmp_dir, "poster"),
                               scale=1.0, log=log)
        if not stills or not os.path.isfile(stills[0].get("out", "")):
            raise RenderError("the renderer returned no still")
        _poster.write_poster(stills[0]["out"], scene, scene_path)
        shutil.move(stills[0]["out"], target)
        return target
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_stills(scene_path: str, frames: list, out_prefix: str = "", *, scale: float = 0.5,
                  log: Callable[[str], None] = lambda _m: None, timeout: float = 600) -> list:
    """Render single frames to PNG. Returns ``[{"frame", "out"}, ...]``."""
    path, scene = _checked(scene_path)
    prefix = out_prefix or os.path.join(os.path.dirname(path), "_preview", _stem(path))
    frames = [int(f) for f in frames]
    job = {"mode": "still", "scene": path, "out": os.path.abspath(prefix), "frames": frames, "scale": scale}
    res = _run_job(job, log, timeout)
    return res.get("stills", [])
