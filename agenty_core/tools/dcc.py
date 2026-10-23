"""Build a model in a 3D app from one script file, and check its proportions against a photo by numbers.

Two tools, one adapter per app (Blender, Maya, Houdini):

* ``dcc_build``            run the WHOLE build script file in the app, optionally save the scene and render a still.
* ``dcc_silhouette_check`` project key points of the model through the scene camera and compare them with where the
                           same points sit in the reference photo. Pure numbers, no images.

Two modes:

* ``live``  talk to the app the user has open. Blender: the Blender Lab MCP add-on's socket (localhost:9876,
            JSON ``{"type":"execute","code":...}`` + NUL). Maya: its commandPort (7001, python source type).
            Houdini has no live mode here, see below.
* ``batch`` start a fresh headless process: ``blender -b``, ``mayapy``, ``hython``. The build script runs, the scene
            is saved to a file the tool returns, and later batch calls (check, render) open that file.

Houdini live: not supported. The Houdini MCP bridge (houdinimcp, port 9877) is a third-party plug-in whose
protocol is its own and is not installed here, and ``hrpyc`` needs rpyc inside Houdini's Python and a server
started by hand in the user's session. Neither is something this tool can rely on, so Houdini is batch only.

Coordinates are always the app's OWN world coordinates (Blender Z up, Maya and Houdini Y up); nothing is
converted, so the build script and the check points must use the same convention.

The projection maths (``project_point`` and the ``intrinsics_*`` converters) is plain Python and does not need
the app: every adapter returns its camera as raw numbers plus the app's own native projection, and the results
are cross-checked against the pure function.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool

APPS = ("blender", "maya", "houdini")
MODES = ("live", "batch")
_EXT = {"blender": ".blend", "maya": ".ma", "houdini": ".hip"}
_RESULT_MARK = "__DCC_RESULT__"
_DRIVERS = Path(__file__).with_name("dcc_drivers")

# ---------------------------------------------------------------------------------------------------------------
# Projection maths (pure)
# ---------------------------------------------------------------------------------------------------------------
# A camera, in the app's own world coordinates:
#   position           world position of the camera
#   right, up, back    the camera's local axes in world coordinates (unit vectors). The camera looks along -back.
#   fx, fy             focal length / film width, focal length / film height  (the film as the FRAME shows it,
#                      after the app's sensor-fit rules and the render aspect)
#   shift_x, shift_y   lens shift as fractions of the frame width / height. Positive moves the FRAME right / up,
#                      so the subject moves left / down in the picture (Blender's sign).
# Output: x (0 left .. 1 right), y measured from the TOP (0 top .. 1 bottom), in_front.


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def project_point(point, cam: dict) -> tuple:
    """World point -> (x, y_from_top, in_front) as fractions of the camera frame."""
    d = [point[i] - cam["position"][i] for i in range(3)]
    cx, cy, depth = _dot(d, cam["right"]), _dot(d, cam["up"]), -_dot(d, cam["back"])
    in_front = depth > 1e-9
    z = depth if abs(depth) > 1e-12 else 1e-12
    x = 0.5 + cam["fx"] * cx / z - cam.get("shift_x", 0.0)
    y_bottom = 0.5 + cam["fy"] * cy / z - cam.get("shift_y", 0.0)
    return x, 1.0 - y_bottom, in_front


def intrinsics_blender(c: dict) -> dict:
    """Blender camera data -> fx, fy, shift. Follows BKE_camera_sensor_fit: AUTO fits the larger side."""
    aw = c["res_x"] * c.get("pixel_aspect_x", 1.0)
    ah = c["res_y"] * c.get("pixel_aspect_y", 1.0)
    declared = c.get("sensor_fit", "AUTO")
    fit = declared if declared != "AUTO" else ("HORIZONTAL" if aw >= ah else "VERTICAL")
    sensor = c["sensor_height_mm"] if declared == "VERTICAL" else c["sensor_width_mm"]
    if fit == "HORIZONTAL":
        film_w, film_h, fit_dim = sensor, sensor * ah / aw, sensor
    else:
        film_w, film_h, fit_dim = sensor * aw / ah, sensor, sensor
    return {"fx": c["focal_mm"] / film_w, "fy": c["focal_mm"] / film_h,
            "shift_x": c.get("shift_x", 0.0) * fit_dim / film_w,
            "shift_y": c.get("shift_y", 0.0) * fit_dim / film_h}


def intrinsics_maya(c: dict) -> dict:
    """Maya camera shape -> fx, fy, shift. filmFit 0 fill, 1 horizontal, 2 vertical, 3 overscan; film in inches."""
    dev = c["device_aspect"]
    film_aspect = c["film_w_in"] / c["film_h_in"]
    fit = int(c.get("film_fit", 1))
    if fit == 0:
        fit = 1 if dev >= film_aspect else 2
    elif fit == 3:
        fit = 2 if dev >= film_aspect else 1
    if fit == 1:
        film_w = c["film_w_in"] * 25.4
        film_h = film_w / dev
    else:
        film_h = c["film_h_in"] * 25.4
        film_w = film_h * dev
    # The camera's `overscan` attribute is left out: Maya's own rendering frustum ignores it too.
    return {"fx": c["focal_mm"] / film_w, "fy": c["focal_mm"] / film_h,
            "shift_x": c.get("film_offset_x_in", 0.0) * 25.4 / film_w * _MAYA_SHIFT_SIGN[0],
            "shift_y": c.get("film_offset_y_in", 0.0) * 25.4 / film_h * _MAYA_SHIFT_SIGN[1]}


# Maya film offset vs the frame shift of this module's convention: the same sign (checked against Maya's own
# MFnCamera.getRenderingFrustum in mayapy, see the integration test).
_MAYA_SHIFT_SIGN = (1.0, 1.0)


def intrinsics_houdini(c: dict) -> dict:
    """Houdini camera -> fx, fy, shift. The aperture is the frame width; the window offset is in frame widths."""
    film_w = c["aperture_mm"] * c.get("win_size_x", 1.0)
    film_h = c["aperture_mm"] * c["res_y"] / (c["res_x"] * c.get("pixel_aspect", 1.0)) * c.get("win_size_y", 1.0)
    return {"fx": c["focal_mm"] / film_w, "fy": c["focal_mm"] / film_h,
            "shift_x": c.get("win_x", 0.0) / c.get("win_size_x", 1.0),
            "shift_y": c.get("win_y", 0.0) / c.get("win_size_y", 1.0)}


_INTRINSICS = {"blender": intrinsics_blender, "maya": intrinsics_maya, "houdini": intrinsics_houdini}


def camera_from_raw(app: str, raw: dict) -> dict:
    """The adapter's raw camera numbers -> the canonical camera ``project_point`` takes."""
    cam = {k: raw[k] for k in ("position", "right", "up", "back")}
    cam.update(_INTRINSICS[app](raw))
    return cam


# ---------------------------------------------------------------------------------------------------------------
# Result shaping (pure)
# ---------------------------------------------------------------------------------------------------------------


def _hint(dx: float, dy: float, in_front: bool) -> str:
    if not in_front:
        return "behind the camera: the camera faces away from it, or the camera is placed on the wrong side of the model"
    if abs(dy) >= abs(dx):
        if dy < 0:
            return "projected %.2f too high: model too tall or camera too low (or lens shift moves the frame down)" % -dy
        return "projected %.2f too low: model too short or camera too high (or lens shift moves the frame up)" % dy
    if dx < 0:
        return "projected %.2f too far left: part sits too far left or too wide, or the camera is yawed / shifted" % -dx
    return "projected %.2f too far right: part sits too far right or too wide, or the camera is yawed / shifted" % dx


def shape_check(points: list, projected: list, tolerance: float) -> dict:
    """Compare projections with the photo's positions. ``projected`` is [(x, y_top, in_front)] per point."""
    rows = []
    for i, (p, (x, y, front)) in enumerate(zip(points, projected)):
        row = {"index": i, "name": p["name"], "world": p["world"], "projected": [round(x, 4), round(y, 4)],
               "in_front": bool(front)}
        t = p.get("target")
        if t is not None:
            dx, dy = x - t[0], y - t[1]
            dist = math.hypot(dx, dy)
            ok = bool(front) and dist <= tolerance
            row.update(target=t, dx=round(dx, 4), dy=round(dy, 4), distance=round(dist, 4), **{"pass": ok})
            row["hint"] = "" if ok else _hint(dx, dy, bool(front))
        rows.append(row)
    judged = [r for r in rows if "distance" in r]
    judged.sort(key=lambda r: (r["pass"], -r["distance"]))       # failures first, worst first
    unjudged = [r for r in rows if "distance" not in r]
    failed = [r for r in judged if not r["pass"]]
    out = {"tolerance": tolerance, "overall_pass": bool(judged) and not failed, "points_checked": len(judged),
           "failed": len(failed), "points": judged + unjudged}
    if failed:
        out["worst"] = [{"name": r["name"], "distance": r["distance"], "hint": r["hint"]} for r in failed]
    if len(judged) >= 3 and failed:
        mx = sum(r["dx"] for r in judged) / len(judged)
        my = sum(r["dy"] for r in judged) / len(judged)
        if math.hypot(mx, my) > tolerance * 0.7 and all(
                math.hypot(r["dx"] - mx, r["dy"] - my) <= tolerance for r in judged if r["in_front"]):
            out["overall_hint"] = ("every point is off by about (%+.3f, %+.3f) the same way: this is the camera "
                                   "(position, aim or lens shift), not the model's proportions" % (mx, my))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Finding the apps
# ---------------------------------------------------------------------------------------------------------------

_ENV = {"blender": "BLENDER_PATH", "maya": "MAYAPY_PATH", "houdini": "HYTHON_PATH"}
_PATH_NAMES = {"blender": ("blender",), "maya": ("mayapy",), "houdini": ("hython",)}
_GLOBS = {
    "blender": ["C:/Program Files/Blender Foundation/Blender */blender.exe",
                "/Applications/Blender.app/Contents/MacOS/Blender", "/usr/bin/blender", "/snap/bin/blender"],
    "maya": ["C:/Program Files/Autodesk/Maya*/bin/mayapy.exe",
             "/Applications/Autodesk/maya*/Maya.app/Contents/bin/mayapy", "/usr/autodesk/maya*/bin/mayapy"],
    "houdini": ["C:/Program Files/Side Effects Software/Houdini */bin/hython.exe",
                "/opt/hfs*/bin/hython",
                "/Applications/Houdini/Houdini*/Frameworks/Houdini.framework/Versions/Current/Resources/bin/hython"],
}


def _version_key(path: str):
    return [int(n) for n in re.findall(r"\d+", path)]


def find_app(app: str) -> Optional[str]:
    """The executable for batch mode: $BLENDER_PATH / $MAYAPY_PATH / $HYTHON_PATH, then PATH, then the usual
    install folders (the newest version wins). None when it is not installed."""
    env = os.environ.get(_ENV[app], "").strip().strip('"')
    if env and os.path.isfile(env):
        return env
    for n in _PATH_NAMES[app]:
        hit = shutil.which(n)
        if hit:
            return hit
    found = []
    for pattern in _GLOBS[app]:
        found += [p for p in glob.glob(pattern) if os.path.isfile(p)]
    return max(found, key=_version_key) if found else None


def _live_endpoint(app: str) -> tuple:
    if app == "blender":
        return os.environ.get("BLENDER_MCP_HOST", "localhost"), int(os.environ.get("BLENDER_MCP_PORT", "9876"))
    return os.environ.get("MAYA_COMMANDPORT_HOST", "localhost"), int(os.environ.get("MAYA_COMMANDPORT_PORT", "7001"))


def port_open(app: str, timeout: float = 1.0) -> bool:
    host, port = _live_endpoint(app)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------------------------
# Running a job in the app
# ---------------------------------------------------------------------------------------------------------------


class DccError(Exception):
    pass


def driver_source(app: str) -> str:
    """The text that runs inside the app: shared helpers + that app's driver (defines ``run(job)``)."""
    return (_DRIVERS / "_common.py").read_text(encoding="utf-8") + "\n\n" + \
        (_DRIVERS / f"{app}_driver.py").read_text(encoding="utf-8")


_BATCH_TAIL = '''
import sys as _sys, json as _json
_job = _json.load(open(_sys.argv[-1], encoding="utf-8"))
_res = run(_job)
with open(_job["result_path"], "w", encoding="utf-8") as _f:
    _json.dump(_res, _f)
'''


def _batch_wrapper(app: str) -> str:
    head, tail = "", _BATCH_TAIL
    if app == "maya":
        head = "import maya.standalone\nmaya.standalone.initialize(name='python')\n"
        tail += "\n_sys.stdout.flush()\n_sys.stderr.flush()\nimport os as _os\n_os._exit(0)\n"
    return head + driver_source(app) + "\n" + tail


def _live_wrapper(app: str, job: dict) -> str:
    src = driver_source(app)
    body = (f"__g = {{'__name__': '__dcc__'}}\n"
            f"exec(compile({src!r}, '<dcc_driver>', 'exec'), __g)\n"
            f"import json as __j\n"
            f"__r = __j.dumps(__g['run'](__j.loads({json.dumps(job)!r})))\n")
    if app == "maya":
        return body + "import builtins as __b\n__b.__dcc_result = __r\n"
    return body + f"print({_RESULT_MARK!r} + __r)\n"


def _socket_roundtrip(host: str, port: int, payload: bytes, timeout: float) -> bytes:
    with socket.create_connection((host, port), timeout=10) as s:
        s.settimeout(timeout)
        s.sendall(payload)
        buf = bytearray()
        while b"\0" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
    return bytes(buf).partition(b"\0")[0]


def _run_live(app: str, job: dict, timeout: float) -> dict:
    host, port = _live_endpoint(app)
    job = dict(job, live=True)
    code = _live_wrapper(app, job)
    try:
        if app == "blender":
            raw = _socket_roundtrip(host, port, json.dumps(
                {"type": "execute", "code": code, "strict_json": False}).encode("utf-8") + b"\0", timeout)
            if not raw:
                raise DccError("empty answer from Blender")
            reply = json.loads(raw.decode("utf-8"))
            if reply.get("status") not in ("ok", "success", None) and "message" in reply and _RESULT_MARK not in str(
                    reply.get("stdout", "")):
                raise DccError(f"Blender refused the code: {reply.get('message')}")
            text = str(reply.get("stdout", "")) + "\n" + str(reply.get("result", ""))
            i = text.rfind(_RESULT_MARK)
            if i < 0:
                raise DccError("Blender gave no result: " + json.dumps(reply)[:600])
            return json.loads(text[i + len(_RESULT_MARK):].splitlines()[0])
        if app == "maya":
            run_line = f"exec(compile({code!r}, '<dcc>', 'exec'))\n"
            first = _socket_roundtrip(host, port, run_line.encode("utf-8"), timeout)
            second = _socket_roundtrip(host, port, b"__import__('builtins').__dcc_result\n", timeout)
            text = second.decode("utf-8", "replace").strip()
            if not text or text.startswith("Error") or "Traceback" in text:
                raise DccError("Maya gave no result: " + (text or first.decode("utf-8", "replace"))[:600])
            if text[0] in "'\"":
                import ast
                text = ast.literal_eval(text)
            return json.loads(text)
    except ConnectionRefusedError:
        raise DccError(f"{app} is not listening on {host}:{port}: open it with the "
                       f"{'MCP add-on server' if app == 'blender' else 'commandPort'} running, or use mode='batch'")
    except (OSError, socket.timeout) as e:
        raise DccError(f"could not talk to {app} on {host}:{port}: {e}")
    raise DccError(f"{app} has no live mode; use mode='batch'")


def _run_batch(app: str, job: dict, timeout: float) -> dict:
    exe = find_app(app)
    if not exe:
        raise DccError(f"{app} is not installed (set ${_ENV[app]} to its {'blender' if app == 'blender' else _PATH_NAMES[app][0]} executable)")
    work = Path(tempfile.mkdtemp(prefix="agenty_dcc_"))
    try:
        driver = work / "driver.py"
        driver.write_text(_batch_wrapper(app), encoding="utf-8")
        job = dict(job, live=False, result_path=str(work / "result.json"))
        jf = work / "job.json"
        jf.write_text(json.dumps(job), encoding="utf-8")
        env = dict(os.environ)
        if app == "blender":
            env.pop("OCIO", None)          # a studio OCIO config breaks Blender's views
            cmd = [exe, "--factory-startup", "-b", "--python-exit-code", "1", "--python", str(driver), "--", str(jf)]
        else:
            cmd = [exe, str(driver), str(jf)]
            if app == "houdini":
                env.setdefault("HFS", str(Path(exe).resolve().parent.parent))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, env=env, stdin=subprocess.DEVNULL, creationflags=flags)
        except subprocess.TimeoutExpired:
            raise DccError(f"{app} batch run timed out after {int(timeout)} s")
        rf = Path(job["result_path"])
        if not rf.exists():
            tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-25:])
            raise DccError(f"{app} exited with code {p.returncode} and no result:\n{tail}")
        return json.loads(rf.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_job(app: str, mode: str, job: dict, timeout: float = 900) -> dict:
    return _run_live(app, job, timeout) if mode == "live" else _run_batch(app, job, timeout)


# ---------------------------------------------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------------------------------------------


def _err(message: str, **extra) -> str:
    return json.dumps({"ok": False, "error": message, **extra})


def _tail(text: str, lines: int = 40, chars: int = 4000) -> str:
    t = "\n".join((text or "").strip().splitlines()[-lines:])
    return t[-chars:]


def _bad_app_mode(app: str, mode: str) -> Optional[str]:
    if app not in APPS:
        return f"app must be one of {', '.join(APPS)}, not {app!r}"
    if mode not in MODES:
        return f"mode must be 'live' or 'batch', not {mode!r}"
    if app == "houdini" and mode == "live":
        return ("Houdini has no live mode here (its MCP bridge is a third-party plug-in and hrpyc needs a hand-started "
                "server): use mode='batch', which builds in a fresh hython and saves a .hip")
    return None


def default_scene_path(script_path: str, app: str) -> str:
    p = Path(script_path).resolve()
    return str(p.parent / "_dcc" / (p.stem + _EXT[app]))


@tool
def dcc_build(script_path: str, app: str = "blender", mode: str = "live", save_to: str = "", render_to: str = "",
              render_engine: str = "fast", resolution: str = "") -> str:
    """Run a whole build script file in a 3D app (Blender, Maya or Houdini), then optionally save and render a still.

    This is the ONLY way a model should be rebuilt: keep one script, change its constants, call this again. The
    script is expected to clear and rebuild its own collection / group, so every run is clean. This tool never
    patches the scene itself and refuses a missing file. Coordinates in the script are the app's own (Blender Z up,
    Maya and Houdini Y up).

    Modes: "live" runs in the app the user has open (Blender through its MCP add-on socket on 9876, Maya through its
    commandPort on 7001; Houdini has no live mode) and changes THEIR scene. "batch" runs in a fresh headless process
    (blender -b / mayapy / hython): it opens save_to if that file already exists (else starts empty), runs the
    script and saves the scene, and returns the path; later batch calls (dcc_silhouette_check, renders) open that file.

    Args:
        script_path: absolute path of the build script (.py) to execute in full.
        app: "blender", "maya" or "houdini".
        mode: "live" (the open app) or "batch" (headless, saves a scene file).
        save_to: where to save the scene (a .blend / .ma / .mb / .hip). Batch: defaults to <script folder>/_dcc/<name>.<ext>.
            Live: empty means do not save; a path saves a COPY and leaves the open session on the user's own file.
        render_to: a .png path to render a still of the scene camera to. Empty means no render.
        render_engine: "fast" (Blender Workbench, Maya hardware / playblast, Houdini OpenGL) or "final"
            (Blender Cycles, Maya Arnold, Houdini Karma).
        resolution: render size as "WIDTHxHEIGHT" (default: the scene's own).

    Returns:
        JSON: {ok, app, mode, stdout_tail, error (with the script's own line), traceback, objects: {count, top_level,
        collections}, saved, render: {path, engine, resolution, exists | error}}.
    """
    bad = _bad_app_mode(app, mode)
    if bad:
        return _err(bad)
    sp = Path(script_path).expanduser()
    if not sp.is_file():
        return _err(f"build script not found: {script_path}")
    if render_engine not in ("fast", "final"):
        return _err("render_engine must be 'fast' or 'final'")
    if resolution and not re.fullmatch(r"\s*\d+\s*[xX]\s*\d+\s*", resolution):
        return _err("resolution must look like 1280x720")
    if mode == "batch" and not save_to:
        save_to = default_scene_path(str(sp), app)
    job = {"action": "build", "script_path": str(sp.resolve()), "scene_in": save_to if mode == "batch" else "",
           "save_to": str(Path(save_to).expanduser().resolve()) if save_to else "",
           "render_to": str(Path(render_to).expanduser().resolve()) if render_to else "",
           "render_engine": render_engine, "resolution": resolution.strip()}
    if mode == "batch":
        job["scene_in"] = job["save_to"]
    t0 = time.time()
    try:
        res = run_job(app, mode, job, timeout=3600 if (render_to and render_engine == "final") else 900)
    except DccError as e:
        return _err(str(e), app=app, mode=mode)
    out = {"ok": bool(res.get("ok")), "app": app, "mode": mode, "stdout_tail": _tail(res.get("stdout", "")),
           "error": res.get("error", ""), "traceback": res.get("traceback", ""),
           "objects": res.get("objects"), "saved": res.get("saved", ""), "render": res.get("render"),
           "seconds": round(time.time() - t0, 1)}
    return json.dumps(out)


def _clean_points(points) -> list:
    if isinstance(points, str):
        points = json.loads(points)
    if not isinstance(points, list) or not points:
        raise ValueError("points must be a non-empty list of {name, world:[x,y,z], target:[x,y_from_top]}")
    out = []
    for i, p in enumerate(points):
        if not isinstance(p, dict):
            raise ValueError(f"points[{i}] must be an object")
        w = p.get("world")
        if not (isinstance(w, (list, tuple)) and len(w) == 3 and all(isinstance(v, (int, float)) for v in w)):
            raise ValueError(f"points[{i}] ({p.get('name', '?')}): 'world' must be [x, y, z] numbers")
        t = p.get("target")
        if t is not None and not (isinstance(t, (list, tuple)) and len(t) == 2
                                  and all(isinstance(v, (int, float)) for v in t)):
            raise ValueError(f"points[{i}] ({p.get('name', '?')}): 'target' must be [x_frac, y_from_top_frac]")
        out.append({"name": str(p.get("name") or f"p{i}"), "world": [float(v) for v in w],
                    "target": [float(v) for v in t] if t is not None else None})
    return out


@tool
def dcc_silhouette_check(points: list, app: str = "blender", mode: str = "live", scene: str = "",
                         camera: str = "", tolerance: float = 0.03) -> str:
    """Check a model's proportions against the reference photo BY NUMBERS: project key points through the scene
    camera and compare with where they sit in the photo. No images, so it cannot hallucinate.

    Give 6-12 key points of the model (ground contact, deck corner, eave, ridge end, mast top, left and right
    extents...) with their position in the world and where the same spot is in the photo (measured off a grid, as
    fractions of the picture: x from the left, y from the TOP). `world` is in the APP'S OWN world coordinates:
    Blender Z up, Maya and Houdini Y up; nothing is converted.

    The scene camera is used (Houdini: the first camera under /obj), or the one named in `camera`. Lens shift, sensor
    fit and the render aspect are honoured. In "batch" mode pass `scene` = the file dcc_build returned in `saved`.

    Args:
        points: list of {"name": str, "world": [x, y, z], "target": [x_frac, y_from_top_frac]}.
        app: "blender", "maya" or "houdini".
        mode: "live" (the open app, scene argument ignored) or "batch" (open the scene file headless).
        scene: batch only: path of the saved scene file.
        camera: name of a camera object (Houdini: node path); empty means the scene's camera.
        tolerance: largest allowed distance, in fractions of the frame (0.03 = 3 % of the picture).

    Returns:
        JSON: {ok, overall_pass, points: [{name, projected, target, dx, dy, distance, pass, in_front, hint}] with the
        worst failures first, worst: [...], overall_hint, camera, cross_check}. dx, dy are projected minus target;
        a negative dy means the model point sits higher in the frame than in the photo.
    """
    bad = _bad_app_mode(app, mode)
    if bad:
        return _err(bad)
    try:
        pts = _clean_points(points)
        tol = float(tolerance)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        return _err(str(e))
    if mode == "batch":
        if not scene:
            return _err("batch mode needs scene= the file dcc_build returned in 'saved'")
        if not Path(scene).expanduser().is_file():
            return _err(f"scene file not found: {scene}")
    job = {"action": "check", "points": [p["world"] for p in pts], "camera": camera,
           "scene_in": str(Path(scene).expanduser().resolve()) if scene else ""}
    try:
        res = run_job(app, mode, job, timeout=600)
    except DccError as e:
        return _err(str(e), app=app, mode=mode)
    if not res.get("ok"):
        return _err(res.get("error", "the check failed"), app=app, mode=mode, traceback=res.get("traceback", ""))
    raw = res["camera"]
    native = [tuple(n) for n in res["native"]]
    out = {"ok": True, "app": app, "mode": mode}
    cam_info = {"name": raw.get("name"), "kind": raw.get("kind"), "focal_mm": raw.get("focal_mm"),
                "resolution": [raw.get("res_x"), raw.get("res_y")]}
    warnings = []
    cross = None
    if raw.get("kind", "PERSP") != "PERSP":
        warnings.append(f"{raw.get('kind')} camera: the app's own projection is used, no cross-check")
    else:
        canon = camera_from_raw(app, raw)
        pure = [project_point(p["world"], canon) for p in pts]
        diffs = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(native, pure) if a[2] and b[2]]
        cross = {"max_abs_diff": round(max(diffs), 6) if diffs else 0.0}
        cross["agree"] = cross["max_abs_diff"] < 0.002
        if not cross["agree"]:
            warnings.append("the app's own projection and the pure camera maths differ by "
                            f"{cross['max_abs_diff']:.4f}: the numbers below are the app's (trusted) values")
    out.update(shape_check(pts, native, tol))
    out["camera"] = cam_info
    if cross is not None:
        out["cross_check"] = cross
    if warnings:
        out["warnings"] = warnings
    return json.dumps(out)
