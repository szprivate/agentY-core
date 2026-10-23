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
import hashlib
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
# Viewpoint (yaw) and the build gate (pure)
# ---------------------------------------------------------------------------------------------------------------

DETAIL_MARKER = "# --- DETAIL ---"
STAGES = ("blockout", "detail", "look")


def camera_space(point, cam: dict) -> tuple:
    """World point -> (right, up, depth) in the camera's own frame (depth grows away from the camera)."""
    d = [point[i] - cam["position"][i] for i in range(3)]
    return _dot(d, cam["right"]), _dot(d, cam["up"]), -_dot(d, cam["back"])


def pair_yaw(a, b) -> dict:
    """Yaw of the segment a-b in the ground plane relative to the image plane, from two camera-space points
    (right, up, depth). ``yaw_deg = atan2(dz, dx)`` with the segment turned so that dx >= 0 (the left end first):
    positive means the RIGHT end is farther away (a facade seen from the front-left), negative the left end is.
    0 is a segment parallel to the image plane (a camera straight in front of a flat facade). ``nearer`` is the
    end ("a" or "b") closer to the camera."""
    (ax, _, az), (bx, _, bz) = a, b
    left, right = ((ax, az), (bx, bz)) if ax <= bx else ((bx, bz), (ax, az))
    yaw = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    return {"yaw_deg": yaw, "nearer": "a" if az <= bz else "b"}


def yaw_check(points: list, yaw_pairs: list, cam: dict) -> list:
    """Judge each {"pair": [nameA, nameB], "target_yaw_deg": x, "tolerance_deg": 8} against the camera.
    Returns rows with a ``pass`` and a ``hint`` so they join the pass / fail count."""
    by_name = {p["name"]: p["world"] for p in points}
    rows = []
    for yp in yaw_pairs:
        a, b = str(yp["pair"][0]), str(yp["pair"][1])
        tgt, tol = float(yp["target_yaw_deg"]), float(yp.get("tolerance_deg", 8.0))
        row = {"pair": [a, b], "name": f"yaw {a} -> {b}", "target_yaw_deg": tgt, "tolerance_deg": tol}
        ca, cb = camera_space(by_name[a], cam), camera_space(by_name[b], cam)
        if ca[2] <= 1e-9 or cb[2] <= 1e-9:
            row.update(yaw_deg=None, hint="an end of the pair is behind the camera: the camera faces away from it "
                                          "or sits on the wrong side of the model")
            row["pass"] = False
        else:
            y = pair_yaw(ca, cb)
            yaw = y["yaw_deg"]
            nearer = a if y["nearer"] == "a" else b
            ok = abs(yaw - tgt) <= tol
            hint = ""
            if not ok:
                hint = ("facade yaw %.0f deg vs photo %.0f deg: " % (yaw, tgt)
                        + ("move the camera left / rotate the model so the right end recedes more" if yaw < tgt else
                           "move the camera right / rotate the model so the right end recedes less")
                        + " (in the model %s is the nearer end)" % nearer)
            row.update(yaw_deg=round(yaw, 1), nearer=nearer, hint=hint)
            row["pass"] = ok
        rows.append(row)
    return rows


def _clean_yaw_pairs(yaw_pairs, points: list) -> list:
    if isinstance(yaw_pairs, str):
        yaw_pairs = json.loads(yaw_pairs)
    if not isinstance(yaw_pairs, list):
        raise ValueError("yaw_pairs must be a list of {pair:[nameA,nameB], target_yaw_deg, tolerance_deg}")
    names = {p["name"] for p in points}
    out = []
    for i, yp in enumerate(yaw_pairs):
        pr = yp.get("pair") if isinstance(yp, dict) else None
        if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
            raise ValueError(f"yaw_pairs[{i}]: 'pair' must be [nameA, nameB]")
        missing = [str(n) for n in pr if str(n) not in names]
        if missing:
            raise ValueError(f"yaw_pairs[{i}]: {missing} not among the points (give both ends as points with 'world')")
        if not isinstance(yp.get("target_yaw_deg"), (int, float)):
            raise ValueError(f"yaw_pairs[{i}]: 'target_yaw_deg' must be a number")
        out.append({"pair": [str(pr[0]), str(pr[1])], "target_yaw_deg": float(yp["target_yaw_deg"]),
                    "tolerance_deg": float(yp.get("tolerance_deg", 8.0))})
    return out


def script_hashes(path: str) -> dict:
    """``sha``: the whole script. ``blockout_sha``: the script ABOVE the ``# --- DETAIL ---`` line (trailing
    whitespace and line endings ignored); with no marker it is the whole script, so any edit changes it."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    cut = next((i for i, ln in enumerate(lines) if ln.strip().startswith(DETAIL_MARKER)), None)
    top = lines if cut is None else lines[:cut]

    def h(ls):
        return hashlib.sha1("\n".join(ls).encode("utf-8")).hexdigest()[:16]
    return {"sha": h(lines), "blockout_sha": h(top), "has_marker": cut is not None}


def check_file_path(script_path: str) -> Path:
    return Path(str(script_path) + ".check.json")


def gate_decision(script_path: str, stage: str, force_reason: str = "") -> dict:
    """Decide whether ``stage`` may be built. Returns {"allowed", "gate", "error"}.
    blockout: always. detail / look: the last ``<script>.check.json`` must have overall_pass AND the same
    ``blockout_sha`` as the script now (so only lines below the DETAIL marker may have changed; with no marker
    any edit needs a fresh check). ``force_reason`` lets it through, recorded in ``gate``."""
    if stage == "blockout":
        return {"allowed": True, "gate": {"stage": stage}}
    rec = None
    why = ""
    cf = check_file_path(script_path)
    if not cf.is_file():
        why = (f"no silhouette check recorded for this script ({cf.name} is missing): run dcc_silhouette_check with "
               f"script_path=<this script> and get overall_pass first")
    else:
        try:
            rec = json.loads(cf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            why = f"{cf.name} is unreadable: run dcc_silhouette_check again"
    summary = None
    if rec is not None:
        summary = {k: rec.get(k) for k in ("overall_pass", "n_pass", "n_total", "worst_distance", "timestamp")}
        now = script_hashes(script_path)
        if not rec.get("overall_pass"):
            fails = rec.get("failing") or []
            lines = "; ".join("%s: %s" % (f.get("name"), f.get("hint") or "off by %s" % f.get("distance"))
                              for f in fails[:8])
            why = ("the last silhouette check FAILED (%s of %s pass, worst %s): %s. Fix the block-out constants, "
                   "run dcc_build with stage='blockout' and check again"
                   % (rec.get("n_pass"), rec.get("n_total"), rec.get("worst_distance"), lines or "see the check"))
        elif rec.get("blockout_sha") != now["blockout_sha"]:
            why = ("the script changed since the passing check%s: run dcc_silhouette_check again"
                   % ("" if now["has_marker"] else " (it has no '%s' line, so any edit counts; put the marker above "
                      "the detail code so detail edits do not need a re-check)" % DETAIL_MARKER))
    if not why:
        return {"allowed": True, "gate": {"stage": stage, "forced": False, "last_check": summary}}
    if force_reason and force_reason.strip():
        return {"allowed": True, "gate": {"stage": stage, "forced": True, "reason": force_reason.strip(),
                                          "why_blocked": why, "last_check": summary}}
    return {"allowed": False, "error": f"stage '{stage}' refused: {why}",
            "gate": {"stage": stage, "forced": False, "last_check": summary}}


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
        # Shut Maya down before the hard exit: os._exit on a live maya.standalone crashed
        # every run (a crash dump, a [Recovered] scene in Temp, the Autodesk error dialog).
        tail += ("\n_sys.stdout.flush()\n_sys.stderr.flush()\n"
                 "try:\n    import maya.standalone as _ms\n    _ms.uninitialize()\nexcept Exception:\n    pass\n"
                 "import os as _os\n_os._exit(0)\n")
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
            if app == "maya":
                # No Autodesk error-report dialog, usage survey or in-product messages for a
                # headless run nobody is watching.
                env.update(MAYA_DISABLE_CER="1", MAYA_DISABLE_CIP="1", MAYA_DISABLE_CLIC_IPM="1")
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
              render_engine: str = "", resolution: str = "", stage: str = "blockout",
              force_reason: str = "") -> str:
    """Run a whole build script file in a 3D app (Blender, Maya or Houdini), then optionally save and render a still.

    This is the ONLY way a model should be rebuilt: keep one script, change its constants, call this again. The
    script is expected to clear and rebuild its own collection / group, so every run is clean. This tool never
    patches the scene itself and refuses a missing file. Coordinates in the script are the app's own (Blender Z up,
    Maya and Houdini Y up).

    In Blender build scripts prefer bmesh / bpy.data over bpy.ops (ops need a window/area context; live mode runs
    them under a VIEW_3D override, batch has none).

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
        render_engine: "fast" (Blender Workbench, Maya hardware / playblast, Houdini OpenGL: unlit grey shapes,
            no shadows - right for a blockout) or "final" (Blender Cycles, Maya Arnold, Houdini Karma: materials,
            sun and shadows). Empty (default) picks by stage: "final" for stage="look", "fast" otherwise.
        resolution: render size as "WIDTHxHEIGHT" (default: the scene's own).
        stage: "blockout" (default, always allowed), "detail" or "look". Set it on EVERY build. detail and look are
            GATED: they run only if the last dcc_silhouette_check(script_path=<this script>) passed (rule below),
            otherwise the build is refused (ok false, nothing run) with the failing points and their hints.
        force_reason: a non-empty sentence that lets a detail / look build through the gate anyway. The result then
            carries gate {forced: true, reason, last_check}. Only for a point that is truly unmeasurable: say which
            point and why. Never to skip a failing check.

    The gate rule: the check writes <script>.check.json with overall_pass and two hashes of the script. A detail /
    look build needs overall_pass AND an unchanged ``blockout_sha``, the hash of the script text ABOVE a
    ``# --- DETAIL ---`` line (whitespace ignored). Edits below the marker never need a re-check; any edit above
    it does. A script with no marker hashes whole: any edit needs a fresh passing check.

    Returns:
        JSON: {ok, gate, app, mode, stdout_tail, error (with the script's own line), traceback, objects: {count, top_level,
        collections}, saved, render: {path, engine, resolution, exists | error}}.
    """
    bad = _bad_app_mode(app, mode)
    if bad:
        return _err(bad)
    sp = Path(script_path).expanduser()
    if not sp.is_file():
        return _err(f"build script not found: {script_path}")
    if stage not in STAGES:
        return _err(f"stage must be one of {', '.join(STAGES)}, not {stage!r}")
    # A "look" render judged with the fast (unlit, shadowless) engine sent an agent chasing
    # its lighting for a dozen passes: the look stage renders final unless told otherwise.
    render_engine = (render_engine or "").strip().lower() or ("final" if stage == "look" else "fast")
    if render_engine not in ("fast", "final"):
        return _err("render_engine must be 'fast' or 'final'")
    gate = gate_decision(str(sp.resolve()), stage, force_reason)
    if not gate["allowed"]:
        return _err(gate["error"], gate=gate["gate"], stage=stage)
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
    out = {"ok": bool(res.get("ok")), "gate": gate["gate"], "app": app, "mode": mode, "stdout_tail": _tail(res.get("stdout", "")),
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
                         camera: str = "", tolerance: float = 0.03, yaw_pairs: Optional[list] = None,
                         script_path: str = "") -> str:
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
        yaw_pairs: viewpoint check, [{"pair": [nameA, nameB], "target_yaw_deg": 21, "tolerance_deg": 8}]. Both names
            are points above (only their world positions are used). The yaw of the segment in the ground plane,
            relative to the image plane, is computed from the two points in CAMERA space: atan2(depth difference,
            sideways difference) with the left end first. Positive: the right end is farther away (a facade seen
            from the front-left); 0: the camera is straight in front of a flat facade. Use the yaw
            measure_reference gave for the same edge. A wrong camera side or a straight-on camera fails here, with a
            hint. Each pair counts as one more point in overall_pass / n_pass. Assumes a level camera.
        script_path: the build script this check is for. When given, the result is written to
            <script_path>.check.json, which is what dcc_build(stage="detail" / "look") reads as its gate. ALWAYS pass it.

    Returns:
        JSON: {ok, overall_pass, points: [{name, projected, target, dx, dy, distance, pass, in_front, hint}] with the
        worst failures first, worst: [...], n_pass, n_total, yaw_pairs: [{pair, yaw_deg, target_yaw_deg, nearer,
        pass, hint}], overall_hint, camera, cross_check, check_file}. dx, dy are projected minus target;
        a negative dy means the model point sits higher in the frame than in the photo.
    """
    bad = _bad_app_mode(app, mode)
    if bad:
        return _err(bad)
    try:
        pts = _clean_points(points)
        tol = float(tolerance)
        yps = _clean_yaw_pairs(yaw_pairs, pts) if yaw_pairs else []
    except (ValueError, TypeError, KeyError, AttributeError, json.JSONDecodeError) as e:
        return _err(str(e))
    if script_path and not Path(script_path).expanduser().is_file():
        return _err(f"script_path not found: {script_path}")
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
    if yps:
        rows = yaw_check(pts, yps, {k: raw[k] for k in ("position", "right", "up", "back")})
        out["yaw_pairs"] = rows
        failed_yaw = [r for r in rows if not r["pass"]]
        out["failed"] += len(failed_yaw)
        out["overall_pass"] = out["failed"] == 0
        if failed_yaw:
            out["worst"] = out.get("worst", []) + [{"name": r["name"], "distance": None, "hint": r["hint"]}
                                                   for r in failed_yaw]
    out["n_total"] = out["points_checked"] + len(yps)
    out["n_pass"] = out["n_total"] - out["failed"]
    out["camera"] = cam_info
    if cross is not None:
        out["cross_check"] = cross
    if warnings:
        out["warnings"] = warnings
    if script_path:
        out["check_file"] = _write_check_file(script_path, out, scene, camera)
    return json.dumps(out)


def _write_check_file(script_path: str, out: dict, scene: str, camera: str) -> str:
    sp = Path(script_path).expanduser().resolve()
    cf = check_file_path(str(sp))
    judged = [p for p in out.get("points", []) if "distance" in p]
    failing = [{"name": p["name"], "distance": p["distance"], "hint": p["hint"]} for p in judged if not p["pass"]]
    failing += [{"name": r["name"], "distance": None, "hint": r["hint"]}
                for r in out.get("yaw_pairs", []) if not r["pass"]]
    worst = max([f["distance"] for f in failing if f["distance"] is not None], default=None)
    rec = {"overall_pass": bool(out["overall_pass"]), "worst_distance": worst, "n_pass": out["n_pass"],
           "n_total": out["n_total"], "tolerance": out["tolerance"], "failing": failing, "points": out["points"],
           "yaw_pairs": out.get("yaw_pairs", []), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "scene": scene, "camera_name": camera, "camera": out["camera"], "script": str(sp), **script_hashes(str(sp))}
    try:
        cf.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    except OSError as e:
        out.setdefault("warnings", []).append(f"could not write {cf}: {e}")
        return ""
    return str(cf)
