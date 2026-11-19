"""Track a plate's camera in the ComfyUI Image Viewer's matchmove (its 3D space's Matchmove menu).

The viewer leads a person through seven steps - plate, lens, masks, tracks, solve, objects, into the scene -
and this tool does the same steps for the agent. Every action is a request to the open page, the road
animate_scene takes: the viewer's ``POST /imageviewer/rig/agent`` hands it over, the page does it on its 3D tab
(one undo step each, saved into the tab's ``.mm.json``) and answers with a summary of the matchmove - the
plate, the lens, the tracks a frame, the solve's error, its worst tracks and frames - so the track can be
judged after every step (``js/imageViewer_previzAgentMatchmove.js`` in the viewer). The skill ``matchmove``
says how to get a good track.

Coordinates on the plate are its own pixels: x right, y down, from the top left corner; frames count from 0.
"""

from __future__ import annotations

import json
from typing import Optional

from agenty_core._compat import tool
from agenty_core.tools.animate_character import ask_viewer

# What the page answers (js/imageViewer_previzAgentMatchmove.js MATCHMOVE_AGENT_ACTIONS, without the prefix;
# the viewer's rig_routes.AGENT_ACTIONS lists them too).
ACTIONS = (
    "info", "open", "camera", "lens", "lens_export", "mask", "mask_remove", "track", "track_add", "track_delete",
    "track_set", "solve", "object_add", "object_track", "object_solve", "build", "orient", "geometry", "save",
)
NEEDS = {
    "mask_remove": ("mask",), "track_add": ("points",), "track_set": ("ids",), "orient": ("how", "points"),
    "geometry": ("kind", "points"),
}
LENS_HOW = ("grid", "lines", "comp", "set", "none")
SOLVE_MODES = ("auto", "free", "tripod")
REFINE = ("auto", "on", "off")
ORIENT_HOW = ("ground", "origin", "x", "z", "scale")
GEOMETRY = ("plane", "box", "locator")


def _error(msg: str) -> str:
    return json.dumps({"status": "error", "error": msg})


def _points(pts, what: str) -> None:
    if not isinstance(pts, list) or len(pts) < 3 or not all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in pts):
        raise ValueError(f"{what} is 3 or more [x, y] points in plate pixels")


def request_body(action: str, options: Optional[dict] = None) -> dict:
    """What the viewer's /imageviewer/rig/agent takes for a matchmove action."""
    if action not in ACTIONS:
        raise ValueError(f"action is one of: {', '.join(ACTIONS)}")
    opts = dict(options or {})
    opts.pop("action", None)
    for k in NEEDS.get(action, ()):
        if opts.get(k) in (None, "", []):
            raise ValueError(f"{action} needs options.{k}")
    if action == "open" and not (opts.get("plate") or opts.get("path") or opts.get("file")):
        raise ValueError("open needs options.plate (a film, or one frame of a numbered sequence) or options.file (a .mm.json)")
    if action == "lens":
        how = opts.get("how")
        if how not in LENS_HOW:
            raise ValueError(f"options.how is one of: {', '.join(LENS_HOW)}")
        if how == "set" and opts.get("k1") is None:
            raise ValueError("lens set needs options.k1 (and k2, center)")
        if how == "comp" and not opts.get("path"):
            raise ValueError("lens comp needs options.path (a .comp.json with a Lens Distortion node)")
        for line in opts.get("lines") or []:
            _points(line, "each of options.lines")
    if action == "lens_export":
        if opts.get("to", "comp") not in ("comp", "stmap"):
            raise ValueError("options.to is comp or stmap")
        if opts.get("mode", "undistort") not in ("undistort", "redistort"):
            raise ValueError("options.mode is undistort or redistort")
    if action in ("mask", "object_add"):
        keys = opts.get("keys")
        if action == "mask" and not opts.get("mask") and not keys:
            raise ValueError("a new mask needs options.keys: [{frame, points: [[x, y], ...]}]")
        for k in keys or []:
            if not isinstance(k, dict) or k.get("frame", k.get("f")) is None:
                raise ValueError("each key is {frame, points: [[x, y], ...]}")
            _points(k.get("points", k.get("pts")), "a key's points")
        if opts.get("mode") not in (None, "exclude", "object"):
            raise ValueError("options.mode is exclude or object")
    if action == "track_add":
        if not all(isinstance(p, dict) and "x" in p and "y" in p for p in opts["points"]):
            raise ValueError("options.points is [{x, y, frame}] in plate pixels")
    if action == "track_delete" and not any(opts.get(k) not in (None, "", []) for k in ("ids", "worst_share", "error_over", "shorter_than", "all")):
        raise ValueError("track_delete needs ids, worst_share, error_over, shorter_than or all")
    if action == "solve":
        if opts.get("mode") not in (None, *SOLVE_MODES):
            raise ValueError(f"options.mode is one of: {', '.join(SOLVE_MODES)}")
        if opts.get("refine") not in (None, *REFINE):
            raise ValueError(f"options.refine is one of: {', '.join(REFINE)}")
    if action == "orient":
        if opts["how"] not in ORIENT_HOW:
            raise ValueError(f"options.how is one of: {', '.join(ORIENT_HOW)}")
        if opts["how"] == "scale" and not opts.get("distance"):
            raise ValueError("orient scale needs options.distance (how far apart the two points are, scene units)")
    if action == "geometry" and opts["kind"] not in GEOMETRY:
        raise ValueError(f"options.kind is one of: {', '.join(GEOMETRY)}")
    if opts.get("from") is not None and opts.get("to") is not None and int(opts["to"]) < int(opts["from"]):
        raise ValueError("options.to must not come before options.from")
    return {"action": f"matchmove_{action}", **opts}


@tool
def matchmove(action: str, options: Optional[dict] = None, timeout_seconds: float = 1200.0) -> str:
    """Track a plate's camera (and moving objects) in the ComfyUI Image Viewer's matchmove, step by step.

    Read the skill "matchmove" first: it is the order of the steps and what makes a track good. The browser
    must be open with the viewer (a 3D tab is brought up; the Matchmove and Plate panels open). Plate
    coordinates are pixels of the plate, x right, y down from the top left; frames count from 0. Every answer
    carries "matchmove": the plate, camera, lens, masks, objects, the tracks (total, median length, fewest on a
    frame, thin_frames runs), the solve (rms_px, verdict, focal_px / focal_mm, worst_tracks [[id, px]],
    frames_over_1px, unsolved_frames) and what was built.

    Args:
        action: One of -
            "info" - the summary only.
            "open" {plate (a film, or one frame of a numbered sequence: the whole sequence is taken) | file (a
            .mm.json to go on with)} - a new plate starts a new matchmove; the camera facts the file holds
            (focal length, sensor) are read and answered as facts.
            "camera" {focal_mm (null = not known), sensor [w, h] mm (the RECORDED part of the sensor), known (the
            solve holds a known focal length unless solve refine is on)}.
            "lens" {how: grid {path (default the plate), frames [n, ...] (several, the chart in other places),
            kind auto | checker | lines} | lines {lines: [[[x, y], ...], ...] (3+ points along something straight
            in the world each; 6+ lines out to the corners)} | comp {path (a .comp.json's first Lens Distortion
            node)} | set {k1, k2, center [x, y] px} | none; fit_center, k2 (true / false / null = if the data
            allows)} - the answer's lens.fit_error_px and coverage say how good it is.
            "lens_export" {to: comp (a .comp.json: the plate -> Lens Distortion -> Output) | stmap (EXR ST map),
            mode undistort | redistort, name, open (comp: open it in the Comp space)} -> result.path.
            "mask" {keys [{frame, points [[x, y], ...]}] (the shape moves straight between keys, held outside them),
            name, mode exclude (no tracks in it: people, cars, water, sky, reflections) | object, object (its
            name: tracks in it are that object's), mask (an existing one to change), replace_keys, on};
            "mask_remove" {mask}.
            "track" {features (8-1000, default 200), min_ncc (0.3-0.99, default 0.75: how alike a feature must
            stay), min_dist (px between features), from, to} - finds and follows features; run again to add more
            (it keeps the tracks there). "track_add" {points [{x, y, frame}], object} - tracks by hand, followed
            both ways. "track_delete" {ids, worst_share (0-0.5 of the solved tracks, worst first), error_over (px),
            shorter_than (frames), all}. "track_set" {ids, on, object (a name, or "camera")}.
            "solve" {mode auto | free | tripod, refine auto | on | off (the focal length)} - the camera of every
            frame and a 3D point per track; under 0.5 px rms is very good, under 1 usable.
            "object_add" {name, keys (its mask)}, "object_track" {object}, "object_solve" {object, scale}.
            "build" {backplate (default true: the undistorted plate on the camera)} - a "Matchmove" group, the
            camera keyed every frame and looked through, the plate, the points as locators, the objects.
            "orient" {how: ground (3+ points on the floor) | origin (1+: their middle) | x | z (2: that axis from
            the first to the second) | scale (2, distance = how far apart in scene units, metres), points [track
            ids with a 3D point - take them from solve.worst_tracks' opposite end or info]}.
            "geometry" {kind plane (3+) | box | locator, points [track ids]} - geometry on solved points to check the
            track sticks.
            "save" - the .mm.json (saved after every change anyway) -> result.path.
        options: The action's settings (above).
        timeout_seconds: How long to wait for the page (tracking and solving a long plate take minutes).

    Returns:
        JSON: {"status": "ok", "action", "result": {..., "matchmove": {...}}} or {"status": "error", "error"}.
    """
    try:
        body = request_body(action, options)
    except (ValueError, TypeError) as exc:
        return _error(str(exc))
    return ask_viewer(body, timeout_seconds, what="the matchmove")
