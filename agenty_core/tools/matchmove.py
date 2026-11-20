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
    "ground", "ai_mask", "measure", "phone_export", "frame", "pin",
)
NEEDS = {
    "mask_remove": ("mask",), "track_add": ("points",), "track_set": ("ids",), "orient": ("how", "points"),
    "geometry": ("kind", "points"),
}
LENS_HOW = ("grid", "lines", "comp", "set", "none")
SOLVE_MODES = ("auto", "free", "tripod")
REFINE = ("auto", "on", "off")
DEPTH_SOURCES = ("off", "phone", "ai")
DEPTH_EDGES = (True, "phone", "ai")
ORIENT_HOW = ("ground", "origin", "x", "z", "scale")
GEOMETRY = ("plane", "box", "locator")
GROUND_HOW = ("auto", "points", "ai", "depth", "phone")
AI_MASK_ENGINES = ("objects", "background")
MEASURE_KINDS = ("distance", "plane", "line", "right")
TRACK_DIRECTIONS = ("both", "forward", "backward")
PIN_MODES = ("off", "align", "hold")
FRAME_MOST = 16           # frames one "frame" call looks at
FRAME_SIZE = 1600         # longest side of a frame picture, px
FRAME_TRACKS_LISTED = 80  # tracks per frame named in the answer (all are drawn)


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
    if action == "track":
        if opts.get("direction") not in (None, *TRACK_DIRECTIONS):
            raise ValueError(f"options.direction is one of: {', '.join(TRACK_DIRECTIONS)}")
        for k in ("start",):
            if opts.get(k) is not None and not (isinstance(opts[k], (int, float)) and opts[k] >= 0):
                raise ValueError("options.start is a frame (0 or more) or null (the first frame of the range)")
        seeds = opts.get("seeds")
        if seeds is not None and seeds != "thin" and not (
                isinstance(seeds, list) and seeds and all(isinstance(f, (int, float)) and f >= 0 for f in seeds)):
            raise ValueError('options.seeds is a list of frames to find new points on, or "thin"')
        ext = opts.get("extend")
        if ext is not None and not (isinstance(ext, bool) or (isinstance(ext, list) and ext)):
            raise ValueError("options.extend is true (every track on) or a list of track ids")
    if action == "frame":
        fr = opts.get("frames")
        if fr is not None and not (isinstance(fr, list) and fr and all(isinstance(f, (int, float)) and f >= 0 for f in fr)):
            raise ValueError("options.frames is a list of plate frames (from 0)")
        if fr is not None and len(fr) > FRAME_MOST:
            raise ValueError(f"at most {FRAME_MOST} frames at a time")
        if opts.get("frame") is not None and not (isinstance(opts["frame"], (int, float)) and opts["frame"] >= 0):
            raise ValueError("options.frame is a plate frame (from 0)")
    if action == "track_delete" and not any(opts.get(k) not in (None, "", [], False) for k in ("ids", "worst_share", "error_over", "shorter_than", "all", "depth_edges")):
        raise ValueError("track_delete needs ids, worst_share, error_over, shorter_than, all or depth_edges")
    if action == "track_delete" and opts.get("depth_edges") not in (None, False, *DEPTH_EDGES):
        raise ValueError('options.depth_edges is true (the depth the solve uses, else the phone\'s, else AI), "phone" or "ai"')
    if action == "solve":
        if opts.get("mode") not in (None, *SOLVE_MODES):
            raise ValueError(f"options.mode is one of: {', '.join(SOLVE_MODES)}")
        if opts.get("refine") not in (None, *REFINE):
            raise ValueError(f"options.refine is one of: {', '.join(REFINE)}")
        if opts.get("depth") not in (None, *DEPTH_SOURCES):
            raise ValueError(f"options.depth is one of: {', '.join(DEPTH_SOURCES)}")
        ds = opts.get("depth_strength")
        if ds is not None and not (isinstance(ds, (int, float)) and 0.1 <= ds <= 10):
            raise ValueError("options.depth_strength is 0.1 .. 10 (1: as much as the depth agrees with itself)")
    if action == "orient":
        if opts["how"] not in ORIENT_HOW:
            raise ValueError(f"options.how is one of: {', '.join(ORIENT_HOW)}")
        if opts["how"] == "scale" and not opts.get("distance"):
            raise ValueError("orient scale needs options.distance (how far apart the two points are, scene units)")
    if action == "ground":
        how = opts.get("how", "auto")
        if how not in GROUND_HOW:
            raise ValueError(f"options.how is one of: {', '.join(GROUND_HOW)}")
        if how == "points" and len(opts.get("points") or []) < 3:
            raise ValueError("ground points needs options.points: 3 or more track ids on the floor")
    if action == "ai_mask":
        if opts.get("engine", "objects") not in AI_MASK_ENGINES:
            raise ValueError(f"options.engine is one of: {', '.join(AI_MASK_ENGINES)}")
        if opts.get("engine", "objects") == "objects" and not str(opts.get("prompt") or "").strip():
            raise ValueError("ai_mask with engine objects needs options.prompt (what to mask: person, car, water ...)")
    if action in ("ground", "ai_mask") and opts.get("scale") is not None and not (0.1 <= float(opts["scale"]) <= 1):
        raise ValueError("options.scale is the share of the plate's size the model runs at, 0.1 .. 1")
    if action == "measure":
        add = opts.get("add")
        if not add and opts.get("remove") in (None, "") and not opts.get("clear"):
            raise ValueError("measure needs options.add {kind, ids, metres}, options.remove (an id) or options.clear")
        if add:
            if not isinstance(add, dict) or add.get("kind") not in MEASURE_KINDS:
                raise ValueError(f"options.add.kind is one of: {', '.join(MEASURE_KINDS)}")
            need = {"distance": 2, "line": 3, "plane": 3, "right": 3}[add["kind"]]
            if len(add.get("ids") or []) < need:
                raise ValueError(f"a {add['kind']} needs {need}{'' if add['kind'] in ('distance', 'right') else ' or more'} track ids")
            if add["kind"] == "distance" and not (float(add.get("metres") or 0) > 0):
                raise ValueError("a distance needs options.add.metres (how far apart the two tracks are in the world)")
    if action == "solve" and opts.get("geometry") not in (None, *PIN_MODES):
        raise ValueError(f"options.geometry is one of: {', '.join(PIN_MODES)}")
    if action == "pin":
        add = opts.get("add")
        if not add and opts.get("remove") in (None, "", []) and not opts.get("clear") and opts.get("mode") is None:
            raise ValueError("pin needs options.add {track, point [x, y, z]} | {track, item, vertex}, options.remove, options.clear or options.mode")
        if opts.get("mode") not in (None, *PIN_MODES):
            raise ValueError(f"options.mode is one of: {', '.join(PIN_MODES)}")
        for p in (add if isinstance(add, list) else [add] if add else []):
            if not isinstance(p, dict) or p.get("track") in (None, ""):
                raise ValueError("each pin is {track, point [x, y, z]} or {track, item, vertex}")
            pt = p.get("point")
            if pt is not None:
                if not (isinstance(pt, (list, tuple)) and len(pt) == 3 and all(isinstance(v, (int, float)) for v in pt)):
                    raise ValueError("options.add.point is [x, y, z] in the 3D space's units")
            elif not p.get("item"):
                raise ValueError("a pin needs options.add.point [x, y, z] or options.add.item (an item's id or name; vertex: one of its vertices)")
            if p.get("vertex") is not None and not (isinstance(p["vertex"], int) and p["vertex"] >= 0):
                raise ValueError("options.add.vertex is a vertex index (0 or more) of the item's first mesh")
    if action == "camera" and opts.get("pixel_aspect") is not None and not (0.2 <= float(opts["pixel_aspect"]) <= 5):
        raise ValueError("options.pixel_aspect is a pixel's width over its height, 0.2 .. 5 (2 = a 2x anamorphic squeeze)")
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
    frames_over_1px, unsolved_frames, focal_check, units), measurements, the phone (an iPhone film or photo: its
    camera, rotation and depth) and what was built.

    Args:
        action: One of -
            "info" - the summary only. START with it: the user may have loaded a plate already (the turn's
            [VIEWER] lines name it as "matchmove plate=..."); go on with that one, "open" only for another.
            "frame" {frames [n, ...] (at most 16) | frame (default the frame shown), tracks (default true: the
            tracks drawn on - green solved under 1 px, yellow not solved, red over 1 px or off - and listed with
            id, x, y), undistort (the plate through the lens instead; nothing drawn), size (longest side, px)} -
            the frames as pictures to LOOK at: result.frames[i].path (and result.contact_sheet for several);
            hand them to analyze_image (file_paths=[...]) or view_image.
            "open" {plate (a film, or one frame of a numbered sequence: the whole sequence is taken) | file (a
            .mm.json to go on with)} - a new plate starts a new matchmove; the camera facts the file holds
            (focal length, sensor) are read and answered as facts.
            "camera" {preset (an id of the viewer's camera list, e.g. iphone14-wide, iphone14pro-wide, iphone-ultra,
            iphone-tele77: a phone as its 35 mm equivalent on the plate's shape), focal_mm (null = not known), sensor
            [w, h] mm (the RECORDED part of the sensor), known (the solve starts from a known focal length; "auto"
            refine still refines it), pixel_aspect (1 square, 2 a 2x anamorphic squeeze)}.
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
            stay), min_dist (px between features), from, to, direction both | forward | backward (default
            forward), start (the frame tracking starts on, null = from), seeds ([frames]: new points found on those
            frames and followed the direction's way(s); "thin": on the middle of every stretch with too few
            tracks, both ways), extend (true: carry every track on past its ends; or [track ids])} - finds and
            follows features; run again to add more (it keeps the tracks there). A hard shot (things coming
            into view, a fast turn, a cut-off start): seeds "thin" after the first pass, extend, direction both. "track_add" {points [{x, y, frame}], object} - tracks by hand, followed
            both ways. "track_delete" {ids, worst_share (0-0.5 of the solved tracks, worst first), error_over (px),
            shorter_than (frames), all, depth_edges (true | "phone" | "ai": the tracks the depth says not to trust -
            on a depth edge, a corner in front of something far, or jumping in depth; answered as depth_edges
            [[id, edge | jumps]]), edge_tol (0.15)}. "track_set" {ids, on, object (a name, or "camera")}.
            "solve" {mode auto | free | tripod, refine auto | on | off (the focal length), depth off | phone (an
            iPhone Cinematic film's depth track) | ai (a depth model on 12 frames over the tracks, run first and
            kept), depth_strength (0.1-10, 1), depth_workflow, depth_scale (0.1-1), depth_again, geometry off |
            align | hold (how the pins are used, see "pin")} - the camera of
            every frame and a 3D point per track; under 0.5 px rms is very good, under 1 usable. With depth the
            points are pulled into the depth the plate was seen at (each frame scaled onto the solve, so no units
            needed): a shot with little camera movement gets its focal length and proportions from it, and one
            solved inside out (near and far swapped) is solved again from the depth. solve.depth says how well
            they agree (order: rank correlation, agreement: share off) or why the depth was left out.
            "object_add" {name, keys (its mask)}, "object_track" {object}, "object_solve" {object, scale}.
            "build" {backplate (default true: the undistorted plate on the camera)} - a "Matchmove" group, the
            camera keyed every frame and looked through, the plate, the points as locators, the objects.
            "orient" {how: ground (3+ points on the floor) | origin (1+: their middle) | x | z (2: that axis from
            the first to the second) | scale (2, distance = how far apart in scene units, metres), points [track
            ids with a 3D point - take them from solve.worst_tracks' opposite end or info]}.
            "geometry" {kind plane (3+) | box | locator, points [track ids]} - geometry on solved points to check the
            track sticks.
            "measure" {add {kind distance | plane | line | right, ids [track ids; a right angle's in order end,
            corner, end], metres (a distance)} | remove (its id) | clear} - real measurements the NEXT solve keeps:
            a distance sets the scale (1 unit = 1 m) and two or more fix the proportions; a squashed or stretched
            solve is put right by them. The answer's measurements say whether each holds.
            "pin" {add {track, point [x, y, z]} | {track, item (id or name), vertex (index; else its origin)} or a
            list of those, remove [track ids], clear, mode off | align (the solve moved, turned and scaled onto
            the pins) | hold (default: the pins also pull inside the solve, so camera and focal length follow -
            mends a squashed solve)} - tracks pinned to known spots of geometry ALREADY in the 3D space (a box set
            where the real one stands, an imported set model, a scan, surveyed points) in its axes and units.
            With 3+ pins on, not on one line, the NEXT solve lands in that space: solve.surveyed, units "scene",
            solve.geometry.pins says how far each lands off (left out = disagreed with the rest), and build puts
            the camera straight into the geometry's world (do not orient / ground afterwards). Tripod solves
            leave the pins out. The answer's pins list them.
            "ground" {how auto (every solved point) | points (3+ track ids on the floor) | ai (SAM 3 looks for
            prompt, default "ground") | depth (a depth model on 5 frames; metres true sets 1 unit = 1 m with the
            metric model) | phone (an iPhone Cinematic film's own depth track: no model, no ComfyUI job, not in
            metres), scale (ai / depth: share of the plate's size the model runs at, 0.1..1), tolerance (% of the
            scene's size), lift (scene units), flip} - after build: the world set level on the ground, at height 0.
            "ai_mask" {engine objects (SAM 3: prompt = what to mask) | background (a background remover: the
            subject; films use the video model), step (every Nth frame), from, to, object (an object's name: its
            mask), workflow, scale} - one ComfyUI job, a matte for the whole stretch; track again after it.
            "phone_export" - an iPhone Cinematic film's depth track as a 16-bit PNG sequence -> result.depth.
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
    if action == "frame":
        return _look(body, timeout_seconds)
    return ask_viewer(body, timeout_seconds, what="the matchmove")


def _track_colour(t: dict):
    if t.get("on") is False:
        return (230, 60, 60)
    e = t.get("error_px")
    if e is None:
        return (240, 210, 60)
    return (90, 220, 110) if e <= 1 else (235, 70, 70)


def _look(body: dict, timeout_seconds: float) -> str:
    """matchmove "frame": the page names the plate, its lens and each frame's tracks (matchmove_frame); the frames
    themselves come from the viewer's mm_frame route as JPEGs, the tracks drawn on, saved for analyze_image."""
    from agenty_core.tools.animate_character import _client
    from agenty_core.tools.workspace_tools import _frames_dir, _sheet

    raw = ask_viewer(body, timeout_seconds, what="the matchmove")
    try:
        out = json.loads(raw)
    except ValueError:
        return raw
    if out.get("status") != "ok":
        return raw
    res = out.get("result") or {}
    mm = res.get("matchmove") or {}
    plate = mm.get("plate") or {}
    if not plate.get("path"):
        return _error("no plate yet: matchmove_open with plate = the film or one frame of the sequence")
    undistort = bool(body.get("undistort")) and bool(res.get("lens"))
    size = int(body.get("size") or FRAME_SIZE)
    folder = None
    paths, labels = [], []
    try:
        from PIL import Image, ImageDraw
    except ImportError:  # pragma: no cover - Pillow is a dependency
        Image = None
    import io
    for fr in res.get("frames") or []:
        q = {"path": plate["path"], "frame": int(fr["frame"]), "size": max(256, min(size, 4096))}
        if undistort:
            L = res["lens"]
            q.update(k1=L.get("k1") or 0, k2=L.get("k2") or 0)
            if isinstance(L.get("center"), list) and len(L["center"]) >= 2:
                q.update(cx=L["center"][0], cy=L["center"][1])
        try:
            resp = _client().get("/imageviewer/mm_frame", params=q, raw=True)
            data = resp.content
        except Exception as exc:  # noqa: BLE001
            fr["error"] = f"the frame was not read: {exc}"
            continue
        if folder is None:
            stem = str(plate["path"]).replace("\\", "/").split("/")[-1].split(".")[0] or "plate"
            folder = _frames_dir(f"matchmove_{stem}")
        target = folder / f"frame_{int(fr['frame']):05d}.jpg"
        tracks = fr.get("tracks") or []
        if Image is not None and tracks and not undistort and plate.get("width"):
            im = Image.open(io.BytesIO(data)).convert("RGB")
            k = im.width / float(plate["width"])
            draw = ImageDraw.Draw(im)
            r = max(3, round(im.width / 300))
            for t in tracks:
                x, y = t["x"] * k, t["y"] * k
                draw.ellipse((x - r, y - r, x + r, y + r), outline=_track_colour(t), width=max(1, r // 2))
                if len(tracks) <= 60:
                    draw.text((x + r + 2, y - r - 2), str(t["id"]), fill=_track_colour(t))
            im.save(target, quality=90)
        else:
            target.write_bytes(data)
        fr["path"] = str(target)
        fr["tracks_on_frame"] = len(tracks)
        if len(tracks) > FRAME_TRACKS_LISTED:
            fr["tracks"] = sorted(tracks, key=lambda t: -(t.get("error_px") or 0))[:FRAME_TRACKS_LISTED]
            fr["tracks_note"] = f"the {FRAME_TRACKS_LISTED} worst listed; all {len(tracks)} drawn"
        paths.append(target)
        labels.append(f"frame {fr['frame']}")
    if not paths:
        return _error("no frame of the plate could be read through ComfyUI (is the viewer up to date?)")
    if len(paths) > 1:
        try:
            res["contact_sheet"] = str(_sheet(paths, labels, folder / "contact_sheet.jpg"))
        except Exception as exc:  # noqa: BLE001
            res["contact_sheet_error"] = str(exc)
    res["look"] = ("hand result.frames[i].path to analyze_image (file_paths=[...] for several)"
                   + ("; the plate is undistorted, so no tracks are drawn" if undistort else
                      "; tracks: green solved under 1 px, yellow not solved, red over 1 px or switched off"))
    return json.dumps(out)
