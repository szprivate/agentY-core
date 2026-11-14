"""Work on the 3D scene open in the ComfyUI Image Viewer beyond one character's rig.

animate_character rigs and animates a character; this is the rest of the 3D space: the Clip
Sequencer (animation as clips on tracks), cameras and turntables, renders and playblasts of the
scene on screen, keys on any channel, animation-layer and display-layer settings, IK/FK and
spaces, custom attributes and set-driven keys, a model brought in, and a USD stage of the shot.

Every action is a request to the open page, the same road as animate_character: the viewer's
``POST /imageviewer/rig/agent`` hands it over, the page does it on its 3D scene (one undo step
each, as the same edit by hand) and reports back (``js/imageViewer_previzAgentScene.js``). The
browser with the viewer has to be open on the 3D scene.

``render_turntables`` is the whole "a turntable of each clip" job in one call: one turntable rig
keyed with a full turn over every clip's frames, then a render of each clip through it.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from agenty_core._compat import tool
from agenty_core.tools.animate_character import ask_viewer

# What the page answers (js/imageViewer_previzAgentScene.js SCENE_AGENT_ACTIONS; the viewer's
# rig_routes.AGENT_ACTIONS lists them too).
ACTIONS = (
    "scene_info", "scene_set", "add_model", "add_shape", "add_camera", "set_camera", "turntable", "render", "playblast",
    "key_set", "key_delete", "key_list",
    "layer_set", "layer_remove", "layer_merge_down",
    "display_layer_add", "display_layer_set", "display_layer_list",
    "clip_list", "clip_from_keys", "clip_place", "clip_set", "clip_split", "clip_remove", "clip_match", "clip_bake",
    "clip_track_add", "clip_track_set",
    "ikfk_switch", "ikfk_match", "space_set", "attr_add", "sdk_key", "sdk_remove",
    "export_usd",
    # The Terrain menu (js/imageViewer_previzAgentTerrain.js TERRAIN_AGENT_ACTIONS).
    "terrain_add", "terrain_set", "terrain_sculpt", "terrain_paint", "terrain_clear", "terrain_height",
    "terrain_erode", "scatter_add", "scatter_set", "scatter_paint",
)
# What each action needs before it is worth sending (the page checks the rest).
NEEDS = {
    "add_model": ("path",), "set_camera": ("camera",),
    "key_set": ("item", "prop"), "key_delete": ("item",), "key_list": ("item",),
    "layer_set": ("layer",), "layer_remove": ("layer",), "layer_merge_down": ("layer",),
    "display_layer_add": ("items",), "display_layer_set": ("layer",),
    "clip_place": ("source",), "clip_set": ("clip",), "clip_split": ("clip",), "clip_remove": ("clip",),
    "clip_match": ("clip",), "clip_track_set": ("track",),
    "ikfk_switch": ("control",), "ikfk_match": ("control",), "space_set": ("control",),
    "attr_add": ("item", "name"), "sdk_key": ("driver", "driven"), "sdk_remove": ("item",),
    "terrain_sculpt": ("strokes",), "terrain_paint": ("strokes",), "scatter_paint": ("strokes",),
}
SCULPT_TOOLS = ("raise", "lower", "smooth", "flatten", "restore")
SCATTER_TYPES = ("pine", "tree", "bush", "grass", "rock", "column", "lamp", "model")
ERODE_MODES = ("both", "hydraulic", "thermal")
ERODE_FOLLOW = ("flow", "deposit", "erosion", "wetness", "dry")
RENDER_FORMATS = ("mp4", "mov", "webm", "png")
SHADINGS = ("original", "clay", "normal", "wireframe")


def _error(msg: str) -> str:
    return json.dumps({"status": "error", "error": msg})


def request_body(action: str, options: Optional[dict] = None) -> dict:
    """What the viewer's /imageviewer/rig/agent takes for a scene action."""
    if action not in ACTIONS:
        raise ValueError(f"action is one of: {', '.join(ACTIONS)}")
    opts = dict(options or {})
    opts.pop("action", None)
    for k in NEEDS.get(action, ()):
        if opts.get(k) in (None, "", []):
            raise ValueError(f"{action} needs options.{k}")
    if action in ("render", "playblast"):
        if opts.get("format") and opts["format"] not in RENDER_FORMATS:
            raise ValueError(f"options.format is one of: {', '.join(RENDER_FORMATS)}")
        if opts.get("shading") and opts["shading"] not in SHADINGS:
            raise ValueError(f"options.shading is one of: {', '.join(SHADINGS)}")
    if action in ("terrain_sculpt", "terrain_paint", "scatter_paint"):
        strokes = opts.get("strokes")
        if not isinstance(strokes, list) or not all(isinstance(st, dict) and st.get("points") for st in strokes):
            raise ValueError(f"{action}: options.strokes is a list of {{points: [[x, z], ...], radius, strength}}")
        if action == "terrain_sculpt":
            for st in strokes:
                if st.get("tool", opts.get("tool", "raise")) not in SCULPT_TOOLS:
                    raise ValueError(f"a stroke's tool is one of: {', '.join(SCULPT_TOOLS)}")
    if action == "scatter_add" and opts.get("type", "tree") not in SCATTER_TYPES:
        raise ValueError(f"options.type is one of: {', '.join(SCATTER_TYPES)}")
    if action == "terrain_erode":
        if opts.get("mode") is not None and opts["mode"] not in ERODE_MODES:
            raise ValueError(f"options.mode is one of: {', '.join(ERODE_MODES)}")
        for kind in (opts.get("follow") or {}).values():
            if kind and kind not in ERODE_FOLLOW:
                raise ValueError(f"options.follow takes one of: {', '.join(ERODE_FOLLOW)}")
    if action == "export_usd" and opts.get("format") not in (None, "usda", "usdc"):
        raise ValueError("options.format is usda or usdc")
    if opts.get("from") is not None and opts.get("to") is not None and int(opts["to"]) < int(opts["from"]):
        raise ValueError("options.to must not come before options.from")
    return {"action": action, **opts}


@tool
def animate_scene(action: str, options: Optional[dict] = None, timeout_seconds: float = 900.0) -> str:
    """Work on the 3D scene open in the ComfyUI Image Viewer: clips, cameras, turntables, renders, keys, layers, terrain.

    animate_character does a character's rig and AI motion; this does the rest of the 3D space. The browser
    must be open on the 3D scene. get_viewer_state() shows its shot, cameras, clips and layers (scene3d).
    Items, clips, tracks, sources and layers are named by their name (or id). Every change is one undo step.
    The 3D space is Y UP, in metres: [x, y, z] is x across, y the HEIGHT, z along the ground (a camera 1.6 m
    above flat ground at the origin is [0, 1.6, 0]); never Z up as in some DCCs. On a terrain, take the ground's
    height from terrain_height and add to y.

    Args:
        action: One of -
            Scene: "scene_info" (everything: items, cameras, clips, layers), "scene_set" {length, fps, frame},
            "add_model" {path, name, position, rotation, scale} (a model file into this scene).
            "add_shape" {type box|sphere|plane|cylinder|cone|torus|group, name, position, rotation, scale (its
            size in m), color "#rrggbb", parent} (a solid stands on its base: y is its bottom; a cylinder's axis
            is Y - rotation [0, 0, 90] for a wheel). A stand-in (a car, a prop) is a "group" with shapes under
            it (their numbers in the group's space), the group keyed with key_set.
            Cameras (Y up: position and look_at are [x, height, z]): "add_camera" {name, position [x,y,z],
            look_at [x,y,z] (or rotation, degrees XYZ), fov,
            resolution [w,h], parent, active (default true)}, "set_camera" {camera, position, look_at or
            rotation, fov, resolution, active, frame (a camera that animates gets a key there)},
            "turntable" {target (item or list; default every model), clips ("all" or a list of clip names: a full
            turn over EACH clip's frames) or from / to, degrees (360), start_yaw (0 = in front of a character facing
            +Z), distance (m; 0 = fit the subject), height (m above the look point; default a fifth of the
            distance), look_height (m above the subject's centre), fov, resolution, name ("Turntable"), active}
            - one pivot + camera rig ("Turntable cam"), framed on the subject's box over each range; follow true
            (follow_step, frames, default 4) for a subject that MOVES (a car driving): the turn stays centred on it
            frame by frame, at one frame's size - an orbit round a driving car is this, not set_camera keys.
            Render: "render" {camera (name; "" = the free view; default the active one), clip (that clip's frames)
            or from / to, name (the file: output/previz/<name>.<format> by the export rule), format (mp4 | mov |
            webm | png), shading (original | clay | normal | wireframe), width / height (else the camera's),
            scale (%), background, add_node (a loader node on the canvas, default the scene's setting)} ->
            result.path; "playblast" {camera, clip or from / to, scale, format, shading, hud, note} (the view as drawn
            with a heads-up display, into the History).
            Keys: "key_set" {item, prop (position | rotation | scale | pivot | fov | ikfk | space | attr:<name> |
            bs:<shape>), frame, value, ease (smooth | linear | hold), axis (0-2 or x/y/z: that axis only), keys (a list
            of {frame, value, ease, axis} instead), layer (an animation layer's name; "base" = the base)} - the value is
            what the item shows with every layer and clip on top; "key_delete" {item, prop, frame / frames (none = all)};
            "key_list" {item, frame} (keys per channel and the values at the frame).
            Animation layers (animate_character layer_add / layer_list make and list them): "layer_set" {layer,
            weight (0-1; with frame + key: true a weight key), mute, solo, lock, mode, name, active}, "layer_remove",
            "layer_merge_down" {layer}.
            Display layers: "display_layer_add" {name, items, visible, mode (normal | template | reference), color},
            "display_layer_set" {layer, visible, mode, color, name, add [items], remove [items], delete}, "display_layer_list".
            Clip Sequencer: "clip_list" (sources, tracks, clips with their frames), "clip_from_keys" {model (all its
            controls) or items, from, to (default the keys' range), name, at, track, keep_keys (leave the keys on the
            items too), place (false = a source only)}, "clip_place" {source, at, track, then any clip_set field},
            "clip_set" {clip, name, start, length, in, out, speed, loop ("" | cycle | pingpong), weight, blend_in,
            blend_out, pre, post, mute, accumulate, target (another character), track}, "clip_split" {clip, at},
            "clip_remove", "clip_match" {clip} (lines it up where the clip before it on the character ended),
            "clip_bake" {clip (one) or none (all clips into keys)}, "clip_track_add" {name, mode}, "clip_track_set"
            {track, name, mute, solo, lock, weight, mode, delete}. AI takes land as clips with
            animate_character(action="motion", options={"as_clip": true}).
            Rig extras: "ikfk_switch" {control (a limb's IK, pole or FK control), to (ik | fk), frame} (no jump),
            "ikfk_match" {control, side (fk | ik), frame}, "space_set" {control, space ("parent", an item, or an index),
            frame}, "attr_add" {item, name, min, max, value} (a keyable number, attr:<name>), "sdk_key" {driver {item,
            prop, axis}, driver_value, driven [{item, prop, axis, value}], frame} (a set-driven key: the driver at
            driver_value gives the driven values), "sdk_remove" {item, index or prop}.
            USD: "export_usd" {name, format (usda | usdc)} - the whole shot as a stage (constraints and clips baked).
            Terrain (the Terrain menu; places are world metres [x, z], Y up; each stroke is one undo step and
            shows in the view as it lands): "terrain_add" {name, size [w, d] (m, default 200 x 200), height (m of
            relief, default 20), seed, roughness (0.05-0.95), segments (8-512), position, rotation, layers [{name,
            color}] for ground / rock / cliff / peak} -> the terrain's name and settings; "terrain_set" {terrain,
            same fields}; "terrain_sculpt" {terrain, strokes [{tool (raise | lower | smooth | flatten | restore),
            points [[x, z], ...] (a polyline), radius (m), strength (0-1), hardness (0-1), passes (1-20; more =
            stronger), to (flatten: metres above the terrain's base; default the height under the first point),
            amount (raise / lower: metres in all)}], pause (ms between strokes, default 150)} - a road is a
            flatten stroke along it, a ridge a raise with a WIDE radius (a tenth of the terrain or more) and amount
            set to its height in metres (passes at full strength stack up into a spike); "terrain_paint" {terrain, strokes [{layer (0-3 or its name),
            points, radius, strength, erase}]} (the ground layers, over the slope and height rules);
            "terrain_clear" {terrain, what (sculpt | layer_paint | scatter_paint + scatter)}; "terrain_height"
            {terrain, points [[x, z], ...]} -> the ground's world y there (stand a model or a camera on it);
            "terrain_erode" {terrain, steps (default 1000; 300-1500 is usual), mode (both | hydraulic |
            thermal), rain, evaporation, capacity, dissolve, deposit, talus (deg), thermal, apply (default true),
            layers (paint rock / ground from the result, default true), follow {scatter: flow | deposit | erosion |
            wetness | dry}, timeout_seconds} - Terrain > Erode… run live in the view, then applied as sculpting.
            Scatters: "scatter_add" {type (pine | tree | bush | grass | rock | column | lamp | model + path),
            terrain, name, count (how many on the whole terrain), scale [min, max], max_slope (deg), layer (-1 =
            any, else the ground layer it grows on), color, seed, wind}; "scatter_set" {scatter, same fields};
            "scatter_paint" {scatter, strokes [{points, radius, strength, erase}]} - once painted it grows only
            where painted.
        options: The action's fields, as listed above.
        timeout_seconds: How long to wait for the viewer (a render takes as long as its frames).

    Returns:
        JSON: {"status": "ok", "action", "result": {...}} or {"status": "error", "error"}.
    """
    try:
        body = request_body(action, options)
    except (ValueError, TypeError) as exc:
        return _error(str(exc))
    return ask_viewer(body, timeout_seconds, what="the 3D scene's agent actions")


def _safe(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", str(name or "").lower()).strip("_") or "clip"


@tool
def render_turntables(
    clips: Optional[list] = None,
    target: str = "",
    degrees: float = 360.0,
    name: str = "",
    format: str = "mp4",
    shading: str = "original",
    width: int = 0,
    height: int = 0,
    distance: float = 0.0,
    height_m: Optional[float] = None,
    start_yaw: float = 0.0,
    fov: float = 0.0,
    add_node: bool = False,
    timeout_seconds: float = 1800.0,
) -> str:
    """Render a turntable of each animation clip in the viewer's 3D scene: the camera goes once round the subject
    over every clip's frames, one film per clip.

    Make the clips first (animate_character action="motion" with options={"as_clip": true}, or
    animate_scene "clip_from_keys" / "clip_place"); the browser must be open on the 3D scene.

    Args:
        clips: Clip names to render; empty = every clip on an unmuted track, in timeline order.
        target: The item(s) to turn around (a name, or names joined by commas); empty = every model.
        degrees: How far the camera turns over each clip (360 = once round).
        name: Prefix of the files (default the shot's name): <name>_<clip>_turntable.<format>.
        format: mp4 | mov | webm | png.
        shading: original | clay | normal | wireframe.
        width: Render width in px (0 = the camera's resolution, 1920 x 1080 by default).
        height: Render height in px.
        distance: Metres from the subject (0 = fit it, over the whole clip).
        height_m: Metres the camera stands above the look point (default a fifth of the distance).
        start_yaw: Where the turn starts, degrees (0 = in front of a character facing +Z).
        fov: The camera's vertical field of view (0 = 35).
        add_node: Put a loader node for each film on the canvas.
        timeout_seconds: How long each render may take.

    Returns:
        JSON: {"status": "ok", "camera", "renders": [{clip, from, to, path}], "failed": [...]} or an error.
    """
    tt = {"clips": list(clips) if clips else "all", "degrees": degrees, "start_yaw": start_yaw}
    if target:
        tt["target"] = [t.strip() for t in str(target).split(",") if t.strip()]
    if distance and distance > 0:
        tt["distance"] = distance
    if height_m is not None:
        tt["height"] = height_m
    if fov and fov > 0:
        tt["fov"] = fov
    made = json.loads(ask_viewer(request_body("turntable", tt), 120, what="the 3D scene's agent actions"))
    if made.get("status") != "ok":
        return json.dumps(made)
    res = made.get("result") or {}
    cam = res.get("camera")
    prefix = _safe(name) if name else ""
    renders, failed = [], []
    for rg in res.get("ranges") or []:
        clip = rg.get("clip") or f"{rg.get('from')}-{rg.get('to')}"
        opts = {"camera": cam, "from": rg.get("from"), "to": rg.get("to"), "format": format, "shading": shading,
                "add_node": bool(add_node),
                "name": "_".join(x for x in (prefix, _safe(clip), "turntable") if x)}
        if width > 0 and height > 0:
            opts["width"], opts["height"] = int(width), int(height)
        out = json.loads(ask_viewer(request_body("render", opts), timeout_seconds, what="the 3D scene's agent actions"))
        if out.get("status") == "ok":
            r = out.get("result") or {}
            renders.append({"clip": clip, "from": rg.get("from"), "to": rg.get("to"), "path": r.get("path"),
                            "where": r.get("where"), "frames": r.get("frames")})
        else:
            failed.append({"clip": clip, "error": out.get("error")})
    status = "ok" if renders else "error"
    out = {"status": status, "camera": cam, "renders": renders, "failed": failed}
    if not renders:
        out["error"] = "no turntable rendered" + (f": {failed[0]['error']}" if failed else "")
    return json.dumps(out)
