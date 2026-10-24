"""Validation of a motion-graphics scene (``*.mgscene.json``, ``"mgscene": 1``).

The same rules, with the same issue codes, as ``validateScene`` in
``renderer/src/mgscene.js`` - ``renderer/test/validate_cases.json`` holds the cases
both must agree on and a test runs them through each. Pure Python, no Node needed:
the agent can check a scene before anything is installed or rendered.

An issue is ``{"code", "path", "message"}``. Errors make the scene unrenderable;
warnings are things that probably are not what was meant.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

SCENE_VERSION = 1
LAYER_TYPES = ("text", "shape", "image", "video", "group", "component")
SHAPES = ("rect", "ellipse", "path")
ALIGNS = ("left", "center", "right")
FITS = ("contain", "cover", "fill")
REVEAL_DIRS = ("right", "left", "down", "up")
EASES = ("linear", "smooth", "hold", "easeIn", "easeOut", "easeInOut", "backOut", "expoOut")
BLENDS = ("normal", "multiply", "screen", "overlay", "add", "darken", "lighten", "color-dodge", "difference")

# prop -> (kind, min, max)
PROPS: dict[str, tuple[str, float | None, float | None]] = {
    "anchor": ("vec2", None, None),
    "position": ("vec2", None, None),
    "scale": ("vec2", None, None),
    "rotation": ("num", None, None),
    "opacity": ("num", 0, 100),
    "fill": ("color", None, None),
    "stroke": ("color", None, None),
    "strokeWidth": ("num", 0, None),
    "size": ("vec2", None, None),
    "radius": ("num", 0, None),
    "fontSize": ("num", 1, None),
    "tracking": ("num", None, None),
    "reveal": ("num", 0, 1),
    "trimStart": ("num", 0, 1),
    "trimEnd": ("num", 0, 1),
}
_TRANSFORM = ["anchor", "position", "scale", "rotation", "opacity"]
PROPS_BY_TYPE: dict[str, list[str]] = {
    "text": _TRANSFORM + ["fill", "stroke", "strokeWidth", "fontSize", "tracking", "reveal"],
    "shape": _TRANSFORM + ["fill", "stroke", "strokeWidth", "size", "radius", "reveal", "trimStart", "trimEnd"],
    "image": _TRANSFORM + ["size", "radius", "reveal"],
    "video": _TRANSFORM + ["size", "radius", "reveal"],
    "group": list(_TRANSFORM),
    "component": _TRANSFORM + ["size"],
}

_HEX = re.compile(r"^#([0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$")
_RGB = re.compile(r"^rgba?\(\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*(?:,\s*[\d.]+\s*)?\)$")


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x: Any) -> bool:
    return (isinstance(x, int) and not isinstance(x, bool)) or (isinstance(x, float) and x.is_integer())


def is_color(s: Any) -> bool:
    if not isinstance(s, str):
        return False
    t = s.strip().lower()
    return t == "transparent" or bool(_HEX.match(t)) or bool(_RGB.match(t))


def _check_value(prop: str, v: Any) -> bool:
    kind = PROPS[prop][0]
    if kind == "num":
        return _num(v)
    if kind == "vec2":
        return isinstance(v, list) and len(v) == 2 and all(_num(x) for x in v)
    return v is not None and is_color(v)


def _ease_ok(e: Any) -> bool:
    if e is None:
        return True
    if isinstance(e, str):
        return e in EASES
    return (isinstance(e, list) and len(e) == 4 and all(_num(x) for x in e)
            and 0 <= e[0] <= 1 and 0 <= e[2] <= 1)


def validate_scene(scene: Any) -> dict:
    """``{"ok": bool, "errors": [...], "warnings": [...]}`` for a parsed scene."""
    errors: list[dict] = []
    warnings: list[dict] = []

    def E(code, path, msg):
        errors.append({"code": code, "path": path, "message": msg})

    def W(code, path, msg):
        warnings.append({"code": code, "path": path, "message": msg})

    if not isinstance(scene, dict):
        E("E_SCENE", "", "the scene is not a JSON object")
        return {"ok": False, "errors": errors, "warnings": warnings}
    if scene.get("mgscene") != SCENE_VERSION or isinstance(scene.get("mgscene"), bool):
        E("E_VERSION", "mgscene", f'"mgscene" must be {SCENE_VERSION}')
    c = scene.get("comp")
    duration = math.inf
    if not isinstance(c, dict):
        E("E_COMP", "comp", "comp {width, height, fps, duration} is missing")
    else:
        for k, lo, hi in (("width", 16, 8192), ("height", 16, 8192), ("fps", 1, 120), ("duration", 1, 360000)):
            v = c.get(k)
            if not _int(v) or v < lo or v > hi:
                E("E_COMP", f"comp.{k}", f"comp.{k} must be an integer from {lo} to {hi}")
        if _int(c.get("duration")):
            duration = c["duration"]
        bg = c.get("background")
        if bg is not None and not is_color(bg):
            E("E_COLOR", "comp.background", "comp.background is not a colour")

    assets = scene.get("assets", {})
    if not isinstance(assets, dict):
        E("E_ASSET", "assets", "assets must be an object")
        assets = {}
    else:
        for aid, a in assets.items():
            if not isinstance(a, dict) or a.get("type") not in ("image", "video", "font"):
                E("E_ASSET", f"assets.{aid}", 'an asset needs type "image", "video" or "font"')
            elif not isinstance(a.get("src"), str) or not a["src"]:
                E("E_ASSET", f"assets.{aid}.src", "an asset needs a src (path or url)")

    layers = scene.get("layers")
    if not isinstance(layers, list):
        E("E_LAYERS", "layers", "layers must be an array")
        return {"ok": False, "errors": errors, "warnings": warnings}

    ids: dict[str, dict] = {}
    for i, layer in enumerate(layers):
        p = f"layers[{i}]"
        if not isinstance(layer, dict):
            E("E_LAYER", p, "a layer must be an object")
            continue
        lid = layer.get("id")
        if not isinstance(lid, str) or not lid:
            E("E_LAYER_ID", f"{p}.id", "every layer needs a string id")
        elif lid in ids:
            E("E_LAYER_ID", f"{p}.id", f'duplicate layer id "{lid}"')
        else:
            ids[lid] = layer

    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
        p = f"layers[{i}]"
        name = layer.get("id") or p
        ltype = layer.get("type")
        if ltype not in LAYER_TYPES:
            E("E_LAYER_TYPE", f"{p}.type", f'layer "{name}": type must be one of {", ".join(LAYER_TYPES)}')
            continue
        parent = layer.get("parent")
        if parent is not None:
            if parent not in ids:
                E("E_PARENT", f"{p}.parent", f'layer "{name}": parent "{parent}" does not exist')
            else:
                cur, hops = layer, 0
                while cur is not None and cur.get("parent") and hops < 1000:
                    hops += 1
                    cur = ids.get(cur["parent"])
                    if cur is layer:
                        E("E_PARENT", f"{p}.parent", f'layer "{name}": parents form a loop')
                        break
        for k in ("in", "out"):
            if k in layer and not _num(layer[k]):
                E("E_RANGE", f"{p}.{k}", f'layer "{name}": {k} must be a frame number')
        if _num(layer.get("in")) and _num(layer.get("out")) and layer["out"] <= layer["in"]:
            E("E_RANGE", f"{p}.out", f'layer "{name}": out must be after in')
        if _num(layer.get("in")) and layer["in"] >= duration:
            W("W_RANGE", f"{p}.in", f'layer "{name}" starts after the composition ends')
        if "blend" in layer and layer["blend"] not in BLENDS:
            E("E_BLEND", f"{p}.blend", f'layer "{name}": blend must be one of {", ".join(BLENDS)}')
        allowed = PROPS_BY_TYPE[ltype]
        for prop in PROPS:
            if prop not in layer:
                continue
            if prop not in allowed:
                W("W_PROP", f"{p}.{prop}", f'layer "{name}": {prop} has no effect on a {ltype} layer')
                continue
            if layer[prop] is None and prop in ("fill", "stroke"):
                continue
            if not _check_value(prop, layer[prop]):
                E("E_COLOR" if PROPS[prop][0] == "color" else "E_VALUE", f"{p}.{prop}",
                  f'layer "{name}": {prop} is not a valid {PROPS[prop][0]}')
        tracks = layer.get("tracks")
        if "tracks" in layer and not isinstance(tracks, dict):
            E("E_TRACK", f"{p}.tracks", "tracks must be an object of arrays")
        elif isinstance(tracks, dict):
            for prop, keys in tracks.items():
                tp = f"{p}.tracks.{prop}"
                if prop not in PROPS or prop not in allowed:
                    E("E_TRACK_PROP", tp, f'layer "{name}": {prop} cannot be animated on a {ltype} layer')
                    continue
                if not isinstance(keys, list) or not keys:
                    E("E_TRACK", tp, "a track is a non-empty array of keys")
                    continue
                last, ordered = -math.inf, True
                for j, k in enumerate(keys):
                    kp = f"{tp}[{j}]"
                    if not isinstance(k, dict) or not _num(k.get("f")):
                        E("E_KEY", kp, "a key needs a numeric f (comp frame)")
                        continue
                    if not _check_value(prop, k.get("v")):
                        E("E_COLOR" if PROPS[prop][0] == "color" else "E_KEY", f"{kp}.v",
                          f"{prop} key value is not a valid {PROPS[prop][0]}")
                    if not _ease_ok(k.get("ease")):
                        E("E_EASE", f"{kp}.ease", f"ease must be one of {', '.join(EASES)} or [x1,y1,x2,y2]")
                    if k["f"] < last:
                        ordered = False
                    if k["f"] == last:
                        E("E_KEY", kp, f"two keys on frame {k['f']}")
                    last = max(last, k["f"])
                if not ordered:
                    W("W_KEY_ORDER", tp, "keys are not sorted by frame (they are used sorted)")
                if math.isfinite(duration) and any(isinstance(k, dict) and _num(k.get("f")) and k["f"] > duration for k in keys):
                    W("W_RANGE", tp, "a key lies after the end of the composition")
        if ltype == "text" and not isinstance(layer.get("text"), str):
            E("E_TEXT", f"{p}.text", f'text layer "{name}" needs a text string')
        if ltype == "shape":
            kind = layer.get("shape", "rect")
            if kind not in SHAPES:
                E("E_SHAPE", f"{p}.shape", f"shape must be one of {', '.join(SHAPES)}")
            if kind == "path" and (not isinstance(layer.get("d"), str) or not layer["d"].strip()):
                E("E_SHAPE", f"{p}.d", f'path shape "{name}" needs d (SVG path data)')
        if ltype in ("image", "video"):
            a = assets.get(layer.get("asset")) if isinstance(layer.get("asset"), str) else None
            if not layer.get("asset") or not a:
                E("E_ASSET_REF", f"{p}.asset", f'{ltype} layer "{name}" needs asset naming an entry of assets')
            elif isinstance(a, dict) and a.get("type") != ltype:
                E("E_ASSET_REF", f"{p}.asset", f'layer "{name}": asset "{layer["asset"]}" is a {a.get("type")}, not a {ltype}')
        if ltype == "component" and (not isinstance(layer.get("component"), str) or not layer["component"]):
            E("E_COMPONENT", f"{p}.component", f'component layer "{name}" needs the name of a registered component')
        if "align" in layer and layer["align"] not in ALIGNS:
            E("E_VALUE", f"{p}.align", f"align must be one of {', '.join(ALIGNS)}")
        if "fit" in layer and layer["fit"] not in FITS:
            E("E_VALUE", f"{p}.fit", f"fit must be one of {', '.join(FITS)}")
        if "revealDir" in layer and layer["revealDir"] not in REVEAL_DIRS:
            E("E_VALUE", f"{p}.revealDir", f"revealDir must be one of {', '.join(REVEAL_DIRS)}")
        if "font" in layer and not isinstance(layer["font"], str):
            E("E_VALUE", f"{p}.font", "font is a family name or a font asset id")

    return {"ok": not errors, "errors": errors, "warnings": warnings}


def load_scene(path: str) -> Any:
    """Parse a scene file. Raises ValueError with a readable message."""
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise ValueError(f"no such file: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"not valid JSON ({exc.msg} at line {exc.lineno} column {exc.colno})") from None
