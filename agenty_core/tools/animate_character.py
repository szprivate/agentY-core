"""Rig and animate a character in the ComfyUI Image Viewer's 3D space.

The rig lives in the viewer's page (its controls are scene items; see the viewer's
``imageViewer_rigData.js``), so every action is a request to the open page: the
viewer's ``POST /imageviewer/rig/agent`` hands it over, the page does it on its 3D
scene, and this tool polls ``GET /imageviewer/rig/agent?id=`` for the answer. The
browser with the viewer has to be open, on the 3D scene that holds the character.

Actions:

* ``rig``         controls on a skinned model (humanoid FK + IK + COG when it reads as
                  a humanoid, FK only otherwise - creatures, props);
* ``autorig``     a model with no skeleton gets one (UniRig: skeleton + skin weights,
                  any character), then controls;
* ``motion``      motion from text (NVIDIA Kimodo), keyed on the controls from ``from_frame``
                  to ``to_frame``, starting where - and as - the character stands;
* ``inbetween``   the frames between the keyed poses in the range filled in (the keys are
                  the constraints; the two outer ones are kept);
* ``clean_feet``  the standing feet pinned and kept above the floor, re-keyed;
* ``export``      FBX (skeleton, skin, keys every frame) or Alembic (the meshes baked);
* ``remove_rig``;
* ``restyle``      the character's motion done another way (``prompt`` = the style; Kimodo keeps the path and
                  timing, and as ``strength`` says the hands and feet or the whole body);
* ``video_motion`` a person's motion from a film (``media_path``; MediaPipe pose), keyed from ``from_frame``;
* ``lipsync``      the mouth's blendshapes keyed from a sound or a film's sound (``media_path``);
* ``layer_add`` / ``layer_list``                                    animation layers;
* ``constrain`` / ``remove_constraint`` / ``list_constraints`` / ``bake_constraints``   constraints;
* ``pose_list`` / ``pose_save`` / ``pose_apply`` / ``pose_delete`` / ``flip_pose``     the pose library.

Humanoids only for ``motion`` / ``inbetween`` / ``clean_feet``. UniRig and Kimodo run on
the ComfyUI machine (set up with the viewer's scripts/rig/install_rig_tools.py).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from agenty_core._compat import tool

logger = logging.getLogger("agenty_core.animate_character")

ACTIONS = ("rig", "autorig", "motion", "inbetween", "clean_feet", "export", "remove_rig",
           "restyle", "video_motion", "lipsync",
           "layer_add", "layer_list", "constrain", "remove_constraint", "list_constraints", "bake_constraints",
           "pose_list", "pose_save", "pose_apply", "pose_delete", "flip_pose")
TAKES = ("motion", "inbetween", "restyle", "video_motion")          # what can go onto a new animation layer
STRENGTHS = ("loose", "medium", "close")
CONSTRAINTS = ("parent", "point", "orient", "aim", "scale")
# Extra fields an action passes through from ``options`` (as the viewer's page reads them).
OPTION_KEYS = {
    "video_motion": ("film_from", "film_to", "step", "smooth", "in_place", "clean"),
    "lipsync": ("start", "seconds", "strength", "lead"),
    "layer_add": ("mode", "weight", "select"),
    "constrain": ("type", "driven", "targets", "maintain_offset", "weights", "skip", "aim"),
    "remove_constraint": ("driven", "constraint"),
    "list_constraints": ("item",),
    "bake_constraints": ("items", "remove"),
    "pose_apply": ("blend", "mirror", "key"),
}


def _client():
    from agenty_core.utils.comfyui_client import get_client
    return get_client()


def _error(msg: str) -> str:
    return json.dumps({"status": "error", "error": msg})


def request_body(action, model="", prompt="", from_frame=None, to_frame=None, seed=None, feet=True,
                 format="fbx", humanoid=True, from_pose=True, name="", media_path="", strength="medium",
                 layer=False, options=None) -> dict:
    """What the viewer's /imageviewer/rig/agent takes for an action (fields it does not use left out)."""
    if action not in ACTIONS:
        raise ValueError(f"action is one of: {', '.join(ACTIONS)}")
    body = {"action": action}
    if model:
        body["model"] = model
    if action in ("motion", "inbetween"):
        if action == "motion" and not str(prompt or "").strip():
            raise ValueError("motion needs a prompt: what the character does (\"A person walks forward and waves.\")")
        if prompt:
            body["prompt"] = str(prompt).strip()
        body["feet"] = bool(feet)
        if action == "motion":
            body["from_pose"] = bool(from_pose)
    options = dict(options or {})
    if action == "restyle":
        if not str(prompt or "").strip():
            raise ValueError("restyle needs a prompt: the style (\"A tired old man walks slowly.\")")
        if strength not in STRENGTHS:
            raise ValueError(f"strength is one of: {', '.join(STRENGTHS)}")
        body.update(prompt=str(prompt).strip(), strength=strength, feet=bool(feet))
    if action in ("video_motion", "lipsync"):
        if not str(media_path or "").strip():
            raise ValueError(f"{action} needs media_path: the {'film' if action == 'video_motion' else 'sound (or film)'}")
        body["path"] = str(media_path).strip()
        if action == "video_motion":
            body["feet"] = bool(feet)
            if prompt:
                body["prompt"] = str(prompt).strip()
    if action == "constrain" and options.get("type", "parent") not in CONSTRAINTS:
        raise ValueError(f"options.type is one of: {', '.join(CONSTRAINTS)}")
    if action == "constrain" and not (options.get("driven") and options.get("targets")):
        raise ValueError("constrain needs options.driven (the item that follows) and options.targets (a list)")
    if action in ("layer_add", "pose_save", "pose_apply", "pose_delete"):
        if action != "layer_add" and not name:
            raise ValueError(f"{action} needs name: the pose's name")
        if name:
            body["name"] = name
    for k in OPTION_KEYS.get(action, ()):
        if k in options:
            body[k] = options[k]
    if action in TAKES and layer:
        body["layer"] = "new"
    if action in ("motion", "inbetween", "clean_feet", "export", "restyle", "video_motion", "lipsync", "bake_constraints"):
        if from_frame is not None:
            body["from"] = int(from_frame)
        if to_frame is not None:
            body["to"] = int(to_frame)
        if from_frame is not None and to_frame is not None and int(to_frame) <= int(from_frame):
            raise ValueError("to_frame must come after from_frame")
    if action in ("motion", "inbetween", "autorig", "restyle") and seed is not None:
        body["seed"] = int(seed)
    if action in ("rig", "autorig"):
        body["humanoid"] = bool(humanoid)
    if action == "export":
        fmt = str(format or "fbx").lower().lstrip(".")
        if fmt not in ("fbx", "abc"):
            raise ValueError("format is fbx or abc")
        body["format"] = fmt
        if name:
            body["name"] = name
    return body


@tool
def animate_character(
    action: str,
    model: str = "",
    prompt: str = "",
    from_frame: Optional[int] = None,
    to_frame: Optional[int] = None,
    seed: Optional[int] = None,
    feet: bool = True,
    format: str = "fbx",
    humanoid: bool = True,
    from_pose: bool = True,
    name: str = "",
    media_path: str = "",
    strength: str = "medium",
    layer: bool = False,
    options: Optional[dict] = None,
    timeout_seconds: float = 900.0,
) -> str:
    """Rig, animate (AI motion from text, in-betweens, restyle, motion from video, lip sync), clean up or export a
    character in the viewer's 3D scene; also its animation layers, constraints and pose library.

    Works on a model item in the 3D scene open in the ComfyUI Image Viewer (the browser must be open on it).
    get_viewer_state() lists the scene's characters: id, name, whether each has a skeleton, a rig, and keys.

    Args:
        action: "rig" (controls on a skinned model), "autorig" (UniRig skeleton + skin for a model without
            one, then controls), "motion" (motion from `prompt`, keyed on the controls), "inbetween" (fill the
            frames between the keyed poses in the range; the keys are hit), "clean_feet" (pin standing feet,
            keep them above the floor), "export" (FBX or Alembic by the export rule), "remove_rig".
        model: The model item's name or id; empty = the selected one (or the only one).
        prompt: For "motion": what the character does, in plain sentences, one per segment
            ("A person walks forward. Then waves with the right hand."). Optional for "inbetween".
        from_frame: First frame (motion: where it starts, default the playhead; inbetween / clean_feet /
            export: the range, default the keyed range or the whole shot).
        to_frame: Last frame.
        seed: For repeatable results (motion, inbetween, autorig).
        feet: For motion / inbetween: pin the feet where they stand and keep them above the floor (default on).
        format: For export: "fbx" (skeleton, skin, keys on every frame) or "abc" (Alembic: baked meshes).
        humanoid: For rig / autorig: humanoid controls (IK legs and arms, COG) when the skeleton reads as one.
        from_pose: For motion: start from the pose the character is in at from_frame (default on).
        name: For export: the file's name (default the model's). For layer_add: the layer's name. For
            pose_save / pose_apply / pose_delete: the pose's name.
        media_path: For video_motion: the film (one person, whole body, a still camera). For lipsync: a sound
            file or a film with sound. A path ComfyUI's machine can read (its input/output folders or allowed ones).
        strength: For restyle: "loose" (the path and timing only), "medium" (also hands and feet), "close"
            (the whole body every third of a second: least change).
        layer: For motion / inbetween / restyle / video_motion: put the result on a new animation layer, so the
            keys under it stay (pose to pose, then AI on top).
        options: Extra fields per action - video_motion: film_from, film_to, step, smooth, in_place, clean
            (Kimodo clean-up); lipsync: start, seconds, strength, lead; layer_add: mode ("override" | "additive"),
            weight; constrain: type (parent | point | orient | aim | scale), driven (item name/id), targets (list),
            maintain_offset, weights, skip, aim; remove_constraint: driven, constraint; list_constraints: item;
            bake_constraints: items, remove; pose_apply: blend, mirror, key.
        timeout_seconds: How long to wait for the viewer (auto-rigging and motion take a minute or two).

    Returns:
        JSON: {"status": "ok", "action", "result": {...}} - e.g. keys and frames written, the exported path,
        whether the rig is humanoid - or {"status": "error", "error"}.
    """
    t0 = time.time()
    try:
        body = request_body(action, model, prompt, from_frame, to_frame, seed, feet, format, humanoid, from_pose, name,
                            media_path, strength, layer, options)
    except ValueError as exc:
        return _error(str(exc))
    try:
        ask = _client().post("/imageviewer/rig/agent", json_data=body)
    except Exception as exc:  # noqa: BLE001
        if "404" in str(exc):
            return _error("the ComfyUI Image Viewer is not installed, or too old for character rigs "
                          "(update ComfyUI-ImageViewer and restart ComfyUI)")
        return _error(f"could not reach the viewer through ComfyUI: {exc}")
    if not isinstance(ask, dict) or "id" not in ask:
        return _error(f"unexpected answer from the viewer: {ask!r}")
    state = ask
    deadline = t0 + float(timeout_seconds)
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            state = _client().get("/imageviewer/rig/agent", params={"id": ask["id"]})
        except Exception as exc:  # noqa: BLE001
            logger.warning("rig request poll failed: %s", exc)
            continue
        if state.get("state") in ("done", "error"):
            break
        if state.get("state") == "sent" and time.time() - t0 > 30:
            return _error("no viewer took the request: open the ComfyUI Image Viewer on the 3D scene with the "
                          "character (the 3D space), then try again")
    if state.get("state") == "done":
        return json.dumps({"status": "ok", "action": action, "result": state.get("result") or {},
                           "seconds_taken": round(time.time() - t0, 1)})
    if state.get("state") == "error":
        return _error(state.get("error") or "the viewer could not do it")
    return _error(f"timed out waiting for the viewer (last state: {state.get('state') or 'unknown'})")
