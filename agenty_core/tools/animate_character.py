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
* ``remove_rig``.

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

ACTIONS = ("rig", "autorig", "motion", "inbetween", "clean_feet", "export", "remove_rig")


def _client():
    from agenty_core.utils.comfyui_client import get_client
    return get_client()


def _error(msg: str) -> str:
    return json.dumps({"status": "error", "error": msg})


def request_body(action, model="", prompt="", from_frame=None, to_frame=None, seed=None, feet=True,
                 format="fbx", humanoid=True, from_pose=True, name="") -> dict:
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
    if action in ("motion", "inbetween", "clean_feet", "export"):
        if from_frame is not None:
            body["from"] = int(from_frame)
        if to_frame is not None:
            body["to"] = int(to_frame)
        if from_frame is not None and to_frame is not None and int(to_frame) <= int(from_frame):
            raise ValueError("to_frame must come after from_frame")
    if action in ("motion", "inbetween", "autorig") and seed is not None:
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
    timeout_seconds: float = 900.0,
) -> str:
    """Rig, animate (AI motion from text, AI in-betweens), clean up or export a character in the viewer's 3D scene.

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
        name: For export: the file's name (default the model's).
        timeout_seconds: How long to wait for the viewer (auto-rigging and motion take a minute or two).

    Returns:
        JSON: {"status": "ok", "action", "result": {...}} - e.g. keys and frames written, the exported path,
        whether the rig is humanoid - or {"status": "error", "error"}.
    """
    t0 = time.time()
    try:
        body = request_body(action, model, prompt, from_frame, to_frame, seed, feet, format, humanoid, from_pose, name)
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
