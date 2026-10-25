"""Execution modes: where the OPEN-WEIGHT models of a run are executed.

``local``   (default, today's behaviour) - on this machine's ComfyUI. Never touches Comfy Cloud.
``hybrid``  - locally when the workflow's models are on disk AND fit the GPU, else on Comfy Cloud when it has them.
``cloud``   - on Comfy Cloud; no GPU needed here.

Partner (API) nodes - Seedance, GPT Image, Kling, Nano Banana, ... - are NOT affected by the mode. They need no GPU,
they run through the local ComfyUI in every mode, under the usual partner-credit checks. A workflow that mixes partner
nodes with open-weight models is routed by its open-weight part alone.

Which mode is in force, strongest first:

1. the user's per-turn override ("run this in the cloud", "keep it local") - set through the
   ``set_execution_mode_for_turn`` tool, valid for this turn (it expires after :data:`OVERRIDE_TTL_S`);
2. the environment variable ``AGENTY_EXECUTION_MODE`` (a deployment pins it);
3. the host's ``execution.mode`` setting (the Settings page; the installer writes it);
4. ``local``.

The one list of things Comfy Cloud cannot do is :data:`CLOUD_LIMITS`; the agent guidance and the tools that are
local-only read it, so the wording lives in a single place.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

MODES = ("local", "hybrid", "cloud")
DEFAULT_MODE = "local"
OVERRIDE_TTL_S = 30 * 60.0
_SLOT = "execution_mode_override"

# The host registers where its settings live: fn() -> the ``execution`` table (a dict).
_settings_provider: Optional[Callable[[], dict]] = None


def set_settings_provider(fn: Optional[Callable[[], dict]]) -> None:
    global _settings_provider
    _settings_provider = fn


def settings() -> dict:
    """The ``execution`` settings table (``{}`` when no host registered one)."""
    if _settings_provider is None:
        return {}
    try:
        got = _settings_provider()
        return got if isinstance(got, dict) else {}
    except Exception:  # noqa: BLE001 - a broken settings read must not break a local run
        return {}


def normalize(value) -> Optional[str]:
    v = str(value or "").strip().strip('"').lower()
    return v if v in MODES else None


def configured_mode() -> str:
    """Environment, then setting, then ``local`` (no per-turn override)."""
    try:
        from agenty_core.utils.secrets import get_secret
        env = normalize(get_secret("AGENTY_EXECUTION_MODE", ""))
    except Exception:  # noqa: BLE001
        env = normalize(os.environ.get("AGENTY_EXECUTION_MODE"))
    return env or normalize(settings().get("mode")) or DEFAULT_MODE


# ── per-turn override ────────────────────────────────────────────────────────
def set_turn_override(mode: Optional[str]) -> Optional[str]:
    """Set (or with ``None``/"default" clear) this turn's override. Returns the stored mode."""
    from agenty_core.utils import turn_scope
    m = normalize(mode)
    turn_scope.current().set(_SLOT, {"mode": m, "at": time.time()} if m else None)
    return m


def turn_override() -> Optional[str]:
    from agenty_core.utils import turn_scope
    got = turn_scope.current().get(_SLOT)
    if not isinstance(got, dict):
        return None
    if time.time() - float(got.get("at", 0)) > OVERRIDE_TTL_S:
        return None
    return normalize(got.get("mode"))


def effective_mode() -> str:
    return turn_override() or configured_mode()


def describe() -> dict:
    return {"mode": effective_mode(), "configured": configured_mode(), "turn_override": turn_override()}


# Phrases that mean "this one goes to the cloud" / "keep it on this machine". The agent normally sets the override
# itself through the tool; hosts may also run a message through :func:`parse_override`.
_CLOUD_PHRASES = ("in the cloud", "on the cloud", "on comfy cloud", "use comfy cloud", "use the cloud",
                  "run it in cloud", "run in cloud", "cloud run", "send it to the cloud")
_LOCAL_PHRASES = ("keep it local", "keep this local", "run it locally", "run this locally", "run locally",
                  "on my machine", "on my gpu", "don't use the cloud", "do not use the cloud", "no cloud")


def parse_override(text: str) -> Optional[str]:
    t = (text or "").lower()
    if any(p in t for p in _LOCAL_PHRASES):
        return "local"
    if any(p in t for p in _CLOUD_PHRASES):
        return "cloud"
    return None


# ── what Comfy Cloud cannot do ───────────────────────────────────────────────
# One list. ``tools`` are the agent tools that cannot run without this machine; ``advice`` is what to tell the user.
CLOUD_LIMITS = [
    {"id": "sam3_masks", "label": "SAM3 masks / finding things in pictures",
     "tools": ("annotate_image",),
     "advice": "SAM3 runs on this machine's GPU. In cloud mode switch the turn to local, or describe the region "
               "and use a partner/edit model instead of a mask."},
    {"id": "splat_training", "label": "Gaussian splat training (Brush)",
     "tools": ("train_splat", "splat_training_status"),
     "advice": "Brush trains on a local GPU and cannot run on Comfy Cloud. Use a machine with a GPU in local "
               "mode, or an Atlas / Marble world (hosted) instead."},
    {"id": "lora_training_local", "label": "Local LoRA training",
     "tools": ("train_lora_local",),
     "advice": "Local LoRA training (ai-toolkit / musubi) needs this machine's GPU. A hosted trainer (fal) works "
               "in any mode, and asks for its price first."},
    {"id": "custom_nodes", "label": "Custom nodes Comfy Cloud does not have",
     "tools": (),
     "advice": "Comfy Cloud only has its own node set. A workflow with other custom nodes stays local, or the "
               "run is refused with the node names."},
    {"id": "canvas_only", "label": "Canvas-only features",
     "tools": (),
     "advice": "Work that reads or writes the user's open ComfyUI canvas (hooks, canvas edits, screenshots) "
               "happens on this machine's ComfyUI whatever the mode; only the model run moves."},
]
_BY_TOOL = {t: lim for lim in CLOUD_LIMITS for t in lim["tools"]}


def cloud_limits_text() -> str:
    return "\n".join(f"- {lim['label']}: {lim['advice']}" for lim in CLOUD_LIMITS)


def local_only_refusal(tool_name: str) -> Optional[dict]:
    """A clear tool result when *tool_name* needs this machine and the turn is in cloud mode, else ``None``.

    Only ``cloud`` refuses: in ``hybrid`` there may well be a local GPU, and ``local`` is today's behaviour.
    """
    lim = _BY_TOOL.get(tool_name)
    if not lim or effective_mode() != "cloud":
        return None
    return {"status": "error", "error_kind": "local_only", "limit": lim["id"],
            "error": f"{lim['label']} cannot run in cloud mode. {lim['advice']}",
            "hint": "Say so to the user. If they want it, set_execution_mode_for_turn('local') and retry."}
