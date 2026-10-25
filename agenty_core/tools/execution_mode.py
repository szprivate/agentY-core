"""Tools for the execution modes (``local | hybrid | cloud``): look at the mode, change it for one turn.

The modes only decide where OPEN-WEIGHT models run. Partner (API) nodes run in the local ComfyUI in every mode.
See :mod:`agenty_core.utils.execution_mode` and docs/execution_modes.md.
"""
from __future__ import annotations

import json

from agenty_core._compat import tool
from agenty_core.utils import execution_mode as em


@tool
def set_execution_mode_for_turn(mode: str, reason: str = "") -> str:
    """Override where open-weight models run for THIS turn only.

    Call it when the user says where a run should happen: "run this in the cloud" / "use Comfy Cloud" ->
    ``"cloud"``; "keep it local" / "run it on my GPU" -> ``"local"``; "let it decide" -> ``"hybrid"``;
    ``"default"`` drops the override. It lasts for this turn (30 minutes at most) and does not change the setting.
    Partner (API) nodes are not affected: they run in the local ComfyUI in every mode.

    Args:
        mode: "local", "hybrid", "cloud" or "default".
        reason: the user's words, for the log.
    """
    if str(mode or "").strip().lower() in ("default", "none", "off", ""):
        em.set_turn_override(None)
        return json.dumps({"status": "ok", "mode": em.effective_mode(), "override": None})
    if em.normalize(mode) is None:
        return json.dumps({"status": "error", "error": f"unknown mode {mode!r}: use local, hybrid, cloud or default"})
    em.set_turn_override(mode)
    out = {"status": "ok", "mode": em.effective_mode(), "override": em.turn_override(), "reason": reason}
    if em.effective_mode() != "local":
        from agenty_core.utils import comfy_cloud
        if not comfy_cloud.has_key():
            out["warning"] = "No Comfy API key (COMFYUI_API_KEY): a run that needs the cloud will be refused."
    return json.dumps(out)


@tool
def get_execution_mode() -> str:
    """Where open-weight models run right now: the effective mode, the configured one, any per-turn override,
    whether a Comfy API key is set, the Comfy credit balance (a free read), and what Comfy Cloud cannot do."""
    from agenty_core import partner_credits as pc
    from agenty_core.utils import comfy_cloud
    d = em.describe()
    d["comfy_key_set"] = comfy_cloud.has_key()
    bal = pc.fetch_balance() if d["comfy_key_set"] else None
    d["balance_usd"] = None if bal is None else bal["effective_usd"]
    d["cloud_cannot"] = [lim["label"] for lim in em.CLOUD_LIMITS]
    d["note"] = "Partner (API) nodes run in the local ComfyUI in every mode."
    return json.dumps(d)
