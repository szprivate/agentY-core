"""Where does this run execute? The deterministic router behind ``local | hybrid | cloud`` (no LLM involved).

Partner (API) nodes never decide anything: they need no GPU and run through the local ComfyUI in every mode, under
the partner-credit checks. Only the OPEN-WEIGHT part of a workflow - the model files its nodes load - is routed.

``local``   -> local. Nothing is ever sent to Comfy Cloud.
``hybrid``  -> a) every model is on disk AND the estimate fits the GPU's free VRAM: local;
               b) else, Comfy Cloud has every model and node class: cloud;
               c) else :class:`RouteRefused` saying what is missing (and what to do about it).
``cloud``   -> a workflow with no open-weight part (partner nodes + basic nodes only) runs in the local ComfyUI, since
               that needs no GPU; every other workflow goes to Comfy Cloud, or is refused when the cloud lacks a
               model or node class.

The result is a :class:`Decision`; ``backend`` is ``local`` or ``comfy_cloud`` (the Deadline farm, when switched
on, takes over what would run locally - the executors apply that, as before).

All probes (what is on disk, the VRAM, what the cloud has) are injectable, so the decision is unit-testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from agenty_core.partner_credits import CostGateError

logger = logging.getLogger("agenty_core.exec_route")

MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")
DEFAULT_MARGIN_GB = 2.0
GB = 1024 ** 3
# Modules whose nodes are ComfyUI's own (no extra pack to install on the cloud).
_CORE_MODULES = ("nodes", "comfy_extras", "comfy_api")


class RouteRefused(CostGateError):
    """The run cannot happen anywhere in this mode. Subclasses :class:`CostGateError`, so every call site that
    already turns a refused submission into a tool answer (``payload()``) does so for this too."""

    def __init__(self, message: str, decision: "Decision"):
        super().__init__("cannot_run_here", message, {}, None)
        self.decision = decision

    def payload(self) -> dict:
        d = self.decision
        return {"status": "error", "error_kind": "cannot_run_here", "error": str(self), "mode": d.mode,
                "missing_models": d.missing_local, "missing_on_cloud_models": d.missing_cloud_models,
                "missing_on_cloud_nodes": d.missing_cloud_nodes, "hint": d.hint}


@dataclass
class Decision:
    backend: str = "local"                  # local | comfy_cloud
    mode: str = "local"
    reason: str = ""
    open_weight: bool = False               # does the workflow load model files at all
    partner_nodes: bool = False
    models: list = field(default_factory=list)
    missing_local: list = field(default_factory=list)
    missing_cloud_models: list = field(default_factory=list)
    missing_cloud_nodes: list = field(default_factory=list)
    need_bytes: Optional[int] = None
    free_bytes: Optional[int] = None
    hint: str = ""

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("backend", "mode", "reason", "open_weight", "partner_nodes",
                                              "missing_local", "missing_cloud_models", "missing_cloud_nodes",
                                              "need_bytes", "free_bytes")}


# ── reading the workflow ─────────────────────────────────────────────────────
def _is_api_info(info: Optional[dict]) -> bool:
    info = info or {}
    return bool(info.get("api_node")) or str(info.get("python_module") or "").startswith("comfy_api_nodes")


def _is_core_info(info: Optional[dict]) -> bool:
    mod = str((info or {}).get("python_module") or "")
    return bool(mod) and (mod in _CORE_MODULES or mod.split(".")[0] in _CORE_MODULES) \
        and not mod.startswith("comfy_api_nodes")


def analyse(workflow: dict, info_for: Callable[[str], Optional[dict]]) -> dict:
    """``{"models": [{name, node_id, class_type, input}], "api": [class], "core": [class], "custom": [class]}``."""
    models, api, core, custom = [], set(), set(), set()
    for nid, node in (workflow or {}).items():
        if not isinstance(node, dict) or not node.get("class_type"):
            continue
        cls = str(node["class_type"])
        info = info_for(cls)
        if _is_api_info(info):
            api.add(cls)
            continue
        (core if _is_core_info(info) else custom).add(cls)
        for key, val in (node.get("inputs") or {}).items():
            if isinstance(val, str) and val.lower().endswith(MODEL_EXTS):
                models.append({"name": val, "node_id": str(nid), "class_type": cls, "input": key})
    return {"models": models, "api": sorted(api), "core": sorted(core), "custom": sorted(custom)}


def estimate_need(sizes: list, margin_gb: float = DEFAULT_MARGIN_GB) -> Optional[int]:
    """VRAM a run needs, from the model file sizes: ComfyUI loads and offloads stage by stage, so the peak is
    about the largest file plus a quarter of the rest, plus a margin for activations. ``None`` when no size is
    known."""
    known = sorted((s for s in sizes if s), reverse=True)
    if not known:
        return None
    return int(known[0] + 0.25 * sum(known[1:]) + margin_gb * GB)


# ── probes ───────────────────────────────────────────────────────────────────
class Probes:
    """What the router asks about the world. Replace any attribute with a callable in tests."""

    def local_available(self) -> bool:
        try:
            from agenty_core.utils.comfyui_client import get_client
            return isinstance(get_client().get("/system_stats"), dict)
        except Exception:  # noqa: BLE001
            return False

    def node_info(self, class_type: str) -> Optional[dict]:
        """Local ComfyUI's entry for a class; the cloud's when ComfyUI is not running."""
        try:
            from agenty_core import partner_credits as pc
            info = pc._node_info(class_type)  # noqa: SLF001 - cached local lookup
            if info:
                return info
        except Exception:  # noqa: BLE001
            pass
        try:
            from agenty_core.utils import comfy_cloud
            if comfy_cloud.has_key():
                return comfy_cloud.get_client().node_info(class_type)
        except Exception:  # noqa: BLE001
            pass
        return None

    def local_model(self, name: str) -> dict:
        """``{"present": bool, "bytes": int|None}`` for a model file name."""
        try:
            from agenty_core.tools import comfyui as c
            rel = c.installed_model(name)
            if not rel:
                return {"present": False, "bytes": None}
            size = None
            for dirs in (c._model_dirs() or {}).values():  # noqa: SLF001
                for d in dirs:
                    p = Path(d) / rel
                    if p.is_file():
                        size = p.stat().st_size
                        break
                if size:
                    break
            return {"present": True, "bytes": size}
        except Exception:  # noqa: BLE001
            return {"present": False, "bytes": None}

    def free_vram(self) -> Optional[int]:
        """Free VRAM in bytes; 0 when the machine has no usable GPU; ``None`` when unknown."""
        try:
            from agenty_core.utils.comfyui_client import get_client
            stats = get_client().get("/system_stats")
            devs = stats.get("devices") or []
            gpu = [d for d in devs if str(d.get("type", "")).lower() not in ("cpu", "")]
            if not gpu:
                return 0
            d = gpu[0]
            free = int(d.get("vram_free") or 0) + int(d.get("torch_vram_free") or 0)
            total = int(d.get("vram_total") or 0)
            return min(free, total) if total else free
        except Exception:  # noqa: BLE001
            return None

    def cloud_key(self) -> bool:
        from agenty_core.utils import comfy_cloud
        return comfy_cloud.has_key()

    def cloud_models(self) -> set:
        from agenty_core.utils import comfy_cloud
        return set(comfy_cloud.get_client().model_names())

    def cloud_nodes(self) -> set:
        from agenty_core.utils import comfy_cloud
        return set(comfy_cloud.get_client().nodes())


def _margin_gb() -> float:
    try:
        from agenty_core.utils import execution_mode
        return float(execution_mode.settings().get("vram_margin_gb", DEFAULT_MARGIN_GB))
    except Exception:  # noqa: BLE001
        return DEFAULT_MARGIN_GB


# ── the decision ─────────────────────────────────────────────────────────────
def decide(workflow: dict, *, mode: Optional[str] = None, probes: Optional[Probes] = None,
           raise_on_refuse: bool = True) -> Decision:
    """Route one API-format workflow. Raises :class:`RouteRefused` when there is no place to run it (unless
    *raise_on_refuse* is False, which returns the decision with ``backend == "none"``)."""
    from agenty_core.utils import execution_mode as em
    mode = em.normalize(mode) or em.effective_mode()
    probes = probes or Probes()
    d = Decision(mode=mode)

    if mode == "local":
        # Today's behaviour, to the letter: nothing is analysed, nothing leaves this machine.
        d.reason = "local mode"
        return d

    facts = analyse(workflow, probes.node_info)
    d.models = facts["models"]
    d.open_weight = bool(facts["models"])
    d.partner_nodes = bool(facts["api"])
    names = sorted({m["name"] for m in facts["models"]})
    no_open_weight = not facts["models"] and not facts["custom"]

    def refuse(msg: str, hint: str) -> Decision:
        d.backend = "none"
        d.reason = msg
        d.hint = hint
        if raise_on_refuse:
            raise RouteRefused(msg, d)
        return d

    local_up = probes.local_available()

    if no_open_weight:
        # Partner nodes + basic nodes: no GPU involved, so the local ComfyUI does it (as in local mode).
        if local_up or not probes.cloud_key():
            d.reason = ("partner/basic nodes only - runs in the local ComfyUI (no GPU needed)"
                        if d.partner_nodes else "no model files in the workflow - runs locally")
            return d
        d.backend = "comfy_cloud"
        d.reason = "local ComfyUI is not reachable; partner/basic nodes run on Comfy Cloud"
        return d

    # ── an open-weight part exists ──
    present, sizes = {}, []
    for n in names:
        info = probes.local_model(n) if local_up else {"present": False, "bytes": None}
        present[n] = info
        sizes.append(info.get("bytes"))
    d.missing_local = [n for n in names if not present[n]["present"]]
    free = probes.free_vram() if local_up else 0
    d.free_bytes = free
    d.need_bytes = estimate_need(sizes, _margin_gb())
    fits = free is None or d.need_bytes is None or d.need_bytes <= free

    if mode == "hybrid" and local_up and not d.missing_local and fits:
        d.reason = (f"hybrid: all {len(names)} model(s) on disk and they fit the GPU"
                    + (f" (~{d.need_bytes / GB:.1f} GB of {free / GB:.1f} GB free)" if d.need_bytes and free else ""))
        return d

    # ── cloud (hybrid fallback, or cloud mode) ──
    why_not_local = ("not installed here: " + ", ".join(d.missing_local) if d.missing_local
                     else "local ComfyUI is not reachable" if not local_up
                     else f"needs ~{(d.need_bytes or 0) / GB:.1f} GB of VRAM, {(free or 0) / GB:.1f} GB free"
                     if not fits else "")
    if not probes.cloud_key():
        return refuse(
            f"{mode} mode wants Comfy Cloud for this workflow ({why_not_local or 'cloud mode'}), but there is "
            "no Comfy API key.",
            "Set COMFYUI_API_KEY (Settings > Authentication), or run it locally: "
            + ("download the missing models (find_hf_file + download_hf_model check the drive and size first), "
               if d.missing_local else "free up VRAM, ")
            + "or use a partner model for this step (partner_run).")
    try:
        cloud_models, cloud_nodes = probes.cloud_models(), probes.cloud_nodes()
    except Exception as exc:  # noqa: BLE001
        return refuse(f"Comfy Cloud could not be asked what it has ({exc}).",
                      "Try again in a moment; or run it locally in local mode.")
    d.missing_cloud_models = [n for n in names if Path(n.replace("\\", "/")).name.lower() not in cloud_models]
    used = set(facts["core"]) | set(facts["custom"]) | set(facts["api"])
    d.missing_cloud_nodes = sorted(c for c in used if c not in cloud_nodes)
    if not d.missing_cloud_models and not d.missing_cloud_nodes:
        d.backend = "comfy_cloud"
        d.reason = f"{mode}: {why_not_local or 'cloud mode'} - Comfy Cloud has every model and node"
        return d

    parts = []
    if d.missing_cloud_models:
        parts.append("models Comfy Cloud lacks: " + ", ".join(d.missing_cloud_models))
    if d.missing_cloud_nodes:
        parts.append("nodes Comfy Cloud lacks: " + ", ".join(d.missing_cloud_nodes))
    hint = []
    if d.missing_local:
        hint.append("download the missing local models (find_hf_file + download_hf_model: they check the drive "
                    "and free space first)")
    elif local_up and not fits:
        hint.append("free GPU memory (free_memory) and retry")
    hint.append("or pick a partner-model alternative (partner_run needs no GPU)")
    hint.append("or swap to models the cloud has")
    return refuse(f"{mode} mode cannot run this anywhere: {why_not_local or 'cloud mode'}; " + "; ".join(parts) + ".",
                  "; ".join(hint) + ".")
