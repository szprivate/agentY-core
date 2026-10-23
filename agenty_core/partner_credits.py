"""Cost checks for paid ComfyUI partner (API) nodes - "Comfy credits".

Three jobs, all offline-testable:

1. **Estimate.** Every paid (V3 API) node publishes a ``price_badge`` in
   ``/object_info``: a JSONata expression over the node's widget values and its
   connected inputs. :func:`estimate_workflow` evaluates it against an API-format
   workflow and returns per-node and total ``{min_usd, max_usd}``. A node whose
   price cannot be worked out (token-priced, evaluation error, missing values) is
   reported as *unknown* - never guessed.
2. **Balance.** :func:`fetch_balance` reads ``api.comfy.org/customers/balance``
   (cached 60 s). ``effective`` is what can be spent right now (it is below
   ``prepaid`` while charges are pending).
3. **Gate + spend.** :func:`preflight` runs before a submission: a graph that
   costs more than the balance is refused (``insufficient_credits``); one at or
   above the confirm threshold needs an explicit ``confirm_cost`` (``confirm_cost``);
   anything cheaper runs. :func:`track` / :func:`settle` measure what a paid run
   actually cost from the balance before and after.

A graph with no paid node costs nothing here: the per-class schema lookup is a
local ComfyUI call that is cached, and the balance endpoint is never touched.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("agentY.partner_credits")

BALANCE_URL = "https://api.comfy.org/customers/balance"
BALANCE_TTL_S = 60.0
DEFAULT_CONFIRM_ABOVE_USD = 1.0
# How long to wait before the second balance read that follows a paid run
# (charges are posted a moment after the job reports done).
SETTLE_DELAY_S = 4.0

# ── test / host seams ────────────────────────────────────────────────────────
# _http_get(url, headers) -> dict | None ; _info_provider(class_type) -> dict | None
_http_get: Callable[[str, dict], Any] | None = None
_info_provider: Callable[[str], Any] | None = None
_threshold_provider: Callable[[], float | None] | None = None
_sleep: Callable[[float], None] = time.sleep


def set_threshold_provider(fn: Callable[[], float | None] | None) -> None:
    """A host registers where the ``confirm_above_usd`` setting lives."""
    global _threshold_provider
    _threshold_provider = fn


def confirm_threshold() -> float:
    """USD at or above which a run needs an explicit confirmation."""
    if _threshold_provider is not None:
        try:
            v = _threshold_provider()
            if v is not None:
                return max(0.0, float(v))
        except Exception:  # noqa: BLE001
            pass
    try:
        return max(0.0, float(os.environ.get("AGENTY_CONFIRM_ABOVE_USD", DEFAULT_CONFIRM_ABOVE_USD)))
    except ValueError:
        return DEFAULT_CONFIRM_ABOVE_USD


# ═══════════════════════════════════════════════════════════════════════════
# Price estimation
# ═══════════════════════════════════════════════════════════════════════════

_info_cache: dict[str, dict | None] = {}
_info_lock = threading.Lock()
_compiled: dict[str, Any] = {}


def _node_info(class_type: str) -> dict | None:
    """The ``/object_info`` entry of one node class (cached; None if unknown)."""
    with _info_lock:
        if class_type in _info_cache:
            return _info_cache[class_type]
    info: dict | None = None
    try:
        if _info_provider is not None:
            got = _info_provider(class_type)
        else:
            from agenty_core.utils.comfyui_client import get_client
            from urllib.parse import quote
            got = get_client().get(f"/object_info/{quote(class_type, safe='')}")
            if isinstance(got, dict) and class_type in got:
                got = got[class_type]
        info = got if isinstance(got, dict) else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("partner_credits: no schema for %s (%s)", class_type, exc)
        # A transient failure is not cached: the next graph tries again.
        return None
    with _info_lock:
        _info_cache[class_type] = info
    return info


def clear_caches() -> None:
    with _info_lock:
        _info_cache.clear()
    _compiled.clear()
    _balance_cache.update(at=0.0, value=None)


def _is_link(v: Any) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and isinstance(v[0], (str, int)) and isinstance(v[1], int)
            and not isinstance(v[0], bool))


def _schema_default(info: dict, name: str, values: dict) -> Any:
    """The default of widget *name* (``model.resolution`` = inside the selected
    option of the dynamic combo ``model``); None when there is none."""
    inp = info.get("input") or {}
    flat: dict = {}
    for grp in ("required", "optional"):
        flat.update(inp.get(grp) or {})
    if "." not in name:
        spec = flat.get(name)
    else:
        head, _, rest = name.partition(".")
        spec = None
        head_spec = flat.get(head)
        if isinstance(head_spec, (list, tuple)) and len(head_spec) > 1 and isinstance(head_spec[1], dict):
            for opt in head_spec[1].get("options") or []:
                if isinstance(opt, dict) and opt.get("key") == values.get(head):
                    sub = opt.get("inputs") or {}
                    spec = {**(sub.get("required") or {}), **(sub.get("optional") or {})}.get(rest)
                    break
    if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("default")
    return None


def _context_for(node_inputs: dict, badge: dict, info: dict) -> dict:
    """The JSONata context ComfyUI's frontend builds: widget values, which named
    inputs are connected, and how many links each input group has."""
    deps = badge.get("depends_on") or {}
    values = {k: v for k, v in (node_inputs or {}).items() if not _is_link(v)}
    widgets: dict = {}
    for w in deps.get("widgets") or []:
        name = w.get("name") if isinstance(w, dict) else str(w)
        if not name:
            continue
        if name in values:
            widgets[name] = values[name]
        else:
            d = _schema_default(info, name, values)
            if d is not None:
                widgets[name] = d
    links = [k for k, v in (node_inputs or {}).items() if _is_link(v)]
    inputs = {n: {"connected": n in links} for n in (deps.get("inputs") or [])}
    groups = {g: sum(1 for k in links if k.startswith(g + ".")) for g in (deps.get("input_groups") or [])}
    return {"widgets": widgets, "inputs": inputs, "inputGroups": groups}


def _evaluate(expr: str, ctx: dict) -> Any:
    import jsonata  # jsonata-python
    comp = _compiled.get(expr)
    if comp is None:
        comp = _compiled[expr] = jsonata.Jsonata(expr)
    return comp.evaluate(ctx)


def _read_price(res: Any) -> tuple[float, float] | None:
    """(min, max) USD for one run from a badge result, or None when the result is
    not a per-run price (token-priced ``list_usd``, plain ``text``, undefined)."""
    if not isinstance(res, dict):
        return None
    kind = res.get("type")
    try:
        if kind == "usd" and res.get("usd") is not None:
            v = float(res["usd"])
            return v, v
        if kind == "range_usd":
            lo, hi = float(res["min_usd"]), float(res["max_usd"])
            return min(lo, hi), max(lo, hi)
    except (TypeError, ValueError, KeyError):
        return None
    return None


def estimate_node(class_type: str, node_inputs: dict, info: dict | None = None) -> dict | None:
    """Price one node. None = not a paid node. Otherwise
    ``{"min_usd","max_usd","known","approximate","reason"}``."""
    info = info if info is not None else _node_info(class_type)
    badge = (info or {}).get("price_badge")
    if not isinstance(badge, dict) or not badge.get("expr"):
        return None
    out = {"min_usd": None, "max_usd": None, "known": False, "approximate": False, "reason": ""}
    if badge.get("engine", "jsonata") != "jsonata":
        out["reason"] = f"unsupported price engine {badge.get('engine')!r}"
        return out
    try:
        res = _evaluate(badge["expr"], _context_for(node_inputs, badge, info))
    except Exception as exc:  # noqa: BLE001
        out["reason"] = f"price expression failed ({type(exc).__name__})"
        return out
    got = _read_price(res)
    if got is None:
        out["reason"] = ("priced per token or per use, not per run" if isinstance(res, dict)
                         else "price depends on values that are not set")
        return out
    out.update(min_usd=round(got[0], 6), max_usd=round(got[1], 6), known=True,
               approximate=bool(((res.get("format") or {}) if isinstance(res, dict) else {}).get("approximate")))
    return out


def estimate_workflow(workflow: dict, info_for: Callable[[str], Any] | None = None) -> dict:
    """Estimate an API-format workflow ``{node_id: {class_type, inputs}}``.

    Returns ``{"paid": bool, "min_usd", "max_usd", "nodes": [...], "unknown": [...]}``.
    Totals count only nodes whose price is known; ``unknown`` lists the rest.
    """
    nodes: list[dict] = []
    unknown: list[dict] = []
    lo = hi = 0.0
    paid = False
    for nid, node in (workflow or {}).items():
        if not isinstance(node, dict) or not node.get("class_type"):
            continue
        ct = str(node["class_type"])
        info = info_for(ct) if info_for else _node_info(ct)
        est = estimate_node(ct, node.get("inputs") or {}, info if isinstance(info, dict) else None) \
            if isinstance(info, dict) else None
        if est is None:
            continue
        paid = True
        row = {"node_id": str(nid), "class_type": ct, **est}
        nodes.append(row)
        if est["known"]:
            lo += est["min_usd"]
            hi += est["max_usd"]
        else:
            unknown.append({"node_id": str(nid), "class_type": ct, "reason": est["reason"]})
    return {"paid": paid, "min_usd": round(lo, 4), "max_usd": round(hi, 4),
            "nodes": nodes, "unknown": unknown}


# ═══════════════════════════════════════════════════════════════════════════
# Balance
# ═══════════════════════════════════════════════════════════════════════════

_balance_cache: dict = {"at": 0.0, "value": None}
_balance_lock = threading.Lock()


def _api_key() -> str:
    try:
        from agenty_core.utils.secrets import get_secret
        return get_secret("COMFYUI_API_KEY")
    except Exception:  # noqa: BLE001
        return ""


def fetch_balance(force: bool = False) -> dict | None:
    """``{"effective_usd","prepaid_usd","currency"}`` or None when unknown
    (no key, offline, odd answer). Cached for 60 s unless *force*."""
    with _balance_lock:
        if (not force and _balance_cache["value"] is not None
                and time.time() - _balance_cache["at"] < BALANCE_TTL_S):
            return _balance_cache["value"]
    key = _api_key()
    if not key and _http_get is None:
        return None
    try:
        headers = {"X-API-KEY": key, "Accept": "application/json"}
        if _http_get is not None:
            data = _http_get(BALANCE_URL, headers)
        else:
            import requests
            r = requests.get(BALANCE_URL, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
        eff = data.get("effective_balance_cents")
        pre = data.get("prepaid_balance_cents")
        if eff is None and pre is None:
            return None
        value = {"effective_usd": round((eff if eff is not None else pre) / 100.0, 4),
                 "prepaid_usd": round((pre if pre is not None else eff) / 100.0, 4),
                 "currency": data.get("currency") or "usd"}
    except Exception as exc:  # noqa: BLE001 - never print the key
        logger.debug("partner_credits: balance unavailable (%s)", type(exc).__name__)
        return None
    with _balance_lock:
        _balance_cache.update(at=time.time(), value=value)
    return value


# ═══════════════════════════════════════════════════════════════════════════
# Gate
# ═══════════════════════════════════════════════════════════════════════════

class CostGateError(RuntimeError):
    """A submission refused on cost. ``error_kind`` is ``insufficient_credits``
    or ``confirm_cost``; the message says what to do next."""

    def __init__(self, error_kind: str, message: str, estimate: dict, balance: dict | None):
        super().__init__(message)
        self.error_kind = error_kind
        self.estimate = estimate
        self.balance = balance

    def payload(self) -> dict:
        return {"status": "error", "error_kind": self.error_kind, "error": str(self),
                "estimate_usd": {"min": self.estimate.get("min_usd"), "max": self.estimate.get("max_usd")},
                "balance_usd": (self.balance or {}).get("effective_usd"),
                "paid_nodes": [{k: n[k] for k in ("node_id", "class_type", "min_usd", "max_usd")}
                               for n in self.estimate.get("nodes", []) if n.get("known")]}


_confirm_all: contextvars.ContextVar[bool] = contextvars.ContextVar("agenty_confirm_cost", default=False)
_confirmed_paths: dict[str, float] = {}


@contextlib.contextmanager
def confirmed(flag: bool = True):
    """Within this block the user's go-ahead is given (``confirm_cost=True``)."""
    tok = _confirm_all.set(bool(flag) or _confirm_all.get())
    try:
        yield
    finally:
        _confirm_all.reset(tok)


def _norm(path: str) -> str:
    try:
        return str(Path(path).resolve()).lower()
    except Exception:  # noqa: BLE001
        return str(path).lower()


def confirm_path(path: str, max_usd: float) -> None:
    """Remember the go-ahead for one workflow file up to *max_usd* - for the paths
    where the run happens later (end of turn) than the call that was confirmed."""
    _confirmed_paths[_norm(path)] = max(_confirmed_paths.get(_norm(path), 0.0), float(max_usd))


def _money(v: float) -> str:
    return f"${v:,.2f}"


def _range(est: dict) -> str:
    lo, hi = est["min_usd"], est["max_usd"]
    return _money(hi) if abs(hi - lo) < 0.005 else f"{_money(lo)}-{_money(hi)}"


def preflight(workflow: dict, *, confirm: bool = False, workflow_path: str = "") -> dict:
    """Check a workflow before it is submitted.

    Returns the estimate (``paid`` False: nothing was looked up beyond local
    schemas). Raises :class:`CostGateError` for ``insufficient_credits`` or
    ``confirm_cost``.
    """
    est = estimate_workflow(workflow)
    if not est["paid"]:
        return est
    return _gate(est, confirm, workflow_path)


def _gate(est: dict, confirm: bool, workflow_path: str) -> dict:
    bal = fetch_balance()
    est["balance"] = bal
    names = ", ".join(sorted({n["class_type"] for n in est["nodes"]}))
    est["summary"] = (f"{_range(est)} on partner credits ({names})" if est["max_usd"] > 0
                      else f"price unknown ({names})")
    if bal is not None and est["max_usd"] > bal["effective_usd"] + 1e-9:
        raise CostGateError(
            "insufficient_credits",
            f"Not run: the estimate is up to {_money(est['max_usd'])} ({names}) but the Comfy credit "
            f"balance is {_money(bal['effective_usd'])}. Nothing was submitted; the user has to top up "
            "or pick a cheaper setting (lower resolution, shorter duration).", est, bal)
    ok = (confirm or _confirm_all.get()
          or _confirmed_paths.get(_norm(workflow_path), -1.0) >= est["max_usd"] - 1e-9)
    thr = confirm_threshold()
    if est["max_usd"] >= thr and est["max_usd"] > 0 and not ok:
        raise CostGateError(
            "confirm_cost",
            f"Not run: this costs {_range(est)} on partner credits ({names}), at or above the "
            f"{_money(thr)} confirmation level"
            + (f"; balance {_money(bal['effective_usd'])}" if bal else "")
            + ". Tell the user the estimate and ask whether to go ahead; only when they say yes, "
              "run it again with confirm_cost=True.", est, bal)
    return est


def preflight_many(paths: list[str], *, confirm: bool = False) -> dict:
    """:func:`preflight` for several workflow files that run together: the gate
    judges their SUM (ten cheap variants are one expensive decision)."""
    total = {"paid": False, "min_usd": 0.0, "max_usd": 0.0, "nodes": [], "unknown": []}
    all_confirmed = True
    for path in paths:
        try:
            wf = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - the submit step reports an unreadable file
            continue
        est = estimate_workflow(wf)
        if est["paid"]:
            if _confirmed_paths.get(_norm(str(path)), -1.0) < est["max_usd"] - 1e-9:
                all_confirmed = False
            total["paid"] = True
            total["min_usd"] = round(total["min_usd"] + est["min_usd"], 4)
            total["max_usd"] = round(total["max_usd"] + est["max_usd"], 4)
            total["nodes"] += est["nodes"]
            total["unknown"] += est["unknown"]
    if total["paid"]:
        _gate(total, confirm or all_confirmed, "")
    return total


def estimate_report(workflow: dict) -> dict:
    """The estimate plus balance and threshold, for the agent to quote (never raises)."""
    est = estimate_workflow(workflow)
    if est["paid"]:
        est["balance"] = fetch_balance()
    est["confirm_above_usd"] = confirm_threshold()
    est["needs_confirmation"] = bool(est["paid"] and est["max_usd"] >= est["confirm_above_usd"])
    return est


# ═══════════════════════════════════════════════════════════════════════════
# Actual spend
# ═══════════════════════════════════════════════════════════════════════════

_track_lock = threading.Lock()
_inflight: dict[str, dict] = {}
_watermark: float | None = None     # effective balance (cents) after the last settle
_listeners: list[Callable[[float, dict], None]] = []
_TURN_SLOT = "partner_cost_usd"


def add_listener(fn: Callable[[float, dict], None]) -> None:
    """A host hears every measured spend: ``fn(usd, info)``."""
    if fn not in _listeners:
        _listeners.append(fn)


def _cents() -> float | None:
    b = fetch_balance(force=True)
    return None if b is None else round(b["effective_usd"] * 100.0, 4)


def track(prompt_id: str, report: dict | None) -> None:
    """Note a paid submission so :func:`settle` can measure it."""
    global _watermark
    if not prompt_id or not report or not report.get("paid"):
        return
    with _track_lock:
        if not _inflight:
            _watermark = _cents()      # nothing else running: a clean baseline
        _inflight[prompt_id] = {"min_usd": report.get("min_usd"), "max_usd": report.get("max_usd"),
                                "at": time.time()}


def _log(entry: dict) -> None:
    try:
        from agenty_core.paths import project_root
        p = project_root() / ".logs" / "partner_cost.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001
        pass


def settle(prompt_id: str) -> float | None:
    """What a tracked paid run cost, from the balance now against the balance at
    submission (shared fairly when runs overlap: each settle takes the change since
    the previous one). None when the run was not tracked or the balance is unknown.
    Blocks ~4 s when the first read shows no change yet."""
    global _watermark
    with _track_lock:
        meta = _inflight.get(prompt_id)
    if meta is None:
        return None
    before = _watermark
    after = _cents()
    if before is None or after is None:
        with _track_lock:
            _inflight.pop(prompt_id, None)
        return None
    if after >= before - 1e-9:                     # charge not posted yet
        _sleep(SETTLE_DELAY_S)
        again = _cents()
        after = again if again is not None else after
    usd = round(max(0.0, before - after) / 100.0, 4)
    with _track_lock:
        _inflight.pop(prompt_id, None)
        _watermark = after
    try:
        from agenty_core.utils import turn_scope
        sc = turn_scope.current()
        with sc.lock:
            sc.set(_TURN_SLOT, round(float(sc.get(_TURN_SLOT, 0.0)) + usd, 4))
    except Exception:  # noqa: BLE001
        pass
    info = {"prompt_id": prompt_id, "estimate_min_usd": meta.get("min_usd"),
            "estimate_max_usd": meta.get("max_usd")}
    _log({"ts": time.time(), "usd": usd, **info})
    for fn in list(_listeners):
        try:
            fn(usd, info)
        except Exception:  # noqa: BLE001
            pass
    return usd


def turn_total() -> float:
    """Partner credits measured so far in the current turn (USD)."""
    try:
        from agenty_core.utils import turn_scope
        return float(turn_scope.current().get(_TURN_SLOT, 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def gate_submission(workflow: dict, workflow_path: str = "") -> dict:
    """The check every submission path runs just before POSTing /prompt."""
    return preflight(workflow, workflow_path=workflow_path)
