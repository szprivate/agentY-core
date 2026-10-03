"""Which items in ComfyUI's queue this agent put there.

Stop used to mean ``POST /interrupt``, which ends the job that is *running* and
nothing else. A run is usually several prompts deep — a batch member per variant,
a repaired graph re-queued behind the original — so stopping killed the current
one and then watched ComfyUI cheerfully start the next. Pressing stop repeatedly
until the queue drained was the only way to actually stop.

Clearing the whole queue instead would be worse. The user queues their own work in
the same ComfyUI, and a stop button that threw away somebody's overnight batch
because an agent was also running is not a stop button anybody would press twice.

So we keep note. Every prompt agentY submits is recorded here, and stopping
deletes exactly the pending items in this ledger — never one we did not put there.
Identity by prompt id rather than by ``client_id``: a client_id is generated fresh
per submission (it is the websocket correlation key), so it identifies a job, not
an owner. The ids we hold are the ones we were handed by ``POST /prompt``, which
makes "ours" a fact rather than a guess.

The ledger is per process and deliberately not persisted. A prompt id from a host
that has since exited is not something this host should be deleting.

Each id is kept with its owner: the conversation (turn scope's ``thread_id``) whose
turn submitted it. A host running several conversations at once stops one of them
by owner, so a Stop in one chat does not delete or interrupt another chat's
renders. Ids recorded outside any conversation have the owner "".
"""

from __future__ import annotations

import threading
from collections import OrderedDict

# Bounded so a long-lived host does not accumulate ids forever. Well above any
# real run: the biggest batches are tens of members, not hundreds.
_MAX = 512

_LOCK = threading.Lock()
# prompt_id -> owner, in submission order. An OrderedDict is the cheap way to have
# both "is this ours" in O(1) and "drop the oldest" when the cap is reached.
_OURS: "OrderedDict[str, str]" = OrderedDict()


def _current_owner() -> str:
    try:
        from agenty_core.utils import turn_scope
        return turn_scope.current().thread_id
    except Exception:  # noqa: BLE001
        return ""


def remember(prompt_id: str, owner: str | None = None) -> str:
    """Record a prompt id as ours — owned by *owner*, by default the conversation
    whose turn is running. Returns it, so this can wrap a submit call."""
    pid = str(prompt_id or "").strip()
    if not pid:
        return pid
    who = _current_owner() if owner is None else str(owner)
    with _LOCK:
        _OURS[pid] = who
        _OURS.move_to_end(pid)
        while len(_OURS) > _MAX:
            _OURS.popitem(last=False)
    return pid


def forget(prompt_id: str) -> None:
    with _LOCK:
        _OURS.pop(str(prompt_id or "").strip(), None)


def is_ours(prompt_id: str, owner: str | None = None) -> bool:
    """Whether we submitted *prompt_id* — and, given *owner*, whether that
    conversation did."""
    with _LOCK:
        pid = str(prompt_id or "").strip()
        if pid not in _OURS:
            return False
        return owner is None or _OURS[pid] == owner


def owner_of(prompt_id: str) -> str | None:
    """The conversation that submitted *prompt_id*, or None if it is not ours."""
    with _LOCK:
        return _OURS.get(str(prompt_id or "").strip())


def ours(owner: str | None = None) -> list[str]:
    with _LOCK:
        return [pid for pid, who in _OURS.items() if owner is None or who == owner]


def clear() -> None:
    """Forget everything. For tests, and for a host that is shutting down."""
    with _LOCK:
        _OURS.clear()


def prompt_id_of(entry) -> str:
    """The prompt id in one ComfyUI queue entry.

    An entry is ``[number, prompt_id, prompt, extra_data, outputs]``. Handled
    defensively because this is the one place a shape change in ComfyUI's API
    would turn "stop the agent's jobs" into "delete something else".
    """
    if isinstance(entry, dict):
        return str(entry.get("prompt_id") or "")
    if isinstance(entry, (list, tuple)) and len(entry) > 1:
        return str(entry[1] or "")
    return ""


def cancel_ours(client=None, owner: str | None = None) -> dict:
    """Delete our PENDING prompts from ComfyUI's queue. Never touches the user's.

    With *owner*, only that conversation's: the others' prompts count as kept, and
    ``running_is_ours`` says whether the running job is that conversation's.

    Returns ``{ok, deleted, kept, running, running_is_ours}``. ``ok`` is False when
    the queue could not be read at all — the caller then has no basis for deciding
    whose job is running, and should fall back to a plain interrupt.
    """
    out = {"ok": False, "deleted": [], "kept": 0, "running": [], "running_is_ours": False}
    try:
        if client is None:
            from agenty_core.utils.comfyui_client import get_client
            client = get_client()
        queue = client.get("/queue")
    except Exception:  # noqa: BLE001
        return out
    if not isinstance(queue, dict):
        return out

    pending = [prompt_id_of(e) for e in (queue.get("queue_pending") or [])]
    running = [prompt_id_of(e) for e in (queue.get("queue_running") or [])]
    mine = [pid for pid in pending if pid and is_ours(pid, owner)]
    out.update({
        "ok": True,
        "kept": len([pid for pid in pending if pid and not is_ours(pid, owner)]),
        "running": [pid for pid in running if pid],
        "running_is_ours": any(is_ours(pid, owner) for pid in running if pid),
    })
    if not mine:
        return out
    try:
        # One call with every id: ComfyUI processes the list atomically, and
        # deleting one at a time leaves the queue advancing between requests —
        # an item can start running in the gap and then be missed entirely.
        client.post("/queue", json_data={"delete": mine})
        out["deleted"] = mine
    except Exception:  # noqa: BLE001
        # Reported as read-but-not-deleted rather than as a failure to read: the
        # caller's fallback (interrupt anyway) is right either way.
        pass
    for pid in mine:
        forget(pid)
    return out
