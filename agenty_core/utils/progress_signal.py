"""
progress_signal – Thread-safe buffer for mid-tool progress lines.

Sync tools (e.g. download_hf_model) push formatted progress lines here
via ``push()``.  The async pipeline drains them via ``drain()`` and yields
each line as a ``{"data": "..."}`` event so Chainlit can display them.

Usage from a tool:
    from agenty_core.utils.progress_signal import push as push_progress
    push_progress("⬇️ [████░░░░░░] 45% …")

Usage from the pipeline:
    from agenty_core.utils.progress_signal import drain as drain_progress
    for line in drain_progress():
        yield {"data": line}

One buffer per turn (:mod:`agenty_core.utils.turn_scope`): with several
conversations running at once, a download's progress belongs in the chat that
started it.
"""

from __future__ import annotations

from collections import deque

from agenty_core.utils import turn_scope

# Bounded so a long-running MCP server (whose consumer no longer drains this
# buffer) cannot accumulate progress lines without limit.
_MAX = 200


def _buffer(scope) -> deque:
    return scope.slot("progress_signal", lambda: deque(maxlen=_MAX))


def push(line: str) -> None:
    """Append a progress line to the current turn's buffer (thread-safe)."""
    scope = turn_scope.current()
    with scope.lock:
        _buffer(scope).append(line)


def drain(scope=None) -> list[str]:
    """Atomically read and clear the buffered lines of *scope* (default: the
    current turn's). Empty when nothing was pushed since the last drain."""
    scope = scope or turn_scope.current()
    with scope.lock:
        buf = _buffer(scope)
        if not buf:
            return []
        out = list(buf)
        buf.clear()
        return out
