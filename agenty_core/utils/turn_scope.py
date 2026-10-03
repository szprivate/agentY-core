"""Which turn the code running right now belongs to.

A host used to run one turn at a time, so the buffers a turn fills — the canvas
patches it pushes, its tool activity, its progress lines, the workflows it
submitted — could live in module-level variables: whoever drained them was the
one turn there was. With several conversations running at once those buffers are
shared by all of them, and one chat's canvas edits, progress and Stop would land
in another's.

So each turn runs inside a :class:`Scope`, carried by a context variable. Python
copies context into asyncio tasks and into ``asyncio.to_thread`` (which is how
Strands runs a tool), so code deep inside a tool reaches its own turn's scope
without being handed it. A thread a turn starts itself does not inherit context:
start it through :func:`bind`.

Code outside any turn — the CLI, a test, a background watcher — gets one shared
default scope, which is exactly the single-turn behaviour there always was.

A buffer becomes per turn with :meth:`Scope.slot`::

    def push(event):
        buf = turn_scope.current().slot("canvas_patch", list)
        ...

and a reader in another thread (the HTTP stream that forwards a turn's events)
names the scope it is reading for, instead of relying on its own context.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any, Callable


class Scope:
    """One turn's private state: named slots, created on first use."""

    __slots__ = ("request_id", "thread_id", "_slots", "_lock")

    def __init__(self, request_id: str = "", thread_id: str = "") -> None:
        self.request_id = str(request_id or "")
        self.thread_id = str(thread_id or "")
        self._slots: dict[str, Any] = {}
        self._lock = threading.RLock()

    def slot(self, name: str, factory: Callable[[], Any]) -> Any:
        """The value in slot *name*, made by *factory* the first time."""
        with self._lock:
            if name not in self._slots:
                self._slots[name] = factory()
            return self._slots[name]

    def get(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self._slots.get(name, default)

    def set(self, name: str, value: Any) -> None:
        with self._lock:
            self._slots[name] = value

    @property
    def lock(self) -> threading.RLock:
        """A lock for the slots' contents (each slot's own buffer is not locked)."""
        return self._lock

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Scope(request_id={self.request_id!r}, thread_id={self.thread_id!r})"


DEFAULT = Scope()
_current: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("agenty_turn_scope",
                                                                        default=None)


def current() -> Scope:
    """The scope of the turn this code runs for, or the shared default."""
    return _current.get() or DEFAULT


def enter(scope: Scope) -> contextvars.Token:
    """Make *scope* current in this context. Pair with :func:`leave`."""
    return _current.set(scope)


def leave(token: contextvars.Token) -> None:
    try:
        _current.reset(token)
    except ValueError:  # a token from another context: nothing of ours to undo
        _current.set(None)


def bind(fn: Callable) -> Callable:
    """*fn* wrapped to run in the context it was bound in — for a thread a turn
    starts itself (threads do not inherit context; asyncio tasks do)."""
    ctx = contextvars.copy_context()

    def _run(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return _run
