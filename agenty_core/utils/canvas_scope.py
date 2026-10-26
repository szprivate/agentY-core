"""Which ComfyUI page a canvas tool may touch, for the turn it runs in.

The ComfyUI extension's ``/agent/*`` routes broadcast to EVERY open page unless
told which one (``client_id``, the page's websocket id). A host that runs turns
nobody streams (a turn API, Slack) records on the turn's scope what its canvas is:

``canvas_client``    the page this turn belongs to: aim the routes at it;
``canvas_detached``  the turn has no page at all: touch no canvas.

A caller that sets neither (a plain MCP client) gets what it always got.
"""
from __future__ import annotations

from agenty_core.utils import turn_scope

NO_CANVAS = ("no canvas attached to this turn - it has no ComfyUI page of its own and none "
             "shows this conversation, so there is no open workflow to touch. Save the "
             "workflow to a file and open it with open_workflow_in_canvas instead; do not "
             "retry this tool.")


def client_id() -> str:
    """The page the current turn is bound to ('' when it is not bound to one)."""
    return str(turn_scope.current().get("canvas_client", "") or "")


def detached() -> bool:
    """True when the current turn has no page and must not touch any."""
    s = turn_scope.current()
    return bool(s.get("canvas_detached")) and not client_id()


def aim(body: dict) -> dict:
    """*body* with the turn's page added, for a ``/agent/*`` request."""
    cid = client_id()
    return {**body, "client_id": cid} if cid else body
