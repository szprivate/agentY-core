"""
comfyui_progress – WebSocket-based live progress streamer for ComfyUI jobs.

Connects to ``ws://<comfyui>/ws?clientId=<uuid>`` and yields one-line status
updates (queue position, per-node progress bars, errors) as ComfyUI emits
events.  This is the sole completion path for ComfyUI jobs in agentY.

Yields:
    str  — human-readable status line (progress bar, node-start, etc.)
    dict — terminal result, exactly one of:
              {"history": <stripped_history_dict>}     on success
              {"error": str, "details"?: dict}         on error / timeout

The caller treats any dict yield as the end of the stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator
from urllib.parse import urlparse

logger = logging.getLogger("agentY.comfyui_progress")


def _bar(value: int, max_value: int, width: int = 20) -> str:
    if max_value <= 0:
        return ""
    pct = int(value / max_value * 100)
    filled = int(width * value / max_value)
    return f"[{'█' * filled}{'░' * (width - filled)}] {value}/{max_value} ({pct}%)"


def _has_outputs(entry) -> bool:
    """True if a /history entry has at least one saved output file."""
    if not isinstance(entry, dict):
        return False
    for node_out in (entry.get("outputs") or {}).values():
        if isinstance(node_out, dict) and any(
            node_out.get(k) for k in ("images", "gifs", "videos", "audio")
        ):
            return True
    return False


def _exception_is_interrupt(exc_type: str, message: str) -> bool:
    """True when a node exception is really a cancellation, not a workflow fault.

    Interrupting the queue mid-run doesn't always surface as the top-level
    ``execution_interrupted`` event: a node that's actively working raises an
    interrupt EXCEPTION instead, reported as a normal ``execution_error``. Core
    nodes raise ``InterruptProcessingException``; the API nodes (OpenAI/Kling/…)
    raise ``comfy_api_nodes…ProcessingInterrupted: "Task cancelled"``. Both carry
    "interrupt" in the exception type; match the API-node "task cancelled" message
    too. Recognising these keeps a deliberate stop out of the repair/heal loop.
    """
    et = (exc_type or "").lower()
    m = (message or "").lower()
    if "interrupt" in et:                       # InterruptProcessingException, ProcessingInterrupted
        return True
    return "task cancelled" in m or "task canceled" in m


def _status_is_interrupt(status_info: dict) -> bool:
    """True when a history ``status_str == "error"`` is actually an interruption.

    When the user or agent stops the queue, ComfyUI marks the prompt's status as
    ``error`` but the stop shows up in the status ``messages`` array — either as an
    ``execution_interrupted`` entry, or as an ``execution_error`` carrying an
    interrupt/"Task cancelled" exception (API nodes). Neither is a workflow fault,
    so distinguishing them keeps a deliberate stop out of the repair/heal loop.
    """
    for msg in (status_info.get("messages") or []):
        # messages are ``[event_type, data]`` pairs.
        if not isinstance(msg, (list, tuple)) or not msg:
            continue
        et = str(msg[0]).lower()
        if "interrupt" in et or "cancel" in et:
            return True
        data = msg[1] if len(msg) > 1 and isinstance(msg[1], dict) else {}
        if _exception_is_interrupt(data.get("exception_type", ""),
                                   data.get("exception_message", "")):
            return True
    return False


def _check_history(client, prompt_id: str):
    """Inline check of /history/{prompt_id}.  Returns terminal dict or None.

    A ``completed`` status is only treated as terminal once the prompt's output
    files have actually been written into ``/history`` — ComfyUI marks the job
    completed a moment before the outputs land, so returning early here would
    yield a history with empty ``outputs`` ("no output files found").  Returning
    None keeps the caller polling until the outputs appear.
    """
    from agenty_core.tools.comfyui import _strip_history

    try:
        raw = client.get(f"/history/{prompt_id}")
    except Exception as exc:
        logger.debug("history check failed: %s", exc)
        return None
    if not isinstance(raw, dict) or prompt_id not in raw:
        return None
    entry = raw[prompt_id]
    status_info = entry.get("status", {})
    if status_info.get("status_str") == "error":
        if _status_is_interrupt(status_info):
            return {"interrupted": True, "error": "Execution interrupted"}
        return {"error": "ComfyUI job failed", "details": _strip_history(raw)}
    if status_info.get("completed") and _has_outputs(entry):
        return {"history": _strip_history(raw)}
    return None


def _prompt_in_queue(client, prompt_id: str):
    """Is *prompt_id* currently running or pending in the ComfyUI queue?

    Returns True (queued), False (not queued), or None (couldn't check). ComfyUI
    ``/queue`` entries are lists shaped ``[number, prompt_id, prompt, ...]``.
    """
    try:
        q = client.get("/queue")
    except Exception as exc:  # noqa: BLE001
        logger.debug("queue check failed: %s", exc)
        return None
    if not isinstance(q, dict):
        return None
    for key in ("queue_running", "queue_pending"):
        for entry in (q.get(key) or []):
            if isinstance(entry, (list, tuple)) and len(entry) > 1 and str(entry[1]) == str(prompt_id):
                return True
    return False


async def _check_terminal(client, prompt_id: str):
    """Terminal-state check that does NOT rely solely on /history.

    Extends ``_check_history`` with a **queue** check so a job that left the queue
    without a recorded success/error (ComfyUI sometimes drops such a prompt from
    /history entirely) is detected in seconds instead of hanging until the hard
    timeout. Returns a terminal dict (``{"history": …}`` or ``{"error": …}``) or
    ``None`` (still running / can't tell — keep waiting).
    """
    hist = _check_history(client, prompt_id)
    if hist is not None:
        return hist
    # Still queued (or the queue can't be read) → not terminal.
    if _prompt_in_queue(client, prompt_id) is not False:
        return None
    # Not in the queue and /history isn't terminal: the job has finished/failed.
    # Give lagging outputs a brief window before declaring the outcome.
    for _ in range(6):  # ~2.4 s
        await asyncio.sleep(0.4)
        hist = _check_history(client, prompt_id)
        if hist is not None:
            return hist
        if _prompt_in_queue(client, prompt_id):  # re-queued (rare) → keep waiting
            return None
    return {
        "error": "ComfyUI job left the queue without producing a result — it "
                 "likely failed to execute (check the ComfyUI log).",
        "details": {"prompt_id": prompt_id, "reason": "queue_drained_no_output"},
    }


# A long job is legitimate — a video render can take an hour — so the wait has
# no cap by default. What it must not do is go quiet or miss its own ending:
# the terminal state is re-checked on a wall clock whatever else ComfyUI is
# saying, and a heartbeat says where the job stands when nothing else has.
_CHECK_EVERY = 30.0
_HEARTBEAT = 60.0
_POLL_EVERY = 5.0         # without a socket


def _queue_status(client, prompt_id: str, started: float) -> str:
    """One line on where *prompt_id* stands, for the heartbeat."""
    import time
    mins = int((time.monotonic() - started) // 60)
    try:
        q = client.get("/queue")
    except Exception:  # noqa: BLE001
        q = None
    if isinstance(q, dict):
        if any(isinstance(e, (list, tuple)) and len(e) > 1 and str(e[1]) == str(prompt_id)
               for e in q.get("queue_running") or []):
            return f"⏳ Still running in ComfyUI — {mins} min so far"
        pending = [str(e[1]) for e in q.get("queue_pending") or []
                   if isinstance(e, (list, tuple)) and len(e) > 1]
        if str(prompt_id) in pending:
            ahead = pending.index(str(prompt_id)) + len(q.get("queue_running") or [])
            return f"⏳ Queued in ComfyUI — {ahead} job(s) ahead, waiting {mins} min"
    return f"⏳ Waiting for ComfyUI — {mins} min so far"


async def _poll_until_done(client, prompt_id: str, started: float, *, timeout: float | None = None):
    """Follow a job by polling when the WebSocket is not there to follow it.

    A dropped socket (ComfyUI restarted, the network blinked) used to end the
    wait with "WebSocket connection failed" while the job itself kept running,
    so its result was never collected. Polling the terminal state is slower to
    notice the end, and enough.
    """
    import time
    last_line = time.monotonic()
    while timeout is None or time.monotonic() - started < timeout:
        await asyncio.sleep(_POLL_EVERY)
        fallback = await _check_terminal(client, prompt_id)
        if fallback is not None:
            yield fallback
            return
        if time.monotonic() - last_line >= _HEARTBEAT:
            last_line = time.monotonic()
            yield _queue_status(client, prompt_id, started)
    yield {"error": f"ComfyUI job did not finish within {timeout:.0f}s"}


async def _fetch_history_with_outputs(client, prompt_id: str, *, attempts: int = 10, delay: float = 0.4):
    """Fetch /history for a finished prompt, retrying until its outputs appear.

    ComfyUI emits ``execution_success`` a beat before the prompt's output files
    are written into ``/history``; a single immediate fetch then returns an entry
    with empty ``outputs`` — the intermittent "no output files found" bug.  Retry
    a handful of times (cheap, no LLM tokens) until outputs are present, then
    return the stripped history.  Falls back to whatever was last fetched after
    the final attempt so a genuinely output-less workflow still terminates.
    """
    from agenty_core.tools.comfyui import _strip_history

    raw = None
    for attempt in range(1, attempts + 1):
        try:
            raw = client.get(f"/history/{prompt_id}")
        except Exception as exc:
            logger.debug("history fetch failed (attempt %d/%d): %s", attempt, attempts, exc)
            raw = None
        if isinstance(raw, dict) and _has_outputs(raw.get(prompt_id, {})):
            if attempt > 1:
                logger.info(
                    "executor: outputs appeared in /history after %d attempt(s)", attempt
                )
            return _strip_history(raw)
        if attempt < attempts:
            await asyncio.sleep(delay)
    logger.warning(
        "executor: /history for %s still has no outputs after %d attempts (%.1fs)",
        prompt_id, attempts, attempts * delay,
    )
    return _strip_history(raw) if isinstance(raw, dict) else {}


async def stream_comfyui_job(
    prompt_id: str,
    client_id: str,
    *,
    timeout: float | None = None,
    node_titles: dict[str, str] | None = None,
    console: bool | None = None,
) -> AsyncGenerator:
    """Stream live progress for *prompt_id* via the ComfyUI WebSocket.

    Args:
        prompt_id:   Returned by POST /prompt.
        client_id:   The same client_id passed to /prompt; used to subscribe.
        timeout:     Optional cap on total wait time (seconds). None (default):
                     wait as long as the job is queued or running — renders can
                     legitimately take an hour. The job's state is re-checked
                     every 30 s regardless, so an end that the socket never
                     reported is still noticed.
        node_titles: Optional mapping of node_id -> display name, used to
                     annotate progress messages with human-readable names.
        console:     Relay ComfyUI's own terminal output alongside the progress
                     lines (model loads, warnings, node prints).  ``None`` =
                     let ``AGENTY_COMFY_CONSOLE`` decide, which defaults to on.

    Yields:
        Progress strings, then a single terminal dict.
    """
    import websockets

    from agenty_core.utils.comfyui_client import get_client
    from agenty_core.utils.comfyui_console import attach as _console_attach

    client = get_client()
    parsed = urlparse(client.base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/ws?clientId={client_id}"

    headers: list[tuple[str, str]] = []
    if client.api_key:
        headers.append(("Authorization", f"Bearer {client.api_key}"))

    # If the job already completed before we got here (common in batch flows),
    # short-circuit without opening a socket.
    pre = _check_history(client, prompt_id)
    if pre is not None:
        yield pre
        return

    import time as _time

    last_progress_pct: int = -1
    last_emit_loop_t: float = 0.0
    started = _time.monotonic()
    last_check = started      # wall clock of the last terminal-state check
    last_line = started       # wall clock of the last line about THIS job
    ws_lost = False
    RECV_TIMEOUT = 5.0  # seconds — also drives periodic history fallback check

    # ComfyUI's own terminal, relayed for the life of this job. Best-effort and
    # entirely optional: a tap that cannot subscribe simply never yields a line.
    tap = _console_attach(console)

    try:
        connect_kwargs: dict = {"max_size": None}
        if headers:
            # websockets >= 12 uses additional_headers
            connect_kwargs["additional_headers"] = headers

        async with websockets.connect(ws_url, **connect_kwargs) as ws:
            # Two sources feed this loop: the job's event socket and the console
            # tap. Both are held as futures and waited on together, so console
            # output appears as it is printed without putting a receive that may
            # be mid-message on a cancellation treadmill.
            recv_fut: asyncio.Future | None = None
            console_fut: asyncio.Future | None = None
            loop = asyncio.get_running_loop()
            last_heard = loop.time()
            try:
                while timeout is None or _time.monotonic() - started < timeout:
                    if recv_fut is None:
                        recv_fut = asyncio.ensure_future(ws.recv())
                    if tap is not None and console_fut is None:
                        console_fut = asyncio.ensure_future(tap.get())

                    await asyncio.wait(
                        {f for f in (recv_fut, console_fut) if f is not None},
                        timeout=RECV_TIMEOUT,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if console_fut is not None and console_fut.done():
                        try:
                            _lines = console_fut.result()
                        except Exception as _exc:  # noqa: BLE001 — console trouble
                            logger.debug("console relay dropped: %s", _exc)
                            _lines = []          # must never take a run down
                            tap.close()
                            tap = None
                        console_fut = None
                        for _line in _lines:
                            yield f"🖥 {_line}"

                    # No event for RECV_TIMEOUT — re-check the terminal state in
                    # case we missed the completion/error event (server restart,
                    # the prompt finished between pre-check and ws.connect, or the
                    # job errored and was dropped from /history). The queue-aware
                    # check ends the wait as soon as the prompt leaves the queue,
                    # instead of hanging until the hard timeout. Timed from the
                    # last thing the JOB socket said, not from "the wait expired",
                    # so a chatty console — which wakes the wait early, over and
                    # over — can never starve the check.
                    if loop.time() - last_heard >= RECV_TIMEOUT:
                        last_heard = loop.time()
                        last_check = _time.monotonic()
                        fallback = await _check_terminal(client, prompt_id)
                        if fallback is not None:
                            yield fallback
                            return

                    # ...and on a wall clock, whatever the socket is saying.
                    # Other jobs' traffic resets `last_heard` above, so while
                    # ComfyUI was busy with anything else the check never ran
                    # and a missed ending meant waiting for as long as that
                    # lasted, with nothing on screen.
                    _now = _time.monotonic()
                    if _now - last_check >= _CHECK_EVERY:
                        last_check = _now
                        fallback = await _check_terminal(client, prompt_id)
                        if fallback is not None:
                            yield fallback
                            return
                    if _now - last_line >= _HEARTBEAT:
                        last_line = _now
                        yield _queue_status(client, prompt_id, started)

                    if recv_fut is None or not recv_fut.done():
                        continue
                    last_heard = loop.time()
                    try:
                        raw = recv_fut.result()
                    except Exception as _exc:  # noqa: BLE001 — socket closed / dropped
                        logger.warning("comfyui_progress: WebSocket lost for prompt_id=%s (%s) "
                                       "— following the job by polling", prompt_id, _exc)
                        recv_fut = None
                        ws_lost = True
                        break
                    recv_fut = None

                    if isinstance(raw, (bytes, bytearray)):
                        # Binary preview frames — not used for progress.
                        continue

                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")
                    data = msg.get("data", {}) or {}
                    msg_prompt_id = data.get("prompt_id")

                    # Filter to our prompt where the message carries one.
                    if msg_prompt_id and msg_prompt_id != prompt_id:
                        continue
                    if msg_prompt_id == prompt_id:
                        last_line = _time.monotonic()   # news of this job: no heartbeat due

                    if msg_type == "status":
                        qrem = (
                            data.get("status", {})
                            .get("exec_info", {})
                            .get("queue_remaining")
                        )
                        if qrem is not None and qrem > 0:
                            yield f"⏳ Queue: {qrem} job(s) ahead"

                    elif msg_type == "execution_start":
                        yield "▶ Execution started"
                        last_progress_pct = -1

                    elif msg_type == "execution_cached":
                        cached = data.get("nodes", []) or []
                        if cached:
                            yield f"💾 {len(cached)} node(s) served from cache"

                    elif msg_type == "executing":
                        node = data.get("node")
                        if node is None:
                            # null node = prompt finished (older protocol); confirm via history
                            fallback = _check_history(client, prompt_id)
                            if fallback is not None:
                                yield fallback
                                return
                        else:
                            last_progress_pct = -1
                            _title = node_titles.get(str(node), "") if node_titles else ""
                            _node_label = f" · {_title}" if _title else ""
                            yield f"🎨 Running node {node}{_node_label}"

                    elif msg_type == "progress":
                        value = int(data.get("value", 0) or 0)
                        max_v = int(data.get("max", 0) or 0)
                        node = data.get("node")
                        if max_v > 0:
                            pct = int(value / max_v * 100)
                            loop_t = loop.time()
                            # Throttle: emit on first/last step, ≥10% jump, or ≥1s elapsed.
                            is_endpoint = value <= 1 or value >= max_v
                            big_jump = pct - last_progress_pct >= 10
                            time_due = loop_t - last_emit_loop_t >= 1.0
                            if is_endpoint or big_jump or time_due:
                                if node:
                                    _title = node_titles.get(str(node), "") if node_titles else ""
                                    node_label = f" — node {node}" + (f" · {_title}" if _title else "")
                                else:
                                    node_label = ""
                                yield f"🎨 {_bar(value, max_v)}{node_label}"
                                last_progress_pct = pct
                                last_emit_loop_t = loop_t

                    elif msg_type == "execution_success":
                        # Outputs can lag this event by a moment — retry /history
                        # until they appear instead of fetching once.
                        yield {"history": await _fetch_history_with_outputs(client, prompt_id)}
                        return

                    elif msg_type == "execution_error":
                        err_msg = data.get("exception_message", "Unknown error")
                        exc_type = data.get("exception_type", "")
                        node_type = data.get("node_type", "?")
                        node_id = data.get("node_id", "?")
                        # A cancellation surfaced as a node exception (API nodes raise
                        # ProcessingInterrupted "Task cancelled" when the queue is
                        # stopped mid-run) is a deliberate stop, not a workflow fault —
                        # flag it so the executor skips it instead of repairing it.
                        if _exception_is_interrupt(exc_type, err_msg):
                            yield "🛑 Execution interrupted"
                            yield {"interrupted": True, "error": "Execution interrupted"}
                            return
                        yield f"❌ Error in {node_type} (node {node_id}): {err_msg}"
                        yield {
                            "error": "ComfyUI execution failed",
                            "details": {
                                "node_id": node_id,
                                "node_type": node_type,
                                "exception_type": exc_type,
                                "exception_message": err_msg,
                                "traceback": data.get("traceback", []),
                            },
                        }
                        return

                    elif msg_type == "execution_interrupted":
                        yield "🛑 Execution interrupted"
                        # NOT a workflow fault — flag it so the executor skips it
                        # instead of recording an error and firing the repair/heal loop.
                        yield {"interrupted": True, "error": "Execution interrupted"}
                        return

                if not ws_lost:
                    # Only reached with a cap the caller asked for.
                    yield {"error": f"ComfyUI job did not finish within {timeout:.0f}s"}
                    return
            finally:
                for _fut in (recv_fut, console_fut):
                    if _fut is not None:
                        _fut.cancel()

    except Exception as exc:  # noqa: BLE001 — could not connect, or the socket broke
        logger.warning("comfyui_progress: WebSocket failed for prompt_id=%s (%s) "
                       "— following the job by polling", prompt_id, exc)
        ws_lost = True
    finally:
        if tap is not None:
            tap.close()

    # The job outlives a socket that could not be opened or was dropped: keep
    # following it, by polling, until it really ends.
    if ws_lost:
        async for ev in _poll_until_done(client, prompt_id, started, timeout=timeout):
            yield ev
