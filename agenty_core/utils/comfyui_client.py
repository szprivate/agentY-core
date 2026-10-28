"""
HTTP client wrapper for communicating with the ComfyUI server.

Provides a singleton client that handles authentication and base URL configuration.
The ComfyUI API key is read directly from the .env file.
"""

import contextlib
import contextvars
import json
import logging
import os
import threading
import time
from pathlib import Path

import requests

from agenty_core.utils.secrets import get_secret
from agenty_core.paths import project_root


_DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"

# Error bodies are folded into the exception message, which ends up in toasts and
# chat lines, so they are clipped. A single ``value_not_in_list`` detail can carry
# every model name ComfyUI knows about.
_MAX_DETAIL = 200
_MAX_BODY = 500


def _clip(text: object, limit: int) -> str:
    """*text* as one whitespace-collapsed line, no longer than *limit*."""
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _reason(entry: dict) -> str:
    """One ComfyUI error record as ``message (details)``."""
    message = _clip(entry.get("message") or entry.get("type") or "error", _MAX_DETAIL)
    details = _clip(entry.get("details"), _MAX_DETAIL)
    return f"{message} ({details})" if details and details != message else message


def describe_node_errors(node_errors) -> str:
    """ComfyUI's per-node rejection dict as one readable line, or ''.

    Shared by the two places that meet it, which is the point: a rejected
    ``/prompt`` (400, via :func:`describe_error_response`) and an ACCEPTED one
    that still rejected some outputs (200, ``node_errors`` alongside the
    ``prompt_id``). The second shape used to be dropped on the floor at both
    ends — validation called it valid, submission returned only the id — so a
    graph could lose an entire output branch with nobody ever seeing the reason
    ComfyUI had already written down.
    """
    if not isinstance(node_errors, dict) or not node_errors:
        return ""
    parts: list[str] = []
    for node_id, info in node_errors.items():
        if not isinstance(info, dict):
            continue
        label = f"node {node_id} ({info.get('class_type') or '?'})"
        reasons = "; ".join(
            _reason(e) for e in (info.get("errors") or []) if isinstance(e, dict)
        )
        parts.append(f"{label}: {reasons}" if reasons else label)
    return " | ".join(parts)


def describe_error_response(resp: requests.Response) -> str:
    """Summarise an error response body, or '' when it says nothing useful.

    ComfyUI answers a rejected ``/prompt`` with 400 and a JSON body holding the
    reason — ``{"error": {"type", "message", "details"}, "node_errors": {id:
    {"class_type", "errors": [...]}}}`` (see its ``execution.validate_prompt``).
    ``raise_for_status`` builds its exception from the status line alone, so
    without this every rejection reads ``400 Client Error: Bad Request`` and the
    actual cause — no output node, a missing required input, a model name that
    isn't installed — is thrown away. Other endpoints answer with plain text or a
    different JSON shape, both of which fall through to a clipped dump.
    """
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 — a non-JSON body is still worth showing
        return _clip(getattr(resp, "text", ""), _MAX_BODY)
    if not isinstance(body, dict):
        return _clip(body, _MAX_BODY)

    parts: list[str] = []
    error = body.get("error")
    if isinstance(error, dict):
        parts.append(_reason(error))
    elif error:
        parts.append(_clip(error, _MAX_DETAIL))

    described = describe_node_errors(body.get("node_errors"))
    if described:
        parts.append(described)

    if not parts:  # some routes answer {"message": ...} / {"detail": ...}
        for key in ("message", "detail", "reason"):
            if body.get(key):
                parts.append(_clip(body[key], _MAX_DETAIL))
                break
    return _clip(" | ".join(parts), _MAX_BODY)


def raise_for_status(resp: requests.Response) -> None:
    """``resp.raise_for_status()``, but keep what the server said about why.

    The raised error stays a :class:`requests.HTTPError` carrying the same
    ``response``/``request``, so existing handlers that switch on
    ``exc.response.status_code`` are unaffected — only the message grows.
    """
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = describe_error_response(resp)
        if not detail:
            raise
        raise requests.HTTPError(f"{exc} — {detail}", response=resp) from None


def parse_argv_dir_flag(argv: list, flag: str) -> str | None:
    """Extract a directory value passed to ComfyUI as ``--flag=VALUE`` or ``--flag VALUE``.

    ``/system_stats`` echoes the server's ``sys.argv`` verbatim, so a
    space-separated flag arrives as two consecutive list elements
    (``["--input-directory", "W:\\..."]``) while the ``=`` form arrives as a
    single element (``["--input-directory=W:\\..."]``).  Earlier code only
    handled the ``=`` form, so space-separated launch flags were silently
    missed and callers fell back to ComfyUI's stock install defaults.  Handle
    both forms; return ``None`` when the flag is absent.
    """
    for i, arg in enumerate(argv):
        if not isinstance(arg, str):
            continue
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
        if arg == flag and i + 1 < len(argv) and isinstance(argv[i + 1], str):
            return argv[i + 1]
    return None


# A ComfyUI that is not there must cost one short wait, not 120 s per call. The
# read timeout stays long (a busy server can be slow to answer), but the CONNECT
# timeout is short, and after a failed connect the client answers "down" at once
# for a few seconds instead of every caller paying for its own attempt.
_CONNECT_TIMEOUT = 3.05
_READ_TIMEOUT = 120
_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
_DOWN_FOR = 5.0
_down_until: dict[str, float] = {}
_down_lock = threading.Lock()


def _refuse_if_down(base_url: str) -> None:
    with _down_lock:
        until = _down_until.get(base_url, 0.0)
    if until > time.monotonic():
        raise requests.ConnectionError(f"ComfyUI at {base_url} is not reachable (retrying shortly)")


def _send(base_url: str, method, url: str, **kw):
    """``method(url, **kw)`` with a short connect timeout and a fail-fast window."""
    _refuse_if_down(base_url)
    try:
        resp = method(url, **kw)
    except (requests.ConnectionError, requests.ConnectTimeout):
        with _down_lock:
            _down_until[base_url] = time.monotonic() + _DOWN_FOR
        raise
    with _down_lock:
        _down_until.pop(base_url, None)
    return resp


def forget_down() -> None:
    """Drop the fail-fast memory (a ComfyUI was just started, or a test needs a clean slate)."""
    with _down_lock:
        _down_until.clear()


# ── Prompt hooks: the one seam every queued graph passes ───────────────────────
# A graph reaches ComfyUI's queue from many places (the executor, batches, the
# canvas tap, Atlas renders, the python node, reference measuring...), and all of
# them go through ``post("/prompt")``. A host that has to act before ComfyUI
# starts using the graphics card - agentY stops a local LLM server that holds the
# whole card - registers a hook here instead of patching every caller.
#
# ``hook(prompt)`` gets the API-format graph about to be queued. It may block (it
# runs on the caller's thread, before the POST) and may return a callable, which
# is called once the POST has returned or failed. A hook that raises is logged
# and skipped: a render never fails because a hook did.
_prompt_hooks: list = []
_hooks_suspended = contextvars.ContextVar("agenty_prompt_hooks_suspended", default=False)
_hook_log = logging.getLogger("agenty_core.comfyui_client")


def add_prompt_hook(hook) -> None:
    """Call *hook(prompt)* before every graph is queued (idempotent)."""
    if hook not in _prompt_hooks:
        _prompt_hooks.append(hook)


def remove_prompt_hook(hook) -> None:
    with contextlib.suppress(ValueError):
        _prompt_hooks.remove(hook)


@contextlib.contextmanager
def prompt_hooks_suspended():
    """A ``/prompt`` POST inside this block skips the hooks - for a submit that is
    not a render (server-side validation queues the graph and deletes it again)."""
    token = _hooks_suspended.set(True)
    try:
        yield
    finally:
        _hooks_suspended.reset(token)


def _run_prompt_hooks(json_data) -> list:
    if not _prompt_hooks or _hooks_suspended.get():
        return []
    prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
    afters = []
    for hook in list(_prompt_hooks):
        try:
            after = hook(prompt)
        except Exception as exc:  # noqa: BLE001 - a hook must never cost the render
            _hook_log.warning("prompt hook %r failed: %s", hook, exc)
            continue
        if callable(after):
            afters.append(after)
    return afters


class ComfyUIClient:
    """HTTP client for the ComfyUI REST API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        self.base_url = (base_url or self._load_base_url()).rstrip("/")
        self.api_key = api_key or get_secret("COMFYUI_API_KEY")

    @staticmethod
    def _load_base_url() -> str:
        # An MCP host / .mcpb bundle can inject the ComfyUI URL via env.
        env_url = os.environ.get("COMFYUI_URL")
        if env_url:
            return env_url
        config_path = project_root() / "config" / "settings.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.loads("".join(ln for ln in f if not ln.lstrip().startswith("//")))
            return config.get("comfyui_url", _DEFAULT_COMFYUI_URL)
        return _DEFAULT_COMFYUI_URL

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get(
        self,
        path: str,
        params: dict | None = None,
        stream: bool = False,
        raw: bool = False,
    ) -> requests.Response | dict | list | str:
        """Send a GET request. Returns parsed JSON unless raw=True."""
        url = f"{self.base_url}{path}"
        resp = _send(
            self.base_url, requests.get, url,
            headers=self._headers(), params=params, stream=stream, timeout=_TIMEOUT,
        )
        raise_for_status(resp)
        if raw or stream:
            return resp
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def post(
        self,
        path: str,
        json_data: dict | None = None,
        data: dict | None = None,
        files: dict | None = None,
    ) -> dict | str:
        """Send a POST request. Returns parsed JSON when possible.

        A ``/prompt`` POST (a graph queued) runs the prompt hooks first."""
        url = f"{self.base_url}{path}"
        headers = self._headers()
        if files:
            # Let requests set content-type with boundary for multipart
            headers.pop("Accept", None)
        afters = _run_prompt_hooks(json_data) if path.rstrip("/") == "/prompt" else []
        try:
            resp = _send(
                self.base_url, requests.post, url,
                headers=headers, json=json_data, data=data, files=files, timeout=_TIMEOUT,
            )
        finally:
            for after in afters:
                try:
                    after()
                except Exception as exc:  # noqa: BLE001
                    _hook_log.warning("prompt hook follow-up failed: %s", exc)
        raise_for_status(resp)
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def patch(self, path: str, json_data: dict | None = None,
              timeout: float = 120) -> dict | str:
        """Send a PATCH request. Returns parsed JSON when possible.

        ``timeout`` is settable because the one caller (the console-log
        subscription) unsubscribes from a teardown path, where waiting two
        minutes on an unreachable ComfyUI would stall shutdown.
        """
        url = f"{self.base_url}{path}"
        resp = _send(self.base_url, requests.patch, url, headers=self._headers(),
                     json=json_data, timeout=(min(_CONNECT_TIMEOUT, timeout), timeout))
        raise_for_status(resp)
        try:
            return resp.json()
        except ValueError:
            return resp.text

    def delete(self, path: str) -> dict | str:
        """Send a DELETE request."""
        url = f"{self.base_url}{path}"
        resp = _send(self.base_url, requests.delete, url, headers=self._headers(),
                     timeout=_TIMEOUT)
        raise_for_status(resp)
        try:
            return resp.json()
        except ValueError:
            return resp.text


# ── Singleton ──────────────────────────────────────────────────────────────────

_client: ComfyUIClient | None = None


def get_client() -> ComfyUIClient:
    """Return (and lazily create) the singleton ComfyUI client."""
    global _client  # noqa: PLW0603
    if _client is None:
        _client = ComfyUIClient()
    return _client


# ── ComfyUI directory resolution ───────────────────────────────────────────────
# /system_stats echoes the server's argv, which is fixed for the life of that
# process, so the parsed directories are memoised. Callers that write generated
# artifacts use these to keep output inside the user's ComfyUI install rather
# than inside an agent checkout, where it would dirty the repo (blocking the
# startup updater) and be invisible to ComfyUI's own browsers.

_DIR_CACHE_LOADED = False
_COMFY_USER_DIR: Path | None = None
_COMFY_OUTPUT_DIR: Path | None = None


def reset_comfyui_dir_cache() -> None:
    """Forget the cached directories (call when ComfyUI is restarted)."""
    global _DIR_CACHE_LOADED, _COMFY_USER_DIR, _COMFY_OUTPUT_DIR  # noqa: PLW0603
    _DIR_CACHE_LOADED = False
    _COMFY_USER_DIR = None
    _COMFY_OUTPUT_DIR = None


def _load_comfyui_dirs() -> None:
    global _DIR_CACHE_LOADED, _COMFY_USER_DIR, _COMFY_OUTPUT_DIR  # noqa: PLW0603
    if _DIR_CACHE_LOADED:
        return
    try:
        stats = get_client().get("/system_stats")
        argv = stats.get("system", {}).get("argv", []) if isinstance(stats, dict) else []
        usr = parse_argv_dir_flag(argv, "--user-directory")
        if usr:
            _COMFY_USER_DIR = Path(usr).resolve()
        out = parse_argv_dir_flag(argv, "--output-directory")
        if out:
            _COMFY_OUTPUT_DIR = Path(out).resolve()
    except Exception:  # noqa: BLE001 — ComfyUI being down is a fallback, not an error
        pass
    _DIR_CACHE_LOADED = True


def comfyui_user_dir() -> Path | None:
    """ComfyUI's ``--user-directory``, or None when it can't be determined."""
    _load_comfyui_dirs()
    return _COMFY_USER_DIR


def comfyui_output_dir() -> Path | None:
    """ComfyUI's ``--output-directory``, or None when it can't be determined."""
    _load_comfyui_dirs()
    return _COMFY_OUTPUT_DIR


def comfyui_agent_workflows_dir() -> Path | None:
    """Where agent-generated workflows belong: ``<user>/default/workflows/agentY/``.

    ComfyUI's workflow browser reads ``<user-directory>/default/workflows`` — note
    the ``default`` profile segment. Writing to ``<user-directory>/workflows`` (as
    an earlier version did) creates a folder the browser never shows. The
    ``agentY`` subfolder keeps generated graphs browsable without mixing them into
    hand-made ones.

    Returns None when ComfyUI isn't reachable, so callers can fall back.
    """
    user = comfyui_user_dir()
    if user is None:
        return None
    return user / "default" / "workflows" / "agentY"
