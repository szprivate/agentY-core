"""Mask secret values in text before it goes back to a model.

The name checks in :mod:`agenty_core.utils.path_guard` stop the plain ways of
reading ``.env``; they cannot stop ``python -c`` spelling the name differently,
a log line that happens to echo a header, or an error message quoting a URL with
a key in it. This is the backstop: every tool result is passed through
:func:`redact` on its way to the model, and whatever looks like a credential
leaves as ``hf_Y…oK``.

Two nets:
- the values actually configured: every value in the project's ``.env`` (and the
  same names in ``os.environ``), matched exactly, and the agentY session token;
- the shapes credentials come in (``sk-…``, ``sk-ant-…``, ``hf_…``, ``ghp_…``,
  ``xox?-…``, ``AIza…``, ``Bearer …``), for keys that are not in ``.env``.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

_MIN_LEN = 8            # shorter values (ports, flags, "true") are not secrets

_PATTERNS = [
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bsk-(?:proj-|or-v1-|live_)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bxapp-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
]
_BEARER = re.compile(r"(\b(?:Bearer|Token)\s+)([A-Za-z0-9._~+/=\-]{16,})", re.I)
# KEY=value / "api_key": "value" where the name says it is a secret
_NAMED = re.compile(
    r"""((?:api[_-]?key|secret|token|password|passwd|authorization)["']?\s*[:=]\s*["']?)"""
    r"""([^\s"',;]{8,})""", re.I)

_lock = threading.Lock()
_cache: dict = {"key": None, "values": []}


def mask(value: str) -> str:
    """``hf_YabcdefoK`` -> ``hf_Y…oK``: enough to tell keys apart, not to use one."""
    v = str(value)
    if len(v) <= 8:
        return "…"
    return f"{v[:4]}…{v[-2:]}"


def _env_file() -> Path | None:
    try:
        from agenty_core.paths import project_root
        return project_root() / ".env"
    except Exception:  # noqa: BLE001
        return None


def _token_file() -> Path | None:
    try:
        from agenty_core.paths import project_root
        return project_root() / ".agenty_token"
    except Exception:  # noqa: BLE001
        return None


def _mtime(p: Path | None) -> float:
    try:
        return p.stat().st_mtime if p else 0.0
    except OSError:
        return 0.0


def known_values() -> list[str]:
    """The configured secret values, longest first. Re-read when .env changes."""
    env, tok = _env_file(), _token_file()
    key = (str(env), _mtime(env), str(tok), _mtime(tok))
    with _lock:
        if _cache["key"] == key:
            return _cache["values"]
        values: set[str] = set()
        names: set[str] = set()
        if env is not None and env.is_file():
            try:
                from dotenv import dotenv_values
                for k, v in dotenv_values(env).items():
                    names.add(k)
                    if v and len(v.strip()) >= _MIN_LEN:
                        values.add(v.strip())
            except Exception:  # noqa: BLE001
                pass
        for k in names:
            v = os.environ.get(k, "")
            if v and len(v.strip()) >= _MIN_LEN:
                values.add(v.strip())
        for k in ("AGENTY_SESSION_TOKEN",):
            v = os.environ.get(k, "")
            if len(v) >= _MIN_LEN:
                values.add(v)
        try:
            if tok is not None and tok.is_file():
                t = tok.read_text(encoding="utf-8").strip()
                if len(t) >= _MIN_LEN:
                    values.add(t)
        except OSError:
            pass
        # Values that are not secrets but are long enough to pass the length
        # test - a URL, a path, a model name - would make every mention of them
        # unreadable. Keep only what looks like a key: no spaces, no slashes.
        out = sorted((v for v in values if " " not in v and "/" not in v and "\\" not in v),
                     key=len, reverse=True)
        _cache["key"], _cache["values"] = key, out
        return out


def redact(text):
    """*text* with every secret value masked. Non-strings come back unchanged."""
    if not isinstance(text, str) or len(text) < _MIN_LEN:
        return text
    out = text
    for v in known_values():
        if v in out:
            out = out.replace(v, mask(v))
    for pat in _PATTERNS:
        out = pat.sub(lambda m: mask(m.group(0)), out)
    out = _BEARER.sub(lambda m: m.group(1) + mask(m.group(2)), out)
    out = _NAMED.sub(lambda m: m.group(1) + (m.group(2) if "…" in m.group(2) else mask(m.group(2))), out)
    return out


def redact_obj(obj):
    """Redact every string inside a JSON-like structure (dicts, lists)."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(v) for v in obj)
    return obj
