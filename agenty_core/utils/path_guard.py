"""Files the LLM-facing tools must not read, send or overwrite.

The file tools take whatever path the model names, and the model can be talked
into naming anything: a web page, an image description or a workflow file is
enough to ask it to read ``.env`` and pass the text to ``upload_file_to_url`` or
``send_to_slack``. Containing every tool to a few roots would break the ordinary
"read this file I pointed you at"; refusing the handful of files that hold
credentials, and the handful whose contents decide what this machine RUNS, does
not.

Refused for reading and sending (:func:`refuse_read`):
  ``.env`` (and ``.env.*`` except ``.example``), ``.agenty_token``,
  ``settings.local.json``, anything under a hidden credential folder (``.ssh``,
  ``.mcp_tokens``, ``.aws``, ``.gnupg``, ``.azure``, ``.kube``, ``.docker``), ``.git-credentials``,
  ``.netrc``, ``.pypirc``, private keys (``id_rsa`` & co, ``*.pem``, ``*.key``,
  ``*.pfx``, ``*.p12``).

Refused for writing (:func:`refuse_write`), on top of the above:
  ``mcp.json`` (its ``command`` entries are started as programs),
  ``settings.default.toml``, ``requirements*.txt`` / ``requirements.lock``
  (installed by the launcher at every start), ``.agenty_host.json`` (what
  ComfyUI's start button runs), the launchers
  ``run_agent.ps1`` / ``run_agent.sh``, git hooks, shell start-up files and the
  Windows Startup folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SECRET_NAMES = frozenset({
    ".env", ".agenty_token", "settings.local.json", ".git-credentials", ".netrc",
    ".pypirc", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials",
})
_SECRET_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker",
                          ".mcp_tokens"})   # agentY: MCP OAuth tokens (config/.mcp_tokens)
_SECRET_SUFFIXES = (".pem", ".key", ".pfx", ".p12")

_RUN_NAMES = frozenset({
    "mcp.json", "settings.default.toml", "requirements.lock", ".agenty_host.json",
    "run_agent.ps1", "run_agent.sh", ".bashrc", ".bash_profile", ".zshrc",
    ".profile", "profile.ps1", "microsoft.powershell_profile.ps1",
})
_REQUIREMENTS = re.compile(r"^requirements[\w.-]*\.txt$", re.I)


def _parts(path) -> tuple[str, list[str]]:
    """(lower-case file name, lower-case path parts) of *path*, links resolved."""
    raw = str(path or "").strip().strip('"')
    p = Path(os.path.expandvars(os.path.expanduser(raw)))
    try:
        p = p.resolve()
    except (OSError, RuntimeError, ValueError):
        try:
            p = Path(os.path.abspath(str(p)))
        except (OSError, ValueError):
            return "", []
    parts = [s.lower() for s in p.parts]
    return (parts[-1] if parts else ""), parts


def is_secret(path) -> bool:
    """Does *path* name a file that holds a credential?"""
    name, parts = _parts(path)
    if not name:
        return False
    if name in _SECRET_NAMES or name.endswith(_SECRET_SUFFIXES):
        return True
    if name.startswith(".env.") and not name.endswith(".example"):
        return True
    if name.startswith(("id_rsa", "id_ed25519", "id_ecdsa")):
        return True
    return any(part in _SECRET_DIRS for part in parts[:-1])


def decides_what_runs(path) -> bool:
    """Does *path* name a file whose contents are later executed or installed?"""
    name, parts = _parts(path)
    if name in _RUN_NAMES or _REQUIREMENTS.match(name):
        return True
    joined = "/".join(parts)
    if ".git/hooks/" in joined:
        return True
    return "start menu/programs/startup/" in joined


def refuse_read(path, tool: str = "") -> str:
    """Why *tool* may not read or send *path*, or "" when it may."""
    if is_secret(path):
        who = f"[{tool}] " if tool else ""
        return (f"{who}Refused: {path} holds credentials (API keys, tokens or keys). "
                "Tools never read or send it; ask the user to look at it themselves.")
    return ""


def refuse_write(path, tool: str = "") -> str:
    """Why *tool* may not write *path*, or "" when it may."""
    why = refuse_read(path, tool)
    if why:
        return why
    if decides_what_runs(path):
        who = f"[{tool}] " if tool else ""
        return (f"{who}Refused: {path} decides what this machine runs or installs "
                "(MCP servers, launchers, requirements, start-up files). Tools do not "
                "write it; propose the change to the user instead.")
    return ""
