"""
Lightweight file I/O tools that always return plain text.

These avoid the Bedrock-style ``document`` content blocks that some file-read
helpers emit, returning simple strings instead so the result passes cleanly
back to the host model over MCP.
"""

import json
from pathlib import Path

from agenty_core._compat import tool
from agenty_core.paths import project_root
from agenty_core.tools._batch import as_list as _as_list, one_or_many as _one_or_many


@tool
def read_text_file(path: list | str) -> str:
    """Read one or more text files from disk and return their contents as plain strings.

    Use this tool to inspect configuration files, JSON templates, markdown
    documents, or any other UTF-8 text file.  Binary files are not supported.

    Args:
        path: Absolute or relative paths to read, e.g.
            ["a/workflow.json", "b/workflow.json"]. Pass every file you need in
            one call. A single path (as a plain string) returns that file's text
            on its own; several return JSON ``{"files": {path: text}}``, with the
            error message in place of the text for any file that cannot be read.

    Returns:
        The full text contents of the file, or an error message if the file
        cannot be opened.
    """
    paths = _as_list(path)
    return _one_or_many(paths, _read_text_file_one, "files")


def _read_text_file_one(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"[read_text_file] File not found: {path}"
    except PermissionError:
        return f"[read_text_file] Permission denied: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"[read_text_file] Error reading {path}: {exc}"


@tool
def write_text_file(path: str, content: str) -> str:
    """Write *content* to a file on disk (UTF-8).  Parent directories are
    created automatically.  Any existing file at *path* is overwritten.

    Use this tool to persist JSON, markdown, or any other plain-text data.
    Prefer this over ``run_script`` for simple file-write operations — it
    always uses the correct workspace root regardless of process CWD.

    Args:
        path: Absolute path, OR a path relative to the agentY workspace root
              (e.g. ``output_workflows/multiprompt.json``).
        content: The text to write.  Must already be a string; pass
                 ``json.dumps(data, indent=2)`` for JSON payloads.

    Returns:
        A JSON string ``{"ok": true, "path": "<absolute_path>", "bytes": N}``
        on success, or ``{"ok": false, "error": "<message>"}`` on failure.
    """
    try:
        # Resolve relative paths against the consuming app's root.
        _WORKSPACE_ROOT = project_root()

        p = Path(path)
        if not p.is_absolute():
            p = _WORKSPACE_ROOT / p

        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True, "path": str(p), "bytes": len(content.encode("utf-8"))})
    except PermissionError as exc:
        return json.dumps({"ok": False, "error": f"Permission denied: {exc}"})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc)})
