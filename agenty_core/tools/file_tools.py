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


# How much of ONE file a single read returns unless the caller asks for more.
# Unbounded, a read of the 2.5 MB recipe database put ~700,000 tokens into a
# specialist's context (2026-10-05): every later call of that agent re-sent it,
# each taking minutes, and a 20-minute turn was 15 minutes of that.
DEFAULT_READ_CHARS = 60_000
MAX_READ_CHARS = 200_000
_FIND_MAX_LINES = 60
_FIND_LINE_CHARS = 300


@tool
def read_text_file(path: list | str, offset: int = 0, max_chars: int = 0,
                   find: str = "") -> str:
    """Read one or more text files from disk and return their contents as plain strings.

    Use this tool to inspect configuration files, JSON templates, markdown
    documents, or any other UTF-8 text file.  Binary files are not supported.

    A read returns at most 60,000 characters of each file. A longer file comes
    back cut, with a note saying how long it is and how to go on. **For a big
    file, do not page through it — use ``find``**: "is X in this file?" is one
    cheap call that way. For a ComfyUI workflow JSON, ``inspect_workflow_file``
    gives the graph without the bulk.

    Args:
        path: Absolute or relative paths to read, e.g.
            ["a/workflow.json", "b/workflow.json"]. Pass every file you need in
            one call. A single path (as a plain string) returns that file's text
            on its own; several return JSON ``{"files": {path: text}}``, with the
            error message in place of the text for any file that cannot be read.
        offset: Character position to start from — to continue a file that came
            back cut (the note gives the next offset).
        max_chars: How much to return per file. 0 = 60,000; the ceiling is 200,000.
        find: Return only the lines containing this text (case-insensitive), each
            with its line number — up to 60 lines. The way to look something up
            in a large file.

    Returns:
        The text (or the matching lines), or an error message if the file
        cannot be opened.
    """
    paths = _as_list(path)
    return _one_or_many(
        paths, lambda one: _read_text_file_one(one, offset, max_chars, find), "files")


def _read_text_file_one(path: str, offset: int = 0, max_chars: int = 0,
                        find: str = "") -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"[read_text_file] File not found: {path}"
    except PermissionError:
        return f"[read_text_file] Permission denied: {path}"
    except Exception as exc:  # noqa: BLE001
        return f"[read_text_file] Error reading {path}: {exc}"
    if str(find or "").strip():
        return _find_lines(text, str(find).strip(), path)
    start = max(0, int(offset or 0))
    limit = int(max_chars or 0)
    limit = DEFAULT_READ_CHARS if limit <= 0 else min(limit, MAX_READ_CHARS)
    piece = text[start:start + limit]
    end = start + len(piece)
    if start == 0 and end >= len(text):
        return text
    left = len(text) - end
    note = (f"\n\n[read_text_file] Showing characters {start:,}-{end:,} of {len(text):,}"
            + (f"; {left:,} more. Continue with offset={end}, or — better for a file "
               f"this size — pass find=\"<text>\" to get only the lines you need."
               if left else " (the end of the file)."))
    return piece + note


def _find_lines(text: str, needle: str, path: str) -> str:
    """The lines of *text* containing *needle*, numbered, each clipped."""
    low = needle.lower()
    hits = []
    total = 0
    for n, line in enumerate(text.splitlines(), 1):
        if low not in line.lower():
            continue
        total += 1
        if len(hits) < _FIND_MAX_LINES:
            shown = line.strip()
            if len(shown) > _FIND_LINE_CHARS:
                # Keep the part of a long line (minified JSON) around the match.
                at = shown.lower().find(low)
                lo = max(0, at - _FIND_LINE_CHARS // 2)
                shown = ("…" if lo else "") + shown[lo:lo + _FIND_LINE_CHARS] + "…"
            hits.append(f"{n}: {shown}")
    if not hits:
        return (f"[read_text_file] No line of {path} contains \"{needle}\" "
                f"({len(text):,} characters searched).")
    head = f"[read_text_file] {total} line(s) of {path} contain \"{needle}\""
    if total > len(hits):
        head += f" — the first {len(hits)} shown; narrow the text to see the rest"
    return head + ":\n" + "\n".join(hits)


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
