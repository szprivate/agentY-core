"""agentY – @tool access to the per-project memory store.

The store itself is described in ``src/utils/project_memory.py``: one fact per
file, in a folder inside ComfyUI's user directory, so it switches with the project
when the pipeline switches ComfyUI's user/input/output directories.

These two tools are the *on-demand* half. The always-on half is the
``[PROJECT MEMORY]`` block the pipeline injects into every turn — the agent sees
what exists without asking, and calls ``project_memory_read`` only for the entries
it actually needs in full.
"""

from __future__ import annotations

from agenty_core._compat import tool
from agenty_core.utils.project_memory import (KNOWN_TYPES, delete_entry,
                                              list_entries, read_entry,
                                              store_dir, write_entry)

# A host can hold writes off for a while (agentY does, while a benchmark or a
# test is running): a callable returning the seconds the pause has left, 0 when
# writes are on. Project memory is injected into every later turn, so a note a
# test run writes — wrong or not — steers the user's real work afterwards.
_write_pause = lambda: 0.0  # noqa: E731


def set_write_pause(fn) -> None:
    """Have *fn* (seconds of pause left, 0 = writing) decide whether writes land."""
    global _write_pause
    _write_pause = fn or (lambda: 0.0)


@tool
def project_memory_read(name: str = "") -> str:
    """Read a fact established for the CURRENT project, in full, by name.

    The turn already carries a ``[PROJECT MEMORY]`` block listing what exists;
    call this when you need the whole entry rather than its first line — the
    character's actual prompt, the full style guide, the reference path.

    Call it **before writing a prompt for anything the project has already
    defined**. A character described once has a stored description; reinventing it
    is how shot 4 stops matching shot 1.

    Args:
        name: The entry name as listed in the block, e.g. "hero" or "grade".
              Capitalisation and spacing don't matter. Leave empty to list
              everything currently on record.

    Returns:
        The entry's full text, a listing when no name is given, or a clear notice
        when the project has no such entry.
    """
    key = str(name or "").strip()
    if not key:
        entries = list_entries()
        if not entries:
            return "(this project has no stored memory yet)"
        return "\n".join(f"- {e.name} ({e.type}) — {e.summary}" for e in entries)

    entry = read_entry(key)
    if entry is None:
        known = ", ".join(e.name for e in list_entries()) or "(nothing stored yet)"
        return f"No project entry named '{key}'. On record: {known}"
    return f"# {entry.name} ({entry.type})\n\n{entry.body}"


@tool
def project_memory_write(name: str, content: str, type: str = "note") -> str:
    """Establish a fact for the CURRENT project, so later turns keep it.

    This is project state, not a lesson: what the hero looks like, how the grade
    is described, which reference image is the locked one, what aspect ratio this
    production delivers in. It lives beside the project and switches with it, so
    write things that are true of *this* project and would be wrong for the next.
    Use ``memory_write`` instead for things true of the user across all projects.

    Write it the way you would want to read it cold, and put the most identifying
    line first — that first line is what every later turn sees in the injected
    block. Writing the same name again REPLACES the entry, which is how you update
    a fact; there is no append.

    Store reference images as a path relative to ComfyUI's input directory when you
    can, so the entry survives the project moving between machines.

    Args:
        name: Short handle to recall it by, e.g. "hero", "grade", "aspect-ratio".
        content: The fact itself. A prompt fragment, a guide, a path — whatever a
                 later turn needs verbatim.
        type: One of "technical", "character", "style", "reference", "note".
              "technical" entries are injected into every turn IN FULL (aspect
              ratios, resolutions, fps), so keep those short; everything else is
              listed by name and read on demand.

    Returns:
        Confirmation of what was stored and where, or a clear notice if it wasn't.
    """
    key = str(name or "").strip()
    body = str(content or "").strip()
    if not key or not body:
        return "Nothing stored — both a name and content are required."
    try:
        paused = float(_write_pause() or 0)
    except Exception:  # noqa: BLE001 — a broken guard must not lose the user's write
        paused = 0.0
    if paused > 0:
        return ("Project memory writes are paused for this session (a benchmark or test run "
                f"is in progress). NOT saved: {key}")
    kind = str(type or "note").strip().lower()
    if kind not in KNOWN_TYPES:
        kind = "note"

    entry = write_entry(key, body, type=kind)
    if entry is None:
        d = store_dir()
        where = f" (store: {d})" if d else " — ComfyUI did not report a user directory"
        return f"Could not write project memory '{key}'{where}. NOT saved."
    return f"Stored in project memory as '{entry.name}' ({entry.type}) → {entry.path}"


@tool
def project_memory_forget(name: str) -> str:
    """Remove a fact from the current project's memory.

    Use it when a fact stops being true — a character redesigned, a locked
    reference replaced. Correcting a fact is a ``project_memory_write`` with the
    same name; this is for facts that should simply no longer be there.

    Args:
        name: The entry name to remove.

    Returns:
        Confirmation, or a notice that there was nothing by that name.
    """
    key = str(name or "").strip()
    if not key:
        return "Nothing removed — no name given."
    try:
        if float(_write_pause() or 0) > 0:
            return f"Project memory writes are paused for this session. NOT removed: {key}"
    except Exception:  # noqa: BLE001
        pass
    return (f"Removed '{key}' from project memory."
            if delete_entry(key) else f"No project entry named '{key}'.")
