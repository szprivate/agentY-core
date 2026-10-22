"""Workflow-template registry tools — register / unregister custom templates.

In-process equivalents of ``scripts/add_workflow.ps1`` and
``scripts/remove_workflow.ps1``. They parse a ComfyUI workflow JSON into the
custom-templates ``index.json`` — the sole catalog surfaced by
``get_workflow_catalog`` (name, models, io, and description) — and, on removal,
delete the matching ``skills/<kebab>`` directory. No PowerShell or subprocess is
involved.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agenty_core._compat import tool
from agenty_core.utils.workflow_parser import (
    _custom_index_path,
    _project_root,
    parse_workflow,
    workflow_remove,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _name_in_index(name: str, index_path: Path) -> bool:
    """Return True if a template called *name* already exists in *index_path*."""
    if not index_path.exists():
        return False
    try:
        raw = index_path.read_text(encoding="utf-8").strip()
        index = json.loads(raw) if raw else []
    except Exception:
        return False
    for group in index or []:
        for tpl in group.get("templates", []):
            if tpl.get("name") == name:
                return True
    return False


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def register_workflow_template(workflow_file: str, index_path: str = "", description: str = "",
                               local: bool = False) -> str:
    """Register a custom ComfyUI workflow JSON as a reusable template.

    In-process equivalent of ``scripts/add_workflow.ps1``, extended so a workflow
    can be registered directly from any location (e.g. a file just pasted/saved
    by the user) without manually placing it first. It:
      1. Loads the workflow and, if it is in ComfyUI **graph/export format**
         (rather than API format), auto-converts it to API format using the same
         converter as ``_load_workflow`` (``_convert_graph_to_api``).
      2. Copies the (converted) workflow JSON into the custom-templates directory
         as ``<name>.json`` so ``get_workflow_template`` can load it later.
      3. Parses it and appends/updates its entry (name, models, io) in the
         custom-templates ``index.json`` — which is also where its catalog
         ``description`` lives, so it appears in ``get_workflow_catalog()`` (the
         flat ``config/workflow_templates.json`` catalog is retired).

    The template name is always the file stem (e.g. ``my_flow`` for
    ``my_flow.json``). Registering a name that already exists is refused, exactly
    like the script.

    Note: graph→API conversion maps widget values to named inputs using the live
    ComfyUI node schema. If ComfyUI is unreachable the conversion still succeeds
    but raw widget values are kept under ``__widgets_values`` — re-register once
    ComfyUI is running for a fully named API workflow.

    ``local=True`` keeps it on THIS MACHINE only: it goes to the git-ignored
    ``local/`` folder of the custom templates (its own ``index.json``), skips the
    shared catalog and never feeds the recipe database. Use it for a job that was
    solved here and should be the fast path next time ("save this as a template
    for me"); the name may differ from the file stem when the file is a throw-away.
    Matching requests are offered the saved job first (retrieval prefers it).

    Args:
        workflow_file: Path to the ComfyUI workflow ``.json`` (API or graph
            format). Relative paths resolve against the repository root.
        index_path: Optional override for the ``index.json`` path. Empty (default)
            uses the configured custom-templates index. The JSON is copied next to
            this index.
        description: Catalog description. With ``local=True`` write one: what the
            job does, which inputs change (files, the prompt) and the key settings;
            when empty one is generated from the graph.
        local: Save as a machine-local template (see above) instead of a shared one.

    Returns:
        JSON summary with the parsed entry, the index path written, the path the
        JSON was copied to, whether it was converted from graph format, and
        whether the catalog was updated — or ``{"error": ...}``.
    """
    if local:
        from agenty_core.tools import local_templates
        return json.dumps(local_templates.register_local_file(workflow_file, "", description),
                          ensure_ascii=False, default=str)
    try:
        # Lazy import to avoid import-time side effects / any cycle.
        from agenty_core.tools.comfyui import _convert_graph_to_api, _is_graph_format

        root = _project_root()
        p = Path(workflow_file)
        if not p.is_absolute():
            p = root / workflow_file
        if not p.exists():
            return json.dumps({"error": f"Workflow file not found: {workflow_file}"})

        name = p.stem
        idx = Path(index_path) if index_path else _custom_index_path()

        # Pre-check: refuse duplicates (mirrors workflow_parser._main / the script).
        if _name_in_index(name, idx):
            return json.dumps({"error": f"Template '{name}' already exists in {idx}"})

        try:
            with open(p, encoding="utf-8") as f:
                workflow = json.load(f)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid workflow JSON: {e}"})

        # Auto-convert graph/export format → API format.
        converted = False
        if _is_graph_format(workflow):
            workflow = _convert_graph_to_api(workflow)
            converted = True

        # Copy the (converted) workflow into the custom-templates directory so
        # get_workflow_template / _fetch_template can load it by name.
        dest_dir = idx.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{name}.json"
        if converted:
            # Persist the converted API-format workflow.
            dest.write_text(
                json.dumps(workflow, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        elif p.resolve() != dest.resolve():
            # Already API format and not already in place — copy verbatim.
            shutil.copy2(p, dest)

        entry = parse_workflow(
            workflow,
            name=name,
            description=description or None,
            update_index=True,
            index_path=index_path or None,
        )

        return json.dumps(
            {
                "registered": name,
                "index_path": str(idx),
                "copied_to": str(dest),
                "converted_from_graph_format": converted,
                "entry": entry,
                "message": (
                    f"Registered '{name}'"
                    + (" (converted graph→API format)" if converted else "")
                    + f". Copied workflow to {dest}. Set a 'description' on the "
                    "template's index.json entry so get_workflow_catalog() describes it."
                ),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def unregister_workflow_template(name: str, index_path: str = "") -> str:
    """Unregister a custom workflow template by name.

    In-process equivalent of ``scripts/remove_workflow.ps1``. It:
      1. Removes every template named *name* from the custom-templates
         ``index.json`` (dropping a group that becomes empty) — which also drops
         it from ``get_workflow_catalog()``.
      2. Deletes the matching skill directory ``skills/<kebab>`` where ``<kebab>``
         is *name* lowercased with underscores turned into hyphens.

    Args:
        name: The template name (file stem) to remove, e.g. ``my_flow``.
        index_path: Optional override for the ``index.json`` path. Empty (default)
            uses the configured custom-templates index.

    Returns:
        JSON summary of what was removed — or ``{"error": ...}``.
    """
    try:
        if not name:
            return json.dumps({"error": "Template name is required"})

        # A machine-local template (register_workflow_template(local=True)) is removed
        # from its own folder; nothing shared is touched.
        from agenty_core.tools import local_templates
        if not index_path and any(e.get("name") == name for e in local_templates._read_entries()):
            local_templates.forget_local(name)
            return json.dumps({"removed": name, "local": True,
                               "message": f"Unregistered local template '{name}'."}, ensure_ascii=False)

        root = _project_root()
        idx_written = workflow_remove(name, index_path=index_path or None)

        # Remove the skill directory (kebab-case derived from the name).
        kebab = name.lower().replace("_", "-")
        skill_dir = root / "skills" / kebab
        skill_removed = False
        if skill_dir.is_dir():
            shutil.rmtree(skill_dir)
            skill_removed = True

        return json.dumps(
            {
                "removed": name,
                "index_path": str(idx_written),
                "skill_dir": str(skill_dir),
                "skill_dir_removed": skill_removed,
                "message": (
                    f"Unregistered '{name}'. Index updated; "
                    f"skill dir {'removed' if skill_removed else 'not found'}."
                ),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})
