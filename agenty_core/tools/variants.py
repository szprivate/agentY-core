"""``run_variants`` - N independent variants of one workflow, built and queued together.

The slow way, seen in a real turn: ``duplicate_workflow`` four times, ``update_workflow``
four times, run four times - sixteen model round trips (about 5 s each, replaying the
whole conversation every time) for four independent jobs. Here it is one call:

1. the base workflow is duplicated N times (``duplicate_workflow(count=N)``: fresh
   seeds in each copy);
2. each copy gets its own overrides through ``update_workflow`` (so every copy is
   validated exactly like a hand-patched one);
3. with ``run=True`` all copies are queued TOGETHER (ComfyUI works the queue in
   order; nothing waits for the previous variant's answer) and the outputs come back
   per variant.

It reuses the existing pieces and adds no execution machinery of its own: hosts that
have their own executor (agentY, agentY-mcp) register it with :func:`set_batch_hook`
and get their QA, healing, output tagging and viewer push for free.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from agenty_core._compat import tool

logger = logging.getLogger("agentY.variants")

MAX_VARIANTS = 24

# Host hook: ``fn([workflow_path, ...]) -> {"outputs": {path: [files]}, ...}``; None = headless default.
_batch_hook: Optional[Callable[[list], object]] = None


def set_batch_hook(fn: Optional[Callable[[list], object]]) -> None:
    global _batch_hook
    _batch_hook = fn


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            got = json.loads(value)
            return got if isinstance(got, list) else [got]
        except ValueError:
            return []
    return []


def variant_patches(variant: dict) -> tuple[list, list, list]:
    """``(patches, add_nodes, remove_nodes)`` of one variant spec.

    A variant may say ``set`` - ``{"6.text": "a warm lamp", "12.image": "b.png"}`` (node id,
    a dot, the input name; the node id comes first, so ``"3.model.resolution"`` is node 3's
    ``model.resolution``) - and/or the full ``patches`` / ``add_nodes`` / ``remove_nodes`` of
    ``update_workflow``."""
    patches = [p for p in _as_list(variant.get("patches")) if isinstance(p, dict)]
    for key, value in (variant.get("set") or {}).items() if isinstance(variant.get("set"), dict) else []:
        node_id, _, input_name = str(key).partition(".")
        if node_id and input_name:
            patches.append({"node_id": node_id, "input_name": input_name, "value": value})
    return patches, _as_list(variant.get("add_nodes")), _as_list(variant.get("remove_nodes"))


def _problems_of(res: dict) -> list:
    """The reasons an update_workflow answer was not ok, as short strings."""
    found = [str(x) for x in (res.get("problems") or [])]
    found += [str(x) for x in (res.get("node_errors") or [])]
    found += [str(x) for x in (res.get("local_errors") or [])]
    srv = res.get("server_errors")
    if srv:
        found.append(str(srv)[:300])
    if res.get("error"):
        found.append(str(res["error"]))
    return (found or ["the workflow did not validate"])[:6]


def prepare_variants(workflow_path: str, variants: list) -> dict:
    """Duplicate *workflow_path* once per variant and patch each copy.

    ``{"status": "ok"|"partial"|"error", "variants": [{index, label, workflow_path, status,
    problems}], "ready": [paths of the variants that are valid]}``. A variant that fails
    validation is reported and left out of ``ready``; the others go on."""
    from agenty_core.tools import comfyui as C

    specs = [v if isinstance(v, dict) else {"set": {}} for v in _as_list(variants)]
    if not specs:
        return {"status": "error", "error": "no variants given: pass a list of "
                'variant objects, e.g. [{"label": "warm", "set": {"6.text": "..."}}]'}
    if len(specs) > MAX_VARIANTS:
        return {"status": "error", "error": f"at most {MAX_VARIANTS} variants per call (got {len(specs)}); "
                                            "use start_batch_job for larger sets"}
    dup = json.loads(C.duplicate_workflow(workflow_path, count=len(specs)))
    paths = dup.get("new_paths") or ([dup["new_path"]] if dup.get("new_path") else [])
    if len(paths) != len(specs):
        return {"status": "error", "error": dup.get("error") or "could not duplicate the workflow",
                "detail": dup}

    out, ready = [], []
    token = C.local_only().set(True)      # the run is the server-side check; do not submit every copy twice
    try:
        for i, (spec, path) in enumerate(zip(specs, paths)):
            patches, add_nodes, remove_nodes = variant_patches(spec)
            entry = {"index": i + 1, "label": str(spec.get("label") or f"variant {i + 1}"),
                     "workflow_path": path, "status": "ok", "problems": []}
            if patches or add_nodes or remove_nodes:
                try:
                    res = json.loads(C.update_workflow(path, json.dumps(patches), json.dumps(add_nodes),
                                                       json.dumps(remove_nodes)))
                except Exception as exc:  # noqa: BLE001
                    res = {"status": "error", "problems": [f"{type(exc).__name__}: {exc}"]}
                if res.get("status") != "ok":
                    entry["status"] = "error"
                    entry["problems"] = _problems_of(res)
            if entry["status"] == "ok":
                ready.append(path)
            out.append(entry)
    finally:
        C.local_only().reset(token)
    status = "ok" if len(ready) == len(out) else ("partial" if ready else "error")
    return {"status": status, "variants": out, "ready": ready}


def _default_batch(paths: list) -> dict:
    """No host hook: submit everything, then wait for each (ComfyUI works the queue in order)."""
    from agenty_core.tools.partner_run import _default_run
    workers = max(1, min(len(paths), 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_default_run, paths))
    return {"outputs": {p: (r.get("outputs") or []) for p, r in zip(paths, results)},
            "errors": {p: r.get("error") for p, r in zip(paths, results) if r.get("status") == "error"}}


@tool
def run_variants(workflow_path: str, variants: list, run: bool = True) -> str:
    """Make and run N independent variants of ONE workflow in a single call.

    For "give me 4 versions", "try three lighting setups", "same shot with each of
    these prompts / seeds / images": instead of duplicate -> update -> run once per
    variant (many steps, each replaying the whole conversation), this duplicates the
    workflow N times, patches each copy with its own overrides, validates them and -
    with ``run=True`` - queues them all together. Outputs come back per variant.

    ``variants`` is a list; each variant is an object:
    ``{"label": "warm lamp", "set": {"6.text": "a warm lamp", "12.image": "b.png"}}`` -
    ``set`` keys are ``<node id>.<input name>``; or the full update_workflow shape:
    ``{"patches": [{"node_id": "6", "input_name": "text", "value": "..."}],
    "add_nodes": [...], "remove_nodes": [...]}``. An empty variant ``{}`` is just a
    fresh-seeded copy ("4 takes of the same thing"). Up to 24 variants.

    Use it only when the variants are independent of each other. A variant that fails
    validation is reported with its problems and skipped; the rest still run.

    Args:
        workflow_path: Path of the validated base workflow.
        variants: List of variant objects (see above).
        run: True (default) queues and runs them all; False only builds and validates.
    """
    prepared = prepare_variants(workflow_path, variants)
    if prepared.get("status") == "error" or not run or not prepared.get("ready"):
        return json.dumps({**prepared, "ran": False})
    hook = _batch_hook or _default_batch
    try:
        ran = hook(list(prepared["ready"]))
    except Exception as exc:  # noqa: BLE001
        ran = {"errors": {"*": f"{type(exc).__name__}: {exc}"}}
    if isinstance(ran, str):
        try:
            ran = json.loads(ran)
        except ValueError:
            ran = {"result": ran}
    outs = (ran or {}).get("outputs") if isinstance(ran, dict) else None
    if isinstance(outs, dict):
        for v in prepared["variants"]:
            if v["workflow_path"] in outs:
                v["outputs"] = outs[v["workflow_path"]]
    return json.dumps({**prepared, "ran": True, "run": {k: v for k, v in (ran or {}).items() if k != "outputs"}
                       if isinstance(ran, dict) else ran}, default=str)
