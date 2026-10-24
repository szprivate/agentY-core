"""On-disk formats of LoRA training: the contract between the tools, the runner and the bEpic viewer.

Everything lives under ComfyUI's output folder, in ``agent/``:

    agent/lora_datasets/<name>/
        manifest.json         the dataset (below). The viewer's Dataset panel edits this file.
        review.json           the review gate (below)
        items/                the prepared files (images / clips), already cropped and resized
        trainer/              what the backend reads (written at train time from the manifest)
        contact_sheet.jpg     all included items with their captions (for Slack / MCP review)
    agent/lora_runs/<job>/
        job.json              the job as started (dataset, model, backend, preset, steps, command, pid ...)
        progress.json         live state, rewritten by the runner every few seconds (below)
        events.jsonl          one JSON per line: {"id": n, "t": epoch, "kind": started|sample|checkpoint|finished|
                              failed|stopped, "text": "..."}. notify_bus / toasts read this.
        samples.json          [{"step": n, "files": [{"prompt": "...", "file": "samples/....png"}]}]
        samples/              sample images / clips the backend rendered during training
        control.json          written by anyone (viewer, tools): {"action": "stop", "keep": true|false}
        train.log             the backend's output
        result.json           after the run: {"lora": path, "checkpoints": [...], "registered": {...}}

manifest.json::

    {"version": 1, "name": str, "subject": str, "kind": "character|style|product|motion|concept",
     "model": str, "trigger_word": str, "caption_style": str,
     "status": "draft|review|approved|training|trained|cancelled",
     "approved_at": epoch|null, "created": iso, "buckets": [[w, h], ...],
     "summary": {"count": n, "included": n, "removed": n, "warnings": [str, ...]},
     "items": [{"id": "i001", "file": "items/i001.png" (relative to the dataset folder), "kind": "image|video",
                "caption": str, "included": bool,
                "crop": [x, y, w, h] | null     (normalised 0..1 on the ORIGINAL file; null = whole picture),
                "original": "path of the source file", "size": [w, h], "frames": n|null, "fps": n|null,
                "source": "file|folder|canvas|web", "note": "why kept / why removed",
                "removed_by": "agent|user|null"}]}

A removed item is ``included: false`` and keeps its file, so it can be restored. ``trigger_word`` is part of
every caption the trainer sees (the writer prepends it when absent).

review.json::

    {"state": "pending|approved|changes|cancelled", "requested": epoch, "updated": epoch, "note": str,
     "manifest_sha": str (of the manifest at approval), "by": "viewer|chat|slack"}

``approved`` is the only state ``train_lora`` accepts, and only while manifest.json still hashes to
``manifest_sha`` (an edit after approval needs a new approval).

progress.json::

    {"job": str, "status": "starting|running|done|failed|stopped", "step": n, "total": n, "pct": 0..1,
     "eta_s": n|null, "it_s": float|null, "elapsed_s": n, "loss": float|null, "lr": float|null, "vram_gb": float|null,
     "loss_series": [[step, loss], ...] (at most 400 points), "updated": epoch, "message": str,
     "backend": str, "model": str, "dataset": str, "last_checkpoint": path|null}
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip()).strip("_")[:60]


def agent_root(output_dir: str = "") -> Path:
    """``<ComfyUI output>/agent`` (or ``<output_dir>/agent``)."""
    if str(output_dir or "").strip():
        base = Path(os.path.expanduser(str(output_dir).strip().strip('"')))
    else:
        base = None
        try:
            from agenty_core.utils.comfyui_client import comfyui_output_dir
            b = comfyui_output_dir()
            base = Path(b) if b else None
        except Exception:  # noqa: BLE001
            base = None
        if base is None:
            from agenty_core.paths import project_root
            base = project_root() / "output"
    return base / "agent"


def datasets_root(output_dir: str = "") -> Path:
    d = agent_root(output_dir) / "lora_datasets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_root(output_dir: str = "") -> Path:
    d = agent_root(output_dir) / "lora_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_dataset(ref: str, output_dir: str = "") -> Path:
    """A dataset folder from its name or a path (to the folder or its manifest.json)."""
    p = Path(os.path.expanduser(str(ref or "").strip().strip('"')))
    if p.name == "manifest.json":
        p = p.parent
    if p.is_dir() and (p / "manifest.json").exists():
        return p
    q = datasets_root(output_dir) / slug(str(ref))
    if (q / "manifest.json").exists():
        return q
    raise FileNotFoundError(f"no LoRA dataset '{ref}' (looked in {q})")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data: Any) -> None:
    """Atomic: a reader (the viewer polls these files) never sees half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def manifest_sha(manifest: dict) -> str:
    """Hash of what training would see: trigger word, model and every included item's file, caption and crop."""
    core = {"trigger_word": manifest.get("trigger_word", ""), "model": manifest.get("model", ""),
            "items": [[i.get("file"), i.get("caption", ""), i.get("crop")]
                      for i in manifest.get("items", []) if i.get("included", True)]}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()


def load_manifest(ds: Path) -> dict:
    m = read_json(Path(ds) / "manifest.json")
    if not isinstance(m, dict) or "items" not in m:
        raise ValueError(f"{ds}/manifest.json is missing or not a dataset manifest")
    return m


def save_manifest(ds: Path, manifest: dict) -> None:
    items = manifest.get("items", [])
    inc = [i for i in items if i.get("included", True)]
    manifest.setdefault("summary", {}).update(count=len(items), included=len(inc), removed=len(items) - len(inc))
    write_json(Path(ds) / "manifest.json", manifest)


def load_review(ds: Path) -> dict:
    return read_json(Path(ds) / "review.json", {"state": "none"}) or {"state": "none"}


def is_approved(ds: Path) -> tuple[bool, str]:
    """(ok, why not). Approved only while the manifest still matches what was approved."""
    r = load_review(ds)
    if r.get("state") != "approved":
        return False, f"the dataset is not approved (review state: {r.get('state', 'none')})"
    try:
        sha = manifest_sha(load_manifest(ds))
    except ValueError as exc:
        return False, str(exc)
    if r.get("manifest_sha") and r["manifest_sha"] != sha:
        return False, "the dataset was edited after it was approved; it needs a new approval"
    return True, ""


def approve(ds: Path, by: str = "chat", note: str = "") -> dict:
    m = load_manifest(ds)
    r = {"state": "approved", "requested": load_review(ds).get("requested", time.time()), "updated": time.time(),
         "note": note, "manifest_sha": manifest_sha(m), "by": by}
    write_json(Path(ds) / "review.json", r)
    m["status"], m["approved_at"] = "approved", r["updated"]
    save_manifest(ds, m)
    return r


def event(run: Path, kind: str, text: str) -> None:
    """Append to events.jsonl (ids count up; readers remember the last id they saw)."""
    f = Path(run) / "events.jsonl"
    n = 0
    try:
        n = sum(1 for _ in f.open("rb"))
    except OSError:
        pass
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": n + 1, "t": time.time(), "kind": kind, "text": text}) + "\n")


def read_events(run: Path, after: int = 0) -> list[dict]:
    out = []
    try:
        for line in (Path(run) / "events.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("id", 0) > after:
                out.append(e)
    except OSError:
        pass
    return out
