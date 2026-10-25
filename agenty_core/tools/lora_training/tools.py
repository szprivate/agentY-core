"""The LoRA training tools (agentY and agentY-mcp both register these)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from agenty_core._compat import tool

from . import backends as B
from . import dataset as D
from . import formats as F
from . import jobs as J
from . import preflight as PF
from .presets import PRESETS, get_preset, public_summary

logger = logging.getLogger("agenty_core.lora.tools")


def _guard(fn):
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except J.LoraError as exc:
            return exc.render()
        except (FileNotFoundError, KeyError, ValueError) as exc:
            return json.dumps({"error": str(exc).strip("'\""), "error_kind": "args"})
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed", fn.__name__)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}", "error_kind": "api"})
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


def _open_in_viewer(manifest_path: str) -> dict:
    try:
        from agenty_core.tools.show_in_viewer import open_in_viewer
        return open_in_viewer([manifest_path], label="LoRA dataset", focus=True)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@_guard
def _prepare(sources, subject, kind, model, name, trigger_word, caption_style, max_items, output_dir):
    m = D.prepare_dataset(sources, subject, kind, model, name=name, trigger_word=trigger_word,
                          caption_style=caption_style, max_items=max_items, output_dir=output_dir)
    ds = F.datasets_root(output_dir) / m["name"]
    removed = [{"id": i["id"], "why": i["note"]} for i in m["items"] if not i["included"]]
    out = {"status": "needs_captions" if m.get("needs_captions") else "ok", "dataset": m["name"], "folder": str(ds),
           "manifest": str(ds / "manifest.json"), "items": m["summary"].get("included"), "removed": removed[:30],
           "trigger_word": m["trigger_word"], "model": m["model"], "buckets": m["buckets"],
           "warnings": m["summary"].get("warnings", []), "contact_sheet": str(ds / "contact_sheet.jpg")}
    if m.get("needs_captions"):
        out["needs_captions"] = m["needs_captions"]
        out["next"] = (f"{len(m['needs_captions'])} items have no caption. Look at them (view_image / analyze_image on "
                       f"<folder>/items/<id>.jpg, or the contact sheet) and write the captions with lora_dataset_set_captions, "
                       f"following the caption style: {m['caption_style']}. Then lora_dataset_review.")
    else:
        out["next"] = "Captions are written. Call lora_dataset_review so the user can check the dataset; do not train before they approve."
    return json.dumps(out)


@tool
def lora_dataset_prepare(sources: list, subject: str, kind: str = "character", model: str = "z_image", name: str = "",
                         trigger_word: str = "", caption_style: str = "", max_items: int = 200,
                         output_dir: str = "") -> str:
    """Build a LoRA training dataset: collect pictures / clips, drop what is unusable, crop and resize them to the
    model's buckets, split long clips, caption everything and write a manifest. Does NOT train.

    Removes duplicates (near-identical shots), blurry and too-small pictures, and anything the captioner flags as
    carrying a watermark or overlay text. Every item gets a crop box (normalised, editable in the viewer), a caption
    and a note on why it was kept or removed. The result is a dataset folder under ``<output>/agent/lora_datasets/``.
    Always follow with ``lora_dataset_review``: training starts only after the USER approves the dataset.

    Args:
        sources: files, folders, globs or image URLs (a list, or one comma-separated string). Canvas outputs: pass their
            file paths. Web search results: only with the user's consent, download them first (download_image).
        subject: what the LoRA teaches, in a few words ("my dog Rex", "ink wash illustration style", "the Aviator sneaker").
        kind: character | style | product | motion | concept. Changes the caption rules and the size warnings.
        model: the base model: z_image, flux2_klein_4b, flux2_klein_9b, flux2_dev, flux1_dev, qwen_image, wan22_t2v_14b,
            wan22_5b, ltx2, sdxl, sd15 (names like "Wan 2.2" or "flux.2 klein" resolve too). Video models also need clips.
        name: dataset name (default: from the subject).
        trigger_word: the word that calls the LoRA; default: made up from the name. Added to every caption by the trainer.
        caption_style: override the model's caption style.
        max_items: stop collecting after this many usable items.
        output_dir: save under this folder instead of ComfyUI's output.

    Returns:
        JSON: status ok | needs_captions (no captioner is set up here: write them with lora_dataset_set_captions),
        dataset, manifest path, contact_sheet, items, removed (with reasons), warnings, next. Or {"error", "error_kind"}.
    """
    return _prepare(sources, subject, kind, model, name, trigger_word, caption_style, max_items, output_dir)


@_guard
def _set_captions(dataset, captions, trigger_word, output_dir):
    ds = F.resolve_dataset(dataset, output_dir)
    r = D.set_captions(ds, captions or {}, trigger_word)
    return json.dumps({"status": "ok" if not r["missing"] else "incomplete", "missing_captions": r["missing"],
                       "unknown_ids": r["unknown_ids"],
                       "next": "Call lora_dataset_review." if not r["missing"] else "Some items still have no caption."})


@tool
def lora_dataset_set_captions(dataset: str, captions: dict, trigger_word: str = "", output_dir: str = "") -> str:
    """Write or replace captions of a LoRA dataset (item id -> caption), and optionally change its trigger word.

    Use it when ``lora_dataset_prepare`` returned ``needs_captions``, or to apply the user's wording changes. An edit
    after the user approved the dataset takes the approval back: it needs a new ``lora_dataset_review``.

    Args:
        dataset: the dataset name (or folder / manifest path).
        captions: {"i001": "a woman standing in a kitchen, daylight", ...}. Do not repeat the trigger word.
        trigger_word: set a new trigger word.
        output_dir: where the datasets are, if not ComfyUI's output.

    Returns:
        JSON with missing_captions (ids still empty) and next.
    """
    return _set_captions(dataset, captions, trigger_word, output_dir)


@_guard
def _review(dataset, action, note, user_message, open_viewer, output_dir):
    ds = F.resolve_dataset(dataset, output_dir)
    action = (action or "open").lower()
    if action == "approve":
        if not str(user_message or "").strip():
            raise J.LoraError("approve needs user_message: the user's own words approving the dataset (e.g. 'approve'). "
                              "Never approve on the user's behalf.", "args")
        r = F.approve(ds, by="chat", note=f"user said: {user_message}")
        return json.dumps({"status": "approved", "dataset": ds.name, "approved_by": "chat",
                           "next": "Call train_lora now."})
    if action in ("changes", "cancel"):
        state = "changes" if action == "changes" else "cancelled"
        F.write_json(ds / "review.json", {**F.load_review(ds), "state": state, "note": note, "updated": __import__("time").time()})
        return json.dumps({"status": state, "dataset": ds.name})
    if action == "status":
        r = F.load_review(ds)
        ok, why = F.is_approved(ds)
        return json.dumps({"status": r.get("state"), "approved": ok, "why_not": why, "note": r.get("note", ""),
                           "dataset": ds.name})
    info = D.request_review(ds, note)
    viewer = _open_in_viewer(info["manifest"]) if open_viewer else {"skipped": True}
    shown = bool(viewer) and not viewer.get("error") and not viewer.get("skipped") and not viewer.get("refused")         and (bool(viewer.get("lora")) or viewer.get("ok") is not False)
    msg = ("The dataset is open in the viewer's Dataset panel: the user can edit captions, drop items and press "
           "'Approve & train'. END YOUR TURN NOW with a short message saying that, and wait; their approval (or note) "
           "arrives as the next message. Do not call train_lora before it." if shown else
           "The viewer could not be opened. Show the user the contact sheet (send_to_slack / show_in_viewer / the chat), "
           "list the trigger word and warnings, and ask them to reply 'approve' or what to change. END YOUR TURN and wait. "
           "When they approve, call lora_dataset_review(action='approve', user_message=<their words>).")
    return json.dumps({"status": "waiting_for_user", "dataset": ds.name, "contact_sheet": info["contact_sheet"],
                       "manifest": info["manifest"], "trigger_word": info["trigger_word"], "items": info["items"],
                       "warnings": info["summary"].get("warnings", []), "viewer": viewer, "next": msg})


@tool
def lora_dataset_review(dataset: str, action: str = "open", note: str = "", user_message: str = "",
                        open_viewer: bool = True, output_dir: str = "") -> str:
    """The user's gate before any training: show them the dataset and wait for their approval.

    ``action="open"`` (default) builds a contact sheet with all captions, opens the viewer's Dataset panel (grid,
    inline caption edit, include/exclude, crop, "Approve & train" / "Ask the agent") and returns
    ``waiting_for_user``: end your turn then. The user's approval arrives as a message. If the viewer is not available
    (Slack, Claude Desktop), send them the contact sheet and wait for "approve" in the chat. ``train_lora`` refuses an
    unapproved dataset, and one edited after approval.

    Args:
        dataset: the dataset name (or folder / manifest path).
        action: open | status (is it approved?) | approve (the user said so in chat: pass their words in user_message) |
            changes (record that they asked for changes) | cancel.
        note: for open: a line for the user; for changes: what they asked for.
        user_message: for approve: the user's own words. Never approve on their behalf.
        open_viewer: open the Dataset panel (open only).
        output_dir: where the datasets are, if not ComfyUI's output.

    Returns:
        JSON: status waiting_for_user | approved | changes | cancelled, contact_sheet, warnings, next.
    """
    return _review(dataset, action, note, user_message, open_viewer, output_dir)


@_guard
def _train(dataset, model, backend, preset, steps, rank, learning_rate, name, free_vram, allow_download, confirm_cost,
           output_dir, dry_run, extra):
    ov = dict(extra or {})
    ov.update(steps=steps, rank=rank, learning_rate=learning_rate)
    out = J.start(dataset, model=model, backend=backend, preset_overrides=ov, name=name, free_vram=free_vram,
                  allow_download=allow_download, confirm_cost=confirm_cost, output_dir=output_dir, dry_run=dry_run)
    if out.get("status") == "started":
        out["next"] = ("Training runs in the background (a detached process: it survives a restart). Tell the user it "
                       "started and where to watch it (the viewer's Dataset panel, Training tab); you get a notification at "
                       "each sample and at the end. Check with lora_training_status only when asked.")
    return json.dumps(out)


@tool
def train_lora(dataset: str, model: str = "", backend: str = "auto", preset: str = "", steps: int = 0, rank: int = 0,
               learning_rate: float = 0.0, name: str = "", free_vram: bool = True, allow_download: bool = False,
               confirm_cost: bool = False, dry_run: bool = False, output_dir: str = "", options: Optional[dict] = None) -> str:
    """Start training a LoRA on an APPROVED dataset, in the background. Returns a job id at once.

    Refuses until the user approved the dataset (``lora_dataset_review``). Runs the pre-flight first: free GPU memory
    (it frees ComfyUI's), disk space (never C:), the base model, and the backend, which is installed into its own
    venv on first use. Local backends: ai-toolkit (default) and musubi-tuner; ``backend="fal"`` trains hosted and
    COSTS MONEY: the price is estimated first, and above the confirmation level you get ``confirm_cost`` back - tell the
    user the price, and only after a yes call again with ``confirm_cost=True``. If the base model must be downloaded
    (``allow_download``), say the size to the user first. Use ``dry_run=True`` to see the pre-flight and the plan without starting.

    Args:
        dataset: the approved dataset (name, folder or manifest).
        model: base model (default: the dataset's). See lora_dataset_prepare for the names.
        backend: auto | ai-toolkit | musubi | fal.
        preset: reserved (each model has one preset; override single values with steps / rank / learning_rate / options).
        steps: training steps (0 = the preset's).
        rank: LoRA rank (0 = the preset's).
        learning_rate: 0 = the preset's.
        name: job name (default: the dataset's; a version suffix is added).
        free_vram: ask ComfyUI to unload its models first.
        allow_download: the user agreed to download the base model (ai-toolkit fetches it from Hugging Face).
        confirm_cost: the user agreed to the hosted price.
        dry_run: run the checks and return the plan, start nothing.
        output_dir: save runs under this folder instead of ComfyUI's output.
        options: more overrides: batch_size, optimizer, seed, save_every, sample_every, frames, fps, resolutions, low_vram,
            sample_prompts (list).

    Returns:
        JSON: status started (job, run_dir, warnings) | dry_run | or {"error", "error_kind": not_approved | preflight |
        confirm_cost | args}.
    """
    if str(backend or "auto").lower() != "fal":
        from agenty_core.utils import execution_mode
        lim = execution_mode.local_only_refusal("train_lora_local")
        if lim:                       # cloud mode: the local trainers need this machine's GPU (fal does not)
            return json.dumps(lim)
    return _train(dataset, model, backend, preset, steps, rank, learning_rate, name, free_vram, allow_download,
                  confirm_cost, output_dir, dry_run, options)


@_guard
def _status(job, wait_s, log_lines, output_dir):
    if not str(job or "").strip():
        loc = {n: B.get(n).locate() for n in ("ai-toolkit", "musubi")}
        return json.dumps({"runs": J.runs(output_dir)[:20],
                           "backends": {n: {"installed": v["installed"], "root": v["root"]} for n, v in loc.items()},
                           "gpu": PF.gpu_memory(), "presets": [public_summary({**p, "key": k}) for k, p in PRESETS.items()]})
    return json.dumps(J.status(job, wait_s, output_dir, log_lines))


@tool
def lora_training_status(job: str = "", wait_s: int = 0, log_lines: int = 0, output_dir: str = "") -> str:
    """Progress of a LoRA training: step / total, ETA, it/s, loss, learning rate, VRAM, the latest samples, new events.
    When it is done: the registered LoRA, a comparison sheet (first / middle / last checkpoint samples) and a test plan
    (prompts, seed, strengths) for rendering with and without the LoRA.

    Call it without ``job`` to list recent runs, the installed backends, the GPU and every model preset
    (steps, rank, VRAM need, caption style).

    Args:
        job: the job id train_lora returned. Empty: overview.
        wait_s: wait up to this long for it to finish.
        log_lines: include the last N lines of the backend log.
        output_dir: where the runs are, if not ComfyUI's output.
    """
    return _status(job, wait_s, log_lines, output_dir)


@_guard
def _stop(job, keep, output_dir):
    return json.dumps(J.stop(job, keep, output_dir))


@tool
def lora_training_stop(job: str, keep_last_checkpoint: bool = True, output_dir: str = "") -> str:
    """Stop a running LoRA training. With ``keep_last_checkpoint`` (default) the newest saved checkpoint is copied into
    ComfyUI's loras folder and registered as a partial result; otherwise nothing is kept. Only stops the job's own
    process tree.

    Args:
        job: the job id.
        keep_last_checkpoint: keep the newest checkpoint as the result.
        output_dir: where the runs are, if not ComfyUI's output.
    """
    return _stop(job, keep_last_checkpoint, output_dir)
