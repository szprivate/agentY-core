"""Getting image files in and out — the parts that do not depend on the host.

Three things every host needs and none of them needs its own version of:

* **staging a local file** into ComfyUI's input dir, so a ``LoadImage`` can read
  it (with the resolution and idempotency rules that make a bare canvas filename
  work);
* **fetching a web image** straight into that same input dir;
* **PUT-ing a local file to a presigned URL**, which is the upload half of every
  MCP creation flow.

What is deliberately NOT here is anything about showing an image to a model.
``analyze_image`` and ``view_image`` diverge for a real reason — the Strands app
delegates to a Vision Agent to keep pixels out of the orchestrator's context,
while an MCP host looks at the image itself — so they stay in each repo. The
byte-wrangling underneath them is shared, in
:mod:`agenty_core.utils.image_bytes`.
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests
from PIL import Image

from agenty_core._compat import tool
from agenty_core.paths import project_root
from agenty_core.utils.comfyui_client import get_client
from agenty_core.utils.image_bytes import detect_format, downsize as _downsize

# Extension → Content-Type for the PUT header when the caller doesn't pass one.
# Set by the host so a downloaded image reaches wherever that host shows things
# — in agentY, the chat panel and the ComfyUI canvas. Injected rather than
# imported: this layer is shared with the MCP server, which has no canvas to drop
# a node onto and leaves it unset. Mirrors ``src/tools/annotate.py``.
_output_sink: Optional[Callable[[str], None]] = None


def set_output_sink(fn: Optional[Callable[[str], None]]) -> None:
    """Register the callable that publishes a downloaded file to the host."""
    global _output_sink
    _output_sink = fn


def set_turn_output_sink(fn: Optional[Callable[[str], None]]) -> None:
    """Publish this turn's downloads through *fn*. A host running several
    conversations has one sink per conversation, and the module-wide one would
    hand every download to whichever was registered last."""
    from agenty_core.utils import turn_scope
    turn_scope.current().set("image_io.output_sink", fn)


def _publish(path: str) -> None:
    """Hand *path* to the host, if one is listening. Never raises."""
    from agenty_core.utils import turn_scope
    sink = turn_scope.current().get("image_io.output_sink") or _output_sink
    if not path or sink is None:
        return
    try:
        sink(path)
    except Exception as exc:  # noqa: BLE001 — delivery must never fail a download
        print(f"[download_image] could not publish {path}: {exc}")


_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff", ".mp4": "video/mp4",
    ".webm": "video/webm", ".mov": "video/quicktime", ".pdf": "application/pdf",
}


def comfy_input_dir() -> Optional[str]:
    """Best-effort lookup of ComfyUI's configured input directory, or ``None``."""
    try:
        from agenty_core.tools.comfyui import get_comfyui_dirs
        info = json.loads(get_comfyui_dirs()) or {}
        v = info.get("input_dir")
        if v and v != "unknown":
            return v
    except Exception:  # noqa: BLE001 — best-effort; the callers all cope with None
        pass
    return None


def resolve_local_image(file_path: str) -> Optional[str]:
    """Resolve a possibly-bare image path to an existing file on disk.

    Tries, in order: the path as given (absolute or relative to CWD, expanding
    ``~``), then CWD-joined, then ``<ComfyUI input dir>/<basename>``. That last
    step is what makes a canvas-selected image work: a ``LoadImage`` widget stores
    its image by *bare filename*, and that file lives in ComfyUI's input dir, not
    the agent's CWD. Returns the resolved path, or ``None`` if it is nowhere.
    """
    if not file_path:
        return None
    p = Path(file_path).expanduser()
    if p.is_file():
        return str(p)
    cwd_p = Path(os.getcwd()) / file_path
    if cwd_p.is_file():
        return str(cwd_p)
    in_dir = comfy_input_dir()
    if in_dir:
        staged = Path(in_dir) / os.path.basename(file_path)
        if staged.is_file():
            return str(staged)
    return None


def stage_image(file_path: str, subfolder: str = "", image_type: str = "input",
                overwrite: bool = False) -> dict:
    """Upload one local image into ComfyUI's input dir. Returns a plain dict.

    The ComfyUI ``{name, subfolder, type}`` response, an idempotency skip note, or
    ``{"error": ...}``. Uploading through the API rather than copying the file
    keeps this working when ComfyUI is on another machine.
    """
    resolved = resolve_local_image(file_path)
    if resolved is None:
        return {"error": f"File not found: {file_path}"}

    filename = os.path.basename(resolved)

    # Idempotency: if this file is already staged in ComfyUI's input dir, don't
    # re-upload it — return its bare filename instead. Covers (a) pointing at a
    # file that already lives in the input dir, (b) a canvas LoadImage widget that
    # references an input by bare name, and (c) re-staging a file that was already
    # staged this turn. Only for flat input uploads and when not overwriting.
    # Best-effort: any failure falls through to a normal upload.
    if image_type == "input" and not subfolder and not overwrite:
        try:
            input_dir = comfy_input_dir()
            if input_dir:
                staged = os.path.join(input_dir, filename)
                same_path = os.path.abspath(resolved) == os.path.abspath(staged)
                if os.path.isfile(staged) and (
                    same_path or os.path.getsize(staged) == os.path.getsize(resolved)
                ):
                    return {
                        "name": filename, "subfolder": "", "type": "input",
                        "note": "already staged in ComfyUI input dir; upload skipped",
                    }
        except Exception:  # noqa: BLE001
            pass

    try:
        with open(resolved, "rb") as f:
            files = {"image": (filename, f, "image/png")}
            data = {"type": image_type, "overwrite": str(overwrite).lower()}
            if subfolder:
                data["subfolder"] = subfolder
            return get_client().post("/upload/image", data=data, files=files)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@tool
def download_image(image_url: str, subfolder: str = "", downsize: bool = True) -> str:
    """Download a web image straight into ComfyUI's input folder so a workflow can load it.

    Use this right after an image search to fetch a reference you found (pass the
    result's ``image_url``). The image is uploaded into ComfyUI's input directory
    and can then be referenced directly by a ``LoadImage`` node using the returned
    ``name`` and ``subfolder`` — no separate upload call is needed.

    A browser User-Agent is sent so hosts that block hot-linking still serve it.

    Args:
        image_url: Direct http/https URL of the image.
        subfolder: Input-dir subfolder to store the image in. Empty (the default)
                   puts it in the input root, which is the safe choice: LoadImage
                   on some ComfyUI builds cannot read input subdirectories.
        downsize:  When True (default), oversized images are downscaled to the
                   5 MB / 1568 px limit so they stay usable everywhere (matching
                   how user-uploaded images are handled). False keeps the original.

    The downloaded image is put in front of the user automatically — in agentY it
    is dropped onto their ComfyUI canvas as a loader node and shown in the chat,
    exactly as a generated image is. So downloading IS showing: do not follow it
    by building or editing a workflow to display the picture, and do not offer to.

    Returns:
        JSON ``{"name", "subfolder", "type", "saved_to", "width", "height",
        "size_bytes", "source_url"}`` on success, or ``{"error": "<message>"}``.
        ``name`` + ``subfolder`` are what a ``LoadImage`` node references;
        ``saved_to`` is the resolved on-disk path.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/png,image/jpeg,*/*;q=0.8",
        }
        resp = requests.get(image_url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.content
        mime = resp.headers.get("content-type", "")

        # Resolve the format from content-type / URL extension, then magic bytes.
        img_fmt = detect_format(image_url.split("?")[0], mime)
        if img_fmt is None:
            if data[:4] == b"\x89PNG":
                img_fmt = "png"
            elif data[:3] == b"\xff\xd8\xff":
                img_fmt = "jpeg"
            elif data[:6] in (b"GIF87a", b"GIF89a"):
                img_fmt = "gif"
            elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
                img_fmt = "webp"
        if img_fmt is None:
            return json.dumps(
                {"error": f"URL did not return a recognised image (content-type={mime!r})."}
            )

        # Optionally normalise to the size/edge limits. downsize only targets
        # png/jpeg; gif/webp are uploaded as they are.
        ext = img_fmt
        if downsize and img_fmt in ("png", "jpeg"):
            try:
                data, ext = _downsize(data, img_fmt)
            except Exception:  # noqa: BLE001
                ext = img_fmt  # keep the original bytes if downsizing fails

        suffix = "jpg" if ext == "jpeg" else ext
        base = image_url.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(base).stem)[:48] or "reference"
        filename = f"{stem}_{uuid.uuid4().hex[:8]}.{suffix}"

        # Upload via the API (filesystem-agnostic; works whether ComfyUI is local
        # or remote). ComfyUI creates the subfolder and returns the authoritative
        # {name, subfolder, type}.
        files = {"image": (filename, io.BytesIO(data), f"image/{ext}")}
        form: dict = {"type": "input", "overwrite": "false"}
        if subfolder:
            form["subfolder"] = subfolder
        up = get_client().post("/upload/image", data=form, files=files)
        if not isinstance(up, dict) or "name" not in up:
            return json.dumps({"error": f"Unexpected /upload/image response: {up!r}"})

        saved_to = ""
        try:
            input_dir = comfy_input_dir() or ""
            if input_dir:
                saved_to = str(Path(input_dir) / up.get("subfolder", "") / up["name"])
        except Exception:  # noqa: BLE001
            pass

        try:
            with Image.open(io.BytesIO(data)) as im:
                width, height = im.size
        except Exception:  # noqa: BLE001
            width = height = None

        # Show it. Downloading an image is an explicit act — the caller wanted
        # this picture — so the host gets to put it in front of the user, the
        # same as anything a render produced. Without this the file lands in
        # ComfyUI's input directory and is mentioned in a tool result, which no
        # one is looking at: agents were reduced to building a workflow of
        # LoadImage nodes to make a downloaded reference visible.
        #
        # Only what actually decodes. The format is taken from the content-type
        # or the URL's extension, so a hotlink block or a login page served at
        # `…/photo.jpg` still arrives looking like a JPEG — and putting THAT on
        # someone's canvas is a node that shows nothing and fails when it runs.
        # `width` is None exactly when PIL could not read the bytes.
        if width:
            _publish(saved_to)

        return json.dumps({
            "name": up.get("name"),
            "subfolder": up.get("subfolder", subfolder),
            "type": up.get("type", "input"),
            "saved_to": saved_to,
            "width": width,
            "height": height,
            "size_bytes": len(data),
            "source_url": image_url,
        })
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


@tool
def upload_file_to_url(url: str, file_path: str, content_type: str = "") -> str:
    """HTTP PUT a local file's raw bytes to a presigned upload URL.

    Use this for the **presigned-PUT step of an MCP upload flow** (e.g. Magnific:
    ``creations_create_upload`` returns a ``proxyUploadUrl`` + a server-side
    ``path``; PUT the file here, then call ``creations_finalize_upload`` with that
    ``path``). It reads ``file_path`` from disk and sends the body as raw binary —
    do **not** hand-write a ``python -c`` / ``curl`` script for this (multi-line
    inline scripts silently fail to execute via ``run_script`` on Windows).

    The returned ``status`` and ``ok`` are the **actual HTTP result**, so you can
    verify the upload really landed before finalizing — an unverified finalize is
    what produces "Upload not found … Did the PUT succeed?".

    Args:
        url: The presigned PUT target (e.g. Magnific's ``proxyUploadUrl``).
        file_path: Local path to the file to upload (absolute preferred; a
                   relative path is resolved against the project root).
        content_type: Value for the ``Content-Type`` header. Defaults to the type
                      inferred from the file extension (falls back to
                      ``application/octet-stream``).

    Returns:
        JSON ``{"ok", "status", "file_path", "size_bytes", "content_type",
        "response"}`` — ``ok`` is true only for a 2xx status. ``response`` is the
        server body (truncated). On a local failure (missing file, network error)
        returns ``{"ok": false, "error": "<message>"}``.
    """
    try:
        p = Path(file_path)
        if not p.is_absolute():
            # The consuming app's root, not this package's — each host calls
            # set_project_root() at startup, which is what makes a relative path
            # mean the same thing here as it does everywhere else in that app.
            p = (project_root() / file_path).resolve()
        if not p.exists():
            return json.dumps({"ok": False, "error": f"file not found: {p}"})
        if not p.is_file():
            return json.dumps({"ok": False, "error": f"not a file: {p}"})

        ct = content_type.strip() or _MIME_BY_EXT.get(
            p.suffix.lower(), "application/octet-stream"
        )
        data = p.read_bytes()
        resp = requests.put(url, data=data, headers={"Content-Type": ct}, timeout=120)
        body = (resp.text or "")[:500]
        return json.dumps({
            "ok": 200 <= resp.status_code < 300,
            "status": resp.status_code,
            "file_path": str(p),
            "size_bytes": len(data),
            "content_type": ct,
            "response": body,
        })
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": str(exc), "file_path": file_path})
