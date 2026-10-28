"""Is the ComfyUI behind a running job still there, and still doing anything?

The wait for a job (``comfyui_progress``) used to treat "cannot read /queue or
/history" as "still running", so a ComfyUI that crashed mid-render left the wait
polling a dead port for hours. ``Health`` records when ComfyUI last answered; the
wait gives up once it has been silent for ``dead_after()`` seconds. ``Activity``
records when the JOB last showed a sign of life (progress event, console line,
queue change) so a hung-but-listening ComfyUI is also given up on, after a
generous quiet period (``stall_after()``), not a wall clock alone.

Also: what a stopped run left behind. ``partial_outputs`` lists the media files
written since the job started and ``mp4_has_moov`` flags a truncated video (an
mp4 whose index was never written plays nowhere).
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path

MEDIA_EXT = {".mp4", ".mov", ".webm", ".mkv", ".gif", ".png", ".jpg", ".jpeg", ".webp",
             ".wav", ".mp3", ".flac", ".glb", ".obj", ".ply"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def dead_after() -> float:
    """Seconds of unbroken silence from ComfyUI's HTTP API after which it counts as gone."""
    return _env_float("AGENTY_COMFY_DEAD_AFTER_S", 60.0)


def stall_after() -> float:
    """Seconds without any sign of life from the running job before it counts as hung (0 = never).

    Generous by default: a video model can sit in one node (loading, VAE decode)
    for many minutes without a progress event."""
    return _env_float("AGENTY_COMFY_STALL_MIN", 45.0) * 60.0


class Health:
    """When did ComfyUI's HTTP API last answer / first stop answering."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.down_since: float | None = None

    def ok(self) -> None:
        self.down_since = None

    def fail(self) -> None:
        if self.down_since is None:
            self.down_since = self._clock()

    def dead(self, limit: float | None = None) -> bool:
        limit = dead_after() if limit is None else limit
        return self.down_since is not None and self._clock() - self.down_since >= limit

    def down_for(self) -> float:
        return 0.0 if self.down_since is None else self._clock() - self.down_since


class Activity:
    """When did the job last show a sign of life."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.last = clock()
        self._queue_sig = None
        self.running = False

    def touch(self) -> None:
        self.last = self._clock()

    def queue_seen(self, sig) -> None:
        if sig != self._queue_sig:
            self._queue_sig = sig
            self.touch()

    def stalled(self, limit: float | None = None) -> bool:
        limit = stall_after() if limit is None else limit
        return limit > 0 and self._clock() - self.last >= limit

    def quiet_for(self) -> float:
        return self._clock() - self.last


def mp4_has_moov(path: str | Path) -> bool | None:
    """True/False for an ISO-BMFF file (mp4/mov); None when it is not one or unreadable.

    Walks the top-level boxes: a recording cut short has ``ftyp``/``mdat`` and no ``moov``."""
    p = Path(path)
    if p.suffix.lower() not in {".mp4", ".mov", ".m4v"}:
        return None
    try:
        size_total = p.stat().st_size
        with open(p, "rb") as f:
            pos, seen = 0, set()
            while pos + 8 <= size_total:
                f.seek(pos)
                head = f.read(16)
                if len(head) < 8:
                    break
                size, kind = struct.unpack(">I4s", head[:8])
                if kind == b"moov":
                    return True
                seen.add(kind)
                if size == 1 and len(head) >= 16:
                    size = struct.unpack(">Q", head[8:16])[0]
                elif size == 0:
                    size = size_total - pos
                if size < 8:
                    break
                pos += size
            return False if seen else None
    except OSError:
        return None


def partial_outputs(since_wall: float, root: str | Path | None = None) -> list[dict]:
    """Media files written under ComfyUI's output folder since *since_wall* (``time.time()``)."""
    if root is None:
        try:
            from agenty_core.utils.comfyui_client import comfyui_output_dir
            root = comfyui_output_dir()
        except Exception:  # noqa: BLE001
            root = None
    if not root:
        return []
    found: list[dict] = []
    try:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if Path(name).suffix.lower() not in MEDIA_EXT:
                    continue
                full = Path(dirpath) / name
                try:
                    st = full.stat()
                except OSError:
                    continue
                if st.st_mtime >= since_wall - 2:
                    item = {"path": str(full), "bytes": st.st_size}
                    if mp4_has_moov(full) is False:
                        item["truncated"] = True
                    found.append(item)
    except OSError:
        pass
    found.sort(key=lambda d: d["path"])
    return found[:50]


def describe_partial(items: list[dict]) -> str:
    if not items:
        return "No output file was written."
    parts = []
    for it in items:
        tag = " (TRUNCATED / unplayable: no moov atom)" if it.get("truncated") else ""
        parts.append(f"{it['path']}{tag}")
    return "Partial output on disk: " + "; ".join(parts) + "."


def stopped_result(prompt_id: str, started_wall: float, *, reason: str, detail: str) -> dict:
    """The terminal dict for a job whose ComfyUI went away or hung."""
    items = partial_outputs(started_wall)
    headline = ("ComfyUI stopped responding / crashed during the run" if reason == "comfyui_unreachable"
                else "ComfyUI stopped making progress during the run (hung)")
    return {
        "error": f"{headline}: {detail} {describe_partial(items)} "
                 "Do not retry blindly; ask the user to restart ComfyUI, check its log, then re-run.",
        "comfyui_stopped": True,
        "details": {"prompt_id": prompt_id, "reason": reason, "partial_outputs": items},
    }
