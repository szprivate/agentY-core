"""Several images on one labelled sheet, so one vision call can look at all of them.

Used by ``analyze_image(file_paths=[...])`` (agentY and agentY-mcp): N separate
vision calls cost N round trips of 12-21 s each; one tiled picture with a number
on every tile costs one. The numbers are drawn INTO the picture, so the model's
answer can say "image 3" and the caller can map that back to a file.
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

MAX_TILES = 16


def layout(count: int, columns: int | None = None) -> tuple[int, int]:
    """``(columns, rows)`` for *count* tiles: roughly square unless *columns* is given."""
    count = max(1, int(count))
    cols = int(columns) if columns else math.ceil(math.sqrt(count))
    cols = max(1, min(cols, count))
    return cols, math.ceil(count / cols)


def build_contact_sheet(paths: list, labels: list | None = None, cell: int = 512,
                        columns: int | None = None, out_path: str = "") -> dict:
    """Tile *paths* into one JPEG with a number (or label) on each tile.

    Returns ``{"path", "tiles": [{"n", "file", "label"}], "columns", "rows", "skipped": [...]}``.
    Images that cannot be opened are skipped and named. Raises ``ValueError`` when
    no image could be read.
    """
    from PIL import Image, ImageDraw, ImageFont

    paths = [str(p) for p in paths][:MAX_TILES]
    tiles, skipped = [], []
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((cell, cell))
                tiles.append((p, (labels[i] if labels and i < len(labels) and labels[i] else str(len(tiles) + 1)), im.copy()))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"file": p, "error": f"{type(exc).__name__}: {exc}"})
    if not tiles:
        raise ValueError("none of the images could be read: " + "; ".join(s["file"] for s in skipped))

    cols, rows = layout(len(tiles), columns)
    pad, bar = 6, 22
    w, h = cell + pad, cell + pad + bar
    sheet = Image.new("RGB", (cols * w + pad, rows * h + pad), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    out_tiles = []
    for idx, (src, label, im) in enumerate(tiles):
        cx, cy = idx % cols, idx // cols
        x0, y0 = pad + cx * w, pad + cy * h
        draw.text((x0 + 2, y0 + 2), f"{idx + 1}", fill=(255, 190, 60), font=font)
        draw.text((x0 + 24, y0 + 2), os.path.basename(label)[:48], fill=(220, 220, 220), font=font)
        sheet.paste(im, (x0 + (cell - im.width) // 2, y0 + bar + (cell - im.height) // 2))
        out_tiles.append({"n": idx + 1, "file": src, "label": label})

    if not out_path:
        fd, out_path = tempfile.mkstemp(prefix="agenty_contact_", suffix=".jpg")
        os.close(fd)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=88)
    return {"path": str(out_path), "tiles": out_tiles, "columns": cols, "rows": rows,
            "skipped": skipped}


def sheet_question(question: str, tiles: list) -> str:
    """The question to put with the sheet: says what the numbers are."""
    names = "; ".join(f"{t['n']} = {os.path.basename(t['file'])}" for t in tiles)
    return (f"This is ONE contact sheet of {len(tiles)} separate images, numbered in the "
            f"top-left of each tile ({names}). Answer for each image by its number"
            f"{', and compare them where it helps' if len(tiles) > 1 else ''}. "
            f"{question or 'Describe each image.'}")
