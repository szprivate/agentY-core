"""Looking around: node source, folders, workflow files, media, video frames, web pages.

Six read-only tools for what agents kept doing with run_script. From the logs
(643 run_script calls, July–October): reading a node's Python to see how it
works (30%), listing folders for the newest render or an input (21%), searching
saved workflows (15%), probing media with ffprobe/PIL (6%), pulling frames for
a contact sheet (3%), fetching a README (1%). Each is one call here, with an
answer shaped for the question rather than a page of stdout.
"""

from __future__ import annotations

import ast
import datetime as _dt
import html
import json
import logging
import os
import re
import shutil
import struct
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import requests

from agenty_core._compat import tool

logger = logging.getLogger(__name__)


def _out(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _mb(n) -> float | None:
    try:
        return round(int(n) / 1e6, 2)
    except (TypeError, ValueError):
        return None


def _when(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


# ── ComfyUI's folders ────────────────────────────────────────────────────────

def _comfy_dirs() -> dict:
    try:
        from agenty_core.tools.comfyui import get_comfyui_dirs
        d = json.loads(get_comfyui_dirs())
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _comfy_root() -> Path | None:
    try:
        from agenty_core.tools.comfyui import _comfyui_root
        return _comfyui_root()
    except Exception:  # noqa: BLE001
        return None


def _agent_dirs() -> dict:
    try:
        from agenty_core.tools.comfyui import get_agent_output_dirs
        d = json.loads(get_agent_output_dirs())
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _resolve(path: str) -> Path:
    """A path, or one of ComfyUI's folders by name: "input", "output", "temp",
    "user", "workflows", "agent" — optionally followed by a subpath
    ("output/spec/spec_0070")."""
    p = str(path or "").strip().strip('"')
    head, _, rest = p.replace("\\", "/").partition("/")
    key = head.lower()
    base: Path | None = None
    if key in ("input", "output", "user", "temp"):
        d = _comfy_dirs()
        if key == "temp":
            out = d.get("output_dir")
            base = Path(out).parent / "temp" if out and out != "unknown" else None
        else:
            v = d.get(f"{key}_dir")
            base = Path(v) if v and v != "unknown" else None
    elif key == "workflows":
        u = _comfy_dirs().get("user_dir")
        base = Path(u) / "default" / "workflows" if u and u != "unknown" else None
    elif key == "agent":
        out = _comfy_dirs().get("output_dir")
        base = Path(out) / "agent" if out and out != "unknown" else None
    if base is not None:
        return base / rest if rest else base
    return Path(p)


# ── get_node_source ──────────────────────────────────────────────────────────

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "tests", "test",
              "web", "js", "docs", "example_workflows"}


def _py_files(target: Path, limit: int = 600) -> list[Path]:
    if target.is_file():
        return [target]
    out = []
    for dirpath, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        out.extend(Path(dirpath) / f for f in files if f.endswith(".py"))
        if len(out) >= limit:
            break
    return out


def _module_target(node_class: str) -> tuple[Path | None, str]:
    """The file or node-pack folder that defines *node_class*, from ComfyUI's
    own record (``python_module`` in /object_info)."""
    try:
        from agenty_core.utils.comfyui_client import get_client
        info = get_client().get(f"/object_info/{node_class}") or {}
        mod = (info.get(node_class) or {}).get("python_module", "") if isinstance(info, dict) else ""
    except Exception:  # noqa: BLE001
        mod = ""
    root = _comfy_root()
    if not mod or root is None:
        return None, mod
    if mod.startswith("custom_nodes."):
        return root / "custom_nodes" / mod.split(".", 1)[1], mod
    as_file = root / (mod.replace(".", "/") + ".py")
    if as_file.is_file():
        return as_file, mod
    as_pkg = root / mod.replace(".", "/")
    return (as_pkg if as_pkg.is_dir() else None), mod


def _class_for(node_class: str, files: list[Path]) -> tuple[Path, str] | None:
    """(file, Python class name) that implements node *node_class*."""
    key = re.escape(node_class)
    mapping = re.compile(r"""["']%s["']\s*:\s*([A-Za-z_][\w.]*)""" % key)
    node_id = re.compile(r"""node_id\s*=\s*["']%s["']""" % key)
    plain = re.compile(r"^class\s+%s\b" % key, re.M)
    hits_plain, hits_id, hits_map = [], [], []
    for f in files:
        try:
            src = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if plain.search(src):
            hits_plain.append((f, node_class))
        m = node_id.search(src)
        if m:
            # V3 schema: the class whose define_schema holds this node_id.
            before = src[:m.start()]
            cls = re.findall(r"^class\s+(\w+)", before, re.M)
            if cls:
                hits_id.append((f, cls[-1]))
        for m in mapping.finditer(src):
            hits_map.append((f, m.group(1).rsplit(".", 1)[-1]))
    # A mapping names the class; find the file that defines it.
    for _f, cls in hits_map:
        for f in files:
            try:
                if re.search(r"^class\s+%s\b" % re.escape(cls), f.read_text(encoding="utf-8", errors="replace"), re.M):
                    return f, cls
            except OSError:
                continue
    return (hits_id or hits_plain or [None])[0]


def _class_source(path: Path, cls: str, helpers: bool, max_chars: int) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
    node = top.get(cls)
    if node is None:
        return {"error": f"class {cls} not found in {path}"}
    lines = src.splitlines()

    def seg(n) -> str:
        start = min([d.lineno for d in getattr(n, "decorator_list", [])] + [n.lineno])
        return "\n".join(lines[start - 1:n.end_lineno])

    parts = [(cls, node.lineno, seg(node))]
    used = []
    if helpers:
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
                {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        # Bases defined in the same file too (the parent's execute is often the point).
        names |= {b.id for b in node.bases if isinstance(b, ast.Name)}
        for name in sorted(names):
            n = top.get(name)
            if n is not None and name != cls:
                parts.append((name, n.lineno, seg(n)))
                used.append(name)
    text, kept, cut = "", [], []
    for name, line, body in parts:
        block = f"# {path.name}:{line}  {name}\n{body}\n\n"
        if len(text) + len(block) > max_chars and kept:
            cut.append(name)
            continue
        text += block[: max(0, max_chars - len(text))]
        kept.append(name)
    out = {"file": str(path), "class": cls, "line": node.lineno, "source": text.rstrip()}
    if used:
        out["helpers_included"] = [n for n in kept if n != cls]
    if cut:
        out["helpers_left_out"] = cut
        out["hint"] = "Ask again with query=<name> for a helper left out."
    return out


def _search_code(files: list[Path], query: str, context: int, limit: int = 20) -> list[dict]:
    try:
        rx = re.compile(query, re.I)
    except re.error:
        rx = re.compile(re.escape(query), re.I)
    out = []
    for f in files:
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            if rx.search(line):
                lo, hi = max(0, i - context), min(len(lines), i + context + 1)
                out.append({"file": str(f), "line": i + 1,
                            "code": "\n".join(f"{j + 1:5}  {lines[j]}" for j in range(lo, hi))})
                if len(out) >= limit:
                    return out
    return out


@tool
def get_node_source(node_class: str = "", query: str = "", path: str = "",
                    context: int = 12, max_chars: int = 14000) -> str:
    """How a ComfyUI node — or any part of ComfyUI — actually works: its Python.

    Use this instead of reading .py files with run_script.
    - node_class alone: the class that implements the node (found the way ComfyUI
      itself knows it, core or custom node pack), with the helper functions and
      base classes it uses from the same file.
    - node_class + query: lines matching `query` (a regex or plain text) in that
      node's file or pack, with `context` lines around each.
    - query + path: search a file or folder — relative to ComfyUI's folder
      ("comfy/model_base.py", "comfy/ldm/lightricks", "custom_nodes/ComfyUI-KJNodes")
      or absolute. query alone searches ComfyUI's core (nodes.py, comfy/, comfy_extras/).

    For a node's inputs/outputs (not its code) use get_node_schema.
    """
    node_class, query = str(node_class or "").strip(), str(query or "").strip()
    root = _comfy_root()
    if not node_class and not query:
        return _out({"ok": False, "error": "Give node_class, or a query to search for."})
    if path:
        target = Path(path)
        if not target.is_absolute() and root is not None:
            target = root / path
        if not target.exists():
            return _out({"ok": False, "error": f"No such file or folder: {target}"})
        files = _py_files(target)
    elif node_class:
        target, module = _module_target(node_class)
        if target is None:
            return _out({"ok": False, "error": f"ComfyUI does not know a node called {node_class!r}"
                         + (f" (module {module})" if module else "") + ".",
                         "hint": "search_nodes finds the right name."})
        files = _py_files(target)
    else:
        if root is None:
            return _out({"ok": False, "error": "ComfyUI's folder is not known; pass path."})
        files = [root / "nodes.py"] + _py_files(root / "comfy") + _py_files(root / "comfy_extras")
    if query:
        hits = _search_code(files, query, max(0, min(int(context or 0), 60)))
        return _out({"ok": True, "query": query, "searched_files": len(files), "matches": hits,
                     **({} if hits else {"hint": "No match. Try a shorter or different word."})})
    found = _class_for(node_class, files)
    if not found:
        return _out({"ok": False, "error": f"Could not find the class for {node_class} in {target}."})
    f, cls = found
    try:
        return _out({"ok": True, "node_class": node_class,
                     **_class_source(f, cls, True, max(2000, min(int(max_chars or 14000), 40000)))})
    except SyntaxError as exc:
        return _out({"ok": False, "error": f"{f} does not parse: {exc}"})


# ── list_files ───────────────────────────────────────────────────────────────

_SEQ = re.compile(r"^(?P<head>.*?[._]?)(?P<num>\d{3,8})(?P<tail>\.[A-Za-z0-9]+)$")


def _collapse(entries: list[dict]) -> list[dict]:
    """Numbered frames become one entry per sequence: name.####.exr 1001-1120."""
    groups: dict[tuple, list] = {}
    rest = []
    for e in entries:
        m = _SEQ.match(e["name"])
        if m:
            groups.setdefault((str(Path(e["name"]).parent), m.group("head").rsplit("/", 1)[-1],
                               m.group("tail"), len(m.group("num"))), []).append((int(m.group("num")), e))
        else:
            rest.append(e)
    for (parent, head, tail, width), items in groups.items():
        if len(items) < 3:
            rest.extend(e for _n, e in items)
            continue
        nums = sorted(n for n, _e in items)
        missing = sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))
        pat = f"{head}{'#' * width}{tail}"
        rest.append({"name": pat if parent in (".", "") else f"{parent}/{pat}",
                     "sequence": True, "frames": f"{nums[0]}-{nums[-1]}", "count": len(nums),
                     "missing": missing[:20] + (["…"] if len(missing) > 20 else []),
                     "size_mb": round(sum(e.get("size_mb") or 0 for _n, e in items), 2),
                     "modified": max(e["modified"] for _n, e in items)})
    return rest


@tool
def list_files(path: str, pattern: str = "", recursive: bool = False,
               newest_first: bool = True, limit: int = 60) -> str:
    """What is in a folder: files with size and date, newest first, numbered frames
    collapsed into one line per sequence (render.####.exr 1001-1120, 2 missing).

    Use this instead of dir / ls / os.listdir / glob in run_script.

    Args:
        path: A folder — or ComfyUI's by name: "input", "output", "temp",
            "workflows", "agent" (the agent's outputs), optionally with a subpath
            ("output/spec/spec_0070/videos"). A file path answers for that file.
        pattern: Glob or substring of the name ("*.mp4", "concept_", "*beauty*.exr").
        recursive: Include subfolders.
        newest_first: Sort by date (default) rather than name.
        limit: Max entries (default 60, max 400).
    """
    import fnmatch
    base = _resolve(path)
    if base.is_file():
        st = base.stat()
        return _out({"ok": True, "path": str(base), "exists": True, "is_file": True,
                     "size_mb": _mb(st.st_size), "modified": _when(st.st_mtime)})
    if not base.is_dir():
        return _out({"ok": True, "path": str(base), "exists": False})
    limit = max(1, min(int(limit or 60), 400))
    pat = str(pattern or "").lower()
    match = (lambda n: fnmatch.fnmatch(n.lower(), pat)) if any(c in pat for c in "*?[") \
        else (lambda n: pat in n.lower())
    files, folders, scanned = [], [], 0
    walker = os.walk(base) if recursive else [(str(base), [d.name for d in os.scandir(base) if d.is_dir()],
                                               [f.name for f in os.scandir(base) if f.is_file()])]
    for dirpath, dirs, names in walker:
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if not recursive:
            for d in dirs:
                full = Path(dirpath) / d
                try:
                    folders.append({"name": d + "/", "modified": _when(full.stat().st_mtime)})
                except OSError:
                    pass
        for n in names:
            scanned += 1
            if pat and not match(n):
                continue
            full = Path(dirpath) / n
            try:
                st = full.stat()
            except OSError:
                continue
            files.append({"name": str(full.relative_to(base)).replace("\\", "/"),
                          "size_mb": _mb(st.st_size), "modified": _when(st.st_mtime)})
        if scanned > 50000:
            break
    files = _collapse(files)
    key = (lambda e: e["modified"]) if newest_first else (lambda e: e["name"].lower())
    files.sort(key=key, reverse=bool(newest_first))
    folders.sort(key=key, reverse=bool(newest_first))
    result = {"ok": True, "path": str(base), "file_count": len(files), "files": files[:limit]}
    if folders and not pat:
        result["folders"] = folders[:limit]
    if len(files) > limit:
        result["more"] = len(files) - limit
    return _out(result)


# ── find_workflows ───────────────────────────────────────────────────────────

def _workflow_roots(folder: str) -> list[tuple[str, Path]]:
    folder = (folder or "all").lower()
    roots: list[tuple[str, Path]] = []
    user = _comfy_dirs().get("user_dir")
    if folder in ("all", "user", "agent") and user and user != "unknown":
        wf = Path(user) / "default" / "workflows"
        roots.append(("agent" if folder == "agent" else "user", wf / "agentY" if folder == "agent" else wf))
    if folder in ("all", "templates"):
        try:
            from agenty_core.tools.comfyui import _custom_templates_dir, _official_templates_dir
            roots += [("templates", _custom_templates_dir()), ("templates", _official_templates_dir())]
        except Exception:  # noqa: BLE001
            pass
    root = _comfy_root()
    if folder in ("all", "examples") and root is not None:
        cn = root / "custom_nodes"
        if cn.is_dir():
            for pack in cn.iterdir():
                for sub in ("example_workflows", "examples", "workflows", "workflow"):
                    if (pack / sub).is_dir():
                        roots.append(("examples", pack / sub))
    if folder not in ("all", "user", "agent", "templates", "examples"):
        roots.append(("folder", _resolve(folder)))
    return [(k, p) for k, p in roots if p and Path(p).is_dir()]


@tool
def find_workflows(contains: str = "", folder: str = "all", newest_first: bool = True,
                   limit: int = 20) -> str:
    """Find workflow JSON files by what is in them: node types, model files,
    titles, prompt text — every word must appear. Newest first.

    Use this instead of globbing and grepping workflow files in run_script; then
    inspect_workflow_file on a hit says what it does.

    Args:
        contains: Words that must all be in the workflow ("LTXVAddGuide",
            "wan2.2 EmptyHunyuanLatentVideo", "Panel 3"). Empty lists the newest.
        folder: "user" (ComfyUI's saved workflows, the agent's included), "agent"
            (only the agent's), "templates", "examples" (node packs' example
            workflows), "all" (default), or a folder path.
        newest_first: Sort by date (default) rather than name.
        limit: Max results (default 20, max 100).
    """
    terms = [t.lower() for t in str(contains or "").split() if t]
    limit = max(1, min(int(limit or 20), 100))
    hits, scanned = [], 0
    for kind, root in _workflow_roots(folder):
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if not f.lower().endswith(".json"):
                    continue
                full = Path(dirpath) / f
                try:
                    st = full.stat()
                    if st.st_size > 8_000_000:
                        continue
                    scanned += 1
                    text = full.read_text(encoding="utf-8", errors="replace") if terms else ""
                except OSError:
                    continue
                low = text.lower()
                if terms and not all(t in low for t in terms):
                    continue
                entry = {"path": str(full), "where": kind, "modified": _when(st.st_mtime),
                         "size_kb": round(st.st_size / 1e3, 1)}
                if terms:
                    try:
                        data = json.loads(text)
                        nodes = data.get("nodes") if isinstance(data, dict) else None
                        types = [n.get("type") for n in nodes] if isinstance(nodes, list) else \
                            [v.get("class_type") for v in data.values() if isinstance(v, dict)] if isinstance(data, dict) else []
                        entry["nodes"] = len([t for t in types if t])
                        entry["matching_node_types"] = sorted({t for t in types if t and any(x in t.lower() for x in terms)})[:10]
                    except Exception:  # noqa: BLE001
                        pass
                hits.append(entry)
    hits.sort(key=(lambda e: e["modified"]) if newest_first else (lambda e: e["path"].lower()),
              reverse=bool(newest_first))
    result = {"ok": True, "searched": scanned, "count": len(hits), "workflows": hits[:limit],
              "folders": [str(p) for _k, p in _workflow_roots(folder)]}
    if len(hits) > limit:
        result["more"] = len(hits) - limit
    return _out(result)


# ── media_info ───────────────────────────────────────────────────────────────

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".mxf")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")


def _ffmpeg() -> str | None:
    try:
        from agenty_core.tools.video import _ffmpeg_exe
        return _ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return shutil.which("ffmpeg")


def _probe_video(p: Path) -> dict:
    probe = shutil.which("ffprobe")
    if probe:
        r = subprocess.run([probe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(p)],
                           capture_output=True, text=True, timeout=60)
        data = json.loads(r.stdout or "{}")
        out: dict = {"container": (data.get("format") or {}).get("format_name"),
                     "duration_s": _f((data.get("format") or {}).get("duration")),
                     "size_mb": _mb((data.get("format") or {}).get("size"))}
        for s in data.get("streams") or []:
            if s.get("codec_type") == "video" and "video" not in out:
                fps = _rate(s.get("avg_frame_rate") or s.get("r_frame_rate"))
                frames = s.get("nb_frames")
                out["video"] = {"codec": s.get("codec_name"), "width": s.get("width"),
                                "height": s.get("height"), "fps": fps, "pix_fmt": s.get("pix_fmt"),
                                "frames": int(frames) if str(frames or "").isdigit()
                                else (round(fps * out["duration_s"]) if fps and out.get("duration_s") else None),
                                "color": {k: s.get(k) for k in ("color_space", "color_transfer", "color_primaries") if s.get(k)}}
            elif s.get("codec_type") == "audio" and "audio" not in out:
                out["audio"] = {"codec": s.get("codec_name"), "sample_rate": s.get("sample_rate"),
                                "channels": s.get("channels")}
        out.setdefault("audio", None)
        return out
    import cv2
    cap = cv2.VideoCapture(str(p))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        return {"video": {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                          "fps": round(fps, 3) if fps else None, "frames": frames},
                "duration_s": round(frames / fps, 3) if fps and frames else None,
                "audio": "unknown (no ffprobe)"}
    finally:
        cap.release()


def _f(v):
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def _rate(v) -> float | None:
    try:
        a, _, b = str(v).partition("/")
        return round(float(a) / float(b or 1), 3) if float(b or 1) else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _exr_header(p: Path) -> dict:
    """The few EXR header attributes that matter, read without OpenEXR."""
    with open(p, "rb") as fh:
        head = fh.read(65536)
    if head[:4] != b"\x76\x2f\x31\x01":
        return {"error": "not an OpenEXR file"}
    out: dict = {"format": "exr", "multipart": bool(head[5] & 0x10), "deep": bool(head[5] & 0x08)}
    i = 8
    comp = ["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab"]
    while i < len(head) and head[i] != 0:
        name_end = head.index(b"\0", i)
        name = head[i:name_end].decode("latin-1")
        type_end = head.index(b"\0", name_end + 1)
        typ = head[name_end + 1:type_end].decode("latin-1")
        size = struct.unpack_from("<i", head, type_end + 1)[0]
        val = head[type_end + 5:type_end + 5 + size]
        if typ == "chlist":
            chans, j = [], 0
            while j < len(val) and val[j] != 0:
                e = val.index(b"\0", j)
                cname = val[j:e].decode("latin-1")
                ptype = struct.unpack_from("<i", val, e + 1)[0]
                chans.append(f"{cname}:{['uint', 'half', 'float'][ptype] if 0 <= ptype < 3 else ptype}")
                j = e + 1 + 16
            out["channels"] = chans
        elif typ == "box2i" and name in ("dataWindow", "displayWindow"):
            x0, y0, x1, y1 = struct.unpack_from("<iiii", val)
            out[name] = {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}
        elif typ == "compression":
            out["compression"] = comp[val[0]] if val and val[0] < len(comp) else val[0]
        elif typ == "string" and name in ("chromaticities", "owner", "comments", "software"):
            out[name] = val.decode("latin-1", "replace")[:200]
        elif typ == "chromaticities" and len(val) >= 32:
            out["chromaticities"] = [round(x, 4) for x in struct.unpack_from("<8f", val)]
        i = type_end + 5 + size
    if "dataWindow" in out:
        out["width"], out["height"] = out["dataWindow"]["width"], out["dataWindow"]["height"]
    return out


def _image_info(p: Path) -> dict:
    from PIL import Image
    with Image.open(p) as im:
        out = {"format": im.format, "width": im.width, "height": im.height, "mode": im.mode,
               "channels": list(im.getbands()),
               "bit_depth": 16 if im.mode.startswith("I;16") or im.mode in ("I", "I;16B") else
               32 if im.mode == "F" else 8,
               "frames": getattr(im, "n_frames", 1),
               "icc_profile": bool(im.info.get("icc_profile"))}
        if "A" in im.getbands() or "transparency" in im.info:
            import numpy as np
            a = np.asarray(im.convert("RGBA").getchannel("A"))
            clear = a < 128
            out["alpha"] = {"min": int(a.min()), "max": int(a.max()),
                            "transparent_fraction": round(float(clear.mean()), 4),
                            "partly_transparent_fraction": round(float(((a > 0) & (a < 255)).mean()), 4)}
            if clear.any():
                ys, xs = np.nonzero(clear)
                out["alpha"]["transparent_box"] = {"x": int(xs.min()), "y": int(ys.min()),
                                                   "width": int(xs.max() - xs.min() + 1),
                                                   "height": int(ys.max() - ys.min() + 1)}
            out["alpha"]["note"] = ("ComfyUI's LoadImage turns alpha into a MASK of 1-alpha: the "
                                    "transparent part is white (1) in the mask.")
    return out


@tool
def media_info(path: str) -> str:
    """Facts about a video, an image, an EXR or an image sequence — without
    opening it in a script.

    - video: codec, size, fps, frame count, duration, pixel format, colour tags, audio
    - image: size, mode, channels, bit depth; alpha — how much is transparent and
      where (what a LoadImage mask will be)
    - EXR: channels and their types, data/display window, compression
    - a folder or a "name.####.exr" pattern: the sequence's frame range and gaps,
      and the first frame's facts

    Args:
        path: The file, a folder of frames, or a sequence pattern. ComfyUI's folders
            work by name ("input/plate.mov", "output/…").
    """
    p = _resolve(path)
    try:
        if p.is_dir() or "#" in p.name or "%0" in p.name:
            folder = p if p.is_dir() else p.parent
            listing = json.loads(list_files(str(folder), pattern="" if p.is_dir() else
                                            re.sub(r"(#+|%0\d+d)", "*", p.name), limit=400))
            seqs = [e for e in listing.get("files", []) if e.get("sequence")]
            if not seqs:
                return _out({"ok": False, "error": f"No numbered frames in {folder}."})
            seq = seqs[0]
            head, tail = re.split(r"#+", seq["name"].rsplit("/", 1)[-1], maxsplit=1)
            first = min((f for f in os.listdir(folder)
                         if f.startswith(head) and f.endswith(tail) and _SEQ.match(f)),
                        key=lambda f: int(_SEQ.match(f).group("num")))
            facts = json.loads(media_info(str(folder / first)))
            facts.pop("ok", None)
            return _out({"ok": True, "path": str(folder), "sequence": seq, "first_frame": facts})
        if not p.is_file():
            return _out({"ok": False, "error": f"No such file: {p}"})
        ext = p.suffix.lower()
        if ext == ".exr":
            info = _exr_header(p)
        elif ext in VIDEO_EXT and ext != ".gif":
            info = _probe_video(p)
        else:
            info = _image_info(p)
        info["size_mb"] = info.get("size_mb") or _mb(p.stat().st_size)
        return _out({"ok": True, "path": str(p), **info})
    except Exception as exc:  # noqa: BLE001
        return _out({"ok": False, "path": str(p), "error": f"{type(exc).__name__}: {exc}"})


# ── video_frames ─────────────────────────────────────────────────────────────

def _frames_dir(stem: str) -> Path:
    images = _agent_dirs().get("images")
    base = Path(images) if images else Path.cwd() / "output" / "frames"
    d = base / "frames" / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)[:60]}_{_dt.datetime.now():%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sheet(paths: list[Path], labels: list[str], out: Path, width: int = 1600) -> Path:
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in paths]
    cols = 3 if len(ims) > 4 else min(len(ims), 2) if len(ims) > 1 else 1
    rows = -(-len(ims) // cols)
    cw = width // cols
    ch = int(cw * ims[0].height / max(1, ims[0].width))
    sheet = Image.new("RGB", (cw * cols, (ch + 22) * rows), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for k, (im, label) in enumerate(zip(ims, labels)):
        x, y = (k % cols) * cw, (k // cols) * (ch + 22)
        sheet.paste(im.resize((cw, ch)), (x, y + 22))
        draw.text((x + 6, y + 4), label, fill=(230, 230, 230))
    sheet.save(out)
    return out


@tool
def video_frames(path: str, count: int = 6, frames: list | None = None,
                 times: list | None = None, contact_sheet: bool = True) -> str:
    """Pull still frames out of a video (or pick them from an image sequence) to
    look at — evenly spaced, or the frames / times you name — and, by default,
    one contact sheet with every frame labelled. Hand the paths to view_image or
    analyze_image.

    Use this instead of ffmpeg in run_script.

    Args:
        path: A video, a folder of frames, or a "name.####.png" pattern.
        count: How many evenly spaced frames when neither frames nor times is
            given (default 6, max 24).
        frames: Frame numbers (0-based for videos; the file numbers for a sequence).
        times: Times in seconds (videos).
        contact_sheet: Also save one image with all frames in a grid (default on).
    """
    p = _resolve(path)
    count = max(1, min(int(count or 6), 24))
    try:
        if p.is_dir() or "#" in p.name:
            folder = p if p.is_dir() else p.parent
            files = sorted(f for f in os.listdir(folder) if _SEQ.match(f) and
                           (p.is_dir() or f.endswith(p.suffix)))
            if not files:
                return _out({"ok": False, "error": f"No numbered frames in {folder}."})
            nums = {int(_SEQ.match(f).group("num")): f for f in files}
            order = sorted(nums)
            pick = [n for n in (frames or []) if n in nums] if frames else \
                [order[round(i * (len(order) - 1) / max(1, count - 1))] for i in range(min(count, len(order)))]
            out_paths = [folder / nums[n] for n in dict.fromkeys(pick)]
            labels = [f"frame {n}" for n in dict.fromkeys(pick)]
            out_dir = None
            if out_paths and out_paths[0].suffix.lower() in (".exr", ".hdr", ".dpx", ".tif", ".tiff"):
                # Not viewable as they are: convert, linear -> sRGB for scene-linear EXR.
                ffmpeg = _ffmpeg()
                if not ffmpeg:
                    return _out({"ok": False, "error": "ffmpeg is needed to view these frames."})
                out_dir = _frames_dir(folder.name)
                shown = []
                for src_frame, label in zip(out_paths, labels):
                    target = out_dir / f"{src_frame.stem}.png"
                    pre = ["-apply_trc", "iec61966_2_1"] if src_frame.suffix.lower() == ".exr" else []
                    subprocess.run([ffmpeg, "-v", "error", "-y", *pre, "-i", str(src_frame), str(target)],
                                   capture_output=True, text=True, timeout=120)
                    if target.is_file():
                        shown.append(target)
                if not shown:
                    return _out({"ok": False, "error": "These frames could not be converted for viewing."})
                result = {"ok": True, "source": str(folder),
                          "frames": [{"label": lab, "path": str(x), "from": str(s)}
                                     for lab, x, s in zip(labels, shown, out_paths)],
                          "note": "Converted for viewing (EXR: linear to sRGB, clipped)."}
                out_paths = shown
            else:
                result = {"ok": True, "source": str(folder),
                          "frames": [{"label": lab, "path": str(x)} for lab, x in zip(labels, out_paths)]}
            if contact_sheet and len(out_paths) > 1:
                sheet_dir = out_dir or _frames_dir(folder.name)
                result["contact_sheet"] = str(_sheet(out_paths, labels, sheet_dir / "contact_sheet.jpg"))
            return _out(result)
        if not p.is_file():
            return _out({"ok": False, "error": f"No such file: {p}"})
        info = _probe_video(p)
        v = info.get("video") or {}
        fps, total = v.get("fps") or 24.0, v.get("frames") or 0
        if times:
            want = [(float(t), f"{float(t):.2f}s") for t in times]
        elif frames:
            want = [(int(n) / fps, f"frame {int(n)}") for n in frames]
        else:
            n = max(1, total or int((info.get("duration_s") or 1) * fps))
            idx = sorted({round(i * (n - 1) / max(1, count - 1)) for i in range(min(count, n))})
            want = [(i / fps, f"frame {i}") for i in idx]
        ffmpeg = _ffmpeg()
        if not ffmpeg:
            return _out({"ok": False, "error": "ffmpeg is not available."})
        out_dir = _frames_dir(p.stem)
        made, labels = [], []
        for k, (t, label) in enumerate(want):
            target = out_dir / f"{k:02d}_{re.sub(r'[^0-9a-z.]+', '_', label)}.png"
            # +half a frame: a seek to a frame boundary can land on the one before;
            # for the last frame that runs off the end, so then on it, then before.
            for at in (t + 0.5 / fps, t, t - 0.5 / fps):
                r = subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, at):.4f}",
                                    "-i", str(p), "-frames:v", "1", str(target)],
                                   capture_output=True, text=True, timeout=120)
                if r.returncode == 0 and target.is_file():
                    made.append(target)
                    labels.append(label)
                    break
        if not made:
            return _out({"ok": False, "error": "No frame could be read.", "video": v})
        result = {"ok": True, "source": str(p), "fps": fps, "total_frames": total,
                  "frames": [{"label": lab, "path": str(m)} for lab, m in zip(labels, made)]}
        if contact_sheet and len(made) > 1:
            result["contact_sheet"] = str(_sheet(made, labels, out_dir / "contact_sheet.jpg"))
        return _out(result)
    except Exception as exc:  # noqa: BLE001
        return _out({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ── read_web_page ────────────────────────────────────────────────────────────

class _Text(HTMLParser):
    """Readable text from HTML: headings, paragraphs, lists, code; no scripts,
    navigation or footers — and only the page's own content (<main>/<article>)
    when it marks one."""
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "svg", "form", "aside", "iframe"}
    BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "pre", "blockquote",
             "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.main: list[str] = []
        self.in_main = 0
        self.skip = 0
        self.title = ""
        self._in_title = False

    def _put(self, text: str) -> None:
        self.parts.append(text)
        if self.in_main:
            self.main.append(text)

    def handle_starttag(self, tag, attrs):
        if tag in ("main", "article"):
            self.in_main += 1
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.BLOCK and not self.skip:
            self._put("\n")
            if tag in ("h1", "h2", "h3", "h4"):
                self._put("#" * int(tag[1]) + " ")
            elif tag == "li":
                self._put("- ")

    def handle_endtag(self, tag):
        if tag in ("main", "article"):
            self.in_main = max(0, self.in_main - 1)
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self.BLOCK and not self.skip:
            self._put("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self._put(data)

    @staticmethod
    def _tidy(parts: list[str]) -> str:
        t = re.sub(r"[ \t\r\f\v]+", " ", "".join(parts))
        lines = [ln.strip() for ln in t.split("\n")]
        # Menus leave bullets and heading marks with nothing after them; a bullet
        # whose text landed on the next line is joined back to it.
        out: list[str] = []
        for ln in lines:
            if re.fullmatch(r"[-#|·•\s]*", ln):
                if out and out[-1] in ("-", "#", "##", "###", "####"):
                    out.pop()
                if ln in ("-", "#", "##", "###", "####"):
                    out.append(ln)
                continue
            if out and out[-1] in ("-", "#", "##", "###", "####"):
                ln = out.pop() + " " + ln
            out.append(ln)
        if out and out[-1] in ("-", "#", "##", "###", "####"):
            out.pop()
        return "\n".join(out).strip()

    def text(self) -> str:
        main = self._tidy(self.main)
        return main if len(main) > 400 else self._tidy(self.parts)


_UA = {"User-Agent": "Mozilla/5.0 (agentY; +https://github.com/szprivate)"}


def _github(url: str, max_chars: int) -> dict | None:
    m = re.match(r"https?://github\.com/([^/]+)/([^/#?]+)(?:/(tree|blob)/([^/]+)(?:/(.*))?)?", url)
    if not m:
        return None
    owner, repo, kind, ref, sub = m.group(1), m.group(2).removesuffix(".git"), m.group(3), m.group(4), m.group(5) or ""
    api = f"https://api.github.com/repos/{owner}/{repo}"
    meta = requests.get(api, headers=_UA, timeout=30)
    if meta.status_code != 200:
        return {"ok": False, "error": f"GitHub: {meta.status_code} for {owner}/{repo}"}
    info = meta.json()
    ref = ref or info.get("default_branch") or "main"
    if kind == "blob":
        raw = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{sub}", headers=_UA, timeout=30)
        return {"ok": raw.ok, "url": url, "kind": "github file", "text": raw.text[:max_chars],
                "truncated": len(raw.text) > max_chars}
    listing = requests.get(f"{api}/contents/{sub}", params={"ref": ref}, headers=_UA, timeout=30)
    files = [f"{e['name']}{'/' if e.get('type') == 'dir' else ''}" for e in listing.json()] \
        if listing.ok and isinstance(listing.json(), list) else []
    readme = ""
    for name in ("README.md", "readme.md", "README.rst", "README.txt", "README"):
        if name in files or not files:
            r = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{(sub + '/') if sub else ''}{name}",
                             headers=_UA, timeout=30)
            if r.ok:
                readme = r.text
                break
    return {"ok": True, "url": url, "kind": "github repo", "repo": f"{owner}/{repo}", "branch": ref,
            "description": info.get("description"), "stars": info.get("stargazers_count"),
            "updated": info.get("pushed_at"), "files": files[:200],
            "readme": readme[:max_chars], "truncated": len(readme) > max_chars}


@tool
def read_web_page(url: str, max_chars: int = 12000) -> str:
    """Read a web page as text — after web_search, to read what it found: a model
    guide, a node pack's documentation, a paper's page. A GitHub repo link gives
    its description, file list and README; a GitHub file link gives the file.

    Use this instead of urllib / requests in run_script.

    Args:
        url: The page (http/https).
        max_chars: How much text to return (default 12000, max 40000).
    """
    url = str(url or "").strip()
    if not re.match(r"https?://", url):
        return _out({"ok": False, "error": "Give an http(s) URL."})
    max_chars = max(1000, min(int(max_chars or 12000), 40000))
    try:
        gh = _github(url, max_chars)
        if gh is not None:
            return _out(gh)
        r = requests.get(url, headers=_UA, timeout=30)
        ctype = r.headers.get("Content-Type", "")
        if r.status_code >= 400:
            return _out({"ok": False, "url": url, "status": r.status_code,
                         "error": f"The page answered {r.status_code}."})
        if "html" in ctype or r.text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            parser = _Text()
            parser.feed(r.text)
            text = parser.text()
            title = html.unescape(parser.title.strip())
        elif any(t in ctype for t in ("text", "json", "markdown", "xml")):
            text, title = r.text, ""
        else:
            return _out({"ok": False, "url": url, "error": f"Not a text page ({ctype or 'unknown type'})."})
        return _out({"ok": True, "url": r.url, "title": title, "text": text[:max_chars],
                     "truncated": len(text) > max_chars, "chars": len(text)})
    except Exception as exc:  # noqa: BLE001
        return _out({"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"})
