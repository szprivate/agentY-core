"""Finding models: on Hugging Face, on this machine, and in a workflow file.

Five read-only tools for the questions an agent asks while setting up a
workflow: which repos have a file like this, what is in this repo, can I
download that file and where would it go, what do I already have, and what does
this workflow load. They answer in one call each what agents used to piece
together with run_script — in one recorded session, 66 hand-written scripts
(urllib against the Hub API, HEAD requests, os.walk over the model folders,
json.load on example workflows) where these make it about ten calls.

Hub notes the tools build on:
* the model search matches substrings of the repo id, so "ltx 2.5 gemma" finds
  nothing while "ltx-2.5" finds everything; :func:`hf_search` tries the forms a
  repo id actually takes, and turns a publisher named in the query into the
  ``author`` filter;
* ``full=true`` returns each repo's file list (no sizes); the tree endpoint has
  sizes;
* a HEAD on ``/resolve/`` answers without downloading: 302 with
  ``X-Linked-Size`` for a file that is there, 403/401 with ``X-Error-Code:
  GatedRepo`` when the token has not been granted the repo, 404 when it is not.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
from pathlib import Path

import requests

from agenty_core._compat import tool
from agenty_core.tools.huggingface import (
    HF_API_BASE,
    _folder_paths,
    _hf_headers,
    _models_base_dir,
    _resolve_download_dir,
)
from agenty_core.utils.model_node_mapping import guess_folder_from_filename

logger = logging.getLogger(__name__)

HF = "https://huggingface.co"
MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx")

# Publishers people name in a query ("Comfy-Org ltx", "lightricks ltx 2.5"). The
# search would look for that word inside repo names and find nothing.
KNOWN_AUTHORS = {a.lower(): a for a in (
    "Comfy-Org", "Lightricks", "Kijai", "black-forest-labs", "Wan-AI", "city96",
    "QuantStack", "comfyicu", "stabilityai", "Qwen", "tencent", "lllyasviel",
    "Kwai-Kolors", "bytedance-research", "ByteDance", "alibaba-pai", "nvidia",
    "google", "openai", "microsoft", "meta-llama", "unsloth", "lodestones",
)}


def _out(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _gb(size) -> float | None:
    try:
        return round(int(size) / 1e9, 2)
    except (TypeError, ValueError):
        return None


def _matches(name: str, pattern: str) -> bool:
    """A glob when it has wildcards, else a case-insensitive substring — of the
    whole path or of the file name."""
    if not pattern:
        return True
    name_l, pat = name.lower(), pattern.lower()
    if any(c in pat for c in "*?["):
        return fnmatch.fnmatch(name_l, pat) or fnmatch.fnmatch(name_l.rsplit("/", 1)[-1], pat)
    return pat in name_l


# Folder names ComfyUI loads models from. A repo laid out like ComfyUI's models
# folder (Comfy-Org's, Lightricks', most repackagers') says where a file goes.
COMFY_FOLDERS = {"checkpoints", "diffusion_models", "unet", "text_encoders", "clip", "vae",
                 "loras", "controlnet", "clip_vision", "upscale_models", "latent_upscale_models",
                 "embeddings", "model_patches", "audio_encoders", "style_models", "gligen",
                 "hypernetworks", "photomaker", "vae_approx", "sams", "ultralytics"}


def _comfy_folder(path: str) -> str:
    parts = [x.lower() for x in path.replace("\\", "/").split("/")[:-1]]
    for part in reversed(parts):
        if part in COMFY_FOLDERS or part in (_folder_paths() or {}):
            return {"unet": "diffusion_models", "clip": "text_encoders"}.get(part, part)
    try:
        guessed = guess_folder_from_filename(path.rsplit("/", 1)[-1]) or ""
    except Exception:  # noqa: BLE001
        guessed = ""
    return guessed.split("models/", 1)[-1].strip("/") if guessed else ""


def _query_forms(query: str) -> list[str]:
    """What a repo id containing these words would contain: as typed, joined
    with hyphens, the longest words alone."""
    q = " ".join(query.split())
    if not q:
        return [""]
    forms = [q]
    words = q.split(" ")
    if len(words) > 1:
        forms.append("-".join(words))
        # "ltx 2.5" -> "ltx-2.5"; "wan 2.2 i2v" -> "wan-2.2", "wan2.2"
        forms.append("-".join(words[:2]))
        forms.append("".join(words[:2]))
        forms.extend(sorted(words, key=len, reverse=True)[:2])
    seen, out = set(), []
    for f in forms:
        if f and f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def _split_author(query: str, author: str) -> tuple[str, str]:
    if author:
        return query, author
    words = query.split()
    for i, w in enumerate(words):
        a = KNOWN_AUTHORS.get(w.lower().rstrip("/"))
        if a:
            return " ".join(words[:i] + words[i + 1:]), a
    return query, ""


@tool
def hf_search(query: str = "", author: str = "", file_pattern: str = "", limit: int = 10) -> str:
    """Find Hugging Face repos — and, with file_pattern, the files in them you want.

    Use this instead of scripting the Hub API. One call answers "which repos have
    an LTX-2.5 text encoder" (query="ltx-2.5", file_pattern="*gemma*") or "what
    does Comfy-Org publish for Wan" (author="Comfy-Org", query="wan").

    Args:
        query: Words from the repo name (e.g. "ltx-2.5", "wan 2.2", "flux kontext").
            A publisher named here (Comfy-Org, Lightricks, Kijai…) becomes `author`.
        author: Only repos of this user/organisation (e.g. "Comfy-Org").
        file_pattern: Only repos containing a matching file, listed with those
            files. A glob ("*gemma*.safetensors") or a plain substring ("ingredients").
        limit: Max repos to return (default 10, max 30).

    Returns JSON: repos (id, downloads, gated, pipeline_tag, matching files with
    the ComfyUI models folder each belongs in). Sizes and access: hf_repo / hf_file.
    """
    query, author = _split_author(str(query or ""), str(author or "").strip())
    limit = max(1, min(int(limit or 10), 30))
    want_files = bool(file_pattern)
    repos: dict[str, dict] = {}
    tried = []
    for form in _query_forms(query):
        params = {"limit": 100 if want_files else max(limit, 20), "sort": "downloads", "direction": "-1"}
        if form:
            params["search"] = form
        if author:
            params["author"] = author
        if want_files:
            params["full"] = "true"
        tried.append(form or f"author={author}")
        try:
            r = requests.get(HF_API_BASE, headers=_hf_headers(), params=params, timeout=30)
            r.raise_for_status()
            found = r.json()
        except Exception as exc:  # noqa: BLE001
            return _out({"ok": False, "error": f"Hub search failed: {exc}", "tried": tried})
        for m in found:
            rid = m.get("id") or m.get("modelId")
            if not rid or rid in repos:
                continue
            files = [s.get("rfilename", "") for s in (m.get("siblings") or [])]
            hits = [f for f in files if _matches(f, file_pattern)] if want_files else []
            if want_files and not hits:
                continue
            entry = {"repo_id": rid, "downloads": m.get("downloads", 0),
                     "gated": m.get("gated") or False, "pipeline_tag": m.get("pipeline_tag", "")}
            if want_files:
                entry["files"] = [{"path": f, "comfy_folder": _comfy_folder(f)} for f in hits[:15]]
                if len(hits) > 15:
                    entry["more_files"] = len(hits) - 15
            repos[rid] = entry
        if len(repos) >= limit:
            break
        if repos and not want_files:
            break          # the first form that finds anything is the right one
        if not query:
            break
    ranked = sorted(repos.values(), key=lambda e: -int(e.get("downloads") or 0))[:limit]
    result = {"ok": True, "count": len(ranked), "repos": ranked, "searched": tried}
    if author:
        result["author"] = author
    if not ranked:
        result["hint"] = ("Nothing matched. The Hub matches words inside repo NAMES: try the "
                          "model family as it is spelled in repo ids (\"ltx-2.5\", \"wan2.2\"), "
                          "drop descriptive words, or search a publisher with author=… and a "
                          "file_pattern.")
    return _out(result)


def _tree(repo_id: str, revision: str = "main") -> list[dict]:
    out, url = [], f"{HF_API_BASE}/{repo_id}/tree/{revision}"
    params = {"recursive": "true"}
    for _ in range(20):                     # pages of 1000; a model repo rarely needs two
        r = requests.get(url, headers=_hf_headers(), params=params, timeout=30)
        r.raise_for_status()
        out.extend(e for e in r.json() if e.get("type") == "file")
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        if not nxt:
            break
        url, params = nxt.group(1), None
    return out


def _head(repo_id: str, path: str) -> requests.Response:
    return requests.head(f"{HF}/{repo_id}/resolve/main/{path}", headers=_hf_headers(),
                         allow_redirects=False, timeout=30)


def _access(resp: requests.Response, repo_id: str) -> dict:
    code = resp.status_code
    err = resp.headers.get("X-Error-Code", "")
    if code in (200, 302, 307):
        return {"access": "ok"}
    if err == "GatedRepo" or code in (401, 403):
        has_token = "Authorization" in _hf_headers()
        return {"access": "gated",
                "how": (f"Gated: your Hugging Face account has not been granted {repo_id}. "
                        f"The user must open {HF}/{repo_id}, accept its terms, and (if not "
                        "done yet) put a read token in HF_TOKEN.") if has_token else
                       (f"Gated and no HF_TOKEN is set: accept the terms at {HF}/{repo_id} "
                        "and set HF_TOKEN.")}
    if code == 404 or err in ("EntryNotFound", "RepoNotFound"):
        return {"access": "missing"}
    return {"access": "unknown", "status": code}


@tool
def hf_repo(repo_id: str, path_pattern: str = "", readme: bool = False) -> str:
    """What is in one Hugging Face repo: its files with sizes, whether you can
    download them, and (readme=True) the model card.

    Args:
        repo_id: e.g. "Lightricks/LTX-2.5" or a huggingface.co URL.
        path_pattern: Only files matching this glob or substring ("text_encoders/",
            "*fp8*", "gemma").
        readme: Also return the model card (first ~4000 characters, without the
            metadata header).
    """
    repo_id = _repo_from(repo_id)
    if not repo_id:
        return _out({"ok": False, "error": "Give a repo id like 'Lightricks/LTX-2.5'."})
    try:
        meta = requests.get(f"{HF_API_BASE}/{repo_id}", headers=_hf_headers(), timeout=30)
        if meta.status_code == 404:
            return _out({"ok": False, "error": f"No repo {repo_id} on Hugging Face.",
                         "hint": "hf_search finds the right id."})
        meta.raise_for_status()
        info = meta.json()
        files = _tree(repo_id)
    except Exception as exc:  # noqa: BLE001
        return _out({"ok": False, "error": f"Could not read {repo_id}: {exc}"})
    shown = [f for f in files if _matches(f.get("path", ""), path_pattern)]
    listing = []
    for f in shown[:150]:
        path = f.get("path", "")
        size = (f.get("lfs") or {}).get("size") or f.get("size")
        row = {"path": path, "size_gb": _gb(size)}
        if path.lower().endswith(MODEL_EXTENSIONS):
            row["comfy_folder"] = _comfy_folder(path)
        listing.append(row)
    result = {"ok": True, "repo_id": repo_id, "gated": info.get("gated") or False,
              "downloads": info.get("downloads", 0), "pipeline_tag": info.get("pipeline_tag", ""),
              "license": (info.get("cardData") or {}).get("license", "") if isinstance(info.get("cardData"), dict) else "",
              "file_count": len(files), "files": listing}
    if len(shown) > 150:
        result["more_files"] = len(shown) - 150
    if info.get("gated"):
        probe = next((f["path"] for f in shown if f.get("path", "").lower().endswith(MODEL_EXTENSIONS)), None)
        if probe:
            try:
                result.update(_access(_head(repo_id, probe), repo_id))
            except Exception:  # noqa: BLE001
                pass
    if readme:
        try:
            r = requests.get(f"{HF}/{repo_id}/raw/main/README.md", headers=_hf_headers(), timeout=30)
            text = r.text if r.ok else ""
            if text.startswith("---"):
                parts = text.split("---", 2)
                text = parts[2] if len(parts) == 3 else text
            result["readme"] = text.strip()[:4000]
        except Exception:  # noqa: BLE001
            result["readme"] = ""
    return _out(result)


def _repo_from(value: str) -> str:
    v = str(value or "").strip()
    m = re.match(r"https?://huggingface\.co/([^/]+/[^/?#]+)", v)
    if m:
        return m.group(1)
    return v.strip("/") if v.count("/") == 1 else ""


@tool
def hf_file(url: str = "", repo_id: str = "", path: str = "") -> str:
    """Check one file on Hugging Face without downloading it: is it there, how big,
    can you download it, where would it go in ComfyUI, and do you have it already.

    When the repo is gated and you have no access, the same file name is looked
    for in other repos (mirrors) — those can usually be downloaded at once.

    Args:
        url: A huggingface.co .../resolve/... or .../blob/... link. Or instead:
        repo_id: e.g. "Lightricks/LTX-2.5", with
        path: the file's path in the repo, e.g. "text_encoders/gemma4-12b-....safetensors".
    """
    if url:
        m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/[^/]+/(.+?)(?:\?.*)?$", url.strip())
        if not m:
            return _out({"ok": False, "error": "Not a huggingface.co file link (…/resolve/main/<path>)."})
        repo_id, path = m.group(1), m.group(2)
    repo_id, path = _repo_from(repo_id), str(path or "").strip("/")
    if not repo_id or not path:
        return _out({"ok": False, "error": "Give a url, or repo_id and path."})
    name = path.rsplit("/", 1)[-1]
    try:
        resp = _head(repo_id, path)
    except Exception as exc:  # noqa: BLE001
        return _out({"ok": False, "error": f"Could not reach Hugging Face: {exc}"})
    result = {"ok": True, "repo_id": repo_id, "path": path,
              "url": f"{HF}/{repo_id}/resolve/main/{path}", **_access(resp, repo_id)}
    size = resp.headers.get("X-Linked-Size") or (resp.headers.get("Content-Length")
                                                 if resp.status_code == 200 else None)
    if size:
        result["size_gb"] = _gb(size)
    try:
        dest, _src = _resolve_download_dir("", "", filename=name,
                                           hf_subfolder=path.rsplit("/", 1)[0] if "/" in path else "")
        result["would_download_to"] = str(dest)
    except Exception:  # noqa: BLE001
        pass
    local = _local_files(name, exact=True)
    if local:
        result["already_installed"] = local[:3]
    if result["access"] in ("gated", "missing"):
        mirrors = _mirrors(name, repo_id)
        if mirrors:
            result["same_file_elsewhere"] = mirrors
    if result["access"] == "missing":
        result["hint"] = f"Not in {repo_id}. hf_repo(\"{repo_id}\") lists what is there."
    return _out(result)


def _mirrors(name: str, repo_id: str) -> list[dict]:
    """Other repos with a file of exactly this name, ungated ones first.

    The Hub's search matches repo NAMES, so a file name finds nothing: search the
    family the file belongs to — the repo's own name ("LTX-2.5") and the first
    words of the file name ("gemma4-12b") — and look through those repos' files.
    """
    stem = name.rsplit(".", 1)[0]
    words = [w for w in re.split(r"[-_.]", stem) if w]
    queries = [repo_id.split("/", 1)[-1], "-".join(words[:2]), "-".join(words[:3])]
    found, seen = [], {repo_id}
    for q in dict.fromkeys(x for x in queries if x):
        try:
            r = requests.get(HF_API_BASE, headers=_hf_headers(), timeout=30,
                             params={"search": q, "full": "true", "limit": 50})
            r.raise_for_status()
            repos = r.json()
        except Exception:  # noqa: BLE001
            continue
        for m in repos:
            rid = m.get("id") or m.get("modelId")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            for sib in m.get("siblings") or []:
                rf = sib.get("rfilename", "")
                if rf.rsplit("/", 1)[-1] == name:
                    found.append({"repo_id": rid, "path": rf, "gated": m.get("gated") or False,
                                  "downloads": m.get("downloads", 0),
                                  "url": f"{HF}/{rid}/resolve/main/{rf}"})
                    break
        if len(found) >= 5:
            break
    found.sort(key=lambda f: (bool(f["gated"]), -int(f.get("downloads") or 0)))
    return found[:5]


# ── this machine ─────────────────────────────────────────────────────────────

def _model_roots() -> list[tuple[str, Path]]:
    """(category, folder) for every folder ComfyUI loads models from."""
    roots: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for cat, paths in (_folder_paths() or {}).items():
        if cat in ("custom_nodes", "configs"):
            continue
        for p in paths:
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen and os.path.isdir(p):
                seen.add(key)
                roots.append((cat, Path(p)))
    if not roots:
        base = _models_base_dir()
        if base.is_dir():
            roots = [(d.name, d) for d in base.iterdir() if d.is_dir()]
    return roots


def _local_index() -> list[tuple[str, str, str]]:
    """Every model file ComfyUI can load: (category, name in its folder, full path)."""
    out = []
    for cat, root in _model_roots():
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.lower().endswith(MODEL_EXTENSIONS):
                    full = os.path.join(dirpath, f)
                    out.append((cat, os.path.relpath(full, root).replace("\\", "/"),
                                full.replace("\\", "/")))
    return out


def _local_files(pattern: str, *, exact: bool = False, category: str = "",
                 limit: int = 100, index: list | None = None) -> list[dict]:
    out = []
    for cat, name, full in (index if index is not None else _local_index()):
        if category and cat != category.lower():
            continue
        base = name.rsplit("/", 1)[-1]
        if exact and base != pattern:
            continue
        if not exact and not _matches(name, pattern):
            continue
        try:
            size = os.path.getsize(full)
        except OSError:
            size = None
        out.append({"name": name, "category": cat, "path": full, "size_gb": _gb(size)})
        if len(out) >= limit:
            break
    return out


@tool
def find_local_models(pattern: str, category: str = "") -> str:
    """Which model files this ComfyUI already has, by part of the name — every
    model folder it loads from, extra_model_paths included.

    Args:
        pattern: Substring or glob of the file name or its path in the folder
            ("ltx-2.5", "*gemma*", "loras/*wan*").
        category: Only this folder kind ("loras", "vae", "text_encoders",
            "diffusion_models", "checkpoints", …).

    Returns each file's `name` as a loader node expects it, its category and size.
    """
    if not str(pattern or "").strip():
        return _out({"ok": False, "error": "Give part of a file name, e.g. 'ltx-2.5' or '*gemma*'."})
    found = _local_files(pattern, category=category)
    result = {"ok": True, "count": len(found), "files": found}
    if not found:
        result["hint"] = ("Not installed. hf_search(file_pattern=…) finds it on Hugging Face; "
                          "download_hf_model puts it in the right folder.")
    return _out(result)


# ── a workflow file ──────────────────────────────────────────────────────────

_INPUT_LOADERS = ("LoadImage", "LoadVideo", "LoadAudio", "VHS_LoadVideo", "LoadImageMask",
                  "VHS_LoadAudio", "LoadVideoUpload")


def _is_model_value(v) -> bool:
    return isinstance(v, str) and v.lower().endswith(MODEL_EXTENSIONS)


@tool
def inspect_workflow_file(path: str) -> str:
    """Read any ComfyUI workflow JSON (an example workflow, a template, a saved
    one; UI or API format) and say what it does: its nodes — those inside
    subgraphs too — the models each loader loads, the input files, the outputs,
    and the text of its notes.

    Args:
        path: The .json file.
    """
    p = Path(str(path or "").strip().strip('"'))
    if not p.is_file():
        return _out({"ok": False, "error": f"No such file: {p}"})
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return _out({"ok": False, "error": f"Not a JSON workflow: {exc}"})

    nodes: list[dict] = []        # {type, title, values, where}
    subgraphs: list[dict] = []
    declared: list[dict] = []     # properties.models: the author's own links

    def take(node: dict, where: str) -> None:
        props = node.get("properties") or {}
        for m in props.get("models") or []:
            if isinstance(m, dict) and m.get("name"):
                declared.append({"name": m.get("name"), "url": m.get("url", ""),
                                 "directory": m.get("directory", "")})
        nodes.append({"type": node.get("type") or node.get("class_type") or "?",
                      "title": node.get("title") or "",
                      "values": node.get("widgets_values") if "widgets_values" in node
                      else node.get("inputs"), "where": where})

    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        defs = (data.get("definitions") or {}).get("subgraphs") or []
        sub_names = {s.get("id"): s.get("name") or s.get("id") for s in defs}
        for n in data["nodes"]:
            take(n, "")
        for s in defs:
            inner = s.get("nodes") or []
            subgraphs.append({"name": s.get("name") or s.get("id"), "nodes": len(inner)})
            for n in inner:
                take(n, s.get("name") or s.get("id"))
        # A node whose type is a subgraph id is that subgraph placed in the graph.
        for n in nodes:
            if n["type"] in sub_names:
                n["type"] = f"subgraph:{sub_names[n['type']]}"
    elif isinstance(data, dict):            # API format: {"id": {"class_type", "inputs"}}
        for n in data.values():
            if isinstance(n, dict) and "class_type" in n:
                take(n, "")
    else:
        return _out({"ok": False, "error": "Not a ComfyUI workflow (no nodes)."})

    def values(n) -> list:
        v = n["values"]
        if isinstance(v, dict):
            return [x for x in v.values() if not isinstance(x, list)]
        return v if isinstance(v, list) else []

    models, inputs, outputs, notes = [], [], [], []
    seen = set()
    # The loaders themselves first; a subgraph placed in the graph repeats the
    # values it exposes, so it only adds a model no loader inside named.
    for n in sorted(nodes, key=lambda n: n["type"].startswith("subgraph:")):
        t = n["type"]
        for v in values(n):
            if _is_model_value(v) and v not in seen:
                seen.add(v)
                models.append({"node": t, "file": v, "in": n["where"] or "main graph"})
    for n in nodes:
        t = n["type"]
        if t in _INPUT_LOADERS or t.startswith("Load") and any(
                isinstance(v, str) and re.search(r"\.(png|jpe?g|webp|exr|mp4|mov|webm|wav|mp3|flac)$", v, re.I)
                for v in values(n)):
            inputs.append({"node": t, "value": next((v for v in values(n) if isinstance(v, str) and v), "")})
        if t.startswith(("Save", "Preview", "VHS_VideoCombine", "CreateVideo")) or "Save" in t:
            outputs.append({"node": t, "in": n["where"] or "main graph"})
        if t in ("MarkdownNote", "Note") and values(n):
            text = str(values(n)[0]).strip()
            if text:
                notes.append(text[:700])
    have = {name.rsplit("/", 1)[-1] for _c, name, _f in _local_index()} if models else set()
    for m in models:
        m["installed"] = m["file"].replace("\\", "/").rsplit("/", 1)[-1] in have
    counts: dict[str, int] = {}
    for n in nodes:
        if n["type"] not in ("MarkdownNote", "Note", "Reroute"):
            counts[n["type"]] = counts.get(n["type"], 0) + 1
    return _out({
        "ok": True, "file": str(p), "node_count": sum(counts.values()),
        "node_types": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:60]),
        "subgraphs": subgraphs, "models": models[:60],
        "declared_downloads": declared[:30], "inputs": inputs[:20], "outputs": outputs[:20],
        "notes": notes[:6],
    })
