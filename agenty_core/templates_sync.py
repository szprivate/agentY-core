"""Keep the official template corpus in step with Comfy-Org/workflow_templates.

Two mirrors live under ``comfyui_workflow_templates_official/``:

    blueprints/   <- upstream blueprints/  (headless subgraphs, one per capability)
    templates/    <- upstream templates/   (runnable graphs, incl. the partner-API set)

Both are byte-exact mirrors, so syncing is an overwrite, never a merge — there is
nothing local to preserve inside them. The one exception is name collisions with
the *custom* folder: several upstream templates have a locally modified twin
there, and ``_fetch_template`` resolves custom first, so mirroring the upstream
copy as well would put two workflows with one name into the corpus. Those names
are skipped and reported instead.

Only ``.json`` is mirrored. Upstream also ships per-template thumbnails (~770
webp/mp4 files); nothing here renders them, and they are what makes the upstream
repo 3.8 GB.

Drift is detected by git blob SHA over LF-normalised bytes — comparing raw bytes
reports every file as changed on a ``core.autocrlf=true`` checkout, because the
working tree is CRLF while upstream blobs are LF.

    python -m agenty_core.templates_sync              # report drift, write nothing
    python -m agenty_core.templates_sync --apply      # write it
    python -m agenty_core.templates_sync --from-comfyui http://127.0.0.1:8188

``--from-comfyui`` mirrors ``templates/`` from what that ComfyUI install serves
instead of from GitHub — see :func:`sync_from_comfyui`. agentY runs it on startup.

A local ComfyUI install usually already carries most of the template JSONs (the
``comfyui_workflow_templates_json`` wheel). Files whose SHA matches upstream are
copied from there instead of downloaded; the SHA check means a seeded file is
provably identical to the one the download would have produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from agenty_core.paths import corpus_root

REPO = "Comfy-Org/workflow_templates"
BRANCH = "main"
API = f"https://api.github.com/repos/{REPO}"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
MIRRORS = ("blueprints", "templates")


def _get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "agenty-core-templates-sync",
    })
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def blob_sha(data: bytes) -> str:
    """Git blob SHA of *data*, newline-normalised the way git stores it."""
    norm = data.replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(norm) + norm).hexdigest()  # noqa: S324


def upstream_index(log=print) -> Dict[str, Dict[str, str]]:
    """{mirror: {filename: blob_sha}} for the .json files in each upstream folder."""
    root = _get_json(f"{API}/git/trees/{BRANCH}")
    by_path = {e["path"]: e for e in root["tree"]}
    out: Dict[str, Dict[str, str]] = {}
    for m in MIRRORS:
        if m not in by_path:
            raise SystemExit(f"upstream has no '{m}/' folder — repo layout changed")
        tree = _get_json(f"{API}/git/trees/{by_path[m]['sha']}")
        if tree.get("truncated"):
            raise SystemExit(f"upstream '{m}/' tree came back truncated; cannot sync safely")
        out[m] = {e["path"]: e["sha"] for e in tree["tree"]
                  if e["type"] == "blob" and e["path"].endswith(".json")}
        log(f"[upstream] {m}/: {len(out[m])} json files")
    return out


def local_index(base: Path) -> Dict[str, str]:
    if not base.is_dir():
        return {}
    return {p.name: blob_sha(p.read_bytes()) for p in base.glob("*.json")}


def custom_names(log=print, root: Path | None = None) -> set:
    """Template names that already exist in the custom folder (they win)."""
    d = (Path(root) if root else corpus_root()) / "comfyui_workflow_templates_custom" / "templates"
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob("*.json") if p.name != "index.json"}


def seed_pool(extra: Iterable[Path] = ()) -> Dict[str, Path]:
    """{filename: path} of template JSONs already on disk, from an installed
    ComfyUI templates wheel. Found automatically when this runs inside ComfyUI's
    environment; otherwise point ``--seed-from`` at
    ``<ComfyUI>/.venv/Lib/site-packages/comfyui_workflow_templates_json/templates``.
    Seeding is only an accelerator: content is SHA-verified before use, so a
    seeded file is provably identical to what the download would have produced."""
    pool: Dict[str, Path] = {}
    roots: List[Path] = [Path(p) for p in extra]
    for p in sys.path:
        cand = Path(p) / "comfyui_workflow_templates_json" / "templates"
        if cand.is_dir():
            roots.append(cand)
    for r in roots:
        for f in r.glob("*.json"):
            pool.setdefault(f.name, f)
    return pool


def plan(mirror: str, up: Dict[str, str], base: Path, skip: set) -> Tuple[list, list, list, list]:
    """(add, change, delete, skipped) filenames for one mirror."""
    loc = local_index(base)
    skipped = sorted(f for f in up
                     if mirror == "templates" and Path(f).stem in skip)
    want = {f: s for f, s in up.items() if f not in skipped}
    add = sorted(f for f in want if f not in loc)
    change = sorted(f for f in want if f in loc and loc[f] != want[f])
    delete = sorted(f for f in loc if f not in want)
    return add, change, delete, skipped


def fetch(mirror: str, name: str, want_sha: str, pool: Dict[str, Path], log=print) -> bytes:
    src = pool.get(name)
    if src is not None:
        data = src.read_bytes().replace(b"\r\n", b"\n")
        if blob_sha(data) == want_sha:
            return data
    url = f"{RAW}/{mirror}/{urllib.parse.quote(name)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    got = blob_sha(data)
    if got != want_sha:
        raise SystemExit(f"{mirror}/{name}: downloaded content does not match the "
                         f"listed sha ({got[:8]} != {want_sha[:8]})")
    return data


# ── From a running ComfyUI ──────────────────────────────────────────────────────
#
# GitHub's main branch is not the corpus anyone can run. A template published
# there today can need a node the local ComfyUI does not have yet, and a template
# the local ComfyUI already ships can be missing from a mirror synced before it
# existed — MiniMax H3's local reference-to-video was, and with nothing to start
# from the agent built that graph from scratch and guessed its sampler. The
# templates an install serves are by construction the ones written for the nodes
# it has, so that is what templates/ follows. blueprints/ is not served that way
# and keeps its GitHub sync.

SYNC_STATE = Path("config") / "official_templates_sync.json"
# A ComfyUI serves some 500 templates. An answer far short of that is a broken or
# foreign endpoint, and mirroring it would delete most of the corpus.
MIN_SERVED_TEMPLATES = 50

_sync_lock = threading.Lock()


class ComfyUIUnreachable(RuntimeError):
    """ComfyUI did not answer at all; nothing was read or written."""


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "agenty-core-templates-sync"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _read_state(root: Path) -> dict:
    try:
        return json.loads((root / SYNC_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sync_from_comfyui(base_url: str, *, root: Path | None = None, get=None, log=print,
                      force: bool = False, workers: int = 8) -> dict:
    """Make ``templates/`` match the templates the ComfyUI at *base_url* serves.

    Cheap when nothing changed: ComfyUI reports its templates package version in
    ``/system_stats``, and a version already mirrored returns ``{"status":
    "current"}`` after that single request. Otherwise every served template is
    fetched — all of them before anything is written, so a failure midway leaves
    the mirror exactly as it was — compared by LF-normalised blob SHA, and the
    differences written: added, changed, and removed where that ComfyUI no longer
    ships one. Names with a *custom* twin are skipped, as in the GitHub sync.

    Returns ``{status, version, added, changed, removed, skipped_custom}``.
    Raises :class:`ComfyUIUnreachable` when ComfyUI does not answer.
    """
    get = get or _http_get
    base_url = base_url.rstrip("/")
    root = Path(root) if root else corpus_root()
    mirror = root / "comfyui_workflow_templates_official" / "templates"
    with _sync_lock:
        try:
            stats = json.loads(get(f"{base_url}/system_stats"))
        except (OSError, ValueError) as exc:        # URLError is an OSError
            raise ComfyUIUnreachable(f"{base_url}: {exc}") from exc
        version = str(((stats or {}).get("system") or {}).get("installed_templates_version") or "")
        if version and not force and _read_state(root).get("templates_version") == version:
            return {"status": "current", "version": version}

        index_bytes = get(f"{base_url}/templates/index.json")
        names = sorted({t["name"] for group in json.loads(index_bytes)
                        if isinstance(group, dict)
                        for t in (group.get("templates") or [])
                        if isinstance(t, dict) and isinstance(t.get("name"), str)})
        if len(names) < MIN_SERVED_TEMPLATES:
            raise RuntimeError(f"{base_url} lists only {len(names)} templates; refusing "
                               f"to mirror that over the corpus")
        custom = custom_names(root=root)
        skipped = [n for n in names if n in custom]
        wanted = [f"{n}.json" for n in names if n not in custom]
        extra_indexes = sorted(p.name for p in mirror.glob("index*.json")
                               if p.name != "index.json")

        def _fetch(fname: str) -> bytes:
            return get(f"{base_url}/templates/{urllib.parse.quote(fname)}")

        fetched: Dict[str, bytes] = {"index.json": index_bytes}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for fname, data in zip(wanted, pool.map(_fetch, wanted)):
                fetched[fname] = data
            # Locale indexes are extras: one this ComfyUI does not serve stays as
            # it is rather than failing the sync.
            for fname, fut in [(f, pool.submit(_fetch, f)) for f in extra_indexes]:
                try:
                    fetched[fname] = fut.result()
                except (OSError, ValueError):
                    pass

        local = local_index(mirror)
        added = sorted(f for f in fetched if f not in local)
        changed = sorted(f for f in fetched if f in local and local[f] != blob_sha(fetched[f]))
        removed = sorted(f for f in local
                         if f not in fetched and not f.startswith("index"))
        mirror.mkdir(parents=True, exist_ok=True)
        for f in added + changed:
            (mirror / f).write_bytes(fetched[f].replace(b"\r\n", b"\n"))
        for f in removed:
            (mirror / f).unlink()
        state = root / SYNC_STATE
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"templates_version": version, "templates": len(names),
                                     "skipped_custom": len(skipped)}, indent=2) + "\n",
                         encoding="utf-8")
        log(f"[comfyui] templates {version or '?'}: +{len(added)} ~{len(changed)} "
            f"-{len(removed)} ({len(skipped)} skipped: custom twin)")
        return {"status": "synced", "version": version, "added": added,
                "changed": changed, "removed": removed, "skipped_custom": skipped}


# ── Example workflows shipped with custom node packs ───────────────────────────
#
# A node pack documents itself with example workflows: a folder in the pack named
# ``example_workflows`` (ComfyUI's convention; ``workflow``, ``workflows``,
# ``example`` and ``examples`` are accepted too). ComfyUI scans its custom_nodes
# folders for exactly these, lists them at ``/api/workflow_templates`` and serves
# the files beside it — that is what its own template browser shows per pack.
#
# For a pack's nodes these are the only templates there are: the official corpus
# covers core and partner nodes, so a request needing WanVideoWrapper or SAM3 had
# nothing to start from and was built from node schemas alone.
#
# They are mirrored into ``templates/node_packs/`` inside the custom corpus, so
# everything that reads custom templates finds them, and that folder is
# git-ignored: what is installed is a fact about this machine, and the files are
# other people's work. Named ``<pack>__<workflow>``, because packs reuse names
# ("basic", "example") and a template is looked up by file name alone.

NODE_PACK_DIR = Path("comfyui_workflow_templates_custom") / "templates" / "node_packs"
EXAMPLE_FOLDERS = ("example_workflows", "example", "examples", "workflow", "workflows")


def _safe(text: str) -> str:
    import re  # noqa: PLC0415
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._-")


def node_pack_template_name(pack: str, workflow: str) -> str:
    """The template name of *workflow* from node pack *pack*."""
    return f"{_safe(pack)}__{_safe(workflow)}"


def _is_workflow(data) -> bool:
    """A ComfyUI graph or API prompt — packs also keep settings JSON in these folders."""
    if not isinstance(data, dict) or not data:
        return False
    if isinstance(data.get("nodes"), list):
        return True
    return all(isinstance(v, dict) and "class_type" in v for v in data.values())


def _index_entry(name: str, pack: str, workflow: str, data: dict) -> dict:
    """The catalog entry of one example: what it loads and saves, and whose it is."""
    entry = {"name": name, "title": workflow,
             "description": f"Example workflow \"{workflow}\" shipped with the {pack} "
                            f"custom node pack — how its nodes are meant to be wired.",
             "models": [], "io": {"inputs": [], "outputs": []}}
    try:
        from agenty_core.utils.workflow_parser import parse_workflow  # noqa: PLC0415
        api = data
        if isinstance(data.get("nodes"), list):
            from agenty_core.tools.comfyui import _convert_graph_to_api  # noqa: PLC0415
            api = _convert_graph_to_api(data)
        parsed = parse_workflow(api, name=name, update_index=False)["templates"][0]
        entry["models"], entry["io"] = parsed.get("models", []), parsed.get("io", entry["io"])
    except Exception:  # noqa: BLE001 — a graph the parser cannot read is still a usable template
        pass
    return entry


def sync_node_pack_examples(base_url: str, *, root: Path | None = None, get=None,
                            log=print, workers: int = 8, describe=_index_entry) -> dict:
    """Mirror the example workflows of the node packs installed in the ComfyUI at
    *base_url* into ``templates/node_packs/`` and write that folder's index.json.

    A pure mirror: an example a pack no longer ships (or a pack that was removed)
    goes, a changed one is replaced. Files that are not workflows, and files the
    ComfyUI does not serve (a pack that failed to load), are skipped.

    Returns ``{status, packs, added, changed, removed, skipped}``; raises
    :class:`ComfyUIUnreachable` when ComfyUI does not answer.
    """
    get = get or _http_get
    base_url = base_url.rstrip("/")
    root = Path(root) if root else corpus_root()
    mirror = root / NODE_PACK_DIR
    with _sync_lock:
        try:
            listing = json.loads(get(f"{base_url}/api/workflow_templates"))
        except (OSError, ValueError) as exc:
            raise ComfyUIUnreachable(f"{base_url}: {exc}") from exc
        if not isinstance(listing, dict):
            raise RuntimeError(f"{base_url}/api/workflow_templates did not return a map")
        wanted = [(str(pack), str(wf)) for pack, names in sorted(listing.items())
                  if isinstance(names, list) for wf in sorted(set(names))]

        def _fetch(item):
            pack, wf = item
            try:
                raw = get(f"{base_url}/api/workflow_templates/"
                          f"{urllib.parse.quote(pack)}/{urllib.parse.quote(wf)}.json")
                data = json.loads(raw.decode("utf-8-sig"))
            except (OSError, ValueError):
                return None
            return (raw, data) if _is_workflow(data) else None

        fetched: Dict[str, bytes] = {}
        entries: Dict[str, List[dict]] = {}
        skipped: List[str] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for (pack, wf), got in zip(wanted, pool.map(_fetch, wanted)):
                name = node_pack_template_name(pack, wf)
                if got is None or not name or f"{name}.json" in fetched:
                    skipped.append(f"{pack}/{wf}")
                    continue
                fetched[f"{name}.json"] = got[0].replace(b"\r\n", b"\n")
                entries.setdefault(pack, []).append((name, wf, got[1]))

        local = {f: s for f, s in local_index(mirror).items() if f != "index.json"}
        added = sorted(f for f in fetched if f not in local)
        changed = sorted(f for f in fetched if f in local and local[f] != blob_sha(fetched[f]))
        removed = sorted(f for f in local if f not in fetched)
        if not (added or changed or removed) and (mirror / "index.json").is_file():
            return {"status": "current", "packs": len(entries), "added": [], "changed": [],
                    "removed": [], "skipped": skipped}
        mirror.mkdir(parents=True, exist_ok=True)
        for f in added + changed:
            (mirror / f).write_bytes(fetched[f])
        for f in removed:
            (mirror / f).unlink()
        index = [{"moduleName": pack, "title": f"{pack} (node pack examples)",
                  "templates": [describe(name, pack, wf, data) for name, wf, data in items]}
                 for pack, items in sorted(entries.items())]
        tmp = mirror / "index.json.tmp"
        tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, mirror / "index.json")
        log(f"[comfyui] node pack examples from {len(entries)} pack(s): "
            f"+{len(added)} ~{len(changed)} -{len(removed)}")
        return {"status": "synced", "packs": len(entries), "added": added, "changed": changed,
                "removed": removed, "skipped": skipped}


def _drop_template_caches() -> None:
    """Forget what this process read from the old mirror."""
    try:
        from agenty_core.tools import comfyui as C  # noqa: PLC0415
    except Exception:  # noqa: BLE001 — nothing loaded, nothing to forget
        return
    C._index_cache = None
    C._official_index_cache = None
    C._template_cache.clear()
    C.clear_tool_caches()


def _regenerate_recipes(root: Path) -> dict:
    from agenty_core.workflow_recipes.cli import build_arg_parser, run  # noqa: PLC0415
    args = build_arg_parser().parse_args([
        "--no-fetch",
        "--custom-folder", str(root / "comfyui_workflow_templates_custom"),
        "--official-folder", str(root / "comfyui_workflow_templates_official"),
        "--out", str(root / "config" / "workflow_recipes.json"),
        "--object-info-cache", str(root / "config" / "workflow_recipes_object_info_cache.json"),
    ])
    db = (run(args) or {}).get("database")
    if db is None:
        return {}
    return {"workflow_count": db.workflow_count, "task_count": len(db.tasks),
            "recipe_count": db.recipe_count}


def refresh_node_pack_examples(base_url: str, *, root: Path | None = None, get=None,
                               log=print, regenerate=None) -> dict:
    """:func:`sync_node_pack_examples`, then — only when an example changed — the
    recipe database is rebuilt and this process's template caches are dropped."""
    result = sync_node_pack_examples(base_url, root=root, get=get, log=log)
    if result.get("status") != "synced":
        return result
    root = Path(root) if root else corpus_root()
    try:
        result["recipes"] = (regenerate or (lambda: _regenerate_recipes(root)))()
    except Exception as exc:  # noqa: BLE001 — the mirror is right even if the DB is not
        result["recipes"] = {"error": str(exc)}
    _drop_template_caches()
    return result


def refresh_from_comfyui(base_url: str, *, root: Path | None = None, get=None,
                         log=print, regenerate=None) -> dict:
    """:func:`sync_from_comfyui`, then bring everything that reads the mirror up to date.

    Only when a template actually changed: the node-schema cache the recipe
    generator reads is refreshed from the same ComfyUI (new templates use nodes the
    old cache has no signature for — and a failed fetch keeps the old cache rather
    than building with none), the recipe database is rebuilt, and this process's
    template caches are dropped so the next lookup reads the new files.

    The sync result, plus ``recipes``: the rebuild's counts, or ``{"error": ...}``.
    """
    result = sync_from_comfyui(base_url, root=root, get=get, log=log)
    if result.get("status") != "synced" or not (
            result["added"] or result["changed"] or result["removed"]):
        return result
    root = Path(root) if root else corpus_root()
    get = get or _http_get
    cache = root / "config" / "workflow_recipes_object_info_cache.json"
    try:
        data = get(f"{base_url.rstrip('/')}/object_info")
        json.loads(data)
        tmp = cache.with_name(cache.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cache)
    except (OSError, ValueError) as exc:
        log(f"[comfyui] node schemas not refreshed ({exc}); rebuilding with the cached ones")
    if regenerate is None:
        def regenerate() -> dict:
            from agenty_core.workflow_recipes.cli import build_arg_parser, run  # noqa: PLC0415
            args = build_arg_parser().parse_args([
                "--no-fetch",
                "--custom-folder", str(root / "comfyui_workflow_templates_custom"),
                "--official-folder", str(root / "comfyui_workflow_templates_official"),
                "--out", str(root / "config" / "workflow_recipes.json"),
                "--object-info-cache", str(cache),
            ])
            db = (run(args) or {}).get("database")
            if db is None:
                return {}
            return {"workflow_count": db.workflow_count, "task_count": len(db.tasks),
                    "recipe_count": db.recipe_count}
    try:
        result["recipes"] = regenerate()
    except Exception as exc:  # noqa: BLE001 — the mirror is right even if the DB is not
        result["recipes"] = {"error": str(exc)}
    _drop_template_caches()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="templates_sync",
        description="Mirror the official ComfyUI workflow templates into the corpus.")
    ap.add_argument("--apply", action="store_true",
                    help="write the changes (default: report only)")
    ap.add_argument("--seed-from", default=None,
                    help="extra directory of template JSONs to seed from (content is "
                         "SHA-verified before use)")
    ap.add_argument("--root", default=None, help="corpus root override")
    ap.add_argument("--from-comfyui", default=None, metavar="URL",
                    help="mirror templates/ from the ComfyUI at URL instead of GitHub, "
                         "then rebuild the recipe DB (always writes)")
    args = ap.parse_args(argv)

    if args.from_comfyui:
        res = refresh_from_comfyui(args.from_comfyui,
                                   root=Path(args.root).resolve() if args.root else None)
        print(json.dumps({k: (len(v) if isinstance(v, list) else v)
                          for k, v in res.items()}, indent=2))
        return 0

    root = Path(args.root).resolve() if args.root else corpus_root()
    official = root / "comfyui_workflow_templates_official"
    print(f"[corpus] {official}")

    up = upstream_index()
    skip = custom_names()
    pool = seed_pool([args.seed_from] if args.seed_from else [])
    print(f"[seed] {len(pool)} template json files available locally")

    total = 0
    for mirror in MIRRORS:
        base = official / mirror
        add, change, delete, skipped = plan(mirror, up[mirror], base, skip)
        total += len(add) + len(change) + len(delete)
        print(f"\n[{mirror}] +{len(add)} add  ~{len(change)} change  "
              f"-{len(delete)} delete  ({len(skipped)} skipped: custom twin)")
        for tag, names in (("add", add), ("change", change), ("delete", delete)):
            for n in names[:10]:
                print(f"    {tag:6} {n}")
            if len(names) > 10:
                print(f"    {tag:6} ... and {len(names) - 10} more")
        if not args.apply:
            continue
        base.mkdir(parents=True, exist_ok=True)
        seeded = 0
        for n in add + change:
            data = fetch(mirror, n, up[mirror][n], pool)
            if n in pool:
                seeded += 1
            (base / n).write_bytes(data)
        for n in delete:
            (base / n).unlink()
        print(f"    wrote {len(add) + len(change)} files ({seeded} seeded locally), "
              f"removed {len(delete)}")

    if not args.apply:
        print(f"\n{total} file(s) would change. Re-run with --apply to write them.")
    else:
        print("\nNow regenerate the recipe DB:")
        print("    python -m agenty_core.workflow_recipes.cli --no-fetch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
