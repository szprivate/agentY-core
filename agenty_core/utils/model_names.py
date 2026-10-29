"""Which models a request names - matched against the model names the corpus itself uses.

Two guesses this replaces, both wrong in the benchmark of night_20261029:

* **Pinning by template-name words.** A template counted as "named" when all the
  leftover words of its FILE name appeared in the request, so "Qwen Image Edit
  2511 ... 8 steps" pinned ``image_qwen_image_2512_with_2steps_lora`` (EX04).
* **Versions read off folder and file names.** "LTX2/ltx-2.5-..." read as LTX 2.0
  (CX15), and ``ltxv_text_to_video`` passed for an "LTX-2" request (CX10).

The vocabulary here is the one the recipe system already has, per template:

* the model names its catalog entry states (``index.json`` ``models``:
  "Qwen Image Edit 2511", "LTX-2.5", "LTX-0.9.5", "Wan2.2" ...), minus the ones
  that are a VAE / text encoder / LoRA file of that template;
* the main model files it loads (``diffusion_models`` / ``checkpoints`` /
  ``unet``), by file name - never the folder they sit in;
* the recipe it belongs to (``config/workflow_recipes.json``): its task and its
  model family ("LTX-2", "WAN 2.2", "Qwen Image"). The family stands in as a name
  only for a template that states no other.

Both sides are spelled the same way by :func:`tokens` (case, ``-``/``_``/``.``/
space, ``v`` prefixes, "ltx2.5" / "wan2_2" / "sd15" joins). A name matches when
its tokens appear *contiguously* in the other side's tokens, and a version is one
token ("2.5", "2511", "0.9.5"), so versions only ever match exactly: LTX 2 is not
LTX 2.5, 2511 is not 2512. Numbers that are not part of a model name ("8 steps",
"CFG 1", "1024x1024") are never read.

Pure apart from reading the corpus; nothing here calls ComfyUI.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable

# The folders a graph's MAIN model comes from. VAEs, text encoders and LoRAs are
# chosen to go with it, and their names carry other versions ("wan_2.1_vae" in a
# Wan 2.2 graph, "qwen_2.5_vl" in every Qwen-Image graph).
_MAIN_DIRS = {"diffusion_models", "checkpoints", "unet", "unet_gguf"}
_MAIN_INPUTS = {"ckpt_name", "unet_name"}
_MAIN_LOADERS = {"CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader", "UnetLoaderGGUF",
                 "ImageOnlyCheckpointLoader", "DiffusionModelLoaderKJ", "LoaderGGUF"}
_MODEL_EXT = re.compile(r"\.(safetensors|sft|ckpt|pt|pth|bin|gguf)$", re.I)

# Sampler settings. A model name that starts with one ("Seed 2.0") is not told
# apart from a setting ("seed 2"), so it never counts as a name.
_SETTING_WORDS = {"seed", "steps", "step", "cfg", "denoise", "sampler", "scheduler", "frames", "fps"}
# What a number counts when it is not a version: "wan 8 steps", "flux 4 images".
_UNIT_WORDS = _SETTING_WORDS | {"frame", "second", "seconds", "sec", "s", "px", "x", "image", "images",
                                "shots", "shot", "panels", "clips", "variations", "versions", "times",
                                "minutes", "min", "loras", "pass", "passes"}
# Precision / format prefixes: "fp8" and "bf16" are not "fp 8".
_NO_SPLIT = {"fp", "bf", "int", "uint", "nf", "mp", "rank", "e"}
# The kinds of file that go WITH a model. A catalog entry sometimes lists its VAE
# or text encoder among its models ("wan_2.1_vae" on a Wan 2.2 graph); that is not
# the model the graph is, and its version is not the graph's.
_COMPONENT_WORDS = {"vae", "vl", "clip", "umt", "t5xxl", "encoder", "lora", "loras", "upscaler", "embeddings"}
# A name that is a null placeholder in the catalog, not a model.
_NOT_A_NAME = {"none", "null", "n/a", ""}

_TASK_MEDIA = {
    "text_to_image": "image", "image_edit": "image", "inpaint_outpaint": "image",
    "image_edit_with_controlnet": "image", "image_tools": "image", "character": "image",
    "text_to_video": "video", "image_to_video": "video", "video_to_video": "video",
    "first_last_frame_to_video": "video", "video_inpaint": "video", "video_tools": "video",
    "3d": "3d", "audio": "audio", "upscale": None, "preprocessors_estimation": None,
    "text_tools": "text",
}
# What a request's task word accepts of the recipe tasks.
_TASK_FITS = {
    "first_last_frame_to_video": {"first_last_frame_to_video"},
    "image_to_video": {"image_to_video"},
    "video_to_video": {"video_to_video", "video_inpaint"},
    "text_to_video": {"text_to_video"},
    "inpaint_outpaint": {"inpaint_outpaint", "image_edit"},
    "image_edit": {"image_edit", "image_edit_with_controlnet", "inpaint_outpaint", "character"},
    "text_to_image": {"text_to_image"},
    "upscale": {"upscale"},
    "image_edit_with_controlnet": {"image_edit_with_controlnet"},
}
# Request wording -> task, first match wins (specific before general: "upscaler"
# in a text-to-video request is a stage, not the job).
_TASK_WORDS: list[tuple[str, re.Pattern]] = [
    ("first_last_frame_to_video", re.compile(
        r"\bflf2?v?\b|first[\s/-]*(?:and[\s-]*|&[\s-]*)?last[\s-]*frames?|start[\s-]*(?:and|&|/)[\s-]*end[\s-]*frames?")),
    ("image_to_video", re.compile(
        r"image[\s-]*to[\s-]*video|\bi2v\b|img2vid|animate (?:this|the|my|an?|each|these) "
        r"(?:image|photo|picture|still|frame)s?|(?:image|photo|picture|still) (?:is|as) the first frame|"
        r"the first frame")),
    ("video_to_video", re.compile(r"video[\s-]*to[\s-]*video|\bv2v\b|vid2vid|video[\s-]*edit")),
    ("text_to_video", re.compile(r"text[\s-]*to[\s-]*video|\bt2v\b")),
    ("inpaint_outpaint", re.compile(r"\b(?:in|out)paint")),
    ("image_edit_with_controlnet", re.compile(r"control[\s-]?net")),
    ("image_edit", re.compile(r"\bedit(?:s|ed|ing)?\b|\bi2i\b|img2img|image[\s-]*to[\s-]*image")),
    ("text_to_image", re.compile(r"text[\s-]*to[\s-]*image|\bt2i\b")),
    ("upscale", re.compile(r"\bupscal")),
]


# ---------------------------------------------------------------------------
# spelling
# ---------------------------------------------------------------------------

def _strip_folders(text: str) -> str:
    """*text* without the folder part of paths: "LTX2/ltx-2.5-22b.safetensors" is
    the file "ltx-2.5-22b.safetensors"; the folder "LTX2" names nothing."""
    return re.sub(r"[\w.\-]+[\\/]+(?=[\w.\-])", " ", str(text or ""))


def _norm_version(tok: str) -> str:
    """"2.0" -> "2", "1.0" -> "1": a whole-number release is written both ways."""
    if re.fullmatch(r"\d+(?:\.\d+)+", tok):
        while tok.endswith(".0"):
            tok = tok[:-2]
    return tok


def tokens(text: str) -> list[str]:
    """The words of *text*, spelled the one way model names and requests are compared.

    ``"Qwen-Image-Edit 2511"`` -> ``["qwen", "image", "edit", "2511"]``;
    ``"ltx2.5"``, ``"LTX 2.5"``, ``"video_ltx2_5"`` -> ``[..., "ltx", "2.5"]``;
    ``"Wan2.2"``, ``"wan 2.2"``, ``"wan22"`` -> ``["wan", "2.2"]``; ``"LTX-2"`` ->
    ``["ltx", "2"]``; ``"v1.0"`` -> ``["1"]``; ``"Stable Diffusion XL"`` -> ``["sdxl"]``.
    """
    s = _strip_folders(text).lower()
    s = re.sub(r"\.(?:safetensors|sft|ckpt|pth?|bin|gguf)\b", " ", s)
    s = re.sub(r"stable[\s_-]*diffusion[\s_-]*xl\b", "sdxl", s)
    s = re.sub(r"stable[\s_-]*diffusion", "sd", s)
    s = re.sub(r"(?<![a-z0-9])sd[\s_-]+xl(?![a-z])", "sdxl", s)
    # "sd-v1-5": the SD 1.x files write their version with a dash.
    s = re.sub(r"(?<![a-z0-9])v(\d)-(\d)(?![0-9])", r"v\1.\2", s)
    # "wan2_2", "ltx2_5", "ace_step_1_5": a version written with "_" between its parts.
    s = re.sub(r"(?<![\d.])(\d{1,2})_(\d{1,2})(?![0-9a-z])", r"\1.\2", s)
    out: list[str] = []
    for raw in re.split(r"[^a-z0-9.]+", s):
        tok = raw.strip(".")
        if not tok:
            continue
        # "flux.1", "sam.3": a dot between letters and digits is a separator.
        parts = re.split(r"(?<=[a-z])\.(?=\d)|(?<=\d)\.(?=[a-z])", tok)
        for p in parts:
            m = re.fullmatch(r"v(\d+(?:\.\d+)*)", p)
            if m:
                out.append(_norm_version(m.group(1)))
                continue
            m = re.fullmatch(r"([a-z]{2,})(\d+(?:\.\d+)*)", p)
            if m and m.group(1) not in _NO_SPLIT:
                word, num = m.group(1), m.group(2)
                if re.fullmatch(r"\d\d", num):          # "wan22", "sd15", "sd35"
                    num = num[0] + "." + num[1]
                out.extend([word, _norm_version(num)])
                continue
            out.append(_norm_version(p) if re.fullmatch(r"[\d.]+", p) else p)
    return out


def is_version(tok: str) -> bool:
    """A version token: "2", "2.5", "0.9.5", "2511" - not "14b", "720p", "4k"."""
    return bool(re.fullmatch(r"\d+(?:\.\d+)*", tok or ""))


def _contains(hay: tuple, needle: tuple) -> bool:
    n = len(needle)
    if not n or n > len(hay):
        return False
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


# ---------------------------------------------------------------------------
# the vocabulary
# ---------------------------------------------------------------------------

def main_model_files(data: Any) -> list[str]:
    """File names (no folders) of the main model(s) a graph loads (UI or API format)."""
    found: list[str] = []
    if not isinstance(data, dict):
        return found
    try:
        from agenty_core.utils.template_models import models_in_workflow
        for name, info in models_in_workflow(data).items():
            if str(info.get("directory") or "").split("/")[0] in _MAIN_DIRS:
                found.append(name)
    except Exception:  # noqa: BLE001
        pass

    def walk(nodes):
        for n in nodes or []:
            if isinstance(n, dict) and n.get("type") in _MAIN_LOADERS:
                for w in n.get("widgets_values") or []:
                    if isinstance(w, str) and _MODEL_EXT.search(w):
                        found.append(w)
                        break
    if isinstance(data.get("nodes"), list):
        walk(data.get("nodes"))
        for sg in ((data.get("definitions") or {}).get("subgraphs") or []):
            walk(sg.get("nodes"))
    else:
        for n in data.values():
            if isinstance(n, dict) and isinstance(n.get("inputs"), dict):
                for k, v in n["inputs"].items():
                    if k in _MAIN_INPUTS and isinstance(v, str):
                        found.append(v)
    out = []
    for f in found:
        base = str(f).replace("\\", "/").rsplit("/", 1)[-1].split(" #", 1)[0].strip()
        if base and base not in out:
            out.append(base)
    return out


def side_model_files(data: Any) -> set[str]:
    """Stems of the files a graph loads that are NOT its main model (VAE, encoders, LoRAs)."""
    out: set[str] = set()
    try:
        from agenty_core.utils.template_models import models_in_workflow
        for name, info in models_in_workflow(data if isinstance(data, dict) else {}).items():
            if str(info.get("directory") or "").split("/")[0] not in _MAIN_DIRS:
                out.add(" ".join(tokens(name)))
    except Exception:  # noqa: BLE001
        pass
    return out


class Vocabulary:
    """Per template: its model names (token tuples), recipe task and recipe family."""

    def __init__(self, entries: dict[str, dict]):
        # entries: {template: {"models": [str], "files": [str], "side": set[str],
        #                      "task": str|None, "family": str|None}}
        self.templates: dict[str, dict] = {}
        counts: dict[tuple, int] = {}
        for tpl, e in entries.items():
            side = set(e.get("side") or ())
            names: list[tuple] = []
            shown: dict[tuple, str] = {}
            for raw in list(e.get("models") or []) + list(e.get("files") or []):
                t = tuple(tokens(raw))
                if (not t or " ".join(t) in side or str(raw).strip().lower() in _NOT_A_NAME
                        or _COMPONENT_WORDS & set(t)):
                    continue
                # A file's model part ends at its version: "qwen_image_edit_2511_bf16"
                # names Qwen Image Edit 2511, whatever precision the file is in.
                cut = next((i for i, x in enumerate(t) if i and is_version(x)), None)
                for form in ((t[:cut + 1], t) if cut is not None and cut + 1 < len(t) else (t,)):
                    if form not in names:
                        names.append(form)
                        shown.setdefault(form, str(raw) if form == t else " ".join(form))
            fam = e.get("family")
            if fam and fam.lower() != "generic":
                ft = tuple(tokens(fam))
                if ft and not names:
                    names.append(ft)
                    shown.setdefault(ft, fam)
            self.templates[tpl] = {"names": names, "shown": shown, "task": e.get("task"),
                                   "family": fam}
            for t in set(names) | ({tuple(tokens(fam))} if fam and fam.lower() != "generic" else set()):
                counts[t] = counts.get(t, 0) + 1
        # The names a request can name a model by. A file name is long and never
        # typed; the catalog's model names and the recipe families are what people
        # say. Kept: a name with a version ("Wan2.2"), a recipe family ("Kling"),
        # or a one-word name several templates carry ("SDXL", "Seedance") - a
        # one-word name only one or two templates carry ("Lens", "FILM") is as
        # likely an ordinary word in a request.
        family_heads = set()
        for e in entries.values():
            fam = e.get("family")
            if fam and fam.lower() != "generic":
                ft = tokens(fam)
                if ft:
                    family_heads.add(ft[0])
        self.family_heads = family_heads
        vocab: dict[tuple, str] = {}
        for e in self.templates.values():
            for t in e["names"]:
                vocab.setdefault(t, e["shown"].get(t, " ".join(t)))
        for e in entries.values():
            fam = e.get("family")
            if fam and fam.lower() != "generic":
                vocab.setdefault(tuple(tokens(fam)), fam)
        self.vocab: dict[tuple, str] = {}
        for t, shown in vocab.items():
            if t[0] in _SETTING_WORDS or is_version(t[0]) or len(t) > 6:
                continue
            versioned = any(is_version(x) for x in t[1:])
            if versioned or t[0] in family_heads or len(t) > 1 or (counts.get(t, 0) >= 3 and len(t[0]) >= 4):
                self.vocab[t] = shown

    def names_of(self, template: str) -> list[tuple]:
        return list((self.templates.get(template) or {}).get("names") or [])

    def task_of(self, template: str) -> str | None:
        return (self.templates.get(template) or {}).get("task")


def _corpus_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        from agenty_core.tools import comfyui as C
        dirs = [Path(C._custom_templates_dir()), Path(C._official_templates_dir())]
    except Exception:  # noqa: BLE001
        pass
    try:
        from agenty_core.paths import corpus_root
        root = corpus_root()
        for sub in ("comfyui_workflow_templates_custom", "comfyui_workflow_templates_official"):
            p = root / sub
            if not any(p == d or p in d.parents or d in p.parents for d in dirs):
                dirs.append(p)
    except Exception:  # noqa: BLE001
        pass
    # The official corpus root holds templates/ and blueprints/; the custom dir
    # IS templates/. Walk from the corpus folder so both are covered.
    out: list[Path] = []
    for d in dirs:
        top = d.parent if d.name == "templates" else d
        if top.is_dir() and top not in out:
            out.append(top)
    return out


def _recipe_db_path() -> Path | None:
    try:
        from agenty_core.paths import corpus_root
        p = corpus_root() / "config" / "workflow_recipes.json"
        return p if p.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def recipe_index(db: dict | None) -> dict[str, dict]:
    """``{template: {"task", "family"}}`` from a recipe DB (task id without the API prefix)."""
    out: dict[str, dict] = {}
    for task in (db or {}).get("tasks") or []:
        tid = str(task.get("id") or "")
        tid = tid[len("api_partner_nodes_"):] if tid.startswith("api_partner_nodes_") else tid
        for m in task.get("models") or []:
            for name in m.get("member_files") or []:
                out.setdefault(str(name), {"task": tid, "family": m.get("model")})
    return out


def build_entries(dirs: Iterable[Path], db: dict | None) -> dict[str, dict]:
    """Vocabulary entries for every template under *dirs*, with the recipe DB's task/family."""
    recipes = recipe_index(db)
    entries: dict[str, dict] = {}
    files: dict[str, Path] = {}
    for d in dirs:
        for idx in sorted(Path(d).rglob("index.json")):
            try:
                data = json.loads(idx.read_text(encoding="utf-8-sig"))
            except Exception:  # noqa: BLE001
                continue
            for cat in data if isinstance(data, list) else []:
                if not isinstance(cat, dict):
                    continue
                for t in list(cat.get("templates") or []) + list(cat.get("blueprints") or []):
                    name = t.get("name") if isinstance(t, dict) else None
                    if name and name not in entries:
                        entries[name] = {"models": [str(m) for m in (t.get("models") or [])
                                                    if isinstance(m, str)]}
        for p in sorted(Path(d).rglob("*.json")):
            if p.name == "index.json" or p.stem in files:
                continue
            files[p.stem] = p
    for name in set(entries) | set(recipes):
        e = entries.setdefault(name, {"models": []})
        e.update(recipes.get(name) or {})
        p = files.get(name)
        if p is not None:
            try:
                wf = json.loads(p.read_text(encoding="utf-8-sig"))
                e["files"] = main_model_files(wf)
                e["side"] = side_model_files(wf)
            except Exception:  # noqa: BLE001
                pass
    return entries


_lock = threading.Lock()
_cached: tuple | None = None


_RECHECK_S = 10.0


def _stamp(paths: list[Path]) -> tuple:
    out = []
    for p in paths:
        try:
            out.append((str(p), p.stat().st_mtime))
        except OSError:
            out.append((str(p), None))
    return tuple(out)


def vocabulary() -> Vocabulary:
    """The corpus vocabulary, rebuilt when an index.json or the recipe DB changes.

    Called once per catalog line and per request, so the check is cheap: the files
    it was built from are stat'ed at most every few seconds (a template sync rewrites
    the index files and the recipe DB together)."""
    global _cached
    import time
    now = time.monotonic()
    with _lock:
        if _cached is not None:
            stamp, vocab, paths, checked = _cached
            if now - checked < _RECHECK_S:
                return vocab
            if _stamp(paths) == stamp:
                _cached = (stamp, vocab, paths, now)
                return vocab
        dirs = _corpus_dirs()
        db_path = _recipe_db_path()
        paths = [idx for d in dirs for idx in sorted(d.rglob("index.json"))]
        if db_path is not None:
            paths.append(db_path)
        stamp = _stamp(paths)
        db = None
        if db_path is not None:
            try:
                db = json.loads(db_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                db = None
        vocab = Vocabulary(build_entries(dirs, db))
        _cached = (stamp, vocab, paths, now)
        return vocab


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------

def named_models(text: str, vocab: Vocabulary | None = None) -> list[dict]:
    """The models *text* names: ``[{"name", "tokens", "family", "version"}]``.

    The longest name wins where names overlap ("Qwen Image Edit 2511" over
    "Qwen-Image"). A name without a version followed by a number that is not a
    count of something ("Wan 2.4", "LTX 3" - not "Wan 8 steps") is reported with
    that version, so a version no template has still rejects the others instead
    of reading as "any Wan".
    """
    vocab = vocab or vocabulary()
    toks = tokens(text)
    hits: list[tuple[int, int, tuple]] = []
    for name in vocab.vocab:
        n = len(name)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) == name:
                hits.append((i, i + n, name))
    # keep the hits no longer hit covers
    keep = [h for h in hits
            if not any(o is not h and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0])
                       for o in hits)]
    out: list[dict] = []
    seen: set = set()
    for start, end, name in sorted(keep):
        t = tuple(name)
        ver = next((x for x in t[1:] if is_version(x)), "")
        nxt = toks[end] if end < len(toks) else ""
        after = toks[end + 1] if end + 1 < len(toks) else ""
        if not ver and is_version(nxt) and len(nxt) <= 4 and after not in _UNIT_WORDS:
            t = t + (nxt,)
            ver = nxt
        if t in seen:
            continue
        seen.add(t)
        out.append({"name": vocab.vocab.get(name, " ".join(name)), "tokens": t,
                    "family": t[0], "version": ver})
    return out


def request_task(text: str) -> str | None:
    """The recipe task a request's wording asks for, or None when it does not say."""
    low = str(text or "").lower()
    for task, rx in _TASK_WORDS:
        if rx.search(low):
            return task
    return None


def _request_media(text: str) -> str | None:
    try:
        from agenty_core.utils.template_fit import request_wants
        return request_wants(text).get("kind")
    except Exception:  # noqa: BLE001
        return None


def task_fits(want_task: str | None, want_media: str | None, template_task: str | None) -> bool:
    """Whether a template of *template_task* does the job the request says."""
    if want_task:
        return template_task in _TASK_FITS.get(want_task, {want_task})
    if want_media and template_task:
        media = _TASK_MEDIA.get(template_task)
        return media is None or media == want_media
    return True


def carries(names: Iterable[tuple], named: tuple) -> bool:
    """Whether any of a template's *names* contains the *named* model, to the version."""
    return any(_contains(tuple(n), tuple(named)) for n in names)


def model_versions(names: Iterable[tuple]) -> list[tuple[str, str]]:
    """``[(family, version)]`` of a template's model names (family = first word)."""
    out: list[tuple[str, str]] = []
    for n in names:
        n = tuple(n)
        ver = next((x for x in n[1:] if is_version(x)), "")
        if ver and n and (n[0], ver) not in out:
            out.append((n[0], ver))
    return out


def template_model_names(template: str, data: Any = None, vocab: Vocabulary | None = None) -> list[tuple]:
    """The model names of *template*: the corpus's, else (a local graph) its main model files."""
    try:
        names = (vocab or vocabulary()).names_of(template)
    except Exception:  # noqa: BLE001
        names = []
    if names:
        return names
    if data is not None:
        side = side_model_files(data)
        return [tuple(tokens(f)) for f in main_model_files(data)
                if " ".join(tokens(f)) not in side and not _COMPONENT_WORDS & set(tokens(f))]
    return []


def pin_templates(text: str, catalog: Iterable[str] | None = None, vocab: Vocabulary | None = None,
                  cap: int = 8) -> dict | None:
    """The templates a request holds the researcher to, by the models it names.

    ``{"named": [...], "task": ..., "templates": [...]}`` - the templates that carry
    the most of the named models (to the version), whose recipe task fits what
    the request says (edit / text-to-image / image-to-video ...) and whose model
    names share the most words with it. None when nothing is named, nothing fits,
    or the fit is no choice: more than *cap* templates spread over more than two
    recipes ("FLUX.1" fits every FLUX.1 graph; every Qwen Image Edit 2511 edit
    graph sits in one recipe and is a fair pick).
    """
    vocab = vocab or vocabulary()
    named = named_models(text, vocab)
    if not named:
        return None
    task = request_task(text)
    media = None if task else _request_media(text)
    words = set(tokens(text))
    pool = list(catalog) if catalog is not None else list(vocab.templates)
    best: list[str] = []
    best_score: tuple = (0, 0)
    for tpl in pool:
        names = vocab.names_of(tpl)
        if not names:
            continue
        carried = [m for m in named if carries(names, m["tokens"])]
        if not carried or not task_fits(task, media, vocab.task_of(tpl)):
            continue
        # A template whose recipe puts it in ANOTHER family loads the named model
        # only as a part (Lotus depth on an SD 1.5 base, Bernini-R on Wan 2.2): not
        # a template "for" that model.
        fam = (vocab.templates.get(tpl) or {}).get("family") or ""
        if fam and fam.lower() != "generic":
            head = (tokens(fam) or [""])[0]
            if not any(m["family"] == head for m in carried):
                continue
        # Then the template whose model names say most of what the request says:
        # "FLUX.1 fill" is flux1-fill-dev, "base and refiner" the refiner example.
        said = max(len(set(n) & words) for n in names)
        score = (len(carried), said)
        if score > best_score:
            best, best_score = [tpl], score
        elif score == best_score:
            best.append(tpl)
    recipes = {(vocab.task_of(t), (vocab.templates.get(t) or {}).get("family")) for t in best}
    if not best or (len(best) > cap and len(recipes) > 2) or len(best) > 3 * cap:
        return None
    return {"named": [m["name"] for m in named], "task": task, "templates": sorted(best)}
