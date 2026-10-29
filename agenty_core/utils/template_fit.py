"""Does this template do what the request asks - by what it IS, not what it is called?

Retrieval matches on names and descriptions, and both mislead: a template named
for "Seedance" and "storyboard to video" came back for "animate each starting
frame with Seedance 2.5 first-frame, 480p, no audio, cut into one film" - although
it draws an 8-panel board first, drives Seedance 2.0 through a reference node, and
renders 720p with audio. (The catalog even files video templates under
``mediaType: image``.) Nothing in the name said so; the graph does.

:func:`template_facts` reads a template's graph (UI or API format, subgraphs
included): what it outputs, whether it draws or consumes a storyboard and how many
panels, which model versions, resolutions and audio setting it carries, whether it
takes a first frame, a reference or only text. :func:`request_wants` reads the same
things out of a request. :func:`check` compares them:

* ``reject`` - structural mismatches nobody can patch with a widget value: the
  wrong output kind, a storyboard step the request never asked for, another model
  version than the one named, a text-only template for a job that starts from
  frames, a reference-mode template for a first-frame job;
* ``demote`` - parameters a widget can change (resolution, audio): the template
  still works, the reason says what to set;
* ``fit`` - nothing found.

Pure and offline: no ComfyUI needed. Whatever a template does not state (audio of
a UI-format graph without schemas) is ``None`` and never counts against it.
"""
from __future__ import annotations

import re
from typing import Any

_VIDEO_SAVERS = {"SaveVideo", "VHS_VideoCombine", "SaveWEBM", "SaveAnimatedWEBP", "SaveAnimatedPNG",
                 "VHS_SaveVideo", "PreviewVideo", "SaveGif"}
_IMAGE_SAVERS = {"SaveImage", "PreviewImage", "SaveImageWebsocket", "SaveImageExtended",
                 "Image Save", "easy imageSave"}
_AUDIO_SAVERS = {"SaveAudio", "SaveAudioMP3", "SaveAudioOpus", "PreviewAudio", "VHS_SaveAudio"}
_MODEL_SAVERS = {"SaveGLB", "SaveGLTF", "Save3D", "Preview3D", "Save3DModel"}
_NOTE_TYPES = {"Note", "MarkdownNote"}
_LOADERS = {"LoadImage": "image", "LoadImageMask": "image", "LoadVideo": "video", "VHS_LoadVideo": "video",
            "LoadAudio": "audio"}

# family -> regex of its name; a version number may follow.
_FAMILIES = {
    "seedance": r"seedance",
    "seedream": r"seedream",
    "kling": r"kling",
    "veo": r"\bveo",
    "wan": r"\bwan",
    "ltx": r"\bltx",
    "flux": r"\bflux",
    "gpt image": r"gpt[\s_-]*image",
    "nano banana": r"nano[\s_-]*banana",
    "hailuo": r"hailuo|minimax",
    "runway": r"runway",
    "sora": r"\bsora",
}
_RES_RE = re.compile(r"(?<![\w.])(240|360|480|540|576|720|1080|1440|2160)\s?p\b|\b([248])\s?k\b", re.I)
_NO_AUDIO = re.compile(r"\b(no|without|silent|mute|muted|disable[d]?|turn(?:ed)? off)\s+(?:the\s+)?"
                       r"(audio|sound|sounds|music|voice|voices|dialogue|dialog)\b|\bsilent\b|\bmuted\b", re.I)
_WITH_AUDIO = re.compile(r"\bwith\s+(audio|sound|music|voice|dialogue|dialog)\b|\b(synchroni[sz]ed|generate|add)\s+"
                         r"(audio|sound)\b", re.I)
_FIRST_FRAME = re.compile(r"first[\s-]*frame|start(?:ing)?[\s-]*frame|from (?:the |a |each )?(?:first|start)|"
                          r"\bflf\b|first[\s/-]*(?:and[\s/-]*)?last|start(?:ing)?[\s-]*(?:and|&|/)[\s-]*end", re.I)
_FROM_FRAMES = re.compile(r"animate (?:each|every|the|this|these|my|a|an)\b.*?\b(frame|image|still|picture|photo)s?\b|"
                          r"\bimage[\s-]*to[\s-]*video\b|\bi2v\b", re.I)
_STRONG_VIDEO = re.compile(r"\b(animate|animation|animated|i2v|t2v|v2v|flf|film|clip|clips|footage|seedance|kling|"
                           r"veo|hailuo|sora|image[\s-]*to[\s-]*video|text[\s-]*to[\s-]*video)\b", re.I)
_REFERENCE = re.compile(r"\breference[\s-]*(image|video|to[\s-]*video|mode)s?\b|\br2v\b|\bconsistent (character|subject)",
                        re.I)
_STORYBOARD = re.compile(r"story[\s-]?board", re.I)
_VIDEO_WORDS = re.compile(r"\b(video|videos|clip|clips|film|films|movie|animate|animation|animated|footage|"
                          r"i2v|t2v|v2v|flf|motion|seedance|kling|veo|hailuo|sora)\b", re.I)
_IMAGE_WORDS = re.compile(r"\b(image|images|picture|pictures|photo|photos|poster|still|stills|portrait|"
                          r"illustration|render|artwork|thumbnail|logo)\b", re.I)
_AUDIO_WORDS = re.compile(r"\b(music|song|soundtrack|voice[\s-]?over|speech|tts|sound effect)s?\b", re.I)
_MODEL3D_WORDS = re.compile(r"\b(3d model|mesh|glb|gltf|splat|gaussian|3d asset)\b", re.I)


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------

def _versions(text: str, only: str | None = None) -> list[tuple[str, str]]:
    """``[(family, "2.5")]`` for each family named with a version in *text*."""
    out = []
    low = str(text or "").lower().replace("_", " ")
    for fam, rx in _FAMILIES.items():
        if only and fam != only:
            continue
        for m in re.finditer(rx + r"[\s-]*(?:v|version\s*)?(\d{1,2}(?:[.\s]\d(?!\d))?)?", low):
            ver = (m.group(1) or "").replace(" ", ".")
            if ver and "." not in ver and fam in ("seedance", "kling", "wan", "ltx", "veo"):
                ver = ver + ".0"
            out.append((fam, ver))
    # The first LTX-Video releases name their version after the size
    # ("ltx-video-2b-v0.9", "ltxv-13b-0.9.7"). Unread, ltxv_text_to_video counted as
    # versionless, passed for an "LTX-2" request and was built from twice
    # (CX10, night_20261029): an LTXV 0.9 graph with a T5 encoder.
    if not only or only == "ltx":
        for m in re.finditer(r"\bltx[\s-]*v(?:ideo)?[\s-]*(?:\d+(?:\.\d+)?b[\s-]*)?v?(0\.\d+)", low):
            out.append(("ltx", m.group(1)))
    return out


def _no_folders(text: str) -> str:
    """*text* without the folder part of model paths. "LTX2/ltx-2.5-22b-....safetensors"
    names LTX 2.5; read with its folder it also "named" LTX 2.0, and every LTX template
    was then marked NOT FOR THIS REQUEST (CX15, night_20261029)."""
    return re.sub(r"[\w.-]+[\\/]+(?=[\w.-])", " ", str(text or ""))


def _fit_version(ver: str) -> str:
    """A whole-number version as this module has always written it: "2" -> "2.0"
    (a 4-digit release like Qwen's "2511" stays as it is)."""
    return ver + ".0" if re.fullmatch(r"\d{1,2}", ver or "") else ver


def _named_versions(text: str) -> list[tuple[str, str]] | None:
    """``[(family, version)]`` the request names, by the corpus's own model names
    (:mod:`agenty_core.utils.model_names`); None when that vocabulary is unavailable."""
    try:
        from agenty_core.utils import model_names
        named = model_names.named_models(text)
    except Exception:  # noqa: BLE001
        return None
    return list(dict.fromkeys((m["family"], _fit_version(m["version"])) for m in named if m["version"]))


def _template_versions(name: str, data: Any) -> list[tuple[str, str]] | None:
    """``[(family, version)]`` of a template's model names (catalog names + main model
    files), or None when it states none - never read off a folder or a template file name."""
    try:
        from agenty_core.utils import model_names
        names = model_names.template_model_names(name, data)
    except Exception:  # noqa: BLE001
        return None
    if not names:
        return None
    return [(f, _fit_version(v)) for f, v in model_names.model_versions(names)]


def request_wants(text: str) -> dict[str, Any]:
    """What a request asks for, as far as its words say."""
    t = str(text or "")
    low = t.lower()
    pos = {}
    for kind_name, rx in (("video", _VIDEO_WORDS), ("3d", _MODEL3D_WORDS), ("audio", _AUDIO_WORDS),
                          ("image", _IMAGE_WORDS)):
        m = rx.search(low)
        if m:
            pos[kind_name] = m.start()
    # "animate each starting frame": the product is a video though images are named; a
    # strong video word wins, otherwise whatever is named first ("an image of a video game").
    if _STRONG_VIDEO.search(low):
        kind = "video"
    else:
        kind = min(pos, key=pos.get) if pos else None
    res = sorted({(m.group(1) + "p") if m.group(1) else (m.group(2) + "k") for m in _RES_RE.finditer(t)})
    audio = False if _NO_AUDIO.search(low) else (True if _WITH_AUDIO.search(low) else None)
    shots = None
    m = re.search(r"\b(\d{1,3})\s*(?:-)?\s*(shots?|panels?|clips?|scenes?)\b", low)
    if m:
        shots = int(m.group(1))
    # The model versions named: matched against the model names the corpus uses
    # ("Qwen Image Edit 2511", "LTX-2.5", "Wan2.2"), so "8 steps" or a folder
    # ("LTX2/ltx-2.5-...") never reads as a version. The old family regexes only
    # stand in when that vocabulary cannot be read.
    models = _named_versions(t)
    if models is None:
        models = list(dict.fromkeys((f, v) for f, v in _versions(_no_folders(low)) if v))
    return {
        "kind": kind,
        "models": models,
        "families": sorted({f for f, _v in _versions(low)}),
        "resolutions": res,
        "audio": audio,
        "storyboard": bool(_STORYBOARD.search(low)),
        "first_frame": bool(_FIRST_FRAME.search(low)),
        "from_frames": bool(_FIRST_FRAME.search(low) or _FROM_FRAMES.search(low)),
        "reference": bool(_REFERENCE.search(low)),
        "shots": shots,
        "multishot": bool(re.search(r"multi[\s-]?shot|multiple shots", low)),
    }


# ---------------------------------------------------------------------------
# the template
# ---------------------------------------------------------------------------

def _graph_nodes(data: dict) -> list[dict]:
    """``[{id, type, title, widgets, inputs}]`` of a UI graph (subgraphs included) or an API prompt."""
    out: list[dict] = []
    if isinstance(data.get("nodes"), list):
        def _walk(nodes):
            for n in nodes or []:
                if not isinstance(n, dict):
                    continue
                out.append({"id": n.get("id"), "type": str(n.get("type") or ""),
                            "title": str(n.get("title") or ""),
                            "widgets": [w for w in (n.get("widgets_values") or [])
                                        if not isinstance(w, (dict, list))] if isinstance(n.get("widgets_values"), list)
                            else [], "inputs": {}})
        _walk(data.get("nodes"))
        for sg in ((data.get("definitions") or {}).get("subgraphs") or []):
            _walk(sg.get("nodes"))
        return out
    for nid, n in data.items():
        if isinstance(n, dict) and "class_type" in n:
            ins = n.get("inputs") if isinstance(n.get("inputs"), dict) else {}
            out.append({"id": nid, "type": str(n.get("class_type")),
                        "title": str((n.get("_meta") or {}).get("title") or ""),
                        "widgets": [v for v in ins.values() if isinstance(v, (str, int, float, bool))],
                        "inputs": ins})
    return out


def template_facts(data: dict, name: str = "", description: str = "") -> dict[str, Any]:
    """What the template does, read off its graph. Never raises; unknowns are ``None`` / empty."""
    facts: dict[str, Any] = {"name": name, "kinds": [], "primary": None, "storyboard": False, "panels": None,
                             "models": [], "resolutions": [], "audio": None, "capability": None,
                             "inputs": {}, "multishot": False, "classes": []}
    try:
        nodes = _graph_nodes(data if isinstance(data, dict) else {})
    except Exception:  # noqa: BLE001
        return facts
    classes = {n["type"] for n in nodes}
    facts["classes"] = sorted(classes)
    kinds = []
    if classes & _VIDEO_SAVERS:
        kinds.append("video")
    if classes & _MODEL_SAVERS or any(c.startswith(("Meshy", "Tripo", "Rodin", "Hunyuan3D")) for c in classes):
        kinds.append("3d")
    if classes & _AUDIO_SAVERS:
        kinds.append("audio")
    if classes & _IMAGE_SAVERS:
        kinds.append("image")
    facts["kinds"] = kinds
    facts["primary"] = kinds[0] if kinds else None

    inputs: dict[str, int] = {}
    for n in nodes:
        k = _LOADERS.get(n["type"])
        if k:
            inputs[k] = inputs.get(k, 0) + 1
    facts["inputs"] = inputs

    texts, notes = [], []
    for n in nodes:
        bucket = notes if n["type"] in _NOTE_TYPES else texts
        bucket.append(n["title"])
        bucket.extend(str(w) for w in n["widgets"] if isinstance(w, str))
    blob = " ".join(texts)
    low_name = (name or "").lower()

    # model versions: from the template's model names - the catalog's ("LTX-0.9.5",
    # "Qwen-Image 2512") and its main model files, without their folders. Only a
    # graph that states none (an API graph of a custom pack) is read from its
    # widget values and description, folders stripped. Never from the template's
    # file name: "image_qwen_image_2512_with_2steps_lora" names a LoRA's step count.
    seen = []
    for src in (blob, name, description):
        for fam, ver in _versions(src):
            if (fam, ver) not in seen:
                seen.append((fam, ver))
    stated = _template_versions(name, data)
    if stated is None:
        stated = []
        for src in (_no_folders(blob), description):
            for fam, ver in _versions(src):
                if ver and (fam, ver) not in stated:
                    stated.append((fam, ver))
    facts["models"] = stated
    facts["families"] = sorted({f for f, _v in seen})

    res = set()
    for n in nodes:
        for w in n["widgets"]:
            if isinstance(w, str) and len(w) <= 8:
                m = _RES_RE.fullmatch(w.strip())
                if m:
                    res.add((m.group(1) + "p") if m.group(1) else (m.group(2) + "k"))
    facts["resolutions"] = sorted(res)

    # audio: only an API-format graph names its widgets
    audio = None
    for n in nodes:
        for key, val in (n["inputs"] or {}).items():
            if str(key).rsplit(".", 1)[-1] == "generate_audio" and isinstance(val, bool):
                audio = val if audio is None else (audio or val)
    facts["audio"] = audio

    # storyboard: drawn or consumed by the graph, or said so by its name / notes / description
    all_text = " ".join([low_name, blob.lower(), " ".join(notes).lower(), (description or "").lower()])
    if _STORYBOARD.search(all_text):
        facts["storyboard"] = True
        m = re.search(r"(\d+)[\s-]*(?:[a-z-]+\s+){0,2}(?:panels?|frames?|shots?)\b", all_text)
        m2 = re.search(r"(\d+)[\s-]*panel", all_text)
        pick = m2 or m
        if pick:
            facts["panels"] = int(pick.group(1))

    # capability by name and nodes
    probe = low_name + " " + " ".join(c.lower() for c in classes)
    if re.search(r"flf|first[_\s-]?last|firstlast|start[_\s-]?end|i2v_flf", probe):
        facts["capability"] = "first_last_frame"
    elif re.search(r"r2v|reference2v|reference[_\s-]to|referenceto", probe):
        facts["capability"] = "reference"
    elif re.search(r"i2v|image[_\s-]?to[_\s-]?video|img2vid", probe):
        facts["capability"] = "image_to_video"
    elif re.search(r"t2v|text[_\s-]?to[_\s-]?video", probe):
        facts["capability"] = "text_to_video"
    elif re.search(r"v2v|video[_\s-]?(?:edit|to[_\s-]?video)", probe):
        facts["capability"] = "video_to_video"
    facts["multishot"] = bool(re.search(r"multi[\s_-]?shot", all_text))
    return facts


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------

def check(request: str | dict, facts: dict) -> dict[str, Any]:
    """``{"verdict": "fit"|"demote"|"reject", "reasons": [...], "adjust": [...]}``.

    *request* is the request text (or an already computed :func:`request_wants`)."""
    want = request if isinstance(request, dict) else request_wants(request)
    reasons: list[str] = []
    adjust: list[str] = []

    # -- kind of product -----------------------------------------------------
    prim, kinds = facts.get("primary"), facts.get("kinds") or []
    wk = want.get("kind")
    if wk and prim and wk not in kinds:
        reasons.append(f"it produces {'/'.join(kinds)}, the request is for {wk}")

    # -- storyboard step ----------------------------------------------------
    if facts.get("storyboard") and not want.get("storyboard"):
        n = facts.get("panels")
        reasons.append("it draws a storyboard board first" + (f" ({n} panels)" if n else "")
                       + " and animates that, which the request does not ask for")
    # -- model version -------------------------------------------------------
    have = facts.get("models") or []
    named: dict[str, list[str]] = {}
    for fam, ver in want.get("models") or []:
        named.setdefault(fam, [])
        if ver not in named[fam]:
            named[fam].append(ver)
    for fam, vers in named.items():
        mine = sorted({v for f, v in have if f == fam})
        if mine and not set(vers) & set(mine):
            reasons.append(f"it uses {fam.title()} {'/'.join(mine)}, the request names "
                           f"{fam.title()} {'/'.join(vers)}")
    # -- how the job starts ----------------------------------------------------
    cap = facts.get("capability")
    if want.get("from_frames") and not want.get("reference"):
        if cap == "text_to_video":
            reasons.append("it is text-to-video (no start frame input), the request starts from frames")
        elif cap in (None, "image_to_video", "first_last_frame") and (facts.get("inputs") or {}).get("image", 1) == 0:
            reasons.append("it takes no image input, the request starts from frames")
    if want.get("first_frame") and not want.get("reference") and cap == "reference":
        reasons.append("it is a reference-to-video template, the request is a first-frame job")
    if want.get("reference") and cap in ("text_to_video",):
        reasons.append("it is text-to-video, the request needs reference images")
    if facts.get("multishot") and not want.get("multishot") and want.get("shots") in (None, 1):
        adjust.append("it builds a multi-shot sequence; the request is for a single shot")

    # -- parameters (patchable) --------------------------------------------------
    wr, hr = want.get("resolutions") or [], facts.get("resolutions") or []
    if wr and hr and not set(wr) & set(hr):
        adjust.append(f"resolution is {'/'.join(hr)} in the template, the request asks {'/'.join(wr)} - set it")
    wa, ha = want.get("audio"), facts.get("audio")
    if wa is False and ha is True:
        adjust.append("audio is ON in the template, the request wants none - set generate_audio to false")
    if wa is True and ha is False:
        adjust.append("audio is OFF in the template, the request wants it - set generate_audio to true")

    verdict = "reject" if reasons else ("demote" if adjust else "fit")
    return {"verdict": verdict, "reasons": reasons, "adjust": adjust}


# ---------------------------------------------------------------------------
# by template name
# ---------------------------------------------------------------------------

_cache: dict[str, tuple] = {}


def facts_for_template(name: str, description: str = "") -> dict[str, Any] | None:
    """Facts of the corpus template *name* (custom, then official, then local), cached by file mtime."""
    from pathlib import Path
    try:
        from agenty_core.tools import comfyui as C
        found = None
        for base in (C._custom_templates_dir(), C._official_templates_dir()):
            found = C._find_template_file(Path(base), name)
            if found is not None:
                break
        if found is None:
            return None
        stamp = found.stat().st_mtime
        hit = _cache.get(str(found))
        if hit and hit[0] == stamp:
            return hit[1]
        import json
        data = json.loads(found.read_text(encoding="utf-8-sig"))
        facts = template_facts(data, name, description)
        _cache[str(found)] = (stamp, facts)
        return facts
    except Exception:  # noqa: BLE001 - a template that cannot be read is simply not judged
        return None


def check_template(request: str, name: str, description: str = "") -> dict[str, Any] | None:
    """:func:`check` for the corpus template *name*; ``None`` when it cannot be read."""
    facts = facts_for_template(name, description)
    return None if facts is None else {**check(request, facts), "template": name}


def one_line(verdict: dict) -> str:
    """The reasons of a verdict in one line, for a catalog entry or a message."""
    parts = list(verdict.get("reasons") or []) or list(verdict.get("adjust") or [])
    return "; ".join(parts)
