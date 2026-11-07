"""Read and change a cut in the ComfyUI Image Viewer's Edit panel: ``edit_cut``.

"Cut the voice-over into the edit", "put the music under shots 3 to 5", "trim the last
shot", "take clip 4 out". The cut is the ``.otio`` the Edit panel has open (the viewer's
state names it) or any other one. The viewer does the work (``POST /imageviewer/cut``,
reached through ComfyUI's HTTP API like every other core tool): it changes the cut the
way the panel would, saves it, and tells the open Edit panels, which take the new cut in
as one undo step - so a change made here is never lost to the panel's next save, and is
not made by hand-writing the OTIO (which the panel would not see, and would then save
over).

Tracks are named as the panel shows them: V1 is the cut itself (clips one after
another), A1 the bed laid under the whole film, V2.. / A2.. clips placed at a moment of
the film. Clips are numbered from 1 on each track, in the order they play.
"""

from __future__ import annotations

import json
import os

from agenty_core._compat import tool

#: Most clips listed per track in a read (the rest are counted).
LIST_CAP = 60


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _guard(path: str) -> str:
    from agenty_core.utils.path_guard import refuse_read
    return refuse_read(path, "edit_cut")


def _r(v) -> float:
    try:
        return round(float(v or 0), 3)
    except (TypeError, ValueError):
        return 0.0


def _row(n: int, c: dict, at: float) -> dict:
    words = ""
    if isinstance(c.get("title"), dict):
        words = str(c["title"].get("text") or "").strip().splitlines()[0:1]
        words = words[0] if words else ""
    name = (f'title "{words[:60]}"' if words else "title") if isinstance(c.get("title"), dict)         else str(c.get("name") or os.path.basename(str(c.get("file") or "")))
    out = {"clip": n, "name": name[:80],
           "file": str(c.get("file") or ""), "at": _r(at), "end": _r(at + float(c.get("length") or 0)),
           "in": _r(c.get("in")), "length": _r(c.get("length"))}
    if isinstance(c.get("title"), dict):
        out["is_title"] = True
    if c.get("volume") not in (None, 1, 1.0):
        out["volume"] = _r(c.get("volume"))
    if c.get("speed") not in (None, 1, 1.0):
        out["speed"] = _r(c.get("speed"))
    if float(c.get("transition") or 0) > 0:
        out["dissolve"] = _r(c.get("transition"))
    if float(c.get("gap") or 0) > 0:
        out["gap_before"] = _r(c.get("gap"))
    _fades(out, c)
    fx = _effects(c.get("afx"))
    if fx:
        out["effects"] = fx
    return out


_AFX_NAMES = {"eq": "EQ", "comp": "Compressor", "limit": "Limiter", "reverb": "Reverb", "delay": "Delay",
              "gain": "Gain", "pan": "Pan"}


def _effects(chain) -> list[str]:
    """Sound effects of a chain as display names, a bypassed one marked: ``["EQ", "Reverb (bypassed)"]``."""
    out = []
    for fx in chain if isinstance(chain, list) else []:
        if isinstance(fx, dict):
            name = _AFX_NAMES.get(str(fx.get("type")), str(fx.get("type") or "effect"))
            out.append(name + (" (bypassed)" if fx.get("on") is False else ""))
    return out


def _fades(out: dict, c: dict) -> None:
    """Audio fades (seconds of the clip) and the crossfade into the clip, when it has any."""
    if _r(c.get("fadeIn")) > 0:
        out["fade_in"] = _r(c.get("fadeIn"))
    if _r(c.get("fadeOut")) > 0:
        out["fade_out"] = _r(c.get("fadeOut"))
    x = c.get("xfade")
    if isinstance(x, dict) and _r(x.get("len")) > 0:
        out["crossfade_in"] = {"length": _r(x.get("len")), "curve": str(x.get("curve") or "+3"),
                               "align": str(x.get("align") or "centre")}


def _crossfades(clips: list[dict]) -> list[dict]:
    """The audio crossfades of a track's rows: which clip they come into, from which, how long."""
    out = []
    for i, c in enumerate(clips):
        x = c.get("crossfade_in")
        if x:
            prev = clips[i - 1] if i else None
            out.append({"from_clip": prev["clip"] if prev else None, "into_clip": c["clip"],
                        "between": f"{prev['name']} -> {c['name']}" if prev else f"-> {c['name']}", **x})
    return out


def locked_tracks(edit: dict) -> set[str]:
    """Names (upper case) of the tracks the Edit panel has locked, texts as ``T``."""
    e = edit if isinstance(edit, dict) else {}
    out = {str(k).upper() for k, v in (e.get("lock") or {}).items() if v}
    for t in e.get("tracks") or []:
        if isinstance(t, dict) and t.get("lock"):
            out.add(str(t.get("id") or t.get("name") or "").upper())
    return out


def _locked_refusal(lock: set[str], ops: list) -> str:
    """Why the first op naming a locked track cannot be done ("" = none does)."""
    for i, o in enumerate(ops, 1):
        t = str(o.get("track") or "").strip().upper() if isinstance(o, dict) else ""
        if t and t in lock:
            return (f"not changed: change {i} ({o.get('op')}): track {t} is locked in the Edit panel - "
                    "ask the user to unlock it there (the lock icon on the track), or use another track")
    return ""


def _said(msg) -> str:
    """The viewer's refusal, as is; one about a locked track gets the way out added."""
    text = str(msg)[:400]
    if "locked" in text.lower() and "unlock" not in text.lower():
        text += " - ask the user to unlock the track in the Edit panel"
    return text


def outline(edit: dict) -> list[dict]:
    """The tracks of a cut (the panel's form, as the viewer answers it) as compact rows:
    ``[{"track", "kind", "clips": [{"clip", "name", "file", "at", "end", "in", "length"}]}]``."""
    e = edit if isinstance(edit, dict) else {}
    tracks = []
    rows, at = [], 0.0
    for i, c in enumerate(e.get("clips") or []):
        if not isinstance(c, dict):
            continue
        at += float(c.get("gap") or 0)
        if i:
            at -= float(c.get("transition") or 0)
        rows.append(_row(len(rows) + 1, c, max(0.0, at)))
        at += float(c.get("length") or 0)
    tracks.append({"track": "V1", "kind": "video", "what": "the cut: clips one after another", "clips": rows})
    bed = [{"clip": n, "name": str(s.get("name") or os.path.basename(str(s.get("file") or "")))[:80],
            "file": str(s.get("file") or "")}
           for n, s in enumerate((x for x in e.get("sound") or [] if isinstance(x, dict)), 1)]
    for row, s in zip(bed, (x for x in e.get("sound") or [] if isinstance(x, dict))):
        if _r(s.get("fadeIn")) > 0:
            row["fade_in"] = _r(s.get("fadeIn"))
        if s.get("fadeOut") is not None and _r(s.get("fadeOut")) != 1.0:   # 1 s is agentY's default
            row["fade_out"] = _r(s.get("fadeOut"))
    tracks.append({"track": "A1", "kind": "audio", "what": "the bed under the whole film", "clips": bed})
    for t in e.get("tracks") or []:
        if not isinstance(t, dict):
            continue
        clips = [c for c in t.get("clips") or [] if isinstance(c, dict)]
        row = {"track": str(t.get("id") or t.get("name") or "?"), "kind": t.get("kind") or "video",
               "clips": [_row(n, c, float(c.get("at") or 0)) for n, c in enumerate(clips, 1)]}
        if t.get("off"):
            row["off"] = True
        if t.get("lock"):
            row["locked"] = True
        if t.get("auto") is False:
            row["auto_select_off"] = True
        if _effects(t.get("afx")):
            row["effects"] = _effects(t.get("afx"))
        tracks.append(row)
    for t in tracks[:2]:
        if (e.get("lock") or {}).get(t["track"]):
            t["locked"] = True
        if (e.get("auto") or {}).get(t["track"]) is False:
            t["auto_select_off"] = True
    bed_fx = _effects((e.get("afx") or {}).get("A1") if isinstance(e.get("afx"), dict) else None)
    if bed_fx:
        tracks[1]["effects"] = bed_fx
    for t in tracks:
        xs = _crossfades(t["clips"])
        if xs:
            t["crossfades"] = xs
    for t in tracks:
        if len(t["clips"]) > LIST_CAP:
            t["more"] = len(t["clips"]) - LIST_CAP
            t["clips"] = t["clips"][:LIST_CAP]
    return tracks


def marked_range(state: dict) -> tuple[int, int] | None:
    """The in/out the user marked on the cut's timeline (film frames, both included), from the
    viewer's state - only while the tab on screen is the Edit panel's cut (the marks of any other
    tab are frames of that tab's clip, not of the cut)."""
    s = state if isinstance(state, dict) else {}
    edit = s.get("edit") or {}
    tab = s.get("tab") or {}
    if not edit.get("otio") or not (tab.get("mode") == "edit" or (edit.get("tab") and edit.get("tab") == tab.get("key"))):
        return None
    r = (s.get("timeline") or {}).get("range") or {}
    try:
        return int(r["start"]), int(r["end"])
    except (KeyError, TypeError, ValueError):
        return None


def _open_cut() -> tuple[str, str, tuple[int, int] | None]:
    """``(the .otio the Edit panel has open, why not, the in/out marked on it or None)``."""
    from agenty_core.tools.viewer_state import read_state
    try:
        got = json.loads(read_state())
    except Exception as e:  # noqa: BLE001
        return "", f"the viewer's state could not be read ({e})", None
    if got.get("status") != "ok":
        return "", got.get("error") or "the viewer's state could not be read", None
    state = got.get("state") or {}
    edit = state.get("edit") or {}
    if not edit.get("otio"):
        if edit.get("open"):
            return "", ("the cut open in the Edit panel has no file yet - ask the user to save it "
                        "(Save in the Edit panel), or pass otio="), None
        return "", "no cut is open in the viewer's Edit panel - pass otio= (the cut's .otio path)", None
    return str(edit["otio"]), "", marked_range(state)


#: Most pieces of a marked range handed on (one per clip it touches, on every picture track).
MARKED_CAP = 20

#: What the agent is told about a marked range's pieces.
MARKED_NOTE = ("pieces: one per clip under the marked in/out. src_in/src_out are frames of the clip's FILE "
               "(0 = its first frame) and src_in_s/src_seconds its seconds - the viewer worked them out "
               "through the clip's place, in point, speed and loop and the file's own fps; use them as they "
               "are (video_frames(frames=[src_in, src_out]), a trim from src_in_s for src_seconds), never "
               "work them out yourself. rec_in/rec_out are frames of the cut. A piece with covered_by is "
               "under a clip of a higher track there.")


def _marked_of(r: dict, start: int, end: int) -> dict:
    """The ``marked`` part of a /imageviewer/cut answer, for the agent (or why there is none)."""
    m = r.get("marked") if isinstance(r, dict) else None
    if not isinstance(m, dict):
        return {"error": ("the viewer in this ComfyUI does not map a marked range to its sources yet - update "
                          "ComfyUI-ImageViewer under <ComfyUI>/custom_nodes/ and restart ComfyUI"),
                "start": start, "end": end}
    out = dict(m)
    pieces = [p for p in m.get("pieces") or [] if isinstance(p, dict)]
    out["pieces"] = pieces[:MARKED_CAP]
    if len(pieces) > MARKED_CAP:
        out["more_pieces"] = len(pieces) - MARKED_CAP
    if not pieces:
        out["note"] = "no clip with a picture lies under the marked in/out (a gap, or a switched-off clip)"
    else:
        out["note"] = MARKED_NOTE
    return out


def marked_sources(otio: str, start: int, end: int) -> dict:
    """The marked in/out (*start*..*end*, film frames of the cut in *otio*) in the sources: the
    viewer's ``timeline_io.source_map`` through ``POST /imageviewer/cut`` (a read; nothing is written).
    ``{"start", "end", "frames", "fps", "start_s", "end_s", "pieces": [...], "note"}`` or
    ``{"error"}``."""
    from urllib.parse import urlencode
    from agenty_core.utils import canvas_scope as _cs
    from agenty_core.utils.comfyui_client import get_client
    why = _guard(otio)
    if why:
        return {"error": why}
    cid = _cs.client_id()
    route = "/imageviewer/cut" + (("?" + urlencode({"client_id": cid})) if cid else "")
    try:
        r = get_client().post(route, json_data={"otio": otio, "ops": [],
                                                "range": {"start": int(start), "end": int(end)}})
    except Exception as e:  # noqa: BLE001
        return {"error": f"the cut could not be read ({str(e)[:160]})"}
    if not isinstance(r, dict) or not r.get("ok"):
        return {"error": _said((r or {}).get("error") if isinstance(r, dict) else r)}
    return _marked_of(r, int(start), int(end))


@tool
def edit_cut(otio: str = "", ops: list | None = None, frame_in: int = -1, frame_out: int = -1) -> str:
    """Read the cut open in the ComfyUI Image Viewer's Edit panel (or any .otio cut), and change it:
    lay a voice-over, music or a clip on a track at a time, move or trim a clip, take one out.

    The in/out the user MARKED on the cut ("the shot between my marks", "recreate what I
    marked"): a read returns ``marked`` - for every clip under the marks its file and the
    source frames ``src_in``/``src_out`` (frames of THAT FILE, at its own fps) and
    ``src_in_s``/``src_seconds``, worked out by the viewer through the clip's place, in point,
    speed, loop and split pieces. Use them as they are; never compute source frames yourself.
    The marks are taken from the viewer while its Edit tab is on screen; ``frame_in`` /
    ``frame_out`` (frames of the cut, both included) give a range yourself.

    The viewer saves the change the way its Edit panel does and the open panel shows it at
    once (one Ctrl+Z undoes it). Never write the .otio yourself (run_script): the panel would
    not see the change and its next save would drop it.

    Tracks, as the panel shows them: V1 = the cut (clips one after another), A1 = the bed
    under the whole film, V2.. / A2.. = clips placed at a moment of the film. Clips are
    numbered from 1 on each track, in play order. Times are seconds of the film.

    Args:
        otio: The cut's .otio path; "" for the cut open in the Edit panel.
        ops: Leave empty to READ the cut (tracks and clips with their times). To change it,
            a list (all done, or none):
            ``{"op": "add", "file": <audio/video/image path>, "at": 8.25, "track": "",
            "in": 0, "length": <s>, "volume": 1.0, "name": ""}`` - a clip at ``at`` on
            ``track`` ("A2", "V2", "new", or "" for the first track of the file's kind with
            room - sound goes on A2.., picture on V2..). ``length`` defaults to the rest of the
            file. To lay a voice-over in pieces, add each piece (with ``in``/``length`` into the
            one file) at its own ``at``.
            ``{"op": "set", "track": "A2", "clip": 1, "at": .., "in": .., "length": .., "volume": ..,
            "name": ..}`` - move / trim / re-level; on V1 ``in``/``length`` trim and ripple, and
            ``gap`` sets empty time before a clip (V1 clips have no ``at``).
            ``{"op": "remove", "track": "V1", "clip": 3}`` - take a clip out (V1 closes up).

    A track the user LOCKED in the panel takes no change: such an op is refused (ask the user to
    unlock it); an add with no track named skips locked tracks.

    Returns JSON ``{status, otio, length, tracks: [{track, kind, locked?, auto_select_off?, clips:
    [{clip, name, file, at, end, in, length, fade_in?, fade_out?, crossfade_in?}], crossfades?:
    [{from_clip, into_clip, length, curve, align}]}], marked?: {start, end, fps, pieces: [{track,
    clip, file, rec_in, rec_out, src_in, src_out, src_frames, src_in_s, src_out_s, src_seconds,
    src_fps, speed, loop?, runs?, covered_by?}]}, said, pushed}``; ``said`` lists what was
    changed, ``pushed`` whether the open panels were told. Audio fades are seconds of the clip; a
    crossfade lies on the clip coming into a cut. Sound effects (EQ, Compressor, Reverb ...) are
    shown too - ``effects`` on a clip or a track, e.g. ``["EQ", "Reverb (bypassed)"]`` - but are
    not changeable through edit_cut: the user sets them in the panel.
    """
    from agenty_core.utils import canvas_scope as _cs
    ops = ops or []
    if isinstance(ops, dict):
        ops = [ops]
    if isinstance(ops, str):
        try:
            ops = json.loads(ops)
        except ValueError:
            return _j({"status": "error", "error": "ops is a list of changes (JSON)"})
        if isinstance(ops, dict):
            ops = [ops]
    if not isinstance(ops, list):
        return _j({"status": "error", "error": "ops is a list of changes"})
    otio = str(otio or "").strip()
    try:
        fin, fout = int(frame_in), int(frame_out)
    except (TypeError, ValueError):
        return _j({"status": "error", "error": "frame_in / frame_out are frames of the cut (whole numbers)"})
    marks = (fin, fout) if fin >= 0 and fout >= 0 else None
    if (fin >= 0) != (fout >= 0):
        return _j({"status": "error", "error": "give both frame_in and frame_out (frames of the cut, both included)"})
    if not otio:
        if _cs.detached():
            return _j({"status": "error", "error": _cs.NO_CANVAS + " Pass otio= (the cut's .otio path)."})
        otio, why, shown = _open_cut()
        if not otio:
            return _j({"status": "error", "error": why})
        if marks is None and not ops:
            marks = shown
    if not otio.lower().endswith(".otio"):
        return _j({"status": "error", "error": f"{os.path.basename(otio)} is not a cut (.otio)"})
    for path in [otio] + [str(o.get("file")) for o in ops if isinstance(o, dict) and o.get("file")]:
        why = _guard(path)
        if why:
            return _j({"status": "error", "error": why})
    from urllib.parse import urlencode
    from agenty_core.utils.comfyui_client import get_client
    cid = _cs.client_id()
    route = "/imageviewer/cut" + (("?" + urlencode({"client_id": cid})) if cid else "")
    if any(isinstance(o, dict) and str(o.get("track") or "").strip() for o in ops):
        # Refuse a change to a locked track before sending it (the viewer refuses too).
        try:
            pre = get_client().post(route, json_data={"otio": otio, "ops": []})
            why = _locked_refusal(locked_tracks(pre.get("edit") if isinstance(pre, dict) else {}), ops)
        except Exception:  # noqa: BLE001 - the real call below says what is wrong
            why = ""
        if why:
            return _j({"status": "error", "error": why})
    body = {"otio": otio, "ops": ops}
    if marks is not None:
        body["range"] = {"start": marks[0], "end": marks[1]}
    try:
        r = get_client().post(route, json_data=body)
    except Exception as e:  # noqa: BLE001
        resp = getattr(e, "response", None)
        status = getattr(resp, "status_code", None)
        if status == 404 or (status is None and "404" in str(e)):
            try:
                said = resp.json().get("error") if resp is not None else ""
            except Exception:  # noqa: BLE001
                said = ""
            if said:
                return _j({"status": "error", "error": said})
            return _j({"status": "error", "error": (
                "the viewer in this ComfyUI predates edit_cut (no /imageviewer/cut route) - update "
                "ComfyUI-ImageViewer under <ComfyUI>/custom_nodes/ and restart ComfyUI.")})
        if resp is not None:
            try:
                said = resp.json().get("error")
            except Exception:  # noqa: BLE001
                said = ""
            if said:
                return _j({"status": "error", "error": _said(said)})
        return _j({"status": "error", "error": f"could not reach ComfyUI ({str(e)[:160]}). Is it running?"})
    if not isinstance(r, dict) or not r.get("ok"):
        return _j({"status": "error", "error": _said((r or {}).get("error") if isinstance(r, dict) else r)})
    out = {"status": "ok", "otio": r.get("otio") or otio, "length": _r(r.get("length")),
           "tracks": outline(r.get("edit") or {})}
    ed = r.get("edit") if isinstance(r.get("edit"), dict) else {}
    if (ed.get("lock") or {}).get("T"):
        out["texts_locked"] = True
    if (ed.get("auto") or {}).get("T") is False:
        out["texts_auto_select_off"] = True
    if marks is not None:
        out["marked"] = _marked_of(r, *marks)
    if ops:
        out["said"] = [str(s)[:200] for s in (r.get("said") or [])][:50]
        out["pushed"] = bool(r.get("pushed"))
        if r.get("notes"):
            out["notes"] = [str(n)[:160] for n in r["notes"]][:5]
    return _j(out)
