"""Build, read, render and farm out comps of the ComfyUI Image Viewer's Comp space (``*.comp.json``).

``edit_comp`` reads a comp or changes it (``POST /imageviewer/comp_ops``: the viewer's ``comp_api.Comp.apply``,
all or nothing, saved, and an open Comp tab takes it in); ``render_comp`` renders its Write nodes in ComfyUI
(``/imageviewer/comp_write``) or sends them to the Deadline farm (the viewer plans the job, ``comp_farm`` with
``dry_run``, and ``deadline_farm.submit_comp`` submits it from here). Never write the .comp.json by hand: the
open tab would not see it.
"""

from __future__ import annotations

import json
import logging
import threading
import time

from agenty_core._compat import tool

_OPS_HELP = "ops is a list of changes (JSON)"
_log = logging.getLogger(__name__)

# A Write job of a comp with a Render3D asks a viewer page for each 3D frame (the viewer's comp_routes.page_3d).
# When no page has taken the frame after _HEADLESS_AFTER s, a headless browser is started on ComfyUI's page
# (?imageviewer_serve3d=1), which draws the frames; the job waits up to _WAIT3D s for it to come up.
_HEADLESS_AFTER = 4.0
_WAIT3D = 120


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _said(e) -> str:
    resp = getattr(e, "response", None)
    if resp is not None:
        try:
            return str(resp.json().get("error") or "")[:400]
        except Exception:  # noqa: BLE001
            return ""
    return ""


def _status(e):
    return getattr(getattr(e, "response", None), "status_code", None)


def _fail(e, what: str) -> str:
    said = _said(e)
    if said:
        return _j({"status": "error", "error": said})
    if _status(e) == 404 or (_status(e) is None and "404" in str(e)):
        return _j({"status": "error", "error": (
            f"the viewer in this ComfyUI cannot {what} yet (no such route) - update ComfyUI-ImageViewer under "
            "<ComfyUI>/custom_nodes/ and restart ComfyUI.")})
    return _j({"status": "error", "error": f"could not reach ComfyUI ({str(e)[:160]}). Is it running?"})


def _open_comp() -> str:
    """The comp open in the viewer's Comp tab ("" when none)."""
    try:
        from agenty_core.tools.viewer_state import read_state
        got = json.loads(read_state())
    except Exception:  # noqa: BLE001
        return ""
    key = str(((got.get("state") or {}).get("tab") or {}).get("key") or "")
    return key[5:] if key.startswith("comp:") and key.endswith(".comp.json") else ""


def _path(path: str) -> tuple[str, str]:
    path = (path or "").strip() or _open_comp()
    if not path:
        return "", "no comp is open in the viewer - pass path= (a .comp.json)"
    if not path.lower().endswith(".comp.json"):
        return "", "path must be a *.comp.json"
    from agenty_core.utils.path_guard import refuse_read
    why = refuse_read(path, "edit_comp")
    return ("", why) if why else (path, "")


def _clean_ops(ops):
    if ops is None:
        return [], ""
    if isinstance(ops, str):
        if not ops.strip():
            return [], ""
        try:
            ops = json.loads(ops)
        except ValueError:
            return None, _OPS_HELP
    if isinstance(ops, dict):
        ops = [ops]
    return (ops, "") if isinstance(ops, list) else (None, _OPS_HELP)


def _run_ai(path: str, op: dict, wait_s: float) -> dict:
    """One run_ai op: the open viewer page runs the AI node (comp_routes' /comp_agent), saves the comp, answers."""
    from agenty_core.utils.comfyui_client import get_client
    c = get_client()
    body = {"action": "ai_run", "path": path, "node": str(op.get("node") or "")}
    for k in ("first", "last"):
        if op.get(k) is not None:
            try:
                body[k] = int(op[k])
            except (TypeError, ValueError):
                return {"node": body["node"], "state": "error", "error": f"{k} must be a frame number"}
    try:
        r = c.post("/imageviewer/comp_agent", json_data=body)
    except Exception as e:  # noqa: BLE001
        return {"node": body["node"], "state": "error", "error": json.loads(_fail(e, "run AI nodes for an agent"))["error"]}
    aid = r.get("id")
    end = time.time() + max(5.0, float(wait_s))
    last = r
    while time.time() < end:
        time.sleep(1.0)
        try:
            last = c.get("/imageviewer/comp_agent", params={"id": aid})
        except Exception as e:  # noqa: BLE001
            return {"node": body["node"], "state": "error", "error": json.loads(_fail(e, "run AI nodes for an agent"))["error"]}
        if last.get("state") in ("done", "error"):
            break
    out = {"node": body["node"], "state": last.get("state"), **({"result": last["result"]} if last.get("result") else {})}
    if last.get("error"):
        out["error"] = last["error"]
    if last.get("state") not in ("done", "error"):
        out["note"] = last.get("note") or "still running in the viewer; read the comp later (its node's results)"
    return out


@tool
def edit_comp(path: str = "", ops: list | None = None, create: bool = False, width: int = 1920, height: int = 1080,
              fps: float = 24, frames: int = 100, wait_s: int = 1800) -> str:
    """Read or change a comp of the ComfyUI Image Viewer's Comp space (a node graph of pixel operations in a
    ``*.comp.json``): add nodes, wire them, set and key their parameters, add Write nodes. The viewer saves the
    change and an open Comp tab shows it at once.

    Args:
        path: The .comp.json; "" for the comp open in the viewer's Comp tab.
        ops: Leave empty to READ it (format, output, nodes with their inputs and the parameters that differ from
            the defaults, keyed ones). To change it, a list, all done or none:
            ``{"op": "add", "type": "read|constant|merge|grade|blur|transform|write|metadata|...", "name", "inputs": [..],
              "x", "y", <param>: <value>}`` - its id comes back in ``ids``; later ops can name it "$0", "$1" (the
              first, second node added by this call; counted from 0).
            ``{"op": "connect", "node": .., "from": <node>, "port": <input name or index>}``,
            ``{"op": "disconnect", "node": .., "port": ..}``,
            ``{"op": "set", "node": .., "params": {..}}`` (also "name", "disable"),
            ``{"op": "key", "node": .., "param": .., "frame": .., "value": .., "ease": "smooth|linear|hold|..."}``,
            ``{"op": "unkey", "node": .., "param": .., "frame": ..}``, ``{"op": "remove", "node": ..}``,
            ``{"op": "output", "node": ..}`` (what the Output node shows), ``{"op": "format", "width", "height", "fps", "frames"}``.
            Nodes are named by id or by their (unique) name. A Read takes stills, films, sequences
            (shot.####.exr) and other comps (a precomp). A Write: file (#### = the frame number), file_type
            exr|png|tiff|jpg|mov|mp4|webm, channels rgba|rgb|all (all: every extra channel as EXR layers),
            depth, compression, first / last (comp frames, from 0), handles, first_number (1001 = comp frame 0).
            Some node types beyond the list: depthblur (inputs src, depth; channel, focus = the depth in focus
            on the map's own scale - read it off the map where the subject is -, dof, falloff, size, layers,
            filter gaussian | lens), defocus, glow, lensflare, lumakeyer, chromakeyer, erode, shuffle,
            colorcorrect. A Read's offset slides the footage: +10 starts it 10 frames later, -16 skips its
            first 16 frames. Filters and colour nodes have a last input "mask" (with mask_channel, mask_invert,
            mix): only where the mask is do they act.
            Depth of field: Read the shot, Read its depth (an AI Depth result or a depth film) and its subject's
            matte; depthblur(src = shot, depth = depth read, focus at the subject's depth) with mask = the matte
            and mask_invert true, so the subject stays sharp. Never a plain blur over the whole picture.
            The read-back lists every type there is as ``node_types``: look there before saying one is missing.
            AI nodes (ai_depth, ai_matte = BiRefNet-style cut-out, ai_upscale, ai_inpaint) make their pictures in
            a ComfyUI run; until run, their ``results`` is 0 and they show nothing. Run one with
            ``{"op": "run_ai", "node": .., "first"?: .., "last"?: ..}`` (comp frames, default all): the viewer
            open in the browser runs it on the node's own workflow, saves the comp and answers how many frames
            came back (or which model files are missing). It runs after the other ops of the call, so one call
            can add the node and run it. So depth of field is: read, ai_depth on it, ai_matte on it, run_ai both,
            then depthblur as above - no separate ComfyUI workflows needed.
            3D: ``{"op": "card_setup", "node": <a picture>, "renderer"?: "3D space|path traced|built-in",
            "camera"?: {translate, rotate, focal}}`` puts the picture on a Card in front of a Camera through a Render3D
            (the Render3D takes over what read the picture); it adds three nodes ($n counts them). Other 3D nodes:
            card3d / geo3d (shape cube|sphere|cylinder; translate, rotate [deg, X then Y then Z], scale, uniform),
            project3d (inputs img, camera; faces front|both) feeding a card / shape, scene3d (up to 4), scenefile3d
            (file = a .usda / .usdc / .json scene of the viewer's 3D space, or "tab:<3D tab name>"; offset),
            light3d (light point|spot|directional|ambient, color, intensity, cone), camera3d (focal, haperture,
            near, far; with a SceneFile in its scene input, from_scene = that scene's camera, followed every
            frame), render3d (inputs scene, camera, bg; renderer "3D space" (default) | "path traced" (rays,
            bounces) | "built-in" (cards and shapes only, unlit); depth.z written). Y is up, the camera looks down
            -Z, a card is one unit wide. The read-back's ``warnings`` names 3D nodes that draw nothing as wired:
            fix those. 3D-space and path traced frames are made by the viewer's open Comp tab; without one a
            render falls back to the built-in renderer and its notes say so.
        create: Make the comp when the file is not there (with width / height / fps / frames).
        wait_s: For run_ai: how long to wait for the viewer (max 3600 s each).

    Returns:
        JSON: {"status": "ok", "path", "comp": {format, output, nodes: [...]}, "ids"?: [new node ids]} or
        {"status": "error", "error"}.
    """
    path, why = _path(path)
    if why:
        return _j({"status": "error", "error": why})
    ops, why = _clean_ops(ops)
    if why:
        return _j({"status": "error", "error": why})
    runs = [o for o in ops if isinstance(o, dict) and o.get("op") == "run_ai"]
    ops = [o for o in ops if not (isinstance(o, dict) and o.get("op") == "run_ai")]
    if runs and not ops and not create:
        ran = [_run_ai(path, o, min(3600, int(wait_s))) for o in runs]
        bad = [x for x in ran if x.get("state") != "done"]
        return _j({"status": "error" if bad else "ok", "path": path, "ran": ran,
                   **({"error": "; ".join(f"{x['node']}: {x.get('error') or x.get('note')}" for x in bad)} if bad else {})})
    from agenty_core.utils.comfyui_client import get_client
    try:
        if not ops and not create:
            r = get_client().get("/imageviewer/comp_summary", params={"path": path})
            return _j({"status": "ok", "path": path, "comp": r})
        body = {"path": path, "ops": ops}
        if create:
            body["create"] = {"width": int(width), "height": int(height), "fps": float(fps), "frames": int(frames)}
        r = get_client().post("/imageviewer/comp_ops", json_data=body)
    except Exception as e:  # noqa: BLE001
        return _fail(e, "edit comps")
    if not isinstance(r, dict) or r.get("error"):
        return _j({"status": "error", "error": str((r or {}).get("error") if isinstance(r, dict) else r)[:400]})
    out = {"status": "ok", "path": r.get("path") or path, "ids": r.get("ids") or [], "comp": r.get("comp")}
    if runs:
        # "$0" names the first node this call added, as in the other ops.
        ids = out["ids"]
        for o in runs:
            ref = str(o.get("node") or "")
            if ref.startswith("$") and ref[1:].isdigit() and int(ref[1:]) < len(ids):
                o = {**o, "node": ids[int(ref[1:])]}
            out.setdefault("ran", []).append(_run_ai(out["path"], o, min(3600, int(wait_s))))
        bad = [x for x in out["ran"] if x.get("state") != "done"]
        if bad:
            out["status"] = "error"
            out["error"] = "; ".join(f"{x['node']}: {x.get('error') or x.get('note')}" for x in bad)
    return _j(out)


def _have_playwright() -> bool:
    try:
        import playwright.sync_api  # type: ignore[import]  # noqa: F401
        return True
    except ImportError:
        return False


class _Headless3D:
    """A headless browser on ComfyUI's page that draws a comp job's Render3D frames, for as long as it is needed.
    Everything Playwright does happens in its own thread (its sync API is bound to the thread it started in)."""

    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/?imageviewer_serve3d=1"
        self.stop = threading.Event()
        self.error = ""
        self.started = False
        self.thread = threading.Thread(target=self._run, name="comp-headless-3d", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import]
            with sync_playwright() as p:
                browser, last = None, None
                args = ["--ignore-gpu-blocklist", "--enable-unsafe-swiftshader", "--use-angle=default"]
                for channel in ("chrome", "msedge", None):
                    try:
                        kw = {"headless": True, "args": args}
                        if channel:
                            kw["channel"] = channel
                        browser = p.chromium.launch(**kw)
                        break
                    except Exception as e:  # noqa: BLE001
                        last = e
                if browser is None:
                    raise RuntimeError(f"no browser to start (tried Chrome, Edge, Playwright's chromium): {last}")
                try:
                    page = browser.new_page(viewport={"width": 1280, "height": 800})
                    page.goto(self.url, wait_until="domcontentloaded", timeout=60000)
                    self.started = True
                    while not self.stop.is_set():
                        page.wait_for_timeout(500)
                finally:
                    browser.close()
        except Exception as e:  # noqa: BLE001 - the job falls back to Comp's own renderer
            self.error = str(e).splitlines()[0][:300] if str(e) else type(e).__name__
            _log.warning("render_comp: headless browser for 3D frames: %s", self.error)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=15)


def _close_when_done(c, jid: str, browser: "_Headless3D", limit_s: float = 6 * 3600) -> None:
    end = time.time() + limit_s
    try:
        while time.time() < end and not browser.stop.is_set():
            if c.get("/imageviewer/comp_write", params={"job": jid}).get("state") != "running":
                break
            time.sleep(2.0)
    except Exception:  # noqa: BLE001 - ComfyUI went away: nothing left to draw for
        pass
    browser.close()


@tool
def render_comp(path: str = "", writes: str = "", first: int = -1, last: int = -1, handles: int = -1,
                farm: bool = False, chunk: int = 10, pool: str = "", group: str = "", priority: int = -1,
                wait_s: int = 600, dry_run: bool = False) -> str:
    """Render a comp's Write nodes - each writes its own files (EXR sequences with layers, PNG, films) - in
    ComfyUI, or send them to the Deadline render farm as one job in chunks of frames.

    Args:
        path: The .comp.json; "" for the comp open in the viewer. It is rendered as SAVED (edit_comp saves).
        writes: Write nodes by id or name, comma separated; "" = all that are switched on.
        first, last: comp frames (from 0); -1 = each Write's own range.
        handles: extra frames each side; -1 = each Write's own.
        farm: send to Deadline instead of rendering here (needs Deadline set up for agentY, and the comp, its
            footage and its outputs on paths the farm machines can reach - the answer warns about any that are not).
        chunk, pool, group, priority: the farm job's frames per task, pool, group, priority (0-100).
        wait_s: rendering here, wait this long for the files (max 3600); the job goes on if it takes longer.
            A comp with a Render3D draws its 3D frames with the viewer's 3D space renderer in a browser page: an
            open ComfyUI page with the viewer does it, and when none answers a headless browser is started on
            ComfyUI's page for the job (needs Playwright, as reverse image search does); without either, Comp's own
            renderer draws them (cards and shapes, unlit) and the notes say so.
        dry_run: say what would be written (and, with farm, the job) without rendering.

    Returns:
        JSON: here: {"status": "ok", "writes": [{name, frames, pattern, files}], "notes"} (or "running" with a job id);
        farm: {"status": "submitted", "job_id", "tasks", "warnings"}; or {"status": "error", "error"}.
    """
    path, why = _path(path)
    if why:
        return _j({"status": "error", "error": why})
    body = {"path": path, "writes": [w.strip() for w in (writes or "").split(",") if w.strip()] or None}
    for k, v in (("first", first), ("last", last), ("handles", handles)):
        if int(v) >= 0:
            body[k] = int(v)
    from agenty_core.utils.comfyui_client import get_client
    c = get_client()
    if farm:
        try:
            r = c.post("/imageviewer/comp_farm", json_data={**body, "dry_run": True, "chunk": int(chunk), "pool": pool,
                                                             "group": group, "priority": int(priority)})
        except Exception as e:  # noqa: BLE001
            return _fail(e, "send comps to the farm")
        if dry_run:
            return _j({"status": "ok", "dry_run": True, "request": r.get("request"), "warnings": r.get("warnings") or [],
                       "writes": r.get("writes") or []})
        from agenty_core.tools import deadline_farm as DF
        try:
            out = DF.submit_comp(r.get("request") or {})
        except DF.FarmError as e:
            return e.render()
        return _j({**out, "warnings": r.get("warnings") or []})
    try:
        if dry_run:
            r = c.post("/imageviewer/comp_write_plan", json_data=body)
            return _j({"status": "ok", "dry_run": True, "writes": r.get("writes") or [], "refused": r.get("refused")})
        headless_ok = _have_playwright()
        if headless_ok:
            body["wait3d"] = _WAIT3D
        r = c.post("/imageviewer/comp_write", json_data=body)
        jid = r.get("job")
        end = time.time() + max(0, min(3600, int(wait_s)))
        browser = None
        try:
            while True:
                s = c.get("/imageviewer/comp_write", params={"job": jid})
                if s.get("state") == "done":
                    res = s.get("result") or {}
                    out = {"status": "ok", "writes": res.get("writes") or [], "notes": res.get("notes") or []}
                    if browser is not None:
                        if browser.started and not browser.error:
                            out["drawn_3d_by"] = "a headless browser on ComfyUI's page"
                        if browser.error:
                            out["notes"].append(f"the headless browser for the 3D frames did not start: {browser.error}")
                    return _j(out)
                if s.get("state") != "running":
                    return _j({"status": "error", "error": s.get("error") or f"the render {s.get('state')}"})
                # A 3D frame no viewer page has taken: start a page of our own to draw it.
                if (headless_ok and browser is None and s.get("need3d") and not s.get("owner3d")
                        and float(s.get("waiting3d") or 0) >= _HEADLESS_AFTER):
                    browser = _Headless3D(c.base_url)
                if time.time() >= end:
                    if browser is not None:     # it draws on until the job ends
                        threading.Thread(target=_close_when_done, args=(c, jid, browser), daemon=True).start()
                        browser = None
                    return _j({"status": "running", "job": jid, "done": s.get("done"), "total": s.get("total"),
                               "hint": "it renders on; call again later to see the files"})
                time.sleep(1.0)
        finally:
            if browser is not None:
                browser.close()
    except Exception as e:  # noqa: BLE001
        return _fail(e, "render Write nodes")
