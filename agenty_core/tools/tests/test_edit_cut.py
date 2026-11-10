"""edit_cut: read and change the cut in the ComfyUI Image Viewer's Edit panel through POST /imageviewer/cut.

ComfyUI is mocked. The round trip runs the viewer's own timeline_io (the checkout beside this one,
or IMAGE_VIEWER_DIR) behind the mocked route, so a voice-over added at 8 s is read back from the
.otio file at 8 s on A2.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import edit_cut as ec

_HERE = Path(__file__).resolve()
VIEWER = Path(os.environ.get("IMAGE_VIEWER_DIR") or (_HERE.parents[3].parent / "ComfyUI-ImageViewer"))
# The viewer's Python modules: in src/ in newer checkouts, at the checkout root in older ones.
VIEWER_PY = VIEWER / "src" if (VIEWER / "src" / "__init__.py").is_file() else VIEWER


def _viewer_tio():
    name = "image_viewer_tio_for_core_tests"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(VIEWER_PY)]
        sys.modules[name] = pkg
    return importlib.import_module(name + ".timeline_io")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.client = mock.Mock()
        p = mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=self.client)
        p.start()
        self.addCleanup(p.stop)
        for name, value in (("client_id", ""), ("detached", False)):
            q = mock.patch(f"agenty_core.utils.canvas_scope.{name}", return_value=value)
            q.start()
            self.addCleanup(q.stop)


EDIT = {"name": "cut1", "fps": 24, "clips": [
    {"ref": 0, "name": "a", "file": "D:/o/a.mp4", "in": 1, "length": 3, "transition": 0, "volume": 1},
    {"ref": 1, "name": "b", "file": "D:/o/b.mp4", "in": 0, "length": 2.5, "transition": 0.5, "volume": 1, "gap": 1}],
    "texts": [], "sound": [{"name": "bed", "file": "D:/o/music.wav"}],
    "tracks": [{"id": "A2", "kind": "audio", "name": "A2", "off": False,
                "clips": [{"ref": [2, 1], "name": "vo", "file": "D:/o/vo.mp3", "in": 0, "length": 6, "at": 8, "volume": 1.2}]}]}


class Reading(Base):
    def test_outline_gives_times_in_the_film_on_every_track(self):
        tracks = ec.outline(EDIT)
        self.assertEqual([t["track"] for t in tracks], ["V1", "A1", "A2"])
        v1 = tracks[0]["clips"]
        self.assertEqual([(c["clip"], c["at"], c["end"]) for c in v1], [(1, 0.0, 3.0), (2, 3.5, 6.0)])
        self.assertEqual((v1[1]["dissolve"], v1[1]["gap_before"]), (0.5, 1.0))
        self.assertEqual(tracks[1]["clips"], [{"clip": 1, "name": "bed", "file": "D:/o/music.wav"}])
        self.assertEqual(tracks[2]["clips"][0], {"clip": 1, "name": "vo", "file": "D:/o/vo.mp3", "at": 8.0, "end": 14.0,
                                                 "in": 0.0, "length": 6.0, "volume": 1.2})
        self.assertNotIn("ref", json.dumps(tracks))

    def test_a_title_clip_with_no_file_is_described_by_its_words(self):
        edit = dict(EDIT, tracks=EDIT["tracks"] + [{"id": "V2", "kind": "video", "name": "V2", "clips": [
            {"ref": [3, 1], "name": "", "file": "", "in": 0, "length": 2, "at": 1.5,
             "title": {"text": "Mara Lindqvist\nKitchen", "font": "Inter"}},
            {"ref": [3, 2], "name": "", "file": "", "in": 0, "length": 1, "at": 5, "title": {"text": ""}}]}])
        v2 = ec.outline(edit)[3]["clips"]
        self.assertEqual((v2[0]["name"], v2[0]["file"], v2[0]["at"], v2[0]["end"], v2[0]["is_title"]),
                         ('title "Mara Lindqvist"', "", 1.5, 3.5, True))
        self.assertEqual(v2[1]["name"], "title")
        json.dumps(ec.outline(edit))

    def test_locks_fades_and_crossfades_are_shown(self):
        edit = json.loads(json.dumps(EDIT))
        edit["lock"] = {"V1": False, "A1": True, "T": True}
        edit["auto"] = {"V1": False, "A1": True, "T": True}
        edit["clips"][1].update({"fadeOut": 0.5, "xfade": {"len": 1, "curve": "0", "align": "start"}})
        edit["tracks"][0].update({"lock": True, "auto": False})
        tracks = {t["track"]: t for t in ec.outline(edit)}
        self.assertNotIn("locked", tracks["V1"])
        self.assertTrue(tracks["V1"]["auto_select_off"])
        self.assertTrue(tracks["A1"]["locked"])
        self.assertTrue(tracks["A2"]["locked"] and tracks["A2"]["auto_select_off"])
        self.assertEqual(tracks["V1"]["clips"][1]["fade_out"], 0.5)
        self.assertEqual(tracks["V1"]["crossfades"], [{
            "from_clip": 1, "into_clip": 2, "between": "a -> b", "length": 1.0, "curve": "0", "align": "start"}])
        self.assertNotIn("crossfades", tracks["A2"])
        self.client.post.return_value = {"ok": True, "otio": "W:/c.otio", "edit": edit, "length": 14}
        out = json.loads(ec.edit_cut("W:/c.otio"))
        self.assertTrue(out["texts_locked"])

    def test_sound_effects_are_shown_on_clips_and_tracks(self):
        edit = json.loads(json.dumps(EDIT))
        edit["clips"][0]["afx"] = [{"type": "eq", "on": True, "p": {}}, {"type": "reverb", "on": False, "p": {}}]
        edit["tracks"][0]["afx"] = [{"type": "comp", "on": True, "p": {}}]
        edit["afx"] = {"A1": [{"type": "limit", "on": True, "p": {}}]}
        tracks = {t["track"]: t for t in ec.outline(edit)}
        self.assertEqual(tracks["V1"]["clips"][0]["effects"], ["EQ", "Reverb (bypassed)"])
        self.assertNotIn("effects", tracks["V1"]["clips"][1])
        self.assertEqual(tracks["A1"]["effects"], ["Limiter"])
        self.assertEqual(tracks["A2"]["effects"], ["Compressor"])
        self.assertNotIn("effects", tracks["V1"])
        json.dumps(ec.outline(edit))

    def test_the_open_cut_comes_from_the_viewer_state_and_a_read_writes_nothing(self):
        state = {"status": "ok", "state": {"open": True, "edit": {"open": True, "otio": "W:/x/cut1.otio", "clips": 2}}}
        self.client.post.return_value = {"ok": True, "otio": "W:/x/cut1.otio", "edit": EDIT, "length": 14,
                                         "saved": False, "said": []}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
            out = json.loads(ec.edit_cut())
        args, kwargs = self.client.post.call_args
        self.assertEqual(args[0], "/imageviewer/cut")
        self.assertEqual(kwargs["json_data"], {"otio": "W:/x/cut1.otio", "ops": []})
        self.assertEqual((out["status"], out["otio"], out["length"]), ("ok", "W:/x/cut1.otio", 14.0))
        self.assertNotIn("said", out)

    def test_the_marks_on_the_cut_are_sent_and_their_sources_come_back(self):
        state = {"status": "ok", "state": {
            "open": True, "tab": {"key": "edit:a", "mode": "edit"},
            "timeline": {"fps": 24, "range": {"start": 242, "end": 273}},
            "edit": {"open": True, "tab": "edit:a", "otio": "W:/x/cut1.otio",
                     "selected": [{"name": "m", "file": "W:/m.mp4", "in": 3.458, "out": 4.792, "track": "V1"}]}}}
        piece = {"track": "V1", "clip": 5, "file": "W:/m.mp4", "rec_in": 242, "rec_out": 265,
                 "src_in": 83, "src_out": 106, "src_in_s": 3.4583, "src_seconds": 1.0}
        self.client.post.return_value = {"ok": True, "otio": "W:/x/cut1.otio", "edit": EDIT, "length": 14,
                                         "marked": {"start": 242, "end": 273, "fps": 24, "pieces": [piece]}}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
            out = json.loads(ec.edit_cut())
        self.assertEqual(self.client.post.call_args[1]["json_data"]["range"], {"start": 242, "end": 273})
        self.assertEqual(out["marked"]["pieces"], [piece])
        self.assertIn("never work them out yourself", out["marked"]["note"])
        # Marks on another tab are frames of that tab's clip, not of the cut: none sent.
        state["state"]["tab"] = {"key": "named:agentY", "mode": "video"}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
            out = json.loads(ec.edit_cut())
        self.assertNotIn("range", self.client.post.call_args[1]["json_data"])
        self.assertNotIn("marked", out)

    def test_a_range_given_by_hand_and_an_old_viewer(self):
        self.client.post.return_value = {"ok": True, "otio": "W:/c.otio", "edit": EDIT, "length": 14}
        out = json.loads(ec.edit_cut("W:/c.otio", frame_in=10, frame_out=20))
        self.assertEqual(self.client.post.call_args[1]["json_data"]["range"], {"start": 10, "end": 20})
        self.assertIn("update ComfyUI-ImageViewer", out["marked"]["error"])
        self.assertIn("both", json.loads(ec.edit_cut("W:/c.otio", frame_in=10))["error"])
        self.assertIn("whole numbers", json.loads(ec.edit_cut("W:/c.otio", frame_in="x", frame_out=3))["error"])

    def test_viewer_state_carries_the_marked_sources(self):
        from agenty_core.tools import viewer_state as vs
        raw = {"status": "ok", "state": {
            "open": True, "tab": {"key": "edit:a", "mode": "edit"},
            "timeline": {"fps": 24, "range": {"start": 242, "end": 273}},
            "edit": {"open": True, "tab": "edit:a", "otio": "W:/x/cut1.otio",
                     "selected": [{"name": "m", "file": "W:/m.mp4", "in": 3.458, "out": 4.792}]}}}
        self.client.post.return_value = {"ok": True, "otio": "W:/x/cut1.otio", "edit": EDIT, "length": 14,
                                         "marked": {"start": 242, "end": 273, "pieces": [{"src_in": 83}]}}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(raw)):
            got = json.loads(vs.get_viewer_state())
        edit = got["state"]["edit"]
        self.assertEqual(edit["marked"]["pieces"], [{"src_in": 83}])
        self.assertIn("WHOLE clip", edit["selected_note"])
        args, kwargs = self.client.post.call_args
        self.assertEqual(kwargs["json_data"], {"otio": "W:/x/cut1.otio", "ops": [], "range": {"start": 242, "end": 273}})
        # No marks: nothing asked of the cut.
        self.client.post.reset_mock()
        raw["state"]["timeline"] = {"fps": 24}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(raw)):
            got = json.loads(vs.get_viewer_state())
        self.assertNotIn("marked", got["state"]["edit"])
        self.client.post.assert_not_called()
        self.assertIn("(cut frames; source frames: edit_cut marked)",
                      vs.describe_lines({"open": True, "tab": {"mode": "edit"},
                                         "timeline": {"frames": 600, "current": 3, "range": {"start": 1, "end": 9}}}))

    def test_no_open_cut_an_unsaved_one_or_no_canvas_is_said(self):
        for edit, want in (({}, "no cut is open"), ({"open": True}, "no file yet")):
            state = {"status": "ok", "state": {"open": True, "edit": edit}}
            with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
                self.assertIn(want, json.loads(ec.edit_cut())["error"])
        with mock.patch("agenty_core.utils.canvas_scope.detached", return_value=True):
            self.assertIn("otio=", json.loads(ec.edit_cut())["error"])
        self.client.post.assert_not_called()


class Changing(Base):
    def test_ops_are_posted_with_the_client_id_and_said_comes_back(self):
        self.client.post.return_value = {"ok": True, "otio": "W:/c.otio", "edit": EDIT, "length": 14, "saved": True,
                                         "said": ["added vo on A2 at 8.0s for 6.0s"], "pushed": True}
        op = {"op": "add", "file": "D:/o/vo.mp3", "at": 8}
        with mock.patch("agenty_core.utils.canvas_scope.client_id", return_value="tab 1"):
            out = json.loads(ec.edit_cut("W:/c.otio", op))           # one op as a dict is a list of one
        args, kwargs = self.client.post.call_args
        self.assertEqual(args[0], "/imageviewer/cut?client_id=tab+1")
        self.assertEqual(kwargs["json_data"]["ops"], [op])
        self.assertEqual((out["said"], out["pushed"]), (["added vo on A2 at 8.0s for 6.0s"], True))
        json.loads(ec.edit_cut("W:/c.otio", json.dumps([op])))      # and as JSON text
        self.assertEqual(self.client.post.call_args[1]["json_data"]["ops"], [op])

    def test_an_op_on_a_locked_track_is_refused_before_it_is_sent(self):
        edit = json.loads(json.dumps(EDIT))
        edit["tracks"][0]["lock"] = True
        self.client.post.return_value = {"ok": True, "otio": "W:/c.otio", "edit": edit, "length": 14}
        out = json.loads(ec.edit_cut("W:/c.otio", [{"op": "set", "track": "a2", "clip": 1, "at": 9}]))
        self.assertEqual(out["status"], "error")
        self.assertIn("A2 is locked", out["error"])
        self.assertIn("unlock", out["error"])
        for call in self.client.post.call_args_list:
            self.assertEqual(call[1]["json_data"]["ops"], [])        # nothing with ops went out
        # An unlocked track goes through (read first, then the change).
        self.client.post.reset_mock()
        out = json.loads(ec.edit_cut("W:/c.otio", [{"op": "set", "track": "V1", "clip": 1, "length": 2}]))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(self.client.post.call_args[1]["json_data"]["ops"][0]["track"], "V1")

    def test_a_viewer_refusal_about_a_locked_track_comes_through(self):
        resp = mock.Mock(status_code=422)
        resp.json.return_value = {"ok": False, "error": "add: track A2 is locked in the Edit panel"}
        err = Exception("422")
        err.response = resp
        self.client.post.side_effect = err
        out = json.loads(ec.edit_cut("W:/c.otio", [{"op": "add", "file": "D:/o/vo.mp3"}]))
        self.assertIn("A2 is locked", out["error"])
        self.client.post.side_effect = None
        self.client.post.return_value = {"ok": False, "error": "remove: V1 is locked"}
        out = json.loads(ec.edit_cut("W:/c.otio", [{"op": "remove", "file": "x"}]))
        self.assertIn("locked", out["error"])
        self.assertIn("unlock", out["error"])

    def test_refusals_the_route_gives_and_an_old_viewer(self):
        resp = mock.Mock(status_code=422)
        resp.json.return_value = {"ok": False, "error": "not changed: change 1 (add): A1 takes no clip placed at a time"}
        self.client.post.side_effect = Exception("422")
        self.client.post.side_effect.response = resp
        self.assertIn("A1 takes no clip", json.loads(ec.edit_cut("W:/c.otio", [{"op": "add"}]))["error"])
        gone = mock.Mock(status_code=404)
        gone.json.side_effect = ValueError("not json")
        err = Exception("404 Client Error")
        err.response = gone
        self.client.post.side_effect = err
        self.assertIn("predates edit_cut", json.loads(ec.edit_cut("W:/c.otio"))["error"])
        self.assertIn("not a cut", json.loads(ec.edit_cut("W:/c.mp4"))["error"])
        self.assertIn("Refused", json.loads(ec.edit_cut("W:/c.otio", [{"op": "add", "file": str(self.tmp / ".env")}]))["error"])
        self.assertEqual(json.loads(ec.edit_cut("W:/c.otio", "not json"))["status"], "error")


@unittest.skipUnless((VIEWER_PY / "timeline_io.py").is_file(), "needs the ComfyUI-ImageViewer checkout")
class RoundTrip(Base):
    """edit_cut -> (POST /imageviewer/cut, done here by the viewer's timeline_io) -> the .otio -> edit_cut."""

    def setUp(self):
        super().setUp()
        self.tio = _viewer_tio()
        if not hasattr(self.tio, "change"):
            self.skipTest("the viewer checkout predates timeline_io.change")
        for name in ("shot_01.mp4", "shot_02.mp4", "vo.mp3"):
            (self.tmp / name).write_bytes(b"\0" * 16)
        self.shot1, self.shot2, self.vo = (str(self.tmp / n).replace("\\", "/") for n in ("shot_01.mp4", "shot_02.mp4", "vo.mp3"))
        doc = self.tio.rebuild(self.tio.new_doc("cut1", 24), {"clips": [
            {"file": self.shot1, "name": "shot_01", "in": 0, "length": 8, "transition": 0},
            {"file": self.shot2, "name": "shot_02", "in": 2, "length": 10, "transition": 0}]}, "", lambda p: True)
        self.otio = str(self.tmp / "cut1.otio")
        Path(self.otio).write_text(json.dumps(doc), encoding="utf-8")
        self.client.post.side_effect = self.route

    def route(self, path, json_data=None, **_):
        """What viewer_api._imageviewer_cut does, minus the fence and the push."""
        body = json_data or {}
        if body.get("ops"):
            edit, said = self.tio.change(body["otio"], body["ops"], "", lambda p: True, {self.vo: 12.0})
        else:
            edit, said = self.tio.read_editable(body["otio"]), []
        out = {"ok": True, "otio": body["otio"], "edit": edit, "length": self.tio.edit_length(edit),
               "saved": bool(body.get("ops")), "said": said, "pushed": True}
        if body.get("range") and hasattr(self.tio, "source_map"):
            r = body["range"]
            out["marked"] = self.tio.source_map(edit, r["start"], r["end"], {self.shot1: 24.0, self.shot2: 30.0})
        return out

    def test_the_marked_in_out_comes_back_as_frames_of_the_files(self):
        if not hasattr(self.tio, "source_map"):
            self.skipTest("the viewer checkout predates timeline_io.source_map")
        # shot_02 starts 8 s into the film, 2 s into its file; it is 30 fps, the cut 24.
        # Marked: film frames 204..227 (8.5 s .. 9.5 s) = 2.5 s .. 3.5 s of shot_02 = its frames 75..104.
        state = {"status": "ok", "state": {"open": True, "tab": {"key": "edit:x", "mode": "edit"},
                                           "timeline": {"fps": 24, "range": {"start": 204, "end": 227}},
                                           "edit": {"open": True, "tab": "edit:x", "otio": self.otio}}}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
            out = json.loads(ec.edit_cut())
        p, = out["marked"]["pieces"]
        self.assertEqual((p["clip"], p["file"], p["rec_in"], p["rec_out"]), (2, self.shot2, 204, 227))
        self.assertEqual((p["src_fps"], p["src_in"], p["src_out"], p["src_in_s"], p["src_seconds"]),
                         (30.0, 75, 104, 2.5, 1.0))
        self.assertIn("never", out["marked"]["note"])
        # A range over the cut between the two shots: two pieces, each in its own file.
        out = json.loads(ec.edit_cut(self.otio, frame_in=180, frame_out=203))
        a, b = out["marked"]["pieces"]
        self.assertEqual((a["file"], a["src_in"], a["src_out"]), (self.shot1, 180, 191))
        self.assertEqual((b["file"], b["src_in"], b["src_out"]), (self.shot2, 60, 74))

    def test_a_vo_at_8s_on_A2_reads_back_from_the_file(self):
        out = json.loads(ec.edit_cut(self.otio, [{"op": "add", "file": self.vo, "at": 8.0, "length": 9.5}]))
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(out["said"], ["added vo on A2 at 8.0s for 9.5s"])
        back = json.loads(ec.edit_cut(self.otio))
        self.assertNotIn("marked", back)                  # nothing marked, nothing mapped
        a2, = [t for t in back["tracks"] if t["track"] == "A2"]
        self.assertEqual([(c["file"], c["at"], c["end"]) for c in a2["clips"]], [(self.vo, 8.0, 17.5)])
        v1, = [t for t in back["tracks"] if t["track"] == "V1"]
        self.assertEqual([(c["file"], c["at"], c["end"]) for c in v1["clips"]], [(self.shot1, 0.0, 8.0), (self.shot2, 8.0, 18.0)])
        # In the OTIO file itself: a track A2 after A1 with 8 s of Gap and then the clip.
        doc = json.loads(Path(self.otio).read_text(encoding="utf-8"))
        tracks = {t["name"]: t for t in doc["tracks"]["children"]}
        self.assertEqual(list(tracks), ["V1", "A1", "A2"])
        gap, clip = tracks["A2"]["children"]
        self.assertEqual((gap["OTIO_SCHEMA"], gap["source_range"]["duration"]["value"]), ("Gap.1", 192.0))
        self.assertTrue(clip["media_reference"]["target_url"].endswith("/vo.mp3"))

    def test_a_vo_in_pieces_then_trimmed_and_one_taken_out(self):
        pieces = [{"op": "add", "file": self.vo, "in": 0.0, "length": 2.0, "at": 0.0},
                  {"op": "add", "file": self.vo, "in": 2.75, "length": 4.25, "at": 2.0},
                  {"op": "add", "file": self.vo, "in": 8.0, "at": 6.25}]           # to the end of the file
        out = json.loads(ec.edit_cut(self.otio, pieces))
        a2, = [t for t in out["tracks"] if t["track"] == "A2"]
        self.assertEqual([(c["at"], c["in"], c["length"]) for c in a2["clips"]], [(0.0, 0.0, 2.0), (2.0, 2.75, 4.25), (6.25, 8.0, 4.0)])
        self.assertEqual([t["track"] for t in out["tracks"]], ["V1", "A1", "A2"], "they fit one after another: one track")
        # A piece that would overlap the one before goes on a track of its own, not over it.
        over = json.loads(ec.edit_cut(self.otio, [{"op": "add", "file": self.vo, "at": 5.0, "length": 1.0}]))
        self.assertEqual([(t["track"], len(t["clips"])) for t in over["tracks"][2:]], [("A2", 3), ("A3", 1)])
        ec.edit_cut(self.otio, [{"op": "remove", "track": "A3", "clip": 1}])
        out = json.loads(ec.edit_cut(self.otio, [{"op": "set", "track": "A2", "clip": 3, "length": 3.0},
                                                 {"op": "remove", "track": "A2", "clip": 1},
                                                 {"op": "set", "track": "V1", "clip": 2, "length": 6.0}]))
        a2, = [t for t in out["tracks"] if t["track"] == "A2"]
        self.assertEqual([(c["at"], c["length"]) for c in a2["clips"]], [(2.0, 4.25), (6.25, 3.0)])
        self.assertEqual(out["length"], 14.0)

    def test_insert_from_the_source_markers_a_dissolve_a_retime_and_the_master(self):
        if not hasattr(self.tio, "_ops"):
            self.skipTest("the viewer checkout predates insert / markers / mix")
        out = json.loads(ec.edit_cut(self.otio, [
            {"op": "insert", "file": self.shot2, "in": 4.0, "length": 2.0, "at": 3.0},
            {"op": "add_marker", "at": 1.0, "name": "hand", "note": "fix the fingers", "color": "red"},
            {"op": "set", "track": "V1", "clip": 4, "transition": 1.0, "transition_look": {"type": "dip"}},
            {"op": "set", "track": "V1", "clip": 2, "speed": 2.0},
            {"op": "mix", "track": "master", "gain_db": -6}]))
        self.assertEqual(out["status"], "ok", out)
        v1, = [t for t in out["tracks"] if t["track"] == "V1"]
        self.assertEqual([(c["name"], c["at"], c["length"]) for c in v1["clips"]],
                         [("shot_01", 0.0, 3.0), ("shot_02", 3.0, 1.0), ("shot_01", 4.0, 5.0), ("shot_02", 8.0, 10.0)])
        self.assertEqual(v1["clips"][1]["speed"], 2.0)
        self.assertEqual((v1["clips"][3]["dissolve"], v1["clips"][3]["dissolve_look"]), (1.0, {"type": "dip", "color": "#000000"}))
        self.assertEqual(out["markers"], [{"marker": 1, "at": 1.0, "name": "hand", "note": "fix the fingers", "color": "red"}])
        self.assertEqual(out["master"], {"fader_db": -6.0})
        back = json.loads(ec.edit_cut(self.otio))
        self.assertEqual(back["markers"], out["markers"])
        self.assertEqual(back["master"], out["master"])
        out = json.loads(ec.edit_cut(self.otio, [{"op": "remove_marker", "marker": 1},
                                                 {"op": "mix", "track": "master", "gain_db": 0, "mute": True}]))
        self.assertNotIn("markers", out)
        self.assertEqual(out["master"], {"muted": True})


if __name__ == "__main__":
    unittest.main()
