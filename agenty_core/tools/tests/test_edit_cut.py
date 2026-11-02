"""edit_cut: read and change the cut in the bEpic viewer's Edit panel through POST /bepic/cut.

ComfyUI is mocked. The round trip runs the viewer's own timeline_io (the checkout beside this one,
or BEPIC_VIEWER_DIR) behind the mocked route, so a voice-over added at 8 s is read back from the
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
VIEWER = Path(os.environ.get("BEPIC_VIEWER_DIR") or (_HERE.parents[3].parent / "ComfyUI-ImageViewer"))


def _viewer_tio():
    name = "bepic_viewer_tio_for_core_tests"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(VIEWER)]
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

    def test_the_open_cut_comes_from_the_viewer_state_and_a_read_writes_nothing(self):
        state = {"status": "ok", "state": {"open": True, "edit": {"open": True, "otio": "W:/x/cut1.otio", "clips": 2}}}
        self.client.post.return_value = {"ok": True, "otio": "W:/x/cut1.otio", "edit": EDIT, "length": 14,
                                         "saved": False, "said": []}
        with mock.patch("agenty_core.tools.viewer_state.get_viewer_state", return_value=json.dumps(state)):
            out = json.loads(ec.edit_cut())
        args, kwargs = self.client.post.call_args
        self.assertEqual(args[0], "/bepic/cut")
        self.assertEqual(kwargs["json_data"], {"otio": "W:/x/cut1.otio", "ops": []})
        self.assertEqual((out["status"], out["otio"], out["length"]), ("ok", "W:/x/cut1.otio", 14.0))
        self.assertNotIn("said", out)

    def test_no_open_cut_an_unsaved_one_or_no_canvas_is_said(self):
        for edit, want in (({}, "no cut is open"), ({"open": True}, "no file yet")):
            state = {"status": "ok", "state": {"open": True, "edit": edit}}
            with mock.patch("agenty_core.tools.viewer_state.get_viewer_state", return_value=json.dumps(state)):
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
        self.assertEqual(args[0], "/bepic/cut?client_id=tab+1")
        self.assertEqual(kwargs["json_data"]["ops"], [op])
        self.assertEqual((out["said"], out["pushed"]), (["added vo on A2 at 8.0s for 6.0s"], True))
        json.loads(ec.edit_cut("W:/c.otio", json.dumps([op])))      # and as JSON text
        self.assertEqual(self.client.post.call_args[1]["json_data"]["ops"], [op])

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


@unittest.skipUnless((VIEWER / "timeline_io.py").is_file(), "needs the ComfyUI-ImageViewer checkout")
class RoundTrip(Base):
    """edit_cut -> (POST /bepic/cut, done here by the viewer's timeline_io) -> the .otio -> edit_cut."""

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
        """What viewer_api._bepic_cut does, minus the fence and the push."""
        body = json_data or {}
        if body.get("ops"):
            edit, said = self.tio.change(body["otio"], body["ops"], "", lambda p: True, {self.vo: 12.0})
        else:
            edit, said = self.tio.read_editable(body["otio"]), []
        return {"ok": True, "otio": body["otio"], "edit": edit, "length": self.tio.edit_length(edit),
                "saved": bool(body.get("ops")), "said": said, "pushed": True}

    def test_a_vo_at_8s_on_A2_reads_back_from_the_file(self):
        out = json.loads(ec.edit_cut(self.otio, [{"op": "add", "file": self.vo, "at": 8.0, "length": 9.5}]))
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(out["said"], ["added vo on A2 at 8.0s for 9.5s"])
        back = json.loads(ec.edit_cut(self.otio))
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


if __name__ == "__main__":
    unittest.main()
