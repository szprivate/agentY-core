"""edit_image: read and change a picture in layers in the bEpic viewer's Paint tab through POST /bepic/paint/doc.

ComfyUI is mocked. The round trip runs the viewer's own paint_routes / paint_ops (the checkout beside this one,
or BEPIC_VIEWER_DIR) behind the mocked route, so a layer added here is read back from the .ora file.
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

from agenty_core.tools import edit_image as ei

_HERE = Path(__file__).resolve()
VIEWER = Path(os.environ.get("BEPIC_VIEWER_DIR") or (_HERE.parents[3].parent / "ComfyUI-ImageViewer"))


def _viewer(mod):
    name = "bepic_viewer_paint_for_core_tests"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(VIEWER)]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.{mod}")


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


class Arguments(Base):
    def test_a_file_that_is_not_a_picture_is_refused_before_asking(self):
        got = json.loads(ei.edit_image(doc="D:/o/cut.otio"))
        self.assertEqual(got["status"], "error")
        self.client.post.assert_not_called()

    def test_ops_as_a_json_string_and_a_single_dict(self):
        self.client.post.return_value = {"path": "D:/o/a.ora", "outline": {"w": 1}, "said": ["x"], "written": True, "token": "t"}
        got = json.loads(ei.edit_image(doc="D:/o/a.ora", ops='{"op": "flatten"}'))
        self.assertEqual(got["status"], "ok")
        body = self.client.post.call_args.kwargs["json_data"]
        self.assertEqual(body["ops"], [{"op": "flatten"}])
        self.assertTrue(got["pushed"] and got["written"])

    def test_the_open_picture_is_taken_from_the_viewer_unless_it_has_unsaved_changes(self):
        state = {"status": "ok", "state": {"paint": {"open": True, "path": "D:/o/p.ora", "dirty": True}}}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
            got = json.loads(ei.edit_image(ops=[{"op": "flatten"}]))
        self.assertEqual(got["status"], "error")
        self.assertIn("save", got["error"])
        state["state"]["paint"]["dirty"] = False
        self.client.post.return_value = {"path": "D:/o/p.ora", "outline": {}}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
            got = json.loads(ei.edit_image())
        self.assertEqual(got["status"], "ok")
        self.assertEqual(self.client.post.call_args.kwargs["json_data"]["doc"], "D:/o/p.ora")

    def test_nothing_open(self):
        state = {"status": "ok", "state": {}}
        with mock.patch("agenty_core.tools.viewer_state.read_state", return_value=json.dumps(state)):
            got = json.loads(ei.edit_image())
        self.assertIn("pass doc=", got["error"])

    def test_the_viewers_refusal_is_passed_on(self):
        err = Exception("400")
        err.response = mock.Mock(status_code=400, json=lambda: {"error": "change 2 (explode): unknown op"})
        self.client.post.side_effect = err
        got = json.loads(ei.edit_image(doc="D:/o/a.ora", ops=[{"op": "explode"}]))
        self.assertEqual(got["error"], "change 2 (explode): unknown op")


@unittest.skipUnless((VIEWER / "paint_routes.py").is_file(), "needs the viewer checkout (BEPIC_VIEWER_DIR)")
class RoundTrip(Base):
    def setUp(self):
        super().setUp()
        try:
            import numpy as np  # noqa: F401
            from PIL import Image
        except ImportError:
            self.skipTest("needs numpy and Pillow")
        self.R = _viewer("paint_routes")
        self.P = _viewer("paint_io")
        rules = _viewer("export_rules")
        out = self.tmp / "out"
        out.mkdir()
        self.png = str(self.tmp / "pic.png")
        Image.new("RGBA", (40, 30), (0, 0, 200, 255)).save(self.png)
        self.logo = str(self.tmp / "logo.png")
        Image.new("RGBA", (8, 8), (255, 255, 0, 255)).save(self.logo)

        def post(route, json_data):
            answer, tok = self.R.agent_ops(json_data["doc"], json_data.get("ops"), resolve=os.path.abspath,
                                           is_allowed=lambda p: True, may_serve=lambda p: True,
                                           export_rules=rules, output_dir=str(out))
            return {**answer, "token": tok}
        self.client.post.side_effect = post

    def test_a_picture_becomes_an_ora_with_the_new_layers(self):
        got = json.loads(ei.edit_image(doc=self.png, ops=[
            {"op": "add_layer", "file": self.logo, "x": 30, "y": 2, "name": "Logo"},
            {"op": "add_adjustment", "type": "hue_saturation", "params": {"saturation": -30}},
            {"op": "add_text", "text": "Title", "size": 10}]))
        self.assertEqual(got["status"], "ok", got)
        self.assertTrue(got["doc"].endswith(".ora"))
        self.assertEqual([r["name"] for r in got["outline"]["layers"]], ["Title", "Hue/Saturation", "Logo", "Background"])
        doc = self.P.read_ora(got["doc"])
        self.assertEqual(len(doc.layers), 4)
        self.assertTrue(os.path.isfile(self.png))          # the picture itself is not written over
        # Read back, then changed again in place.
        again = json.loads(ei.edit_image(doc=got["doc"], ops=[{"op": "set_layer", "layer": "Logo", "opacity": 50},
                                                               {"op": "move_layer", "layer": 1, "to": "bottom"}]))
        self.assertEqual(again["doc"], got["doc"])
        rows = json.loads(ei.edit_image(doc=got["doc"]))["outline"]["layers"]
        self.assertEqual(rows[-1]["name"], "Title")
        self.assertEqual(next(r for r in rows if r["name"] == "Logo")["opacity"], 50.0)


if __name__ == "__main__":
    unittest.main()
