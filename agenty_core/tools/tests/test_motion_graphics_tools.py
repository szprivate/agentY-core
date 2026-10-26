"""mg_scene_validate / mg_render / mg_scene_preview_frame: the tool layer around the Remotion renderer.

The real render test runs only when the renderer's npm packages are installed (it is skipped otherwise)."""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from agenty_core.motion_graphics import render, schema
from agenty_core.tools import motion_graphics as mg

mg.AUTO_OPEN = False          # these tests must never push anything into a viewer that happens to be running

EXAMPLES = os.path.join(os.path.dirname(schema.__file__), "examples")
LOWER = os.path.join(EXAMPLES, "lower_third.mgscene.json")


def write(folder, scene, name="x.mgscene.json"):
    p = os.path.join(folder, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(scene, fh)
    return p


class ValidateTool(unittest.TestCase):
    def test_summary_of_a_good_scene(self):
        out = json.loads(mg.mg_scene_validate(LOWER))
        self.assertTrue(out["ok"])
        self.assertEqual(out["summary"]["size"], [1920, 1080])
        self.assertEqual(out["summary"]["seconds"], 3.0)
        self.assertIn("name (text)", out["summary"]["layers"])

    def test_errors_say_where(self):
        d = tempfile.mkdtemp()
        p = write(d, {"mgscene": 1, "comp": {"width": 1280, "height": 720, "fps": 30, "duration": 30},
                      "layers": [{"id": "t", "type": "text"}]})
        out = json.loads(mg.mg_scene_validate(p))
        self.assertFalse(out["ok"])
        self.assertEqual(out["errors"][0]["code"], "E_TEXT")
        self.assertEqual(out["errors"][0]["path"], "layers[0].text")

    def test_missing_and_broken_files(self):
        self.assertEqual(json.loads(mg.mg_scene_validate("C:/nope/x.mgscene.json"))["errors"][0]["code"], "E_FILE")
        d = tempfile.mkdtemp()
        p = os.path.join(d, "bad.mgscene.json")
        open(p, "w").write("{not json")
        self.assertIn("not valid JSON", json.loads(mg.mg_scene_validate(p))["errors"][0]["message"])


class AutoOpen(unittest.TestCase):
    """A scene the agent writes opens in the viewer's Motion editor by itself - once."""

    def setUp(self):
        mg.AUTO_OPEN = True
        mg._opened.clear()
        self.addCleanup(setattr, mg, "AUTO_OPEN", False)

    def test_first_valid_check_pushes_the_scene_once(self):
        from agenty_core.tools import show_in_viewer as siv
        with mock.patch.object(siv, "open_in_viewer", return_value={"ok": True, "motion": [LOWER]}) as push:
            first = json.loads(mg.mg_scene_validate(LOWER))
            json.loads(mg.mg_scene_validate(LOWER))
        push.assert_called_once_with([os.path.abspath(LOWER)], focus=True)
        self.assertTrue(first["opened_in_viewer"])

    def test_an_invalid_scene_is_not_pushed_and_a_missing_viewer_is_retried(self):
        from agenty_core.tools import show_in_viewer as siv
        d = tempfile.mkdtemp()
        bad = write(d, {"mgscene": 1, "comp": {"width": 1280, "height": 720, "fps": 30, "duration": 30}, "layers": [{"type": "text"}]})
        with mock.patch.object(siv, "open_in_viewer") as push:
            mg.mg_scene_validate(bad)
        push.assert_not_called()
        with mock.patch.object(siv, "open_in_viewer", return_value={"error": "could not reach the viewer"}) as push:
            mg.mg_scene_validate(LOWER)
            mg.mg_scene_validate(LOWER)
        self.assertEqual(push.call_count, 2)

    def test_off_means_off(self):
        from agenty_core.tools import show_in_viewer as siv
        mg.AUTO_OPEN = False
        with mock.patch.object(siv, "open_in_viewer") as push:
            mg.mg_scene_validate(LOWER)
        push.assert_not_called()


class RenderTools(unittest.TestCase):
    def test_an_invalid_scene_is_never_rendered(self):
        d = tempfile.mkdtemp()
        p = write(d, {"mgscene": 1, "comp": {"width": 1280, "height": 720, "fps": 30, "duration": 30}, "layers": [{"type": "text"}]})
        with mock.patch.object(render, "_run_job") as run:
            out = json.loads(mg.mg_render(p))
        run.assert_not_called()
        self.assertIn("not valid", out["error"])

    def test_render_job_and_default_output(self):
        with mock.patch.object(render, "_run_job", return_value={"out": "O.mp4", "frames": 90, "width": 1920, "height": 1080}) as run:
            out = json.loads(mg.mg_render(LOWER, frame_start=10, scale=0.5, poster=False))
        job = run.call_args.args[0]
        self.assertEqual(job["mode"], "video")
        self.assertTrue(job["out"].endswith("lower_third.mp4"))
        self.assertEqual((job["range"], job["scale"]), ([10, 89], 0.5))
        self.assertEqual(out["path"], "O.mp4")

    def test_preview_clamps_frames_and_builds_a_sheet(self):
        d = tempfile.mkdtemp()
        from PIL import Image
        stills = []
        for f in (0, 45, 89):
            pth = os.path.join(d, f"s_f{f:04d}.png")
            Image.new("RGB", (64, 36), (f, 0, 0)).save(pth)
            stills.append({"frame": f, "out": pth})
        with mock.patch.object(render, "render_stills", return_value=stills) as rs:
            out = json.loads(mg.mg_scene_preview_frame(LOWER, frames=[-5, 45, 9999]))
        self.assertEqual(rs.call_args.args[1], [0, 45, 89])
        self.assertTrue(os.path.isfile(out["sheet"]))

    def test_no_node_is_a_plain_sentence(self):
        with mock.patch("shutil.which", return_value=None):
            out = json.loads(mg.mg_render(LOWER))
        self.assertIn("Node.js", out["error"])


@unittest.skipUnless(shutil.which("node") and render.is_installed(), "the renderer's npm packages are not installed")
class RealRender(unittest.TestCase):
    def test_renders_a_few_frames_to_mp4(self):
        d = tempfile.mkdtemp()
        out = json.loads(mg.mg_render(LOWER, os.path.join(d, "o.mp4"), frame_start=0, frame_end=5, scale=0.25))
        self.assertNotIn("error", out, out)
        self.assertEqual(out["frames"], 6)
        self.assertGreater(os.path.getsize(out["path"]), 1000)

    def test_local_image_asset_is_served_to_the_render(self):
        from PIL import Image
        d = tempfile.mkdtemp()
        Image.new("RGB", (64, 64), (255, 0, 0)).save(os.path.join(d, "red.png"))
        scene = {"mgscene": 1, "comp": {"width": 160, "height": 90, "fps": 10, "duration": 5, "background": "#000000"},
                 "assets": {"red": {"type": "image", "src": "red.png", "width": 64, "height": 64}},
                 "layers": [{"id": "i", "type": "image", "asset": "red", "position": [80, 45], "size": [64, 64]}]}
        p = write(d, scene)
        res = render.render_stills(p, [2], os.path.join(d, "still"), scale=1)
        from PIL import Image as I
        px = I.open(res[0]["out"]).convert("RGB").getpixel((80, 45))
        self.assertGreater(px[0], 200)
        self.assertLess(px[1], 40)


if __name__ == "__main__":
    unittest.main()
