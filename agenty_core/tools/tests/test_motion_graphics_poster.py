"""The poster PNG: scene round trip through PNG chunks, and render_scene's poster hand-off (no Node)."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

from PIL import Image

from agenty_core.motion_graphics import poster, render

SCENE = {"mgscene": 1, "comp": {"width": 16, "height": 9, "fps": 10, "duration": 20, "background": "#000000"},
         "layers": [{"id": "t", "type": "text", "text": "Grüße ✓"}]}


class PosterRoundTrip(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_write_then_read(self):
        png = os.path.join(self.d, "p.png")
        Image.new("RGB", (16, 9), (10, 20, 30)).save(png)
        poster.write_poster(png, SCENE, os.path.join(self.d, "a.mgscene.json"))
        scene, path = poster.read_scene_from_png(png)
        self.assertEqual(scene, SCENE)
        self.assertEqual(path, os.path.abspath(os.path.join(self.d, "a.mgscene.json")))
        self.assertEqual(Image.open(png).convert("RGB").getpixel((0, 0)), (10, 20, 30))
        self.assertFalse(os.path.exists(png + ".tmp"))

    def test_plain_png_has_no_scene(self):
        png = os.path.join(self.d, "plain.png")
        Image.new("RGB", (4, 4)).save(png)
        self.assertIsNone(poster.read_scene_from_png(png))
        self.assertIsNone(poster.read_scene_from_png(os.path.join(self.d, "missing.png")))

    def test_poster_frame(self):
        self.assertEqual(poster.poster_frame(SCENE), 10)
        s = {"comp": dict(SCENE["comp"], poster=3)}
        self.assertEqual(poster.poster_frame(s), 3)
        s = {"comp": dict(SCENE["comp"], poster=99)}
        self.assertEqual(poster.poster_frame(s), 10)


class RenderSceneHandOff(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self.scene_path = os.path.join(self.d, "a.mgscene.json")

    def _run(self, stills, **kw):
        with mock.patch.object(render, "_checked", return_value=(self.scene_path, SCENE)), \
                mock.patch.object(render, "_run_job", return_value={"out": os.path.join(self.d, "a.mp4")}), \
                mock.patch.object(render, "render_stills", side_effect=stills):
            return render.render_scene(self.scene_path, **kw)

    def test_poster_written_next_to_video(self):
        def stills(path, frames, prefix, **_kw):
            out = prefix + "-10.png"
            Image.new("RGB", (16, 9)).save(out)
            self.assertEqual(frames, [10])
            return [{"frame": 10, "out": out}]
        res = self._run(stills)
        self.assertEqual(res["poster"], os.path.join(self.d, "a.png"))
        self.assertEqual(poster.read_scene_from_png(res["poster"])[0], SCENE)

    def test_still_failure_keeps_the_video_result(self):
        res = self._run(render.RenderError("boom"))
        self.assertNotIn("poster", res)
        self.assertIn("boom", res["poster_error"])
        self.assertTrue(res["path"].endswith("a.mp4"))

    def test_opt_out(self):
        res = self._run(AssertionError("must not render"), poster=False)
        self.assertNotIn("poster", res)
        self.assertNotIn("poster_error", res)


if __name__ == "__main__":
    unittest.main()
