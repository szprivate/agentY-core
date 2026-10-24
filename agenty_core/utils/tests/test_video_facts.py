"""video_facts: measured clip signals on synthetic frames (static, panning, zooming, empty, frozen, black)."""

import unittest

import numpy as np
from PIL import Image

from agenty_core.utils import video_facts as vf


def texture(w=512, h=288, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (h // 8, w // 8, 3), dtype=np.uint8)
    im = Image.fromarray(base).resize((w, h), Image.BICUBIC)
    fine = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
    return np.clip(np.asarray(im).astype(int) // 2 + fine, 0, 255).astype(np.uint8)


class Clips(unittest.TestCase):
    def test_static_camera(self):
        t = texture()
        rng = np.random.default_rng(1)
        frames = [np.clip(t.astype(int) + rng.integers(-1, 2, t.shape), 0, 255).astype(np.uint8) for _ in range(12)]
        m = vf.measure_frames(frames, fps=24)
        self.assertEqual(m["camera"]["state"], "static")
        self.assertFalse(m["empty_fraction"]["void_frames"])

    def test_pan_right_is_content_moving_left(self):
        t = texture(1024, 288)
        frames = [t[:, 8 * i:8 * i + 512] for i in range(12)]          # window slides right = camera pans right
        m = vf.measure_frames(frames, fps=24)
        self.assertEqual(m["camera"]["state"], "moving")
        self.assertIn("pan right", m["camera"]["direction"])
        self.assertLess(m["motion"]["content_shift_pct_width_per_s"]["x"], -5)

    def test_pan_left(self):
        t = texture(1024, 288)
        frames = [t[:, 400 - 8 * i:400 - 8 * i + 512] for i in range(12)]
        m = vf.measure_frames(frames, fps=24)
        self.assertIn("pan left", m["camera"]["direction"])

    def test_tilt(self):
        t = texture(512, 600)
        frames = [t[8 * i:8 * i + 288, :] for i in range(12)]          # window slides down = camera tilts down
        m = vf.measure_frames(frames, fps=24)
        self.assertIn("tilt down", m["camera"]["direction"])

    def test_push_in(self):
        t = Image.fromarray(texture(512, 288))
        frames = []
        for i in range(12):
            s = 1 + 0.02 * i
            w, h = 512 / s, 288 / s
            x0, y0 = (512 - w) / 2, (288 - h) / 2
            frames.append(np.asarray(t.crop((x0, y0, x0 + w, y0 + h)).resize((512, 288), Image.BICUBIC)))
        m = vf.measure_frames(frames, fps=24)
        self.assertEqual(m["camera"]["state"], "moving")
        self.assertIn("push in", m["camera"]["direction"])

    def test_empty_frames_are_void(self):
        flat = np.full((288, 512, 3), (40, 44, 52), np.uint8)
        m = vf.measure_frames([flat] * 6)
        self.assertEqual(len(m["empty_fraction"]["void_frames"]), 6)
        self.assertTrue(any("VOID" in v for v in m["verdict"]))

    def test_textured_frames_are_not_void(self):
        m = vf.measure_frames([texture(seed=s) for s in range(5)])
        self.assertLess(m["empty_fraction"]["max"], 0.2)
        self.assertFalse(any("VOID" in v for v in m["verdict"]))

    def test_half_empty_frame(self):
        t = texture()
        t[:, 256:] = 30
        m = vf.measure_frames([t] * 3)
        self.assertGreater(m["empty_fraction"]["mean"], 0.4)
        self.assertLess(m["empty_fraction"]["mean"], 0.6)

    def test_frozen_and_black(self):
        t = texture(1024, 288)
        frames = [t[:, 8 * i:8 * i + 512] for i in range(4)] + [t[:, 24:536]] * 4 + [np.zeros((288, 512, 3), np.uint8)]
        m = vf.measure_frames(frames)
        self.assertGreaterEqual(m["motion"]["longest_frozen_run"], 3)
        self.assertEqual(m["brightness"]["black_frames"], [8])

    def test_soft_frame(self):
        t = texture()
        blur = np.asarray(Image.fromarray(t).resize((64, 36)).resize((512, 288), Image.BILINEAR))
        m = vf.measure_frames([t, t, blur, t, t])
        self.assertEqual(m["sharpness"]["soft_frames"], [2])

    def test_summary_text_and_rule(self):
        m = vf.measure_frames([texture()] * 3)
        txt = vf.summary_text(m)
        self.assertIn("MEASURED", txt)
        self.assertIn("trust", m["rule"])

    def test_images_and_errors(self):
        self.assertEqual(vf.measure_frames([])["frames"], 0)
        self.assertIn("error", vf.measure_video("does/not/exist.mp4"))


if __name__ == "__main__":
    unittest.main()
