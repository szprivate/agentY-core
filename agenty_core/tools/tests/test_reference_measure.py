"""reference_measure maths on synthetic arrays (no ComfyUI): region hints, plane fit, sampling, pairs, analyse()."""

import json
import math
import unittest
from unittest import mock

import numpy as np

from agenty_core.tools import reference_measure as rm


def synthetic(cam_h=2.0, fov_y=40.0, W=200, H=300, wall_z=10.0):
    """Flat ground y=-cam_h below the horizon, a wall at z=wall_z above it, camera level, Y up Z forward."""
    f = (H / 2) / math.tan(math.radians(fov_y) / 2)
    v, u = np.mgrid[0:H, 0:W]
    ray = np.stack([(u + .5 - W / 2) / f, -(v + .5 - H / 2) / f, np.ones_like(u, dtype=float)], axis=-1)
    pts = np.zeros((H, W, 3))
    below = ray[..., 1] < 0
    t = np.where(below, -cam_h / np.where(below, ray[..., 1], -1), 0)
    pts[below] = (ray * t[..., None])[below]
    pts[~below] = (ray * (wall_z / ray[..., 2:3]))[~below]
    fx = 2 * math.degrees(math.atan(W / 2 / f))
    return {"points": pts, "mask": np.ones((H, W), bool), "width": W * rm.DECIMATION, "height": H * rm.DECIMATION,
            "fov_y": fov_y, "fov_x": fx}


class RegionTests(unittest.TestCase):
    def test_hints(self):
        self.assertEqual(rm.parse_region(""), (0.0, 0.75, 1.0, 1.0))
        self.assertEqual(rm.parse_region("bottom 40%"), (0.0, 0.6, 1.0, 1.0))
        self.assertEqual(rm.parse_region("rect 0.1,0.7,0.9,1.0"), (0.1, 0.7, 0.9, 1.0))
        self.assertEqual(rm.parse_region("top third")[3], 1 / 3)


class PlaneTests(unittest.TestCase):
    def test_flat_ground_with_noise(self):
        rng = np.random.default_rng(0)
        g = np.stack([rng.uniform(-5, 5, 3000), np.full(3000, -1.7) + rng.normal(0, .01, 3000), rng.uniform(2, 20, 3000)], 1)
        n, d, share = rm.fit_plane(g)
        self.assertAlmostEqual(n[1], 1.0, places=2)
        self.assertAlmostEqual(abs(d), 1.7, delta=0.03)
        self.assertGreater(share, 0.9)

    def test_ignores_outlier_wall_and_rejects_vertical(self):
        rng = np.random.default_rng(1)
        g = np.stack([rng.uniform(-5, 5, 2000), np.full(2000, -2.0), rng.uniform(2, 20, 2000)], 1)
        wall = np.stack([rng.uniform(-5, 5, 400), rng.uniform(-2, 5, 400), np.full(400, 15.0)], 1)
        n, d, _ = rm.fit_plane(np.vstack([g, wall]))
        self.assertAlmostEqual(abs(d), 2.0, delta=0.05)
        self.assertIsNone(rm.fit_plane(wall))

    def test_level_matrix_takes_normal_to_up(self):
        n = np.array([0.0, math.cos(0.2), -math.sin(0.2)])
        self.assertTrue(np.allclose(rm.level_matrix(n) @ n, [0, 1, 0], atol=1e-9))


class SamplingTests(unittest.TestCase):
    def test_median_skips_invalid_and_reports_none(self):
        pts = np.zeros((20, 20, 3))
        pts[..., 2] = 5.0
        pts[10, 10] = [100, 100, 100]                      # an outlier pixel
        valid = np.ones((20, 20), bool)
        xyz, n = rm.sample_point(pts, valid, 10 / 19, 10 / 19, radius_px=2)
        self.assertEqual(list(xyz), [0, 0, 5.0])
        self.assertEqual(n, 25)
        valid[:] = False
        xyz, n = rm.sample_point(pts, valid, 0.5, 0.5, radius_px=2)
        self.assertIsNone(xyz)
        self.assertEqual(n, 0)

    def test_edges_clamped(self):
        pts = np.ones((10, 10, 3))
        xyz, n = rm.sample_point(pts, np.ones((10, 10), bool), 1.0, 1.0, radius_px=2)
        self.assertEqual(n, 9)


class PairTests(unittest.TestCase):
    def test_metrics(self):
        m = rm.pair_metrics([0, 0, 0], [3, 4, 0])
        self.assertAlmostEqual(m["distance_m"], 5.0)
        self.assertAlmostEqual(m["vertical_diff_m"], 4.0)
        self.assertAlmostEqual(m["horizontal_m"], 3.0)


class AnalyseTests(unittest.TestCase):
    def test_camera_height_horizon_and_points(self):
        data = synthetic(cam_h=2.0)
        r = rm.analyse(data, points=[{"name": "base", "px": [0.5, 0.9]}, {"name": "top", "px": [0.5, 0.2]},
                                     {"name": "x", "px": "bad"}], pairs=[["base", "top"], ["base", "nope"]])
        self.assertAlmostEqual(r["camera_height_m"], 2.0, delta=0.05)
        self.assertAlmostEqual(r["horizon"]["y_from_top"], 0.5, delta=0.01)
        base, top, bad = r["points"]
        self.assertAlmostEqual(base["height_above_ground_m"], 0.0, delta=0.05)
        self.assertAlmostEqual(top["position_m"][2], 10.0, delta=0.1)
        self.assertFalse(bad["valid"])
        self.assertAlmostEqual(r["pairs"][0]["vertical_diff_m"], top["height_above_ground_m"], delta=0.05)
        self.assertIn("error", r["pairs"][1])

    def test_invalid_region_point_reported(self):
        data = synthetic()
        data["mask"][:100, :] = False
        r = rm.analyse(data, points=[{"name": "sky", "px": [0.5, 0.1]}])
        self.assertFalse(r["points"][0]["valid"])

    def test_no_ground_gives_null_height(self):
        data = synthetic()
        data["points"][..., 1] = np.linspace(-3, 3, data["points"].shape[0])[:, None]   # vertical wall everywhere
        data["points"][..., 2] = 8.0
        r = rm.analyse(data)
        self.assertIsNone(r["camera_height_m"])


class ToolTests(unittest.TestCase):
    def test_missing_file(self):
        self.assertEqual(json.loads(rm.measure_reference("/nope.jpg"))["status"], "error")

    def test_cache_hit_skips_comfyui(self):
        with mock.patch.object(rm, "load_or_run", return_value=(synthetic(), True)):
            with mock.patch("os.path.isfile", return_value=True):
                r = json.loads(rm.measure_reference("x.jpg", [{"name": "a", "px": [.5, .9]}]))
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["cached"])


if __name__ == "__main__":
    unittest.main()
