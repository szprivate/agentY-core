"""render_camera_move: the frame of a camera path (picture vs world) and the sanity check against the scene.

A synthetic 3DGS splat of a room (floor at y=0, ceiling at y=3, walls) stands in for a trained Atlas splat;
its picture camera is at y=1.013 (job.json beside the .ply), like the real interior2 splat.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from agenty_core.tools import camera_check as cc
from agenty_core.tools import previz_shot as p
from agenty_core.tools.splat_train import write_ply

PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3"]
FIRST_CAMERA = [[1, 0, 0, 0.0], [0, 1, 0, 1.013], [0, 0, 1, 0.0], [0, 0, 0, 1]]


def room_points(n=30000, seed=2):
    """A room in scene coordinates (Y up): x -2..2, y 0..3, z -6..0 - floor, ceiling and four walls."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, 3))
    pts = np.stack([-2 + 4 * u[:, 0], 3 * u[:, 1], -6 * u[:, 2]], -1)
    face = rng.integers(0, 6, n)
    pts[face == 0, 1] = 0.0
    pts[face == 1, 1] = 3.0
    pts[face == 2, 0] = -2.0
    pts[face == 3, 0] = 2.0
    pts[face == 4, 2] = -6.0
    pts[face == 5, 2] = 0.0
    return pts


def write_splat(folder: Path, name="room.ply", with_job=True):
    pts = room_points()
    a = np.zeros((len(pts), len(PROPS)), dtype="<f4")
    a[:, 0], a[:, 1], a[:, 2] = pts[:, 0], -pts[:, 1], -pts[:, 2]      # the file is Y down
    a[:, 6] = 2.0
    path = folder / name
    write_ply(path, PROPS, a)
    if with_job:
        (folder / "job.json").write_text(json.dumps({"first_camera": FIRST_CAMERA}), encoding="utf-8")
    return str(path)


def run_tool(model, moves=None, states=None, **kw):
    client = mock.Mock()
    client.post.return_value = {"ok": True, "id": "abc", "key": "previz:s", "scene_file": "x.json"}
    client.get.side_effect = states or [{"state": "done", "path": "D:/none/s.mp4"}]
    with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client), mock.patch("time.sleep"):
        out = json.loads(p.render_camera_move(name="s", model=model, moves=moves, **kw))
    return out, client


class SplatFrames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.splat = write_splat(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_is_a_splat_and_pose_comes_from_the_job(self):
        self.assertTrue(cc.is_splat_path(self.splat))
        pose = p._splat_picture_pose(self.splat)
        self.assertEqual(pose["position"], [0.0, 1.013, 0.0])

    def test_relative_moves_start_at_the_picture_camera_not_the_origin(self):
        out, c = run_tool(self.splat, [{"t": 0}, {"t": 2, "forward": 2.0}])
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(out["keys"][0]["position"], [0.0, 1.013, 0.0])
        self.assertEqual(out["keys"][1]["position"], [0.0, 1.013, -2.0])
        self.assertEqual(out["picture_camera"]["position"], [0.0, 1.013, 0.0])
        self.assertTrue(out["check"]["passed"])
        self.assertAlmostEqual(out["scene_bounds"]["floor_y"], 0.0, delta=0.1)
        self.assertAlmostEqual(out["scene_bounds"]["top_y"], 3.0, delta=0.1)

    def test_absolute_keys_need_a_frame(self):
        out, _ = run_tool(self.splat, [{"t": 0, "position": [0, 0.7, -0.4], "look_at": [0, 0.7, -2]},
                                       {"t": 2, "position": [0, 0.7, -2], "look_at": [0, 0.7, -4]}])
        self.assertEqual(out["status"], "error")
        self.assertIn("frame='world'", out["error"])
        self.assertIn("1.013", out["error"])

    def test_frame_world_keeps_the_numbers(self):
        out, _ = run_tool(self.splat, [{"t": 0, "position": [0, 0.7, -0.4], "look_at": [0, 0.7, -2]},
                                       {"t": 2, "position": [0, 0.7, -2], "look_at": [0, 0.7, -4]}], frame="world")
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(out["keys"][0]["position"], [0.0, 0.7, -0.4])
        self.assertEqual(out["frame"], "world")

    def test_frame_picture_adds_the_picture_camera(self):
        out, _ = run_tool(self.splat, [{"t": 0, "position": [0, -0.3, -0.4], "look_at": [0, -0.3, -2]},
                                       {"t": 2, "position": [0, -0.3, -2], "look_at": [0, -0.3, -4]}], frame="picture")
        self.assertEqual(out["status"], "ok", out)
        self.assertAlmostEqual(out["keys"][0]["position"][1], 0.713, places=3)
        self.assertAlmostEqual(out["keys"][0]["position"][2], -0.4, places=3)

    def test_keys_under_the_floor_are_refused_with_numbers_and_the_frame_hint(self):
        # picture-relative numbers (0.3 m below the eye) sent as world coordinates sit under the floor
        out, c = run_tool(self.splat, [{"t": 0, "position": [0, -0.3, -0.4], "look_at": [0, -0.3, -2]},
                                       {"t": 2, "position": [0, -0.3, -2], "look_at": [0, -0.3, -4]}], frame="world")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["error_kind"], "camera_outside_scene")
        self.assertIn("below the floor", out["error"])
        self.assertIn("y=-0.30", out["error"])
        self.assertIn("frame='picture'", out["hint"])
        self.assertIn("INSIDE", out["hint"])
        self.assertIn("floor_y", out["scene_bounds"])
        self.assertFalse(c.post.called)                         # nothing was sent to the viewer

    def test_outside_the_room_is_refused(self):
        out, _ = run_tool(self.splat, [{"t": 0}, {"t": 2, "forward": 30.0}])
        self.assertEqual(out["status"], "error")
        self.assertIn("outside the scene", out["error"])

    def test_looking_at_nothing_is_measured(self):
        pts = room_points()
        b = cc.scene_bounds(pts, "splat", len(pts))
        rep = cc.check_path([{"frame": 0, "position": [0, 2.5, -3], "forward": [0, 1, 0], "key": True}],
                            pts, b, 54.0, 16 / 9)
        self.assertTrue(any("sees almost nothing" in x for x in rep["problems"]))
        self.assertLess(rep["measured"][0]["gaussians_in_view"], 600)

    def test_check_false_renders_anyway(self):
        out, c = run_tool(self.splat, [{"t": 0}, {"t": 2, "forward": 30.0}], check=False)
        self.assertEqual(out["status"], "ok")
        self.assertTrue(c.post.called)

    def test_orbit_is_checked_too(self):
        out, _ = run_tool(self.splat, None, orbit={"center": [0, 1.2, -3], "degrees": 90, "seconds": 3,
                                                   "distance": [1.0, 1.5], "height": [0, 0.2]}, frame="world")
        self.assertEqual(out["status"], "ok", out)
        far, _ = run_tool(self.splat, None, orbit={"center": [0, 1.2, -3], "degrees": 90, "seconds": 3,
                                                   "distance": [10, 12]}, frame="world")
        self.assertEqual(far["status"], "error")

    def test_splat_without_a_picture_pose_asks_for_atlas_world(self):
        with tempfile.TemporaryDirectory() as d:
            lone = write_splat(Path(d), with_job=False)
            out, _ = run_tool(lone, [{"t": 0}, {"t": 2, "forward": 1.0}])
            self.assertEqual(out["status"], "error")
            self.assertIn("atlas_world", out["error"])
            ok, _ = run_tool(lone, [{"t": 0, "position": [0, 0.7, -1], "look_at": [0, 0.7, -3]},
                                    {"t": 2, "position": [0, 0.7, -2], "look_at": [0, 0.7, -4]}], frame="world")
            self.assertEqual(ok["status"], "ok", ok)

    def test_atlas_world_gives_the_pose(self):
        with tempfile.TemporaryDirectory() as d:
            lone = write_splat(Path(d), with_job=False)
            world = {"frames": [{"id": 0, "camera": {"extrinsics": {"position": [0, 1.013, 0], "quaternion": [0, 0, 0, 1],
                                                                      "coordinateSystem": "rub"},
                                                     "intrinsics": {"fx": 743.0, "fy": 743.0, "cx": 640, "cy": 360,
                                                                    "width": 1280, "height": 720}}}]}
            wd = Path(d) / "w"
            wd.mkdir()
            (wd / "world.json").write_text(json.dumps(world), encoding="utf-8")
            with mock.patch("agenty_core.tools.worldlabs_atlas._world_dir", return_value=wd):
                out, _ = run_tool(lone, [{"t": 0}, {"t": 2, "forward": 1.0}], atlas_world="w")
            self.assertEqual(out["status"], "ok", out)
            self.assertAlmostEqual(out["picture_camera"]["position"][1], 1.013, places=3)
            self.assertAlmostEqual(out["picture_camera"]["fov"], 51.7, delta=0.3)

    def test_bad_frame(self):
        out, _ = run_tool(self.splat, [{"t": 0}, {"t": 1}], frame="local")
        self.assertIn("frame must be", out["error"])


class MeshUnchanged(unittest.TestCase):
    def test_mesh_frames_are_the_same_and_frame_is_optional(self):
        kw = dict(seconds=1, fps=24, fov=None, resolution=(8, 8), ease="linear", ground=False)
        moves = [{"t": 0, "position": [0, 1, 0], "look_at": [0, 1, -3]},
                 {"t": 1, "position": [0, 1, -2], "look_at": [0, 1, -5]}]
        a, _ = p.build_scene(model="D:/x/mesh.glb", moves=moves, **kw)
        b, _ = p.build_scene(model="D:/x/mesh.glb", moves=moves, frame="world", **kw)
        self.assertEqual(a["items"][-1]["tracks"], b["items"][-1]["tracks"])


class RenderMeasured(unittest.TestCase):
    def test_void_and_static_renders_are_flagged(self):
        frames = [np.full((144, 256, 3), 40, np.uint8)] * 8
        with tempfile.TemporaryDirectory() as d:
            f = str(Path(d) / "v.mp4")
            try:
                import imageio.v3 as iio
                iio.imwrite(f, np.stack(frames), fps=24)
            except Exception as exc:  # noqa: BLE001
                self.skipTest(f"no video writer: {exc}")
            res = p._measure_render(f, expect_motion=True)
        self.assertTrue(res["measured"]["empty_fraction"]["void_frames"])
        self.assertTrue(any("void" in w for w in res["render_warnings"]))
        self.assertTrue(any("static" in w for w in res["render_warnings"]))


if __name__ == "__main__":
    unittest.main()
