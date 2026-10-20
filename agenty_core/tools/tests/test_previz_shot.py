"""previz_shot offline: the camera maths and the call to the viewer, the ComfyUI client mocked."""

import json
import unittest
from unittest import mock

from agenty_core.tools import previz_shot as p


def close(a, b, tol=1e-6):
    return all(abs(x - y) < tol for x, y in zip(a, b))


class MathTests(unittest.TestCase):
    def test_euler_roundtrip(self):
        for rot in ([0, 0, 0], [-3.4, 12, 0], [20, -40, 5], [-70, 80, 10]):
            back = p.matrix_to_euler(p.euler_to_matrix(rot), rot)
            self.assertTrue(close(rot, back, 1e-4), (rot, back))

    def test_look_at_looks_at_the_target(self):
        m = p.look_at_matrix([0, 2, 10], [0, 2, 0])
        fwd = [-x for x in p._col(m, 2)]
        self.assertTrue(close(fwd, [0, 0, -1]))
        self.assertTrue(close(p.matrix_to_euler(m), [0, 0, 0], 1e-6))

    def test_forward_is_horizontal_even_when_pitched(self):
        keys = p.camera_pose_keys([0, 1.7, 0], [-20, 0, 0], [{"t": 0}, {"t": 5, "forward": 10}], 24)
        self.assertTrue(close(keys[1][1], [0, 1.7, -10]))
        self.assertEqual(keys[1][0], 120)

    def test_up_right_and_yaw(self):
        keys = p.camera_pose_keys([0, 0, 0], [0, 0, 0], [{"t": 0}, {"t": 1, "right": 2, "up": 3, "yaw": 90}], 10)
        self.assertTrue(close(keys[1][1], [2, 3, 0]))
        fwd = [-x for x in p._col(p.euler_to_matrix(keys[1][2]), 2)]
        self.assertTrue(close(fwd, [-1, 0, 0], 1e-6))

    def test_world_reference_camera_is_the_start(self):
        keys = p.camera_pose_keys([5, 1.7, 60], [-3.4, 0, 0], [{"t": 0}, {"t": 2, "forward": 4}], 24, 65.0)
        self.assertTrue(close(keys[0][1], [5, 1.7, 60]))
        self.assertTrue(close(keys[0][2], [-3.4, 0, 0], 1e-4))
        self.assertEqual(keys[0][3], 65.0)


class SceneTests(unittest.TestCase):
    WORLD = {"version": 1, "fps": 24, "length": 240, "walk": {}, "items": [
        {"id": "env", "kind": "environment"}, {"id": "terrain", "kind": "terrain"},
        {"id": "scatter_pine", "kind": "scatter"}, {"id": "hero", "kind": "depthmesh"},
        {"id": "scene", "kind": "model", "src": {}},
        {"id": "refcam", "kind": "camera", "position": [0, 1.7, 60], "rotation": [-3, 0, 0], "fov": 65}]}

    def test_world_scene_keeps_the_set_only(self):
        scene, summary = p.build_scene(world=self.WORLD, moves=[{"t": 0}, {"t": 5, "forward": 40, "up": 1.5}],
                                       seconds=5, fps=24, fov=None, resolution=(864, 480), ease="linear")
        self.assertEqual([i["id"] for i in scene["items"]], ["env", "scene", "refcam", "shotcam"])
        self.assertEqual(scene["activeCamera"], "shotcam")
        self.assertEqual(scene["length"], 121)
        cam = scene["items"][-1]
        self.assertEqual(cam["fov"], 65.0)
        self.assertEqual(cam["tracks"]["position"][1]["f"], 120)
        self.assertNotIn("fov", cam["tracks"])
        self.assertNotIn("walk", scene)
        self.assertEqual(summary["frames"], 121)

    def test_depth_view_stays_without_a_scene_model(self):
        world = json.loads(json.dumps(self.WORLD))
        world["items"] = [i for i in world["items"] if i["id"] != "scene"]
        scene, _ = p.build_scene(world=world, moves=[{"t": 0}, {"t": 1}], seconds=1, fps=24, fov=None,
                                 resolution=(864, 480), ease="smooth")
        self.assertIn("hero", [i["id"] for i in scene["items"]])

    def test_bare_mesh_and_fov_track(self):
        scene, _ = p.build_scene(model="D:/x/mesh.glb", moves=[{"t": 0, "fov": 40}, {"t": 2, "fov": 55}],
                                 seconds=2, fps=24, fov=None, resolution=(832, 480), ease="linear")
        self.assertEqual(scene["items"][0]["src"]["format"], "glb")
        self.assertIn("fov", scene["items"][-1]["tracks"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            p.build_scene(model="m.glb", moves=[{"t": 0}], seconds=1, fps=24, fov=None, resolution=(8, 8), ease="")
        with self.assertRaises(ValueError):
            p.build_scene(moves=[{"t": 0}, {"t": 1}], seconds=1, fps=24, fov=None, resolution=(8, 8), ease="")


class ToolTests(unittest.TestCase):
    def run_tool(self, states, **kw):
        client = mock.Mock()
        client.get.side_effect = [SceneTests.WORLD] + states
        client.post.return_value = {"ok": True, "id": "abc", "key": "previz:s", "scene_file": "x.json"}
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client), \
                mock.patch("time.sleep"):
            out = json.loads(p.render_camera_move(name="s", world="w",
                                                  moves=[{"t": 0}, {"t": 5, "forward": 30}], **kw))
        return out, client

    def test_renders_and_waits(self):
        out, c = self.run_tool([{"state": "rendering", "done": 3}, {"state": "done", "path": "D:/o/s.mp4"}])
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["path"], "D:/o/s.mp4")
        body = c.post.call_args.kwargs["json_data"]
        self.assertEqual(c.post.call_args.args[0], "/bepic/previz_shot")
        self.assertEqual(body["render"]["camera"], "shotcam")
        self.assertEqual(body["scene"]["length"], 121)

    def test_render_error_is_reported(self):
        out, _ = self.run_tool([{"state": "error", "message": "no 3D view"}])
        self.assertEqual(out["status"], "error")
        self.assertIn("no 3D view", out["error"])

    def test_no_render_returns_at_once(self):
        out, c = self.run_tool([], render=False)
        self.assertFalse(out["rendered"])
        self.assertNotIn("render", c.post.call_args.kwargs["json_data"])

    def test_old_viewer(self):
        client = mock.Mock()
        client.get.return_value = SceneTests.WORLD
        client.post.side_effect = RuntimeError("404 Not Found")
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
            out = json.loads(p.render_camera_move(name="s", world="w", moves=[{"t": 0}, {"t": 1}]))
        self.assertIn("too old", out["error"])


if __name__ == "__main__":
    unittest.main()
