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


class GroundFitTests(unittest.TestCase):
    def test_fits_a_tilted_floor_and_levels_it(self):
        import numpy as np
        rng = np.random.default_rng(0)
        x = rng.uniform(-20, 20, 4000)
        z = rng.uniform(-40, -3, 4000)
        tilt = np.radians(4.0)
        y = -1.8 + np.tan(tilt) * (z + 20) * 0 + (-z - 20) * np.tan(tilt) * -1   # slopes with distance
        floor = np.stack([x, y, z], 1)
        wall = np.stack([rng.uniform(-5, 5, 500), rng.uniform(-1, 8, 500), np.full(500, -30.0)], 1)
        fit = p.fit_ground_plane(np.concatenate([floor, wall]))
        self.assertIsNotNone(fit)
        n, c0, share = fit
        self.assertAlmostEqual(float(np.degrees(np.arccos(n[1]))), 4.0, delta=0.6)
        self.assertGreater(share, 0.6)
        rot = p.level_rotation(n)
        up = p.euler_to_matrix(rot)
        level = [sum(up[i][k] * n[k] for k in range(3)) for i in range(3)]
        self.assertTrue(close(level, [0, 1, 0], 1e-6))

    def test_nothing_to_fit(self):
        import numpy as np
        self.assertIsNone(p.fit_ground_plane(np.zeros((10, 3))))


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

    def test_ground_plane_at_the_models_ground(self):
        world = json.loads(json.dumps(self.WORLD))
        for i in world["items"]:
            if i["id"] == "scene":
                i["scene_model"] = {"ground_y": 1.35}
        scene, _ = p.build_scene(world=world, moves=[{"t": 0}, {"t": 1}], seconds=1, fps=24, fov=None,
                                 resolution=(864, 480), ease="linear")
        g = next(i for i in scene["items"] if i["id"] == "ground")
        self.assertEqual(g["position"][1], 1.35)
        scene, _ = p.build_scene(world=world, moves=[{"t": 0}, {"t": 1}], seconds=1, fps=24, fov=None,
                                 resolution=(864, 480), ease="linear", ground=False)
        self.assertNotIn("ground", [i["id"] for i in scene["items"]])

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




class OrbitRig(unittest.TestCase):
    """orbit: a pivot on the subject turns about Y only; the camera is its child on +Z."""

    def _scene(self, **orbit):
        from agenty_core.tools import previz_shot as P
        spec = {"center": [1.0, 1.7, -3.0], "degrees": 180, "seconds": 5, "distance": [0.5, 7.0],
                "height": [0.0, 1.4], **orbit}
        return P.orbit_items(spec, [0.0, 1.6, 0.0], [0.0, 0.0, 0.0], 24, 40.0, (1280, 720), "linear")

    def test_pivot_turns_about_y_only(self):
        pivot, cam, length, _ = self._scene()
        self.assertEqual(length, 121)
        rot = pivot["tracks"]["rotation"]
        self.assertEqual(len(rot), 2)
        self.assertTrue(all(k["v"][0] == 0 and k["v"][2] == 0 for k in rot))
        self.assertAlmostEqual(rot[1]["v"][1] - rot[0]["v"][1], 180.0)
        self.assertEqual(cam["parent"], pivot["id"])

    def test_starts_facing_the_picture_camera(self):
        # the picture's camera stands at +Z of the centre (and to its left): yaw0 points local +Z at it
        pivot, _cam, _l, _ = self._scene()
        import math
        self.assertAlmostEqual(pivot["tracks"]["rotation"][0]["v"][1], math.degrees(math.atan2(-1.0, 3.0)), places=2)

    def test_camera_moves_straight_back_and_up_and_aims_at_the_centre(self):
        import math
        _p, cam, _l, _ = self._scene()
        pos, rot = cam["tracks"]["position"], cam["tracks"]["rotation"]
        self.assertEqual(pos[0]["v"], [0.0, 0.0, 0.5])
        self.assertEqual(pos[-1]["v"], [0.0, 1.4, 7.0])
        for p, r in zip(pos, rot):
            self.assertEqual(p["v"][0], 0.0)
            self.assertEqual(r["v"][1:], [0.0, 0.0])                     # no yaw, no roll of its own
            self.assertAlmostEqual(r["v"][0], -math.degrees(math.atan2(p["v"][1], p["v"][2])), places=2)
        self.assertEqual({k["ease"] for k in pos}, {"linear"})

    def test_build_scene_uses_the_rig(self):
        from agenty_core.tools import previz_shot as P
        scene, summary = P.build_scene(model="", world={"version": 1, "items": [
            {"id": "refcam", "kind": "camera", "position": [0, 1.6, 0], "rotation": [0, 0, 0], "fov": 40}]},
            moves=[], seconds=0, fps=24, fov=None, resolution=(864, 480), ease="linear", ground=False,
            orbit={"center": {"forward": 3, "up": 0.1}, "distance": [1, 5]})
        ids = [i["id"] for i in scene["items"]]
        self.assertIn("orbitpivot", ids)
        self.assertEqual(scene["activeCamera"], "shotcam")
        self.assertEqual(summary["orbit"]["center"], [0.0, 1.7, -3.0])



class OrbitReport(unittest.TestCase):
    def test_world_path_goes_round_to_behind(self):
        from agenty_core.tools import previz_shot as P
        _p, _c, _l, summary = P.orbit_items({"center": [0, 1.7, -1], "degrees": 180, "distance": [0.5, 7],
                                             "height": [0, 1.4]}, [0, 1.6, 0], [0, 0, 0], 24, 40, (1280, 720))
        path = summary["orbit"]["camera_world_path"]
        self.assertEqual(path["start"], [0.0, 1.7, -0.5])        # in front of him, towards the picture's camera
        self.assertAlmostEqual(path["end"][2], -8.0, places=3)     # 7 m behind him
        self.assertAlmostEqual(path["end"][1], 3.1, places=3)


if __name__ == "__main__":
    unittest.main()


class EulerFilterAndOrbitKeysTest(unittest.TestCase):
    def test_filter_picks_the_solution_next_to_the_previous_key(self):
        import agenty_core.tools.previz_shot as P
        prev = [-106.514, -69.045, -107.613]
        # the same rotation as the flipped solution agentY's whip keys jumped to
        flipped = [prev[0] + 180, 180 - prev[1], prev[2] + 180]
        got = P.matrix_to_euler(P.euler_to_matrix(flipped), prev)
        for a, b in zip(got, prev):
            self.assertAlmostEqual(a, b, places=3)

    def test_filter_unwraps_a_turn_past_180(self):
        import agenty_core.tools.previz_shot as P
        prev = [0.0, 0.0, 170.0]
        got = P.matrix_to_euler(P.euler_to_matrix([0.0, 0.0, -170.0]), prev)
        self.assertAlmostEqual(got[2], 190.0, places=3)

    def test_orbit_keys_drive_only_the_pivot_y_angle(self):
        import agenty_core.tools.previz_shot as P
        keys = [{"t": 0, "degrees": 0}, {"t": 1.3, "degrees": 15}, {"t": 1.65, "degrees": 105, "ease": "smooth"},
                {"t": 2.95, "degrees": 120}, {"t": 6, "degrees": 120, "ease": "hold"}]
        pivot, cam, length, summary = P.orbit_items(
            {"center": [0, 0.1, 0], "keys": keys, "distance": [2, 2], "height": [0.8, 0.8]},
            [0, 1, 2], [0, 0, 0], 24, 40.0, (1920, 1080), "linear")
        rot = pivot["tracks"]["rotation"]
        self.assertEqual([k["f"] for k in rot], [0, 31, 40, 71, 144])
        self.assertTrue(all(k["v"][0] == 0 and k["v"][2] == 0 for k in rot))
        y0 = rot[0]["v"][1]
        self.assertEqual([round(k["v"][1] - y0, 3) for k in rot], [0, 15, 105, 120, 120])
        # ease of the segment leaving a key = the next key's ease
        self.assertEqual([k["ease"] for k in rot], ["linear", "smooth", "linear", "hold", "hold"])
        self.assertEqual(length, 145)
        # the camera never moves toward the subject: constant distance and height
        self.assertTrue(all(k["v"] == cam["tracks"]["position"][0]["v"] for k in cam["tracks"]["position"]))

    def test_orbit_keys_refuse_a_half_turn_step(self):
        import agenty_core.tools.previz_shot as P
        with self.assertRaises(ValueError):
            P.orbit_items({"center": [0, 0, 0], "keys": [{"t": 0, "degrees": 0}, {"t": 1, "degrees": 200}]},
                          [0, 1, 2], [0, 0, 0], 24, 40.0, (1920, 1080))
