"""atlas_render_camera_move offline: the path maths, the plan and its price, the cost gate, the task sequence
against a stubbed Atlas API, resume without paying twice, and the assembly through a fake viewer."""

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from agenty_core import partner_credits as pc
from agenty_core.tools import atlas_camera_move as m
from agenty_core.tools import worldlabs_atlas as w
from agenty_core.tools.tests.test_worldlabs_atlas import Base, CAM, done_op, frame

MOVES = [{"t": 0}, {"t": 2, "forward": 1.5, "right": 0.3, "yaw": 12}]


class PathTests(unittest.TestCase):
    def world(self):
        return {"name": "w", "frames": [{"id": 0, "image": "v.png", "depth": "d.exr", "camera": CAM}]}

    def test_waypoints_and_pose(self):
        wps = m.waypoints(self.world(), MOVES)
        self.assertEqual([x["t"] for x in wps], [0.0, 2.0])
        R, p, _f = m.pose_at(wps, 1.0)
        self.assertAlmostEqual(p[2], -0.75, places=4)              # halfway forward (down -Z)
        self.assertAlmostEqual(m.pose_at(wps, 5.0)[1][2], -1.5, places=4)      # clamped to the end
        self.assertAlmostEqual(m.pose_at(wps, 1.0, "smooth")[1][2], -0.75, places=4)
        self.assertLess(-m.pose_at(wps, 0.5, "smooth")[1][2], 0.375)           # eases in

    def test_keys_from_viewer_matrices(self):
        eye = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        keys = [{"t": 0, "viewer": {"camera_world": eye + [0, 0, 0, 1], "fov": 40, "aspect": 16 / 9, "gated": True}},
                {"t": 1, "viewer": {"camera_world": eye + [0, 0, -2, 1], "fov": 40, "aspect": 16 / 9, "gated": True}}]
        wps = m.waypoints(self.world(), keys=keys)
        self.assertAlmostEqual(wps[1]["p"][2], -2.0, places=5)
        self.assertAlmostEqual(wps[0]["fov"], 40.0, places=3)

    def test_world_frame_absolute_keys_are_converted(self):
        cam = {**CAM, "extrinsics": {**CAM["extrinsics"], "position": [0, 1.0, 0]}}
        world = {"name": "w", "frames": [{"id": 0, "image": "v.png", "depth": "d.exr", "camera": cam}]}
        mv = [{"t": 0, "position": [0, 1.0, 0], "look_at": [0, 1.0, -3]},
              {"t": 1, "position": [0, 1.0, -2], "look_at": [0, 1.0, -5]}]
        wps = m.waypoints(world, mv, frame="world")
        self.assertAlmostEqual(wps[0]["p"][1], 0.0, places=5)
        self.assertAlmostEqual(wps[1]["p"][2], -2.0, places=5)

    def test_bad_paths(self):
        for kw in ({}, {"moves": [{"t": 0}]}, {"moves": [{"t": 1}, {"t": 1, "forward": 2}]}):
            with self.assertRaises(w.AtlasError):
                m.waypoints(self.world(), **kw)

    def test_batches(self):
        self.assertEqual(m.keyframe_times(2.0, 0.5), [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual([len(b) for b in m.split_batches(list(range(9)), 4)], [3, 3, 3])
        self.assertEqual([len(b) for b in m.split_batches(list(range(5)), 8)], [5])

    def test_plan_and_price(self):
        wps = m.waypoints(self.world(), MOVES)
        plan = m.make_plan(self.world(), wps, 24.0, "standard", "linear")
        self.assertEqual((plan["frames"], plan["keyframes"], plan["tasks"]), (49, 5, 2))
        e = plan["estimate"]
        self.assertEqual((e["min_credits"], e["max_credits"]), (200, 500))
        self.assertAlmostEqual(e["max_usd"], 0.5)
        self.assertEqual(m.make_plan(self.world(), wps, 24.0, "draft", "linear")["tasks"], 1)
        with mock.patch.dict(os.environ, {"WLT_ATLAS_CREDITS_PER_TASK": "250"}):
            self.assertAlmostEqual(m.make_plan(self.world(), wps, 24.0, "standard", "linear")["estimate"]["max_usd"], 1.25)

    def test_pure_rotation_and_length_are_refused(self):
        wps = m.waypoints(self.world(), [{"t": 0}, {"t": 2, "yaw": 40}])
        with self.assertRaises(w.AtlasError):
            m.make_plan(self.world(), wps, 24.0, "standard", "linear")
        with self.assertRaises(w.AtlasError):
            m.make_plan(self.world(), m.waypoints(self.world(), [{"t": 0}, {"t": 40, "forward": 5}]), 24.0, "draft", "linear")

    def test_gate(self):
        wps = m.waypoints(self.world(), MOVES)
        plan = m.make_plan(self.world(), wps, 24.0, "standard", "linear")      # up to $0.50
        with self.assertRaises(pc.CostGateError) as cm:
            m.gate(plan, 0, False)
        self.assertEqual(cm.exception.error_kind, "confirm_cost")
        m.gate(plan, 0, True)
        with pc.confirmed():
            m.gate(plan, 0, False)
        with self.assertRaises(w.AtlasError) as cm2:
            m.gate(plan, 0.2, True)
        self.assertEqual(cm2.exception.kind, "over_cap")


class FakeClient:
    def __init__(self, tmp):
        self.posts, self.tmp = [], tmp

    def post(self, path, json_data=None, **kw):
        self.posts.append((path, json_data))
        return {"id": "job1", "key": "previz:x"}

    def get(self, path, params=None, **kw):
        p = Path(self.tmp) / "clip.mp4"
        p.write_bytes(b"mp4")
        return {"state": "done", "path": str(p), "where": "tab"}


class RunTests(Base):
    def setUp(self):
        super().setUp()
        out = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)], "promptUsed": "a cabin"}),
        }, name="cabin", prompt="a cabin")
        self.assertEqual(out["status"], "ok", out)
        self.calls.clear()
        self.client = FakeClient(self.tmp.name)
        for p in (mock.patch.object(w, "_publish", lambda path: None),
                  mock.patch("agenty_core.tools.previz_shot._measure_render",
                             lambda path, expect_motion=True: {"measured": {"ok": 1}}),
                  mock.patch("agenty_core.utils.comfyui_client.get_client", lambda: self.client)):
            p.start()
            self.addCleanup(p.stop)
        self.n = 0

    def gen(self, body):
        cams = body["targetCameras"]
        self.n += 1
        frames = [{**frame(10 * self.n + j), "camera": c} for j, c in enumerate(cams)]
        return done_op(f"op_g{self.n}", {"frames": frames, "promptUsed": "a cabin"})

    def go(self, **kw):
        a, b = self.api({("POST", "/tasks:atlasGenerate"): self.gen})
        args = dict(name="push", world="cabin", moves=MOVES, quality="standard", confirm_cost=True,
                    output_dir=self.tmp.name)
        args.update(kw)
        with a, b:
            return json.loads(m.render_move(**args))

    def submits(self):
        return [c for c in self.calls if c[0] == "POST" and c[1].endswith("tasks:atlasGenerate")]

    def test_dry_run_spends_nothing(self):
        out = self.go(dry_run=True, confirm_cost=False)
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["spent"], 0)
        self.assertEqual(out["plan"]["tasks"], 2)
        self.assertEqual(self.submits(), [])
        self.assertFalse((Path(self.tmp.name) / "cabin__push").exists())

    def test_gate_blocks_without_confirm(self):
        out = self.go(confirm_cost=False)
        self.assertEqual(out["error_kind"], "confirm_cost")
        self.assertEqual(out["plan"]["tasks"], 2)
        self.assertEqual(self.submits(), [])

    def test_full_run_sequence_and_assembly(self):
        progress = []
        out = self.go(on_progress=progress.append)
        self.assertEqual(out["status"], "ok", out)
        subs = self.submits()
        self.assertEqual(len(subs), 2)
        self.assertEqual([len(s[2]["targetCameras"]) for s in subs], [3, 2])
        self.assertTrue(all(s[2]["returnDepth"] for s in subs))
        # the second task is conditioned on a view painted by the first
        ctx2 = [c["imageAsset"]["assetId"] for c in subs[1][2]["contextFrames"]]
        self.assertTrue(any(a.startswith("asset_img1") for a in ctx2), ctx2)
        self.assertEqual(out["tasks_used"], 2)
        wd = Path(self.tmp.name) / "cabin__push"
        wj = json.loads((wd / "world.json").read_text())
        self.assertEqual(len(wj["frames"]), 1 + 5)
        orig = json.loads((Path(self.tmp.name) / "cabin" / "world.json").read_text())
        self.assertEqual(len(orig["frames"]), 1)                                   # the original is untouched
        path, body = self.client.posts[0]
        self.assertEqual(path, "/bepic/previz_shot")
        scene = body["scene"]
        self.assertEqual(scene["length"], 49)
        self.assertEqual(len(scene["items"][1]["tracks"]["position"]), 49)
        self.assertTrue(scene["items"][0]["src"]["path"].endswith("world.glb"))
        self.assertTrue(body["render"]["useCameraRes"])
        self.assertEqual(out["measured"], {"ok": 1})
        self.assertTrue(Path(out["path"] + ".agenty.json").exists())
        self.assertEqual(progress[0]["tasks_total"], 2)

    def test_resume_does_not_pay_twice(self):
        a, b = self.api({("POST", "/tasks:atlasGenerate"): self.gen})
        with mock.patch.object(m, "_viewer_render", side_effect=w.AtlasError("viewer closed", "timeout")), a, b:
            first = json.loads(m.render_move(name="push", world="cabin", moves=MOVES, confirm_cost=True,
                                             output_dir=self.tmp.name))
        self.assertEqual(first["error_kind"], "timeout")
        self.assertEqual(len(self.submits()), 2)
        out = self.go()
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(len(self.submits()), 2)                                   # no new task
        self.assertEqual(out["tasks_paid_this_run"], 0)
        other = self.go(moves=[{"t": 0}, {"t": 2, "forward": 3}])
        self.assertEqual(other["error_kind"], "args")                              # same name, other path

    def test_frame_count_mismatch_is_an_error_and_resumable(self):
        a, b = self.api({("POST", "/tasks:atlasGenerate"): lambda body: done_op("op_x", {"frames": [frame(50)]})})
        with a, b:
            out = json.loads(m.render_move(name="push", world="cabin", moves=MOVES, confirm_cost=True,
                                           output_dir=self.tmp.name))
        self.assertEqual(out["error_kind"], "api")
        self.assertIn("resume", out)

    def test_tool_wrapper_dry_run(self):
        out = json.loads(m.atlas_render_camera_move(name="p", world="cabin", moves=MOVES, dry_run=True,
                                                    output_dir=self.tmp.name))
        self.assertEqual(out["status"], "dry_run")


class SwitchTest(unittest.TestCase):
    def tearDown(self):
        m.set_enabled_provider(None)

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(m.ENABLE_ENV, None)
            self.assertFalse(m.camera_move_enabled())

    def test_provider_and_env(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(m.ENABLE_ENV, None)
            m.set_enabled_provider(lambda: True)
            self.assertTrue(m.camera_move_enabled())
        with mock.patch.dict(os.environ, {m.ENABLE_ENV: "0"}):
            self.assertFalse(m.camera_move_enabled())
        m.set_enabled_provider(None)
        with mock.patch.dict(os.environ, {m.ENABLE_ENV: "1"}):
            self.assertTrue(m.camera_move_enabled())


if __name__ == "__main__":
    unittest.main()
