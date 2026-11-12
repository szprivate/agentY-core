"""animate_scene / render_turntables offline: the requests they build and their waits, the ComfyUI client mocked."""

import json
import unittest
from unittest import mock

from agenty_core.tools import animate_character as ac
from agenty_core.tools import animate_scene as a
from agenty_core.tools import viewer_state as vs


class _Page:
    """Stands in for ComfyUI + the open page: every POST gets an id, its answer comes from `answer(body)`."""

    def __init__(self, answer):
        self.answer = answer
        self.posted = []
        self.asks = {}

    def post(self, path, json_data=None, **kw):
        self.posted.append((path, json_data))
        rid = f"r{len(self.posted)}"
        self.asks[rid] = self.answer(json_data)
        return {"id": rid, "state": "sent"}

    def get(self, path, params=None, **kw):
        return self.asks[params["id"]]


class AnimateSceneTests(unittest.TestCase):
    def run_with(self, page, fn, **kw):
        with mock.patch.object(ac, "_client", return_value=page), mock.patch.object(ac.time, "sleep"):
            return json.loads(fn(**kw))

    def test_request_bodies(self):
        self.assertEqual(a.request_body("clip_set", {"clip": "Walk", "speed": 2}), {"action": "clip_set", "clip": "Walk", "speed": 2})
        self.assertEqual(a.request_body("scene_info"), {"action": "scene_info"})
        self.assertEqual(a.request_body("turntable", {"clips": "all", "action": "render"}), {"action": "turntable", "clips": "all"})
        for action, opts in (("nope", {}), ("key_set", {"item": "Hero"}), ("clip_set", {}),
                             ("render", {"format": "gif"}), ("render", {"from": 10, "to": 2}),
                             ("export_usd", {"format": "usdz"}), ("sdk_key", {"driver": {"item": "a"}})):
            with self.assertRaises(ValueError, msg=(action, opts)):
                a.request_body(action, opts)

    def test_an_action_goes_to_the_page_and_its_answer_comes_back(self):
        page = _Page(lambda b: {"state": "done", "result": {"clip": "c1", "from": 30, "to": 53}})
        out = self.run_with(page, a.animate_scene, action="clip_place", options={"source": "Walk 1", "at": 30})
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result"]["from"], 30)
        self.assertEqual(page.posted, [("/imageviewer/rig/agent", {"action": "clip_place", "source": "Walk 1", "at": 30})])
        bad = self.run_with(page, a.animate_scene, action="clip_set", options={})
        self.assertEqual(bad["status"], "error")
        self.assertIn("options.clip", bad["error"])
        err = self.run_with(_Page(lambda b: {"state": "error", "error": "no clip named x"}), a.animate_scene,
                            action="clip_remove", options={"clip": "x"})
        self.assertEqual(err, {"status": "error", "error": "no clip named x"})

    def test_an_old_viewer_says_so(self):
        class Old(_Page):
            def post(self, path, json_data=None, **kw):
                raise RuntimeError("400 Bad Request: action is one of rig, motion")
        out = self.run_with(Old(None), a.animate_scene, action="scene_info")
        self.assertIn("does not know the action 'scene_info'", out["error"])

    def test_render_turntables_makes_one_rig_and_renders_each_clip(self):
        def answer(b):
            if b["action"] == "turntable":
                return {"state": "done", "result": {"camera": "Turntable cam", "ranges": [
                    {"clip": "Hero walk", "from": 0, "to": 23}, {"clip": "Fast walk", "from": 30, "to": 41}]}}
            if b["from"] == 30:
                return {"state": "error", "error": "the renderer produced no image"}
            return {"state": "done", "result": {"path": f"D:/out/previz/{b['name']}.mp4", "frames": b["to"] - b["from"] + 1}}
        page = _Page(answer)
        out = self.run_with(page, a.render_turntables, clips=["Hero walk", "Fast walk"], target="Hero, Prop", name="Shot 1",
                            width=640, height=360, shading="clay")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["camera"], "Turntable cam")
        self.assertEqual(out["renders"], [{"clip": "Hero walk", "from": 0, "to": 23, "path": "D:/out/previz/shot_1_hero_walk_turntable.mp4",
                                           "where": None, "frames": 24}])
        self.assertEqual(out["failed"], [{"clip": "Fast walk", "error": "the renderer produced no image"}])
        tt = page.posted[0][1]
        self.assertEqual(tt, {"action": "turntable", "clips": ["Hero walk", "Fast walk"], "degrees": 360.0, "start_yaw": 0.0,
                              "target": ["Hero", "Prop"]})
        r = page.posted[1][1]
        self.assertEqual(r, {"action": "render", "camera": "Turntable cam", "from": 0, "to": 23, "format": "mp4",
                             "shading": "clay", "add_node": False, "name": "shot_1_hero_walk_turntable", "width": 640, "height": 360})
        none = self.run_with(_Page(lambda b: {"state": "error", "error": "the scene has no clips"}), a.render_turntables)
        self.assertEqual(none["status"], "error")
        self.assertIn("no clips", none["error"])

    def test_animate_character_new_options(self):
        self.assertEqual(ac.request_body("motion", prompt="A walk.", options={"as_clip": True}),
                         {"action": "motion", "prompt": "A walk.", "feet": True, "from_pose": True, "as_clip": True})
        self.assertEqual(ac.request_body("export", format="usd"), {"action": "export", "format": "usd"})
        self.assertEqual(ac.request_body("add_motion", model="Hero", media_path="D:/mocap/run.fbx", from_frame=12,
                                         options={"how": "bake"}),
                         {"action": "add_motion", "model": "Hero", "path": "D:/mocap/run.fbx", "how": "bake", "from": 12})
        with self.assertRaises(ValueError):
            ac.request_body("add_motion")
        with self.assertRaises(ValueError):
            ac.request_body("export", format="obj")

    def test_viewer_state_line(self):
        text = vs.describe_lines({"open": True, "scene3d": {
            "fps": 24, "length": 60, "active_camera": "Turntable cam", "cameras": ["Shot cam", "Turntable cam"],
            "clips": [{"name": "Walk", "track": "AI takes", "from": 0, "to": 23}, {"name": "Wave", "track": "T2", "from": 30, "to": 41, "mute": True}],
            "anim_layers": [{"name": "Fix", "mode": "additive", "weight": 0.5, "mute": True}],
            "display_layers": [{"name": "set", "visible": False, "mode": "template"}]}})
        line = [x for x in text.splitlines() if x.startswith("3d scene")][0]
        self.assertEqual(line, "3d scene 60 frames @ 24 fps; cameras Shot cam, Turntable cam (looking through Turntable cam); "
                               "clips Walk 0-23, Wave 30-41 muted; anim layers Fix (additive 0.5, muted); "
                               "display layers set hidden template")


if __name__ == "__main__":
    unittest.main()
