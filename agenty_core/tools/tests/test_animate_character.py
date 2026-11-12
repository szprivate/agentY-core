"""animate_character offline: the request it builds and its wait for the page, the ComfyUI client mocked."""

import json
import unittest
from unittest import mock

from agenty_core.tools import animate_character as a


class _Client:
    def __init__(self, states):
        self.states = list(states)
        self.posted = []

    def post(self, path, json_data=None, **kw):
        self.posted.append((path, json_data))
        return {"id": "r1", "state": "sent"}

    def get(self, path, params=None, **kw):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]


class AnimateCharacterTests(unittest.TestCase):
    def call(self, client, **kw):
        with mock.patch.object(a, "_client", return_value=client), mock.patch.object(a.time, "sleep"):
            return json.loads(a.animate_character(**kw))

    def test_motion_request(self):
        c = _Client([{"state": "running"}, {"state": "done", "result": {"keys": 40, "frames": 49}}])
        out = self.call(c, action="motion", model="Hero", prompt="A person walks.", from_frame=0, to_frame=48, seed=3)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result"]["keys"], 40)
        self.assertEqual(c.posted, [("/imageviewer/rig/agent", {
            "action": "motion", "model": "Hero", "prompt": "A person walks.", "feet": True, "from_pose": True,
            "from": 0, "to": 48, "seed": 3})])

    def test_new_actions(self):
        self.assertEqual(a.request_body("restyle", prompt=" A soldier marches. ", strength="close", layer=True, seed=2),
                         {"action": "restyle", "prompt": "A soldier marches.", "strength": "close", "feet": True,
                          "layer": "new", "seed": 2})
        self.assertEqual(a.request_body("video_motion", media_path="D:/films/dance.mp4", from_frame=10,
                                        options={"clean": True, "step": 2, "nonsense": 1}),
                         {"action": "video_motion", "path": "D:/films/dance.mp4", "feet": True, "clean": True, "step": 2,
                          "from": 10})
        self.assertEqual(a.request_body("lipsync", model="Hero", media_path="line.wav", options={"lead": 2}),
                         {"action": "lipsync", "model": "Hero", "path": "line.wav", "lead": 2})
        self.assertEqual(a.request_body("constrain", options={"type": "aim", "driven": "Head", "targets": ["Ball"]}),
                         {"action": "constrain", "type": "aim", "driven": "Head", "targets": ["Ball"]})
        self.assertEqual(a.request_body("layer_add", name="Fixes", options={"mode": "additive"}),
                         {"action": "layer_add", "name": "Fixes", "mode": "additive"})
        self.assertEqual(a.request_body("pose_apply", name="Run 1", options={"mirror": True}),
                         {"action": "pose_apply", "name": "Run 1", "mirror": True})
        # a layer is only for takes
        self.assertNotIn("layer", a.request_body("lipsync", media_path="x.wav", layer=True))
        for kw in ({"action": "restyle"}, {"action": "restyle", "prompt": "x", "strength": "huge"},
                   {"action": "video_motion"}, {"action": "lipsync"}, {"action": "pose_save"},
                   {"action": "constrain", "options": {"type": "glue", "driven": "a", "targets": ["b"]}},
                   {"action": "constrain", "options": {"driven": "a"}}):
            with self.assertRaises(ValueError, msg=kw):
                a.request_body(**kw)

    def test_pose(self):
        self.assertEqual(a.request_body("pose", model="Hero", from_frame=12, layer=True,
                                        options={"preset": ["sit", "arms_crossed"], "joints": {"head": {"turn": 20}}, "x": 1}),
                         {"action": "pose", "model": "Hero", "frame": 12, "preset": ["sit", "arms_crossed"],
                          "joints": {"head": {"turn": 20}}, "layer": "new"})
        self.assertEqual(a.request_body("pose", prompt=" A person sits and reads. ", seed=4),
                         {"action": "pose", "prompt": "A person sits and reads.", "seed": 4})
        self.assertEqual(a.request_body("pose_presets"), {"action": "pose_presets"})
        with self.assertRaises(ValueError):
            a.request_body("pose")
        # On the floor, within the joints' range, or read off a picture.
        self.assertEqual(a.request_body("pose", options={"joints": {"hips": {"side": -70}}, "contacts": ["hand.R"], "limits": False}),
                         {"action": "pose", "joints": {"hips": {"side": -70}}, "contacts": ["hand.R"], "limits": False})
        self.assertEqual(a.request_body("pose", media_path=" D:/refs/dancer.png ", options={"via": "sam3d", "fov": 40, "facing": "picture"}),
                         {"action": "pose", "image": "D:/refs/dancer.png", "via": "sam3d", "fov": 40, "facing": "picture"})
        with self.assertRaises(ValueError):
            a.request_body("pose", media_path="x.png", options={"via": "openpose"})
        self.assertEqual(a.request_body("video_motion", media_path="D:/films/walk.mp4", options={"via": "sam3d", "fov": 45}),
                         {"action": "video_motion", "path": "D:/films/walk.mp4", "feet": True, "via": "sam3d", "fov": 45})
        with self.assertRaises(ValueError):
            a.request_body("video_motion", media_path="walk.mp4", options={"via": "openpose"})

    def test_export_and_checks(self):
        self.assertEqual(a.request_body("export", format=".ABC", to_frame=10),
                         {"action": "export", "format": "abc", "to": 10})
        for kw in ({"action": "dance"}, {"action": "motion"}, {"action": "export", "format": "obj"},
                   {"action": "inbetween", "from_frame": 10, "to_frame": 5}):
            out = self.call(_Client([{}]), **kw)
            self.assertEqual(out["status"], "error", kw)

    def test_errors_and_no_page(self):
        out = self.call(_Client([{"state": "error", "error": "no model named X in the 3D scene"}]), action="rig", model="X")
        self.assertEqual(out, {"status": "error", "error": "no model named X in the 3D scene"})
        with mock.patch.object(a.time, "time", side_effect=[0, 0, 31, 31]):
            out = self.call(_Client([{"state": "sent"}]), action="rig")
        self.assertIn("no viewer took the request", out["error"])


if __name__ == "__main__":
    unittest.main()
