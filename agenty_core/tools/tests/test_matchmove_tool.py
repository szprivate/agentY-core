"""matchmove offline: the requests it builds and refuses, and the wait on the page, the ComfyUI client mocked."""

import json
import unittest
from unittest import mock

from agenty_core.tools import animate_character as ac
from agenty_core.tools import matchmove as m


class _Page:
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


class MatchmoveToolTests(unittest.TestCase):
    def test_request_bodies(self):
        self.assertEqual(m.request_body("info"), {"action": "matchmove_info"})
        self.assertEqual(m.request_body("open", {"plate": "/a/plate.1001.exr", "action": "x"}),
                         {"action": "matchmove_open", "plate": "/a/plate.1001.exr"})
        self.assertEqual(m.request_body("solve", {"mode": "tripod"})["action"], "matchmove_solve")
        self.assertEqual(m.request_body("lens", {"how": "set", "k1": -0.05})["k1"], -0.05)
        self.assertEqual(m.request_body("mask", {"keys": [{"frame": 0, "points": [[0, 0], [9, 0], [9, 9]]}]})["action"],
                         "matchmove_mask")
        self.assertEqual(m.request_body("ground", {"how": "phone"}), {"action": "matchmove_ground", "how": "phone"})
        self.assertEqual(m.request_body("phone_export"), {"action": "matchmove_phone_export"})
        self.assertEqual(m.request_body("ai_mask", {"engine": "background"})["action"], "matchmove_ai_mask")
        self.assertEqual(m.request_body("measure", {"add": {"kind": "distance", "ids": ["t1", "t2"], "metres": 1.2}})["add"]["metres"], 1.2)
        self.assertEqual(m.request_body("camera", {"preset": "iphone14-wide"})["preset"], "iphone14-wide")
        for action, opts in (("nope", {}), ("open", {}), ("lens", {"how": "magic"}), ("lens", {"how": "set"}),
                             ("lens", {"how": "comp"}), ("lens", {"how": "lines", "lines": [[[0, 0], [1, 1]]]}),
                             ("mask", {}), ("mask", {"keys": [{"frame": 0, "points": [[0, 0]]}]}),
                             ("mask", {"keys": [{"points": [[0, 0], [1, 0], [1, 1]]}]}),
                             ("track_add", {"points": [{"x": 1}]}), ("track_delete", {}), ("solve", {"mode": "orbit"}),
                             ("orient", {"how": "ground"}), ("orient", {"how": "scale", "points": ["t1", "t2"]}),
                             ("geometry", {"kind": "cone", "points": ["t1"]}), ("lens_export", {"to": "nk"}),
                             ("track", {"from": 10, "to": 2}), ("ground", {"how": "lidar"}),
                             ("ground", {"how": "points", "points": ["t1"]}), ("ground", {"how": "depth", "scale": 2}),
                             ("ai_mask", {}), ("ai_mask", {"engine": "magic"}), ("measure", {}),
                             ("measure", {"add": {"kind": "distance", "ids": ["t1", "t2"]}}),
                             ("measure", {"add": {"kind": "plane", "ids": ["t1", "t2"]}}),
                             ("camera", {"pixel_aspect": 9})):
            with self.assertRaises(ValueError, msg=(action, opts)):
                m.request_body(action, opts)

    def test_the_page_answers_through_the_rig_agent_road(self):
        page = _Page(lambda body: {"state": "done", "result": {"matchmove": {"solve": {"rms_px": 0.4}}, "got": body}})
        with mock.patch.object(ac, "_client", return_value=page), mock.patch.object(ac.time, "sleep"):
            out = json.loads(m.matchmove("solve", {"mode": "auto"}, timeout_seconds=5))
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(page.posted[0], ("/imageviewer/rig/agent", {"action": "matchmove_solve", "mode": "auto"}))
        self.assertEqual(out["result"]["matchmove"]["solve"]["rms_px"], 0.4)

    def test_a_refused_request_never_reaches_the_page(self):
        page = _Page(lambda body: {"state": "done"})
        with mock.patch.object(ac, "_client", return_value=page):
            out = json.loads(m.matchmove("orient", {"how": "spin", "points": ["a"]}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(page.posted, [])

    def test_the_actions_are_the_viewers(self):
        """The viewer's list (MATCHMOVE_AGENT_ACTIONS) is matchmove_ + these; checked against a checkout beside this one."""
        import os
        import re
        js = os.path.join(os.path.dirname(__file__), *[".."] * 4, "ComfyUI-ImageViewer", "js",
                          "imageViewer_previzAgentMatchmove.js")
        if not os.path.isfile(js):
            self.skipTest("no ComfyUI-ImageViewer checkout beside agentY-core")
        src = open(js, encoding="utf-8").read()
        block = src.split("MATCHMOVE_AGENT_ACTIONS = [", 1)[1].split("]", 1)[0]
        self.assertEqual(sorted(re.findall(r'"matchmove_(\w+)"', block)), sorted(m.ACTIONS))


if __name__ == "__main__":
    unittest.main()
