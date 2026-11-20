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
        self.assertEqual(m.request_body("track", {"direction": "both", "start": 12, "seeds": "thin", "extend": True}),
                         {"action": "matchmove_track", "direction": "both", "start": 12, "seeds": "thin", "extend": True})
        self.assertEqual(m.request_body("track", {"seeds": [0, 40], "extend": ["t1"]})["seeds"], [0, 40])
        self.assertEqual(m.request_body("frame", {"frames": [0, 5]}), {"action": "matchmove_frame", "frames": [0, 5]})
        self.assertEqual(m.request_body("solve", {"depth": "ai", "depth_strength": 2}),
                         {"action": "matchmove_solve", "depth": "ai", "depth_strength": 2})
        self.assertEqual(m.request_body("track_delete", {"depth_edges": True})["depth_edges"], True)
        self.assertEqual(m.request_body("pin", {"add": [{"track": "t1", "point": [1, 2, 3.5]}, {"track": "t2", "item": "Box", "vertex": 4}],
                                                "mode": "align"})["add"][1]["vertex"], 4)
        self.assertEqual(m.request_body("pin", {"clear": True}), {"action": "matchmove_pin", "clear": True})
        self.assertEqual(m.request_body("solve", {"geometry": "hold"})["geometry"], "hold")
        for action, opts in (("track", {"direction": "sideways"}), ("track", {"seeds": "some"}), ("track", {"seeds": []}),
                             ("track", {"start": -1}), ("track", {"extend": "yes"}), ("frame", {"frames": []}),
                             ("frame", {"frames": list(range(17))}), ("frame", {"frame": -2}),
                             ("nope", {}), ("open", {}), ("lens", {"how": "magic"}), ("lens", {"how": "set"}),
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
                             ("camera", {"pixel_aspect": 9}), ("solve", {"depth": "lidar"}),
                             ("solve", {"depth": "phone", "depth_strength": 50}), ("track_delete", {"depth_edges": "maybe"}),
                             ("track_delete", {"depth_edges": False}), ("pin", {}), ("pin", {"mode": "glue"}),
                             ("pin", {"add": {"point": [0, 0, 0]}}), ("pin", {"add": {"track": "t1"}}),
                             ("pin", {"add": {"track": "t1", "point": [0, 0]}}),
                             ("pin", {"add": {"track": "t1", "item": "Box", "vertex": -1}}), ("solve", {"geometry": "glue"})):
            with self.assertRaises(ValueError, msg=(action, opts)):
                m.request_body(action, opts)

    def test_the_page_answers_through_the_rig_agent_road(self):
        page = _Page(lambda body: {"state": "done", "result": {"matchmove": {"solve": {"rms_px": 0.4}}, "got": body}})
        with mock.patch.object(ac, "_client", return_value=page), mock.patch.object(ac.time, "sleep"):
            out = json.loads(m.matchmove("solve", {"mode": "auto"}, timeout_seconds=5))
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(page.posted[0], ("/imageviewer/rig/agent", {"action": "matchmove_solve", "mode": "auto"}))
        self.assertEqual(out["result"]["matchmove"]["solve"]["rms_px"], 0.4)

    def test_frame_saves_the_plate_frames_with_their_tracks_drawn(self):
        import io
        import tempfile
        from pathlib import Path
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (320, 180), (10, 10, 10)).save(buf, "JPEG")
        jpeg = buf.getvalue()
        answer = {"state": "done", "result": {
            "frames": [{"frame": 0, "tracks": [{"id": "t1", "x": 100, "y": 50, "error_px": 0.3}]}, {"frame": 9, "tracks": []}],
            "lens": None, "undistort": False,
            "matchmove": {"plate": {"path": "D:/in/plate.####.png", "frames": 10, "width": 640, "height": 360}}}}
        page = _Page(lambda body: answer)
        got = []

        def get(path, params=None, raw=False, **kw):
            if path == "/imageviewer/mm_frame":
                got.append(params)
                return mock.Mock(content=jpeg)
            return page.asks[params["id"]]
        page.get = get
        tmp = tempfile.mkdtemp()
        with mock.patch.object(ac, "_client", return_value=page), mock.patch.object(ac.time, "sleep"), \
                mock.patch("agenty_core.tools.workspace_tools._frames_dir", return_value=Path(tmp)):
            out = json.loads(m.matchmove("frame", {"frames": [0, 9]}, timeout_seconds=5))
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(page.posted[0], ("/imageviewer/rig/agent", {"action": "matchmove_frame", "frames": [0, 9]}))
        self.assertEqual([q["frame"] for q in got], [0, 9])
        self.assertEqual(got[0]["path"], "D:/in/plate.####.png")
        fr = out["result"]["frames"]
        self.assertTrue(all(Path(f["path"]).is_file() for f in fr))
        im = Image.open(fr[0]["path"])
        self.assertGreater(im.getpixel((50 + 3, 25))[1], 60)        # the green ring of t1 (half size)
        self.assertTrue(Path(out["result"]["contact_sheet"]).is_file())
        self.assertIn("analyze_image", out["result"]["look"])

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
