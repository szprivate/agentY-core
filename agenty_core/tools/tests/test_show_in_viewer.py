"""show_in_viewer offline: the ComfyUI client is mocked."""

import json
import os
import unittest
from unittest import mock

from agenty_core.tools import show_in_viewer as s


def run(args, answer=None, exc=None):
    client = mock.Mock()
    if exc:
        client.post.side_effect = exc
    else:
        client.post.return_value = answer
    with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client):
        out = json.loads(s.show_in_viewer(**args))
    return out, client


class ShowInViewerTests(unittest.TestCase):
    def tearDown(self):
        s.set_default_tab_provider(None)

    def test_posts_to_push_with_defaults(self):
        out, c = run({"paths": ["clip3.mp4"]}, {"ok": True, "shown": 1, "refused": [], "opened": True})
        (route,) = c.post.call_args.args
        body = c.post.call_args.kwargs["json_data"]
        self.assertEqual(route, "/bepic/push")
        self.assertEqual(body["paths"], [os.path.abspath("clip3.mp4")])
        self.assertEqual(body["tab_name"], "agentY")
        self.assertTrue(body["focus"])
        self.assertEqual(body["panel"], "")
        self.assertEqual(out["shown"], 1)
        self.assertNotIn("error", out)

    def test_named_tab_panel_and_provider(self):
        s.set_default_tab_provider(lambda: "Mine")
        _, c = run({"paths": ["a.mp4"], "panel": "Edit"}, {"ok": True, "shown": 1})
        self.assertEqual(c.post.call_args.kwargs["json_data"]["tab_name"], "Mine")
        self.assertEqual(c.post.call_args.kwargs["json_data"]["panel"], "edit")
        _, c = run({"paths": ["a.png"], "tab_name": " side  by side "}, {"ok": True, "shown": 1})
        self.assertEqual(c.post.call_args.kwargs["json_data"]["tab_name"], "side by side")

    def test_nothing_shown_is_an_error(self):
        out, _ = run({"paths": ["/etc/x.png"]}, {"ok": True, "shown": 0, "refused": ["/etc/x.png"]})
        self.assertIn("error", out)
        self.assertEqual(out["refused"], ["/etc/x.png"])

    def test_a_motion_scene_opened_is_not_an_error(self):
        out, _ = run({"paths": ["C:/x/title.mgscene.json"]}, {"ok": True, "shown": 0, "motion": ["C:/x/title.mgscene.json"]})
        self.assertNotIn("error", out)
        self.assertEqual(out["motion"], ["C:/x/title.mgscene.json"])

    def test_old_viewer_404(self):
        out, _ = run({"paths": ["a.png"]}, exc=RuntimeError("404 Not Found"))
        self.assertIn("too old", out["error"])

    def test_no_paths(self):
        out, c = run({"paths": []})
        self.assertIn("error", out)
        c.post.assert_not_called()

    def test_single_string_path(self):
        _, c = run({"paths": "a.png"}, {"ok": True, "shown": 1})
        self.assertEqual(len(c.post.call_args.kwargs["json_data"]["paths"]), 1)


if __name__ == "__main__":
    unittest.main()
