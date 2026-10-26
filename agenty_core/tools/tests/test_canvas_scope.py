"""The canvas tools only touch the page their turn belongs to.

The extension's /agent/* routes broadcast to every open ComfyUI page. A host that
runs turns nobody streams records on the turn's scope which page it may touch
(`canvas_client`), or that it may touch none (`canvas_detached`); the tools obey.
A caller that sets neither (a plain MCP client) behaves as it always did.
"""

import json
import unittest
from unittest import mock

from agenty_core.tools import comfyui
from agenty_core.utils import canvas_scope, turn_scope

GRAPH = {"nodes": [{"id": 1, "type": "KSampler"}], "links": []}


class _Client:
    def __init__(self):
        self.posts, self.gets = [], []

    def post(self, path, json_data=None):
        self.posts.append((path, json_data))
        return {"ok": True, "applied": ["text"], "node": "x"}

    def get(self, path):
        self.gets.append(path)
        return {"ok": True, "nodes": [{"id": 1}]}


class CanvasScopeTest(unittest.TestCase):

    def setUp(self):
        self.client = _Client()
        self.enterContext(mock.patch.object(comfyui, "get_client", lambda: self.client))
        self.enterContext(mock.patch.object(comfyui, "_load_workflow", lambda p: GRAPH))
        self.enterContext(mock.patch.object(comfyui, "_is_graph_format", lambda w: True))
        tok = turn_scope.enter(turn_scope.Scope("r", "t"))
        self.addCleanup(turn_scope.leave, tok)

    def _loads(self):
        return [p for p in self.client.posts if p[0] == "/agent/load_workflow"]

    def test_a_turn_with_no_canvas_saves_the_workflow_and_pushes_nothing(self):
        turn_scope.current().set("canvas_detached", True)
        out = json.loads(comfyui.open_workflow_in_canvas("w.json", name="mine"))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["saved_as"], "agent/mine")
        self.assertFalse(out["opened_on_canvas"])
        self.assertTrue(out["no_canvas"])
        self.assertEqual(self._loads(), [], "nothing reached any open page")
        self.assertTrue(any(p[0].startswith("/userdata/") for p in self.client.posts))

    def test_a_bound_turn_aims_the_load_at_its_own_page(self):
        turn_scope.current().set("canvas_client", "sid-1")
        comfyui.open_workflow_in_canvas("w.json")
        self.assertEqual(self._loads()[0][1]["client_id"], "sid-1")

    def test_an_unscoped_caller_still_broadcasts(self):
        comfyui.open_workflow_in_canvas("w.json")
        self.assertNotIn("client_id", self._loads()[0][1])

    def test_node_params_and_selection_follow_the_same_rule(self):
        turn_scope.current().set("canvas_detached", True)
        for res in (comfyui.set_canvas_node_params("1", {"text": "x"}), comfyui.get_canvas_selection()):
            self.assertIn("no canvas attached", json.loads(res)["error"])
        self.assertEqual(self.client.posts, [])
        self.assertEqual(self.client.gets, [])

        turn_scope.current().set("canvas_detached", False)
        turn_scope.current().set("canvas_client", "sid-2")
        comfyui.set_canvas_node_params("1", {"text": "x"})
        comfyui.get_canvas_selection()
        self.assertEqual(self.client.posts[-1][1]["client_id"], "sid-2")
        self.assertIn("client_id=sid-2", self.client.gets[-1])

    def test_detached_means_a_turn_with_no_page_of_its_own(self):
        self.assertFalse(canvas_scope.detached())
        turn_scope.current().set("canvas_detached", True)
        self.assertTrue(canvas_scope.detached())
        turn_scope.current().set("canvas_client", "sid-3")
        self.assertFalse(canvas_scope.detached(), "a page is bound: not detached")


if __name__ == "__main__":
    unittest.main()
