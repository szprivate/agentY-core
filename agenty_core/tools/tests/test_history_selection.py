"""History-panel picks resolved to files the agent can use.

The page (agentY-comfyuiConnect) sends ``{viewer, assets}``; ``resolve`` turns it
into one list with absolute paths, the run id and the sidecar's role, and
``describe_lines`` is what the agentY host lists at the top of a turn.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import history_selection as hs
from agenty_core.utils import turn_scope

RUN = "0f8fad5b-d9cb-469f-a165-70867728950e"


class Resolve(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.dirs = {"input": str(root / "input"), "output": str(root / "output"),
                     "temp": str(root / "temp")}
        for d in self.dirs.values():
            Path(d).mkdir()
        (root / "output" / "shots").mkdir()
        self.hero = root / "output" / "shots" / "hero_001.png"
        self.hero.write_bytes(b"x")
        (root / "output" / "shots" / "hero_001.png.agenty.json").write_text(
            json.dumps({"role": "hero sheet, front", "made_by": "agentY"}), encoding="utf-8")
        (root / "input" / "ref.jpg").write_bytes(b"x")
        (root / "temp" / "prev_00001_.png").write_bytes(b"x")

    def test_nothing_picked_is_an_empty_answer(self):
        for sel in (None, {}, {"viewer": {"items": []}}):
            self.assertEqual(hs.resolve(sel, dirs=self.dirs)["items"], [])
        self.assertEqual(hs.describe_lines({"items": []}), ("", ""))

    def test_viewer_items_keep_their_path_and_gain_the_sidecar_role(self):
        sel = {"viewer": {"tab": "named:agentY", "tab_label": "agentY", "mode": "multi", "count": 2,
                          "items": [{"index": 0, "path": str(self.hero), "kind": "image",
                                     "prompt_id": RUN, "agent_tag": "hero"},
                                    {"index": 3, "filename": "hero_001.png", "subfolder": "shots",
                                     "type": "output", "label": "Hero"}]}}
        res = hs.resolve(sel, dirs=self.dirs)
        self.assertEqual(res["count"], 2)
        a, b = res["items"]
        self.assertEqual(a["role"], "hero sheet, front")
        self.assertTrue(a["exists"])
        self.assertEqual(a["source"], "viewer")
        self.assertEqual(Path(b["path"]), self.hero, "filename/subfolder/type joined onto output")
        self.assertEqual(res["panels"][0]["how"], "multi")

    def test_asset_ids_of_every_shape(self):
        out_id = f'{RUN}-' + json.dumps(["9", "shots", "hero_001.png"])
        lookups = []

        def lookup(pid):
            lookups.append(pid)
            return {pid: {"outputs": {"9": {"images": [
                {"filename": "prev_00001_.png", "subfolder": "", "type": "temp"},
                {"filename": "hero_001.png", "subfolder": "shots", "type": "output"}]}}}}

        sel = {"assets": {"count": 4, "ids": [RUN, out_id, "input-3-ref.jpg", "some-asset-uuid"]}}
        res = hs.resolve(sel, dirs=self.dirs, history_lookup=lookup)
        run, one, inp, other = res["items"]
        self.assertEqual(lookups, [RUN], "only a bare run id needs ComfyUI's history")
        self.assertEqual(run["prompt_id"], RUN)
        self.assertEqual(Path(run["path"]), self.hero)
        self.assertEqual(len(run["run_outputs"]), 2)
        self.assertEqual(Path(one["path"]), self.hero)
        self.assertEqual(one["prompt_id"], RUN)
        self.assertEqual(one["role"], "hero sheet, front")
        self.assertEqual(Path(inp["path"]), Path(self.dirs["input"]) / "ref.jpg")
        self.assertTrue(other["unresolved"])
        self.assertNotIn("path", other)

    def test_block_lines_are_capped_with_a_count(self):
        items = [{"source": "viewer", "path": f"D:/o/{i}.png", "kind": "image"} for i in range(20)]
        where, lines = hs.describe_lines(
            {"count": 20, "items": items,
             "panels": [{"panel": "bEpic viewer history", "tab": "agentY", "how": "multi", "count": 20}]},
            cap=5)
        self.assertEqual(where, 'bEpic viewer history, tab "agentY" (selected)')
        rows = lines.splitlines()
        self.assertEqual(len(rows), 6)
        self.assertIn("15 more", rows[-1])
        self.assertTrue(rows[0].startswith("1. image: D:/o/0.png"))

    def test_block_line_carries_role_tag_run_and_missing(self):
        _, lines = hs.describe_lines({"items": [{"path": "D:/x.mp4", "kind": "video", "role": "intro",
                                                 "agent_tag": "intro", "prompt_id": RUN,
                                                 "exists": False}]})
        self.assertIn('is: "intro"', lines)
        self.assertIn("#intro", lines)
        self.assertIn(RUN, lines)
        self.assertIn("FILE MISSING", lines)


class Tool(unittest.TestCase):

    def setUp(self):
        tok = turn_scope.enter(turn_scope.Scope("r", "t"))
        self.addCleanup(turn_scope.leave, tok)
        self.gets = []

    def _client(self, answer):
        test = self

        class C:
            def get(self, path):
                test.gets.append(path)
                return answer
        return C()

    def test_reads_the_route_aimed_at_the_turns_page(self):
        turn_scope.current().set("canvas_client", "sid-7")
        ans = {"ok": True, "selection": {"viewer": {"items": [{"path": "D:/a.png"}], "count": 1}}}
        with mock.patch("agenty_core.utils.comfyui_client.get_client", lambda: self._client(ans)):
            out = json.loads(hs.get_history_selection())
        self.assertEqual(self.gets, ["/agent/history_selection?client_id=sid-7"])
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["items"][0]["path"], "D:/a.png")

    def test_nothing_picked_says_how_to_pick(self):
        with mock.patch("agenty_core.utils.comfyui_client.get_client",
                        lambda: self._client({"ok": True, "selection": None})):
            out = json.loads(hs.get_history_selection())
        self.assertEqual(out["count"], 0)
        self.assertIn("Ctrl+click", out["hint"])

    def test_a_turn_with_no_canvas_asks_nobody(self):
        turn_scope.current().set("canvas_detached", True)
        out = json.loads(hs.get_history_selection())
        self.assertIn("no canvas attached", out["error"])
        self.assertEqual(self.gets, [])


if __name__ == "__main__":
    unittest.main()
