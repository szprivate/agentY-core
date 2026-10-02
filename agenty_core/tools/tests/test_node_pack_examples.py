"""Example workflows shipped with node packs are mirrored into the custom corpus.

A pack documents itself with an ``example_workflows`` folder; ComfyUI lists those
at /api/workflow_templates and serves the files. For a pack's nodes they are the
only templates there are, and the agent had none of them.
"""

import json
import tempfile
import unittest
from pathlib import Path

from agenty_core import templates_sync as ts

BASE = "http://comfy"
GRAPH = {"nodes": [{"id": 1, "type": "SAM3Segment"}], "links": []}
API = {"1": {"class_type": "KSampler", "inputs": {}}}


def _server(listing, files):
    def get(url, timeout=60):
        if url == f"{BASE}/api/workflow_templates":
            return json.dumps(listing).encode()
        key = url.replace(f"{BASE}/api/workflow_templates/", "")
        if key in files:
            data = files[key]
            return data if isinstance(data, bytes) else json.dumps(data).encode()
        raise OSError(f"404 {url}")
    return get


def _describe(name, pack, wf, data):
    return {"name": name, "title": wf, "description": f"{pack}", "models": [], "io": {}}


class NodePackExamples(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mirror = self.root / ts.NODE_PACK_DIR

    def _sync(self, listing, files):
        return ts.sync_node_pack_examples(BASE, root=self.root, get=_server(listing, files),
                                          log=lambda *_: None, describe=_describe)

    def test_examples_land_under_the_packs_name_with_an_index(self):
        res = self._sync({"ComfyUI-SAM3": ["image seg"], "Impact": ["basic"]},
                         {"ComfyUI-SAM3/image%20seg.json": GRAPH, "Impact/basic.json": API})
        self.assertEqual(res["status"], "synced")
        self.assertEqual(res["added"], ["ComfyUI-SAM3__image_seg.json", "Impact__basic.json"])
        self.assertEqual(json.loads((self.mirror / "Impact__basic.json").read_text("utf-8")), API)
        index = json.loads((self.mirror / "index.json").read_text("utf-8"))
        self.assertEqual([g["moduleName"] for g in index], ["ComfyUI-SAM3", "Impact"])
        self.assertEqual(index[0]["templates"][0]["name"], "ComfyUI-SAM3__image_seg")

    def test_two_packs_with_the_same_workflow_name_do_not_collide(self):
        res = self._sync({"A": ["basic"], "B": ["basic"]}, {"A/basic.json": API, "B/basic.json": GRAPH})
        self.assertEqual(res["added"], ["A__basic.json", "B__basic.json"])

    def test_what_is_not_a_workflow_or_not_served_is_skipped(self):
        res = self._sync({"A": ["settings", "broken", "gone", "ok"]},
                         {"A/settings.json": {"theme": "dark"}, "A/broken.json": b"{not json",
                          "A/ok.json": API})
        self.assertEqual(res["added"], ["A__ok.json"])
        self.assertEqual(sorted(res["skipped"]), ["A/broken", "A/gone", "A/settings"])

    def test_a_second_run_with_nothing_new_writes_nothing(self):
        self._sync({"A": ["ok"]}, {"A/ok.json": API})
        stamp = (self.mirror / "index.json").stat().st_mtime_ns
        res = self._sync({"A": ["ok"]}, {"A/ok.json": API})
        self.assertEqual(res["status"], "current")
        self.assertEqual((self.mirror / "index.json").stat().st_mtime_ns, stamp)

    def test_a_changed_example_is_replaced_and_a_removed_pack_goes(self):
        self._sync({"A": ["ok"], "B": ["x"]}, {"A/ok.json": API, "B/x.json": API})
        res = self._sync({"A": ["ok"]}, {"A/ok.json": GRAPH})
        self.assertEqual((res["changed"], res["removed"]), (["A__ok.json"], ["B__x.json"]))
        self.assertFalse((self.mirror / "B__x.json").exists())
        index = json.loads((self.mirror / "index.json").read_text("utf-8"))
        self.assertEqual([g["moduleName"] for g in index], ["A"])

    def test_the_users_own_templates_are_never_touched(self):
        own = self.root / "comfyui_workflow_templates_custom" / "templates"
        own.mkdir(parents=True)
        (own / "mine.json").write_text("{}", encoding="utf-8")
        (own / "index.json").write_text("[]", encoding="utf-8")
        self._sync({"A": ["ok"]}, {"A/ok.json": API})
        self._sync({}, {})
        self.assertEqual(sorted(p.name for p in own.glob("*.json")), ["index.json", "mine.json"])

    def test_a_silent_comfyui_is_unreachable_not_an_empty_corpus(self):
        self._sync({"A": ["ok"]}, {"A/ok.json": API})

        def dead(url, timeout=60):
            raise OSError("refused")
        with self.assertRaises(ts.ComfyUIUnreachable):
            ts.sync_node_pack_examples(BASE, root=self.root, get=dead, log=lambda *_: None)
        self.assertTrue((self.mirror / "A__ok.json").exists())

    def test_refresh_rebuilds_recipes_only_when_something_changed(self):
        calls = []
        kw = dict(root=self.root, get=_server({"A": ["ok"]}, {"A/ok.json": API}),
                  log=lambda *_: None, regenerate=lambda: calls.append(1) or {"recipe_count": 3})
        self.assertEqual(ts.refresh_node_pack_examples(BASE, **kw)["recipes"], {"recipe_count": 3})
        self.assertNotIn("recipes", ts.refresh_node_pack_examples(BASE, **kw))
        self.assertEqual(calls, [1])


class CatalogReadsThem(unittest.TestCase):

    def test_the_custom_index_includes_the_node_pack_examples(self):
        from agenty_core.tools import comfyui as C
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "node_packs").mkdir()
            (base / "index.json").write_text(json.dumps([{"templates": [{"name": "mine"}]}]), "utf-8")
            (base / "node_packs" / "index.json").write_text(
                json.dumps([{"title": "A (node pack examples)", "templates": [{"name": "A__ok"}]}]), "utf-8")
            (base / "node_packs" / "A__ok.json").write_text(json.dumps(API), "utf-8")
            old_dir, old_cache = C._custom_templates_dir, C._index_cache
            C._custom_templates_dir, C._index_cache = (lambda: base), None
            try:
                self.assertEqual([t["name"] for t in C._load_index()], ["mine", "A__ok"])
                self.assertEqual(C._find_template_file(base, "A__ok"), base / "node_packs" / "A__ok.json")
            finally:
                C._custom_templates_dir, C._index_cache = old_dir, old_cache


if __name__ == "__main__":
    unittest.main()
