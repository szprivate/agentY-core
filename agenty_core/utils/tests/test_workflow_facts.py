"""A template's description leads with what it does and what is distinctive.

Only the first sentence of a description, cut to 60 characters, is shown when a
template is picked. "Example workflow X shipped with pack Y" spent it on nothing.
"""

import json
import tempfile
import unittest
from pathlib import Path

from agenty_core import templates_sync as ts
from agenty_core.utils import workflow_facts as wf

OBJECT_INFO = {
    "SAM3Grounding": {"python_module": "custom_nodes.ComfyUI-SAM3", "description": "Segment by text."},
    "LoadSAM3Model": {"python_module": "custom_nodes.ComfyUI-SAM3"},
    "LoadImage": {"python_module": "nodes"},
    "PreviewImage": {"python_module": "nodes"},
    "KlingOmniPro": {"python_module": "comfy_api_nodes.nodes_kling", "api_node": True},
}
SAM3 = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "LoadSAM3Model", "inputs": {}},
    "3": {"class_type": "SAM3Grounding", "inputs": {"model": ["2", 0], "image": ["1", 0]}},
    "4": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
}


class Facts(unittest.TestCase):

    def test_nodes_are_sorted_by_whose_they_are(self):
        f = wf.workflow_facts(SAM3, "image_seg_text", object_info=OBJECT_INFO)
        self.assertEqual(sorted(f["pack_nodes"]["ComfyUI-SAM3"]), ["LoadSAM3Model", "SAM3Grounding"])
        self.assertEqual(f["node_descriptions"], {"SAM3Grounding": "Segment by text."})
        self.assertEqual((f["inputs"], f["outputs"]), ({"image": 1}, {"image": 1}))
        self.assertNotIn("LoadImage", f["classes"])        # plumbing tells nothing apart

    def test_api_nodes_are_recognised(self):
        f = wf.workflow_facts({"1": {"class_type": "KlingOmniPro", "inputs": {}}}, "x",
                              object_info=OBJECT_INFO)
        self.assertEqual(f["api_nodes"], ["KlingOmniPro"])
        self.assertTrue(wf.describe_from_facts(f).startswith("[API] "))

    def test_notes_of_a_ui_graph_are_kept(self):
        ui = {"nodes": [{"id": 1, "type": "MarkdownNote", "widgets_values": ["Segments  by\ntext prompt"]}],
              "links": []}
        self.assertEqual(wf.workflow_facts(ui, "x", object_info={})["notes"], "Segments by text prompt")

    def test_garbage_yields_empty_facts_not_an_error(self):
        self.assertEqual(wf.workflow_facts("nope", "x", object_info={})["classes"], [])
        self.assertEqual(wf.workflow_facts({"nodes": [None, 3]}, "x", object_info={})["pack_nodes"], {})


class DescribeFromFacts(unittest.TestCase):

    def test_the_first_sentence_names_the_operation_and_the_packs_nodes(self):
        f = wf.workflow_facts(SAM3, "image_seg_text", object_info=OBJECT_INFO)
        f["task"] = "segmentation"
        line = wf.describe_from_facts(f, pack="ComfyUI-SAM3", workflow="image_seg_text")
        first = line.split(". ")[0]
        self.assertTrue(first.startswith("[Local] Segmentation with "), first)
        self.assertIn("SAM3Grounding", first)
        self.assertIn("1 image in -> 1 image out", line)
        self.assertIn("shipped with the ComfyUI-SAM3", line)

    def test_without_a_task_it_still_says_something_true(self):
        line = wf.describe_from_facts({"media": "video", "classes": ["Foo"]})
        self.assertTrue(line.startswith("[Local] Video workflow with Foo."), line)


class WrittenDescriptionsSurviveASync(unittest.TestCase):
    """agentY's model pass writes a description and the file's sha beside it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.index = self.root / ts.NODE_PACK_DIR / "index.json"

    def _sync(self, files):
        listing = {"A": sorted(n.split("/")[1][:-5] for n in files)}

        def get(url, timeout=60):
            if url.endswith("/api/workflow_templates"):
                return json.dumps(listing).encode()
            return json.dumps(files[url.split("/api/workflow_templates/")[1]]).encode()
        return ts.sync_node_pack_examples("http://c", root=self.root, get=get, log=lambda *_: None,
                                          describe=lambda n, p, w, d: {"name": n, "description": "graph",
                                                                       "description_source": "graph"})

    def _entries(self):
        return {t["name"]: t for g in json.loads(self.index.read_text("utf-8")) for t in g["templates"]}

    def test_kept_for_an_unchanged_file_dropped_for_a_changed_one(self):
        v1 = {"1": {"class_type": "KSampler", "inputs": {}}}
        self._sync({"A/one.json": v1, "A/two.json": v1})
        index = json.loads(self.index.read_text("utf-8"))
        for t in index[0]["templates"]:
            raw = (self.index.parent / f"{t['name']}.json").read_bytes()
            t.update({"description": "written", "description_source": "llm",
                      "description_sha": ts.blob_sha(raw)})
        self.index.write_text(json.dumps(index), "utf-8")

        self._sync({"A/one.json": v1, "A/two.json": {"1": {"class_type": "Other", "inputs": {}}},
                    "A/three.json": v1})
        got = self._entries()
        self.assertEqual(got["A__one"]["description"], "written")
        self.assertEqual(got["A__two"]["description"], "graph")
        self.assertEqual(got["A__three"]["description"], "graph")


if __name__ == "__main__":
    unittest.main()
