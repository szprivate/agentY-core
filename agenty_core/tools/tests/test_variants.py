"""run_variants: N independent variants duplicated, patched and queued together."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools import variants as V

OI = {
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}]}},
                       "output": ["CONDITIONING"]},
    "KSampler": {"input": {"required": {"seed": ["INT", {"default": 0, "min": 0, "max": 2**31}],
                                        "positive": ["CONDITIONING"]}}, "output": ["LATENT"]},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}},
                  "output": [], "output_node": True},
}
BASE = {
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a lamp"}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["6", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "x"}},
}


class Variants(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.base = Path(self.d.name) / "base.json"
        self.base.write_text(json.dumps(BASE))
        for target, val in (("_get_object_info", OI), ("_workflows_dir", Path(self.d.name))):
            p = mock.patch.object(C, target, return_value=val)
            p.start()
            self.addCleanup(p.stop)
        srv = mock.patch.object(C, "_server_validate", side_effect=AssertionError("server validation must be skipped"))
        srv.start()
        self.addCleanup(srv.stop)
        self.addCleanup(V.set_batch_hook, None)

    def prompts(self, result):
        return [json.loads(Path(v["workflow_path"]).read_text())["6"]["inputs"]["text"] for v in result["variants"]]

    def test_set_shorthand_patches_each_copy(self):
        r = V.prepare_variants(str(self.base), [{"label": "a", "set": {"6.text": "a warm lamp"}},
                                                {"label": "b", "set": {"6.text": "a cold lamp"}}, {}])
        self.assertEqual(r["status"], "ok", r)
        self.assertEqual(self.prompts(r), ["a warm lamp", "a cold lamp", "a lamp"])
        self.assertEqual(len(set(v["workflow_path"] for v in r["variants"])), 3)
        self.assertEqual(json.loads(self.base.read_text())["6"]["inputs"]["text"], "a lamp")   # base untouched

    def test_each_copy_has_its_own_seed(self):
        r = V.prepare_variants(str(self.base), [{}, {}, {}])
        seeds = {json.loads(Path(v["workflow_path"]).read_text())["3"]["inputs"]["seed"] for v in r["variants"]}
        self.assertEqual(len(seeds), 3)

    def test_the_full_patch_shape_works(self):
        r = V.prepare_variants(str(self.base), [{"patches": [{"node_id": "6", "input_name": "text", "value": "x"}]}])
        self.assertEqual(self.prompts(r), ["x"])

    def test_a_bad_variant_is_reported_and_the_rest_still_run(self):
        r = V.prepare_variants(str(self.base), [{"set": {"6.text": "ok"}}, {"set": {"99.text": "no such node"}}])
        self.assertEqual(r["status"], "partial", r)
        self.assertEqual(len(r["ready"]), 1)
        self.assertEqual(r["variants"][1]["status"], "error")
        self.assertTrue(r["variants"][1]["problems"])

    def test_no_variants_and_too_many_are_errors(self):
        self.assertEqual(V.prepare_variants(str(self.base), [])["status"], "error")
        self.assertEqual(V.prepare_variants(str(self.base), [{}] * (V.MAX_VARIANTS + 1))["status"], "error")

    def test_run_queues_all_ready_copies_in_ONE_hook_call(self):
        calls = []
        V.set_batch_hook(lambda paths: calls.append(list(paths)) or
                         {"outputs": {p: [f"out_{i}.png"] for i, p in enumerate(paths)}})
        out = json.loads(V.run_variants(str(self.base), [{"set": {"6.text": "a"}}, {"set": {"6.text": "b"}},
                                                         {"set": {"6.text": "c"}}, {"set": {"6.text": "d"}}]))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 4)
        self.assertTrue(out["ran"])
        self.assertEqual([v["outputs"] for v in out["variants"]], [["out_0.png"], ["out_1.png"], ["out_2.png"], ["out_3.png"]])

    def test_run_false_only_builds(self):
        V.set_batch_hook(lambda paths: self.fail("must not run"))
        out = json.loads(V.run_variants(str(self.base), [{}, {}], run=False))
        self.assertFalse(out["ran"])
        self.assertEqual(len(out["ready"]), 2)

    def test_variants_may_arrive_as_json_text(self):
        r = V.prepare_variants(str(self.base), json.dumps([{"set": {"6.text": "j"}}]))
        self.assertEqual(self.prompts(r), ["j"])


if __name__ == "__main__":
    unittest.main()
