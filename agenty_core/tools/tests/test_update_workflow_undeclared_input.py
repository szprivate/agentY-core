"""update_workflow refuses an input name the node does not declare.

A wire patched onto ``VAEEncode.image`` (its inputs are ``pixels`` and ``vae``)
came back ``status: ok`` and was written, so the rewiring the agent meant never
happened and the run used the old graph.

    python -m unittest agenty_core.tools.tests.test_update_workflow_undeclared_input
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools.tests.test_autogrow_addressing import H3_INFO, h3_workflow

OBJECT_INFO = {
    "VAEEncode": {"input": {"required": {"pixels": ["IMAGE", {}], "vae": ["VAE", {}]}},
                  "output": ["LATENT"]},
    "LoadImage": {"input": {"required": {"image": [["a.png"], {}]}}, "output": ["IMAGE"]},
    "ImpactSwitch": {"input": {"required": {"select": ["INT", {"default": 1}]},
                               "optional": {"input1": ["*", {}]}}, "output": ["*"]},
    "KSampler": {"input": {"required": {"seed": ["INT", {}]},
                           "hidden": {"unique_id": "UNIQUE_ID"}}, "output": ["LATENT"]},
    "MiniMaxH3ReferenceToVideo": H3_INFO,
}


def wf():
    return {
        "21": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "300": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
        "22": {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["8", 0]}},
        "5": {"class_type": "ImpactSwitch", "inputs": {"select": 1, "input1": ["21", 0]}},
        "6": {"class_type": "KSampler", "inputs": {"seed": 1}},
    }


class UndeclaredInput(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for target, value in (("_get_object_info", OBJECT_INFO), ("_server_validate", {}),
                              ("_workflows_dir", self.tmp / "agentY"),
                              ("_corpus_root", self.tmp / "corpus")):
            p = mock.patch.object(C, target, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        self.path = self.tmp / "work" / "w.json"
        self.path.parent.mkdir(parents=True)

    def run_patch(self, workflow, patches=None, add=None):
        self.path.write_text(json.dumps(workflow), encoding="utf-8")
        out = json.loads(C.update_workflow(
            str(self.path), json.dumps(patches or []), json.dumps(add or [])))
        return out, json.loads(self.path.read_text(encoding="utf-8"))

    def test_the_vaeencode_case_is_rejected_and_not_written(self):
        out, saved = self.run_patch(wf(), [{"node_id": "22", "input_name": "image",
                                            "value": ["300", 0]}])
        self.assertEqual(out["status"], "error")
        err = " ".join(out["node_errors"])
        self.assertIn("'image' is not an input of VAEEncode", err)
        self.assertIn("pixels", err)
        self.assertIn("vae", err)
        self.assertNotIn("image", saved["22"]["inputs"])
        self.assertEqual(saved["22"]["inputs"]["pixels"], ["21", 0])

    def test_a_near_miss_gets_a_did_you_mean(self):
        out, _ = self.run_patch(wf(), [{"node_id": "22", "input_name": "pixel",
                                        "value": ["300", 0]}])
        self.assertIn("Did you mean 'pixels'", " ".join(out["node_errors"]))

    def test_the_right_name_still_patches(self):
        out, saved = self.run_patch(wf(), [{"node_id": "22", "input_name": "pixels",
                                            "value": ["300", 0]}])
        self.assertEqual(out["node_errors"], [])
        self.assertEqual(saved["22"]["inputs"]["pixels"], ["300", 0])

    def test_adding_a_node_with_an_undeclared_input_is_rejected(self):
        out, saved = self.run_patch(wf(), add=[{"node_id": "40", "class_type": "VAEEncode",
                                                "inputs": {"image": ["300", 0]}}])
        self.assertTrue(any("'image' is not an input of VAEEncode" in e
                            for e in out["node_errors"]), out)
        self.assertNotIn("40", saved)

    def test_autogrow_dotted_slots_are_accepted(self):
        w = h3_workflow()
        out, saved = self.run_patch(w, [{"node_id": "9", "input_name": "ref_images.ref_image_1",
                                         "value": ["7", 0]}])
        self.assertEqual(out["node_errors"], [])
        self.assertEqual(saved["9"]["inputs"]["ref_images.ref_image_1"], ["7", 0])

    def test_a_bad_autogrow_slot_is_rejected(self):
        out, _ = self.run_patch(h3_workflow(), [{"node_id": "9", "input_name": "ref_images.ref_image_99",
                                                 "value": ["7", 0]}])
        self.assertTrue(any("not an input of MiniMaxH3ReferenceToVideo" in e
                            for e in out["node_errors"]), out)

    def test_a_kwargs_style_numbered_input_and_hidden_inputs_pass(self):
        out, saved = self.run_patch(wf(), [
            {"node_id": "5", "input_name": "input2", "value": ["300", 0]},
            {"node_id": "6", "input_name": "unique_id", "value": "6"}])
        self.assertEqual(out["node_errors"], [])
        self.assertEqual(saved["5"]["inputs"]["input2"], ["300", 0])

    def test_patch_workflow_refuses_it_too(self):
        self.path.write_text(json.dumps(wf()), encoding="utf-8")
        out = json.loads(C.patch_workflow(str(self.path), json.dumps(
            [{"node_id": "22", "input_name": "image", "value": ["300", 0]}])))
        self.assertIn("is not an input of VAEEncode", json.dumps(out))
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("image", saved["22"]["inputs"])

    def test_without_schemas_nothing_is_judged(self):
        with mock.patch.object(C, "_get_object_info", return_value={}):
            out, saved = self.run_patch(wf(), [{"node_id": "22", "input_name": "image",
                                                "value": ["300", 0]}])
        self.assertEqual(out["node_errors"], [])
        self.assertIn("image", saved["22"]["inputs"])


if __name__ == "__main__":
    unittest.main()
