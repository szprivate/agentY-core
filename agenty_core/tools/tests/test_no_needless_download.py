"""A file ComfyUI already has is not downloaded.

Two ways the agent got to download_hf_model for one: a preprocessor's weights
(an option of the node itself, kept in that pack's folder, in no model list -
check_model said "False"), and an installed text encoder a loader failed to
read once during a storage hiccup.

    python -m unittest agenty_core.tools.tests.test_no_needless_download
"""

import json
import unittest
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools import huggingface as hf

OBJECT_INFO = {
    "DepthAnythingV2Preprocessor": {"input": {"required": {
        "image": ["IMAGE", {}],
        "ckpt_name": [["depth_anything_v2_vitl.pth", "depth_anything_v2_vitb.pth"], {}]}}},
    "OtherDepthNode": {"input": {"optional": {
        "ckpt_name": ["COMBO", {"options": ["depth_anything_v2_vitl.pth"]}]}}},
    "KSampler": {"input": {"required": {"steps": ["INT", {"default": 20}]}}},
}
INVENTORY = {"t5xxl_fp16.safetensors": "Flux-Dev\t5xxl_fp16.safetensors"}


def unwrap(t):
    return getattr(t, "func", getattr(t, "_tool_func", t))


class NoNeedlessDownload(unittest.TestCase):

    def setUp(self):
        for target, name, value in ((C, "_get_object_info", lambda: OBJECT_INFO),
                                    (C, "_model_inventory_index", lambda: dict(INVENTORY))):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_check_model_knows_a_nodes_own_option(self):
        out = json.loads(unwrap(C.check_model)(["depth_anything_v2_vitl.pth", "t5xxl_fp16.safetensors",
                                                "not_here.safetensors"]))
        self.assertEqual(out["depth_anything_v2_vitl.pth"], "depth_anything_v2_vitl.pth")
        self.assertEqual(out["t5xxl_fp16.safetensors"], "Flux-Dev\t5xxl_fp16.safetensors")
        self.assertEqual(out["not_here.safetensors"], "False")
        note = out["_notes"]["depth_anything_v2_vitl.pth"]
        self.assertIn("DepthAnythingV2Preprocessor.ckpt_name", note)
        self.assertIn("OtherDepthNode.ckpt_name", note)
        self.assertIn("do not download", note)

    def test_check_model_takes_one_plain_name(self):
        out = json.loads(unwrap(C.check_model)("t5xxl_fp16.safetensors"))
        self.assertEqual(out, {"t5xxl_fp16.safetensors": "Flux-Dev\t5xxl_fp16.safetensors"})

    def test_neither_is_downloaded(self):
        with mock.patch.object(hf.requests, "get", side_effect=AssertionError("it went to the network")):
            managed = json.loads(unwrap(hf.download_hf_model)("some/repo", "depth_anything_v2_vitl.pth",
                                                             destination_folder="annotator"))
            have = json.loads(unwrap(hf.download_hf_model)("some/repo", "t5xxl_fp16.safetensors"))
        self.assertTrue(managed["skipped"] and "keeps and fetches" in managed["message"])
        self.assertTrue(have["skipped"] and "read error" in have["message"])
        self.assertEqual(have["path"], "Flux-Dev\t5xxl_fp16.safetensors")

    def test_a_lookup_that_fails_does_not_block_a_real_download(self):
        with mock.patch.object(C, "_model_inventory_index", side_effect=RuntimeError("no inventory")), \
                mock.patch.object(C, "_get_object_info", side_effect=RuntimeError("ComfyUI is down")), \
                mock.patch.object(hf, "_resolve_download_dir", side_effect=RuntimeError("reached the download")):
            out = json.loads(unwrap(hf.download_hf_model)("some/repo", "new_model.safetensors"))
        self.assertIn("reached the download", json.dumps(out))


if __name__ == "__main__":
    unittest.main()
