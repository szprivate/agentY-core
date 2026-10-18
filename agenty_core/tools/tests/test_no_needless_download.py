"""A file ComfyUI already has is not downloaded.

Three ways the agent got to download_hf_model for one: a preprocessor's weights
(an option of the node itself, kept in that pack's folder, in no model list -
check_model said "False"), an installed text encoder a loader failed to
read once during a storage hiccup, and a partner/API node's model — which is not
a file at all but the name of a model running on somebody else's servers.

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


# A partner/API node advertises its model on a COMFY_DYNAMICCOMBO_V3 menu whose
# options are DICTS keyed "key", not strings. The hatch above read only the two
# string shapes, so every one of those menus was invisible: 180 such inputs on a
# live server, all dict-keyed. check_model answered "False" for "MiniMax H3" and
# a repair agent went to HuggingFace for weights of a model that never runs here.
PARTNER_OBJECT_INFO = dict(OBJECT_INFO, **{
    "MinimaxHailuo03ReferenceNode": {
        "api_node": True,
        "category": "partner/video/MiniMax",
        "python_module": "comfy_api_nodes.nodes_minimax",
        "input": {"required": {
            "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                {"key": "MiniMax H3", "inputs": {"required": {"prompt": ["STRING", {}]}}},
                {"key": "MiniMax H2"}]}],
            "seed": ["INT", {"default": 0}]}}},
    # Same menu shape on a node that is NOT an API node: still nothing to
    # download, but for the other reason — so it must not be called remote.
    "LocalDynamicComboNode": {
        "input": {"required": {
            "weights": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "local_thing.pth"}]}]}}},
})


class PartnerNodeModelsAreNotFiles(unittest.TestCase):

    def setUp(self):
        for target, name, value in ((C, "_get_object_info", lambda: PARTNER_OBJECT_INFO),
                                    (C, "_model_inventory_index", lambda: dict(INVENTORY))):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)

    def test_a_partner_nodes_own_menu_value_is_recognised(self):
        got = C.node_managed_option("MiniMax H3")
        self.assertIsNotNone(got, "the dict-keyed menu is still invisible")
        self.assertEqual(got["node_class"], "MinimaxHailuo03ReferenceNode")
        self.assertEqual(got["value"], "MiniMax H3")
        self.assertTrue(got["remote"])

    def test_check_model_no_longer_says_false_for_it(self):
        out = json.loads(unwrap(C.check_model)(["MiniMax H3"]))
        self.assertEqual(out["MiniMax H3"], "MiniMax H3")
        note = out["_notes"]["MiniMax H3"]
        self.assertIn("partner's servers", note)
        self.assertIn("nothing to download", note.lower())

    def test_the_note_does_not_claim_the_node_fetches_weights(self):
        """A preprocessor does fetch its own weights; an API node fetches nothing."""
        note = json.loads(unwrap(C.check_model)(["MiniMax H3"]))["_notes"]["MiniMax H3"]
        self.assertNotIn("keeps and fetches", note)

    def test_a_local_nodes_dict_keyed_menu_is_found_but_not_called_remote(self):
        got = C.node_managed_option("local_thing.pth")
        self.assertIsNotNone(got)
        self.assertEqual(got["node_class"], "LocalDynamicComboNode")
        self.assertFalse(got["remote"])
        note = json.loads(unwrap(C.check_model)(["local_thing.pth"]))["_notes"]["local_thing.pth"]
        self.assertIn("keeps and fetches", note)

    def test_the_string_menus_still_work(self):
        """The two shapes that always worked must not regress."""
        self.assertIsNotNone(C.node_managed_option("depth_anything_v2_vitl.pth"))
        self.assertFalse(C.node_managed_option("depth_anything_v2_vitl.pth")["remote"])

    def test_a_name_no_node_offers_is_still_a_miss(self):
        self.assertIsNone(C.node_managed_option("not_a_menu_value.safetensors"))

    def test_download_hf_model_refuses_a_partner_model(self):
        with mock.patch.object(hf.requests, "get", side_effect=AssertionError("it went to the network")):
            out = json.loads(unwrap(hf.download_hf_model)("some/repo", "MiniMax H3"))
        self.assertTrue(out["skipped"])
        self.assertIn("partner's servers", out["message"])
        self.assertNotIn("keeps and fetches", out["message"])


class MenuShapeMatrix(unittest.TestCase):
    """_menu_option_labels reads every shape ComfyUI sends, and survives junk."""

    def test_bare_list_of_strings(self):
        self.assertEqual(C._menu_option_labels([["a.ckpt", "b.ckpt"]]), ["a.ckpt", "b.ckpt"])

    def test_combo_with_string_options(self):
        self.assertEqual(C._menu_option_labels(["COMBO", {"options": ["a.ckpt"]}]), ["a.ckpt"])

    def test_dynamic_combo_with_dict_options(self):
        self.assertEqual(C._menu_option_labels(
            ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "MiniMax H3"}, {"key": "H2"}]}]),
            ["MiniMax H3", "H2"])

    def test_a_dict_option_keyed_something_else(self):
        self.assertEqual(C._menu_option_labels(
            ["COMFY_DYNAMICCOMBO_V3", {"options": [{"label": "L"}, {"value": "V"}, {"text": "T"}]}]),
            ["L", "V", "T"])

    def test_options_that_are_not_labels_at_all_are_skipped(self):
        self.assertEqual(C._menu_option_labels(
            ["COMFY_DYNAMICCOMBO_V3", {"options": [{"inputs": {}}, {"key": 7}, "real.ckpt"]}]),
            ["real.ckpt"])

    def test_junk_never_raises(self):
        for junk in (None, [], "COMBO", ["COMBO"], ["COMBO", "not-a-dict"], 7):
            self.assertEqual(C._menu_option_labels(junk), [])

    def test_what_counts_as_an_api_node(self):
        self.assertTrue(C._is_api_node({"api_node": True}))
        self.assertTrue(C._is_api_node({"python_module": "comfy_api_nodes.nodes_x"}))
        self.assertFalse(C._is_api_node({"api_node": False, "python_module": "nodes"}))
        self.assertFalse(C._is_api_node({"category": "partner/video"}))  # category alone is not the flag
        self.assertFalse(C._is_api_node(None))
        self.assertFalse(C._is_api_node({}))


if __name__ == "__main__":
    unittest.main()
