"""save_workflow / open_workflow_in_canvas refuse a workflow ComfyUI would draw red."""
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from agenty_core.tools import comfyui as C

OI = {
    "UNETLoader": {"input": {"required": {
        "unet_name": [["real_unet.safetensors", "other.safetensors"]],
        "weight_dtype": [["default", "fp8"]]}}, "output": ["MODEL"], "output_name": ["MODEL"]},
    "KSampler": {"input": {"required": {
        "model": ["MODEL"], "seed": ["INT", {"default": 0, "control_after_generate": True}],
        "steps": ["INT", {"default": 20}], "cfg": ["FLOAT", {"default": 8.0}],
        "sampler_name": [["euler", "dpmpp_2m"]], "scheduler": [["simple", "karras"]],
        "denoise": ["FLOAT", {"default": 1.0}]}}, "output": ["LATENT"], "output_name": ["LATENT"]},
}


def ksampler(wv):
    return {"id": 2, "type": "KSampler", "inputs": [{"name": "model", "type": "MODEL", "link": 1}],
            "outputs": [], "widgets_values": wv}


def ui(nodes):
    return {"version": 0.4, "last_node_id": 9, "last_link_id": 1, "links": [], "nodes": nodes}


class Guard(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(C, "_get_object_info", return_value=OI)
        p.start(); self.addCleanup(p.stop)

    def test_clean_ui_graph(self):
        g = ui([ksampler([5, "fixed", 20, 7.0, "euler", "simple", 1.0])])
        self.assertEqual(C.check_workflow_against_schema(g), [])

    def test_missing_class_and_model_and_frontend_only_note(self):
        g = ui([{"id": 1, "type": "Nope", "widgets_values": []},
                {"id": 3, "type": "MarkdownNote", "widgets_values": ["hi"]},
                {"id": 4, "type": "UNETLoader", "widgets_values": ["ltx-invented.safetensors", "default"]}])
        pr = C.check_workflow_against_schema(g)
        self.assertEqual(len(pr), 2)
        self.assertIn("node 1 Nope: no such class", pr[0])
        self.assertIn("unet_name 'ltx-invented.safetensors'", pr[1])
        self.assertIn("real_unet.safetensors", pr[1])

    def test_shifted_widget_values_flagged(self):
        # seed/control missing: every later value lands one slot early
        pr = C.check_workflow_against_schema(ui([ksampler([20, 7.0, "euler", "simple", 1.0])]))
        self.assertTrue(any("sampler_name" in p or "left over" in p or "scheduler" in p for p in pr), pr)

    def test_display_text_on_widgetless_node_is_not_a_value(self):
        oi = dict(OI, GetImageSize={"input": {"required": {"image": ["IMAGE", {}]}}, "output": ["INT"]})
        node = {"id": 7, "type": "GetImageSize", "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
                "widgets_values": ["width: 1365, height: 768 batch size: 1"]}
        with mock.patch.object(C, "_get_object_info", return_value=oi):
            self.assertEqual(C.check_workflow_against_schema(ui([node])), [])

    def test_api_format(self):
        api = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "bad.safetensors", "weight_dtype": "fp8"}},
               "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1, "steps": 20, "cfg": 7,
                                                           "sampler_name": "euler", "scheduler": "simple",
                                                           "denoise": 1}}}
        pr = C.check_workflow_against_schema(api)
        self.assertEqual(len(pr), 1)
        self.assertIn("unet_name", pr[0])

    def test_unreachable_object_info_means_unchecked(self):
        with mock.patch.object(C, "_get_object_info", side_effect=RuntimeError("down")):
            self.assertIsNone(C.check_workflow_against_schema(ui([])))

    def test_save_workflow_refuses_then_forces_then_warns(self):
        bad = {"1": {"class_type": "Nope", "inputs": {}}}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(C, "_workflows_dir", return_value=pathlib.Path(d)):
            r = json.loads(C.save_workflow.__wrapped__(json.dumps(bad), "a")
                           if hasattr(C.save_workflow, "__wrapped__") else C.save_workflow(json.dumps(bad), "a"))
            self.assertEqual(r["status"], "problems")
            self.assertFalse(list(pathlib.Path(d).iterdir()))
            r = json.loads(C.save_workflow(json.dumps(bad), "a", True))
            self.assertIn("workflow_path", r)
            self.assertTrue(r["problems"])
            with mock.patch.object(C, "_get_object_info", side_effect=RuntimeError("down")):
                r = json.loads(C.save_workflow(json.dumps(bad), "b"))
            self.assertIn("not reachable", r["warning"])


if __name__ == "__main__":
    unittest.main()
