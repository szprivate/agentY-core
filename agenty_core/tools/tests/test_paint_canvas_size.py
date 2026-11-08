"""A generation for a picture in the viewer's Paint tab comes out at the canvas's size.

``assembly_deterministic.set_size_inputs`` sets plain width / height and a size preset with a Custom
option (gpt-image's ``model.size`` + ``model.custom_width`` / ``custom_height``), held to the inputs'
range and step; ``comfyui._paint_canvas_size`` reads the size of a Paint snapshot named in the briefing.
"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import assembly_deterministic as A
from agenty_core.tools import comfyui as C

FIX = Path(__file__).parent / "fixtures" / "partner_nodes_object_info.json"
TEMPLATE = Path(__file__).resolve().parents[3] / "comfyui_workflow_templates_custom" / "templates" / "api_i2i_imageEdit_OpenAi_GPT2.json"


class SetSize(unittest.TestCase):
    def test_a_preset_with_custom_is_set_to_custom_at_the_size_the_inputs_allow(self):
        spec = json.loads(FIX.read_text())["OpenAIGPTImageNodeV2"]["input"]
        node = {"class_type": "OpenAIGPTImageNodeV2", "inputs": {
            "prompt": "x", "model": "gpt-image-2.5-flare", "model.size": "2160x3840",
            "model.custom_width": 1024, "model.custom_height": 1024}}
        done = A.set_size_inputs(node, 1920, 1080, spec["required"], spec.get("optional"))
        self.assertTrue(done)
        ins = node["inputs"]
        self.assertEqual(ins["model.size"], "Custom")
        self.assertEqual((ins["model.custom_width"], ins["model.custom_height"]), (1920, 1088))   # multiples of 16
        # Too big for the inputs: scaled down with its shape, inside 480..3840.
        A.set_size_inputs(node, 8000, 4000, spec["required"], spec.get("optional"))
        self.assertEqual((ins["model.custom_width"], ins["model.custom_height"]), (3840, 1920))

    def test_plain_width_and_height_and_wires_left_alone(self):
        sp = ["INT", {"min": 64, "max": 4096, "step": 8}]
        node = {"inputs": {"width": 512, "height": ["7", 0]}}
        A.set_size_inputs(node, 1000, 700, {"width": sp, "height": sp})
        self.assertEqual(node["inputs"], {"width": 1000, "height": ["7", 0]})

    def test_a_preset_without_custom_is_not_touched(self):
        node = {"inputs": {"size": "1024x1024"}}
        self.assertEqual(A.set_size_inputs(node, 1920, 1080, {"size": ["COMBO", {"options": ["1024x1024", "1536x1024"]}]}), [])
        self.assertEqual(node["inputs"], {"size": "1024x1024"})

    @unittest.skipUnless(TEMPLATE.is_file(), "needs the custom templates")
    def test_the_gpt_image_edit_template(self):
        spec = json.loads(FIX.read_text())["OpenAIGPTImageNodeV2"]["input"]
        wf = json.loads(TEMPLATE.read_text())
        node = next(n for n in wf.values() if n.get("class_type") == "OpenAIGPTImageNodeV2")
        node["inputs"]["model"] = "gpt-image-2.5-flare"     # the fixture's option names
        A.set_size_inputs(node, 1600, 1200, spec["required"], spec.get("optional"))
        self.assertEqual((node["inputs"]["model.size"], node["inputs"]["model.custom_width"], node["inputs"]["model.custom_height"]),
                         ("Custom", 1600, 1200))


class PaintSnapshot(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("needs Pillow")
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        os.makedirs(os.path.join(self.tmp, "paint"))
        self.name = "board_20261108_151500_picture.png"
        Image.new("RGB", (1920, 1080)).save(os.path.join(self.tmp, "paint", self.name))
        Image.new("RGB", (640, 480)).save(os.path.join(self.tmp, "other.png"))

    def test_the_size_of_a_paint_snapshot_in_the_briefing(self):
        with mock.patch("agenty_core.tools.image_io.comfy_input_dir", return_value=self.tmp):
            self.assertEqual(C._paint_canvas_size([{"node_id": "5", "filename": self.name}]), (1920, 1080))
            self.assertEqual(C._paint_canvas_size([{"node_id": "5", "filename": "paint/" + self.name}]), (1920, 1080))
            self.assertIsNone(C._paint_canvas_size([{"node_id": "5", "filename": "other.png"}]))
            self.assertIsNone(C._paint_canvas_size([]))


if __name__ == "__main__":
    unittest.main()
