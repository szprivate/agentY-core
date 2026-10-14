"""A model missing under one name is often installed under another: say so, and
download - when a download is the answer - into the extra model path, not
ComfyUI's own folder."""

import json
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as c
from agenty_core.tools import huggingface as h

INDEX = {Path(e).name.lower(): e for e in (
    "SD15\control_sd15_canny.pth", "SD15\control_sd15_depth.pth", "SDXL\controlnet-canny-sdxl.safetensors",
    "FLUX1\flux1-dev-kontext_fp8_scaled.safetensors", "FLUX1\flux1-dev.safetensors",
    "RealESRGAN_x4plus.safetensors")}


class Alternatives(unittest.TestCase):

    def test_the_same_kind_under_another_name(self):
        self.assertEqual(c._installed_alternatives("control_v11p_sd15_canny.pth", INDEX),
                         ["SD15\control_sd15_canny.pth"])

    def test_only_the_closest(self):
        self.assertEqual(c._installed_alternatives("flux1-dev-kontext.safetensors", INDEX),
                         ["FLUX1\flux1-dev-kontext_fp8_scaled.safetensors"])

    def test_a_different_model_is_no_alternative(self):
        self.assertEqual(c._installed_alternatives("RealESRGAN_x2plus.pth", INDEX), [])
        self.assertEqual(c._installed_alternatives("control_v11p_sd15_openpose.pth", INDEX), [])
        self.assertEqual(c._installed_alternatives("model.safetensors", INDEX), [])

    def test_check_model_says_it(self):
        fn = getattr(c.check_model, "func", None) or getattr(c.check_model, "original", None) or c.check_model
        with mock.patch.object(c, "_model_inventory_index", return_value=INDEX), \
                mock.patch.object(c, "node_managed_option", return_value=None):
            got = json.loads(fn(["control_v11p_sd15_canny.pth"]))
        self.assertEqual(got["control_v11p_sd15_canny.pth"], "False")
        self.assertEqual(got["_installed_alternatives"]["control_v11p_sd15_canny.pth"],
                         ["SD15\control_sd15_canny.pth"])
        self.assertIn("instead of downloading", got["_notes"]["control_v11p_sd15_canny.pth"])


class WhereADownloadGoes(unittest.TestCase):
    PATHS = {"controlnet": ["D:\AI\comfyui\models\controlnet", "D:\AI\comfyui\models\t2i_adapter",
                            "L:\software\comfyui\Models\controlnet"]}

    def test_the_extra_model_path_not_comfyuis_own_folder(self):
        with mock.patch.object(h, "_folder_paths", return_value=self.PATHS), \
                mock.patch.object(h, "_configured_models_dir", return_value=None):
            got, source = h._category_dir("controlnet")
        self.assertEqual(Path(got), Path("L:/software/comfyui/Models/controlnet"))

    def test_a_configured_dir_wins(self):
        with mock.patch.object(h, "_folder_paths", return_value=self.PATHS), \
                mock.patch.object(h, "_configured_models_dir", return_value=Path("D:/AI/comfyui/models")):
            got, source = h._category_dir("controlnet")
        self.assertEqual(Path(got), Path("D:/AI/comfyui/models/controlnet"))
        self.assertEqual(source, "configured_models_dir")


if __name__ == "__main__":
    unittest.main()
