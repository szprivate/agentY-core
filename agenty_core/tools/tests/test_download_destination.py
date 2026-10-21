"""A download goes where ComfyUI itself saves: the is_default extra_model_paths
section's folder (listed first by /internal/folder_paths), else ComfyUI's own."""
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import huggingface as h

DEFAULT_FIRST = {  # sample yaml: comfyui section with base_path L:/..., is_default: true
    "vae": ["L:/software/comfyui/Models/vae", "D:/AI/comfyui/models/vae"],
    "geometry_estimation": ["L:/software/comfyui/Models/geometry_estimation",
                            "D:/AI/comfyui/models/geometry_estimation"],
    "text_encoders": ["L:/software/comfyui/Models/text_encoders", "L:/software/comfyui/Models/clip",
                      "D:/AI/comfyui/models/text_encoders", "D:/AI/comfyui/models/clip"],
}
NO_DEFAULT = {"vae": ["D:/AI/comfyui/models/vae", "L:/software/comfyui/Models/vae"]}


def cat(paths, category):
    with mock.patch.object(h, "_folder_paths", return_value=paths), \
            mock.patch.object(h, "_configured_models_dir", return_value=None):
        return h._category_dir(category)


class Destination(unittest.TestCase):
    def test_default_section_wins(self):
        self.assertEqual(Path(cat(DEFAULT_FIRST, "vae")[0]), Path("L:/software/comfyui/Models/vae"))

    def test_custom_type(self):
        self.assertEqual(Path(cat(DEFAULT_FIRST, "geometry_estimation")[0]),
                         Path("L:/software/comfyui/Models/geometry_estimation"))

    def test_multi_folder_type_uses_first(self):
        self.assertEqual(Path(cat(DEFAULT_FIRST, "text_encoders")[0]),
                         Path("L:/software/comfyui/Models/text_encoders"))

    def test_no_default_means_comfyuis_own(self):
        self.assertEqual(Path(cat(NO_DEFAULT, "vae")[0]), Path("D:/AI/comfyui/models/vae"))

    def test_unknown_type_falls_back_to_base_dir(self):
        with mock.patch.object(h, "_models_base_dir", return_value=Path("D:/m")):
            got, src = cat({}, "nothing")
        self.assertEqual((Path(got), src), (Path("D:/m/nothing"), "models_base_dir"))

    def test_c_drive_still_refused(self):
        with mock.patch.object(h, "_folder_paths", return_value={}), \
                mock.patch.object(h, "_models_base_dir", return_value=Path("C:/m")):
            with self.assertRaises(RuntimeError):
                h._ensure_not_c_drive(Path("C:/m/vae"), "x")

    def test_failed_lookup_not_cached(self):
        h._FOLDER_PATHS_CACHE = None
        with mock.patch("agenty_core.utils.comfyui_client.get_client", side_effect=RuntimeError):
            h._folder_paths()
        self.assertIsNone(h._FOLDER_PATHS_CACHE)



class AliasFolder(unittest.TestCase):
    def test_the_folder_named_after_the_type_wins_on_the_default_drive(self):
        fp = {"diffusion_models": ["L:/m/unet", "L:/m/diffusion_models", "D:/c/models/unet"]}
        with mock.patch.object(h, "_folder_paths", return_value=fp),                 mock.patch.object(h, "_configured_models_dir", return_value=None):
            self.assertEqual(h._category_dir("diffusion_models")[0], Path("L:/m/diffusion_models"))

if __name__ == "__main__":
    unittest.main()
