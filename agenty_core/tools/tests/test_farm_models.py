"""Where a farm machine downloads a model from (agenty_core.utils.farm_models)."""
import json
import tempfile
import unittest
from pathlib import Path

from agenty_core.utils import farm_models as fm

LISTING = [
    {"id": "Comfy-Org/MOSS-Image", "siblings": [
        {"rfilename": "README.md"},
        {"rfilename": "diffusion_models/moss_image_sft_bf16.safetensors"},
        {"rfilename": "text_encoders/qwen3.5_9b_int8_convrot.safetensors"},
        {"rfilename": "vae/flux2-vae.safetensors"}]},
    {"id": "Comfy-Org/Flux-2-Dev", "siblings": [
        {"rfilename": "split_files/vae/flux2-vae.safetensors"}]},
    {"id": "Comfy-Org/Odd", "siblings": None},
]


class Resolve(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = []

        def http(url, params):
            self.calls.append(url)
            if url.endswith("/api/models"):
                return LISTING
            if url.endswith("/api/models/Comfy-Org/Odd"):
                return {"siblings": [{"rfilename": "loras/odd.safetensors"}]}
            if url.endswith("/api/models/me/private"):
                return {"siblings": [{"rfilename": "loras/mine.safetensors"}]}
            raise RuntimeError(url)
        fm.set_root(self.root)
        fm.set_http(http)

    def tearDown(self):
        fm.set_http(None)
        fm.set_root(None)
        self.tmp.cleanup()

    def test_comfy_org_repos_are_searched_by_folder_and_name(self):
        self.assertEqual(fm.resolve("diffusion_models", "moss_image_sft_bf16.safetensors"),
                         "https://huggingface.co/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/"
                         "moss_image_sft_bf16.safetensors")
        self.assertIn("/vae/flux2-vae.safetensors", fm.resolve("vae", "flux2-vae.safetensors"))
        self.assertEqual(fm.resolve("loras", "odd.safetensors"),
                         "https://huggingface.co/Comfy-Org/Odd/resolve/main/loras/odd.safetensors")
        self.assertEqual(fm.resolve("loras", "nowhere.safetensors"), "")

    def test_a_name_with_a_subfolder_is_found_by_its_file_name(self):
        self.assertTrue(fm.resolve("diffusion_models", "moss/moss_image_sft_bf16.safetensors").endswith(
            "diffusion_models/moss_image_sft_bf16.safetensors"))

    def test_the_index_is_read_once_and_kept(self):
        fm.resolve("vae", "flux2-vae.safetensors")
        fm.resolve("vae", "flux2-vae.safetensors")
        self.assertEqual(sum(1 for u in self.calls if u.endswith("/api/models")), 1)
        self.assertTrue((self.root / ".cache" / "farm_model_index.json").is_file())

    def test_the_users_list_and_the_workflows_urls_come_first(self):
        (self.root / "config").mkdir()
        (self.root / "config" / "farm_models.json").write_text(json.dumps({
            "models": {"vae/flux2-vae.safetensors": "https://example.org/vae.safetensors"},
            "repos": ["me/private"]}))
        self.assertEqual(fm.resolve("vae", "flux2-vae.safetensors"), "https://example.org/vae.safetensors")
        self.assertEqual(fm.resolve("loras", "mine.safetensors"),
                         "https://huggingface.co/me/private/resolve/main/loras/mine.safetensors")
        self.assertEqual(fm.resolve("loras", "x.safetensors", {"loras/x.safetensors": "https://example.org/x"}),
                         "https://example.org/x")
        self.assertEqual(fm.resolve("loras", "x.safetensors", {"loras/x.safetensors": "http://insecure/x"}), "")


if __name__ == "__main__":
    unittest.main()
