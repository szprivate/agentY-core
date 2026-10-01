"""A missing model file is swapped for one of its own family, not the nearest words.

An LTX-2.5 template on a machine with only LTX-2.3: the video VAE
"ltx-2.5-video-vae-bf16" tied on word overlap between "LTX2/ltx-2-3-22b-VAE"
and "hunyuan_video_vae_bf16" (video, vae, bf16), the Hunyuan one came first,
and the graph crashed when it ran.

    python -m unittest agenty_core.tools.tests.test_snap_family
"""

import unittest

from agenty_core.tools.assembly_deterministic import _model_brand, snap_combo

VAES = ["hunyuan_video_vae_bf16.safetensors", "LTX2\\ltx-2-3-22b-VAE.safetensors",
        "LTX2\\ltx-2-3-22b-audio_vae.safetensors", "ae.safetensors", "taeltx_2.safetensors",
        "wan2.2_vae.safetensors"]


class SnapFamilyTest(unittest.TestCase):

    def test_a_model_stays_in_its_family(self):
        self.assertEqual(snap_combo("ltx-2.5-video-vae-bf16.safetensors", VAES, fallback_first=False),
                         "LTX2\\ltx-2-3-22b-VAE.safetensors")
        self.assertEqual(snap_combo("ltx-2.5-audio-vae-bf16.safetensors", VAES, fallback_first=False),
                         "LTX2\\ltx-2-3-22b-audio_vae.safetensors")

    def test_no_file_of_the_family_keeps_the_old_choice(self):
        """Not a missing-model verdict: that path can download, so it is not
        widened here."""
        self.assertEqual(snap_combo("hidream_vae_video.safetensors", VAES, fallback_first=False),
                         "hunyuan_video_vae_bf16.safetensors")

    def test_enums_are_untouched(self):
        self.assertEqual(snap_combo("ltx nearest", ["bilinear", "nearest-exact"]), "nearest-exact")

    def test_the_brand(self):
        self.assertEqual(_model_brand("ltx-2.5-video-vae-bf16"), "ltx")
        self.assertEqual(_model_brand("flux1-dev-fp8"), "flux1")
        self.assertEqual(_model_brand("wan2.2_vae"), "wan2")
        self.assertIsNone(_model_brand("ae"))
        self.assertIsNone(_model_brand("model"))
        self.assertIsNone(_model_brand("sd_xl_base_1.0"))


if __name__ == "__main__":
    unittest.main()
