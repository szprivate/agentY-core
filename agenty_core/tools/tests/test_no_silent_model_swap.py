"""A missing model is never quietly swapped for another one.

A gated LTX LoRA ("…-layout-to-render") that could not be downloaded became the
installed "…-sdr-to-hdr" one — same words, a different effect — and the graph
validated and ran. A LoRA now only matches its own file (in any folder); other
model and enum substitutions still happen but are reported.

    python -m unittest agenty_core.tools.tests.test_no_silent_model_swap
"""

import unittest

from agenty_core.tools.assembly_deterministic import harden_node_inputs

LORAS = ["LTX2\\ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
         "WAN22\\wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"]
LORA_REQ = {"model": ["MODEL"], "lora_name": [LORAS], "strength_model": ["FLOAT", {"default": 1.0}]}
VAES = ["LTX2\\ltx-2-3-22b-VAE.safetensors", "hunyuan_video_vae_bf16.safetensors"]
VAE_REQ = {"vae_name": [VAES]}
CLIP_REQ = {"clip_name": [["umt5.safetensors"]], "type": [["stable_diffusion", "ltxv", "wan"]]}


def _node(**inputs):
    return {"class_type": "X", "inputs": dict(inputs)}


class NoSilentSwap(unittest.TestCase):

    def test_a_missing_lora_stays_missing(self):
        n = _node(model=["1", 0], lora_name="ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors")
        missing, notes = [], []
        harden_node_inputs(n, LORA_REQ, missing, None, None, notes)
        self.assertEqual(n["inputs"]["lora_name"], "ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors")
        self.assertEqual(missing, ["ltx-2.5-22b-ic-lora-layout-to-render-1.0.safetensors"])
        self.assertEqual(notes, [])

    def test_the_same_lora_in_another_folder_is_found_quietly(self):
        n = _node(model=["1", 0], lora_name="wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors")
        notes = []
        harden_node_inputs(n, LORA_REQ, [], None, None, notes)
        self.assertEqual(n["inputs"]["lora_name"], LORAS[1])
        self.assertEqual(notes, [])

    def test_another_models_stand_in_is_reported(self):
        n = _node(vae_name="ltx-2.5-video-vae-bf16.safetensors")
        notes = []
        harden_node_inputs(n, VAE_REQ, [], None, None, notes)
        self.assertEqual(n["inputs"]["vae_name"], VAES[0])
        self.assertEqual(len(notes), 1)
        self.assertIn("not installed", notes[0])
        self.assertIn("ltx-2-3-22b-VAE", notes[0])

    def test_an_unknown_enum_value_is_reported(self):
        n = _node(clip_name="umt5.safetensors", type="ltxav")
        notes = []
        harden_node_inputs(n, CLIP_REQ, [], None, None, notes)
        self.assertEqual(n["inputs"]["type"], "ltxv")
        self.assertTrue(any("'ltxav'" in x and "unknown value" in x for x in notes))


if __name__ == "__main__":
    unittest.main()
