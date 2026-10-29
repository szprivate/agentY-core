"""Which models a request names, matched against the corpus's own model names.

The regressions of night_20261029: EX04 ("Qwen Image Edit 2511 ... 8 steps" pinned a
Qwen-Image 2512 graph), CX10 (an LTX-Video 0.9 template for "LTX-2"), CX15 (a folder
"LTX2/" read as LTX 2.0, so the LTX 2.5 template was rejected).

    python -m unittest agenty_core.utils.tests.test_model_names
"""
import re
import unittest

from agenty_core.utils import model_names as M
from agenty_core.utils import template_fit as T

EX04 = ("Build a Qwen Image Edit 2511 workflow that uses the Lightning 8-step LoRA: 8 steps, CFG 1. "
        "Load a photo and apply 'replace the sky with a dramatic orange sunset, keep the buildings'. "
        "Save the edited image.")
CX10 = ("Build an LTX-2 text-to-video workflow for 'a hummingbird hovering at a red flower, slow motion' "
        "and save the video.")
CX15 = ("Build a local LTX 2.5 text-to-video workflow (the distilled LTX 2.5 model, not the LTX API nodes) "
        "for 'a red kite flying over windswept sand dunes'. Render it in two stages: sample at 512x288, "
        "upscale the latent 2x with the LTX 2.5 latent spatial upscaler and refine it with a short second "
        "pass, for a 1024x576 result. 97 frames at 24 fps. Save the video. "
        "Files: LTX2/ltx-2.5-22b-distilled.safetensors")

# A corpus in miniature: catalog model names, main model files, recipe task/family.
ENTRIES = {
    "image_qwen_image_edit_2511": {"models": ["Qwen-Image-Edit"], "task": "image_edit", "family": "Qwen Image",
                                   "files": ["qwen_image_edit_2511_bf16.safetensors"]},
    "image_qwen_image_edit_2509": {"models": ["Qwen-Image"], "task": "image_edit", "family": "Qwen Image",
                                   "files": ["qwen_image_edit_2509_fp8_e4m3fn.safetensors"]},
    "image_qwen_image_2512_with_2steps_lora": {"models": ["Qwen-Image 2512"], "task": "text_to_image",
                                               "family": "Qwen Image",
                                               "files": ["qwen_image_2512_fp8_e4m3fn.safetensors"]},
    "image_qwen_Image_2512": {"models": ["Qwen", "Qwen-Image"], "task": "text_to_image", "family": "Qwen Image",
                              "files": ["qwen_image_2512_fp8_e4m3fn.safetensors"]},
    "video_ltx2_t2v": {"models": ["LTX-2", "Lightricks"], "task": "text_to_video", "family": "LTX-2",
                       "files": ["ltx-2-19b-dev-fp8.safetensors"]},
    "video_ltx2_3_t2v": {"models": ["LTX-2.3", "Lightricks"], "task": "text_to_video", "family": "LTX-2",
                         "files": ["ltx-2.3-22b-dev-fp8.safetensors"]},
    "video_ltx2_5_t2v": {"models": ["LTX-2.5"], "task": "text_to_video", "family": "LTX-2",
                         "files": ["ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"]},
    "video_ltx2_5_i2v": {"models": ["LTX-2.5"], "task": "image_to_video", "family": "LTX-2", "files": []},
    "ltxv_text_to_video": {"models": ["LTX-0.9.5"], "task": "text_to_video", "family": "LTX-2",
                           "files": ["ltx-video-2b-v0.9.safetensors"]},
    "video_wan2_2_14B_t2v": {"models": ["Wan2.2", "Wan", "wan_2.1_vae"], "task": "text_to_video",
                             "family": "WAN 2.2",
                             "files": ["wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"],
                             "side": {"wan 2.1 vae"}},
    "video_wan2_1_t2v": {"models": ["Wan2.1", "Wan"], "task": "text_to_video", "family": "WAN",
                         "files": ["wan2.1_t2v_14B_bf16.safetensors"]},
    "film_grain": {"models": [], "task": "image_tools", "family": "Generic"},
    "frame_interpolation": {"models": ["FILM"], "task": "video_tools", "family": "Generic"},
    "upscale_using_model": {"models": [], "task": "upscale", "family": "Generic"},
}


def vocab():
    return M.Vocabulary(ENTRIES)


class Spelling(unittest.TestCase):
    def test_one_spelling_for_names_and_requests(self):
        self.assertEqual(M.tokens("Qwen-Image-Edit 2511"), ["qwen", "image", "edit", "2511"])
        for s in ("ltx2.5", "LTX 2.5", "LTX-2.5", "ltx_2_5", "LTX-v2.5"):
            self.assertEqual(M.tokens(s), ["ltx", "2.5"], s)
        for s in ("wan 2.2", "Wan2.2", "wan2_2", "wan22", "WAN-2.2"):
            self.assertEqual(M.tokens(s), ["wan", "2.2"], s)
        self.assertEqual(M.tokens("LTX-2"), ["ltx", "2"])
        self.assertEqual(M.tokens("Seedance 2.0"), ["seedance", "2"])
        self.assertEqual(M.tokens("flux dev"), ["flux", "dev"])
        self.assertEqual(M.tokens("Stable Diffusion XL"), ["sdxl"])
        self.assertEqual(M.tokens("sd_xl_base_1.0.safetensors"), ["sdxl", "base", "1"])
        self.assertEqual(M.tokens("sd-v1-5-inpainting"), ["sd", "1.5", "inpainting"])

    def test_a_folder_is_never_a_version(self):
        self.assertEqual(M.tokens("LTX2/ltx-2.5-22b.safetensors"), ["ltx", "2.5", "22b"])
        self.assertEqual(M.tokens("LTX2\\ltx-2.5-22b.safetensors"), ["ltx", "2.5", "22b"])

    def test_precision_and_task_words_stay_whole(self):
        self.assertEqual(M.tokens("fp8 bf16 int8 t2v hunyuan3d"), ["fp8", "bf16", "int8", "t2v", "hunyuan3d"])


class WhatARequestNames(unittest.TestCase):
    def named(self, text):
        return [(m["name"], m["version"]) for m in M.named_models(text, vocab())]

    def test_the_longest_name_wins(self):
        self.assertEqual(self.named(EX04), [("qwen image edit 2511", "2511")])  # from the model file
        self.assertEqual(self.named("a Qwen Image 2512 text-to-image"), [("Qwen-Image 2512", "2512")])

    def test_versions_are_exact(self):
        self.assertEqual(self.named(CX10), [("LTX-2", "2")])
        self.assertIn(("LTX-2.5", "2.5"), self.named(CX15))
        self.assertNotIn(("LTX-2", "2"), self.named(CX15))
        self.assertEqual(self.named("wan 2.2 text to video"), [("Wan2.2", "2.2")])
        self.assertEqual(self.named("wan2.1 text to video"), [("Wan2.1", "2.1")])

    def test_a_version_no_template_has_is_still_read(self):
        self.assertEqual(self.named("wan 2.4 text to video"), [("Wan", "2.4")])

    def test_settings_and_loose_numbers_are_not_models(self):
        self.assertEqual(self.named("8 steps, CFG 1, seed 2, 1024x1024, 97 frames at 24 fps"), [])
        self.assertEqual(self.named("a red fox in the snow, 35mm film grain"), [])


class Pinning(unittest.TestCase):
    def pin(self, text):
        out = M.pin_templates(text, vocab=vocab())
        return out["templates"] if out else None

    def test_ex04_pins_the_edit_2511_graph_and_never_2512(self):
        self.assertEqual(self.pin(EX04), ["image_qwen_image_edit_2511"])

    def test_qwen_image_2512_is_not_edit_2511(self):
        self.assertEqual(self.pin("a Qwen Image 2512 text-to-image of a cat"),
                         ["image_qwen_Image_2512", "image_qwen_image_2512_with_2steps_lora"])

    def test_cx10_is_ltx_2_never_ltxv_0_9(self):
        self.assertEqual(self.pin(CX10), ["video_ltx2_t2v"])

    def test_cx15_is_the_ltx_2_5_text_to_video(self):
        self.assertEqual(self.pin(CX15), ["video_ltx2_5_t2v"])

    def test_wan_2_2_is_not_wan_2_1(self):
        self.assertEqual(self.pin("a Wan 2.2 text-to-video using both the high and low models"),
                         ["video_wan2_2_14B_t2v"])
        self.assertEqual(self.pin("wan 2.1 text to video"), ["video_wan2_1_t2v"])

    def test_the_task_must_fit(self):
        self.assertEqual(self.pin("LTX 2.5 image-to-video of this photo"), ["video_ltx2_5_i2v"])
        self.assertIsNone(self.pin("Qwen Image 2512, edit this photo"))

    def test_no_model_named_no_pin(self):
        for text in ("a red fox in the snow", "a cinematic photo, 35mm film grain",
                     "upscale it using the model", "8 steps, cfg 1"):
            self.assertIsNone(self.pin(text), text)

    def test_a_family_named_loosely_pins_only_when_it_is_a_choice(self):
        fams = ["WAN", "WAN VACE", "WAN 2.2"]
        many = {f"t{i}": {"models": ["Wan"], "task": "text_to_video", "family": fams[i % 3]} for i in range(12)}
        out = M.pin_templates("a wan text to video", vocab=M.Vocabulary(many))
        self.assertIsNone(out)


class TheVersionCheck(unittest.TestCase):
    """template_fit reads versions from the model names, not from folders or file names."""

    def facts(self, name):
        names = vocab().names_of(name)
        return {"models": [(f, T._fit_version(v)) for f, v in M.model_versions(names)]}

    def test_the_vae_of_a_wan_2_2_graph_is_not_its_version(self):
        self.assertEqual(self.facts("video_wan2_2_14B_t2v")["models"], [("wan", "2.2")])

    def test_ltxv_0_9_is_rejected_for_ltx_2(self):
        with unittest.mock.patch.object(M, "vocabulary", vocab):
            self.assertEqual(T.check(CX10, self.facts("ltxv_text_to_video"))["verdict"], "reject")
            self.assertEqual(T.check(CX10, self.facts("video_ltx2_t2v"))["verdict"], "fit")

    def test_ltx_2_5_fits_cx15(self):
        with unittest.mock.patch.object(M, "vocabulary", vocab):
            self.assertEqual(T.check(CX15, self.facts("video_ltx2_5_t2v"))["reasons"], [])
            self.assertEqual(T.check(CX15, self.facts("video_ltx2_3_t2v"))["verdict"], "reject")

    def test_edit_2511_rejects_2512(self):
        with unittest.mock.patch.object(M, "vocabulary", vocab):
            v = T.check(EX04, self.facts("image_qwen_image_2512_with_2steps_lora"))
            self.assertEqual(v["reasons"], ["it uses Qwen 2512, the request names Qwen 2511"])
            self.assertEqual(T.check(EX04, self.facts("image_qwen_image_edit_2511"))["reasons"], [])


class TheRealCorpus(unittest.TestCase):
    """The same cases against the recipe DB and catalog as they are."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.v = M.vocabulary()
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(f"no corpus: {exc}")
        if not cls.v.templates or "video_ltx2_5_t2v" not in cls.v.templates:
            raise unittest.SkipTest("the corpus is not here")

    def test_the_recipe_db_gives_task_and_family(self):
        self.assertEqual(self.v.task_of("video_ltx2_5_t2v"), "text_to_video")
        self.assertEqual(self.v.templates["video_ltx2_5_t2v"]["family"], "LTX-2")

    def test_ex04(self):
        out = M.pin_templates(EX04, vocab=self.v)
        self.assertIsNotNone(out)
        # Every pin is a Qwen Image Edit 2511 edit graph (the ones whose model names
        # also say "Lightning 8 steps" first); none is a 2512 graph.
        for t in out["templates"]:
            self.assertTrue(M.carries(self.v.names_of(t), ("qwen", "image", "edit", "2511")), t)
            self.assertEqual(self.v.task_of(t), "image_edit", t)
        self.assertTrue(all("2512" not in t for t in out["templates"]), out)
        out = M.pin_templates("Qwen Image Edit 2511: replace the sky", vocab=self.v)
        self.assertIn("image_qwen_image_edit_2511", out["templates"])

    def test_cx10(self):
        out = M.pin_templates(CX10, vocab=self.v)
        self.assertIn("video_ltx2_t2v", out["templates"])
        self.assertNotIn("ltxv_text_to_video", out["templates"])
        v = T.check_template(CX10, "ltxv_text_to_video")
        self.assertEqual(v["verdict"], "reject", v)

    def test_cx15(self):
        out = M.pin_templates(CX15, vocab=self.v)
        self.assertIn("video_ltx2_5_t2v", out["templates"])
        v = T.check_template(CX15, "video_ltx2_5_t2v")
        self.assertNotEqual(v["verdict"], "reject", v)
        self.assertEqual(T.check_template(CX15, "video_ltx2_3_t2v")["verdict"], "reject")

    def test_qwen_versions(self):
        out = M.pin_templates("a Qwen Image 2512 text-to-image of a cat", vocab=self.v)
        self.assertIn("image_qwen_image_2512_with_2steps_lora", out["templates"])
        self.assertTrue(all("2511" not in t for t in out["templates"]), out)

    def test_wan_versions(self):
        out = M.pin_templates("a Wan 2.2 text-to-video", vocab=self.v)
        self.assertIn("video_wan2_2_14B_t2v", out["templates"])
        self.assertFalse([t for t in out["templates"] if re.search(r"wan_?2[._]1", t)], out)
        self.assertEqual(T.check_template("a Wan 2.2 text-to-video", "video_wan2_2_14B_t2v")["verdict"], "fit")
        self.assertEqual(T.check_template("a Wan 2.1 text-to-video", "video_wan2_2_14B_t2v")["verdict"], "reject")

    def test_nothing_named(self):
        self.assertIsNone(M.pin_templates("a red fox in the snow, 35mm film grain, 8 steps", vocab=self.v))


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
