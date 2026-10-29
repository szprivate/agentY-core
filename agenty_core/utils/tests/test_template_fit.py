"""Template matching checks what a template IS (kind, storyboard, model version, ...), not its name.

    python -m unittest agenty_core.utils.tests.test_template_fit
"""
import json
import unittest
from pathlib import Path

from agenty_core.utils import template_fit as T

REQUEST = ("animate each starting frame with Seedance 2.5 first-frame, 480p, no audio, "
           "cut into one film")
OFFICIAL = Path(__file__).resolve().parents[2].parent / "comfyui_workflow_templates_official" / "templates"


def _real(name):
    f = OFFICIAL / f"{name}.json"
    if not f.is_file():
        raise unittest.SkipTest(f"{name} is not in this corpus")
    return T.template_facts(json.loads(f.read_text(encoding="utf-8-sig")), name)


class RequestWords(unittest.TestCase):
    def test_the_regression_request(self):
        w = T.request_wants(REQUEST)
        self.assertEqual(w["kind"], "video")
        self.assertEqual(w["models"], [("seedance", "2.5")])
        self.assertEqual(w["resolutions"], ["480p"])
        self.assertIs(w["audio"], False)
        self.assertTrue(w["first_frame"])
        self.assertFalse(w["storyboard"])

    def test_an_image_of_a_video_game_is_an_image(self):
        self.assertEqual(T.request_wants("an image of a video game hero")["kind"], "image")

    def test_with_audio_and_storyboard(self):
        w = T.request_wants("a storyboard of 8 shots, then a video with audio")
        self.assertTrue(w["storyboard"])
        self.assertIs(w["audio"], True)
        self.assertEqual(w["shots"], 8)

    def test_versions_are_normalised(self):
        self.assertEqual(T.request_wants("wan 2.2 text to video")["models"], [("wan", "2.2")])
        self.assertEqual(T.request_wants("seedance 2")["models"], [("seedance", "2.0")])


class TheRegression(unittest.TestCase):
    """'animate each starting frame with Seedance 2.5 first-frame, 480p, no audio, cut into one
    film' must NOT pick template_seedance2_storyboard_to_video."""

    def test_the_storyboard_template_is_rejected_with_reasons(self):
        facts = _real("template_seedance2_storyboard_to_video")
        self.assertTrue(facts["storyboard"])
        self.assertEqual(facts["panels"], 8)
        v = T.check(REQUEST, facts)
        self.assertEqual(v["verdict"], "reject")
        text = " ".join(v["reasons"])
        self.assertIn("storyboard", text)
        self.assertIn("8 panels", text)
        self.assertIn("2.0", text)

    def test_the_first_frame_template_for_that_version_is_not_rejected(self):
        v = T.check(REQUEST, _real("api_seedance2_5_flf2v"))
        self.assertNotEqual(v["verdict"], "reject", v)
        self.assertTrue(any("480p" in a for a in v["adjust"]))

    def test_other_versions_and_modes_are_rejected(self):
        for name, needle in (("api_seedance2_0_flf2v", "2.0"), ("api_seedance2_5_t2v", "text-to-video"),
                             ("api_seedance2_5_r2v", "reference")):
            v = T.check(REQUEST, _real(name))
            self.assertEqual(v["verdict"], "reject", name)
            self.assertIn(needle, " ".join(v["reasons"]), name)


class Synthetic(unittest.TestCase):
    def api(self, **node_inputs):
        return {"1": {"class_type": "ByteDance2FirstLastFrameNode",
                      "inputs": {"model": "Seedance 2.5", "model.resolution": "720p",
                                 "model.generate_audio": True, **node_inputs}},
                "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}},
                "3": {"class_type": "LoadImage", "inputs": {"image": "a.png"}}}

    def test_kind_mismatch_is_a_reject(self):
        facts = T.template_facts({"1": {"class_type": "SaveImage", "inputs": {}}}, "txt2img")
        v = T.check("make a video of a fox running", facts)
        self.assertEqual(v["verdict"], "reject")
        self.assertIn("video", v["reasons"][0])

    def test_audio_and_resolution_only_demote(self):
        facts = T.template_facts(self.api(), "api_seedance2_5_flf2v")
        self.assertIs(facts["audio"], True)
        v = T.check(REQUEST, facts)
        self.assertEqual(v["verdict"], "demote", v)
        self.assertTrue(any("generate_audio" in a for a in v["adjust"]))
        self.assertTrue(any("480p" in a for a in v["adjust"]))

    def test_a_matching_template_fits(self):
        facts = T.template_facts(self.api(**{"model.resolution": "480p", "model.generate_audio": False}),
                                 "api_seedance2_5_flf2v")
        self.assertEqual(T.check(REQUEST, facts)["verdict"], "fit")

    def test_the_unknown_is_never_held_against_a_template(self):
        facts = T.template_facts({"1": {"class_type": "KSampler", "inputs": {}}}, "mystery")
        self.assertEqual(T.check("anything at all in seedance 2.5 480p", facts)["verdict"], "fit")

    def test_a_storyboard_request_accepts_a_storyboard_template(self):
        facts = {"name": "x", "kinds": ["video", "image"], "primary": "video", "storyboard": True, "panels": 8,
                 "models": [], "resolutions": [], "audio": None, "capability": None, "inputs": {},
                 "multishot": False}
        self.assertEqual(T.check("make an 8 panel storyboard and animate it into a video", facts)["verdict"], "fit")

    def test_text_only_template_for_a_frames_job(self):
        facts = {"name": "t2v", "kinds": ["video"], "primary": "video", "capability": "text_to_video",
                 "storyboard": False, "models": [], "resolutions": [], "audio": None, "inputs": {},
                 "multishot": False}
        self.assertEqual(T.check("animate this image", facts)["verdict"], "reject")


if __name__ == "__main__":
    unittest.main()


class ModelPathsInTheRequest(unittest.TestCase):
    """CX15, night_20261029: the request listed its files as "LTX2/ltx-2.5-....safetensors";
    the folder read as "LTX 2.0" and every LTX template, the 2.5 one too, was rejected."""

    REQ = ("Local LTX-2.5 text-to-video. Files: LTX2/ltx-2.5-22b-distilled.safetensors, "
           "LTX2\gemma4-12b-with-proj-ltx-2.5.safetensors")

    def test_a_folder_is_not_a_version(self):
        self.assertEqual(T.request_wants(self.REQ)["models"], [("ltx", "2.5")])

    def test_the_named_version_fits_and_another_is_rejected(self):
        def facts(ver):
            return {"primary": "video", "kinds": ["video"], "models": [("ltx", ver)]}
        self.assertEqual(T.check(self.REQ, facts("2.5"))["verdict"], "fit")
        v = T.check(self.REQ, facts("2.0"))
        self.assertEqual(v["reasons"], ["it uses Ltx 2.0, the request names Ltx 2.5"])

    def test_two_versions_named_reject_neither(self):
        v = T.check("compare LTX 2.3 against LTX 2.5", {"models": [("ltx", "2.5")]})
        self.assertEqual(v["reasons"], [])


class TheFirstLtxVideoReleases(unittest.TestCase):
    """CX10, night_20261029: ltxv_text_to_video (LTX-Video 0.9, a T5 encoder) passed for
    an "LTX-2" request because its checkpoint name carries the version after the size."""

    def test_the_version_after_the_size_is_read(self):
        got = {v for f, v in T._versions("ltx-video-2b-v0.9.safetensors and ltxv-13b-0.9.7-dev") if v}
        self.assertEqual(got, {"0.9"})

    def test_an_ltx_2_request_rejects_it(self):
        facts = {"models": [("ltx", "0.9")]}
        v = T.check("Build an LTX-2 text-to-video workflow", facts)
        self.assertEqual(v["verdict"], "reject")
