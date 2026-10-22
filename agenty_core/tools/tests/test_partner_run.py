"""partner_run builds a partner node's graph from its own schema (fixtures from the live ComfyUI).

    python -m unittest agenty_core.tools.tests.test_partner_run
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools import partner_run as P

FIX = Path(__file__).parent / "fixtures"
OI = json.loads((FIX / "partner_nodes_object_info.json").read_text(encoding="utf-8"))
OI.update(json.loads((FIX / "seedance_ref_v2_object_info.json").read_text(encoding="utf-8")))


def build(node, inputs=None, settings=None, save="auto", name="t"):
    with mock.patch.object(P, "_stage", side_effect=lambda p: Path(p).name):
        return P.build_partner_workflow(node, inputs, settings, save, name, object_info=OI)


class Resolving(unittest.TestCase):
    def test_class_display_and_partial_names(self):
        self.assertEqual(P.resolve_node_type("OpenAIGPTImage1", OI)[0], "OpenAIGPTImage1")
        self.assertEqual(P.resolve_node_type("OpenAI GPT Image 2", OI)[0], "OpenAIGPTImage1")
        self.assertEqual(P.resolve_node_type("Seedance 2.5 First-Last-Frame to Video", OI)[0],
                         "ByteDance2FirstLastFrameNode")

    def test_ambiguous_name_lists_candidates(self):
        cls, cands = P.resolve_node_type("Seedance", OI)
        self.assertIsNone(cls)
        self.assertGreaterEqual(len(cands), 2)

    def test_unknown_name_builds_nothing(self):
        out = build("Frobnicator 9000")
        self.assertFalse(out["ok"])


class GPTImage(unittest.TestCase):
    def test_text_to_image_gets_a_save_node(self):
        r = build("OpenAIGPTImage1", settings={"prompt": "a red fox", "quality": "high"})
        self.assertTrue(r["ok"], r)
        wf = r["workflow"]
        node = wf[r["node_id"]]
        self.assertEqual(node["inputs"]["prompt"], "a red fox")
        self.assertEqual(node["inputs"]["quality"], "high")
        saver = wf[r["saver"]]
        self.assertEqual(saver["class_type"], "SaveImage")
        self.assertEqual(saver["inputs"]["images"], [r["node_id"], 0])
        self.assertTrue(saver["inputs"]["filename_prefix"].startswith("agent/images/"))

    def test_reference_image_gets_a_loader_wired_to_image(self):
        r = build("OpenAIGPTImage1", inputs={"image": "D:/x/ref.png"}, settings={"prompt": "edit"})
        self.assertTrue(r["ok"], r)
        lid, _slot = r["workflow"][r["node_id"]]["inputs"]["image"]
        self.assertEqual(r["workflow"][lid]["class_type"], "LoadImage")
        self.assertEqual(r["workflow"][lid]["inputs"]["image"], "ref.png")

    def test_a_bad_option_is_an_error_listing_the_options(self):
        r = build("OpenAIGPTImage1", settings={"prompt": "x", "quality": "ultra"})
        self.assertFalse(r["ok"])
        self.assertIn("low", " ".join(r["errors"]))

    def test_out_of_range_number_is_clamped_with_a_warning(self):
        r = build("OpenAIGPTImage1", settings={"prompt": "x", "n": 99})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["workflow"][r["node_id"]]["inputs"]["n"], 8)
        self.assertTrue(any("'n'" in w for w in r["warnings"]))

    def test_unknown_setting_is_named(self):
        r = build("OpenAIGPTImage1", settings={"prompt": "x", "colour_grade": "warm"})
        self.assertFalse(r["ok"])
        self.assertIn("colour_grade", r["errors"][0])


class SeedanceReference(unittest.TestCase):
    def test_dynamic_combo_and_autogrow_slots_use_dotted_names(self):
        r = build("ByteDance2ReferenceNodeV2",
                  inputs={"model.reference_images": ["a.png", "b.png"], "model.reference_videos": "m.mp4"},
                  settings={"model": "Seedance 2.5", "prompt": "x", "resolution": "480p",
                            "duration": 6, "generate_audio": False})
        self.assertTrue(r["ok"], r)
        ins = r["workflow"][r["node_id"]]["inputs"]
        self.assertEqual(ins["model"], "Seedance 2.5")
        self.assertEqual(ins["model.resolution"], "480p")
        self.assertEqual(ins["model.duration"], 6)
        self.assertIs(ins["model.generate_audio"], False)
        a, b = ins["model.reference_images.image_1"], ins["model.reference_images.image_2"]
        self.assertEqual(r["workflow"][a[0]]["class_type"], "LoadImage")
        self.assertEqual(r["workflow"][b[0]]["inputs"]["image"], "b.png")
        v = ins["model.reference_videos.video_1"]
        self.assertEqual(r["workflow"][v[0]]["class_type"], "LoadVideo")
        self.assertEqual(r["workflow"][r["saver"]]["class_type"], "SaveVideo")
        self.assertIn("seed", ins)

    def test_another_option_changes_which_settings_exist(self):
        r = build("ByteDance2ReferenceNodeV2", settings={"model": "Seedance 2.5 Draft", "prompt": "x"})
        self.assertTrue(r["ok"], r)

    def test_a_setting_is_checked_against_the_option_that_was_chosen(self):
        """4k exists only on some models: it is judged by the model named, not the node's default."""
        opts = {o["key"]: o for o in OI["ByteDance2ReferenceNodeV2"]["input"]["required"]["model"][1]["options"]}
        only_2_0 = [r for r in opts["Seedance 2.0"]["inputs"]["required"]["resolution"][1]["options"]
                    if r not in opts["Seedance 2.5"]["inputs"]["required"]["resolution"][1]["options"]]
        if not only_2_0:
            self.skipTest("both options offer the same resolutions in this schema")
        r = build("ByteDance2ReferenceNodeV2", settings={"resolution": only_2_0[0], "model": "Seedance 2.0",
                                                         "prompt": "x"})
        self.assertTrue(r["ok"], r)

    def test_group_overflow_is_an_error(self):
        r = build("ByteDance2ReferenceNodeV2", inputs={"model.reference_videos": ["v.mp4"] * 11},
                  settings={"prompt": "x"})
        self.assertFalse(r["ok"])

    def test_model_name_is_matched_loosely(self):
        r = build("ByteDance2ReferenceNodeV2", settings={"model": "seedance 2.5", "prompt": "x"})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["workflow"][r["node_id"]]["inputs"]["model"], "Seedance 2.5")


class FirstLastFrame(unittest.TestCase):
    def test_first_frame_only(self):
        r = build("ByteDance2FirstLastFrameNode", inputs={"first_frame": "D:/f/start.png"},
                  settings={"model": "Seedance 2.5", "prompt": "slow push in", "resolution": "480p",
                            "generate_audio": False})
        self.assertTrue(r["ok"], r)
        ins = r["workflow"][r["node_id"]]["inputs"]
        self.assertIn("first_frame", ins)
        self.assertNotIn("last_frame", ins)
        self.assertEqual(ins["model.resolution"], "480p")
        self.assertIs(ins["model.generate_audio"], False)

    def test_required_file_missing_is_named(self):
        r = build("KlingStartEndFrameNode", inputs={"start_frame": "s.png"}, settings={"prompt": "x"})
        self.assertFalse(r["ok"])
        self.assertIn("end_frame", " ".join(r["errors"]))

    def test_kling_start_end_builds_with_both_frames(self):
        r = build("KlingStartEndFrameNode", inputs={"start_frame": "s.png", "end_frame": "e.png"},
                  settings={"prompt": "x", "negative_prompt": ""})
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["workflow"][r["saver"]]["class_type"], "SaveVideo")


class Outputs(unittest.TestCase):
    def test_a_node_that_is_its_own_output_gets_no_saver(self):
        r = build("MeshyImageToModelNode", inputs={"image": "obj.png"}, settings={})
        self.assertTrue(r["ok"], r)
        self.assertIsNone(r.get("saver"))

    def test_save_kind_must_match_the_output(self):
        r = build("OpenAIGPTImage1", settings={"prompt": "x"}, save="video")
        self.assertFalse(r["ok"])

    def test_a_url_is_refused(self):
        r = build("OpenAIGPTImage1", inputs={"image": "https://x/y.png"}, settings={"prompt": "x"})
        self.assertFalse(r["ok"])


class TheTool(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        for target, val in (("_get_object_info", OI), ("_workflows_dir", Path(self.d.name))):
            p = mock.patch.object(C, target, return_value=val)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(P, "_stage", side_effect=lambda x: Path(x).name)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(P.set_run_hook, None)

    def test_run_false_builds_validates_and_writes(self):
        out = json.loads(P.partner_run("OpenAIGPTImage1", settings={"prompt": "a fox"}, run=False))
        self.assertEqual(out["status"], "ready", out)
        self.assertFalse(out["ran"])
        wf = json.loads(Path(out["workflow_path"]).read_text())
        self.assertEqual(wf[out["node_id"]]["class_type"], "OpenAIGPTImage1")

    def test_run_goes_through_the_host_hook(self):
        seen = []
        P.set_run_hook(lambda path: seen.append(path) or json.dumps({"status": "done", "outputs": ["x.png"]}))
        out = json.loads(P.partner_run("OpenAIGPTImage1", settings={"prompt": "a fox"}))
        self.assertEqual(seen, [out["workflow_path"]])
        self.assertEqual(out["run"]["outputs"], ["x.png"])

    def test_an_invalid_build_never_runs(self):
        seen = []
        P.set_run_hook(lambda path: seen.append(path))
        out = json.loads(P.partner_run("OpenAIGPTImage1", settings={"quality": "ultra"}))
        self.assertEqual(out["status"], "error")
        self.assertEqual(seen, [])

    def test_inputs_may_arrive_as_json_text(self):
        known = OI["LoadImage"]["input"]["required"]["image"][0][0]   # what this ComfyUI lists
        out = json.loads(P.partner_run("OpenAIGPTImage1", inputs=json.dumps({"image": known}),
                                       settings='{"prompt": "x"}', run=False))
        self.assertEqual(out["status"], "ready", out)

    def test_seedance_first_frame_job_validates_locally(self):
        known = OI["LoadImage"]["input"]["required"]["image"][0][0]
        out = json.loads(P.partner_run(
            "Seedance 2.5 First-Last-Frame to Video", inputs={"first_frame": known},
            settings={"model": "Seedance 2.5", "prompt": "slow push in", "resolution": "480p",
                      "generate_audio": False}, name="film_shot1", run=False))
        self.assertEqual(out["status"], "ready", out)
        wf = json.loads(Path(out["workflow_path"]).read_text())
        self.assertEqual(wf[out["save_node"]]["class_type"], "SaveVideo")
        self.assertIn("film_shot1", wf[out["save_node"]]["inputs"]["filename_prefix"])


if __name__ == "__main__":
    unittest.main()
