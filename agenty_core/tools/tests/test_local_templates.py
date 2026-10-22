"""Solved jobs become machine-local templates: stored apart, read like any template, matched conservatively.

    python -m unittest agenty_core.tools.tests.test_local_templates
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools import local_templates as L
from agenty_core.tools import workflow_registry as R
from agenty_core.utils import template_fit

SEEDANCE_FF = {
    "1": {"class_type": "ByteDance2FirstLastFrameNode",
          "inputs": {"model": "Seedance 2.5", "model.prompt": "slow push in on the hero",
                     "model.resolution": "480p", "model.generate_audio": False, "model.duration": 5,
                     "first_frame": ["2", 0], "seed": 5, "watermark": False},
          "_meta": {"title": "Seedance first frame"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "hero.png"}, "_meta": {"title": "Start frame"}},
    "3": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0], "filename_prefix": "agent/videos/x"}},
}
GPT_IMAGE = {
    "1": {"class_type": "OpenAIGPTImage1", "inputs": {"prompt": "a red fox in the snow at dawn", "quality": "high"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "agent/images/fox"}},
}


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = Path(self.d.name)
        (self.root / "official").mkdir()
        for target, val in (("_custom_templates_dir", self.root), ("_official_templates_dir", self.root / "official")):
            p = mock.patch.object(C, target, return_value=val)
            p.start()
            self.addCleanup(p.stop)
        C._index_cache = None
        C.clear_tool_caches()
        template_fit._cache.clear()
        self.addCleanup(setattr, C, "_index_cache", None)


class Store(Base):
    def test_saved_in_a_local_folder_with_its_own_index(self):
        got = L.save_local(SEEDANCE_FF, request="animate each starting frame with Seedance 2.5, 480p, no audio")
        self.assertTrue(got["created"])
        self.assertEqual(Path(got["path"]).parent, self.root / "local")
        index = json.loads((self.root / "local" / "index.json").read_text())
        entry = index[0]["templates"][0]
        self.assertEqual(entry["name"], got["name"])
        self.assertEqual(entry["output_kind"], "video")
        kinds = {(p["kind"], p["input"]) for p in entry["parameters"]}
        self.assertIn(("image", "image"), kinds)
        self.assertIn(("text", "model.prompt"), kinds)
        self.assertIn(("setting", "model.resolution"), kinds)
        self.assertTrue(entry["local"])
        # nothing shared was written
        self.assertFalse((self.root / "index.json").exists())

    def test_the_same_structure_again_updates_instead_of_duplicating(self):
        a = L.save_local(SEEDANCE_FF, request="animate frames with seedance 2.5")
        again = json.loads(json.dumps(SEEDANCE_FF))
        again["2"]["inputs"]["image"] = "other.png"
        b = L.save_local(again, request="another one")
        self.assertEqual(a["name"], b["name"])
        self.assertFalse(b["created"])
        self.assertEqual(len(L._read_entries()), 1)
        self.assertEqual(L._read_entries()[0]["uses"], 2)

    def test_read_side_finds_it_without_new_code(self):
        got = L.save_local(GPT_IMAGE, name="fox_poster", description="[API] Fox image via GPT Image. prompt -> image.")
        catalog = json.loads(C.get_workflow_catalog())
        self.assertIn(got["name"], catalog)
        self.assertIn("Fox image", catalog[got["name"]])
        self.assertIsNotNone(C._fetch_template(got["name"]))
        self.assertIn(got["name"], [t["name"] for t in C._load_index()])

    def test_register_workflow_template_local_skips_the_shared_index(self):
        f = self.root / "my_flow.json"
        f.write_text(json.dumps(GPT_IMAGE))
        out = json.loads(R.register_workflow_template(str(f), description="Fox poster via GPT Image.", local=True))
        self.assertTrue(out["local"], out)
        self.assertFalse((self.root / "index.json").exists())
        self.assertTrue((self.root / "local" / f"{out['registered']}.json").is_file())
        self.assertIn("Fox poster", L._read_entries()[0]["description"])

    def test_local_names_lists_the_saved_jobs(self):
        got = L.save_local(GPT_IMAGE, name="fox")
        self.assertEqual(L.local_names(), {got["name"]})

    def test_unregister_removes_a_local_template(self):
        got = L.save_local(GPT_IMAGE, name="fox")
        out = json.loads(R.unregister_workflow_template(got["name"]))
        self.assertTrue(out.get("local"))
        self.assertEqual(L._read_entries(), [])
        self.assertFalse(Path(got["path"]).exists())

    def test_the_recipe_generator_never_sees_the_local_folder(self):
        import agenty_core.workflow_recipes.parser as P
        L.save_local(GPT_IMAGE, name="fox")
        (self.root / "shared.json").write_text(json.dumps(GPT_IMAGE))
        graphs = P.Corpus._parse_folders({"custom": str(self.root)}, None, {}, log=lambda *a, **k: None)
        names = {g.name for g in graphs}
        self.assertIn("shared", names)
        self.assertFalse(any(n.startswith("local_") for n in names))

    def test_the_folder_is_ignored_by_git(self):
        gi = (Path(__file__).resolve().parents[3] / ".gitignore").read_text()
        self.assertIn("comfyui_workflow_templates_custom/templates/local/", gi)


class Matching(Base):
    def setUp(self):
        super().setUp()
        L.save_local(SEEDANCE_FF, request="animate each starting frame with Seedance 2.5 first frame, 480p, no audio")
        L.save_local(GPT_IMAGE, request="make a poster image of a red fox with GPT Image")

    def names(self, request):
        return [m["name"] for m in L.match_local(request)]

    def test_the_same_job_with_new_inputs_matches(self):
        got = L.match_local("animate these start frames with Seedance 2.5 first-frame at 480p, no audio")
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["output_kind"], "video")
        self.assertEqual(got[0]["verdict"], "fit")

    def test_another_kind_of_output_never_matches(self):
        self.assertEqual(self.names("make a video of a red fox with GPT Image"), [])
        got = self.names("a poster image of a red fox with GPT Image")
        self.assertEqual(len(got), 1)
        self.assertIn("fox", got[0])

    def test_a_named_model_the_job_does_not_run_never_matches(self):
        self.assertEqual(self.names("animate these start frames with Kling 3.0 first frame"), [])

    def test_another_model_version_is_rejected_by_the_fit_check(self):
        self.assertEqual(self.names("animate these start frames with Seedance 2.0 first frame"), [])

    def test_unrelated_wording_does_not_match(self):
        self.assertEqual(self.names("upscale this video to 4k"), [])

    def test_catalog_block_names_the_new_inputs(self):
        block = L.catalog_block("animate these start frames with Seedance 2.5 first-frame at 480p, no audio")
        self.assertIn("YOUR SAVED JOBS", block)
        self.assertIn("image 'image'", block)
        self.assertEqual(L.catalog_block("upscale this video to 4k"), "")


class Limits(Base):
    def test_the_oldest_job_goes_when_the_folder_is_full(self):
        with mock.patch.object(L, "MAX_LOCAL", 2):
            for i in range(3):
                wf = json.loads(json.dumps(GPT_IMAGE))
                wf["1"]["class_type"] = f"OpenAIGPTImage{i}"
                L.save_local(wf, name=f"job{i}")
        names = [e["name"] for e in L._read_entries()]
        self.assertEqual(len(names), 2)
        self.assertNotIn("local_job0", names)


if __name__ == "__main__":
    unittest.main()


class LocalOnly(Base):
    def test_a_partner_job_is_never_offered_to_a_local_request(self):
        L.save_local(GPT_IMAGE, request="make a poster image of a red fox with GPT Image")
        self.assertTrue(L._read_entries()[0]["uses_api"])
        self.assertEqual(len(L.match_local("make a poster image of a red fox with GPT Image")), 1)
        self.assertEqual(L.match_local("make a poster image of a red fox with GPT Image", local_only=True), [])
