"""A picture, film or sound file is never looked for or fetched on Hugging Face (promo run 2026-11-15)."""
import json
import os
import tempfile
import unittest
from unittest import mock

from agenty_core.tools import huggingface as H
from agenty_core.tools import model_research as R


def _call(t, *a, **k):
    fn = getattr(t, "__wrapped__", None) or getattr(t, "_tool_func", None) or t
    return fn(*a, **k)


class LocalInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inp = os.path.join(self.tmp.name, "input")
        self.out = os.path.join(self.tmp.name, "output", "imageviewer", "videos")
        os.makedirs(self.inp)
        os.makedirs(self.out)
        open(os.path.join(self.out, "hills_previz_v005.mp4"), "wb").close()
        dirs = json.dumps({"input_dir": self.inp, "output_dir": os.path.join(self.tmp.name, "output")})
        self.p = mock.patch("agenty_core.tools.comfyui.get_comfyui_dirs", return_value=dirs)
        self.p.start()
        self.net = mock.patch.object(H.requests, "get", side_effect=AssertionError("no network call"))
        self.net.start()

    def tearDown(self):
        self.net.stop()
        self.p.stop()
        self.tmp.cleanup()

    def test_find_hf_file_refuses_a_film_and_names_the_local_copy(self):
        r = json.loads(_call(H.find_hf_file, "hills_previz_v005.mp4"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["refused"], "input_file")
        self.assertTrue(r["local"][0].endswith("output/imageviewer/videos/hills_previz_v005.mp4"))
        self.assertIn("upload_image", r["hint"])

    def test_download_refuses_media_and_starts_no_job(self):
        with mock.patch.object(H._threading, "Thread", side_effect=AssertionError("no download")):
            r = json.loads(_call(H.download_hf_model, "Lightricks/LTX-2", "sample.mp4", destination_folder="examples"))
        self.assertFalse(r["ok"])
        self.assertIn("not a model", r["error"])
        self.assertNotIn("local", r)
        self.assertIn("list_files", r["hint"])

    def test_download_refuses_dotdot_folder(self):
        with mock.patch.object(H._threading, "Thread", side_effect=AssertionError("no download")):
            r = json.loads(_call(H.download_hf_model, "a/b", "x.safetensors", destination_folder="../examples"))
        self.assertFalse(r["ok"])

    def test_hf_search_and_hf_file_refuse(self):
        with mock.patch.object(R.requests, "get", side_effect=AssertionError("no network call")):
            self.assertFalse(json.loads(_call(R.hf_search, query="hills_previz_v005.mp4"))["ok"])
            self.assertFalse(json.loads(_call(R.hf_search, query="ltx", file_pattern="*look_frame.png"))["ok"])
        self.assertFalse(json.loads(_call(R.hf_file, repo_id="Lightricks/LTX-2", path="examples/a.mp4"))["ok"])

    def test_model_files_still_pass(self):
        self.assertIsNone(H.input_file_refusal("ltx-2.5-22b-dev-fp8.safetensors", "x"))
        self.assertIsNone(H.input_file_refusal("model.gguf", "x"))


if __name__ == "__main__":
    unittest.main()
