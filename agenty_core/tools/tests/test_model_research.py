"""The model-research tools, offline: the Hub faked, the model folders temporary."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import model_research as mr


class _Resp:
    def __init__(self, status=200, body=None, headers=None, text=""):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = text
        self.ok = status < 400

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


REPOS = [
    {"id": "Lightricks/LTX-2.5", "downloads": 9, "gated": "auto", "pipeline_tag": "image-to-video",
     "siblings": [{"rfilename": "text_encoders/gemma4-ltx.safetensors"},
                  {"rfilename": "vae/ltx-vae.safetensors"}]},
    {"id": "comfyicu/LTX-2.5", "downloads": 5, "gated": False,
     "siblings": [{"rfilename": "text_encoders/gemma4-ltx.safetensors"}]},
    {"id": "someone/ltx-2.5-lora", "downloads": 1, "gated": False,
     "siblings": [{"rfilename": "lora.safetensors"}]},
]


def _search(params):
    q = (params.get("search") or "").lower()
    author = (params.get("author") or "").lower()
    out = [r for r in REPOS if q in r["id"].lower() and (not author or r["id"].lower().startswith(author + "/"))]
    if params.get("full") != "true":
        out = [{k: v for k, v in r.items() if k != "siblings"} for r in out]
    return out


class Searching(unittest.TestCase):

    def setUp(self):
        self.seen = []

        def get(url, headers=None, params=None, timeout=None):
            self.seen.append(dict(params or {}))
            return _Resp(200, _search(params or {}))
        p = mock.patch.object(mr.requests, "get", side_effect=get)
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(mr, "_folder_paths", return_value={})
        p.start()
        self.addCleanup(p.stop)

    def test_words_become_the_form_repo_ids_take(self):
        out = json.loads(mr.hf_search("ltx 2.5"))
        self.assertEqual(out["repos"][0]["repo_id"], "Lightricks/LTX-2.5")
        self.assertIn("ltx-2.5", out["searched"])

    def test_a_publisher_in_the_query_becomes_the_author(self):
        out = json.loads(mr.hf_search("comfyicu ltx-2.5"))
        self.assertEqual(out["author"], "comfyicu")
        self.assertEqual([r["repo_id"] for r in out["repos"]], ["comfyicu/LTX-2.5"])

    def test_file_pattern_lists_the_files_with_their_folder(self):
        out = json.loads(mr.hf_search("ltx-2.5", file_pattern="*gemma*"))
        ids = [r["repo_id"] for r in out["repos"]]
        self.assertEqual(ids, ["Lightricks/LTX-2.5", "comfyicu/LTX-2.5"])     # not the lora repo
        f = out["repos"][0]["files"][0]
        self.assertEqual((f["path"], f["comfy_folder"]), ("text_encoders/gemma4-ltx.safetensors", "text_encoders"))

    def test_nothing_found_says_how_to_search(self):
        out = json.loads(mr.hf_search("a model that does not exist"))
        self.assertEqual(out["count"], 0)
        self.assertIn("repo NAMES", out["hint"])


class OneFile(unittest.TestCase):

    def head(self, status, headers):
        return mock.patch.object(mr.requests, "head", return_value=_Resp(status, headers=headers))

    def setUp(self):
        for name, value in (("_folder_paths", {}), ("_local_index", []),
                            ("_resolve_download_dir", (Path("/models/text_encoders"), "x"))):
            p = mock.patch.object(mr, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(mr.requests, "get",
                              side_effect=lambda url, headers=None, params=None, timeout=None:
                              _Resp(200, _search(params or {})))
        p.start()
        self.addCleanup(p.stop)

    def test_there_with_its_size(self):
        with self.head(302, {"X-Linked-Size": "26263860594"}):
            out = json.loads(mr.hf_file(url="https://huggingface.co/comfyicu/LTX-2.5/resolve/main/text_encoders/gemma4-ltx.safetensors"))
        self.assertEqual(out["access"], "ok")
        self.assertEqual(out["size_gb"], 26.26)
        self.assertEqual(out["would_download_to"], str(Path("/models/text_encoders")))

    def test_gated_names_a_mirror_we_can_download(self):
        with self.head(403, {"X-Error-Code": "GatedRepo"}):
            out = json.loads(mr.hf_file(repo_id="Lightricks/LTX-2.5", path="text_encoders/gemma4-ltx.safetensors"))
        self.assertEqual(out["access"], "gated")
        self.assertIn("https://huggingface.co/Lightricks/LTX-2.5", out["how"])   # where to unlock it
        self.assertEqual(out["same_file_elsewhere"][0]["repo_id"], "comfyicu/LTX-2.5")
        self.assertFalse(out["same_file_elsewhere"][0]["gated"])

    def test_missing(self):
        with self.head(404, {"X-Error-Code": "EntryNotFound"}):
            out = json.loads(mr.hf_file(repo_id="comfyicu/LTX-2.5", path="nope.safetensors"))
        self.assertEqual(out["access"], "missing")
        self.assertIn("hf_repo", out["hint"])


class OnThisMachine(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for rel in ("loras/ltx/ltx-2.5-ic-lora.safetensors", "vae/ltx-2.5-vae.safetensors",
                    "vae/readme.txt", "text_encoders/gemma4.gguf"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"x" * 10)
        roots = [(c, root / c) for c in ("loras", "vae", "text_encoders")]
        p = mock.patch.object(mr, "_model_roots", return_value=roots)
        p.start()
        self.addCleanup(p.stop)
        self.root = root

    def test_by_part_of_the_name(self):
        out = json.loads(mr.find_local_models("ltx-2.5"))
        names = {(f["category"], f["name"]) for f in out["files"]}
        self.assertEqual(names, {("loras", "ltx/ltx-2.5-ic-lora.safetensors"), ("vae", "ltx-2.5-vae.safetensors")})

    def test_a_glob_and_a_category(self):
        out = json.loads(mr.find_local_models("*gemma*", category="text_encoders"))
        self.assertEqual([f["name"] for f in out["files"]], ["gemma4.gguf"])
        self.assertEqual(json.loads(mr.find_local_models("*gemma*", category="vae"))["count"], 0)

    def test_a_workflow_file(self):
        wf = {
            "nodes": [
                {"id": 1, "type": "LoadImage", "widgets_values": ["ref.png", "image"]},
                {"id": 2, "type": "sub-1", "widgets_values": ["ltx-2.5-vae.safetensors"]},
                {"id": 3, "type": "SaveVideo", "widgets_values": ["out"]},
                {"id": 4, "type": "MarkdownNote", "widgets_values": ["Put the LoRA in loras/ltx"]},
            ],
            "definitions": {"subgraphs": [{"id": "sub-1", "name": "Load Models", "nodes": [
                {"id": 10, "type": "VAELoader", "widgets_values": ["ltx-2.5-vae.safetensors"],
                 "properties": {"models": [{"name": "ltx-2.5-vae.safetensors",
                                            "url": "https://huggingface.co/x/y/resolve/main/vae/ltx-2.5-vae.safetensors",
                                            "directory": "vae"}]}},
                {"id": 11, "type": "LoraLoaderModelOnly", "widgets_values": ["missing-lora.safetensors", 1.0]},
            ]}]},
        }
        path = self.root / "wf.json"
        path.write_text(json.dumps(wf), encoding="utf-8")
        out = json.loads(mr.inspect_workflow_file(str(path)))
        self.assertEqual(out["subgraphs"], [{"name": "Load Models", "nodes": 2}])
        self.assertIn("subgraph:Load Models", out["node_types"])
        models = {m["file"]: (m["node"], m["installed"]) for m in out["models"]}
        self.assertEqual(models, {"ltx-2.5-vae.safetensors": ("VAELoader", True),
                                  "missing-lora.safetensors": ("LoraLoaderModelOnly", False)})
        self.assertEqual(out["inputs"], [{"node": "LoadImage", "value": "ref.png"}])
        self.assertEqual(out["outputs"][0]["node"], "SaveVideo")
        self.assertEqual(out["declared_downloads"][0]["directory"], "vae")
        self.assertIn("loras/ltx", out["notes"][0])

    def test_an_api_format_workflow(self):
        path = self.root / "api.json"
        path.write_text(json.dumps({"1": {"class_type": "UNETLoader",
                                          "inputs": {"unet_name": "x.safetensors"}}}), encoding="utf-8")
        out = json.loads(mr.inspect_workflow_file(str(path)))
        self.assertEqual(out["models"][0]["file"], "x.safetensors")


if __name__ == "__main__":
    unittest.main()
