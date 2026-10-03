"""The looking-around tools, offline: a fake ComfyUI tree, temporary folders and
media, the web mocked."""

import json
import os
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import workspace_tools as w

NODES_LT = '''
import torch

def get_noise_mask(latent):
    return latent

class _Base:
    def common(self):
        return 1

class LTXVAddGuide(_Base):
    @classmethod
    def execute(cls, latent):
        mask = get_noise_mask(latent)
        return append_keyframe(mask)

def append_keyframe(x):
    """Put the guide frame in."""
    return x

def unrelated():
    return 2
'''

PACK = '''
class ResizeV2:
    def go(self):
        return helper_scale()

def helper_scale():
    return 3

NODE_CLASS_MAPPINGS = {"ImageResizeKJv2": ResizeV2}
'''


class _Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class NodeSource(_Tmp):

    def setUp(self):
        super().setUp()
        (self.root / "comfy_extras").mkdir()
        (self.root / "comfy_extras" / "nodes_lt.py").write_text(NODES_LT, encoding="utf-8")
        pack = self.root / "custom_nodes" / "ComfyUI-KJNodes" / "nodes"
        pack.mkdir(parents=True)
        (pack / "image_nodes.py").write_text(PACK, encoding="utf-8")
        (self.root / "comfy").mkdir()
        (self.root / "comfy" / "model_base.py").write_text("x = 1\nout['minimax_payload'] = 2\n", encoding="utf-8")
        (self.root / "nodes.py").write_text("class LoadImage: pass\n", encoding="utf-8")
        p = mock.patch.object(w, "_comfy_root", return_value=self.root)
        p.start()
        self.addCleanup(p.stop)
        modules = {"LTXVAddGuide": "comfy_extras.nodes_lt", "ImageResizeKJv2": "custom_nodes.ComfyUI-KJNodes"}
        client = mock.Mock()
        client.get.side_effect = lambda path: {path.rsplit("/", 1)[-1]: {"python_module": modules[path.rsplit("/", 1)[-1]]}} \
            if path.rsplit("/", 1)[-1] in modules else {}
        p = mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client)
        p.start()
        self.addCleanup(p.stop)

    def test_a_core_node_with_the_helpers_and_base_it_uses(self):
        out = json.loads(w.get_node_source("LTXVAddGuide"))
        self.assertTrue(out["ok"], out)
        self.assertIn("class LTXVAddGuide(_Base)", out["source"])
        self.assertEqual(set(out["helpers_included"]), {"_Base", "get_noise_mask", "append_keyframe"})
        self.assertNotIn("def unrelated", out["source"])

    def test_a_node_packs_node_found_through_its_mapping(self):
        out = json.loads(w.get_node_source("ImageResizeKJv2"))
        self.assertEqual(out["class"], "ResizeV2")
        self.assertIn("def helper_scale", out["source"])

    def test_a_query_in_a_nodes_file_and_in_core(self):
        out = json.loads(w.get_node_source("LTXVAddGuide", query="def append_keyframe", context=1))
        self.assertEqual(out["matches"][0]["line"], 17)
        self.assertIn("Put the guide frame in", out["matches"][0]["code"])
        out = json.loads(w.get_node_source(query="minimax_payload", path="comfy/model_base.py"))
        self.assertEqual(out["matches"][0]["line"], 2)
        out = json.loads(w.get_node_source(query="minimax_payload"))       # core by default
        self.assertEqual(len(out["matches"]), 1)

    def test_an_unknown_node(self):
        out = json.loads(w.get_node_source("NoSuchNode"))
        self.assertFalse(out["ok"])
        self.assertIn("search_nodes", out["hint"])

    def test_it_stays_within_its_budget(self):
        out = json.loads(w.get_node_source("LTXVAddGuide", max_chars=2000))
        self.assertLessEqual(len(out["source"]), 2000)


class ListingFiles(_Tmp):

    def setUp(self):
        super().setUp()
        for n in [1001, 1002, 1003, 1005]:
            (self.root / f"beauty.{n}.exr").write_bytes(b"x")
        (self.root / "notes.txt").write_text("hi")
        time.sleep(0.02)
        (self.root / "latest.mp4").write_bytes(b"xx")
        (self.root / "sub").mkdir()
        (self.root / "sub" / "deep.png").write_bytes(b"x")

    def test_sequences_collapse_and_newest_comes_first(self):
        out = json.loads(w.list_files(str(self.root)))
        names = [f["name"] for f in out["files"]]
        self.assertEqual(names[0], "latest.mp4")
        seq = next(f for f in out["files"] if f.get("sequence"))
        self.assertEqual((seq["name"], seq["frames"], seq["count"], seq["missing"]),
                         ("beauty.####.exr", "1001-1005", 4, [1004]))
        self.assertEqual(out["folders"][0]["name"], "sub/")

    def test_pattern_and_recursion(self):
        out = json.loads(w.list_files(str(self.root), pattern="*.png", recursive=True))
        self.assertEqual([f["name"] for f in out["files"]], ["sub/deep.png"])

    def test_comfyuis_folders_by_name(self):
        with mock.patch.object(w, "_comfy_dirs", return_value={"output_dir": str(self.root)}):
            out = json.loads(w.list_files("output/sub"))
        self.assertEqual([f["name"] for f in out["files"]], ["deep.png"])

    def test_a_missing_folder_or_a_single_file(self):
        self.assertFalse(json.loads(w.list_files(str(self.root / "nope")))["exists"])
        self.assertTrue(json.loads(w.list_files(str(self.root / "notes.txt")))["is_file"])


class FindingWorkflows(_Tmp):

    def test_every_word_must_be_in_it(self):
        wf = self.root / "wf"
        wf.mkdir()
        (wf / "a.json").write_text(json.dumps({"nodes": [{"type": "LTXVAddGuide"}, {"type": "UNETLoader",
                                   "widgets_values": ["wan2.2_t2v.safetensors"]}]}), encoding="utf-8")
        (wf / "b.json").write_text(json.dumps({"nodes": [{"type": "LTXVAddGuide"}]}), encoding="utf-8")
        out = json.loads(w.find_workflows("ltxvaddguide wan2.2", folder=str(wf)))
        self.assertEqual([Path(h["path"]).name for h in out["workflows"]], ["a.json"])
        self.assertEqual(out["workflows"][0]["matching_node_types"], ["LTXVAddGuide"])
        self.assertEqual(json.loads(w.find_workflows("", folder=str(wf)))["count"], 2)


def _exr(path: Path, w_: int = 64, h: int = 32) -> None:
    def attr(name, typ, val):
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(val)) + val
    chans = b"".join(c.encode() + b"\0" + struct.pack("<i", 1) + b"\0\0\0\0" + struct.pack("<ii", 1, 1)
                     for c in ("B", "G", "R")) + b"\0"
    box = struct.pack("<iiii", 0, 0, w_ - 1, h - 1)
    head = (b"\x76\x2f\x31\x01" + bytes([2, 0, 0, 0]) + attr("channels", "chlist", chans)
            + attr("compression", "compression", bytes([3])) + attr("dataWindow", "box2i", box)
            + attr("displayWindow", "box2i", box) + b"\0")
    path.write_bytes(head + b"\0" * 64)


class Media(_Tmp):

    def test_an_image_with_alpha_says_what_the_mask_will_be(self):
        from PIL import Image
        im = Image.new("RGBA", (100, 50), (255, 0, 0, 255))
        im.paste((0, 0, 0, 0), (10, 10, 30, 20))          # a 20x10 hole
        im.save(self.root / "m.png")
        out = json.loads(w.media_info(str(self.root / "m.png")))
        self.assertEqual((out["width"], out["height"], out["mode"]), (100, 50, "RGBA"))
        self.assertEqual(out["alpha"]["transparent_fraction"], 0.04)
        self.assertEqual(out["alpha"]["transparent_box"], {"x": 10, "y": 10, "width": 20, "height": 10})

    def test_an_exr_header(self):
        _exr(self.root / "r.exr")
        out = json.loads(w.media_info(str(self.root / "r.exr")))
        self.assertEqual(out["channels"], ["B:half", "G:half", "R:half"])
        self.assertEqual((out["width"], out["height"], out["compression"]), (64, 32, "zip"))

    def test_a_sequence(self):
        for n in (1, 2, 4):
            _exr(self.root / f"plate.{n:04d}.exr")
        out = json.loads(w.media_info(str(self.root / "plate.####.exr")))
        self.assertEqual(out["sequence"]["frames"], "1-4")
        self.assertEqual(out["sequence"]["missing"], [3])
        self.assertEqual(out["first_frame"]["width"], 64)

    def _video(self) -> Path:
        import imageio.v3 as iio
        import numpy as np
        path = self.root / "clip.mp4"
        frames = np.stack([np.full((64, 96, 3), i * 10, dtype=np.uint8) for i in range(12)])
        iio.imwrite(path, frames, fps=12, codec="libx264", plugin="pyav") if False else \
            iio.imwrite(path, frames, fps=12)
        return path

    def test_a_video_and_its_frames(self):
        try:
            path = self._video()
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"cannot write a test video here: {exc}")
        info = json.loads(w.media_info(str(path)))
        self.assertEqual((info["video"]["width"], info["video"]["height"]), (96, 64))
        self.assertEqual(info["video"]["frames"], 12)
        with mock.patch.object(w, "_agent_dirs", return_value={"images": str(self.root / "agent")}):
            out = json.loads(w.video_frames(str(path), count=3))
        self.assertTrue(out["ok"], out)
        self.assertEqual([f["label"] for f in out["frames"]], ["frame 0", "frame 6", "frame 11"])
        self.assertTrue(Path(out["contact_sheet"]).is_file())


class _R:
    def __init__(self, status=200, text="", ctype="text/html", body=None, url=""):
        self.status_code, self.text, self.headers, self._body, self.url = status, text, {"Content-Type": ctype}, body, url
        self.ok = status < 400

    def json(self):
        return self._body


class WebPages(unittest.TestCase):

    def test_the_pages_own_content_without_menus(self):
        page = ("<html><head><title>Guide</title><script>x()</script></head><body>"
                "<nav><ul><li></li><li>Home</li></ul></nav><main><h1>LTX prompts</h1>"
                + "<p>Give it long descriptive prompts. </p>" * 20 + "<ul><li></li><li>Tip</li></ul></main>"
                "<footer>©</footer></body></html>")
        with mock.patch.object(w.requests, "get", return_value=_R(text=page, url="https://x/guide")):
            out = json.loads(w.read_web_page("https://x/guide"))
        self.assertEqual(out["title"], "Guide")
        self.assertTrue(out["text"].startswith("# LTX prompts"))
        self.assertIn("- Tip", out["text"])
        self.assertNotIn("Home", out["text"])
        self.assertNotIn("x()", out["text"])
        self.assertNotIn("\n-\n", out["text"])

    def test_a_github_repo(self):
        def get(url, headers=None, timeout=None, params=None):
            if url.endswith("/repos/Lightricks/ComfyUI-LTXVideo"):
                return _R(body={"description": "LTX for ComfyUI", "default_branch": "master", "stargazers_count": 5})
            if "/contents/" in url:
                return _R(body=[{"name": "README.md", "type": "file"}, {"name": "example_workflows", "type": "dir"}])
            if url.endswith("/master/README.md"):
                return _R(text="# LTXVideo\nInstall it.")
            return _R(status=404)
        with mock.patch.object(w.requests, "get", side_effect=get):
            out = json.loads(w.read_web_page("https://github.com/Lightricks/ComfyUI-LTXVideo"))
        self.assertEqual(out["kind"], "github repo")
        self.assertEqual(out["files"], ["README.md", "example_workflows/"])
        self.assertIn("Install it", out["readme"])

    def test_an_error_page(self):
        with mock.patch.object(w.requests, "get", return_value=_R(status=404)):
            self.assertFalse(json.loads(w.read_web_page("https://x/missing"))["ok"])


if __name__ == "__main__":
    unittest.main()
