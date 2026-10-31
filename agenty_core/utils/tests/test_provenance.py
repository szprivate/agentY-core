"""Provenance v1: the graph that made a file, kept beside it (sidecar + .prov.json.gz),
read back from a PNG's text chunks, a video's container metadata, or ComfyUI's history.
"""
import gzip
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from agenty_core.utils import provenance as pv

FAKE_KEY = "sk-test-FAKEKEY1234567890abcdef"

GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 912334, "steps": 30, "cfg": 7, "sampler_name": "euler",
                                               "scheduler": "normal", "denoise": 1, "model": ["4", 0],
                                               "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl_base.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 576, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cozy beach house at dusk", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "beach", "images": ["8", 0]}},
    "12": {"class_type": "KlingImage2VideoNode", "inputs": {"prompt": "walk in", "duration": 5,
                                                              "api_key": FAKE_KEY, "image": ["10", 0]}},
}


def make_png(path, prompt=None, workflow=None, kinds=("tEXt",)):
    """A real 1x1 PNG with text chunks written by hand."""
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    for key, val in (("prompt", prompt), ("workflow", workflow)):
        if val is None:
            continue
        text = json.dumps(val).encode()
        if "iTXt" in kinds:
            out += chunk(b"iTXt", key.encode() + b"\x00\x01\x00\x00\x00" + zlib.compress(text))
        elif "zTXt" in kinds:
            out += chunk(b"zTXt", key.encode() + b"\x00\x00" + zlib.compress(text))
        else:
            out += chunk(b"tEXt", key.encode() + b"\x00" + text)
    out += chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00")) + chunk(b"IEND", b"")
    Path(path).write_bytes(out)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.no_net = mock.patch("agenty_core.utils.comfyui_client.get_client", side_effect=RuntimeError("no net"))
        self.no_net.start()
        self.addCleanup(self._stop)

    def _stop(self):
        try:
            self.no_net.stop()
        except RuntimeError:
            pass


class Sidecar(Base):
    def test_write_then_read_merges_and_keeps_the_role(self):
        f = self.tmp / "a.png"
        f.write_bytes(b"x")
        Path(str(f) + ".agenty.json").write_text(json.dumps({"role": "hero", "made_by": "agentY"}))
        ok = pv.write(f, GRAPH, prompt_id="pid1", output_node="9", output_index=0, workflow_file="w.json")
        self.assertTrue(ok)
        doc = json.loads(Path(str(f) + ".agenty.json").read_text())
        self.assertEqual(doc["role"], "hero")
        p = doc["provenance"]
        self.assertEqual((p["v"], p["source"], p["prompt_id"], p["output_node"]), (1, "agentY", "pid1", "9"))
        self.assertEqual(p["graph_file"], "a.png.prov.json.gz")
        self.assertEqual(p["seeds"], [{"node": "3", "input": "seed", "value": 912334}])
        self.assertIn({"node": "4", "input": "ckpt_name", "value": "sdxl_base.safetensors"}, p["models"])
        self.assertEqual(p["partner"][0]["class"], "KlingImage2VideoNode")
        blob = json.loads(gzip.decompress(Path(str(f) + ".prov.json.gz").read_bytes()))
        self.assertEqual(blob["v"], 1)
        self.assertEqual(p["graph_sha1"], pv.canonical_sha1(blob["prompt"]))
        got = pv.provenance_of(f)
        self.assertEqual((got["found"], got["source"], got["confidence"]), (True, "sidecar", "exact"))
        self.assertEqual(got["prompt_id"], "pid1")
        self.assertEqual(got["graph"]["3"]["inputs"]["seed"], 912334)

    def test_write_never_raises(self):
        self.assertFalse(pv.write(self.tmp / "no" / "x.png", {}))
        self.assertFalse(pv.write(None, GRAPH))

    def test_a_second_write_replaces_only_provenance(self):
        f = self.tmp / "a.png"
        f.write_bytes(b"x")
        pv.write(f, GRAPH, prompt_id="one")
        doc = json.loads(Path(str(f) + ".agenty.json").read_text())
        doc["role"] = "keep"
        Path(str(f) + ".agenty.json").write_text(json.dumps(doc))
        pv.write(f, GRAPH, prompt_id="two", source="regenerate",
                 derived_from={"file": "o.png", "prompt_id": "one", "changes": ["3.seed: 1 -> 2"]})
        doc = json.loads(Path(str(f) + ".agenty.json").read_text())
        self.assertEqual(doc["role"], "keep")
        self.assertEqual(doc["provenance"]["prompt_id"], "two")
        self.assertEqual(doc["provenance"]["derived_from"]["changes"], ["3.seed: 1 -> 2"])

    def test_capture_from_a_history_entry_in_the_background(self):
        a, b = self.tmp / "o_00001_.png", self.tmp / "o_00002_.png"
        for f in (a, b):
            f.write_bytes(b"x")
        hist = {"p9": {"prompt": [1, "p9", GRAPH, {"api_key_comfy_org": "comfyui-SECRET0123456789"}, []],
                       "outputs": {"9": {"images": [{"filename": a.name}, {"filename": b.name}]}}}}
        t = pv.capture([a, b], hist, "p9", "w.json", source="agentY-mcp")
        self.assertIsNotNone(t)
        pv.wait_pending()
        rec = pv.recorded(b)
        self.assertEqual((rec["source"], rec["prompt_id"], rec["output_node"], rec["output_index"]),
                         ("agentY-mcp", "p9", "9", 1))
        self.assertNotIn("SECRET0123456789", gzip.decompress(Path(str(b) + ".prov.json.gz").read_bytes()).decode())
        self.assertEqual(pv.capture([a], {}, "p9", background=False), 0)
        self.assertIsNone(pv.capture([a], {"p9": {"prompt": "junk"}}, "p9"))
        self.assertEqual(pv.recorded(self.tmp / "none.png"), {})

    def test_the_write_waits_for_a_host_holding_the_sidecar_lock(self):
        import threading
        f = self.tmp / "a.png"
        f.write_bytes(b"x")
        done = threading.Event()
        with pv.SIDECAR_LOCK:
            t = threading.Thread(target=lambda: (pv.write(f, GRAPH, prompt_id="p"), done.set()))
            t.start()
            self.assertFalse(done.wait(0.3))          # blocked while the host has the sidecar
            Path(str(f) + ".agenty.json").write_text(json.dumps({"role": "hero"}))
        t.join(10)
        doc = json.loads(Path(str(f) + ".agenty.json").read_text())
        self.assertEqual((doc["role"], doc["provenance"]["prompt_id"]), ("hero", "p"))

    def test_move_takes_the_sidecars_along(self):
        f = self.tmp / "a.png"
        f.write_bytes(b"x")
        pv.write(f, GRAPH)
        g = self.tmp / "sub" / "b.png"
        pv.move_with_sidecars(f, g)
        self.assertFalse(f.exists())
        self.assertEqual(pv.provenance_of(g)["source"], "sidecar")
        self.assertEqual(json.loads(Path(str(g) + ".agenty.json").read_text())["provenance"]["graph_file"],
                         "b.png.prov.json.gz")


class Redaction(Base):
    def test_a_key_never_reaches_the_files(self):
        g = json.loads(json.dumps(GRAPH))
        g["6"]["inputs"]["text"] = f"remember {FAKE_KEY} for later"
        f = self.tmp / "a.png"
        f.write_bytes(b"x")
        pv.write(f, g, workflow_ui={"nodes": [{"id": 12, "type": "KlingImage2VideoNode",
                                              "inputs": [{"name": "api_key", "widget": {"name": "api_key"}}],
                                              "widgets_values": [FAKE_KEY, 5]}], "links": []})
        for suf in (".agenty.json", ".prov.json.gz"):
            raw = Path(str(f) + suf).read_bytes()
            if suf.endswith("gz"):
                raw = gzip.decompress(raw)
            self.assertNotIn(FAKE_KEY.encode(), raw, suf)
            self.assertNotIn(b"FAKEKEY1234567890", raw, suf)
        got = pv.provenance_of(f)
        self.assertEqual(got["graph"]["12"]["inputs"]["api_key"], "<redacted>")
        self.assertNotIn(FAKE_KEY, json.dumps(got["graph"]))
        self.assertNotIn(FAKE_KEY, json.dumps(pv.summarize(got["graph"])))

    def test_redact_graph_is_a_copy(self):
        r = pv.redact_graph(GRAPH)
        self.assertEqual(GRAPH["12"]["inputs"]["api_key"], FAKE_KEY)
        self.assertEqual(r["12"]["inputs"]["api_key"], "<redacted>")
        self.assertEqual(r["3"]["inputs"]["seed"], 912334)


class Embedded(Base):
    def test_png_text_chunk_fallback(self):
        for kinds in (("tEXt",), ("zTXt",), ("iTXt",)):
            f = self.tmp / f"p_{kinds[0]}.png"
            make_png(f, prompt=GRAPH, workflow={"nodes": [], "links": []}, kinds=kinds)
            got = pv.provenance_of(f)
            self.assertEqual((got["found"], got["source"]), (True, "png"), kinds)
            self.assertEqual(got["confidence"], "high")
            self.assertEqual(got["output_node"], "9")      # the single SaveImage
            self.assertEqual(got["graph"]["3"]["inputs"]["steps"], 30)
            self.assertEqual(got["graph"]["12"]["inputs"]["api_key"], "<redacted>")

    def test_png_with_two_save_nodes_is_medium_unless_the_prefix_tells(self):
        g = json.loads(json.dumps(GRAPH))
        g["10"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "other", "images": ["8", 0]}}
        f = self.tmp / "ComfyUI_00001_.png"
        make_png(f, prompt=g)
        self.assertEqual(pv.provenance_of(f)["confidence"], "medium")
        f2 = self.tmp / "beach_00001_.png"
        make_png(f2, prompt=g)
        got = pv.provenance_of(f2)
        self.assertEqual((got["confidence"], got["output_node"]), ("high", "9"))

    def test_plain_png_without_metadata_is_not_found(self):
        f = self.tmp / "x.png"
        make_png(f)
        self.assertFalse(pv.provenance_of(f, use_history=False)["found"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_mp4_metadata_fallback_like_vhs_and_savevideo(self):
        g = {"1": {"class_type": "KSampler", "inputs": {"seed": 5}},
             "2": {"class_type": "VHS_VideoCombine", "inputs": {"filename_prefix": "clip", "images": ["1", 0]}}}
        for label, tags in (("vhs", ["prompt=" + json.dumps(g), "workflow=" + json.dumps({"nodes": []})]),
                            ("comment", ["comment=" + json.dumps({"prompt": g, "workflow": {"nodes": []}})])):
            f = self.tmp / f"clip_{label}.mp4"
            cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.2:r=10",
                   "-pix_fmt", "yuv420p"]
            for t in tags:
                cmd += ["-metadata", t]
            r = subprocess.run(cmd + ["-movflags", "use_metadata_tags", str(f)], capture_output=True)
            if r.returncode != 0:
                self.skipTest("ffmpeg could not make the file: " + r.stderr.decode()[:100])
            got = pv.provenance_of(f)
            self.assertEqual((got["found"], got["source"], got["confidence"]), (True, "video", "high"), label)
            self.assertEqual(got["graph"]["1"]["inputs"]["seed"], 5)
            self.assertEqual(got["output_node"], "2")


class History(Base):
    ENTRY = {"prompt": [7, "pid9", GRAPH, {"api_key_comfy_org": FAKE_KEY}, ["9"]],
             "outputs": {"9": {"images": [{"filename": "beach_00001_.png", "subfolder": "", "type": "output"}]}}}

    def test_graph_from_history_never_returns_extra_data(self):
        g = pv.graph_from_history(self.ENTRY)
        self.assertEqual(g["3"]["inputs"]["seed"], 912334)
        self.assertNotIn("api_key_comfy_org", json.dumps(g))
        self.assertEqual(pv.output_position(self.ENTRY, "D:/x/beach_00001_.png"), ("9", 0))
        self.assertEqual(pv.output_position(self.ENTRY, "other.png"), ("", 0))

    def _client(self):
        c = mock.Mock()
        c.get.return_value = {"pid9": self.ENTRY}
        return c

    def test_history_fallback(self):
        self._stop()
        f = self.tmp / "beach_00001_.png"
        f.write_bytes(b"x")
        c = self._client()
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=c):
            got = pv.provenance_of(f, prompt_id="pid9")
            self.assertEqual((got["source"], got["confidence"], got["output_node"]), ("history", "high", "9"))
            self.assertNotIn(FAKE_KEY, json.dumps(got["graph"]))
            g2 = pv.provenance_of(self.tmp / "unrelated.png", prompt_id="pid9")
            self.assertEqual((g2["source"], g2["confidence"]), ("history", "medium"))
        c.get.assert_called_with("/history/pid9")
        self.assertFalse(Path(str(f) + ".prov.json.gz").exists())      # in memory only

    def test_prompt_id_from_the_sidecar(self):
        self._stop()
        f = self.tmp / "beach_00001_.png"
        f.write_bytes(b"x")
        Path(str(f) + ".agenty.json").write_text(json.dumps({"role": "r", "prompt_id": "pid9"}))
        with mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=self._client()):
            self.assertEqual(pv.provenance_of(f)["source"], "history")

    def test_not_found(self):
        f = self.tmp / "z.png"
        f.write_bytes(b"x")
        got = pv.provenance_of(f, prompt_id="nope")
        self.assertEqual((got["found"], got["source"], got["confidence"]), (False, "none", "none"))
        self.assertIn("can't recreate exactly", got["note"])


if __name__ == "__main__":
    unittest.main()
