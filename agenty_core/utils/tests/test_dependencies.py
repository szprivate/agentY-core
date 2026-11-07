"""What a file was made from, and what was made from it.

A plate gets a depth pass, the depth pass feeds a render. Pinned here, on real
files in a scratch ComfyUI layout (no ComfyUI, no GPU):

* provenance.write records the inputs (path + fingerprint), the kind and the
  settings that decide reuse, and the input learns what was made from it;
* the same pass with the same frames is found again, a different frame range is not,
  and a render is never offered for reuse (a new seed is a new picture);
* re-grading the plate makes the depth pass out of date, and the render built on
  it too;
* the copy agentY stages into ComfyUI's input dir counts as the same file;
* a file written before this existed is worked out from its stored graph;
* graph() walks up and down from any file.

    python -m unittest agenty_core.utils.tests.test_dependencies
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from agenty_core.utils import dependencies as dp
from agenty_core.utils import provenance as pv


def depth_graph(frames=97):
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "plate.mp4", "frame_load_cap": frames,
                                                         "skip_first_frames": 0}},
        "2": {"class_type": "DepthAnythingV2Preprocessor",
              "inputs": {"image": ["1", 0], "ckpt_name": "depth_anything_v2_vitl.pth", "resolution": 1024}},
        "3": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["2", 0], "filename_prefix": "depth"}},
    }


def mask_graph(prompt="person"):
    return {
        "1": {"class_type": "VHS_LoadVideo", "inputs": {"video": "plate.mp4", "frame_load_cap": 97}},
        "2": {"class_type": "SAM3VideoSegmentation", "inputs": {"images": ["1", 0], "prompt": prompt}},
        "3": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["2", 0], "filename_prefix": "mask"}},
    }


def render_graph(depth_name):
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": f"{depth_name} [output]"}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "3": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "latent_image": ["1", 0], "seed": 5}},
        "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "render"}},
    }


class Deps(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.inp, self.out = self.tmp / "input", self.tmp / "output"
        self.inp.mkdir()
        self.out.mkdir()
        dp.set_roots({"input": str(self.inp), "output": str(self.out), "temp": str(self.tmp / "temp")})
        self.plate = self.inp / "plate.mp4"
        self.plate.write_bytes(b"plate-v1" * 1000)

    def tearDown(self):
        dp.set_roots(None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, name, graph, node="3"):
        p = self.out / name
        p.write_bytes(name.encode() * 50)
        self.assertTrue(pv.write(p, graph, output_node=node, workflow_file=f"/wf/{name}.json"))
        return p

    def test_write_records_inputs_kind_and_settings(self):
        depth = self.make("depth_00001.mp4", depth_graph())
        d = dp.recorded_deps(depth)
        self.assertEqual(d["kind"], "depth")
        self.assertEqual([i["path"] for i in d["inputs"]], [str(self.plate.resolve())])
        self.assertEqual(d["inputs"][0]["fp"], dp.fingerprint(self.plate))
        self.assertEqual(d["params"]["frame_load_cap"], 97)
        self.assertEqual(d["params"]["ckpt_name"], "depth_anything_v2_vitl.pth")
        made = dp.derived_of(self.plate)
        self.assertEqual([m["path"] for m in made], [str(depth.resolve())])
        self.assertTrue((self.inp / "agent_metadata" / "plate.mp4.derived.json").is_file())

    def test_the_same_pass_is_found_and_a_different_one_is_not(self):
        depth = self.make("depth_00001.mp4", depth_graph())
        self.make("mask_00001.mp4", mask_graph("person"))
        hits = dp.existing_for_graph(depth_graph())
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["kind"], "depth")
        self.assertEqual(hits[0]["existing"], [str(depth.resolve())])
        self.assertEqual(dp.existing_for_graph(depth_graph(frames=49)), [])
        self.assertEqual(len(dp.existing_for_graph(mask_graph("person"))), 1)
        self.assertEqual(dp.existing_for_graph(mask_graph("red car")), [])

    def test_a_render_is_never_offered_for_reuse(self):
        self.make("depth_00001.mp4", depth_graph())
        r = self.make("render_00001.png", render_graph("depth_00001.mp4"), node="4")
        self.assertEqual(dp.recorded_deps(r)["kind"], "render")
        self.assertEqual(dp.existing_for_graph(render_graph("depth_00001.mp4")), [])

    def test_a_changed_plate_makes_everything_after_it_out_of_date(self):
        depth = self.make("depth_00001.mp4", depth_graph())
        r = self.make("render_00001.png", render_graph("depth_00001.mp4"), node="4")
        self.assertEqual(dp.stale_reason(depth), "")
        time.sleep(0.01)
        self.plate.write_bytes(b"plate-v2-regraded" * 1000)
        self.assertIn("plate.mp4 changed", dp.stale_reason(depth))
        self.assertIn("depth_00001.mp4 is out of date", dp.stale_reason(r))
        # ...and a pass of the old plate is no longer offered.
        self.assertEqual(dp.existing_for_graph(depth_graph()), [])

    def test_a_staged_copy_is_the_same_file(self):
        depth = self.make("depth_00001.mp4", depth_graph())
        (self.inp / "agentY").mkdir()
        staged = self.inp / "agentY" / "depth_00001.mp4"
        shutil.copyfile(depth, staged)
        g = render_graph("agentY/depth_00001.mp4")
        g["1"]["inputs"]["image"] = "agentY/depth_00001.mp4"
        r = self.make("render_00001.png", g, node="4")
        self.assertEqual([m["path"] for m in dp.derived_of(depth)], [str(r.resolve())])

    def test_an_old_file_is_worked_out_from_its_graph(self):
        depth = self.make("depth_00001.mp4", depth_graph())
        sc = pv.sidecar_path(depth)
        doc = json.loads(sc.read_text())
        del doc["provenance"]["deps"]
        sc.write_text(json.dumps(doc))
        os.remove(dp.derived_path(self.plate))
        self.assertEqual(dp.recorded_deps(depth), {})
        self.assertEqual(dp.reindex(self.out), 1)
        self.assertEqual(dp.recorded_deps(depth)["kind"], "depth")
        self.assertEqual([m["path"] for m in dp.derived_of(self.plate)], [str(depth.resolve())])

    def test_graph_walks_up_and_down(self):
        depth = self.make("depth_00001.mp4", depth_graph())
        r = self.make("render_00001.png", render_graph("depth_00001.mp4"), node="4")
        g = dp.graph(depth)
        ids = {n["id"]: n for n in g["nodes"]}
        self.assertEqual(set(ids), {str(self.plate.resolve()), str(depth.resolve()), str(r.resolve())})
        self.assertTrue(ids[str(depth.resolve())]["root"])
        self.assertEqual(ids[str(r.resolve())]["kind"], "render")
        edges = {(e["from"], e["to"]) for e in g["edges"]}
        self.assertEqual(edges, {(str(self.plate.resolve()), str(depth.resolve())),
                                 (str(depth.resolve()), str(r.resolve()))})

    def test_prompts_come_with_the_file_that_was_made_from_them(self):
        self.make("depth_00001.mp4", depth_graph())
        g = render_graph("depth_00001.mp4")
        g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": "a kitchen at dawn, warm light"}}
        g["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": "blurry"}}
        g["3"]["inputs"].update({"positive": ["5", 0], "negative": ["6", 0]})
        # A second output in the same graph, with a prompt of its own that is not this file's.
        g["7"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 1], "text": "another shot"}}
        g["8"] = {"class_type": "KSampler", "inputs": {"model": ["2", 0], "positive": ["7", 0], "seed": 1}}
        g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "other"}}
        r = self.make("render_00001.png", g, node="4")
        self.assertEqual([(x["role"], x["text"]) for x in dp.prompts_of(r)],
                         [("positive", "a kitchen at dawn, warm light"), ("negative", "blurry")])
        ids = {n["id"]: n for n in dp.graph(r)["nodes"]}
        self.assertEqual(len(ids[str(r.resolve())]["prompts"]), 2)
        self.assertNotIn("prompts", ids[str(self.plate.resolve())])
        # A pass's own text (SAM3's subject) is its prompt too.
        m = self.make("mask_00001.mp4", mask_graph("person"))
        self.assertEqual([x["text"] for x in dp.prompts_of(m)], ["person"])

    def test_classify(self):
        self.assertEqual(dp.kind_of_class("DWPreprocessor"), "pose")
        self.assertEqual(dp.kind_of_class("CannyEdgePreprocessor"), "edges")
        self.assertEqual(dp.kind_of_class("HEDPreprocessor"), "edges")
        self.assertEqual(dp.kind_of_class("CachedImageLoader"), "")
        self.assertEqual(dp.kind_of_class("ImageUpscaleWithModel"), "upscale")
        self.assertEqual(dp.kind_of_class("Sam2Segmentation"), "mask")
        self.assertEqual(dp.kind_of_class("ImageScaleBy"), "")
        self.assertEqual(dp.kind_of_class("DraftImage"), "")
        self.assertEqual(dp.classify(mask_graph(), "3")["kind"], "mask")

    def test_model_files_and_prefixes_are_not_inputs(self):
        self.assertEqual(dp.resolve_value("sdxl.safetensors"), "")
        self.assertEqual(dp.resolve_value("agentY/render"), "")
        self.assertEqual(dp.resolve_value("plate.mp4 [input]"), str(self.plate.resolve()))
        self.assertEqual(dp.resolve_value(str(self.plate)), str(self.plate.resolve()))

    def test_fingerprint_matches_a_copy_and_not_an_edit(self):
        copy = self.tmp / "copy.mp4"
        shutil.copyfile(self.plate, copy)
        self.assertEqual(dp.fingerprint(copy), dp.fingerprint(self.plate))
        copy.write_bytes(b"other")
        self.assertNotEqual(dp.fingerprint(copy), dp.fingerprint(self.plate))
        self.assertEqual(dp.fingerprint(self.tmp / "missing.mp4"), "")


if __name__ == "__main__":
    unittest.main()
