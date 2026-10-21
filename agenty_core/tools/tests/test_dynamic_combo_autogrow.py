"""A dynamic combo brings its own inputs, autogrow groups among them.

ByteDance2ReferenceNodeV2 (Seedance 2.5 reference-to-video) has ONE schema-level
input beside seed and watermark: ``model``, a ``COMFY_DYNAMICCOMBO_V3`` whose
options ("Seedance 2.5", "Seedance 2.5 Draft", ...) each carry prompt, resolution,
... and the autogrow groups ``reference_images`` (image_1..30), ``reference_videos``
and ``reference_audios``. The prompt addresses them ``model.prompt`` and
``model.reference_images.image_1``. The schema the agent was shown listed none of
them, so it believed the node had no reference slots and asked the user to place
the node by hand - although ComfyUI accepts that prompt as is.

Built on the node's real /object_info entry (fixtures/).

    python -m unittest agenty_core.tools.tests.test_dynamic_combo_autogrow
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comfyui as C
from agenty_core.tools.assembly_deterministic import (
    declared_input_keys,
    describe_input_names,
    expand_input_specs,
    normalize_autogrow_keys,
    undeclared_wires,
)

FIXTURE = Path(__file__).parent / "fixtures" / "seedance_ref_v2_object_info.json"
INFO = json.loads(FIXTURE.read_text(encoding="utf-8"))["ByteDance2ReferenceNodeV2"]
REQUIRED = INFO["input"]["required"]
OPTIONAL = INFO["input"].get("optional") or {}
OBJECT_INFO = {
    "ByteDance2ReferenceNodeV2": INFO,
    "LoadImage": {"input": {"required": {"image": [["a.png"], {}]}},
                  "output": ["IMAGE", "MASK"], "output_name": ["IMAGE", "MASK"]},
    "LoadVideo": {"input": {"required": {"file": [["a.mp4"], {}]}},
                  "output": ["VIDEO"], "output_name": ["VIDEO"]},
}


def seedance(option="Seedance 2.5 Draft", **extra):
    inputs = {"model": option, "model.prompt": "a cat", "model.resolution": "480p",
              "model.ratio": "16:9", "model.duration": 6, "model.generate_audio": False,
              "model.task_type": "reference", "model.output_format": "mp4",
              "seed": 7, "watermark": False}
    inputs.update(extra)
    return {"class_type": "ByteDance2ReferenceNodeV2", "inputs": inputs}


class TheExpansion(unittest.TestCase):

    def test_the_selected_option_brings_dotted_inputs_and_nested_slots(self):
        specs = expand_input_specs(REQUIRED, OPTIONAL, {"model": "Seedance 2.5 Draft"})
        for name in ("model.prompt", "model.duration", "model.reference_images",
                     "model.reference_images.image_1", "model.reference_images.image_30",
                     "model.reference_videos.video_10", "model.auto_downscale"):
            self.assertIn(name, specs)
        self.assertEqual(specs["model.reference_images.image_1"][0], "IMAGE")
        self.assertEqual(specs["model.reference_videos.video_1"][0], "VIDEO")
        self.assertNotIn("model.reference_images.image_31", specs)

    def test_an_unknown_option_contributes_nothing(self):
        self.assertNotIn("model.prompt", expand_input_specs(REQUIRED, OPTIONAL, {"model": "nope"}))

    def test_without_a_value_the_first_option_is_in_force(self):
        self.assertIn("model.prompt", expand_input_specs(REQUIRED, OPTIONAL, {}))

    def test_the_names_for_a_message_collapse_each_group_to_a_range(self):
        shown = describe_input_names(expand_input_specs(REQUIRED, OPTIONAL, {"model": "Seedance 2.5"}))
        self.assertIn("model.reference_images.image_1..image_30", shown)
        self.assertIn("model.reference_videos.video_1..video_10", shown)
        self.assertFalse(any(n.endswith("image_2") for n in shown), shown)


class Declared(unittest.TestCase):

    def test_the_prompt_from_the_working_run_has_no_undeclared_wire(self):
        node = seedance(**{"model.reference_videos.video_1": ["v", 0],
                           "model.reference_images.image_1": ["i1", 0],
                           "model.reference_images.image_2": ["i2", 0]})
        keys, _ = declared_input_keys(node, REQUIRED, OPTIONAL)
        self.assertIn("model.reference_images.image_2", keys)
        self.assertEqual(undeclared_wires(node, REQUIRED, OPTIONAL, INFO), [])

    def test_a_slot_of_an_unknown_group_is_reported(self):
        node = seedance(**{"model.reference_pictures.image_1": ["i1", 0],
                           "model.reference_images.image_31": ["i1", 0]})
        self.assertEqual(sorted(undeclared_wires(node, REQUIRED, OPTIONAL, INFO)),
                         ["model.reference_images.image_31", "model.reference_pictures.image_1"])

    def test_a_bare_slot_name_is_moved_to_its_full_address(self):
        node = seedance(image_1=["i1", 0])
        normalize_autogrow_keys(node, REQUIRED, OPTIONAL)
        self.assertEqual(node["inputs"]["model.reference_images.image_1"], ["i1", 0])
        self.assertNotIn("image_1", node["inputs"])


class TheSchemaTheAgentSees(unittest.TestCase):

    def test_the_dynamic_combo_lists_every_option_with_its_dotted_inputs(self):
        entry = C._parse_inputs_schema(REQUIRED)["model"]
        self.assertTrue(entry["dynamic"])
        self.assertIn("Seedance 2.5 Draft", entry["options"])
        draft = entry["option_inputs"]["Seedance 2.5 Draft"]
        self.assertIn("model.prompt", draft)
        self.assertIn("model.reference_images", draft)
        self.assertEqual(draft["model.reference_images"]["connect_as"][0],
                         "model.reference_images.image_1")
        self.assertTrue(draft["model.auto_downscale"]["optional"])

    def test_get_node_schema_carries_it(self):
        client = mock.Mock()
        client.get.return_value = {"ByteDance2ReferenceNodeV2": INFO}
        with mock.patch.object(C, "get_client", return_value=client):
            out = json.loads(C._get_node_schema_one("ByteDance2ReferenceNodeV2"))
        options = out["input_required"]["model"]["option_inputs"]["Seedance 2.5"]
        self.assertIn("model.reference_videos", options)


class ThroughTheTools(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for target, value in (("_get_object_info", OBJECT_INFO), ("_server_validate", {}),
                              ("_workflows_dir", self.tmp / "agentY"),
                              ("_corpus_root", self.tmp / "corpus")):
            patcher = mock.patch.object(C, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def write(self, node):
        wf = {"i1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
              "i2": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
              "v": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}},
              "9": node}
        path = self.tmp / "work" / "s.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(wf), encoding="utf-8")
        return path

    def test_the_working_prompt_validates_locally(self):
        node = seedance(**{"model.reference_videos.video_1": ["v", 0],
                           "model.reference_images.image_1": ["i1", 0],
                           "model.reference_images.image_2": ["i2", 0]})
        out = json.loads(C.validate_workflow(str(self.write(node))))
        self.assertEqual(out["local_errors"], [])

    def test_an_invented_slot_lists_the_valid_dotted_names_compactly(self):
        node = seedance(**{"model.reference_pictures.image_1": ["i1", 0]})
        errors = json.loads(C.validate_workflow(str(self.write(node))))["local_errors"]
        hit = [e for e in errors if "'model.reference_pictures.image_1'" in e]
        self.assertEqual(len(hit), 1, errors)
        self.assertIn("model.reference_images.image_1..image_30", hit[0])
        self.assertNotIn("image_17", hit[0])


class TheCanvasRoundTrip(unittest.TestCase):
    """API -> litegraph -> API keeps every dotted wire and widget."""

    def setUp(self):
        patcher = mock.patch.object(C, "_get_object_info", return_value=OBJECT_INFO)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_nested_slots_are_typed_labelled_and_survive_the_trip(self):
        api = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
               "2": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}},
               "3": seedance(**{"model.reference_images.image_1": ["1", 0],
                                "model.reference_videos.video_1": ["2", 0]})}
        graph = C._api_to_graph(api)
        node = next(n for n in graph["nodes"] if n["type"] == "ByteDance2ReferenceNodeV2")
        socket = {i["name"]: i for i in node["inputs"]}
        self.assertEqual(socket["model.reference_images.image_1"]["type"], "IMAGE")
        self.assertEqual(socket["model.reference_images.image_1"]["label"], "image_1")
        self.assertEqual(socket["model.reference_videos.video_1"]["type"], "VIDEO")
        back = C._convert_graph_to_api(graph)
        inputs = next(n for n in back.values()
                      if n["class_type"] == "ByteDance2ReferenceNodeV2")["inputs"]
        self.assertEqual(inputs["model.reference_images.image_1"], ["1", 0])
        self.assertEqual(inputs["model.reference_videos.video_1"], ["2", 0])
        self.assertEqual(inputs["model"], "Seedance 2.5 Draft")
        self.assertEqual(inputs["model.prompt"], "a cat")
        self.assertEqual(inputs["model.duration"], 6)


if __name__ == "__main__":
    unittest.main()
