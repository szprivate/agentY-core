"""The wiring a recipe hands a from-scratch build.

A recipe's ``connection_patterns`` are ``(from_role, to_role, data_type)`` triples
agreed across every member, and they cannot address a wire: not which instance (two
samplers in a two-stage graph share one triple), not which input (``CFGGuider``
takes `positive` and `negative`, both CONDITIONING), not which output slot (a
checkpoint loader's VAE is slot 2). Measured on this corpus, a recipe with four or
more members shows a median of ZERO of them — the intersection of a rich group is
empty — so a 35-node build was being asked for with no wiring at all.

``_reference_wiring`` answers with the edge list of the one member the rest of the
build spec comes from, addressed by class and instance, plus the optional inputs
that member wires — the ones nothing will complain about and which carry the whole
point (an unwired ``CreateVideo.audio`` is a silent video out of an audio model).

Runs under pytest or directly (``python test_reference_wiring.py``).
"""
import unittest

from agenty_core.tools.comfyui import _reference_wiring

INFO = {
    "CheckpointLoaderSimple": {"output": ["MODEL", "CLIP", "VAE"]},
    "CLIPTextEncode": {"output": ["CONDITIONING"],
                       "input": {"required": {"clip": ["CLIP"], "text": ["STRING"]}}},
    "KSampler": {"output": ["LATENT"],
                 "input": {"required": {"model": ["MODEL"], "positive": ["CONDITIONING"],
                                        "negative": ["CONDITIONING"],
                                        "latent_image": ["LATENT"]}}},
    "VAEDecode": {"output": ["IMAGE"],
                  "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}},
    "VAEDecodeAudio": {"output": ["AUDIO"],
                       "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}},
    "CreateVideo": {"output": ["VIDEO"],
                    "input": {"required": {"images": ["IMAGE"], "fps": ["FLOAT"]},
                              "optional": {"audio": ["AUDIO"], "codec": ["COMBO"]}}},
    "SaveVideo": {"output": [], "output_node": True,
                  "input": {"required": {"video": ["VIDEO"]}}},
    "ComfySwitchNode": {"output": ["*"],
                        "input": {"optional": {"on_true": ["*"], "on_false": ["*"]}}},
}

# The shape of the graph the real bug came from: audio decoded and wired in.
GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": "a cat"}},
    "3": {"class_type": "KSampler",
          "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
                     "latent_image": ["9", 0]}},
    "4": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
    "5": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["3", 0], "vae": ["1", 2]}},
    "6": {"class_type": "CreateVideo",
          "inputs": {"images": ["4", 0], "fps": 24, "audio": ["5", 0]}},
    "7": {"class_type": "SaveVideo", "inputs": {"video": ["6", 0]}},
    "9": {"class_type": "EmptyLatent", "inputs": {}},
}


class TheEdgeList(unittest.TestCase):

    def test_every_wire_carries_instance_slot_and_input_name(self):
        edges, _load = _reference_wiring(GRAPH, INFO)
        self.assertIn("CheckpointLoaderSimple#0:2 -> VAEDecode#0.vae", edges)
        self.assertIn("CheckpointLoaderSimple#0:1 -> CLIPTextEncode#0.clip", edges)
        self.assertIn("VAEDecodeAudio#0:0 -> CreateVideo#0.audio", edges)
        self.assertIn("CreateVideo#0:0 -> SaveVideo#0.video", edges)
        self.assertEqual(len(edges), 12)   # every wired input in GRAPH, and no more

    def test_one_source_can_feed_two_inputs_of_the_same_type(self):
        # The case a role triple collapses: positive and negative from one encoder.
        edges, _ = _reference_wiring(GRAPH, INFO)
        self.assertIn("CLIPTextEncode#0:0 -> KSampler#0.positive", edges)
        self.assertIn("CLIPTextEncode#0:0 -> KSampler#0.negative", edges)

    def test_instances_are_numbered_in_node_order(self):
        wf = {"7": {"class_type": "CLIPTextEncode", "inputs": {"text": "b"}},
              "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a"}},
              "9": {"class_type": "KSampler",
                    "inputs": {"positive": ["2", 0], "negative": ["7", 0]}}}
        edges, _ = _reference_wiring(wf, INFO)
        self.assertIn("CLIPTextEncode#0:0 -> KSampler#0.positive", edges)
        self.assertIn("CLIPTextEncode#1:0 -> KSampler#0.negative", edges)

    def test_annotations_and_reroutes_are_not_part_of_the_shape(self):
        wf = dict(GRAPH, N={"class_type": "Note", "inputs": {}},
                  R={"class_type": "Reroute", "inputs": {"": ["4", 0]}})
        edges, _ = _reference_wiring(wf, INFO)
        self.assertFalse([e for e in edges if "Note" in e or "Reroute" in e])

    def test_a_wire_from_a_node_that_is_not_there_is_left_out(self):
        wf = {"1": {"class_type": "VAEDecode", "inputs": {"samples": ["99", 0]}},
              "2": {"class_type": "SaveVideo", "inputs": {"video": ["1", 0]}}}
        edges, _ = _reference_wiring(wf, INFO)
        self.assertEqual(edges, ("VAEDecode#0:0 -> SaveVideo#0.video",))

    def test_a_long_graph_is_capped(self):
        wf = {str(i): {"class_type": "VAEDecode", "inputs": {"samples": [str(i - 1), 0]}}
              for i in range(1, 40)}
        wf["0"] = {"class_type": "KSampler", "inputs": {}}
        edges, _ = _reference_wiring(wf, INFO, cap=5)
        self.assertEqual(len(edges), 5)

    def test_nothing_to_say_about_a_non_graph(self):
        self.assertEqual(_reference_wiring(None, INFO), ((), ()))
        self.assertEqual(_reference_wiring({}, INFO), ((), ()))


class TheLoadBearingInputs(unittest.TestCase):

    def test_an_optional_input_a_working_graph_wires_is_named(self):
        _edges, load = _reference_wiring(GRAPH, INFO)
        self.assertEqual(load, ("CreateVideo#0.audio",))

    def test_required_inputs_are_not_named(self):
        # They cannot be forgotten silently — validation already refuses those.
        _edges, load = _reference_wiring(GRAPH, INFO)
        self.assertFalse([x for x in load if ".images" in x or ".samples" in x])

    def test_a_switch_or_math_node_is_not_worth_naming(self):
        wf = {"1": {"class_type": "VAEDecode", "inputs": {}},
              "2": {"class_type": "ComfySwitchNode", "inputs": {"on_true": ["1", 0]}},
              "3": {"class_type": "CreateVideo", "inputs": {"images": ["2", 0], "fps": 24}}}
        _edges, load = _reference_wiring(wf, INFO)
        self.assertEqual(load, ())

    def test_without_schemas_the_edges_still_come_back(self):
        # ComfyUI down: the shape needs no schema, and nothing is claimed about
        # which inputs were optional.
        edges, load = _reference_wiring(GRAPH, {})
        self.assertIn("VAEDecodeAudio#0:0 -> CreateVideo#0.audio", edges)
        self.assertEqual(load, ())


class TheBuildSpec(unittest.TestCase):

    def test_the_recipe_view_carries_the_wiring_keys(self):
        # The leaf view merges _recipe_build_spec's keys; a rename there would
        # silently drop the wiring from every recipe.
        import inspect

        from agenty_core.tools import comfyui
        src = inspect.getsource(comfyui._recipe_build_spec)
        for key in ("reference_member", "reference_wiring", "load_bearing_inputs"):
            self.assertIn(key, src)
        view = inspect.getsource(comfyui._recipe_leaf_view)
        self.assertIn("reference_wiring", view)          # named in how_to_build
        self.assertIn("load_bearing_inputs", view)

    def test_an_empty_spec_still_has_the_keys(self):
        from agenty_core.tools.comfyui import _recipe_build_spec
        spec = _recipe_build_spec([], "Nothing", set())
        self.assertEqual(spec["reference_wiring"], [])
        self.assertEqual(spec["load_bearing_inputs"], [])
        self.assertEqual(spec["reference_member"], "")


if __name__ == "__main__":
    unittest.main()
