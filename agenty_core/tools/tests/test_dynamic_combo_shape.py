"""A graph the validator called valid and ComfyUI could not run.

The failure, five times in one run, from ComfyUI's own /history:

    ByteDanceSeedreamNodeV3.execute() missing 1 required positional argument: 'model'

The agent wrote the node's settings as an object::

    "model": {"model": "seedream 5.0 pro", "size_preset": "…", "width": 1312, …}

The API format spells them as dotted siblings — `model` beside
`model.size_preset` — and ComfyUI binds by those names, so an object binds
nothing. `update_workflow` said `"status": "ok"`, because local validation checks
class names, missing required inputs and link targets and never the SHAPE of a
value. The agent read "valid", submitted, and met a TypeError naming neither the
input nor the shape it wanted — so it tried to repair a graph that was never
structurally wrong, over and over, until the user stopped it.

Three things were missing and all three are mechanical:

* the object is unpacked into the dotted keys ComfyUI binds;
* the widgets an option contributes have specs, which nothing had hoisted — so a
  required `model.prompt_optimization` was never defaulted and a `model.height`
  of 736 was never measured against its minimum of 1024;
* a `width`/`height` pair sitting beside a chosen size preset is what made that
  minimum bite. The node says "Select Custom to use the width and height below",
  so unless the preset is Custom they are noise — and clamping them instead would
  have produced 1312x1024, a shape nobody asked for.

    python -m unittest discover -s agenty_core/tools/tests
"""
import unittest

from agenty_core.tools.assembly_deterministic import (drop_overridden_dimensions,
                                                      dynamic_sub_specs,
                                                      flatten_dynamic_combos,
                                                      harden_node_inputs)

PRESETS = ["1024x1024 (1:1)", "1312x736 (16:9)", "Custom"]
REQUIRED = {
    "prompt": ["STRING", {}],
    "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [
        {"key": "pro", "inputs": {"required": {
            "size_preset": ["COMBO", {"options": PRESETS}],
            "width": ["INT", {"default": 2048, "min": 1024, "max": 4096}],
            "height": ["INT", {"default": 2048, "min": 1024, "max": 4096}],
            "prompt_optimization": ["COMBO", {"options": ["standard", "fast"]}],
        }}},
        {"key": "lite", "inputs": {"required": {
            "size_preset": ["COMBO", {"options": ["2048x2048 (1:1)"]}]}}},
    ]}],
}
OPTIONAL = {"seed": ["INT", {"default": 0}]}


def node(inputs):
    return {"class_type": "N", "inputs": dict(inputs)}


class TheObjectIsUnpacked(unittest.TestCase):

    def test_the_option_key_becomes_the_value_and_widgets_go_dotted(self):
        n = node({"model": {"model": "pro", "size_preset": "1312x736 (16:9)"}})
        errors = flatten_dynamic_combos(n, REQUIRED, OPTIONAL)
        self.assertEqual(errors, [])
        self.assertEqual(n["inputs"]["model"], "pro")
        self.assertEqual(n["inputs"]["model.size_preset"], "1312x736 (16:9)")

    def test_a_top_level_input_swept_into_the_object_is_hoisted_back(self):
        n = node({"model": {"model": "pro", "seed": 7}})
        flatten_dynamic_combos(n, REQUIRED, OPTIONAL)
        self.assertEqual(n["inputs"]["seed"], 7)
        self.assertNotIn("model.seed", n["inputs"])

    def test_a_widget_the_option_does_not_offer_is_dropped_and_named(self):
        notes = []
        n = node({"model": {"model": "lite", "width": 100}})
        flatten_dynamic_combos(n, REQUIRED, OPTIONAL, notes)
        self.assertNotIn("model.width", n["inputs"])
        self.assertIn("dropped width", " ".join(notes))

    def test_an_object_naming_no_option_is_an_error_not_a_guess(self):
        n = node({"model": {"model": "turbo"}})
        errors = flatten_dynamic_combos(n, REQUIRED, OPTIONAL)
        self.assertEqual(len(errors), 1)
        self.assertIn("'pro'", errors[0])
        # Left alone: picking one would run a model nobody asked for.
        self.assertIsInstance(n["inputs"]["model"], dict)

    def test_a_plain_value_is_untouched(self):
        n = node({"model": "pro", "model.size_preset": "1024x1024 (1:1)"})
        before = dict(n["inputs"])
        self.assertEqual(flatten_dynamic_combos(n, REQUIRED, OPTIONAL), [])
        self.assertEqual(n["inputs"], before)


class TheOptionsWidgetsHaveSpecs(unittest.TestCase):

    def test_they_are_hoisted_under_dotted_names(self):
        req, _opt = dynamic_sub_specs(node({"model": "pro"}), REQUIRED, OPTIONAL)
        self.assertIn("model.size_preset", req)
        self.assertIn("model.prompt_optimization", req)

    def test_only_for_the_option_actually_selected(self):
        req, _opt = dynamic_sub_specs(node({"model": "lite"}), REQUIRED, OPTIONAL)
        self.assertIn("model.size_preset", req)
        self.assertNotIn("model.width", req)

    def test_a_missing_required_widget_is_defaulted(self):
        n = node({"model": "pro", "model.size_preset": "Custom"})
        harden_node_inputs(n, REQUIRED, None, OPTIONAL)
        self.assertEqual(n["inputs"]["model.prompt_optimization"], "standard")

    def test_an_out_of_range_widget_is_brought_into_range(self):
        # 736 against a minimum of 1024 — the value that failed validation.
        n = node({"model": "pro", "model.size_preset": "Custom",
                  "model.height": 736})
        harden_node_inputs(n, REQUIRED, None, OPTIONAL)
        self.assertGreaterEqual(n["inputs"]["model.height"], 1024)


class APresetDecidesTheSize(unittest.TestCase):

    def test_width_and_height_beside_a_preset_are_dropped(self):
        n = node({"model": "pro", "model.size_preset": "1312x736 (16:9)",
                  "model.width": 1312, "model.height": 736})
        dropped = []
        drop_overridden_dimensions(n, REQUIRED, OPTIONAL, dropped)
        self.assertNotIn("model.width", n["inputs"])
        self.assertNotIn("model.height", n["inputs"])
        self.assertEqual(len(dropped), 2)

    def test_custom_keeps_them_because_custom_means_use_them(self):
        n = node({"model": "pro", "model.size_preset": "Custom",
                  "model.width": 1312, "model.height": 736})
        drop_overridden_dimensions(n, REQUIRED, OPTIONAL)
        self.assertEqual(n["inputs"]["model.width"], 1312)

    def test_hardening_drops_them_rather_than_clamping_them(self):
        """The distinction that matters, and the one the bug turned on.

        Clamping 1312x736 to the 1024 minimum yields 1312x1024 — valid, and not
        16:9. Dropping them leaves the preset in charge and the re-injected
        defaults (2048) merely present and ignored. So the defaults are the
        proof: a clamp could never produce them.
        """
        n = node({"model": "pro", "model.size_preset": "1312x736 (16:9)",
                  "model.width": 1312, "model.height": 736})
        harden_node_inputs(n, REQUIRED, None, OPTIONAL)
        self.assertEqual(n["inputs"]["model.width"], 2048)
        self.assertEqual(n["inputs"]["model.height"], 2048)
        self.assertEqual(n["inputs"]["model.size_preset"], "1312x736 (16:9)")

    def test_a_menu_with_no_custom_option_is_left_alone(self):
        # The convention is the literal "Custom"; without it, no such rule.
        n = node({"model": "lite", "model.size_preset": "2048x2048 (1:1)",
                  "model.width": 100})
        drop_overridden_dimensions(n, REQUIRED, OPTIONAL)
        self.assertEqual(n["inputs"]["model.width"], 100)


class TheWholeRepairTogether(unittest.TestCase):
    """The reported node, start to finish."""

    def test_an_object_becomes_a_graph_that_binds(self):
        n = node({"prompt": "x", "model": {
            "model": "pro", "size_preset": "1312x736 (16:9)",
            "width": 1312, "height": 736, "images": {}, "seed": 47629,
            "watermark": False}})
        notes = []
        harden_node_inputs(n, REQUIRED, None, OPTIONAL, None, notes)
        got = n["inputs"]
        self.assertEqual(got["model"], "pro")
        self.assertEqual(got["model.size_preset"], "1312x736 (16:9)")
        self.assertEqual(got["seed"], 47629)          # hoisted, not buried
        self.assertEqual(got["model.prompt_optimization"], "standard")
        self.assertGreaterEqual(got["model.width"], 1024)   # in range, ignored
        self.assertNotIn("images", got)               # the empty placeholder
        self.assertTrue(notes)

    def test_nothing_is_reported_when_nothing_needed_doing(self):
        n = node({"prompt": "x", "model": "pro",
                  "model.size_preset": "Custom", "model.width": 2048,
                  "model.height": 2048, "model.prompt_optimization": "fast"})
        notes = []
        harden_node_inputs(n, REQUIRED, None, OPTIONAL, None, notes)
        self.assertEqual(notes, [])


class TheOptionsOwnInputWins(unittest.TestCase):
    """SaveVideo declares `codec` twice: top-level, and inside its format option.

    Written as `{"format": "auto", "codec": "auto"}`, the codec went to the
    top-level input and `format.codec` - which the option requires - was reported
    missing. The agent spent three update_workflow calls finding the dotted form.
    """

    SAVE_REQUIRED = {
        "filename_prefix": ["STRING", {}],
        "format": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "auto", "inputs": {"required": {
                "codec": ["COMBO", {"options": ["auto", "h264"]}]}}}]}],
    }
    SAVE_OPTIONAL = {"codec": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto"}]}]}

    def test_a_name_the_option_declares_goes_under_the_option(self):
        n = node({"format": {"format": "auto", "codec": "auto"}})
        self.assertEqual(flatten_dynamic_combos(n, self.SAVE_REQUIRED, self.SAVE_OPTIONAL), [])
        self.assertEqual(n["inputs"], {"format": "auto", "format.codec": "auto"})

    def test_hardening_then_has_nothing_missing(self):
        n = node({"filename_prefix": "v", "format": {"format": "auto", "codec": "auto"}})
        self.assertEqual(harden_node_inputs(n, self.SAVE_REQUIRED, None, self.SAVE_OPTIONAL), [])



class AMissingDynamicComboIsDefaulted(unittest.TestCase):
    """A SaveVideo with no `format` passed ComfyUI's validation, rendered the
    whole video and failed at the save: "execute() missing ... 'format'". Three
    benchmark cases lost a first attempt (and a full render) to it."""

    CODEC = ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto", "inputs": {"required": {}}},
                                                   {"key": "h264", "inputs": {"required": {}}}]}]
    REQUIRED = {
        "video": ["VIDEO", {}],
        "filename_prefix": ["STRING", {"default": "video/ComfyUI"}],
        "format": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "auto", "inputs": {"required": {"codec": CODEC}}},
            {"key": "mp4", "inputs": {"required": {"codec": CODEC}}}]}],
    }

    def test_format_and_the_codec_it_requires_are_filled_in(self):
        n = node({"video": ["2", 0], "filename_prefix": "v"})
        notes = []
        self.assertEqual(harden_node_inputs(n, self.REQUIRED, None, {}, None, notes), [])
        self.assertEqual(n["inputs"]["format"], "auto")
        self.assertEqual(n["inputs"]["format.codec"], "auto")
        self.assertTrue(any("format" in x for x in notes))

    def test_a_chosen_format_is_kept_and_only_its_codec_added(self):
        n = node({"video": ["2", 0], "filename_prefix": "v", "format": "mp4"})
        self.assertEqual(harden_node_inputs(n, self.REQUIRED, None, {}), [])
        self.assertEqual((n["inputs"]["format"], n["inputs"]["format.codec"]), ("mp4", "auto"))

    def test_the_synthesized_save_node_has_them_too(self):
        from agenty_core.tools.assembly_deterministic import ensure_output_node
        oi = {"CreateVideo": {"output": ["VIDEO"], "output_node": False},
              "SaveVideo": {"output": [], "output_node": True, "input": {"required": self.REQUIRED}}}
        wf = {"1": {"class_type": "CreateVideo", "inputs": {}}}
        new = ensure_output_node(wf, oi)
        self.assertEqual(wf[new]["inputs"]["format"], "auto")
        self.assertEqual(wf[new]["inputs"]["format.codec"], "auto")



class AnInputOneLetterOff(unittest.TestCase):
    """`SaveImage.image` beside `images` passed validation and crashed the node:
    "save_images() got an unexpected keyword argument 'image'"."""

    REQUIRED = {"images": ["IMAGE", {}], "filename_prefix": ["STRING", {"default": "ComfyUI"}]}

    def test_the_stray_one_is_dropped_when_the_real_one_is_set(self):
        n = node({"images": ["8", 0], "image": ["8", 0], "filename_prefix": "x"})
        notes = []
        self.assertEqual(harden_node_inputs(n, self.REQUIRED, None, {}, None, notes), [])
        self.assertEqual(n["inputs"], {"images": ["8", 0], "filename_prefix": "x"})
        self.assertTrue(any("'image'" in x for x in notes))

    def test_it_takes_the_real_name_when_that_is_missing(self):
        n = node({"image": ["8", 0], "filename_prefix": "x"})
        self.assertEqual(harden_node_inputs(n, self.REQUIRED, None, {}), [])
        self.assertEqual(n["inputs"]["images"], ["8", 0])
        self.assertNotIn("image", n["inputs"])

    def test_other_undeclared_inputs_are_left_alone(self):
        n = node({"images": ["8", 0], "filename_prefix": "x", "lora_3": "a.safetensors", "format.codec": "auto"})
        harden_node_inputs(n, self.REQUIRED, None, {})
        self.assertIn("lora_3", n["inputs"])
        self.assertIn("format.codec", n["inputs"])


class ALegacyPrimitiveNodeIsInlined(unittest.TestCase):
    """A template's frontend-only "Song Duration = 120" PrimitiveNode stayed in
    the API graph: invalid, replaced by a repair agent on every build, and the
    length the user asked for was never applied to the inputs it fed."""

    def test_its_value_lands_on_every_input_it_fed(self):
        from agenty_core.tools.comfyui import _inline_primitive_nodes
        wf = {"99": {"class_type": "PrimitiveNode", "inputs": {"__widgets_values": [120, "fixed"]}},
              "98": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": ["99", 0], "batch_size": 1}},
              "94": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"duration": ["99", 0], "seed": ["109", 0]}}}
        self.assertEqual(_inline_primitive_nodes(wf), ["99"])
        self.assertNotIn("99", wf)
        self.assertEqual(wf["98"]["inputs"]["seconds"], 120)
        self.assertEqual(wf["94"]["inputs"], {"duration": 120, "seed": ["109", 0]})


if __name__ == "__main__":
    unittest.main()
