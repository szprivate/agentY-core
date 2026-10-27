"""A scalar must never end up in a socket that only takes a wire.

The bug this pins: ``video_wan2_2_14B_i2v`` drives ``WanImageToVideo.length``
from a ``ComfyMathExpression``, so ``length`` is wired and drops out of the
positional widget alignment. Everything after it shifted one place — ``batch_size``
took the frame count 81, and the leftover ``1`` landed in ``clip_vision_output``,
a ``CLIP_VISION_OUTPUT`` socket. Nothing rejected it. ComfyUI's own validation
returned ``valid: true``, the graph queued, Flux rendered for six minutes, and
then the sampler reached into that ``1`` for ``.penultimate_hidden_states``.

Two defences, pinned separately because either alone leaves the hole open:

* **alignment** — ``CLIP_VISION_OUTPUT`` is in ``LINK_ONLY_TYPES``, so the socket
  never claims a widget slot in the first place;
* **repair** — ``harden_node_inputs`` deletes such a value if it arrives by some
  other route (a hand-edited graph, a custom template, an LLM patch).

And one guard against over-correcting: a socketless *display* widget (COLOR,
CURVE, IMAGECOMPARE) is not link-only and must keep its value. Deciding "is this
a wire?" by "does some node output this type?" would strip those — COLOR and
CURVE are outputs too — which is why ``LINK_ONLY_TYPES`` stays curated.

Runs under pytest or directly (``python test_link_only_alignment.py``).
"""
import json
import unittest

from agenty_core.tools import comfyui as C
from agenty_core.tools.assembly_deterministic import (
    LINK_ONLY_TYPES,
    harden_node_inputs,
    link_only_scalars,
    strip_link_only_scalars,
)
from agenty_core.tools.comfyui import _map_widget_values, _schema_widget_slots

# The live /object_info shape for WanImageToVideo (ComfyUI 0.3.45).
WAN_I2V_SCHEMA = {
    "required": {
        "positive": ["CONDITIONING", {}],
        "negative": ["CONDITIONING", {}],
        "vae": ["VAE", {}],
        "width": ["INT", {"default": 832, "min": 16, "max": 16384, "step": 16}],
        "height": ["INT", {"default": 480, "min": 16, "max": 16384, "step": 16}],
        "length": ["INT", {"default": 81, "min": 1, "max": 16384, "step": 4}],
        "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
    },
    "optional": {
        "clip_vision_output": ["CLIP_VISION_OUTPUT", {}],
        "start_image": ["IMAGE", {}],
    },
}
# What the template serialises: width, height, length, batch_size.
WAN_I2V_WIDGET_VALUES = [640, 640, 81, 1]
# positive/negative/vae/start_image are wired, and so are width/height/length —
# the template drives them from primitives and a ComfyMathExpression.
WAN_I2V_LINKED = {"positive", "negative", "vae", "start_image",
                  "width", "height", "length"}

# A socketless display widget: not a wire, legitimately carries a value.
PAINTER_SCHEMA = {
    "required": {"bg_color": ["COLOR", {}], "opacity": ["INT", {"default": 100}]},
    "optional": {},
}


class LinkOnlyTypesTest(unittest.TestCase):
    def test_clip_vision_output_is_link_only(self):
        self.assertIn("CLIP_VISION_OUTPUT", LINK_ONLY_TYPES)

    def test_comfyui_shares_the_one_set(self):
        # Alignment and repair reading two different sets is what let the bug
        # through; they are the same object now.
        self.assertIs(C._LINK_ONLY_TYPES, LINK_ONLY_TYPES)

    def test_socketless_display_widgets_are_not_link_only(self):
        for typ in ("COLOR", "CURVE", "IMAGECOMPARE", "BOUNDING_BOXES"):
            self.assertNotIn(typ, LINK_ONLY_TYPES)


class AlignmentTest(unittest.TestCase):
    def test_the_socket_claims_no_widget_slot(self):
        names = [n for n, _ in _schema_widget_slots(WAN_I2V_SCHEMA, set())]
        self.assertNotIn("clip_vision_output", names)
        self.assertEqual(names, ["width", "height", "length", "batch_size"])

    def test_wired_length_no_longer_poisons_the_socket(self):
        # The reading that actually shipped the bug: every wired input excluded,
        # leaving batch_size to soak up the shift. It may still mis-assign
        # batch_size — that is a separate alignment question — but the socket
        # must come out untouched.
        slots = _schema_widget_slots(WAN_I2V_SCHEMA, WAN_I2V_LINKED)
        mapped, _leftover = _map_widget_values(slots, WAN_I2V_WIDGET_VALUES,
                                               WAN_I2V_LINKED)
        self.assertNotIn("clip_vision_output", mapped)

    def test_including_wired_slots_maps_every_value_correctly(self):
        slots = _schema_widget_slots(WAN_I2V_SCHEMA, set())
        mapped, leftover = _map_widget_values(slots, WAN_I2V_WIDGET_VALUES, set())
        self.assertEqual(mapped, {"width": 640, "height": 640,
                                  "length": 81, "batch_size": 1})
        self.assertEqual(leftover, [])


class LinkOnlyScalarsTest(unittest.TestCase):
    def _node(self):
        return {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0],
            "start_image": ["108", 0], "width": 832, "height": 832,
            "length": 81, "batch_size": 1, "clip_vision_output": 1}}

    def _specs(self, schema=None):
        schema = schema or WAN_I2V_SCHEMA
        specs = dict(schema["optional"])
        specs.update(schema["required"])
        return specs

    def test_finds_the_scalar(self):
        self.assertEqual(link_only_scalars(self._node(), self._specs()),
                         [("clip_vision_output", 1)])

    def test_a_real_wire_is_not_a_scalar(self):
        node = self._node()
        node["inputs"]["clip_vision_output"] = ["50", 0]
        self.assertEqual(link_only_scalars(node, self._specs()), [])

    def test_an_absent_socket_is_not_a_scalar(self):
        node = self._node()
        del node["inputs"]["clip_vision_output"]
        self.assertEqual(link_only_scalars(node, self._specs()), [])

    def test_an_explicit_null_is_not_a_scalar(self):
        node = self._node()
        node["inputs"]["clip_vision_output"] = None
        self.assertEqual(link_only_scalars(node, self._specs()), [])

    def test_strip_removes_the_key_and_nothing_else(self):
        node = self._node()
        stripped: list = []
        self.assertEqual(strip_link_only_scalars(node, self._specs(), stripped),
                         ["clip_vision_output"])
        self.assertNotIn("clip_vision_output", node["inputs"])
        self.assertEqual(node["inputs"]["batch_size"], 1)
        self.assertEqual(node["inputs"]["start_image"], ["108", 0])
        self.assertEqual(stripped, ["WanImageToVideo.clip_vision_output = 1"])

    def test_a_display_widget_keeps_its_value(self):
        node = {"class_type": "Painter",
                "inputs": {"bg_color": "#ffffff", "opacity": 100}}
        strip_link_only_scalars(node, self._specs(PAINTER_SCHEMA))
        self.assertEqual(node["inputs"]["bg_color"], "#ffffff")


class HardenTest(unittest.TestCase):
    def test_hardening_strips_and_reports(self):
        node = {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0],
            "start_image": ["108", 0], "width": 832, "height": 832,
            "length": 81, "batch_size": 1, "clip_vision_output": 1}}
        stripped: list = []
        harden_node_inputs(node, WAN_I2V_SCHEMA["required"], [],
                           WAN_I2V_SCHEMA["optional"], stripped)
        self.assertNotIn("clip_vision_output", node["inputs"])
        self.assertEqual(stripped, ["WanImageToVideo.clip_vision_output = 1"])

    def test_hardening_leaves_a_clean_node_alone(self):
        node = {"class_type": "WanImageToVideo", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0],
            "start_image": ["108", 0], "width": 832, "height": 832,
            "length": 81, "batch_size": 1}}
        before = dict(node["inputs"])
        stripped: list = []
        harden_node_inputs(node, WAN_I2V_SCHEMA["required"], [],
                           WAN_I2V_SCHEMA["optional"], stripped)
        self.assertEqual(node["inputs"], before)
        self.assertEqual(stripped, [])

    def test_hardening_still_takes_the_older_four_argument_call(self):
        # Callers that predate the `stripped` out-param must keep working.
        node = {"class_type": "WanImageToVideo", "inputs": {"clip_vision_output": 1}}
        harden_node_inputs(node, {}, [], WAN_I2V_SCHEMA["optional"])
        self.assertNotIn("clip_vision_output", node["inputs"])


class ValidateReportsItTest(unittest.TestCase):
    """validate_workflow does not mutate, so it has to say what is wrong."""

    # Stub producers so the graph is coherent and the only thing under test is
    # the link-only check.
    OBJECT_INFO = {
        "WanImageToVideo": {"input": WAN_I2V_SCHEMA},
        "StubConditioning": {"input": {"required": {}}, "output": ["CONDITIONING"]},
        "StubVAE": {"input": {"required": {}}, "output": ["VAE"]},
    }

    def setUp(self):
        self._orig_oi = C._get_object_info
        self._orig_sv = C._server_validate
        C._get_object_info = lambda: self.OBJECT_INFO
        C._server_validate = lambda _wf: {}

    def tearDown(self):
        C._get_object_info = self._orig_oi
        C._server_validate = self._orig_sv

    def _graph(self, **extra_inputs):
        return {
            "6": {"class_type": "StubConditioning", "inputs": {}},
            "7": {"class_type": "StubConditioning", "inputs": {}},
            "8": {"class_type": "StubVAE", "inputs": {}},
            "5": {"class_type": "WanImageToVideo", "inputs": {
                "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0],
                "width": 832, "height": 832, "length": 81, "batch_size": 1,
                **extra_inputs}},
        }

    def _validate(self, workflow):
        import json
        orig_load = C._load_workflow
        C._load_workflow = lambda _p: workflow
        try:
            fn = getattr(C.validate_workflow, "func", C.validate_workflow)
            return json.loads(fn("ignored.json"))
        finally:
            C._load_workflow = orig_load

    def test_a_scalar_in_a_socket_is_reported_with_its_fix(self):
        out = self._validate(self._graph(clip_vision_output=1))
        self.assertFalse(out["valid"])
        joined = " ".join(out["local_errors"])
        self.assertIn("clip_vision_output", joined)
        self.assertIn("link-only socket", joined)
        self.assertIn("CLIP_VISION_OUTPUT", joined)

    def test_the_same_graph_without_it_is_valid(self):
        out = self._validate(self._graph())
        self.assertEqual(out["local_errors"], [])
        self.assertTrue(out["valid"])


# CreateVideo, the second node the same shift broke — this time landing in a
# plain INT widget, where the link-only rule cannot help. fps is wired, so it
# drops out of the slot list and bit_depth takes the frame rate 16 against a max
# of 10. ComfyUI rejected the whole video branch at submission and reported the
# run as a success.
CREATE_VIDEO_SCHEMA = {
    "required": {"images": ["IMAGE", {}],
                 "fps": ["FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0}]},
    "optional": {"audio": ["AUDIO", {}],
                 "bit_depth": ["INT", {"default": 8, "min": 8, "max": 10, "step": 2}]},
}
CREATE_VIDEO_WIDGET_VALUES = [16, 8]        # fps, bit_depth
CREATE_VIDEO_LINKED = {"images", "fps"}     # fps driven by a primitive


class RangeAwareScoringTest(unittest.TestCase):
    """Type alone cannot tell a shifted reading from an aligned one.

    Both put a number in a numeric slot, so both scored the same and the
    tie-break took the shifted one. The declared range breaks the tie for free.
    """

    def test_a_value_past_its_max_does_not_fit(self):
        self.assertFalse(C._value_fits(["INT", {"min": 8, "max": 10}], 16))

    def test_a_value_below_its_min_does_not_fit(self):
        self.assertFalse(C._value_fits(["INT", {"min": 1, "max": 64}], 0))

    def test_a_value_inside_its_range_fits(self):
        self.assertTrue(C._value_fits(["INT", {"min": 8, "max": 10}], 8))

    def test_a_slot_with_no_declared_range_still_fits_any_number(self):
        self.assertTrue(C._value_fits(["INT", {}], 999999))

    def test_a_link_is_always_acceptable(self):
        self.assertTrue(C._value_fits(["INT", {"min": 8, "max": 10}], ["12", 0]))

    def test_the_shifted_reading_now_loses(self):
        excl = _schema_widget_slots(CREATE_VIDEO_SCHEMA, CREATE_VIDEO_LINKED)
        incl = _schema_widget_slots(CREATE_VIDEO_SCHEMA, set())
        shifted, _ = _map_widget_values(excl, CREATE_VIDEO_WIDGET_VALUES,
                                        CREATE_VIDEO_LINKED)
        aligned, _ = _map_widget_values(incl, CREATE_VIDEO_WIDGET_VALUES,
                                        CREATE_VIDEO_LINKED)
        self.assertEqual(shifted, {"bit_depth": 16})    # what used to win
        self.assertEqual(aligned, {"bit_depth": 8})     # what is actually meant
        self.assertGreater(C._mapping_score(aligned, incl),
                           C._mapping_score(shifted, excl))


class WiredWidgetAlignmentTest(unittest.TestCase):
    """End to end on the template that broke twice."""

    def setUp(self):
        # The conversion reads node schemas from /object_info. Recorded schemas of the classes under test
        # keep this off the live ComfyUI (down, or a different version): the result must not depend on it.
        from unittest import mock
        fixture = {"WanImageToVideo": {"input": WAN_I2V_SCHEMA}, "CreateVideo": {"input": CREATE_VIDEO_SCHEMA}}
        self.enterContext(mock.patch.object(C, "_get_object_info", lambda: fixture))

    def _convert(self, name="video_wan2_2_14B_i2v"):
        import json as _json
        import pathlib
        root = pathlib.Path(C.__file__).resolve().parents[2]
        path = root / "comfyui_workflow_templates_official" / "templates" / f"{name}.json"
        if not path.is_file():
            self.skipTest(f"corpus template {name} not present")
        return C._convert_graph_to_api(_json.loads(path.read_text(encoding="utf-8")))

    def _node(self, api, class_type):
        for node in api.values():
            if node.get("class_type") == class_type:
                return node
        self.fail(f"{class_type} not in the converted graph")

    def test_create_video_gets_the_bit_depth_not_the_frame_rate(self):
        node = self._node(self._convert(), "CreateVideo")
        self.assertEqual(node["inputs"]["bit_depth"], 8)
        # Every value landed somewhere, so there is nothing left over to park.
        self.assertEqual((node.get("_meta") or {}).get("unmapped_widgets"), None)

    def test_wan_image_to_video_gets_the_batch_size_not_the_frame_count(self):
        node = self._node(self._convert(), "WanImageToVideo")
        self.assertEqual(node["inputs"]["batch_size"], 1)
        self.assertNotIn("clip_vision_output", node["inputs"])


class UpdateWorkflowPersistsTheStrip(unittest.TestCase):
    """Saying "stripped" while the file still carries it would be worse than silence.

    ``update_workflow`` saves *before* its hardening pass, so the removal has to
    be written back or the next submission still carries the value that crashes
    the sampler minutes into a render.
    """

    OBJECT_INFO = {"WanImageToVideo": {"input": WAN_I2V_SCHEMA},
                   "StubConditioning": {"input": {"required": {}}, "output": ["CONDITIONING"]},
                   "StubVAE": {"input": {"required": {}}, "output": ["VAE"]}}

    def setUp(self):
        self.saved: list = []
        self._orig = {n: getattr(C, n) for n in
                      ("_get_object_info", "_server_validate", "_save_workflow",
                       "_load_workflow")}
        self.graph = {
            "6": {"class_type": "StubConditioning", "inputs": {}},
            "7": {"class_type": "StubConditioning", "inputs": {}},
            "8": {"class_type": "StubVAE", "inputs": {}},
            "5": {"class_type": "WanImageToVideo", "inputs": {
                "positive": ["6", 0], "negative": ["7", 0], "vae": ["8", 0],
                "width": 832, "height": 832, "length": 81, "batch_size": 1}},
        }
        C._get_object_info = lambda: self.OBJECT_INFO
        C._server_validate = lambda _wf: {}
        C._load_workflow = lambda _p: self.graph
        C._save_workflow = lambda wf, name=None: (
            self.saved.append(json.loads(json.dumps(wf))) or "saved.json")

    def tearDown(self):
        for name, fn in self._orig.items():
            setattr(C, name, fn)

    def _update(self):
        fn = getattr(C.update_workflow, "func", C.update_workflow)
        return json.loads(fn(workflow_path="whatever.json"))

    def test_the_cleaned_graph_is_written_back(self):
        self.graph["5"]["inputs"]["clip_vision_output"] = 1
        out = self._update()
        self.assertIn("stripped WanImageToVideo.clip_vision_output = 1 (link-only socket)",
                      out["applied_patches"])
        # Saved twice: once before hardening, once after — and the LAST write is
        # the one the executor reads.
        self.assertNotIn("clip_vision_output", self.saved[-1]["5"]["inputs"])
        self.assertEqual(self.saved[-1]["5"]["inputs"]["batch_size"], 1)
        self.assertEqual(self.saved[-1]["5"]["inputs"]["start_image"]
                         if "start_image" in self.saved[-1]["5"]["inputs"] else None, None)

    def test_a_clean_graph_is_not_re_saved(self):
        out = self._update()
        self.assertEqual(out["applied_patches"], [])
        self.assertEqual(len(self.saved), 1)


if __name__ == "__main__":
    unittest.main()
