"""A brainbriefing's prompt goes into the primitive that feeds the encoder, not over the link."""
import unittest

from agenty_core.tools.comfyui import _write_prompt_text


def _graph():
    return {
        "5508": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "template prompt"},
                 "_meta": {"title": "Prompt (positive)"}},
        "2483": {"class_type": "CLIPTextEncode", "inputs": {"text": ["5508", 0], "clip": ["1", 0]}},
        "5504": {"class_type": "TextEncodeGemma", "inputs": {"text": ["5508", 0]}},
        "9": {"class_type": "StringConcatenate", "inputs": {"string_a": "a", "string_b": ["5508", 0]}},
        "2612": {"class_type": "CLIPTextEncode", "inputs": {"text": ["9", 0], "clip": ["1", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "plain", "clip": ["1", 0]}},
    }


class PromptPrimitiveTests(unittest.TestCase):
    def test_a_wired_prompt_goes_into_its_primitive(self):
        g = _graph()
        where = _write_prompt_text(g, "2483", "text", "a car racing")
        self.assertEqual(g["5508"]["inputs"]["value"], "a car racing")
        self.assertEqual(g["2483"]["inputs"]["text"], ["5508", 0])   # the link stays
        self.assertEqual(g["5504"]["inputs"]["text"], ["5508", 0])   # so the other encoder reads it too
        self.assertIn("5508.inputs.value", where)

    def test_a_text_that_is_made_from_others_is_replaced(self):
        g = _graph()
        self.assertEqual(_write_prompt_text(g, "2612", "text", "no rack"), "2612.inputs.text")
        self.assertEqual(g["2612"]["inputs"]["text"], "no rack")

    def test_a_literal_text_is_written_as_before(self):
        g = _graph()
        self.assertEqual(_write_prompt_text(g, "7", "text", "x"), "7.inputs.text")
        self.assertEqual(g["7"]["inputs"]["text"], "x")


if __name__ == "__main__":
    unittest.main()
