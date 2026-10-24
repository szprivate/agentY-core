"""The scene validator (Python) agrees with the shared JS one on every case in
renderer/test/validate_cases.json, and the shipped example scenes are valid."""

import json
import os
import unittest

from agenty_core.motion_graphics import schema

RENDERER = os.path.join(os.path.dirname(schema.__file__), "renderer")
CASES = os.path.join(RENDERER, "test", "validate_cases.json")
EXAMPLES = os.path.join(os.path.dirname(schema.__file__), "examples")


class ValidatorAgreesWithJs(unittest.TestCase):
    def test_cases(self):
        with open(CASES, encoding="utf-8") as fh:
            cases = json.load(fh)
        self.assertGreater(len(cases), 20)
        for c in cases:
            r = schema.validate_scene(c["scene"])
            self.assertEqual(sorted(e["code"] for e in r["errors"]), sorted(c["errors"]), c["name"])
            self.assertEqual(sorted(e["code"] for e in r["warnings"]), sorted(c["warnings"]), c["name"])
            self.assertEqual(r["ok"], not c["errors"], c["name"])

    def test_booleans_are_not_numbers(self):
        scene = {"mgscene": 1, "comp": {"width": 1280, "height": 720, "fps": 30, "duration": 60},
                 "layers": [{"id": "a", "type": "group", "rotation": True}]}
        self.assertEqual([e["code"] for e in schema.validate_scene(scene)["errors"]], ["E_VALUE"])

    def test_load_scene_errors_read_well(self):
        with self.assertRaises(ValueError):
            schema.load_scene(os.path.join(RENDERER, "nope.json"))


class ExampleScenes(unittest.TestCase):
    def test_every_example_is_valid(self):
        names = [n for n in os.listdir(EXAMPLES) if n.endswith(".mgscene.json")]
        self.assertGreaterEqual(len(names), 3)
        for n in names:
            r = schema.validate_scene(schema.load_scene(os.path.join(EXAMPLES, n)))
            self.assertTrue(r["ok"], f"{n}: {r['errors']}")
            self.assertEqual(r["warnings"], [], n)


if __name__ == "__main__":
    unittest.main()
