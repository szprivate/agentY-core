"""duplicate_workflow(count=N): N fresh-seeded copies in ONE call (one round trip)."""
import json
import tempfile
import unittest
from pathlib import Path

from agenty_core.tools import comfyui as C


def _dup(*a, **k):
    return json.loads(C.duplicate_workflow(*a, **k))


class DuplicateCount(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.addCleanup(self._d.cleanup)
        self.base = Path(self._d.name) / "wf.json"
        self.base.write_text(json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 1}}}))

    def test_default_is_unchanged(self):
        out = _dup(str(self.base))
        self.assertIn("new_path", out)
        self.assertNotIn("new_paths", out)

    def test_count_makes_all_copies_in_one_call(self):
        out = _dup(str(self.base), count=3)
        self.assertEqual(out["count"], 3)
        self.assertEqual(len(set(out["new_paths"])), 3)
        self.assertEqual(out["new_path"], out["new_paths"][0])
        for p in out["new_paths"]:
            self.assertTrue(Path(p).exists())

    def test_missing_source_is_an_error(self):
        self.assertIn("error", _dup(str(self.base) + ".nope", count=3))

    def test_count_is_clamped(self):
        self.assertEqual(_dup(str(self.base), count=0).get("new_paths"), None)
        self.assertEqual(_dup(str(self.base), count="junk").get("count"), None)


if __name__ == "__main__":
    unittest.main()
