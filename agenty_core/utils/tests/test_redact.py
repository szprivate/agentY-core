"""Secrets are masked in text that goes back to a model (agenty_core.utils.redact)."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils import redact as R


class Redact(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / ".env").write_text(
            "DASHSCOPE_API_KEY=plainvalue1234567890\nCOMFYUI_URL=http://127.0.0.1:8188\n"
            "PORT=5000\n", encoding="utf-8")
        (root / ".agenty_token").write_text("tok_ABCDEFGHIJKLMNOPQRST\n", encoding="utf-8")
        p = mock.patch("agenty_core.paths.project_root", return_value=root)
        p.start()
        self.addCleanup(p.stop)
        R._cache["key"] = None

    def test_configured_values_are_masked_exactly(self):
        out = R.redact("key is plainvalue1234567890, url http://127.0.0.1:8188 port 5000")
        self.assertNotIn("plainvalue1234567890", out)
        self.assertIn("plai…90", out)
        self.assertIn("http://127.0.0.1:8188", out)      # not a secret: kept readable
        self.assertIn("5000", out)

    def test_session_token_is_masked(self):
        self.assertNotIn("tok_ABCDEFGHIJKLMNOPQRST", R.redact("x tok_ABCDEFGHIJKLMNOPQRST y"))

    def test_token_shapes_are_masked(self):
        for secret in ("sk-ant-api03-" + "a" * 40, "sk-proj-" + "b" * 40, "hf_" + "C" * 34,
                       "ghp_" + "d" * 36, "xoxb-1234567890-abcdefghij", "AIza" + "e" * 35):
            out = R.redact(f"value: {secret} end")
            self.assertNotIn(secret, out, secret)
        out = R.redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123")
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz0123", out)

    def test_ordinary_text_is_untouched(self):
        text = "Rendered 4 images to D:/out/ComfyUI_00001_.png with max_tokens: 4096"
        self.assertEqual(R.redact(text), text)

    def test_structures(self):
        out = R.redact_obj({"a": ["hf_" + "Z" * 30, 3], "b": None})
        self.assertNotIn("Z" * 30, str(out))
        self.assertEqual(out["a"][1], 3)


if __name__ == "__main__":
    unittest.main()
