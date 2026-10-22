"""run_script(code=...) writes + runs a snippet in one call, vetted like a command."""
import glob
import json
import os
import tempfile
import unittest

from agenty_core.tools.shell import run_script


def _snips():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "agenty_snippet_*.py")))


class RunScriptCode(unittest.TestCase):
    def test_code_runs_and_prints(self):
        out = json.loads(run_script(code="print('hi', 2+3)"))
        self.assertEqual((out["exit_code"], out["stdout"]), (0, "hi 5"))

    def test_code_gets_args(self):
        out = json.loads(run_script(code="import sys; print(sys.argv[1:])", args="a b"))
        self.assertEqual(out["stdout"], "['a', 'b']")

    def test_command_and_code_are_exclusive(self):
        out = json.loads(run_script(command="python -V", code="print(1)"))
        self.assertTrue(out.get("refused"))

    def test_args_outside_roots_refused(self):
        bad = "C:/Windows/win.ini" if os.name == "nt" else "/etc/passwd"
        out = json.loads(run_script(code="print(1)", args=bad))
        self.assertTrue(out.get("refused"))

    def test_temp_file_removed(self):
        before = _snips()
        run_script(code="print(1)")
        self.assertLessEqual(_snips(), before)

    def test_plain_command_unchanged(self):
        out = json.loads(run_script(command="python -c \"print(7)\""))
        self.assertEqual(out["stdout"], "7")


if __name__ == "__main__":
    unittest.main()
