"""find_app_installs: reading installs from an environment and a fake folder tree (never the real machine)."""
import os
import tempfile
import unittest
from pathlib import Path

from agenty_core.tools import app_installs as ai


def touch(p: Path) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return str(p)


class Finder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_newest_first_and_version_from_folder_name(self):
        touch(self.tmp / "Blender 3.5" / "blender.exe")
        touch(self.tmp / "Blender 5.2" / "blender.exe")
        got = ai.find_installs("blender", system=False, extra_globs=[str(self.tmp / "Blender */blender.exe")])
        self.assertEqual([r["version"] for r in got], ["5.2", "3.5"])

    def test_pick_applies_min_version_and_reports_the_others(self):
        touch(self.tmp / "Blender 3.5" / "blender.exe")
        touch(self.tmp / "Blender 5.2" / "blender.exe")
        got = ai.find_installs("blender", system=False, extra_globs=[str(self.tmp / "Blender */blender.exe")])
        chosen, others, _ = ai.pick(got, "5.1")
        self.assertEqual(chosen["version"], "5.2")
        self.assertEqual([o["version"] for o in others], ["3.5"])
        chosen, others, why = ai.pick(got, "6.0")
        self.assertIsNone(chosen)
        self.assertIn("6.0", why)
        self.assertEqual(ai.pick([], "1")[2], "not installed")

    def test_env_variable_names_the_executable(self):
        exe = touch(self.tmp / "blender-5.1.2-windows-x64" / "blender.exe")
        got = ai.find_installs("blender", env={"BLENDER_PATH": exe}, system=False)
        self.assertEqual(got[0]["version"], "5.1.2")
        self.assertEqual(got[0]["source"], "env:BLENDER_PATH")

    def test_nuke_folder_version_and_year_apps(self):
        touch(self.tmp / "Nuke15.0v4" / "Nuke15.0.exe")
        self.assertEqual(ai.find_installs("nuke", system=False, extra_globs=[str(self.tmp / "Nuke*" / "Nuke*.exe")])[0]["version"], "15.0.4")
        touch(self.tmp / "Adobe After Effects 2025" / "Support Files" / "AfterFX.exe")
        got = ai.find_installs("after-effects", system=False,
                               extra_globs=[str(self.tmp / "Adobe After Effects */Support Files/AfterFX.exe")])
        self.assertEqual(got[0]["version"], "2025")

    def test_aliases_and_unknown(self):
        self.assertEqual(ai.canonical("AE"), "after-effects")
        self.assertEqual(ai.canonical("Resolve"), "davinci-resolve")
        with self.assertRaises(KeyError):
            ai.find_installs("gimp")
        self.assertIn("error", ai.find_app_installs(["gimp"])["gimp"])

    def test_version_tuple(self):
        self.assertEqual(ai.version_tuple("6000.0.23f1"), (6000, 0, 23))
        self.assertEqual(ai.version_tuple(""), ())


if __name__ == "__main__":
    unittest.main()
