"""find_app_installs: reading installs from an environment and a fake folder tree (never the real machine)."""
import os
import plistlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


# What Blender's Info.plist looks like (trimmed to the keys that matter), as shipped in the .dmg.
BLENDER_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Blender</string>
  <key>CFBundleShortVersionString</key><string>5.1.0</string>
  <key>CFBundleVersion</key><string>5.1.0</string>
</dict></plist>
"""


def mac_bundle(root: Path, plist: bytes | str | None = BLENDER_PLIST, name="Blender") -> str:
    """A fake ``Blender.app`` under *root*; returns the executable's path."""
    exe = touch(root / f"{name}.app" / "Contents" / "MacOS" / "Blender")
    if plist is not None:
        data = plist.encode("utf-8") if isinstance(plist, str) else plist
        (root / f"{name}.app" / "Contents" / "Info.plist").write_bytes(data)
    return exe


class MacVersionWithoutStarting(unittest.TestCase):
    """GitHub issue 78 A: a Blender that was never started reported "found ?, but 5.1 or newer is needed"."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # Starting the app fails, as it does on a never-opened Mac (Gatekeeper); the mock records any attempt.
        p = mock.patch.object(subprocess, "run", side_effect=OSError("blocked"))
        self.started = p.start()
        self.addCleanup(p.stop)

    def finds(self, exe):
        return ai.find_installs("blender", system=False, extra_globs=[exe])

    def test_the_version_comes_from_the_bundles_info_plist(self):
        exe = mac_bundle(self.tmp)
        self.assertEqual(self.finds(exe)[0]["version"], "5.1.0")
        self.started.assert_not_called()                    # read from the file; the app was not started

    def test_a_binary_plist_reads_too(self):
        exe = mac_bundle(self.tmp, plistlib.dumps({"CFBundleShortVersionString": "5.2.1"}, fmt=plistlib.FMT_BINARY))
        self.assertEqual(self.finds(exe)[0]["version"], "5.2.1")

    def test_the_short_string_wins_and_the_bundle_version_is_the_fallback(self):
        self.assertEqual(ai.plist_version(str(self._plist({"CFBundleShortVersionString": "5.1.0", "CFBundleVersion": "99"}))), "5.1.0")
        self.assertEqual(ai.plist_version(str(self._plist({"CFBundleVersion": "4.5.3"}))), "4.5.3")
        self.assertEqual(ai.plist_version(str(self._plist({"CFBundleShortVersionString": "n/a"}))), "")

    def _plist(self, data):
        f = self.tmp / "p.plist"
        f.write_bytes(plistlib.dumps(data))
        return f

    def test_a_pre_release_string_is_kept_as_it_is(self):
        exe = mac_bundle(self.tmp, plistlib.dumps({"CFBundleShortVersionString": "5.2.0 Beta"}))
        self.assertEqual(ai.version_tuple(self.finds(exe)[0]["version"]), (5, 2, 0))

    def test_a_folder_name_with_the_version_still_goes_first(self):
        exe = mac_bundle(self.tmp / "Blender 4.2", plistlib.dumps({"CFBundleShortVersionString": "9.9"}))
        self.assertEqual(self.finds(exe)[0]["version"], "4.2")

    def test_broken_or_missing_plist_is_unknown_not_a_crash(self):
        self.assertEqual(self.finds(mac_bundle(self.tmp / "a", "this is not a plist"))[0]["version"], "")
        self.assertEqual(self.finds(mac_bundle(self.tmp / "b", None))[0]["version"], "")
        self.assertEqual(ai.plist_version(str(self.tmp / "nowhere.plist")), "")

    def test_an_unknown_version_is_not_too_old(self):
        exe = mac_bundle(self.tmp, None)                    # never started, no readable plist
        got = self.finds(exe)
        self.assertEqual(got[0]["version"], "")
        chosen, others, why = ai.pick(got, "5.1")
        self.assertIs(chosen, got[0])                       # offered, not refused
        self.assertIn("unknown", why)
        self.assertNotIn("but version", why)

    def test_a_known_install_that_meets_the_minimum_beats_an_unknown_one(self):
        unknown = mac_bundle(self.tmp / "u", None)
        known = mac_bundle(self.tmp / "k", plistlib.dumps({"CFBundleShortVersionString": "5.2.0"}))
        got = ai.find_installs("blender", system=False, extra_globs=[unknown, known])
        chosen, others, _ = ai.pick(got, "5.1")
        self.assertEqual(chosen["version"], "5.2.0")
        self.assertEqual([o["version"] for o in others], [""])

    def test_a_version_that_is_known_and_too_old_is_still_too_old(self):
        exe = mac_bundle(self.tmp, plistlib.dumps({"CFBundleShortVersionString": "4.2.1"}))
        chosen, _, why = ai.pick(self.finds(exe), "5.1")
        self.assertIsNone(chosen)
        self.assertIn("4.2.1", why)
        self.assertIn("5.1 or newer", why)

    def test_the_app_is_asked_for_its_version_only_as_a_last_resort(self):
        exe = mac_bundle(self.tmp, None)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(stdout="Blender 5.1.0\n\tbuild date: x")) as run:
            self.assertEqual(self.finds(exe)[0]["version"], "5.1.0")
            self.assertEqual(run.call_args[0][0][1], "--version")
        with mock.patch.object(subprocess, "run", side_effect=OSError("blocked by Gatekeeper")):
            self.assertEqual(self.finds(exe)[0]["version"], "")

    def test_the_usual_mac_locations_are_searched_for_blender(self):
        globs = " ".join(ai._APPS["blender"]["globs"])
        self.assertIn("/Applications/Blender.app/Contents/MacOS/Blender", globs)
        self.assertIn("/Applications/Blender*.app/Contents/MacOS/Blender", globs)
        self.assertIn("~/Applications/Blender*.app/Contents/MacOS/Blender", globs)


if __name__ == "__main__":
    unittest.main()
