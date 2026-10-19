"""media_refs: which files a tool result really names."""
import json
import os
import tempfile
import time
import unittest

from agenty_core.utils import media_refs as mr


class FindLocal(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.mp4 = os.path.join(self.d, "my render.mp4")
        open(self.mp4, "wb").write(b"x")
        self.png = os.path.join(self.d, "a.png")
        open(self.png, "wb").write(b"x")

    def test_whole_string_path_with_spaces(self):
        self.assertEqual(mr.find_local({"content": [{"text": self.mp4}]}), [self.mp4])

    def test_path_in_prose_and_json(self):
        text = f"Rendered ok. Output: {self.mp4}\nStill: {self.png} done"
        self.assertEqual(sorted(mr.find_local(text)), sorted([self.mp4, self.png]))
        js = json.dumps({"outputLocation": self.png})      # backslashes escaped
        self.assertEqual(mr.find_local({"content": [{"text": js}]}), [self.png])

    def test_missing_and_non_media_are_ignored(self):
        txt = os.path.join(self.d, "n.txt")
        open(txt, "w").write("x")
        self.assertEqual(mr.find_local(f"{txt} {os.path.join(self.d, 'ghost.mp4')}"), [])

    def test_min_mtime_skips_echoed_inputs(self):
        old = time.time() - 3600
        os.utime(self.png, (old, old))
        self.assertEqual(mr.find_local(f"{self.png} {self.mp4}", min_mtime=time.time() - 60), [self.mp4])

    def test_prose_between_two_drive_paths(self):
        self.assertEqual(mr.find_local(f"D:/nope and then {self.png}"), [self.png])


class FindUrls(unittest.TestCase):
    def test_only_media_urls(self):
        text = "see https://x.io/docs and https://cdn.x.io/out/clip.mp4?sig=1, ok https://x.io/a.png."
        self.assertEqual(mr.find_urls(text), ["https://cdn.x.io/out/clip.mp4?sig=1", "https://x.io/a.png"])

    def test_kind(self):
        self.assertEqual(mr.kind_of("https://x/a.MP4?x=1"), "video")
        self.assertEqual(mr.kind_of("a.txt"), "")


if __name__ == "__main__":
    unittest.main()
