"""read_text_file must not be able to flood an agent's context.

It returned whole files. A specialist asked "is X in the recipe database?" read
the 2.5 MB file to find out, and from then on every one of its model calls
carried ~700,000 tokens and took minutes (2026-10-05: 15 of a 20-minute turn).
A read is capped, says how much it left out, and ``find`` answers the lookup
that the big read was for.

Runs under pytest or directly (``python test_read_text_file_cap.py``).
"""
import json
import os
import tempfile
import unittest

from agenty_core.tools import file_tools as F


def _call(*args, **kw):
    fn = getattr(F.read_text_file, "__wrapped__", None) or F.read_text_file
    return fn(*args, **kw)


class Fixture(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="agenty_read_")

    def _file(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path


class TheCap(Fixture):

    def test_a_small_file_comes_back_exactly(self):
        path = self._file("a.txt", "hello\nworld\n")
        self.assertEqual(_call(path), "hello\nworld\n")

    def test_a_big_file_is_cut_and_says_so(self):
        path = self._file("big.json", "x" * (F.DEFAULT_READ_CHARS * 5))
        out = _call(path)
        self.assertLess(len(out), F.DEFAULT_READ_CHARS + 400)
        self.assertIn(f"of {F.DEFAULT_READ_CHARS * 5:,}", out)
        self.assertIn(f"offset={F.DEFAULT_READ_CHARS}", out)
        self.assertIn("find=", out)

    def test_offset_continues_where_the_note_said(self):
        text = "".join(f"{i:07d}\n" for i in range(20_000))       # 160,000 chars
        path = self._file("rows.txt", text)
        first = _call(path)
        second = _call(path, offset=F.DEFAULT_READ_CHARS)
        body = lambda out: out.split("\n\n[read_text_file]")[0]
        self.assertEqual(body(first) + body(second), text[:2 * F.DEFAULT_READ_CHARS])

    def test_the_last_piece_says_it_is_the_end(self):
        path = self._file("rows.txt", "y" * (F.DEFAULT_READ_CHARS + 10))
        self.assertIn("the end of the file", _call(path, offset=F.DEFAULT_READ_CHARS))

    def test_asking_for_more_has_a_ceiling(self):
        path = self._file("huge.txt", "z" * (F.MAX_READ_CHARS * 3))
        self.assertLess(len(_call(path, max_chars=10 ** 9)), F.MAX_READ_CHARS + 400)

    def test_each_file_of_a_batch_is_capped(self):
        a = self._file("a.txt", "a" * (F.DEFAULT_READ_CHARS * 2))
        b = self._file("b.txt", "short")
        files = json.loads(_call([a, b]))["files"]
        self.assertLess(len(files[a]), F.DEFAULT_READ_CHARS + 400)
        self.assertEqual(files[b], "short")


class Find(Fixture):

    def test_it_returns_only_the_matching_lines_with_numbers(self):
        path = self._file("r.json", "\n".join(
            ['"LTX-2.5_I2V"' if i == 500 else f'"other_{i}"' for i in range(50_000)]))
        out = _call(path, find="ltx-2.5")
        self.assertIn('501: "LTX-2.5_I2V"', out)
        self.assertLess(len(out), 400)

    def test_no_match_says_so(self):
        path = self._file("r.json", '{"a": 1}')
        self.assertIn("No line", _call(path, find="HDR"))

    def test_many_matches_are_counted_and_capped(self):
        path = self._file("r.txt", "hit\n" * 500)
        out = _call(path, find="hit")
        self.assertIn("500 line(s)", out)
        self.assertEqual(out.count("\n"), F._FIND_MAX_LINES)

    def test_a_long_line_shows_the_part_around_the_match(self):
        path = self._file("min.json", "a" * 5000 + "NEEDLE" + "b" * 5000)
        out = _call(path, find="needle")
        self.assertIn("NEEDLE", out)
        self.assertLess(len(out), 600)


if __name__ == "__main__":
    unittest.main()
