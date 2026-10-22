import os
import tempfile
import unittest

from PIL import Image

from agenty_core.utils import contact_sheet as cs


class ContactSheet(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)

    def _img(self, name, size=(300, 200), color=(200, 0, 0)):
        p = os.path.join(self.d.name, name)
        Image.new("RGB", size, color).save(p)
        return p

    def test_layout_is_roughly_square(self):
        self.assertEqual(cs.layout(1), (1, 1))
        self.assertEqual(cs.layout(4), (2, 2))
        self.assertEqual(cs.layout(5), (3, 2))
        self.assertEqual(cs.layout(6, columns=2), (2, 3))

    def test_sheet_has_every_tile_numbered(self):
        paths = [self._img(f"a{i}.png") for i in range(5)]
        r = cs.build_contact_sheet(paths, cell=128, out_path=os.path.join(self.d.name, "s.jpg"))
        self.assertEqual([t["n"] for t in r["tiles"]], [1, 2, 3, 4, 5])
        with Image.open(r["path"]) as im:
            self.assertGreater(im.width, 128 * 2)
        self.assertEqual((r["columns"], r["rows"]), (3, 2))

    def test_unreadable_image_is_skipped_and_named(self):
        bad = os.path.join(self.d.name, "nope.png")
        r = cs.build_contact_sheet([self._img("ok.png"), bad], cell=64)
        self.assertEqual(len(r["tiles"]), 1)
        self.assertEqual(r["skipped"][0]["file"], bad)
        os.remove(r["path"])

    def test_nothing_readable_raises(self):
        with self.assertRaises(ValueError):
            cs.build_contact_sheet([os.path.join(self.d.name, "x.png")])

    def test_question_names_the_tiles(self):
        r = cs.build_contact_sheet([self._img("a.png"), self._img("b.png")], cell=64)
        q = cs.sheet_question("Which is sharper?", r["tiles"])
        self.assertIn("1 = a.png", q)
        self.assertIn("Which is sharper?", q)
        os.remove(r["path"])

    def test_caps_at_max_tiles(self):
        paths = [self._img(f"c{i}.png", size=(20, 20)) for i in range(cs.MAX_TILES + 3)]
        r = cs.build_contact_sheet(paths, cell=32)
        self.assertEqual(len(r["tiles"]), cs.MAX_TILES)
        os.remove(r["path"])


if __name__ == "__main__":
    unittest.main()
