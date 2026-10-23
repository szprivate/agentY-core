"""reverse_search offline: parsing, dedupe and the failure paths - no browser, no network."""

import base64
import builtins
import json
import unittest
from unittest import mock

from agenty_core.tools import reverse_search as rs


class Helpers(unittest.TestCase):
    def test_dedupe_keeps_external_pages_once(self):
        links = [
            {"href": "https://yandex.com/images/search?x=1", "title": "nav"},
            {"href": "https://i.pinimg.com/736x/ab/cd.jpg", "title": "736 на 935"},
            {"href": "https://i.pinimg.com/736x/ab/cd.jpg#again", "title": "dup"},
            {"href": "https://example.com/a", "title": "A", "img": "https://x/t.jpg", "w": 10, "h": 20},
            {"href": "javascript:void(0)", "title": "x"},
        ]
        out = rs._dedupe_results(links, "yandex", 10)
        self.assertEqual([r["page_url"] for r in out], ["https://i.pinimg.com/736x/ab/cd.jpg", "https://example.com/a"])
        pin = out[0]
        self.assertEqual((pin["width"], pin["height"], pin["title"]), (736, 935, ""))
        self.assertEqual(pin["image_url"], pin["page_url"])   # a picture file links to itself
        self.assertEqual(out[1]["image_url"], "https://x/t.jpg")

    def test_max_results_cap(self):
        links = [{"href": f"https://e.com/{i}"} for i in range(30)]
        self.assertEqual(len(rs._dedupe_results(links, "bing", 5)), 5)

    def test_bing_redirect_is_unwrapped(self):
        target = "https://www.alamy.com/photo.html"
        packed = "a1" + base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        wrapped = f"https://www.bing.com/ck/a?!&&p=abc&u={packed}&ntb=1"
        out = rs._dedupe_results([{"href": wrapped, "title": "alamy.com"}], "bing", 5)
        self.assertEqual(out[0]["page_url"], target)

    def test_similar_images_skip_icons_and_known(self):
        raw = {"images": [
            {"src": "https://a/1.jpg", "w": 200, "h": 100},
            {"src": "https://a/icon.png", "w": 16, "h": 16},
            {"src": "https://a/known.jpg", "w": 300, "h": 300},
            {"src": "https://r.bing.com/rp/x.svg", "w": 300, "h": 300},
            {"src": "data:image/gif;base64,AAAA", "w": 300, "h": 300},
        ]}
        self.assertEqual(rs._similar_images(raw, {"https://a/known.jpg"}, 10), ["https://a/1.jpg"])

    def test_best_guess_and_captcha(self):
        self.assertEqual(rs._best_guess({"title": "Stilt House - Search", "text": ""}, "bing"), "Stilt House")
        self.assertEqual(rs._best_guess({"title": "Bing Images", "text": ""}, "bing"), "")
        self.assertTrue(rs._looks_like_captcha("https://www.google.com/sorry/index?x", {}))
        self.assertFalse(rs._looks_like_captcha("https://yandex.com/images/search", {"title": "t", "text": "ok"}))


class Failures(unittest.TestCase):
    def test_missing_playwright(self):
        real_import = builtins.__import__

        def fake(name, *a, **k):
            if name.startswith("playwright"):
                raise ImportError(name)
            return real_import(name, *a, **k)

        with mock.patch("builtins.__import__", side_effect=fake):
            out = json.loads(rs.reverse_image_search("https://example.com/a.jpg"))
        self.assertEqual(out["error_kind"], "missing_dependency")
        self.assertIn("pip install playwright", out["hint"])

    def test_bad_inputs_do_not_raise(self):
        pw = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"playwright": pw, "playwright.sync_api": pw}):
            self.assertEqual(json.loads(rs.reverse_image_search(""))["error_kind"], "bad_input")
            self.assertEqual(json.loads(rs.reverse_image_search("/no/such/file.jpg"))["error_kind"], "bad_input")
            self.assertEqual(json.loads(rs.reverse_image_search("https://e.com/a.jpg", engines=["altavista"]))["error_kind"],
                             "bad_input")


if __name__ == "__main__":
    unittest.main()
