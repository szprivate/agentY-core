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
            {"href": "https://example.com/a", "title": "A", "img": "https://x/t.jpg", "w": 200, "h": 300},
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


class SimilarAndDownloads(unittest.TestCase):
    def test_yandex_tile_carries_full_size(self):
        href = "https://yandex.com/images/search?pos=1&img_url=" + "https%3A%2F%2Fi.pinimg.com%2Foriginals%2Fa.jpg" + "&cbir_page=similar"
        raw = {"tiles": [{"href": href, "thumb": "https://avatars.mds.yandex.net/i?id=x", "w": 250, "h": 300},
                         {"href": href, "thumb": "https://avatars.mds.yandex.net/i?id=y", "w": 250, "h": 300}]}
        out = rs._similar_entries(raw, "yandex", 40)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["image_url"], "https://i.pinimg.com/originals/a.jpg")
        self.assertTrue(out[0]["full_size"])
        self.assertTrue(out[0]["thumb_url"].startswith("https://avatars"))

    def test_bing_m_attribute_murl_preferred(self):
        m = json.dumps({"murl": "https://site.com/big.jpg", "purl": "https://site.com/p", "turl": "https://th.bing.com/th/id/OIP.x"})
        out = rs._similar_entries({"tiles": [{"href": "", "m": m, "thumb": "https://th.bing.com/th/id/OIP.x?w=180", "w": 180, "h": 200}]}, "bing", 5)
        self.assertEqual((out[0]["image_url"], out[0]["page_url"]), ("https://site.com/big.jpg", "https://site.com/p"))

    def test_favicons_are_not_result_images(self):
        self.assertTrue(rs._is_favicon("https://th.bing.com/th/id/ODF.abc?w=20&h=20"))
        self.assertTrue(rs._is_favicon("https://th.bing.com/th?q=x&w=32&h=32"))
        self.assertFalse(rs._is_favicon("https://th.bing.com/th/id/OIP.abc?w=180&h=270", 180, 270))
        out = rs._dedupe_results([{"href": "https://e.com/a", "title": "A", "img": "https://th.bing.com/th/id/ODF.x?w=20&h=20", "w": 20, "h": 20}], "bing", 5)
        self.assertEqual((out[0]["image_url"], out[0]["width"]), ("", 0))

    def test_near_duplicate(self):
        a = [i * 8 % 256 for i in range(256)]
        self.assertTrue(rs._near_duplicate(a, [min(255, v + 5) for v in a]))
        self.assertFalse(rs._near_duplicate(a, list(reversed(a))))

    def test_download_filters(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow missing")
        import io, os, tempfile

        def png(size, seed):
            im = Image.new("RGB", size)
            im.putdata([((x * seed) % 256, (x // size[0] * 7 * seed) % 256, seed * 3 % 256) for x in range(size[0] * size[1])])
            b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()

        query, big, small = png((300, 300), 1), png((400, 300), 5), png((100, 100), 9)
        store = {"https://q/q.png": query, "dupq": query, "big": big, "big2": big, "small": small}

        def fake_fetch(urls, headers):
            for u in urls:
                if u in store:
                    return u, "image/png", store[u] + b"\x00" * 2000
            return None

        cands = [{"urls": [k], "page_url": "p", "kind": "similar", "engine": "yandex"} for k in ("dupq", "small", "big", "big2")]
        with mock.patch.object(rs, "_fetch", fake_fetch), tempfile.TemporaryDirectory() as d:
            out = rs._download_all(cands, d, 16, query_image="https://q/q.png")
            self.assertEqual([x["source_url"] for x in out], ["big"])
            self.assertTrue(os.path.basename(out[0]["path"]).startswith("yandex_similar_01"))
            self.assertEqual((out[0]["width"], out[0]["duplicate_of_query"]), (400, False))


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
