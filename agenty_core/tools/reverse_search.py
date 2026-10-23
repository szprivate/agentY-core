"""
Reverse image search - find other photos of the same object or place.

``web_search_images`` only searches by text. ``reverse_image_search`` starts from
a picture (a URL or a file on disk) and asks Yandex and Bing's visual search
what else looks like it: the same photo on other pages, other shots of the same
building, a best-guess label for what it shows.

It drives the machine's own Chrome (or Edge) through Playwright, because those
engines answer a real browser and refuse plain HTTP clients. Playwright is an
optional dependency: without it the tool answers with a ``missing_dependency``
error instead of failing. Google Lens is not tried by default - it puts a
captcha in front of automated browsers - but ``engines=["google"]`` attempts it
and reports ``error_kind: "captcha"`` when it is stopped.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.parse
from typing import Any, Optional

from agenty_core._compat import tool

logger = logging.getLogger(__name__)

_SUPPORTED = ("yandex", "bing", "google")
_PAGE_TIMEOUT_MS = 45_000
_SETTLE_MS = 9_000
# Each engine's own hosts: links into them are navigation, not results.
_ENGINE_HOSTS = {
    "yandex": ("yandex.", "ya.ru"),
    "bing": ("bing.", "microsoft.", "msn.", "live.com", "office.com"),
    "google": ("google.", "gstatic.", "googleusercontent.", "youtube."),
}
_CONSENT_BUTTONS = ("Accept", "Accept all", "Allow all", "Alle akzeptieren", "I agree", "Agree")
_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif")

# Runs in the page: every link with its text and the picture inside it, plus the
# loose pictures (the "similar images" grids are often plain <img> in a link back
# to the engine).
_EXTRACT_JS = """
() => {
  const abs = (u) => { try { return new URL(u, location.href).href } catch (e) { return '' } };
  const out = {links: [], images: []};
  for (const a of document.querySelectorAll('a[href]')) {
    const img = a.querySelector('img');
    const text = ((a.getAttribute('aria-label') || a.title || a.innerText || (img && img.alt) || '') + '').trim();
    out.links.push({
      href: a.href, title: text.slice(0, 160),
      img: img ? abs(img.currentSrc || img.src || img.getAttribute('data-src') || '') : '',
      w: img ? (img.naturalWidth || 0) : 0, h: img ? (img.naturalHeight || 0) : 0,
    });
  }
  for (const img of document.querySelectorAll('img')) {
    const src = abs(img.currentSrc || img.src || img.getAttribute('data-src') || '');
    if (src.startsWith('http')) out.images.push({src, w: img.naturalWidth || 0, h: img.naturalHeight || 0,
                                                 alt: (img.alt || '').slice(0, 160)});
  }
  out.title = document.title || '';
  out.text = (document.body && document.body.innerText || '').slice(0, 6000);
  return out;
}
"""


# --------------------------------------------------------------------------
# pure helpers (no browser, no network) - covered by tests
# --------------------------------------------------------------------------
def _host(url: str) -> str:
    try:
        return (urllib.parse.urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _is_engine_host(url: str, engine: str) -> bool:
    h = _host(url)
    return any(p in h or h == p.strip(".") for p in _ENGINE_HOSTS.get(engine, ()))


def _unwrap_redirect(url: str) -> str:
    """Bing and Google wrap outbound links; return the real target when one is named."""
    try:
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query)
        # Bing: /ck/a?...&u=a1<base64url of the target>
        packed = (qs.get("u") or [""])[0]
        if packed.startswith("a1") and "bing." in _host(url):
            try:
                target = base64.urlsafe_b64decode(packed[2:] + "=" * (-len(packed[2:]) % 4)).decode("utf-8", "replace")
                if target.startswith(("http://", "https://")):
                    return target
            except Exception:  # noqa: BLE001
                pass
        for key in ("url", "u", "q", "r", "target"):
            val = (qs.get(key) or [""])[0]
            if val.startswith(("http://", "https://")) and _host(val) != _host(url):
                return val
    except ValueError:
        pass
    return url


def _dedupe_results(raw_links: list, engine: str, max_results: int) -> list:
    """Keep external-host links only, one per page URL, in page order."""
    seen: set = set()
    out: list = []
    for link in raw_links:
        href = _unwrap_redirect(str(link.get("href", "")))
        if not href.startswith(("http://", "https://")) or _is_engine_host(href, engine):
            continue
        key = href.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        title = str(link.get("title", "") or "").replace("\n", " ").strip()
        width, height = int(link.get("w") or 0), int(link.get("h") or 0)
        # Yandex labels a picture link with its size ("736 x 935", "736 на 935"): data, not a title.
        dims = re.fullmatch(r"(\d{2,5})\s*(?:[x×]|\w{1,3})\s*(\d{2,5})", title)
        if dims:
            width, height, title = int(dims.group(1)), int(dims.group(2)), ""
        image_url = str(link.get("img", "") or "")
        # A link straight to a picture file is the picture itself, at full size.
        if urllib.parse.urlparse(href).path.lower().endswith(_IMAGE_EXT):
            image_url = href
        out.append({"page_url": href, "image_url": image_url, "title": title,
                    "width": width, "height": height})
        if len(out) >= max_results:
            break
    return out


def _similar_images(raw: dict, exclude: set, limit: int) -> list:
    """Picture URLs on the page that no result already carries (icons and sprites dropped)."""
    seen = set(exclude)
    out: list = []
    for img in raw.get("images", []):
        src = str(img.get("src", ""))
        if not src.startswith(("http://", "https://")) or src in seen:
            continue
        if max(int(img.get("w") or 0), int(img.get("h") or 0)) < 90:
            continue
        low = src.lower().split("?", 1)[0]
        # the engine's own chrome and the query picture echoed back are not "similar" pictures
        if low.endswith(".svg") or "/images/blob" in low or "get-images-cbir" in low:
            continue
        seen.add(src)
        out.append(src)
        if len(out) >= limit:
            break
    return out


def _best_guess(raw: dict, engine: str) -> str:
    """The engine's own label for the picture, when its page shows one."""
    text = str(raw.get("text", ""))
    title = str(raw.get("title", ""))
    if engine == "yandex":
        m = re.search(r"(?:Image appears to contain|Looks like|Image is similar to)\s*[:\n]+\s*([^\n]{2,120})", text, re.I)
        if m:
            return m.group(1).strip()
    if engine == "bing":
        m = re.search(r"(?:Looks like|Best guess(?: for this image)?)\s*[:\n]+\s*([^\n]{2,120})", text, re.I)
        if m:
            return m.group(1).strip()
        # Bing redirects to a text search titled "<guess> - Search" when it recognises the picture.
        m = re.match(r"(.{3,120}?)\s+-\s+(?:Search|Bing)\b", title)
        if m and "visual search" not in m.group(1).lower():
            return m.group(1).strip()
    return ""


def _looks_like_captcha(url: str, raw: dict) -> bool:
    low = (url + " " + str(raw.get("title", "")) + " " + str(raw.get("text", ""))[:600]).lower()
    return any(s in low for s in ("/sorry/", "captcha", "unusual traffic", "showcaptcha", "are you not a robot"))


def _is_url(image: str) -> bool:
    return image.lower().startswith(("http://", "https://"))


def _err(message: str, kind: str, **extra: Any) -> str:
    return json.dumps({"error": message, "error_kind": kind, **extra}, ensure_ascii=False)


# --------------------------------------------------------------------------
# browser driving
# --------------------------------------------------------------------------
def _launch(p):
    """Prefer the system Chrome, then Edge, then Playwright's own chromium."""
    last: Optional[Exception] = None
    args = ["--window-position=-2400,0", "--disable-blink-features=AutomationControlled"]
    for channel in ("chrome", "msedge", None):
        try:
            kwargs: dict = {"headless": False, "args": args}
            if channel:
                kwargs["channel"] = channel
            return p.chromium.launch(**kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"no usable browser (tried Chrome, Edge, bundled chromium): {last}")


def _accept_consent(page) -> None:
    for label in _CONSENT_BUTTONS:
        try:
            btn = page.get_by_role("button", name=label)
            if btn.count():
                btn.first.click(timeout=2000)
                page.wait_for_timeout(2500)
                return
        except Exception:  # noqa: BLE001
            continue


def _upload(page, engine: str, path: str) -> None:
    """Hand a local file to an engine's own upload field."""
    if engine == "yandex":
        page.goto("https://yandex.com/images/", timeout=_PAGE_TIMEOUT_MS)
        page.wait_for_timeout(2500)
        _accept_consent(page)
        # the camera button reveals a file input; setting files on it starts the search
        for sel in ('button[aria-label*="image" i]', '[class*="CbirButton"]', '[class*="cbir" i] button',
                    'button[class*="camera" i]'):
            try:
                if page.locator(sel).count():
                    page.locator(sel).first.click(timeout=3000)
                    break
            except Exception:  # noqa: BLE001
                continue
        page.wait_for_timeout(1000)
        page.locator('input[type="file"]').first.set_input_files(path, timeout=15_000)
    elif engine == "bing":
        # the file input only listens once the camera button has opened the upload pane
        # (a first ordinary search sets the cookies Bing wants - without it the upload is "unable to process")
        page.goto("https://www.bing.com/search?q=image+search", timeout=_PAGE_TIMEOUT_MS)
        page.wait_for_timeout(3000)
        _accept_consent(page)
        page.goto("https://www.bing.com/images", timeout=_PAGE_TIMEOUT_MS)
        page.wait_for_timeout(2500)
        page.locator("#sb_sbi").first.click(timeout=8000)
        page.wait_for_timeout(1500)
        page.locator('input[type="file"]').first.set_input_files(path, timeout=15_000)
    elif engine == "google":
        page.goto("https://lens.google.com/upload", timeout=_PAGE_TIMEOUT_MS)
        page.wait_for_timeout(2500)
        _accept_consent(page)
        page.locator('input[type="file"]').first.set_input_files(path, timeout=15_000)
    else:
        raise ValueError(engine)


def _search_url(engine: str, image_url: str) -> str:
    q = urllib.parse.quote(image_url, safe="")
    if engine == "yandex":
        return "https://yandex.com/images/search?rpt=imageview&url=" + q
    if engine == "bing":
        return ("https://www.bing.com/images/search?view=detailv2&iss=sbi&form=SBIVSP"
                "&sbisrc=UrlPaste&q=imgurl:" + q)
    return "https://lens.google.com/uploadbyurl?url=" + q


def _run_engine(browser, engine: str, image: str, max_results: int) -> dict:
    page = browser.new_page(viewport={"width": 1400, "height": 1100}, locale="en-US")
    page.set_default_timeout(_PAGE_TIMEOUT_MS)
    try:
        if _is_url(image):
            page.goto(_search_url(engine, image), timeout=_PAGE_TIMEOUT_MS)
        else:
            _upload(page, engine, image)
        page.wait_for_timeout(_SETTLE_MS)
        _accept_consent(page)
        try:
            page.mouse.wheel(0, 900)
            page.wait_for_timeout(2500)
        except Exception:  # noqa: BLE001
            pass
        raw = page.evaluate(_EXTRACT_JS)
        url = page.url
        if _looks_like_captcha(url, raw):
            return {"engine": engine, "page_url": url, "error": "the engine put a captcha in front of the search",
                    "error_kind": "captcha", "results": [], "similar_images": []}
        results = _dedupe_results(raw.get("links", []), engine, max_results)
        known = {r["image_url"] for r in results if r["image_url"]}
        return {
            "engine": engine,
            "page_url": url,
            "best_guess": _best_guess(raw, engine),
            "results": results,
            "similar_images": _similar_images(raw, known, max_results),
        }
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass


def _download_all(urls: list, dest: str, limit: int) -> list:
    import requests

    os.makedirs(dest, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    saved: list = []
    for url in urls:
        if len(saved) >= limit:
            break
        try:
            r = requests.get(url, headers=headers, timeout=20)
            ctype = r.headers.get("content-type", "").split(";")[0].strip()
            if r.status_code != 200 or not ctype.startswith("image/") or len(r.content) < 2000:
                continue
            ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                   "image/gif": ".gif", "image/avif": ".avif"}.get(ctype, "")
            if not ext:
                path_ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
                ext = path_ext if path_ext in _IMAGE_EXT else ".jpg"
            path = os.path.join(dest, f"reverse_{len(saved) + 1:02d}{ext}")
            with open(path, "wb") as fh:
                fh.write(r.content)
            saved.append({"source_url": url, "path": path, "size_bytes": len(r.content)})
        except Exception as exc:  # noqa: BLE001
            logger.info("reverse_image_search: download of %s failed: %s", url[:100], exc)
    return saved


@tool
def reverse_image_search(
    image: str,
    engines: Optional[list] = None,
    max_results: int = 20,
    download_dir: str = "",
) -> str:
    """Reverse image search: find other photos of the same object or place, or where a picture is from.

    Use this when you have ONE picture and need more of the same subject - other
    angles of a building, other shots of a location, bigger or cleaner copies of
    a photo - or when you must identify what or where something is. It is the
    opposite of ``web_search_images``, which searches by words. Feed what you
    find to ``download_image`` (or set ``download_dir`` here) and look at the
    pictures with your vision tool.

    It opens the machine's Chrome in a window parked off-screen and asks Yandex
    and Bing's visual search; a run takes about 20-40 seconds per engine. Yandex
    is the better one for exact copies and other photos of the same thing; Bing
    often names the subject ("best_guess") and lists related pages. Google Lens
    ("google") is blocked by a captcha for automated browsers and only reports
    that.

    Args:
        image: A public image URL, or a path to a local image file (it is
               uploaded to the engine).
        engines: Which engines to ask, from "yandex", "bing", "google".
                 Default ["yandex", "bing"].
        max_results: Most results to keep per engine (1-50). Default 20.
        download_dir: When set, the top picture URLs found are downloaded into
                      this folder (failures skipped) so they can be viewed.
                      Empty (default) downloads nothing.

    Returns:
        JSON ``{"image", "engines": [{"engine", "page_url", "best_guess",
        "results": [{"page_url", "image_url", "title", "width", "height"}],
        "similar_images": [url, ...]} ...], "downloaded": [{"source_url",
        "path"}]}``. Results are links to OTHER sites (the engine's own links
        are left out), deduplicated by page. ``image_url`` is the picture shown
        on the engine's results page and can be a thumbnail. An engine that was
        blocked or failed carries its own ``error`` and ``error_kind``
        ("captcha", "timeout", "failed"). On a whole-call failure returns
        ``{"error": "<message>", "error_kind": "..."}``; ``"missing_dependency"``
        means Playwright is not installed.
    """
    try:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import]
        except ImportError:
            return _err("Playwright is not installed.", "missing_dependency",
                        hint="pip install playwright (uses the system Chrome; no browser download)")

        image = (image or "").strip()
        if not image:
            return _err("image is empty - pass an image URL or a file path.", "bad_input")
        if not _is_url(image):
            image = os.path.abspath(os.path.expanduser(image))
            if not os.path.isfile(image):
                return _err(f"no such file: {image}", "bad_input")
        wanted: list = []
        for e in (engines or ["yandex", "bing"]):
            e = str(e).strip().lower()
            if e in _SUPPORTED and e not in wanted:
                wanted.append(e)
        if not wanted:
            return _err(f"no supported engine in {engines!r}; use {list(_SUPPORTED)}.", "bad_input")
        max_results = max(1, min(int(max_results or 20), 50))

        reports: list = []
        try:
            with sync_playwright() as p:
                try:
                    browser = _launch(p)
                except Exception as exc:  # noqa: BLE001
                    return _err(str(exc)[:300], "no_browser",
                                hint="install Google Chrome or Microsoft Edge, or run `playwright install chromium`")
                try:
                    for engine in wanted:
                        try:
                            reports.append(_run_engine(browser, engine, image, max_results))
                        except Exception as exc:  # noqa: BLE001
                            kind = "timeout" if "imeout" in type(exc).__name__ + str(exc) else "failed"
                            logger.warning("reverse_image_search %s failed: %s", engine, exc)
                            reports.append({"engine": engine, "error": str(exc).splitlines()[0][:300],
                                            "error_kind": kind, "results": [], "similar_images": []})
                finally:
                    try:
                        browser.close()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception as exc:  # noqa: BLE001
            if not reports:
                return _err(str(exc).splitlines()[0][:300], "failed")

        out: dict = {"image": image, "engines": reports}
        if download_dir:
            # full-size picture files first, then thumbnails and similar-image tiles
            urls: list = []
            for rep in reports:
                urls.extend(r["image_url"] for r in rep.get("results", [])
                            if r.get("image_url") == r.get("page_url"))
            for rep in reports:
                urls.extend(r["image_url"] for r in rep.get("results", []) if r.get("image_url"))
                urls.extend(rep.get("similar_images", []))
            urls = list(dict.fromkeys(urls))
            try:
                out["downloaded"] = _download_all(urls[:40], download_dir, limit=min(max_results, 10))
            except Exception as exc:  # noqa: BLE001
                out["downloaded"] = []
                out["download_error"] = str(exc)[:200]
        return json.dumps(out, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reverse_image_search failed: %s", exc)
        return _err(str(exc)[:300], "failed")
