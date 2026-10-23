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

import asyncio
import base64
import concurrent.futures
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
_SIMILAR_LIMIT = 40
_MIN_SIDE = 250          # smaller downloads are thumbnails, not usable references
_DOWNLOAD_LIMIT = 16
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
  // Yandex similar-grid tiles carry the full-size original in the href's img_url parameter;
  // Bing image tiles carry it in the `m` attribute's JSON (murl).
  out.tiles = [];
  for (const a of document.querySelectorAll('a[href*="img_url="], a[m*="murl"], [m*="murl"]')) {
    const img = a.querySelector('img');
    const el = a.closest('a[href]') || a;
    out.tiles.push({href: el.href || '', m: a.getAttribute('m') || '',
      thumb: img ? abs(img.currentSrc || img.src || img.getAttribute('data-src') || '') : '',
      w: img ? (img.naturalWidth || 0) : 0, h: img ? (img.naturalHeight || 0) : 0});
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
        elif _is_favicon(image_url, link.get("w", 0), link.get("h", 0)):
            image_url = ""
            if not dims:
                width = height = 0
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


def _is_favicon(url: str, w: int = 0, h: int = 0) -> bool:
    """Bing shows each result's site icon (13-32 px, ``ODF.`` ids) - never a picture of the subject."""
    if not url or url.startswith("data:"):
        return True
    if "/th/id/ODF." in url or "id=ODF." in url:
        return True
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    try:
        qw = int((qs.get("w") or ["0"])[0])
    except ValueError:
        qw = 0
    known = [v for v in (int(w or 0), int(h or 0), qw) if v]
    return bool(known) and max(known) <= 32


def _tile_entry(tile: dict, engine: str) -> Optional[dict]:
    """One similar-image tile -> {image_url (full size), thumb_url, page_url, width, height}."""
    thumb = str(tile.get("thumb", "") or "")
    full = page = ""
    width, height = int(tile.get("w") or 0), int(tile.get("h") or 0)
    if engine == "yandex":
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(str(tile.get("href", ""))).query)
        full = (qs.get("img_url") or [""])[0]
    elif engine == "bing" and tile.get("m"):
        try:
            m = json.loads(tile["m"])
        except (ValueError, TypeError):
            m = {}
        full, page = str(m.get("murl") or ""), str(m.get("purl") or "")
        thumb = thumb or str(m.get("turl") or "")
    if not full.startswith(("http://", "https://")):
        full = ""
    if thumb and not thumb.startswith(("http://", "https://")):
        thumb = ""
    if full:
        width, height = 0, 0   # the tile's size is the thumbnail's, not the original's
    elif not thumb:
        return None
    return {"image_url": full or thumb, "thumb_url": thumb, "page_url": page,
            "width": width, "height": height, "full_size": bool(full)}


def _similar_entries(raw: dict, engine: str, limit: int) -> list:
    """Structured similar-image entries from the page's tiles, full-size originals first."""
    seen: set = set()
    out: list = []
    for tile in raw.get("tiles", []):
        e = _tile_entry(tile, engine)
        if not e or e["image_url"] in seen or _is_favicon(e["thumb_url"], tile.get("w", 0), tile.get("h", 0)) and not e["full_size"]:
            continue
        seen.add(e["image_url"])
        out.append(e)
    out.sort(key=lambda e: not e["full_size"])   # stable: page order within each group
    return out[:limit]


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


def _yandex_similar_grid(page, raw: dict) -> list:
    """Open Yandex's full "Similar images" page, scroll for more, return its tiles."""
    link = next((str(l.get("href", "")) for l in raw.get("links", [])
                 if "cbir_page=similar" in str(l.get("href", ""))), "")
    if not link:
        return []
    page.goto(link, timeout=_PAGE_TIMEOUT_MS)
    page.wait_for_timeout(4000)
    for _ in range(5):
        try:
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(1500)
            more = page.locator("button.CbirSimilarList-MoreButton, a.CbirSimilarList-MoreButton")
            if more.count():
                more.first.click(timeout=1500)
        except Exception:  # noqa: BLE001
            pass
    grid = page.evaluate(_EXTRACT_JS)
    return _similar_entries(grid, "yandex", _SIMILAR_LIMIT)


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
                    "error_kind": "captcha", "results": [], "similar": [], "similar_images": []}
        results = _dedupe_results(raw.get("links", []), engine, max_results)
        known = {r["image_url"] for r in results if r["image_url"]}
        report = {"engine": engine, "page_url": url, "best_guess": _best_guess(raw, engine), "results": results}
        similar = _similar_entries(raw, engine, _SIMILAR_LIMIT)
        if engine == "yandex":
            # the strip on the result page is 4 thumbnails; the full grid is behind "Similar images"
            try:
                grid = _yandex_similar_grid(page, raw)
                if grid:
                    similar = grid
            except Exception as exc:  # noqa: BLE001
                logger.info("reverse_image_search: yandex similar grid failed: %s", exc)
        report["similar"] = similar
        # plain URL list kept for compatibility: full-size originals first, then thumbnails
        urls = [e["image_url"] for e in similar]
        urls += [e["thumb_url"] for e in similar if e["thumb_url"]]
        urls += _similar_images(raw, known | set(urls), max_results)
        report["similar_images"] = list(dict.fromkeys(urls))
        return report
    finally:
        try:
            page.close()
        except Exception:  # noqa: BLE001
            pass


def _ahash(img) -> list:
    """16x16 grayscale pixels - a cheap fingerprint for near-duplicate checks."""
    return list(img.convert("L").resize((16, 16)).getdata())


def _near_duplicate(a: list, b: list, threshold: float = 10.0) -> bool:
    """True when two fingerprints differ by less than ``threshold`` grey levels on average."""
    if not a or not b or len(a) != len(b):
        return False
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    # compare after removing overall brightness, so a lightened copy still matches
    diff = sum(abs((x - ma) - (y - mb)) for x, y in zip(a, b)) / len(a)
    return diff < threshold


def _fetch(urls: list, headers: dict):
    import requests

    for url in urls:
        if not url:
            continue
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") and len(r.content) >= 2000:
                return url, r.headers.get("content-type", "").split(";")[0].strip(), r.content
        except Exception as exc:  # noqa: BLE001
            logger.info("reverse_image_search: download of %s failed: %s", url[:100], exc)
    return None


def _download_all(cands: list, dest: str, limit: int, query_image: str = "") -> list:
    """Download candidates ``{urls, page_url, kind, engine}`` in order.

    Skips pictures under ``_MIN_SIDE`` px, the query picture itself (near-duplicates)
    and repeated content; returns one record per saved file.
    """
    import hashlib
    import io

    from PIL import Image

    os.makedirs(dest, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
    qhash: list = []
    try:
        if _is_url(query_image):
            got = _fetch([query_image], headers)
            qdata = got[2] if got else b""
        else:
            with open(query_image, "rb") as fh:
                qdata = fh.read()
        if qdata:
            qhash = _ahash(Image.open(io.BytesIO(qdata)))
    except Exception as exc:  # noqa: BLE001
        logger.info("reverse_image_search: query picture unreadable for duplicate check: %s", exc)

    from concurrent.futures import ThreadPoolExecutor

    todo = cands[:max(limit * 4, 40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(lambda c: _fetch(c["urls"], headers), todo))

    saved: list = []
    hashes: set = set()
    prints: list = []
    counters: dict = {}
    for cand, got in zip(todo, fetched):
        if len(saved) >= limit:
            break
        if not got:
            continue
        url, ctype, data = got
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            width, height = img.size
            if min(width, height) < _MIN_SIDE:
                continue
            digest = hashlib.sha1(data).hexdigest()
            if digest in hashes:
                continue
            fp = _ahash(img)
            if qhash and _near_duplicate(fp, qhash):
                continue
            if any(_near_duplicate(fp, h, 4.0) for h in prints):   # the same photo again, resized
                continue
            hashes.add(digest)
            prints.append(fp)
            ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
                   "image/gif": ".gif", "image/avif": ".avif"}.get(ctype, ".jpg")
            engine = cand.get("engine") or "web"
            counters[engine] = counters.get(engine, 0) + 1
            path = os.path.join(dest, f"{engine}_{cand.get('kind', 'result')}_{counters[engine]:02d}{ext}")
            with open(path, "wb") as fh:
                fh.write(data)
            saved.append({"source_url": url, "page_url": cand.get("page_url", ""), "path": path,
                          "width": width, "height": height, "kind": cand.get("kind", "result"),
                          "size_bytes": len(data), "duplicate_of_query": False})
        except Exception as exc:  # noqa: BLE001
            logger.info("reverse_image_search: %s unusable: %s", url[:100], exc)
    return saved


@tool
def reverse_image_search(
    image: str,
    engines: Optional[list] = None,
    max_results: int = 20,
    download_dir: str = "",
    detail: str = "compact",
) -> str:
    """Reverse image search: find other photos of the same object or place, or where a picture is from.

    Use this when you have ONE picture and need more of the same subject - other
    angles of a building, other shots of a location, bigger or cleaner copies of
    a photo - or when you must identify what or where something is. It is the
    opposite of ``web_search_images``, which searches by words.

    It opens the machine's Chrome in a window parked off-screen and asks Yandex
    and Bing's visual search; a run takes about 1 minute. ``similar`` holds OTHER
    PHOTOS OF THE SUBJECT (Yandex's full "Similar images" grid, ~40 items, with
    full-size originals) - that is where other angles of the same object usually
    are. ``results`` are pages that show the same or a similar picture. Google
    Lens ("google") is blocked by a captcha and only reports that.

    Args:
        image: A public image URL, or a path to a local image file (uploaded).
        engines: From "yandex", "bing", "google". Default ["yandex", "bing"].
        max_results: Most ``results`` to keep per engine (1-50). Default 20.
        download_dir: When set, up to 16 pictures are downloaded there so they
                      can be viewed: full-size ``similar`` originals first;
                      thumbnails (<250 px), copies of the query picture and
                      repeated content are skipped. Empty downloads nothing.
        detail: "compact" (default): per engine only best_guess, counts, ~5 top
                page titles, plus ``downloaded`` and a ``next`` hint (a few KB; LOOK
                at the downloaded pictures). "full": everything below (~50 KB).

    Returns:
        compact: ``{"image", "engines": [{"engine", "best_guess", "n_results",
        "n_similar", "top": ["title - host"], "error"?, "error_kind"?}],
        "dir", "downloaded": [{"file", "width", "height", "kind", "source"}] (or
        ``image_urls`` when download_dir is empty), "next"}``. full:
        JSON ``{"image", "engines": [{"engine", "page_url", "best_guess",
        "results": [{"page_url", "image_url", "title", "width", "height"}],
        "similar": [{"image_url" (full size when known), "thumb_url", "page_url",
        "width", "height", "full_size"}], "similar_images": [url, ...]}],
        "downloaded": [{"source_url", "page_url", "path", "width", "height",
        "kind": "similar"|"result", "duplicate_of_query": false}]}``. Result
        ``image_url`` is the result's own picture or empty (never a favicon).
        A failed engine carries ``error`` and ``error_kind`` ("captcha",
        "timeout", "failed"); a whole-call failure returns ``{"error",
        "error_kind"}`` ("missing_dependency": Playwright is not installed).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _reverse_image_search(image, engines, max_results, download_dir, detail)
    # Playwright's sync API refuses to run on a thread that has an event loop (the MCP
    # server calls tools from one), so the browser work goes to a worker thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_reverse_image_search, image, engines, max_results, download_dir, detail).result()


def _download_candidates(reports: list) -> list:
    """Download order: full-size similar originals (engines interleaved), then full-size
    result pictures, then similar thumbnails, then result thumbnails."""
    sim_full: list = []
    for rep in reports:
        sim_full.append([{"urls": [e["image_url"], e["thumb_url"]], "page_url": e.get("page_url", ""),
                          "kind": "similar", "engine": rep["engine"]}
                         for e in rep.get("similar", []) if e.get("full_size")])
    inter: list = []
    for i in range(max((len(x) for x in sim_full), default=0)):
        inter.extend(x[i] for x in sim_full if i < len(x))
    res_full, res_thumb, sim_thumb = [], [], []
    for rep in reports:
        for r in rep.get("results", []):
            if r.get("image_url") and r["image_url"] == r.get("page_url"):
                res_full.append({"urls": [r["image_url"]], "page_url": r["page_url"], "kind": "result", "engine": rep["engine"]})
            elif r.get("image_url"):
                res_thumb.append({"urls": [r["image_url"]], "page_url": r["page_url"], "kind": "result", "engine": rep["engine"]})
        sim_thumb.extend({"urls": [e["image_url"]], "page_url": e.get("page_url", ""), "kind": "similar",
                          "engine": rep["engine"]} for e in rep.get("similar", []) if not e.get("full_size"))
    out, seen = [], set()
    for c in inter + res_full + sim_thumb + res_thumb:
        if c["urls"][0] not in seen:
            seen.add(c["urls"][0])
            out.append(c)
    return out


_NEXT_HINT = ("Look at the downloaded pictures; keep only those that show the same object; "
              "ignore copies of the query photo.")


def _compact(out: dict) -> dict:
    """What an agent needs to act on: counts, a few page titles, and the downloaded pictures."""
    engines = []
    for rep in out.get("engines", []):
        e: dict = {"engine": rep.get("engine", ""), "best_guess": rep.get("best_guess", ""),
                   "n_results": len(rep.get("results", [])), "n_similar": len(rep.get("similar", []))}
        if rep.get("error"):
            e["error"] = str(rep["error"])[:160]
            e["error_kind"] = rep.get("error_kind", "failed")
        top = []
        for r in rep.get("results", []):
            title = (r.get("title") or "").strip()
            host = _host(r.get("page_url", ""))
            if title and title.lower().removeprefix("www.") != host.removeprefix("www."):
                line = f"{title[:80]} — {host}"
            else:
                line = host
            if line and line not in top:
                top.append(line)
            if len(top) >= 5:
                break
        e["top"] = top
        engines.append(e)
    res: dict = {"image": out.get("image", ""), "engines": engines}
    if out.get("downloaded"):
        paths = [d.get("path", "") for d in out["downloaded"]]
        res["dir"] = os.path.dirname(paths[0]) if paths else ""      # path = dir + file
        res["downloaded"] = [{"file": os.path.basename(d.get("path", "")), "width": d.get("width"),
                              "height": d.get("height"), "kind": d.get("kind", ""),
                              "source": _host(d.get("page_url") or d.get("source_url", ""))}
                             for d in out["downloaded"]]
    elif "downloaded" not in out:
        urls: list = []
        for rep in out.get("engines", []):
            urls += [x["image_url"] for x in rep.get("similar", []) if x.get("full_size") and x.get("image_url")]
            urls += [x["image_url"] for x in rep.get("results", []) if x.get("image_url") and x["image_url"] == x.get("page_url")]
        res["image_urls"] = list(dict.fromkeys(urls))[:8]
    else:
        res["downloaded"] = []
    if out.get("download_error"):
        res["download_error"] = out["download_error"]
    if res.get("downloaded") or res.get("image_urls"):
        res["next"] = _NEXT_HINT
    else:
        res["next"] = "Nothing found or downloaded; check the engine errors, retry once, or continue without it."
    return res


def _reverse_image_search(image, engines, max_results, download_dir, detail="compact") -> str:
    try:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import]
        except ImportError:
            return _err("Playwright is not installed.", "missing_dependency",
                        hint="Reverse image search is not installed. Re-run agentY's installer and enable it, or: .venv\\Scripts\\python.exe -m pip install playwright (uses your installed Chrome or Edge; no browser download)")

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
                                            "error_kind": kind, "results": [], "similar": [], "similar_images": []})
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
            try:
                out["downloaded"] = _download_all(_download_candidates(reports), download_dir,
                                                  limit=_DOWNLOAD_LIMIT, query_image=image)
            except Exception as exc:  # noqa: BLE001
                out["downloaded"] = []
                out["download_error"] = str(exc)[:200]
        if str(detail or "compact").strip().lower() == "full":
            return json.dumps(out, ensure_ascii=False, indent=2)
        return json.dumps(_compact(out), ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reverse_image_search failed: %s", exc)
        return _err(str(exc)[:300], "failed")
