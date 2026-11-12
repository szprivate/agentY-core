"""spaitial_echo offline: every HTTP call is a stub, nothing is billed."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from agenty_core.tools import spaitial_echo as e

BASE = "https://api.spaitial.ai"


class Resp:
    def __init__(self, status=200, body=None, content=b"", headers=None):
        self.status_code, self._b, self.content, self.headers = status, body if body is not None else {}, content, headers or {}
        self.text = json.dumps(self._b)

    def json(self):
        return self._b


def png_file(d, name="room.png"):
    p = Path(d) / name
    Image.new("RGB", (64, 36), (90, 120, 60)).save(p, "PNG")
    return str(p)


class Fake:
    """A SpAItial API: answers by method + path, records every call (and which headers it got)."""

    def __init__(self, polls=1, fail=None, models=None, export_polls=1):
        self.calls, self.polls, self.fail, self.export_polls = [], polls, fail, export_polls
        self.models = models if models is not None else {"models": [{"id": "echo-2", "pricing": {"credits": 160}},
                                                                    {"id": "echo-hq", "pricing": {"credits": 800}}]}
        self.exports = {}

    def request(self, method, url, headers=None, json=None, files=None, params=None, timeout=None, **_):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}), "json": json, "files": files})
        path = url[len(BASE):] if url.startswith(BASE) else url
        if path == "/v1/models":
            return Resp(200, self.models)
        if path == "/v1/files" and method == "POST":
            return Resp(201, {"file_id": "file_1", "status": "available"})
        if path == "/v1/worlds" and method == "POST":
            return Resp(202, {"request_id": "req_1", "status": "PENDING"})
        if path.endswith("/status"):
            self.polls -= 1
            if self.polls > 0:
                return Resp(200, {"status": "PROCESSING", "progress": 0.4})
            if self.fail:
                return Resp(200, {"status": "FAILED", "error": {"code": self.fail, "message": "nope"}})
            return Resp(200, {"status": "COMPLETED", "progress": 1})
        if path == "/v1/worlds/requests/req_1":
            return Resp(200, {"request_id": "req_1", "status": "COMPLETED", "world": {
                "id": "w_1", "title": "room", "splat_url": f"{BASE}/v1/worlds/requests/req_1/splat",
                "splat_format": "spz", "thumbnail_url": "https://cdn.example/thumb.jpg",
                "panorama_url": f"{BASE}/v1/worlds/requests/req_1/panorama",
                "viewer_url": "https://app.spaitial.ai/w/w_1"}})
        if "/exports/" in path:
            kind = path.rsplit("/", 1)[-1]
            if method == "POST":
                self.exports[kind] = self.export_polls
                return Resp(202, {"status": "PROCESSING"})
            if kind not in self.exports:
                return Resp(404, {"error": {"code": "REQUEST_NOT_FOUND", "message": "no export"}})
            self.exports[kind] -= 1
            if self.exports[kind] > 0:
                return Resp(200, {"status": "PROCESSING"})
            return Resp(200, {"status": "READY", "download_url": f"{BASE}/v1/worlds/requests/req_1/exports/{kind}?download=1"})
        return Resp(404, {"error": {"code": "NOT_FOUND", "message": path}})

    def get(self, url, headers=None, timeout=None, allow_redirects=True):
        self.calls.append({"method": "GET", "url": url, "headers": dict(headers or {}), "redirects": allow_redirects})
        if url.startswith(BASE):
            return Resp(302, headers={"Location": "https://storage.example/signed/" + url.rsplit("/", 1)[-1]})
        return Resp(200, content=b"BYTES:" + url.encode())


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = self.tmp.name
        p = mock.patch.dict(os.environ, {"SPAITIAL_API_KEY": "spt_live_test", "SPAITIAL_BASE_URL": ""})
        p.start()
        self.addCleanup(p.stop)
        for target, value in ((e, "_sleep"), ):
            p = mock.patch.object(target, value, lambda s: None)
            p.start()
            self.addCleanup(p.stop)
        self.opened = []
        p = mock.patch.object(e, "_open_viewer", lambda path: self.opened.append(str(path)) or {"tab": "t1"})
        p.start()
        self.addCleanup(p.stop)

    def use(self, fake):
        for name in ("request", "get"):
            p = mock.patch.object(e.requests, name, getattr(fake, name))
            p.start()
            self.addCleanup(p.stop)
        return fake

    def run_json(self, fn, **kw):
        return json.loads(fn(output_dir=self.out, **kw))


class CreateTests(Base):
    def test_no_key_is_an_auth_error(self):
        with mock.patch.dict(os.environ, {"SPAITIAL_API_KEY": ""}), mock.patch.object(e, "_key", lambda: ""):
            out = self.run_json(e.echo_create_world, name="x", prompt="a room")
        self.assertEqual(out["error_kind"], "auth")

    def test_dry_run_sends_nothing_billable(self):
        fake = self.use(Fake())
        out = self.run_json(e.echo_create_world, name="room", prompt="a sunlit courtyard", dry_run=True)
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["estimate"]["credits"], 160)
        self.assertEqual(out["estimate"]["source"], "api")
        self.assertEqual(out["request"]["input"], {"type": "text", "prompt": "a sunlit courtyard"})
        self.assertEqual([c["url"] for c in fake.calls], [f"{BASE}/v1/models"])

    def test_prompt_world_end_to_end(self):
        fake = self.use(Fake(polls=3))
        out = self.run_json(e.echo_create_world, name="court yard", prompt="a sunlit courtyard")
        self.assertEqual(out["status"], "ok", out)
        wd = Path(self.out) / "court_yard"
        self.assertEqual(out["world_dir"], str(wd))
        self.assertTrue((wd / "splat.spz").read_bytes().startswith(b"BYTES:"))
        self.assertTrue((wd / "thumbnail.jpg").exists())
        world = json.loads((wd / "world.json").read_text())
        self.assertEqual(world["status"], "ready")
        self.assertEqual(world["request_id"], "req_1")
        self.assertEqual(world["viewer_url"], "https://app.spaitial.ai/w/w_1")
        post = next(c for c in fake.calls if c["url"] == f"{BASE}/v1/worlds")
        self.assertEqual(post["json"], {"input": {"type": "text", "prompt": "a sunlit courtyard"}, "model": "echo-2",
                                        "title": "court_yard", "output_format": "spz"})
        self.assertIn("Idempotency-Key", post["headers"])
        self.assertEqual(self.opened, [str(wd / "splat.spz")])

    def test_the_key_never_goes_to_the_storage_host(self):
        fake = self.use(Fake())
        self.run_json(e.echo_create_world, name="room", prompt="a room")
        for c in fake.calls:
            if c["url"].startswith(BASE):
                self.assertEqual(c["headers"].get("Authorization"), "Bearer spt_live_test")
            else:
                self.assertNotIn("Authorization", c["headers"])
        api_gets = [c for c in fake.calls if c.get("redirects") is not None and c["url"].startswith(BASE)]
        self.assertTrue(api_gets and all(c["redirects"] is False for c in api_gets))

    def test_a_local_picture_is_uploaded_and_the_prompt_left_out(self):
        fake = self.use(Fake())
        img = png_file(self.out)
        out = self.run_json(e.echo_create_world, name="pic", image=img, prompt="ignored words", is_pano="false")
        self.assertEqual(out["status"], "ok", out)
        up = next(c for c in fake.calls if c["url"] == f"{BASE}/v1/files")
        self.assertEqual(up["files"]["file"][0], "room.png")
        post = next(c for c in fake.calls if c["url"] == f"{BASE}/v1/worlds")
        self.assertEqual(post["json"]["input"], {"type": "file_id", "file_id": "file_1", "is_pano": False})
        self.assertTrue(any("left out" in n for n in out["notes"]))

    def test_an_https_picture_is_passed_as_a_url(self):
        fake = self.use(Fake())
        self.run_json(e.echo_create_world, name="u", image="https://example.com/a.jpg")
        post = next(c for c in fake.calls if c["url"] == f"{BASE}/v1/worlds")
        self.assertEqual(post["json"]["input"], {"type": "url", "image_url": "https://example.com/a.jpg"})
        self.assertFalse(any(c["url"] == f"{BASE}/v1/files" for c in fake.calls))

    def test_mesh_export_is_made_and_downloaded(self):
        fake = self.use(Fake(export_polls=2))
        out = self.run_json(e.echo_create_world, name="m", prompt="a room", mesh="mesh-simplified")
        self.assertEqual(out["mesh"], str(Path(self.out) / "m" / "mesh_simplified.ply"))
        self.assertTrue(Path(out["mesh"]).exists())
        self.assertTrue(any(c["method"] == "POST" and c["url"].endswith("/exports/mesh-simplified") for c in fake.calls))

    def test_a_failed_job_is_recorded_and_may_be_made_again(self):
        self.use(Fake(fail="INTERNAL_ERROR"))
        out = self.run_json(e.echo_create_world, name="f", prompt="a room")
        self.assertEqual(out["error_kind"], "execution")
        world = json.loads((Path(self.out) / "f" / "world.json").read_text())
        self.assertEqual(world["status"], "failed")
        self.use(Fake())
        again = self.run_json(e.echo_create_world, name="f", prompt="a room")
        self.assertEqual(again["status"], "ok", again)
        world = json.loads((Path(self.out) / "f" / "world.json").read_text())
        self.assertEqual(len(world["failed_attempts"]), 1)

    def test_a_ready_world_is_protected(self):
        self.use(Fake())
        self.run_json(e.echo_create_world, name="keep", prompt="a room")
        out = self.run_json(e.echo_create_world, name="keep", prompt="a room")
        self.assertEqual(out["error_kind"], "args")

    def test_moderation_and_credits_are_classified(self):
        for status, code, kind in ((402, "INSUFFICIENT_CREDITS", "balance"), (403, "MODERATION_REJECTED", "content_refusal"),
                                   (401, "UNAUTHORIZED", "auth"), (429, "RATE_LIMIT_EXCEEDED", "rate_limit")):
            self.assertEqual(e._http_error(Resp(status, {"error": {"code": code, "message": "m"}})).kind, kind)

    def test_a_model_the_key_may_not_use_is_refused_before_spending(self):
        fake = self.use(Fake(models={"models": [{"id": "echo-2"}]}))
        out = self.run_json(e.echo_create_world, name="hq", prompt="a room", model="echo-hq")
        self.assertEqual(out["error_kind"], "scope")
        self.assertFalse(any(c["url"] == f"{BASE}/v1/worlds" for c in fake.calls))

    def test_bad_arguments(self):
        self.use(Fake())
        for kw in ({"prompt": ""}, {"prompt": "x", "model": "echo-9"}, {"prompt": "x", "format": "glb"},
                   {"prompt": "x", "mesh": "obj"}, {"image": "http://example.com/a.png"}):
            out = self.run_json(e.echo_create_world, name="b", **kw)
            self.assertEqual(out["error_kind"], "args", kw)


class ExportTests(Base):
    def test_a_timed_out_world_is_collected_later(self):
        self.use(Fake(polls=99))
        with mock.patch.object(e.time, "monotonic", side_effect=[0.0, 0.0, 99999.0, 99999.0]):
            out = self.run_json(e.echo_create_world, name="slow", prompt="a room", timeout_s=10)
        self.assertEqual(out["error_kind"], "timeout")
        self.use(Fake())
        got = self.run_json(e.echo_export, name="slow")
        self.assertEqual(got["status"], "ok", got)
        self.assertTrue(Path(got["file"]).exists())

    def test_floorplan_export(self):
        self.use(Fake())
        self.run_json(e.echo_create_world, name="fp", prompt="a room", open_in_viewer=False)
        got = self.run_json(e.echo_export, name="fp", what="floorplan")
        self.assertTrue(got["file"].endswith("floorplan.png"))
        self.assertEqual(self.opened, [])

    def test_unknown_world(self):
        self.use(Fake())
        out = self.run_json(e.echo_export, name="nope")
        self.assertEqual(out["error_kind"], "args")


if __name__ == "__main__":
    unittest.main()
