"""dashscope_generate offline: HTTP mocked, files into a temp dir."""

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import dashscope_generate as d

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")


class Resp:
    def __init__(self, status=200, body=None, content=b""):
        self.status_code, self._b, self.content = status, body if body is not None else {}, content
        self.text = json.dumps(self._b)

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test", "DASHSCOPE_BASE_URL": "",
                                              "DASHSCOPE_PAYG_API_KEY": "", "DASHSCOPE_PAYG_BASE_URL": ""})
        env.start()
        self.addCleanup(env.stop)
        s = mock.patch.object(d, "_sleep", lambda s: None)
        s.start()
        self.addCleanup(s.stop)

    def run_tool(self, post, get, **kw):
        with mock.patch.object(d.requests, "post", side_effect=post), \
                mock.patch.object(d.requests, "get", side_effect=get):
            return json.loads(d.dashscope_generate(output_dir=self.tmp.name, **kw))


class VideoTests(Base):
    def test_i2v_submit_poll_download(self):
        img = Path(self.tmp.name) / "in.png"
        img.write_bytes(PNG)
        posted = []
        polls = iter(["PENDING", "RUNNING", "SUCCEEDED"])

        def post(url, **kw):
            posted.append((url, kw))
            return Resp(200, {"output": {"task_id": "T1", "task_status": "PENDING"}})

        def get(url, **kw):
            if url.endswith("/tasks/T1"):
                st = next(polls)
                out = {"task_status": st}
                if st == "SUCCEEDED":
                    out["video_url"] = "https://x/y.mp4?sig=1"
                return Resp(200, {"output": out, "usage": {"duration": 5, "SR": 480}, "request_id": "R"})
            return Resp(200, content=b"MP4DATA")

        out = self.run_tool(post, get, kind="video", model="wan2.2-i2v-flash", prompt="go",
                            image_paths=[str(img)], resolution="480P")
        self.assertEqual(out["usage"]["duration"], 5)
        self.assertEqual(Path(out["files"][0]).read_bytes(), b"MP4DATA")
        self.assertTrue(Path(out["files"][0] + ".agenty.json").is_file())
        url, kw = posted[0]
        self.assertTrue(url.startswith("https://dashscope-intl.aliyuncs.com/api/v1/"))
        self.assertEqual(kw["headers"]["X-DashScope-Async"], "enable")
        self.assertTrue(kw["json"]["input"]["img_url"].startswith("data:image/png;base64,"))
        self.assertEqual(kw["json"]["parameters"]["resolution"], "480P")
        self.assertNotIn("duration", kw["json"]["parameters"])

    def test_new_family_uses_media(self):
        body, bad = d._video_body("happyhorse-1.0-i2v", "p", ["https://a/b.png"], "720P", "", 5, "n", -1, False, "")
        self.assertEqual(bad, "")
        self.assertEqual(body["input"]["media"], [{"type": "first_frame", "url": "https://a/b.png"}])
        self.assertNotIn("negative_prompt", body["input"])

    def test_base_url_host_is_used(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_BASE_URL": "https://ws-1.maas.aliyuncs.com/compatible-mode/v1"}):
            self.assertEqual(d._api_root(), "https://ws-1.maas.aliyuncs.com/api/v1")

    def test_payg_key_and_url_preferred(self):
        env = {"DASHSCOPE_API_KEY": "sk-sp-plan", "DASHSCOPE_BASE_URL": "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
               "DASHSCOPE_PAYG_API_KEY": "sk-pay", "DASHSCOPE_PAYG_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(d._api_key(), "sk-pay")
            self.assertEqual(d._api_root(), "https://dashscope.aliyuncs.com/api/v1")

    def test_plan_key_alone_is_refused_with_the_fix(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-sp-plan"}):
            out = json.loads(d.dashscope_generate("video", "wan2.2-i2v-flash", "x"))
        self.assertEqual(out["error_kind"], "auth")
        self.assertIn("DASHSCOPE_PAYG_API_KEY", out["error"])

    def test_unknown_model_local(self):
        out = json.loads(d.dashscope_generate("video", "sora-2", "x"))
        self.assertEqual(out["error_kind"], "unknown_model")

    def test_unknown_model_from_api(self):
        out = self.run_tool(
            lambda *a, **k: Resp(400, {"code": "InvalidParameter", "message": "Model not exist."}),
            None, kind="video", model="wan2.9-t2v", prompt="x")
        self.assertEqual(out["error_kind"], "unknown_model")

    def test_balance_error(self):
        out = self.run_tool(
            lambda *a, **k: Resp(400, {"code": "Arrearage",
                                       "message": "Access denied, please make sure your account is in good standing."}),
            None, kind="video", model="wan2.7-t2v", prompt="x")
        self.assertEqual(out["error_kind"], "balance")

    def test_content_refusal_in_task(self):
        def post(*a, **k):
            return Resp(200, {"output": {"task_id": "T", "task_status": "PENDING"}})

        def get(url, **k):
            return Resp(200, {"output": {"task_status": "FAILED", "code": "DataInspectionFailed",
                                         "message": "Input data may contain inappropriate content."}})
        out = self.run_tool(post, get, kind="video", model="wan2.7-t2v", prompt="x")
        self.assertEqual(out["error_kind"], "content_refusal")
        self.assertTrue(out["content_refusal"])
        self.assertEqual(out["task_id"], "T")

    def test_timeout_keeps_task_id(self):
        def post(*a, **k):
            return Resp(200, {"output": {"task_id": "T9", "task_status": "PENDING"}})

        def get(url, **k):
            return Resp(200, {"output": {"task_status": "RUNNING"}})
        with mock.patch.object(d.time, "monotonic", side_effect=[0, 0, 5, 1000]):
            out = self.run_tool(post, get, kind="video", model="wan2.7-t2v", prompt="x", timeout_s=10)
        self.assertEqual(out["error_kind"], "timeout")
        self.assertEqual(out["task_id"], "T9")

    def test_missing_input_image(self):
        out = json.loads(d.dashscope_generate("video", "wan2.2-i2v-flash", "x", image_paths=["nope.png"]))
        self.assertEqual(out["error_kind"], "args")

    def test_no_key(self):
        with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "", "ALIBABA_API_KEY": ""}):
            out = json.loads(d.dashscope_generate("image", "qwen-image-2.0", "x"))
        self.assertEqual(out["error_kind"], "auth")


class ImageTests(Base):
    def test_sync_image(self):
        sent = {}

        def post(url, **kw):
            sent.update(kw)
            return Resp(200, {"output": {"choices": [{"message": {"content": [{"image": "https://x/a.png"}]}}]},
                              "usage": {"width": 512, "height": 512, "image_count": 1}, "request_id": "R"})

        out = self.run_tool(post, lambda url, **k: Resp(200, content=PNG), kind="image",
                            model="qwen-image-2.0", prompt="apple", size="512*512")
        self.assertEqual(out["usage"]["image_count"], 1)
        self.assertEqual(Path(out["files"][0]).read_bytes(), PNG)
        self.assertEqual(sent["json"]["parameters"]["size"], "512*512")
        self.assertEqual(sent["json"]["input"]["messages"][0]["content"], [{"text": "apple"}])

    def test_async_image(self):
        def post(url, **kw):
            self.assertTrue(url.endswith("/text2image/image-synthesis"))
            return Resp(200, {"output": {"task_id": "I1", "task_status": "PENDING"}})

        def get(url, **kw):
            if "/tasks/" in url:
                return Resp(200, {"output": {"task_status": "SUCCEEDED",
                                             "results": [{"url": "https://x/1.png"}, {"url": "https://x/2.png"}]},
                                  "usage": {"image_count": 2}})
            return Resp(200, content=PNG)
        out = self.run_tool(post, get, kind="image", model="wan2.5-t2i-preview", prompt="x", n=2)
        self.assertEqual(len(out["files"]), 2)

    def test_fetch(self):
        def get(url, **kw):
            if "/tasks/" in url:
                return Resp(200, {"output": {"task_status": "SUCCEEDED", "video_url": "https://x/v.mp4"}})
            return Resp(200, content=b"V")
        with mock.patch.object(d.requests, "get", side_effect=get):
            out = json.loads(d.dashscope_fetch("T1", output_dir=self.tmp.name))
        self.assertEqual(Path(out["files"][0]).read_bytes(), b"V")


if __name__ == "__main__":
    unittest.main()
