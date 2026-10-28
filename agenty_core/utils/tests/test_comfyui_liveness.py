"""A ComfyUI that dies or hangs mid-run ends the wait, with the partial output named.

Before: an unreadable /queue and /history counted as "still running", so a crash
left the wait polling a dead port for 7.8 hours.

    python -m unittest agenty_core.utils.tests.test_comfyui_liveness
"""

import asyncio
import os
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.utils.tests.test_comfyui_long_wait import ChattyWS, DONE


class DeadClient:
    base_url = "http://127.0.0.1:8188"
    api_key = ""

    def __init__(self):
        self.dead = False
        self.running = ["p1"]

    def get(self, path, **kw):
        if self.dead:
            raise ConnectionError("connection refused")
        if path.startswith("/history"):
            return {}
        if path == "/queue":
            return {"queue_running": [[0, p] for p in self.running], "queue_pending": []}
        return {}


def box(kind: bytes, payload: bytes = b"") -> bytes:
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


class Mp4Test(unittest.TestCase):
    def test_moov_detection(self):
        from agenty_core.utils import comfyui_liveness as L
        with tempfile.TemporaryDirectory() as d:
            good, bad, other = Path(d, "g.mp4"), Path(d, "b.mp4"), Path(d, "x.png")
            good.write_bytes(box(b"ftyp", b"isom") + box(b"mdat", b"x" * 50) + box(b"moov", b"y"))
            bad.write_bytes(box(b"ftyp", b"isom") + box(b"mdat", b"x" * 50)[:30])
            other.write_bytes(b"png")
            self.assertTrue(L.mp4_has_moov(good))
            self.assertFalse(L.mp4_has_moov(bad))
            self.assertIsNone(L.mp4_has_moov(other))

    def test_partial_outputs_flags_truncated_video(self):
        from agenty_core.utils import comfyui_liveness as L
        with tempfile.TemporaryDirectory() as d:
            old = Path(d, "old.png")
            old.write_bytes(b"1")
            os.utime(old, (1000, 1000))
            Path(d, "sub").mkdir()
            Path(d, "sub", "v.mp4").write_bytes(box(b"ftyp", b"isom") + box(b"mdat", b"z" * 40)[:20])
            items = L.partial_outputs(time.time() - 60, d)
            self.assertEqual([Path(i["path"]).name for i in items], ["v.mp4"])
            self.assertTrue(items[0]["truncated"])
            self.assertIn("TRUNCATED", L.describe_partial(items))


class WaitTest(unittest.TestCase):
    def setUp(self):
        from agenty_core.utils import comfyui_client, comfyui_progress as P
        self.P, self.client = P, DeadClient()
        self.ws = ChattyWS()
        fake = type(sys)("websockets")
        fake.connect = lambda url, **kw: self.ws
        self.enterContext(mock.patch.dict(sys.modules, {"websockets": fake}))
        self.enterContext(mock.patch.object(comfyui_client, "get_client", lambda: self.client))
        self.enterContext(mock.patch.object(P, "_CHECK_EVERY", 0.2))
        self.enterContext(mock.patch.object(P, "_HEARTBEAT", 3600))
        self.enterContext(mock.patch.object(P, "_POLL_EVERY", 0.1))
        self.enterContext(mock.patch.dict(os.environ, {"AGENTY_COMFY_DEAD_AFTER_S": "0.5"}))

    def _run(self, limit=8):
        async def go():
            out = []
            gen = self.P.stream_comfyui_job("p1", "c1", console=False)
            try:
                async for ev in gen:
                    out.append(ev)
                    if isinstance(ev, dict):
                        break
            finally:
                await gen.aclose()
            return out
        return asyncio.run(asyncio.wait_for(go(), limit))

    def test_comfyui_dying_mid_run_ends_the_wait(self):
        self.ws = ChattyWS(fail_after=2)         # socket drops...

        async def die():
            await asyncio.sleep(0.1)
            self.client.dead = True              # ...and the API refuses
        with mock.patch.object(self.P, "_CHECK_EVERY", 0.2):
            async def go():
                asyncio.ensure_future(die())
                out = []
                gen = self.P.stream_comfyui_job("p1", "c1", console=False)
                try:
                    async for ev in gen:
                        out.append(ev)
                        if isinstance(ev, dict):
                            break
                finally:
                    await gen.aclose()
                return out
            out = asyncio.run(asyncio.wait_for(go(), 8))
        res = out[-1]
        self.assertTrue(res.get("comfyui_stopped"), res)
        self.assertIn("stopped responding / crashed", res["error"])
        self.assertEqual(res["details"]["reason"], "comfyui_unreachable")

    def test_a_hiccup_shorter_than_the_limit_is_survived(self):
        self.client.dead = True
        with mock.patch.dict(os.environ, {"AGENTY_COMFY_DEAD_AFTER_S": "30"}):
            with self.assertRaises(asyncio.TimeoutError):
                self._run(limit=1.5)

    def test_a_running_job_that_goes_silent_is_given_up_on(self):
        # Socket only carries other jobs' traffic; ours is "running" but says nothing.
        with mock.patch.dict(os.environ, {"AGENTY_COMFY_STALL_MIN": "0.01"}):   # 0.6 s
            res = self._run()[-1]
        self.assertEqual(res["details"]["reason"], "comfyui_stalled")
        self.assertIn("hung", res["error"])

    def test_stall_cap_is_off_with_zero(self):
        with mock.patch.dict(os.environ, {"AGENTY_COMFY_STALL_MIN": "0"}):
            with self.assertRaises(asyncio.TimeoutError):
                self._run(limit=1.5)

    def test_a_job_that_reports_progress_is_not_stalled(self):
        class Ours(ChattyWS):
            async def recv(self):
                import json
                await asyncio.sleep(0.05)
                return json.dumps({"type": "progress", "data": {"prompt_id": "p1", "value": 1, "max": 9}})
        self.ws = Ours()
        with mock.patch.dict(os.environ, {"AGENTY_COMFY_STALL_MIN": "0.01"}):
            with self.assertRaises(asyncio.TimeoutError):
                self._run(limit=1.5)


if __name__ == "__main__":
    unittest.main()
