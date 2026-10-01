"""Waiting for a ComfyUI job: as long as it takes, but never blind.

Renders can take an hour, so the wait has no cap. What it must not do is miss
its job's ending or go quiet. The fallback check of the job's state was timed
from the last socket message of ANY job, so while ComfyUI was busy with other
work it never ran: a job whose ending the socket did not report kept the turn
waiting, with nothing on screen, for as long as ComfyUI stayed busy. And a
dropped socket ended the wait with an error while the job itself ran on.

    python -m unittest agenty_core.utils.tests.test_comfyui_long_wait
"""

import asyncio
import json
import sys
import unittest
from unittest import mock

DONE = {"p1": {"status": {"status_str": "success", "completed": True},
               "outputs": {"9": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}}}}


class Client:
    base_url = "http://127.0.0.1:8188"
    api_key = ""

    def __init__(self):
        self.history, self.running, self.pending = {}, ["p1"], []

    def get(self, path, **kw):
        if path.startswith("/history"):
            return self.history
        if path == "/queue":
            return {"queue_running": [[0, p] for p in self.running],
                    "queue_pending": [[1, p] for p in self.pending]}
        return {}


class ChattyWS:
    """Another job's progress, forever — and nothing about ours."""

    def __init__(self, fail_after=None):
        self.n, self.fail_after = 0, fail_after

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def recv(self):
        await asyncio.sleep(0.05)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise ConnectionError("socket dropped")
        return json.dumps({"type": "progress", "data": {"prompt_id": "other", "value": self.n, "max": 10**6}})


class LongWaitTest(unittest.TestCase):

    def setUp(self):
        from agenty_core.utils import comfyui_client, comfyui_progress as P
        self.P, self.client = P, Client()
        self.ws = ChattyWS()
        fake = type(sys)("websockets")
        fake.connect = lambda url, **kw: self.ws
        self.enterContext(mock.patch.dict(sys.modules, {"websockets": fake}))
        self.enterContext(mock.patch.object(comfyui_client, "get_client", lambda: self.client))
        self.enterContext(mock.patch.object(P, "_CHECK_EVERY", 0.3, create=True))
        self.enterContext(mock.patch.object(P, "_HEARTBEAT", 3600, create=True))
        self.enterContext(mock.patch.object(P, "_POLL_EVERY", 0.1, create=True))

    def _run(self, finish_after: float, limit: float = 5):
        async def go():
            async def finish():
                await asyncio.sleep(finish_after)
                self.client.history, self.client.running = DONE, []
            asyncio.ensure_future(finish())
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

    def test_an_unreported_ending_is_noticed_while_comfyui_is_busy(self):
        out = self._run(finish_after=0.4)
        self.assertIn("history", out[-1])

    def test_a_long_job_says_where_it_stands(self):
        with mock.patch.object(self.P, "_HEARTBEAT", 0.3), mock.patch.object(self.P, "_CHECK_EVERY", 1.0):
            self.client.running, self.client.pending = ["x"], ["p1"]
            out = self._run(finish_after=1.2)
        beats = [e for e in out if isinstance(e, str) and e.startswith("⏳")]
        self.assertTrue(beats, out)
        self.assertIn("1 job(s) ahead", beats[0])

    def test_a_dropped_socket_does_not_end_a_running_job(self):
        self.ws = ChattyWS(fail_after=2)
        out = self._run(finish_after=0.6)
        self.assertIn("history", out[-1], "followed by polling, not failed")

    def test_no_socket_at_all_is_still_followed(self):
        def refuse(url, **kw):
            raise OSError("connection refused")
        sys.modules["websockets"].connect = refuse
        out = self._run(finish_after=0.4)
        self.assertIn("history", out[-1])

    def test_there_is_no_cap_unless_asked_for(self):
        import inspect
        self.assertIsNone(inspect.signature(self.P.stream_comfyui_job).parameters["timeout"].default)


if __name__ == "__main__":
    unittest.main()
