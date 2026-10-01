"""Stop reaches a model download.

A download runs in a worker thread that cancelling the turn cannot reach, so
pressing Stop left a multi-GB fetch running and the turn waiting on it.

    python -m unittest agenty_core.tools.tests.test_download_stop
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import huggingface as hf


class SlowResponse:
    """200 chunks, 50 ms apart — ten seconds of download."""

    status_code = 200
    headers = {"content-length": str(200 * 1024)}

    def __init__(self):
        self.closed = False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        for _ in range(200):
            if self.closed:
                raise ConnectionError("response closed")
            time.sleep(0.05)
            yield b"x" * 1024

    def close(self):
        self.closed = True


class DownloadStopTest(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        for name, value in (("_resolve_download_dir", lambda *a, **k: (self.dir, "test")),
                            ("_ensure_not_c_drive", lambda d, s: (d, s)),
                            ("get_secret", lambda *a, **k: None),
                            ("_push_progress", lambda *a, **k: None),
                            ("_free_gb", lambda *a, **k: 10_000.0),
                            ("_refresh_model_cache", lambda *a, **k: None)):
            self.enterContext(mock.patch.object(hf, name, value, create=True))
        self.resp = SlowResponse()
        self.enterContext(mock.patch.object(hf.requests, "get", lambda *a, **k: self.resp))
        hf.clear_download_cancel()
        self.addCleanup(hf.clear_download_cancel)
        self.download = getattr(hf.download_hf_model, "func",
                                getattr(hf.download_hf_model, "_tool_func", hf.download_hf_model))

    def test_stop_ends_a_running_download_and_keeps_the_partial(self):
        out = {}
        t = threading.Thread(target=lambda: out.update(r=self.download("org/model", "big.safetensors")))
        t.start()
        time.sleep(0.4)
        self.assertEqual(hf.cancel_downloads(), 1, "the running download was found")
        t.join(3)
        self.assertFalse(t.is_alive(), "Stop did not end the download")
        result = json.loads(out["r"])
        self.assertTrue(result.get("cancelled"), result)
        self.assertFalse((self.dir / "big.safetensors").exists())
        partial = self.dir / "big.safetensors.downloading"
        self.assertTrue(partial.exists() and partial.stat().st_size > 0, "the partial is kept to resume")

    def test_a_stop_before_the_download_starts_is_honoured(self):
        hf.cancel_downloads()
        result = json.loads(self.download("org/model", "big.safetensors"))
        self.assertTrue(result.get("cancelled"), result)

    def test_the_next_turn_downloads_again(self):
        hf.cancel_downloads()
        hf.clear_download_cancel()
        self.resp.iter_content = lambda chunk_size=1: iter([b"x" * 1024] * 3)
        result = json.loads(self.download("org/model", "small.safetensors"))
        self.assertTrue(result.get("ok"), result)


if __name__ == "__main__":
    unittest.main()
