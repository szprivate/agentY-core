"""Downloads as jobs waited on in slices, so a message reaches the agent meanwhile."""

import json
import os
import threading
import time
import unittest
from unittest import mock

from agenty_core.tools import huggingface as hf


class _Fake:
    """Stands in for the transfer: grows `done` until released."""

    def __init__(self, total=10_000_000_000, quick=False):
        self.total, self.quick = total, quick
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, model_id, filename, node_class_type="", destination_folder="",
                 subfolder="", job=None):
        self.calls += 1
        job.update(to=f"D:/models/{filename}", total=self.total, done=0, resumed_from=0)
        if not self.quick:
            while not self.release.wait(0.02):
                job["done"] = min(self.total, job["done"] + 100_000_000)
        return json.dumps({"ok": True, "path": f"D:/models/{filename}", "size_mb": 1})


class DownloadsInSlices(unittest.TestCase):

    def setUp(self):
        hf._jobs.clear()
        self.addCleanup(hf._jobs.clear)
        env = mock.patch.dict(os.environ, {"AGENTY_DOWNLOAD_WAIT_S": "0.3"})
        env.start()
        self.addCleanup(env.stop)

    def fake(self, **kw):
        f = _Fake(**kw)
        p = mock.patch.object(hf, "_download_blocking", side_effect=f)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(f.release.set)
        return f

    def test_a_small_file_comes_back_done(self):
        self.fake(quick=True)
        out = json.loads(hf.download_hf_model("org/repo", "small.safetensors"))
        self.assertTrue(out["ok"])
        self.assertNotIn("status", out)

    def test_a_big_one_reports_progress_and_the_wait_picks_it_up(self):
        f = self.fake()
        out = json.loads(hf.download_hf_model("org/repo", "big.safetensors", subfolder="te"))
        self.assertEqual(out["status"], "downloading")
        self.assertEqual(out["to"], "D:/models/big.safetensors")
        self.assertGreater(out["done_gb"], 0)
        self.assertEqual(out["total_gb"], 10.0)
        self.assertIn("answer them now", out["what_to_do"])
        job_id = out["job_id"]
        # Asked again for the same file: the same job, not a second download.
        again = json.loads(hf.download_hf_model("org/repo", "big.safetensors", subfolder="te"))
        self.assertEqual(again["job_id"], job_id)
        self.assertEqual(f.calls, 1)
        self.assertEqual([j["job_id"] for j in hf.download_progress()], [job_id])
        f.release.set()
        done = json.loads(hf.wait_for_download(job_id, 5))
        self.assertTrue(done["ok"])
        self.assertEqual(done["path"], "D:/models/big.safetensors")
        self.assertEqual(hf.download_progress(), [])

    def test_waiting_without_an_id_takes_the_running_one(self):
        f = self.fake()
        hf.download_hf_model("org/repo", "big.safetensors")
        out = json.loads(hf.wait_for_download("", 1))
        self.assertEqual(out["status"], "downloading")
        f.release.set()
        self.assertTrue(json.loads(hf.wait_for_download("", 5))["ok"])
        self.assertFalse(json.loads(hf.wait_for_download("nope", 1))["ok"])

    def test_code_that_needs_the_file_waits_to_the_end(self):
        f = self.fake()
        threading.Timer(0.8, f.release.set).start()
        out = json.loads(hf.download_to_completion("org/repo", "big.safetensors"))
        self.assertTrue(out["ok"])

    def test_the_host_hears_when_it_ends(self):
        f = self.fake()
        heard = []
        hf.add_download_listener(heard.append)
        self.addCleanup(lambda: hf._listeners.remove(heard.append))
        hf.download_hf_model("org/repo", "big.safetensors")
        f.release.set()
        deadline = time.time() + 5
        while not heard and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(heard[0]["filename"], "big.safetensors")
        self.assertTrue(json.loads(heard[0]["result"])["ok"])


if __name__ == "__main__":
    unittest.main()
