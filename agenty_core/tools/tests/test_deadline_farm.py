"""deadline_farm offline: config, the dedicated ComfyUI pool guard, submit files (no keys in them), input staging,
status / outputs, and the Web Service transport - all against a fake deadlinecommand / fake HTTP."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import deadline_farm as df

PROMPT = {"1": {"class_type": "LoadImage", "inputs": {"image": "cat.png"}},
          "2": {"class_type": "KSampler", "inputs": {"seed": 5, "model": ["3", 0]}}}


class FakeFarm:
    """A fake deadlinecommand: pools, workers, a job store; records submitted files."""

    def __init__(self):
        self.members = {"comfyui": ["W1", "W2"]}
        self.states = {"W1": "Idle", "W2": "Offline"}
        self.submitted = []
        self.job = {}
        self.error_file = None

    @property
    def pools(self):
        return ["none"] + list(self.members)

    def __call__(self, args, timeout=60):
        a = [str(x) for x in args]
        c = a[0]
        if c == "-GetPoolNames":
            return "\n".join(self.pools)
        if c == "-GetGroupNames":
            return "none\ngpu"
        if c == "-GetSlaveNamesInPool":
            return "\n".join(self.members.get(a[1], []))
        if c == "-GetSlaveInfo":
            return "SlaveState=" + self.states.get(a[1], "Offline")
        if c == "-GetSlaves":
            return "\n\n".join("Name=%s\nSlaveState=%s" % (n, s) for n, s in self.states.items())
        if c == "-GetRepositoryRoot":
            return r"\\farm\deadline"
        if c == "-GetJob":
            return "\n".join("%s=%s" % (k, v) for k, v in self.job.items())
        if c == "-GetJobErrorReportFilenames":
            return self.error_file or ""
        if not c.startswith("-"):                    # a submit: job info, plugin info, aux
            ji, pi, aux = (Path(x) for x in a)
            self.submitted.append({"job": ji.read_text(encoding="utf-8"), "plugin": pi.read_text(encoding="utf-8"),
                                   "aux": aux.read_text(encoding="utf-8")})
            return "Result=Success\nJobID=abc123"
        raise AssertionError(a)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "share"
        env = {"DEADLINE_OUTPUT_DIR": str(self.out), "DEADLINE_COMFYUI_PATH": r"D:\ComfyUI",
               "DEADLINE_WEBSERVICE_URL": "", "DEADLINE_COMFYUI_POOL": ""}
        p = mock.patch.dict(os.environ, env)
        p.start()
        self.addCleanup(p.stop)
        self.farm = FakeFarm()
        for target, val in (("_dc", self.farm), ("_dc_exe", lambda: "deadlinecommand")):
            q = mock.patch.object(df, target, val)
            q.start()
            self.addCleanup(q.stop)
        self.inp = Path(self.tmp.name) / "input"
        self.inp.mkdir()
        (self.inp / "cat.png").write_bytes(b"png")
        q = mock.patch.object(df, "_comfy_dirs", lambda: {"input_dir": str(self.inp), "output_dir": ""})
        q.start()
        self.addCleanup(q.stop)


class Config(unittest.TestCase):
    def test_not_configured(self):
        with mock.patch.object(df, "_dc_exe", lambda: None), mock.patch.object(df, "_ws_url", lambda: ""):
            for fn in (df._info_tool, lambda: df._submit_tool(PROMPT), lambda: df._status_tool("x")):
                r = json.loads(fn())
                self.assertEqual(r["error_kind"], "config")
                self.assertIn("Deadline not configured", r["error"])


class Pool(Base):
    def test_default_pool_name(self):
        self.assertEqual(df.comfyui_pool(), "comfyui")
        with mock.patch.dict(os.environ, {"DEADLINE_COMFYUI_POOL": "gpu_comfy"}):
            self.assertEqual(df.comfyui_pool(), "gpu_comfy")

    def test_info_reports_pool(self):
        r = json.loads(df._info_tool())
        self.assertTrue(r["ready"])
        self.assertEqual(r["comfyui_pool"]["worker_count"], 2)
        self.assertEqual(r["comfyui_pool"]["online_count"], 1)
        self.assertEqual(r["comfyui_pool"]["workers"][0], {"name": "W1", "state": "Idle"})

    def test_info_pool_missing(self):
        self.farm.members = {}
        r = json.loads(df._info_tool())
        self.assertFalse(r["ready"])
        self.assertFalse(r["comfyui_pool"]["exists"])
        self.assertIn("-AddPool comfyui", r["issues"][0])

    def test_submit_refused_without_pool_or_workers(self):
        self.farm.members = {}
        r = json.loads(df._submit_tool(PROMPT))
        self.assertEqual(r["error_kind"], "config")
        self.assertIn("does not exist", r["error"])
        self.assertIn("SetPoolsForSlave", r["error"])
        self.farm.members = {"comfyui": []}
        r = json.loads(df._submit_tool(PROMPT))
        self.assertIn("no workers", r["error"])
        self.assertEqual(self.farm.submitted, [])


class Submit(Base):
    def test_submit_goes_to_comfyui_pool(self):
        r = json.loads(df._submit_tool(PROMPT, name="Cat test"))
        self.assertEqual(r["status"], "submitted")
        self.assertEqual(r["job_id"], "abc123")
        self.assertEqual(r["pool"], "comfyui")
        sub = self.farm.submitted[0]
        self.assertIn("Plugin=ComfyUI", sub["job"])
        self.assertIn("Pool=comfyui", sub["job"])
        self.assertIn("ConcurrentTasks=1", sub["job"])
        self.assertIn("ComfyUIPath=D:\\ComfyUI", sub["plugin"])
        self.assertIn("SeedMode=keep", sub["plugin"])
        self.assertIn("Input0=cat.png", sub["plugin"])
        self.assertEqual(json.loads(sub["aux"])["1"]["inputs"]["image"], "cat.png")
        self.assertTrue((Path(r["output_dir"]) / "_inputs" / "cat.png").is_file())

    def test_explicit_pool_override(self):
        self.farm.members["big"] = ["W9"]
        r = json.loads(df._submit_tool(PROMPT, pool="big"))
        self.assertEqual(r["pool"], "big")
        self.assertIn("Pool=big", self.farm.submitted[0]["job"])

    def test_batch_randomizes_seeds_and_no_keys_in_files(self):
        wf = {"prompt": PROMPT, "extra_data": {"api_key_comfy_org": "SECRET-KEY"}}
        with mock.patch.dict(os.environ, {"COMFY_ORG_API_KEY": "SECRET-KEY"}):
            r = json.loads(df._submit_tool(json.dumps(wf), batch=4))
        sub = self.farm.submitted[0]
        self.assertEqual(r["tasks"], 4)
        self.assertIn("Frames=0-3", sub["job"])
        self.assertIn("SeedMode=randomize", sub["plugin"])
        for blob in sub.values():
            self.assertNotIn("SECRET-KEY", blob)

    def test_inputs_absolute_and_annotated(self):
        other = Path(self.tmp.name) / "elsewhere"
        other.mkdir()
        (other / "ref.jpg").write_bytes(b"jpg")
        wf = {"1": {"class_type": "LoadImage", "inputs": {"image": str(other / "ref.jpg")}},
              "2": {"class_type": "LoadImage", "inputs": {"image": "cat.png [input]"}},
              "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "a photo.png of nothing"}}}
        r = json.loads(df._submit_tool(wf))
        aux = json.loads(self.farm.submitted[0]["aux"])
        self.assertEqual(aux["1"]["inputs"]["image"], "ref.jpg")
        self.assertEqual(aux["2"]["inputs"]["image"], "cat.png")
        self.assertEqual(aux["3"]["inputs"]["text"], "a photo.png of nothing")
        self.assertEqual(sorted(r["inputs_staged"]), ["cat.png", "ref.jpg"])

    def test_bad_workflows(self):
        self.assertEqual(json.loads(df._submit_tool({"nodes": [], "links": []}))["error_kind"], "args")
        self.assertEqual(json.loads(df._submit_tool("not json"))["error_kind"], "args")
        self.assertEqual(json.loads(df._submit_tool({"1": {"inputs": {}}}))["error_kind"], "args")

    def test_needs_output_dir(self):
        with mock.patch.dict(os.environ, {"DEADLINE_OUTPUT_DIR": ""}):
            r = json.loads(df._submit_tool(PROMPT))
        self.assertEqual(r["error_kind"], "config")


class Status(Base):
    def test_completed_with_outputs(self):
        d = Path(self.tmp.name) / "o"
        d.mkdir()
        (d / ".agenty_farm_abc123_0.json").write_text(json.dumps({"files": [str(d / "a_t0000.png")]}))
        self.farm.job = {"JobStatus": "Completed", "TaskCount": 1, "CompletedChunks": 1, "ErrorReports": 0,
                         "OutputDirectories": str(d), "Name": "x", "Pool": "comfyui"}
        r = json.loads(df._status_tool("abc123"))
        self.assertEqual(r["state"], "Completed")
        self.assertEqual(r["progress"], 1.0)
        self.assertEqual(r["outputs"], [str(d / "a_t0000.png")])

    def test_failed_with_error_text(self):
        rep = Path(self.tmp.name) / "rep.txt"
        rep.write_text("header\nError Message: ComfyUI execution_error in KSampler: OOM\nmore")
        self.farm.error_file = str(rep)
        self.farm.job = {"Status": "Failed", "TaskCount": 2, "CompletedChunks": 0, "FailedChunks": 1,
                         "ErrorReports": 3, "OutputDirectories": "", "Name": "x"}
        r = json.loads(df._status_tool("abc123"))
        self.assertEqual(r["state"], "Failed")
        self.assertIn("OOM", r["errors"])

    def test_queued(self):
        self.farm.job = {"JobStatus": "Queued", "TaskCount": 3, "CompletedChunks": 0, "QueuedChunks": 3,
                         "RenderingChunks": 0, "OutputDirectories": ""}
        r = json.loads(df._status_tool("abc123", 0))
        self.assertEqual(r["state"], "Queued")
        self.assertIn("free worker", r["note"])

    def test_unknown_job(self):
        self.farm.job = {}
        self.assertEqual(json.loads(df._status_tool("nope"))["error_kind"], "args")


class WebService(Base):
    URL = {"DEADLINE_WEBSERVICE_URL": "http://farm:8082"}

    def test_ws_used_when_reachable_and_falls_back_when_not(self):
        def ws(method, path, **kw):
            return ["none", "ws-pool"]

        def down(*a, **k):
            raise df.FarmError("down")

        with mock.patch.dict(os.environ, self.URL), mock.patch.object(df, "_ws", ws):
            self.assertEqual(df._pools(), ["none", "ws-pool"])
        with mock.patch.dict(os.environ, self.URL), mock.patch.object(df, "_ws", down):
            self.assertEqual(df._pools(), ["none", "comfyui"])     # deadlinecommand answered

    def test_ws_submit_body(self):
        seen = {}

        def ws(method, path, **kw):
            if path == "/api/pools":
                return ["comfyui"]
            seen["body"] = kw["json"]
            return {"_id": "wsjob1"}

        ok = {"pool": "comfyui", "exists": True, "workers": [{"name": "W", "state": "Idle"}],
              "worker_count": 1, "online_count": 1}
        with mock.patch.dict(os.environ, self.URL), mock.patch.object(df, "_ws", ws), \
                mock.patch.object(df, "_pool_status", lambda p: ok):
            r = json.loads(df._submit_tool(PROMPT))
        self.assertEqual(r["job_id"], "wsjob1")
        self.assertEqual(seen["body"]["JobInfo"]["Pool"], "comfyui")
        self.assertEqual(seen["body"]["JobInfo"]["Plugin"], "ComfyUI")


if __name__ == "__main__":
    unittest.main()
