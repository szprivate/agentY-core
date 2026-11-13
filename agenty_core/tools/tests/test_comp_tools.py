"""edit_comp / render_comp (the viewer's Comp routes, ComfyUI mocked) and deadline_farm.submit_comp (a fake
deadlinecommand): a comp's Write nodes as a CommandLine job running the viewer's comp_render.py in chunks."""
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import comp as ct
from agenty_core.tools import deadline_farm as df
from agenty_core.tools.tests.test_deadline_farm import Base as FarmBase, FakeFarm

REQ = {"comp": "//share/shot/shot.comp.json", "writes": ["Write1", "Write2"], "frames": "0-14", "frame_offset": 2,
       "chunk": 5, "first": -2, "last": 12, "pool": "", "group": "", "priority": -1, "name": "shot.comp.json",
       "python": "/here/python", "script": "/here/scripts/comp_render.py", "outputs": ["//share/shot/out"]}


class CompFarm(FakeFarm):
    def __call__(self, args, timeout=60):
        a = [str(x) for x in args]
        if not a[0].startswith("-") and len(a) == 2:
            self.submitted.append({"job": Path(a[0]).read_text(encoding="utf-8"), "plugin": Path(a[1]).read_text(encoding="utf-8")})
            return "Result=Success\nJobID=comp42"
        return super().__call__(args, timeout)


class SubmitComp(FarmBase):
    def setUp(self):
        super().setUp()
        self.farm = CompFarm()
        q = mock.patch.object(df, "_dc", self.farm)
        q.start()
        self.addCleanup(q.stop)

    def test_a_commandline_job_in_chunks_with_the_offset(self):
        with mock.patch.dict(os.environ, {"DEADLINE_COMP_PYTHON": r"\\share\comfy\python.exe",
                                          "DEADLINE_COMP_SCRIPT": r"\\share\viewer\scripts\comp_render.py"}):
            r = df.submit_comp(REQ)
        self.assertEqual((r["job_id"], r["tasks"], r["pool"], r["frames"]), ("comp42", 3, "comfyui", "0-14"))
        job, plug = self.farm.submitted[0]["job"], self.farm.submitted[0]["plugin"]
        self.assertIn("Plugin=CommandLine", job)
        self.assertIn("Frames=0-14", job)
        self.assertIn("ChunkSize=5", job)
        self.assertIn("OutputDirectory0=//share/shot/out", job)
        self.assertIn(r"Executable=\\share\comfy\python.exe", plug)
        self.assertIn('--writes "Write1,Write2" --frames <STARTFRAME>-<ENDFRAME> --frame-offset 2 --quiet', plug)
        self.assertIn(r'"\\share\viewer\scripts\comp_render.py" "//share/shot/shot.comp.json"', plug)

    def test_refused_without_a_pool_or_with_bad_frames(self):
        with self.assertRaises(df.FarmError):
            df.submit_comp(dict(REQ, pool="nope"))
        with self.assertRaises(df.FarmError):
            df.submit_comp(dict(REQ, frames="-2-12"))
        self.assertEqual(self.farm.submitted, [])


class Tools(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()
        p = mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=self.client)
        p.start()
        self.addCleanup(p.stop)

    def test_read_change_and_the_open_comp(self):
        self.client.get.return_value = {"format": {"width": 64}, "nodes": []}
        got = json.loads(ct.edit_comp(path="D:/c/shot.comp.json"))
        self.assertEqual(got["comp"]["format"]["width"], 64)
        self.client.get.assert_called_with("/imageviewer/comp_summary", params={"path": "D:/c/shot.comp.json"})
        self.client.post.return_value = {"path": "D:/c/shot.comp.json", "ids": ["Blur1"], "comp": {}}
        with mock.patch.object(ct, "_open_comp", return_value="D:/c/shot.comp.json"):
            got = json.loads(ct.edit_comp(ops='[{"op": "add", "type": "blur"}]'))
        self.assertEqual(got["ids"], ["Blur1"])
        body = self.client.post.call_args.kwargs["json_data"]
        self.assertEqual(body["ops"], [{"op": "add", "type": "blur"}])
        self.assertNotIn("create", body)
        with mock.patch.object(ct, "_open_comp", return_value=""):
            self.assertIn("no comp is open", ct.edit_comp())
        self.assertIn("comp.json", ct.edit_comp(path="D:/c/x.png"))

    def test_render_here_waits_for_the_files(self):
        self.client.post.return_value = {"job": "j1"}
        self.client.get.side_effect = [{"state": "running", "done": 1, "total": 2},
                                       {"state": "done", "result": {"writes": [{"name": "main", "frames": 2}], "notes": []}}]
        with mock.patch.object(ct.time, "sleep"):
            got = json.loads(ct.render_comp(path="D:/c/shot.comp.json", writes="main", handles=2))
        self.assertEqual(got["writes"][0]["name"], "main")
        self.assertEqual(self.client.post.call_args.kwargs["json_data"], {"path": "D:/c/shot.comp.json", "writes": ["main"], "handles": 2})

    def test_render_on_the_farm_is_planned_by_the_viewer_and_submitted_here(self):
        self.client.post.return_value = {"request": REQ, "warnings": ["the comp is on this machine's own disk"]}
        with mock.patch.object(df, "submit_comp", return_value={"ok": True, "status": "submitted", "job_id": "x"}) as sub:
            got = json.loads(ct.render_comp(path="D:/c/shot.comp.json", farm=True, chunk=5))
        sub.assert_called_once_with(REQ)
        self.assertEqual(got["job_id"], "x")
        self.assertEqual(got["warnings"], ["the comp is on this machine's own disk"])
        self.assertTrue(self.client.post.call_args.kwargs["json_data"]["dry_run"])


if __name__ == "__main__":
    unittest.main()
