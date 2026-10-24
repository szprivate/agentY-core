"""farm_run: routing a normal run to Deadline - the switch, the reference, the stream's protocol, errors, cancel."""

import asyncio
import os
import unittest
from unittest import mock

from agenty_core.tools import deadline_farm as df
from agenty_core.utils import farm_run as fr


def run(coro):
    return asyncio.run(coro)


async def collect(gen):
    out = []
    async for x in gen:
        out.append(x)
    return out


class Switch(unittest.TestCase):
    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {"DEADLINE_ROUTE_RUNS": ""}):
            self.assertFalse(fr.enabled())
            self.assertFalse(fr.enabled(None))

    def test_the_hosts_setting_turns_it_on(self):
        with mock.patch.dict(os.environ, {"DEADLINE_ROUTE_RUNS": ""}):
            self.assertTrue(fr.enabled(True))

    def test_the_environment_wins_both_ways(self):
        with mock.patch.dict(os.environ, {"DEADLINE_ROUTE_RUNS": "1"}):
            self.assertTrue(fr.enabled(False))
        with mock.patch.dict(os.environ, {"DEADLINE_ROUTE_RUNS": "0"}):
            self.assertFalse(fr.enabled(True))

    def test_references(self):
        self.assertTrue(fr.is_farm_ref("farm:abc"))
        self.assertFalse(fr.is_farm_ref("1234-prompt-id"))
        self.assertEqual(fr.job_id_of("farm:abc"), "abc")
        self.assertEqual(fr.job_id_of("plain"), "plain")


class Submit(unittest.TestCase):
    def test_returns_a_farm_reference_for_one_task(self):
        with mock.patch.object(df, "submit", return_value={"job_id": "J1"}) as m:
            ref = fr.submit({"1": {"class_type": "X", "inputs": {}}}, name="poster")
        self.assertEqual(ref, "farm:J1")
        kw = m.call_args.kwargs
        self.assertEqual((kw["batch"], kw["randomize_seed"], kw["name"]), (1, False, kw["name"]))
        self.assertRegex(kw["name"], r"^poster [0-9a-f]{6}$")

    def test_a_refusal_becomes_a_runtime_error_the_executor_already_handles(self):
        with mock.patch.object(df, "submit", side_effect=df.FarmError("pool 'comfyui' does not exist", "config")):
            with self.assertRaises(RuntimeError) as cm:
                fr.submit({"1": {"class_type": "X", "inputs": {}}})
        self.assertIn("does not exist", str(cm.exception))


class Stream(unittest.TestCase):
    def _status(self, *states):
        it = iter(states)
        return mock.patch.object(df, "job_status", side_effect=lambda jid, w=0: next(it))

    def test_queued_then_rendering_then_history(self):
        done = {"state": "Completed", "outputs": [r"\\farm\out\a_t0000.png", r"\\farm\out\b_t0000.mp4"]}
        with self._status({"state": "Queued", "tasks": {}}, {"state": "Rendering", "progress": 0.5,
                                                              "tasks": {"completed": 0, "total": 1}}, done):
            out = run(collect(fr.stream("farm:J1", poll_s=0)))
        lines = [x for x in out if isinstance(x, str)]
        self.assertTrue(any("queued" in l for l in lines))
        self.assertTrue(any("rendering" in l and "50%" in l for l in lines))
        final = out[-1]
        self.assertEqual(len([x for x in out if isinstance(x, dict)]), 1, "exactly one terminal dict")
        files = [i["filename"] for i in final["history"]["farm:J1"]["outputs"]["farm"]["images"]]
        self.assertEqual(files, done["outputs"])

    def test_the_history_is_what_the_executors_extract(self):
        h = fr.to_history("farm:J1", ["C:/o/a.png"])
        # same walk as executor._extract_output_files
        found = []
        for entry in h.values():
            for node_id, node_out in entry.get("outputs", {}).items():
                for key in ("images", "gifs", "videos", "audio"):
                    for item in node_out.get(key, []):
                        found.append((node_id, item["filename"]))
        self.assertEqual(found, [("farm", "C:/o/a.png")])
        self.assertEqual(fr.to_history("farm:J1", [])["farm:J1"]["outputs"], {})

    def test_failure_carries_the_node_and_exception_for_the_heal_loop(self):
        err = ("Error Message: ComfyUI execution_error in KSampler (node 3): "
               "torch.OutOfMemoryError: CUDA out of memory")
        with self._status({"state": "Failed", "errors": err}):
            out = run(collect(fr.stream("farm:J1", poll_s=0)))
        final = out[-1]
        self.assertEqual(final["error"], "Deadline job failed")
        d = final["details"]
        self.assertEqual((d["node_id"], d["node_type"], d["exception_type"]),
                         ("3", "KSampler", "torch.OutOfMemoryError"))
        self.assertIn("out of memory", d["exception_message"])
        self.assertEqual(d["job_id"], "J1")

    def test_an_error_text_that_is_not_the_plugins_still_reaches_the_repair(self):
        d = fr.error_details("ComfyUI not installed on RND03", "J2")
        self.assertEqual(d["exception_message"], "ComfyUI not installed on RND03")
        self.assertEqual(d["node_id"], "")

    def test_a_job_suspended_in_monitor_is_an_interruption_not_a_defect(self):
        with self._status({"state": "Suspended"}):
            out = run(collect(fr.stream("farm:J1", poll_s=0)))
        self.assertTrue(out[-1]["interrupted"])

    def test_a_few_failed_polls_are_forgiven_many_are_not(self):
        calls = {"n": 0}

        def flaky(jid, w=0):
            calls["n"] += 1
            if calls["n"] <= 3:
                raise df.FarmError("share busy")
            return {"state": "Completed", "outputs": ["x.png"]}

        with mock.patch.object(df, "job_status", side_effect=flaky):
            out = run(collect(fr.stream("farm:J1", poll_s=0)))
        self.assertIn("history", out[-1])
        with mock.patch.object(df, "job_status", side_effect=df.FarmError("down")):
            out = run(collect(fr.stream("farm:J1", poll_s=0)))
        self.assertIn("lost contact", out[-1]["error"])

    def test_a_time_limit_suspends_the_job(self):
        with mock.patch.object(df, "job_status", return_value={"state": "Queued", "tasks": {}}), \
                mock.patch.object(df, "cancel_job") as c:
            out = run(collect(fr.stream("farm:J1", poll_s=0.01, max_wait_s=0.05)))
        self.assertIn("did not finish", out[-1]["error"])
        c.assert_called_once_with("J1")

    def test_pressing_stop_suspends_the_job(self):
        async def go():
            with mock.patch.object(df, "job_status", return_value={"state": "Queued", "tasks": {}}), \
                    mock.patch.object(df, "cancel_job") as c:
                async def consume():
                    async for _ in fr.stream("farm:J1", poll_s=0.01):
                        pass
                t = asyncio.ensure_future(consume())
                await asyncio.sleep(0.1)
                t.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await t
                return c
        c = run(go())
        c.assert_called_once_with("J1")

    def test_closing_after_a_finished_job_leaves_it_alone(self):
        async def go():
            with mock.patch.object(df, "job_status", return_value={"state": "Completed", "outputs": ["a.png"]}), \
                    mock.patch.object(df, "cancel_job") as c:
                gen = fr.stream("farm:J1", poll_s=0)
                async for ev in gen:
                    if isinstance(ev, dict):
                        break
                await gen.aclose()          # what the executors do after the terminal dict
                return c
        c = run(go())
        c.assert_not_called()


class Fetch(unittest.TestCase):
    def test_unset_means_the_files_stay_on_the_share(self):
        with mock.patch.dict(os.environ, {"DEADLINE_FETCH_DIR": ""}):
            self.assertEqual(fr.fetch_outputs(["//farm/o/a.png"], "J1"), ["//farm/o/a.png"])

    def test_set_copies_them_next_to_the_other_outputs(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            share, local = Path(d) / "share", Path(d) / "local"
            share.mkdir()
            (share / "a.png").write_bytes(b"x")
            with mock.patch.dict(os.environ, {"DEADLINE_FETCH_DIR": str(local)}):
                out = fr.fetch_outputs([str(share / "a.png"), str(share / "gone.png")], "J1")
            self.assertEqual(out[0], str(local / "J1" / "a.png"))
            self.assertEqual(Path(out[0]).read_bytes(), b"x")
            self.assertEqual(out[1], str(share / "gone.png"))     # not copyable: used in place

    def test_the_stream_hands_on_the_fetched_paths(self):
        with mock.patch.object(df, "job_status", return_value={"state": "Completed", "outputs": ["a.png"]}),                 mock.patch.object(fr, "fetch_outputs", return_value=["/local/a.png"]):
            out = run(collect(fr.stream("farm:J1", poll_s=0)))
        files = [i["filename"] for i in out[-1]["history"]["farm:J1"]["outputs"]["farm"]["images"]]
        self.assertEqual(files, ["/local/a.png"])


class Cancel(unittest.TestCase):
    def test_suspends_through_deadlinecommand(self):
        calls = []
        with mock.patch.object(df, "_require_configured"), mock.patch.object(df, "_use_ws", return_value=False), \
                mock.patch.object(df, "_dc", side_effect=lambda a, timeout=60: calls.append(a) or "ok"):
            r = df.cancel_job("J9")
        self.assertEqual(calls, [["-SuspendJob", "J9"]])
        self.assertTrue(r["suspended"])

    def test_a_farm_that_does_not_answer_is_reported_not_raised(self):
        with mock.patch.object(df, "_require_configured"), mock.patch.object(df, "_use_ws", return_value=False), \
                mock.patch.object(df, "_dc", side_effect=df.FarmError("nope")):
            r = df.cancel_job("J9")
        self.assertFalse(r["suspended"])

    def test_the_wait_limit_setting(self):
        with mock.patch.dict(os.environ, {"DEADLINE_MAX_WAIT_MINUTES": "10"}):
            self.assertEqual(fr.max_wait_seconds(), 600)
        with mock.patch.dict(os.environ, {"DEADLINE_MAX_WAIT_MINUTES": "x"}):
            self.assertEqual(fr.max_wait_seconds(), 240 * 60)


if __name__ == "__main__":
    unittest.main()
