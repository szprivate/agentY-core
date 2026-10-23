"""Cost checks for paid ComfyUI partner nodes (agenty_core.partner_credits).

Offline: the schemas come from a fixture of real /object_info price badges, the
balance endpoint is a stub, and nothing is ever submitted.

Runs under pytest or directly (``python test_partner_credits.py``).
"""
import json
import tempfile
import unittest
from pathlib import Path

from agenty_core import partner_credits as pc

FIX = json.loads((Path(__file__).parent / "fixtures" / "price_badges.json").read_text(encoding="utf-8"))
# Not paid at all: a local node answers price_badge None.
INFO = {**FIX, "KSampler": {"price_badge": None, "input": {"required": {}}}}


def seedance(res="720p", dur=6, video=False, task="auto", model="Seedance 2.5", ratio="16:9"):
    inputs = {"model": model, "model.resolution": res, "model.ratio": ratio,
              "model.duration": dur, "model.task_type": task, "model.prompt": "x"}
    if video:
        inputs["model.reference_videos.video_1"] = ["5", 0]
    wf = {"9": {"class_type": "ByteDance2ReferenceNodeV2", "inputs": inputs}}
    if video:
        wf["5"] = {"class_type": "LoadVideo", "inputs": {"file": "clip.mp4"}}
    return wf


class Base(unittest.TestCase):
    def setUp(self):
        pc.clear_caches()
        self.balance = {"effective_balance_cents": 5000, "prepaid_balance_cents": 5000, "currency": "usd"}
        self.balance_calls = []
        self.info_calls = []

        def http(url, headers):
            self.balance_calls.append(url)
            assert "X-API-KEY" in headers
            return dict(self.balance)

        def info(ct):
            self.info_calls.append(ct)
            return INFO.get(ct)

        pc._http_get, pc._info_provider = http, info
        pc._threshold_provider = None
        pc._sleep = lambda s: None
        self._orig_log, pc._log = pc._log, lambda entry: self.logged.append(entry)
        self.logged = []
        pc._inflight.clear()
        pc._watermark = None
        pc._confirmed_paths.clear()

    def tearDown(self):
        pc._http_get = pc._info_provider = pc._threshold_provider = None
        pc._sleep = __import__("time").sleep
        pc._listeners.clear()
        pc._log = self._orig_log
        pc.clear_caches()


class PriceEstimate(Base):
    def test_seedance_25_resolutions_no_video(self):
        got = {r: pc.estimate_workflow(seedance(r))["max_usd"] for r in ("480p", "720p", "1080p")}
        self.assertAlmostEqual(got["480p"], 0.8881, places=3)
        self.assertAlmostEqual(got["720p"], 1.9968, places=3)
        self.assertAlmostEqual(got["1080p"], 4.9126, places=3)
        est = pc.estimate_workflow(seedance("720p"))
        self.assertEqual(est["min_usd"], est["max_usd"])        # a plain usd price
        self.assertTrue(est["paid"])
        self.assertEqual(est["unknown"], [])

    def test_seedance_25_with_reference_video_is_a_range(self):
        # 720p, 6 s, one reference video wired in (an input GROUP count of 1)
        est = pc.estimate_workflow(seedance("720p", 6, video=True))
        self.assertAlmostEqual(est["min_usd"], 1.1943, places=3)
        self.assertAlmostEqual(est["max_usd"], 7.1248, places=3)   # the formula adds 30 s to the top
        e480 = pc.estimate_workflow(seedance("480p", 6, video=True))
        self.assertAlmostEqual(e480["min_usd"], 0.5312, places=3)
        e1080 = pc.estimate_workflow(seedance("1080p", 6, video=True))
        self.assertAlmostEqual(e1080["min_usd"], 2.9392, places=3)
        self.assertLess(est["min_usd"], est["max_usd"])

    def test_known_reference_video_length_tightens_the_worst_case(self):
        calls = []
        orig = pc._video_seconds
        pc._video_seconds = lambda name: calls.append(name) or 6.0
        try:
            est = pc.estimate_workflow(seedance("720p", 6, video=True))
            self.assertEqual(calls, ["clip.mp4"])
            self.assertAlmostEqual(est["min_usd"], 1.1943, places=3)
            self.assertAlmostEqual(est["max_usd"], 2.3804, places=3)     # 6 s out + 6 s in, not +30 s
            self.assertEqual(est["nodes"][0]["refined_by_video_s"], 6.0)
            # nothing known about the clip: the 30 s worst case stays
            pc._video_seconds = lambda name: None
            self.assertAlmostEqual(pc.estimate_workflow(seedance("720p", 6, video=True))["max_usd"], 7.1248, places=3)
            # a clip that comes from another node (no file to read) is unknown too
            wf = seedance("720p", 6, video=True)
            wf["5"]["inputs"] = {"images": ["2", 0]}
            pc._video_seconds = lambda name: 6.0
            self.assertAlmostEqual(pc.estimate_workflow(wf)["max_usd"], 7.1248, places=3)
        finally:
            pc._video_seconds = orig

    def test_video_seconds_reads_the_file_duration(self):
        d = Path(tempfile.mkdtemp())
        (d / "clip.mp4").write_bytes(b"x")
        import agenty_core.tools.workspace_tools as W
        o = (W.media_info, W._resolve)
        W.media_info = lambda p: json.dumps({"ok": True, "duration_s": 6.5})
        W._resolve = lambda c: d / "clip.mp4" if c == "clip.mp4" else d / "nope"
        pc._video_len_cache.clear()
        try:
            self.assertEqual(pc._video_seconds("clip.mp4"), 6.5)
            self.assertIsNone(pc._video_seconds("missing.mp4"))
        finally:
            W.media_info, W._resolve = o

    def test_connected_input_changes_the_price(self):
        base = {"1": {"class_type": "Flux2ProImageNode", "inputs": {"width": 1024, "height": 1024}}}
        self.assertAlmostEqual(pc.estimate_workflow(base)["max_usd"], 0.03, places=4)
        base["1"]["inputs"]["images"] = ["4", 0]
        est = pc.estimate_workflow(base)
        self.assertAlmostEqual(est["min_usd"], 0.045, places=4)
        self.assertAlmostEqual(est["max_usd"], 0.15, places=4)

    def test_missing_widget_falls_back_to_the_schema_default(self):
        wf = seedance("720p", 5)
        del wf["9"]["inputs"]["model.ratio"]            # default 16:9 -> same as explicit
        a = pc.estimate_workflow(wf)["max_usd"]
        b = pc.estimate_workflow(seedance("720p", 5))["max_usd"]
        self.assertAlmostEqual(a, b, places=6)

    def test_price_unknown_is_not_guessed(self):
        # token-priced node (list_usd per 1K tokens): paid, but no per-run price
        wf = {"3": {"class_type": "ClaudeNode", "inputs": {"model": "Sonnet 5"}}}
        est = pc.estimate_workflow(wf)
        self.assertTrue(est["paid"])
        self.assertEqual(est["max_usd"], 0.0)
        self.assertEqual(len(est["unknown"]), 1)
        self.assertIn("token", est["unknown"][0]["reason"])
        # widgets the formula needs are absent and have no default: undefined -> unknown
        bad = {"9": {"class_type": "ByteDance2ReferenceNodeV2", "inputs": {}}}
        e2 = pc.estimate_workflow(bad)
        self.assertEqual(len(e2["unknown"]), 1)
        self.assertEqual(e2["max_usd"], 0.0)

    def test_expression_error_is_unknown(self):
        broken = {"price_badge": {"engine": "jsonata", "depends_on": {"widgets": [], "inputs": [], "input_groups": []},
                                  "expr": "(("}}
        est = pc.estimate_node("X", {}, broken)
        self.assertFalse(est["known"])
        self.assertIn("failed", est["reason"])


class Gate(Base):
    def test_local_graph_makes_no_network_call(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        est = pc.preflight(wf)
        self.assertFalse(est["paid"])
        self.assertEqual(self.balance_calls, [])

    def test_insufficient_credits_only_when_even_the_minimum_does_not_fit(self):
        self.balance["effective_balance_cents"] = 100      # $1.00 against $1.19 - $7.12
        with self.assertRaises(pc.CostGateError) as cm:
            pc.preflight(seedance("720p", 6, video=True), confirm=True)   # even confirmed
        self.assertEqual(cm.exception.error_kind, "insufficient_credits")
        msg = str(cm.exception)
        self.assertIn("$1.19", msg)
        self.assertIn("$1.00", msg)
        self.assertEqual(cm.exception.payload()["error_kind"], "insufficient_credits")

    def test_top_of_range_above_balance_asks_instead_of_refusing(self):
        # known 6 s clip: $1.19 - $2.39; balance $2.11 sits inside the range
        orig = pc._video_seconds
        pc._video_seconds = lambda name: 6.0
        self.balance["effective_balance_cents"] = 211
        try:
            with self.assertRaises(pc.CostGateError) as cm:
                pc.preflight(seedance("720p", 6, video=True))
            self.assertEqual(cm.exception.error_kind, "confirm_cost")
            self.assertIn("may not be covered", str(cm.exception))
            self.assertIn("$1.19-$2.38", str(cm.exception))
            self.assertIn("$2.11", str(cm.exception))
            pc.preflight(seedance("720p", 6, video=True), confirm=True)
            # under the confirmation level but possibly short: still asks
            pc._threshold_provider = lambda: 10.0
            with self.assertRaises(pc.CostGateError) as cm2:
                pc.preflight(seedance("720p", 6, video=True))
            self.assertEqual(cm2.exception.error_kind, "confirm_cost")
        finally:
            pc._video_seconds = orig

    def test_confirm_threshold(self):
        # $2.00 >= default $1.00 -> asks
        with self.assertRaises(pc.CostGateError) as cm:
            pc.preflight(seedance("720p", 6))
        self.assertEqual(cm.exception.error_kind, "confirm_cost")
        self.assertIn("confirm_cost=True", str(cm.exception))
        self.assertIn("$2.00", str(cm.exception))
        # with the go-ahead it passes
        est = pc.preflight(seedance("720p", 6), confirm=True)
        self.assertTrue(est["paid"])
        # a go-ahead given for the FILE carries to the later submission
        with self.assertRaises(pc.CostGateError):
            pc.preflight(seedance("720p", 6), workflow_path="a.json")
        pc.confirm_path("a.json", float("inf"))
        pc.preflight(seedance("720p", 6), workflow_path="a.json")
        # the contextvar form
        with pc.confirmed():
            pc.preflight(seedance("720p", 6))

    def test_threshold_is_a_setting(self):
        pc._threshold_provider = lambda: 5.0
        pc.preflight(seedance("720p", 6))                       # $2 < $5: runs
        pc._threshold_provider = lambda: 0.5
        with self.assertRaises(pc.CostGateError):
            pc.preflight(seedance("480p", 5))                   # ~$0.74 >= $0.50

    def test_below_threshold_runs_and_reports_the_estimate(self):
        wf = {"1": {"class_type": "Flux2ProImageNode", "inputs": {"width": 1024, "height": 1024}}}
        est = pc.preflight(wf)
        self.assertTrue(est["paid"])
        self.assertAlmostEqual(est["max_usd"], 0.03, places=4)
        self.assertEqual(est["balance"]["effective_usd"], 50.0)

    def test_unknown_balance_does_not_block(self):
        pc._http_get = lambda u, h: (_ for _ in ()).throw(OSError("offline"))
        pc.clear_caches()
        pc._info_provider = lambda ct: INFO.get(ct)
        wf = {"1": {"class_type": "Flux2ProImageNode", "inputs": {"width": 1024, "height": 1024}}}
        self.assertTrue(pc.preflight(wf)["paid"])
        self.assertIsNone(pc.fetch_balance(force=True))

    def test_balance_cached_for_a_minute(self):
        pc.fetch_balance()
        pc.fetch_balance()
        self.assertEqual(len(self.balance_calls), 1)
        pc.fetch_balance(force=True)
        self.assertEqual(len(self.balance_calls), 2)

    def test_variants_are_judged_by_their_sum(self):
        d = Path(tempfile.mkdtemp())
        paths = []
        for i in range(3):
            p = d / f"v{i}.json"
            p.write_text(json.dumps(seedance("480p", 4)), encoding="utf-8")   # ~$0.59 each
            paths.append(str(p))
        pc.preflight(json.loads(Path(paths[0]).read_text()))     # one alone: under $1
        with self.assertRaises(pc.CostGateError) as cm:
            pc.preflight_many(paths)                              # three together: over
        self.assertEqual(cm.exception.error_kind, "confirm_cost")
        pc.preflight_many(paths, confirm=True)


class ActualSpend(Base):
    def test_spend_is_the_balance_delta(self):
        heard = []
        pc.add_listener(lambda usd, info: heard.append(usd))
        est = pc.estimate_workflow(seedance("720p", 6))
        t0 = pc.turn_total()
        pc.track("p1", est)                                   # balance 5000 c read here
        self.balance["effective_balance_cents"] = 5000 - 238  # the charge posts
        self.assertEqual(pc.settle("p1"), 2.38)
        self.assertEqual(heard, [2.38])
        self.assertEqual(self.logged[-1]["usd"], 2.38)
        self.assertAlmostEqual(pc.turn_total() - t0, 2.38, places=4)
        self.assertIsNone(pc.settle("p1"))                    # settled once

    def test_pending_charge_is_picked_up_by_the_second_read(self):
        est = pc.estimate_workflow(seedance("720p", 6))
        pc.track("p2", est)
        reads = []

        def later(s):                                          # the charge lands during the wait
            self.balance["effective_balance_cents"] = 4800
            reads.append(s)
        pc._sleep = later
        before = pc.turn_total()
        self.assertEqual(pc.settle("p2"), 2.0)
        self.assertEqual(reads, [pc.SETTLE_DELAY_S])
        self.assertAlmostEqual(pc.turn_total() - before, 2.0, places=4)

    def test_local_run_is_not_tracked(self):
        pc.track("p3", {"paid": False})
        self.assertIsNone(pc.settle("p3"))
        self.assertEqual(self.balance_calls, [])

    def test_overlapping_runs_share_the_total(self):
        est = pc.estimate_workflow(seedance("480p", 4))
        pc.track("a", est)
        pc.track("b", est)                                     # baseline stays at the first read
        self.balance["effective_balance_cents"] = 4940
        first = pc.settle("a")
        self.balance["effective_balance_cents"] = 4880
        second = pc.settle("b")
        self.assertAlmostEqual(first + second, 1.2, places=4)


class SubmitGate(Base):
    """The check inside comfyui_exec._submit_workflow, with ComfyUI stubbed."""

    def _run(self, wf, **kw):
        from agenty_core import comfyui_exec as X
        from agenty_core.utils import comfyui_client as C
        posts = []

        class Client:
            api_key = "k"

            def post(self, path, json_data=None, **_):
                posts.append(path)
                return {"prompt_id": "pid-1"}

        orig = C.get_client
        C.get_client = lambda: Client()
        d = Path(tempfile.mkdtemp())
        f = d / "w.json"
        f.write_text(json.dumps(wf), encoding="utf-8")
        try:
            return X._submit_workflow(str(f)), posts, str(f)
        finally:
            C.get_client = orig

    def test_gate_stops_before_the_post(self):
        with self.assertRaises(pc.CostGateError):
            self._run(seedance("720p", 6))
        # (the stub client was never reached for a refusal)

    def test_local_graph_posts_without_balance_call(self):
        pid, posts, _ = self._run({"1": {"class_type": "KSampler", "inputs": {}}})
        self.assertEqual(pid, "pid-1")
        self.assertEqual(posts, ["/prompt"])
        self.assertEqual(self.balance_calls, [])

    def test_paid_run_below_threshold_is_tracked(self):
        pid, posts, _ = self._run({"1": {"class_type": "Flux2ProImageNode",
                                         "inputs": {"width": 1024, "height": 1024}}})
        self.assertIn("pid-1", pc._inflight)
        self.balance["effective_balance_cents"] = 4997
        self.assertEqual(pc.settle("pid-1"), 0.03)


class ToolLevel(Base):
    """The tools an agent calls, with ComfyUI stubbed out."""

    def _file(self, wf):
        f = Path(tempfile.mkdtemp()) / "w.json"
        f.write_text(json.dumps(wf), encoding="utf-8")
        return str(f)

    def test_submit_prompt_refuses_and_names_the_kind(self):
        from agenty_core.tools.comfyui import submit_prompt
        got = json.loads(submit_prompt(self._file(seedance("720p", 6))))
        self.assertEqual(got["error_kind"], "confirm_cost")
        self.assertAlmostEqual(got["estimate_usd"]["max"], 2.0, places=2)
        self.assertEqual(got["balance_usd"], 50.0)

    def test_submit_prompt_with_confirmation_posts_and_tracks(self):
        from agenty_core.tools import comfyui as T
        posts = []

        class Client:
            api_key = ""

            def post(self, path, json_data=None, **_):
                posts.append(json_data)
                return {"prompt_id": "pid-9"}
        orig = T.get_client
        T.get_client = lambda: Client()
        try:
            got = json.loads(T.submit_prompt(self._file(seedance("720p", 6)), confirm_cost=True))
        finally:
            T.get_client = orig
        self.assertEqual(got["prompt_id"], "pid-9")
        self.assertAlmostEqual(got["estimate_usd"]["max"], 2.0, places=2)
        self.assertEqual(len(posts), 1)
        self.assertIn("pid-9", pc._inflight)

    def test_estimate_workflow_cost_tool(self):
        from agenty_core.tools.comfyui import estimate_workflow_cost
        got = json.loads(estimate_workflow_cost(self._file(seedance("720p", 6, video=True))))
        self.assertTrue(got["paid"])
        self.assertAlmostEqual(got["min_usd"], 1.1943, places=3)
        self.assertEqual(got["confirm_above_usd"], 1.0)
        self.assertTrue(got["needs_confirmation"])
        local = json.loads(estimate_workflow_cost(self._file({"1": {"class_type": "KSampler", "inputs": {}}})))
        self.assertFalse(local["paid"])

    def test_run_variants_sums_and_asks(self):
        from agenty_core.tools import variants as V
        d = Path(tempfile.mkdtemp())
        paths = []
        for i in range(3):
            p = d / f"v{i}.json"
            p.write_text(json.dumps(seedance("480p", 4)), encoding="utf-8")
            paths.append(str(p))
        orig = V.prepare_variants
        V.prepare_variants = lambda wp, vs: {"status": "ok", "ready": paths, "variants": []}
        ran = []
        V.set_batch_hook(lambda ps: ran.append(ps) or {"outputs": {}})
        try:
            got = json.loads(V.run_variants("x.json", [{}, {}, {}]))
            self.assertEqual(got["error_kind"], "confirm_cost")
            self.assertEqual(ran, [])
            got = json.loads(V.run_variants("x.json", [{}, {}, {}], confirm_cost=True))
            self.assertTrue(got["ran"])
            self.assertEqual(len(ran), 1)
            self.assertGreater(got["estimate_usd"]["max"], 1.5)
        finally:
            V.prepare_variants = orig
            V.set_batch_hook(None)

    def test_partner_run_estimate_and_gate(self):
        from agenty_core.tools import partner_run as PR
        path = self._file(seedance("720p", 6))
        orig = PR.prepare_partner_workflow
        PR.prepare_partner_workflow = lambda *a, **k: {"status": "ready", "workflow_path": path, "node": "Seedance"}
        ran = []
        PR.set_run_hook(lambda p: ran.append(p) or {"status": "done", "outputs": ["a.mp4"]})
        try:
            dry = json.loads(PR.partner_run("ByteDance2ReferenceNodeV2", run=False))
            self.assertAlmostEqual(dry["estimate_usd"]["max"], 2.0, places=2)
            self.assertFalse(dry["ran"])
            refused = json.loads(PR.partner_run("ByteDance2ReferenceNodeV2"))
            self.assertEqual(refused["error_kind"], "confirm_cost")
            self.assertEqual(ran, [])
            ok = json.loads(PR.partner_run("ByteDance2ReferenceNodeV2", confirm_cost=True))
            self.assertEqual(ok["run"]["status"], "done")
            self.assertEqual(len(ran), 1)
        finally:
            PR.prepare_partner_workflow = orig
            PR.set_run_hook(None)

    def test_partner_run_default_run_reports_actual_spend(self):
        from agenty_core import comfyui_exec as X
        from agenty_core import batch_runner as B
        from agenty_core.tools import partner_run as PR
        wf = {"1": {"class_type": "Flux2ProImageNode", "inputs": {"width": 1024, "height": 1024}}}
        path = self._file(wf)
        o = (X._submit_workflow, B._wait_for_completion, X._extract_output_files)
        def fake_submit(p, client_id=""):
            est = pc.gate_submission(json.loads(Path(p).read_text()), p)
            pc.track("pid-7", est)
            return "pid-7"
        def fake_wait(client, pid, timeout=0):
            self.balance["effective_balance_cents"] = 4997          # the charge posts
            return {"history": {}}
        X._submit_workflow, B._wait_for_completion, X._extract_output_files = fake_submit, fake_wait, lambda h: []
        try:
            from agenty_core.utils import comfyui_client as C
            oc = C.get_client
            C.get_client = lambda: object()
            try:
                got = PR._default_run(path)
            finally:
                C.get_client = oc
        finally:
            X._submit_workflow, B._wait_for_completion, X._extract_output_files = o
        self.assertEqual(got["partner_cost_usd"], 0.03)


if __name__ == "__main__":
    unittest.main()
