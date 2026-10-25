"""Execution modes: the deterministic router, mode/override precedence, the cost gate for cloud runs, and the
cloud client against stubbed responses. No network, no ComfyUI, no credits.

    python -m unittest agenty_core.tools.tests.test_execution_modes
"""
import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core import partner_credits as pc
from agenty_core.utils import cloud_run, comfy_cloud, exec_route, execution_mode as em, turn_scope

GB = 1024 ** 3
MODULE = {"KSampler": "nodes", "CheckpointLoaderSimple": "nodes", "SaveImage": "nodes", "LoadImage": "nodes",
          "UNETLoader": "nodes", "CLIPLoader": "nodes", "LoraLoader": "nodes",
          "SeedreamNode": "comfy_api_nodes.nodes_bytedance", "KlingNode": "comfy_api_nodes.nodes_kling",
          "WeirdPackNode": "custom_nodes.weird"}

PARTNER = {"1": {"class_type": "SeedreamNode", "inputs": {"prompt": "a cat"}},
           "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
OPEN = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}}
MIXED = {**OPEN, "4": {"class_type": "SeedreamNode", "inputs": {"prompt": "x"}}}
FLUX = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "t5.safetensors"}}}


class FakeProbes(exec_route.Probes):
    def __init__(self, *, local_up=True, models=None, free=24 * GB, key=True, cloud_models=(), cloud_nodes=None):
        self.up, self.models, self.free, self.key = local_up, models or {}, free, key
        self.cm = set(cloud_models)
        self.cn = set(MODULE) if cloud_nodes is None else set(cloud_nodes)

    def local_available(self): return self.up
    def node_info(self, c): return {"python_module": MODULE[c]} if c in MODULE else None
    def local_model(self, n): return self.models.get(n, {"present": False, "bytes": None})
    def free_vram(self): return self.free
    def cloud_key(self): return self.key
    def cloud_models(self): return self.cm
    def cloud_nodes(self): return self.cn


class Untouchable(exec_route.Probes):
    def __getattribute__(self, name):
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        raise AssertionError(f"local mode must not probe anything ({name})")


class LocalMode(unittest.TestCase):
    def test_local_never_probes_and_never_goes_to_the_cloud(self):
        for wf in (PARTNER, OPEN, MIXED):
            d = exec_route.decide(wf, mode="local", probes=Untouchable())
            self.assertEqual(d.backend, "local")

    def test_partner_only_in_local_mode_stays_local(self):
        self.assertEqual(exec_route.decide(PARTNER, mode="local", probes=FakeProbes()).backend, "local")

    def test_mixed_in_local_mode_stays_local(self):
        self.assertEqual(exec_route.decide(MIXED, mode="local", probes=FakeProbes(models={})).backend, "local")


class Hybrid(unittest.TestCase):
    SDXL = {"sdxl.safetensors": {"present": True, "bytes": 6 * GB}}

    def test_fits_goes_local(self):
        d = exec_route.decide(OPEN, mode="hybrid", probes=FakeProbes(models=self.SDXL, free=12 * GB))
        self.assertEqual(d.backend, "local")

    def test_does_not_fit_goes_to_the_cloud_when_it_has_everything(self):
        d = exec_route.decide(OPEN, mode="hybrid", probes=FakeProbes(
            models=self.SDXL, free=4 * GB, cloud_models={"sdxl.safetensors"}))
        self.assertEqual(d.backend, "comfy_cloud")
        self.assertIn("VRAM", d.reason)

    def test_not_on_disk_goes_to_the_cloud_when_it_has_it(self):
        d = exec_route.decide(OPEN, mode="hybrid", probes=FakeProbes(cloud_models={"sdxl.safetensors"}))
        self.assertEqual(d.backend, "comfy_cloud")
        self.assertEqual(d.missing_local, ["sdxl.safetensors"])

    def test_cloud_lacks_a_model_is_a_clear_refusal(self):
        with self.assertRaises(exec_route.RouteRefused) as cm:
            exec_route.decide(OPEN, mode="hybrid", probes=FakeProbes(models=self.SDXL, free=2 * GB))
        p = cm.exception.payload()
        self.assertEqual(p["error_kind"], "cannot_run_here")
        self.assertEqual(p["missing_on_cloud_models"], ["sdxl.safetensors"])
        self.assertIn("partner", p["hint"])

    def test_cloud_lacks_a_node_class(self):
        wf = {**OPEN, "9": {"class_type": "WeirdPackNode", "inputs": {}}}
        with self.assertRaises(exec_route.RouteRefused) as cm:
            exec_route.decide(wf, mode="hybrid", probes=FakeProbes(
                free=2 * GB, models=self.SDXL, cloud_models={"sdxl.safetensors"},
                cloud_nodes=set(MODULE) - {"WeirdPackNode"}))
        self.assertEqual(cm.exception.decision.missing_cloud_nodes, ["WeirdPackNode"])

    def test_no_key_says_so(self):
        with self.assertRaises(exec_route.RouteRefused) as cm:
            exec_route.decide(OPEN, mode="hybrid", probes=FakeProbes(key=False))
        self.assertIn("COMFYUI_API_KEY", cm.exception.payload()["hint"])

    def test_partner_only_stays_local(self):
        d = exec_route.decide(PARTNER, mode="hybrid", probes=FakeProbes(free=0, key=False))
        self.assertEqual(d.backend, "local")
        self.assertTrue(d.partner_nodes)

    def test_mixed_is_routed_by_its_open_weight_part_only(self):
        fits = exec_route.decide(MIXED, mode="hybrid", probes=FakeProbes(models=self.SDXL, free=12 * GB))
        self.assertEqual(fits.backend, "local")
        away = exec_route.decide(MIXED, mode="hybrid", probes=FakeProbes(
            models=self.SDXL, free=2 * GB, cloud_models={"sdxl.safetensors"}))
        self.assertEqual(away.backend, "comfy_cloud")

    def test_the_estimate_is_peak_not_sum(self):
        need = exec_route.estimate_need([20 * GB, 10 * GB], 2)
        self.assertLess(need, 30 * GB)
        self.assertGreater(need, 20 * GB)
        self.assertIsNone(exec_route.estimate_need([None]))

    def test_two_stages_that_fit_one_after_the_other(self):
        d = exec_route.decide(FLUX, mode="hybrid", probes=FakeProbes(
            models={"flux.safetensors": {"present": True, "bytes": 12 * GB},
                    "t5.safetensors": {"present": True, "bytes": 9 * GB}}, free=17 * GB))
        self.assertEqual(d.backend, "local")

    def test_local_comfyui_down_uses_the_cloud(self):
        d = exec_route.decide(OPEN, mode="hybrid",
                              probes=FakeProbes(local_up=False, cloud_models={"sdxl.safetensors"}))
        self.assertEqual(d.backend, "comfy_cloud")


class CloudMode(unittest.TestCase):
    def test_open_weight_goes_to_the_cloud(self):
        d = exec_route.decide(OPEN, mode="cloud", probes=FakeProbes(cloud_models={"sdxl.safetensors"}))
        self.assertEqual(d.backend, "comfy_cloud")

    def test_partner_only_runs_in_the_local_comfyui(self):
        self.assertEqual(exec_route.decide(PARTNER, mode="cloud", probes=FakeProbes(free=0)).backend, "local")

    def test_partner_only_falls_back_to_the_cloud_without_a_local_comfyui(self):
        d = exec_route.decide(PARTNER, mode="cloud", probes=FakeProbes(local_up=False))
        self.assertEqual(d.backend, "comfy_cloud")

    def test_a_model_the_cloud_lacks_is_refused(self):
        with self.assertRaises(exec_route.RouteRefused):
            exec_route.decide(OPEN, mode="cloud", probes=FakeProbes())

    def test_refusal_can_be_returned_instead_of_raised(self):
        d = exec_route.decide(OPEN, mode="cloud", probes=FakeProbes(), raise_on_refuse=False)
        self.assertEqual(d.backend, "none")


class Precedence(unittest.TestCase):
    def setUp(self):
        em.set_turn_override(None)
        self.addCleanup(em.set_turn_override, None)
        self.addCleanup(em.set_settings_provider, None)
        env = mock.patch.dict(os.environ, {"AGENTY_EXECUTION_MODE": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_default_is_local(self):
        em.set_settings_provider(None)
        self.assertEqual(em.effective_mode(), "local")

    def test_setting_then_env_then_override(self):
        em.set_settings_provider(lambda: {"mode": "hybrid"})
        self.assertEqual(em.effective_mode(), "hybrid")
        with mock.patch.dict(os.environ, {"AGENTY_EXECUTION_MODE": "cloud"}):
            self.assertEqual(em.effective_mode(), "cloud")
            em.set_turn_override("local")
            self.assertEqual(em.effective_mode(), "local")
        em.set_turn_override(None)
        self.assertEqual(em.effective_mode(), "hybrid")

    def test_a_bad_value_is_ignored(self):
        em.set_settings_provider(lambda: {"mode": "turbo"})
        self.assertEqual(em.effective_mode(), "local")
        self.assertIsNone(em.set_turn_override("turbo"))

    def test_the_override_is_per_turn(self):
        em.set_turn_override("cloud")
        tok = turn_scope._current.set(turn_scope.Scope("other-turn"))
        try:
            self.assertEqual(em.effective_mode(), "local")
        finally:
            turn_scope._current.reset(tok)
        self.assertEqual(em.effective_mode(), "cloud")

    def test_the_override_expires(self):
        em.set_turn_override("cloud")
        with mock.patch("agenty_core.utils.execution_mode.time.time", return_value=9e12):
            self.assertEqual(em.effective_mode(), "local")

    def test_phrases(self):
        self.assertEqual(em.parse_override("please run this in the cloud"), "cloud")
        self.assertEqual(em.parse_override("keep it local"), "local")
        self.assertIsNone(em.parse_override("make it blue"))

    def test_the_tools(self):
        from agenty_core.tools import execution_mode as t
        out = json.loads(t.set_execution_mode_for_turn("cloud", "user said so"))
        self.assertEqual(out["override"], "cloud")
        self.assertEqual(em.effective_mode(), "cloud")
        self.assertEqual(json.loads(t.set_execution_mode_for_turn("nonsense"))["status"], "error")
        t.set_execution_mode_for_turn("default")
        self.assertEqual(em.effective_mode(), "local")

    def test_cloud_mode_refuses_local_only_tools_with_the_reason(self):
        self.assertIsNone(em.local_only_refusal("train_splat"))
        em.set_turn_override("cloud")
        r = em.local_only_refusal("train_splat")
        self.assertEqual(r["error_kind"], "local_only")
        self.assertIn("Brush", r["error"])
        self.assertIsNone(em.local_only_refusal("execute_workflow"))
        em.set_turn_override("hybrid")
        self.assertIsNone(em.local_only_refusal("train_splat"))


class CostGate(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, pc, "_http_get", pc._http_get)
        self.addCleanup(em.set_settings_provider, None)
        pc.clear_caches()
        self.addCleanup(pc.clear_caches)
        self.addCleanup(setattr, pc, "_threshold_provider", pc._threshold_provider)
        pc.set_threshold_provider(lambda: 1.0)
        self.client = comfy_cloud.CloudClient(key="k", http=mock.Mock())
        self.client.node_info = lambda c: None
        p = mock.patch.object(comfy_cloud, "_client", self.client)
        p.start()
        self.addCleanup(p.stop)

    def balance(self, usd):
        pc._http_get = lambda url, headers: {"effective_balance_cents": usd * 100, "prepaid_balance_cents": usd * 100}
        pc.clear_caches()

    def test_below_the_minimum_is_refused(self):
        self.balance(0.2)
        with self.assertRaises(pc.CostGateError) as cm:
            cloud_run.gate(OPEN)
        self.assertEqual(cm.exception.error_kind, "insufficient_credits")

    def test_a_covered_cheap_run_goes(self):
        self.balance(20)
        est = cloud_run.gate(OPEN)
        self.assertTrue(est["paid"])
        # image workflow: 20-120 s at ~$0.077/min
        self.assertAlmostEqual(est["min_usd"], round(0.077 / 60 * 20, 4))
        self.assertAlmostEqual(est["max_usd"], round(0.077 / 60 * 120, 4))

    def test_a_video_workflow_is_estimated_longer(self):
        video = {"1": {"class_type": "WanImageToVideo", "inputs": {}}}
        lo, hi = cloud_run.gpu_time_usd(video)
        self.assertAlmostEqual(hi, round(0.077 / 60 * 600, 4))
        self.assertGreater(hi, cloud_run.gpu_time_usd(OPEN)[1])

    def test_above_the_threshold_needs_the_users_yes(self):
        self.balance(20)
        em.set_settings_provider(lambda: {"cloud_gpu_usd_per_second": 0.05})
        with self.assertRaises(pc.CostGateError) as cm:
            cloud_run.gate(OPEN)
        self.assertEqual(cm.exception.error_kind, "confirm_cost")
        with pc.confirmed():
            self.assertTrue(cloud_run.gate(OPEN)["paid"])

    def test_no_gpu_time_is_charged_to_a_run_without_models(self):
        self.balance(20)
        self.assertFalse(cloud_run.gate(PARTNER, open_weight=False)["paid"])


class FakeResp:
    def __init__(self, status=200, body=None, content=b"", headers=None):
        self.status_code, self._b, self.content, self.headers = status, body, content, headers or {}
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._b is None:
            raise ValueError
        return self._b


class FakeHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, headers=None, **kw):
        self.calls.append((method, url, dict(headers or {}), kw))
        for (m, suffix), resp in self.routes.items():
            if m == method and url.split("?")[0].endswith(suffix):
                return resp(kw) if callable(resp) else resp
        return FakeResp(404, {"error": "nope"})


class Client(unittest.TestCase):
    def mk(self, routes, **kw):
        self.http = FakeHttp(routes)
        self.cache = Path(tempfile.mkdtemp()) / "c.json"
        return comfy_cloud.CloudClient(key="SECRET-KEY", http=self.http, cache_path=self.cache, **kw)

    def test_submit_sends_the_key_header_and_returns_the_id(self):
        c = self.mk({("POST", "/api/prompt"): FakeResp(200, {"prompt_id": "p1", "node_errors": {}})})
        self.assertEqual(c.submit({"1": {"class_type": "X", "inputs": {}}}), "p1")
        method, url, headers, kw = self.http.calls[0]
        self.assertEqual(headers["X-API-Key"], "SECRET-KEY")
        self.assertEqual(kw["json"]["extra_data"]["api_key_comfy_org"], "SECRET-KEY")
        self.assertTrue(url.startswith("https://cloud.comfy.org/api/prompt"))

    def test_node_errors_are_a_refusal(self):
        c = self.mk({("POST", "/api/prompt"): FakeResp(200, {"prompt_id": "p", "node_errors": {"3": {"errors": []}}})})
        with self.assertRaises(comfy_cloud.CloudError) as cm:
            c.submit({})
        self.assertEqual(cm.exception.kind, "invalid")

    def test_http_errors_are_classified_and_never_leak_the_key(self):
        for code, kind in ((401, "auth"), (402, "credits"), (429, "subscription"), (400, "invalid"), (500, "http")):
            c = self.mk({("GET", "/api/user"): FakeResp(code, {"message": "boom"})})
            with self.assertRaises(comfy_cloud.CloudError) as cm:
                c.user()
            self.assertEqual(cm.exception.kind, kind)
            self.assertNotIn("SECRET-KEY", str(cm.exception))

    def test_a_network_failure_does_not_leak_the_key(self):
        http = mock.Mock()
        http.request.side_effect = ConnectionError("failed for https://x?token=SECRET-KEY")
        c = comfy_cloud.CloudClient(key="SECRET-KEY", http=http)
        with self.assertRaises(comfy_cloud.CloudError) as cm:
            c.user()
        self.assertNotIn("SECRET-KEY", str(cm.exception))

    def test_no_key_is_its_own_error(self):
        c = comfy_cloud.CloudClient(key="", http=FakeHttp({}))
        with self.assertRaises(comfy_cloud.CloudError) as cm:
            c.user()
        self.assertEqual(cm.exception.kind, "no_key")

    def test_states_are_normalised_from_both_vocabularies(self):
        for raw, want in (("waiting_to_dispatch", "queued"), ("queued_waiting", "queued"), ("in_progress", "running"),
                          ("executing", "running"), ("completed", "done"), ("success", "done"), ("error", "error"),
                          ("non_retryable_error", "error"), ("lost", "error"), ("cancelled", "cancelled"),
                          ("???", "unknown")):
            self.assertEqual(comfy_cloud.normalize_state(raw), want)

    def test_download_follows_the_redirect_without_the_key(self):
        c = self.mk({("GET", "/api/view"): FakeResp(302, headers={"Location": "https://storage.example/signed?x=1"}),
                     ("GET", "/signed"): FakeResp(200, content=b"PNGDATA")})
        dest = Path(tempfile.mkdtemp()) / "out" / "a.png"
        c.download("a.png", dest)
        self.assertEqual(dest.read_bytes(), b"PNGDATA")
        view, signed = self.http.calls
        self.assertIn("X-API-Key", view[2])
        self.assertFalse(view[3]["allow_redirects"])
        self.assertNotIn("X-API-Key", signed[2])

    def test_upload_returns_the_cloud_name(self):
        c = self.mk({("POST", "/api/upload/image"): FakeResp(200, {"name": "cat.png", "subfolder": "", "type": "input"})})
        f = Path(tempfile.mkdtemp()) / "cat.png"
        f.write_bytes(b"x")
        self.assertEqual(c.upload(f), "cat.png")
        self.assertEqual(self.http.calls[0][3]["data"]["type"], "input")

    def test_models_and_nodes_are_cached(self):
        c = self.mk({
            ("GET", "/api/experiment/models"): FakeResp(200, [{"name": "checkpoints", "folders": ["checkpoints"]}]),
            ("GET", "/api/experiment/models/checkpoints"): FakeResp(200, [{"name": "sub/SDXL.safetensors", "pathIndex": 0}]),
            ("GET", "/api/object_info"): FakeResp(200, {
                "KSampler": {"python_module": "nodes"},
                "Seedream": {"python_module": "comfy_api_nodes.x", "price_badge": {"expr": "1"}, "input": {}}})})
        self.assertTrue(c.has_model("SDXL.safetensors"))
        self.assertFalse(c.has_model("other.safetensors"))
        self.assertTrue(c.has_node("KSampler"))
        self.assertTrue(c.nodes()["Seedream"]["api_node"])
        n = len(self.http.calls)
        c.has_model("SDXL.safetensors")
        c.has_node("KSampler")
        self.assertEqual(len(self.http.calls), n)                      # served from memory
        c2 = comfy_cloud.CloudClient(key="k", http=FakeHttp({}), cache_path=self.cache)
        self.assertTrue(c2.has_model("sdxl.safetensors"))             # and from disk, in a new process

    def test_the_cache_expires(self):
        t = [1000.0]
        c = self.mk({("GET", "/api/object_info"): FakeResp(200, {"KSampler": {}})}, clock=lambda: t[0])
        c.nodes()
        t[0] += comfy_cloud.CACHE_TTL_S + 1
        c.nodes()
        self.assertEqual(len([x for x in self.http.calls if x[1].endswith("/api/object_info")]), 2)


class Run(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(setattr, pc, "_http_get", pc._http_get)
        pc._http_get = lambda url, headers: {"effective_balance_cents": 5000, "prepaid_balance_cents": 5000}
        pc.clear_caches()
        pc._inflight.clear()
        pc._watermark = None
        self.addCleanup(pc._inflight.clear)
        self.addCleanup(setattr, pc, "_threshold_provider", pc._threshold_provider)
        pc.set_threshold_provider(lambda: 1.0)
        self.addCleanup(pc.clear_caches)

    def client(self, statuses, detail, routes=None):
        def status(kw):
            return FakeResp(200, {"status": statuses.pop(0) if len(statuses) > 1 else statuses[0]})
        r = {("POST", "/api/prompt"): FakeResp(200, {"prompt_id": "J1"}),
             ("GET", "/api/job/J1/status"): status,
             ("GET", "/api/jobs/J1"): FakeResp(200, detail),
             ("GET", "/api/view"): FakeResp(200, content=b"IMG"),
             ("POST", "/api/queue"): FakeResp(200, {}), ("POST", "/api/interrupt"): FakeResp(200, {})}
        r.update(routes or {})
        self.http = FakeHttp(r)
        c = comfy_cloud.CloudClient(key="k", http=self.http, cache_path=Path(tempfile.mkdtemp()) / "c.json")
        c.node_info = lambda cls: None
        p = mock.patch.object(comfy_cloud, "_client", c)
        p.start()
        self.addCleanup(p.stop)
        return c

    def drive(self, client, ref):
        async def go():
            return [x async for x in cloud_run.stream(ref, poll_s=0, client=client)]
        return asyncio.run(go())

    def test_submit_stream_and_history(self):
        c = self.client(["pending", "in_progress", "completed"], {
            "status": "completed",
            "outputs": {"9": {"images": [{"filename": "r.png", "subfolder": "", "type": "output"}]}}})
        ref = cloud_run.submit(OPEN, name="t", dest_dir=self.out, client=c)
        self.assertEqual(ref, "cloud:J1")
        events = self.drive(c, ref)
        files = [i["filename"] for i in events[-1]["history"][ref]["outputs"]["farm"]["images"]]
        self.assertEqual(Path(files[0]).read_bytes(), b"IMG")
        self.assertTrue(Path(files[0]).is_relative_to(self.out))
        self.assertTrue(any("running" in str(e) for e in events if isinstance(e, str)))

    def test_a_cloud_failure_reaches_the_heal_loop_like_a_local_one(self):
        c = self.client(["error"], {"status": "failed", "execution_error": {
            "node_id": "2", "node_type": "KSampler", "exception_type": "OOMError",
            "exception_message": "out of memory", "traceback": ["t"]}})
        end = self.drive(c, "cloud:J1")[-1]
        self.assertEqual(end["details"]["node_id"], "2")
        self.assertEqual(end["details"]["exception_type"], "OOMError")
        self.assertTrue(end["details"]["cloud"])

    def test_cancelled_is_an_interruption(self):
        c = self.client(["cancelled"], {})
        self.assertTrue(self.drive(c, "cloud:J1")[-1]["interrupted"])

    def test_inputs_are_uploaded_and_renamed_on_a_copy(self):
        img = self.out / "in.png"
        img.write_bytes(b"x")
        wf = {"9": {"class_type": "LoadImage", "inputs": {"image": str(img)}}, **OPEN}
        c = self.client(["completed"], {"outputs": {}},
                        {("POST", "/api/upload/image"): FakeResp(200, {"name": "up_in.png", "subfolder": ""})})
        cloud_run.submit(wf, dest_dir=self.out, client=c)
        sent = [x for x in self.http.calls if x[1].endswith("/api/prompt")][0][3]["json"]["prompt"]
        self.assertEqual(sent["9"]["inputs"]["image"], "up_in.png")
        self.assertEqual(wf["9"]["inputs"]["image"], str(img))

    def test_the_cost_is_measured_and_reported(self):
        c = self.client(["completed"], {"outputs": {}})
        bal = iter([5000, 5000, 5000, 4990, 4990, 4990])
        pc._http_get = lambda url, headers: {"effective_balance_cents": next(bal, 4990), "prepaid_balance_cents": 5000}
        with mock.patch.object(pc, "_sleep", lambda s: None):
            ref = cloud_run.submit(OPEN, dest_dir=self.out, client=c)
            line = cloud_run.settle_line(ref)
        self.assertIn("ran_on: comfy_cloud", line)
        self.assertIn("cost $0.10", line)
        self.assertEqual(cloud_run.runs_this_turn()[-1]["ran_on"], "comfy_cloud")

    def test_stop_cancels_the_paid_job(self):
        c = self.client(["pending"], {})

        async def go():
            g = cloud_run.stream("cloud:J1", poll_s=0.01, client=c)
            await g.__anext__()
            t = asyncio.ensure_future(g.__anext__())
            await asyncio.sleep(0.05)
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        asyncio.run(go())
        self.assertTrue(any(x[1].endswith("/api/queue") for x in self.http.calls))


if __name__ == "__main__":
    unittest.main()
