"""gpu_pool offline: fake ComfyUI machines and fake vast.ai / RunPod APIs behind a swapped transport."""

import json
import time
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from agenty_core.utils import gpu_pool as gp

WF = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "z_image_turbo.safetensors"}},
      "2": {"class_type": "KSampler", "inputs": {"seed": 100, "model": ["1", 0]}},
      "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}}}


class R:
    def __init__(self, status=200, body=None, raw=None):
        self.status_code = status
        self.content = raw if raw is not None else json.dumps(body if body is not None else {}).encode()

    def json(self):
        return json.loads(self.content.decode() or "null")


class FakeComfy:
    def __init__(self, nodes=("CheckpointLoaderSimple", "KSampler", "SaveImage"), models=("z_image_turbo.safetensors",),
                 fail=False):
        self.nodes, self.models, self.fail = set(nodes), list(models), fail
        self.prompts, self.auth = {}, []

    def __call__(self, method, path, headers, body):
        self.auth.append((headers or {}).get("Authorization"))
        if path == "/system_stats":
            return R(body={"devices": [{"name": "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync"}],
                           "system": {"comfyui_version": "0.4.0"}})
        if path == "/queue":
            return R(body={"queue_running": [], "queue_pending": []})
        if path == "/object_info":
            return R(body={n: {} for n in self.nodes})
        if path.startswith("/models/"):
            return R(body=self.models if path.endswith("checkpoints") else [])
        if path == "/prompt":
            pid = f"p{len(self.prompts)}"
            self.prompts[pid] = body["prompt"]
            return R(body={"prompt_id": pid, "node_errors": {}})
        if path.startswith("/history/"):
            pid = path.rsplit("/", 1)[1]
            if self.fail:
                return R(body={pid: {"status": {"status_str": "error", "completed": False, "messages": [
                    ["execution_error", {"exception_message": "CUDA out of memory"}]]}, "outputs": {}}})
            return R(body={pid: {"status": {"status_str": "success", "completed": True},
                                 "outputs": {"3": {"images": [{"filename": f"{pid}.png", "subfolder": "", "type": "output"}]}}}})
        if path.startswith("/view"):
            return R(raw=b"PNG" + path.encode())
        if path in ("/interrupt",):
            return R(body={})
        return R(404, raw=b"no")


class Net:
    """Routes by host: machines by their netloc, vast / runpod by API host."""

    def __init__(self):
        self.machines, self.calls = {}, []
        self.vast_instances = {}
        self.rented = []

    def __call__(self, method, url, headers=None, json_body=None, data=None, files=None, timeout=30):
        u = urlparse(url)
        self.calls.append((method, url))
        path = u.path + (("?" + u.query) if u.query else "")
        if u.netloc == "console.vast.ai":
            return self.vast(method, u.path[len("/api/v0"):], json_body)
        if u.netloc in self.machines:
            return self.machines[u.netloc](method, path, headers, json_body)
        return R(502, raw=b"down")

    def vast(self, method, path, body):
        if path == "/bundles/":
            return R(body={"offers": [
                {"id": 11, "gpu_name": "RTX 5090", "dph_total": 0.47, "geolocation": "Germany, DE", "reliability2": 0.99,
                 "gpu_ram": 32768, "inet_down": 900, "disk_space": 200},
                {"id": 12, "gpu_name": "RTX 5090", "dph_total": 0.41, "geolocation": "Texas, US", "reliability2": 0.98},
                {"id": 13, "gpu_name": "RTX 5090", "dph_total": 1.90, "geolocation": "Sweden, SE", "reliability2": 0.99}]})
        if path.startswith("/asks/"):
            self.rented.append(body)
            iid = str(500 + len(self.rented))
            self.vast_instances[iid] = {"id": int(iid), "actual_status": "loading", "dph_total": 0.47}
            return R(body={"success": True, "new_contract": int(iid)})
        if path == "/instances/":
            return R(body={"instances": list(self.vast_instances.values())})
        if path.startswith("/instances/"):
            iid = path.strip("/").split("/")[1]
            if method == "DELETE":
                self.vast_instances.pop(iid, None)
            else:
                self.vast_instances[iid]["actual_status"] = "running" if body["state"] == "running" else "exited"
            return R(body={"success": True})
        return R(404, raw=b"?")


class FakeWS:
    def __init__(self, msgs, url, headers):
        self.msgs, self.url, self.headers, self.closed = msgs, url, headers, False

    def recv(self, timeout=None):
        if self.msgs:
            return self.msgs.pop(0)
        time.sleep(0.01)
        raise TimeoutError

    def close(self):
        self.closed = True


JPEG = b"\xff\xd8\xff\xe0" + b"x" * 40


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        gp.set_state_path(Path(self.tmp.name) / "pool.json")
        self.net = Net()
        gp.set_transport(self.net)
        self.keys = {"VAST_API_KEY": "vk-secret"}
        gp.set_secret_reader(lambda n: self.keys.get(n, ""))
        gp._last_provider_poll = 0
        gp.listeners.clear()
        self.ws = []
        gp.set_ws_connect(self.fake_ws)
        self.dest = str(Path(self.tmp.name) / "out")

    def fake_ws(self, url, headers):
        conn = FakeWS(list(self.ws), url, headers)
        self.ws.clear()
        self.ws_conns = getattr(self, "ws_conns", []) + [conn]
        return conn

    def tearDown(self):
        gp.set_ws_connect(None)
        gp.set_transport(None)
        gp.set_secret_reader(None)
        gp.set_state_path(None)
        self.tmp.cleanup()

    def add(self, host, comfy=None):
        self.net.machines[host] = comfy or FakeComfy()
        return gp.add_url(f"http://{host}", token="tok-" + host)

    def run_ticks(self, n=4):
        for _ in range(n):
            gp.tick(dt=1)

    def test_a_batch_fans_out_with_its_own_seeds_and_comes_back(self):
        a, b = self.add("a:8188"), self.add("b:8188")
        got = []
        gp.listeners.append(lambda job, task, files: got.append(files))
        job = gp.submit(WF, name="cars", batch=4, seed_mode="increment", dest=self.dest)
        self.run_ticks(6)
        st = gp.state()
        j = st["jobs"][0]
        self.assertEqual(j["state"], "done")
        self.assertEqual(j["counts"], {"done": 4})
        seeds = sorted(p["2"]["inputs"]["seed"] for m in self.net.machines.values() for p in m.prompts.values())
        self.assertEqual(seeds, [100, 101, 102, 103])
        self.assertEqual({len(m.prompts) for m in self.net.machines.values()}, {2})     # both machines worked
        files = [f for fs in got for f in fs]
        self.assertEqual(len(files), 4)
        self.assertTrue(all(Path(f).is_file() and job["id"] in f for f in files))
        self.assertIn("Bearer tok-a:8188", self.net.machines["a:8188"].auth)

    def test_the_state_never_holds_a_key_or_a_token(self):
        self.add("a:8188")
        text = json.dumps(gp.state())
        self.assertNotIn("vk-secret", text)
        self.assertNotIn("tok-a", text)
        self.assertTrue(gp.state()["keys"]["vast"])
        self.assertFalse(gp.state()["keys"]["runpod"])

    def test_a_failed_render_is_tried_once_more_on_another_machine(self):
        self.add("bad:1", FakeComfy(fail=True))
        gp.submit(WF, batch=1, dest=self.dest)
        gp.tick(dt=1)                      # bad is the only one ready: it takes the task
        self.add("good:1")
        self.run_ticks(5)
        t = gp.state()["jobs"][0]["tasks"][0]
        self.assertEqual(t["state"], "done")
        self.assertEqual(t["tries"], 2)

    def test_parity_names_missing_nodes_and_models(self):
        m = self.add("a:1", FakeComfy(nodes=("KSampler", "SaveImage", "CheckpointLoaderSimple"), models=()))
        res = gp.check_parity(m["id"], dict(WF, **{"4": {"class_type": "SZ_GetPath", "inputs": {}}}))
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing_nodes"], ["SZ_GetPath"])
        self.assertEqual(res["missing_models"], ["checkpoints/z_image_turbo.safetensors"])

    def test_vast_offers_are_eu_only_and_marked_when_too_dear(self):
        rows = gp.offers("vast")
        self.assertEqual([o["offer_id"] for o in rows], ["11", "13"])     # Texas left out
        self.assertEqual([o["too_dear"] for o in rows], [False, True])
        gp.set_settings({"eu_only": False})
        self.assertEqual(len(gp.offers("vast")), 3)

    def test_the_api_address_comes_from_env_or_the_default(self):
        self.assertEqual(gp._api_url("vast"), "https://console.vast.ai/api/v0")
        self.assertEqual(gp._api_url("runpod"), "https://rest.runpod.io/v1")
        self.keys["VAST_API_URL"] = "https://vast.example/api/v0/ "
        with self.assertRaises(gp.PoolError):      # the fake net only answers console.vast.ai
            gp.offers("vast")
        self.assertTrue(self.net.calls[-1][1].startswith("https://vast.example/api/v0/bundles/"))

    def test_gpu_names_match_with_wildcards_and_without_exact_case(self):
        m = gp.gpu_matches
        self.assertTrue(m("", "RTX 4090"))
        self.assertTrue(m("rtx 5090", "RTX 5090"))
        self.assertTrue(m("5090", "RTX 5090"))
        self.assertTrue(m("RTX 50*", "RTX 5070 Ti"))
        self.assertTrue(m("*pro 6000*", "RTX PRO 6000 WS"))
        self.assertTrue(m("H?00*", "H100 SXM"))
        self.assertFalse(m("RTX 50*", "RTX 4090"))
        self.assertFalse(m("5090", "RTX 4090"))

    def test_vast_offers_take_a_wildcard(self):
        self.assertEqual([o["offer_id"] for o in gp.offers("vast", gpu="rtx 50*")], ["11", "13"])
        self.assertEqual(gp.offers("vast", gpu="*4090*"), [])
        q = [c for c in self.net.calls if c[1].endswith("/bundles/")]
        self.assertTrue(q)

    def test_a_machine_terminated_on_the_provider_site_leaves_the_farm(self):
        gp.set_settings({"image": "example/comfy:1"})
        m = gp.rent("vast", "11", usd_per_hour=0.47, gpu="RTX 5090")
        gp._poll_providers(force=True)
        self.assertEqual(len(gp.state()["machines"]), 1)
        self.net.vast_instances.clear()                     # destroyed on cloud.vast.ai
        gp._poll_providers(force=True)                      # just rented: still in its grace period
        self.assertEqual(len(gp.state()["machines"]), 1)
        later = time.time() + gp.GONE_GRACE_SECONDS + 5
        with mock.patch.object(gp, "_now", return_value=later):
            gp._poll_providers(force=True)
        self.assertEqual(gp.state()["machines"], [])
        self.assertNotIn(m["id"], json.dumps(gp.state()["machines"]))

    def test_renting_needs_an_image_a_key_and_room_under_the_cap(self):
        with self.assertRaisesRegex(gp.PoolError, "machine image"):
            gp.rent("vast", "11", usd_per_hour=0.47)
        gp.set_settings({"image": "example/comfy:1"})
        with self.assertRaisesRegex(gp.PoolError, "limit"):
            gp.rent("vast", "13", usd_per_hour=1.9)
        with self.assertRaisesRegex(gp.PoolError, "RUNPOD_API_KEY"):
            gp.rent("runpod", "x", usd_per_hour=0.5)
        m = gp.rent("vast", "11", usd_per_hour=0.47, gpu="RTX 5090")
        self.assertEqual(self.net.rented[0]["image"], "example/comfy:1")
        self.assertEqual(m["state"], "starting")
        self.assertNotIn("token", m)

    def test_spend_is_billed_and_the_cap_stops_rented_machines(self):
        gp.set_settings({"image": "example/comfy:1", "daily_cap_usd": 1.0, "idle_stop_minutes": 0})
        m = gp.rent("vast", "11", usd_per_hour=0.47)
        iid = self.net.vast_instances and next(iter(self.net.vast_instances))
        self.net.vast_instances[iid].update(actual_status="running", public_ipaddr="1.2.3.4",
                                            ports={"8188/tcp": [{"HostPort": "40000"}]})
        self.net.machines["1.2.3.4:40000"] = FakeComfy()
        gp.tick(dt=3600)                    # one hour at $0.47
        st = gp.state()
        self.assertAlmostEqual(st["spend"]["today"], 0.47, places=3)
        self.assertEqual(st["machines"][0]["state"], "ready")
        gp._last_provider_poll = 0
        gp.tick(dt=3600 * 2)                # over the $1 cap
        st = gp.state()
        self.assertEqual(st["machines"][0]["state"], "stopped")
        self.assertIn("cap", st["machines"][0]["error"])
        self.assertEqual(self.net.vast_instances[iid]["actual_status"], "exited")
        with self.assertRaisesRegex(gp.PoolError, "cap"):
            gp.start(m["id"])

    def test_an_idle_rented_machine_is_stopped(self):
        gp.set_settings({"image": "example/comfy:1", "idle_stop_minutes": 1})
        gp.rent("vast", "11", usd_per_hour=0.47)
        iid = next(iter(self.net.vast_instances))
        self.net.vast_instances[iid].update(actual_status="running", public_ipaddr="5.6.7.8",
                                            ports={"8188/tcp": [{"HostPort": "1"}]})
        self.net.machines["5.6.7.8:1"] = FakeComfy()
        gp.tick(dt=1)
        mid = gp.state()["machines"][0]["id"]
        gp._load()["machines"][mid]["idle_since"] -= 120
        gp.tick(dt=1)
        self.assertEqual(gp.state()["machines"][0]["state"], "stopped")

    def test_cancel_and_retry(self):
        gp.submit(WF, batch=3, dest=self.dest)
        j = gp.state()["jobs"][0]
        self.assertEqual(gp.cancel(j["id"])["state"], "cancelled")
        self.add("a:1")
        r = gp.retry(j["id"])
        self.assertEqual(r["counts"], {"queued": 3})
        self.run_ticks(8)
        self.assertEqual(gp.state()["jobs"][0]["state"], "done")

    def test_actions_answer_with_the_state_and_refuse_unknown_ones(self):
        res = gp.action({"action": "add_url", "url": "http://c:8188", "name": "office"})
        self.assertEqual(res["state"]["machines"][0]["name"], "office")
        with self.assertRaises(gp.PoolError):
            gp.action({"action": "rm -rf"})
        with self.assertRaises(gp.PoolError):
            gp.action({"action": "add_url", "url": "file:///etc/passwd"})
        with self.assertRaises(gp.PoolError):
            gp.submit({"x": 1})

    def test_a_rented_machine_is_not_removed_only_terminated(self):
        gp.set_settings({"image": "example/comfy:1"})
        m = gp.rent("vast", "11", usd_per_hour=0.47)
        with self.assertRaisesRegex(gp.PoolError, "terminated"):
            gp.remove(m["id"])
        gp.terminate(m["id"])
        self.assertEqual(gp.state()["machines"], [])
        self.assertEqual(self.net.vast_instances, {})


    def test_a_preview_frame_is_found_by_its_magic_bytes(self):
        self.assertEqual(gp.preview_of(b"\x00\x00\x00\x01\x00\x00\x00\x01" + JPEG), (JPEG, "image/jpeg"))
        meta = json.dumps({"node_id": "2", "prompt_id": "p0"}).encode()
        frame = b"\x00\x00\x00\x04" + len(meta).to_bytes(4, "big") + meta + b"\x89PNG" + b"y" * 30
        self.assertEqual(gp.preview_of(frame)[1], "image/png")
        self.assertIsNone(gp.preview_of(b"\x00" * 40))

    def test_the_running_render_shows_its_node_steps_and_preview(self):
        m = self.add("a:8188")
        gp.submit(WF, name="cars", batch=1, dest=self.dest)
        fake = self.net.machines["a:8188"]
        self.net.machines["a:8188"] = lambda method, path, h, b: (R(body={}) if path.startswith("/history/") else fake(method, path, h, b))
        gp.tick(dt=1)
        mm = gp._load()["machines"][m["id"]]
        for msg in [json.dumps({"type": "execution_start", "data": {"prompt_id": "p0"}}),
                    json.dumps({"type": "execution_cached", "data": {"prompt_id": "p0", "nodes": ["1"]}}),
                    json.dumps({"type": "executing", "data": {"node": "2", "prompt_id": "p0"}}),
                    json.dumps({"type": "progress", "data": {"value": 12, "max": 30, "prompt_id": "p0", "node": "2"}}),
                    json.dumps({"type": "progress", "data": {"value": 29, "max": 30, "prompt_id": "OTHER"}}),
                    b"\x00\x00\x00\x01\x00\x00\x00\x01" + JPEG]:
            gp.handle_ws_message(mm, msg)
        live = gp.state()["machines"][0]["live"]
        self.assertEqual((live["class_type"], live["value"], live["max"], live["nodes_done"], live["nodes_total"]),
                         ("KSampler", 12, 30, 1, 3))
        self.assertTrue(live["preview"].startswith("data:image/jpeg;base64,"))

    def test_the_feed_is_heard_with_the_pools_own_client_id_and_token(self):
        m = self.add("a:8188")
        fake = self.net.machines["a:8188"]
        self.net.machines["a:8188"] = lambda method, path, h, b: (R(body={}) if path.startswith("/history/") else fake(method, path, h, b))
        self.ws += [json.dumps({"type": "execution_start", "data": {"prompt_id": "p0"}}),
                    json.dumps({"type": "progress", "data": {"value": 3, "max": 8, "prompt_id": "p0"}})]
        gp.submit(WF, batch=1, dest=self.dest)
        gp.tick(dt=1)
        for _ in range(100):
            if gp.live_of(m["id"]).get("value") == 3:
                break
            time.sleep(0.02)
        conn = self.ws_conns[0]
        self.assertTrue(conn.url.startswith("ws://a:8188/ws?clientId=agentY-farm-"))
        self.assertEqual(conn.headers, {"Authorization": "Bearer tok-a:8188"})
        self.assertEqual(gp.live_of(m["id"])["value"], 3)
        sent = fake.prompts["p0"]
        self.assertIn("2", sent)
        self.assertTrue(any(c[1].endswith("/prompt") for c in self.net.calls))


if __name__ == "__main__":
    unittest.main()
