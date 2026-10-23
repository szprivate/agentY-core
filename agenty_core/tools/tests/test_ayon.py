"""ayon tools with the Ayon host, the Ayon server and ComfyUI mocked.

The RPC frames below are the ones the addon's own code produces (ayon_comfyui api/rpc_server.py, the
wsrpc_aiohttp it runs on, and wsrpc.js on the browser side), not invented ones:

* a client call            {"id": 3, "method": "ayonComfyUI.pingAyonMenu", "params": {"message": "x"}}
* the answer               {"result": "x", "id": 3}                      (handler._send(result=..., id=serial))
* an error answer          {"error": {"type": "...", "message": "..."}, "id": 3}
* the server's keepalive   {"id": 1, "method": "ping", "params": {"seq": 12.5}}   -> client echoes {"id": 1, "result": {"seq": 12.5}}
* the static page          page_templater.HTML_TMP with {{port}} and {{source}} filled in
"""

import json
import sys
import types
import unittest
from unittest import mock

from agenty_core.tools import ayon

# page_templater.template_html(webui_port=55056, comfy_url="http://127.0.0.1:8188")
STATIC_PAGE = """<!DOCTYPE html>
<script lang="text/javascript" type="module" src="../static/js/proxy.js">
</script>
<script lang="text/javascript">
window.rpc_port = 55056;
</script>
<body>
    <iframe id="server_iframe" src="http://127.0.0.1:8188"></iframe><br>
</body>"""

SESSION = {"static_url": "http://127.0.0.1:5454", "comfy_url": "http://127.0.0.1:8188", "rpc_port": 55056,
           "rpc_ok": True, "rpc_error": "", "same_comfyui_as_agenty": True}
LAUNCH = {"project": "i002_ai_rnd", "folder_path": "agentRND/agentRND_0010", "task": "ai",
          "workdir": "C:/work/ai", "server_url": "https://bepic.ayon.app"}


class FakeSocket:
    """What websockets.sync.client.connect returns: scripted incoming frames, recorded outgoing ones."""

    def __init__(self, incoming):
        self.incoming, self.sent = list(incoming), []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self, timeout=None):
        if not self.incoming:
            raise TimeoutError()
        item = self.incoming.pop(0)
        return item if isinstance(item, str) else json.dumps(item)


class StaticPage(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(ayon.parse_static_page(STATIC_PAGE), ("http://127.0.0.1:8188", 55056))
        self.assertEqual(ayon.parse_static_page("<html>other</html>"), (None, None))

    def test_discover(self):
        resp = mock.Mock(status_code=200, text=STATIC_PAGE)
        with mock.patch.object(ayon.requests, "get", return_value=resp) as g:
            s = ayon.discover_session()
        self.assertEqual(s, {"static_url": "http://127.0.0.1:5454", "comfy_url": "http://127.0.0.1:8188", "rpc_port": 55056})
        self.assertEqual(g.call_args.args[0], "http://127.0.0.1:5454")

    def test_discover_nothing_listening(self):
        with mock.patch.object(ayon.requests, "get", side_effect=ayon.requests.ConnectionError()):
            self.assertIsNone(ayon.discover_session())

    def test_discover_some_other_page_on_the_port(self):
        with mock.patch.object(ayon.requests, "get", return_value=mock.Mock(status_code=200, text="<html></html>")):
            self.assertIsNone(ayon.discover_session())


class Rpc(unittest.TestCase):
    def call(self, incoming, method="ayonComfyUI.pingAyonMenu", params=None, timeout=1.0):
        sock = FakeSocket(incoming)
        with mock.patch.object(ayon, "_ws_connect", return_value=sock) as c:
            try:
                out = ayon.rpc_call(55056, method, params, timeout=timeout)
            finally:
                self.connect = c
        return out, sock

    def test_ping_round_trip(self):
        out, sock = self.call([{"result": "agentY", "id": 3}], params={"message": "agentY"})
        self.assertEqual(out, "agentY")
        self.assertEqual(sock.sent, [{"id": 3, "method": "ayonComfyUI.pingAyonMenu", "params": {"message": "agentY"}}])
        self.assertEqual(self.connect.call_args.args[0], "ws://127.0.0.1:55056/ws/")

    def test_server_keepalive_is_answered_before_the_result(self):
        out, sock = self.call([{"id": 1, "method": "ping", "params": {"seq": 12.5}}, {"result": "ok", "id": 3}])
        self.assertEqual(out, "ok")
        self.assertIn({"id": 1, "result": {"seq": 12.5}}, sock.sent)

    def test_other_server_calls_are_refused_not_answered(self):
        _, sock = self.call([{"id": 5, "method": "getPublishNodes", "params": {}}, {"result": 1, "id": 3}])
        refusal = next(m for m in sock.sent if m.get("id") == 5)
        self.assertEqual(refusal["error"]["type"], "NotImplementedError")

    def test_error_answer(self):
        with self.assertRaises(ayon.AyonRpcError) as cm:
            self.call([{"error": {"type": "NotImplementedError", "message": "Method call of x is not implemented"}, "id": 3}])
        self.assertIn("NotImplementedError", str(cm.exception))

    def test_no_answer_times_out(self):
        with self.assertRaises(ayon.AyonRpcError) as cm:
            self.call([], timeout=0.2)
        self.assertIn("did not answer", str(cm.exception))

    def test_unreachable(self):
        with mock.patch.object(ayon, "_ws_connect", side_effect=OSError("refused")):
            with self.assertRaises(ayon.AyonRpcError):
                ayon.rpc_call(55056, "ayonComfyUI.pingAyonMenu")

    def test_request_tool_frame(self):
        out, sock = self.call([{"result": "publisher scheduled in qt_thread", "id": 3}],
                              method="ayonComfyUI.requestToolByName", params={"tool_name": "publisher"})
        self.assertEqual(sock.sent[0], {"id": 3, "method": "ayonComfyUI.requestToolByName",
                                        "params": {"tool_name": "publisher"}})
        self.assertEqual(out, "publisher scheduled in qt_thread")


class LaunchContext(unittest.TestCase):
    def fake_psutil(self, env):
        conn = types.SimpleNamespace(status="LISTEN", laddr=types.SimpleNamespace(port=8188), pid=4242)
        mod = types.SimpleNamespace(
            net_connections=lambda kind="tcp": [conn],
            Process=lambda pid: types.SimpleNamespace(environ=lambda: env))
        return mock.patch.dict(sys.modules, {"psutil": mod})

    def test_only_the_allow_list_is_read(self):
        env = {"AYON_PROJECT_NAME": "i002_ai_rnd", "AYON_FOLDER_PATH": "agentRND/agentRND_0010", "AYON_TASK_NAME": "ai",
               "AYON_WORKDIR": "C:/work/ai", "AYON_API_KEY": "SECRET", "PATH": "x"}
        with self.fake_psutil(env):
            got = ayon.launch_context_of("http://127.0.0.1:8188")
        self.assertEqual(got, {"project": "i002_ai_rnd", "folder_path": "agentRND/agentRND_0010", "task": "ai",
                               "workdir": "C:/work/ai"})
        self.assertNotIn("SECRET", json.dumps(got))

    def test_not_an_ayon_process(self):
        with self.fake_psutil({"PATH": "x"}):
            self.assertIsNone(ayon.launch_context_of("http://127.0.0.1:8188"))


class Documents(unittest.TestCase):
    def test_context_doc_is_made_wellformed(self):
        self.assertEqual(ayon.parse_context_doc(""), {"context": {}, "instances": [], "containers": []})
        self.assertEqual(ayon.parse_context_doc('{"context": {"a": 1}, "instances": null})')["instances"], [])
        self.assertEqual(ayon.parse_context_doc("not json")["containers"], [])

    def test_merge_unique(self):
        out = ayon._merge_unique([{"k": 1, "v": "old"}, {"k": 2}], [{"k": 1, "v": "new"}, {"k": 3}], "k")
        self.assertEqual([(i["k"], i.get("v")) for i in out], [(1, "new"), (2, None), (3, None)])

    def test_container_matches_containerise(self):
        c = ayon.build_container("imageMain", "agentRND_0010", "rep1", [{"name": "a.png", "subfolder": "AYON/x", "type": "input"}], "ImageLoader")
        self.assertEqual(c["schema"], "ayon:container-3.0")
        self.assertEqual(c["id"], "ayon.load.container")
        self.assertEqual(c["container_name"], "imageMain_CON")
        self.assertEqual(c["representation"], "rep1")

    def test_image_instance_matches_creator(self):
        i = ayon.build_instance("image", "image", "Main", LAUNCH)
        self.assertEqual(i["id"], "ayon.create.instance")
        self.assertEqual(i["creator_identifier"], "io.ayon.creators.comfyui.image")
        self.assertEqual(i["productName"], "ayon_ayon_imageMain")
        self.assertEqual((i["folderPath"], i["task"], i["projectName"]), ("agentRND/agentRND_0010", "ai", "i002_ai_rnd"))
        self.assertEqual(set(i["creator_attributes"]), {"keep_metadata", "use_unique_name", "prefix", "unique_name",
                                                        "compression_level", "force_recook_on_publish"})

    def test_video_and_model_attributes(self):
        v = ayon.build_instance("video", "render", "Main", LAUNCH)
        self.assertEqual(v["creator_identifier"], "io.ayon.creators.comfyui.render")
        self.assertEqual(v["creator_attributes"]["formatcodec"], "mp4/h264")
        m = ayon.build_instance("model", "model", "Main", LAUNCH)
        self.assertEqual(m["creator_attributes"]["fallback_format"], "glb")

    def test_product_name_override_and_no_unique_part(self):
        self.assertEqual(ayon.build_instance("image", "image", "Hero", LAUNCH, product_name="imageHero")["productName"], "imageHero")
        self.assertEqual(ayon.build_instance("image", "image", "Hero", LAUNCH, unique_name="", use_unique_name=False)["productName"],
                         "ayon_imageHero")


class Paths(unittest.TestCase):
    def test_expand_single_and_sequence(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            for n in (1, 2, 3):
                (Path(d) / f"s.{n:04d}.exr").write_bytes(b"x")
            self.assertEqual(len(ayon.expand_files(str(Path(d) / "s.%04d.exr"))), 3)
            self.assertEqual(len(ayon.expand_files(str(Path(d) / "s.####.exr"))), 3)
            self.assertEqual(ayon.expand_files(str(Path(d) / "s.0001.exr")), [str(Path(d) / "s.0001.exr")])
            self.assertEqual(ayon.expand_files(str(Path(d) / "missing.png")), [])


class FakeClient:
    base_url = "http://127.0.0.1:8188"

    def __init__(self, selection=None):
        self.posts, self.selection = [], selection or []

    def get(self, path, **kw):
        assert path == "/agent/canvas_selection"
        return {"ok": True, "nodes": self.selection}

    def post(self, path, json_data=None, data=None, files=None):
        self.posts.append((path, json_data, data))
        if path == "/upload/image":
            return {"name": "imageMain_v003.png", "subfolder": "AYON\\agentRND_0010_imageMain", "type": "input"}
        return {"ok": True, "applied": list((json_data or {}).get("params", {})), "node": "n"}


CTX_NODE_PROMPT = {"7": {"class_type": "AYON Context", "inputs": {"ayon_context_info": json.dumps(
    {"context": {"project_name": "i002_ai_rnd", "folder_path": "agentRND/agentRND_0010", "task_name": "ai"},
     "instances": [], "containers": []})}},
    "12": {"class_type": "VAEDecode", "inputs": {}}}


class ToolBase(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.placed = []
        self.addCleanup(ayon.set_canvas_provider, None)
        self.addCleanup(ayon.set_canvas_placer, None)
        ayon._last_written.clear()
        for name, value in (("_session", (dict(SESSION), "")), ("launch_context_of", dict(LAUNCH)),
                            ("_client_for", self.client)):
            p = mock.patch.object(ayon, name, **({"return_value": value} if name != "_client_for" else {"return_value": value}))
            p.start()
            self.addCleanup(p.stop)

    def placer(self, ops, reason):
        self.placed.append((ops, reason))
        return True


class NotLaunched(unittest.TestCase):
    def test_every_tool_says_so(self):
        with mock.patch.object(ayon, "discover_session", return_value=None):
            for out in (ayon.ayon_context(), ayon.ayon_load("x.png", "imageMain"), ayon.ayon_mark_publish("12"),
                        ayon.ayon_publish(), ayon.ayon_open_tool("loader")):
                self.assertEqual(json.loads(out)["status"], "not_launched_through_ayon")
        self.assertFalse(json.loads(ayon.ayon_context())["launched_through_ayon"])


class Context(ToolBase):
    def test_launch_context_and_canvas(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        out = json.loads(ayon.ayon_context())
        self.assertEqual(out["launch_context"]["task"], "ai")
        self.assertEqual(out["canvas"]["context_node"], "7")
        self.assertEqual(out["canvas"]["source"], "canvas")
        self.assertNotIn("folder", out)
        self.assertTrue(any("ayon-mcp" in n for n in out["notes"]))

    def test_selection_is_the_fallback_and_a_missing_node_is_said(self):
        self.client.selection = []
        out = json.loads(ayon.ayon_context())
        self.assertIsNone(out["canvas"]["context_node"])
        self.assertTrue(any("No AYON Context node" in n for n in out["notes"]))

    def test_context_switched_in_ayon_is_flagged(self):
        other = json.loads(CTX_NODE_PROMPT["7"]["inputs"]["ayon_context_info"])
        other["context"]["task_name"] = "comp"
        prompt = {"7": {"class_type": "AYON Context", "inputs": {"ayon_context_info": json.dumps(other)}}}
        ayon.set_canvas_provider(lambda: prompt)
        out = json.loads(ayon.ayon_context())
        self.assertTrue(any("another context" in n for n in out["notes"]))


class Load(ToolBase):
    def run_load(self, **kw):
        return json.loads(ayon.ayon_load(str(self.png()), "imageMain", representation_id="r1", folder_path="", version=3, **kw))

    def png(self):
        import tempfile
        from pathlib import Path
        d = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, d, True)
        p = Path(d) / "imageMain_v003.png"
        p.write_bytes(b"\x89PNG")
        return p

    def test_dry_run_uploads_and_changes_nothing(self):
        out = self.run_load(dry_run=True)
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["resolved"]["version"], 3)
        self.assertEqual(out["would_place"], "AYON Load Image")
        self.assertEqual(self.client.posts, [])
        self.assertEqual(self.placed, [])

    def test_load_adds_the_node_and_registers_the_container(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        ayon.set_canvas_placer(self.placer)
        out = self.run_load()
        self.assertEqual(out["status"], "loaded")
        ops, _ = self.placed[0]
        self.assertEqual(ops[0]["class_type"], "AYON Load Image")
        self.assertEqual(ops[0]["node_id"], "13")
        info = json.loads(ops[0]["params"]["ayon_container_info"])
        self.assertEqual(info["image_upload_info"][0]["subfolder"], "AYON/agentRND_0010_imageMain")
        self.assertEqual((info["loader"], info["representation"]), ("ImageLoader", "r1"))
        upload = next(p for p in self.client.posts if p[0] == "/upload/image")
        self.assertEqual(upload[2], {"type": "input", "subfolder": "AYON/agentRND_0010_imageMain"})
        ctx = next(p for p in self.client.posts if p[0] == "/agent/set_node_params")
        self.assertEqual(ctx[1]["node_id"], "7")
        doc = json.loads(ctx[1]["params"]["ayon_context_info"])
        self.assertEqual([c["container_uuid"] for c in doc["containers"]], [info["container_uuid"]])
        self.assertTrue(out["registered"]["registered"])

    def test_without_a_placer_the_node_data_is_handed_back(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        out = self.run_load()
        self.assertEqual(out["status"], "prepared")
        self.assertEqual(out["placed"]["paste"]["class_type"], "AYON Load Image")
        self.assertIn("container_uuid", out["placed"]["paste"]["value"])

    def test_fills_a_node_the_user_placed(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        out = self.run_load(load_node_id="21")
        self.assertEqual(out["placed"], {"how": "filled_existing", "node_id": "21"})
        self.assertEqual(self.client.posts[1][1]["node_id"], "21")

    def test_missing_file_is_an_error_not_a_crash(self):
        out = json.loads(ayon.ayon_load("C:/nowhere/x.png", "imageMain"))
        self.assertEqual(out["status"], "error")
        self.assertIn("does not exist", out["message"])


class MarkPublish(ToolBase):
    def test_adds_wires_and_registers(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        ayon.set_canvas_placer(self.placer)
        out = json.loads(ayon.ayon_mark_publish("12", kind="image", variant="Main"))
        self.assertEqual(out["status"], "marked")
        ops, _ = self.placed[0]
        self.assertEqual(ops[0]["class_type"], "AYON Image Save")
        self.assertEqual(ops[0]["near"], "12")
        self.assertEqual(ops[1], {"op": "connect", "from": "12", "output": 0, "to": "13", "input": "images_in"})
        info = json.loads(ops[0]["params"]["ayon_info"])
        self.assertEqual(info["productName"], "ayon_ayon_imageMain")
        doc = json.loads(next(p for p in self.client.posts if p[0] == "/agent/set_node_params")[1]["params"]["ayon_context_info"])
        self.assertEqual([i["instance_id"] for i in doc["instances"]], [info["instance_id"]])

    def test_two_marks_in_one_turn_both_stay_registered(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)      # the host's snapshot never learns of the first
        ayon.set_canvas_placer(self.placer)
        ayon.ayon_mark_publish("12", variant="A")
        ayon.ayon_mark_publish("12", variant="B")
        last = json.loads([p for p in self.client.posts if p[0] == "/agent/set_node_params"][-1][1]["params"]["ayon_context_info"])
        self.assertEqual(sorted(i["variant"] for i in last["instances"]), ["A", "B"])

    def test_without_a_placer_nothing_is_claimed(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        out = json.loads(ayon.ayon_mark_publish("12"))
        self.assertEqual(out["status"], "partly_marked")
        self.assertEqual(out["placed"]["paste"]["widget"], "ayon_info")
        self.assertTrue(out["registered"]["registered"])        # the context node exists, so that part works

    def test_no_context_node_and_no_placer_says_publisher_will_not_see_it(self):
        out = json.loads(ayon.ayon_mark_publish("12"))
        self.assertFalse(out["registered"]["registered"])
        self.assertTrue(any("Publisher will not list it" in t for t in out["todo"]))

    def test_adds_the_context_node_when_the_canvas_has_none(self):
        ayon.set_canvas_placer(self.placer)
        out = json.loads(ayon.ayon_mark_publish("12"))
        self.assertTrue(out["registered"]["added_context_node"])
        ctx_op = self.placed[-1][0][0]
        self.assertEqual(ctx_op["class_type"], "AYON Context")
        self.assertEqual(json.loads(ctx_op["params"]["ayon_context_info"])["context"]["task_name"], "ai")

    def test_fills_an_existing_save_node(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        out = json.loads(ayon.ayon_mark_publish("12", save_node_id="30"))
        self.assertEqual(out["placed"], {"how": "filled_existing", "node_id": "30"})

    def test_unknown_node_and_kind(self):
        ayon.set_canvas_provider(lambda: CTX_NODE_PROMPT)
        self.assertEqual(json.loads(ayon.ayon_mark_publish("99"))["status"], "error")
        self.assertEqual(json.loads(ayon.ayon_mark_publish("12", kind="sound"))["status"], "error")


class Publish(ToolBase):
    def test_nothing_registered(self):
        self.assertEqual(json.loads(ayon.ayon_publish())["status"], "nothing_to_publish")

    def test_opens_the_publisher_and_does_not_claim_a_publish(self):
        doc = {"context": {}, "instances": [{"instance_id": "i1", "productName": "ayon_ayon_imageMain"}], "containers": []}
        ayon.set_canvas_provider(lambda: {"7": {"class_type": "AYON Context", "inputs": {"ayon_context_info": json.dumps(doc)}}})
        with mock.patch.object(ayon, "rpc_call", return_value="publisher scheduled in qt_thread") as rpc:
            out = json.loads(ayon.ayon_publish())
        rpc.assert_called_once_with(55056, "ayonComfyUI.requestToolByName", {"tool_name": "publisher"})
        self.assertEqual((out["status"], out["instances"], out["headless"]), ("publisher_opened", ["ayon_ayon_imageMain"], False))
        self.assertIn("Press Publish", out["needs_user"])

    def test_open_tool(self):
        with mock.patch.object(ayon, "rpc_call", return_value="loader scheduled in qt_thread") as rpc:
            self.assertEqual(json.loads(ayon.ayon_open_tool("loader"))["status"], "requested")
        rpc.assert_called_once_with(55056, "ayonComfyUI.requestToolByName", {"tool_name": "loader"})
        self.assertEqual(json.loads(ayon.ayon_open_tool("quit"))["status"], "error")


if __name__ == "__main__":
    unittest.main()
