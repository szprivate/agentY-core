"""worldlabs_atlas offline: HTTP mocked, files into a temp dir, depth reading stubbed."""

import io
import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from agenty_core.tools import worldlabs_atlas as w


def png_bytes(size=(1280, 720), color=(120, 90, 60)):
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


class Resp:
    def __init__(self, status=200, body=None, content=b"", headers=None):
        self.status_code, self._b, self.content = status, body if body is not None else {}, content
        self.headers = headers or {}
        self.text = json.dumps(self._b)

    def json(self):
        return self._b


CAM = {"extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "coordinateSystem": "rub"},
       "intrinsics": {"fx": 900, "fy": 900, "cx": 640, "cy": 360, "width": 1280, "height": 720}}


def frame(i, depth=True):
    f = {"imageAsset": {"assetId": f"asset_img{i}", "url": f"https://dl/img{i}.png"}, "camera": CAM}
    if depth:
        f["depth"] = {"depthAsset": {"assetId": f"asset_d{i}", "url": f"https://dl/d{i}.exr"},
                      "confidenceAsset": {"assetId": f"asset_c{i}", "url": f"https://dl/c{i}.png"}}
    return f


def done_op(op_id, response):
    return {"id": op_id, "done": True, "error": None, "response": response, "metadata": {"task": "x"}}


def read_glb(path):
    raw = Path(path).read_bytes()
    magic, ver, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67 and ver == 2 and total == len(raw)
    jl, jt = struct.unpack("<II", raw[12:20])
    gltf = json.loads(raw[20:20 + jl])
    bl, bt = struct.unpack("<II", raw[20 + jl:28 + jl])
    assert jt == 0x4E4F534A and bt == 0x004E4942
    return gltf, raw[28 + jl:28 + jl + bl]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for target, val in [(os.environ, {"WLT_API_KEY": "wl_test", "WORLDLABS_API_KEY": "", "WLT_API_BASE_URL": ""})]:
            p = mock.patch.dict(target, val)
            p.start()
            self.addCleanup(p.stop)
        for n, v in [("_sleep", lambda s: None)]:
            p = mock.patch.object(w, n, v)
            p.start()
            self.addCleanup(p.stop)
        # depth: a gently tilted plane 4-5 units away, whatever the file says
        p = mock.patch.object(w, "_read_exr", lambda path: np.tile(np.linspace(4.0, 5.0, 1280, dtype="float32"), (720, 1)))
        p.start()
        self.addCleanup(p.stop)
        self.calls = []

    def api(self, routes, downloads=None):
        """routes: {(METHOD, path-suffix): callable(body)->Resp|dict}"""
        def request(method, url, headers=None, json=None, params=None, data=None, timeout=None, **kw):
            self.calls.append((method, url, json, headers))
            path = url.split("/api/v2", 1)[-1] if "/api/v2" in url else url
            for (m, suffix), fn in routes.items():
                if m == method and path.endswith(suffix):
                    out = fn(json) if callable(fn) else fn
                    return out if isinstance(out, Resp) else Resp(200, out)
            return Resp(404, {"detail": f"no route {method} {path}"})

        def get(url, **kw):
            data = (downloads or {}).get(url)
            if data is None:
                data = png_bytes() if url.endswith((".png", ".jpg")) else b"EXR"
            return Resp(200, content=data)

        return mock.patch.object(w.requests, "request", side_effect=request), mock.patch.object(w.requests, "get", side_effect=get)

    def run_tool(self, fn, routes, **kw):
        a, b = self.api(routes)
        with a, b:
            return json.loads(fn(output_dir=self.tmp.name, **kw))


class CreateTests(Base):
    def test_image_to_world(self):
        img = Path(self.tmp.name) / "in.png"
        img.write_bytes(png_bytes((64, 64)))
        waits = iter([{"id": "op_1", "done": False}, done_op("op_1", {"frames": [frame(0)], "poseRefinementAccepted": False})])
        out = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:images2PosedRGBD"): {"id": "op_1", "done": False},
            ("POST", "/operations/op_1:wait"): lambda b: next(waits),
        }, name="Room 1", image_paths=[str(img)])
        self.assertEqual(out["status"], "ok", out)
        sub = next(c for c in self.calls if c[1].endswith("tasks:images2PosedRGBD"))
        self.assertEqual(sub[3]["WLT-Api-Key"], "wl_test")
        self.assertTrue(sub[3]["Idempotency-Key"])
        self.assertIn("base64", sub[2]["frames"][0]["imageAsset"])
        self.assertEqual(sub[2]["targetResolution"], [1280, 720])
        wd = Path(out["world_dir"])
        self.assertEqual(wd.name, "Room_1")
        for f in ("view_000.png", "depth_000.exr", "conf_000.png", "world.glb", "world.json",
                  "view_000.png.agenty.json", "world.glb.agenty.json"):
            self.assertTrue((wd / f).exists(), f)
        gltf, binary = read_glb(out["glb"])
        self.assertEqual(len(gltf["meshes"]), 1)
        self.assertGreater(out["mesh"]["triangles"], 1000)
        pos = gltf["accessors"][0]
        # the first camera is the origin; the plane lies in front of it (-Z), 4..5 units away
        self.assertAlmostEqual(pos["max"][2], -4.0, delta=0.05)
        self.assertAlmostEqual(pos["min"][2], -5.0, delta=0.05)
        side = json.loads((wd / "world.glb.agenty.json").read_text())
        self.assertEqual(side["provider"], "worldlabs-atlas")
        self.assertIsNone(out["usage"])
        self.assertFalse(any("pending" in str(p) for p in wd.glob("pending/*.json")))

    def test_prompt_to_world_asks_for_depth(self):
        out = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)], "promptUsed": "a courtyard", "aspectRatio": "16:9"}),
        }, name="court", prompt="a stone courtyard")
        self.assertEqual(out["status"], "ok", out)
        sub = next(c for c in self.calls if c[1].endswith("atlasTextToImage"))
        self.assertTrue(sub[2]["returnDepth"])
        self.assertEqual(out["prompt_used"], "a courtyard")

    def test_timeout_is_resumable(self):
        routes = {("POST", "/tasks:atlasTextToImage"): {"id": "op_slow", "done": False},
                  ("POST", "/operations/op_slow:wait"): {"id": "op_slow", "done": False}}
        clock = iter([0, 0, 0, 0, 1000, 1000, 1000, 1000])
        with mock.patch.object(w.time, "monotonic", lambda: next(clock, 2000)):
            out = self.run_tool(w.atlas_create_world, routes, name="slow", prompt="x", timeout_s=5)
        self.assertEqual(out["error_kind"], "timeout")
        self.assertEqual(out["operation_id"], "op_slow")
        self.assertTrue(list(Path(self.tmp.name).glob("slow/pending/op_slow.json")))
        # later: it is done
        fin = self.run_tool(w.atlas_fetch, {
            ("GET", "/operations/op_slow"): done_op("op_slow", {"frames": [frame(0)], "promptUsed": "p"}),
        }, operation_id="op_slow")
        self.assertEqual(fin["status"], "ok", fin)
        self.assertTrue((Path(self.tmp.name) / "slow" / "world.glb").exists())

    def test_existing_name_refused_and_args(self):
        (Path(self.tmp.name) / "a").mkdir()
        (Path(self.tmp.name) / "a" / "world.json").write_text("{}")
        self.assertEqual(self.run_tool(w.atlas_create_world, {}, name="a", prompt="x")["error_kind"], "args")
        self.assertEqual(self.run_tool(w.atlas_create_world, {}, name="b")["error_kind"], "args")

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"WLT_API_KEY": "", "WORLDLABS_API_KEY": ""}), \
                mock.patch("agenty_core.utils.secrets.get_secret", lambda k, d="": ""):
            out = json.loads(w.atlas_create_world(name="x", prompt="y", output_dir=self.tmp.name))
        self.assertEqual(out["error_kind"], "auth")
        self.assertNotIn("wl_test", json.dumps(out))


class ErrorTests(Base):
    def submit_error(self, resp):
        return self.run_tool(w.atlas_create_world, {("POST", "/tasks:atlasTextToImage"): resp}, name="e", prompt="x")

    def test_http_classes(self):
        cases = [(401, {}, "auth"), (402, {"code": "CREDITS_UNAVAILABLE"}, "balance"),
                 (402, {"code": "SPEND_LIMIT_EXCEEDED"}, "spend_limit"), (403, {}, "scope"),
                 (404, {}, "not_found"), (422, {"detail": [{"loc": ["body"]}]}, "args")]
        for status, body, kind in cases:
            self.assertEqual(self.submit_error(Resp(status, body))["error_kind"], kind, status)

    def test_rate_limit_retries_then_fails(self):
        out = self.submit_error(Resp(429, {"code": "RATE_LIMIT_EXCEEDED"}, headers={"Retry-After": "7"}))
        self.assertEqual(out["error_kind"], "rate_limit")
        self.assertEqual(sum(1 for c in self.calls if "tasks:" in c[1]), 4)

    def test_concurrency_limit_is_its_own_kind_with_a_hint(self):
        resp = Resp(429, {"code": "CONCURRENCY_LIMIT_EXCEEDED"}, headers={"Retry-After": "30"})
        out = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): resp,
            ("GET", "/operations"): {"operations": [{"id": "op_run", "done": False, "createdAtSecs": 1,
                                                      "metadata": {"task": "atlasGenerate"}}]}},
            name="c", prompt="x")
        self.assertEqual(out["error_kind"], "concurrency")
        self.assertEqual(out["in_flight"][0]["id"], "op_run")
        self.assertIn("op_run", out["hint"])
        n = sum(1 for c in self.calls if "tasks:" in c[1])
        self.assertGreater(n, 1)   # retried inside the budget, not given up at once

    def test_operation_errors(self):
        def failed(reason, code):
            return {"id": "op_f", "done": True, "response": None,
                    "error": {"code": code, "message": "m", "details": [{"reason": reason}]}}
        for reason, code, kind in [("CONTENT_POLICY", 3, "content_refusal"), ("EXECUTION_FAILED", 13, "execution"),
                                   ("EXECUTION_TIMEOUT", 4, "execution_timeout")]:
            out = self.submit_error(Resp(200, failed(reason, code)))
            self.assertEqual(out["error_kind"], kind)
        self.assertTrue(self.submit_error(Resp(200, failed("CONTENT_POLICY", 3)))["content_refusal"])


class ExpandTests(Base):
    def make_world(self):
        out = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)], "promptUsed": "a courtyard"}),
        }, name="court", prompt="a courtyard")
        self.assertEqual(out["status"], "ok", out)
        self.calls.clear()

    def test_expand_sends_context_ids_and_world_frame_cameras(self):
        self.make_world()
        gen = {"frames": [{**frame(1), "camera": {**CAM, "extrinsics": {**CAM["extrinsics"], "position": [0, 0, -2]}}}],
               "promptUsed": "a courtyard"}
        out = self.run_tool(w.atlas_expand_world, {("POST", "/tasks:atlasGenerate"): done_op("op_g", gen)},
                            name="court", moves=[{"forward": 2}])
        self.assertEqual(out["status"], "ok", out)
        body = next(c for c in self.calls if c[1].endswith("tasks:atlasGenerate"))[2]
        ctx = body["contextFrames"][0]
        self.assertEqual(ctx["imageAsset"], {"assetId": "asset_img0"})
        self.assertEqual(ctx["depth"], {"depthAsset": {"assetId": "asset_d0"}})
        tgt = body["targetCameras"][0]["extrinsics"]["position"]
        self.assertAlmostEqual(tgt[2], -2.0, places=4)            # forward = along -Z of the first camera
        self.assertTrue(body["returnDepth"])
        self.assertFalse(body["enhancePrompt"])                    # reuses promptUsed
        self.assertEqual(body["prompt"], "a courtyard")
        self.assertEqual(len(out["frames"]), 2)
        gltf, _ = read_glb(out["glb"])
        self.assertEqual(len(gltf["meshes"]), 2)

    def test_expand_follows_the_first_cameras_orientation(self):
        self.make_world()
        wj = Path(self.tmp.name) / "court" / "world.json"
        world = json.loads(wj.read_text())
        s = math.sqrt(0.5)       # first camera turned 90 degrees about Y (looks along -X)
        world["frames"][0]["camera"]["extrinsics"]["quaternion"] = [0, s, 0, s]
        wj.write_text(json.dumps(world))
        cams = w._target_cameras(json.loads(wj.read_text()), [{"t": 0, "forward": 3}], 0)
        px, py, pz = cams[0]["extrinsics"]["position"]
        self.assertAlmostEqual(px, -3.0, places=4)
        self.assertAlmostEqual(pz, 0.0, places=4)

    def test_expand_needs_a_world(self):
        out = self.run_tool(w.atlas_expand_world, {}, name="nope", moves=[{"forward": 1}])
        self.assertEqual(out["error_kind"], "args")


class MeshTests(unittest.TestCase):
    def test_plane_geometry_and_jump_cut(self):
        K = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 25.0, "width": 100, "height": 50}
        cam = w.normalize_camera({"extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1]}, "intrinsics": K})
        depth = np.full((50, 100), 3.0, dtype="float32")
        pos, uv, tri = w.frame_mesh((100, 50), K, depth, None, cam, cam, stride=5)
        self.assertTrue(np.allclose(pos[:, 2], -3.0))
        # a pixel left of centre maps to -x; above centre to +y
        self.assertLess(pos[:, 0].min(), 0)
        self.assertGreater(pos[:, 1].max(), 0)
        self.assertTrue(uv.min() >= 0 and uv.max() <= 1)
        self.assertEqual(tri.max() + 1, len(pos))
        depth[:, 50:] = 30.0                                          # a cliff down the middle
        _p, _u, tri2 = w.frame_mesh((100, 50), K, depth, None, cam, cam, stride=5)
        self.assertLess(len(tri2), len(tri))
        # invalid depth leaves holes, never NaNs
        depth[10:20, 10:20] = np.nan
        p3, _u3, _t3 = w.frame_mesh((100, 50), K, depth, None, cam, cam, stride=5)
        self.assertTrue(np.isfinite(p3).all())

    def test_second_camera_lands_in_first_cameras_frame(self):
        K = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 25.0, "width": 100, "height": 50}
        base = w.normalize_camera({"extrinsics": {"position": [10, 0, 0], "quaternion": [0, 0, 0, 1]}, "intrinsics": K})
        other = w.normalize_camera({"extrinsics": {"position": [10, 0, -2], "quaternion": [0, 0, 0, 1]}, "intrinsics": K})
        pos, _u, _t = w.frame_mesh((100, 50), K, np.full((50, 100), 3.0, "float32"), None, other, base, stride=10)
        self.assertTrue(np.allclose(pos[:, 2], -5.0, atol=1e-4))      # 2 further along -Z plus 3 deep

    def test_rdf_cameras_convert_to_rub(self):
        K = {"fx": 1, "fy": 1, "cx": 0, "cy": 0, "width": 2, "height": 2}
        n = w.normalize_camera({"extrinsics": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1],
                                               "coordinateSystem": "rdf"}, "intrinsics": K})
        self.assertEqual([round(n["R"][i][i]) for i in range(3)], [1, -1, -1])

    def test_quaternion_roundtrip(self):
        for q in ([0, 0, 0, 1], [0.5, 0.5, 0.5, 0.5], [0, math.sqrt(0.5), 0, math.sqrt(0.5)], [1, 0, 0, 0]):
            back = w._mat_to_quat(w._quat_to_mat(q))
            same = all(abs(a - b) < 1e-6 for a, b in zip(q, back)) or all(abs(a + b) < 1e-6 for a, b in zip(q, back))
            self.assertTrue(same, (q, back))

    def test_log_depth_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "d.png"
            a = np.zeros((4, 4), dtype="uint8")
            a[:, 0], a[:, 3] = 255, 0                                 # white = zMin, black = zMax
            Image.fromarray(a).save(p)
            z = w._decode_depth(p, {"zMin": 1.0, "zMax": 100.0})
            self.assertAlmostEqual(float(z[0, 0]), 1.0, places=4)
            self.assertAlmostEqual(float(z[0, 3]), 100.0, places=2)

    def test_glb_is_well_formed(self):
        with tempfile.TemporaryDirectory() as d:
            part = {"positions": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], "f4"),
                    "uvs": np.array([[0, 0], [1, 0], [0, 1]], "f4"), "indices": np.array([[0, 1, 2]], "u4"),
                    "image": png_bytes((4, 4)), "mime": "image/png"}
            stats = w.write_glb(Path(d) / "a.glb", [part, part])
            gltf, binary = read_glb(Path(d) / "a.glb")
            self.assertEqual(stats["triangles"], 2)
            for v in gltf["bufferViews"]:
                self.assertEqual(v["byteOffset"] % 4, 0)
                self.assertLessEqual(v["byteOffset"] + v["byteLength"], len(binary))
            for a in gltf["accessors"]:
                self.assertLess(a["bufferView"], len(gltf["bufferViews"]))
            self.assertEqual(gltf["buffers"][0]["byteLength"], len(binary))

    @unittest.skipUnless(__import__("importlib").util.find_spec("OpenEXR"), "OpenEXR not installed")
    def test_exr_roundtrip(self):
        import OpenEXR
        with tempfile.TemporaryDirectory() as d:
            a = (np.random.rand(8, 16) * 9 + 1).astype("float32")
            p = Path(d) / "x.exr"
            OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION}, {"Y": a}).write(str(p))
            self.assertTrue(np.allclose(w._read_exr(p), a))


class ExrPreflightTests(Base):
    """A billed job must not finish into an unreadable result: the EXR reader is checked before the submit."""

    def no_readers(self):
        real = __import__
        def fake(name, *a, **k):
            if name in ("OpenEXR", "cv2"):
                raise ImportError(name)
            return real(name, *a, **k)
        p = mock.patch("builtins.__import__", fake)
        p.start()
        self.addCleanup(p.stop)

    def test_create_and_expand_stop_before_the_billed_call(self):
        routes = {("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)]}),
                  ("POST", "/tasks:atlasGenerate"): done_op("op_g", {"frames": []})}
        self.assertEqual(self.run_tool(w.atlas_create_world, routes, name="p", prompt="x")["status"], "ok")
        self.calls.clear()
        self.no_readers()
        for fn, kw in [(w.atlas_create_world, {"name": "q", "prompt": "x"}),
                       (w.atlas_expand_world, {"name": "p", "moves": [{"forward": 1}]})]:
            out = self.run_tool(fn, routes, **kw)
            self.assertEqual(out["error_kind"], "dependency", out)
        self.assertFalse([c for c in self.calls if "/tasks:" in c[1]], self.calls)

    def test_other_depth_tasks_share_the_check_but_splat_to_mesh_skips_it(self):
        self.no_readers()
        with mock.patch.object(w, "_request", side_effect=AssertionError("billed")):
            for task in (w._TASK_I2P, w._TASK_T2I, w._TASK_GEN):
                with self.assertRaises(w.AtlasError) as cm:
                    w._submit(task, {})
                self.assertEqual(cm.exception.kind, "dependency")
        with mock.patch.object(w, "_request", return_value={"id": "op"}) as req:
            self.assertEqual(w._submit(w._TASK_S2M, {}), {"id": "op"})
            req.assert_called_once()


class OtherTests(Base):
    def test_rebuild_is_offline_and_changes_detail(self):
        out = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)]}),
        }, name="r", prompt="x", mesh_detail=8)
        light = out["mesh"]["triangles"]
        self.calls.clear()
        with mock.patch.object(w.requests, "request", side_effect=AssertionError("no network")):
            out2 = json.loads(w.atlas_rebuild_world_mesh(name="r", mesh_detail=4, output_dir=self.tmp.name))
        self.assertGreater(out2["mesh"]["triangles"], light * 3)

    def test_splat_to_mesh_upload_flow(self):
        spz = Path(self.tmp.name) / "scene.spz"
        spz.write_bytes(b"spz-bytes")
        puts = []

        def put(method, url, data=None, headers=None, **kw):
            puts.append((method, url, headers))
            return Resp(200)

        routes = {("POST", "/assets"): {"id": "asset_s", "state": "CREATED"},
                  ("POST", "/assets/asset_s:createUploadUrl"): {"uploadUrl": "https://up/x", "method": "PUT", "headers": {"x-a": "1"}},
                  ("POST", "/assets/asset_s:completeUpload"): {"id": "asset_s"},
                  ("POST", "/tasks:splats2Mesh"): done_op("op_m", {"mesh": {"assetId": "asset_m", "url": "https://dl/m.glb"}, "anchorPositions": [[0, 0, 0]]})}
        a, b = self.api(routes, downloads={"https://dl/m.glb": b"glTF"})

        def dispatch(method, url, **kw):
            return put(method, url, **kw) if url.startswith("https://up/") else a.kwargs["side_effect"](method, url, **kw)
        with mock.patch.object(w.requests, "request", side_effect=dispatch), b:
            out = json.loads(w.atlas_splat_to_mesh(splat_path=str(spz), output_dir=self.tmp.name, target_faces=5000))
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(puts[0][0], "PUT")
        self.assertNotIn("WLT-Api-Key", puts[0][2])                  # the signed URL never gets the key
        self.assertEqual(puts[0][2]["x-a"], "1")
        body = next(c for c in self.calls if c[1].endswith("tasks:splats2Mesh"))[2]
        self.assertEqual(body["splatAsset"], {"assetId": "asset_s"})
        self.assertEqual(body["targetFaces"], 5000)
        self.assertTrue(Path(out["glb"]).exists())

    def test_list_worlds_and_operations(self):
        self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)]}),
        }, name="l", prompt="x")
        out = json.loads(w.atlas_list(what="worlds", output_dir=self.tmp.name))
        self.assertEqual([x["name"] for x in out["worlds"]], ["l"])
        a, b = self.api({("GET", "/operations"): {"operations": [
            {"id": "op_9", "done": True, "metadata": {"task": "atlasGenerate"}, "error": None, "createdAtSecs": 5}]}})
        with a, b:
            ops = json.loads(w.atlas_list(what="operations", task="atlasGenerate", status="succeeded", output_dir=self.tmp.name))
        self.assertEqual(ops["operations"][0]["task"], "atlasGenerate")
        self.assertEqual(self.calls[-1][1].split("?")[0].rsplit("/", 1)[-1], "operations")

    def test_download_replaces_an_expired_url(self):
        calls = []

        def get(url, **kw):
            calls.append(url)
            return Resp(403) if "old" in url else Resp(200, content=b"ok")
        with mock.patch.object(w.requests, "get", side_effect=get), \
                mock.patch.object(w, "_request", return_value={"readUrl": "https://new/x"}):
            dest = Path(self.tmp.name) / "f.bin"
            w._download({"assetId": "asset_1", "url": "https://old/x"}, dest)
        self.assertEqual(calls, ["https://old/x", "https://new/x"])
        self.assertEqual(dest.read_bytes(), b"ok")


if __name__ == "__main__":
    unittest.main()
