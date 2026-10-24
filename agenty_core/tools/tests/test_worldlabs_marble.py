"""worldlabs_marble offline: every HTTP call is a stub, nothing is billed."""

import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from agenty_core.tools import splat_formats as sf
from agenty_core.tools import worldlabs_atlas as wl
from agenty_core.tools import worldlabs_marble as m
from agenty_core.tools.tests.test_splat_formats import synthetic

BASE = "https://api.worldlabs.ai/marble/v1"


def png(color=(90, 120, 60)):
    b = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(b, "PNG")
    return b.getvalue()


class Resp:
    def __init__(self, status=200, body=None, content=b"", headers=None):
        self.status_code, self._b, self.content, self.headers = status, body if body is not None else {}, content, headers or {}
        self.text = json.dumps(self._b)

    def json(self):
        return self._b


def spz_bytes(k=3):
    d = tempfile.mkdtemp()
    props, a = synthetic(n=30, k=k)
    sf.write_spz(Path(d) / "t.spz", props, a)
    return (Path(d) / "t.spz").read_bytes()


def world_body(wid="w_1"):
    return {"world_id": wid, "display_name": "x", "world_marble_url": f"https://marble.worldlabs.ai/world/{wid}",
            "model": "marble-1.1-plus", "assets": {
                "caption": "a room", "thumbnail_url": "https://dl/thumb.jpg",
                "imagery": {"pano_url": "https://dl/pano.png"},
                "mesh": {"collider_mesh_url": "https://dl/collider.glb"},
                "splats": {"spz_urls": {"100k": "https://dl/100k.spz", "500k": "https://dl/500k.spz", "full_res": "https://dl/full.spz"},
                           "semantics_metadata": {"metric_scale_factor": 2.0, "ground_plane_offset": 0.5}}}}


def done_op(op_id="op_1", wid="w_1"):
    return {"operation_id": op_id, "done": True, "response": world_body(wid), "cost": {"total_credits": 1700, "line_items": []},
            "metadata": {"world_id": wid}}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = self.tmp.name
        p = mock.patch.dict(os.environ, {"WLT_MARBLE_API_KEY": "wl_marble_test", "WLT_API_KEY": "wl_atlas_test",
                                         "WORLDLABS_API_KEY": "", "WORLDLABS_MARBLE_API_KEY": "",
                                         "WLT_MARBLE_BASE_URL": "", "WLT_API_BASE_URL": ""})
        p.start()
        self.addCleanup(p.stop)
        for mod in (m, wl):
            p = mock.patch.object(mod, "_sleep", lambda s: None)
            p.start()
            self.addCleanup(p.stop)
        p = mock.patch.object(m, "_open_viewer", lambda path: {"tab": "t", "path": str(path)})
        p.start()
        self.addCleanup(p.stop)
        self.calls = []
        self.puts = []
        self.spz = spz_bytes()

    def stub(self, routes, downloads=None):
        """routes: {(METHOD, path): body | list of bodies | callable(json) -> body | Resp}; Marble paths are
        relative to its base, Atlas ones are prefixed ATLAS."""
        def request(method, url, headers=None, json=None, params=None, data=None, timeout=None, **kw):
            self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
            if url.startswith("https://up/"):
                self.puts.append((method, url, headers, data))
                return Resp(200)
            if url.startswith(BASE):
                path = url[len(BASE):]
            elif "atlas-beta" in url:
                path = "ATLAS" + url.split("/api/v2", 1)[-1]
            else:
                path = url
            for (mm, p), fn in routes.items():
                if mm == method and path == p:
                    out = (fn.pop(0) if len(fn) > 1 else fn[0]) if isinstance(fn, list) else fn
                    out = out(json) if callable(out) else out
                    return out if isinstance(out, Resp) else Resp(200, out)
            return Resp(404, {"detail": f"no route {method} {path}"})

        def get(url, **kw):
            self.calls.append({"method": "GET*", "url": url})
            data = (downloads or {}).get(url)
            if data is None:
                data = self.spz if url.endswith(".spz") else b"glTF" if url.endswith(".glb") else png()
            return Resp(200, content=data)

        return mock.patch.object(m.requests, "request", side_effect=request), mock.patch.object(m.requests, "get", side_effect=get)

    def call(self, fn, routes, downloads=None, **kw):
        a, b = self.stub(routes, downloads)
        with a, b:
            return json.loads(fn(output_dir=self.out, **kw))

    def posts(self, path):
        return [c for c in self.calls if c["method"] == "POST" and c["url"].endswith(path)]


PREP = {("POST", "/media-assets:prepare_upload"): lambda b: {
    "media_asset": {"media_asset_id": "ma_" + b["file_name"].split(".")[0]},
    "upload_info": {"upload_url": "https://up/" + b["file_name"], "upload_method": "PUT",
                    "required_headers": {"x-ms-blob-type": "BlockBlob"}}}}


def routes_for_create(extra=None):
    r = {("GET", "/credits"): {"remaining_credits": 20000},
         ("POST", "/worlds:generate"): {"operation_id": "op_1", "done": False, "metadata": {"world_id": "w_1"}},
         ("GET", "/operations/op_1"): [{"operation_id": "op_1", "done": False}, done_op()],
         ("GET", "/worlds/w_1"): world_body(),
         **PREP}
    r.update(extra or {})
    return r


class CreateTests(Base):
    def test_single_image_world(self):
        img = Path(self.out) / "room.png"
        img.write_bytes(png())
        out = self.call(m.marble_create_world, routes_for_create(), name="room", images=[str(img)], prompt="a cosy room")
        self.assertEqual(out["status"], "ok", out)
        body = self.posts("/worlds:generate")[0]["json"]
        self.assertEqual(body["model"], "marble-1.1-plus")
        self.assertEqual(body["world_prompt"]["type"], "image")
        self.assertEqual(body["world_prompt"]["image_prompt"], {"source": "media_asset", "media_asset_id": "ma_room"})
        self.assertEqual(body["world_prompt"]["text_prompt"], "a cosy room")
        self.assertEqual(self.puts[0][2], {"x-ms-blob-type": "BlockBlob"})         # required headers only, no API key
        self.assertTrue(Path(out["splat"]).name == "splat_full_res.spz" and Path(out["splat"]).exists())
        self.assertEqual(out["cost"]["total_credits"], 1700)
        self.assertEqual(out["semantics"], {"metric_scale_factor": 2.0, "ground_plane_offset": 0.5})
        self.assertEqual(out["splat_count"], 30)
        self.assertIn("viewer", out)
        wj = json.loads((Path(out["world_dir"]) / "world.json").read_text())
        self.assertEqual((wj["status"], wj["world_id"], wj["provider"]), ("ready", "w_1", "worldlabs-marble"))
        side = json.loads(Path(out["splat"] + ".agenty.json").read_text())
        self.assertEqual(side["provider"], "worldlabs-marble")
        key_headers = {c["headers"]["WLT-Api-Key"] for c in self.calls if c.get("headers") and not c["url"].startswith("https://up/")}
        self.assertEqual(key_headers, {"wl_marble_test"})                          # the Marble key, not the Atlas one

    def test_url_inputs_and_multi_image(self):
        out = self.call(m.marble_create_world, routes_for_create(), name="two", images=["https://x/a.jpg", "https://x/b.jpg"],
                        azimuths=[0, 180], reconstruct_images=True, model="marble-1.1", seed=7, collider_mesh=True,
                        open_in_viewer=False)
        self.assertEqual(out["status"], "ok", out)
        wp = self.posts("/worlds:generate")[0]["json"]
        self.assertEqual(wp["world_prompt"]["type"], "multi-image")
        self.assertEqual(wp["world_prompt"]["multi_image_prompt"][1], {"azimuth": 180, "content": {"source": "uri", "uri": "https://x/b.jpg"}})
        self.assertTrue(wp["world_prompt"]["reconstruct_images"])
        self.assertEqual(wp["seed"], 7)
        self.assertTrue((Path(out["world_dir"]) / "collider.glb").exists())
        self.assertNotIn("viewer", out)
        self.assertEqual(self.posts("/media-assets:prepare_upload"), [])

    def test_dry_run_spends_and_uploads_nothing(self):
        img = Path(self.out) / "room.png"
        img.write_bytes(png())
        out = self.call(m.marble_create_world, routes_for_create(), name="d", images=[str(img)], dry_run=True)
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["spent"], 0)
        self.assertEqual((out["estimate"]["min"], out["estimate"]["max"]), (1580, 3080))
        self.assertEqual([c for c in self.calls if c["method"] == "POST"], [])
        self.assertFalse((Path(self.out) / "d" / "world.json").exists())

    def test_estimates(self):
        self.assertEqual(m.estimate_credits("marble-1.0-draft", "image")["min"], 230)
        self.assertEqual(m.estimate_credits("marble-1.1", "multi")["max"], 1600)
        self.assertEqual(m.estimate_credits("marble-1.1-plus", "image", pano=True)["max"], 3000)
        self.assertEqual(m.estimate_credits("marble-1.1", "text")["usd_min"], round(1580 / 1250, 2))

    def test_refuses_when_balance_too_low(self):
        out = self.call(m.marble_create_world, routes_for_create({("GET", "/credits"): {"remaining_credits": 100}}),
                        name="poor", images=["https://x/a.jpg"])
        self.assertEqual(out["error_kind"], "balance")
        self.assertEqual(self.posts("/worlds:generate"), [])

    def test_wrong_model_and_args(self):
        for kw in ({"model": "marble-2"}, {"images": None}, {"images": ["https://x/a"] * 5}, {"format": "obj"}):
            args = {"name": "a", "images": ["https://x/a.jpg"], **kw}
            out = self.call(m.marble_create_world, routes_for_create(), **args)
            self.assertEqual(out["error_kind"], "args", (kw, out))
        self.assertEqual(self.posts("/worlds:generate"), [])

    def test_401_names_the_separate_key(self):
        os.environ["WLT_MARBLE_API_KEY"] = ""
        out = self.call(m.marble_create_world, {("GET", "/credits"): Resp(401, {"message": "Unauthorized"}),
                                                ("POST", "/worlds:generate"): Resp(401, {"message": "Unauthorized"})},
                        name="a", images=["https://x/a.jpg"])
        self.assertEqual(out["error_kind"], "auth")
        self.assertIn("WLT_MARBLE_API_KEY", out["error"])

    def test_no_key_at_all(self):
        with mock.patch("agenty_core.utils.secrets.get_secret", return_value=""):
            out = json.loads(m.marble_create_world(name="a", images=["https://x/a.jpg"], output_dir=self.out))
        self.assertEqual(out["error_kind"], "auth")

    def test_content_refusal(self):
        r = routes_for_create({("GET", "/operations/op_1"): {"operation_id": "op_1", "done": True,
                                                              "error": {"code": 3, "message": "Prompt violates content policy"}}})
        out = self.call(m.marble_create_world, r, name="bad", images=["https://x/a.jpg"])
        self.assertEqual(out["error_kind"], "content_refusal")
        wj = json.loads((Path(self.out) / "bad" / "world.json").read_text())
        self.assertEqual(wj["status"], "failed")

    def test_name_taken(self):
        self.call(m.marble_create_world, routes_for_create(), name="same", images=["https://x/a.jpg"], open_in_viewer=False)
        out = self.call(m.marble_create_world, routes_for_create(), name="same", images=["https://x/a.jpg"])
        self.assertEqual(out["error_kind"], "args")

    def test_ply_format_uses_the_free_export(self):
        ply = Path(self.out) / "src.ply"
        props, a = synthetic(n=10, k=0)
        sf.write_ply(ply, props, a)
        r = routes_for_create({("POST", "/worlds/w_1:export"): {"operation_id": "ex", "done": True,
                                                                  "response": {"url": "https://dl/full.ply", "format": "ply"}}})
        out = self.call(m.marble_create_world, r, {"https://dl/full.ply": ply.read_bytes()}, name="p",
                        images=["https://x/a.jpg"], format="ply")
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(out["ply_source"], "api")
        self.assertEqual(self.posts("/worlds/w_1:export")[0]["json"], {"asset_type": "splats", "format": "ply", "resolution": "full_res"})
        self.assertTrue(Path(out["ply"]).exists())
        self.assertTrue(out["viewer"]["path"].endswith(".ply"))


def atlas_world_dir(root: Path, name="room"):
    """An Atlas world folder with posed views looking front, 2 degrees off front, left, back and right."""
    wd = root / name
    wd.mkdir(parents=True)

    def cam(yaw_deg):
        h = math.radians(yaw_deg) / 2           # a turn LEFT about +Y
        return {"extrinsics": {"position": [0, 0, 0], "quaternion": [0, math.sin(h), 0, math.cos(h)], "coordinateSystem": "rub"},
                "intrinsics": {"fx": 900, "fy": 900, "cx": 640, "cy": 360, "width": 1280, "height": 720}}
    frames = []
    for i, yaw in enumerate([0, 2, 90, 180, -90]):           # -90 = turn right
        (wd / f"view_{i:03d}.png").write_bytes(png((i * 40, 50, 50)))
        frames.append({"id": i, "image": f"view_{i:03d}.png", "camera": wl._canon_camera(cam(yaw))})
    (wd / "world.json").write_text(json.dumps({"name": name, "prompt_used": "a tidy room", "frames": frames}))
    return wd


class AtlasWorldTests(Base):
    def test_azimuths_from_camera_yaws(self):
        awd = atlas_world_dir(Path(self.out) / "atlas_here")
        w = json.loads((awd / "world.json").read_text())
        base = w["frames"][0]["camera"]
        az = [m.frame_azimuth(base, f["camera"]) for f in w["frames"]]
        self.assertEqual(az, [0.0, 358.0, 270.0, 180.0, 90.0])         # left turn = 270, right turn = 90

    def test_world_from_atlas_views(self):
        atlas_world_dir(Path(self.out))
        out = self.call(m.marble_create_world, routes_for_create(), name="room_marble", atlas_world="room", open_in_viewer=False)
        self.assertEqual(out["status"], "ok", out)
        body = self.posts("/worlds:generate")[0]["json"]["world_prompt"]
        self.assertEqual(body["type"], "multi-image")
        self.assertEqual(body["text_prompt"], "a tidy room")
        self.assertEqual(sorted(p["azimuth"] for p in body["multi_image_prompt"]), [0.0, 90.0, 180.0, 270.0])   # spread round; the near-duplicate is left out
        self.assertEqual(len(self.posts("/media-assets:prepare_upload")), 4)
        wj = json.loads((Path(out["world_dir"]) / "world.json").read_text())
        self.assertEqual(wj["input"]["atlas_world"], "room")

    def test_unknown_atlas_world(self):
        out = self.call(m.marble_create_world, routes_for_create(), name="x", atlas_world="nope")
        self.assertEqual(out["error_kind"], "args")


class ExportTests(Base):
    def make_world(self, status="ready"):
        wd = Path(self.out) / "w"
        wd.mkdir()
        (wd / "world.json").write_text(json.dumps({"name": "w", "provider": "worldlabs-marble", "world_id": "w_1", "status": status,
                                                   "operation_id": "op_1", "model": "marble-1.1-plus",
                                                   "semantics": {"metric_scale_factor": 2.0, "ground_plane_offset": 0.5},
                                                   "request": {"resolution": "full_res", "format": "spz", "collider_mesh": False},
                                                   "files": {}}))
        return wd

    def downloads(self):
        return len([c for c in self.calls if c["method"] == "GET*"])

    def test_spz_500k_and_collider(self):
        self.make_world()
        r = {("GET", "/worlds/w_1"): world_body()}
        out = self.call(m.marble_export, r, name="w", resolution="500k", collider_mesh=True)
        self.assertEqual(out["status"], "ok", out)
        self.assertTrue(out["file"].endswith("splat_500k.spz"))
        self.assertTrue((Path(self.out) / "w" / "collider.glb").exists())
        self.assertEqual(out["splat_count"], 30)
        n = self.downloads()
        self.call(m.marble_export, r, name="w", resolution="500k")
        self.assertEqual(self.downloads(), n)                    # cached: no re-download
        self.call(m.marble_export, r, name="w", resolution="500k", refresh=True)
        self.assertGreater(self.downloads(), n)

    def test_ply_falls_back_to_local_conversion(self):
        self.make_world()
        r = {("GET", "/worlds/w_1"): world_body(), ("POST", "/worlds/w_1:export"): Resp(422, {"detail": "no such resolution"})}
        out = self.call(m.marble_export, r, name="w", format="ply", resolution="500k")
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(out["ply_source"], "local")
        props, a = sf.read_ply_any(out["ply"])
        self.assertEqual(len(a), 30)
        self.assertEqual(props[:3], ["x", "y", "z"])
        _, raw = sf.read_spz(Path(self.out) / "w" / "splat_500k.spz")
        np.testing.assert_allclose(a, raw, atol=1e-6)                # raw axes, no flip, no scaling

    def test_ply_metric(self):
        self.make_world()
        r = {("GET", "/worlds/w_1"): world_body(), ("POST", "/worlds/w_1:export"): Resp(404, {"detail": "x"})}
        out = self.call(m.marble_export, r, name="w", format="ply", metric=True)
        self.assertTrue(out["ply"].endswith("splat_full_res_metric.ply"))
        props, a = sf.read_ply_any(out["ply"])
        _, raw = sf.read_spz(Path(self.out) / "w" / "splat_full_res.spz")
        np.testing.assert_allclose(a[:, 0], raw[:, 0] * 2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(a[:, 1], raw[:, 1] * 2 - 0.5, rtol=1e-5, atol=1e-5)

    def test_billing_error_on_export_is_not_swallowed(self):
        self.make_world()
        r = {("GET", "/worlds/w_1"): world_body(), ("POST", "/worlds/w_1:export"): Resp(402, {"detail": "no credits"})}
        out = self.call(m.marble_export, r, name="w", format="ply")
        self.assertEqual(out["error_kind"], "balance")

    def test_collects_a_pending_world(self):
        self.make_world("pending")
        r = {("GET", "/operations/op_1"): [{"operation_id": "op_1", "done": False}, done_op()], ("GET", "/worlds/w_1"): world_body()}
        out = self.call(m.marble_export, r, name="w")
        self.assertEqual(out["status"], "ok", out)
        wj = json.loads((Path(self.out) / "w" / "world.json").read_text())
        self.assertEqual(wj["status"], "ready")
        self.assertEqual(out["cost"]["total_credits"], 1700)

    def test_still_running_times_out_with_name(self):
        self.make_world("pending")
        ticks = iter([0, 0, 5, 9999, 99999, 99999])
        with mock.patch.object(m.time, "monotonic", side_effect=lambda: next(ticks)):
            out = self.call(m.marble_export, {("GET", "/operations/op_1"): {"operation_id": "op_1", "done": False}}, name="w",
                            timeout_s=10)
        self.assertEqual(out["error_kind"], "timeout")
        self.assertEqual(out["name"], "w")

    def test_unknown_world_lists_the_known(self):
        self.make_world()
        out = self.call(m.marble_export, {}, name="zzz")
        self.assertEqual(out["error_kind"], "args")
        self.assertIn("have: w", out["error"])


class SplatToMeshTests(Base):
    def atlas_routes(self):
        return {("POST", "ATLAS/assets"): {"id": "asset_s"},
                ("POST", "ATLAS/assets/asset_s:createUploadUrl"): {"uploadUrl": "https://up/s", "method": "PUT", "headers": {}},
                ("POST", "ATLAS/assets/asset_s:completeUpload"): {"id": "asset_s"},
                ("POST", "ATLAS/tasks:splats2Mesh"): {"id": "op_m", "done": True, "metadata": {"task": "splats2Mesh"},
                                                      "response": {"mesh": {"assetId": "am", "url": "https://dl/m.glb"}}}}

    def run_mesh(self, extra=None, **kw):
        a, b = self.stub({**self.atlas_routes(), **(extra or {})}, {"https://dl/m.glb": b"glTF"})
        with a, b:
            return json.loads(wl.atlas_splat_to_mesh(output_dir=self.out, **kw))

    def marble_world(self):
        wd = Path(self.out) / "scene"
        wd.mkdir()
        (wd / "world.json").write_text(json.dumps({"name": "scene", "provider": "worldlabs-marble", "world_id": "w_1",
                                                   "status": "ready", "files": {}}))
        return wd

    def test_marble_world_by_name(self):
        wd = self.marble_world()
        (wd / "splat_full_res.spz").write_bytes(self.spz)
        out = self.run_mesh(splat_path="scene")
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(Path(out["glb"]), wd / "mesh.glb")
        self.assertEqual(self.puts[0][3], self.spz)                                   # the world's own SPZ went up
        self.assertEqual(json.loads((wd / "world.json").read_text())["files"]["mesh"], "mesh.glb")

    def test_marble_world_without_spz_fetches_it(self):
        wd = self.marble_world()
        out = self.run_mesh({("GET", "/worlds/w_1"): world_body()}, splat_path="scene")
        self.assertEqual(out["status"], "ok", out)
        self.assertTrue((wd / "splat_full_res.spz").exists())

    def test_ply_is_converted_to_spz_first(self):
        ply = Path(self.out) / "trained.ply"
        props, a = synthetic(n=12, k=3)
        sf.write_ply(ply, props, a)
        out = self.run_mesh(splat_path=str(ply))
        self.assertEqual(out["status"], "ok", out)
        sent = self.puts[0][3]
        self.assertEqual(sent[:2], b"\x1f\x8b")                                          # gzip: an SPZ
        f = Path(self.out) / "x.spz"
        f.write_bytes(sent)
        self.assertEqual(sf.spz_header(f)["count"], 12)

    def test_spz_path_and_bad_input(self):
        f = Path(self.out) / "own.spz"
        f.write_bytes(self.spz)
        self.assertEqual(self.run_mesh(splat_path=str(f))["status"], "ok")
        self.assertEqual(self.puts[0][3], self.spz)
        t = Path(self.out) / "x.obj"
        t.write_bytes(b"o")
        self.assertEqual(self.run_mesh(splat_path=str(t))["error_kind"], "args")
        self.assertEqual(self.run_mesh(splat_path="nothing_here")["error_kind"], "args")

    def test_atlas_fetch_finds_a_marble_mesh_job(self):
        wd = Path(self.out) / "scene"
        (wd / "pending").mkdir(parents=True)
        (wd / "pending" / "op_zz.json").write_text("{}")
        found, _ = wl._find_pending("op_zz", self.out)
        self.assertEqual(found, wd)


if __name__ == "__main__":
    unittest.main()
