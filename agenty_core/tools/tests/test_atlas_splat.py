"""atlas_splat offline: the view-consistency check, the posedImages2Splats request (frames, cameras, depth), the
cost gate, polling, the SPZ download and its viewer-axes PLY, and collecting a timed-out job with atlas_fetch."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from agenty_core.tools import atlas_splat as asp
from agenty_core.tools import splat_formats as sf
from agenty_core.tools import worldlabs_atlas as w

K = {"fx": 640.0, "fy": 640.0, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720}
DH, DW = 72, 128                     # depth grid (scaled to the camera grid by the code)


def cam(pos, quat=(0, 0, 0, 1)):
    return {"extrinsics": {"position": list(pos), "quaternion": list(quat), "coordinateSystem": "rub"},
            "intrinsics": dict(K)}


def wall(z=4.0):
    return np.full((DH, DW), z, dtype="float32")


def png_bytes(size=(1280, 720)):
    b = io.BytesIO()
    Image.new("RGB", size, (120, 90, 60)).save(b, "PNG")
    return b.getvalue()


def spz_bytes(tmp: Path) -> bytes:
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    a = np.zeros((4, len(props)), "float32")
    a[:, 0:3] = [[0.5, 1.0, -2.0], [0.0, 2.0, -3.0], [1.0, 0.5, -1.0], [-1.0, 1.5, -4.0]]
    a[:, 7:10] = -3.0
    a[:, 10] = 1.0
    p = tmp / "fixture.spz"
    sf.write_spz(p, props, a)
    return p.read_bytes()


class Resp:
    def __init__(self, status=200, body=None, content=b""):
        self.status_code, self._b, self.content, self.headers = status, body if body is not None else {}, content, {}
        self.text = json.dumps(self._b)

    def json(self):
        return self._b


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        p = mock.patch.dict(os.environ, {"WLT_API_KEY": "wl_test", "WORLDLABS_API_KEY": "", "WLT_API_BASE_URL": ""})
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch.object(w, "_sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)
        self.depths = {}
        p = mock.patch.object(w, "_read_exr", lambda path: self.depths[Path(path).name])
        p.start()
        self.addCleanup(p.stop)
        self.calls, self.puts = [], []
        self.wd = self.make_world()

    def make_world(self, sizes=None):
        """Three views of a wall 4 m away. 1 = moved 0.3 m right (agrees); 2 = same place as 0, but a box
        1 m from the camera fills its middle that view 0 does not show (disagrees). 2 has no stored assets."""
        wd = self.root / "room"
        wd.mkdir(parents=True, exist_ok=True)
        frames = []
        for i, pos in enumerate([(0, 1, 0), (0.3, 1, 0), (0, 1, 0)]):
            d = wall()
            if i == 2:
                d[20:52, 40:88] = 1.0
            self.depths[f"depth_{i:03d}.exr"] = d
            (wd / f"depth_{i:03d}.exr").write_bytes(b"EXR")
            size = (sizes or {}).get(i, (1280, 720))
            (wd / f"view_{i:03d}.png").write_bytes(png_bytes(size))
            fr = {"id": i, "image": f"view_{i:03d}.png", "depth": f"depth_{i:03d}.exr", "camera": cam(pos),
                  "source": "generated"}
            if i < 2:
                fr.update(image_asset=f"asset_img{i}", depth_asset=f"asset_d{i}")
            frames.append(fr)
        (wd / "world.json").write_text(json.dumps({"version": 1, "name": "room", "frames": frames, "operations": []}))
        return wd

    def api(self, routes, downloads=None):
        def request(method, url, headers=None, json=None, params=None, data=None, timeout=None, **kw):
            if url.startswith("https://upload/"):
                self.puts.append((method, url, len(data or b"")))
                return Resp(200)
            self.calls.append((method, url, json, headers))
            path = url.split("/api/v2", 1)[-1]
            for (m, suffix), fn in routes.items():
                if m == method and path.endswith(suffix):
                    out = fn(json) if callable(fn) else fn
                    return out if isinstance(out, Resp) else Resp(200, out)
            return Resp(404, {"detail": f"no route {method} {path}"})

        def get(url, **kw):
            return Resp(200, content=(downloads or {}).get(url, b""))

        return mock.patch.object(w.requests, "request", side_effect=request), mock.patch.object(w.requests, "get", side_effect=get)

    def uploads(self):
        n = iter(range(100))
        return {("POST", "/assets"): lambda b: {"id": f"asset_up{next(n)}"},
                ("POST", ":createUploadUrl"): {"uploadUrl": "https://upload/x", "method": "PUT", "headers": {}},
                ("POST", ":completeUpload"): {}}

    def call(self, routes, downloads=None, **kw):
        a, b = self.api(routes, downloads)
        kw.setdefault("open_in_viewer", False)
        with a, b:
            return json.loads(asp.atlas_views_to_splat(world="room", output_dir=str(self.root), **kw))

    def submits(self):
        return [c for c in self.calls if c[1].endswith("tasks:posedImages2Splats")]


class ConsistencyTests(Base):
    def test_the_view_with_an_object_the_others_lack_is_left_out(self):
        frames = w._load_world(self.wd)["frames"]
        r = asp.view_consistency(self.wd, frames)
        self.assertEqual(r["kept"], [0, 1])
        self.assertEqual([e["id"] for e in r["excluded"]], [2])
        self.assertIn(r["excluded"][0]["with"], (0, 1))
        self.assertGreater(r["excluded"][0]["conflict"], 0.05)
        self.assertLess(r["views"][1]["worst"], 0.01)

    def test_a_surface_behind_is_occlusion_not_conflict(self):
        """Projecting a far surface into a view that sees a nearer one is occlusion, not a conflict; only a
        surface IN FRONT of what the other view sees counts."""
        f0 = w._load_world(self.wd)["frames"][0]
        self.depths["depth_far.exr"] = wall(6.0)
        s0 = asp._view_samples(self.wd, f0, 8, 0.2)
        far = asp._view_samples(self.wd, dict(f0, depth="depth_far.exr"), 8, 0.2)
        shared, conflict = asp._project(far, s0, 0.15, 0.2)      # far wall behind s0's wall: occluded, fine
        self.assertGreater(shared, 0)
        self.assertEqual(conflict, 0)
        shared, conflict = asp._project(s0, far, 0.15, 0.2)      # s0's wall in front of what far sees: conflict
        self.assertEqual(conflict, shared)

    def test_ids_parse(self):
        self.assertEqual(asp._ids("0-3, 7,9-10"), [0, 1, 2, 3, 7, 9, 10])
        self.assertEqual(asp._ids([3, 1, 3]), [1, 3])
        self.assertEqual(asp._ids(""), [])


class RequestTests(Base):
    def test_dry_run_spends_nothing_and_prices_it(self):
        out = self.call({}, dry_run=True)
        self.assertEqual(out["status"], "dry_run")
        self.assertEqual(out["plan"]["frames"], [0, 1])
        self.assertEqual(out["plan"]["estimate"]["credits"], 5000)
        self.assertEqual([e["id"] for e in out["plan"]["consistency"]["excluded"]], [2])
        self.assertEqual(self.calls, [])
        draft = self.call({}, dry_run=True, refine=False)
        self.assertEqual(draft["plan"]["estimate"]["credits"], 500)

    def test_without_confirmation_it_asks(self):
        out = self.call({})
        self.assertEqual(out["error_kind"], "confirm_cost")
        self.assertIn("5,000", out["error"])
        self.assertEqual(self.calls, [])

    def test_request_shape_polling_and_download(self):
        spz = spz_bytes(self.root)
        waits = iter([{"id": "op_s", "done": False},
                      {"id": "op_s", "done": True, "error": None,
                       "response": {"refinedSplats": {"500000": {"assetId": "asset_a", "url": "https://dl/a.spz"},
                                                      "2000000": {"assetId": "asset_b", "url": "https://dl/b.spz"}}}}])
        routes = {("POST", "/tasks:posedImages2Splats"): {"id": "op_s", "done": False},
                  ("POST", "/operations/op_s:wait"): lambda b: next(waits), **self.uploads()}
        out = self.call(routes, {"https://dl/b.spz": spz}, frames="0-2", check_consistency=False, confirm_cost=True,
                       prompt="a wooden cabin")
        self.assertEqual(out["status"], "ok", out)
        body = self.submits()[0][2]
        self.assertTrue(self.submits()[0][3]["Idempotency-Key"])
        self.assertEqual(len(body["posedFrames"]), 3)
        self.assertTrue(body["refine"])
        self.assertEqual(body["prompt"], "a wooden cabin")
        f0, f2 = body["posedFrames"][0], body["posedFrames"][2]
        self.assertEqual(f0["imageAsset"], {"assetId": "asset_img0"})
        self.assertEqual(f0["depth"], {"depthAsset": {"assetId": "asset_d0"}})
        # the world camera goes in unchanged: camera-to-world, RUB, the world's own frame
        self.assertEqual(f0["camera"]["extrinsics"]["coordinateSystem"], "rub")
        self.assertEqual(f0["camera"]["extrinsics"]["position"], [0.0, 1.0, 0.0])
        self.assertEqual(body["posedFrames"][1]["camera"]["extrinsics"]["position"], [0.3, 1.0, 0.0])
        self.assertEqual(f0["camera"]["intrinsics"]["width"], 1280)
        # frame 2 had no stored assets: its image and depth EXR were uploaded first
        self.assertTrue(f2["imageAsset"]["assetId"].startswith("asset_up"))
        self.assertTrue(f2["depth"]["depthAsset"]["assetId"].startswith("asset_up"))
        grants = [c[2] for c in self.calls if c[1].endswith(":createUploadUrl")]
        self.assertIn("image/x-exr", [g["contentType"] for g in grants])
        # the largest refined level is the one kept; a viewer-axes PLY sits next to it
        self.assertEqual(out["splat_level"], "2000000")
        self.assertTrue(Path(out["spz"]).exists())
        props, a = sf.read_ply_any(out["ply"])
        _, raw = sf.read_spz(out["spz"])
        np.testing.assert_allclose(a[:, 0], raw[:, 0], atol=1e-3)
        np.testing.assert_allclose(a[:, 1], -raw[:, 1], atol=1e-3)      # turned 180 degrees about X
        np.testing.assert_allclose(a[:, 2], -raw[:, 2], atol=1e-3)
        self.assertTrue(Path(out["ply_lite"]).exists())
        world = w._load_world(self.wd)
        self.assertEqual(world["splats"][0]["operation"], "op_s")
        self.assertEqual(world["splats"][0]["frames"], [0, 1, 2])
        self.assertFalse(list(self.wd.glob("pending/*.json")))

    def test_consistency_check_runs_before_upload(self):
        routes = {("POST", "/tasks:posedImages2Splats"): {"id": "op_c", "done": True, "error": None,
                                                         "response": {"draftSplat": {"assetId": "asset_d", "url": "https://dl/d.spz"}}}}
        out = self.call(routes, {"https://dl/d.spz": spz_bytes(self.root)}, confirm_cost=True, refine=False)
        self.assertEqual(out["status"], "ok", out)
        body = self.submits()[0][2]
        self.assertEqual(len(body["posedFrames"]), 2)                     # frame 2 never left the machine
        self.assertFalse(body["refine"])
        self.assertFalse([c for c in self.calls if c[1].endswith("/assets")])
        self.assertEqual(out["splat_level"], "draft")
        self.assertEqual([e["id"] for e in out["excluded"]], [2])

    def test_timeout_then_atlas_fetch_finishes_it(self):
        routes = {("POST", "/tasks:posedImages2Splats"): {"id": "op_t", "done": False},
                  ("POST", "/operations/op_t:wait"): {"id": "op_t", "done": False}}
        clock = iter([0, 0, 0, 0, 1000, 1000, 1000, 1000])
        with mock.patch.object(w.time, "monotonic", lambda: next(clock, 2000)):
            out = self.call(routes, confirm_cost=True, timeout_s=5)
        self.assertEqual(out["error_kind"], "timeout")
        self.assertEqual(out["operation_id"], "op_t")
        self.assertTrue((self.wd / "pending" / "op_t.json").exists())
        a, b = self.api({("GET", "/operations/op_t"): {"id": "op_t", "done": True, "error": None, "response": {
            "refinedSplats": {"2000000": {"assetId": "asset_r", "url": "https://dl/r.spz"}}}}},
            {"https://dl/r.spz": spz_bytes(self.root)})
        with a, b, mock.patch.object(asp, "finish_splat", wraps=asp.finish_splat):
            job = json.loads((self.wd / "pending" / "op_t.json").read_text())
            job["open_in_viewer"] = False
            (self.wd / "pending" / "op_t.json").write_text(json.dumps(job))
            fin = json.loads(w.atlas_fetch(operation_id="op_t", output_dir=str(self.root)))
        self.assertEqual(fin["status"], "ok", fin)
        self.assertTrue(fin["spz"].endswith("room_posed_splat.spz"))

    def test_mixed_image_sizes_are_refused(self):
        self.depths.clear()
        self.wd = self.make_world(sizes={1: (1024, 578)})
        out = self.call({}, dry_run=True)
        self.assertEqual(out["error_kind"], "args")
        self.assertIn("one image size", out["error"])

    def test_missing_frames_and_failures(self):
        self.assertEqual(self.call({}, frames="0,7", dry_run=True)["error_kind"], "args")
        routes = {("POST", "/tasks:posedImages2Splats"): {"id": "op_f", "done": True, "response": None,
                                                         "error": {"code": 13, "message": "boom"}}}
        out = self.call(routes, confirm_cost=True)
        self.assertEqual(out["error_kind"], "execution")

    def test_masks_are_uploaded_per_frame(self):
        md = self.wd / "masks"
        md.mkdir()
        Image.new("L", (1280, 720), 255).save(md / "mask_001.png")
        routes = {("POST", "/tasks:posedImages2Splats"): {"id": "op_m", "done": True, "error": None,
                                                         "response": {"draftSplat": {"assetId": "x", "url": "https://dl/m.spz"}}},
                  **self.uploads()}
        out = self.call(routes, {"https://dl/m.spz": spz_bytes(self.root)}, confirm_cost=True, masks="masks")
        self.assertEqual(out["status"], "ok", out)
        frames = self.submits()[0][2]["posedFrames"]
        self.assertNotIn("maskAsset", frames[0])
        self.assertTrue(frames[1]["maskAsset"]["assetId"].startswith("asset_up"))


if __name__ == "__main__":
    unittest.main()
