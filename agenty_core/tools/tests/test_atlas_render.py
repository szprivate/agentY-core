"""atlas_render offline: the camera-frame conversion with known poses, the request Atlas gets, the files,
the upscale workflow (ComfyUI client mocked) and the resume path."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from agenty_core.tools import atlas_render as r
from agenty_core.tools import previz_shot as ps
from agenty_core.tools import worldlabs_atlas as w
from agenty_core.tools.tests.test_worldlabs_atlas import Base, CAM, Resp, done_op, frame, png_bytes, read_glb


def col_major(R, t, s=1.0):
    """three.js matrix.elements of T * R * S (uniform scale s)."""
    return [R[0][0] * s, R[1][0] * s, R[2][0] * s, 0, R[0][1] * s, R[1][1] * s, R[2][1] * s, 0,
            R[0][2] * s, R[1][2] * s, R[2][2] * s, 0, t[0], t[1], t[2], 1]


def close(a, b, tol=1e-6):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol)


def rot_y(deg):
    return ps.euler_to_matrix([0, deg, 0])


EYE = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


class ConversionTests(unittest.TestCase):
    def test_identity_model_keeps_the_pose(self):
        R, t = ps.euler_to_matrix([10, 35, -5]), [1.0, 2.0, -3.0]
        R2, t2 = r.viewer_pose_to_mesh(col_major(R, t), col_major(EYE, [0, 0, 0]))
        self.assertTrue(close(R2, R) and close(t2, t))
        R3, t3 = r.viewer_pose_to_mesh(col_major(R, t))
        self.assertTrue(close(R3, R) and close(t3, t))

    def test_a_default_viewer_camera_looks_down_minus_z(self):
        R, _ = r.viewer_pose_to_mesh(col_major(EYE, [0, 0, 5]))
        self.assertTrue(close([-v for v in r._col(R, 2)], [0, 0, -1]))

    def test_model_transform_is_undone(self):
        for pos, rot, s in (([0, 0, 0], [0, 90, 0], 1.0), ([3, -1, 2], [15, -40, 5], 2.5), ([0, 5, 0], [0, 0, 0], 0.1)):
            Rm = ps.euler_to_matrix(rot)
            for Rc, tc in ((ps.euler_to_matrix([-8, 30, 0]), [0.5, 1.7, -2.0]), (EYE, [0, 0, 0])):
                # the camera in the model's own frame (Rc, tc); in the scene it is model matrix * that
                Mc = np.array(col_major(Rc, tc)).reshape(4, 4).T
                Mm = np.array(col_major(Rm, pos, s)).reshape(4, 4).T
                world = Mm @ Mc
                R2, t2 = r.viewer_pose_to_mesh(list(world.T.reshape(-1)), col_major(Rm, pos, s))
                self.assertTrue(close(R2, Rc, 1e-6), (pos, rot, s))
                self.assertTrue(close(t2, tc, 1e-6), (pos, rot, s))

    def test_model_turned_90_degrees_by_hand(self):
        # model turned 90 deg about Y: model +X maps to world -Z, model +Z to world +X.
        # A camera at world (0, 0, 5) therefore stands at model (5, 0, 0)? No: model = Rm^T * world.
        Rm = rot_y(90)
        _R2, t2 = r.viewer_pose_to_mesh(col_major(EYE, [0, 0, 5]), col_major(Rm, [0, 0, 0]))
        self.assertTrue(close(t2, [-5.0, 0, 0]) or close(t2, [5.0, 0, 0]))
        self.assertTrue(close(np.array(Rm).T @ np.array([0, 0, 5.0]), t2))
        self.assertTrue(close(Rm[0], [0, 0, 1]))            # three.js Ry(90): x' = z  (so model +Z -> world +X)

    def test_orthonormal_drops_scale(self):
        R = ps.euler_to_matrix([20, 30, 40])
        self.assertTrue(close(r.orthonormal((np.array(R) * 3.0).tolist()), R))

    def test_fov_contain_and_gate(self):
        self.assertAlmostEqual(r.viewer_fov(50, 16 / 9, False), 50.0, places=4)
        self.assertAlmostEqual(r.viewer_fov(50, 16 / 9, True), 50.0, places=4)
        self.assertAlmostEqual(r.viewer_fov(50, 1.0, False), 50.0, places=4)      # taller viewport: vertical one stays
        wide = r.viewer_fov(50, 3.0, False)                                       # wider viewport: widen to hold its sides
        self.assertGreater(wide, 50.0)
        self.assertAlmostEqual(math.tan(math.radians(wide) / 2) * 16 / 9, math.tan(math.radians(25)) * 3.0, places=6)
        self.assertAlmostEqual(r.viewer_fov(35, 0.5, True), 35.0, places=4)       # a gate is the picture itself
        self.assertEqual(r.viewer_fov(500, 1.0), r.MAX_FOV)


def world_with(cams):
    return {"name": "w", "frames": [{"id": i, "image": f"view_{i:03d}.png", "depth": f"depth_{i:03d}.exr",
                                     "camera": c} for i, c in enumerate(cams)]}


def api_cam(R, t, fx=900.0):
    return w.camera_to_api(R, t, {"fx": fx, "fy": fx, "cx": 640, "cy": 360, "width": 1280, "height": 720})


class FrameTests(unittest.TestCase):
    def setUp(self):
        # first view: not at the origin, turned 90 deg about Y; second view: moved and turned more
        self.c0 = api_cam(rot_y(90), [1.0, 1.5, -2.0])
        self.c1 = api_cam(ps.euler_to_matrix([-5, 120, 0]), [2.0, 1.4, -4.0])
        self.world = world_with([self.c0, self.c1])

    def test_mesh_frame_roundtrip(self):
        base = w.normalize_camera(self.c0)
        R, t = w.camera_to_mesh_pose(base, self.c1)
        back = w.mesh_pose_to_camera(base, base["K"], R, t)
        n1, nb = w.normalize_camera(self.c1), w.normalize_camera(back)
        self.assertTrue(close(n1["R"], nb["R"], 1e-5) and close(n1["t"], nb["t"], 1e-5))
        R0, t0 = w.camera_to_mesh_pose(base, self.c0)       # the first camera IS the origin, looking down -Z
        self.assertTrue(close(R0, EYE) and close(t0, [0, 0, 0]))

    def test_viewer_pose_on_a_views_camera_reproduces_its_pixels(self):
        """A mesh vertex built the way build_world_mesh builds it must land on its own pixel through the
        camera a viewer pose converts to (viewer node scaled, turned and moved on top of that)."""
        base = w.normalize_camera(self.c0)
        cam1 = w.normalize_camera(self.c1)
        depth = np.full((720, 1280), 4.0, dtype="float32")
        pos, uv, _tri = w.frame_mesh((1280, 720), cam1["K"], depth, None, cam1, base, stride=40)
        R, t = w.camera_to_mesh_pose(base, self.c1)
        Rm, tm, s = ps.euler_to_matrix([0, 25, 0]), [3.0, -1.0, 7.0], 2.0
        Mm = np.array(col_major(Rm, tm, s)).reshape(4, 4).T
        Mc = np.array(col_major(R, t)).reshape(4, 4).T
        spec = {"viewer": {"camera_world": list((Mm @ Mc).T.reshape(-1)), "model_world": col_major(Rm, tm, s),
                           "fov": math.degrees(2 * math.atan(360 / 900.0)), "aspect": 16 / 9, "gated": True}}
        api, _pose, _fov = r.resolve_camera(self.world, spec)
        got = w.normalize_camera(api)
        self.assertTrue(close(got["R"], cam1["R"], 1e-5) and close(got["t"], cam1["t"], 1e-5))
        self.assertEqual((api["intrinsics"]["width"], api["intrinsics"]["height"]), (1280, 720))
        self.assertAlmostEqual(api["intrinsics"]["fy"], 900.0, places=3)
        world_pts = pos.astype("float64") @ np.array(base["R"]).T + np.array(base["t"])
        pc = (world_pts - np.array(got["t"])) @ np.array(got["R"])         # rows of R^T (p - t)
        u = got["K"]["fx"] * pc[:, 0] / -pc[:, 2] + got["K"]["cx"]
        v = -got["K"]["fy"] * pc[:, 1] / -pc[:, 2] + got["K"]["cy"]
        self.assertTrue(close(u, uv[:, 0] * 1280, 1e-2), np.abs(u - uv[:, 0] * 1280).max())
        self.assertTrue(close(v, uv[:, 1] * 720, 1e-2), np.abs(v - uv[:, 1] * 720).max())

    def test_relative_move_matches_the_expand_tools_cameras(self):
        mv = {"forward": 2.0, "right": 0.5, "up": 0.3, "yaw": 25, "pitch": -4}
        api, _p, _f = r.resolve_camera(self.world, mv)
        ref = w._target_cameras(self.world, [{**mv, "t": 0.0}], 0.0)[0]
        a, b = w.normalize_camera(api), w.normalize_camera(ref)
        self.assertTrue(close(a["R"], b["R"]) and close(a["t"], b["t"]))

    def test_forward_is_along_the_first_cameras_view_in_the_world_frame(self):
        # first camera turned 90 deg about Y looks along -X in the world; "forward 3" lands 3 m that way
        api, _p, _f = r.resolve_camera(self.world, {"forward": 3.0})
        got = w.normalize_camera(api)["t"]
        self.assertTrue(close(got, [1.0 - 3.0, 1.5, -2.0], 1e-5), got)

    def test_look_at_and_matrix_agree(self):
        a, (R, _t), _ = r.resolve_camera(self.world, {"position": [1, 1.6, -2], "look_at": [1, 1.6, -9]})
        self.assertTrue(close([-v for v in r._col(R, 2)], [0, 0, -1]))
        m = [[R[0][0], R[0][1], R[0][2], 1], [R[1][0], R[1][1], R[1][2], 1.6],
             [R[2][0], R[2][1], R[2][2], -2], [0, 0, 0, 1]]
        b, _p, _f = r.resolve_camera(self.world, {"matrix": m})
        na, nb = w.normalize_camera(a), w.normalize_camera(b)
        self.assertTrue(close(na["R"], nb["R"]) and close(na["t"], nb["t"]))

    def test_fov_overrides_and_is_clamped(self):
        api, _p, _f = r.resolve_camera(self.world, {"forward": 1, "fov": 40})
        self.assertAlmostEqual(math.degrees(2 * math.atan(360 / api["intrinsics"]["fy"])), 40.0, places=3)
        _a, _p, fov = r.resolve_camera(self.world, {"forward": 1}, fov=300)
        self.assertEqual(fov, r.MAX_FOV)

    def test_bad_specs(self):
        for spec in ({}, {"position": [0, 0, 0]}, {"position": [1, 1, 1], "look_at": [1, 1, 1]}, {"matrix": [1, 2]},
                     {"viewer": {"camera_world": [1, 2, 3]}}):
            with self.assertRaises(w.AtlasError):
                r.resolve_camera(self.world, spec)

    def test_context_is_the_nearest_views(self):
        cams = [api_cam(EYE, [0, 0, -z]) for z in (0, 10, 20, 30, 40)]
        target = api_cam(EYE, [0, 0, -19])
        self.assertEqual([f["id"] for f in r._pick_context(world_with(cams), target, 3)], [2, 1, 3])


class RenderTests(Base):
    def make_world(self):
        out = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)], "promptUsed": "a courtyard"}),
        }, name="court", prompt="a courtyard")
        self.assertEqual(out["status"], "ok", out)
        self.calls.clear()
        self.published = []
        p = mock.patch.object(w, "_publish", lambda path: self.published.append(str(path)))
        p.start()
        self.addCleanup(p.stop)

    def gen(self, pos=(0, 0, -2)):
        cam = {**CAM, "extrinsics": {**CAM["extrinsics"], "position": list(pos)}}
        return done_op("op_g", {"frames": [{**frame(1), "camera": cam}], "promptUsed": "a courtyard"})

    def test_render_sends_a_1280x720_camera_and_saves_the_picture(self):
        self.make_world()
        out = self.run_tool(r.atlas_render_view, {("POST", "/tasks:atlasGenerate"): self.gen()},
                            world="court", camera={"forward": 2.0, "yaw": 20})
        self.assertEqual(out["status"], "ok", out)
        body = next(c for c in self.calls if c[1].endswith("tasks:atlasGenerate"))[2]
        self.assertEqual(len(body["targetCameras"]), 1)
        ic = body["targetCameras"][0]["intrinsics"]
        self.assertEqual((ic["width"], ic["height"]), (1280, 720))
        self.assertAlmostEqual(body["targetCameras"][0]["extrinsics"]["position"][2], -2.0, places=4)
        self.assertEqual(body["contextFrames"][0]["imageAsset"], {"assetId": "asset_img0"})
        self.assertFalse(body["returnDepth"])
        self.assertEqual(body["prompt"], "a courtyard")
        self.assertFalse(body["enhancePrompt"])
        png = Path(out["path"])
        self.assertTrue(png.exists() and png.parent.name == "renders" and png.name == "render_000.png")
        self.assertTrue(Path(str(png) + ".agenty.json").exists())
        self.assertEqual(self.published, [str(png)])
        self.assertEqual(out["size"], [1280, 720])
        self.assertIn("camera_mesh_frame", out)
        wj = json.loads((Path(self.tmp.name) / "court" / "world.json").read_text())
        self.assertEqual(len(wj["frames"]), 1)                                    # not added
        self.assertEqual(len(wj["renders"]), 1)
        out2 = self.run_tool(r.atlas_render_view, {("POST", "/tasks:atlasGenerate"): self.gen()}, world="court",
                             camera={"yaw": 10})
        self.assertTrue(out2["path"].endswith("render_001.png"))

    def test_world_by_path_and_add_to_world(self):
        self.make_world()
        glb_path = str(Path(self.tmp.name) / "court" / "world.glb")
        out = self.run_tool(r.atlas_render_view, {("POST", "/tasks:atlasGenerate"): self.gen()},
                            world=glb_path, camera={"forward": 2.0}, add_to_world=True)
        self.assertEqual(out["status"], "ok", out)
        body = next(c for c in self.calls if c[1].endswith("tasks:atlasGenerate"))[2]
        self.assertTrue(body["returnDepth"])
        wj = json.loads((Path(self.tmp.name) / "court" / "world.json").read_text())
        self.assertEqual(len(wj["frames"]), 2)
        self.assertEqual(out["glb"], glb_path)
        self.assertEqual(len(read_glb(glb_path)[0]["meshes"]), 2)

    def test_errors(self):
        self.make_world()
        self.assertEqual(self.run_tool(r.atlas_render_view, {}, world="nope", camera={"forward": 1})["error_kind"], "args")
        self.assertEqual(self.run_tool(r.atlas_render_view, {}, world="court", camera={})["error_kind"], "args")
        self.assertEqual(self.run_tool(r.atlas_render_view, {}, world="court", camera={"forward": 1},
                                       upscale=3)["error_kind"], "args")
        refusal = {"id": "x", "done": True, "error": {"code": 3, "message": "no", "details": [{"reason": "CONTENT_POLICY"}]}}
        out = self.run_tool(r.atlas_render_view, {("POST", "/tasks:atlasGenerate"): refusal}, world="court",
                            camera={"forward": 1})
        self.assertEqual(out["error_kind"], "content_refusal")

    def test_timeout_then_atlas_fetch_finishes_the_render_including_the_upscale(self):
        self.make_world()
        routes = {("POST", "/tasks:atlasGenerate"): {"id": "op_slow", "done": False},
                  ("POST", "/operations/op_slow:wait"): {"id": "op_slow", "done": False}}
        clock = iter([0, 0, 0, 0, 1000, 1000, 1000, 1000])
        with mock.patch.object(w.time, "monotonic", lambda: next(clock, 2000)):
            out = self.run_tool(r.atlas_render_view, routes, world="court", camera={"forward": 1}, timeout_s=5,
                                upscale=2)
        self.assertEqual(out["error_kind"], "timeout")
        self.assertEqual(out["operation_id"], "op_slow")
        gen = self.gen()
        gen["id"] = "op_slow"

        def fake_up(src, dest, scale, model="", timeout_s=0):
            dest.write_bytes(png_bytes((2560, 1440)))
            return {"model": "m", "scale": scale}
        with mock.patch.object(r, "upscale_image", fake_up):
            fin = self.run_tool(w.atlas_fetch, {("GET", "/operations/op_slow"): gen}, operation_id="op_slow")
        self.assertEqual(fin["status"], "ok", fin)
        self.assertTrue(fin["path"].endswith("render_000_x2.png"))
        self.assertEqual(fin["size"], [2560, 1440])
        self.assertTrue(fin["original"].endswith("render_000.png"))
        self.assertEqual(self.published, [fin["path"]])

    def test_upscale_failure_keeps_the_render(self):
        self.make_world()
        with mock.patch.object(r, "upscale_image", side_effect=w.AtlasError("ComfyUI is down", "dependency")):
            out = self.run_tool(r.atlas_render_view, {("POST", "/tasks:atlasGenerate"): self.gen()}, world="court",
                                camera={"forward": 1}, upscale=4)
        self.assertEqual(out["status"], "ok")
        self.assertTrue(out["path"].endswith("render_000.png"))
        self.assertIn("ComfyUI is down", out["upscale_error"])

    def test_open_in_viewer_posts_the_mesh_and_a_16x9_camera(self):
        self.make_world()
        posted = {}

        class Client:
            def post(self, path, json_data=None, **kw):
                posted.update(path=path, body=json_data)
                return {"ok": True, "key": "previz:atlas_court", "scene_file": "x.json"}
        with mock.patch("agenty_core.utils.comfyui_client.get_client", lambda: Client()):
            out = json.loads(r.atlas_open_in_viewer("court", output_dir=self.tmp.name))
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(posted["path"], "/bepic/previz_shot")
        items = posted["body"]["scene"]["items"]
        self.assertEqual([i["kind"] for i in items], ["model", "camera"])
        self.assertTrue(items[0]["src"]["path"].endswith("world.glb"))
        self.assertEqual(items[1]["resolution"], [1280, 720])
        self.assertEqual(posted["body"]["scene"]["activeCamera"], items[1]["id"])
        self.assertEqual(posted["body"]["name"], "atlas_court")


class UpscaleTests(unittest.TestCase):
    def test_native_scale_and_choice(self):
        self.assertEqual(r.native_scale("4x-AnimeSharp.pth"), 4)
        self.assertEqual(r.native_scale("RealESRGAN_x2plus.pth"), 2)
        self.assertEqual(r.native_scale("4x_NMKD-Superscale-SP_178000_G.pth"), 4)
        names = ["4x-AnimeSharp.pth", "4x_NMKD-Superscale-SP_178000_G.pth"]
        self.assertEqual(r.pick_upscaler(names, 4), "4x_NMKD-Superscale-SP_178000_G.pth")   # photo style first
        self.assertEqual(r.pick_upscaler(names, 2), "4x_NMKD-Superscale-SP_178000_G.pth")
        self.assertEqual(r.pick_upscaler(names, 4, "anime"), "4x-AnimeSharp.pth")
        with self.assertRaises(w.AtlasError):
            r.pick_upscaler(names, 4, "nothing")
        with self.assertRaises(w.AtlasError):
            r.pick_upscaler([], 4)
        self.assertEqual(r.pick_upscaler(["4x_b.pth", "2x_a.pth"], 2), "2x_a.pth")          # native factor preferred

    def test_workflow_shape(self):
        wf = r.build_upscale_workflow("in.png", "4x.pth", 2, 4)
        self.assertEqual(wf["4"]["class_type"], "ImageScaleBy")
        self.assertAlmostEqual(wf["4"]["inputs"]["scale_by"], 0.5)
        self.assertEqual(wf["5"]["inputs"]["images"], ["4", 0])
        wf = r.build_upscale_workflow("in.png", "4x.pth", 4, 4)
        self.assertNotIn("4", wf)
        self.assertEqual(wf["5"]["inputs"]["images"], ["3", 0])

    def test_upscale_through_the_core_client(self):
        tmp = Path(tempfile.mkdtemp())
        src, dest = tmp / "a.png", tmp / "a_x2.png"
        src.write_bytes(png_bytes((64, 36)))
        seen = {}

        class Client:
            def get(self, path, params=None, raw=False, **kw):
                if path == "/models/upscale_models":
                    return ["4x-AnimeSharp.pth", "4x_NMKD-Superscale-SP_178000_G.pth"]
                if path.startswith("/history/"):
                    return {"pid1": {"status": {"status_str": "success"}, "outputs": {"5": {"images": [
                        {"filename": "o.png", "subfolder": "", "type": "output"}]}}}}
                if path == "/view":
                    seen["view"] = params
                    return Resp(200, content=png_bytes((128, 72)))
                raise AssertionError(path)

            def post(self, path, json_data=None, **kw):
                seen["wf"] = json_data["prompt"]
                return {"prompt_id": "pid1"}
        with mock.patch("agenty_core.utils.comfyui_client.get_client", lambda: Client()), \
                mock.patch("agenty_core.tools.image_io.stage_image",
                           lambda p, **k: {"name": "a.png", "subfolder": "", "type": "input"}):
            info = r.upscale_image(src, dest, 2)
        self.assertEqual(info["model"], "4x_NMKD-Superscale-SP_178000_G.pth")
        self.assertEqual(seen["wf"]["2"]["inputs"]["model_name"], info["model"])
        self.assertEqual(seen["wf"]["1"]["inputs"]["image"], "a.png")
        self.assertEqual(seen["view"]["filename"], "o.png")
        self.assertTrue(dest.exists())


if __name__ == "__main__":
    unittest.main()
