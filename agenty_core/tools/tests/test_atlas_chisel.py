"""atlas_chisel offline: the request follows the current atlasChisel shape (targetFrames / contextFrames),
masks and pictures land on the 1280x720 grid, the files, and the resume path."""

import base64
import io
import json
from pathlib import Path
from unittest import mock

from PIL import Image

from agenty_core.tools import atlas_chisel as c
from agenty_core.tools import worldlabs_atlas as w
from agenty_core.tools.tests.test_worldlabs_atlas import Base, CAM, done_op, frame, png_bytes
from agenty_core.utils.provenance import sidecar_path


def decode(asset):
    return Image.open(io.BytesIO(base64.b64decode(asset["base64"])))


def chisel_op(op_id="op_c"):
    return done_op(op_id, {"frames": [{"imageAsset": {"assetId": "asset_out", "url": "https://dl/out.png"},
                                       "camera": CAM}], "promptUsed": "a meadow", "requestId": "r1"})


class ChiselTests(Base):
    def setUp(self):
        super().setUp()
        self.published = []
        p = mock.patch.object(w, "_publish", lambda path: self.published.append(str(path)))
        p.start()
        self.addCleanup(p.stop)

    def file(self, name, data):
        p = Path(self.tmp.name) / name
        p.write_bytes(data)
        return str(p)

    def body(self):
        return next(x for x in self.calls if x[1].endswith("tasks:atlasChisel"))[2]

    def test_masked_edit_of_a_picture_uses_target_frames(self):
        img = self.file("in.png", png_bytes((1920, 1080)))
        m = Image.new("L", (1920, 1080), 0)
        m.paste(255, (0, 0, 960, 1080))                     # left half white = change it
        buf = io.BytesIO()
        m.save(buf, "PNG")
        mask = self.file("mask.png", buf.getvalue())
        out = self.run_tool(c.atlas_chisel, {("POST", "/tasks:atlasChisel"): chisel_op()},
                            prompt="remove the fence", image_path=img, mask_path=mask, seed=7)
        self.assertEqual(out["status"], "ok", out)
        b = self.body()
        self.assertNotIn("targetCameras", b)
        self.assertNotIn("sourceFrames", b)
        self.assertEqual(len(b["targetFrames"]), 1)
        tf = b["targetFrames"][0]
        self.assertEqual((tf["camera"]["intrinsics"]["width"], tf["camera"]["intrinsics"]["height"]), (1280, 720))
        self.assertEqual(decode(tf["imageAsset"]).size, (1280, 720))
        mk = decode(tf["maskAsset"])
        self.assertEqual((mk.size, mk.mode), ((1280, 720), "L"))
        self.assertEqual(mk.getpixel((10, 360)), 0)          # Atlas: black = generate
        self.assertEqual(mk.getpixel((1270, 360)), 255)      # white = keep
        self.assertEqual(b["contextFrames"], [])
        self.assertEqual(b["prompt"], "remove the fence")
        self.assertEqual(b["modelParameters"], {"seed": 7})
        self.assertFalse(b["returnDepth"])
        self.assertNotIn("voxelSize", b)
        png = Path(out["path"])
        self.assertTrue(png.exists() and png.name == "chisel_000.png" and png.parent.name == "chisel")
        self.assertTrue(sidecar_path(png).exists())
        self.assertEqual(self.published, [str(png)])
        self.assertEqual(out["mode"], "edit")

    def test_mask_white_is_keep_is_sent_as_is(self):
        img = self.file("in.png", png_bytes())
        mask = self.file("mask.png", png_bytes(color=(255, 255, 255)))
        self.run_tool(c.atlas_chisel, {("POST", "/tasks:atlasChisel"): chisel_op()},
                      prompt="x", image_path=img, mask_path=mask, mask_white_is="keep")
        self.assertEqual(decode(self.body()["targetFrames"][0]["maskAsset"]).getpixel((5, 5)), 255)

    def test_depth_guided_log_png_with_voxels(self):
        depth = self.file("depth.png", png_bytes())
        out = self.run_tool(c.atlas_chisel, {("POST", "/tasks:atlasChisel"): chisel_op()},
                            prompt="a gothic chapel in a meadow", depth_path=depth, z_min=10, z_max=150,
                            voxel_size=0.1, fov=40, enhance_prompt=False)
        self.assertEqual(out["status"], "ok", out)
        b = self.body()
        d = b["targetFrames"][0]["depth"]
        self.assertEqual((d["zMin"], d["zMax"]), (10.0, 150.0))
        self.assertIn("base64", d["logDepthAsset"])
        self.assertNotIn("imageAsset", b["targetFrames"][0])
        self.assertEqual(b["voxelSize"], 0.1)
        self.assertFalse(b["enhancePrompt"])
        self.assertEqual(out["mode"], "depth")

    def test_depth_exr_is_uploaded(self):
        depth = self.file("depth.exr", b"EXR")
        routes = {("POST", "/tasks:atlasChisel"): chisel_op(), ("POST", "/assets"): {"id": "asset_dep"},
                  ("POST", "/assets/asset_dep:createUploadUrl"): {"uploadUrl": "https://up/x", "method": "PUT"},
                  ("PUT", "https://up/x"): {}, ("POST", "/assets/asset_dep:completeUpload"): {}}
        out = self.run_tool(c.atlas_chisel, routes, prompt="a hall", depth_path=depth)
        self.assertEqual(out["status"], "ok", out)
        self.assertEqual(self.body()["targetFrames"][0]["depth"], {"depthAsset": {"assetId": "asset_dep"}})

    def test_edit_a_world_frame_with_context_and_style(self):
        made = self.run_tool(w.atlas_create_world, {
            ("POST", "/tasks:atlasTextToImage"): done_op("op_t", {"frames": [frame(0)], "promptUsed": "a meadow"}),
        }, name="field", prompt="a meadow")
        self.assertEqual(made["status"], "ok", made)
        self.calls.clear()
        mask = self.file("mask.png", png_bytes(color=(0, 0, 0)))
        style = self.file("style.png", png_bytes((64, 64)))
        out = self.run_tool(c.atlas_chisel, {("POST", "/tasks:atlasChisel"): chisel_op()},
                            prompt="add a well", world="field", frame=0, mask_path=mask, style_image=style)
        self.assertEqual(out["status"], "ok", out)
        b = self.body()
        self.assertEqual(b["targetFrames"][0]["camera"]["extrinsics"]["position"], CAM["extrinsics"]["position"])
        self.assertIn("imageAsset", b["targetFrames"][0])
        self.assertIn("base64", b["styleReferenceImageAsset"])
        self.assertEqual(b["contextFrames"], [])           # the only view is the one being edited
        wj = json.loads((Path(self.tmp.name) / "field" / "world.json").read_text())
        self.assertEqual(wj["chisel"][0]["file"], "chisel/chisel_000.png")
        self.assertEqual(wj["operations"][-1]["task"], "atlasChisel")
        # a new camera in the same world takes that view as context (picture + camera, no depth)
        self.calls.clear()
        out = self.run_tool(c.atlas_chisel, {("POST", "/tasks:atlasChisel"): chisel_op("op_d")},
                            prompt="the meadow", world="field", camera={"forward": 1.0},
                            depth_path=self.file("d.png", png_bytes()), z_min=1, z_max=20)
        self.assertEqual(out["status"], "ok", out)
        ctx = self.body()["contextFrames"]
        self.assertEqual(ctx, [{"imageAsset": {"assetId": "asset_img0"}, "camera": CAM}])
        self.assertTrue(out["path"].endswith("chisel_001.png"))

    def test_errors_before_anything_is_billed(self):
        img = self.file("in.png", png_bytes())
        for kw in ({"prompt": "", "image_path": img},                       # no prompt
                   {"prompt": "x"},                                         # nothing to build on
                   {"prompt": "x", "mask_path": img},                       # mask without picture
                   {"prompt": "x", "depth_path": self.file("d.png", png_bytes())},   # log PNG without z range
                   {"prompt": "x", "depth_path": self.file("d.png", png_bytes()), "z_min": 1, "z_max": 9,
                    "voxel_size": 0.9},
                   {"prompt": "x", "image_path": img, "mask_path": img, "mask_white_is": "both"},
                   {"prompt": "x", "frame": 0, "mask_path": img},           # frame without world
                   {"prompt": "x", "camera": {"forward": 1}, "image_path": img}):
            out = self.run_tool(c.atlas_chisel, {}, **kw)
            self.assertEqual(out.get("error_kind"), "args", (kw, out))
        self.assertFalse([x for x in self.calls if "tasks:" in x[1]])

    def test_timeout_then_atlas_fetch_finishes(self):
        img = self.file("in.png", png_bytes())
        routes = {("POST", "/tasks:atlasChisel"): {"id": "op_slow", "done": False},
                  ("POST", "/operations/op_slow:wait"): {"id": "op_slow", "done": False}}
        clock = iter([0, 0, 0, 0, 1000, 1000, 1000, 1000])
        with mock.patch.object(w.time, "monotonic", lambda: next(clock, 2000)):
            out = self.run_tool(c.atlas_chisel, routes, prompt="x", image_path=img, mask_path=img, timeout_s=5)
        self.assertEqual(out["error_kind"], "timeout")
        self.assertEqual(out["operation_id"], "op_slow")
        out = self.run_tool(w.atlas_fetch, {("GET", "/operations/op_slow"): chisel_op("op_slow")},
                            operation_id="op_slow")
        self.assertEqual(out["status"], "ok", out)
        self.assertTrue(Path(out["path"]).exists())
        self.assertFalse((Path(self.tmp.name) / "chisel" / "pending" / "op_slow.json").exists())

    def test_camera_of_another_size_is_put_on_the_grid(self):
        cam = {"extrinsics": CAM["extrinsics"],
               "intrinsics": {"fx": 1400, "fy": 1400, "cx": 960, "cy": 540, "width": 1920, "height": 1080}}
        K = c._on_grid(cam)["intrinsics"]
        self.assertEqual((K["width"], K["height"], K["cx"], K["cy"]), (1280, 720, 640.0, 360.0))
        self.assertAlmostEqual(K["fy"], 1400 * 720 / 1080)


if __name__ == "__main__":
    import unittest
    unittest.main()
