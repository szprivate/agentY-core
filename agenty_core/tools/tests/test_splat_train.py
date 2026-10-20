"""splat_train + atlas_capture offline: the 180-degree axis turn (positions, rotations, SH colour), the floater
cleanup, the Atlas -> Nerfstudio export, the background job with a fake Brush, and the all-round capture plan."""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from agenty_core.tools import atlas_capture as ac
from agenty_core.tools import splat_train as st
from agenty_core.tools import worldlabs_atlas as wl

# -- a 3DGS SH evaluation (the reference renderer's constants) ----------------------
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]


def sh_basis(d):
    x, y, z = d
    xx, yy, zz = x * x, y * y, z * z
    return np.array([-C1 * y, C1 * z, -C1 * x,
                     C2[0] * x * y, C2[1] * y * z, C2[2] * (2 * zz - xx - yy), C2[3] * x * z, C2[4] * (xx - yy),
                     C3[0] * y * (3 * xx - yy), C3[1] * x * y * z, C3[2] * y * (4 * zz - xx - yy),
                     C3[3] * z * (2 * zz - 3 * xx - 3 * yy), C3[4] * x * (4 * zz - xx - yy),
                     C3[5] * z * (xx - yy), C3[6] * x * (xx - 3 * yy)])


def quat_mat(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def splat_props(sh_rest=15):
    return (["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * sh_rest)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"])


def random_splats(n=50, seed=0):
    rng = np.random.default_rng(seed)
    props = splat_props()
    return props, rng.normal(size=(n, len(props))).astype("float32")


class FlipTests(unittest.TestCase):
    F = np.diag([1.0, -1.0, -1.0])

    def test_twice_is_identity(self):
        props, a = random_splats()
        b = st.flip_x(props, st.flip_x(props, a))
        ix = {p: i for i, p in enumerate(props)}
        rot = [ix[f"rot_{j}"] for j in range(4)]
        other = [i for i in range(len(props)) if i not in rot]
        self.assertTrue(np.allclose(b[:, other], a[:, other]))
        self.assertTrue(np.allclose(np.abs(b[:, rot]), np.abs(a[:, rot])))   # q and -q are the same turn

    def test_positions_and_rotations_turn_with_the_scene(self):
        props, a = random_splats()
        b = st.flip_x(props, a)
        ix = {p: i for i, p in enumerate(props)}
        for k in range(len(a)):
            p0 = a[k, [ix["x"], ix["y"], ix["z"]]]
            p1 = b[k, [ix["x"], ix["y"], ix["z"]]]
            self.assertTrue(np.allclose(p1, self.F @ p0))
            R0 = quat_mat(*a[k, [ix[f"rot_{j}"] for j in range(4)]])
            R1 = quat_mat(*b[k, [ix[f"rot_{j}"] for j in range(4)]])
            self.assertTrue(np.allclose(R1, self.F @ R0, atol=1e-5))

    def test_view_dependent_colour_is_the_same_seen_from_the_turned_direction(self):
        props, a = random_splats(5)
        b = st.flip_x(props, a)
        ix = {p: i for i, p in enumerate(props)}
        rng = np.random.default_rng(1)
        for k in range(len(a)):
            for _ in range(10):
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                for c in range(3):
                    coef0 = a[k, [ix[f"f_rest_{c * 15 + j}"] for j in range(15)]]
                    coef1 = b[k, [ix[f"f_rest_{c * 15 + j}"] for j in range(15)]]
                    self.assertAlmostEqual(float(sh_basis(d) @ coef0), float(sh_basis(self.F @ d) @ coef1), places=4)

    def test_colour_only_copy(self):
        props, a = random_splats()
        p2, b = st.color_only(props, a)
        self.assertFalse(any(p.startswith("f_rest") for p in p2))
        self.assertEqual(b.shape, (len(a), len(p2)))
        self.assertEqual(st.sh_rest_count(props), 15)
        self.assertEqual(st.sh_rest_count(p2), 0)

    def test_ply_round_trip(self):
        props, a = random_splats()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "s.ply"
            st.write_ply(f, props, a, ("hello",))
            p2, b = st.read_ply(f)
        self.assertEqual(p2, props)
        self.assertTrue(np.array_equal(a, b))


class PruneTests(unittest.TestCase):
    def test_drops_huge_near_camera_and_invisible(self):
        props = splat_props(0)
        ix = {p: i for i, p in enumerate(props)}
        a = np.zeros((4, len(props)), dtype="float32")
        a[:, ix["opacity"]] = 3.0                       # opaque
        a[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]] = math.log(0.05)
        a[:, ix["z"]] = -5.0
        a[1, ix["scale_0"]] = math.log(3.0)             # huge
        a[2, ix["z"]] = -0.3                            # in front of the camera ...
        a[2, ix["scale_1"]] = math.log(0.5)             # ... and big
        a[3, ix["opacity"]] = -8.0                      # invisible
        kept, dropped = st.prune(props, a, [[0.0, 0.0, 0.0]])
        self.assertEqual(dropped, 3)
        self.assertEqual(len(kept), 1)


def write_world(wd: Path, n=3):
    """A small Atlas-like world: n views side by side, 64x36 pictures, log-depth PNGs at 2 m."""
    wd.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(n):
        Image.new("RGB", (64, 36), (40 * i, 100, 200)).save(wd / f"view_{i:03d}.png")
        v = (math.log(2.0) - math.log(10.0)) / (math.log(0.5) - math.log(10.0))
        Image.fromarray(np.full((36, 64), int(v * 255), dtype="uint8")).save(wd / f"depth_{i:03d}.png")
        cam = wl.camera_to_api(ac._ry(15 * i), [0.3 * i, 1.5, 0.0],
                               {"fx": 60.0, "fy": 60.0, "cx": 32.0, "cy": 18.0, "width": 64, "height": 36})
        frames.append({"id": i, "image": f"view_{i:03d}.png", "depth": f"depth_{i:03d}.png",
                       "depth_log": {"zMin": 0.5, "zMax": 10.0}, "camera": cam, "source": "posed" if i == 0 else "generated"})
    (wd / "world.json").write_text(json.dumps({"version": 1, "name": wd.name, "frames": frames, "operations": []}))


class ExportTests(unittest.TestCase):
    def test_atlas_world_to_nerfstudio(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d) / "room"
            write_world(wd)
            info = st.export_atlas_dataset(wd, Path(d) / "ds", stride=4)
            t = json.loads((Path(d) / "ds" / "transforms.json").read_text())
            self.assertEqual(info["views"], 3)
            self.assertEqual(len(t["frames"]), 4)                      # + the held-out copy, first
            self.assertEqual(t["frames"][0]["file_path"], "images/000_eval.png")
            self.assertEqual(t["frames"][0]["transform_matrix"], t["frames"][1]["transform_matrix"])
            self.assertTrue((Path(d) / "ds" / t["frames"][2]["file_path"]).exists())
            m = np.array(t["frames"][2]["transform_matrix"])
            self.assertTrue(np.allclose(m[:3, 3], [0.3, 1.5, 0.0], atol=1e-5))
            pts = (Path(d) / "ds" / "points.ply").read_bytes()
            self.assertIn(b"element vertex %d" % info["points"], pts)
            # every unprojected point of view 0 lies ~2 m in front of its camera (along -Z)
            head_end = pts.index(b"end_header\n") + 11
            rec = np.frombuffer(pts[head_end:], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
            self.assertGreater(len(rec), 0)
            self.assertTrue(np.all(rec["z"] < -1.0))

    def test_a_single_picture_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d) / "one"
            write_world(wd, 1)
            with self.assertRaises(st.SplatError) as cm:
                st.export_atlas_dataset(wd, Path(d) / "ds")
            self.assertEqual(cm.exception.kind, "args")


class FakeProc:
    """Stands in for Brush: writes its result when created, exits at once."""
    def __init__(self, args, **kw):
        export = Path(args[args.index("--export-path") + 1])
        steps = int(args[args.index("--total-steps") + 1])
        export.mkdir(parents=True, exist_ok=True)
        (export / f"eval_{steps}").mkdir()
        Image.new("RGB", (8, 8)).save(export / f"eval_{steps}" / "000_eval.png")
        props, a = random_splats(20)
        ix = {p: i for i, p in enumerate(props)}
        a[:, ix["opacity"]] = 3.0
        a[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]] = math.log(0.05)
        a[:, ix["z"]] = -5.0
        a[0, ix["scale_0"]] = math.log(4.0)                         # one floater
        st.write_ply(export / f"export_{steps}.ply", props, a)
        self.args, self.pid = args, 4242

    def poll(self):
        return 0


class JobTests(unittest.TestCase):
    def test_train_finishes_cleans_and_opens(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d) / "atlas" / "room"
            write_world(wd)
            client = mock.Mock()
            client.post.return_value = {"key": "tab1", "scene_file": "x.json"}
            with mock.patch.object(st, "brush_exe", return_value=Path(d) / "brush_app.exe"), \
                    mock.patch.object(st.subprocess, "Popen", FakeProc), \
                    mock.patch("agenty_core.utils.comfyui_client.get_client", return_value=client), \
                    mock.patch.object(wl, "_publish"):
                out = json.loads(st.train_splat(str(wd), name="room", steps=1000, wait_s=5,
                                                output_dir=str(Path(d) / "splats")))
            self.assertEqual(out["status"], "ok", out)
            self.assertEqual(out["splats"], 19)
            self.assertEqual(out["dropped_floaters"], 1)
            self.assertTrue(Path(out["splat"]).exists() and Path(out["splat_lite"]).exists())
            self.assertTrue(Path(out["preview"]).exists())
            self.assertEqual(out["viewer"]["tab"], "tab1")
            calls = [c.args[0] for c in client.post.call_args_list]
            self.assertIn("/free", calls)                                   # VRAM freed first
            self.assertIn("/bepic/previz_shot", calls)
            scene = client.post.call_args_list[-1].kwargs["json_data"]["scene"]
            self.assertTrue(scene["items"][0]["src"]["path"].endswith("room_lite.ply"))
            job = json.loads((Path(d) / "splats" / "room" / "job.json").read_text())
            args = job["command"]
            self.assertEqual(args[args.index("--eval-split-every") + 1], "5")     # 3 views + 2: only the copy is held out
            props, a = st.read_ply(Path(out["splat"]))
            self.assertTrue(np.all(a[:, props.index("z")] > 0))              # turned: z = -(-5)
            # status of a finished job just reports it
            again = json.loads(st.splat_training_status("room", output_dir=str(Path(d) / "splats")))
            self.assertEqual(again["splat"], out["splat"])

    def test_running_then_failed(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d) / "room"
            write_world(wd)

            class Running(FakeProc):
                def __init__(self, args, **kw):
                    Path(args[args.index("--export-path") + 1]).mkdir(parents=True, exist_ok=True)
                    (Path(args[args.index("--export-path") + 1]) / "eval_500").mkdir()
                    self.pid, self.code = 99, None

                def poll(self):
                    return self.code
            with mock.patch.object(st, "brush_exe", return_value=Path(d) / "b.exe"), \
                    mock.patch.object(st.subprocess, "Popen", Running), \
                    mock.patch.object(st, "_free_comfyui_vram", return_value=False):
                out = json.loads(st.train_splat(str(wd), name="r2", steps=2000, wait_s=0, output_dir=d + "/s"))
                self.assertEqual(out["status"], "running")
                self.assertEqual(out["step"], 500)
                busy = json.loads(st.train_splat(str(wd), name="r2", steps=2000, wait_s=0, output_dir=d + "/s"))
                self.assertEqual(busy["error_kind"], "busy")
                st._PROCS["r2"].code = 1
                failed = json.loads(st.splat_training_status("r2", output_dir=d + "/s"))
            self.assertEqual(failed["error_kind"], "execution")

    def test_unknown_source(self):
        with tempfile.TemporaryDirectory() as d:
            out = json.loads(st.train_splat(d, name="x", output_dir=d))
            self.assertEqual(out["error_kind"], "args")

    def test_brush_path_override(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "brush_app.exe"
            exe.write_bytes(b"")
            with mock.patch.dict(os.environ, {"BRUSH_PATH": str(exe)}):
                self.assertEqual(st.brush_exe(), exe)
            with mock.patch.dict(os.environ, {"BRUSH_PATH": str(exe) + "x"}):
                with self.assertRaises(st.SplatError):
                    st.brush_exe()


class CapturePlanTests(unittest.TestCase):
    def test_ring_goes_all_round_in_growing_yaw(self):
        views = ac.plan_ring([0.0, 0.0, -4.0], 1.0, 30.0, ceiling_floor=False, inner=False)
        yaws = [v["yaw"] for v in views]
        self.assertEqual(len(yaws), 12)
        self.assertEqual(sorted(round(y % 360) for y in yaws), list(range(0, 360, 30)))
        self.assertEqual([abs(y) for y in yaws], sorted(abs(y) for y in yaws))     # outwards from the picture
        for v in views:                                                            # behind the centre, looking out
            d = ac.look_dir(v["yaw"])
            self.assertTrue(np.allclose(np.array(v["position"]) + np.array(d), [0, 0, -4.0], atol=1e-9))

    def test_look_direction_conventions(self):
        self.assertTrue(np.allclose(ac.look_dir(0), [0, 0, -1]))
        self.assertTrue(np.allclose(ac.look_dir(90), [-1, 0, 0], atol=1e-9))    # yaw > 0 turns left
        self.assertGreater(ac.look_dir(0, 40)[1], 0)                            # pitch > 0 looks up
        self.assertGreater(ac.look_dir(180, 40)[1], 0)                          # also when turned round

    def test_full_plan_batches(self):
        views = ac.plan_ring([0, 0, -4.0])
        self.assertEqual(len(views), 12 + 6 + 6)
        bs = ac.batches(views)
        self.assertTrue(all(len(b) <= 8 for b in bs))
        self.assertEqual(sum(len(b) for b in bs), len(views))

    def test_center_forms(self):
        self.assertEqual(ac._center({}, Path("."), {"forward": 3, "right": 1}), [1.0, 0.0, -3.0])
        self.assertEqual(ac._center({}, Path("."), [1, 2, 3]), [1.0, 2.0, 3.0])

    def test_capture_requests(self):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d) / "room"
            write_world(wd, 2)
            sent = []

            def fake_run(job, op, timeout_s):
                sent.append(job["cameras"])
                w = wl._load_world(wd)
                for c in job["cameras"]:
                    i = len(w["frames"])
                    w["frames"].append({**w["frames"][0], "id": i, "camera": c, "source": "generated"})
                wl._save_world(wd, w)
                return json.dumps({"status": "ok"})
            with mock.patch.object(wl, "_api_key", return_value="k"), \
                    mock.patch.object(wl, "_submit", return_value={"id": "op"}) as sub, \
                    mock.patch.object(wl, "_run", side_effect=fake_run), \
                    mock.patch.object(wl, "build_world_mesh", return_value={"vertices": 1}), \
                    mock.patch.object(wl, "context_frames", side_effect=lambda wd_, ctx: [f["id"] for f in ctx]):
                w = wl._load_world(wd)
                w["glb"] = "world.glb"
                wl._save_world(wd, w)
                out = json.loads(ac.atlas_capture_around(str(wd), center={"forward": 2}, prompt="empty, no people"))
            self.assertEqual(out["status"], "ok", out)
            self.assertEqual(out["views_added"], 24)
            self.assertEqual(sum(len(c) for c in sent), 24)
            bodies = [c.args[1] for c in sub.call_args_list]
            self.assertTrue(all(b["prompt"] == "empty, no people" for b in bodies))
            self.assertTrue(all(len(b["contextFrames"]) <= 4 for b in bodies))
            self.assertTrue(all(b["returnDepth"] for b in bodies))


if __name__ == "__main__":
    unittest.main()
