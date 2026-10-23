"""dcc_build / dcc_silhouette_check without any app: the projection maths, the result shaping, the argument
checks, the driver sources and the live-socket reply parsing (socket mocked)."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core.tools import dcc


def cam_looking_down_minus_z(position=(0, 0, 10), focal=50.0, film_w=36.0, film_h=18.0, sx=0.0, sy=0.0):
    return {"position": list(position), "right": [1, 0, 0], "up": [0, 1, 0], "back": [0, 0, 1],
            "fx": focal / film_w, "fy": focal / film_h, "shift_x": sx, "shift_y": sy}


class ProjectPoint(unittest.TestCase):
    def test_point_on_the_optical_axis_is_the_frame_centre(self):
        x, y, front = dcc.project_point([0, 0, 0], cam_looking_down_minus_z())
        self.assertAlmostEqual(x, 0.5)
        self.assertAlmostEqual(y, 0.5)
        self.assertTrue(front)

    def test_hand_computed_offset(self):
        # 1 m right and 0.5 m up at 10 m, 50 mm on a 36 x 18 film: x = 0.5 + 50/36 * 0.1, y_bottom = 0.5 + 50/18 * 0.05
        x, y, _ = dcc.project_point([1, 0.5, 0], cam_looking_down_minus_z())
        self.assertAlmostEqual(x, 0.5 + 50 / 36 * 0.1, places=9)
        self.assertAlmostEqual(y, 1 - (0.5 + 50 / 18 * 0.05), places=9)

    def test_y_is_measured_from_the_top(self):
        _, y_up, _ = dcc.project_point([0, 1, 0], cam_looking_down_minus_z())
        _, y_down, _ = dcc.project_point([0, -1, 0], cam_looking_down_minus_z())
        self.assertLess(y_up, 0.5)
        self.assertGreater(y_down, 0.5)

    def test_behind_the_camera_is_not_in_front(self):
        self.assertFalse(dcc.project_point([0, 0, 20], cam_looking_down_minus_z())[2])

    def test_lens_shift_moves_the_frame_so_the_subject_moves_the_other_way(self):
        # positive shift_y moves the frame UP: the centre point goes DOWN in the picture (y from top grows)
        _, y, _ = dcc.project_point([0, 0, 0], cam_looking_down_minus_z(sy=0.1))
        self.assertAlmostEqual(y, 0.6)
        x, _, _ = dcc.project_point([0, 0, 0], cam_looking_down_minus_z(sx=0.1))
        self.assertAlmostEqual(x, 0.4)

    def test_rotated_camera(self):
        # camera at the origin looking along +Y (Blender rotation 90 deg about X): right = +X, up = +Z, back = -Y
        cam = {"position": [0, 0, 0], "right": [1, 0, 0], "up": [0, 0, 1], "back": [0, -1, 0],
               "fx": 50 / 36, "fy": 50 / 18, "shift_x": 0, "shift_y": 0}
        x, y, front = dcc.project_point([1, 5, 0.5], cam)
        self.assertAlmostEqual(x, 0.5 + 50 / 36 / 5)
        self.assertAlmostEqual(y, 1 - (0.5 + 50 / 18 * 0.1))
        self.assertTrue(front)


class Intrinsics(unittest.TestCase):
    def test_blender_landscape_auto(self):
        i = dcc.intrinsics_blender({"focal_mm": 50, "sensor_width_mm": 36, "sensor_height_mm": 24, "sensor_fit": "AUTO",
                                    "res_x": 1000, "res_y": 500})
        self.assertAlmostEqual(i["fx"], 50 / 36)
        self.assertAlmostEqual(i["fy"], 50 / 18)

    def test_blender_portrait_auto_fits_the_height(self):
        i = dcc.intrinsics_blender({"focal_mm": 50, "sensor_width_mm": 36, "sensor_height_mm": 24, "sensor_fit": "AUTO",
                                    "res_x": 500, "res_y": 1000})
        self.assertAlmostEqual(i["fy"], 50 / 36)          # AUTO uses sensor_width for the larger side
        self.assertAlmostEqual(i["fx"], 50 / 18)

    def test_blender_shift_is_a_fraction_of_the_fit_side(self):
        # landscape: shift_y is a fraction of the WIDTH, so as a fraction of the height it is twice as large here
        i = dcc.intrinsics_blender({"focal_mm": 50, "sensor_width_mm": 36, "sensor_height_mm": 24, "sensor_fit": "AUTO",
                                    "res_x": 1000, "res_y": 500, "shift_x": 0.1, "shift_y": 0.1})
        self.assertAlmostEqual(i["shift_x"], 0.1)
        self.assertAlmostEqual(i["shift_y"], 0.2)

    def test_maya_fit_modes(self):
        base = {"focal_mm": 50, "film_w_in": 36 / 25.4, "film_h_in": 24 / 25.4, "device_aspect": 2.0}
        horiz = dcc.intrinsics_maya(dict(base, film_fit=1))
        self.assertAlmostEqual(horiz["fx"], 50 / 36)
        self.assertAlmostEqual(horiz["fy"], 50 / 18)
        vert = dcc.intrinsics_maya(dict(base, film_fit=2))
        self.assertAlmostEqual(vert["fy"], 50 / 24)
        # Fill with a wider picture than the film (2.0 > 1.5) fits horizontally; Overscan fits vertically
        self.assertAlmostEqual(dcc.intrinsics_maya(dict(base, film_fit=0))["fx"], horiz["fx"])
        self.assertAlmostEqual(dcc.intrinsics_maya(dict(base, film_fit=3))["fy"], vert["fy"])

    def test_maya_film_offset_in_inches_becomes_a_frame_fraction(self):
        i = dcc.intrinsics_maya({"focal_mm": 50, "film_w_in": 36 / 25.4, "film_h_in": 18 / 25.4, "device_aspect": 2.0,
                                 "film_fit": 1, "film_offset_x_in": 0.1})
        self.assertAlmostEqual(i["shift_x"], 0.1 * 25.4 / 36)

    def test_houdini_aperture_is_the_frame_width(self):
        i = dcc.intrinsics_houdini({"focal_mm": 50, "aperture_mm": 36, "res_x": 1000, "res_y": 500})
        self.assertAlmostEqual(i["fx"], 50 / 36)
        self.assertAlmostEqual(i["fy"], 50 / 18)

    def test_camera_from_raw_feeds_project_point(self):
        raw = {"position": [0, 0, 10], "right": [1, 0, 0], "up": [0, 1, 0], "back": [0, 0, 1], "focal_mm": 50,
               "aperture_mm": 36, "res_x": 1000, "res_y": 500}
        x, y, _ = dcc.project_point([1, 0.5, 0], dcc.camera_from_raw("houdini", raw))
        self.assertAlmostEqual(x, 0.5 + 50 / 36 * 0.1)
        self.assertAlmostEqual(y, 1 - (0.5 + 50 / 18 * 0.05))


class ShapeCheck(unittest.TestCase):
    PTS = [{"name": "ridge", "world": [0, 0, 0], "target": [0.5, 0.30]},
           {"name": "deck", "world": [1, 0, 0], "target": [0.6, 0.60]},
           {"name": "mast", "world": [2, 0, 0], "target": [0.7, 0.10]}]

    def test_pass_and_fail_with_worst_first(self):
        res = dcc.shape_check(self.PTS, [(0.51, 0.31, True), (0.60, 0.68, True), (0.70, 0.10, True)], 0.03)
        self.assertFalse(res["overall_pass"])
        self.assertEqual(res["failed"], 1)
        self.assertEqual(res["points"][0]["name"], "deck")           # the failure comes first
        self.assertFalse(res["points"][0]["pass"])
        self.assertAlmostEqual(res["points"][0]["dy"], 0.08)
        self.assertIn("too low", res["points"][0]["hint"])
        self.assertEqual(res["worst"][0]["name"], "deck")
        self.assertTrue(all(p["pass"] for p in res["points"][1:]))

    def test_all_within_tolerance(self):
        res = dcc.shape_check(self.PTS, [(0.5, 0.3, True), (0.6, 0.6, True), (0.7, 0.1, True)], 0.03)
        self.assertTrue(res["overall_pass"])
        self.assertNotIn("worst", res)

    def test_too_high_hint_matches_the_spec_wording(self):
        res = dcc.shape_check(self.PTS[:1], [(0.5, 0.22, True)], 0.03)
        self.assertIn("projected 0.08 too high: model too tall or camera too low", res["points"][0]["hint"])

    def test_sideways_hints(self):
        left = dcc.shape_check(self.PTS[:1], [(0.40, 0.30, True)], 0.03)["points"][0]["hint"]
        right = dcc.shape_check(self.PTS[:1], [(0.60, 0.30, True)], 0.03)["points"][0]["hint"]
        self.assertIn("too far left", left)
        self.assertIn("too far right", right)

    def test_behind_the_camera_always_fails(self):
        res = dcc.shape_check(self.PTS[:1], [(0.5, 0.3, False)], 0.03)
        self.assertFalse(res["overall_pass"])
        self.assertIn("behind the camera", res["points"][0]["hint"])

    def test_common_offset_is_blamed_on_the_camera(self):
        projected = [(p["target"][0], p["target"][1] + 0.08, True) for p in self.PTS]
        res = dcc.shape_check(self.PTS, projected, 0.03)
        self.assertFalse(res["overall_pass"])
        self.assertIn("camera", res["overall_hint"])

    def test_point_without_target_is_only_projected(self):
        res = dcc.shape_check([{"name": "a", "world": [0, 0, 0], "target": None}], [(0.2, 0.3, True)], 0.03)
        self.assertEqual(res["points"][0]["projected"], [0.2, 0.3])
        self.assertNotIn("pass", res["points"][0])
        self.assertFalse(res["overall_pass"])                         # nothing was judged


class Arguments(unittest.TestCase):
    def test_missing_script_is_refused(self):
        r = json.loads(dcc.dcc_build("/definitely/not/here.py"))
        self.assertFalse(r["ok"])
        self.assertIn("not found", r["error"])

    def test_houdini_live_is_refused_with_the_reason(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("pass\n")
        try:
            r = json.loads(dcc.dcc_build(f.name, app="houdini", mode="live"))
        finally:
            Path(f.name).unlink()
        self.assertFalse(r["ok"])
        self.assertIn("batch", r["error"])

    def test_unknown_app_mode_engine_resolution(self):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write("pass\n")
        try:
            self.assertIn("app must be", json.loads(dcc.dcc_build(f.name, app="cinema4d"))["error"])
            self.assertIn("mode must be", json.loads(dcc.dcc_build(f.name, mode="cloud"))["error"])
            self.assertIn("render_engine", json.loads(dcc.dcc_build(f.name, render_engine="slow"))["error"])
            self.assertIn("1280x720", json.loads(dcc.dcc_build(f.name, resolution="big"))["error"])
        finally:
            Path(f.name).unlink()

    def test_check_needs_a_scene_in_batch(self):
        pts = [{"name": "a", "world": [0, 0, 0], "target": [0.5, 0.5]}]
        self.assertIn("scene=", json.loads(dcc.dcc_silhouette_check(pts, mode="batch"))["error"])
        self.assertIn("not found", json.loads(dcc.dcc_silhouette_check(pts, mode="batch", scene="/no/such.blend"))["error"])

    def test_bad_points_are_explained(self):
        self.assertIn("non-empty", json.loads(dcc.dcc_silhouette_check([]))["error"])
        bad = [{"name": "a", "world": [1, 2], "target": [0.5, 0.5]}]
        self.assertIn("'world'", json.loads(dcc.dcc_silhouette_check(bad))["error"])
        bad = [{"name": "a", "world": [1, 2, 3], "target": [0.5]}]
        self.assertIn("'target'", json.loads(dcc.dcc_silhouette_check(bad))["error"])

    def test_points_may_arrive_as_a_json_string(self):
        pts = dcc._clean_points(json.dumps([{"name": "a", "world": [1, 2, 3], "target": [0.5, 0.5]}]))
        self.assertEqual(pts[0]["world"], [1.0, 2.0, 3.0])

    def test_default_scene_path_sits_next_to_the_script(self):
        p = Path(dcc.default_scene_path("/x/output/scripts/house_build.py", "blender"))
        self.assertEqual((p.parent.name, p.name), ("_dcc", "house_build.blend"))


class Drivers(unittest.TestCase):
    def test_every_driver_compiles_and_defines_run(self):
        for app in dcc.APPS:
            src = dcc.driver_source(app)
            compile(src, f"{app}_driver", "exec")
            self.assertIn("def run(job)", src)

    def test_batch_wrappers_compile(self):
        for app in dcc.APPS:
            compile(dcc._batch_wrapper(app), f"{app}_batch", "exec")

    def test_live_wrapper_carries_the_job_and_compiles(self):
        code = dcc._live_wrapper("blender", {"action": "check", "points": [[0, 0, 0]], "live": True})
        compile(code, "live", "exec")
        self.assertIn(dcc._RESULT_MARK, code)

    def test_exec_build_script_reports_the_scripts_own_line(self):
        g = {}
        exec(compile((Path(dcc.__file__).with_name("dcc_drivers") / "_common.py").read_text(encoding="utf-8"), "c", "exec"), g)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.py"
            p.write_text("x = 1\nprint('hello')\nraise ValueError('boom')\n", encoding="utf-8")
            ok, out, err, tb = g["exec_build_script"](str(p))
        self.assertFalse(ok)
        self.assertEqual(out.strip(), "hello")
        self.assertIn("ValueError: boom", err)
        self.assertIn("script line 3", err)
        self.assertIn("raise ValueError", err)


class FindApp(unittest.TestCase):
    def test_env_var_wins(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as f:
            pass
        try:
            with mock.patch.dict("os.environ", {"BLENDER_PATH": f.name}):
                self.assertEqual(dcc.find_app("blender"), f.name)
        finally:
            Path(f.name).unlink()

    def test_newest_version_is_taken(self):
        with mock.patch("agenty_core.tools.dcc.shutil.which", return_value=None), \
                mock.patch.dict("os.environ", {}, clear=False), \
                mock.patch("agenty_core.tools.dcc.glob.glob", return_value=[
                    "C:/Program Files/Side Effects Software/Houdini 20.5.1/bin/hython.exe",
                    "C:/Program Files/Side Effects Software/Houdini 21.0.631/bin/hython.exe"]), \
                mock.patch("agenty_core.tools.dcc.os.path.isfile", side_effect=lambda p: "hython" in p):
            os_env = {k: v for k, v in dcc.os.environ.items() if k != "HYTHON_PATH"}
            with mock.patch.dict("os.environ", os_env, clear=True):
                self.assertIn("21.0.631", dcc.find_app("houdini"))


class LiveReplies(unittest.TestCase):
    """The socket is mocked: this is the reply parsing only. The real Blender socket is in test_dcc_integration."""

    def test_blender_reply_with_result_in_stdout(self):
        res = {"ok": True, "native": [[0.5, 0.5, True]], "camera": {}}
        reply = json.dumps({"status": "ok", "stdout": "noise\n" + dcc._RESULT_MARK + json.dumps(res) + "\n"}).encode()
        with mock.patch("agenty_core.tools.dcc._socket_roundtrip", return_value=reply) as m:
            out = dcc._run_live("blender", {"action": "check"}, 5)
        self.assertEqual(out, res)
        sent = json.loads(m.call_args[0][2].decode().rstrip("\0"))
        self.assertEqual(sent["type"], "execute")
        self.assertFalse(sent["strict_json"])

    def test_blender_error_reply(self):
        reply = json.dumps({"status": "error", "message": "NameError: x"}).encode()
        with mock.patch("agenty_core.tools.dcc._socket_roundtrip", return_value=reply):
            with self.assertRaises(dcc.DccError):
                dcc._run_live("blender", {"action": "check"}, 5)

    def test_maya_two_step_reply(self):
        res = {"ok": True, "native": []}
        with mock.patch("agenty_core.tools.dcc._socket_roundtrip",
                        side_effect=[b"", repr(json.dumps(res)).encode()]) as m:
            out = dcc._run_live("maya", {"action": "check"}, 5)
        self.assertEqual(out, res)
        self.assertIn(b"__dcc_result", m.call_args_list[1][0][2])

    def test_closed_port_is_explained(self):
        with mock.patch("agenty_core.tools.dcc._socket_roundtrip", side_effect=ConnectionRefusedError):
            with self.assertRaises(dcc.DccError) as cm:
                dcc._run_live("blender", {"action": "check"}, 5)
        self.assertIn("not listening", str(cm.exception))

    def test_tool_returns_an_error_json_when_the_app_is_down(self):
        pts = [{"name": "a", "world": [0, 0, 0], "target": [0.5, 0.5]}]
        with mock.patch("agenty_core.tools.dcc._socket_roundtrip", side_effect=ConnectionRefusedError):
            r = json.loads(dcc.dcc_silhouette_check(pts, app="blender", mode="live"))
        self.assertFalse(r["ok"])
        self.assertIn("batch", r["error"])


class CrossCheck(unittest.TestCase):
    def test_disagreement_between_app_and_pure_maths_is_reported(self):
        raw = {"name": "C", "kind": "PERSP", "position": [0, 0, 10], "right": [1, 0, 0], "up": [0, 1, 0],
               "back": [0, 0, 1], "focal_mm": 50, "sensor_width_mm": 36, "sensor_height_mm": 24, "sensor_fit": "AUTO",
               "res_x": 1000, "res_y": 500}
        pts = [{"name": "a", "world": [1, 0.5, 0], "target": [0.5, 0.5]}]
        good = dcc.project_point(pts[0]["world"], dcc.camera_from_raw("blender", raw))
        with mock.patch("agenty_core.tools.dcc.run_job", return_value={"ok": True, "camera": raw, "native": [list(good)]}):
            r = json.loads(dcc.dcc_silhouette_check(pts))
        self.assertTrue(r["cross_check"]["agree"])
        with mock.patch("agenty_core.tools.dcc.run_job",
                        return_value={"ok": True, "camera": raw, "native": [[good[0] + 0.05, good[1], True]]}):
            r = json.loads(dcc.dcc_silhouette_check(pts))
        self.assertFalse(r["cross_check"]["agree"])
        self.assertTrue(r["warnings"])


RAW = {"name": "C", "kind": "PERSP", "position": [0, 0, 10], "right": [1, 0, 0], "up": [0, 1, 0], "back": [0, 0, 1],
       "focal_mm": 50, "sensor_width_mm": 36, "sensor_height_mm": 24, "sensor_fit": "AUTO", "res_x": 1000, "res_y": 500}


def fake_check(pts, camera=RAW):
    """What run_job returns for a check: the raw camera and the 'native' projections (pure maths here)."""
    cam = dcc.camera_from_raw("blender", camera)
    return {"ok": True, "camera": camera, "native": [list(dcc.project_point(p["world"], cam)) for p in pts]}


def on_target(world):
    return {"name": "x", "world": world,
            "target": list(dcc.project_point(world, dcc.camera_from_raw("blender", RAW))[:2])}


def good_points():
    pts = []
    for i, w in enumerate(([-2, 0, 0], [2, 0, -4], [0, 3, -2], [0, 0, -2])):
        p = on_target(w)
        p["name"] = f"p{i}"
        pts.append(p)
    return pts


class YawMaths(unittest.TestCase):
    def test_sign_and_magnitude(self):
        # camera-space (right, up, depth): the right end farther -> positive
        self.assertAlmostEqual(dcc.pair_yaw((0, 0, 5), (4, 0, 5))["yaw_deg"], 0.0)
        self.assertAlmostEqual(dcc.pair_yaw((0, 0, 5), (4, 0, 9))["yaw_deg"], 45.0)
        self.assertAlmostEqual(dcc.pair_yaw((4, 0, 9), (0, 0, 5))["yaw_deg"], 45.0)      # pair order is irrelevant
        self.assertAlmostEqual(dcc.pair_yaw((0, 0, 9), (4, 0, 5))["yaw_deg"], -45.0)
        self.assertEqual(dcc.pair_yaw((0, 0, 5), (4, 0, 9))["nearer"], "a")
        self.assertEqual(dcc.pair_yaw((4, 0, 9), (0, 0, 5))["nearer"], "b")

    def test_yaw_check_against_the_camera(self):
        cam = {k: RAW[k] for k in ("position", "right", "up", "back")}
        pts = [{"name": "L", "world": [-2, 0, 0]}, {"name": "R", "world": [2, 0, -4]}]
        ok = dcc.yaw_check(pts, [{"pair": ["L", "R"], "target_yaw_deg": 40, "tolerance_deg": 8}], cam)[0]
        self.assertTrue(ok["pass"])
        self.assertAlmostEqual(ok["yaw_deg"], 45.0, places=1)
        self.assertEqual(ok["nearer"], "L")
        bad = dcc.yaw_check(pts, [{"pair": ["L", "R"], "target_yaw_deg": 0}], cam)[0]
        self.assertFalse(bad["pass"])
        self.assertIn("facade yaw 45 deg vs photo 0 deg", bad["hint"])
        self.assertIn("right", bad["hint"])
        behind = dcc.yaw_check([{"name": "L", "world": [0, 0, 0]}, {"name": "R", "world": [1, 0, 20]}],
                               [{"pair": ["L", "R"], "target_yaw_deg": 0}], cam)[0]
        self.assertFalse(behind["pass"])

    def test_bad_yaw_pairs_are_explained(self):
        pts = good_points()
        for bad, word in (([{"pair": ["p0", "zz"], "target_yaw_deg": 1}], "not among"),
                          ([{"pair": ["p0"], "target_yaw_deg": 1}], "[nameA, nameB]"),
                          ([{"pair": ["p0", "p1"]}], "target_yaw_deg")):
            with mock.patch("agenty_core.tools.dcc.run_job", return_value=fake_check(pts)):
                self.assertIn(word, json.loads(dcc.dcc_silhouette_check(pts, yaw_pairs=bad))["error"])


class YawInTheCheck(unittest.TestCase):
    def run_check(self, yaw_pairs, **kw):
        pts = good_points()
        with mock.patch("agenty_core.tools.dcc.run_job", return_value=fake_check(pts)):
            return json.loads(dcc.dcc_silhouette_check(pts, yaw_pairs=yaw_pairs, **kw))

    def test_matching_yaw_passes_and_counts(self):
        r = self.run_check([{"pair": ["p0", "p1"], "target_yaw_deg": 45, "tolerance_deg": 5}])
        self.assertTrue(r["overall_pass"], r)
        self.assertEqual((r["n_pass"], r["n_total"]), (5, 5))
        self.assertEqual(r["yaw_pairs"][0]["nearer"], "p0")

    def test_straight_on_camera_fails_the_check_with_a_hint(self):
        r = self.run_check([{"pair": ["p0", "p1"], "target_yaw_deg": -20, "tolerance_deg": 8}])
        self.assertFalse(r["overall_pass"])
        self.assertEqual((r["n_pass"], r["n_total"]), (4, 5))
        self.assertTrue(any("facade yaw" in w["hint"] for w in r["worst"]))

    def test_no_yaw_pairs_changes_nothing(self):
        r = self.run_check(None)
        self.assertTrue(r["overall_pass"])
        self.assertEqual(r["n_total"], 4)
        self.assertNotIn("yaw_pairs", r)


class Gate(unittest.TestCase):
    MARK = dcc.DETAIL_MARKER

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sp = Path(self.tmp.name) / "b.py"
        self.sp.write_text("W = 4.0\n" + self.MARK + "\nDOORS = 2\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, pts=None, yaw=None):
        pts = pts or good_points()
        with mock.patch("agenty_core.tools.dcc.run_job", return_value=fake_check(pts)):
            return json.loads(dcc.dcc_silhouette_check(pts, yaw_pairs=yaw, script_path=str(self.sp)))

    def build(self, **kw):
        with mock.patch("agenty_core.tools.dcc.run_job", return_value={"ok": True, "objects": {}}) as m:
            r = json.loads(dcc.dcc_build(str(self.sp), **kw))
        return r, m.call_count

    def test_blockout_is_always_allowed(self):
        r, n = self.build()
        self.assertTrue(r["ok"])
        self.assertEqual(n, 1)
        self.assertEqual(r["gate"]["stage"], "blockout")

    def test_bad_stage(self):
        self.assertIn("stage must be", json.loads(dcc.dcc_build(str(self.sp), stage="polish"))["error"])

    def test_no_check_refuses_and_runs_nothing(self):
        for stage in ("detail", "look"):
            r, n = self.build(stage=stage)
            self.assertFalse(r["ok"])
            self.assertEqual(n, 0)
            self.assertIn("no silhouette check", r["error"])

    def test_check_writes_the_record(self):
        out = self.check()
        rec = json.loads(Path(out["check_file"]).read_text(encoding="utf-8"))
        self.assertTrue(rec["overall_pass"])
        self.assertEqual((rec["n_pass"], rec["n_total"]), (4, 4))
        for k in ("timestamp", "sha", "blockout_sha", "points", "camera", "worst_distance", "scene"):
            self.assertIn(k, rec)
        self.assertTrue(out["check_file"].endswith("b.py.check.json"))

    def test_passing_check_opens_detail_and_look(self):
        self.check()
        for stage in ("detail", "look"):
            r, n = self.build(stage=stage)
            self.assertTrue(r["ok"], r)
            self.assertEqual(n, 1)
            self.assertFalse(r["gate"]["forced"])
            self.assertTrue(r["gate"]["last_check"]["overall_pass"])

    def test_failing_check_refuses_with_the_failing_points_and_hints(self):
        pts = good_points()
        pts[1]["target"][1] += 0.2
        out = self.check(pts)
        self.assertFalse(out["overall_pass"])
        r, n = self.build(stage="detail")
        self.assertFalse(r["ok"])
        self.assertEqual(n, 0)
        self.assertIn("p1", r["error"])
        self.assertIn("too", r["error"])                    # the hint text
        self.assertIn("Fix the block-out constants", r["error"])

    def test_a_failing_yaw_pair_blocks_too(self):
        self.check(yaw=[{"pair": ["p0", "p1"], "target_yaw_deg": 0}])
        r, n = self.build(stage="detail")
        self.assertFalse(r["ok"])
        self.assertIn("facade yaw", r["error"])

    def test_edit_below_the_marker_keeps_the_check_valid(self):
        self.check()
        self.sp.write_text("W = 4.0\n" + self.MARK + "\nDOORS = 5\nWINDOWS = 9\n", encoding="utf-8")
        r, n = self.build(stage="detail")
        self.assertTrue(r["ok"], r)

    def test_edit_above_the_marker_needs_a_new_check(self):
        self.check()
        self.sp.write_text("W = 4.5\n" + self.MARK + "\nDOORS = 2\n", encoding="utf-8")
        r, n = self.build(stage="detail")
        self.assertFalse(r["ok"])
        self.assertEqual(n, 0)
        self.assertIn("changed since", r["error"])
        self.check()
        self.assertTrue(self.build(stage="detail")[0]["ok"])

    def test_whitespace_and_line_endings_do_not_count_as_a_change(self):
        self.check()
        self.sp.write_bytes(b"W = 4.0   \r\n" + self.MARK.encode() + b"\r\nDOORS = 2\r\n")
        self.assertTrue(self.build(stage="detail")[0]["ok"])

    def test_without_a_marker_any_edit_needs_a_new_check(self):
        self.sp.write_text("W = 4.0\nDOORS = 2\n", encoding="utf-8")
        self.check()
        self.assertTrue(self.build(stage="detail")[0]["ok"])
        self.sp.write_text("W = 4.0\nDOORS = 3\n", encoding="utf-8")
        r, n = self.build(stage="detail")
        self.assertFalse(r["ok"])
        self.assertIn(self.MARK, r["error"])

    def test_force_reason_lets_it_through_and_says_so(self):
        pts = good_points()
        pts[0]["target"][0] += 0.2
        self.check(pts)
        r, n = self.build(stage="look", force_reason="  stilt base hidden behind the dune, unmeasurable  ")
        self.assertTrue(r["ok"], r)
        self.assertEqual(n, 1)
        g = r["gate"]
        self.assertTrue(g["forced"])
        self.assertEqual(g["reason"], "stilt base hidden behind the dune, unmeasurable")
        self.assertFalse(g["last_check"]["overall_pass"])
        self.assertIn("FAILED", g["why_blocked"])

    def test_blank_force_reason_does_not_force(self):
        r, n = self.build(stage="detail", force_reason="   ")
        self.assertFalse(r["ok"])

    def test_force_on_a_passing_gate_is_not_marked_forced(self):
        self.check()
        r, _ = self.build(stage="detail", force_reason="not needed")
        self.assertFalse(r["gate"]["forced"])

    def test_unreadable_check_file(self):
        dcc.check_file_path(str(self.sp)).write_text("{nope", encoding="utf-8")
        r, n = self.build(stage="detail")
        self.assertFalse(r["ok"])
        self.assertIn("unreadable", r["error"])


if __name__ == "__main__":
    unittest.main()
