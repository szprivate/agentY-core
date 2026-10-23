"""dcc_build + dcc_silhouette_check against the REAL apps.

Per app, batch mode (a fresh blender -b / mayapy / hython): a 2 m cube at a known spot and a camera at a known pose
are built by script, the scene is saved, and the silhouette check must return the analytically expected frame
fractions of the cube's eight corners within 0.005. Skipped when the app is not installed.

One more test talks to the Blender session a user has open (the MCP add-on socket on 9876): it builds into a
temporary collection, checks, and deletes it again. Skipped when the port is closed.
"""

import json
import tempfile
import unittest
from pathlib import Path

from agenty_core.tools import dcc

TOL = 0.005
FOCAL, FILM_W, RES = 50.0, 36.0, (1000, 500)
FILM_H = FILM_W * RES[1] / RES[0]


def expected(right, up, depth, shift=(0.0, 0.0)):
    """Plain pinhole maths, written out here on purpose: x from the left, y from the TOP."""
    x = 0.5 + FOCAL / FILM_W * right / depth - shift[0]
    y_bottom = 0.5 + FOCAL / FILM_H * up / depth - shift[1]
    return [x, 1.0 - y_bottom]


def corners_points(world_corners, expect):
    return [{"name": f"c{i}", "world": w, "target": expect(w)} for i, w in enumerate(world_corners)]


def assert_all_pass(test, res):
    test.assertTrue(res["ok"], res)
    test.assertIn("cross_check", res)
    test.assertTrue(res["cross_check"]["agree"], res["cross_check"])
    test.assertEqual(len(res["points"]), 8)
    for p in res["points"]:
        test.assertTrue(p["in_front"])
        test.assertLess(abs(p["dx"]), TOL, p)
        test.assertLess(abs(p["dy"]), TOL, p)
        test.assertTrue(p["pass"], p)
    test.assertTrue(res["overall_pass"])


class Base(unittest.TestCase):
    app = ""

    def setUp(self):
        if not dcc.find_app(self.app):
            self.skipTest(f"{self.app} is not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def script(self, text, name="build.py"):
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def build(self, text, **kw):
        res = json.loads(dcc.dcc_build(self.script(text), app=self.app, mode="batch",
                                       save_to=str(self.dir / f"scene{dcc._EXT[self.app]}"), **kw))
        self.assertTrue(res["ok"], res)
        self.assertTrue(Path(res["saved"]).is_file(), res)
        return res

    def check(self, res, points, **kw):
        return json.loads(dcc.dcc_silhouette_check(points, app=self.app, mode="batch", scene=res["saved"], **kw))


# Cube 2 m centred at CUBE; camera at CAM looking along the world axis noted per app.
class TestBlenderBatch(Base):
    app = "blender"
    CUBE, CAM = (1.0, 6.0, 2.0), (0.0, -4.0, 1.5)           # Z up; the camera looks along +Y
    SCRIPT = """
import bpy, math
for c in list(bpy.data.collections):
    if c.name == "dcc_test": bpy.data.collections.remove(c)
col = bpy.data.collections.new("dcc_test"); bpy.context.scene.collection.children.link(col)
bpy.ops.mesh.primitive_cube_add(size=2, location=%(cube)r)
o = bpy.context.active_object; o.name = "Cube"
for c in o.users_collection: c.objects.unlink(o)
col.objects.link(o)
cd = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cd); col.objects.link(cam)
cam.location = %(cam)r; cam.rotation_euler = (math.radians(90), 0, 0)
cd.lens = 50; cd.sensor_width = 36; cd.sensor_fit = "AUTO"; cd.shift_x = %(sx)r; cd.shift_y = %(sy)r
s = bpy.context.scene; s.camera = cam; s.render.resolution_x = 1000; s.render.resolution_y = 500
print("built", len(col.objects))
"""

    def corners(self):
        return [[self.CUBE[0] + dx, self.CUBE[1] + dy, self.CUBE[2] + dz]
                for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]

    def exp(self, w, shift=(0.0, 0.0)):
        return expected(w[0] - self.CAM[0], w[2] - self.CAM[2], w[1] - self.CAM[1], shift)

    def test_cube_corners(self):
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM, "sx": 0.0, "sy": 0.0})
        self.assertEqual(res["stdout_tail"].strip(), "built 2")
        self.assertEqual(res["objects"]["count"], 2)
        assert_all_pass(self, self.check(res, corners_points(self.corners(), self.exp)))

    def test_lens_shift(self):
        # landscape: shift_y is a fraction of the frame WIDTH, so 0.1 of the width is 0.2 of the height
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM, "sx": 0.05, "sy": 0.05})
        assert_all_pass(self, self.check(res, corners_points(self.corners(), lambda w: self.exp(w, (0.05, 0.10)))))

    def test_script_error_names_the_scripts_line(self):
        res = json.loads(dcc.dcc_build(self.script("import bpy\nx = 1\nraise RuntimeError('bad dimension')\n"),
                                       app="blender", mode="batch", save_to=str(self.dir / "e.blend")))
        self.assertFalse(res["ok"])
        self.assertIn("RuntimeError: bad dimension", res["error"])
        self.assertIn("script line 3", res["error"])
        self.assertEqual(res["saved"], "")

    def test_fast_render_writes_a_picture(self):
        out = self.dir / "r.png"
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM, "sx": 0.0, "sy": 0.0},
                         render_to=str(out), resolution="200x100")
        self.assertTrue(res["render"]["exists"], res["render"])
        self.assertEqual(res["render"]["resolution"], [200, 100])
        self.assertTrue(out.stat().st_size > 100)

    def test_a_wrong_point_fails_with_a_hint(self):
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM, "sx": 0.0, "sy": 0.0})
        w = self.corners()[0]
        t = self.exp(w)
        out = self.check(res, [{"name": "corner", "world": w, "target": [t[0], t[1] - 0.1]}])
        self.assertFalse(out["overall_pass"])
        self.assertIn("too low", out["points"][0]["hint"])


class TestMayaBatch(Base):
    app = "maya"
    CUBE, CAM = (1.0, 2.0, 0.0), (0.0, 1.5, 10.0)            # Y up; the camera looks along -Z
    SCRIPT = """
import maya.cmds as cmds
for n in ("dcc_grp",):
    if cmds.objExists(n): cmds.delete(n)
cube = cmds.polyCube(w=2, h=2, d=2, name="Cube")[0]
cmds.move(*%(cube)r, cube)
cam, shape = cmds.camera(name="Cam")
cmds.move(*%(cam)r, cam)
cmds.setAttr(shape + ".focalLength", 50)
cmds.setAttr(shape + ".horizontalFilmAperture", 36 / 25.4)
cmds.setAttr(shape + ".verticalFilmAperture", 18 / 25.4)
cmds.setAttr(shape + ".horizontalFilmOffset", %(hfo)r)
cmds.setAttr(shape + ".verticalFilmOffset", %(vfo)r)
cmds.setAttr(shape + ".renderable", 1)
# set only width and height, as a script would: deviceAspectRatio is NOT updated by this (the tool must not rely on it)
cmds.setAttr("defaultResolution.width", 1000); cmds.setAttr("defaultResolution.height", 500)
cmds.group(cube, cam, name="dcc_grp")
print("built maya")
"""

    def corners(self):
        return [[self.CUBE[0] + dx, self.CUBE[1] + dy, self.CUBE[2] + dz]
                for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]

    def exp(self, w, shift=(0.0, 0.0)):
        return expected(w[0] - self.CAM[0], w[1] - self.CAM[1], self.CAM[2] - w[2], shift)

    def test_cube_corners(self):
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM, "hfo": 0.0, "vfo": 0.0})
        self.assertEqual(res["stdout_tail"].strip(), "built maya")
        self.assertEqual(res["objects"]["top_level"], ["dcc_grp"])
        out = self.check(res, corners_points(self.corners(), self.exp))
        assert_all_pass(self, out)

    def test_film_offset(self):
        # 0.1 inch of film offset on a 36 mm (1.417 in) wide frame: 0.0706 of the width, 0.1 / 0.7087 of the height
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM, "hfo": 0.1, "vfo": 0.05})
        sh = (0.1 * 25.4 / FILM_W, 0.05 * 25.4 / FILM_H)
        assert_all_pass(self, self.check(res, corners_points(self.corners(), lambda w: self.exp(w, sh))))

    def test_script_error(self):
        res = json.loads(dcc.dcc_build(self.script("import maya.cmds as cmds\ncmds.polyCube()\nundefined_name\n"),
                                       app="maya", mode="batch", save_to=str(self.dir / "e.ma")))
        self.assertFalse(res["ok"])
        self.assertIn("NameError", res["error"])
        self.assertIn("script line 3", res["error"])

    def test_fast_render_writes_a_picture(self):
        out = self.dir / "r.png"
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM, "hfo": 0.0, "vfo": 0.0},
                         render_to=str(out), resolution="200x100")
        self.assertTrue(res["render"].get("exists"), res["render"])


class TestHoudiniBatch(Base):
    app = "houdini"
    CUBE, CAM = (1.0, 2.0, 0.0), (0.0, 1.5, 10.0)            # Y up; the camera looks along -Z
    SCRIPT = """
import hou
obj = hou.node("/obj")
for n in obj.children(): n.destroy()
geo = obj.createNode("geo", "Cube")
for c in geo.children(): c.destroy()
box = geo.createNode("box"); box.parmTuple("size").set((2, 2, 2)); box.parmTuple("t").set(%(cube)r)
cam = obj.createNode("cam", "Cam"); cam.parmTuple("t").set(%(cam)r)
cam.parm("focal").set(50); cam.parm("aperture").set(36); cam.parm("resx").set(1000); cam.parm("resy").set(500)
print("built houdini")
"""

    def corners(self):
        return [[self.CUBE[0] + dx, self.CUBE[1] + dy, self.CUBE[2] + dz]
                for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]

    def exp(self, w):
        return expected(w[0] - self.CAM[0], w[1] - self.CAM[1], self.CAM[2] - w[2])

    def test_cube_corners(self):
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM})
        self.assertEqual(res["stdout_tail"].strip(), "built houdini")
        self.assertEqual(sorted(res["objects"]["top_level"]), ["Cam", "Cube"])
        assert_all_pass(self, self.check(res, corners_points(self.corners(), self.exp)))

    def test_named_camera(self):
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM})
        assert_all_pass(self, self.check(res, corners_points(self.corners(), self.exp), camera="/obj/Cam"))

    def test_script_error(self):
        res = json.loads(dcc.dcc_build(self.script("import hou\nhou.node('/obj')\nundefined_name\n"),
                                       app="houdini", mode="batch", save_to=str(self.dir / "e.hip")))
        self.assertFalse(res["ok"])
        self.assertIn("script line 3", res["error"])

    def test_fast_render_writes_a_picture(self):
        out = self.dir / "r.png"
        res = self.build(self.SCRIPT % {"cube": self.CUBE, "cam": self.CAM}, render_to=str(out), resolution="200x100")
        self.assertTrue(res["render"].get("exists"), res["render"])


class TestBlenderLive(unittest.TestCase):
    """Against the Blender the user has open. Builds into a temporary collection and deletes it again."""

    COL = "agenty_dcc_test_tmp"
    CUBE, CAM = (1.0, 6.0, 2.0), (0.0, -4.0, 1.5)
    BUILD = """
import bpy, math
COL = %(col)r
def _clear():
    c = bpy.data.collections.get(COL)
    if c:
        for o in list(c.objects):
            data = o.data
            bpy.data.objects.remove(o, do_unlink=True)
            if data is not None and data.users == 0:
                (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.cameras).remove(data)
        bpy.data.collections.remove(c)
_clear()
col = bpy.data.collections.new(COL); bpy.context.scene.collection.children.link(col)
me = bpy.data.meshes.new(COL + "_mesh")
import bmesh
bm = bmesh.new(); bmesh.ops.create_cube(bm, size=2.0); bm.to_mesh(me); bm.free()
ob = bpy.data.objects.new(COL + "_cube", me); ob.location = %(cube)r; col.objects.link(ob)
cd = bpy.data.cameras.new(COL + "_cam"); cam = bpy.data.objects.new(COL + "_cam", cd); col.objects.link(cam)
cam.location = %(cam)r; cam.rotation_euler = (math.radians(90), 0, 0)
cd.lens = 50; cd.sensor_width = 36; cd.sensor_fit = "HORIZONTAL"
print("live built")
"""
    CLEAN = """
import bpy
COL = %(col)r
c = bpy.data.collections.get(COL)
if c:
    for o in list(c.objects):
        data = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        if data is not None and data.users == 0:
            (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.cameras).remove(data)
    bpy.data.collections.remove(c)
print("cleaned")
"""

    def setUp(self):
        if not dcc.port_open("blender"):
            self.skipTest("no Blender listening on the MCP add-on socket")
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        if hasattr(self, "tmp"):
            clean = self.d / "clean.py"
            clean.write_text(self.CLEAN % {"col": self.COL}, encoding="utf-8")
            json.loads(dcc.dcc_build(str(clean), app="blender", mode="live"))
            self.tmp.cleanup()

    def test_build_check_and_clean_up(self):
        b = self.d / "build.py"
        b.write_text(self.BUILD % {"col": self.COL, "cube": self.CUBE, "cam": self.CAM}, encoding="utf-8")
        res = json.loads(dcc.dcc_build(str(b), app="blender", mode="live"))
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["stdout_tail"].strip(), "live built")
        self.assertIn(self.COL, res["objects"]["collections"])
        corners = [[self.CUBE[0] + dx, self.CUBE[1] + dy, self.CUBE[2] + dz]
                   for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
        cam_name = self.COL + "_cam"
        # the user's own scene camera and resolution are left alone: ask for ours by name, read the frame back
        first = json.loads(dcc.dcc_silhouette_check(
            [{"name": "c", "world": corners[0], "target": [0.5, 0.5]}], app="blender", mode="live", camera=cam_name))
        self.assertTrue(first["ok"], first)
        res_x, res_y = first["camera"]["resolution"]
        film_h = FILM_W * res_y / res_x                        # HORIZONTAL fit: frame height follows the render aspect

        def exp(w):
            depth = w[1] - self.CAM[1]
            return [0.5 + FOCAL / FILM_W * (w[0] - self.CAM[0]) / depth,
                    1.0 - (0.5 + FOCAL / film_h * (w[2] - self.CAM[2]) / depth)]
        out = json.loads(dcc.dcc_silhouette_check(corners_points(corners, exp), app="blender", mode="live",
                                                  camera=cam_name))
        assert_all_pass(self, out)

    def test_cleanup_leaves_no_trace(self):
        b = self.d / "build.py"
        b.write_text(self.BUILD % {"col": self.COL, "cube": self.CUBE, "cam": self.CAM}, encoding="utf-8")
        json.loads(dcc.dcc_build(str(b), app="blender", mode="live"))
        c = self.d / "clean.py"
        c.write_text(self.CLEAN % {"col": self.COL}, encoding="utf-8")
        res = json.loads(dcc.dcc_build(str(c), app="blender", mode="live"))
        self.assertTrue(res["ok"], res)
        self.assertNotIn(self.COL, res["objects"]["collections"])
        self.assertFalse([n for n in res["objects"]["top_level"] if n.startswith(self.COL)])


if __name__ == "__main__":
    unittest.main()
