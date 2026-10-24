"""splat_formats: SPZ <-> PLY on a tiny synthetic splat, plus a hand-packed SPZ read independently of the writer."""

import gzip
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agenty_core.tools import splat_formats as sf


def synthetic(n=40, k=3, seed=1):
    rng = np.random.default_rng(seed)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(3 * k)] + \
            ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    a = np.zeros((n, len(props)), dtype=np.float32)
    ix = {p: i for i, p in enumerate(props)}
    a[:, :3] = rng.uniform(-8, 8, (n, 3))
    a[:, 3:6] = rng.uniform(-1.5, 1.5, (n, 3))
    for i in range(3 * k):
        a[:, ix[f"f_rest_{i}"]] = rng.uniform(-0.6, 0.6, n)
    a[:, ix["opacity"]] = rng.uniform(-3, 3, n)
    a[:, 
      [ix["scale_0"], ix["scale_1"], ix["scale_2"]]] = rng.uniform(-8, -1, (n, 3))
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a[:, [ix["rot_0"], ix["rot_1"], ix["rot_2"], ix["rot_3"]]] = q
    return props, a


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def check(self, k):
        props, a = synthetic(k=k)
        spz = self.dir / f"s{k}.spz"
        head = sf.write_spz(spz, props, a)
        self.assertEqual(head["count"], 40)
        self.assertEqual(sf.spz_header(spz)["sh_degree"], {0: 0, 3: 1, 8: 2, 15: 3}[k])
        props2, b = sf.read_spz(spz)
        self.assertEqual(props2, props)
        ix = {p: i for i, p in enumerate(props)}
        self.assertLess(np.abs(a[:, :3] - b[:, :3]).max(), 1 / 4096)                  # 12 fractional bits
        self.assertLess(np.abs(a[:, 3:6] - b[:, 3:6]).max(), 0.02)                    # colour byte / 0.15
        sig = lambda v: 1 / (1 + np.exp(-v))
        self.assertLess(np.abs(sig(a[:, ix["opacity"]]) - sig(b[:, ix["opacity"]])).max(), 1 / 255)
        self.assertLess(np.abs(a[:, ix["scale_0"]:ix["scale_0"] + 3] - b[:, ix["scale_0"]:ix["scale_0"] + 3]).max(), 1 / 32 + 1e-4)
        qa = a[:, ix["rot_0"]:ix["rot_0"] + 4]
        qb = b[:, ix["rot_0"]:ix["rot_0"] + 4]
        self.assertGreater(np.abs((qa * qb).sum(1)).min(), 0.9995)                    # same rotation (q or -q)
        if k:
            r = [f"f_rest_{i}" for i in range(3 * k)]
            cols = [ix[p] for p in r]
            self.assertLess(np.abs(a[:, cols] - b[:, cols]).max(), 1 / 128)

    def test_round_trip_all_sh_degrees(self):
        for k in (0, 3, 8, 15):
            with self.subTest(sh=k):
                self.check(k)

    def test_spz_to_ply_and_back(self):
        props, a = synthetic(k=3)
        spz = self.dir / "a.spz"
        sf.write_spz(spz, props, a)
        ply = self.dir / "a.ply"
        info = sf.spz_to_ply(spz, ply)
        self.assertEqual(info, {"count": 40, "sh_degree": 3})
        p2, c = sf.read_ply_any(ply)
        self.assertEqual(p2, props)
        spz2 = self.dir / "b.spz"
        sf.ply_to_spz(ply, spz2)
        _, d = sf.read_spz(spz2)
        self.assertLess(np.abs(c - d)[:, :3].max(), 1 / 4096 + 1e-6)

    def test_ply_reader_takes_mixed_types(self):
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("flag", "u1"), ("opacity", "<f8")])
        rec = np.zeros(3, dtype=dt)
        rec["x"], rec["flag"], rec["opacity"] = [1, 2, 3], [7, 8, 9], [0.5, 0.25, 0.125]
        head = ("ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar flag\nproperty double opacity\nend_header\n")
        f = self.dir / "m.ply"
        f.write_bytes(head.encode() + rec.tobytes())
        names, a = sf.read_ply_any(f)
        self.assertEqual(names, ["x", "y", "z", "flag", "opacity"])
        self.assertEqual(a[:, 3].tolist(), [7, 8, 9])
        self.assertEqual(a[:, 4].tolist(), [0.5, 0.25, 0.125])

    def test_hand_packed_v3_point(self):
        """One point packed by hand from the SPZ spec, not by write_spz."""
        frac = 12
        x, y, z = 1.5, -2.25, 0.125
        pos = b"".join(int(round(v * (1 << frac))).to_bytes(3, "little", signed=True) for v in (x, y, z))
        alpha = bytes([255])
        color = bytes([204, 128, 0])                                   # (b/255-0.5)/0.15 = 2.0, ~0.0033, -3.33
        scale = bytes([16 * 7, 16 * 8, 16 * 9])                         # log scales -3, -2, -1
        # rotation: identity (x=y=z=0, w=1): largest = 3 (w); the other three are zero magnitude
        comp = 3 << 30
        rot = comp.to_bytes(4, "little")
        data = struct.pack("<IIIBBBB", sf.SPZ_MAGIC, 3, 1, 0, frac, 0, 0) + pos + alpha + color + scale + rot
        f = self.dir / "h.spz"
        f.write_bytes(gzip.compress(data))
        props, a = sf.read_spz(f)
        r = dict(zip(props, a[0]))
        self.assertAlmostEqual(r["x"], x, 4)
        self.assertAlmostEqual(r["y"], y, 4)
        self.assertAlmostEqual(r["z"], z, 4)
        self.assertAlmostEqual(r["f_dc_0"], 2.0, 3)
        self.assertAlmostEqual(r["f_dc_2"], -0.5 / 0.15, 3)
        self.assertGreater(r["opacity"], 10)
        self.assertEqual([r["scale_0"], r["scale_1"], r["scale_2"]], [-3.0, -2.0, -1.0])
        self.assertEqual([round(float(r[f"rot_{i}"]), 4) for i in range(4)], [1.0, 0.0, 0.0, 0.0])

    def test_hand_packed_v2_rotation(self):
        data = struct.pack("<IIIBBBB", sf.SPZ_MAGIC, 2, 1, 0, 12, 0, 0) + bytes(9) + bytes([128]) + bytes(3) + bytes(3) + \
            bytes([255, 128, 128])
        f = self.dir / "v2.spz"
        f.write_bytes(gzip.compress(data))
        props, a = sf.read_spz(f)
        r = dict(zip(props, a[0]))
        self.assertAlmostEqual(r["rot_1"], 1.0, 4)            # x = 255/127.5 - 1
        self.assertAlmostEqual(r["rot_0"], 0.0, 2)            # w from the unit norm

    def test_bad_files(self):
        f = self.dir / "bad.spz"
        f.write_bytes(gzip.compress(b"not a splat at all, but long enough"))
        with self.assertRaises(sf.SplatFormatError):
            sf.read_spz(f)
        v4 = self.dir / "v4.spz"
        v4.write_bytes(struct.pack("<II", sf.SPZ_MAGIC, 4) + b"\0" * 20)
        with self.assertRaisesRegex(sf.SplatFormatError, "version 4"):
            sf.read_spz(v4)
        with self.assertRaises(sf.SplatFormatError):
            sf.write_spz(self.dir / "x.spz", ["x", "y", "z"], np.zeros((1, 3)))

    def test_metric(self):
        props, a = synthetic(n=5, k=0)
        b = sf.apply_metric(props, a, 2.0, 0.5)
        ix = {p: i for i, p in enumerate(props)}
        np.testing.assert_allclose(b[:, ix["x"]], a[:, ix["x"]] * 2, rtol=1e-6)
        np.testing.assert_allclose(b[:, ix["y"]], a[:, ix["y"]] * 2 - 0.5, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b[:, ix["scale_1"]], a[:, ix["scale_1"]] + math.log(2), rtol=1e-6)
        np.testing.assert_array_equal(b[:, ix["opacity"]], a[:, ix["opacity"]])


if __name__ == "__main__":
    unittest.main()
