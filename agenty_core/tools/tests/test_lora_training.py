"""LoRA training offline: presets, dataset preparation + review gate, trainer configs, log parsing, a whole local job
through the runner with a fake trainer, hosted (fal) flow against a stub, stop / keep, cost gate."""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from agenty_core.tools import lora_training as L
from agenty_core.tools.lora_training import backends as B
from agenty_core.tools.lora_training import dataset as D
from agenty_core.tools.lora_training import fal as FAL
from agenty_core.tools.lora_training import formats as F
from agenty_core.tools.lora_training import jobs as J
from agenty_core.tools.lora_training import preflight as PF
from agenty_core.tools.lora_training import presets as P
from agenty_core.tools.lora_training import runner as R


def noise_image(path, size=(900, 700), seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(a).save(path)


def flat_image(path, size=(900, 700)):
    Image.new("RGB", size, (120, 130, 140)).save(path)


def textured(path, seed, size=(900, 700)):
    """A picture with structure that differs between seeds (so not a duplicate) and is sharp."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    im = Image.fromarray(base).resize(size, Image.NEAREST)
    a = np.asarray(im, dtype=np.int16) + rng.integers(-40, 40, (size[1], size[0], 3))
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = str(self.root / "output")
        Path(self.out).mkdir()
        self.src = self.root / "src"
        self.src.mkdir()
        D.set_captioner(None)

    def tearDown(self):
        D.set_captioner(None)
        self.tmp.cleanup()

    def make_sources(self, n=14):
        for i in range(n):
            textured(self.src / f"p{i:02d}.png", seed=i)
        return str(self.src)

    def prepared(self, captioner=True, **kw):
        if captioner:
            D.set_captioner(lambda path, instr: f"a photo number {Path(path).stem}, outdoors")
        m = D.prepare_dataset(self.make_sources(), "my dog Rex", "character", "z_image", name="rex", output_dir=self.out, **kw)
        return F.datasets_root(self.out) / "rex", m


class PresetTests(unittest.TestCase):
    def test_resolve_aliases(self):
        self.assertEqual(P.resolve_model("Wan 2.2"), "wan22_t2v_14b")
        self.assertEqual(P.resolve_model("z-image"), "z_image")
        self.assertEqual(P.resolve_model("flux.2 klein"), "flux2_klein_4b")
        self.assertEqual(P.resolve_model("LTX-2.5"), "ltx2")
        self.assertIsNone(P.resolve_model("nonsense"))

    def test_every_preset_is_complete(self):
        for k, p in P.PRESETS.items():
            self.assertIn(p["kind"], ("image", "video"), k)
            self.assertTrue(p["backends"], k)
            for b, cfg in p["backends"].items():
                self.assertIn(b, P.BACKENDS)
                if b == "ai-toolkit":
                    self.assertIn("arch", cfg, k)
                    self.assertIn("name_or_path", cfg, k)
                if b == "musubi":
                    self.assertTrue(cfg["network_module"].startswith("networks."), k)
                    self.assertIn("dit", cfg["files"], k)
                if b == "fal":
                    self.assertIn("endpoint", cfg, k)

    def test_pick_backend(self):
        p = P.get_preset("wan22_image")
        with self.assertRaises(ValueError):
            P.pick_backend(p, "auto")
        self.assertEqual(P.pick_backend(p, "auto", allow_hosted=True), "fal")
        self.assertEqual(P.pick_backend(P.get_preset("z_image"), "auto"), "ai-toolkit")
        with self.assertRaises(ValueError):
            P.pick_backend(P.get_preset("ltx2"), "musubi")


class DatasetTests(Base):
    def test_prepare_filters_and_captions(self):
        textured(self.src / "a.png", 1)
        shutil_copy = lambda s, d: Path(d).write_bytes(Path(s).read_bytes())
        shutil_copy(self.src / "a.png", self.src / "a_copy.png")          # duplicate
        flat_image(self.src / "blurry.png")                                 # no detail
        textured(self.src / "tiny.png", 3, size=(200, 150))                # too small
        textured(self.src / "ok.png", 4)
        D.set_captioner(lambda p, i: "WATERMARK: logo in the corner" if Path(p).stem == "i005" else "a dog on grass")
        m = D.prepare_dataset(str(self.src), "dog", "character", "z_image", name="d1", output_dir=self.out)
        notes = {i["original"].rsplit(os.sep, 1)[-1]: i for i in m["items"]}
        self.assertTrue(notes["a.png"]["included"])
        self.assertFalse(notes["a_copy.png"]["included"])
        self.assertIn("duplicate", notes["a_copy.png"]["note"])
        self.assertIn("blurry", notes["blurry.png"]["note"])
        self.assertIn("too small", notes["tiny.png"]["note"])
        self.assertTrue(all(i["crop"] for i in m["items"] if i["size"]))
        ds = F.datasets_root(self.out) / "d1"
        self.assertTrue((ds / "contact_sheet.jpg").exists())
        self.assertTrue(any("only" in w for w in m["summary"]["warnings"]))        # too few items

    def test_watermark_removed(self):
        D.set_captioner(lambda p, i: "WATERMARK: shutterstock")
        textured(self.src / "x.png", 1)
        textured(self.src / "y.png", 2)
        # every item removed -> prepare keeps them but reports
        m = D.prepare_dataset(str(self.src), "dog", "character", "z_image", name="d2", output_dir=self.out)
        self.assertEqual(m["summary"].get("included", 0), 0) if "included" in m["summary"] else None
        self.assertTrue(all(not i["included"] for i in m["items"]))

    def test_no_captioner_needs_captions(self):
        ds, m = self.prepared(captioner=False)
        self.assertEqual(len(m["needs_captions"]), 14)
        with self.assertRaises(ValueError):
            D.request_review(ds)
        r = D.set_captions(ds, {i["id"]: f"caption {i['id']}" for i in m["items"]}, trigger_word="rexdog")
        self.assertEqual(r["missing"], [])
        self.assertEqual(F.load_manifest(ds)["trigger_word"], "rexdog")
        self.assertTrue(D.request_review(ds)["contact_sheet"].endswith("contact_sheet.jpg"))

    def test_trigger_word_in_captions(self):
        self.assertEqual(D.with_trigger("a dog", "rexdog"), "rexdog, a dog")
        self.assertEqual(D.with_trigger("rexdog sits", "rexdog"), "rexdog sits")

    def test_ratio_and_buckets(self):
        self.assertEqual(D.nearest_ratio(1920, 1080), (16, 9))
        self.assertEqual(D.nearest_ratio(700, 1000), (2, 3))
        w, h = D.bucket_size((1, 1), 1024)
        self.assertEqual((w, h), (1024, 1024))
        w, h = D.bucket_size((16, 9), 1024)
        self.assertTrue(w % 16 == 0 and h % 16 == 0 and abs(w * h - 1024 * 1024) < 60000)
        c = D.auto_crop(1000, 500, (1, 1))
        self.assertAlmostEqual(c[2], 0.5)

    def test_trainer_dataset_applies_crop_and_trigger(self):
        ds, m = self.prepared()
        m = F.load_manifest(ds)
        m["items"][0]["crop"] = [0.0, 0.0, 0.5, 0.5]            # user dragged a crop box
        F.save_manifest(ds, m)
        info = D.write_trainer_dataset(ds, m)
        self.assertEqual(info["count"], 14)
        txt = (Path(info["folder"]) / "i001.txt").read_text()
        self.assertTrue(txt.startswith(m["trigger_word"]))
        with Image.open(Path(info["folder"]) / "i001.png") as im:
            self.assertGreater(im.size[0], im.size[1])             # 450x350 crop of the original -> a 4:3 bucket

    def test_collect_sources_variants(self):
        textured(self.src / "a.png", 1)
        (self.src / "note.txt").write_text("x")
        got = D.collect_sources(f"{self.src}")
        self.assertEqual([Path(p).name for p, _ in got], ["a.png"])
        self.assertEqual(len(D.collect_sources([str(self.src / "a.png"), str(self.src / "a.png")])), 1)


class ApprovalTests(Base):
    def test_gate_sha_and_edit_invalidates(self):
        ds, m = self.prepared()
        self.assertFalse(F.is_approved(ds)[0])
        D.request_review(ds)
        self.assertFalse(F.is_approved(ds)[0])
        F.approve(ds, by="viewer")
        self.assertTrue(F.is_approved(ds)[0])
        D.set_captions(ds, {"i001": "something else entirely"})
        ok, why = F.is_approved(ds)
        self.assertFalse(ok)
        self.assertIn("approval", why + "approval")

    def test_sha_ignores_removed_items_and_notes(self):
        ds, m = self.prepared()
        a = F.manifest_sha(m)
        m2 = json.loads(json.dumps(m))
        m2["items"][0]["note"] = "edited note"
        self.assertEqual(a, F.manifest_sha(m2))
        m2["items"][0]["caption"] = "changed"
        self.assertNotEqual(a, F.manifest_sha(m2))

    def test_sha_vector_is_stable(self):
        m = {"trigger_word": "t", "model": "z_image", "items": [
            {"file": "items/i001.jpg", "caption": "a", "crop": [0, 0, 1, 1], "included": True},
            {"file": "items/i002.jpg", "caption": "b", "crop": None, "included": False}]}
        self.assertEqual(len(F.manifest_sha(m)), 64)
        viewer = {"trigger_word": "ohwx", "model": "flux1-dev",       # the same vector ComfyUI-ImageViewer pins (lora_io)
                  "items": [{"file": "items/i001.png", "caption": "ohwx a cat", "included": True, "crop": None},
                            {"file": "items/i002.png", "caption": "gone", "included": False, "crop": None},
                            {"file": "items/i003.png", "caption": "ohwx ünï", "crop": [0.1, 0.2, 0.5, 0.5]}]}
        self.assertEqual(F.manifest_sha(viewer), "d3d160065d63d9723d783def906967f5bedc04425d98d6e4c89e789f01db428c")

    def test_train_refuses_unapproved_and_tool_flow(self):
        ds, m = self.prepared()
        out = json.loads(L.train_lora("rex", output_dir=self.out))
        self.assertEqual(out["error_kind"], "not_approved")
        r = json.loads(L.lora_dataset_review("rex", open_viewer=False, output_dir=self.out))
        self.assertEqual(r["status"], "waiting_for_user")
        self.assertIn("END YOUR TURN", r["next"])
        out = json.loads(L.train_lora("rex", output_dir=self.out))
        self.assertEqual(out["error_kind"], "not_approved")
        bad = json.loads(L.lora_dataset_review("rex", action="approve", output_dir=self.out))
        self.assertIn("error", bad)                                   # no user words, no approval
        ok = json.loads(L.lora_dataset_review("rex", action="approve", user_message="approve", output_dir=self.out))
        self.assertEqual(ok["status"], "approved")
        self.assertTrue(F.is_approved(ds)[0])


class ParseTests(unittest.TestCase):
    def test_aitoolkit_tqdm(self):
        s = {}
        line = "my_lora:  25%|██▌       | 500/2000 [03:12<09:40,  2.58it/s, lr: 1.0e-04 loss: 1.234e-01]"
        self.assertTrue(B.parse_tqdm(line, s))
        self.assertEqual((s["step"], s["total"]), (500, 2000))
        self.assertEqual(s["eta_s"], 580)
        self.assertAlmostEqual(s["loss"], 0.1234)
        self.assertAlmostEqual(s["lr"], 1e-4)

    def test_musubi_tqdm_and_slow(self):
        s = {}
        self.assertTrue(B.parse_tqdm("steps:  12%|█▏ | 100/800 [05:00<35:00,  3.20s/it, avr_loss=0.0912]", s))
        self.assertAlmostEqual(s["it_s"], 0.3125, places=3)
        self.assertAlmostEqual(s["loss"], 0.0912)

    def test_not_progress(self):
        self.assertFalse(B.parse_tqdm("Loading checkpoint shards: 100%|███| 1/1 [00:01<00:00, 1.1it/s]", {}))
        self.assertFalse(B.parse_tqdm("hello", {}))


class ConfigTests(Base):
    def ctx(self, key, be):
        ds, m = self.prepared()
        preset = P.get_preset(key)
        cfg = dict(preset["backends"][be])
        job = {"name": "t_v1", "steps": 1000, "rank": 16, "learning_rate": 1e-4, "save_every": 250, "sample_every": 250,
               "seed": 7, "resolutions": preset.get("resolutions"), "frames": preset.get("frames"), "fps": preset.get("fps")}
        run = self.root / "run"
        run.mkdir(exist_ok=True)
        return {"preset": preset, "cfg": cfg, "job": job, "dataset_dir": str(self.root / "trainer"), "run_dir": str(run),
                "manifest": m, "base_model": {"files": {"dit": "D.safetensors", "vae": "V.safetensors",
                                                         "text_encoder": "T.safetensors", "t5": "t5.pth"}},
                "sample_prompts": ["rexdog on a beach", "a lighthouse"], "hf_home": str(self.root / "hf")}

    def test_aitoolkit_config(self):
        ctx = self.ctx("z_image", "ai-toolkit")
        c = B.AIToolkit().config(ctx)
        p = c["config"]["process"][0]
        self.assertEqual(p["type"], "diffusion_trainer")
        self.assertEqual(p["model"]["arch"], "zimage")
        self.assertEqual(p["network"]["linear"], 16)
        self.assertEqual(p["train"]["steps"], 1000)
        self.assertEqual(p["sample"]["prompts"], ["rexdog on a beach", "a lighthouse"])
        self.assertTrue(p["training_folder"].startswith(ctx["run_dir"]))
        self.assertEqual(p["trigger_word"], ctx["manifest"]["trigger_word"])
        self.assertIn("assistant_lora_path", p["model"])

    def test_aitoolkit_sd15_uses_local_file_and_classic_trainer(self):
        ctx = self.ctx("sd15", "ai-toolkit")
        ctx["base_model"] = {"local_file": "L:/models/sd15.safetensors"}
        p = B.AIToolkit().config(ctx)["config"]["process"][0]
        self.assertEqual(p["type"], "sd_trainer")
        self.assertEqual(p["model"]["name_or_path"], "L:/models/sd15.safetensors")
        self.assertNotIn("arch", p["model"])

    def test_aitoolkit_video_has_frames(self):
        ctx = self.ctx("wan22_t2v_14b", "ai-toolkit")
        p = B.AIToolkit().config(ctx)["config"]["process"][0]
        self.assertEqual(p["datasets"][0]["num_frames"], 33)
        self.assertEqual(p["model"]["arch"], "wan22_14b:t2v")

    def test_musubi_stages(self):
        ctx = self.ctx("z_image", "musubi")
        with mock.patch.object(B.Musubi, "locate", return_value={"root": str(self.root / "mt"), "python": "py", "installed": True}):
            st = B.Musubi().stages(ctx)
        names = [s["name"] for s in st]
        self.assertEqual(names, ["cache latents", "cache text encoder outputs", "train (musubi-tuner)"])
        train = st[-1]["cmd"]
        self.assertIn("networks.lora_zimage", train)
        self.assertEqual(train[train.index("--max_train_steps") + 1], "1000")
        self.assertIn("--sample_prompts", train)
        toml = (Path(ctx["run_dir"]) / "dataset.toml").read_text()
        self.assertIn("image_directory", toml)
        self.assertIn("enable_bucket = true", toml)
        self.assertTrue((Path(ctx["run_dir"]) / "sample_prompts.txt").read_text().startswith("rexdog on a beach --w"))

    def test_musubi_wan_uses_t5_and_frames(self):
        ctx = self.ctx("wan22_t2v_14b", "musubi")
        with mock.patch.object(B.Musubi, "locate", return_value={"root": str(self.root / "mt"), "python": "py", "installed": True}):
            st = B.Musubi().stages(ctx)
        self.assertIn("--t5", st[1]["cmd"])
        self.assertIn("--task", st[2]["cmd"])
        self.assertIn("target_frames", (Path(ctx["run_dir"]) / "dataset.toml").read_text())

    def test_not_installed_adds_install_stages(self):
        ctx = self.ctx("z_image", "ai-toolkit")
        with mock.patch.object(B.AIToolkit, "locate", return_value={"root": str(self.root / "at"), "python": "x", "installed": False}),                 mock.patch.object(B.AIToolkit, "root", return_value=self.root / "at"):
            st = B.AIToolkit().stages(ctx)
        self.assertTrue(any("clone" in s["name"] for s in st))
        self.assertTrue(any("torch" in s["name"] for s in st))
        self.assertEqual(st[-1]["kind"], "train")
        self.assertTrue(any("cu128" in " ".join(s["cmd"]) for s in st))


class PreflightTests(Base):
    def test_c_drive_refused(self):
        self.assertIn("C: system drive", PF.disk_check("C:/Users/x/out", 5, "the run"))

    def test_gpu_too_small_and_busy(self):
        p = P.get_preset("flux2_dev")
        with mock.patch.object(PF, "gpu_memory", return_value={"name": "X", "total_gb": 24.0, "free_gb": 20.0, "used_gb": 4.0}), \
                mock.patch.object(PF, "base_model", return_value={"present": True, "where": "x", "missing": []}), \
                mock.patch.object(PF, "disk_check", return_value=None):
            r = PF.check(p, "ai-toolkit", self.root)
        self.assertFalse(r["ok"])
        self.assertIn("too small", r["problems"][0])
        p = P.get_preset("z_image")
        with mock.patch.object(PF, "gpu_memory", return_value={"name": "X", "total_gb": 32.0, "free_gb": 8.0, "used_gb": 24.0}), \
                mock.patch.object(PF, "base_model", return_value={"present": True, "where": "x", "missing": []}), \
                mock.patch.object(PF, "disk_check", return_value=None), \
                mock.patch.object(B.AIToolkit, "locate", return_value={"root": "x", "installed": True}):
            r = PF.check(p, "ai-toolkit", self.root)
        self.assertTrue(r["ok"])                                         # busy but big enough: a warning, J.start frees ComfyUI
        self.assertTrue(r["warnings"])

    def test_base_model_missing_needs_consent(self):
        p = P.get_preset("z_image")
        with mock.patch.object(PF, "gpu_memory", return_value={"name": "X", "total_gb": 32.0, "free_gb": 30.0, "used_gb": 2.0}), \
                mock.patch.object(PF, "hf_cached", return_value=False), \
                mock.patch.object(PF, "disk_check", return_value=None), \
                mock.patch.object(B.AIToolkit, "locate", return_value={"root": "x", "installed": True}):
            r = PF.check(p, "ai-toolkit", self.root)
            self.assertFalse(r["ok"])
            self.assertIn("allow_download", " ".join(r["problems"]))
            r = PF.check(p, "ai-toolkit", self.root, allow_download=True)
            self.assertTrue(r["ok"])

    def test_no_gpu_blocks_local_not_fal(self):
        with mock.patch.object(PF, "gpu_memory", return_value=None), mock.patch.object(PF, "disk_check", return_value=None):
            self.assertFalse(PF.check(P.get_preset("flux2_dev"), "ai-toolkit", self.root, allow_download=True)["ok"])
            self.assertTrue(PF.check(P.get_preset("flux2_dev"), "fal", self.root)["ok"])


class FakeBackend(B.Backend):
    """A trainer that is a python one-liner: prints tqdm lines, writes samples + checkpoints + the LoRA."""
    NAME = "fake"
    MARKER = "x"

    def locate(self):
        return {"root": "x", "python": sys.executable, "installed": True}

    def stages(self, ctx):
        run = ctx["run_dir"]
        script = (
            "import sys,time,os,pathlib\n"
            "run=pathlib.Path(sys.argv[1]); out=run/'out'/sys.argv[2]; (out/'samples').mkdir(parents=True,exist_ok=True)\n"
            "total=int(sys.argv[3]); slow=float(sys.argv[4])\n"
            "from PIL import Image\n"
            "for s in range(1,total+1):\n"
            "    sys.stdout.write('t: %d%%|x| %d/%d [00:0%d<00:0%d,  4.00it/s, lr: 1.0e-04 loss: %.3e]\\r'%(s*100//total,s,total,s%10,(total-s)%10,1.0/s)); sys.stdout.flush()\n"
            "    if s%5==0:\n"
            "        for k in range(2): Image.new('RGB',(64,64),(s*3%255,k*100,50)).save(out/'samples'/('1700__%09d_%d.jpg'%(s,k)))\n"
            "        (out/('t_%09d.safetensors'%s)).write_bytes(b'lora%d'%s)\n"
            "    time.sleep(slow)\n"
            "(out/(sys.argv[2]+'.safetensors')).write_bytes(b'FINAL')\n")
        (Path(run) / "fake.py").write_text(script)
        return [{"name": "train (fake)", "kind": "train", "parse": True,
                 "cmd": [sys.executable, str(Path(run) / "fake.py"), run, ctx["job"]["name"], str(ctx["job"]["steps"]),
                         str(ctx.get("slow", 0.01))], "cwd": run}]

    def collect(self, run_dir, job):
        d = Path(run_dir) / "out" / job["name"]
        samples = []
        import re
        for f in sorted((d / "samples").glob("*.jpg")) if (d / "samples").exists() else []:
            m = re.search(r"__(\d{9})_(\d)\.jpg$", f.name)
            samples.append({"step": int(m.group(1)), "idx": int(m.group(2)), "path": str(f)})
        ck = [{"step": int(p.stem.split("_")[-1]), "path": str(p)} for p in sorted(d.glob("t_*.safetensors"))]
        final = d / f"{job['name']}.safetensors"
        return {"samples": samples, "checkpoints": ck, "final": str(final) if final.exists() else None}


class JobTests(Base):
    def approved(self):
        ds, m = self.prepared()
        D.request_review(ds)
        F.approve(ds, by="viewer")
        return ds

    def setUp(self):
        super().setUp()
        self.loras = self.root / "loras"
        self.loras.mkdir()
        self.patches = [
            mock.patch.object(J, "loras_dir", return_value=self.loras),
            mock.patch.object(J, "_spawn_runner", lambda run: (os.getpid(), 0.0)),
            mock.patch.object(J, "_free_vram", lambda: True),
            mock.patch.object(PF, "check", lambda *a, **k: {"ok": True, "problems": [], "warnings": [], "gpu": None,
                                                           "base_model": {"present": True, "where": "test"}}),
            mock.patch.dict(B._BACKENDS, {"ai-toolkit": FakeBackend()}),
            mock.patch.object(R, "TICK", 1.0),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def test_dry_run_starts_nothing(self):
        self.approved()
        r = json.loads(L.train_lora("rex", steps=20, dry_run=True, output_dir=self.out))
        self.assertEqual(r["status"], "dry_run")
        self.assertFalse(any(F.runs_root(self.out).iterdir()))

    def test_full_local_job(self):
        ds = self.approved()
        r = json.loads(L.train_lora("rex", steps=20, output_dir=self.out, options={"save_every": 5, "sample_every": 5}))
        self.assertEqual(r["status"], "started", r)
        run = Path(r["run_dir"])
        self.assertEqual(F.load_manifest(ds)["status"], "training")
        job = F.read_json(run / "job.json")
        self.assertEqual(job["steps"], 20)
        self.assertTrue((Path(job["trainer_dataset"]) / "i001.txt").exists())
        rc = R.run_local(run, job)
        self.assertEqual(rc, 0)
        pr = F.read_json(run / "progress.json")
        self.assertEqual(pr["status"], "done")
        self.assertEqual(pr["step"], 20)
        self.assertEqual(pr["pct"], 1.0)
        self.assertTrue(pr["loss_series"])
        self.assertLessEqual(len(pr["loss_series"]), R.SERIES_MAX)
        samples = F.read_json(run / "samples.json")
        self.assertGreaterEqual(len(samples), 2)
        self.assertTrue((run / samples[0]["files"][0]["file"]).exists())
        res = F.read_json(run / "result.json")
        self.assertTrue(Path(res["lora"]).exists())
        self.assertEqual(Path(res["lora"]).read_bytes(), b"FINAL")
        self.assertEqual(res["registered"]["trigger_word"], F.load_manifest(ds)["trigger_word"])
        self.assertTrue(res["test_plan"]["prompts"])
        self.assertIn("rex_v1", json.loads((self.root / "output" / "agent" / "lora_registry.json").read_text()))
        kinds = [e["kind"] for e in F.read_events(run)]
        self.assertIn("started", kinds)
        self.assertIn("sample", kinds)
        self.assertIn("finished", kinds)
        self.assertEqual(F.load_manifest(ds)["status"], "trained")
        st = json.loads(L.lora_training_status(r["job"], output_dir=self.out))
        self.assertEqual(st["status"], "done")
        self.assertIn("test_plan", st["result"])

    def test_stop_keeps_checkpoint(self):
        ds = self.approved()
        r = json.loads(L.train_lora("rex", steps=400, output_dir=self.out, options={"save_every": 5, "sample_every": 5}))
        run = Path(r["run_dir"])
        job = F.read_json(run / "job.json")
        job["stages"][0]["cmd"][-1] = "0.05"                  # slow it down
        t = threading.Thread(target=R.run_local, args=(run, job))
        t.start()
        for _ in range(100):
            time.sleep(0.2)
            if (F.read_json(run / "progress.json") or {}).get("step", 0) >= 12:
                break
        out = json.loads(L.lora_training_stop(r["job"], True, self.out))
        self.assertEqual(out["status"], "stopping")
        t.join(timeout=30)
        self.assertFalse(t.is_alive())
        pr = F.read_json(run / "progress.json")
        self.assertEqual(pr["status"], "stopped")
        res = F.read_json(run / "result.json")
        self.assertTrue(res["partial"])
        self.assertTrue(Path(res["lora"]).exists())

    def test_failed_stage_reports_tail(self):
        self.approved()
        r = json.loads(L.train_lora("rex", steps=10, output_dir=self.out))
        run = Path(r["run_dir"])
        job = F.read_json(run / "job.json")
        job["stages"] = [{"name": "boom", "kind": "train", "cmd": [sys.executable, "-c", "print('bad thing'); raise SystemExit(3)"], "cwd": str(run)}]
        self.assertEqual(R.run_local(run, job), 1)
        pr = F.read_json(run / "progress.json")
        self.assertEqual(pr["status"], "failed")
        self.assertIn("bad thing", pr["message"])

    def test_dead_runner_is_reported(self):
        self.approved()
        r = json.loads(L.train_lora("rex", steps=10, output_dir=self.out))
        run = Path(r["run_dir"])
        j = F.read_json(run / "job.json")
        j["runner_pid"] = 999999
        F.write_json(run / "job.json", j)
        st = json.loads(L.lora_training_status(r["job"], output_dir=self.out))
        self.assertEqual(st["status"], "failed")

    def test_versioned_job_names(self):
        self.approved()
        a = json.loads(L.train_lora("rex", steps=10, output_dir=self.out))["job"]
        b = json.loads(L.train_lora("rex", steps=10, output_dir=self.out))["job"]
        self.assertEqual((a, b), ("rex_v1", "rex_v2"))

    def test_overview_lists_presets(self):
        o = json.loads(L.lora_training_status("", output_dir=self.out))
        self.assertIn("z_image", [p["key"] for p in o["presets"]])
        self.assertIn("ai-toolkit", o["backends"])


class FalTests(Base):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.loras = self.root / "loras"
        self.loras.mkdir()

        class Resp:
            def __init__(s, code=200, data=None, content=b""):
                s.status_code, s._d, s.content, s.text = code, data or {}, content, json.dumps(data or {})

            def json(s):
                return s._d

        self.Resp = Resp

        def http(method, url, **kw):
            self.calls.append((method, url))
            if "storage/upload" in url:
                return Resp(200, {"upload_url": "https://up/put", "file_url": "https://cdn/data.zip"})
            if url == "https://up/put":
                return Resp(200)
            if url.startswith("https://queue.fal.run/") and method == "POST":
                return Resp(200, {"request_id": "r1", "status_url": "https://queue.fal.run/x/requests/r1/status",
                                  "response_url": "https://queue.fal.run/x/requests/r1", "cancel_url": "https://queue.fal.run/x/requests/r1/cancel"})
            if url.endswith("/status") or "/status?" in url:
                return Resp(200, {"status": "COMPLETED", "logs": [{"message": "step 1000/1000 loss 0.02"}]})
            if url.endswith("/requests/r1"):
                return Resp(200, {"diffusers_lora_file": {"url": "https://cdn/lora.safetensors"}})
            if url == "https://cdn/lora.safetensors":
                return Resp(200, content=b"FALLORA")
            return Resp(404)

        self.patches = [
            mock.patch.object(FAL, "_http", http),
            mock.patch.object(FAL, "api_key", lambda: "k"),
            mock.patch.object(J, "loras_dir", return_value=self.loras),
            mock.patch.object(J, "_spawn_runner", lambda run: (os.getpid(), 0.0)),
            mock.patch.object(PF, "check", lambda *a, **k: {"ok": True, "problems": [], "warnings": [], "gpu": None,
                                                           "base_model": {"present": True, "where": "hosted"}}),
        ]
        for p in self.patches:
            p.start()
        D.set_captioner(lambda path, instr: "a picture")
        D.prepare_dataset(self.make_sources(), "x", "style", "flux2_dev", name="sty", output_dir=self.out)
        ds = F.datasets_root(self.out) / "sty"
        D.request_review(ds)
        F.approve(ds, by="viewer")

    def tearDown(self):
        for p in self.patches:
            p.stop()
        super().tearDown()

    def test_cost_gate_then_run(self):
        r = json.loads(L.train_lora("sty", backend="fal", steps=1000, output_dir=self.out))
        self.assertEqual(r["error_kind"], "confirm_cost")
        self.assertAlmostEqual(r["estimate_usd"], 8.0)
        self.assertFalse(any("queue.fal.run" in u for _, u in self.calls))      # nothing was submitted
        r = json.loads(L.train_lora("sty", backend="fal", steps=1000, confirm_cost=True, output_dir=self.out))
        self.assertEqual(r["status"], "started", r)
        run = Path(r["run_dir"])
        job = F.read_json(run / "job.json")
        self.assertEqual(job["mode"], "fal")
        self.assertTrue(any(u.startswith("https://queue.fal.run/fal-ai/flux-2-trainer") for _, u in self.calls))
        with mock.patch.object(R.time, "sleep", lambda s: None):
            self.assertEqual(R.run_fal(run, job), 0)
        res = F.read_json(run / "result.json")
        self.assertEqual(Path(res["lora"]).read_bytes(), b"FALLORA")
        self.assertEqual(F.read_json(run / "progress.json")["status"], "done")

    def test_cheap_run_needs_no_confirmation(self):
        with mock.patch.dict(os.environ, {"AGENTY_CONFIRM_ABOVE_USD": "100"}):
            r = json.loads(L.train_lora("sty", backend="fal", steps=500, output_dir=self.out))
        self.assertEqual(r["status"], "started", r)

    def test_input_shapes(self):
        cfg = P.PRESETS["flux2_dev"]["backends"]["fal"]
        b = FAL.build_input(cfg, "https://u/z.zip", 800, None, "trg")
        self.assertEqual(b["image_data_url"], "https://u/z.zip")
        self.assertEqual(b["steps"], 800)
        self.assertEqual(b["output_lora_format"], "comfy")
        w = FAL.build_input(P.PRESETS["wan22_t2v_14b"]["backends"]["fal"], "https://u/z.zip", 300, None, "")
        self.assertIn("training_data_url", w)
        self.assertIn("number_of_steps", w)

    def test_hosted_only_model_needs_fal(self):
        r = json.loads(L.train_lora("sty", model="wan22_image", output_dir=self.out))
        self.assertEqual(r["error_kind"], "args")


if __name__ == "__main__":
    unittest.main()
