"""Recreate shots with changes: only the requested inputs change, the seed is kept, the
result is versioned beside the original, and no tool answer ever carries the graph.

Real templates from the corpus (API format) are the graphs; ComfyUI is mocked.
"""
from agenty_core.utils.provenance import sidecar_path as pv_sidecar
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agenty_core import partner_credits
from agenty_core.paths import corpus_root
from agenty_core.tools import regenerate as rg
from agenty_core.utils import provenance as pv
from agenty_core.utils.tests.test_provenance import FAKE_KEY

TEMPLATES = Path(corpus_root()) / "comfyui_workflow_templates_custom" / "templates"
SENTINEL = "ZZ-SENTINEL-NOT-A-PARAM-ZZ"
LONG = ("a cozy beach house at dusk, warm light, long shadows, wooden deck, lanterns glowing softly, "
        "calm sea, volumetric haze, film grain, shallow depth of field, ") * 6


def template(name: str) -> dict:
    return json.loads((TEMPLATES / f"{name}.json").read_text(encoding="utf-8"))


def diff_inputs(a: dict, b: dict) -> set:
    """{(node, input)} whose value differs between two graphs."""
    out = set()
    for nid in set(a) | set(b):
        ai = (a.get(nid) or {}).get("inputs") or {}
        bi = (b.get(nid) or {}).get("inputs") or {}
        for k in set(ai) | set(bi):
            if ai.get(k) != bi.get(k):
                out.add((nid, k))
    return out


def tok(text: str) -> int:
    return len(text) // 4


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.posts = []
        client = mock.Mock()
        client.post.side_effect = lambda *a, **k: self.posts.append((a, k)) or {"ok": True}
        client.get.side_effect = RuntimeError("no ComfyUI in tests")
        for target in ("agenty_core.utils.comfyui_client.get_client",):
            p = mock.patch(target, return_value=client)
            p.start()
            self.addCleanup(p.stop)
        self.client = client
        old = partner_credits._info_provider
        partner_credits._info_provider = lambda ct: None
        partner_credits.clear_caches()
        self.addCleanup(setattr, partner_credits, "_info_provider", old)
        self.addCleanup(rg.set_runner, None)
        p = mock.patch.object(rg, "_workflow_dir", return_value=self.tmp / "wf")
        (self.tmp / "wf").mkdir()
        p.start()
        self.addCleanup(p.stop)
        p = mock.patch("agenty_core.tools.comfyui.validate_workflow", return_value='{"valid": true}')
        self.validate = p.start()
        self.addCleanup(p.stop)

    def shot(self, graph, rel="a/shot_03.mp4", node="", prompt_id="pidX", add_sentinel=True) -> str:
        g = copy.deepcopy(graph)
        if add_sentinel:
            g["999"] = {"class_type": "Note", "inputs": {"hidden_blob": SENTINEL}}
        f = self.tmp / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"orig")
        pv.write(f, g, prompt_id=prompt_id, output_node=node, output_index=0, workflow_file="w.json")
        return str(f)

    def runner_that_writes(self, name="out_00001_.mp4", log=None):
        def run(path, *, confirm_cost=False):
            if log is not None:
                log.append((path, confirm_cost))
            out = self.tmp / "comfy_out" / f"{len(log or [])}_{name}"
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(b"new")
            return {"status": "done", "outputs": [str(out)], "prompt_id": f"newpid{len(log or [])}"}
        return run


class Params(Base):
    def test_a_controlnet_between_sampler_and_prompts_keeps_positive_and_negative_apart(self):
        # found on a real output (street_canny_00001_.png): KSampler.positive / .negative are
        # slots 0 / 1 of ControlNetApplyAdvanced, which takes both prompts
        g = {"2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red fox", "clip": ["1", 1]}},
             "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["1", 1]}},
             "5": {"class_type": "KSampler", "inputs": {"seed": 1, "positive": ["11", 0], "negative": ["11", 1]}},
             "11": {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["2", 0], "negative": ["3", 0],
                                                                         "strength": 0.8}}}
        idx = {e["id"]: e["label"] for e in rg.param_index(g)}
        self.assertEqual((idx["2.text"], idx["3.text"]), ("positive", "negative"))
        self.assertEqual(rg._resolve(g, rg.param_index(g), "positive")["id"], "2.text")
        # a chain of two: the role is kept through every pass-through node
        g["12"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["11", 0], "negative": ["11", 1]}}
        g["5"]["inputs"].update(positive=["12", 0], negative=["12", 1])
        idx = {e["id"]: e["label"] for e in rg.param_index(g)}
        self.assertEqual((idx["2.text"], idx["3.text"]), ("positive", "negative"))

    def test_flux_t2i_template(self):
        g = template("image_flux2_klein_text_to_image")
        idx = {e["id"]: e for e in rg.param_index(g)}
        self.assertEqual(idx["76.value"]["label"], "positive")      # the primitive behind the encode node
        self.assertIn("75:73.noise_seed", idx)
        self.assertIn("75:62.steps", idx)
        self.assertIn("75:63.cfg", idx)
        self.assertEqual(idx["75:68.value"]["label"], "width")
        self.assertIn("75:70.unet_name", idx)
        self.assertNotIn("75:74.text", idx)                          # a link, not a param
        f = self.shot(g, "t2i/img_00001_.png", node="9")
        out = json.loads(rg.recreatable_params(f))
        self.assertEqual(out["status"], "ok")
        self.assertIn("76.value (positive)", out["params"])
        self.assertIn("75:73.noise_seed", out["params"])
        self.assertIn("75:70.unet_name flux-2-klein", out["params"])
        self.assertLessEqual(tok(rg._j(out)), 400)

    def test_wan_i2v_and_ltx_templates(self):
        for name, must in (("api_wan2_6_i2v", ["41.prompt (positive)", "41.negative_prompt (negative)",
                                                "41.seed", "41.duration", "42.image"]),
                           ("api_ltxv_image_to_video", ["36.prompt (positive)", "36.duration", "36.fps",
                                                         "47.image"])):
            f = self.shot(template(name), f"{name}/clip.mp4")
            raw = rg.recreatable_params(f)
            out = json.loads(raw)
            self.assertEqual(out["status"], "ok", name)
            for m in must:
                self.assertIn(m, out["params"], (name, out["params"]))
            self.assertLessEqual(tok(raw), 400, name)
            self.assertNotIn(SENTINEL, raw)

    def test_a_synthetic_sdxl_has_the_sampler_line_and_loras(self):
        from agenty_core.utils.tests.test_provenance import GRAPH
        g = copy.deepcopy(GRAPH)
        g["12"] = {"class_type": "LoraLoader", "inputs": {"lora_name": "detail.safetensors", "strength_model": 0.8,
                                                           "strength_clip": 1.0, "model": ["4", 0], "clip": ["4", 1]}}
        f = self.shot(g, "x/s.png", node="9")
        text = json.loads(rg.recreatable_params(f))["params"]
        self.assertIn('6.text (positive): "a cozy beach house at dusk"', text)
        self.assertIn('7.text (negative): "blurry"', text)
        self.assertIn("3.seed 912334 · 3.steps 30 · 3.cfg 7 · 3.sampler_name euler · 3.scheduler normal · 3.denoise 1", text)
        self.assertIn("5.width 1024", text)
        self.assertIn("12.lora_name detail.safetensors (12.strength_model 0.8 · 12.strength_clip 1)", text)

    def test_long_prompts_are_cut_and_full_returns_the_one_value(self):
        g = copy.deepcopy(template("api_wan2_6_i2v"))
        g["41"]["inputs"]["prompt"] = LONG
        f = self.shot(g, "w/clip.mp4")
        raw = rg.recreatable_params(f)
        self.assertLessEqual(tok(raw), 400)
        self.assertIn(f"[{len(' '.join(LONG.split()))} chars]", json.loads(raw)["params"])
        full = json.loads(rg.recreatable_params(f, full="41.prompt"))
        self.assertEqual(full["value"], LONG)
        self.assertNotIn(SENTINEL, rg._j(full))
        self.assertEqual(json.loads(rg.recreatable_params(f, full="positive"))["param"], "41.prompt")

    def test_overlong_lists_drop_inputs_then_models(self):
        g = copy.deepcopy(template("api_wan2_6_i2v"))
        for i in range(40):
            g[f"L{i}"] = {"class_type": "LoadImage", "inputs": {"image": f"some_reference_{i:03d}.png"}}
            g[f"M{i}"] = {"class_type": "LoraLoader", "inputs": {"lora_name": f"lora_number_{i:03d}.safetensors", "strength_model": 0.5}}
        f = self.shot(g, "w/big.mp4")
        raw = rg.recreatable_params(f)
        out = json.loads(raw)
        self.assertLessEqual(tok(raw), 400)
        self.assertIn("+", out["params"].splitlines()[-1])
        self.assertIn("more", out["params"].splitlines()[-1])
        self.assertIn("41.prompt (positive)", out["params"])

    def test_no_provenance(self):
        f = self.tmp / "plain.png"
        f.write_bytes(b"x")
        out = json.loads(rg.recreatable_params(str(f)))
        self.assertEqual(out["status"], "error")
        self.assertIn("can't recreate exactly", out["error"])

    def test_secret_files_are_refused(self):
        out = json.loads(rg.recreatable_params(str(self.tmp / ".env")))
        self.assertEqual(out["status"], "error")
        self.assertIn("Refused", out["error"])


class Versions(Base):
    def test_project_layout(self):
        d = self.tmp / "SEQ" / "SH010" / "render" / "v002"
        d.mkdir(parents=True)
        f = d / "SH010_v002_beauty.mp4"
        f.write_bytes(b"x")
        path, n = rg.next_version_path(str(f))
        self.assertEqual((Path(path), n), (self.tmp / "SEQ" / "SH010" / "render" / "v003" / "SH010_v003_beauty.mp4", 3))
        self.assertFalse(Path(path).parent.exists())                # nothing created
        (self.tmp / "SEQ" / "SH010" / "render" / "v003").mkdir()
        (self.tmp / "SEQ" / "SH010" / "render" / "v003" / "SH010_v003_beauty.mp4").write_bytes(b"x")
        path, n = rg.next_version_path(str(f))
        self.assertEqual((Path(path).parent.name, n), ("v004", 4))

    def test_a_version_token_in_the_name(self):
        f = self.tmp / "shot_0003_v002_upscaled_final.mp4"
        f.write_bytes(b"x")
        path, n = rg.next_version_path(str(f))
        self.assertEqual((Path(path).name, n), ("shot_0003_v003_upscaled_final.mp4", 3))
        Path(path).write_bytes(b"x")
        path, n = rg.next_version_path(str(f))
        self.assertEqual((Path(path).name, n), ("shot_0003_v004_upscaled_final.mp4", 4))
        self.assertEqual(rg.next_version_path(str(self.tmp / "clip-v7.mp4"))[0], str(self.tmp / "clip-v8.mp4"))

    def test_a_plain_name_and_comfyui_counter(self):
        f = self.tmp / "AnimateDiff_00001.mp4"
        path, n = rg.next_version_path(str(f))
        self.assertEqual((Path(path).name, n), ("AnimateDiff_00001_v002.mp4", 2))
        path, n = rg.next_version_path(str(self.tmp / "beach_00004_.png"))
        self.assertEqual(Path(path).name, "beach_00004_v002.png")
        Path(path).write_bytes(b"x")
        self.assertEqual(rg.next_version_path(str(self.tmp / "beach_00004_.png"))[0], str(self.tmp / "beach_00004_v003.png"))


class Regenerate(Base):
    def wan(self, **kw):
        return self.shot(template("api_wan2_6_i2v"), kw.pop("rel", "r/shot_03.mp4"), **kw)

    def test_dry_run_shows_the_diff_and_runs_nothing(self):
        f = self.wan()
        before = sorted(p.name for p in self.tmp.rglob("*"))
        raw = rg.regenerate_from_source(f, [{"param": "steps_not_there", "value": 1}][:0] + [
            {"param": "negative", "value": "ugly"}, {"param": "duration", "value": "8"}], dry_run=True)
        out = json.loads(raw)
        self.assertEqual((out["status"], out["dry_run"]), ("ok", True))
        shot = out["shots"][0]
        self.assertEqual(shot["shot"], "shot_03")
        self.assertEqual(shot["source"], "sidecar")
        self.assertTrue(any(d.startswith("41.negative_prompt (negative):") for d in shot["diff"]), shot["diff"])
        self.assertIn("41.duration: 5 → 8", shot["diff"])
        self.assertRegex(shot["unchanged"], r"^everything else unchanged \(\d+ nodes, \d+ inputs\)$")
        self.assertEqual(shot["cost"], "local: no credits")
        self.assertTrue(shot["next"].endswith("shot_03_v002.mp4"))
        self.assertEqual(self.posts, [])
        self.assertEqual(before, sorted(p.name for p in self.tmp.rglob("*")))     # nothing written
        self.assertLessEqual(tok(rg._j(shot)), 400)
        self.assertNotIn(SENTINEL, raw)

    def test_only_the_requested_input_changes_and_the_seed_is_kept(self):
        f = self.wan()
        plan = rg.plan_regeneration([f], [{"param": "41.duration", "value": 10}])[0]
        orig = pv.provenance_of(f)["graph"]
        self.assertTrue(plan["ok"])
        self.assertEqual(diff_inputs(orig, plan["_graph"]), {("41", "duration")})
        self.assertEqual(plan["_graph"]["41"]["inputs"]["seed"], orig["41"]["inputs"]["seed"])
        self.assertEqual(plan["_graph"]["41"]["inputs"]["duration"], 10)
        self.assertEqual(json.loads(Path(plan["workflow_path"]).read_text(encoding="utf-8")), plan["_graph"])
        self.assertEqual(plan["diff"], ["41.duration: 5 → 10"])

    def test_new_seed_changes_only_seeds(self):
        f = self.shot(template("image_flux2_klein_text_to_image"), "t/img_00001_.png", node="9")
        orig = pv.provenance_of(f)["graph"]
        plan = rg.plan_regeneration([f], None, new_seed=True)[0]
        self.assertTrue(plan["ok"], plan.get("error"))
        self.assertEqual(diff_inputs(orig, plan["_graph"]), {("75:73", "noise_seed")})
        self.assertEqual(len(plan["diff"]), 1)
        self.assertTrue(plan["diff"][0].startswith("75:73.noise_seed: 0 → "))

    def test_prompt_edits_append_and_replace_show_only_the_words(self):
        g = template("api_wan2_6_i2v")
        g["41"]["inputs"]["prompt"] = "a cozy beach house at dusk, warm light"
        f = self.shot(g, "r/shot_05.mp4")
        out = json.loads(rg.regenerate_from_source(f, [{"param": "positive", "replace": ["dusk", "night"]}]))
        self.assertIn("41.prompt (positive): …dusk… → …night…", out["shots"][0]["diff"])
        plan = rg.plan_regeneration([f], [{"param": "prompt", "append": "rain falling"}])[0]
        self.assertEqual(plan["_graph"]["41"]["inputs"]["prompt"], "a cozy beach house at dusk, warm light rain falling")
        self.assertTrue(plan["diff"][0].startswith("41.prompt (positive): + …rain falling…"))
        bad = rg.plan_regeneration([f], [{"param": "positive", "replace": ["not there", "x"]}])[0]
        self.assertFalse(bad["ok"])

    def test_a_primitive_behind_a_prompt_is_the_param(self):
        f = self.shot(template("image_flux2_klein_text_to_image"), "t/img_00001_.png", node="9")
        plan = rg.plan_regeneration([f], [{"param": "positive", "value": "a red fox"}])[0]
        self.assertTrue(plan["ok"], plan.get("error"))
        self.assertEqual(plan["diff"][0].split(":")[0], "76.value (positive)")
        self.assertEqual(plan["_graph"]["76"]["inputs"]["value"], "a red fox")
        link = rg.plan_regeneration([f], [{"param": "75:74.text", "value": "x"}])[0]
        self.assertIn("wired to another node", link["error"])
        self.assertIn("76.value", link["error"])

    def test_refusals(self):
        f = self.wan()
        p = rg.plan_regeneration([f], [])[0]
        self.assertFalse(p["ok"])
        self.assertIn("nothing to change", p["error"])
        same = rg.plan_regeneration([f], [{"param": "41.duration", "value": 5}])[0]
        self.assertFalse(same["ok"])
        self.assertIn("cached", same["error"])
        self.assertIn("no parameter", rg.plan_regeneration([f], [{"param": "41.nope", "value": 1}])[0]["error"])
        self.assertIn("not a number", rg.plan_regeneration([f], [{"param": "duration", "value": "x"}])[0]["error"])
        plain = self.tmp / "plain.png"
        plain.write_bytes(b"x")
        self.assertIn("can't recreate exactly", rg.plan_regeneration([str(plain)], [{"param": "a.b", "value": 1}])[0]["error"])

    def test_ambiguous_alias_lists_the_ids(self):
        g = copy.deepcopy(template("api_wan2_6_i2v"))
        g["77"] = {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 20}}
        f = self.shot(g, "r/two.mp4")
        err = rg.plan_regeneration([f], [{"param": "seed", "value": 1}])[0]["error"]
        self.assertIn("41.seed", err)
        self.assertIn("77.seed", err)

    def test_coercion_to_the_original_type(self):
        f = self.wan()
        g = rg.plan_regeneration([f], [{"param": "41.seed", "value": "123"}, {"param": "41.watermark", "value": "true"},
                                        {"param": "41.generate_audio", "value": "no"}])[0]["_graph"]
        self.assertIs(g["41"]["inputs"]["watermark"], True)
        self.assertIs(g["41"]["inputs"]["generate_audio"], False)
        self.assertEqual(g["41"]["inputs"]["seed"], 123)
        self.assertIsInstance(g["41"]["inputs"]["seed"], int)

    def test_a_change_can_name_its_shot(self):
        a, b = self.wan(rel="r/shot_03.mp4"), self.wan(rel="r/shot_05.mp4")
        plans = rg.plan_regeneration([a, b], [{"param": "duration", "value": 8, "path": "shot_05"},
                                              {"param": "negative", "value": "grain"}])
        self.assertEqual([len(p["diff"]) for p in plans], [1, 2])
        self.assertTrue(plans[0]["diff"][0].startswith("41.negative_prompt (negative)"))
        self.assertIn("41.duration: 5 → 8", plans[1]["diff"])
        self.assertFalse(rg.plan_regeneration([a], [{"param": "duration", "value": 8, "path": "nobody"}])[0]["ok"])

    def test_the_real_run_is_sequential_versioned_and_leaves_provenance(self):
        a, b = self.wan(rel="r/shot_03.mp4", prompt_id="pidA"), self.wan(rel="r/shot_05.mp4", prompt_id="pidB")
        log = []
        rg.set_runner(self.runner_that_writes(log=log))
        raw = rg.regenerate_from_source([a, b], [{"param": "duration", "value": 8}], dry_run=False)
        out = json.loads(raw)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(len(log), 2)
        self.assertEqual([r["status"] for r in out["results"]], ["done", "done"])
        self.assertEqual([Path(r["new_file"]).name for r in out["results"]], ["shot_03_v002.mp4", "shot_05_v002.mp4"])
        self.assertEqual([r["prompt_id"] for r in out["results"]], ["newpid1", "newpid2"])
        self.assertEqual(out["takes"][0], {"from_file": a, "file": out["results"][0]["new_file"], "version": 2})
        self.assertIn("swap_in_takes", out["next_step"])
        self.assertEqual(self.posts, [])                                  # no /prompt from our side
        self.assertTrue(Path(a).is_file())
        new = out["results"][0]["new_file"]
        self.assertEqual(Path(new).read_bytes(), b"new")
        got = pv.provenance_of(new)
        self.assertEqual(got["source"], "sidecar")
        self.assertEqual(got["graph"]["41"]["inputs"]["duration"], 8)
        doc = json.loads(pv_sidecar(new).read_text(encoding="utf-8"))["provenance"]
        self.assertEqual(doc["source"], "regenerate")
        self.assertEqual(doc["prompt_id"], "newpid1")
        self.assertEqual(doc["derived_from"], {"file": a, "prompt_id": "pidA", "changes": ["41.duration: 5 → 8"]})
        self.assertNotIn(SENTINEL, raw)
        for r in out["results"]:
            self.assertLessEqual(tok(rg._j(r)), 400)

    def test_a_runner_that_only_gives_outputs_is_completed_from_their_provenance(self):
        a = self.wan(rel="r/shot_03.mp4", prompt_id="pidA")
        inner = self.runner_that_writes(log=[])

        def run(path, *, confirm_cost=False):
            out = inner(path, confirm_cost=confirm_cost)
            for i, o in enumerate(out["outputs"]):          # what a host's executor records
                pv.capture([o], {"hostpid": {"prompt": [1, "hostpid", {"41": {"class_type": "X", "inputs": {}}}],
                                             "outputs": {"41": {"videos": [{"filename": Path(o).name}]}}}},
                           "hostpid")
            return {"status": "done", "outputs": out["outputs"]}
        rg.set_runner(run)
        out = json.loads(rg.regenerate_from_source([a], [{"param": "duration", "value": 8}], dry_run=False))
        self.assertEqual(out["results"][0]["prompt_id"], "hostpid")
        self.assertEqual(pv.recorded(out["results"][0]["new_file"])["prompt_id"], "hostpid")

    def test_the_host_hears_where_a_produced_file_went(self):
        a = self.wan(rel="r/shot_03.mp4", prompt_id="pidA")
        moved, produced = [], []
        inner = self.runner_that_writes(log=[])

        def run(path, *, confirm_cost=False):
            out = inner(path, confirm_cost=confirm_cost)
            produced.extend(out["outputs"])
            return out
        rg.set_runner(run)
        rg.set_move_listener(lambda src, dst: moved.append((src, dst)))
        self.addCleanup(rg.set_move_listener, None)
        out = json.loads(rg.regenerate_from_source([a], [{"param": "duration", "value": 8}], dry_run=False))
        self.assertEqual(moved, [(produced[0], out["results"][0]["new_file"])])
        # a listener that fails is the host's problem, never the shot's
        rg.set_move_listener(lambda *_: 1 / 0)
        out = json.loads(rg.regenerate_from_source([a], [{"param": "duration", "value": 9}], dry_run=False))
        self.assertEqual(out["status"], "ok")

    def test_confirm_cost_and_errors_are_reported_per_shot(self):
        a, b = self.wan(rel="r/shot_03.mp4"), self.wan(rel="r/shot_05.mp4")
        calls = []

        def run(path, *, confirm_cost=False):
            calls.append(confirm_cost)
            if len(calls) == 1:
                return {"status": "error", "error_kind": "confirm_cost", "error": "costs $2.10", "outputs": [], "prompt_id": ""}
            raise RuntimeError("boom")
        rg.set_runner(run)
        out = json.loads(rg.regenerate_from_source([a, b], [{"param": "duration", "value": 8}], dry_run=False, confirm_cost=True))
        self.assertEqual(calls, [True, True])
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["results"][0]["error_kind"], "confirm_cost")
        self.assertIn("boom", out["results"][1]["error"])
        self.assertIn("confirm_cost=True", out["hint"])

    def test_an_invalid_workflow_stops_the_shot(self):
        f = self.wan()
        self.validate.return_value = json.dumps({"valid": False, "local_errors": ["Node 41: bad"], "server_errors": {}})
        rg.set_runner(lambda *a, **k: self.fail("must not run"))
        out = json.loads(rg.regenerate_from_source(f, [{"param": "duration", "value": 8}], dry_run=False))
        self.assertEqual(out["results"][0]["status"], "error")
        self.assertIn("invalid workflow", out["results"][0]["error"])

    def test_the_output_is_picked_by_node_and_index(self):
        f = self.shot(template("api_wan2_6_i2v"), "r/shot_03.mp4", node="59")
        plan = rg.plan_regeneration([f], [{"param": "duration", "value": 8}])[0]
        o1, o2 = self.tmp / "o1.mp4", self.tmp / "o2.mp4"
        o1.write_bytes(b"one")
        o2.write_bytes(b"two")
        res = rg.finish_regeneration(plan, [str(o1), str(o2)], "np", outputs_by_node={"58": [str(o1)], "59": [str(o2)]})
        self.assertEqual(res["status"], "done")
        self.assertEqual(Path(res["new_file"]).read_bytes(), b"two")
        # no node info: the first file of the same media kind
        f2 = self.shot(template("api_wan2_6_i2v"), "r/shot_09.mp4")
        plan2 = rg.plan_regeneration([f2], [{"param": "duration", "value": 8}])[0]
        png = self.tmp / "frame.png"
        mp4 = self.tmp / "x.mp4"
        png.write_bytes(b"png")
        mp4.write_bytes(b"mp4")
        res2 = rg.finish_regeneration(plan2, [str(png), str(mp4)], "np2")
        self.assertEqual(Path(res2["new_file"]).read_bytes(), b"mp4")

    def test_the_default_runner_polls_history(self):
        f = self.wan()
        out_file = self.tmp / "comfy_out" / "AnimateDiff_00001.mp4"
        out_file.parent.mkdir()
        out_file.write_bytes(b"new")
        hist = {"newid": {"outputs": {"59": {"gifs": [{"filename": out_file.name, "subfolder": "", "type": "output"}]}},
                          "status": {"completed": True, "status_str": "success"}}}
        self.client.get.side_effect = lambda path, *a, **k: hist if path.startswith("/history/newid") else {}
        self.client.post.side_effect = lambda path, json_data=None, **k: {"prompt_id": "newid"}
        with mock.patch("agenty_core.comfyui_exec._get_comfyui_output_dir", return_value=out_file.parent), \
                mock.patch("agenty_core.partner_credits.gate_submission", return_value=None), \
                mock.patch("agenty_core.partner_credits.track"), mock.patch("agenty_core.queue_ledger.remember"):
            out = json.loads(rg.regenerate_from_source(f, [{"param": "duration", "value": 8}], dry_run=False))
        self.assertEqual(out["results"][0]["status"], "done", out)
        self.assertEqual(out["results"][0]["prompt_id"], "newid")
        self.assertTrue(Path(out["results"][0]["new_file"]).name.endswith("shot_03_v002.mp4"))

    def test_a_redacted_secret_in_the_graph_refuses(self):
        g = copy.deepcopy(template("api_wan2_6_i2v"))
        g["41"]["inputs"]["api_key"] = FAKE_KEY
        f = self.shot(g, "r/k.mp4")
        p = rg.plan_regeneration([f], [{"param": "duration", "value": 8}])[0]
        self.assertFalse(p["ok"])
        self.assertIn("redacted", p["error"])
        for suffix in (".agenty.json",):
            self.assertNotIn(FAKE_KEY, pv_sidecar(f).read_text(encoding="utf-8"))
        self.assertNotIn(FAKE_KEY, rg.recreatable_params(f))


class TokenRules(Base):
    def test_no_graph_and_no_sentinel_in_any_answer(self):
        g = template("image_flux2_klein_text_to_image")
        g["76"]["inputs"]["value"] = LONG
        f = self.shot(g, "t/img_00001_.png", node="9")
        answers = [rg.recreatable_params(f),
                   rg.regenerate_from_source(f, [{"param": "positive", "append": "x " * 400}], dry_run=True),
                   rg.regenerate_from_source(f, None, new_seed=True, dry_run=True)]
        rg.set_runner(self.runner_that_writes(name="img.png", log=[]))
        answers.append(rg.regenerate_from_source(f, [{"param": "steps", "value": 31}], dry_run=False))
        graph_text = json.dumps(g)
        for a in answers:
            self.assertNotIn(SENTINEL, a)
            self.assertNotIn("class_type", a)
            self.assertNotIn('"inputs"', a)
            self.assertNotIn(graph_text[:200], a)
            self.assertNotIn(FAKE_KEY, a)
        print(f"\n  token sizes: recreatable_params={tok(answers[0])}, dry-run shot={tok(json.dumps(json.loads(answers[1])['shots'][0]))}")

    def test_many_changes_still_fit_one_shot(self):
        f = self.shot(template("api_wan2_6_i2v"), "r/many.mp4")
        changes = [{"param": "positive", "value": "word " * 300}, {"param": "negative", "value": "bad " * 300},
                   {"param": "duration", "value": 9}, {"param": "seed", "value": 4}, {"param": "resolution", "value": "1080P"}]
        out = json.loads(rg.regenerate_from_source(f, changes, dry_run=True))
        self.assertLessEqual(tok(rg._j(out["shots"][0])), 400)
        refused = json.loads(rg.regenerate_from_source(f, [], dry_run=True))
        self.assertEqual(refused["shots"][0]["status"], "refused")

    def test_describe_plans_strips_internal_keys(self):
        f = self.shot(template("api_wan2_6_i2v"), "r/s.mp4")
        plans = rg.plan_regeneration([f], [{"param": "duration", "value": 8}])
        self.assertIn("_graph", plans[0])
        text = json.dumps(rg.describe_plans(plans, True))
        self.assertNotIn("_graph", text)
        self.assertNotIn("_orig", text)


class Takes(Base):
    def test_posts_the_documented_body(self):
        a, b = self.tmp / "o.mp4", self.tmp / "n.mp4"
        a.write_bytes(b"1")
        b.write_bytes(b"2")
        self.client.post.side_effect = lambda *a, **k: {"ok": True, "saved": True, "applied": 1, "pushed": False, "notes": ["n1"]}
        out = json.loads(rg.swap_in_takes([{"from_file": str(a), "file": str(b), "version": 2, "note": "night"}],
                                          otio=str(self.tmp / "cut.otio"), retime=True))
        args, kwargs = self.client.post.call_args
        self.assertEqual(args[0], "/imageviewer/take")
        self.assertEqual(kwargs["json_data"], {"otio": str(self.tmp / "cut.otio"),
                                               "takes": [{"from_file": str(a), "file": str(b), "version": 2, "note": "night"}],
                                               "retime": True})
        self.assertEqual(out, {"status": "ok", "saved": True, "applied": 1, "pushed": False, "notes": ["n1"]})

    def test_old_viewer_and_bad_input(self):
        a = self.tmp / "o.mp4"
        a.write_bytes(b"1")
        err = Exception("404 Client Error")
        self.client.post.side_effect = err
        out = json.loads(rg.swap_in_takes([{"from_file": str(a), "file": str(a)}]))
        self.assertEqual(out["status"], "error")
        self.assertIn("predates the take route", out["error"])
        self.assertEqual(json.loads(rg.swap_in_takes([{"file": str(a)}]))["status"], "error")
        self.assertEqual(json.loads(rg.swap_in_takes([{"from_file": str(a), "file": str(self.tmp / "missing.mp4")}]))["status"], "error")
        self.assertIn("Refused", json.loads(rg.swap_in_takes([{"from_file": str(self.tmp / ".env"), "file": str(a)}]))["error"])


if __name__ == "__main__":
    unittest.main()
