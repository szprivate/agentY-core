// node --test test/
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import * as M from "../src/mgscene.js";

const here = (p) => new URL(p, import.meta.url);
const close = (a, b, eps = 1e-6) => {
  if (Array.isArray(a)) { a.forEach((x, i) => close(x, b[i], eps)); return; }
  if (typeof a === "string") { assert.equal(a, b); return; }
  assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
};

test("golden vectors", () => {
  const golden = JSON.parse(fs.readFileSync(here("./golden.json")));
  for (const c of golden) {
    c.frames.forEach((f, i) => close(M.valueAt({ tracks: { [c.prop]: c.keys } }, c.prop, f), c.expect[i], 1e-6));
  }
});

test("validate cases", () => {
  const cases = JSON.parse(fs.readFileSync(here("./validate_cases.json")));
  for (const c of cases) {
    const r = M.validateScene(c.scene);
    assert.deepEqual(r.errors.map((e) => e.code).sort(), [...c.errors].sort(), `${c.name}: errors ${JSON.stringify(r.errors)}`);
    assert.deepEqual(r.warnings.map((e) => e.code).sort(), [...c.warnings].sort(), `${c.name}: warnings`);
    assert.equal(r.ok, c.errors.length === 0);
  }
});

test("static values, defaults and clamping", () => {
  const l = { type: "text", position: [10, 20] };
  assert.deepEqual(M.valueAt(l, "position", 5), [10, 20]);
  assert.deepEqual(M.valueAt(l, "scale", 5), [100, 100]);
  assert.equal(M.valueAt(l, "stroke", 5), null);
  const scene = { comp: { width: 100, height: 100, fps: 30, duration: 10 }, layers: [{ id: "a", type: "group", opacity: 250 }] };
  assert.equal(M.evaluateScene(scene, 0)[0].opacity, 1);
});

test("transform: position, rotation about the anchor, scale", () => {
  const m = M.localMatrix([100, 100], 90, [200, 100], [10, 0]);
  close(M.matApply(m, 10, 0), [100, 100]);               // the anchor lands on the position
  close(M.matApply(m, 11, 0), [100, 102]);               // x scaled by 2, turned 90deg: two steps down
  close(M.matApply(M.matInvert(m), 100, 102), [11, 0]);
});

test("parenting passes the transform, not the opacity", () => {
  const scene = { comp: { width: 1920, height: 1080, fps: 30, duration: 60 }, layers: [
    { id: "p", type: "group", position: [100, 0], rotation: 90, opacity: 50 },
    { id: "c", type: "shape", parent: "p", position: [10, 0], opacity: 80 },
  ] };
  const [p, c] = M.evaluateScene(scene, 0);
  close(M.matApply(c.world, 0, 0), [100, 10]);           // child (10,0) in a parent turned 90deg
  assert.equal(c.opacity, 0.8);
  assert.equal(p.opacity, 0.5);
});

test("in/out and hidden decide what is active", () => {
  const scene = { comp: { width: 10, height: 10, fps: 30, duration: 60 }, layers: [
    { id: "a", type: "group", in: 10, out: 20 }, { id: "b", type: "group", hidden: true }] };
  assert.deepEqual([5, 10, 19, 20].map((f) => M.evaluateScene(scene, f)[0].active), [false, true, true, false]);
  assert.equal(M.evaluateScene(scene, 12)[1].active, false);
});

test("layerSvg: text, reveal and trim", () => {
  const scene = { comp: { width: 640, height: 360, fps: 30, duration: 30 }, assets: {}, layers: [
    { id: "t", type: "text", text: "A & B\nline2", fontSize: 40, align: "center", position: [320, 180], tracks: { reveal: [{ f: 0, v: 0, ease: "linear" }, { f: 10, v: 1 }] } },
    { id: "s", type: "shape", shape: "rect", size: [200, 100], radius: 10, fill: null, stroke: "#f60", strokeWidth: 4, trimEnd: 0.5 },
  ] };
  const ctx = { width: 640, height: 360 };
  const at5 = M.frameSvgs(scene, 5, ctx);
  assert.equal(at5.length, 2);
  assert.match(at5[0].svg, /A &amp; B/);
  assert.match(at5[0].svg, /clipPathUnits="objectBoundingBox"/);
  assert.match(at5[0].svg, /width="0\.5"/);
  assert.match(at5[1].svg, /stroke-dasharray="0\.5 1\.5"/);
  assert.doesNotMatch(M.frameSvgs(scene, 10, ctx)[0].svg, /clipPath/);   // fully revealed: no clip
});

test("sceneKeyframes and tangentAt", () => {
  const scene = { layers: [{ id: "a", tracks: { opacity: [{ f: 0, v: 0, ease: "linear" }, { f: 10, v: 100 }], rotation: [{ f: 5, v: 1 }] } }] };
  assert.deepEqual(M.sceneKeyframes(scene), [0, 5, 10]);
  close(M.tangentAt(scene.layers[0], "opacity", 0, 0, "out"), 10);
});

test("colours round trip", () => {
  assert.deepEqual(M.parseColor("#f80"), [255, 136, 0, 1]);
  assert.equal(M.formatColor(M.parseColor("#12ab34")), "#12ab34");
  assert.equal(M.parseColor("nope"), null);
});
