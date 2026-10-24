import fs from "node:fs";
import { CASES } from "./golden_cases.mjs";
import { valueAt } from "../src/mgscene.js";
const out = CASES.map((c) => ({ ...c, expect: c.frames.map((f) => valueAt({ tracks: { [c.prop]: c.keys } }, c.prop, f)) }));
fs.writeFileSync(new URL("./golden.json", import.meta.url), JSON.stringify(out, null, 1));
console.log("wrote", out.length);
