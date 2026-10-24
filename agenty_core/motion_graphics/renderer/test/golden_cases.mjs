// The cases both copies of mgscene.js (this repo's, the viewer's) must answer the same.
// golden.json holds the answers; regenerate with: node test/make_golden.mjs
export const CASES = [
  { name: "linear num", prop: "opacity", keys: [{f:0,v:0,ease:"linear"},{f:10,v:100}], frames: [-3,0,2.5,5,10,14] },
  { name: "smooth num", prop: "opacity", keys: [{f:0,v:0},{f:10,v:100}], frames: [0,2.5,5,7.5,10] },
  { name: "hold", prop: "rotation", keys: [{f:0,v:0,ease:"hold"},{f:10,v:90}], frames: [0,5,9.9,10] },
  { name: "easeIn", prop: "rotation", keys: [{f:0,v:0,ease:"easeIn"},{f:20,v:360}], frames: [0,5,10,15,20] },
  { name: "easeOut", prop: "rotation", keys: [{f:0,v:0,ease:"easeOut"},{f:20,v:360}], frames: [0,5,10,15,20] },
  { name: "backOut overshoots", prop: "scale", keys: [{f:0,v:[0,0],ease:"backOut"},{f:12,v:[100,100]}], frames: [0,3,6,9,12] },
  { name: "custom bezier", prop: "opacity", keys: [{f:0,v:0,ease:[0.25,0.1,0.25,1]},{f:30,v:100}], frames: [0,7,15,22,30] },
  { name: "vec2 linear", prop: "position", keys: [{f:0,v:[0,100],ease:"linear"},{f:10,v:[100,0]}], frames: [0,5,10] },
  { name: "tangents", prop: "opacity", keys: [{f:0,v:0,to:20},{f:10,v:100,ti:0}], frames: [0,2,5,8,10] },
  { name: "colour", prop: "fill", keys: [{f:0,v:"#ff0000",ease:"linear"},{f:10,v:"#0000ff"}], frames: [0,5,10] },
  { name: "colour alpha", prop: "fill", keys: [{f:0,v:"#ffffff00",ease:"linear"},{f:10,v:"#ffffffff"}], frames: [0,5,10] },
  { name: "three keys", prop: "rotation", keys: [{f:0,v:0,ease:"linear"},{f:10,v:90,ease:"hold"},{f:20,v:0}], frames: [0,5,10,15,20,25] },
  { name: "spin over 360", prop: "rotation", keys: [{f:0,v:0,ease:"linear"},{f:10,v:720}], frames: [0,2.5,5,10] },
];
