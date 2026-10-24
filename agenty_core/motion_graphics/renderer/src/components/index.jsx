// Named code components a `component` layer can point at: {"type":"component","component":"nodeTag","params":{...}}.
// A component gets its layer's params as props plus `width`/`height` (the layer's size) and runs on a Sequence
// that starts at the layer's `in`, so useCurrentFrame() inside it counts from the layer's first frame.
// The editor shows a placeholder box for these; the render draws them for real.
import React from 'react';
import {interpolate, useCurrentFrame} from 'remotion';

const clamp = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'};
const ORANGE = '#ff6a1a';

/** A node-style tag: dark plate, hairline border, orange corner brackets, a socket on top, two-weight title, subline. */
const NodeTag = ({light = 'agent', bold = 'Y', sub = '', width, height, accent = ORANGE}) => {
  const f = useCurrentFrame();
  const on = interpolate(f, [0, 8], [0, 1], clamp);
  return (
    <div style={{width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', opacity: on}}>
      <div style={{position: 'relative', background: 'rgba(10,10,10,0.78)', border: '1px solid rgba(235,235,230,0.32)',
        padding: '16px 34px 18px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, fontFamily: 'Arial, sans-serif'}}>
        <span style={{position: 'absolute', left: -1, top: -1, width: 16, height: 16, borderLeft: `2px solid ${accent}`, borderTop: `2px solid ${accent}`}} />
        <span style={{position: 'absolute', right: -1, bottom: -1, width: 16, height: 16, borderRight: `2px solid ${accent}`, borderBottom: `2px solid ${accent}`}} />
        <span style={{position: 'absolute', left: '50%', top: -7, width: 12, height: 12, marginLeft: -6, borderRadius: '50%', background: '#0a0a0a', border: '1.5px solid #ebebe6'}} />
        <div style={{fontSize: 58, color: '#ebebe6', letterSpacing: '0.02em'}}><span style={{fontWeight: 300}}>{light}</span><span style={{fontWeight: 800}}>{bold}</span></div>
        {sub ? <div style={{fontSize: 24, letterSpacing: '0.22em', color: accent, whiteSpace: 'nowrap'}}>{sub}</div> : null}
      </div>
    </div>
  );
};

export const COMPONENTS = {nodeTag: NodeTag};
export const COMPONENT_NAMES = Object.keys(COMPONENTS);
