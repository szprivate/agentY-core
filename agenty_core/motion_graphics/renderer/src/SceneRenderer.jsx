import React, {useEffect, useMemo, useState} from 'react';
import {AbsoluteFill, OffthreadVideo, Sequence, continueRender, delayRender, useCurrentFrame} from 'remotion';
import * as M from './mgscene.js';
import {COMPONENTS} from './components/index.jsx';

// Asset srcs arrive resolved by render.mjs (a url the render serves, or the author's own http(s)/data url).
const assetUrl = (_id, a) => a.src;

/** Hold the first frame until the scene's fonts and pictures are decoded: the SVG layers refer to them by url. */
const useAssets = (scene) => {
  const [handle] = useState(() => delayRender('mgscene assets'));
  useEffect(() => {
    const jobs = [];
    for (const [id, a] of Object.entries(scene.assets || {})) {
      if (!a || !a.src) continue;
      if (a.type === 'font') {
        const face = new FontFace(a.family || id, `url("${a.src}")`, {weight: String(a.weight || 400), style: a.style || 'normal'});
        jobs.push(face.load().then((f) => document.fonts.add(f)));
      } else if (a.type === 'image') {
        const img = new Image();
        img.src = a.src;
        jobs.push(img.decode());
      }
    }
    Promise.allSettled(jobs).then((r) => {
      r.filter((x) => x.status === 'rejected').forEach((x) => console.warn('mgscene asset failed to load:', x.reason));
      continueRender(handle);
    });
  }, [handle, scene]);
};

const VideoLayer = ({res, scene}) => {
  const a = scene.assets[res.layer.asset];
  const style = M.htmlLayerStyle(res);
  const fit = {contain: 'contain', cover: 'cover', fill: 'fill'}[res.layer.fit || 'cover'];
  return (
    <Sequence from={res.layer.in || 0} layout="none">
      <div style={{...style, overflow: 'hidden'}}>
        <OffthreadVideo src={a.src} startFrom={res.layer.startFrom || 0} muted={!res.layer.audio}
          style={{width: '100%', height: '100%', objectFit: fit}} />
      </div>
    </Sequence>
  );
};

const ComponentLayer = ({res}) => {
  const Comp = COMPONENTS[res.layer.component];
  const [w, h] = res.props.size;
  const style = M.htmlLayerStyle(res);
  return (
    <Sequence from={res.layer.in || 0} layout="none">
      <div style={style}>{Comp ? <Comp {...(res.layer.params || {})} width={w} height={h} /> : null}</div>
    </Sequence>
  );
};

export const SceneRenderer = ({scene: raw}) => {
  const scene = useMemo(() => M.normalizeScene(raw), [raw]);
  const frame = useCurrentFrame();
  useAssets(scene);
  const ctx = {width: scene.comp.width, height: scene.comp.height, assetUrl};
  const resolved = M.evaluateScene(scene, frame);
  const fonts = M.fontFaceCss(scene, assetUrl);
  return (
    <AbsoluteFill style={{background: scene.comp.background || 'transparent', isolation: 'isolate', overflow: 'hidden'}}>
      {fonts ? <style>{fonts}</style> : null}
      {resolved.map((res) => {
        if (!res.active) return null;
        if (M.isSvgLayer(res.type)) {
          const svg = M.layerSvg(scene, res, ctx);
          return svg ? <div key={res.id} style={{position: 'absolute', inset: 0, display: 'contents'}} dangerouslySetInnerHTML={{__html: svg}} /> : null;
        }
        if (res.type === 'video' && res.opacity > 0) return <VideoLayer key={res.id} res={res} scene={scene} />;
        if (res.type === 'component' && res.opacity > 0) return <ComponentLayer key={res.id} res={res} />;
        return null;
      })}
    </AbsoluteFill>
  );
};
