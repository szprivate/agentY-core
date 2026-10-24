import React from 'react';
import {Composition} from 'remotion';
import {SceneRenderer} from './SceneRenderer.jsx';
import {normalizeScene} from './mgscene.js';

// The one composition. Size, rate and length come from the scene it is given.
const blank = normalizeScene({layers: []});

export const Root = () => (
  <Composition
    id="SceneRenderer"
    component={SceneRenderer}
    width={blank.comp.width}
    height={blank.comp.height}
    fps={blank.comp.fps}
    durationInFrames={blank.comp.duration}
    defaultProps={{scene: blank}}
    calculateMetadata={({props}) => {
      const c = normalizeScene(props.scene).comp;
      return {width: c.width, height: c.height, fps: c.fps, durationInFrames: c.duration};
    }}
  />
);
