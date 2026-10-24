// node render.mjs <job.json>
// Renders one *.mgscene.json with the SceneRenderer composition. The job file:
//   { "mode": "video" | "still",
//     "scene": "<path to the scene>",         // asset paths in it are relative to this file
//     "out": "<output file>",
//     "frame": 0,                              // still only
//     "format": "mp4" | "webm" | "mov" | "gif",   // video only
//     "crf": 18, "scale": 1, "range": [from, to], "lossless": false, "concurrency": null }
// stdout: one JSON object per line - {"event":"status"|"progress"|"done"|"error", ...}.
// Assets are served from a throw-away local http server (so the bundle is built ONCE per source
// change and cached in .render-cache/, which makes re-rendering an edited scene quick).
import fs from 'node:fs';
import http from 'node:http';
import crypto from 'node:crypto';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const say = (o) => process.stdout.write(JSON.stringify(o) + '\n');

const MIME = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.gif': 'image/gif',
  '.svg': 'image/svg+xml', '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
  '.ttf': 'font/ttf', '.otf': 'font/otf', '.woff': 'font/woff', '.woff2': 'font/woff2',
};

function srcHash() {
  const h = crypto.createHash('sha1');
  const walk = (d) => {
    for (const n of fs.readdirSync(d).sort()) {
      const p = path.join(d, n);
      if (fs.statSync(p).isDirectory()) walk(p); else { h.update(n); h.update(fs.readFileSync(p)); }
    }
  };
  walk(path.join(HERE, 'src'));
  h.update(fs.readFileSync(path.join(HERE, 'package.json')));
  return h.digest('hex').slice(0, 12);
}

function serveAssets(files) {
  const server = http.createServer((req, res) => {
    const id = decodeURIComponent((req.url || '').split('?')[0].split('/')[2] || '');
    const file = files.get(id);
    if (!file || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    const size = fs.statSync(file).size;
    const head = {'Access-Control-Allow-Origin': '*', 'Accept-Ranges': 'bytes',
      'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-store'};
    const m = /bytes=(\d*)-(\d*)/.exec(req.headers.range || '');
    if (m) {
      const start = m[1] ? +m[1] : 0, end = m[2] ? Math.min(+m[2], size - 1) : size - 1;
      res.writeHead(206, {...head, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Content-Length': end - start + 1});
      fs.createReadStream(file, {start, end}).pipe(res);
    } else {
      res.writeHead(200, {...head, 'Content-Length': size});
      fs.createReadStream(file).pipe(res);
    }
  });
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve({server, port: server.address().port})));
}

async function main() {
  const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
  const sceneFile = path.resolve(job.scene);
  const scene = JSON.parse(fs.readFileSync(sceneFile, 'utf8').replace(/^﻿/, ''));
  const sceneDir = path.dirname(sceneFile);

  // Local assets -> urls on our server; http(s)/data urls stay as they are.
  const files = new Map();
  const missing = [];
  const { server, port } = await serveAssets(files);
  for (const [id, a] of Object.entries(scene.assets || {})) {
    if (!a || typeof a.src !== 'string' || /^(https?:|data:)/i.test(a.src)) continue;
    const abs = path.resolve(sceneDir, a.src.replace(/^file:\/\/\/?/, ''));
    if (!fs.existsSync(abs)) { missing.push(`${id}: ${abs}`); continue; }
    files.set(id, abs);
    a.src = `http://127.0.0.1:${port}/a/${encodeURIComponent(id)}/${encodeURIComponent(path.basename(abs))}`;
  }
  if (missing.length) { say({event: 'error', error: `asset file(s) not found: ${missing.join('; ')}`}); server.close(); process.exit(2); }

  const {bundle} = await import('@remotion/bundler');
  const {renderMedia, renderStill, selectComposition, ensureBrowser} = await import('@remotion/renderer');

  say({event: 'status', message: 'checking the render browser'});
  await ensureBrowser();

  const cache = path.join(HERE, '.render-cache');
  fs.mkdirSync(cache, {recursive: true});
  const outDir = path.join(cache, 'bundle-' + srcHash());
  if (!fs.existsSync(path.join(outDir, 'index.html'))) {
    say({event: 'status', message: 'bundling the renderer (first render after a change)'});
    for (const d of fs.readdirSync(cache)) if (d.startsWith('bundle-')) fs.rmSync(path.join(cache, d), {recursive: true, force: true});
    await bundle({entryPoint: path.join(HERE, 'src', 'index.js'), outDir});
  }

  const inputProps = {scene};
  const composition = await selectComposition({serveUrl: outDir, id: 'SceneRenderer', inputProps});
  say({event: 'status', message: `composition ${composition.width}x${composition.height} @ ${composition.fps} fps, ${composition.durationInFrames} frames`});
  fs.mkdirSync(path.dirname(path.resolve(job.out)), {recursive: true});

  if (job.mode === 'still') {
    // One frame (out is the file) or several (out is a prefix: <out>_f0012.png).
    const list = Array.isArray(job.frames) && job.frames.length ? job.frames : [job.frame || 0];
    const outs = [];
    for (const raw of list) {
      const frame = Math.max(0, Math.min(composition.durationInFrames - 1, Math.round(raw)));
      const output = path.resolve(Array.isArray(job.frames) && job.frames.length
        ? `${job.out.replace(/\.(png|jpe?g)$/i, '')}_f${String(frame).padStart(4, '0')}.png` : job.out);
      await renderStill({composition, serveUrl: outDir, output, frame, inputProps,
        imageFormat: /\.jpe?g$/i.test(output) ? 'jpeg' : 'png', scale: job.scale || 1, overwrite: true});
      outs.push({frame, out: output});
      say({event: 'progress', pct: Math.round((outs.length / list.length) * 100)});
    }
    say({event: 'done', stills: outs, width: composition.width, height: composition.height, fps: composition.fps,
      duration: composition.durationInFrames});
  } else {
    const format = job.format || 'mp4';
    const codec = {mp4: 'h264', webm: 'vp9', mov: 'prores', gif: 'gif'}[format];
    if (!codec) throw new Error(`unknown format ${format}`);
    const [from, to] = Array.isArray(job.range) && job.range.length === 2 ? job.range : [0, composition.durationInFrames - 1];
    let last = -1;
    await renderMedia({
      composition, serveUrl: outDir, codec, outputLocation: path.resolve(job.out), inputProps,
      crf: format === 'mp4' || format === 'webm' ? (job.crf ?? 18) : undefined,
      pixelFormat: format === 'mp4' ? 'yuv420p' : undefined,
      proResProfile: format === 'mov' ? '4444' : undefined,
      imageFormat: job.lossless || format === 'mov' || format === 'webm' ? 'png' : 'jpeg',
      jpegQuality: 95, scale: job.scale || 1, frameRange: [from, to], overwrite: true,
      concurrency: job.concurrency || null,
      onProgress: (p) => {
        const pct = Math.round(p.progress * 100);
        if (pct !== last) { last = pct; say({event: 'progress', pct, rendered: p.renderedFrames, encoded: p.encodedFrames}); }
      },
    });
    say({event: 'done', out: path.resolve(job.out), frames: to - from + 1, width: composition.width, height: composition.height,
      fps: composition.fps, bytes: fs.statSync(path.resolve(job.out)).size});
  }
  server.close();
}

main().then(() => process.exit(0)).catch((e) => { say({event: 'error', error: String((e && e.message) || e)}); process.exit(1); });
