import * as THREE from 'three';
import { Bev } from './bev';
import { CATEGORY_COLOR, CATEGORY_LABEL, sampleLabels, type Study } from './data';
import type { SceneObject } from './executor';
import { Viewer } from './viewer';
import { cssVar, onThemeChange, segmented } from './util';
import { heightTone, loadSweep } from './lidar';
import { isDark } from './theme';

/** Figure 1: one keyframe as a bird's-eye view or in 3D, next to the exact prompt rows. */
export function initExplorer(study: Study, startToken: string) {
  const fig = document.getElementById('fig-explorer')!;
  const bev = new Bev(fig.querySelector('[data-role=bev]') as unknown as SVGSVGElement);
  const canvas = fig.querySelector<HTMLCanvasElement>('[data-role=three]')!;
  const prompt = fig.querySelector<HTMLElement>('[data-role=prompt]')!;
  const legend = fig.querySelector<HTMLElement>('[data-role=legend]')!;
  const caption = fig.querySelector<HTMLElement>('[data-role=caption]')!;
  const select = fig.querySelector<HTMLSelectElement>('[data-ctl=sample]')!;
  const labels = sampleLabels(study);
  for (const [t, l] of labels) select.add(new Option(l, t));
  select.value = startToken;

  let token = startToken;
  let colour = 'category';
  let view = 'bev';
  let viewer: Viewer | null = null;
  let boxes: THREE.Group | null = null;
  const lines = new Map<string, HTMLElement>();

  const colourOf = (o: SceneObject) => colour === 'lidar'
    ? (o.lidar_pts ? cssVar('--ink-2') : cssVar('--bad'))
    : CATEGORY_COLOR[o.category] ?? cssVar('--obj');
  const hollow = (o: SceneObject) => colour === 'lidar' && !o.lidar_pts;

  function hover(o: SceneObject | null) {
    bev.hot(o?.token ?? null);
    bev.label(o, o ? `${o.id} · ${CATEGORY_LABEL(o.category)}${o.status ? `, ${o.status}` : ''}${colour === 'lidar' ? ` · ${o.lidar_pts ?? 0} returns` : ''}` : '');
    for (const [t, n] of lines) n.classList.toggle('hot', t === o?.token);
    if (o) lines.get(o.token)?.scrollIntoView({ block: 'nearest' });
  }
  bev.onHover = hover;

  function render() {
    const objects = study.scenes[token].objects;
    bev.setScene(objects);
    bev.paint(colourOf, hollow);
    const want = token;
    loadSweep(token).then((sw) => { if (want === token) { bev.points(sw, isDark()); if (viewer) build3d(); } });
    // the prompt rows, exactly as build_user_prompt formats them
    prompt.replaceChildren();
    lines.clear();
    const head = document.createElement('div');
    head.className = 'pl head';
    head.textContent = `Objects (${objects.length}):\nid, category, x, y, status`;
    prompt.append(head);
    for (const o of objects) {
      const line = document.createElement('div');
      line.className = 'pl';
      if (colour === 'lidar' && !o.lidar_pts) line.classList.add('zero');
      line.textContent = `${o.id}, ${o.category}, ${fmt(o.px)}, ${fmt(o.py)}, ${o.status ?? 'unknown'}`;
      line.addEventListener('pointerenter', () => hover(o));
      line.addEventListener('pointerleave', () => hover(null));
      prompt.append(line);
      lines.set(o.token, line);
    }
    const zero = objects.filter((o) => !o.lidar_pts).length;
    caption.textContent = colour === 'lidar'
      ? `Red, dashed outlines: ${zero} of ${objects.length} annotated objects in this keyframe have no LiDAR return at all.`
      : 'Hover a box or a row to link them.';
    legend.replaceChildren();
    const entries = colour === 'lidar'
      ? [['hit by LiDAR', cssVar('--ink-2')], ['no return', cssVar('--bad')]]
      : [...new Set(objects.map((o) => o.category))].map((c) => [CATEGORY_LABEL(c), CATEGORY_COLOR[c]]);
    for (const [name, c] of entries) {
      const s = document.createElement('span');
      const i = document.createElement('i');
      i.style.background = c;
      s.append(i, name);
      legend.append(s);
    }
    if (viewer) build3d();
  }

  function build3d() {
    if (!viewer) return;
    if (boxes) viewer.scene.remove(boxes);
    boxes = new THREE.Group();
    const objects = study.scenes[token].objects;
    const edge = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
    const cube = new THREE.BoxGeometry(1, 1, 1);
    for (const o of objects) {
      const color = new THREE.Color(colourOf(o));
      const g = new THREE.Group();
      g.position.set(o.x, o.y, o.z);
      g.rotation.z = o.yaw;
      g.scale.set(Math.max(o.l, 0.2), Math.max(o.w, 0.2), Math.max(o.h, 0.2));
      g.add(new THREE.LineSegments(edge, new THREE.LineBasicMaterial({ color })));
      if (!hollow(o)) g.add(new THREE.Mesh(cube, new THREE.MeshLambertMaterial({ color, transparent: true, opacity: 0.55 })));
      boxes.add(g);
    }
    loadSweep(token).then((sw) => {
      if (!sw || !boxes || !viewer) return;
      const dark = isDark();
      const col = new Float32Array(sw.n * 3);
      for (let i = 0; i < sw.n; i++) {
        const [r, g, b] = heightTone(sw.xyz[i * 3 + 2], dark);
        col.set([r, g, b], i * 3);
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(sw.xyz, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      boxes.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.09, vertexColors: true, sizeAttenuation: true })));
      viewer.invalidate();
    });
    const ego = new THREE.Mesh(new THREE.BoxGeometry(4.6, 1.9, 1.6), new THREE.MeshLambertMaterial({ color: cssVar('--ego') }));
    ego.position.z = 0.8;
    boxes.add(ego);
    viewer.scene.add(boxes);
    viewer.invalidate();
  }

  function ensure3d() {
    if (viewer) return;
    viewer = new Viewer(canvas);
    viewer.camera.up.set(0, 0, 1);
    viewer.camera.position.set(-38, 22, 30);
    viewer.controls.target.set(6, 0, 0);
    const rings = new THREE.Group();
    for (let r = 10; r <= 60; r += 10) {
      const geo = new THREE.BufferGeometry().setFromPoints(Array.from({ length: 129 }, (_, i) => {
        const a = (i / 128) * Math.PI * 2;
        return new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), 0);
      }));
      rings.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: cssVar('--rule') })));
    }
    viewer.scene.add(rings);
    build3d();
  }

  select.addEventListener('change', () => { token = select.value; render(); });
  segmented(fig.querySelector('[data-ctl=colour]')!, (v) => { colour = v; render(); });
  segmented(fig.querySelector('[data-ctl=view]')!, (v) => {
    view = v;
    const svg = fig.querySelector<SVGSVGElement>('[data-role=bev]')!;
    svg.style.display = view === 'bev' ? '' : 'none';
    canvas.hidden = view !== '3d';
    if (view === '3d') ensure3d();
  });
  // keep the prompt panel exactly as tall as the scene
  const stage = fig.querySelector<HTMLElement>('.fig-stage')!;
  new ResizeObserver(() => { if (innerWidth > 860) prompt.style.height = `${stage.offsetHeight}px`; else prompt.style.height = ''; }).observe(stage);
  onThemeChange(render);
  render();
}

const fmt = (v: number) => (Object.is(v, -0) ? "-0.0" : Number.isInteger(v) ? `${v}.0` : String(v));
