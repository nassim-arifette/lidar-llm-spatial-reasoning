import { SECTORS, type Layout, type SceneObject } from './executor';

/** Where each relation is drawn under the as-labelled layout: the unsigned angle, so two wedges for each diagonal. */
const AS_LABELLED_WEDGES: [string, number, number][] = [
  ['front', -30, 30], ['front_left', 30, 90], ['front_left', -90, -30],
  ['back_right', 90, 150], ['back_right', -150, -90], ['back', 150, 210],
];
import { el } from './util';
import { heightTone, type Sweep } from './lidar';

/**
 * Bird's-eye view in metres. Ego frame: x forward (screen up), y left (screen left).
 * Screen coordinates: sx = -y, sy = -x, so the SVG viewBox is directly in metres.
 */
export interface BevOptions { range?: number; aspect?: number; rings?: boolean; labels?: boolean }

const toScreen = (x: number, y: number): [number, number] => [-y, -x];

export class Bev {
  readonly svg: SVGSVGElement;
  objects: SceneObject[] = [];
  range: number;
  private aspect: number;
  private gRings: SVGGElement;
  private gPoints: SVGGElement;
  private gSectors: SVGGElement;
  private gObjs: SVGGElement;
  private gOver: SVGGElement;
  private nodes = new Map<string, SVGPolygonElement>();
  onHover?: (o: SceneObject | null) => void;
  onClick?: (o: SceneObject) => void;
  private opts: BevOptions;

  constructor(svg: SVGSVGElement, opts: BevOptions = {}) {
    this.svg = svg;
    this.opts = opts;
    this.range = opts.range ?? 50;
    this.aspect = opts.aspect ?? 1.45;
    this.gRings = el('g', {}, svg);
    this.gPoints = el('g', { 'pointer-events': 'none' }, svg);
    this.gSectors = el('g', {}, svg);
    this.gObjs = el('g', {}, svg);
    this.gOver = el('g', {}, svg);
    this.setView();
  }

  private setView() {
    const R = this.range, A = this.aspect;
    this.svg.setAttribute('viewBox', `${-R * A} ${-R} ${2 * R * A} ${2 * R}`);
    this.svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
    this.drawRings();
  }

  private get fs() { return this.range * 0.034; }

  private drawRings() {
    this.gRings.replaceChildren();
    if (this.opts.rings === false) return;
    const R = this.range;
    for (let r = 10; r <= R * this.aspect * 1.1; r += 10) {
      el('circle', { class: 'ring', cx: 0, cy: 0, r, 'vector-effect': 'non-scaling-stroke' }, this.gRings);
      if (r <= R * 0.95) {
        const t = el('text', { class: 'ring-label', x: r * 0.71 + 0.6, y: -r * 0.71 - 0.6, 'font-size': this.fs * 0.85 }, this.gRings);
        t.textContent = `${r} m`;
      }
    }
    // ego vehicle: a 1.9 m by 4.6 m body with a heading notch
    const ego = el('g', {}, this.gRings);
    el('rect', { x: -0.95, y: -2.6, width: 1.9, height: 4.6, rx: 0.5, fill: 'var(--ego)' }, ego);
    el('path', { d: 'M-0.6,-1.9 L0,-2.9 L0.6,-1.9 Z', fill: 'var(--surface)' }, ego);
  }

  /** Fit the view to the scene: most objects visible, without letting a far outlier shrink everything. */
  fit(objects: SceneObject[], min = 22, max = 62, must: SceneObject[] = []) {
    const extent = (o: SceneObject) => Math.max(Math.abs(o.x), Math.abs(o.y) / this.aspect) + 3;
    const r = objects.map(extent).sort((a, b) => a - b);
    const q = Math.max(r.length ? r[Math.floor(r.length * 0.92)] : min, ...must.map(extent));
    this.range = Math.min(max, Math.max(min, Math.ceil((q * 1.12) / 5) * 5));
    this.setView();
  }

  setScene(objects: SceneObject[], fit = true, must: SceneObject[] = []) {
    this.objects = objects;
    if (fit) this.fit(objects, 22, 62, must);
    this.gObjs.replaceChildren();
    this.nodes.clear();
    for (const o of objects) {
      const c = Math.cos(o.yaw), s = Math.sin(o.yaw);
      const hl = Math.max(o.l, 0.4) / 2, hw = Math.max(o.w, 0.4) / 2;
      const corners = [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]].map(([a, b]) => toScreen(o.x + a * c - b * s, o.y + a * s + b * c));
      const poly = el('polygon', { class: 'obj', points: corners.map((p) => p.join(',')).join(' ') }, this.gObjs);
      poly.dataset.token = o.token;
      poly.addEventListener('pointerenter', () => this.onHover?.(o));
      poly.addEventListener('pointerleave', () => this.onHover?.(null));
      poly.addEventListener('click', () => this.onClick?.(o));
      this.nodes.set(o.token, poly);
    }
    this.gOver.replaceChildren();
  }

  /** Paint objects. fill(o) returns a colour; hollow(o) draws an outline only. */
  paint(fill: (o: SceneObject) => string, hollow?: (o: SceneObject) => boolean) {
    for (const o of this.objects) {
      const n = this.nodes.get(o.token)!;
      const col = fill(o);
      const h = hollow?.(o) ?? false;
      n.setAttribute('fill', h ? 'none' : col);
      n.setAttribute('fill-opacity', h ? '0' : '0.85');
      n.setAttribute('stroke', col);
      n.setAttribute('stroke-dasharray', h ? '2 1.5' : '');
    }
  }

  /** Highlight a set of tokens; everything else is dimmed when dimOthers. */
  select(tokens: Set<string> | null, dimOthers = true) {
    for (const [t, n] of this.nodes) {
      n.classList.toggle('sel', !!tokens?.has(t));
      n.classList.toggle('dim', !!tokens && dimOthers && !tokens.has(t));
    }
    // small objects far away are hard to see: give each selected one a halo
    this.gOver.querySelectorAll('.halo').forEach((h) => h.remove());
    if (!tokens) return;
    for (const o of this.objects) {
      if (!tokens.has(o.token)) continue;
      const [sx, sy] = toScreen(o.x, o.y);
      el('circle', { class: 'halo', cx: sx, cy: sy, r: Math.max(o.l, o.w) / 2 + this.range * 0.017 }, this.gOver);
    }
  }

  hot(token: string | null) {
    for (const [t, n] of this.nodes) n.classList.toggle('hot', t === token);
  }

  /** Light up one or all sectors around an origin (null = ego). */
  sectors(origin: SceneObject | null, only: string | null = null, labels = true, layout: Layout = 'nuscenes_qa') {
    this.gSectors.replaceChildren();
    const ox = origin?.x ?? 0, oy = origin?.y ?? 0;
    const far = this.range * 3;
    const wedges: [string, number, number][] = layout === 'nuscenes_qa_as_labelled'
      ? AS_LABELLED_WEDGES
      : Object.entries(SECTORS).map(([name, [lo, hi]]) => [name, lo, hi]);
    for (const [name, lo, hi] of wedges) {
      if (only && name !== only) continue;
      const end = hi > lo ? hi : hi + 360;
      const pts: [number, number][] = [toScreen(ox, oy)];
      for (let a = lo; a <= end + 1e-9; a += 2) {
        const r = (a * Math.PI) / 180;
        pts.push(toScreen(ox + far * Math.cos(r), oy + far * Math.sin(r)));
      }
      const wedge = el('polygon', { class: 'sector', points: pts.map((p) => p.join(',')).join(' ') }, this.gSectors);
      if (!only) wedge.setAttribute('fill-opacity', String(Object.keys(SECTORS).indexOf(name) % 2 ? 0.55 : 0.15));
      if (labels) {
        const mid = (((lo + end) / 2) * Math.PI) / 180;
        const lr = this.range * 0.78;
        const [lx, ly] = toScreen(ox + lr * Math.cos(mid), oy + lr * Math.sin(mid));
        const t = el('text', { class: 'sector-label', x: lx, y: ly, 'text-anchor': 'middle', 'font-size': this.fs }, this.gSectors);
        t.textContent = name.replace('_', ' ');
      }
    }
  }

  clearSectors() { this.gSectors.replaceChildren(); }

  ring(o: SceneObject | null) {
    this.gOver.querySelectorAll('.anchor-ring').forEach((n) => n.remove());
    if (!o) return;
    const [sx, sy] = toScreen(o.x, o.y);
    el('circle', { class: 'anchor-ring', cx: sx, cy: sy, r: Math.max(o.l, o.w) / 2 + 1.6 }, this.gOver);
  }

  label(o: SceneObject | null, text = '') {
    this.gOver.querySelectorAll('.tip').forEach((n) => n.remove());
    if (!o) return;
    const [sx, sy] = toScreen(o.x, o.y);
    const t = el('text', { class: 'tip', x: sx + Math.max(o.l, o.w) / 2 + 1, y: sy - 1, 'font-size': this.fs }, this.gOver);
    t.textContent = text;
  }

  node(token: string) { return this.nodes.get(token); }

  /** Draw a LiDAR sweep as a raster under the boxes (thousands of SVG circles would be slow). */
  points(sweep: Sweep | null, dark: boolean) {
    this.gPoints.replaceChildren();
    if (!sweep) return;
    const E = 72, PX = 9; // extent in metres, pixels per metre
    const size = 2 * E * PX;
    const c = document.createElement('canvas');
    c.width = c.height = size;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(size, size);
    for (let i = 0; i < sweep.n; i++) {
      const x = sweep.xyz[i * 3], y = sweep.xyz[i * 3 + 1], z = sweep.xyz[i * 3 + 2];
      const px = Math.round((-y + E) * PX), py = Math.round((-x + E) * PX);
      if (px < 0 || py < 0 || px >= size || py >= size) continue;
      const [r, g, b, a] = heightTone(z, dark);
      for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
        const qx = px + dx, qy = py + dy;
        if (qx >= size || qy >= size) continue;
        const o = (qy * size + qx) * 4;
        if (img.data[o + 3] > a * 255) continue;
        img.data[o] = r * 255; img.data[o + 1] = g * 255; img.data[o + 2] = b * 255; img.data[o + 3] = a * 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    el('image', { href: c.toDataURL(), x: -E, y: -E, width: 2 * E, height: 2 * E, preserveAspectRatio: 'none', style: 'image-rendering: auto' }, this.gPoints);
  }
}
