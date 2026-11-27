/**
 * TypeScript port of spatial_reasoning/dsl.py, regions.py (both nuScenes-QA layouts) and the
 * program validation of llm_parser.query_from_json. Parity with the archived Python
 * execution is tested in test/executor.test.ts on every saved program.
 *
 * Coordinates are float32 in the Python code; differences of coordinates are float32
 * too (NumPy scalar arithmetic), while angles are computed in float64. Math.fround
 * reproduces that, so objects on a sector boundary land on the same side.
 */
export interface SceneObject {
  id: string; token: string; category: string; status: string | null;
  x: number; y: number; z: number; w: number; l: number; h: number; yaw: number;
  lidar_pts: number | null; px: number; py: number;
}

export const CATEGORIES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'] as const;
export const GROUPS: Record<string, readonly string[]> = {
  vehicle: ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'motorcycle', 'bicycle'],
  four_wheeler: ['car', 'truck', 'bus', 'trailer', 'construction_vehicle'],
  two_wheeler: ['motorcycle', 'bicycle'],
  person: ['pedestrian'],
  object: CATEGORIES,
};
export const STATUSES = ['moving', 'parked', 'stopped', 'standing', 'sitting lying down', 'with rider', 'without rider'] as const;

/** nuScenes-QA, eq. 2: six 60-degree relations, half-open (low, high] in degrees, CCW from ahead. */
export const SECTORS: Record<string, [number, number]> = {
  front: [-30, 30],
  front_left: [30, 90],
  back_left: [90, 150],
  back: [150, -150],
  back_right: [-150, -90],
  front_right: [-90, -30],
};
export const DIRECTIONS = Object.keys(SECTORS);

/**
 * 'nuscenes_qa' is the definition as published (signed sectors). The released answers follow
 * 'nuscenes_qa_as_labelled' instead: the test is on the unsigned angle a = |bearing|, so both
 * front diagonals are "front left", both rear diagonals are "back right", and "front right" and
 * "back left" hold nothing. Port of regions.sector_contains.
 */
export type Layout = 'nuscenes_qa' | 'nuscenes_qa_as_labelled';
export const UNSIGNED_INTERVALS: Record<string, [number, number] | null> = {
  front: [-1, 30],
  front_left: [30, 90],
  front_right: null,
  back_right: [90, 150],
  back_left: null,
  back: [150, 180],
};
export function sectorContains(name: string, t: number, layout: Layout = 'nuscenes_qa'): boolean {
  if (layout === 'nuscenes_qa_as_labelled') {
    const u = UNSIGNED_INTERVALS[name];
    return u !== null && u[0] < Math.abs(t) && Math.abs(t) <= u[1];
  }
  const s = SECTORS[name];
  return inInterval(t, s[0], s[1]);
}
const MIN_RANGE = 1.0;
const ALIASES: Record<string, string> = {
  ahead: 'front', forward: 'front', in_front: 'front', behind: 'back', rear: 'back',
  left_front: 'front_left', right_front: 'front_right', left_back: 'back_left', rear_left: 'back_left',
  right_back: 'back_right', rear_right: 'back_right',
};

export type Op = 'count' | 'exists' | 'identify' | 'status' | 'same_status' | 'compare_count';
export interface Selector {
  category: string;
  status: string | null;
  negate_status: boolean;
  direction: string | null;
  anchor: Selector | null;
}
export interface Program { op: Op; target: Selector; second: Selector | null; comparison: 'more' | 'fewer' | null }
export type ExecStatus = 'ok' | 'ambiguous' | 'missing' | 'unsupported' | 'no_status';
export interface Result { answer: string | null; status: ExecStatus; selected: SceneObject[]; reason: string; anchor?: SceneObject | null }

const f32 = Math.fround;
const DEG = 180 / Math.PI;

export function normaliseDegrees(a: number): number {
  const w = ((((a + 180) % 360) + 360) % 360) - 180;
  return w === -180 ? 180 : w;
}
function inInterval(t: number, lo: number, hi: number) {
  return lo < hi ? lo < t && t <= hi : t > lo || t <= hi;
}

/** Bearing in degrees of obj seen from origin (ego heading kept), or null when closer than min_range. */
export function bearingFrom(obj: SceneObject, origin?: SceneObject | null): number | null {
  const dx = origin ? f32(obj.x - origin.x) : f32(obj.x - 0);
  const dy = origin ? f32(obj.y - origin.y) : f32(obj.y - 0);
  if (Math.hypot(dx, dy) < MIN_RANGE) return null;
  return normaliseDegrees(Math.atan2(dy, dx) * DEG);
}

export function inDirection(obj: SceneObject, direction: string, origin?: SceneObject | null, layout: Layout = 'nuscenes_qa'): boolean {
  const name = ALIASES[direction] ?? direction;
  if (!SECTORS[name]) throw new Error(`direction '${direction}' does not exist in layout '${layout}'`);
  const t = bearingFrom(obj, origin);
  return t !== null && sectorContains(name, t, layout);
}

export function sectorOf(obj: SceneObject, origin?: SceneObject | null, layout: Layout = 'nuscenes_qa'): string | null {
  const t = bearingFrom(obj, origin);
  if (t === null) return null;
  for (const name of Object.keys(SECTORS)) if (sectorContains(name, t, layout)) return name;
  return null;
}

export function expandCategory(name: string): readonly string[] {
  const key = name.trim().toLowerCase().replace(/ /g, '_');
  if (GROUPS[key]) return GROUPS[key];
  return (CATEGORIES as readonly string[]).includes(key) ? [key] : [];
}

function matches(o: SceneObject, s: Selector): boolean {
  if (s.category !== 'object' && s.category !== 'thing') {
    if (!expandCategory(s.category).includes(o.category)) return false;
  }
  if (s.status) {
    if (o.status === null) return false;
    return s.negate_status ? o.status !== s.status : o.status === s.status;
  }
  return true;
}

export function describe(s: Selector): string {
  const bits: string[] = [];
  if (s.status) bits.push((s.negate_status ? 'not ' : '') + s.status);
  bits.push(s.category);
  let text = bits.join(' ');
  if (s.direction) {
    text += s.anchor ? ` ${s.direction.replace('_', ' ')} of [${describe(s.anchor)}]` : ` ${s.direction.replace('_', ' ')} of the ego vehicle`;
  }
  return text;
}

interface Resolved { objs: SceneObject[]; status: ExecStatus; reason: string; anchor: SceneObject | null }

export function resolve(s: Selector, objects: SceneObject[], depth = 0, layout: Layout = 'nuscenes_qa'): Resolved {
  if (depth > 3) return { objs: [], status: 'unsupported', reason: 'anchor chain too deep', anchor: null };
  if (s.direction && !SECTORS[ALIASES[s.direction] ?? s.direction]) {
    return { objs: [], status: 'unsupported', reason: `direction '${s.direction}' does not exist in layout '${layout}'`, anchor: null };
  }
  let candidates = objects.filter((o) => matches(o, s));
  let reference: SceneObject | null = null;
  if (s.anchor) {
    const a = resolve(s.anchor, objects, depth + 1, layout);
    if (a.status !== 'ok') return { ...a, objs: [] };
    if (!a.objs.length) return { objs: [], status: 'missing', reason: `no ${describe(s.anchor)}`, anchor: null };
    if (a.objs.length > 1) return { objs: [], status: 'ambiguous', reason: `${a.objs.length} objects match [${describe(s.anchor)}]`, anchor: null };
    reference = a.objs[0];
    candidates = candidates.filter((o) => o !== reference);
  }
  if (s.direction) {
    const dir = s.direction;
    candidates = reference
      ? candidates.filter((o) => o !== reference && inDirection(o, dir, reference, layout))
      : candidates.filter((o) => inDirection(o, dir, null, layout));
  }
  return { objs: candidates, status: 'ok', reason: '', anchor: reference };
}

export function execute(p: Program, objects: SceneObject[], layout: Layout = 'nuscenes_qa'): Result {
  const t = resolve(p.target, objects, 0, layout);
  if (t.status !== 'ok') return { answer: null, status: t.status, selected: [], reason: t.reason };
  const objs = t.objs;
  const done = (answer: string | null, status: ExecStatus, selected: SceneObject[], reason = ''): Result =>
    ({ answer, status, selected, reason, anchor: t.anchor });
  if (p.op === 'count') return done(String(objs.length), 'ok', objs);
  if (p.op === 'exists') return done(objs.length ? 'yes' : 'no', 'ok', objs);
  if (p.op === 'identify' || p.op === 'status') {
    if (!objs.length) return done(null, 'missing', [], `no ${describe(p.target)}`);
    if (objs.length > 1) return done(null, 'ambiguous', objs, `${objs.length} objects match [${describe(p.target)}]`);
    const o = objs[0];
    if (p.op === 'identify') return done(o.category.replace(/_/g, ' '), 'ok', objs);
    if (o.status === null) return done(null, 'no_status', objs, 'the annotation carries no attribute');
    return done(o.status, 'ok', objs);
  }
  if (!p.second) return done(null, 'unsupported', [], 'binary operation without a second selector');
  const u = resolve(p.second, objects, 0, layout);
  if (u.status !== 'ok') return { answer: null, status: u.status, selected: [], reason: u.reason };
  if (p.op === 'compare_count') {
    const yes = p.comparison === 'fewer' ? objs.length < u.objs.length : objs.length > u.objs.length;
    return done(yes ? 'yes' : 'no', 'ok', [...objs, ...u.objs], `${objs.length} vs ${u.objs.length}`);
  }
  if (objs.length !== 1 || u.objs.length !== 1) return done(null, 'ambiguous', [...objs, ...u.objs], `${objs.length} and ${u.objs.length} objects match`);
  const a = objs[0].status, b = u.objs[0].status;
  if (a === null || b === null) return done(null, 'no_status', [...objs, ...u.objs], 'one of the objects carries no attribute');
  return done(a === b ? 'yes' : 'no', 'ok', [...objs, ...u.objs]);
}

/** Validate a model's JSON output like llm_parser.query_from_json. Null = explicit refusal. */
export function parseProgram(text: string): Program | null {
  const m = /\{.*\}/s.exec(text ?? '');
  if (!m) throw new Error('no JSON object in the output');
  const blob = JSON.parse(m[0]);
  const sel = (b: any, depth: number): Selector | null => {
    if (b === null || b === undefined) return null;
    if (depth > 2) throw new Error('anchor nested too deep');
    const category = b.category === null || b.category === undefined ? 'object' : String(b.category).trim().toLowerCase().replace(/ /g, '_');
    const direction = b.direction ? String(b.direction).trim().toLowerCase().replace(/ /g, '_') : null;
    return {
      category,
      status: b.status ? String(b.status).trim().toLowerCase() : null,
      negate_status: b.negate_status === true,
      direction,
      anchor: sel(b.anchor, depth + 1),
    };
  };
  if (blob.op === null) return null;
  const target = sel(blob.target, 0);
  if (!target) throw new Error('missing target selector');
  return { op: String(blob.op).trim().toLowerCase() as Op, target, second: sel(blob.second, 0), comparison: blob.comparison ?? null };
}

export function programText(p: Program): string {
  if (p.op === 'compare_count') return `${p.comparison} [${describe(p.target)}] than [${describe(p.second!)}]`;
  if (p.op === 'same_status') return `same status: [${describe(p.target)}] vs [${describe(p.second!)}]`;
  return `${p.op} [${describe(p.target)}]`;
}
