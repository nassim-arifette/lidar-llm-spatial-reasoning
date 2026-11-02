import type { SceneObject } from './executor';

export interface Row {
  question_id: string; sample_token: string; family: string; question: string; reference_answer: string;
  direct_answer: string | null; direct_raw: string | null; direct_correct: boolean; direct_output_tokens: number | null;
  program_raw: string | null; program: string; program_status: string; executed_answer: string | null;
  execution_status: string; execution_reason: string; executed_correct: boolean; selected: string[] | null;
  program_output_tokens: number | null;
}
export interface Side {
  accuracy: number; answered_fraction: number;
  count_errors: { mae: number; under: number; over: number; numeric_coverage: number };
  output_tokens?: number;
}
export interface Run {
  id: string; label: string; runtime: string; model: string;
  summary: { direct: Side; executed: Side; paired: Record<string, number>; by_family: Record<string, any>; scene_dependence: Record<string, number> };
  rows: Row[];
}
export interface Study {
  runs: Run[];
  scenes: Record<string, { scene_token: string; objects: SceneObject[] }>;
  layout: Record<string, [number, number]>;
  reexecuted_programs: number;
}

let promise: Promise<Study> | null = null;
export function loadStudy(): Promise<Study> {
  promise ??= fetch(`${import.meta.env.BASE_URL}data/study.json`).then((r) => {
    if (!r.ok) throw new Error(`study.json: HTTP ${r.status}`);
    return r.json();
  });
  return promise;
}

export const FAMILY_LABEL: Record<string, string> = {
  count_exist_category: 'Count or existence',
  category_status: 'With a status',
  category_ego_region: 'Direction from the ego vehicle',
  category_anchor_relation: 'Direction from another object',
};

export const CATEGORY_COLOR: Record<string, string> = {
  car: '#5b8db8', truck: '#2f5d8a', bus: '#8f6bb3', trailer: '#b07aa1', construction_vehicle: '#c9a227',
  pedestrian: '#e15759', motorcycle: '#f28e2b', bicycle: '#e7a33c', traffic_cone: '#59a14f', barrier: '#9aa0a8',
};
export const CATEGORY_LABEL = (c: string) => c.replace(/_/g, ' ');

/** Short label for a keyframe: drive letter and index within that drive. */
export function sampleLabels(study: Study): Map<string, string> {
  const drives = new Map<string, string[]>();
  for (const [token, s] of Object.entries(study.scenes)) {
    if (!drives.has(s.scene_token)) drives.set(s.scene_token, []);
    drives.get(s.scene_token)!.push(token);
  }
  const out = new Map<string, string>();
  [...drives.entries()].forEach(([, tokens], d) => {
    tokens.forEach((t, i) => out.set(t, `Drive ${'ABC'[d] ?? d + 1}, keyframe ${i + 1} · ${study.scenes[t].objects.length} objects`));
  });
  return out;
}
