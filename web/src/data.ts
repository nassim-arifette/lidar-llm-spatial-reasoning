import type { Layout, SceneObject } from './executor';

export interface Row {
  question_id: string; sample_token: string; family: string; question: string; reference_answer: string;
  direct_answer: string | null; direct_raw: string | null; direct_correct: boolean; direct_output_tokens: number | null;
  program_raw: string | null; program: string; program_status: string; executed_answer: string | null;
  execution_status: string; execution_reason: string; executed_correct: boolean; selected: string[] | null;
  program_output_tokens: number | null;
  /** The same program under the layout the released answers follow (Python re-execution). */
  executed_answer_as_labelled: string | null; execution_status_as_labelled: string; executed_correct_as_labelled: boolean;
  /** Direct answer under the corrected prompt, where that run exists. */
  direct2_answer?: string | null; direct2_correct?: boolean;
}
export interface Side {
  accuracy: number; answered_fraction: number;
  count_errors: { mae: number; under: number; over: number; numeric_coverage: number };
  output_tokens?: number;
}
export interface Run {
  id: string; label: string; runtime: string; model: string; direct2_run: string | null;
  summary_as_labelled: { direct: Side | null; executed: Side };
  summary: { direct: Side; executed: Side; paired: Record<string, number>; by_family: Record<string, any>; scene_dependence: Record<string, number> };
  rows: Row[];
}
export interface Study {
  runs: Run[];
  scenes: Record<string, { scene_token: string; objects: SceneObject[] }>;
  layout: Record<string, [number, number]>;
  reexecuted_programs: number;
}

/**
 * Two ways of scoring the same question. 'published': the direct-1 answer (its prompt states
 * the signed sectors) and the program executed with signed sectors. 'as_labelled': the program
 * executed under the rule the released answers follow, and the direct-2 answer where that run
 * exists; otherwise the direct-1 answer is kept and directPrompt says so.
 */
export type Convention = 'published' | 'as_labelled';
export const LAYOUT: Record<Convention, Layout> = { published: 'nuscenes_qa', as_labelled: 'nuscenes_qa_as_labelled' };
export interface View {
  direct: string | null; directOk: boolean; directPrompt: 'direct-1' | 'direct-2';
  executed: string | null; executedOk: boolean; status: string;
}
export function viewOf(r: Row, c: Convention): View {
  if (c === 'published') {
    return { direct: r.direct_answer, directOk: r.direct_correct, directPrompt: 'direct-1', executed: r.executed_answer, executedOk: r.executed_correct, status: r.execution_status };
  }
  const rerun = r.direct2_correct !== undefined;
  return {
    direct: rerun ? r.direct2_answer ?? null : r.direct_answer, directOk: rerun ? !!r.direct2_correct : r.direct_correct,
    directPrompt: rerun ? 'direct-2' : 'direct-1',
    executed: r.executed_answer_as_labelled, executedOk: r.executed_correct_as_labelled, status: r.execution_status_as_labelled,
  };
}
/** Count-error summary of each mode under a convention (direct-1 summary when direct-2 was not run). */
export function sidesOf(run: Run, c: Convention): { direct: Side; executed: Side } {
  if (c === 'published') return { direct: run.summary.direct, executed: run.summary.executed };
  return { direct: run.summary_as_labelled.direct ?? run.summary.direct, executed: run.summary_as_labelled.executed };
}

interface Interval { estimate: number; low: number; high: number }
export interface Audit {
  laterality: { relations: string[]; sets: { id: string; label: string; n_questions: number; n_samples: number; rows: Record<string, { n: number; empty: number }> }[] };
  checks: { label: string; n: number; signed: number; as_labelled: number; indent?: boolean }[];
  archive: { id: string; label: string; n: number; direct_1: number; program_published: number; program_as_labelled: number }[];
  prompts: { key: string; label: string; archive: string; n: number; cells: Record<string, number> }[];
  validation: {
    runs: { key: string; label: string; conventions: Convention; n: number; n_drives: number; direct: Interval; program: Interval; difference: Interval; drives_ahead: number; drives_behind: number; drives_tied: number }[];
  };
  stats: Record<string, string>;
}

let auditPromise: Promise<Audit> | null = null;
export function loadAudit(): Promise<Audit> {
  auditPromise ??= fetch(`${import.meta.env.BASE_URL}data/audit.json`).then((r) => {
    if (!r.ok) throw new Error(`audit.json: HTTP ${r.status}`);
    return r.json();
  });
  return auditPromise;
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
