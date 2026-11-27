import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { execute, inDirection, parseProgram, programText, sectorOf } from '../src/executor';
import { LAYOUT, viewOf, type Convention, type Row } from '../src/data';

const study = JSON.parse(readFileSync(new URL('../public/data/study.json', import.meta.url), 'utf8'));

describe('TypeScript executor reproduces every archived program', () => {
  for (const run of study.runs) {
    it(run.id, () => {
      let checked = 0;
      for (const row of run.rows) {
        if (row.program_status !== 'ok') continue;
        const program = parseProgram(row.program_raw);
        if (!program) continue;
        const objects = study.scenes[row.sample_token].objects;
        const res = execute(program, objects);
        expect(res.status === 'ok' ? res.answer : null, row.question_id).toBe(row.executed_answer);
        if (res.status === 'ok') expect(res.selected.map((o) => o.token), row.question_id).toEqual(row.selected);
        expect(programText(program), row.question_id).toBe(row.program);
        checked++;
      }
      expect(checked).toBeGreaterThan(150);
    });
  }
});

/**
 * The same programs under the layout the released answers follow, against the Python
 * re-execution saved by spatial_reasoning.rescore_programs. study.json holds 102 of the
 * evaluation keyframes; rows on the others are skipped and the count is asserted.
 */
describe('TypeScript executor matches the Python re-execution under the as-labelled layout', () => {
  const root = new URL('../../spatial_reasoning/results/rescored_as_labelled/', import.meta.url);
  for (const run of study.runs) {
    it(run.id, () => {
      const rows = readFileSync(new URL(`${run.id}/rows_eval.jsonl`, root), 'utf8')
        .split('\n').filter(Boolean).map((line) => JSON.parse(line));
      let checked = 0, changed = 0;
      for (const row of rows) {
        const scene = study.scenes[row.sample_token];
        if (!scene || row.program_status !== 'ok') continue;
        const program = parseProgram(row.program_raw);
        if (!program) continue;
        const res = execute(program, scene.objects, 'nuscenes_qa_as_labelled');
        expect(res.status === 'ok' ? res.answer : null, row.question_id).toBe(row.executed_answer);
        if (res.status === 'ok') expect(res.selected.map((o) => o.token).slice(0, 40), row.question_id).toEqual(row.selected);
        const signed = execute(program, scene.objects);
        if ((signed.status === 'ok' ? signed.answer : null) !== row.executed_answer) changed++;
        checked++;
      }
      expect(checked).toBeGreaterThan(150);
      expect(changed).toBeGreaterThan(10);
    });
  }
});

describe('as-labelled layout', () => {
  const at = (deg: number, d = 20) => ({ x: d * Math.cos((deg * Math.PI) / 180), y: d * Math.sin((deg * Math.PI) / 180) }) as any;
  it('ignores the side', () => {
    for (const [deg, name] of [[60, 'front_left'], [-60, 'front_left'], [120, 'back_right'], [-120, 'back_right'], [10, 'front'], [-170, 'back']] as const) {
      expect(sectorOf(at(deg), null, 'nuscenes_qa_as_labelled')).toBe(name);
    }
    expect(sectorOf(at(-60))).toBe('front_right');
    expect(sectorOf(at(120))).toBe('back_left');
  });
  it('leaves front right and back left empty', () => {
    for (let deg = -179; deg <= 180; deg += 7) {
      expect(inDirection(at(deg), 'front_right', null, 'nuscenes_qa_as_labelled')).toBe(false);
      expect(inDirection(at(deg), 'back_left', null, 'nuscenes_qa_as_labelled')).toBe(false);
    }
  });
});

/**
 * What the page shows under each convention. study.json carries the Python outcome of every
 * archived program under the as-labelled layout; the TypeScript executor has to give the same
 * answer on all 200 questions of each run, and the totals have to be those of audit.json.
 */
describe('both conventions, as the page shows them', () => {
  const audit = JSON.parse(readFileSync(new URL('../public/data/audit.json', import.meta.url), 'utf8'));
  for (const run of study.runs) {
    it(`${run.id}: executed answers and totals`, () => {
      let changed = 0;
      for (const row of run.rows) {
        if (row.executed_answer !== row.executed_answer_as_labelled) changed++;
        if (row.program_status !== 'ok') {
          expect(row.executed_correct_as_labelled, row.question_id).toBe(false);
          continue;
        }
        const program = parseProgram(row.program_raw);
        if (!program) continue;
        const res = execute(program, study.scenes[row.sample_token].objects, LAYOUT.as_labelled);
        expect(res.status === 'ok' ? res.answer : null, row.question_id).toBe(row.executed_answer_as_labelled);
        expect(res.status, row.question_id).toBe(row.execution_status_as_labelled);
      }
      expect(changed).toBeGreaterThan(10);
      const entry = audit.archive.find((a: any) => a.id === run.id);
      const share = (c: Convention, side: 'directOk' | 'executedOk') => run.rows.filter((r: Row) => viewOf(r, c)[side]).length / run.rows.length;
      expect(share('published', 'executedOk')).toBeCloseTo(entry.program_published, 9);
      expect(share('as_labelled', 'executedOk')).toBeCloseTo(entry.program_as_labelled, 9);
      expect(share('published', 'directOk')).toBeCloseTo(entry.direct_1, 9);
      const rerun = audit.prompts.find((p: any) => p.archive === run.id)?.cells.direct_2;
      // direct-2 where it was run; otherwise the direct-1 answer is kept and labelled as such
      expect(share('as_labelled', 'directOk')).toBeCloseTo(rerun ?? entry.direct_1, 9);
      expect(viewOf(run.rows[0], 'as_labelled').directPrompt).toBe(rerun === undefined ? 'direct-1' : 'direct-2');
      expect(viewOf(run.rows[0], 'published').directPrompt).toBe('direct-1');
    });
  }

  it('a question about the front left selects both front diagonals only as labelled', () => {
    const row = study.runs[0].rows.find((r: Row) => r.question_id === 'q_a9e29aab8a21');
    const objects = study.scenes[row.sample_token].objects;
    const program = parseProgram(row.program_raw)!;
    const signed = execute(program, objects, LAYOUT.published);
    const unsigned = execute(program, objects, LAYOUT.as_labelled);
    expect(unsigned.answer).toBe(row.reference_answer);
    expect(signed.answer).not.toBe(row.reference_answer);
    expect(unsigned.selected.some((o) => o.y < 0)).toBe(true);
    expect(signed.selected.every((o) => o.y > 0)).toBe(true);
  });

  it('every quoted number is a formatted string', () => {
    for (const [key, value] of Object.entries(audit.stats)) expect(typeof value, key).toBe('string');
  });
});
