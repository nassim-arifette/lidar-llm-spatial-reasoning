import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { execute, parseProgram, programText } from '../src/executor';

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
