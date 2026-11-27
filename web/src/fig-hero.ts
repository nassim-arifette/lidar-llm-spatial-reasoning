import { loadSweep } from './lidar';
import { isDark } from './theme';
import { Bev } from './bev';
import { LAYOUT, viewOf, type Study } from './data';
import { execute, parseProgram, programText } from './executor';
import { onScreen, reducedMotion } from './util';

/**
 * Qwen2.5-1.5B evaluation questions, scored as the released answers are: direct-2 answers and the
 * archived programs executed under the as-labelled layout. The last one is a failure of the program mode.
 */
const PICKS = ['q_a9e29aab8a21', 'q_386fc3d66a60', 'q_c2df9aa1c9a0', 'q_16a7af6ca052'];
const LAYOUT_USED = LAYOUT.as_labelled;

export function initHero(study: Study) {
  const root = document.getElementById('top')!;
  const bev = new Bev(document.getElementById('hero-bev') as unknown as SVGSVGElement, { aspect: 1 });
  const q = (r: string) => root.querySelector<HTMLElement>(`[data-role=${r}]`)!;
  const run = study.runs[0];
  const rows = PICKS.map((id) => run.rows.find((r) => r.question_id === id)!).filter(Boolean);
  let index = 0;
  let token = 0;
  let visible = true;
  onScreen(root, (v) => { visible = v; });

  const sleep = (ms: number, t: number) => new Promise<boolean>((res) => {
    const start = performance.now();
    const tick = () => {
      if (t !== token) return res(false);
      if (!visible || document.hidden) return setTimeout(tick, 250);
      if (performance.now() - start >= ms) return res(true);
      setTimeout(tick, Math.min(60, ms));
    };
    tick();
  });
  const answer = (node: HTMLElement, text: string, ok: boolean) => {
    const m = document.createElement('span');
    m.className = `mark ${ok ? 'ok' : 'bad'}`;
    m.textContent = ok ? '✓ right' : '✗ wrong';
    node.replaceChildren(document.createTextNode(`${text} `), m);
  };
  const NOTE: Record<string, string> = {
    q_a9e29aab8a21: ' The cars are on both sides: in the released answers, “front left” means either front diagonal.',
    q_16a7af6ca052: ' The program found the right pedestrians, but it answered “how many” to a yes-or-no question.',
  };

  async function play(i: number) {
    const t = ++token;
    const row = rows[i];
    const objects = study.scenes[row.sample_token].objects;
    const program = parseProgram(row.program_raw ?? '');
    const res = program ? execute(program, objects, LAYOUT_USED) : null;
    const view = viewOf(row, 'as_labelled');
    const instant = reducedMotion();
    bev.setScene(objects, true, res ? [...res.selected, ...(res.anchor ? [res.anchor] : [])] : []);
    bev.paint(() => 'var(--obj)');
    bev.select(null);
    loadSweep(row.sample_token).then((sw) => { if (t === token) bev.points(sw, isDark()); });
    bev.clearSectors();
    bev.ring(null);
    for (const k of ['direct', 'program', 'executed', 'reference']) q(k).textContent = ' ';

    const typeInto = async (node: HTMLElement, text: string, speed: number) => {
      if (instant) { node.textContent = text; return true; }
      for (let c = 1; c <= text.length; c++) {
        node.textContent = text.slice(0, c);
        if (!(await sleep(speed, t))) return false;
      }
      return true;
    };
    if (!(await typeInto(q('question'), row.question, 26))) return;
    if (!instant && !(await sleep(450, t))) return;
    answer(q('direct'), view.direct ?? '—', view.directOk);
    if (!instant && !(await sleep(700, t))) return;
    if (!(await typeInto(q('program'), program ? programText(program) : row.program, 14))) return;
    if (!instant && !(await sleep(300, t))) return;
    if (res && program) {
      const dir = program.target.direction;
      if (res.anchor) bev.ring(res.anchor);
      if (dir) bev.sectors(res.anchor ?? null, dir, false, LAYOUT_USED);
      const chosen = new Set<string>();
      bev.select(chosen);
      for (const o of res.selected) {
        chosen.add(o.token);
        bev.select(chosen);
        q('executed').textContent = program.op === 'count' ? String(chosen.size) : '…';
        if (!instant && !(await sleep(140, t))) return;
      }
      if (!res.selected.length) bev.select(new Set());
    }
    answer(q('executed'), view.executed ?? view.status, view.executedOk);
    q('reference').textContent = `Reference answer: ${row.reference_answer}.${NOTE[row.question_id] ?? ''}`;
    if (instant) return;
    if (await sleep(4200, t)) { index = (index + 1) % rows.length; play(index); }
  }

  q('next').addEventListener('click', () => { index = (index + 1) % rows.length; play(index); });
  play(0);
}
