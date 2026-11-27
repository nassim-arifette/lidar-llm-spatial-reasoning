import { Bev } from './bev';
import { CATEGORY_LABEL, LAYOUT, sampleLabels, type Convention, type Study } from './data';
import { CATEGORIES, DIRECTIONS, execute, parseProgram, programText, STATUSES, type Op, type Program, type Selector } from './executor';
import { onThemeChange, segmented } from './util';

const OPS: Op[] = ['count', 'exists', 'identify', 'status'];
const blank = (): Selector => ({ category: 'object', status: null, negate_status: false, direction: null, anchor: null });

/** Figure 3: build a program in the study's schema and execute it live on a keyframe. */
export function initPlayground(study: Study, startToken: string) {
  const fig = document.getElementById('fig-playground')!;
  const bev = new Bev(fig.querySelector('[data-role=bev]') as unknown as SVGSVGElement);
  const form = fig.querySelector<HTMLElement>('[data-role=form]')!;
  const json = fig.querySelector<HTMLElement>('[data-role=json]')!;
  const result = fig.querySelector<HTMLElement>('[data-role=result]')!;
  const sampleSel = fig.querySelector<HTMLSelectElement>('[data-ctl=sample]')!;
  const presetSel = fig.querySelector<HTMLSelectElement>('[data-ctl=preset]')!;
  const labels = sampleLabels(study);
  for (const [t, l] of labels) sampleSel.add(new Option(l, t));

  let token = startToken;
  let convention: Convention = 'as_labelled';
  let program: Program = { op: 'count', target: { ...blank(), category: 'car', direction: 'front' }, second: null, comparison: null };
  let anchorOn = false;
  let anchor: Selector = { ...blank(), category: 'bus' };

  // presets: real evaluation questions of the 1.5B run with a valid, executable program
  const run = study.runs[0];
  const presets = run.rows.filter((r) => r.program_status === 'ok' && r.program_raw && /"op": "(count|exists)"/.test(r.program_raw)).slice(0, 400);
  const seen = new Set<string>();
  presetSel.add(new Option('Choose…', ''));
  for (const r of presets) {
    if (seen.has(r.question)) continue;
    seen.add(r.question);
    presetSel.add(new Option(r.question, r.question_id));
  }

  const mkSelect = (label: string, values: (string | null)[], value: string | null, on: (v: string | null) => void, fmt = (v: string) => v) => {
    const l = document.createElement('label');
    l.textContent = label;
    const s = document.createElement('select');
    for (const v of values) s.add(new Option(v === null ? 'none' : fmt(v), v ?? ''));
    s.value = value ?? '';
    s.addEventListener('change', () => { on(s.value === '' ? null : s.value); update(); });
    l.append(s);
    return l;
  };
  const mkCheck = (label: string, value: boolean, on: (v: boolean) => void) => {
    const l = document.createElement('label');
    l.className = 'check';
    const c = document.createElement('input');
    c.type = 'checkbox';
    c.checked = value;
    c.addEventListener('change', () => { on(c.checked); buildForm(); update(); });
    l.append(c, label);
    return l;
  };
  const cats = ['object', ...CATEGORIES, 'vehicle', 'person'];

  function selectorFields(title: string, s: Selector, withDirection: boolean) {
    const fs = document.createElement('fieldset');
    const lg = document.createElement('legend');
    lg.textContent = title;
    fs.append(lg);
    fs.append(mkSelect('category', cats, s.category, (v) => { s.category = v ?? 'object'; }, CATEGORY_LABEL));
    fs.append(mkSelect('status', [null, ...STATUSES], s.status, (v) => { s.status = v; }));
    if (withDirection) fs.append(mkSelect('direction', [null, ...DIRECTIONS], s.direction, (v) => { s.direction = v; }, (v) => v.replace('_', ' ')));
    fs.append(mkCheck('negate status', s.negate_status, (v) => { s.negate_status = v; }));
    return fs;
  }

  function buildForm() {
    form.replaceChildren();
    const opWrap = document.createElement('fieldset');
    const lg = document.createElement('legend');
    lg.textContent = 'Operation';
    opWrap.append(lg, mkSelect('op', OPS, program.op, (v) => { program.op = (v ?? 'count') as Op; }));
    form.append(opWrap);
    form.append(selectorFields('Target', program.target, true));
    form.append(mkCheck('measure the direction from an anchor object', anchorOn, (v) => { anchorOn = v; }));
    if (anchorOn) form.append(selectorFields('Anchor', anchor, false));
  }

  function update() {
    program.target.anchor = anchorOn ? anchor : null;
    const blob = { op: program.op, target: strip(program.target), second: null, comparison: null };
    json.textContent = JSON.stringify(blob, null, 2);
    const objects = study.scenes[token].objects;
    const layout = LAYOUT[convention];
    const res = execute(program, objects, layout);
    bev.paint(() => 'var(--obj)');
    bev.clearSectors();
    bev.ring(res.anchor ?? null);
    if (program.target.direction && res.status === 'ok') bev.sectors(res.anchor ?? null, program.target.direction, false, layout);
    bev.select(new Set(res.selected.map((o) => o.token)));
    result.replaceChildren();
    const head = document.createElement('div');
    head.textContent = programText(program);
    head.style.color = 'var(--program)';
    head.style.fontFamily = 'var(--mono)';
    head.style.fontSize = '0.75rem';
    const big = document.createElement('span');
    big.className = 'big';
    const note = document.createElement('div');
    if (res.status === 'ok') {
      big.textContent = res.answer ?? '';
      const never = convention === 'as_labelled' && (program.target.direction === 'front_right' || program.target.direction === 'back_left');
      note.textContent = `${res.selected.length} object${res.selected.length === 1 ? '' : 's'} selected.${never ? ` In the released answers nothing is ever to the ${program.target.direction!.replace('_', ' ')}.` : ''}`;
    } else {
      big.textContent = `refused: ${res.status.replace('_', ' ')}`;
      big.classList.add('refused');
      note.textContent = res.reason;
    }
    result.append(head, big, note);
  }

  sampleSel.value = token;
  segmented(fig.querySelector<HTMLElement>('[data-ctl=convention]')!, (value) => { convention = value as Convention; update(); });
  sampleSel.addEventListener('change', () => { token = sampleSel.value; bev.setScene(study.scenes[token].objects); update(); });
  presetSel.addEventListener('change', () => {
    const row = run.rows.find((r) => r.question_id === presetSel.value);
    if (!row?.program_raw) return;
    const p = parseProgram(row.program_raw);
    if (!p) return;
    program = { ...p, second: null, comparison: null };
    anchorOn = !!p.target.anchor;
    anchor = p.target.anchor ?? { ...blank(), category: 'bus' };
    token = row.sample_token;
    sampleSel.value = token;
    bev.setScene(study.scenes[token].objects);
    buildForm();
    update();
  });
  bev.setScene(study.scenes[token].objects);
  buildForm();
  onThemeChange(update);
  update();
}

function strip(s: Selector): Record<string, unknown> {
  return { category: s.category, status: s.status, negate_status: s.negate_status, direction: s.direction, anchor: s.anchor ? strip(s.anchor) : null };
}
