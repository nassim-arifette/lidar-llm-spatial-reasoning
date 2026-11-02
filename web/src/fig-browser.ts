import { loadSweep } from './lidar';
import { isDark } from './theme';
import { Bev } from './bev';
import { CATEGORY_LABEL, FAMILY_LABEL, type Row, type Study } from './data';
import { execute, parseProgram } from './executor';
import { onThemeChange } from './util';

type Quad = 'both' | 'direct_only' | 'executed_only' | 'neither' | '';
const quadOf = (r: Row): Quad => (r.direct_correct ? (r.executed_correct ? 'both' : 'direct_only') : (r.executed_correct ? 'executed_only' : 'neither'));

/** Figure 4: every archived evaluation answer, with a 2×2 paired filter and the scene of the chosen question. */
export function initBrowser(study: Study) {
  const fig = document.getElementById('fig-browser')!;
  const runSel = fig.querySelector<HTMLSelectElement>('[data-ctl=run]')!;
  const famSel = fig.querySelector<HTMLSelectElement>('[data-ctl=family]')!;
  const quadEl = fig.querySelector<HTMLElement>('[data-role=quad]')!;
  const list = fig.querySelector<HTMLElement>('[data-role=list]')!;
  const detail = fig.querySelector<HTMLElement>('[data-role=detail]')!;
  const bev = new Bev(fig.querySelector('[data-role=bev]') as unknown as SVGSVGElement, { aspect: 1.2 });
  study.runs.forEach((r, i) => runSel.add(new Option(r.label, String(i))));
  famSel.add(new Option('All families', ''));
  for (const [k, v] of Object.entries(FAMILY_LABEL)) famSel.add(new Option(v, k));

  let runIdx = 0;
  let quad: Quad = '';
  let current: Row | null = null;

  function drawQuad() {
    const rows = study.runs[runIdx].rows.filter((r) => !famSel.value || r.family === famSel.value);
    const n = (q: Quad) => rows.filter((r) => quadOf(r) === q).length;
    quadEl.replaceChildren();
    const cell = (text: string) => { const d = document.createElement('div'); d.className = 'qh'; d.textContent = text; quadEl.append(d); };
    const btn = (q: Quad, label: string) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(quad === q));
      const big = document.createElement('b');
      big.textContent = String(n(q));
      b.append(big, label);
      b.addEventListener('click', () => { quad = quad === q ? '' : q; drawQuad(); drawList(); });
      quadEl.append(b);
    };
    cell(''); cell('program right'); cell('program wrong');
    cell('direct right'); btn('both', 'both right'); btn('direct_only', 'only direct');
    cell('direct wrong'); btn('executed_only', 'only program'); btn('neither', 'both wrong');
  }

  function drawList() {
    const rows = study.runs[runIdx].rows.filter((r) => (!famSel.value || r.family === famSel.value) && (!quad || quadOf(r) === quad));
    list.replaceChildren();
    const head = document.createElement('div');
    head.className = 'qrow head';
    for (const t of [`Question (${rows.length})`, 'Reference', 'Direct', 'Program']) { const s = document.createElement('span'); s.textContent = t; head.append(s); }
    list.append(head);
    for (const r of rows) {
      const row = document.createElement('div');
      row.className = 'qrow';
      row.setAttribute('role', 'listitem');
      row.tabIndex = 0;
      if (current?.question_id === r.question_id) row.setAttribute('aria-current', 'true');
      const q = document.createElement('span'); q.className = 'q'; q.textContent = r.question;
      const ref = document.createElement('span'); ref.className = 'a'; ref.textContent = r.reference_answer;
      row.append(q, ref, answerCell(r.direct_answer ?? '—', r.direct_correct), answerCell(r.executed_answer ?? (r.program_status !== 'ok' ? 'invalid' : r.execution_status.replace('_', ' ')), r.executed_correct));
      const open = () => { current = r; list.querySelectorAll('[aria-current]').forEach((n) => n.removeAttribute('aria-current')); row.setAttribute('aria-current', 'true'); show(r); };
      row.addEventListener('click', open);
      row.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
      list.append(row);
    }
    if (!current || !rows.includes(current)) { current = rows[0] ?? null; if (current) { list.querySelector('.qrow:not(.head)')?.setAttribute('aria-current', 'true'); show(current); } }
  }

  function show(r: Row) {
    const objects = study.scenes[r.sample_token].objects;
    const p = r.program_raw && r.program_status === 'ok' ? safeParse(r.program_raw) : null;
    const res = p ? execute(p, objects) : null;
    bev.setScene(objects, true, res ? [...res.selected, ...(res.anchor ? [res.anchor] : [])] : []);
    bev.paint(() => 'var(--obj)');
    bev.clearSectors();
    loadSweep(r.sample_token).then((sw) => { if (current === r) bev.points(sw, isDark()); });
    bev.ring(null);
    if (p && res) {
      if (res.anchor) bev.ring(res.anchor);
      if (p.target.direction && res.status === 'ok') bev.sectors(res.anchor ?? null, p.target.direction, false);
      bev.select(new Set(res.selected.map((o) => o.token)));
    } else bev.select(null);
    detail.replaceChildren();
    const add = (k: string, v: string, code = false) => {
      const d = document.createElement('div');
      d.style.marginBottom = '0.5rem';
      const b = document.createElement('b'); b.textContent = k;
      d.append(b);
      const val = document.createElement(code ? 'code' : 'span');
      val.textContent = code ? v : ` ${v}`;
      d.append(val);
      detail.append(d);
    };
    add('Question:', r.question);
    add('Program:', r.program || (r.program_status === 'ok' ? '(refused: not expressible)' : `invalid output: ${(r.program_raw ?? '').slice(0, 160)}`), true);
    if (res && res.status !== 'ok') add('Executor:', `${res.status.replace('_', ' ')}, ${res.reason}`);
    else if (res) add('Executor:', `${res.selected.length} selected${res.anchor ? `, anchor ${res.anchor.id} (${CATEGORY_LABEL(res.anchor.category)})` : ''}`);
    if (r.direct_raw && r.direct_raw.trim() !== (r.direct_answer ?? '')) add('Direct output:', r.direct_raw.slice(0, 220) + (r.direct_raw.length > 220 ? '…' : ''));
  }

  runSel.addEventListener('change', () => { runIdx = +runSel.value; current = null; drawQuad(); drawList(); });
  famSel.addEventListener('change', () => { drawQuad(); drawList(); });
  onThemeChange(() => current && show(current));
  drawQuad();
  drawList();
}

function answerCell(text: string, ok: boolean) {
  const s = document.createElement('span');
  s.className = 'a';
  const m = document.createElement('span');
  m.className = ok ? 'ok-mark' : 'bad-mark';
  m.textContent = ok ? '✓ ' : '✗ ';
  s.append(m, text);
  return s;
}

function safeParse(text: string) {
  try { return parseProgram(text); } catch { return null; }
}
