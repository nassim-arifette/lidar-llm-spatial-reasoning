import { axisBottom, scaleLinear, select, type Selection } from 'd3';
import type { Audit } from './data';
import { cssVar, onThemeChange } from './util';

const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)} %`;
const num = (v: number) => v.toLocaleString('en-US');
const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)}`;

function node<K extends keyof HTMLElementTagNameMap>(tag: K, text = '', className = ''): HTMLElementTagNameMap[K] {
  const n = document.createElement(tag);
  if (text) n.textContent = text;
  if (className) n.className = className;
  return n;
}

/** A plain data table. Cells are strings or nodes; numeric columns are right-aligned by the 'num' class. */
function table(caption: string, head: string[], rows: (string | Node)[][], numeric: number[] = []): HTMLTableElement {
  const t = node('table', '', 'data-table');
  t.append(node('caption', caption));
  const hr = t.createTHead().insertRow();
  head.forEach((h, i) => { const th = node('th', h, numeric.includes(i) ? 'num' : ''); th.scope = 'col'; hr.append(th); });
  const body = t.createTBody();
  for (const cells of rows) {
    const tr = body.insertRow();
    cells.forEach((c, i) => {
      const cell = i === 0 ? node('th') : node('td');
      if (i === 0) (cell as HTMLTableCellElement).scope = 'row';
      if (numeric.includes(i)) cell.className = 'num';
      cell.append(c);
      tr.append(cell);
    });
  }
  return t;
}

/** A value with a proportional bar behind it, for share-of-total columns. */
function barCell(share: number, colour: string, strong = false): HTMLElement {
  const wrap = node('span', '', 'bar-cell');
  const bar = node('i');
  bar.style.width = `${Math.max(0, Math.min(1, share)) * 100}%`;
  bar.style.background = colour;
  wrap.append(bar, node(strong ? 'strong' : 'span', pct(share)));
  return wrap;
}

/** How often a relation is answered 0 or no in the official release: no geometry involved. */
export function initLaterality(audit: Audit) {
  const root = document.querySelector<HTMLElement>('#fig-laterality [data-role=table]')!;
  const sets = audit.laterality.sets.filter((s) => s.id !== 'mini');
  const rows = audit.laterality.relations.map((r) => {
    const never = sets.every((s) => s.rows[r].empty === s.rows[r].n);
    const cells: (string | Node)[] = [never ? node('strong', r) : r];
    for (const s of sets) cells.push(num(s.rows[r].n), barCell(s.rows[r].empty / s.rows[r].n, never ? 'var(--bad)' : 'var(--obj)', never));
    return cells;
  });
  const head = ['Relation named in the question', ...sets.flatMap((s) => [`${s.label}: questions`, `${s.label}: answered 0 or no`])];
  root.replaceChildren(table('Counting and existence questions that name exactly one relation, official NuScenes-QA release.', head, rows, [1, 2, 3, 4]));
}

/** Executor agreement with the released answers under each reading of the relations. */
export function initChecks(audit: Audit) {
  const root = document.querySelector<HTMLElement>('#fig-checks [data-role=table]')!;
  const rows = audit.checks.map((c) => {
    const label = node('span', c.label, c.indent ? 'indent' : '');
    return [label, num(c.n), barCell(c.signed, 'var(--obj)'), barCell(c.as_labelled, 'var(--program-soft)', c.as_labelled > c.signed)];
  });
  root.replaceChildren(table('Share of questions on which the executed reference program gives the released answer.',
    ['Check', 'Questions', 'Signed sectors', 'As the answers use them'], rows, [1, 2, 3]));
}

type Svg = Selection<SVGSVGElement, unknown, null, undefined>;
interface Frame { svg: Svg; W: number; narrow: boolean; x: (v: number) => number; rowY: (i: number) => number; left: number }

/** Shared frame of the two dot charts: a percentage axis and one labelled row per entry, redrawn to the container width. */
function frame(svgEl: SVGSVGElement, labels: { title: string; sub: string }[], opts: { left: number; right: number; domain: [number, number]; rowH: number; narrowRowH: number }): Frame {
  const W = Math.max(300, Math.round(svgEl.parentElement!.clientWidth - 26));
  const narrow = W < 560;
  const left = narrow ? 14 : opts.left, right = narrow ? 14 : opts.right;
  const rowH = narrow ? opts.narrowRowH : opts.rowH;
  const top = 12, bottom = 34;
  const H = top + rowH * labels.length + bottom;
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const svg = select(svgEl);
  svg.selectAll('*').remove();
  const scale = scaleLinear().domain(opts.domain).range([left, W - right]);
  const axis = svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - bottom})`)
    .call(axisBottom(scale).ticks(narrow ? 4 : 7).tickSize(-(H - bottom - top)).tickFormat((d) => `${Math.round(+d * 100)} %`) as any);
  axis.select('.domain').remove();
  axis.selectAll('.tick line').attr('stroke-dasharray', '2 3');
  const rowY = (i: number) => top + rowH * i + (narrow ? rowH * 0.6 : rowH / 2);
  labels.forEach((l, i) => {
    const y = rowY(i);
    if (narrow) {
      const t = svg.append('text').attr('x', left).attr('y', y - rowH * 0.36).attr('font-size', 12.5).attr('fill', cssVar('--ink')).attr('font-weight', 600).text(l.title);
      t.append('tspan').attr('font-weight', 400).attr('fill', cssVar('--muted')).attr('font-size', 11).text(`  ${l.sub}`);
    } else {
      svg.append('text').attr('x', left - 16).attr('y', y - 2).attr('text-anchor', 'end').attr('font-size', 13).attr('fill', cssVar('--ink')).attr('font-weight', 550).text(l.title);
      svg.append('text').attr('x', left - 16).attr('y', y + 12).attr('text-anchor', 'end').attr('font-size', 10.5).attr('fill', cssVar('--muted')).text(l.sub);
    }
  });
  return { svg, W, narrow, x: scale, rowY, left };
}

function dot(svg: Svg, cx: number, cy: number, colour: string, filled: boolean, title: string) {
  svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', filled ? 6.5 : 5.5)
    .attr('fill', filled ? colour : cssVar('--surface')).attr('stroke', colour).attr('stroke-width', filled ? 0 : 2)
    .append('title').text(title);
}

function legend(root: HTMLElement, items: [string, string, boolean][]) {
  root.replaceChildren();
  for (const [text, colour, filled] of items) {
    const s = node('span');
    const i = node('i');
    i.style.borderRadius = '50%';
    i.style.background = filled ? colour : 'transparent';
    i.style.boxShadow = `inset 0 0 0 2px ${colour}`;
    s.append(i, text);
    root.append(s);
  }
}

function redrawOnResize(el: Element, draw: () => void) {
  let width = 0;
  new ResizeObserver((entries) => {
    const w = Math.round(entries[0].contentRect.width);
    if (w !== width) { width = w; draw(); }
  }).observe(el);
  onThemeChange(draw);
}

const SUB: Record<string, string> = {
  llm_modes: 'local, fp16', llm_modes_7b: 'local, fp16', llm_modes_32b: 'local, 4-bit quantised',
  gw_qwen3_32b: 'hosted endpoint', gw_qwen3_next_80b: 'hosted endpoint, mixture of experts', gw_qwen3p8_27b: 'hosted endpoint, output cap not respected',
};

/** The six archived models: each mode under the published convention and under the one the answers follow. */
export function initModels(audit: Audit) {
  const fig = document.getElementById('fig-models')!;
  const svgEl = fig.querySelector<SVGSVGElement>('[data-role=chart]')!;
  const direct2 = new Map(audit.prompts.filter((p) => p.cells.direct_2 !== undefined).map((p) => [p.archive, p.cells.direct_2]));
  const rows = audit.archive.map((a) => ({ ...a, direct_2: direct2.get(a.id) }));

  function draw() {
    const { svg, narrow, x, rowY } = frame(svgEl, rows.map((r) => ({ title: r.label, sub: SUB[r.id] ?? '' })),
      { left: 232, right: 112, domain: [0.35, 1], rowH: 46, narrowRowH: 66 });
    const direct = cssVar('--direct'), program = cssVar('--program');
    rows.forEach((r, i) => {
      const y = rowY(i);
      const dMax = r.direct_2 ?? r.direct_1;
      svg.append('line').attr('x1', x(Math.min(r.direct_1, dMax))).attr('x2', x(r.program_as_labelled)).attr('y1', y).attr('y2', y)
        .attr('stroke', cssVar('--rule')).attr('stroke-width', 4).attr('stroke-linecap', 'round');
      dot(svg, x(r.direct_1), y, direct, false, `${r.label}, direct-1 (published convention stated): ${pct(r.direct_1)}`);
      if (r.direct_2 !== undefined) dot(svg, x(r.direct_2), y, direct, true, `${r.label}, direct-2 (convention as the answers use it): ${pct(r.direct_2)}`);
      dot(svg, x(r.program_published), y, program, false, `${r.label}, program executed as published: ${pct(r.program_published)}`);
      dot(svg, x(r.program_as_labelled), y, program, true, `${r.label}, same program executed as the answers use the relations: ${pct(r.program_as_labelled)}`);
      const dText = r.direct_2 !== undefined ? `${(r.direct_1 * 100).toFixed(1)} → ${pct(r.direct_2)}` : pct(r.direct_1);
      const pText = `${(r.program_published * 100).toFixed(1)} → ${pct(r.program_as_labelled)}`;
      const label = (text: string, colour: string, xx: number, yy: number, anchor: string) =>
        svg.append('text').attr('x', xx).attr('y', yy).attr('text-anchor', anchor).attr('font-size', 11.5).attr('fill', colour).attr('font-weight', 600).text(text);
      if (narrow) {
        label(dText, direct, Math.max(x(r.direct_1) - 6, 14), y - 12, 'start');
        label(pText, program, x(r.program_as_labelled) + 7, y + 21, 'end');
      } else {
        label(dText, direct, x(Math.min(r.direct_1, dMax)) - 12, y + 4, 'end');
        label(pText, program, x(r.program_as_labelled) + 12, y + 4, 'start');
      }
    });
    legend(fig.querySelector<HTMLElement>('[data-role=legend]')!, [
      ['direct-1: published convention stated', direct, false], ['direct-2: convention as the answers use it', direct, true],
      ['program, executed as published', program, false], ['same program, executed as the answers use it', program, true],
    ]);
  }
  svgEl.setAttribute('aria-label', `Accuracy of ${rows.length} models on the same ${rows[0].n} questions under two conventions. `
    + rows.map((r) => `${r.label}: direct ${pct(r.direct_1)}${r.direct_2 !== undefined ? ` then ${pct(r.direct_2)}` : ''}, program ${pct(r.program_published)} then ${pct(r.program_as_labelled)}`).join('. ') + '.');
  fig.querySelector<HTMLElement>('[data-role=table]')!.replaceChildren(table(
    `Accuracy on the same ${rows[0].n} evaluation questions.`,
    ['Model', 'direct-1', 'direct-2', 'Program, as published', 'Program, as the answers use it'],
    rows.map((r) => [r.label, pct(r.direct_1), r.direct_2 !== undefined ? pct(r.direct_2) : 'not rerun', pct(r.program_published), pct(r.program_as_labelled)]), [1, 2, 3, 4]));
  redrawOnResize(svgEl.parentElement!, draw);
  draw();
}

const CONDITIONS: [string, string, string, 'direct' | 'program', boolean][] = [
  ['direct_1', 'direct-1', 'bare answer; prompt states the published sectors', 'direct', false],
  ['direct_2', 'direct-2', 'bare answer; prompt states the rule the answers follow', 'direct', true],
  ['direct_1r', 'direct-1r', 'direct-1 with the working asked for', 'direct', false],
  ['direct_2r', 'direct-2r', 'direct-2 with the working asked for', 'direct', true],
  ['program_published', 'program', 'executed as published', 'program', false],
  ['program_as_labelled', 'program', 'same generations, executed as the answers use the relations', 'program', true],
  ['program_constrained', 'program, constrained', 'decoded under a JSON Schema; a separate condition', 'program', true],
];

/** Every condition that was run on the first comparison, one card per model; a row exists only if its run does. */
export function initPrompts(audit: Audit) {
  const root = document.querySelector<HTMLElement>('#fig-prompts [data-role=cards]')!;
  root.replaceChildren();
  for (const model of audit.prompts) {
    const card = node('div', '', 'cond-card');
    const t = node('table');
    t.append(node('caption', `${model.label} · ${model.n} questions`));
    for (const [key, name, what, mode, corrected] of CONDITIONS) {
      const v = model.cells[key];
      if (v === undefined) continue;
      const tr = t.insertRow();
      const th = node('th');
      th.scope = 'row';
      th.append(node('b', name), node('span', what));
      const td = tr.insertCell();
      const bar = node('i');
      bar.style.width = `${v * 100}%`;
      bar.style.background = `var(--${mode})`;
      if (!corrected) bar.style.opacity = '0.4';
      td.className = 'cond-bar';
      td.append(bar);
      const val = tr.insertCell();
      val.className = 'num';
      val.textContent = pct(v);
      tr.prepend(th);
    }
    card.append(t);
    root.append(card);
  }
}

/** The validation drives: accuracy of each mode with drive-level bootstrap intervals. */
export function initValidation(audit: Audit) {
  const fig = document.getElementById('fig-validation')!;
  const svgEl = fig.querySelector<SVGSVGElement>('[data-role=chart]')!;
  const runs = audit.validation.runs;
  const name = (c: string) => (c === 'published' ? 'as published' : 'as the answers use them');
  const diff = (r: (typeof runs)[number]) => `${signed(r.difference.estimate)} [${signed(r.difference.low)}, ${signed(r.difference.high)}]`;

  function draw() {
    const { svg, W, narrow, x, rowY } = frame(svgEl, runs.map((r) => ({ title: r.label, sub: name(r.conventions) })),
      { left: 190, right: 178, domain: [0.4, 1], rowH: 48, narrowRowH: 78 });
    const direct = cssVar('--direct'), program = cssVar('--program');
    if (!narrow) svg.append('text').attr('x', W - 6).attr('y', 9).attr('text-anchor', 'end').attr('font-size', 10.5).attr('fill', cssVar('--muted')).text('program − direct, points');
    runs.forEach((r, i) => {
      const y = rowY(i);
      const filled = r.conventions === 'as_labelled';
      for (const [e, colour, label] of [[r.direct, direct, 'direct'], [r.program, program, 'program']] as const) {
        svg.append('line').attr('x1', x(e.low)).attr('x2', x(e.high)).attr('y1', y).attr('y2', y).attr('stroke', colour).attr('stroke-width', 2.5).attr('stroke-opacity', 0.55);
        for (const end of [e.low, e.high]) svg.append('line').attr('x1', x(end)).attr('x2', x(end)).attr('y1', y - 5).attr('y2', y + 5).attr('stroke', colour).attr('stroke-width', 1.5).attr('stroke-opacity', 0.55);
        dot(svg, x(e.estimate), y, colour, filled, `${r.label}, ${name(r.conventions)}, ${label}: ${pct(e.estimate)}, 95 % interval ${pct(e.low)} to ${pct(e.high)}`);
        svg.append('text').attr('x', x(e.estimate)).attr('y', y - 12).attr('text-anchor', 'middle').attr('font-size', 11.5).attr('fill', colour).attr('font-weight', 600).text(pct(e.estimate));
      }
      const t = svg.append('text').attr('font-size', 11.5).attr('fill', cssVar('--ink-2')).attr('font-weight', 600);
      if (narrow) t.attr('x', 14).attr('y', y + 22).text(`program − direct: ${diff(r)} points`);
      else t.attr('x', W - 6).attr('y', y + 4).attr('text-anchor', 'end').text(diff(r));
    });
    legend(fig.querySelector<HTMLElement>('[data-role=legend]')!, [
      ['direct, as published', direct, false], ['direct, as the answers use them', direct, true],
      ['program, as published', program, false], ['program, as the answers use them', program, true],
    ]);
  }
  svgEl.setAttribute('aria-label', `Accuracy on ${num(runs[0]?.n ?? 0)} questions from ${runs[0]?.n_drives ?? 0} validation drives, with 95 % drive-level intervals. `
    + runs.map((r) => `${r.label}, ${name(r.conventions)}: direct ${pct(r.direct.estimate)}, program ${pct(r.program.estimate)}, difference ${diff(r)} points`).join('. ') + '.');
  const ci = (e: { estimate: number; low: number; high: number }) => `${pct(e.estimate)} [${(e.low * 100).toFixed(1)}, ${(e.high * 100).toFixed(1)}]`;
  fig.querySelector<HTMLElement>('[data-role=table]')!.replaceChildren(table(
    'Accuracy with 95 % intervals from 10,000 resamples of drives.',
    ['Model', 'Conventions', 'Direct', 'Program', 'Program − direct, points', 'Drives where the program is ahead / behind / tied'],
    runs.map((r) => [r.label, name(r.conventions), ci(r.direct), ci(r.program), diff(r), `${r.drives_ahead} / ${r.drives_behind} / ${r.drives_tied}`]), [2, 3, 4, 5]));
  redrawOnResize(svgEl.parentElement!, draw);
  draw();
}
