import { axisBottom, axisLeft, scaleLinear, select } from 'd3';
import { sidesOf, viewOf, type Convention, type Study } from './data';
import { cssVar, onThemeChange, segmented } from './util';

const isInt = (s: string | null) => s !== null && /^\d+$/.test(s);

/** Predicted against reference counts, both methods, under either convention. */
export function initCounts(study: Study) {
  const fig = document.getElementById('fig-counts')!;
  const svgEl = fig.querySelector<SVGSVGElement>('[data-role=chart]')!;
  const runSel = fig.querySelector<HTMLSelectElement>('[data-ctl=run]')!;
  const legend = fig.querySelector<HTMLElement>('[data-role=legend]')!;
  study.runs.forEach((r, i) => runSel.add(new Option(r.label, String(i))));
  const W = 760, H = 420, M = { l: 48, r: 20, t: 16, b: 44 };
  svgEl.setAttribute('viewBox', `0 0 ${W} ${H}`);

  let convention: Convention = 'as_labelled';

  function draw() {
    const run = study.runs[+runSel.value];
    const views = run.rows.filter((r) => isInt(r.reference_answer)).map((r) => ({ ref: +r.reference_answer, v: viewOf(r, convention) }));
    const rows = views.filter((r) => isInt(r.v.direct) || isInt(r.v.executed));
    // keep the axes on the reference range; rare larger answers are pinned to the top edge
    const max = Math.max(...rows.map((r) => r.ref)) + 2;
    const x = scaleLinear().domain([-0.5, max + 0.5]).range([M.l, W - M.r]);
    const y = scaleLinear().domain([-0.5, max + 0.6]).range([H - M.b, M.t]);
    const svg = select(svgEl);
    svg.selectAll('*').remove();
    svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - M.b})`).call(axisBottom(x).ticks(Math.min(max + 1, 12)).tickFormat((d) => (Number.isInteger(+d) ? String(d) : '')) as any);
    svg.append('g').attr('class', 'axis').attr('transform', `translate(${M.l},0)`).call(axisLeft(y).ticks(Math.min(max + 1, 12)).tickFormat((d) => (Number.isInteger(+d) ? String(d) : '')) as any);
    svg.append('text').attr('x', (M.l + W - M.r) / 2).attr('y', H - 6).attr('text-anchor', 'middle').attr('fill', cssVar('--muted')).attr('font-size', 12).text('reference count');
    svg.append('text').attr('transform', `translate(14,${(M.t + H - M.b) / 2}) rotate(-90)`).attr('text-anchor', 'middle').attr('fill', cssVar('--muted')).attr('font-size', 12).text('answer');
    svg.append('line').attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(max)).attr('y2', y(max)).attr('stroke', cssVar('--rule')).attr('stroke-width', 2);
    // aggregate identical (reference, answer) pairs into one dot sized by count
    for (const [key, colour, dx] of [['direct', cssVar('--direct'), -0.13], ['executed', cssVar('--program'), 0.13]] as const) {
      const bins = new Map<string, number>();
      for (const r of rows) {
        const a = r.v[key];
        if (!isInt(a)) continue;
        const k = `${r.ref},${a}`;
        bins.set(k, (bins.get(k) ?? 0) + 1);
      }
      for (const [k, n] of bins) {
        const [ref, ans] = k.split(',').map(Number);
        const pinned = ans > max;
        svg.append('circle').attr('cx', x(ref + dx)).attr('cy', y(Math.min(ans, max + 0.3))).attr('r', 3.2 + 2.6 * Math.sqrt(n - 1))
          .attr('fill', colour).attr('fill-opacity', 0.75).attr('stroke', cssVar('--surface')).attr('stroke-width', 1)
          .append('title').text(`${key === 'direct' ? 'Direct' : 'Program'}: answered ${ans} when the reference was ${ref} (${n}×)`);
        if (pinned) svg.append('text').attr('x', x(ref + dx) + 9).attr('y', y(max + 0.3) + 4).attr('font-size', 11).attr('fill', colour).text(`answered ${ans}`);
      }
    }
    const sides = sidesOf(run, convention);
    const d = sides.direct.count_errors, e = sides.executed.count_errors;
    const prompt = views[0]?.v.directPrompt ?? 'direct-1';
    svgEl.setAttribute('aria-label', `Predicted against reference counts for ${run.label}, ${convention === 'published' ? 'as published' : 'as the answers use them'}. Mean absolute error: direct (${prompt}) ${d.mae.toFixed(2)}, program and executor ${e.mae.toFixed(2)}.`);
    legend.replaceChildren();
    for (const [name, c, m] of [[`direct (${prompt})`, cssVar('--direct'), d.mae], ['program + executor', cssVar('--program'), e.mae]] as const) {
      const s = document.createElement('span');
      const i = document.createElement('i'); i.style.background = c; i.style.borderRadius = '50%';
      s.append(i, `${name}, mean absolute error ${m.toFixed(2)}`);
      legend.append(s);
    }
  }
  runSel.addEventListener('change', draw);
  segmented(fig.querySelector<HTMLElement>('[data-ctl=convention]')!, (value) => { convention = value as Convention; draw(); });
  onThemeChange(draw);
  draw();
}

/** Verbatim-repeated questions with differing reference answers, scored as the released answers are. */
export function initDependence(study: Study) {
  const root = document.querySelector<HTMLElement>('#fig-dependence [data-role=groups]')!;
  const rows = study.runs[0].rows;
  const groups = new Map<string, typeof rows>();
  for (const r of rows) { if (!groups.has(r.question)) groups.set(r.question, []); groups.get(r.question)!.push(r); }
  const kept = [...groups.values()].filter((g) => new Set(g.map((r) => r.reference_answer)).size > 1);
  root.replaceChildren();
  for (const g of kept) {
    const card = document.createElement('div');
    card.className = 'dep-card';
    const h = document.createElement('h3'); h.textContent = g[0].question;
    const t = document.createElement('table');
    const head = t.insertRow();
    for (const c of ['Asked', 'Reference', 'Direct', 'Program']) { const th = document.createElement('th'); th.textContent = c; head.append(th); }
    g.forEach((r, i) => {
      const tr = t.insertRow();
      tr.insertCell().textContent = String(i + 1);
      tr.insertCell().textContent = r.reference_answer;
      const v = viewOf(r, 'as_labelled');
      for (const [a, ok] of [[v.direct ?? '—', v.directOk], [v.executed ?? '—', v.executedOk]] as const) {
        const td = tr.insertCell();
        const m = document.createElement('span'); m.className = ok ? 'ok-mark' : 'bad-mark'; m.textContent = ok ? '✓ ' : '✗ ';
        td.append(m, String(a));
      }
    });
    card.append(h, t);
    root.append(card);
  }
  // six cards are enough to read the pattern; the rest stay one click away
  const cards = [...root.children] as HTMLElement[];
  if (cards.length > 6) {
    cards.slice(6).forEach((c) => { c.hidden = true; });
    const more = document.createElement('button');
    more.type = 'button';
    more.className = 'btn ghost';
    more.textContent = `Show all ${cards.length} groups`;
    more.addEventListener('click', () => { cards.forEach((c) => { c.hidden = false; }); more.remove(); });
    root.after(more);
    more.style.marginTop = '0.8rem';
  }
}
