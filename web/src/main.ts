import './style.css';
import { loadStudy, type Study } from './data';
import { whenVisible } from './util';
import { initHero } from './fig-hero';
import { initExplorer } from './fig-explorer';
import { initSectors } from './fig-sectors';
import { initPlayground } from './fig-playground';
import { initBrowser } from './fig-browser';
import { initCounts, initDependence, initScale } from './fig-charts';

document.getElementById('theme-toggle')?.addEventListener('click', () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('theme', root.dataset.theme); } catch { /* storage unavailable */ }
  window.dispatchEvent(new Event('themechange'));
});

function failed(id: string, err: unknown) {
  console.error(err);
  const fig = document.getElementById(id);
  if (!fig) return;
  const p = document.createElement('p');
  p.className = 'small';
  p.textContent = 'This figure could not load its data. Serve the site over HTTP (for example with npm run preview) rather than opening the file directly.';
  fig.prepend(p);
}

/** Fill numbers in the text from the archived summaries. */
function stats(study: Study) {
  const s = study.runs[0].summary;
  const pct = (v: number) => `${(v * 100).toFixed(1)} %`;
  const set = (k: string, v: string) => document.querySelectorAll(`[data-stat=${k}]`).forEach((n) => { n.textContent = v; });
  set('direct-acc', pct(s.direct.accuracy));
  set('exec-acc', pct(s.executed.accuracy));
  set('answered', `${Math.round(s.executed.answered_fraction * 100)} %`);
  set('exec-only', String(s.paired.executed_only));
  set('direct-only', String(s.paired.direct_only));
  set('mae-d', s.direct.count_errors.mae.toFixed(2));
  set('mae-e', s.executed.count_errors.mae.toFixed(2));
  set('under', s.direct.count_errors.under.toFixed(2));
}

loadStudy().then((study) => {
  stats(study);
  // the keyframe of the hero's first question: parked cars behind the ego vehicle
  const pick = study.runs[0].rows.find((r) => r.question_id === 'q_0aa6927791ce')?.sample_token ?? Object.keys(study.scenes)[0];
  const guard = (id: string, f: () => void) => { try { f(); } catch (e) { failed(id, e); } };
  guard('top', () => initHero(study));
  const lazy: [string, () => void][] = [
    ['fig-explorer', () => initExplorer(study, pick)],
    ['fig-sectors', () => initSectors(study, pick)],
    ['fig-playground', () => initPlayground(study, pick)],
    ['fig-browser', () => initBrowser(study)],
    ['fig-counts', () => initCounts(study)],
    ['fig-dependence', () => initDependence(study)],
    ['fig-scale', () => initScale(study)],
  ];
  for (const [id, init] of lazy) whenVisible(document.getElementById(id)!, () => guard(id, init));
}).catch((e) => ['top', 'fig-explorer', 'fig-browser'].forEach((id) => failed(id, e)));
