import './style.css';
import { loadAudit, loadStudy, type Audit } from './data';
import { whenVisible } from './util';
import { initHero } from './fig-hero';
import { initExplorer } from './fig-explorer';
import { initSectors } from './fig-sectors';
import { initPlayground } from './fig-playground';
import { initBrowser } from './fig-browser';
import { initCounts, initDependence } from './fig-charts';
import { initChecks, initLaterality, initModels, initPrompts, initValidation } from './fig-audit';

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

/**
 * Write the numbers quoted in the text from audit.json (tools/build_web_audit.py formats them).
 * The HTML already carries the same text, which tools/check_site.py verifies; this keeps a page
 * served with newer data consistent. A [data-if] passage is shown only when its number exists.
 */
function stats(audit: Audit) {
  document.querySelectorAll<HTMLElement>('[data-stat]').forEach((n) => {
    const v = audit.stats[n.dataset.stat!];
    if (v !== undefined) n.textContent = v;
  });
  document.querySelectorAll<HTMLElement>('[data-if]').forEach((n) => { n.hidden = audit.stats[n.dataset.if!] === undefined; });
}

const guard = (id: string, f: () => void) => { try { f(); } catch (e) { failed(id, e); } };
const lazily = (figures: [string, () => void][]) => {
  for (const [id, init] of figures) whenVisible(document.getElementById(id)!, () => guard(id, init));
};

loadAudit().then((audit) => {
  stats(audit);
  lazily([
    ['fig-laterality', () => initLaterality(audit)],
    ['fig-checks', () => initChecks(audit)],
    ['fig-models', () => initModels(audit)],
    ['fig-prompts', () => initPrompts(audit)],
    ['fig-validation', () => initValidation(audit)],
  ]);
}).catch((e) => ['fig-laterality', 'fig-models', 'fig-validation'].forEach((id) => failed(id, e)));

loadStudy().then((study) => {
  // a busy evaluation keyframe: parked cars on both sides behind the ego vehicle
  const pick = study.runs[0].rows.find((r) => r.question_id === 'q_0aa6927791ce')?.sample_token ?? Object.keys(study.scenes)[0];
  guard('top', () => initHero(study));
  lazily([
    ['fig-explorer', () => initExplorer(study, pick)],
    ['fig-sectors', () => initSectors(study, pick)],
    ['fig-playground', () => initPlayground(study, pick)],
    ['fig-browser', () => initBrowser(study)],
    ['fig-counts', () => initCounts(study)],
    ['fig-dependence', () => initDependence(study)],
  ]);
}).catch((e) => ['top', 'fig-explorer', 'fig-browser'].forEach((id) => failed(id, e)));
