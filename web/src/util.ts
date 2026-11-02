/** Read a CSS custom property, e.g. cssVar('--drift'). */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Format a relative quantity: percentages above 1e-4, scientific notation below. */
export function fmtRel(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return '—';
  if (x === 0) return '0';
  if (x >= 1e-4) return `${(x * 100).toFixed(digits)} %`;
  return sci(x);
}

export function sci(x: number, digits = 1): string {
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  const m = x / 10 ** e;
  const sup = String(e).replace('-', '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]);
  return `${m.toFixed(digits)} × 10${sup}`;
}

/** Run init when the element first comes near the viewport. */
export function whenVisible(el: Element, init: () => void, margin = '300px') {
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) { io.disconnect(); init(); }
  }, { rootMargin: margin });
  io.observe(el);
}

/** Track whether an element is on screen (to pause animation loops). */
export function onScreen(el: Element, cb: (visible: boolean) => void) {
  const io = new IntersectionObserver((entries) => cb(entries[0].isIntersecting), { rootMargin: '50px' });
  io.observe(el);
  return io;
}

/** Wire a segmented control: returns a getter and calls onChange with the new value. */
export function segmented(root: HTMLElement, onChange: (value: string) => void) {
  const buttons = Array.from(root.querySelectorAll<HTMLButtonElement>('button'));
  buttons.forEach((b) => b.addEventListener('click', () => {
    buttons.forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
    onChange(b.dataset.value ?? b.dataset.method ?? '');
  }));
}

export function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, parent?: Element) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  parent?.appendChild(node);
  return node;
}

export function onThemeChange(cb: () => void) {
  window.addEventListener('themechange', cb);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', cb);
}

/** Animate a value from its current position, honouring reduced motion. Returns a stop function. */
export function sweep(from: number, to: number, ms: number, step: (v: number) => void, done?: () => void) {
  if (reducedMotion()) { step(to); done?.(); return () => {}; }
  let raf = 0; const t0 = performance.now();
  const tick = (now: number) => {
    const p = Math.min(1, (now - t0) / ms);
    const e = p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2;
    step(from + (to - from) * e);
    if (p < 1) raf = requestAnimationFrame(tick); else done?.();
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}
