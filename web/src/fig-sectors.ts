import { Bev } from './bev';
import { CATEGORY_LABEL, type Study } from './data';
import { bearingFrom, sectorOf, SECTORS, type Layout, type SceneObject } from './executor';
import { onThemeChange } from './util';

/** Six-colour ramp around the compass, so neighbouring sectors stay distinct. */
const SECTOR_COLOR: Record<string, string> = {
  front: '#3949c4', front_left: '#7b5cc4', back_left: '#b5579c', back: '#d9653b', back_right: '#c99a1f', front_right: '#2e9c8f',
};

/** Figure 2: the six NuScenes-QA relations, measured from the ego vehicle or from a clicked object. */
export function initSectors(study: Study, token: string) {
  const fig = document.getElementById('fig-sectors')!;
  const bev = new Bev(fig.querySelector('[data-role=bev]') as unknown as SVGSVGElement);
  const status = fig.querySelector<HTMLElement>('[data-role=status]')!;
  const objects = study.scenes[token].objects;
  let reference: SceneObject | null = null;
  let layout: Layout = 'nuscenes_qa_as_labelled';

  function draw() {
    bev.setScene(objects, reference === null);
    bev.sectors(reference, null, true, layout);
    bev.paint((o) => {
      if (o === reference) return 'var(--ink)';
      const s = sectorOf(o, reference, layout);
      return s ? SECTOR_COLOR[s] : 'var(--obj)';
    });
    bev.ring(reference);
    status.textContent = reference
      ? `Reference: ${reference.id}, a ${reference.status ? `${reference.status} ` : ''}${CATEGORY_LABEL(reference.category)}. Directions keep the ego vehicle’s heading.`
      : 'Reference: the ego vehicle. Click an object to measure from it.';
  }
  bev.onClick = (o) => { reference = o === reference ? null : o; draw(); };
  bev.onHover = (o) => {
    bev.hot(o?.token ?? null);
    if (!o) return bev.label(null);
    const b = bearingFrom(o, reference);
    const s = sectorOf(o, reference, layout);
    bev.label(o, o === reference ? 'reference' : b === null ? 'too close for a direction' : `${CATEGORY_LABEL(o.category)} · bearing ${b.toFixed(1)}° · ${s?.replace('_', ' ')}`);
  };
  fig.querySelector('[data-ctl=reset]')!.addEventListener('click', () => { reference = null; draw(); });
  const toggles = fig.querySelectorAll<HTMLButtonElement>('[data-layout]');
  toggles.forEach((button) => button.addEventListener('click', () => {
    layout = button.dataset.layout as Layout;
    toggles.forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    draw();
  }));
  onThemeChange(draw);
  draw();
  void SECTORS;
}
