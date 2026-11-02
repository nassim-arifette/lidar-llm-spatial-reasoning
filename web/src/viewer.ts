import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { onScreen } from './util';

/** A render-on-demand WebGL view of one canvas. Several views may share one camera. */
export class Viewer {
  readonly scene = new THREE.Scene();
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  readonly camera: THREE.PerspectiveCamera;
  private dirty = true;
  private visible = false;
  private raf = 0;
  private peers: Viewer[] = [];
  onFrame?: (time: number) => boolean | void; // return true to keep animating

  constructor(readonly canvas: HTMLCanvasElement, camera?: THREE.PerspectiveCamera, opts: { zoom?: boolean } = {}) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.camera = camera ?? new THREE.PerspectiveCamera(38, 1, 0.5, 600);
    if (!camera) this.camera.position.set(2.6, 1.7, 3.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.enablePan = false;
    this.controls.enableZoom = opts.zoom ?? true;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 160;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.addEventListener('change', () => this.invalidate());

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8090a0, 2.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(3, 5, 4);
    this.scene.add(sun);

    new ResizeObserver(() => this.resize()).observe(canvas);
    onScreen(canvas, (v) => { this.visible = v; if (v) this.loop(); });
  }

  /** Views sharing a camera re-render together. */
  link(other: Viewer) { this.peers.push(other); other.peers.push(this); }

  invalidate() {
    this.dirty = true;
    for (const p of this.peers) p.dirty = true;
    if (this.visible) this.loop();
  }

  private resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.invalidate();
  }

  private loop() {
    if (this.raf) return;
    const tick = (time: number) => {
      this.raf = 0;
      if (!this.visible) return;
      const moved = this.controls.update();
      const animating = this.onFrame?.(time) === true;
      if (moved || this.dirty || animating) {
        this.dirty = false;
        this.renderer.render(this.scene, this.camera);
      }
      if (moved || animating || this.controlsActive) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  private get controlsActive() {
    // keep ticking briefly so damping can settle
    return (this.controls as unknown as { state: number }).state !== -1;
  }
}

