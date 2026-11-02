/** LIDAR_TOP keyframe sweeps in the ego frame, written by tools/build_lidar_points.py. */
export interface Sweep { n: number; xyz: Float32Array; intensity: Uint8Array }

const cache = new Map<string, Promise<Sweep | null>>();

export function loadSweep(token: string): Promise<Sweep | null> {
  if (!cache.has(token)) {
    cache.set(token, fetch(`${import.meta.env.BASE_URL}lidar/${token}.bin`).then(async (r) => {
      if (!r.ok) return null;
      const buf = await r.arrayBuffer();
      const view = new DataView(buf);
      const n = view.getUint32(0, true);
      const cm = new Int16Array(buf.slice(4, 4 + n * 6));
      const xyz = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) xyz[i] = cm[i] / 100;
      return { n, xyz, intensity: new Uint8Array(buf.slice(4 + n * 6, 4 + n * 7)) };
    }).catch(() => null));
  }
  return cache.get(token)!;
}

/** Height ramp for points: ground faint, raised structure darker. Returns [r,g,b,a] in 0..1. */
export function heightTone(z: number, dark: boolean): [number, number, number, number] {
  const t = Math.min(1, Math.max(0, (z + 1.8) / 4));
  if (dark) return [0.55 + 0.35 * t, 0.62 + 0.3 * t, 0.75 + 0.2 * t, 0.35 + 0.5 * t];
  return [0.32 - 0.2 * t, 0.38 - 0.22 * t, 0.5 - 0.25 * t, 0.25 + 0.55 * t];
}
