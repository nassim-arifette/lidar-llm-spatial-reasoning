"""Export LIDAR_TOP keyframe sweeps for the website, in the same ego frame as the boxes.

nuScenes stores each sweep in the sensor frame as float32 (x, y, z, intensity, ring).
The sensor-to-ego transform is the keyframe's calibrated_sensor record; the boxes in
study.json are in the ego frame of that same keyframe (see nuscenes_adapter.py), so
both overlay exactly.

Output per keyframe: web/public/lidar/<sample_token>.bin, little-endian
  uint32 n, then n x (int16 x, int16 y, int16 z) in centimetres, then n x uint8 intensity.
Points are kept within 70 m horizontally and subsampled deterministically.
"""
from pathlib import Path
import json
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/data/nuscenes"
META = DATA / "v1.0-mini"
OUT = ROOT / "web/public/lidar"
MAX_POINTS = 16000
MAX_RANGE = 70.0


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    study = json.loads((ROOT / "web/public/data/study.json").read_text(encoding="utf-8"))
    wanted = set(study["scenes"])
    table = lambda name: json.loads((META / f"{name}.json").read_text(encoding="utf-8"))
    sensors = {s["token"]: s["channel"] for s in table("sensor")}
    calib = {c["token"]: c for c in table("calibrated_sensor")}
    OUT.mkdir(parents=True, exist_ok=True)
    written, total = 0, 0
    for sd in table("sample_data"):
        if not sd["is_key_frame"] or sd["sample_token"] not in wanted:
            continue
        cs = calib[sd["calibrated_sensor_token"]]
        if sensors[cs["sensor_token"]] != "LIDAR_TOP":
            continue
        raw = np.fromfile(DATA / sd["filename"], dtype=np.float32).reshape(-1, 5)
        xyz = raw[:, :3].astype(np.float64) @ quat_to_matrix(cs["rotation"]).T + np.asarray(cs["translation"])
        keep = np.hypot(xyz[:, 0], xyz[:, 1]) <= MAX_RANGE
        xyz, inten = xyz[keep], raw[keep, 3]
        if len(xyz) > MAX_POINTS:
            idx = np.linspace(0, len(xyz) - 1, MAX_POINTS).round().astype(int)
            xyz, inten = xyz[idx], inten[idx]
        cm = np.clip(np.round(xyz * 100), -32767, 32767).astype("<i2")
        with open(OUT / f"{sd['sample_token']}.bin", "wb") as fh:
            fh.write(struct.pack("<I", len(cm)))
            fh.write(cm.tobytes())
            fh.write(np.clip(inten, 0, 255).astype(np.uint8).tobytes())
        written += 1
        total += len(cm)
    missing = wanted - {p.stem for p in OUT.glob("*.bin")}
    if missing:
        raise SystemExit(f"{len(missing)} keyframes have no sweep on disk")
    print(f"{written} sweeps, {total} points -> {OUT}")


if __name__ == "__main__":
    main()
