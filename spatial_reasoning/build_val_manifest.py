"""A larger frozen evaluation set: the nuScenes-QA validation drives.

The first comparison rests on three evaluation drives. Answers within a drive
are correlated, so its effective sample size is nearer three than two hundred,
and a sign test over questions overstates what it shows. This builds the set
that lets the drive be the unit: every validation drive that had no part in
development, a fixed small number of questions from each.

What is fixed before any model runs:

* **Scenes.** The official nuScenes-QA validation questions on v1.0-trainval,
  minus the drives that also appear in v1.0-mini. Prompts, the layout and the
  reasoning budget were all chosen on mini, so a mini drive is not held out
  however it is labelled.
* **Questions.** ``per_scene`` questions from each drive, dealt round-robin
  over the four narrow families after a seeded shuffle. The selection reads
  the question text and the template, never an answer or a program.
* **Development.** None here. Development stays the mini development split of
  ``first_comparison.json``; this manifest holds evaluation items only.

It also writes the slice of nuScenes metadata those questions need, so a run
does not have to load the 2.5 GB trainval tables. The slice is checked: every
scene built from it must fingerprint identically to the scene built from the
full tables.

    python -m spatial_reasoning.build_val_manifest \\
        --metadata-root data/data/nuscenes/v1.0-trainval \\
        --mini-root data/data/nuscenes/v1.0-mini \\
        --questions data/data/nuscenes_qa/NuScenes_val_questions.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .manifest import (FAMILIES, MANIFEST_VERSION, Manifest, ManifestItem,
                       _question_id, classify, scene_fingerprint)
from .nuscenes_adapter import REQUIRED_TABLES, load_nuscenes_scenes


def _table(root: str, name: str) -> list:
    with open(os.path.join(root, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_metadata_slice(root: str, out: str, samples: set) -> None:
    """Copy only the rows the loader reads for ``samples``."""
    os.makedirs(out, exist_ok=True)
    sample_data = [r for r in _table(root, "sample_data")
                   if r["sample_token"] in samples and r.get("is_key_frame")]
    poses = {r["ego_pose_token"] for r in sample_data}
    annotations = [r for r in _table(root, "sample_annotation")
                   if r["sample_token"] in samples]
    instances = {r["instance_token"] for r in annotations}
    kept = {
        "sample": [r for r in _table(root, "sample") if r["token"] in samples],
        "sample_annotation": annotations,
        "sample_data": sample_data,
        "ego_pose": [r for r in _table(root, "ego_pose") if r["token"] in poses],
        "instance": [r for r in _table(root, "instance") if r["token"] in instances],
    }
    for name in REQUIRED_TABLES:
        rows = kept.get(name)
        if rows is None:
            rows = _table(root, name)
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-root", required=True)
    p.add_argument("--mini-root", required=True)
    p.add_argument("--questions", required=True)
    p.add_argument("--per-scene", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="spatial_reasoning/manifests/val_drives.json")
    p.add_argument("--slice-out", default="data/data/nuscenes/v1.0-trainval-val-drives")
    args = p.parse_args(argv)

    payload = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    rows = payload["questions"] if isinstance(payload, dict) else payload
    scenes, report = load_nuscenes_scenes(args.metadata_root, min_lidar_points=0)
    mini_scenes = {s["token"] for s in _table(args.mini_root, "scene")}

    reasons = collections.Counter()
    pool: Dict[str, Dict[str, List[dict]]] = collections.defaultdict(
        lambda: {family: [] for family in FAMILIES})
    for row in rows:
        scene = scenes.get(row["sample_token"])
        if scene is None:
            reasons["sample_not_in_object_lists"] += 1
            continue
        if scene.scene_token in mini_scenes:
            reasons["drive_is_in_v1.0-mini_used_for_development"] += 1
            continue
        family, reason, flags = classify(row["question"], row["template_type"], "narrow")
        if family is None:
            reasons[reason] += 1
            continue
        pool[scene.scene_token][family].append({**row, "_flags": flags})

    rng = random.Random(args.seed)
    items: List[ManifestItem] = []
    seen = set()
    for scene_token in sorted(pool):
        buckets = pool[scene_token]
        for family in FAMILIES:
            # Sorted, then shuffled: the seed alone decides.
            buckets[family].sort(key=lambda r: _question_id(r["sample_token"], r["question"]))
            rng.shuffle(buckets[family])
        chosen: List[tuple] = []
        cursor = {family: 0 for family in FAMILIES}
        while len(chosen) < args.per_scene:
            progressed = False
            for family in FAMILIES:
                if len(chosen) >= args.per_scene:
                    break
                if cursor[family] < len(buckets[family]):
                    row = buckets[family][cursor[family]]
                    cursor[family] += 1
                    qid = _question_id(row["sample_token"], row["question"])
                    if qid in seen:
                        continue
                    seen.add(qid)
                    chosen.append((family, row, qid))
                    progressed = True
            if not progressed:
                break
        for family, row, qid in chosen:
            items.append(ManifestItem(
                question_id=qid, question=row["question"],
                sample_token=row["sample_token"], scene_token=scene_token,
                split="eval", family=family, template=row["template_type"],
                num_hop=int(row.get("num_hop") or 0),
                answer=str(row["answer"]).strip().lower(), **row["_flags"]))

    samples = {it.sample_token for it in items}
    write_metadata_slice(args.metadata_root, args.slice_out, samples)
    sliced, _ = load_nuscenes_scenes(args.slice_out, min_lidar_points=0)
    for token in samples:
        if scene_fingerprint(sliced[token]) != scene_fingerprint(scenes[token]):
            raise SystemExit(f"metadata slice differs from the full tables at {token}")

    manifest = Manifest(version=MANIFEST_VERSION, items=items, excluded=[], config={
        "source": "nuScenes-QA val (official release) on nuScenes v1.0-trainval annotations",
        "perception": "oracle: ground-truth annotations, not a detector",
        "min_lidar_points": 0,
        "seed": args.seed,
        "coordinate_precision": 1,
        "scope": "narrow",
        "per_scene": args.per_scene,
        "development": "none in this file; development is the dev split of "
                       "first_comparison.json (v1.0-mini)",
        "excluded_drives": "every drive of v1.0-mini",
        "questions_sha256": hashlib.sha256(Path(args.questions).read_bytes()).hexdigest(),
        "n_questions_read": len(rows),
        "exclusion_counts": dict(reasons.most_common()),
        "metadata_slice": {
            "samples": len(samples),
            "check": "every sliced scene fingerprints identically to the full tables"},
    })
    manifest.save(args.out)
    counts = manifest.counts()["eval"]
    print(f"read {len(rows)} questions; excluded {dict(reasons.most_common())}")
    print(f"eval: {counts}")
    print(f"wrote {args.out}\nwrote metadata slice for {len(samples)} samples to {args.slice_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
