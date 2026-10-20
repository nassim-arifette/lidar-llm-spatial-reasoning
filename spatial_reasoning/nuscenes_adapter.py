"""Build object lists from the real nuScenes metadata.

This is the bridge between the released nuScenes annotations and the
object-level representation the rest of the package reasons over. It reads
only the metadata JSONs -- the 4 GB of sensor blobs are never needed, because
perception is fixed and we consume boxes, not point clouds.

What it produces
----------------
One :class:`~spatial_reasoning.objects.Scene` per keyframe sample, with every
annotated object expressed in the ego frame of that sample's LIDAR_TOP pose
($x$ forward, $y$ left, $z$ up).

Two conventions worth stating, because they affect every number downstream:

* **Ground truth, not detections.** nuScenes annotations are the oracle
  condition. A detector condition is obtained by passing these scenes through
  the perturbation model in ``consistency.py``, which is an assumption about
  detector behaviour, not a measurement of one.
* **Visibility filtering.** An annotation with zero LiDAR returns is in the
  label file but invisible to a LiDAR-only system. Counting it would punish a
  method for not seeing what the sensor never captured, so
  ``min_lidar_points`` drops those by default. The number removed is reported.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .objects import Object3D, Scene, canonical_category, global_to_ego, quaternion_to_yaw

#: The metadata tables we actually need.
REQUIRED_TABLES = (
    "sample",
    "sample_annotation",
    "sample_data",
    "ego_pose",
    "instance",
    "category",
    "calibrated_sensor",
    "sensor",
    "attribute",
)

#: nuScenes attribute names mapped onto the status words nuScenes-QA uses in
#: its answers. Getting this mapping wrong would make every status question
#: unanswerable for reasons that have nothing to do with reasoning.
ATTRIBUTE_TO_STATUS = {
    "vehicle.moving": "moving",
    "vehicle.stopped": "stopped",
    "vehicle.parked": "parked",
    "cycle.with_rider": "with rider",
    "cycle.without_rider": "without rider",
    "pedestrian.moving": "moving",
    "pedestrian.standing": "standing",
    "pedestrian.sitting_lying_down": "sitting lying down",
}


def _load_table(root: str, name: str) -> list:
    path = os.path.join(root, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing nuScenes table {name}.json in {root}. Extract the metadata "
            f"with: curl -sL https://www.nuscenes.org/data/v1.0-mini.tgz "
            f"| tar -xz -C <dir> v1.0-mini/"
        )
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class ConversionReport:
    """What the conversion did, so the experiment can quote it."""

    samples: int = 0
    scenes: int = 0
    annotations_total: int = 0
    annotations_kept: int = 0
    dropped_no_lidar_points: int = 0
    dropped_unknown_category: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "samples": self.samples,
            "scenes": self.scenes,
            "annotations_total": self.annotations_total,
            "annotations_kept": self.annotations_kept,
            "dropped_no_lidar_points": self.dropped_no_lidar_points,
            "dropped_unknown_category": self.dropped_unknown_category,
        }


def load_nuscenes_scenes(
    metadata_root: str,
    min_lidar_points: int = 1,
    lidar_channel: str = "LIDAR_TOP",
) -> Tuple[Dict[str, Scene], ConversionReport]:
    """Read the metadata and return ``{sample_token: Scene}`` in the ego frame.

    ``metadata_root`` is the directory holding ``sample.json`` and friends
    (``.../v1.0-mini``).
    """
    tables = {name: _load_table(metadata_root, name) for name in REQUIRED_TABLES}

    category_name = {c["token"]: c["name"] for c in tables["category"]}
    instance_category = {
        i["token"]: category_name.get(i["category_token"]) for i in tables["instance"]
    }
    ego_pose = {e["token"]: e for e in tables["ego_pose"]}
    attribute_status = {
        a["token"]: ATTRIBUTE_TO_STATUS.get(a["name"]) for a in tables["attribute"]
    }

    # Which calibrated sensors are the LiDAR?
    sensor_channel = {s["token"]: s["channel"] for s in tables["sensor"]}
    lidar_calibrated = {
        cs["token"]
        for cs in tables["calibrated_sensor"]
        if sensor_channel.get(cs["sensor_token"]) == lidar_channel
    }

    # The ego pose of a sample is the pose recorded with its LiDAR keyframe.
    pose_of_sample: Dict[str, dict] = {}
    for sd in tables["sample_data"]:
        if sd.get("is_key_frame") and sd.get("calibrated_sensor_token") in lidar_calibrated:
            pose_of_sample[sd["sample_token"]] = ego_pose[sd["ego_pose_token"]]

    annotations_by_sample: Dict[str, list] = {}
    for ann in tables["sample_annotation"]:
        annotations_by_sample.setdefault(ann["sample_token"], []).append(ann)

    report = ConversionReport()
    scenes: Dict[str, Scene] = {}

    for sample in tables["sample"]:
        token = sample["token"]
        pose = pose_of_sample.get(token)
        if pose is None:
            continue

        global_objects: List[Object3D] = []
        for ann in annotations_by_sample.get(token, []):
            report.annotations_total += 1

            if ann.get("num_lidar_pts", 0) < min_lidar_points:
                report.dropped_no_lidar_points += 1
                continue

            raw = instance_category.get(ann["instance_token"])
            name = canonical_category(raw)
            if name is None:
                report.dropped_unknown_category += 1
                continue

            # An annotation carries at most one attribute in nuScenes; objects
            # with none (a traffic cone, a barrier) keep status None and are
            # refused by status questions rather than given a default.
            status = None
            for at in ann.get("attribute_tokens", []):
                status = attribute_status.get(at)
                if status:
                    break

            global_objects.append(Object3D(
                category=name,
                center=ann["translation"],
                size=ann["size"],
                yaw=quaternion_to_yaw(ann["rotation"]),
                score=1.0,
                token=ann["token"],
                status=status,
            ))

        objects = global_to_ego(global_objects, pose["translation"], pose["rotation"])
        report.annotations_kept += len(objects)
        scenes[token] = Scene(
            sample_token=token,
            objects=objects,
            scene_token=sample.get("scene_token"),
            source="ground_truth_nuscenes",
        )

    report.samples = len(scenes)
    report.scenes = len({s.scene_token for s in scenes.values()})
    return scenes, report


# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #

def load_nuscenes_qa(
    path: str,
    scenes: Optional[Dict[str, Scene]] = None,
    token_field: str = "token",
) -> list:
    """Load nuScenes-QA pairs exported as ``[{token, question, answer}, ...]``.

    Items whose sample is absent from ``scenes`` are dropped: a question we
    cannot ground in an object list is not an error of any system, and keeping
    it would silently depress every score equally.
    """
    from .datasets import QAItem  # local import keeps the module graph acyclic

    with open(path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    if isinstance(records, dict):
        records = records.get("questions", list(records.values()))

    items = []
    skipped = 0
    for r in records:
        token = r.get(token_field) or r.get("sample_token")
        if token is None:
            continue
        if scenes is not None and token not in scenes:
            skipped += 1
            continue
        items.append(QAItem(
            question=r.get("question", ""),
            answer=r.get("answer", ""),
            sample_token=token,
            scene_token=(scenes[token].scene_token if scenes else None),
            template=r.get("template_type") or r.get("question_type") or "nuscenes_qa",
        ))
    if skipped:
        print(f"[nuscenes_adapter] dropped {skipped} questions with no matching sample")
    return items


def main(argv: Optional[list] = None) -> int:
    """Convert the metadata to our compact scene format."""
    import argparse

    from .objects import save_simple_scenes

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-root", required=True,
                   help="directory containing sample.json (e.g. .../v1.0-mini)")
    p.add_argument("--out", required=True, help="output JSON in the compact format")
    p.add_argument("--min-lidar-points", type=int, default=1)
    args = p.parse_args(argv)

    scenes, report = load_nuscenes_scenes(args.metadata_root, args.min_lidar_points)
    save_simple_scenes(scenes, args.out)
    print(json.dumps(report.as_dict(), indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
