"""Object-level scene representation built from pre-extracted 3D detections.

The whole point of this module is that perception is *fixed*: we never run a
detector here. We consume detections that were produced once (CenterPoint
outputs shipped with nuScenes-QA, or ground-truth boxes when we want an oracle
perception condition) and turn them into a small, explicit list of objects.

Frame convention (used everywhere in this package)
--------------------------------------------------
Ego frame of the nuScenes LiDAR sample:

    x : forward (direction the ego vehicle faces)
    y : left
    z : up

Angles are in radians, yaw is the rotation around +z, measured from +x towards
+y. Every geometric predicate in ``regions.py`` is expressed in this frame and
nowhere else, so that a "front/left" convention mismatch is a single-file bug
rather than a silent error spread over the codebase.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #

#: Canonical categories we reason about. These are the ten nuScenes detection
#: classes, which is what CenterPoint outputs.
CANONICAL_CATEGORIES: tuple = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
)

#: Coarse groups, because questions often ask about "vehicles" rather than a
#: specific class.
CATEGORY_GROUPS: Dict[str, tuple] = {
    "vehicle": ("car", "truck", "bus", "trailer", "construction_vehicle",
                "motorcycle", "bicycle"),
    "four_wheeler": ("car", "truck", "bus", "trailer", "construction_vehicle"),
    "two_wheeler": ("motorcycle", "bicycle"),
    "person": ("pedestrian",),
    "object": CANONICAL_CATEGORIES,
}

#: nuScenes annotation names (``human.pedestrian.adult``, ...) and a few common
#: spellings mapped onto the canonical set.
_CATEGORY_ALIASES: Dict[str, str] = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
    "trafficcone": "traffic_cone",
    "constructionvehicle": "construction_vehicle",
}


def canonical_category(raw: str) -> Optional[str]:
    """Map a raw category string onto :data:`CANONICAL_CATEGORIES`.

    Returns ``None`` for categories we deliberately ignore (vegetation, debris,
    ...). Returning ``None`` rather than guessing keeps "unknown class" visible
    in the error analysis instead of silently inflating a count.
    """
    if raw is None:
        return None
    key = str(raw).strip().lower().replace(" ", "_")
    if key in CANONICAL_CATEGORIES:
        return key
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    # nuScenes names are dotted; try the last component as a fallback.
    tail = key.split(".")[-1]
    if tail in CANONICAL_CATEGORIES:
        return tail
    return None


def expand_category(name: str) -> tuple:
    """Resolve a category *or* a group name into a tuple of canonical classes."""
    key = str(name).strip().lower().replace(" ", "_")
    if key in CATEGORY_GROUPS:
        return CATEGORY_GROUPS[key]
    canon = canonical_category(key)
    return (canon,) if canon is not None else ()


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class Object3D:
    """One detected (or annotated) object, expressed in the ego frame."""

    category: str
    center: np.ndarray           # (3,) float32 -- x forward, y left, z up
    size: np.ndarray             # (3,) float32 -- width, length, height
    yaw: float = 0.0             # radians, around +z
    score: float = 1.0           # detection confidence; 1.0 for ground truth
    token: Optional[str] = None  # provenance, useful when debugging
    status: Optional[str] = None # nuScenes attribute: moving / parked / ...

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float32).reshape(3)
        self.size = np.asarray(self.size, dtype=np.float32).reshape(3)
        self.yaw = float(self.yaw)
        self.score = float(self.score)

    @property
    def x(self) -> float:
        return float(self.center[0])

    @property
    def y(self) -> float:
        return float(self.center[1])

    @property
    def z(self) -> float:
        return float(self.center[2])

    @property
    def range_xy(self) -> float:
        """Horizontal distance to the ego vehicle origin."""
        return float(math.hypot(self.center[0], self.center[1]))

    @property
    def bearing(self) -> float:
        """Angle to the object in the ego frame: 0 is straight ahead,
        positive towards the left, in ``(-pi, pi]``."""
        return float(math.atan2(self.center[1], self.center[0]))

    def moved_to(self, center: Sequence[float]) -> "Object3D":
        """Return a copy translated to ``center`` (used by counterfactuals)."""
        return replace(self, center=np.asarray(center, dtype=np.float32))


@dataclass
class Scene:
    """A list of objects for one LiDAR sample, plus provenance."""

    sample_token: str
    objects: List[Object3D] = field(default_factory=list)
    scene_token: Optional[str] = None   # the nuScenes *scene*, for split-by-scene
    source: str = "detection"           # "detection" | "ground_truth" | "synthetic"

    def __len__(self) -> int:
        return len(self.objects)

    def filter_score(self, min_score: float) -> "Scene":
        kept = [o for o in self.objects if o.score >= min_score]
        return Scene(self.sample_token, kept, self.scene_token, self.source)

    def of_category(self, names: Iterable[str]) -> List[Object3D]:
        wanted = set()
        for n in names:
            wanted.update(expand_category(n))
        return [o for o in self.objects if o.category in wanted]

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for o in self.objects:
            out[o.category] = out.get(o.category, 0) + 1
        return out


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def quaternion_to_yaw(q: Sequence[float]) -> float:
    """Yaw (rotation about +z) from a ``[w, x, y, z]`` quaternion.

    nuScenes stores rotations in this order, which is *not* the order used by
    scipy; getting it wrong silently mirrors the scene, so it is isolated here.
    """
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_rotation_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def global_to_ego(
    objects: Sequence[Object3D],
    ego_translation: Sequence[float],
    ego_rotation: Sequence[float],
) -> List[Object3D]:
    """Transform objects from the nuScenes *global* frame into the ego frame.

    Detection results released for nuScenes are in global coordinates. Every
    spatial predicate in this package assumes the ego frame, so this conversion
    is mandatory before reasoning. ``ego_rotation`` is a ``[w, x, y, z]``
    quaternion (the ``ego_pose`` record of the sample).
    """
    t = np.asarray(ego_translation, dtype=np.float32).reshape(3)
    ego_yaw = quaternion_to_yaw(ego_rotation)
    r_inv = yaw_to_rotation_matrix(-ego_yaw)
    out: List[Object3D] = []
    for o in objects:
        centered = o.center - t
        out.append(replace(
            o,
            center=(r_inv @ centered).astype(np.float32),
            yaw=o.yaw - ego_yaw,
        ))
    return out


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _object_from_nuscenes_box(box: dict, score_key: str, name_key: str) -> Optional[Object3D]:
    name = canonical_category(box.get(name_key))
    if name is None:
        return None
    size = box.get("size", [1.0, 1.0, 1.0])  # nuScenes order: w, l, h
    return Object3D(
        category=name,
        center=box.get("translation", [0.0, 0.0, 0.0]),
        size=size,
        yaw=quaternion_to_yaw(box.get("rotation", [1.0, 0.0, 0.0, 0.0])),
        score=float(box.get(score_key, 1.0)),
        token=box.get("sample_token"),
    )


class FrameError(ValueError):
    """Raised rather than guessing which frame a detection file is in."""


def load_nuscenes_detections(
    path: str,
    ego_poses: Optional[Dict[str, dict]] = None,
    min_score: float = 0.0,
    assume_ego_frame: bool = False,
) -> Dict[str, Scene]:
    """Load a nuScenes-format detection file (``{"results": {token: [...]}}``).

    ``ego_poses`` maps ``sample_token`` to ``{"translation": [...],
    "rotation": [w, x, y, z]}``; boxes are then converted from the global frame
    to the ego frame.

    **Without poses this raises.** nuScenes submission files are in GLOBAL
    coordinates, so treating them as ego-relative puts every object a kilometre
    from the origin at a bearing measured in the map frame: every directional
    answer is computed in the wrong frame, everything is "far", and comparing
    against ego-frame ground truth charges the entire discrepancy to
    perception. Recording the assumption in ``Scene.source`` was not enough --
    nothing read it. Pass ``assume_ego_frame=True`` only for a file you know is
    already ego-relative.
    """
    if ego_poses is None and not assume_ego_frame:
        raise FrameError(
            f"{path}: nuScenes detection results are in the global frame. Pass "
            f"ego_poses to convert them, or assume_ego_frame=True if this file "
            f"is already ego-relative. Guessing silently produces answers in "
            f"the wrong frame."
        )
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    results = payload.get("results", payload)
    scenes: Dict[str, Scene] = {}
    missing_poses: List[str] = []
    # An empty dict is not 'poses supplied'. Treating it as such converted
    # nothing while claiming the ego frame.
    frame = "detection_ego_from_poses" if ego_poses else "detection_assumed_ego"
    if ego_poses is not None and not ego_poses:
        raise FrameError(
            f"{path}: an empty ego_poses mapping cannot convert anything. Pass "
            f"the poses, or assume_ego_frame=True if the file is already "
            f"ego-relative."
        )

    for token, boxes in results.items():
        objs: List[Object3D] = []
        for box in boxes:
            obj = _object_from_nuscenes_box(box, "detection_score", "detection_name")
            if obj is not None and obj.score >= min_score:
                objs.append(obj)
        if ego_poses is not None:
            pose = ego_poses.get(token)
            if pose is None:
                # A missing pose used to leave that one sample in global
                # coordinates while its neighbours were converted, and both
                # were labelled as ego-frame. A file mixing two frames is
                # worse than one that is uniformly wrong, because nothing
                # downstream can detect it.
                missing_poses.append(token)
                continue
            objs = global_to_ego(objs, pose["translation"], pose["rotation"])
        scenes[token] = Scene(sample_token=token, objects=objs, source=frame)

    if missing_poses:
        raise FrameError(
            f"{path}: no ego pose for {len(missing_poses)} sample(s), e.g. "
            f"{missing_poses[:3]}. Those boxes would stay in the global frame "
            f"while the rest were converted, producing a file in two frames."
        )
    return scenes


def load_simple_scenes(path: str) -> Dict[str, Scene]:
    """Load our own compact, already-ego-frame format.

    ``[{"sample_token": "...", "scene_token": "...", "source": "ground_truth",
        "objects": [{"category": "car", "center": [x, y, z],
                     "size": [w, l, h], "yaw": 0.0, "score": 1.0}]}]``
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    scenes: Dict[str, Scene] = {}
    for entry in payload:
        objs = [
            Object3D(
                category=canonical_category(o["category"]) or o["category"],
                center=o["center"],
                size=o.get("size", [1.0, 1.0, 1.0]),
                yaw=o.get("yaw", 0.0),
                score=o.get("score", 1.0),
                token=o.get("token"),
                status=o.get("status"),
            )
            for o in entry.get("objects", [])
        ]
        token = entry["sample_token"]
        scenes[token] = Scene(
            sample_token=token,
            objects=objs,
            scene_token=entry.get("scene_token"),
            source=entry.get("source", "detection"),
        )
    return scenes


def save_simple_scenes(scenes: Dict[str, Scene], path: str) -> None:
    """Dump scenes in the compact format, so perception is computed once."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = [
        {
            "sample_token": s.sample_token,
            "scene_token": s.scene_token,
            "source": s.source,
            "objects": [
                {
                    "category": o.category,
                    "center": [round(float(v), 4) for v in o.center],
                    "size": [round(float(v), 4) for v in o.size],
                    "yaw": round(float(o.yaw), 5),
                    "score": round(float(o.score), 5),
                    "status": o.status,
                }
                for o in s.objects
            ],
        }
        for s in scenes.values()
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
