"""A synthetic object-level corpus, so the whole study runs without nuScenes.

Why this exists: the original project stalled on 6 GB of RAM and no GPU, and
the nuScenes-QA detection features are a large download. The comparison this
report makes -- learned answer prediction versus explicit geometric
computation, at fixed perception -- does not actually need real point clouds.
It needs object lists, questions, and reference answers.

So this module generates all three, with two properties the real data cannot
offer:

* **gold programs are known**, because each question is generated *from* a
  query. That is what makes the comprehension/geometry split in
  ``evaluate.decompose_errors`` possible at all.
* **ground-truth and "detected" object lists both exist**, the second obtained
  by perturbing the first with a configurable detector-failure model.

What it cannot show: that any of this transfers to real LiDAR. Synthetic
scenes have no occlusion, no class confusion correlated with range, and a
uniform object layout. Results here bound what is worth trying on nuScenes;
they do not replace it. The report says so explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .consistency import PerturbationConfig, perturb_scene
from .datasets import QAItem
from .objects import Object3D, Scene
from .programs import execute
from .question_parser import Operation, Query
from .regions import DEFAULT_REGION_CONFIG, DIRECTIONS, Region, RegionConfig

# --------------------------------------------------------------------------- #
# Surface forms
# --------------------------------------------------------------------------- #

PLURALS: Dict[str, str] = {
    "car": "cars",
    "truck": "trucks",
    "bus": "buses",
    "trailer": "trailers",
    "construction_vehicle": "construction vehicles",
    "pedestrian": "pedestrians",
    "motorcycle": "motorcycles",
    "bicycle": "bicycles",
    "traffic_cone": "traffic cones",
    "barrier": "barriers",
    "vehicle": "vehicles",
}

#: Rough nuScenes-like class frequencies: mostly cars and pedestrians.
CLASS_WEIGHTS: Dict[str, float] = {
    "car": 0.42,
    "pedestrian": 0.18,
    "barrier": 0.11,
    "traffic_cone": 0.09,
    "truck": 0.08,
    "bicycle": 0.04,
    "motorcycle": 0.03,
    "bus": 0.025,
    "trailer": 0.015,
    "construction_vehicle": 0.01,
}

#: Typical (width, length, height) in metres.
CLASS_SIZES: Dict[str, Tuple[float, float, float]] = {
    "car": (1.9, 4.6, 1.6),
    "truck": (2.5, 7.0, 3.0),
    "bus": (2.9, 11.0, 3.5),
    "trailer": (2.9, 12.0, 3.8),
    "construction_vehicle": (2.8, 6.5, 3.2),
    "pedestrian": (0.7, 0.7, 1.7),
    "motorcycle": (0.8, 2.1, 1.5),
    "bicycle": (0.6, 1.7, 1.3),
    "traffic_cone": (0.4, 0.4, 0.9),
    "barrier": (2.5, 0.5, 1.0),
}

DIRECTION_WORDS: Dict[str, str] = {d: d.replace("_", " ") for d in DIRECTIONS}


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #

@dataclass
class SyntheticCorpus:
    """Everything an experiment needs, with perception available two ways."""

    oracle_scenes: Dict[str, Scene] = field(default_factory=dict)
    detected_scenes: Dict[str, Scene] = field(default_factory=dict)
    items: List[QAItem] = field(default_factory=list)
    gold_queries: Dict[str, Query] = field(default_factory=dict)
    detector: Optional[PerturbationConfig] = None

    def summary(self) -> Dict[str, float]:
        n_obj = [len(s.objects) for s in self.oracle_scenes.values()] or [0]
        answers = [it.answer for it in self.items]
        counts = [int(a) for a in answers if a.isdigit()]
        return {
            "scenes": float(len({s.scene_token for s in self.oracle_scenes.values()})),
            "samples": float(len(self.oracle_scenes)),
            "questions": float(len(self.items)),
            "objects_per_sample_mean": float(np.mean(n_obj)),
            "binary_fraction": float(np.mean([a in ("yes", "no") for a in answers])) if answers else 0.0,
            "count_mean": float(np.mean(counts)) if counts else 0.0,
            "count_max": float(max(counts)) if counts else 0.0,
            # Share of questions answerable by always saying "nothing there".
            # Reported because it is the ceiling of the trivial strategy and
            # therefore tells you how much a headline accuracy is worth.
            "trivial_answer_fraction": (
                float(np.mean([a in ("0", "no") for a in answers])) if answers else 0.0
            ),
        }


def _sample_bearing(
    rng: np.random.Generator, config: RegionConfig, boundary_margin: float = 0.05
) -> float:
    """Draw a bearing that is not within ``boundary_margin`` of a sector edge.

    Objects sitting exactly on a sector boundary would make the reference
    answer depend on floating-point rounding, which would show up later as
    irreducible label noise and be mistaken for a reasoning failure.
    """
    for _ in range(64):
        bearing = rng.uniform(-math.pi, math.pi)
        # Distance to the nearest sector edge, read off the ACTIVE layout
        # rather than assuming 45-degree octants.
        theta = math.degrees(bearing)
        margin_deg = math.degrees(boundary_margin)
        edges = [low for low, _ in config.sectors.values()]
        if all(min(abs((theta - e + 180.0) % 360.0 - 180.0) for e in edges) > margin_deg
               for _ in (0,)):
            return float(bearing)
    return float(bearing)


def _make_object(rng: np.random.Generator, config: RegionConfig) -> Object3D:
    categories = list(CLASS_WEIGHTS)
    weights = np.array([CLASS_WEIGHTS[c] for c in categories], dtype=np.float64)
    weights /= weights.sum()
    category = categories[int(rng.choice(len(categories), p=weights))]

    distance = float(rng.uniform(config.min_range + 2.0, 48.0))
    bearing = _sample_bearing(rng, config)
    w, l, h = CLASS_SIZES[category]
    jitter = rng.normal(1.0, 0.07, size=3).clip(0.8, 1.25)

    return Object3D(
        category=category,
        center=[distance * math.cos(bearing), distance * math.sin(bearing), h / 2.0],
        size=[w * jitter[0], l * jitter[1], h * jitter[2]],
        yaw=float(rng.uniform(-math.pi, math.pi)),
        score=1.0,
    )


def generate_scenes(
    n_scenes: int = 8,
    samples_per_scene: int = 12,
    mean_objects: float = 9.0,
    seed: int = 0,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
) -> Dict[str, Scene]:
    """Ground-truth object lists, grouped into scenes so splits can be by scene."""
    rng = np.random.default_rng(seed)
    scenes: Dict[str, Scene] = {}
    for s in range(n_scenes):
        scene_token = f"synthetic_scene_{s:03d}"
        for k in range(samples_per_scene):
            n = int(np.clip(rng.poisson(mean_objects), 1, 30))
            token = f"{scene_token}_sample_{k:03d}"
            scenes[token] = Scene(
                sample_token=token,
                objects=[_make_object(rng, config) for _ in range(n)],
                scene_token=scene_token,
                source="ground_truth",
            )
    return scenes


# --------------------------------------------------------------------------- #
# Questions, generated from programs
# --------------------------------------------------------------------------- #

def _question_text(query: Query) -> str:
    plural = PLURALS.get(query.category, query.category)

    if query.operation is Operation.COMPARE_COUNT:
        other = PLURALS.get(query.second_category, query.second_category)
        word = "fewer" if query.comparison == "fewer" else "more"
        if query.region.kind == "direction":
            where = query.region.direction.replace("_", " ")
            return (f"Are there {word} {plural} than {other} to the {where} "
                    f"of the ego vehicle?")
        return f"Are there {word} {plural} than {other} in the scene?"

    if query.region.kind == "direction":
        where = query.region.direction.replace("_", " ")
        if query.operation is Operation.COUNT:
            return f"How many {plural} are to the {where} of the ego vehicle?"
        return f"Are there any {plural} to the {where} of the ego vehicle?"

    if query.region.kind == "near":
        if query.operation is Operation.COUNT:
            return f"How many {plural} are near the ego vehicle?"
        return f"Are there any {plural} near the ego vehicle?"

    if query.region.kind == "far":
        if query.operation is Operation.COUNT:
            return f"How many {plural} are far from the ego vehicle?"
        return f"Are there any {plural} far from the ego vehicle?"

    if query.operation is Operation.COUNT:
        return f"How many {plural} are there?"
    return f"Are there any {plural} in the scene?"


def _template_name(query: Query) -> str:
    return f"{query.operation.value}::{query.region.kind}"


def _sample_query(rng: np.random.Generator, config: RegionConfig) -> Query:
    # Sample by index: numpy's choice on a list of Enum members would build an
    # object array, which is fragile across numpy versions.
    operations = (Operation.COUNT, Operation.EXISTS, Operation.COMPARE_COUNT)
    operation = operations[int(rng.choice(len(operations), p=[0.55, 0.30, 0.15]))]

    categories = list(CLASS_WEIGHTS) + ["vehicle"]
    category = categories[int(rng.integers(len(categories)))]

    region_kinds = ("any", "direction", "near", "far")
    region_kind = region_kinds[int(rng.choice(len(region_kinds), p=[0.3, 0.5, 0.1, 0.1]))]
    if region_kind == "direction":
        # Sample from the ACTIVE layout. Drawing from the module-global eight
        # made it impossible to generate a corpus under the benchmark's own
        # six-relation convention: 'left' and 'right' do not exist there.
        names = config.directions()
        region = Region(kind="direction",
                        direction=names[int(rng.integers(len(names)))])
    else:
        region = Region(kind=region_kind)

    if operation is Operation.COMPARE_COUNT:
        others = [c for c in categories if c != category]
        return Query(
            operation=operation,
            category=category,
            second_category=others[int(rng.integers(len(others)))],
            comparison="more" if rng.random() < 0.5 else "fewer",
            region=region if region.kind in ("any", "direction") else Region("any"),
        )
    return Query(operation=operation, category=category, region=region)


def _is_trivial(answer: str) -> bool:
    """Answers reachable by always saying "nothing is there"."""
    return answer in ("0", "no")


def generate_corpus(
    n_scenes: int = 8,
    samples_per_scene: int = 12,
    questions_per_sample: int = 6,
    seed: int = 0,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    detector: Optional[PerturbationConfig] = None,
    max_trivial_fraction: float = 0.35,
) -> SyntheticCorpus:
    """Build scenes, then questions whose answers are computed on those scenes.

    ``detector`` is the perception model: the object lists the systems actually
    see are the ground-truth lists passed through it. Leaving it ``None`` gives
    perfect perception, which is useful as an upper bound but is not the
    default setting of the experiments.

    ``max_trivial_fraction`` caps the share of questions whose answer is ``0``
    or ``no``. Without it the corpus is dominated by them: a question about a
    rare class inside a 45-degree sector is almost always answered "none", and
    a system that always says so scores well without looking at anything. That
    is precisely the language-prior shortcut Real-3DQA identifies, and a
    benchmark exhibiting it cannot discriminate between reasoning strategies.
    Set it to ``1.0`` to disable the balancing and reproduce the degenerate
    distribution.
    """
    rng = np.random.default_rng(seed + 991)
    oracle = generate_scenes(n_scenes, samples_per_scene, seed=seed, config=config)

    detected: Dict[str, Scene] = {}
    if detector is None:
        detected = {
            t: Scene(s.sample_token, list(s.objects), s.scene_token, "detection_perfect")
            for t, s in oracle.items()
        }
    else:
        noise_rng = np.random.default_rng(detector.seed)
        for token, scene in oracle.items():
            detected[token] = perturb_scene(scene, detector, noise_rng)

    items: List[QAItem] = []
    gold: Dict[str, Query] = {}

    trivial_budget = int(round(max_trivial_fraction * questions_per_sample))

    for token, scene in oracle.items():
        seen: set = set()
        trivial_used = 0
        attempts = 0
        while len(seen) < questions_per_sample and attempts < questions_per_sample * 40:
            attempts += 1
            query = _sample_query(rng, config)
            text = _question_text(query)
            if text in seen:
                continue
            result = execute(query, scene, config)
            if not result.ok:
                continue
            if _is_trivial(result.answer):
                # Keep a few, because "none" is a legitimate answer; refuse to
                # let them swamp the corpus. The attempt budget above means a
                # sparse scene can still fall short, which is reported in
                # SyntheticCorpus.summary() rather than silently padded.
                if trivial_used >= trivial_budget:
                    continue
                trivial_used += 1
            seen.add(text)
            items.append(QAItem(
                question=text,
                answer=result.answer,
                sample_token=token,
                scene_token=scene.scene_token,
                template=_template_name(query),
            ))
            gold[f"{token}::{text}"] = query

    return SyntheticCorpus(oracle, detected, items, gold, detector)


def default_detector(seed: int = 0) -> PerturbationConfig:
    """A deliberately imperfect detector: misses, double counts, mislocalises.

    The numbers are of the order of what a CenterPoint-class detector gets
    wrong on nuScenes at a moderate score threshold. They are an assumption,
    not a measurement, and the sensitivity sweep in ``consistency.py`` is what
    makes that assumption testable.
    """
    return PerturbationConfig(
        drop_probability=0.12,
        duplicate_probability=0.05,
        position_jitter_m=0.6,
        seed=seed,
    )
