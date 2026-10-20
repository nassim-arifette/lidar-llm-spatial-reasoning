"""Consistency and robustness probes.

A single accuracy number does not say whether a system used the scene. These
probes are the part of the study that is meant to survive even if the headline
comparison is inconclusive.

Three probes:

``contrast pairs``
    The same question text, two samples with different correct answers. A
    system answering from the language prior alone cannot get both right, so
    pair accuracy separates "knows the answer distribution" from "read the
    scene".

``counterfactual moves``
    Move one object from one sector to another and recompute the reference
    answer. A system that tracks geometry must change its answer accordingly.

``perception perturbations``
    Drop, duplicate or jitter detections and watch accuracy degrade. This is
    where a learned counter may beat an exact computation: exact arithmetic on
    a corrupted object list is still exactly wrong.

An honest caveat, stated here and in the report: the reference answers for the
counterfactual probe are produced by the same executor that the symbolic
system uses, so the symbolic system passes that probe by construction. The
probe is informative for the learned systems and is a sanity check for the
symbolic one -- it is not evidence in its favour.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .datasets import QAItem
from .objects import Object3D, Scene
from .programs import execute
from .question_parser import DEFAULT_PARSER, RuleBasedQuestionParser
from .regions import (
    DEFAULT_REGION_CONFIG,
    DIRECTIONS,
    RegionConfig,
    direction_of,
    sample_position_in_direction,
)

PredictFn = Callable[[Sequence[QAItem], Dict[str, Scene]], Sequence[Optional[str]]]


# --------------------------------------------------------------------------- #
# Contrast pairs
# --------------------------------------------------------------------------- #

@dataclass
class ContrastPair:
    question: str
    a: QAItem
    b: QAItem


def build_contrast_pairs(
    items: Sequence[QAItem], max_pairs_per_question: int = 50, seed: int = 0
) -> List[ContrastPair]:
    """Group items by question text and pair up differing answers."""
    rng = np.random.default_rng(seed)
    by_question: Dict[str, List[QAItem]] = {}
    for it in items:
        by_question.setdefault(it.question.strip().lower(), []).append(it)

    pairs: List[ContrastPair] = []
    for question, group in by_question.items():
        by_answer: Dict[str, List[QAItem]] = {}
        for it in group:
            by_answer.setdefault(it.answer, []).append(it)
        answers = sorted(by_answer)
        if len(answers) < 2:
            continue
        made = 0
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                left, right = by_answer[answers[i]], by_answer[answers[j]]
                for k in range(min(len(left), len(right))):
                    if made >= max_pairs_per_question:
                        break
                    pairs.append(ContrastPair(question, left[k], right[k]))
                    made += 1
    rng.shuffle(pairs)
    return pairs


def pair_accuracy(
    predict: PredictFn, pairs: Sequence[ContrastPair], scenes: Dict[str, Scene]
) -> Dict[str, float]:
    """Fraction of pairs where *both* members are answered correctly."""
    if not pairs:
        return {"pair_accuracy": 0.0, "item_accuracy": 0.0, "n_pairs": 0.0}

    items: List[QAItem] = []
    for p in pairs:
        items.extend([p.a, p.b])
    preds = list(predict(items, scenes))

    both, single = 0, 0
    for idx in range(0, len(items), 2):
        ok_a = preds[idx] is not None and str(preds[idx]).lower() == items[idx].answer
        ok_b = preds[idx + 1] is not None and str(preds[idx + 1]).lower() == items[idx + 1].answer
        both += int(ok_a and ok_b)
        single += int(ok_a) + int(ok_b)

    return {
        "pair_accuracy": both / len(pairs),
        "item_accuracy": single / (2.0 * len(pairs)),
        "n_pairs": float(len(pairs)),
    }


# --------------------------------------------------------------------------- #
# Counterfactual moves
# --------------------------------------------------------------------------- #

@dataclass
class Counterfactual:
    item: QAItem
    scene: Scene
    original_answer: str
    new_answer: str
    moved_object: str
    from_direction: Optional[str]
    to_direction: str


def make_counterfactuals(
    items: Sequence[QAItem],
    scenes: Dict[str, Scene],
    parser: Optional[RuleBasedQuestionParser] = None,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    seed: int = 0,
    max_items: int = 500,
    require_answer_change: bool = True,
) -> List[Counterfactual]:
    """Build scenes where moving one object changes the reference answer.

    Only directional questions are usable: for a question with no region the
    answer does not depend on where objects are, so no move can change it.
    """
    parser = parser or DEFAULT_PARSER
    rng = np.random.default_rng(seed)
    out: List[Counterfactual] = []

    for item in items:
        if len(out) >= max_items:
            break
        scene = scenes.get(item.sample_token)
        if scene is None or not scene.objects:
            continue

        parsed = parser.parse(item.question)
        if not parsed.ok or parsed.query.region.kind != "direction":
            continue

        base = execute(parsed.query, scene, config)
        if not base.ok:
            continue

        movable = scene.of_category([parsed.query.category])
        if not movable:
            continue

        obj = movable[int(rng.integers(len(movable)))]
        source = direction_of(obj, config)
        targets = [d for d in DIRECTIONS if d != source]
        rng.shuffle(targets)

        for target in targets:
            distance = float(np.clip(obj.range_xy, config.min_range + 2.0, 45.0))
            new_centre = sample_position_in_direction(target, distance, config, rng)
            new_objects = [
                o.moved_to(new_centre) if o is obj else o for o in scene.objects
            ]
            # The token must be unique per counterfactual, not per sample:
            # a sample carries several questions, so reusing
            # "<token>::cf" would make later counterfactuals overwrite
            # earlier ones in the scene dictionary and silently evaluate
            # every system against the wrong scene.
            new_scene = Scene(
                sample_token=f"{scene.sample_token}::cf{len(out):05d}",
                objects=new_objects,
                scene_token=scene.scene_token,
                source=f"{scene.source}+counterfactual",
            )
            new = execute(parsed.query, new_scene, config)
            if not new.ok:
                continue
            if require_answer_change and new.answer == base.answer:
                continue

            out.append(Counterfactual(
                item=QAItem(
                    question=item.question,
                    answer=new.answer,
                    sample_token=new_scene.sample_token,
                    scene_token=scene.scene_token,
                    template=f"counterfactual::{item.template}",
                ),
                scene=new_scene,
                original_answer=base.answer,
                new_answer=new.answer,
                moved_object=obj.category,
                from_direction=source,
                to_direction=target,
            ))
            break

    return out


def counterfactual_accuracy(
    predict: PredictFn, counterfactuals: Sequence[Counterfactual]
) -> Dict[str, float]:
    """Accuracy on counterfactual scenes, plus how often the answer moved at all.

    ``changed_fraction`` is the diagnostic: a system can score poorly either by
    changing its answer to the wrong value, or by not reacting to the scene at
    all. The second failure is the interesting one.
    """
    if not counterfactuals:
        return {"accuracy": 0.0, "changed_fraction": 0.0, "n": 0.0}

    scenes = {cf.scene.sample_token: cf.scene for cf in counterfactuals}
    items = [cf.item for cf in counterfactuals]
    preds = list(predict(items, scenes))

    correct = sum(
        1 for p, cf in zip(preds, counterfactuals)
        if p is not None and str(p).strip().lower() == cf.new_answer
    )
    changed = sum(
        1 for p, cf in zip(preds, counterfactuals)
        if p is not None and str(p).strip().lower() != cf.original_answer
    )
    n = len(counterfactuals)
    return {"accuracy": correct / n, "changed_fraction": changed / n, "n": float(n)}


# --------------------------------------------------------------------------- #
# Perception perturbations
# --------------------------------------------------------------------------- #

@dataclass
class PerturbationConfig:
    drop_probability: float = 0.0
    duplicate_probability: float = 0.0
    position_jitter_m: float = 0.0
    duplicate_offset_m: float = 0.8
    seed: int = 0

    def describe(self) -> str:
        return (f"drop={self.drop_probability:g} dup={self.duplicate_probability:g} "
                f"jitter={self.position_jitter_m:g}m")


def perturb_scene(scene: Scene, config: PerturbationConfig,
                  rng: Optional[np.random.Generator] = None) -> Scene:
    """Simulate detector failures on an object list.

    This perturbs *detections*, not the point cloud. It therefore models a
    detector that misses, double-counts or mislocalises objects; it does not
    model sensor noise, and the report does not claim otherwise.
    """
    rng = rng or np.random.default_rng(config.seed)
    objects: List[Object3D] = []

    for obj in scene.objects:
        if rng.random() < config.drop_probability:
            continue
        new_obj = copy.deepcopy(obj)
        if config.position_jitter_m > 0:
            offset = rng.normal(0.0, config.position_jitter_m, size=2)
            new_obj = new_obj.moved_to(
                [obj.center[0] + offset[0], obj.center[1] + offset[1], obj.center[2]]
            )
        objects.append(new_obj)

        if rng.random() < config.duplicate_probability:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            d = config.duplicate_offset_m
            objects.append(new_obj.moved_to([
                new_obj.center[0] + d * np.cos(angle),
                new_obj.center[1] + d * np.sin(angle),
                new_obj.center[2],
            ]))

    return Scene(scene.sample_token, objects, scene.scene_token,
                 f"{scene.source}+perturbed({config.describe()})")


def perturb_scenes(scenes: Dict[str, Scene], config: PerturbationConfig) -> Dict[str, Scene]:
    rng = np.random.default_rng(config.seed)
    return {token: perturb_scene(scene, config, rng) for token, scene in scenes.items()}


def robustness_curve(
    predict: PredictFn,
    items: Sequence[QAItem],
    scenes: Dict[str, Scene],
    configs: Sequence[PerturbationConfig],
) -> List[Dict[str, float]]:
    """Accuracy under a sweep of perturbation strengths."""
    from .evaluate import accuracy  # local import keeps module graph acyclic

    curve: List[Dict[str, float]] = []
    for cfg in configs:
        perturbed = perturb_scenes(scenes, cfg)
        preds = predict(items, perturbed)
        curve.append({
            "setting": cfg.describe(),
            "drop_probability": cfg.drop_probability,
            "duplicate_probability": cfg.duplicate_probability,
            "position_jitter_m": cfg.position_jitter_m,
            "accuracy": accuracy(preds, items),
        })
    return curve


def default_perturbation_sweep(seed: int = 0) -> List[PerturbationConfig]:
    """A small, readable sweep: one failure mode varied at a time."""
    sweep = [PerturbationConfig(seed=seed)]
    for p in (0.1, 0.2, 0.3):
        sweep.append(PerturbationConfig(drop_probability=p, seed=seed))
    for p in (0.1, 0.2):
        sweep.append(PerturbationConfig(duplicate_probability=p, seed=seed))
    for j in (0.5, 1.5, 3.0):
        sweep.append(PerturbationConfig(position_jitter_m=j, seed=seed))
    return sweep
