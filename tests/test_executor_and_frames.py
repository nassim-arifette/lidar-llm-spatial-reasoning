"""Regression tests for the four blockers a config-propagation sweep found.

All four shared a shape: a component was fixed, but a *different* path reached
the unfixed version. Unit tests on the fixed component passed throughout.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from spatial_reasoning import programs
from spatial_reasoning.objects import FrameError, Object3D, Scene, load_nuscenes_detections
from spatial_reasoning.question_parser import RuleBasedQuestionParser
from spatial_reasoning.regions import NUSCENES_QA_REGION_CONFIG, Region
from spatial_reasoning.synthetic import generate_corpus

QA = NUSCENES_QA_REGION_CONFIG


def car(x: float, y: float) -> Object3D:
    return Object3D("car", [x, y, 0.8], [2.0, 4.0, 1.6])


# --------------------------------------------------------------------------- #
# Blocker 2: the narrow executor needed the same guard as the DSL one
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scene", [
    Scene("empty", []),
    Scene("no-car", [Object3D("pedestrian", [10.0, 1.0, 0.8], [0.7, 0.7, 1.7])]),
    Scene("with-car", [car(1.0, 10.0)]),
])
def test_narrow_executor_refuses_an_undefined_direction(scene):
    """On an empty scene it answered "0"/OK; on a populated one it raised."""
    query = RuleBasedQuestionParser(allow_relative_regions=True).parse(
        "How many cars are to the left of the ego vehicle?").query
    result = programs.execute(query, scene, QA)
    assert result.status is programs.ExecStatus.UNSUPPORTED
    assert result.answer is None


def test_symbolic_system_abstains_instead_of_crashing():
    system = programs.SymbolicSpatialQA(
        parser=RuleBasedQuestionParser(allow_relative_regions=True),
        config=QA, fallback_answer=None)
    out = system.answer("How many cars are to the left of the ego vehicle?",
                        Scene("s", [car(1.0, 10.0)]))
    assert out.abstained
    assert not out.executed


def test_compare_path_is_guarded_too():
    query = RuleBasedQuestionParser().parse(
        "Are there more cars than trucks to the left of the ego vehicle?").query
    if query is None:
        pytest.skip("grammar declined this phrasing")
    result = programs.execute(query, Scene("s", [car(1.0, 10.0)]), QA)
    assert result.status is programs.ExecStatus.UNSUPPORTED


def test_building_a_predicate_validates_eagerly():
    """The guard lives where every executor passes, not at each call site."""
    with pytest.raises(KeyError):
        Region(kind="direction", direction="left").predicate(QA)
    assert Region(kind="direction", direction="front_left").predicate(QA) is not None


# --------------------------------------------------------------------------- #
# Blockers 3 and 4: the synthetic corpus must honour the active layout
# --------------------------------------------------------------------------- #

def test_corpus_can_be_generated_under_the_benchmark_layout():
    """This used to raise: the generator drew directions from the global eight."""
    corpus = generate_corpus(n_scenes=2, samples_per_scene=3,
                             questions_per_sample=4, seed=0, config=QA)
    assert corpus.items
    for q in corpus.gold_queries.values():
        if q.region.kind == "direction":
            assert q.region.direction in QA.directions()


def test_generated_questions_never_name_an_absent_direction():
    corpus = generate_corpus(n_scenes=2, samples_per_scene=3,
                             questions_per_sample=6, seed=3, config=QA)
    for item in corpus.items:
        assert " left of" not in item.question or "front left" in item.question \
            or "back left" in item.question
        assert " right of" not in item.question or "front right" in item.question \
            or "back right" in item.question


def test_gold_answers_agree_with_the_executor_under_the_same_layout():
    """A corpus built under one convention and scored under another is wrong."""
    from spatial_reasoning.dsl import execute as dsl_execute
    from spatial_reasoning.dsl import lift

    corpus = generate_corpus(n_scenes=2, samples_per_scene=3,
                             questions_per_sample=5, seed=7, config=QA,
                             detector=None)
    for item in corpus.items:
        gold = corpus.gold_queries[f"{item.sample_token}::{item.question}"]
        result = dsl_execute(lift(gold) if not hasattr(gold, "target") else gold,
                             corpus.oracle_scenes[item.sample_token], QA)
        assert result.answer == item.answer, item.question


# --------------------------------------------------------------------------- #
# Blocker 1: the frame of a detection file must not be guessed
# --------------------------------------------------------------------------- #

def _submission(path: str) -> None:
    payload = {"results": {"tok": [{
        "sample_token": "tok",
        "translation": [1010.0, 620.0, 1.0],      # global frame, as released
        "size": [1.9, 4.6, 1.6],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "detection_name": "car",
        "detection_score": 0.9,
    }]}}
    json.dump(payload, open(path, "w", encoding="utf-8"))


def test_loading_a_global_frame_submission_without_poses_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub.json")
        _submission(path)
        with pytest.raises(FrameError):
            load_nuscenes_detections(path)


def test_poses_convert_the_boxes_into_the_ego_frame():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub.json")
        _submission(path)
        poses = {"tok": {"translation": [1000.0, 600.0, 0.0],
                         "rotation": [1.0, 0.0, 0.0, 0.0]}}
        scenes = load_nuscenes_detections(path, ego_poses=poses)
        obj = scenes["tok"].objects[0]
        assert obj.range_xy < 50.0            # not a kilometre away
        assert scenes["tok"].source == "detection_ego_from_poses"


def test_explicit_opt_in_is_still_possible():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub.json")
        _submission(path)
        scenes = load_nuscenes_detections(path, assume_ego_frame=True)
        assert scenes["tok"].source == "detection_assumed_ego"
