"""Regression tests for inputs that were silently accepted instead of refused.

Two defect classes, both of which produce a confident wrong answer rather than
an error: partial data treated as complete (a pose mapping missing an entry),
and a value of the wrong type coerced into a plausible default.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from spatial_reasoning.llm_parser import InvalidProgram, QwenQuestionParser, query_from_json
from spatial_reasoning.objects import FrameError, load_nuscenes_detections
from spatial_reasoning.run_experiments import render_markdown


def _submission(path: str, tokens) -> None:
    results = {
        t: [{
            "sample_token": t,
            "translation": [1000.0 + 5 * i, 600.0, 1.0],
            "size": [1.9, 4.6, 1.6],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "detection_name": "car",
            "detection_score": 0.9,
        }]
        for i, t in enumerate(tokens)
    }
    json.dump({"results": results}, open(path, "w", encoding="utf-8"))


POSE = {"translation": [1000.0, 600.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}


# --------------------------------------------------------------------------- #
# Partial pose data must not produce a file in two frames
# --------------------------------------------------------------------------- #

def test_empty_pose_mapping_is_refused():
    """{} converted nothing while labelling the result as ego-frame."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.json")
        _submission(path, ["a"])
        with pytest.raises(FrameError):
            load_nuscenes_detections(path, ego_poses={})


def test_a_missing_pose_is_refused_rather_than_mixing_frames():
    """'a' was converted and 'b' left in global coordinates, both labelled ego."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.json")
        _submission(path, ["a", "b"])
        with pytest.raises(FrameError) as exc:
            load_nuscenes_detections(path, ego_poses={"a": POSE})
        assert "b" in str(exc.value)


def test_complete_poses_still_work():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "s.json")
        _submission(path, ["a", "b"])
        scenes = load_nuscenes_detections(path, ego_poses={"a": POSE, "b": POSE})
        assert set(scenes) == {"a", "b"}
        for s in scenes.values():
            assert s.objects[0].range_xy < 50.0


# --------------------------------------------------------------------------- #
# A present-but-wrong-typed category must not become "every object"
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [0, False, [], {}, 3.5])
def test_non_string_category_is_rejected(value):
    """str(x or "object") turned all of these into a count of every object."""
    program = json.dumps({"op": "count", "target": {"category": value}})
    with pytest.raises(InvalidProgram):
        query_from_json(program)


def test_absent_category_still_defaults_to_object():
    q = query_from_json(json.dumps({"op": "count", "target": {}}))
    assert q is not None and q.target.category == "object"
    q = query_from_json(json.dumps({"op": "count", "target": {"category": None}}))
    assert q is not None and q.target.category == "object"


# --------------------------------------------------------------------------- #
# Reporting must survive a probe that could not be computed
# --------------------------------------------------------------------------- #

def _minimal_results(changed):
    return {
        "data": {"n_items": 1, "n_samples": 1, "train": 1, "val": 0, "test": 1,
                 "answer_vocabulary": 2, "warnings": []},
        "majority_baseline_accuracy": 0.5,
        "trivial_answer_fraction": 0.5,
        "parser_coverage": {"coverage": 1.0},
        "systems": [{
            "name": "sym", "accuracy": 0.5, "answered_fraction": 1.0,
            "accuracy_when_answered": 0.5, "count_mae": None,
            "count_coverage": {"numeric_fraction": 0.5},
            "cost": {"ms_per_item": 1.0}, "extra": {"num_parameters": 0},
            "by_answer_type": {}, "by_template": {},
        }],
        "contrast_pairs": {"sym": {"pair_accuracy": 0.0, "item_accuracy": 0.0}},
        "counterfactuals": {"n": 0, "note": "", "systems": {
            "sym": {"accuracy": 0.0, "changed_fraction": changed, "n": 0.0}}},
        "robustness": {"sym": []},
        "error_decomposition": {"counts": {"correct": 1}, "fractions": {"correct": 1.0}},
    }


def test_markdown_survives_an_uncomputable_changed_fraction():
    """It used to raise TypeError formatting None, after the JSON was written."""
    out = render_markdown(_minimal_results(None))
    assert "n/a" in out


def test_markdown_still_formats_a_real_changed_fraction():
    assert "0.250" in render_markdown(_minimal_results(0.25))


def test_markdown_reports_the_numeric_share_beside_the_mae():
    out = render_markdown(_minimal_results(0.25))
    assert "numeric share" in out
    assert "count MAE (numeric answers)" in out


# --------------------------------------------------------------------------- #
# Cache reuse is counted once, at consumption
# --------------------------------------------------------------------------- #

def test_foreign_reuse_cannot_exceed_one_per_question():
    """_usable ran twice per question, so the rate could read above 100%."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        a = QwenQuestionParser(model_name="m", cache_path=path, max_new_tokens=160)
        a._cache["Q"] = {"raw": '{"op": null}', "config": a.config_id()}
        a.save_cache()

        b = QwenQuestionParser(model_name="m", cache_path=path,
                               max_new_tokens=96, allow_foreign=True)
        b.warm_cache(["Q"], verbose=False)     # availability check
        b.parse("Q")                           # actual consumption
        assert b.stats.total == 1
        assert b.stats.foreign_used == 1
        assert b.stats.as_dict()["foreign_cache_used"] == 1.0


def test_own_entries_are_not_counted_as_foreign():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        p = QwenQuestionParser(model_name="m", cache_path=path)
        p._cache["Q"] = {"raw": '{"op": null}', "config": p.config_id()}
        p.parse("Q")
        assert p.stats.foreign_used == 0
        assert p.stats.cache_hits == 1
