"""Regression tests for cache provenance, strict validation and the metrics.

Each test corresponds to a defect that let a run describe itself inaccurately:
a cache that forgot which configuration produced it, a validator that accepted
a program meaning something other than what the model wrote, and metrics whose
value depended on the answer distribution rather than on the system.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile

import pytest

from spatial_reasoning.datasets import QAItem
from spatial_reasoning.evaluate import count_answer_coverage, count_mae
from spatial_reasoning.llm_parser import (
    InvalidProgram,
    QwenQuestionParser,
    query_from_json,
)
from spatial_reasoning.regions import NUSCENES_QA_REGION_CONFIG

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="requires torch")


def _program(**over) -> str:
    sel = {"category": "car", "status": None, "negate_status": False,
           "direction": None, "anchor": None}
    sel.update(over.pop("target", {}))
    blob = {"op": "count", "target": sel, "second": None, "comparison": None}
    blob.update(over)
    return json.dumps(blob)


# --------------------------------------------------------------------------- #
# Strict validation
# --------------------------------------------------------------------------- #

def test_negate_status_must_be_a_real_boolean():
    """bool("false") is True; coercion turned a correct emission into a negation."""
    with pytest.raises(InvalidProgram):
        query_from_json(_program(target={"negate_status": "false"}))
    with pytest.raises(InvalidProgram):
        query_from_json(_program(target={"negate_status": 0}))
    assert query_from_json(_program(target={"negate_status": True})) is not None


def test_empty_object_is_malformed_not_a_refusal():
    """{} is broken output; {"op": null} is the model declining. Not the same."""
    with pytest.raises(InvalidProgram):
        query_from_json("{}")
    assert query_from_json('{"op": null}') is None


def test_unknown_fields_are_rejected():
    """An ignored field drops a constraint the model meant to express."""
    with pytest.raises(InvalidProgram):
        query_from_json(_program(target={"radius": 10}))
    with pytest.raises(InvalidProgram):
        query_from_json(_program(nonsense=1))


def test_wrong_types_are_rejected():
    with pytest.raises(InvalidProgram):
        query_from_json(_program(target={"direction": 3}))
    with pytest.raises(InvalidProgram):
        query_from_json(_program(target={"status": ["moving"]}))
    with pytest.raises(InvalidProgram):
        query_from_json(_program(op=7))


# --------------------------------------------------------------------------- #
# Cache provenance
# --------------------------------------------------------------------------- #

def _parser(path, **kw):
    return QwenQuestionParser(model_name="dummy/model", cache_path=path, **kw)


def test_saving_does_not_relabel_foreign_entries():
    """One load/save cycle used to stamp the current config onto everything."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")

        a = _parser(path, max_new_tokens=160)
        a._cache["Q1"] = {"raw": '{"op": null}', "config": a.config_id()}
        a.save_cache()
        id_a = a.config_id()

        # A different token budget is a different configuration.
        b = _parser(path, max_new_tokens=96)
        assert b.config_id() != id_a
        assert b.foreign_entries == 1
        b._cache["Q2"] = {"raw": '{"op": null}', "config": b.config_id()}
        b.save_cache()

        payload = json.load(open(path, encoding="utf-8"))
        assert payload["entries"]["Q1"]["config"] == id_a
        assert payload["entries"]["Q2"]["config"] == b.config_id()
        assert id_a in payload["configs"] and b.config_id() in payload["configs"]


def test_foreign_entries_are_not_reused_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")
        a = _parser(path, max_new_tokens=160)
        a._cache["Q"] = {"raw": '{"op": null}', "config": a.config_id()}
        a.save_cache()

        b = _parser(path, max_new_tokens=96)
        assert b._usable("Q") is None            # would be regenerated

        # _usable is a pure lookup and counts nothing: it is called both to
        # test availability and to fetch, so counting there double-charged the
        # same entry. Consumption through parse() is what increments.
        c = _parser(path, max_new_tokens=96, allow_foreign=True)
        assert c._usable("Q") == '{"op": null}'
        assert c.stats.foreign_used == 0
        c.parse("Q")
        assert c.stats.foreign_used == 1


def test_legacy_flat_cache_keeps_unknown_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.json")
        json.dump({"Q": '{"op": null}'}, open(path, "w", encoding="utf-8"))

        p = _parser(path)
        assert p._cache["Q"]["config"] == "unknown"
        assert p.foreign_entries == 1
        assert p._usable("Q") is None


def test_fingerprint_covers_the_few_shot_examples():
    from spatial_reasoning import llm_parser

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        before = _parser(path).config_id()
        original = llm_parser.FEW_SHOT
        try:
            llm_parser.FEW_SHOT = original[:-1]
            after = _parser(path).config_id()
        finally:
            llm_parser.FEW_SHOT = original
        assert before != after


def test_fingerprint_changes_with_the_direction_set():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "c.json")
        eight = _parser(path).config_id()
        six = _parser(path, directions=NUSCENES_QA_REGION_CONFIG.directions()).config_id()
        assert eight != six


# --------------------------------------------------------------------------- #
# Counting metrics
# --------------------------------------------------------------------------- #

def _count_items():
    return [QAItem("How many cars are there?", "3", "s"),
            QAItem("How many trucks are there?", "5", "s")]


def test_non_numeric_predictions_are_excluded_not_scored_as_zero():
    items = _count_items()
    assert count_mae(["3", "5"], items) == 0.0
    # "yes" used to be charged an error of 5, as though the system said zero.
    assert count_mae(["3", "yes"], items) == 0.0
    assert count_mae(["4", "yes"], items) == 1.0


def test_count_coverage_exposes_the_denominator():
    items = _count_items()
    cov = count_answer_coverage(["3", "yes"], items)
    assert cov["n_count_questions"] == 2
    assert cov["numeric_fraction"] == 0.5
    assert cov["abstained_fraction"] == 0.0

    cov = count_answer_coverage(["3", None], items)
    assert cov["abstained_fraction"] == 0.5


def test_count_mae_is_none_when_nothing_qualifies():
    assert count_mae(["yes", "no"], _count_items()) is None
    assert count_mae([], []) is None


# --------------------------------------------------------------------------- #
# Counterfactual reactivity
# --------------------------------------------------------------------------- #

def test_changed_fraction_compares_the_system_to_itself():
    """It used to compare against the original reference answer instead."""
    from spatial_reasoning.consistency import counterfactual_accuracy, make_counterfactuals
    from spatial_reasoning.objects import Object3D, Scene

    scene = Scene("s0", [
        Object3D("car", [20.0, 1.0, 0.8], [2.0, 4.0, 1.6]),
        Object3D("car", [5.0, 18.0, 0.8], [2.0, 4.0, 1.6]),
    ], scene_token="sc0")
    items = [QAItem("How many cars are to the front of the ego vehicle?",
                    "1", "s0", "sc0")]
    cfs = make_counterfactuals(items, {"s0": scene},
                               config=NUSCENES_QA_REGION_CONFIG, seed=0)
    assert cfs

    # A system that always says "7" is wrong everywhere but never reacts.
    stubborn = lambda its, scns: ["7"] * len(its)
    out = counterfactual_accuracy(stubborn, cfs, original_scenes={"s0": scene})
    assert out["changed_fraction"] == 0.0

    # Without the original scenes the field is None, not a wrong number.
    assert counterfactual_accuracy(stubborn, cfs)["changed_fraction"] is None
