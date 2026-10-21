"""The rule grammar must refuse what it cannot express, not reinterpret it.

Every question below used to parse "successfully" into a program that answered
something else. The archived run shows the cost: "what number of stopped things
are there" was executed as "how many objects are there" and returned 71 against
a gold answer of 10. A single error of that size dominates a mean absolute
error computed over a few hundred counting questions.

nuScenes-QA drops questions whose count answer exceeds 10, so gold counts lie
in [0, 10]. That bound is what makes the diagnosis provable rather than
plausible: a system that only under-counts cannot err by more than 10 on any
item, so an MAE above 10 is evidence of over-counting, not of missing objects.
"""

from __future__ import annotations

import pytest

from spatial_reasoning.evaluate import count_error_decomposition
from spatial_reasoning.datasets import QAItem
from spatial_reasoning.question_parser import DEFAULT_PARSER, ParseStatus


@pytest.mark.parametrize("question, cue", [
    ("How many standing pedestrians are there?", "status"),
    ("What number of stopped things are there?", "status"),
    ("How many moving cars are to the front of the ego vehicle?", "status"),
    ("Are there any parked trucks?", "status"),
    ("Are there any with rider bicycles?", "status"),
])
def test_status_constrained_questions_are_refused(question, cue):
    """Dropping the status counted the whole class: 30 pedestrians against 8."""
    res = DEFAULT_PARSER.parse(question)
    assert res.status is ParseStatus.UNSUPPORTED
    assert cue in res.reason
    assert res.query is None


@pytest.mark.parametrize("question", [
    "How many other things are in the same status as the bicycle?",
    "Do the trailer and the truck have the same status?",
])
def test_attribute_comparisons_are_refused(question):
    assert DEFAULT_PARSER.parse(question).status is ParseStatus.UNSUPPORTED


@pytest.mark.parametrize("question", [
    "There is a trailer; what number of traffic cones are to the back of it?",
    "There is a bus; how many pedestrians are to the front left of it?",
])
def test_preamble_questions_are_refused(question):
    """The grammar took the preamble's class as the target: truck, not pedestrian."""
    res = DEFAULT_PARSER.parse(question)
    assert res.status is ParseStatus.UNSUPPORTED
    assert res.query is None


@pytest.mark.parametrize("question, category", [
    ("How many cars are there?", "car"),
    ("Are there any trailers?", "trailer"),
    ("How many cars are to the front left of the ego vehicle?", "car"),
    ("Are there more cars than trucks in the scene?", "car"),
])
def test_expressible_questions_still_parse(question, category):
    res = DEFAULT_PARSER.parse(question)
    assert res.ok
    assert res.query.category == category


# --------------------------------------------------------------------------- #
# The under/over split
# --------------------------------------------------------------------------- #

def _items(golds):
    return [QAItem(f"How many cars are there? {i}", str(g), "s")
            for i, g in enumerate(golds)]


def test_decomposition_separates_the_two_failure_modes():
    items = _items([5, 5, 5])
    out = count_error_decomposition(["3", "5", "9"], items)
    assert out["under"] == pytest.approx(2 / 3)      # 5 -> 3
    assert out["over"] == pytest.approx(4 / 3)       # 5 -> 9
    assert out["mae"] == pytest.approx(out["under"] + out["over"])
    assert out["n_undercounted"] == 1
    assert out["n_overcounted"] == 1


def test_pure_undercounting_cannot_exceed_the_gold_bound():
    """The argument that rules out 'missing objects' as the whole story."""
    items = _items([10, 10, 10])
    out = count_error_decomposition(["0", "0", "0"], items)
    assert out["over"] == 0.0
    assert out["mae"] <= out["gold_max"]


def test_one_large_overcount_dominates_the_mean():
    items = _items([10] + [1] * 9)
    out = count_error_decomposition(["71"] + ["1"] * 9, items)
    assert out["over"] == pytest.approx(6.1)
    assert out["under"] == 0.0
    assert out["over_share"] == 1.0
    assert out["worst"][0]["error"] == 61


def test_decomposition_is_empty_without_numeric_answers():
    out = count_error_decomposition(["yes", None], _items([1, 2]))
    assert out["n"] == 0 and out["mae"] is None
