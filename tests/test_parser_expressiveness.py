"""The rule grammar must say what the question says.

Every question below used to be mishandled twice over. The first parser
"succeeded" and answered something else: "what number of stopped things are
there" was executed as "how many objects are there" and returned 71 against a
gold answer of 10. Refusing those questions stopped the wrong answers and
replaced them with silence -- 4.4% coverage, with status words alone
accounting for 3767 of 5916 refusals.

Neither is a parser. This module pins the third behaviour: the status word and
the preamble referent go *into* the query, and the two earlier behaviours
survive only as named controls (:class:`Grammar`) so the difference between
them stays measurable.

nuScenes-QA drops questions whose count answer exceeds 10, so gold counts lie
in [0, 10]. That bound is what makes the diagnosis provable rather than
plausible: a system that only under-counts cannot err by more than 10 on any
item, so an MAE above 10 is evidence of over-counting, not of missing objects.
"""

from __future__ import annotations

import pytest

from spatial_reasoning.datasets import QAItem
from spatial_reasoning.dsl import execute as dsl_execute
from spatial_reasoning.dsl import lift
from spatial_reasoning.evaluate import count_error_decomposition
from spatial_reasoning.objects import Object3D, Scene
from spatial_reasoning.programs import execute as narrow_execute
from spatial_reasoning.question_parser import (
    DEFAULT_PARSER,
    LEGACY_PARSER,
    STRICT_PARSER,
    Operation,
    ParseStatus,
)


# --------------------------------------------------------------------------- #
# Status filters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, category, status", [
    ("How many standing pedestrians are there?", "pedestrian", "standing"),
    ("What number of stopped things are there?", "object", "stopped"),
    ("How many moving cars are to the front of the ego vehicle?", "car", "moving"),
    ("Are there any parked trucks?", "truck", "parked"),
    ("Are there any with rider bicycles?", "bicycle", "with rider"),
    ("How many sitting pedestrians are there?", "pedestrian", "sitting lying down"),
])
def test_status_is_carried_into_the_query(question, category, status):
    """Dropping the status counted the whole class: 30 pedestrians against 8."""
    res = DEFAULT_PARSER.parse(question)
    assert res.ok, res.reason
    assert res.query.category == category
    assert res.query.status == status
    assert res.query.negate_status is False


def test_negated_status_is_distinguished_from_the_plain_one():
    res = DEFAULT_PARSER.parse("How many cars are not moving?")
    assert res.ok, res.reason
    assert (res.query.status, res.query.negate_status) == ("moving", True)


def test_status_words_map_onto_the_values_the_annotation_uses():
    """A status the adapter never produces would filter every object away."""
    from spatial_reasoning.nuscenes_adapter import ATTRIBUTE_TO_STATUS
    from spatial_reasoning.question_parser import _STATUS_LEXICON

    produced = set(ATTRIBUTE_TO_STATUS.values())
    assert set(_STATUS_LEXICON.values()) <= produced


# --------------------------------------------------------------------------- #
# Where the attribute attaches
# --------------------------------------------------------------------------- #

def test_status_attaches_to_the_anchor_not_the_subject():
    """"to the left of the parked truck" constrains the truck, not the cars."""
    res = DEFAULT_PARSER.parse("How many cars are to the left of the parked truck?")
    assert res.ok, res.reason
    assert res.query.category == "car"
    assert res.query.status is None
    assert res.query.region.reference_category == "truck"
    assert res.query.anchor_status == "parked"


def test_both_sides_of_a_comparison_keep_their_own_status():
    res = DEFAULT_PARSER.parse("Are there more moving cars than parked trucks?")
    assert res.ok, res.reason
    assert res.query.operation is Operation.COMPARE_COUNT
    assert (res.query.category, res.query.status) == ("car", "moving")
    assert (res.query.second_category, res.query.second_status) == ("truck", "parked")


# --------------------------------------------------------------------------- #
# Preamble clauses and the pronoun that follows them
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, category, direction, anchor", [
    ("There is a trailer; what number of traffic cones are to the back of it?",
     "traffic_cone", "back", "trailer"),
    ("There is a bus; how many pedestrians are to the front left of it?",
     "pedestrian", "front_left", "bus"),
])
def test_preamble_binds_the_pronoun_to_the_referent(question, category, direction, anchor):
    """The grammar used to take the preamble's class as the target."""
    res = DEFAULT_PARSER.parse(question)
    assert res.ok, res.reason
    assert res.query.category == category
    assert res.query.region.kind == "relative"
    assert res.query.region.direction == direction
    assert res.query.region.reference_category == anchor


def test_preamble_status_constrains_the_referent():
    res = DEFAULT_PARSER.parse(
        "There is a parked truck; how many cars are to the back of it?")
    assert res.ok, res.reason
    assert res.query.region.reference_category == "truck"
    assert res.query.anchor_status == "parked"
    assert res.query.status is None


@pytest.mark.parametrize("question, category", [
    ("There is a truck; how many cars are there?", "car"),
    ("There are two buses; what number of pedestrians are there?", "pedestrian"),
    ("There is a bicycle, how many cars are there?", "car"),
])
def test_a_scene_setting_preamble_does_not_become_the_target(question, category):
    """No direction in the main clause: the preamble is context, not an anchor."""
    res = DEFAULT_PARSER.parse(question)
    assert res.ok, res.reason
    assert res.query.category == category
    assert res.query.region.kind == "any"


def test_main_clause_without_its_own_class_is_about_the_referent():
    res = DEFAULT_PARSER.parse("There is a parked car; how many are there?")
    assert res.ok, res.reason
    assert (res.query.category, res.query.status) == ("car", "parked")


def test_a_pronoun_with_no_antecedent_is_refused_not_guessed():
    """Nothing introduced the referent, so there is no origin to measure from."""
    res = DEFAULT_PARSER.parse("How many cars are to the back of it?")
    assert res.status is ParseStatus.UNSUPPORTED
    assert "antecedent" in res.reason


def test_an_ordinary_comma_does_not_truncate_the_question():
    """The split is only taken when the second clause is a question on its own."""
    res = DEFAULT_PARSER.parse("There are more cars than trucks, aren't there?")
    assert not res.ok or res.query.category == "car"


# --------------------------------------------------------------------------- #
# The ego origin
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, reference", [
    ("How many cars are to the left of the bus?", "bus"),
    ("How many cars are to the front of the construction vehicle?",
     "construction_vehicle"),
])
def test_ego_words_are_matched_on_word_boundaries(question, reference):
    """"us" is a substring of "bus".

    A plain substring test read "to the left of the bus" as "to the left of the
    ego vehicle" and answered about the wrong origin without any error.
    """
    res = DEFAULT_PARSER.parse(question)
    assert res.ok, res.reason
    assert res.query.region.kind == "relative"
    assert res.query.region.reference_category == reference


def test_the_ego_vehicle_is_still_recognised_as_the_origin():
    res = DEFAULT_PARSER.parse(
        "How many pedestrians are to the front left of the ego vehicle?")
    assert res.ok and res.query.region.kind == "direction"


# --------------------------------------------------------------------------- #
# What stays outside the grammar
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question", [
    "How many other things are in the same status as the bicycle?",
    "Do the trailer and the truck have the same status?",
])
def test_attribute_comparisons_are_still_refused(question):
    """The executor cannot bind one object's status as a filter over a set.

    This is a missing *operation*, not a missing slot -- CLEVR calls it the
    same-attribute relation and gives it its own function. Until that function
    exists, answering would mean inventing the binding.
    """
    assert DEFAULT_PARSER.parse(question).status is ParseStatus.UNSUPPORTED


def test_colour_is_refused_because_nuscenes_does_not_annotate_it():
    res = DEFAULT_PARSER.parse("Are there any cars of the same colour as the bus?")
    assert res.status is ParseStatus.UNSUPPORTED
    assert "colour" in res.reason


def test_out_of_grammar_question_is_declined_not_guessed():
    res = DEFAULT_PARSER.parse("What colour is the bus?")
    assert not res.ok and res.query is None


# --------------------------------------------------------------------------- #
# The controls still behave as the experiment assumes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("question, cue", [
    ("How many standing pedestrians are there?", "status"),
    ("There is a truck; how many cars are there?", "preamble"),
    ("How many cars are to the back of it?", "pronoun"),
])
def test_strict_control_refuses_with_the_documented_reason(question, cue):
    res = STRICT_PARSER.parse(question)
    assert res.status is ParseStatus.UNSUPPORTED
    assert cue in res.reason


def test_legacy_control_still_reproduces_the_original_error():
    """The measurement is only meaningful if the control is really the old bug."""
    res = LEGACY_PARSER.parse("What number of stopped things are there?")
    assert res.ok and res.query.category == "object"
    assert res.query.status is None          # the word is dropped, as it was

    res = LEGACY_PARSER.parse("There is a truck; how many cars are there?")
    assert res.ok and res.query.category == "truck"   # the preamble wins


# --------------------------------------------------------------------------- #
# The two executors must agree
# --------------------------------------------------------------------------- #

def _obj(category: str, x: float, y: float, status=None) -> Object3D:
    return Object3D(category=category, center=[x, y, 0.8], size=[2.0, 4.0, 1.6],
                    yaw=0.0, score=1.0, status=status)


def _status_scene() -> Scene:
    return Scene("s0", [
        _obj("car", 20.0, 0.0, "moving"),
        _obj("car", 25.0, 1.0, "parked"),
        _obj("car", 30.0, 2.0, "parked"),
        _obj("traffic_cone", 15.0, 0.0, None),
    ])


def test_status_filter_is_applied_by_both_executors():
    """``programs.execute`` and ``dsl.execute`` answer the same programs.

    Two executors that must agree is how the status filter got lost once
    already: the narrow one counted the whole class while the DSL filtered.
    """
    query = DEFAULT_PARSER.parse("How many parked cars are there?").query
    assert narrow_execute(query, _status_scene()).answer == "2"
    assert dsl_execute(lift(query), _status_scene()).answer == "2"


def test_unknown_status_satisfies_neither_the_filter_nor_its_negation():
    """A traffic cone carries no attribute: it is not "standing", and it is
    not "not standing" either. Counting it under the negation would assert
    something the annotation does not say."""
    scene = _status_scene()
    q = DEFAULT_PARSER.parse("How many things are not moving?").query
    assert q.negate_status is True
    # two parked cars; the cone has no status and the moving car is excluded
    assert narrow_execute(q, scene).answer == "2"
    assert dsl_execute(lift(q), scene).answer == "2"


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
