"""The typed functional language, checked without a model.

The expensive run is the one you cannot afford to discover afterwards was
measuring the wrong thing, so everything here runs on a CPU in under a second:
the catalogue's semantics, the type checker, what the parser calls a refusal
as opposed to a malformed output, and the shape of the wide manifest.
"""

from __future__ import annotations

import json

import pytest

from spatial_reasoning.functional import (
    CATEGORY_VOCABULARY,
    STATUSES,
    Failure,
    ProgramError,
    catalogue,
    execute,
    json_schema,
    program_from_json,
)
from spatial_reasoning.functional_parser import (
    FEW_SHOT,
    _extract_json,
    build_system_prompt,
)
from spatial_reasoning.manifest import SCOPES, WIDE_FAMILIES, classify
from spatial_reasoning.objects import Object3D, Scene
from spatial_reasoning.regions import RegionConfig

CONFIG = RegionConfig(layout="nuscenes_qa")
CAT = catalogue(CONFIG)


def _obj(category, x, y, status=None, token=None):
    return Object3D(category=category, center=[x, y, 0.8], size=[2.0, 4.0, 1.6],
                    yaw=0.0, score=1.0, status=status, token=token)


def _scene():
    return Scene("s0", [
        _obj("trailer", 20.0, 0.0, "parked", "t1"),
        _obj("traffic_cone", -18.0, 2.0, None, "c1"),
        _obj("traffic_cone", -22.0, -3.0, None, "c2"),
        _obj("car", 10.0, 18.0, "moving", "k1"),
        _obj("car", -14.0, 9.0, "parked", "k2"),
        _obj("pedestrian", 5.0, 3.0, "standing", "p1"),
    ])


def _program(nodes):
    return program_from_json({"program": nodes}, CAT)


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #

def test_every_signature_has_an_implementation():
    from spatial_reasoning.functional import _APPLY
    assert set(_APPLY) == set(CAT)


def test_the_vocabularies_mirror_what_the_annotation_produces():
    """A constant the adapter never emits would filter every object away."""
    from spatial_reasoning.nuscenes_adapter import ATTRIBUTE_TO_STATUS
    assert set(STATUSES) == set(ATTRIBUTE_TO_STATUS.values())
    assert "object" in CATEGORY_VOCABULARY


def test_the_direction_vocabulary_follows_the_layout():
    """A program written for one sector layout is not a program in another."""
    six = catalogue(RegionConfig(layout="nuscenes_qa"))["relate_ego"].vocabulary
    eight = catalogue(RegionConfig(layout="octant"))["relate_ego"].vocabulary
    assert "left" not in six and "left" in eight


@pytest.mark.parametrize("question, nodes", FEW_SHOT)
def test_every_few_shot_example_type_checks(question, nodes):
    """A prompt that teaches an invalid program teaches the model to write one."""
    assert _program(nodes).nodes


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def test_a_relation_anchored_on_a_resolved_object():
    program = _program([
        {"function": "scene", "value_inputs": [], "inputs": []},
        {"function": "filter_category", "value_inputs": ["trailer"], "inputs": [0]},
        {"function": "unique", "value_inputs": [], "inputs": [1]},
        {"function": "relate", "value_inputs": ["back"], "inputs": [2]},
        {"function": "filter_category", "value_inputs": ["traffic_cone"], "inputs": [3]},
        {"function": "count", "value_inputs": [], "inputs": [4]},
    ])
    assert execute(program, _scene(), CONFIG).answer == "2"


def test_unique_reports_ambiguity_instead_of_choosing():
    """CLEVR prunes ill-posed questions when it generates them.

    On data we did not generate, well-posedness has to be detected. Taking the
    first match would be right about half the time and could not be audited.
    """
    program = _program([
        {"function": "scene", "value_inputs": [], "inputs": []},
        {"function": "filter_category", "value_inputs": ["car"], "inputs": [0]},
        {"function": "unique", "value_inputs": [], "inputs": [1]},
        {"function": "query_status", "value_inputs": [], "inputs": [2]},
    ])
    result = execute(program, _scene(), CONFIG)
    assert result.answer is None
    assert result.failure is Failure.AMBIGUOUS


def test_same_status_excludes_the_object_it_was_asked_about():
    """"the OTHER things in the same status" -- the reference is not one of them."""
    program = _program([
        {"function": "scene", "value_inputs": [], "inputs": []},
        {"function": "filter_category", "value_inputs": ["trailer"], "inputs": [0]},
        {"function": "unique", "value_inputs": [], "inputs": [1]},
        {"function": "same_status", "value_inputs": [], "inputs": [2]},
        {"function": "count", "value_inputs": [], "inputs": [3]},
    ])
    # one parked car; the parked trailer itself is excluded
    assert execute(program, _scene(), CONFIG).answer == "1"


def test_an_unknown_status_satisfies_neither_the_filter_nor_its_negation():
    scene = _scene()
    kept = _program([
        {"function": "scene", "value_inputs": [], "inputs": []},
        {"function": "filter_not_status", "value_inputs": ["moving"], "inputs": [0]},
        {"function": "count", "value_inputs": [], "inputs": [1]},
    ])
    # parked trailer, parked car, standing pedestrian; the two cones have no
    # attribute and are neither moving nor not-moving
    assert execute(kept, scene, CONFIG).answer == "3"


def test_set_operations_do_not_double_count():
    program = _program([
        {"function": "scene", "value_inputs": [], "inputs": []},
        {"function": "filter_category", "value_inputs": ["car"], "inputs": [0]},
        {"function": "or_", "value_inputs": [], "inputs": [1, 1]},
        {"function": "count", "value_inputs": [], "inputs": [2]},
    ])
    assert execute(program, _scene(), CONFIG).answer == "2"


def test_the_trace_records_what_each_step_selected():
    program = _program([
        {"function": "scene", "value_inputs": [], "inputs": []},
        {"function": "filter_category", "value_inputs": ["traffic_cone"], "inputs": [0]},
        {"function": "count", "value_inputs": [], "inputs": [1]},
    ])
    result = execute(program, _scene(), CONFIG)
    assert [step.function for step in result.trace] == ["scene", "filter_category", "count"]
    assert set(result.trace[1].tokens) == {"c1", "c2"}


# --------------------------------------------------------------------------- #
# The type checker
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("nodes, failure", [
    ([{"function": "count", "value_inputs": [], "inputs": []}], Failure.ILL_TYPED),
    ([{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "relate", "value_inputs": ["back"], "inputs": [0]}],
     Failure.ILL_TYPED),                       # relate wants an object, got a set
    ([{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_category", "value_inputs": ["red"], "inputs": [0]},
      {"function": "count", "value_inputs": [], "inputs": [1]}], Failure.ILL_TYPED),
    ([{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "relate_ego", "value_inputs": ["left"], "inputs": [0]},
      {"function": "count", "value_inputs": [], "inputs": [1]}], Failure.ILL_TYPED),
    ([{"function": "scene", "value_inputs": [], "inputs": []}], Failure.ILL_TYPED),
    ([{"function": "nope", "value_inputs": [], "inputs": []}], Failure.MALFORMED),
    ("not json at all", Failure.MALFORMED),
])
def test_an_ill_formed_program_is_refused_with_its_reason(nodes, failure):
    with pytest.raises(ProgramError) as caught:
        program_from_json(nodes if isinstance(nodes, str) else {"program": nodes}, CAT)
    assert caught.value.failure is failure


def test_an_out_of_vocabulary_constant_is_never_coerced():
    """"front-left" under a six-sector layout is a different question."""
    with pytest.raises(ProgramError):
        _program([{"function": "scene", "value_inputs": [], "inputs": []},
                  {"function": "relate_ego", "value_inputs": ["north"], "inputs": [0]},
                  {"function": "count", "value_inputs": [], "inputs": [1]}])


def test_an_input_index_must_point_at_an_earlier_node():
    with pytest.raises(ProgramError):
        _program([{"function": "scene", "value_inputs": [], "inputs": []},
                  {"function": "count", "value_inputs": [], "inputs": [5]}])


# --------------------------------------------------------------------------- #
# Reading the model's output
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", [
    '{"program": [{"function":"scene","value_inputs":[],"inputs":[]}]}',
    '```json\n{"program": [{"function":"scene","value_inputs":[],"inputs":[]}]}\n```',
    'Sure. {"program": [{"function":"scene","value_inputs":[],"inputs":[]}]} done.',
])
def test_the_json_is_recovered_from_whatever_wraps_it(raw):
    assert json.loads(_extract_json(raw))["program"][0]["function"] == "scene"


def test_the_prompt_states_the_layout_it_was_built_for():
    prompt = build_system_prompt(CONFIG)
    for direction in CONFIG.directions():
        assert direction in prompt
    assert "left'" not in prompt.replace("front_left", "").replace("back_left", "")


def test_the_schema_offers_one_variant_per_function():
    schema = json_schema(CONFIG)
    variants = schema["properties"]["program"]["items"]["anyOf"]
    assert len(variants) == len(CAT)


# --------------------------------------------------------------------------- #
# The wide manifest
# --------------------------------------------------------------------------- #

def test_the_wide_scope_admits_what_the_flat_one_cannot_reach():
    for question, template, family in [
        ("There is a parked thing; what is it?", "object", "identify_object"),
        ("The construction vehicle is in what status?", "status", "query_status"),
        ("Do the trailer and the truck have the same status?", "comparison",
         "compare_attribute"),
        ("How many other things are in the same status as the bicycle?", "count",
         "same_attribute_set"),
    ]:
        assert classify(question, template, "narrow")[0] is None
        assert classify(question, template, "wide")[0] == family


def test_colour_is_excluded_in_every_scope():
    """nuScenes does not annotate colour, so no language can answer it."""
    for scope in SCOPES:
        family, reason, _ = classify(
            "Are there any cars of the same colour as the bus?", "exist", scope)
        assert family is None and reason == "colour_is_not_annotated"


def test_an_unknown_scope_is_an_error_not_a_default():
    with pytest.raises(ValueError):
        classify("How many cars are there?", "count", "everything")


def test_the_wide_families_are_disjoint_from_the_narrow_ones():
    from spatial_reasoning.manifest import FAMILIES
    assert not set(FAMILIES) & set(WIDE_FAMILIES)
