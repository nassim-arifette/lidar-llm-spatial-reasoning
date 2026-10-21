"""Integration tests: do the scripts use the components they claim to?

Unit tests on the geometry functions pass whether or not a script bothers to
pass the configuration down. These tests assert the wiring instead: which
layout a script selects, which directions a parser is built with, and that an
invalid program is refused even when no object would have triggered the check.

Every test here corresponds to a defect that shipped and that the unit tests
did not catch.
"""

from __future__ import annotations

import json

import pytest

from spatial_reasoning import run_experiments, run_parser_comparison
from spatial_reasoning.dsl import ExecStatus, Op, Query, Selector, execute
from spatial_reasoning.llm_parser import (
    InvalidProgram,
    build_system_prompt,
    query_from_json,
)
from spatial_reasoning.objects import Object3D, Scene
from spatial_reasoning.regions import (
    NUSCENES_QA_REGION_CONFIG,
    Region,
    RegionConfig,
    validate_region,
)


# --------------------------------------------------------------------------- #
# The scripts select the right layout
# --------------------------------------------------------------------------- #

def test_nuscenes_source_defaults_to_the_benchmark_layout():
    """--source nuscenes must not silently run on the octant convention."""
    parser = run_experiments.build_arg_parser()
    args = parser.parse_args(["--source", "nuscenes",
                              "--metadata-root", "x", "--qa", "y"])
    layout = args.layout or ("nuscenes_qa" if args.source == "nuscenes" else "octant")
    assert layout == "nuscenes_qa"
    assert len(RegionConfig(layout=layout).directions()) == 6


def test_synthetic_source_keeps_the_octant_layout():
    parser = run_experiments.build_arg_parser()
    args = parser.parse_args(["--source", "synthetic"])
    layout = args.layout or ("nuscenes_qa" if args.source == "nuscenes" else "octant")
    assert layout == "octant"


def test_parser_comparison_defaults_to_the_benchmark_layout():
    args = run_parser_comparison.build_arg_parser().parse_args(
        ["--metadata-root", "x", "--qa", "y"])
    assert args.layout == "nuscenes_qa"
    assert len(RegionConfig(layout=args.layout).directions()) == 6


# --------------------------------------------------------------------------- #
# Prompt, validator and executor agree on the legal directions
# --------------------------------------------------------------------------- #

def test_prompt_advertises_exactly_the_active_directions():
    """The direction list is rendered into the prompt, so it must match."""
    six = NUSCENES_QA_REGION_CONFIG.directions()
    prompt = build_system_prompt(six)
    assert json.dumps(list(six)) in prompt
    assert json.dumps(list(RegionConfig().directions())) not in prompt


def test_validator_rejects_a_direction_outside_the_active_layout():
    six = frozenset(NUSCENES_QA_REGION_CONFIG.directions())
    program = json.dumps({
        "op": "count",
        "target": {"category": "car", "status": None, "negate_status": False,
                   "direction": "left", "anchor": None},
        "second": None, "comparison": None,
    })
    with pytest.raises(InvalidProgram):
        query_from_json(program, six)

    # The same program is fine under the eight-sector layout.
    assert query_from_json(program) is not None


def test_validator_checks_anchors_too():
    six = frozenset(NUSCENES_QA_REGION_CONFIG.directions())
    program = json.dumps({
        "op": "count",
        "target": {"category": "car", "direction": "front",
                   "anchor": {"category": "truck", "direction": "right"}},
    })
    with pytest.raises(InvalidProgram):
        query_from_json(program, six)


# --------------------------------------------------------------------------- #
# An invalid program is refused regardless of the scene
# --------------------------------------------------------------------------- #

def _query_left() -> Query:
    return Query(Op.COUNT, Selector(category="car",
                                    region=Region(kind="direction", direction="left")))


@pytest.mark.parametrize("scene", [
    Scene("empty", []),
    Scene("no-car", [Object3D("pedestrian", [10.0, 1.0, 0.8], [0.7, 0.7, 1.7])]),
    Scene("with-car", [Object3D("car", [1.0, 10.0, 0.8], [2.0, 4.0, 1.6])]),
])
def test_undefined_direction_is_refused_whatever_the_scene_holds(scene):
    """The bug: with no candidate object the predicate never ran, so an
    invalid program returned "0" with status OK and scored as correct whenever
    the true answer happened to be zero."""
    result = execute(_query_left(), scene, NUSCENES_QA_REGION_CONFIG)
    assert result.status is ExecStatus.UNSUPPORTED
    assert result.answer is None


def test_validate_region_is_independent_of_any_scene():
    with pytest.raises(KeyError):
        validate_region(Region(kind="direction", direction="left"),
                        NUSCENES_QA_REGION_CONFIG)
    validate_region(Region(kind="direction", direction="front_left"),
                    NUSCENES_QA_REGION_CONFIG)
    validate_region(Region(kind="any"), NUSCENES_QA_REGION_CONFIG)


# --------------------------------------------------------------------------- #
# Counterfactuals enumerate the active layout
# --------------------------------------------------------------------------- #

def test_counterfactuals_do_not_sample_undefined_directions():
    """Enumerating the module-global eight raised under the six-sector layout."""
    from spatial_reasoning.consistency import make_counterfactuals
    from spatial_reasoning.datasets import QAItem

    scene = Scene("s0", [
        Object3D("car", [20.0, 1.0, 0.8], [2.0, 4.0, 1.6]),
        Object3D("car", [5.0, 18.0, 0.8], [2.0, 4.0, 1.6]),
    ], scene_token="sc0")
    items = [QAItem("How many cars are to the front of the ego vehicle?",
                    "1", "s0", "sc0")]

    cfs = make_counterfactuals(items, {"s0": scene},
                               config=NUSCENES_QA_REGION_CONFIG, seed=0)
    for cf in cfs:
        assert cf.to_direction in NUSCENES_QA_REGION_CONFIG.directions()


# --------------------------------------------------------------------------- #
# Reported truncation matches the cap actually applied
# --------------------------------------------------------------------------- #

def test_truncation_rate_is_zero_when_nothing_is_capped():
    from spatial_reasoning.datasets import truncation_rate

    scenes = [Scene(f"s{i}", [Object3D("car", [float(j + 1), 0.0, 0.8],
                                       [2.0, 4.0, 1.6]) for j in range(100)])
              for i in range(3)]
    assert truncation_rate(scenes, None) == 0.0
    assert truncation_rate(scenes, 64) == 1.0
