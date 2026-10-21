"""Regression tests for the four fixes, written before re-running anything.

Each test pins one thing that was wrong and would otherwise come back:

1. the sector layout must match nuScenes-QA, boundaries included;
2. a direction the layout does not define must be refused, never remapped;
3. no prediction may depend on the gold answer;
4. no object may be silently dropped from a scene.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from spatial_reasoning.datasets import QAItem, encode_batch, encode_scene
from spatial_reasoning.dsl import ExecStatus, Op, Query, Selector, execute
from spatial_reasoning.objects import Object3D, Scene
from spatial_reasoning.regions import (
    NUSCENES_QA_REGION_CONFIG,
    _in_interval,
    Region,
    RegionConfig,
    direction_of,
    in_direction,
    sample_position_in_direction,
    sector_span,
)

QA = NUSCENES_QA_REGION_CONFIG
OCTANT = RegionConfig()


def at_angle(degrees: float, distance: float = 20.0, category: str = "car",
             status: str = None) -> Object3D:
    theta = math.radians(degrees)
    return Object3D(category=category,
                    center=[distance * math.cos(theta), distance * math.sin(theta), 0.8],
                    size=[2.0, 4.0, 1.6], status=status)


# --------------------------------------------------------------------------- #
# 1. The nuScenes-QA layout
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "degrees, expected",
    [
        (0.0, "front"), (29.9, "front"), (-29.9, "front"),
        (30.1, "front_left"), (60.0, "front_left"), (89.9, "front_left"),
        (90.1, "back_left"), (120.0, "back_left"), (149.9, "back_left"),
        (150.1, "back"), (180.0, "back"), (-179.9, "back"), (-150.1, "back"),
        (-149.9, "back_right"), (-120.0, "back_right"), (-90.1, "back_right"),
        (-89.9, "front_right"), (-60.0, "front_right"), (-30.1, "front_right"),
    ],
)
def test_nuscenes_qa_sectors(degrees, expected):
    """arXiv:2305.14836 eq. 2: six relations, 60 degrees each."""
    assert direction_of(at_angle(degrees), QA) == expected


@pytest.mark.parametrize(
    "degrees, boundary_of",
    [(30.0, "front"), (90.0, "front_left"), (150.0, "back_left"),
     (-30.0, "front_right"), (-90.0, "back_right")],
)
def test_sector_intervals_are_half_open(degrees, boundary_of):
    """Each interval is (low, high]: the upper edge belongs to the sector.

    Tested on the interval arithmetic directly rather than through an object.
    Building an object at exactly 150 degrees and reading its bearing back
    goes through ``cos``/``sin``/``atan2`` and returns 149.99999999999997, so
    an object-level test at the boundary measures floating-point round-tripping
    instead of the convention.
    """
    low, high = QA.sectors[boundary_of]
    assert _in_interval(degrees, low, high)
    for other, (olow, ohigh) in QA.sectors.items():
        if other != boundary_of:
            assert not _in_interval(degrees, olow, ohigh), other


def test_objects_near_a_boundary_are_stable_on_the_open_side():
    """Just inside the lower edge belongs to the upper sector, and conversely."""
    assert direction_of(at_angle(30.001), QA) == "front_left"
    assert direction_of(at_angle(29.999), QA) == "front"
    assert direction_of(at_angle(-29.999), QA) == "front"
    assert direction_of(at_angle(-30.001), QA) == "front_right"


def test_layout_has_exactly_six_relations():
    assert len(QA.directions()) == 6
    assert "left" not in QA.directions()
    assert "right" not in QA.directions()


def test_every_nuscenes_sector_spans_sixty_degrees():
    for name in QA.directions():
        assert sector_span(name, QA) == pytest.approx(60.0)


@pytest.mark.parametrize("degrees, octant, nuscenes",
                         [(25.0, "front_left", "front"), (80.0, "left", "front_left")])
def test_the_two_layouts_really_disagree(degrees, octant, nuscenes):
    """The exact mismatch that invalidated the first real-data run."""
    obj = at_angle(degrees)
    assert direction_of(obj, OCTANT) == octant
    assert direction_of(obj, QA) == nuscenes


def test_sectors_tile_the_plane_without_overlap():
    rng = np.random.default_rng(0)
    for config in (OCTANT, QA):
        for _ in range(500):
            obj = at_angle(float(rng.uniform(-180, 180)))
            hits = [d for d in config.directions() if in_direction(obj, d, config)]
            assert len(hits) == 1, (config.layout, obj.bearing, hits)


def test_sampled_positions_land_in_their_sector():
    rng = np.random.default_rng(1)
    for config in (OCTANT, QA):
        for name in config.directions():
            for _ in range(20):
                pos = sample_position_in_direction(name, 25.0, config, rng)
                landed = direction_of(
                    Object3D("car", [pos[0], pos[1], 0.8], [2, 4, 1.6]), config)
                assert landed == name, (config.layout, name, landed)


# --------------------------------------------------------------------------- #
# 2. Undefined directions are refused, not remapped
# --------------------------------------------------------------------------- #

def test_left_is_refused_under_the_nuscenes_layout():
    """A cached program saying "left" must not be answered as front_left."""
    scene = Scene("s", [at_angle(80.0)])
    query = Query(Op.COUNT, Selector(category="car",
                                     region=Region(kind="direction", direction="left")))
    result = execute(query, scene, QA)
    assert result.status is ExecStatus.UNSUPPORTED
    assert result.answer is None
    assert "left" in result.reason


def test_left_still_works_under_the_octant_layout():
    scene = Scene("s", [at_angle(80.0)])
    query = Query(Op.COUNT, Selector(category="car",
                                     region=Region(kind="direction", direction="left")))
    assert execute(query, scene, OCTANT).answer == "1"


# --------------------------------------------------------------------------- #
# 3. No prediction may depend on the gold answer
# --------------------------------------------------------------------------- #

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="requires torch")


def test_count_routing_reads_the_question_not_the_label():
    from spatial_reasoning.baselines import asks_for_a_count

    assert asks_for_a_count("How many cars are there?")
    assert asks_for_a_count("There is a trailer; what number of cones are near it?")
    assert not asks_for_a_count("Are there any trailers?")
    assert not asks_for_a_count("The construction vehicle is in what status?")


@needs_torch
def test_predictions_are_invariant_to_the_gold_answers():
    """Corrupt every label at inference; predictions must not move."""
    from spatial_reasoning.baselines import TrainConfig, build_system
    from spatial_reasoning.datasets import AnswerVocabulary, HashingQuestionEncoder

    scene = Scene("s0", [at_angle(10.0), at_angle(100.0, category="truck")],
                  scene_token="sc0")
    scenes = {"s0": scene}
    items = [
        QAItem("How many cars are there?", "1", "s0", "sc0"),
        QAItem("Are there any trucks?", "yes", "s0", "sc0"),
    ]
    vocab = AnswerVocabulary([it.answer for it in items])
    system = build_system("counter", vocab, HashingQuestionEncoder(dim=32),
                          TrainConfig(epochs=1, device="cpu"))
    system.fit(items, scenes)

    before = system.predict(items, scenes)
    corrupted = [QAItem(it.question, "banana", it.sample_token, it.scene_token)
                 for it in items]
    after = system.predict(corrupted, scenes)
    assert before == after


# --------------------------------------------------------------------------- #
# 4. Nothing is silently truncated
# --------------------------------------------------------------------------- #

def test_encode_scene_keeps_every_object_by_default():
    scene = Scene("big", [at_angle(i * 3.0) for i in range(112)])
    feats, mask = encode_scene(scene)
    assert feats.shape[0] == 112
    assert mask.sum() == 112


def test_encode_batch_pads_to_the_largest_scene():
    from spatial_reasoning.datasets import AnswerVocabulary, HashingQuestionEncoder

    scenes = {
        "small": Scene("small", [at_angle(0.0)]),
        "big": Scene("big", [at_angle(i * 3.0) for i in range(90)]),
    }
    items = [QAItem("How many cars are there?", "1", "small"),
             QAItem("How many cars are there?", "90", "big")]
    batch = encode_batch(items, scenes, AnswerVocabulary(["1", "90"]),
                         HashingQuestionEncoder(dim=32))
    assert batch.objects.shape[1] == 90
    assert batch.mask[0].sum() == 1
    assert batch.mask[1].sum() == 90


def test_explicit_cap_still_truncates_when_asked():
    scene = Scene("big", [at_angle(i * 3.0) for i in range(112)])
    feats, mask = encode_scene(scene, max_objects=64)
    assert feats.shape[0] == 64
    assert mask.sum() == 64


# --------------------------------------------------------------------------- #
# Status: unknown is not the same as different
# --------------------------------------------------------------------------- #

def test_unknown_status_does_not_satisfy_a_negated_status():
    """A cone has no attribute; "not standing" must not silently match it."""
    scene = Scene("s", [
        at_angle(10.0, category="pedestrian", status="standing"),
        at_angle(20.0, category="pedestrian", status="moving"),
        at_angle(40.0, category="traffic_cone", status=None),
    ])
    query = Query(Op.COUNT, Selector(category="object", status="standing",
                                     negate_status=True))
    # Only the moving pedestrian: the cone's status is unknown, not different.
    assert execute(query, scene, QA).answer == "1"
