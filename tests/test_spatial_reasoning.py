"""Unit tests for the spatial-reasoning package.

Run from the repository root:

    python -m pytest tests/test_spatial_reasoning.py -q

Only the tests at the bottom need torch; everything else runs with numpy alone.
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from spatial_reasoning.consistency import (
    PerturbationConfig,
    build_contrast_pairs,
    make_counterfactuals,
    perturb_scene,
)
from spatial_reasoning.datasets import (
    AnswerVocabulary,
    HashingQuestionEncoder,
    QAItem,
    encode_scene,
    split_by_scene,
)
from spatial_reasoning.objects import Object3D, Scene, canonical_category, quaternion_to_yaw
from spatial_reasoning.programs import ExecStatus, SymbolicSpatialQA, execute
from spatial_reasoning.question_parser import (
    DEFAULT_PARSER,
    Operation,
    ParseStatus,
    RuleBasedQuestionParser,
)
from spatial_reasoning.regions import (
    direction_of,
    in_direction,
    sample_position_in_direction,
)
from spatial_reasoning.synthetic import default_detector, generate_corpus


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def _obj(category: str, x: float, y: float, score: float = 1.0) -> Object3D:
    return Object3D(category=category, center=[x, y, 0.8],
                    size=[2.0, 4.0, 1.6], yaw=0.0, score=score)


def test_quaternion_to_yaw_identity_and_quarter_turn():
    assert quaternion_to_yaw([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    # 90 degrees about +z
    q = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    assert quaternion_to_yaw(q) == pytest.approx(math.pi / 2, abs=1e-6)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (10.0, 0.0, "front"),
        (0.0, 10.0, "left"),
        (-10.0, 0.0, "back"),
        (0.0, -10.0, "right"),
        (7.0, 7.0, "front_left"),
        (-7.0, 7.0, "back_left"),
        (7.0, -7.0, "front_right"),
        (-7.0, -7.0, "back_right"),
    ],
)
def test_direction_sectors(x, y, expected):
    obj = _obj("car", x, y)
    assert direction_of(obj) == expected
    assert in_direction(obj, expected)


def test_sectors_are_mutually_exclusive():
    """An object belongs to at most one named sector."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        obj = _obj("car", float(rng.uniform(-40, 40)), float(rng.uniform(-40, 40)))
        hits = [d for d in ("front", "front_left", "left", "back_left",
                            "back", "back_right", "right", "front_right")
                if in_direction(obj, d)]
        assert len(hits) <= 1


def test_objects_too_close_belong_to_no_sector():
    obj = _obj("pedestrian", 0.2, 0.1)
    assert direction_of(obj) is None
    assert not in_direction(obj, "front")


def test_sample_position_lands_in_requested_sector():
    rng = np.random.default_rng(3)
    for direction in ("front", "left", "back_right"):
        for _ in range(20):
            pos = sample_position_in_direction(direction, 20.0, rng=rng)
            assert direction_of(_obj("car", float(pos[0]), float(pos[1]))) == direction


def test_canonical_category_maps_nuscenes_names():
    assert canonical_category("human.pedestrian.adult") == "pedestrian"
    assert canonical_category("vehicle.bus.rigid") == "bus"
    assert canonical_category("movable_object.trafficcone") == "traffic_cone"
    assert canonical_category("flat.driveable_surface") is None


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

def test_parse_simple_count():
    res = DEFAULT_PARSER.parse("How many cars are there?")
    assert res.ok
    assert res.query.operation is Operation.COUNT
    assert res.query.category == "car"
    assert res.query.region.kind == "any"


def test_parse_directional_count_is_ego_centric():
    res = DEFAULT_PARSER.parse("How many pedestrians are to the front left of the ego vehicle?")
    assert res.ok
    assert res.query.category == "pedestrian"
    assert res.query.region.kind == "direction"
    assert res.query.region.direction == "front_left"


def test_longest_direction_phrase_wins():
    res = DEFAULT_PARSER.parse("How many cars are to the back right of the ego vehicle?")
    assert res.ok and res.query.region.direction == "back_right"


def test_parse_existence_and_comparison():
    res = DEFAULT_PARSER.parse("Are there any buses in the scene?")
    assert res.ok and res.query.operation is Operation.EXISTS

    res = DEFAULT_PARSER.parse("Are there more cars than trucks in the scene?")
    assert res.ok
    assert res.query.operation is Operation.COMPARE_COUNT
    assert (res.query.category, res.query.second_category) == ("car", "truck")
    assert res.query.comparison == "more"


def test_object_anchored_question_is_declined_when_disabled():
    """The flag is now an ablation, not the default.

    One-hop questions are most of nuScenes-QA, and declining them was the bulk
    of the coverage the parser was losing. They are answered by default because
    ``dsl.resolve`` requires the anchor to match exactly one object and reports
    the failure otherwise -- the reference is resolved explicitly or not at
    all, never by taking the first match.
    """
    parser = RuleBasedQuestionParser(allow_relative_regions=False)
    res = parser.parse("How many cars are to the left of the truck?")
    assert res.status is ParseStatus.UNSUPPORTED
    assert "anchored" in res.reason


def test_object_anchored_question_is_parsed_by_default():
    res = DEFAULT_PARSER.parse("How many cars are to the left of the truck?")
    assert res.ok
    assert res.query.region.kind == "relative"
    assert res.query.region.reference_category == "truck"


def test_out_of_grammar_question_is_declined_not_guessed():
    res = DEFAULT_PARSER.parse("What colour is the bus?")
    assert not res.ok
    assert res.query is None


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #

def _scene() -> Scene:
    return Scene(
        sample_token="s0",
        scene_token="scene0",
        objects=[
            _obj("car", 20.0, 0.0),       # front
            _obj("car", 25.0, 1.0),       # front
            _obj("car", 0.0, 15.0),       # left
            _obj("truck", -20.0, 0.0),    # back
            _obj("pedestrian", 10.0, 10.0),  # front_left
        ],
    )


def test_count_all_and_count_in_region():
    scene = _scene()
    res = execute(DEFAULT_PARSER.parse("How many cars are there?").query, scene)
    assert res.answer == "3"

    q = DEFAULT_PARSER.parse("How many cars are to the front of the ego vehicle?").query
    assert execute(q, scene).answer == "2"


def test_exists_is_consistent_with_count():
    scene = _scene()
    q_count = DEFAULT_PARSER.parse("How many buses are there?").query
    q_exists = DEFAULT_PARSER.parse("Are there any buses in the scene?").query
    assert execute(q_count, scene).answer == "0"
    assert execute(q_exists, scene).answer == "no"


def test_compare_counts():
    scene = _scene()
    q = DEFAULT_PARSER.parse("Are there more cars than trucks in the scene?").query
    assert execute(q, scene).answer == "yes"
    q = DEFAULT_PARSER.parse("Are there fewer cars than trucks in the scene?").query
    assert execute(q, scene).answer == "no"


def test_group_category_covers_subclasses():
    scene = _scene()
    q = DEFAULT_PARSER.parse("How many vehicles are there?").query
    # three cars plus one truck; the pedestrian is not a vehicle
    assert execute(q, scene).answer == "4"


def test_ambiguous_reference_is_refused_not_guessed():
    parser = RuleBasedQuestionParser(allow_relative_regions=True)
    scene = Scene("s1", [_obj("truck", 10.0, 0.0), _obj("truck", -10.0, 0.0),
                         _obj("car", 10.0, 8.0)])
    q = parser.parse("How many cars are to the left of the truck?").query
    res = execute(q, scene)
    assert res.status is ExecStatus.AMBIGUOUS_REFERENCE
    assert res.answer is None


def test_symbolic_system_abstains_outside_grammar():
    system = SymbolicSpatialQA()
    out = system.answer("What colour is the bus?", _scene())
    assert out.abstained and not out.parsed


def test_symbolic_system_fallback_answers_everything():
    system = SymbolicSpatialQA(fallback_answer="no")
    out = system.answer("What colour is the bus?", _scene())
    assert out.answer == "no" and not out.parsed


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

def test_question_encoder_is_deterministic_across_instances():
    a = HashingQuestionEncoder(dim=64).encode("How many cars are there?")
    b = HashingQuestionEncoder(dim=64).encode("How many cars are there?")
    assert np.allclose(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.0)


def test_split_by_scene_keeps_scenes_disjoint():
    items = [
        QAItem(question="How many cars are there?", answer="1",
               sample_token=f"sample{i}", scene_token=f"scene{i % 5}")
        for i in range(50)
    ]
    splits = split_by_scene(items, {}, seed=1)
    splits.pop("warnings", None)
    scenes = {k: {it.scene_token for it in v} for k, v in splits.items()}
    assert not (scenes["train"] & scenes["test"])
    assert not (scenes["train"] & scenes["val"])
    assert not (scenes["val"] & scenes["test"])


def test_encode_scene_pads_and_masks():
    feats, mask = encode_scene(_scene(), max_objects=8)
    assert feats.shape == (8, feats.shape[1])
    assert mask.sum() == 5
    assert np.all(feats[5:] == 0)


def test_answer_vocabulary_round_trip():
    vocab = AnswerVocabulary(["yes", "no", "3"])
    assert vocab.decode(vocab.encode("3")) == "3"
    assert vocab.encode("not-an-answer") == -1


# --------------------------------------------------------------------------- #
# Consistency probes
# --------------------------------------------------------------------------- #

def test_dropping_every_detection_empties_the_scene():
    out = perturb_scene(_scene(), PerturbationConfig(drop_probability=1.0, seed=0))
    assert len(out.objects) == 0


def test_duplication_only_adds_objects():
    out = perturb_scene(_scene(), PerturbationConfig(duplicate_probability=1.0, seed=0))
    assert len(out.objects) == 2 * len(_scene().objects)


def test_contrast_pairs_have_different_answers():
    items = [
        QAItem("How many cars are there?", "1", "a", "s1"),
        QAItem("How many cars are there?", "2", "b", "s2"),
        QAItem("How many cars are there?", "1", "c", "s3"),
    ]
    pairs = build_contrast_pairs(items)
    assert pairs
    for p in pairs:
        assert p.a.answer != p.b.answer


def test_counterfactual_changes_the_reference_answer():
    scene = _scene()
    items = [QAItem("How many cars are to the front of the ego vehicle?", "2", "s0", "scene0")]
    cfs = make_counterfactuals(items, {"s0": scene}, seed=0)
    assert cfs, "a directional question over a non-empty scene should yield one"
    cf = cfs[0]
    assert cf.new_answer != cf.original_answer


def test_counterfactual_tokens_are_unique_per_question():
    """Several questions share a sample; their counterfactual scenes must not.

    With a shared token the scene dictionary keeps only the last scene and
    every other system is scored against the wrong one.
    """
    scene = _scene()
    items = [
        QAItem("How many cars are to the front of the ego vehicle?", "2", "s0", "scene0"),
        QAItem("How many cars are to the left of the ego vehicle?", "1", "s0", "scene0"),
        QAItem("How many trucks are to the back of the ego vehicle?", "1", "s0", "scene0"),
    ]
    cfs = make_counterfactuals(items, {"s0": scene}, seed=0)
    assert len(cfs) >= 2
    tokens = [cf.scene.sample_token for cf in cfs]
    assert len(set(tokens)) == len(tokens)
    assert len({cf.item.sample_token for cf in cfs}) == len(cfs)


def test_symbolic_system_is_exact_on_its_own_counterfactuals():
    """Sanity check documented in the report: the probe is trivial for it.

    If this drops below 1.0 the probe is broken, not the system.
    """
    from spatial_reasoning.consistency import counterfactual_accuracy

    corpus = generate_corpus(n_scenes=2, samples_per_scene=4,
                             questions_per_sample=8, seed=13)
    cfs = make_counterfactuals(corpus.items, corpus.detected_scenes, seed=0)
    assert cfs
    system = SymbolicSpatialQA(fallback_answer="no")

    def predict(items, scenes):
        return [system.answer(it.question, scenes[it.sample_token]).answer
                for it in items]

    assert counterfactual_accuracy(predict, cfs)["accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Synthetic corpus: the key end-to-end invariant
# --------------------------------------------------------------------------- #

def test_synthetic_questions_are_recovered_exactly_by_the_parser():
    """Every generated question must parse back to the program it came from.

    This is what licenses the comprehension/geometry split in the error
    analysis: if the parser cannot recover the gold program on text it was
    designed for, a parsing failure could masquerade as a geometry failure.
    """
    corpus = generate_corpus(n_scenes=2, samples_per_scene=3,
                             questions_per_sample=8, seed=7)
    assert corpus.items
    for item in corpus.items:
        gold = corpus.gold_queries[f"{item.sample_token}::{item.question}"]
        parsed = DEFAULT_PARSER.parse(item.question)
        assert parsed.ok, f"failed to parse generated question: {item.question}"
        assert parsed.query == gold, (
            f"{item.question}\n  parsed: {parsed.query.describe()}"
            f"\n  gold:   {gold.describe()}"
        )


def test_symbolic_system_is_exact_under_perfect_perception():
    """With ground-truth objects, the program answers every question correctly."""
    corpus = generate_corpus(n_scenes=2, samples_per_scene=3,
                             questions_per_sample=8, seed=11, detector=None)
    system = SymbolicSpatialQA()
    for item in corpus.items:
        out = system.answer(item.question, corpus.oracle_scenes[item.sample_token])
        assert out.answer == item.answer, item.question


def test_detector_noise_actually_degrades_the_symbolic_system():
    """Sanity check on the experiment: imperfect perception must cost accuracy.

    If this ever passes at 100%, the detector model is not doing anything and
    the robustness study would be vacuous.
    """
    corpus = generate_corpus(n_scenes=3, samples_per_scene=4,
                             questions_per_sample=8, seed=5,
                             detector=default_detector(5))
    system = SymbolicSpatialQA()
    hits = sum(
        1 for item in corpus.items
        if system.answer(item.question, corpus.detected_scenes[item.sample_token]).answer
        == item.answer
    )
    accuracy = hits / len(corpus.items)
    assert 0.3 < accuracy < 1.0


# --------------------------------------------------------------------------- #
# Learned baselines (need torch)
# --------------------------------------------------------------------------- #

# A module-level importorskip would skip the whole file, including every test
# above that only needs numpy. Mark just the two tests that need torch.
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="learned baselines require torch",
)


@needs_torch
def test_learned_systems_train_and_predict():
    from spatial_reasoning.baselines import TrainConfig, build_system

    corpus = generate_corpus(n_scenes=3, samples_per_scene=4,
                             questions_per_sample=6, seed=2)
    items = corpus.items
    vocab = AnswerVocabulary([it.answer for it in items])
    encoder = HashingQuestionEncoder(dim=64)
    config = TrainConfig(epochs=2, batch_size=16, device="cpu")

    for kind in ("question_only", "deepsets", "counter"):
        system = build_system(kind, vocab, encoder, config)
        system.fit(items[:40], corpus.detected_scenes)
        preds = system.predict(items[:10], corpus.detected_scenes)
        assert len(preds) == 10
        assert all(isinstance(p, str) for p in preds)


@needs_torch
def test_counter_predicts_non_negative_integers_for_count_questions():
    from spatial_reasoning.baselines import TrainConfig, build_system

    corpus = generate_corpus(n_scenes=2, samples_per_scene=3,
                             questions_per_sample=6, seed=4)
    counts = [it for it in corpus.items if it.answer_type == "count"][:20]
    vocab = AnswerVocabulary([it.answer for it in corpus.items])
    system = build_system("counter", vocab, HashingQuestionEncoder(dim=64),
                          TrainConfig(epochs=1, device="cpu"))
    system.fit(corpus.items[:40], corpus.detected_scenes)
    for pred in system.predict(counts, corpus.detected_scenes):
        assert pred.isdigit()
