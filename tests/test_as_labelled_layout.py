"""The convention the released nuScenes-QA answers follow, as a layout.

``nuscenes_qa`` is the published definition and must not move. The new layout
describes the labels: an unsigned angle, so the two front diagonals are both
"front left", the two rear diagonals are both "back right", and "front right"
and "back left" are empty.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from spatial_reasoning.direct_qa import (
    PROMPT_VERSION, build_messages, build_system_prompt, final_answer_line,
    normalise_reasoned, prompt_version)
from spatial_reasoning.dsl import ANY, Op, Query, Selector, execute
from spatial_reasoning.objects import Object3D, Scene
from spatial_reasoning.qwen_runtime import RuntimeConfig, split_thinking
from spatial_reasoning.regions import (
    NUSCENES_QA_AS_LABELLED_REGION_CONFIG, NUSCENES_QA_REGION_CONFIG,
    SECTOR_LAYOUTS, UNSIGNED_LAYOUTS, Region, direction_of, sector_contains,
    in_direction, relative_direction, sample_position_in_direction)

QA = NUSCENES_QA_REGION_CONFIG
LAB = NUSCENES_QA_AS_LABELLED_REGION_CONFIG
ROOT = Path(__file__).resolve().parents[1]


def at(degrees: float, distance: float = 20.0, category: str = "car",
       origin=(0.0, 0.0)) -> Object3D:
    t = math.radians(degrees)
    return Object3D(category=category,
                    center=[origin[0] + distance * math.cos(t),
                            origin[1] + distance * math.sin(t), 0.8],
                    size=[2.0, 4.0, 1.6])


# -- the rule ----------------------------------------------------------------- #

@pytest.mark.parametrize("degrees, expected", [
    (0, "front"), (29, "front"), (-29, "front"),
    (31, "front_left"), (60, "front_left"), (89, "front_left"),
    (-31, "front_left"), (-60, "front_left"), (-89, "front_left"),
    (91, "back_right"), (120, "back_right"), (149, "back_right"),
    (-91, "back_right"), (-120, "back_right"), (-149, "back_right"),
    (151, "back"), (180, "back"), (-151, "back"), (-179, "back"),
])
def test_relation_depends_on_the_unsigned_angle_only(degrees, expected):
    assert direction_of(at(degrees), LAB) == expected
    assert in_direction(at(degrees), expected, LAB)


def test_front_right_and_back_left_are_always_empty():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        obj = at(float(rng.uniform(-180, 180)), float(rng.uniform(1.5, 60)))
        assert not in_direction(obj, "front_right", LAB)
        assert not in_direction(obj, "back_left", LAB)


def test_mirror_images_always_share_a_relation():
    rng = np.random.default_rng(1)
    for _ in range(500):
        d = float(rng.uniform(0.5, 179.5))
        assert direction_of(at(d), LAB) == direction_of(at(-d), LAB)


def test_every_object_has_exactly_one_relation():
    rng = np.random.default_rng(2)
    for _ in range(500):
        obj = at(float(rng.uniform(-180, 180)))
        assert sum(in_direction(obj, d, LAB) for d in LAB.directions()) == 1


@pytest.mark.parametrize("a, relation", [(30.0, "front"), (90.0, "front_left"),
                                          (150.0, "back_right")])
def test_upper_edges_are_closed_on_both_sides(a, relation):
    """a <= 30 is front, a <= 90 front left, a <= 150 back right."""
    for theta in (a, -a):
        assert sector_contains(relation, theta, LAB), theta
        assert sum(sector_contains(n, theta, LAB) for n in LAB.directions()) == 1


def test_axial_relations_agree_with_the_published_layout_away_from_minus_thirty():
    rng = np.random.default_rng(3)
    for _ in range(500):
        obj = at(float(rng.uniform(-180, 180)))
        for name in ("front", "back"):
            assert in_direction(obj, name, LAB) == in_direction(obj, name, QA)


def test_relative_to_an_anchor_uses_the_same_unsigned_test():
    anchor = at(10, 15, "truck")
    centre = (float(anchor.center[0]), float(anchor.center[1]))
    left, right = at(60, 8, origin=centre), at(-60, 8, origin=centre)
    for obj in (left, right):
        assert relative_direction(obj, anchor, "front_left", LAB)
        assert not relative_direction(obj, anchor, "front_right", LAB)
    assert relative_direction(left, anchor, "front_left", QA)
    assert relative_direction(right, anchor, "front_right", QA)


def test_the_two_layouts_count_differently_on_the_same_program():
    scene = Scene(sample_token="s", objects=[at(60), at(-60), at(-45), at(120), at(0)])
    query = Query(Op.COUNT, Selector(category="car",
                                     region=Region("direction", "front_left")))
    assert execute(query, scene, QA).answer == "1"
    assert execute(query, scene, LAB).answer == "3"
    empty = Query(Op.COUNT, Selector(category="car",
                                     region=Region("direction", "front_right")))
    assert execute(empty, scene, QA).answer == "2"
    assert execute(empty, scene, LAB).answer == "0"


def test_the_published_layout_is_untouched_and_names_are_shared():
    assert SECTOR_LAYOUTS["nuscenes_qa"] == {
        "front": (-30.0, 30.0), "front_left": (30.0, 90.0),
        "back_left": (90.0, 150.0), "back": (150.0, -150.0),
        "back_right": (-150.0, -90.0), "front_right": (-90.0, -30.0)}
    # Same names in the same order: a program written under one layout is
    # valid under the other, and the program prompt is the same text.
    assert LAB.directions() == QA.directions()
    assert "nuscenes_qa" not in UNSIGNED_LAYOUTS
    assert sector_contains("front_right", -45.0, QA)
    assert sector_contains("back_left", 120.0, QA)


def test_sampling_refuses_an_empty_relation():
    for name in ("front_right", "back_left"):
        with pytest.raises(ValueError):
            sample_position_in_direction(name, 20.0, LAB)
    rng = np.random.default_rng(4)
    for name in ("front", "front_left", "back_right", "back"):
        pos = sample_position_in_direction(name, 20.0, LAB, rng)
        assert direction_of(Object3D("car", [pos[0], pos[1], 0.8], [2, 4, 1.6]),
                            LAB) == name


# -- prompts ------------------------------------------------------------------ #

def test_direct_1_is_the_archived_prompt():
    """Archived direct answers were produced under exactly this text."""
    text = build_system_prompt(QA)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == "539bfab3eff904a5c19e13073029d0a94196fd9512684fa77743b369d1c0501d"
    assert PROMPT_VERSION == "direct-1"
    objects = [{"id": "a", "category": "car", "x": 1.0, "y": 2.0, "status": "moving"}]
    user = build_messages("How many cars are there?", objects, QA)[1]["content"]
    assert user == ("Objects (1):\nid, category, x, y, status\n"
                    "a, car, 1.0, 2.0, moving\n\n"
                    "Question:\nHow many cars are there?\n\nAnswer:")


def test_prompt_versions_name_convention_and_style():
    assert prompt_version(QA) == "direct-1"
    assert prompt_version(LAB) == "direct-2"
    assert prompt_version(QA, "reasoned") == "direct-1r"
    assert prompt_version(LAB, "reasoned") == "direct-2r"
    with pytest.raises(ValueError):
        prompt_version(QA, "verbose")


def test_direct_2_states_the_unsigned_convention():
    text = build_system_prompt(LAB)
    assert "UNSIGNED" in text and "EITHER side" in text
    assert "front right and back\nleft never contain any object" in text
    assert "bearing in (-90, -30]" not in text
    # Only the convention paragraph differs between direct-1 and direct-2.
    a, b = build_system_prompt(QA).split("\n\n"), text.split("\n\n")
    assert len(a) == len(b)
    assert [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [2]


def test_reasoned_prompt_allows_working_and_differs_only_in_the_format():
    bare, reasoned = (build_system_prompt(LAB, s).split("\n\n")
                      for s in ("bare", "reasoned"))
    assert bare[:-1] == reasoned[:-1]
    assert "Do not show your working" not in reasoned[-1]
    assert "Answer: <answer>" in reasoned[-1]
    user = build_messages("q?", [], LAB, "reasoned")[1]["content"]
    assert user.endswith("\n\nWorking:")


def test_only_the_last_answer_line_is_read():
    raw = "Cars in front: a, c.\nAnswer: 3\nWait, b too.\n**Answer: 4**"
    assert final_answer_line(raw) == "4"
    assert normalise_reasoned(raw, True) == ("4", "ok")
    assert normalise_reasoned("Answer: Yes.", False) == ("yes", "ok")
    # A bare one-line reply is an answer without its prefix.
    assert normalise_reasoned("yes", False) == ("yes", "ok")
    assert normalise_reasoned("3", True) == ("3", "ok")
    # Working with no final line: unreadable, never dug out of the working.
    assert normalise_reasoned("Cars: a, b, c.\nSo there are 3 cars.", True) == (None, "unparseable")
    assert normalise_reasoned("", True) == (None, "unparseable")


def test_thinking_is_kept_out_of_the_answer():
    assert split_thinking("<think>a 3?\nno, 4</think>\n\n4") == ("4", "a 3?\nno, 4")
    assert split_thinking("<think>ran out of budg") == ("", "ran out of budg")
    assert split_thinking("4") == ("4", "")


def test_runtime_fingerprint_is_unchanged_without_the_thinking_switch():
    assert "enable_thinking" not in RuntimeConfig().fingerprint()
    assert RuntimeConfig(enable_thinking=False).fingerprint()["enable_thinking"] is False
    assert RuntimeConfig().identity() != RuntimeConfig(enable_thinking=True).identity()


# -- the labels themselves, with no geometry ------------------------------------ #

def test_released_answers_never_affirm_front_right_or_back_left():
    """Every single-relation count/existence question in the ten mini drives.

    This needs no executor and no coordinates: if the labels used signed
    sectors, some scene among ten drives would have an object to the front
    right. None does, in 100+ questions.
    """
    import collections
    import re

    qa = json.loads((ROOT / "spatial_reasoning/data/nuscenes_qa_mini.json").read_text())
    answers = {(q["token"], q["question"]): q["answer"] for q in qa}
    for name in ("first_comparison", "wide_comparison"):
        manifest = json.loads(
            (ROOT / f"spatial_reasoning/manifests/{name}.json").read_text())
        for item in manifest["items"]:
            answers[(item["sample_token"], item["question"])] = item["answer"]

    relations = ["front left", "front right", "back left", "back right", "front", "back"]
    seen = collections.defaultdict(collections.Counter)
    for (_, question), answer in answers.items():
        q = question.lower()
        if len(re.findall(r"to the (?:front|back)(?: left| right)? of", q)) != 1:
            continue
        relation = next(r for r in relations if re.search(rf"to the {r} of", q))
        if not (re.search(r"how many|what number", q)
                or re.match(r"^(there is [^;]+; )?(are there any|are any)", q)):
            continue
        seen[relation]["n"] += 1
        seen[relation]["empty"] += str(answer) in ("0", "no")

    for relation in ("front right", "back left"):
        assert seen[relation]["n"] >= 50
        assert seen[relation]["empty"] == seen[relation]["n"], relation
    for relation in ("front left", "back right", "front", "back"):
        assert seen[relation]["empty"] < 0.5 * seen[relation]["n"], relation


# -- the schema a constrained decoder is given ---------------------------------- #

def test_flat_schema_accepts_what_the_validator_accepts_on_every_archived_program():
    """No archived program the validator took is outside the schema.

    The converse does not hold and must not: cross-field rules (a second
    selector for a binary operation) are the validator's, not the schema's.
    """
    jsonschema = pytest.importorskip("jsonschema")
    import re
    from spatial_reasoning.llm_parser import (
        FEW_SHOT, InvalidProgram, json_schema, query_from_json)

    directions = QA.directions()
    validator = jsonschema.Draft202012Validator(json_schema(directions))
    for _, program in FEW_SHOT:
        assert not list(validator.iter_errors(program)), program

    seen = accepted = 0
    for path in (ROOT / "spatial_reasoning/results").glob("*/rows_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)["program_raw"]
            try:
                query_from_json(raw, frozenset(directions))
            except InvalidProgram:
                continue
            seen += 1
            blob = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            accepted += not list(validator.iter_errors(blob))
    assert seen > 1000 and accepted == seen


@pytest.mark.parametrize("program", [
    {"op": "count", "target": {"category": "*", "status": None, "negate_status": False,
                               "direction": None, "anchor": None},
     "second": None, "comparison": None},
    {"op": "count", "target": {"category": "car", "status": None, "negate_status": False,
                               "direction": "left", "anchor": None},
     "second": None, "comparison": None},
    {"op": "count", "target": {"category": "car", "status": "red", "negate_status": False,
                               "direction": None, "anchor": None},
     "second": None, "comparison": None},
    {"op": "tally", "target": {"category": "car", "status": None, "negate_status": False,
                               "direction": None, "anchor": None},
     "second": None, "comparison": None},
])
def test_flat_schema_closes_the_vocabulary(program):
    jsonschema = pytest.importorskip("jsonschema")
    from spatial_reasoning.llm_parser import json_schema

    validator = jsonschema.Draft202012Validator(json_schema(QA.directions()))
    assert list(validator.iter_errors(program))
    assert not list(validator.iter_errors({"op": None}))     # a refusal stays expressible


def test_constrained_decoding_has_its_own_cache_identity():
    from spatial_reasoning.llm_parser import QwenQuestionParser

    free = QwenQuestionParser(directions=QA.directions())
    constrained = QwenQuestionParser(directions=QA.directions(), constrained=True)
    assert free.config_fingerprint()["decoding"] == "greedy"
    assert constrained.config_fingerprint()["decoding"] == "greedy+json_schema"
    assert free.config_id() != constrained.config_id()


# -- what counts as a thing ---------------------------------------------------- #

def _write_tables(root, annotations):
    tables = {
        "sample": [{"token": "s", "scene_token": "sc"}],
        "sample_data": [{"sample_token": "s", "is_key_frame": True,
                         "calibrated_sensor_token": "cs", "ego_pose_token": "p"}],
        "ego_pose": [{"token": "p", "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}],
        "calibrated_sensor": [{"token": "cs", "sensor_token": "se"}],
        "sensor": [{"token": "se", "channel": "LIDAR_TOP"}],
        "attribute": [],
        "category": [{"token": "c1", "name": "vehicle.car"},
                     {"token": "c2", "name": "movable_object.pushable_pullable"}],
        "instance": [{"token": "i1", "category_token": "c1"},
                     {"token": "i2", "category_token": "c2"}],
        "sample_annotation": annotations,
    }
    for name, rows in tables.items():
        (root / f"{name}.json").write_text(json.dumps(rows), encoding="utf-8")


def test_things_all_keeps_unnamed_classes_and_adds_the_ego(tmp_path):
    from spatial_reasoning.nuscenes_adapter import load_nuscenes_scenes

    def ann(token, instance, x, y):
        return {"token": token, "sample_token": "s", "instance_token": instance,
                "translation": [x, y, 0], "size": [2, 4, 1.5], "rotation": [1, 0, 0, 0],
                "num_lidar_pts": 0, "attribute_tokens": []}

    _write_tables(tmp_path, [ann("a", "i1", 10, 0), ann("b", "i2", -10, 0)])
    named, _ = load_nuscenes_scenes(str(tmp_path), min_lidar_points=0)
    everything, _ = load_nuscenes_scenes(str(tmp_path), min_lidar_points=0, things="all")
    assert [o.category for o in named["s"].objects] == ["car"]
    assert [o.category for o in everything["s"].objects] == ["car", "other", "ego"]

    def count(scene, category, region=ANY):
        query = Query(Op.COUNT, Selector(category=category, region=region))
        return execute(query, scene, LAB).answer

    # "How many things are there?" = annotations + 1; a named class is unchanged.
    assert count(named["s"], "object") == "1"
    assert count(everything["s"], "object") == "3"
    assert count(everything["s"], "car") == count(named["s"], "car") == "1"
    # The ego is never to the front or back of itself.
    assert count(everything["s"], "object", Region("direction", "front")) == "1"
    assert count(everything["s"], "object", Region("direction", "back")) == "1"
    with pytest.raises(ValueError):
        load_nuscenes_scenes(str(tmp_path), things="everything")


# -- the drive as the unit of uncertainty ---------------------------------------- #

def test_drive_bootstrap_is_wider_than_a_question_bootstrap_when_drives_differ():
    from spatial_reasoning.drive_bootstrap import across, analyse, interval

    # Ten drives, twenty questions each; half the drives are all right, half
    # all wrong. Questions are perfectly correlated within a drive.
    rows = [{"question_id": f"q{d}_{i}", "scene_token": f"d{d}",
             "direct_correct": d % 2 == 0, "executed_correct": True}
            for d in range(10) for i in range(20)]
    e = interval(rows, [float(r["direct_correct"]) for r in rows], 2000, 0)
    assert e["estimate"] == 0.5 and e["n_drives"] == 10 and e["n_questions"] == 200
    # A question-level interval would be about +-0.07; the drive-level one
    # has to reflect ten units, not two hundred.
    assert e["high"] - e["low"] > 0.4
    result = analyse(rows, 2000, 0)
    diff = result["paired_difference"]["executed_minus_direct"]
    assert diff["estimate"] == 0.5
    assert diff["drives_where_second_is_ahead"] == 5 and diff["drives_tied"] == 5
    assert across(rows, rows, 500, 0)["direct"]["estimate"] == 0.0
    with pytest.raises(ValueError):
        across(rows, rows[:-1], 500, 0)
