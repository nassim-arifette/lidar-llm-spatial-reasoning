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
