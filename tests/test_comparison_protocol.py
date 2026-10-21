"""The rules that make the A/B comparison mean something.

Each test here pins one way the comparison could be quietly rigged. They are
cheap and they run without a GPU, because the expensive run is the one you
cannot afford to discover afterwards was measuring the wrong thing.
"""

from __future__ import annotations

import json

import pytest

from spatial_reasoning.compare_llm_modes import (
    Row,
    count_errors,
    paired,
    scene_dependence,
)
from spatial_reasoning.direct_qa import (
    DirectQwenQA,
    asks_for_a_count,
    build_messages,
    build_system_prompt,
    normalise_answer,
)
from spatial_reasoning.manifest import (
    ManifestItem,
    build_manifest,
    classify,
    describe_objects,
    scene_fingerprint,
)
from spatial_reasoning.objects import Object3D, Scene
from spatial_reasoning.qwen_runtime import RuntimeConfig
from spatial_reasoning.regions import RegionConfig

CONFIG = RegionConfig(layout="nuscenes_qa")


def _obj(category, x, y, status=None, token=None):
    return Object3D(category=category, center=[x, y, 0.8], size=[2.0, 4.0, 1.6],
                    yaw=0.0, score=1.0, status=status, token=token)


def _scene(token="s0", scene_token="scene_a", objects=None):
    return Scene(token, objects or [_obj("car", 10.0, 1.0, "parked", "o1"),
                                    _obj("truck", -8.0, 3.0, "moving", "o2")],
                 scene_token)


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #

def test_families_are_assigned_without_consulting_the_parser():
    """Selecting what the grammar happens to parse would pick the easy cases.

    This phrasing matched no existence pattern until the lexicon was widened,
    and it still has to classify: the family is a statement about the
    question, not about whether an implementation copes with it.
    """
    family, reason, flags = classify("Are any trailers visible?", "exist")
    assert family == "count_exist_category", reason
    assert not flags["has_direction"]


@pytest.mark.parametrize("question, template, reason", [
    ("There is a parked thing; what is it?", "object", "template_out_of_scope:object"),
    ("How many other things are in the same status as the bicycle?", "count",
     "attribute_bound_to_another_object"),
    ("Is the status of the car to the back of the park", "count",
     "truncated_question_text"),
    ("How many cars are to the back of it?", "count", "pronoun_without_antecedent"),
])
def test_out_of_scope_questions_are_recorded_with_a_reason(question, template, reason):
    family, got, _ = classify(question, template)
    assert family is None
    assert got == reason


def test_dev_and_eval_never_share_a_scene():
    """Consecutive samples of a drive show the same vehicles.

    Splitting by sample would make the evaluation set a paraphrase of the
    development set, and the held-out score a memory test.
    """
    scenes = {}
    rows = []
    for s in range(6):
        for k in range(10):
            token = f"sample_{s}_{k}"
            scenes[token] = _scene(token, f"scene_{s}")
            rows.append({"sample_token": token, "question": f"How many of the {k} cars are there?",
                         "answer": "1", "template_type": "count", "num_hop": "0"})
    manifest = build_manifest(rows, scenes, n_dev=10, n_eval=20, seed=1,
                              min_eval_scenes=2)
    dev = {it.scene_token for it in manifest.split("dev")}
    ev = {it.scene_token for it in manifest.split("eval")}
    assert dev and ev and not (dev & ev)


def test_the_payload_an_inference_function_sees_has_no_answer():
    item = ManifestItem("q_1", "How many cars are there?", "s0", "scene_a", "dev",
                        "count_exist_category", "count", 0, "7")
    payload = item.payload()
    assert "7" not in json.dumps(payload)
    assert set(payload) == {"question_id", "question", "sample_token"}


# --------------------------------------------------------------------------- #
# What each mode is given
# --------------------------------------------------------------------------- #

def test_the_object_list_does_not_depend_on_the_question():
    """A list filtered by the question would be half the answer."""
    scene = _scene()
    assert describe_objects(scene) == describe_objects(scene)
    prompt_a = build_messages("How many cars are there?", describe_objects(scene), CONFIG)
    prompt_b = build_messages("Are there any trucks?", describe_objects(scene), CONFIG)
    objects_a = prompt_a[1]["content"].split("Question:")[0]
    objects_b = prompt_b[1]["content"].split("Question:")[0]
    assert objects_a == objects_b


def test_both_modes_are_told_the_same_conventions():
    prompt = build_system_prompt(CONFIG)
    for direction in CONFIG.directions():
        assert direction.replace("_", " ") in prompt
    # The rear sector wraps through 180; printed raw it reads as an empty
    # interval, which would be a false statement of what the executor does.
    assert "(150, -150]" not in prompt
    assert "(150, 210]" in prompt
    # Stated across a line break in the prompt, so check the claim in pieces:
    # an unknown status must not be read as a different one.
    flattened = " ".join(prompt.split())
    assert "An unknown status is not a different status" in flattened


def test_the_direct_cache_depends_on_the_scene_and_not_only_the_question():
    """A program does not depend on the objects; an answer does.

    Keying this cache on the question would serve one scene's answer for
    another's -- destroying the control that shows a system reads the scene.
    """
    direct = DirectQwenQA(region_config=CONFIG)
    question = "How many cars are there?"
    one = _scene("s0", "scene_a")
    two = _scene("s1", "scene_b", [_obj("car", 4.0, 2.0, "moving", "o9")])
    assert scene_fingerprint(one) != scene_fingerprint(two)
    assert direct.cache_key(question, one) != direct.cache_key(question, two)


def test_the_generation_budget_is_part_of_the_cache_identity():
    """An earlier run lost 26% of its programs to a 96-token budget.

    A cache that ignores the budget would serve those truncated generations
    back under a configuration that would not have produced them.
    """
    assert (RuntimeConfig(max_new_tokens=96).identity()
            != RuntimeConfig(max_new_tokens=256).identity())


# --------------------------------------------------------------------------- #
# Reading the answers back
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, expects_count, expected", [
    ("3", True, "3"),
    ("There are 3 cars.", True, "3"),
    ("No, there are none.", False, "no"),
    ("Yes, there is one.", False, "yes"),
    ("truck", False, "truck"),
    ("north", False, "north"),         # must not be read as "no"
    ("nothing", False, "nothing"),
])
def test_the_answer_reader_is_shallow_but_fair(raw, expects_count, expected):
    answer, status = normalise_answer(raw, expects_count)
    assert (answer, status) == (expected, "ok")


def test_an_ambiguous_answer_is_reported_not_guessed():
    """Digging a number out of a paragraph would do part of the task.

    The executed branch emits a bare value by construction, so a generous
    reader applied only to the direct branch would be a thumb on the scale in
    the other direction.
    """
    assert normalise_answer("I count 2 of them, or maybe 5", True) == (None, "unparseable")
    assert normalise_answer("", True) == (None, "unparseable")


def test_the_output_format_is_chosen_from_the_question_not_the_answer():
    """Reading the gold answer's type to pick the format is a label leak.

    This project has already made that mistake once, in the learned baselines.
    """
    assert asks_for_a_count("How many cars are there?")
    assert asks_for_a_count("What number of trailers are to the front?")
    assert not asks_for_a_count("Are there any trucks?")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _row(qid, question, reference, direct=None, executed=None, scene="scene_a"):
    row = Row(qid, "s0", scene, "count_exist_category", "count", question, reference)
    row.direct_answer = direct
    row.direct_correct = direct == reference
    row.executed_answer = executed
    row.executed_correct = executed == reference
    return row


def test_counting_errors_separate_under_from_over():
    rows = [_row("a", "How many cars are there?", "5", "3", "9")]
    direct = count_errors(rows, "direct_answer")
    executed = count_errors(rows, "executed_answer")
    assert (direct["under"], direct["over"]) == (2.0, 0.0)
    assert (executed["under"], executed["over"]) == (0.0, 4.0)


def test_counting_errors_report_how_many_answers_they_average_over():
    """A lower error over far fewer answers is not an improvement."""
    rows = [_row("a", "How many cars are there?", "5", "5", None),
            _row("b", "How many cars are there?", "2", "9", None)]
    executed = count_errors(rows, "executed_answer")
    assert executed["n_asked"] == 2
    assert executed["n_numeric"] == 0
    assert executed["numeric_coverage"] == 0.0


def test_the_paired_table_distinguishes_who_was_right():
    rows = [_row("a", "q1?", "1", "1", "1"),
            _row("b", "q2?", "1", "1", "2"),
            _row("c", "q3?", "1", "2", "1"),
            _row("d", "q4?", "1", "2", "2")]
    assert paired(rows) == {"both": 1, "direct_only": 1, "executed_only": 1,
                            "neither": 1}


def test_the_scene_control_finds_one_question_with_two_answers():
    """The same question of two scenes, with different references.

    A system that gives one answer to both is reciting the template's usual
    answer, whatever its aggregate score says.
    """
    rows = [_row("a", "How many cars are there?", "2", "2", "2", scene="scene_a"),
            _row("b", "How many cars are there?", "5", "2", "5", scene="scene_b")]
    control = scene_dependence(rows)
    assert control["n_question_groups"] == 1
    assert control["direct_gave_one_answer_to_all"] == 1
    assert control["direct_correct_on_every_scene"] == 0
    assert control["executed_gave_one_answer_to_all"] == 0
    assert control["executed_correct_on_every_scene"] == 1
