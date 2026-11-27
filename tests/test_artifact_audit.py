"""Reject plausible corruptions of a published experiment's evidence."""

import json
from pathlib import Path

import pytest

from spatial_reasoning import audit_results


@pytest.fixture
def copied_experiment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    relative = "llm_modes/summary_eval.json"
    source = root / "spatial_reasoning/results" / relative
    summary = json.loads(source.read_text(encoding="utf-8"))
    rescored = root / "spatial_reasoning/results/rescored_as_labelled" / relative
    for path in (source, source.with_name("rows_eval.jsonl"), root / summary["manifest"],
                 rescored, rescored.with_name("rows_eval.jsonl")):
        target = tmp_path / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    monkeypatch.setattr(audit_results, "PUBLISHED", (relative,))
    return tmp_path, tmp_path / "spatial_reasoning/results/llm_modes/rows_eval.jsonl"


def test_partial_result_cannot_pass_as_a_complete_run(copied_experiment):
    root, rows = copied_experiment
    lines = rows.read_text(encoding="utf-8").splitlines()
    rows.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        audit_results.audit(root)


def test_correctness_flag_is_checked_against_answer(copied_experiment):
    root, rows = copied_experiment
    lines = rows.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["direct_correct"] = not first["direct_correct"]
    lines[0] = json.dumps(first)
    rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="direct_correct"):
        audit_results.audit(root)


def test_changed_summary_is_not_trusted(copied_experiment):
    root, rows = copied_experiment
    path = rows.with_name("summary_eval.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["summary"]["direct"]["accuracy"] = 1.0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="direct.accuracy"):
        audit_results.audit(root)


# -- rescored results are held to their source --------------------------------- #

def _rescored(root):
    return root / "spatial_reasoning/results/rescored_as_labelled/llm_modes"


def test_rescored_result_passes_untouched(copied_experiment):
    root, _ = copied_experiment
    result = audit_results.audit(root)
    assert result["rescored"][0]["layout"] == "nuscenes_qa_as_labelled"
    assert result["rescored"][0]["n"] == 200


def test_rescoring_may_not_change_what_the_model_wrote(copied_experiment):
    root, _ = copied_experiment
    rows = _rescored(root) / "rows_eval.jsonl"
    lines = rows.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["program_raw"] = first["program_raw"] + " "
    lines[0] = json.dumps(first)
    rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="program_raw"):
        audit_results.audit(root)


def test_rescored_result_is_tied_to_the_archive_it_came_from(copied_experiment):
    root, source_rows = copied_experiment
    source_rows.write_text(source_rows.read_text(encoding="utf-8") + "\n",
                           encoding="utf-8")
    with pytest.raises(ValueError, match="rescored_from.sha256"):
        audit_results.audit(root)


def test_rescored_correctness_is_recomputed(copied_experiment):
    root, _ = copied_experiment
    rows = _rescored(root) / "rows_eval.jsonl"
    lines = rows.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["executed_correct"] = not first["executed_correct"]
    lines[0] = json.dumps(first)
    rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="executed_correct"):
        audit_results.audit(root)
