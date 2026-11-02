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
    for path in (source, source.with_name("rows_eval.jsonl"), root / summary["manifest"]):
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
