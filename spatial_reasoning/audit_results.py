"""Verify the published comparisons from their per-question records, offline.

This checks saved evidence; it does not rerun model inference or certify model
identity. Missing or incomplete experiments must never become result rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .compare_llm_modes import Row, scene_dependence, summarise


PUBLISHED = (
    "llm_modes/summary_dev.json", "llm_modes/summary_eval.json",
    "llm_modes_7b/summary_eval.json", "llm_modes_32b/summary_eval.json",
    "wide_1p5b/summary_eval.json", "gw_qwen3_32b/summary_eval.json",
    "gw_qwen3_next_80b/summary_eval.json", "gw_qwen3p8_27b/summary_eval.json",
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(expected, actual, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{path}: expected an object")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"{path}.{key}: missing")
            _same(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"{path}: list length differs")
        for index, (a, b) in enumerate(zip(expected, actual)):
            _same(a, b, f"{path}[{index}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
                expected, actual, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{path}: saved {expected!r}, recomputed {actual!r}")
    elif expected != actual:
        raise ValueError(f"{path}: saved {expected!r}, recomputed {actual!r}")


def audit(root: Path) -> dict:
    results = root / "spatial_reasoning" / "results"
    checked = []
    for relative in PUBLISHED:
        summary_path = results / relative
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        split = saved["split"]
        rows_path = summary_path.with_name(f"rows_{split}.jsonl")
        raw_rows = [json.loads(line) for line in
                    rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest_path = root / saved["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = {item["question_id"]: item for item in manifest["items"]
                 if item["split"] == split}
        ids = [row["question_id"] for row in raw_rows]
        if len(ids) != len(set(ids)) or set(ids) != set(items):
            raise ValueError(f"{relative}: duplicate, missing, or unexpected question IDs")
        dev_scenes = {i["scene_token"] for i in manifest["items"] if i["split"] == "dev"}
        eval_scenes = {i["scene_token"] for i in manifest["items"] if i["split"] == "eval"}
        if dev_scenes & eval_scenes:
            raise ValueError(f"{relative}: development/evaluation scene overlap")
        for row in raw_rows:
            item = items[row["question_id"]]
            for key in ("question", "sample_token", "scene_token", "family", "template"):
                _same(item[key], row[key], f"{relative}:{row['question_id']}.{key}")
            _same(item["answer"], row["reference_answer"], f"{relative}:reference_answer")
            for mode in ("direct", "executed", "functional"):
                if mode == "functional" and not row.get("functional_status"):
                    continue
                answer = row.get(f"{mode}_answer")
                correct = (answer is not None and
                           str(answer).strip().lower() == str(item["answer"]).strip().lower())
                _same(correct, row[f"{mode}_correct"],
                      f"{relative}:{row['question_id']}.{mode}_correct")
        rows = [Row(**row) for row in raw_rows]
        recomputed = summarise(rows)
        recomputed["scene_dependence"] = scene_dependence(rows)
        _same(saved["summary"], recomputed, f"{relative}.summary")
        checked.append({
            "summary": str(summary_path.relative_to(root)).replace("\\", "/"),
            "sha256": {"summary": _hash(summary_path), "rows": _hash(rows_path),
                       "manifest": _hash(manifest_path)},
            "n": len(rows), "n_scenes": len({r.scene_token for r in rows}),
            "runtime": saved["runtime"],
            "accuracy": {mode: recomputed[mode]["accuracy"] for mode in
                         ("direct", "executed", "functional") if mode in recomputed},
            "by_scene": recomputed["by_scene"],
            "all_rows_have_model_revision": all(r.model_revision for r in rows),
        })
    return {
        "schema_version": 1,
        "status": "verified",
        "checks": ["complete manifest membership", "no duplicate question IDs",
                   "no scene overlap across splits", "question/reference provenance",
                   "correctness recomputed from saved answers", "all saved summary fields",
                   "SHA-256 of rows, summary and manifest"],
        "limits": ["No model inference is rerun.",
                   "Hashes attest these files, not historical execution or provider identity.",
                   "Question-level p-values assume independence; only three evaluation scenes exist."],
        "experiments": checked,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    result = audit(Path(__file__).resolve().parents[1])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(result['experiments'])} complete comparisons "
          f"({sum(e['n'] for e in result['experiments'])} saved rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
