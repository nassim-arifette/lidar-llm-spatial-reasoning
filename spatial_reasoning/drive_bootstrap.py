"""Uncertainty with the drive as the unit, not the question.

Questions from one drive share its vehicles, its road and its annotator, so
they are not independent draws. A sign test over questions treats them as if
they were and is optimistic. Here the resampling unit is the drive: draw
drives with replacement, keep every question of each drawn drive, recompute.

Reported for one file: accuracy of each mode, and the paired difference
between two modes on the same questions. For two files that hold the same
questions (the same model under two prompts, say): the paired difference of
one mode across them.

With three drives this interval is nearly meaningless, which is the honest
answer for the first comparison; it becomes informative on ``val_drives``.

    python -m spatial_reasoning.drive_bootstrap rows_eval.jsonl
    python -m spatial_reasoning.drive_bootstrap new/rows_eval.jsonl --against old/rows_eval.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODES = ("direct", "executed", "functional")


def load(path: str) -> List[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _per_drive(rows: Sequence[dict], values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Sum of ``values`` and number of questions, per drive."""
    total: Dict[str, float] = collections.defaultdict(float)
    count: Dict[str, int] = collections.Counter()
    for row, value in zip(rows, values):
        total[row["scene_token"]] += value
        count[row["scene_token"]] += 1
    drives = sorted(count)
    return (np.array([total[d] for d in drives], dtype=float),
            np.array([count[d] for d in drives], dtype=float))


def interval(rows: Sequence[dict], values: Sequence[float], resamples: int = 10000,
             seed: int = 0, level: float = 0.95) -> Dict[str, float]:
    """Mean of ``values`` over questions, with a drive-level bootstrap interval.

    A ratio estimator: drives contribute different numbers of questions, so
    each replicate divides resampled sums by resampled counts.
    """
    sums, counts = _per_drive(rows, values)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(resamples, len(sums)))
    replicates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(replicates, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": float(sums.sum() / counts.sum()),
            "low": float(low), "high": float(high),
            "n_questions": int(counts.sum()), "n_drives": int(len(sums))}


def modes_in(rows: Sequence[dict]) -> List[str]:
    return [m for m in MODES if m != "functional"
            or any(r.get("functional_status") for r in rows)]


def analyse(rows: Sequence[dict], resamples: int = 10000, seed: int = 0) -> Dict[str, object]:
    modes = modes_in(rows)
    out: Dict[str, object] = {"accuracy": {
        m: interval(rows, [float(bool(r[f"{m}_correct"])) for r in rows], resamples, seed)
        for m in modes}}
    differences = {}
    for i, a in enumerate(modes):
        for b in modes[i + 1:]:
            values = [float(bool(r[f"{b}_correct"])) - float(bool(r[f"{a}_correct"]))
                      for r in rows]
            entry = interval(rows, values, resamples, seed)
            sums, _ = _per_drive(rows, values)
            entry["drives_where_second_is_ahead"] = int((sums > 0).sum())
            entry["drives_where_first_is_ahead"] = int((sums < 0).sum())
            entry["drives_tied"] = int((sums == 0).sum())
            differences[f"{b}_minus_{a}"] = entry
    out["paired_difference"] = differences
    return out


def across(rows: Sequence[dict], other: Sequence[dict], resamples: int = 10000,
           seed: int = 0) -> Dict[str, object]:
    """Same questions in two files: ``rows`` minus ``other``, mode by mode."""
    index = {r["question_id"]: r for r in other}
    if set(index) != {r["question_id"] for r in rows}:
        raise ValueError("the two files do not hold the same questions")
    out = {}
    for mode in modes_in(rows):
        if mode not in modes_in(other):
            continue
        values = [float(bool(r[f"{mode}_correct"]))
                  - float(bool(index[r["question_id"]][f"{mode}_correct"])) for r in rows]
        out[mode] = interval(rows, values, resamples, seed)
    return out


def _show(name: str, e: Dict[str, float]) -> str:
    return f"  {name:<28} {e['estimate']:+.3f}  [{e['low']:+.3f}, {e['high']:+.3f}]"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rows")
    p.add_argument("--against", default=None,
                   help="a second rows file with the same questions")
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rows = load(args.rows)
    result = analyse(rows, args.resamples, args.seed)
    first = next(iter(result["accuracy"].values()))          # type: ignore[union-attr]
    print(f"{first['n_questions']} questions over {first['n_drives']} drives; "
          f"95% intervals from {args.resamples} drive-level resamples")
    for mode, e in result["accuracy"].items():               # type: ignore[union-attr]
        print(_show(mode, e))
    for name, e in result["paired_difference"].items():      # type: ignore[union-attr]
        print(_show(name, e) + f"   drives ahead/behind/tied: "
              f"{e['drives_where_second_is_ahead']}/{e['drives_where_first_is_ahead']}"
              f"/{e['drives_tied']}")
    if args.against:
        result["minus_other_file"] = across(rows, load(args.against),
                                            args.resamples, args.seed)
        print(f"this file minus {args.against}:")
        for mode, e in result["minus_other_file"].items():   # type: ignore[union-attr]
            print(_show(mode, e))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
