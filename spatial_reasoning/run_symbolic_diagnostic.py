"""Two controlled ablations on the symbolic path, with nothing retrained.

``run_experiments`` rebuilds and trains the three learned systems on every
invocation. Using it to ask a question about the parser or the object filter
would burn three trainings and, worse, make the learned rows incomparable with
the ones already measured. This script touches neither.

It answers two questions, and keeps their effects separable:

**A -- what did adding refusals to the grammar cost and buy?**
Identical questions, identical objects, identical geometry. Only the parser
changes: ``strict_refusals=False`` reproduces the behaviour that parsed
anyway and dropped the part it did not understand, against the current one
that refuses.

**B -- does the LiDAR-return filter explain the counting error?**
Identical questions, identical *programs*, identical executor. Only the object
list changes: ``min_lidar_points=1`` against ``0``.

What it reports, for every condition:

* accuracy over all questions, with an abstention counted as an error;
* the answered fraction, and accuracy over answered questions only;
* the counting error split into under- and over-counting (MAE = under + over);
* the same figures restricted to the questions answered numerically in *every*
  condition, because adding objects can make a reference ambiguous and change
  which questions are answerable at all.

A drop in MAE after adding refusals does not mean the system counts better: it
may only mean it stopped answering the cases it got badly wrong. The coverage
column is what tells those apart, so it is never reported without it.

Per-question records -- the program, the answer, the selected object tokens,
the refusal reason -- are written to JSONL so a disagreement can be inspected
rather than guessed at.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .datasets import QAItem
from .dsl import ExecStatus, Query, execute, lift
from .evaluate import count_error_decomposition
from .nuscenes_adapter import load_nuscenes_qa, load_nuscenes_scenes
from .objects import Scene
from .question_parser import RuleBasedQuestionParser
from .regions import SECTOR_LAYOUTS, RegionConfig


# --------------------------------------------------------------------------- #
# One condition
# --------------------------------------------------------------------------- #

def evaluate_condition(
    name: str,
    items: Sequence[QAItem],
    scenes: Dict[str, Scene],
    parser: RuleBasedQuestionParser,
    config: RegionConfig,
    records_path: Optional[str] = None,
) -> Dict:
    """Run one (parser, object-list) combination and archive every answer."""
    start = time.perf_counter()
    predictions: List[Optional[str]] = []
    outcomes = Counter()
    refusal_reasons = Counter()
    exec_failures = Counter()
    records: List[Dict] = []

    for item in items:
        scene = scenes.get(item.sample_token)
        rec: Dict[str, object] = {
            "condition": name,
            "sample_token": item.sample_token,
            "question": item.question,
            "gold": item.answer,
            "template": item.template,
        }

        if scene is None:
            predictions.append(None)
            outcomes["no_scene"] += 1
            rec["status"] = "no_scene"
            records.append(rec)
            continue

        parsed = parser.parse(item.question)
        if not parsed.ok:
            predictions.append(None)
            outcomes["declined_by_parser"] += 1
            refusal_reasons[parsed.reason[:70]] += 1
            rec.update(status="declined_by_parser", reason=parsed.reason)
            records.append(rec)
            continue

        query: Query = lift(parsed.query)
        rec["program"] = query.describe()
        result = execute(query, scene, config)

        if not result.ok:
            predictions.append(None)
            outcomes["declined_by_executor"] += 1
            exec_failures[result.status.value] += 1
            rec.update(status=f"declined_by_executor:{result.status.value}",
                       reason=result.reason)
            records.append(rec)
            continue

        predictions.append(result.answer)
        correct = str(result.answer).strip().lower() == item.answer
        outcomes["correct" if correct else "wrong"] += 1
        rec.update(status="answered", answer=result.answer, correct=correct,
                   selected=[o.token for o in result.selected if o.token][:40],
                   n_selected=len(result.selected))
        records.append(rec)

    if records_path:
        os.makedirs(os.path.dirname(os.path.abspath(records_path)) or ".",
                    exist_ok=True)
        with open(records_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    n = max(len(items), 1)
    answered = sum(1 for p in predictions if p is not None)
    correct = outcomes["correct"]

    return {
        "name": name,
        "n": len(items),
        "accuracy": correct / n,
        "answered_fraction": answered / n,
        "accuracy_when_answered": (correct / answered) if answered else 0.0,
        "outcomes": dict(outcomes),
        "refusal_reasons": dict(refusal_reasons.most_common(8)),
        "executor_failures": dict(exec_failures),
        "count_errors": count_error_decomposition(predictions, items),
        "seconds": round(time.perf_counter() - start, 1),
        "_predictions": predictions,
    }


def restrict_to_common_numeric(
    conditions: Sequence[Dict], items: Sequence[QAItem]
) -> Dict[str, Dict]:
    """Recompute the counting figures on questions numeric in EVERY condition.

    Conditions answer different subsets, so their MAEs are means over different
    populations. Comparing them directly would credit a condition for declining
    the items it would have got wrong. This intersection is the paired figure.
    """
    def numeric(p) -> bool:
        if p is None:
            return False
        try:
            int(str(p).strip())
            return True
        except ValueError:
            return False

    keep = [
        i for i, it in enumerate(items)
        if it.answer_type == "count"
        and all(numeric(c["_predictions"][i]) for c in conditions)
    ]
    paired_items = [items[i] for i in keep]

    out = {}
    for c in conditions:
        preds = [c["_predictions"][i] for i in keep]
        hits = sum(1 for p, it in zip(preds, paired_items)
                   if str(p).strip().lower() == it.answer)
        out[c["name"]] = {
            "n_paired": len(keep),
            "accuracy": hits / len(keep) if keep else 0.0,
            "count_errors": count_error_decomposition(preds, paired_items),
        }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-root", required=True)
    p.add_argument("--qa", nargs="+", required=True)
    p.add_argument("--layout", default="nuscenes_qa", choices=sorted(SECTOR_LAYOUTS))
    p.add_argument("--out", default="outputs/symbolic_diagnostic")
    p.add_argument("--allow-relative", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    config = RegionConfig(layout=args.layout)

    # Two object lists; everything else is shared.
    object_lists = {}
    conversions = {}
    for filt in (1, 0):
        scenes, report = load_nuscenes_scenes(args.metadata_root, min_lidar_points=filt)
        object_lists[filt] = scenes
        conversions[filt] = report.as_dict()
        print(f"min_lidar_points={filt}: {report.annotations_kept} objects over "
              f"{report.samples} samples")

    # Questions are loaded once, against the stricter list, so the item set is
    # identical across conditions. Loading them per-condition would let the
    # object filter change which questions exist.
    items: List[QAItem] = []
    for qa_path in args.qa:
        items.extend(load_nuscenes_qa(qa_path, object_lists[1]))
    if args.limit:
        items = items[:args.limit]
    print(f"questions: {len(items)} over {len({i.sample_token for i in items})} samples")
    print(f"layout: {args.layout} -> {config.directions()}\n")

    conditions = []
    for strict in (True, False):
        for filt in (1, 0):
            name = f"{'strict' if strict else 'legacy'}_parser__min_lidar_{filt}"
            parser = RuleBasedQuestionParser(
                allow_relative_regions=args.allow_relative, strict_refusals=strict)
            res = evaluate_condition(
                name, items, object_lists[filt], parser, config,
                records_path=os.path.join(args.out, f"records_{name}.jsonl"))
            conditions.append(res)
            ce = res["count_errors"]
            print(f"{name:<34} acc={res['accuracy']:.3f} "
                  f"answered={res['answered_fraction']:.3f} "
                  f"acc|ans={res['accuracy_when_answered']:.3f} "
                  f"U={ce['under']!s:.5} O={ce['over']!s:.5} n_num={ce['n']}")

    by_name = {c["name"]: c for c in conditions}

    # Pair within each ablation, not across both. Question B asks what the
    # object filter does at a FIXED parser; intersecting with the other
    # parser's answered set would shrink the population for a reason unrelated
    # to the filter -- on the smoke run that left three questions.
    paired: Dict[str, Dict] = {}
    for strict in ("strict", "legacy"):
        pair = [by_name[f"{strict}_parser__min_lidar_{f}"] for f in (1, 0)]
        paired[f"B_filter_at_{strict}_parser"] = restrict_to_common_numeric(pair, items)
    for filt in (1, 0):
        pair = [by_name[f"{s}_parser__min_lidar_{filt}"] for s in ("strict", "legacy")]
        paired[f"A_parser_at_min_lidar_{filt}"] = restrict_to_common_numeric(pair, items)

    for ablation, group in paired.items():
        n_paired = next(iter(group.values()))["n_paired"]
        print(f"\n{ablation} (paired n={n_paired}):")
        for name, v in group.items():
            ce = v["count_errors"]
            print(f"  {name:<34} acc={v['accuracy']:.3f} "
                  f"U={ce['under']!s:.5} O={ce['over']!s:.5} MAE={ce['mae']!s:.5}")

    payload = {
        "layout": args.layout,
        "conversions": conversions,
        "n_items": len(items),
        "conditions": [{k: v for k, v in c.items() if k != "_predictions"}
                       for c in conditions],
        "paired_numeric": paired,
    }
    path = os.path.join(args.out, "diagnostic.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
