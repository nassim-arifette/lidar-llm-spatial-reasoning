"""Sort the executed system's failures into the four things they can be.

An accuracy of $0.685$ says nothing about what to do next. The $31.5\%$ that
is wrong contains four different problems with four different remedies, and
spending effort on the wrong one is the cost of not separating them:

====================================  =====================================
failure                               what it asks for
====================================  =====================================
output cut off or malformed           a bigger budget, or constrained decoding
program well-formed but says          a better prompt, better examples, then
something other than the question     a more capable model
the question cannot be expressed      a wider target language
program right, answer still wrong     look at the executor and the data
====================================  =====================================

The third and fourth are distinguished without a model: the rule grammar is
deterministic and was checked against the gold answers with no model involved
($91/100$ on development, every disagreement read and categorised). So where
the model's program matches the rule grammar's and the answer is still wrong,
the program is not the problem; where it differs, the difference itself names
what the model got wrong -- the class, the attribute, the direction or the
reference.

That is a triage, not a verdict. A rule-grammar program is a reference, not
ground truth, and the two can agree and both be wrong; those land in the last
bucket, which is the one that gets read by hand.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .dsl import lift
from .question_parser import DEFAULT_PARSER


#: The four buckets, in the order they are decided.
BUCKETS = (
    "malformed_output",          # the model did not emit a usable program
    "refused_at_execution",      # the program ran but could not be resolved
    "program_says_something_else",
    "program_agrees_answer_disagrees",
)


def _reference_program(question: str) -> Optional[str]:
    """What the deterministic grammar makes of the question, or ``None``."""
    parsed = DEFAULT_PARSER.parse(question)
    if not parsed.ok:
        return None
    return lift(parsed.query).describe()


def _difference(model: str, reference: str) -> str:
    """Name the first thing the two programs disagree about.

    Crude on purpose: the point is to count how often each kind of mistake
    happens, not to parse one program into the other.
    """
    model_tokens = set(model.replace("[", " ").replace("]", " ").split())
    reference_tokens = set(reference.replace("[", " ").replace("]", " ").split())
    missing = reference_tokens - model_tokens
    added = model_tokens - reference_tokens

    directions = {"front", "back", "front_left", "front_right",
                  "back_left", "back_right", "left", "right"}
    statuses = {"moving", "parked", "stopped", "standing", "sitting",
                "lying", "rider", "not"}
    operations = {"count", "exists", "compare"}

    if (missing | added) & operations:
        return "wrong_operation"
    if (missing | added) & directions:
        return "wrong_direction"
    if (missing | added) & statuses:
        return "dropped_or_wrong_status"
    if "of" in reference and "of" not in model:
        return "lost_the_reference_object"
    if missing or added:
        return "wrong_category"
    return "same_tokens_different_order"


@dataclass
class Failure:
    question_id: str
    question: str
    family: str
    bucket: str
    detail: str
    reference_answer: str
    executed_answer: Optional[str]
    model_program: str
    reference_program: Optional[str]


def triage(rows: Sequence[Dict[str, object]]) -> List[Failure]:
    out: List[Failure] = []
    for row in rows:
        if row.get("executed_correct"):
            continue
        question = str(row["question"])
        reference = _reference_program(question)
        model_program = str(row.get("program") or "")

        if not row.get("program_valid"):
            bucket, detail = "malformed_output", str(row.get("program_status"))
        elif row.get("executed_answer") is None:
            bucket = "refused_at_execution"
            detail = str(row.get("execution_status"))
        elif reference is None:
            # The grammar cannot express it either, so there is no reference to
            # disagree with. Counted apart rather than blamed on the model.
            bucket, detail = "program_says_something_else", "no_reference_program"
        elif model_program == reference:
            bucket, detail = "program_agrees_answer_disagrees", "see_executor"
        else:
            bucket = "program_says_something_else"
            detail = _difference(model_program, reference)

        out.append(Failure(
            question_id=str(row["question_id"]),
            question=question,
            family=str(row["family"]),
            bucket=bucket,
            detail=detail,
            reference_answer=str(row["reference_answer"]),
            executed_answer=row.get("executed_answer"),  # type: ignore[arg-type]
            model_program=model_program,
            reference_program=reference,
        ))
    return out


def direct_failure_shape(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """How the direct mode is wrong, which is a different question entirely.

    It always answers and its answers are always readable, so there is nothing
    to triage: what matters is whether its counting errors look like a system
    that counted badly or one that did not count.
    """
    gaps: List[int] = []
    low_guesses = collections.Counter()
    for row in rows:
        if row.get("direct_correct"):
            continue
        try:
            gold = int(str(row["reference_answer"]))
            got = int(str(row["direct_answer"]))
        except (TypeError, ValueError):
            continue
        gaps.append(got - gold)
        if gold >= 4:
            low_guesses[got] += 1
    return {
        "n_numeric_failures": len(gaps),
        "under": sum(1 for g in gaps if g < 0),
        "over": sum(1 for g in gaps if g > 0),
        "answers_given_when_the_truth_was_at_least_four":
            dict(low_guesses.most_common(6)),
    }


def summarise(failures: Sequence[Failure]) -> Dict[str, object]:
    by_bucket = collections.Counter(f.bucket for f in failures)
    details = collections.defaultdict(collections.Counter)
    for f in failures:
        details[f.bucket][f.detail] += 1
    return {
        "n_failures": len(failures),
        "buckets": {b: by_bucket.get(b, 0) for b in BUCKETS},
        "details": {b: dict(c.most_common()) for b, c in details.items()},
        "by_family": dict(collections.Counter(f.family for f in failures)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", required=True, help="rows_<split>.jsonl")
    p.add_argument("--out", default=None, help="where to write the triage JSON")
    p.add_argument("--show", type=int, default=6,
                   help="examples to print per bucket")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    with open(args.rows, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]

    failures = triage(rows)
    stats = summarise(failures)
    n = len(rows)

    print(f"{n} questions; the executed system was wrong on {len(failures)} "
          f"({len(failures)/max(n,1):.1%})\n")
    for bucket in BUCKETS:
        count = stats["buckets"][bucket]          # type: ignore[index]
        if not count:
            continue
        print(f"{bucket:<34} {count:>4}  ({count/max(len(failures),1):.0%} of failures)")
        for detail, k in stats["details"][bucket].items():   # type: ignore[index]
            print(f"    {detail:<32} {k}")

    for bucket in BUCKETS:
        examples = [f for f in failures if f.bucket == bucket][:args.show]
        if not examples:
            continue
        print(f"\n--- {bucket} ---")
        for f in examples:
            print(f"  {f.question[:74]}")
            print(f"    model    : {f.model_program or '<none>'}")
            print(f"    reference: {f.reference_program or '<none>'}")
            print(f"    answered {f.executed_answer!r}, gold {f.reference_answer!r}"
                  f"   [{f.detail}]")

    shape = direct_failure_shape(rows)
    print(f"\ndirect mode, for contrast: {shape}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"rows": args.rows, "summary": stats,
                       "direct_failure_shape": shape,
                       "failures": [f.__dict__ for f in failures]},
                      fh, indent=2, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
