"""Metrics, cost measurement, and the error decomposition.

Three things are measured for every system, and the report presents them
together because any one of them alone is misleading:

* **accuracy and counting error** -- is the answer right, and by how much;
* **cost** -- wall-clock time and peak memory, measured the same way for all
  systems, with the one-off cost of perception reported separately;
* **where the errors come from** -- perception, comprehension, or geometry.

The decomposition is the part that a single accuracy table cannot give. It is
only possible because the symbolic system exposes its intermediate query and
its selected objects.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .datasets import QAItem
from .objects import Scene
from .programs import SymbolicSpatialQA
from .question_parser import Query


# --------------------------------------------------------------------------- #
# Basic metrics
# --------------------------------------------------------------------------- #

def accuracy(predictions: Sequence[Optional[str]], items: Sequence[QAItem]) -> float:
    """Exact-match accuracy. An abstention (``None``) counts as an error."""
    if not items:
        return 0.0
    hits = sum(1 for p, it in zip(predictions, items)
               if p is not None and str(p).strip().lower() == it.answer)
    return hits / len(items)


def accuracy_on_answered(
    predictions: Sequence[Optional[str]], items: Sequence[QAItem]
) -> Tuple[float, float]:
    """``(accuracy over answered items, answered fraction)``.

    Reported next to :func:`accuracy` so that a system which abstains often but
    is right when it answers cannot be confused with one that is right overall.
    """
    answered = [(p, it) for p, it in zip(predictions, items) if p is not None]
    if not answered:
        return 0.0, 0.0
    hits = sum(1 for p, it in answered if str(p).strip().lower() == it.answer)
    return hits / len(answered), len(answered) / len(items)


def count_mae(
    predictions: Sequence[Optional[str]], items: Sequence[QAItem]
) -> Optional[float]:
    """Mean absolute error over counting questions the system answered numerically.

    Accuracy treats "4 instead of 3" and "11 instead of 3" as the same failure;
    this does not. Returns ``None`` when nothing qualifies.

    Non-numeric predictions ("yes", an empty string) are **excluded**, not
    scored. They used to be charged an error of ``target``, i.e. treated as if
    the system had answered zero -- a number the system never produced, which
    made the metric depend on the answer distribution rather than on the
    system. They are counted by :func:`count_answer_coverage` instead, so the
    exclusion stays visible.
    """
    errors: List[float] = []
    for p, it in zip(predictions, items):
        if it.answer_type != "count":
            continue
        try:
            target = int(it.answer)
        except ValueError:
            continue
        if p is None:
            continue
        try:
            errors.append(abs(int(str(p).strip()) - target))
        except ValueError:
            continue
    return float(np.mean(errors)) if errors else None


def count_answer_coverage(
    predictions: Sequence[Optional[str]], items: Sequence[QAItem]
) -> Dict[str, float]:
    """How many counting questions the MAE above is actually computed over.

    A system that answers "yes" to half the counting questions can post an
    excellent MAE on the other half. Reporting the denominator alongside the
    mean is what stops that reading as competence.
    """
    counting = [(p, it) for p, it in zip(predictions, items)
                if it.answer_type == "count"]
    if not counting:
        return {"n_count_questions": 0.0, "numeric_fraction": 0.0,
                "abstained_fraction": 0.0}

    numeric = abstained = 0
    for p, _ in counting:
        if p is None:
            abstained += 1
            continue
        try:
            int(str(p).strip())
            numeric += 1
        except ValueError:
            pass
    n = len(counting)
    return {
        "n_count_questions": float(n),
        "numeric_fraction": numeric / n,
        "abstained_fraction": abstained / n,
    }


def breakdown(
    predictions: Sequence[Optional[str]],
    items: Sequence[QAItem],
    key: Callable[[QAItem], str],
) -> Dict[str, Dict[str, float]]:
    """Accuracy grouped by an arbitrary key (answer type, template, scene)."""
    groups: Dict[str, List[Tuple[Optional[str], QAItem]]] = {}
    for p, it in zip(predictions, items):
        groups.setdefault(key(it), []).append((p, it))
    return {
        name: {
            "n": float(len(pairs)),
            "accuracy": accuracy([p for p, _ in pairs], [i for _, i in pairs]),
        }
        for name, pairs in sorted(groups.items())
    }


def majority_baseline(train_items: Sequence[QAItem], test_items: Sequence[QAItem]) -> float:
    """Accuracy of always answering the most frequent training answer.

    The floor below which no result is meaningful.
    """
    counts: Dict[str, int] = {}
    for it in train_items:
        counts[it.answer] = counts.get(it.answer, 0) + 1
    if not counts or not test_items:
        return 0.0
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return sum(1 for it in test_items if it.answer == best) / len(test_items)


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #

@dataclass
class CostMeasurement:
    seconds: float
    peak_memory_mb: float
    items: int

    @property
    def ms_per_item(self) -> float:
        return 1000.0 * self.seconds / max(self.items, 1)

    def as_dict(self) -> Dict[str, float]:
        return {
            "seconds": round(self.seconds, 4),
            "ms_per_item": round(self.ms_per_item, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 3),
            "items": float(self.items),
        }


def measure_cost(fn: Callable[[], object], items: int) -> Tuple[object, CostMeasurement]:
    """Run ``fn`` once, returning its value plus time and peak Python memory.

    ``tracemalloc`` only sees Python allocations, so for the torch models this
    under-reports tensor memory. That limitation is stated in the report rather
    than papered over: the numbers are comparable between the two torch models,
    and the symbolic system's advantage is large enough not to hinge on it.
    """
    tracemalloc.start()
    start = time.perf_counter()
    try:
        value = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, CostMeasurement(elapsed, peak / (1024.0 * 1024.0), items)


# --------------------------------------------------------------------------- #
# Error decomposition
# --------------------------------------------------------------------------- #

ERROR_KINDS = (
    "correct",
    "declined",                # outside the grammar, no answer produced
    "perception",              # right program, wrong object list
    "comprehension",           # wrong program parsed from the question
    "geometry",                # right program and right objects, wrong region maths
    "unattributed",            # wrong, and we cannot tell which stage caused it
)


@dataclass
class ErrorAnalysis:
    counts: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in ERROR_KINDS})
    examples: Dict[str, List[str]] = field(default_factory=lambda: {k: [] for k in ERROR_KINDS})

    def add(self, kind: str, description: str, keep: int = 5) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        bucket = self.examples.setdefault(kind, [])
        if len(bucket) < keep:
            bucket.append(description)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def fractions(self) -> Dict[str, float]:
        t = max(self.total, 1)
        return {k: v / t for k, v in self.counts.items()}


def decompose_errors(
    system: SymbolicSpatialQA,
    items: Sequence[QAItem],
    detected_scenes: Dict[str, Scene],
    oracle_scenes: Optional[Dict[str, Scene]] = None,
    gold_queries: Optional[Dict[str, Query]] = None,
) -> ErrorAnalysis:
    """Attribute each symbolic-system error to a stage.

    The attribution is only as strong as the evidence available:

    * with ``oracle_scenes`` (ground-truth boxes) we can separate perception
      errors from reasoning errors, by re-running the same parsed program on
      the ground-truth object list;
    * with ``gold_queries`` (available for synthetic data, where the question
      was generated from a known program) we can further separate a
      misunderstood question from a correct question evaluated with the wrong
      geometry.

    Without either, errors land in ``"unattributed"`` -- which is the honest
    label, not a failure of the method.
    """
    analysis = ErrorAnalysis()

    for item in items:
        scene = detected_scenes.get(item.sample_token) or Scene(item.sample_token, [])
        result = system.answer(item.question, scene)

        if result.abstained:
            analysis.add("declined", f"{item.question} :: {result.failure}")
            continue

        if str(result.answer).strip().lower() == item.answer:
            analysis.add("correct", item.question)
            continue

        gold_query = (gold_queries or {}).get(_item_key(item))
        if gold_query is not None and result.query is not None and result.query != gold_query:
            analysis.add(
                "comprehension",
                f"{item.question} :: parsed {result.query.describe()} "
                f"but expected {gold_query.describe()}",
            )
            continue

        if oracle_scenes is not None and item.sample_token in oracle_scenes:
            oracle = system.answer(item.question, oracle_scenes[item.sample_token])
            if (not oracle.abstained
                    and str(oracle.answer).strip().lower() == item.answer):
                analysis.add(
                    "perception",
                    f"{item.question} :: {result.answer} on detections, "
                    f"{oracle.answer} on ground-truth boxes (expected {item.answer})",
                )
                continue
            if gold_query is not None:
                analysis.add(
                    "geometry",
                    f"{item.question} :: correct program, wrong result "
                    f"({oracle.answer} vs {item.answer})",
                )
                continue

        analysis.add("unattributed",
                     f"{item.question} :: predicted {result.answer}, expected {item.answer}")

    return analysis


def _item_key(item: QAItem) -> str:
    return f"{item.sample_token}::{item.question}"


# --------------------------------------------------------------------------- #
# One-stop evaluation record
# --------------------------------------------------------------------------- #

@dataclass
class SystemReport:
    """Everything the report tabulates for one system."""

    name: str
    accuracy: float
    answered_fraction: float
    accuracy_when_answered: float
    count_mae: Optional[float]
    count_coverage: Dict[str, float]
    cost: Dict[str, float]
    by_answer_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_template: Dict[str, Dict[str, float]] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "answered_fraction": round(self.answered_fraction, 4),
            "accuracy_when_answered": round(self.accuracy_when_answered, 4),
            "count_mae": None if self.count_mae is None else round(self.count_mae, 4),
            "count_coverage": self.count_coverage,
            "cost": self.cost,
            "by_answer_type": self.by_answer_type,
            "by_template": self.by_template,
            "extra": self.extra,
        }


def evaluate_predictions(
    name: str,
    predictions: Sequence[Optional[str]],
    items: Sequence[QAItem],
    cost: CostMeasurement,
    extra: Optional[Dict[str, object]] = None,
) -> SystemReport:
    acc_answered, answered = accuracy_on_answered(predictions, items)
    return SystemReport(
        name=name,
        accuracy=accuracy(predictions, items),
        answered_fraction=answered,
        accuracy_when_answered=acc_answered,
        count_mae=count_mae(predictions, items),
        count_coverage=count_answer_coverage(predictions, items),
        cost=cost.as_dict(),
        by_answer_type=breakdown(predictions, items, lambda it: it.answer_type),
        by_template=breakdown(predictions, items, lambda it: it.template),
        extra=extra or {},
    )
