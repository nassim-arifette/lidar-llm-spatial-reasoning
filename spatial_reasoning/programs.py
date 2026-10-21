"""Execute a structured query against a list of objects.

This is the "calculation" half of the system. It is deterministic, has no
parameters to learn, and returns a trace of the objects it selected. The trace
is what lets a wrong answer be attributed to perception (the right program ran
on the wrong objects) rather than to reasoning.

Nothing here generates text. A counting question is answered by counting.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .objects import Object3D, Scene
from .question_parser import DEFAULT_PARSER, Operation, Query
from .regions import (
    DEFAULT_REGION_CONFIG,
    RegionConfig,
    select,
    validate_region,
)


class ExecStatus(str, enum.Enum):
    OK = "ok"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"  # "the truck", but there are three
    MISSING_REFERENCE = "missing_reference"      # "the truck", but there are none
    UNSUPPORTED = "unsupported"


@dataclass
class ExecutionResult:
    """Answer plus everything needed to explain it."""

    answer: Optional[str]
    status: ExecStatus = ExecStatus.OK
    selected: List[Object3D] = field(default_factory=list)
    candidates: List[Object3D] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ExecStatus.OK and self.answer is not None

    @property
    def count(self) -> Optional[int]:
        """Integer answer when the query was a count, else ``None``."""
        if self.answer is None:
            return None
        try:
            return int(self.answer)
        except (TypeError, ValueError):
            return None


def _has_status(obj: Object3D, status: Optional[str], negate: bool) -> bool:
    """Whether ``obj`` satisfies an attribute constraint.

    An unknown status satisfies neither the positive nor the negated form: a
    traffic cone carries no attribute, so it is neither "standing" nor "not
    standing", and counting it under the negation would assert something the
    annotation does not say. This matches ``dsl._matches`` deliberately -- the
    two executors answer the same programs and must not disagree.
    """
    if status is None:
        return True
    if obj.status is None:
        return False
    return obj.status != status if negate else obj.status == status


def _of_category_and_status(
    scene: Scene, category: str, status: Optional[str], negate: bool
) -> List[Object3D]:
    return [o for o in scene.of_category([category])
            if _has_status(o, status, negate)]


def _resolve_reference(
    scene: Scene,
    category: str,
    status: Optional[str] = None,
    negate: bool = False,
) -> Tuple[Optional[Object3D], ExecStatus, str]:
    """Resolve a definite reference such as "the truck".

    A definite article presupposes exactly one salient object. When the scene
    contains several, choosing the first one would hide the failure and inflate
    the score, so the query is refused instead. This is the behaviour the
    report argues for: ambiguity is reported, not resolved by accident.

    The attribute narrows the reference rather than the answer: "the back of
    the parked truck" is unambiguous in a scene with one parked and two moving
    trucks, and ignoring the word would refuse a question the data answers.
    """
    matches = _of_category_and_status(scene, category, status, negate)
    described = f"{status} {category}" if status else category
    if not matches:
        return None, ExecStatus.MISSING_REFERENCE, f"no {described} in the scene"
    if len(matches) > 1:
        return (None, ExecStatus.AMBIGUOUS_REFERENCE,
                f"{len(matches)} objects of category {described}; "
                f"the question does not say which one")
    return matches[0], ExecStatus.OK, ""


def execute(
    query: Query,
    scene: Scene,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
) -> ExecutionResult:
    """Run ``query`` on ``scene`` and return the answer with its trace."""

    if query.operation is Operation.COMPARE_COUNT:
        return _execute_compare(query, scene, config)

    # Refuse a program the active layout cannot express before looking at the
    # scene. The rule grammar still produces "left" and "right", which the
    # six-relation nuScenes-QA layout does not define; without this the query
    # either crashes the run or, on an empty candidate list, answers "0".
    try:
        validate_region(query.region, config)
    except KeyError as exc:
        return ExecutionResult(None, ExecStatus.UNSUPPORTED, reason=str(exc).strip('"'))

    reference = None
    if query.region.kind == "relative":
        reference, status, reason = _resolve_reference(
            scene, query.region.reference_category,
            query.anchor_status, query.anchor_negate_status)
        if status is not ExecStatus.OK:
            return ExecutionResult(None, status, reason=reason)

    candidates = _of_category_and_status(
        scene, query.category, query.status, query.negate_status)
    selected = select(candidates, query.region, config, reference)

    if query.operation is Operation.COUNT:
        return ExecutionResult(str(len(selected)), ExecStatus.OK, selected, candidates)

    if query.operation is Operation.EXISTS:
        answer = "yes" if selected else "no"
        return ExecutionResult(answer, ExecStatus.OK, selected, candidates)

    return ExecutionResult(None, ExecStatus.UNSUPPORTED,
                           reason=f"operation {query.operation} not implemented")


def _execute_compare(
    query: Query, scene: Scene, config: RegionConfig
) -> ExecutionResult:
    try:
        validate_region(query.region, config)
    except KeyError as exc:
        return ExecutionResult(None, ExecStatus.UNSUPPORTED, reason=str(exc).strip('"'))

    reference = None
    if query.region.kind == "relative":
        reference, status, reason = _resolve_reference(
            scene, query.region.reference_category,
            query.anchor_status, query.anchor_negate_status)
        if status is not ExecStatus.OK:
            return ExecutionResult(None, status, reason=reason)

    first = select(
        _of_category_and_status(scene, query.category,
                                query.status, query.negate_status),
        query.region, config, reference)
    second = select(
        _of_category_and_status(scene, query.second_category,
                                query.second_status, query.second_negate_status),
        query.region, config, reference)

    if query.comparison == "fewer":
        answer = "yes" if len(first) < len(second) else "no"
    else:
        answer = "yes" if len(first) > len(second) else "no"

    return ExecutionResult(answer, ExecStatus.OK, first + second, first + second,
                           reason=f"{len(first)} vs {len(second)}")


# --------------------------------------------------------------------------- #
# The full symbolic system: parse, then execute
# --------------------------------------------------------------------------- #

@dataclass
class ProgramAnswer:
    """What the symbolic pipeline returns for one (question, scene) pair."""

    answer: Optional[str]
    parsed: bool
    executed: bool
    query: Optional[Query] = None
    execution: Optional[ExecutionResult] = None
    failure: str = ""

    @property
    def abstained(self) -> bool:
        """True when the system declined rather than answered."""
        return self.answer is None


class SymbolicSpatialQA:
    """Parse the question, then compute the answer on the detected objects.

    This is the system the report compares against the learned baselines. It
    has no trained parameters at all.

    ``fallback_answer`` controls what happens when the question falls outside
    the grammar. ``None`` (the default) makes the system abstain, which keeps
    coverage and accuracy separate. Setting it to the training-set majority
    answer makes the system directly comparable to a model that always emits
    something -- both numbers are reported in the experiments.
    """

    def __init__(
        self,
        parser=None,
        config: RegionConfig = DEFAULT_REGION_CONFIG,
        fallback_answer: Optional[str] = None,
    ) -> None:
        self.parser = parser or DEFAULT_PARSER
        self.config = config
        self.fallback_answer = fallback_answer

    def answer(self, question: str, scene: Scene) -> ProgramAnswer:
        parse = self.parser.parse(question)
        if not parse.ok:
            return ProgramAnswer(self.fallback_answer, False, False,
                                 failure=f"parse: {parse.reason}")

        result = execute(parse.query, scene, self.config)
        if not result.ok:
            return ProgramAnswer(self.fallback_answer, True, False,
                                 query=parse.query, execution=result,
                                 failure=f"execute: {result.reason}")

        return ProgramAnswer(result.answer, True, True,
                             query=parse.query, execution=result)

    def answer_batch(self, questions: List[str], scenes: List[Scene]) -> List[ProgramAnswer]:
        return [self.answer(q, s) for q, s in zip(questions, scenes)]
