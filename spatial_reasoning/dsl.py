"""A DSL wide enough for what nuScenes-QA actually asks.

The first version of this package used a narrow grammar: count, exist and
compare, over a category and an ego-centric region. Measured on real
nuScenes-QA questions it covered 31% of them. The other 69% are not harder
geometry -- they are different *questions*:

======================  ======  =================================================
template                share   example
======================  ======  =================================================
``exist``               28%     "Are there any trailers?"
``object``              21%     "There is a parked thing; what is it?"
``comparison``          17%     "Do the trailer and the truck have the same status?"
``status``              16%     "The construction vehicle is in what status?"
``count``               18%     "What number of traffic cones are to the back of it?"
======================  ======  =================================================

So this module widens the target language rather than the parser. A query is
an *operation* over one or two *selectors*, and a selector is a description of
a set of objects: a category, optionally a status, optionally a region that may
itself be anchored on another selector ("to the front left of the parked car").

That shape is deliberately close to the operation set KLDrive gives its agent:
retrieve, filter by relation, count, compare. The point of the experiment is
that the language model chooses the program while the executor computes the
value.

Everything here is still parameter-free and exact. Only the parser changes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from .objects import Object3D, Scene, expand_category
from .regions import DEFAULT_REGION_CONFIG, Region, RegionConfig, select

ANY = Region(kind="any")


class Op(str, enum.Enum):
    """Operations the executor implements."""

    COUNT = "count"               # -> integer
    EXISTS = "exists"             # -> yes / no
    IDENTIFY = "identify"         # -> the category of the one matching object
    STATUS = "status"             # -> the status of the one matching object
    SAME_STATUS = "same_status"   # -> yes / no, two objects share a status
    COMPARE_COUNT = "compare_count"   # -> yes / no, more/fewer of A than B


#: Status values, spelled the way nuScenes-QA spells them in its answers.
STATUSES = ("moving", "parked", "stopped", "standing",
            "sitting lying down", "with rider", "without rider")


@dataclass(frozen=True)
class Selector:
    """A description of a set of objects.

    ``anchor`` makes the description relational: the region is evaluated
    relative to the object the anchor picks out, rather than to the ego
    vehicle. nuScenes-QA calls these one-hop questions.
    """

    category: str = "object"
    status: Optional[str] = None
    negate_status: bool = False
    region: Region = ANY
    anchor: Optional["Selector"] = None

    def describe(self) -> str:
        bits = []
        if self.status:
            bits.append(("not " if self.negate_status else "") + self.status)
        bits.append(self.category)
        text = " ".join(bits)
        if self.region.kind != "any":
            where = self.region.describe()
            if self.anchor is not None:
                where = (f"{self.region.direction.replace('_', ' ')} of "
                         f"[{self.anchor.describe()}]")
            text += f" {where}"
        return text


@dataclass(frozen=True)
class Query:
    operation: Op
    target: Selector
    second: Optional[Selector] = None
    comparison: Optional[str] = None     # "more" | "fewer", for COMPARE_COUNT

    def describe(self) -> str:
        if self.operation is Op.COMPARE_COUNT:
            return (f"{self.comparison} [{self.target.describe()}] than "
                    f"[{self.second.describe()}]")
        if self.operation is Op.SAME_STATUS:
            return (f"same status: [{self.target.describe()}] vs "
                    f"[{self.second.describe()}]")
        return f"{self.operation.value} [{self.target.describe()}]"


class ExecStatus(str, enum.Enum):
    OK = "ok"
    AMBIGUOUS = "ambiguous"       # the description matches several objects
    MISSING = "missing"           # it matches none
    UNSUPPORTED = "unsupported"
    NO_STATUS = "no_status"       # the object carries no attribute


@dataclass
class Result:
    answer: Optional[str]
    status: ExecStatus = ExecStatus.OK
    selected: List[Object3D] = field(default_factory=list)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ExecStatus.OK and self.answer is not None


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def _matches(obj: Object3D, selector: Selector) -> bool:
    if selector.category not in ("object", "thing"):
        if obj.category not in expand_category(selector.category):
            return False
    if selector.status:
        has = (obj.status == selector.status)
        if selector.negate_status:
            if has:
                return False
        elif not has:
            return False
    return True


def resolve(
    selector: Selector,
    scene: Scene,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
    _depth: int = 0,
) -> Tuple[List[Object3D], ExecStatus, str]:
    """Return the objects a selector picks out.

    Anchors are resolved recursively and must be *unique*: "to the left of the
    truck" presupposes exactly one truck. When several match, the query is
    refused rather than resolved by taking the first -- a silent choice there
    would turn a question we cannot answer into a coin flip that sometimes
    scores.
    """
    if _depth > 3:
        return [], ExecStatus.UNSUPPORTED, "anchor chain too deep"

    candidates = [o for o in scene.objects if _matches(o, selector)]

    reference: Optional[Object3D] = None
    if selector.anchor is not None:
        anchors, status, reason = resolve(selector.anchor, scene, config, _depth + 1)
        if status is not ExecStatus.OK:
            return [], status, reason
        if not anchors:
            return [], ExecStatus.MISSING, f"no {selector.anchor.describe()}"
        if len(anchors) > 1:
            return ([], ExecStatus.AMBIGUOUS,
                    f"{len(anchors)} objects match [{selector.anchor.describe()}]")
        reference = anchors[0]
        candidates = [o for o in candidates if o is not reference]

    if selector.region.kind != "any":
        region = selector.region
        if reference is not None:
            region = replace(region, kind="relative",
                             reference_category=selector.anchor.category)
        candidates = select(candidates, region, config, reference)

    return candidates, ExecStatus.OK, ""


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def execute(
    query: Query,
    scene: Scene,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
) -> Result:
    objs, status, reason = resolve(query.target, scene, config)
    if status is not ExecStatus.OK:
        return Result(None, status, reason=reason)

    if query.operation is Op.COUNT:
        return Result(str(len(objs)), ExecStatus.OK, objs)

    if query.operation is Op.EXISTS:
        return Result("yes" if objs else "no", ExecStatus.OK, objs)

    if query.operation in (Op.IDENTIFY, Op.STATUS):
        if not objs:
            return Result(None, ExecStatus.MISSING,
                          reason=f"no {query.target.describe()}")
        if len(objs) > 1:
            return Result(None, ExecStatus.AMBIGUOUS, objs,
                          f"{len(objs)} objects match [{query.target.describe()}]")
        obj = objs[0]
        if query.operation is Op.IDENTIFY:
            return Result(obj.category.replace("_", " "), ExecStatus.OK, objs)
        if obj.status is None:
            return Result(None, ExecStatus.NO_STATUS, objs,
                          "the annotation carries no attribute")
        return Result(obj.status, ExecStatus.OK, objs)

    if query.operation in (Op.SAME_STATUS, Op.COMPARE_COUNT):
        if query.second is None:
            return Result(None, ExecStatus.UNSUPPORTED,
                          reason="binary operation without a second selector")
        others, status2, reason2 = resolve(query.second, scene, config)
        if status2 is not ExecStatus.OK:
            return Result(None, status2, reason=reason2)

        if query.operation is Op.COMPARE_COUNT:
            if query.comparison == "fewer":
                return Result("yes" if len(objs) < len(others) else "no",
                              ExecStatus.OK, objs + others,
                              f"{len(objs)} vs {len(others)}")
            return Result("yes" if len(objs) > len(others) else "no",
                          ExecStatus.OK, objs + others,
                          f"{len(objs)} vs {len(others)}")

        if len(objs) != 1 or len(others) != 1:
            return Result(None, ExecStatus.AMBIGUOUS, objs + others,
                          f"{len(objs)} and {len(others)} objects match")
        a, b = objs[0].status, others[0].status
        if a is None or b is None:
            return Result(None, ExecStatus.NO_STATUS, objs + others,
                          "one of the objects carries no attribute")
        return Result("yes" if a == b else "no", ExecStatus.OK, objs + others)

    return Result(None, ExecStatus.UNSUPPORTED,
                  reason=f"operation {query.operation} not implemented")


# --------------------------------------------------------------------------- #
# Lifting the narrow grammar into this one
# --------------------------------------------------------------------------- #

def lift(old_query) -> Query:
    """Translate a :mod:`question_parser` query into this DSL.

    Both parsers then run on the same executor, so the comparison between the
    rule-based parser and the language-model parser isolates the parser and not
    the thing it parses into.
    """
    from .question_parser import Operation as OldOp

    region = old_query.region
    anchor = None
    if region.kind == "relative":
        anchor = Selector(category=region.reference_category)

    target = Selector(category=old_query.category, region=region, anchor=anchor)

    if old_query.operation is OldOp.COUNT:
        return Query(Op.COUNT, target)
    if old_query.operation is OldOp.EXISTS:
        return Query(Op.EXISTS, target)
    return Query(
        Op.COMPARE_COUNT,
        target,
        Selector(category=old_query.second_category, region=region, anchor=anchor),
        comparison=old_query.comparison,
    )
