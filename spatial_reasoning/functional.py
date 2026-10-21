"""A typed functional program language for nuScenes-QA, after CLEVR.

CLEVR [Johnson et al., CVPR 2017, arXiv:1612.06890] represents every question
as a *functional program* over a small catalogue of elementary operations, and
executes it on the scene graph to obtain the answer. Figure 2 of that paper
gives the catalogue:

======================  ==========================  ======================
inputs                  function                    output
======================  ==========================  ======================
value, objects          ``Filter <attr>``           objects
objects, objects        ``And`` / ``Or``            objects
objects                 ``Exist`` / ``Count``       yes-no / number
object                  ``Query <attr>``            value
value, value            ``Equal``                   yes-no
number, number          ``Equal`` / ``Less`` / ``More``   yes-no
object                  ``Same <attr>``             objects
value, object           ``Relate``                  objects
objects                 ``Unique``                  object
======================  ==========================  ======================

This module is that catalogue, adapted to nuScenes. Three changes, each of
which is a claim about the data rather than a convenience:

**The attributes are different.** CLEVR has size, colour, material and shape.
nuScenes annotates a *category* (ten classes, plus the group names in
``objects.CATEGORY_GROUPS``) and a *status* (``moving``, ``parked``,
``stopped``, ``standing``, ``sitting lying down``, ``with rider``, ``without
rider``). There is no colour, which is why colour questions are unanswerable
here and not merely unimplemented.

**There is an ego vehicle.** CLEVR has no privileged viewpoint, so every
spatial relation is between two objects. nuScenes-QA asks both kinds, so
``relate`` (anchored on a resolved object, CLEVR's ``Relate``) and
``relate_ego`` (anchored on the sensor platform) are separate functions. Their
sectors are the six 60-degree bearings nuScenes-QA uses, not CLEVR's four
half-plane relations.

**``unique`` can fail, and says so.** CLEVR *generates* its questions, so it
prunes programs whose ``unique`` would receive a set of size other than one --
the paper calls those questions ill-posed ("What color is the cube to the right
of the sphere?" when several cubes are) or degenerate (when the reference is
unnecessary). We do not generate the questions and we do not have perfect
perception, so well-posedness cannot be guaranteed in advance: it has to be
*detected*. ``unique`` therefore returns an explicit failure, and the
experiment reports those failures instead of resolving them by taking the
first match. A system that silently picks one is right about half the time and
cannot be audited.

The program itself is a list of nodes in topological order, each naming a
function, its constant arguments and the indices of its inputs -- CLEVR's own
serialisation. It is checked against the signatures below *before* execution,
so a malformed program is a reported error and never a plausible answer.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .objects import CANONICAL_CATEGORIES, CATEGORY_GROUPS, Object3D, Scene, expand_category
from .regions import (
    DEFAULT_REGION_CONFIG,
    RegionConfig,
    in_direction,
    relative_direction,
)

#: The status values the annotation actually produces. Mirrors
#: ``nuscenes_adapter.ATTRIBUTE_TO_STATUS``; a value outside it would filter
#: every object away and look like an empty scene.
STATUSES: Tuple[str, ...] = ("moving", "parked", "stopped", "standing",
                             "sitting lying down", "with rider", "without rider")

#: Every name ``filter_category`` accepts: the ten classes and the group names.
CATEGORY_VOCABULARY: Tuple[str, ...] = tuple(
    sorted(set(CANONICAL_CATEGORIES) | set(CATEGORY_GROUPS)))


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

class Type(str, enum.Enum):
    """The four value kinds a program manipulates, as in CLEVR."""

    OBJECTS = "objects"
    OBJECT = "object"
    NUMBER = "number"
    BOOL = "bool"
    VALUE = "value"     # an attribute value: a category or a status


class Failure(str, enum.Enum):
    """Why a program did not produce an answer.

    These are the cases CLEVR removes at generation time. Keeping them as
    outcomes is the point: each one is a question the system declined, and the
    experiment reports the rate rather than hiding it inside the accuracy.
    """

    AMBIGUOUS = "ambiguous"       # `unique` received more than one object
    MISSING = "missing"           # `unique` received none
    NO_STATUS = "no_status"       # the annotation carries no attribute
    ILL_TYPED = "ill_typed"       # the program does not type-check
    MALFORMED = "malformed"       # it is not a program at all


class ProgramError(Exception):
    """Raised by a function when it cannot return a value."""

    def __init__(self, failure: Failure, message: str) -> None:
        super().__init__(message)
        self.failure = failure
        self.message = message


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Signature:
    """One entry of the function catalogue."""

    name: str
    inputs: Tuple[Type, ...]
    output: Type
    #: Name of the constant argument, and the vocabulary it is drawn from.
    #: ``None`` for functions that take no constant.
    value_input: Optional[str] = None
    vocabulary: Tuple[str, ...] = ()
    doc: str = ""

    @property
    def arity(self) -> int:
        return len(self.inputs)


def _sector_names(config: RegionConfig) -> Tuple[str, ...]:
    return tuple(sorted(config.sectors))


#: Built against the default layout; :func:`catalogue` rebuilds the direction
#: vocabularies for whichever layout an experiment runs under, because the six
#: nuScenes-QA bearings and the eight octants are different languages and a
#: program written in one is not a program in the other.
def catalogue(config: RegionConfig = DEFAULT_REGION_CONFIG) -> Dict[str, Signature]:
    """The function catalogue, with direction vocabularies from ``config``."""
    directions = _sector_names(config)
    signatures = [
        Signature("scene", (), Type.OBJECTS,
                  doc="every detected object in the sample"),

        Signature("filter_category", (Type.OBJECTS,), Type.OBJECTS,
                  "category", CATEGORY_VOCABULARY,
                  doc="keep the objects of a class or group"),
        Signature("filter_status", (Type.OBJECTS,), Type.OBJECTS,
                  "status", STATUSES,
                  doc="keep the objects with this attribute; an object whose "
                      "attribute is unknown is not kept"),
        Signature("filter_not_status", (Type.OBJECTS,), Type.OBJECTS,
                  "status", STATUSES,
                  doc="keep the objects with a different, KNOWN attribute; an "
                      "object with no attribute is neither standing nor not "
                      "standing, so it is dropped either way"),

        Signature("unique", (Type.OBJECTS,), Type.OBJECT,
                  doc="the one object in the set; fails if there are none or "
                      "several, instead of choosing"),

        Signature("relate_ego", (Type.OBJECTS,), Type.OBJECTS,
                  "direction", directions,
                  doc="keep the objects in this bearing sector of the ego "
                      "vehicle"),
        Signature("relate", (Type.OBJECT,), Type.OBJECTS,
                  "direction", directions,
                  doc="the scene objects in this bearing sector of the given "
                      "object, seen from the ego vehicle, excluding it"),

        Signature("and_", (Type.OBJECTS, Type.OBJECTS), Type.OBJECTS,
                  doc="objects in both sets"),
        Signature("or_", (Type.OBJECTS, Type.OBJECTS), Type.OBJECTS,
                  doc="objects in either set"),

        Signature("same_status", (Type.OBJECT,), Type.OBJECTS,
                  doc="the other objects sharing this object's attribute; "
                      "CLEVR's same-attribute relation"),
        Signature("same_category", (Type.OBJECT,), Type.OBJECTS,
                  doc="the other objects of this object's class"),

        Signature("count", (Type.OBJECTS,), Type.NUMBER,
                  doc="how many objects are in the set"),
        Signature("exist", (Type.OBJECTS,), Type.BOOL,
                  doc="whether the set is non-empty"),

        Signature("query_category", (Type.OBJECT,), Type.VALUE,
                  doc="the class of the object"),
        Signature("query_status", (Type.OBJECT,), Type.VALUE,
                  doc="the attribute of the object; fails if it has none"),

        Signature("equal_value", (Type.VALUE, Type.VALUE), Type.BOOL,
                  doc="whether two attribute values are the same"),
        Signature("equal_count", (Type.NUMBER, Type.NUMBER), Type.BOOL,
                  doc="whether two counts are equal"),
        Signature("greater_than", (Type.NUMBER, Type.NUMBER), Type.BOOL,
                  doc="whether the first count exceeds the second"),
        Signature("less_than", (Type.NUMBER, Type.NUMBER), Type.BOOL,
                  doc="whether the first count is below the second"),
    ]
    return {s.name: s for s in signatures}


DEFAULT_CATALOGUE = catalogue()
FUNCTION_NAMES: Tuple[str, ...] = tuple(DEFAULT_CATALOGUE)


# --------------------------------------------------------------------------- #
# Programs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Node:
    """One call. ``inputs`` are indices of earlier nodes, as in CLEVR."""

    function: str
    value_input: Optional[str] = None
    inputs: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {"function": self.function,
                "value_inputs": [self.value_input] if self.value_input else [],
                "inputs": list(self.inputs)}


@dataclass(frozen=True)
class Program:
    """A list of nodes in topological order; the last one is the answer."""

    nodes: Tuple[Node, ...]

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps([n.to_dict() for n in self.nodes], indent=indent)

    def describe(self) -> str:
        """The program as nested calls, which is how a reader checks it."""
        def render(i: int) -> str:
            node = self.nodes[i]
            args = [render(j) for j in node.inputs]
            if node.value_input is not None:
                args.insert(0, repr(node.value_input))
            return f"{node.function}({', '.join(args)})"
        return render(len(self.nodes) - 1) if self.nodes else "<empty>"

    @property
    def question_type(self) -> str:
        """The outermost function, which is how CLEVR labels a question type."""
        return self.nodes[-1].function if self.nodes else "<empty>"


def program_from_json(
    payload: Any, catalogue_: Optional[Dict[str, Signature]] = None
) -> Program:
    """Parse and type-check a program. Raises :class:`ProgramError`.

    Both halves matter. Parsing rejects what is not a program; type-checking
    rejects what is a program but not a *well-typed* one -- ``count`` applied
    to an object, ``relate`` applied to a set. Without the second check an
    ill-typed program would raise somewhere inside execution and be reported as
    an executor failure, which would attribute a parser error to the executor.
    """
    cat = catalogue_ or DEFAULT_CATALOGUE

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProgramError(Failure.MALFORMED, f"not JSON: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("program", payload)
    if not isinstance(payload, list) or not payload:
        raise ProgramError(Failure.MALFORMED, "program must be a non-empty list")

    nodes: List[Node] = []
    outputs: List[Type] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ProgramError(Failure.MALFORMED, f"node {index} is not an object")
        name = raw.get("function")
        if name not in cat:
            raise ProgramError(Failure.MALFORMED,
                               f"node {index}: unknown function {name!r}")
        sig = cat[name]

        values = raw.get("value_inputs") or []
        if isinstance(values, str):
            values = [values]
        if sig.value_input is None:
            if values:
                raise ProgramError(Failure.ILL_TYPED,
                                   f"node {index}: {name} takes no constant")
            value: Optional[str] = None
        else:
            if len(values) != 1:
                raise ProgramError(
                    Failure.ILL_TYPED,
                    f"node {index}: {name} takes exactly one {sig.value_input}")
            value = str(values[0]).strip().lower().replace("-", "_")
            if value not in sig.vocabulary:
                # Never coerced to a default. A near-miss ("front-left" under a
                # six-sector layout, "red") is a different question, and
                # answering it would score a program nobody wrote.
                raise ProgramError(
                    Failure.ILL_TYPED,
                    f"node {index}: {value!r} is not a {sig.value_input}; "
                    f"expected one of {list(sig.vocabulary)}")

        inputs = raw.get("inputs") or []
        if not isinstance(inputs, list) or len(inputs) != sig.arity:
            raise ProgramError(
                Failure.ILL_TYPED,
                f"node {index}: {name} takes {sig.arity} input(s), got {inputs!r}")
        checked: List[int] = []
        for position, j in enumerate(inputs):
            if not isinstance(j, int) or not 0 <= j < index:
                raise ProgramError(
                    Failure.ILL_TYPED,
                    f"node {index}: input {j!r} is not an earlier node")
            if outputs[j] is not sig.inputs[position]:
                raise ProgramError(
                    Failure.ILL_TYPED,
                    f"node {index}: {name} wants {sig.inputs[position].value} "
                    f"as input {position}, node {j} produces {outputs[j].value}")
            checked.append(j)

        nodes.append(Node(name, value, tuple(checked)))
        outputs.append(sig.output)

    if outputs[-1] not in (Type.NUMBER, Type.BOOL, Type.VALUE):
        raise ProgramError(
            Failure.ILL_TYPED,
            f"a program must end in an answer, not {outputs[-1].value}")
    return Program(tuple(nodes))


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    """One executed node, for the trace."""

    index: int
    function: str
    value_input: Optional[str]
    inputs: Tuple[int, ...]
    output: str
    tokens: List[str] = field(default_factory=list)

    def describe(self) -> str:
        call = self.function + (f"[{self.value_input}]" if self.value_input else "")
        return f"{self.index}: {call} -> {self.output}"


@dataclass
class Answer:
    """The result of running a program, with everything needed to explain it."""

    answer: Optional[str]
    failure: Optional[Failure] = None
    reason: str = ""
    trace: List[Step] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failure is None and self.answer is not None

    def explain(self) -> str:
        return "\n".join(step.describe() for step in self.trace)


def _tokens(objects: Sequence[Object3D], limit: int = 40) -> List[str]:
    return [o.token for o in objects if getattr(o, "token", None)][:limit]


def _summarise(value: Any) -> str:
    if isinstance(value, list):
        return f"{len(value)} object(s)"
    if isinstance(value, Object3D):
        return f"object({value.category})"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def execute(
    program: Program,
    scene: Scene,
    config: RegionConfig = DEFAULT_REGION_CONFIG,
) -> Answer:
    """Run ``program`` on ``scene``.

    Returns the answer in the spelling nuScenes-QA uses for its gold answers --
    a bare integer, ``yes``/``no``, or an attribute value with spaces rather
    than underscores -- together with the trace. Nothing here generates text:
    a counting question is answered by counting.
    """
    values: List[Any] = []
    trace: List[Step] = []

    for index, node in enumerate(program.nodes):
        args = [values[j] for j in node.inputs]
        try:
            out = _APPLY[node.function](args, node.value_input, scene, config)
        except ProgramError as exc:
            trace.append(Step(index, node.function, node.value_input,
                              node.inputs, f"<{exc.failure.value}>"))
            return Answer(None, exc.failure, exc.message, trace)
        values.append(out)
        trace.append(Step(index, node.function, node.value_input, node.inputs,
                          _summarise(out),
                          _tokens(out) if isinstance(out, list) else []))

    final = values[-1]
    if isinstance(final, bool):
        return Answer("yes" if final else "no", trace=trace)
    if isinstance(final, int):
        return Answer(str(final), trace=trace)
    return Answer(str(final), trace=trace)


# -- the functions ----------------------------------------------------------- #

def _f_scene(args, value, scene, config):
    return list(scene.objects)


def _f_filter_category(args, value, scene, config):
    wanted = set(expand_category(value))
    return [o for o in args[0] if o.category in wanted]


def _f_filter_status(args, value, scene, config):
    return [o for o in args[0] if o.status == value]


def _f_filter_not_status(args, value, scene, config):
    # An unknown attribute is excluded from both the filter and its negation.
    # Counting an unannotated traffic cone as "not standing" would assert
    # something the data does not say.
    return [o for o in args[0] if o.status is not None and o.status != value]


def _f_unique(args, value, scene, config):
    objects = args[0]
    if not objects:
        raise ProgramError(Failure.MISSING, "no object matches the description")
    if len(objects) > 1:
        raise ProgramError(
            Failure.AMBIGUOUS,
            f"{len(objects)} objects match the description; the question "
            f"presupposes one")
    return objects[0]


def _f_relate_ego(args, value, scene, config):
    return [o for o in args[0] if in_direction(o, value, config)]


def _f_relate(args, value, scene, config):
    reference = args[0]
    return [o for o in scene.objects
            if o is not reference
            and relative_direction(o, reference, value, config, frame="ego")]


def _f_and(args, value, scene, config):
    other = {id(o) for o in args[1]}
    return [o for o in args[0] if id(o) in other]


def _f_or(args, value, scene, config):
    seen = {id(o) for o in args[0]}
    return list(args[0]) + [o for o in args[1] if id(o) not in seen]


def _f_same_status(args, value, scene, config):
    reference = args[0]
    if reference.status is None:
        raise ProgramError(Failure.NO_STATUS,
                           "the reference object carries no attribute, so "
                           "'the same status' has no referent")
    return [o for o in scene.objects
            if o is not reference and o.status == reference.status]


def _f_same_category(args, value, scene, config):
    reference = args[0]
    return [o for o in scene.objects
            if o is not reference and o.category == reference.category]


def _f_count(args, value, scene, config):
    return len(args[0])


def _f_exist(args, value, scene, config):
    return bool(args[0])


def _f_query_category(args, value, scene, config):
    return args[0].category.replace("_", " ")


def _f_query_status(args, value, scene, config):
    status = args[0].status
    if status is None:
        raise ProgramError(Failure.NO_STATUS,
                           "the annotation carries no attribute for this object")
    return status


def _f_equal_value(args, value, scene, config):
    return args[0] == args[1]


def _f_equal_count(args, value, scene, config):
    return args[0] == args[1]


def _f_greater_than(args, value, scene, config):
    return args[0] > args[1]


def _f_less_than(args, value, scene, config):
    return args[0] < args[1]


_APPLY: Dict[str, Callable[[List[Any], Optional[str], Scene, RegionConfig], Any]] = {
    "scene": _f_scene,
    "filter_category": _f_filter_category,
    "filter_status": _f_filter_status,
    "filter_not_status": _f_filter_not_status,
    "unique": _f_unique,
    "relate_ego": _f_relate_ego,
    "relate": _f_relate,
    "and_": _f_and,
    "or_": _f_or,
    "same_status": _f_same_status,
    "same_category": _f_same_category,
    "count": _f_count,
    "exist": _f_exist,
    "query_category": _f_query_category,
    "query_status": _f_query_status,
    "equal_value": _f_equal_value,
    "equal_count": _f_equal_count,
    "greater_than": _f_greater_than,
    "less_than": _f_less_than,
}

assert set(_APPLY) == set(DEFAULT_CATALOGUE), (
    "every signature needs an implementation and vice versa: "
    f"{set(_APPLY) ^ set(DEFAULT_CATALOGUE)}")


# --------------------------------------------------------------------------- #
# The schema a constrained decoder is given
# --------------------------------------------------------------------------- #

def json_schema(config: RegionConfig = DEFAULT_REGION_CONFIG) -> Dict[str, Any]:
    """A JSON Schema describing exactly the well-formed programs.

    This is what makes "constrained generation" mean something stronger than
    "asked politely for JSON". Every function name, category, status and
    direction is an ``enum``, so a decoder that honours the schema cannot emit
    a token outside the language: the 26% invalid-parse rate measured with free
    decoding becomes structurally impossible, and what remains is the model's
    *semantic* error rate, which is the quantity the experiment is about.

    Input arity and types still have to be checked afterwards by
    :func:`program_from_json` -- a JSON Schema cannot express "input 0 must be
    an earlier node whose output type is ``object``".
    """
    cat = catalogue(config)
    variants = []
    for sig in cat.values():
        properties: Dict[str, Any] = {
            "function": {"const": sig.name, "description": sig.doc},
            "inputs": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "minItems": sig.arity,
                "maxItems": sig.arity,
                "description": "indices of earlier nodes, in order",
            },
        }
        if sig.value_input is not None:
            properties["value_inputs"] = {
                "type": "array",
                "items": {"enum": list(sig.vocabulary)},
                "minItems": 1, "maxItems": 1,
            }
            required = ["function", "value_inputs", "inputs"]
        else:
            properties["value_inputs"] = {"type": "array", "maxItems": 0}
            required = ["function", "inputs"]
        variants.append({"type": "object", "properties": properties,
                         "required": required, "additionalProperties": False})

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "nuScenes-QA functional program",
        "description": __doc__.split("\n\n")[0],
        "type": "object",
        "properties": {
            "program": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {"anyOf": variants},
                "description": "nodes in topological order; the last one is "
                               "the answer",
            }
        },
        "required": ["program"],
        "additionalProperties": False,
    }
