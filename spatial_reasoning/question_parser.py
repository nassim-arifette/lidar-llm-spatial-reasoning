"""Question -> structured query.

This is the "understanding" half of the system. It converts a natural-language
question into a small program specification; it never looks at the scene and
never produces an answer. Keeping it scene-blind is what makes the error
decomposition in ``evaluate.py`` possible: a wrong answer is a parsing error if
and only if this module produced the wrong :class:`Query`, independently of
what the detector saw.

The default parser is rule-based. That is a deliberate baseline, not a claim
about language understanding: it covers the templated question forms of
nuScenes-QA and refuses everything else. A refusal is reported as
``ParseStatus.UNSUPPORTED`` rather than guessed, so coverage and accuracy stay
separate numbers.

Three grammars, and why there are three
---------------------------------------

The first version of this parser had no place to put a status word, so it
dropped it: "how many standing pedestrians are there" was executed as "how
many pedestrians are there" (30 against a gold answer of 8), and "what number
of stopped things are there" became "how many objects are there" (71 against
10). The same hole swallowed preamble clauses -- "there is a trailer; what
number of traffic cones are to the back of it?" took *trailer* as the thing to
count, because trailer was the first category in the string.

Refusing those questions stopped the wrong answers but answered almost
nothing: 4.4% of nuScenes-QA, with status alone accounting for 3767 of 5916
refusals. So the fix is not to refuse more carefully, it is to make the
grammar able to say what the question says. :class:`Grammar` keeps all three
behaviours available, because the comparison between them is the measurement:

``Grammar.EXPRESSIVE`` (default)
    Parse the status filter and the preamble referent into the query.
``Grammar.STRICT``
    Refuse anything the narrow grammar cannot hold. Correct but near-silent.
``Grammar.LEGACY``
    The original behaviour: drop what you cannot express and answer anyway.
    Kept only so the cost of doing that can be measured on identical
    questions. It must never be the default.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .regions import ANY_REGION, Region


class Operation(str, enum.Enum):
    """The operations the executor knows how to run."""

    COUNT = "count"            # "how many cars ..."       -> integer
    EXISTS = "exists"          # "are there any cars ..."  -> yes / no
    COMPARE_COUNT = "compare"  # "are there more cars than trucks" -> yes / no


class ParseStatus(str, enum.Enum):
    OK = "ok"
    UNSUPPORTED = "unsupported"   # outside the grammar: we decline to answer
    AMBIGUOUS = "ambiguous"       # inside the grammar but under-specified


class Grammar(str, enum.Enum):
    """How much of a question the parser is allowed to express.

    See the module docstring. ``LEGACY`` reproduces the pre-``1e7c53e``
    behaviour and exists only as an experimental control.
    """

    EXPRESSIVE = "expressive"
    STRICT = "strict"
    LEGACY = "legacy"


@dataclass(frozen=True)
class Query:
    """A structured, executable description of a question."""

    operation: Operation
    category: str                       # canonical class or group name
    region: Region = ANY_REGION
    second_category: Optional[str] = None   # for COMPARE_COUNT
    comparison: Optional[str] = None        # "more" | "fewer"

    #: Attribute constraint on the subject, spelled as ``objects.py`` spells it
    #: ("moving", "parked", "with rider", "sitting lying down", ...).
    #: ``negate_status`` turns it into "not <status>"; an object whose status is
    #: unknown satisfies neither form (see ``dsl._matches``).
    status: Optional[str] = None
    negate_status: bool = False
    second_status: Optional[str] = None
    second_negate_status: bool = False

    #: Attribute constraint on the object the region is anchored to, for
    #: one-hop questions: "to the back of the *parked* truck". The anchor's
    #: category lives in ``region.reference_category``, which is where the
    #: executor already looks for it.
    anchor_status: Optional[str] = None
    anchor_negate_status: bool = False

    def _subject(self) -> str:
        return _spell(self.category, self.status, self.negate_status)

    def _where(self) -> str:
        where = self.region.describe()
        if self.region.kind == "relative" and self.anchor_status:
            anchor = _spell(self.region.reference_category or "object",
                            self.anchor_status, self.anchor_negate_status)
            where = (f"{(self.region.direction or '').replace('_', ' ')} of the "
                     f"{anchor} ({self.region.frame} frame)")
        return where

    def describe(self) -> str:
        if self.operation is Operation.COMPARE_COUNT:
            second = _spell(self.second_category or "object",
                            self.second_status, self.second_negate_status)
            return (f"compare: are there {self.comparison} {self._subject()} "
                    f"than {second} {self._where()}")
        verb = "count" if self.operation is Operation.COUNT else "exists"
        return f"{verb}: {self._subject()} {self._where()}"


def _spell(category: str, status: Optional[str], negate: bool) -> str:
    if not status:
        return category
    return f"{'not ' if negate else ''}{status} {category}"


@dataclass
class ParseResult:
    status: ParseStatus
    query: Optional[Query] = None
    reason: str = ""
    matched_rules: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is ParseStatus.OK and self.query is not None


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

#: Surface forms -> canonical category or group. Order matters only in that
#: longer forms are tried first (see ``_CATEGORY_PATTERN``).
_CATEGORY_LEXICON: Dict[str, str] = {
    "car": "car", "cars": "car", "sedan": "car", "sedans": "car",
    "truck": "truck", "trucks": "truck", "lorry": "truck", "lorries": "truck",
    "bus": "bus", "buses": "bus", "busses": "bus",
    "trailer": "trailer", "trailers": "trailer",
    "construction vehicle": "construction_vehicle",
    "construction vehicles": "construction_vehicle",
    "pedestrian": "pedestrian", "pedestrians": "pedestrian",
    "person": "pedestrian", "people": "pedestrian", "persons": "pedestrian",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle",
    "motorbike": "motorcycle", "motorbikes": "motorcycle",
    "bicycle": "bicycle", "bicycles": "bicycle",
    "bike": "bicycle", "bikes": "bicycle",
    "traffic cone": "traffic_cone", "traffic cones": "traffic_cone",
    "cone": "traffic_cone", "cones": "traffic_cone",
    "barrier": "barrier", "barriers": "barrier",
    # groups
    "vehicle": "vehicle", "vehicles": "vehicle",
    "thing": "object", "things": "object",
    "object": "object", "objects": "object",
}

_CATEGORY_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _CATEGORY_LEXICON), key=len, reverse=True)) + r")\b"
)

#: Surface forms -> the status words nuScenes-QA uses in its answers, which are
#: also the values ``nuscenes_adapter.ATTRIBUTE_TO_STATUS`` produces. A form
#: that does not appear there would filter every object away, so this table is
#: deliberately a mirror of that one and not a superset.
_STATUS_LEXICON: Dict[str, str] = {
    "moving": "moving",
    "parked": "parked",
    "stopped": "stopped",
    "standing": "standing",
    "sitting or lying down": "sitting lying down",
    "sitting lying down": "sitting lying down",
    "lying down": "sitting lying down",
    "sitting": "sitting lying down",
    "lying": "sitting lying down",
    "with rider": "with rider",
    "with a rider": "with rider",
    "without rider": "without rider",
    "without a rider": "without rider",
}

#: ``not`` is the only negation taken: "no moving cars" is a different
#: operation (existence), not a negated filter, and reading it as one would
#: answer "how many cars are not moving" to a yes/no question.
_STATUS_PATTERN = re.compile(
    r"\b(?P<neg>not\s+)?(?P<word>"
    + "|".join(sorted((re.escape(k) for k in _STATUS_LEXICON), key=len, reverse=True))
    + r")\b"
)

#: Multi-word direction phrases, longest first so "front left" wins over "front".
_DIRECTION_PHRASES: List[Tuple[str, str]] = sorted(
    [
        ("front left", "front_left"), ("left front", "front_left"),
        ("front right", "front_right"), ("right front", "front_right"),
        ("back left", "back_left"), ("rear left", "back_left"), ("left back", "back_left"),
        ("back right", "back_right"), ("rear right", "back_right"), ("right back", "back_right"),
        ("in front of", "front"), ("ahead of", "front"), ("front", "front"),
        ("behind", "back"), ("back of", "back"), ("back", "back"), ("rear", "back"),
        ("to the left of", "left"), ("left of", "left"), ("left", "left"),
        ("to the right of", "right"), ("right of", "right"), ("right", "right"),
    ],
    key=lambda kv: len(kv[0]),
    reverse=True,
)

_EGO_WORDS = ("ego vehicle", "ego car", "ego", "us", "me", "the car i am in",
              "our vehicle", "the vehicle")

#: Word boundaries are not decoration here. A plain substring test made "us" a
#: match inside "b-us": "how many cars are to the left of the bus" was read as
#: "to the left of the ego vehicle" and answered about the wrong origin,
#: silently. The same test let "the vehicle" match inside "the construction
#: vehicle".
_EGO_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _EGO_WORDS) + r")\b")

_COUNT_PATTERNS = (
    re.compile(r"\bhow many\b"),
    re.compile(r"\bwhat (?:is the )?number of\b"),
    re.compile(r"\bcount (?:the|how many)\b"),
)

_EXISTS_PATTERNS = (
    re.compile(r"\bare there (?:any|some)?\b"),
    re.compile(r"\bis there (?:any|a|an)?\b"),
    re.compile(r"\bdo(?:es)? (?:the scene|it) (?:contain|have)\b"),
    re.compile(r"\bcan (?:you|we) see\b"),
)

_COMPARE_PATTERN = re.compile(r"\bare there (more|fewer|less)\b.*\bthan\b")

#: "There is a <referent>; <question about something else>". Matched on the
#: RAW question: ``_normalise`` deletes the clause separator this keys on, and
#: matching the normalised text made an earlier version of this rule
#: unreachable. The split is only taken when the second clause is itself a
#: question the grammar recognises, so "are there more cars than trucks,
#: aren't there?" is not cut in half.
_PREAMBLE_RE = re.compile(r"^\s*(there (?:is|are)\b[^;,]*)[;,]\s*(.+)$")

#: A referential pronoun with no noun phrase after the direction word.
_PRONOUN_RE = re.compile(r"\b(it|them)\b")

#: Comparisons of attributes *between* objects. Unlike status filters these are
#: genuinely outside the target language: the executor can test whether two
#: resolved objects share a status, but it has no way to bind one object's
#: status as the filter for a set ("other things in the same status as the
#: bicycle"). Colour is not annotated in nuScenes at all.
_ATTRIBUTE_COMPARISON_RE = re.compile(
    r"\bsame (status|colour|color|attribute)\b")

_NEAR_PATTERN = re.compile(r"\b(nearby|near(?:est)?|close to|closest)\b")
_FAR_PATTERN = re.compile(r"\b(far(?:thest)?|distant|far from|far away)\b")
_RADIUS_PATTERN = re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b")


def _normalise(question: str) -> str:
    q = question.strip().lower()
    q = q.replace("-", " ")
    q = re.sub(r"[^\w\s\.]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _find_categories(text: str) -> List[Tuple[int, str]]:
    """All category mentions as ``(position, canonical_name)``, in order."""
    found: List[Tuple[int, str]] = []
    for match in _CATEGORY_PATTERN.finditer(text):
        found.append((match.start(), _CATEGORY_LEXICON[match.group(1)]))
    return found


def _find_statuses(text: str) -> List[Tuple[int, int, str, bool]]:
    """All status mentions as ``(start, end, canonical_status, negated)``."""
    out: List[Tuple[int, int, str, bool]] = []
    for match in _STATUS_PATTERN.finditer(text):
        word = match.group("word")
        out.append((match.start(), match.end(),
                    _STATUS_LEXICON[word], match.group("neg") is not None))
    return out


def _attach_statuses(
    cats: List[Tuple[int, str]], statuses: List[Tuple[int, int, str, bool]]
) -> Dict[int, Tuple[str, bool]]:
    """Bind each status mention to the category it modifies.

    English puts the attribute before the noun in every nuScenes-QA template
    that has one ("standing pedestrians", "the parked truck"), so a status
    attaches to the first category that starts after it. The predicative form
    ("how many cars are moving") is handled by falling back to the nearest
    category *before* the status, which is the subject of the sentence.

    Binding matters as much as extracting: in "how many cars are to the left of
    the parked truck" the attribute belongs to the anchor, and hanging it on
    the subject instead would count parked cars.
    """
    out: Dict[int, Tuple[str, bool]] = {}
    for start, end, status, negated in statuses:
        after = [i for i, (pos, _) in enumerate(cats) if pos >= end]
        if after:
            index = after[0]
        else:
            before = [i for i, (pos, _) in enumerate(cats) if pos < start]
            if not before:
                continue
            index = before[-1]
        out.setdefault(index, (status, negated))
    return out


#: Compiled once, longest phrase first, with word boundaries so that "back"
#: does not match inside "backpack" and "left" does not match inside "cleft".
_DIRECTION_REGEXES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(phrase) + r"\b"), canon)
    for phrase, canon in _DIRECTION_PHRASES
]


def _find_direction(text: str) -> Optional[Tuple[int, str]]:
    """First direction phrase, as ``(position, canonical_direction)``.

    Ties are broken in favour of the longest phrase: "front left" and "front"
    both start at the same index, and the two-word reading is the right one.
    ``_DIRECTION_PHRASES`` is sorted longest-first and the comparison below is
    strict, so the longer phrase is kept.
    """
    best: Optional[Tuple[int, str]] = None
    for pattern, canon in _DIRECTION_REGEXES:
        match = pattern.search(text)
        if match is not None and (best is None or match.start() < best[0]):
            best = (match.start(), canon)
    return best


def _mentions_ego(text: str) -> bool:
    return _EGO_RE.search(text) is not None


@dataclass(frozen=True)
class _Referent:
    """The noun phrase a preamble clause introduces, for "it" to refer to."""

    category: str
    status: Optional[str] = None
    negate_status: bool = False


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

class RuleBasedQuestionParser:
    """Templated parser for counting / existence / comparison questions.

    Parameters
    ----------
    allow_relative_regions:
        Whether to emit a region anchored on another object ("left of the
        truck"). On by default: ``dsl.resolve`` requires the anchor to match
        exactly one object and reports ``AMBIGUOUS`` otherwise, so the
        reference is resolved explicitly or not at all -- it is never picked by
        taking the first match. Turning it off is what the earlier default did,
        and it declines every one-hop question, which is most of nuScenes-QA.
    grammar:
        See :class:`Grammar`. ``EXPRESSIVE`` is the default; the other two are
        experimental controls.
    """

    def __init__(
        self,
        allow_relative_regions: bool = True,
        grammar: Grammar = Grammar.EXPRESSIVE,
    ) -> None:
        self.allow_relative_regions = allow_relative_regions
        self.grammar = Grammar(grammar)

    @property
    def expressive(self) -> bool:
        return self.grammar is Grammar.EXPRESSIVE

    # -- public ----------------------------------------------------------- #

    def parse(self, question: str) -> ParseResult:
        raw = question.strip().lower()
        referent: Optional[_Referent] = None
        body = raw

        if self.expressive:
            preamble, rest = _split_preamble(raw)
            if preamble is not None:
                referent = _parse_referent(preamble)
                body = rest

        text = _normalise(body)
        if not text:
            return ParseResult(ParseStatus.UNSUPPORTED, reason="empty question")

        # Questions no version of this grammar can express, checked first so
        # they are never half-parsed into something plausible.
        beyond = _beyond_any_grammar(text)
        if beyond:
            return ParseResult(ParseStatus.UNSUPPORTED, reason=beyond)

        # The narrow grammar has no slot for a status or a referent. Under
        # STRICT it says so; under LEGACY it drops them and answers anyway,
        # which is the behaviour the refusals were introduced to stop.
        if self.grammar is Grammar.STRICT:
            narrow = _beyond_narrow_grammar(text, raw)
            if narrow:
                return ParseResult(ParseStatus.UNSUPPORTED, reason=narrow)

        compare = _COMPARE_PATTERN.search(text)
        if compare:
            return self._parse_compare(text, compare.group(1), referent)

        if any(p.search(text) for p in _COUNT_PATTERNS):
            return self._parse_simple(text, Operation.COUNT, ["count"], referent)

        if any(p.search(text) for p in _EXISTS_PATTERNS):
            return self._parse_simple(text, Operation.EXISTS, ["exists"], referent)

        return ParseResult(
            ParseStatus.UNSUPPORTED,
            reason="no supported operation (expected a counting, existence or "
                   "comparison question)",
        )

    # -- internals -------------------------------------------------------- #

    def _parse_simple(
        self,
        text: str,
        op: Operation,
        rules: List[str],
        referent: Optional[_Referent],
    ) -> ParseResult:
        cats = _find_categories(text)
        from_referent = False
        if not cats:
            # A preamble question whose main clause names no class of its own
            # ("there is a parked car; how many are there?") is about the
            # referent itself, attribute included.
            if referent is not None:
                cats = [(0, referent.category)]
                from_referent = True
            else:
                return ParseResult(ParseStatus.UNSUPPORTED,
                                   reason="no known category mentioned",
                                   matched_rules=rules)

        status_by_cat = (_attach_statuses(cats, _find_statuses(text))
                         if self.expressive else {})
        if from_referent and referent is not None:
            # A status in the main clause still wins: it is the narrower claim.
            status_by_cat.setdefault(0, (referent.status, referent.negate_status))

        region, region_rule, status, reason, anchor_index = self._parse_region(
            text, cats, referent)
        if status is not ParseStatus.OK:
            return ParseResult(status, reason=reason, matched_rules=rules + [region_rule])

        subject_status, subject_negate = status_by_cat.get(0, (None, False))
        anchor_status, anchor_negate = self._anchor_attributes(
            anchor_index, status_by_cat, referent)

        return ParseResult(
            ParseStatus.OK,
            Query(operation=op, category=cats[0][1], region=region,
                  status=subject_status, negate_status=subject_negate,
                  anchor_status=anchor_status,
                  anchor_negate_status=anchor_negate),
            matched_rules=rules + [region_rule],
        )

    def _parse_compare(
        self, text: str, word: str, referent: Optional[_Referent]
    ) -> ParseResult:
        cats = _find_categories(text)
        if len(cats) < 2:
            return ParseResult(ParseStatus.UNSUPPORTED,
                               reason="comparison needs two categories",
                               matched_rules=["compare"])
        status_by_cat = (_attach_statuses(cats, _find_statuses(text))
                         if self.expressive else {})

        region, region_rule, status, reason, anchor_index = self._parse_region(
            text, cats, referent)
        if status is not ParseStatus.OK:
            return ParseResult(status, reason=reason, matched_rules=["compare", region_rule])

        comparison = "fewer" if word in ("fewer", "less") else "more"
        subject_status, subject_negate = status_by_cat.get(0, (None, False))
        second_status, second_negate = status_by_cat.get(1, (None, False))
        anchor_status, anchor_negate = self._anchor_attributes(
            anchor_index, status_by_cat, referent)

        return ParseResult(
            ParseStatus.OK,
            Query(
                operation=Operation.COMPARE_COUNT,
                category=cats[0][1],
                second_category=cats[1][1],
                comparison=comparison,
                region=region,
                status=subject_status,
                negate_status=subject_negate,
                second_status=second_status,
                second_negate_status=second_negate,
                anchor_status=anchor_status,
                anchor_negate_status=anchor_negate,
            ),
            matched_rules=["compare", region_rule],
        )

    @staticmethod
    def _anchor_attributes(
        anchor_index: Optional[int],
        status_by_cat: Dict[int, Tuple[str, bool]],
        referent: Optional[_Referent],
    ) -> Tuple[Optional[str], bool]:
        """The status constraining the anchor, from the clause that names it."""
        if anchor_index is None:
            return None, False
        if anchor_index == _ANCHOR_IS_REFERENT:
            if referent is None:
                return None, False
            return referent.status, referent.negate_status
        return status_by_cat.get(anchor_index, (None, False))

    def _parse_region(
        self,
        text: str,
        cats: List[Tuple[int, str]],
        referent: Optional[_Referent],
    ) -> Tuple[Region, str, ParseStatus, str, Optional[int]]:
        radius = _RADIUS_PATTERN.search(text)
        if radius:
            return (Region(kind="radius", radius=float(radius.group(1))),
                    "region:radius", ParseStatus.OK, "", None)

        direction = _find_direction(text)
        if direction is not None:
            dir_pos, dir_name = direction
            kind, anchor_index = self._resolve_anchor(text, dir_pos, cats)

            if kind == "pronoun":
                if self.grammar is Grammar.LEGACY:
                    # The original parser had no pronoun branch: nothing
                    # followed the direction word, so "to the back of it" was
                    # read as "to the back of the ego vehicle" and answered
                    # about the wrong origin. The control has to make that
                    # mistake, or it is not the control.
                    return (Region(kind="direction", direction=dir_name),
                            "region:direction_ego", ParseStatus.OK, "", None)
                # "to the back of it": the anchor is whatever the preamble
                # introduced. Without a preamble there is no antecedent, and
                # guessing one would answer a different question.
                if referent is None:
                    return (ANY_REGION, "region:dangling_pronoun",
                            ParseStatus.UNSUPPORTED,
                            "question refers back to an object with a pronoun "
                            "and no antecedent was introduced", None)
                anchor_category = referent.category
                anchor_index = _ANCHOR_IS_REFERENT
            elif kind == "category":
                anchor_category = cats[anchor_index][1]
            else:
                return (Region(kind="direction", direction=dir_name),
                        "region:direction_ego", ParseStatus.OK, "", None)

            if not self.allow_relative_regions:
                return (ANY_REGION, "region:relative_declined", ParseStatus.UNSUPPORTED,
                        f"question is anchored on another object "
                        f"({anchor_category}); reference resolution is out of "
                        f"scope", None)
            return (Region(kind="relative", direction=dir_name,
                           reference_category=anchor_category, frame="ego"),
                    "region:relative", ParseStatus.OK, "", anchor_index)

        if _NEAR_PATTERN.search(text):
            return Region(kind="near"), "region:near", ParseStatus.OK, "", None
        if _FAR_PATTERN.search(text):
            return Region(kind="far"), "region:far", ParseStatus.OK, "", None

        return ANY_REGION, "region:any", ParseStatus.OK, "", None

    @staticmethod
    def _resolve_anchor(
        text: str, dir_pos: int, cats: List[Tuple[int, str]]
    ) -> Tuple[str, Optional[int]]:
        """What the direction is measured from.

        Returns ``("ego", None)``, ``("category", index_into_cats)`` or
        ``("pronoun", None)``. The anchor is whatever noun phrase follows the
        direction word; the ego vehicle is checked first so that "to the front
        of the ego vehicle" is never read as a one-hop question.
        """
        tail = text[dir_pos:]
        if _mentions_ego(tail):
            return "ego", None
        following = [i for i, (pos, _) in enumerate(cats) if pos > dir_pos]
        if following:
            return "category", following[0]
        if _PRONOUN_RE.search(tail):
            return "pronoun", None
        return "ego", None


#: Sentinel for ``_parse_region``: the anchor is the preamble's referent, whose
#: status comes from the preamble clause rather than from ``status_by_cat``.
_ANCHOR_IS_REFERENT = -1


# --------------------------------------------------------------------------- #
# Clause splitting
# --------------------------------------------------------------------------- #

def _split_preamble(raw: str) -> Tuple[Optional[str], str]:
    """Split "there is a trailer; how many cones ..." into its two clauses.

    Returns ``(preamble, question)``, or ``(None, raw)`` when the question has
    no preamble. The split is refused unless the second clause is recognisable
    as a question on its own, so an ordinary comma inside a single clause does
    not truncate it.
    """
    match = _PREAMBLE_RE.match(raw)
    if match is None:
        return None, raw
    preamble, rest = match.group(1), match.group(2)
    tail = _normalise(rest)
    recognisable = (_COMPARE_PATTERN.search(tail)
                    or any(p.search(tail) for p in _COUNT_PATTERNS)
                    or any(p.search(tail) for p in _EXISTS_PATTERNS))
    if not recognisable:
        return None, raw
    return preamble, rest


def _parse_referent(preamble: str) -> Optional[_Referent]:
    """The noun phrase the preamble introduces, with its status if it has one."""
    text = _normalise(preamble)
    cats = _find_categories(text)
    if not cats:
        return None
    statuses = _attach_statuses(cats, _find_statuses(text))
    status, negate = statuses.get(0, (None, False))
    return _Referent(category=cats[0][1], status=status, negate_status=negate)


# --------------------------------------------------------------------------- #
# What stays out of the grammar
# --------------------------------------------------------------------------- #

def _beyond_any_grammar(text: str) -> Optional[str]:
    """Questions none of the three grammars can express, or ``None``.

    These are not slots the parser is missing: the target language has no
    operation that binds one object's attribute as a filter over a set, and
    nuScenes carries no colour annotation at all. Answering them would require
    inventing data, so they are refused in every mode.
    """
    match = _ATTRIBUTE_COMPARISON_RE.search(text)
    if match:
        attribute = match.group(1)
        if attribute in ("colour", "color"):
            return "question asks about colour, which nuScenes does not annotate"
        return ("question compares attributes between objects; the executor "
                "cannot bind one object's status as a filter over a set")
    return None


def _beyond_narrow_grammar(text: str, raw: str) -> Optional[str]:
    """Reason the *narrow* query shape cannot hold this question.

    Only consulted under ``Grammar.STRICT``. Everything listed here is now
    expressible -- it is kept so that the coverage the narrow shape cost can be
    measured rather than asserted.
    """
    if _find_statuses(text):
        return ("question constrains object status (moving / parked / "
                "stopped / standing / rider); the narrow grammar has no "
                "status filter and would silently count every object of the "
                "class")

    if _PREAMBLE_RE.match(raw):
        return ("question introduces a referent in a preamble clause and "
                "then asks about a different class; the narrow grammar would "
                "take the preamble's class as the target")

    if _PRONOUN_RE.search(text):
        return "question refers back to an object with a pronoun"

    return None


DEFAULT_PARSER = RuleBasedQuestionParser()

#: The two controls, built once so experiments cannot disagree about what they
#: mean.
STRICT_PARSER = RuleBasedQuestionParser(grammar=Grammar.STRICT)
LEGACY_PARSER = RuleBasedQuestionParser(grammar=Grammar.LEGACY)


def parse_question(question: str, parser: Optional[RuleBasedQuestionParser] = None) -> ParseResult:
    """Convenience wrapper around :class:`RuleBasedQuestionParser`."""
    return (parser or DEFAULT_PARSER).parse(question)


def coverage(questions: List[str], parser: Optional[RuleBasedQuestionParser] = None) -> Dict[str, float]:
    """Fraction of questions the grammar accepts, broken down by failure reason.

    Reported alongside accuracy in every experiment: a parser that answers 30%
    of questions perfectly is not comparable to one that answers all of them.
    """
    parser = parser or DEFAULT_PARSER
    total = len(questions) or 1
    reasons: Dict[str, int] = {}
    accepted = 0
    for q in questions:
        res = parser.parse(q)
        if res.ok:
            accepted += 1
        else:
            reasons[res.reason] = reasons.get(res.reason, 0) + 1
    out = {"coverage": accepted / total, "n": float(len(questions))}
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        out[f"declined::{reason}"] = count / total
    return out
