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


@dataclass(frozen=True)
class Query:
    """A structured, executable description of a question."""

    operation: Operation
    category: str                       # canonical class or group name
    region: Region = ANY_REGION
    second_category: Optional[str] = None   # for COMPARE_COUNT
    comparison: Optional[str] = None        # "more" | "fewer"

    def describe(self) -> str:
        if self.operation is Operation.COMPARE_COUNT:
            return (f"compare: are there {self.comparison} {self.category} "
                    f"than {self.second_category} {self.region.describe()}")
        verb = "count" if self.operation is Operation.COUNT else "exists"
        return f"{verb}: {self.category} {self.region.describe()}"


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

#: Attribute words the grammar cannot express. Their presence means the
#: question asks for a subset the executor would not filter, so answering
#: would over-count the whole class.
_STATUS_WORDS_RE = re.compile(
    r"\b(moving|parked|stopped|standing|sitting|lying|"
    r"with rider|without rider|not standing|not moving|"
    r"not parked|not stopped)\b")

#: "There is a <something>; <question about something else>"
_PREAMBLE_RE = re.compile(r"^(there is|there are)\b.*?[;,]")

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
    return any(w in text for w in _EGO_WORDS)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #

class RuleBasedQuestionParser:
    """Templated parser for counting / existence / comparison questions.

    Parameters
    ----------
    allow_relative_regions:
        Whether to emit ``Region(kind="relative")`` when the question is
        anchored on another object ("left of the truck"). Those questions carry
        a reference-resolution problem that is separate from spatial reasoning,
        so they are off by default and reported as unsupported.
    """

    def __init__(self, allow_relative_regions: bool = False) -> None:
        self.allow_relative_regions = allow_relative_regions

    # -- public ----------------------------------------------------------- #

    def parse(self, question: str) -> ParseResult:
        text = _normalise(question)
        if not text:
            return ParseResult(ParseStatus.UNSUPPORTED, reason="empty question")

        # Refuse what the grammar cannot express, BEFORE extracting anything.
        # Without this the parser answered a different question: "how many
        # standing pedestrians are there" became "how many pedestrians are
        # there" (30 against a gold answer of 8), and "what number of stopped
        # things are there" became "how many objects are there" (71 against
        # 10). A refusal costs coverage; a silent reinterpretation costs
        # correctness and looks like a reasoning failure.
        inexpressible = self._why_inexpressible(text)
        if inexpressible:
            return ParseResult(ParseStatus.UNSUPPORTED, reason=inexpressible)

        compare = _COMPARE_PATTERN.search(text)
        if compare:
            return self._parse_compare(text, compare.group(1))

        if any(p.search(text) for p in _COUNT_PATTERNS):
            return self._parse_simple(text, Operation.COUNT, ["count"])

        if any(p.search(text) for p in _EXISTS_PATTERNS):
            return self._parse_simple(text, Operation.EXISTS, ["exists"])

        return ParseResult(
            ParseStatus.UNSUPPORTED,
            reason="no supported operation (expected a counting, existence or "
                   "comparison question)",
        )

    # -- internals -------------------------------------------------------- #

    @staticmethod
    def _why_inexpressible(text: str) -> Optional[str]:
        """Reason this question is outside the grammar, or ``None``.

        Everything listed here used to be parsed anyway, with the unsupported
        part dropped, which produced an answer to a question nobody asked.
        """
        if _STATUS_WORDS_RE.search(text):
            return ("question constrains object status (moving / parked / "
                    "stopped / standing / rider); the grammar has no status "
                    "filter and would silently count every object of the class")

        if "same status" in text or "same colour" in text or "same color" in text:
            return "question compares attributes between objects"

        # "There is a moving truck; how many pedestrians are to the back of it?"
        # The grammar takes the first category it sees, which is the one
        # introduced by the preamble, not the one being counted.
        if _PREAMBLE_RE.search(text):
            return ("question introduces a referent in a preamble clause and "
                    "then asks about a different class; the grammar would take "
                    "the preamble's class as the target")

        if " it?" in text or text.endswith(" it") or " of it" in text:
            return "question refers back to an object with a pronoun"

        return None

    def _parse_simple(self, text: str, op: Operation, rules: List[str]) -> ParseResult:
        cats = _find_categories(text)
        if not cats:
            return ParseResult(ParseStatus.UNSUPPORTED, reason="no known category mentioned",
                               matched_rules=rules)
        category = cats[0][1]

        region, region_rule, status, reason = self._parse_region(text, cats)
        if status is not ParseStatus.OK:
            return ParseResult(status, reason=reason, matched_rules=rules + [region_rule])

        return ParseResult(
            ParseStatus.OK,
            Query(operation=op, category=category, region=region),
            matched_rules=rules + [region_rule],
        )

    def _parse_compare(self, text: str, word: str) -> ParseResult:
        cats = _find_categories(text)
        if len(cats) < 2:
            return ParseResult(ParseStatus.UNSUPPORTED,
                               reason="comparison needs two categories",
                               matched_rules=["compare"])
        region, region_rule, status, reason = self._parse_region(text, cats)
        if status is not ParseStatus.OK:
            return ParseResult(status, reason=reason, matched_rules=["compare", region_rule])

        comparison = "fewer" if word in ("fewer", "less") else "more"
        return ParseResult(
            ParseStatus.OK,
            Query(
                operation=Operation.COMPARE_COUNT,
                category=cats[0][1],
                second_category=cats[1][1],
                comparison=comparison,
                region=region,
            ),
            matched_rules=["compare", region_rule],
        )

    def _parse_region(
        self, text: str, cats: List[Tuple[int, str]]
    ) -> Tuple[Region, str, ParseStatus, str]:
        radius = _RADIUS_PATTERN.search(text)
        if radius:
            return (Region(kind="radius", radius=float(radius.group(1))),
                    "region:radius", ParseStatus.OK, "")

        direction = _find_direction(text)
        if direction is not None:
            dir_pos, dir_name = direction
            # Is the direction anchored on the ego vehicle or on another object?
            anchor = self._resolve_anchor(text, dir_pos, cats)
            if anchor is None:
                return (Region(kind="direction", direction=dir_name),
                        "region:direction_ego", ParseStatus.OK, "")
            if not self.allow_relative_regions:
                return (ANY_REGION, "region:relative_declined", ParseStatus.UNSUPPORTED,
                        f"question is anchored on another object ({anchor}); "
                        f"reference resolution is out of scope")
            return (Region(kind="relative", direction=dir_name,
                           reference_category=anchor, frame="ego"),
                    "region:relative", ParseStatus.OK, "")

        if _NEAR_PATTERN.search(text):
            return Region(kind="near"), "region:near", ParseStatus.OK, ""
        if _FAR_PATTERN.search(text):
            return Region(kind="far"), "region:far", ParseStatus.OK, ""

        return ANY_REGION, "region:any", ParseStatus.OK, ""

    @staticmethod
    def _resolve_anchor(
        text: str, dir_pos: int, cats: List[Tuple[int, str]]
    ) -> Optional[str]:
        """Return the category the direction is relative to, or ``None`` for ego.

        The anchor is whatever noun phrase follows the direction word. If that
        is the ego vehicle (or nothing), the region is ego-centric.
        """
        tail = text[dir_pos:]
        if _mentions_ego(tail):
            return None
        following = [c for pos, c in cats if pos > dir_pos]
        return following[0] if following else None


DEFAULT_PARSER = RuleBasedQuestionParser()


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
