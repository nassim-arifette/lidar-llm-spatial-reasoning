"""The frozen example set both systems are measured on.

The experiment compares two systems that receive the *same* question and the
*same* object list:

**A, direct**  question + objects -> Qwen -> answer.
**B, executed** question + the language description -> the same Qwen ->
program -> executor on the same objects -> answer.

For that comparison to mean anything, the example set has to be fixed before
either system runs, and fixed for reasons that have nothing to do with how
either one performs. This module is that commitment.

Three rules it enforces, each of which is a way the comparison could be
rigged:

**Selection is by what the question asks, never by what a system produces.**
Families are assigned from the question's surface structure -- does it name an
attribute, a direction, a reference object -- using the lexicons directly, not
by calling the parser and keeping what parses. Choosing the questions the rule
grammar happens to handle would quietly select the examples that favour B.

**Development and evaluation never share a scene.** The split is by
``scene_token``, not by sample: consecutive samples of one drive show the same
vehicles, so splitting by sample would put near-identical object lists on both
sides. A scene used to debug the code or the prompts stays in development; it
does not become a held-out test by being renamed.

**The reference answer is for the evaluator.** :meth:`ManifestItem.payload`
returns what an inference function is allowed to see, and it does not include
the answer. The only way to score is to ask the evaluator.

The first scope is deliberately narrow -- the four families below are what the
frozen DSL can represent. Everything else is recorded as out of scope with the
reason, not deleted: "the grammar cannot say this" is a measurement, and the
count of what was excluded belongs in the paper next to the accuracy.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .objects import Scene
from .question_parser import (
    _CATEGORY_PATTERN,
    _DIRECTION_REGEXES,
    _PREAMBLE_RE,
    _PRONOUN_RE,
    _STATUS_PATTERN,
    _normalise,
    _mentions_ego,
)

#: Version of the selection rules. A change here invalidates a manifest, so it
#: is recorded in the file and checked on load.
MANIFEST_VERSION = "1"

#: The four question families of the first comparison, in increasing
#: structural demand. A question that has several of these elements is filed
#: under the most demanding one, and the individual flags are kept so a report
#: can also group by element.
FAMILIES: Tuple[str, ...] = (
    "count_exist_category",      # "How many cars are there?"
    "category_status",           # "How many parked cars are there?"
    "category_ego_region",       # "Are there pedestrians in front of the ego vehicle?"
    "category_anchor_relation",  # "How many cars are behind the truck?"
)

#: Templates the frozen DSL can answer at all. ``object``, ``status`` and
#: ``comparison`` need operations it does not have; they are the second step.
IN_SCOPE_TEMPLATES: Tuple[str, ...] = ("count", "exist")

#: Attribute comparisons between objects: "the same status as the bicycle".
#: Representing these needs a filter bound to another object's attribute,
#: which the frozen DSL has no way to express.
_SAME_ATTRIBUTE_RE = re.compile(r"\bsame (status|colour|color|attribute)\b")


# --------------------------------------------------------------------------- #
# One example
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ManifestItem:
    """One question, pinned to one sample, with its reference answer."""

    question_id: str
    question: str
    sample_token: str
    scene_token: str
    split: str                  # "dev" | "eval"
    family: str
    template: str
    num_hop: int
    answer: str                 # the reference. Not part of `payload`.
    has_status: bool = False
    has_direction: bool = False
    has_anchor: bool = False

    def payload(self) -> Dict[str, object]:
        """Everything an inference function may see. The answer is not here.

        Both systems call this. Passing the item itself would make a leak a
        one-character mistake; this makes it a deliberate one.
        """
        return {"question_id": self.question_id,
                "question": self.question,
                "sample_token": self.sample_token}

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Excluded:
    """A question that was considered and left out, with the reason."""

    question: str
    sample_token: str
    template: str
    reason: str


# --------------------------------------------------------------------------- #
# Reading the question, without reading the parser's mind
# --------------------------------------------------------------------------- #

def _question_id(sample_token: str, question: str) -> str:
    digest = hashlib.sha1(f"{sample_token}|{question}".encode("utf-8"))
    return "q_" + digest.hexdigest()[:12]


def _directions_in(text: str) -> List[Tuple[int, str]]:
    """Every distinct direction phrase, as ``(position, canonical)``.

    Used to spot questions that compose two relations ("in front of me and
    behind the truck"), which are out of scope for the first comparison.
    """
    found: Dict[int, str] = {}
    for pattern, canon in _DIRECTION_REGEXES:
        for match in pattern.finditer(text):
            found.setdefault(match.start(), canon)
    # A longer phrase and the shorter one inside it start at different
    # offsets ("back right" at i, "right" at i+5), so collapse overlaps by
    # keeping the earliest start of each run.
    starts = sorted(found)
    kept: List[Tuple[int, str]] = []
    for start in starts:
        if kept and start - kept[-1][0] < 12:
            continue
        kept.append((start, found[start]))
    return kept


def classify(question: str, template: str) -> Tuple[Optional[str], str, Dict[str, bool]]:
    """Return ``(family, reason, flags)``; ``family`` is ``None`` when excluded.

    Everything here reads the question. Nothing here runs the parser: a family
    is a statement about what is being asked, so that the examples are not
    chosen by whether a particular implementation copes with them.
    """
    raw = question.strip().lower()
    text = _normalise(raw)
    flags = {"has_status": False, "has_direction": False, "has_anchor": False}

    if not raw.endswith("?"):
        # A handful of nuScenes-QA questions are cut off mid-word ("Is the
        # status of the car to the back of the park"). Neither system can be
        # blamed for those and including them would depress both scores.
        return None, "truncated_question_text", flags

    if template not in IN_SCOPE_TEMPLATES:
        return None, f"template_out_of_scope:{template}", flags

    if _SAME_ATTRIBUTE_RE.search(text):
        return None, "attribute_bound_to_another_object", flags

    categories = [m.start() for m in _CATEGORY_PATTERN.finditer(text)]
    if not categories:
        return None, "no_known_category", flags

    flags["has_status"] = _STATUS_PATTERN.search(text) is not None

    directions = _directions_in(text)
    if len(directions) > 1:
        return None, "composes_several_relations", flags
    flags["has_direction"] = bool(directions)

    if directions:
        position = directions[0][0]
        tail = text[position:]
        anchored_on_object = (
            not _mentions_ego(tail)
            and (any(c > position for c in categories)
                 or (_PRONOUN_RE.search(tail) is not None
                     and _PREAMBLE_RE.match(raw) is not None))
        )
        flags["has_anchor"] = anchored_on_object
        if anchored_on_object:
            return "category_anchor_relation", "", flags
        if _PRONOUN_RE.search(tail) is not None:
            # "to the back of it" with nothing introducing "it".
            return None, "pronoun_without_antecedent", flags
        return "category_ego_region", "", flags

    if flags["has_status"]:
        return "category_status", "", flags
    return "count_exist_category", "", flags


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

@dataclass
class Manifest:
    """The frozen example set, plus the record of what was left out."""

    version: str
    items: List[ManifestItem] = field(default_factory=list)
    excluded: List[Excluded] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)

    # -- access ---------------------------------------------------------- #

    def split(self, name: str) -> List[ManifestItem]:
        return [it for it in self.items if it.split == name]

    def by_id(self) -> Dict[str, ManifestItem]:
        return {it.question_id: it for it in self.items}

    def counts(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for name in ("dev", "eval"):
            rows = self.split(name)
            table = collections.Counter(it.family for it in rows)
            table["TOTAL"] = len(rows)
            table["scenes"] = len({it.scene_token for it in rows})
            out[name] = dict(table)
        return out

    def exclusion_reasons(self) -> Dict[str, int]:
        return dict(collections.Counter(e.reason for e in self.excluded).most_common())

    # -- persistence ----------------------------------------------------- #

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = {
            "version": self.version,
            "config": self.config,
            "counts": self.counts(),
            "exclusion_reasons": self.exclusion_reasons(),
            "items": [it.to_dict() for it in self.items],
            "excluded": [asdict(e) for e in self.excluded],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Manifest":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("version") != MANIFEST_VERSION:
            raise ValueError(
                f"manifest {path} was built by selection rules version "
                f"{payload.get('version')!r}, this is {MANIFEST_VERSION!r}. "
                f"Rebuild it rather than comparing across rule sets.")
        return cls(
            version=payload["version"],
            items=[ManifestItem(**row) for row in payload["items"]],
            excluded=[Excluded(**row) for row in payload["excluded"]],
            config=payload.get("config", {}),
        )


def build_manifest(
    qa_rows: Iterable[Dict[str, object]],
    scenes: Dict[str, Scene],
    n_dev: int = 100,
    n_eval: int = 200,
    seed: int = 0,
    min_eval_scenes: int = 3,
    config: Optional[Dict[str, object]] = None,
) -> Manifest:
    """Select the examples, splitting development and evaluation by scene.

    ``n_dev`` and ``n_eval`` are targets. A family with fewer in-scope
    questions than its share contributes everything it has and the shortfall is
    reported rather than made up from another family, because topping up would
    change the mix the numbers are averaged over.

    ``min_eval_scenes`` matters more than it looks on nuScenes v1.0-mini,
    which has ten drives in total and about a hundred and sixty in-scope
    questions each. Two scenes already reach a two-hundred-question target, and
    an evaluation resting on two drives measures those two drives: answers
    within a scene are correlated, so the effective sample size is nearer the
    number of scenes than the number of questions. Results are reported per
    scene for the same reason.
    """
    rng = random.Random(seed)
    kept: List[Tuple[str, Dict[str, object], Dict[str, bool]]] = []
    excluded: List[Excluded] = []

    for row in qa_rows:
        question = str(row.get("question", ""))
        template = str(row.get("template_type") or row.get("template") or "")
        sample_token = str(row.get("sample_token") or row.get("token") or "")

        scene = scenes.get(sample_token)
        if scene is None:
            excluded.append(Excluded(question, sample_token, template,
                                     "sample_not_in_object_lists"))
            continue

        family, reason, flags = classify(question, template)
        if family is None:
            excluded.append(Excluded(question, sample_token, template, reason))
            continue
        kept.append((family, row, flags))

    # Split by scene. A drive's consecutive samples show the same vehicles, so
    # splitting by sample would put near-duplicate object lists on both sides
    # and make the evaluation set a paraphrase of the development set.
    scene_of: Dict[str, str] = {}
    for _, row, _ in kept:
        token = str(row.get("sample_token") or row.get("token"))
        scene_of[token] = scenes[token].scene_token or token
    scene_tokens = sorted({scene_of[str(r.get("sample_token") or r.get("token"))]
                           for _, r, _ in kept})
    rng.shuffle(scene_tokens)

    # Enough scenes for the evaluation target, the rest for development.
    per_scene: Dict[str, int] = collections.Counter(
        scene_of[str(r.get("sample_token") or r.get("token"))] for _, r, _ in kept)
    eval_scenes: set = set()
    running = 0
    for token in scene_tokens:
        if running >= n_eval and len(eval_scenes) >= min_eval_scenes:
            break
        eval_scenes.add(token)
        running += per_scene[token]
    dev_scenes = set(scene_tokens) - eval_scenes
    if not dev_scenes:
        raise RuntimeError(
            f"every scene went to evaluation ({len(eval_scenes)} scenes, "
            f"{running} questions); lower n_eval or widen the scope")

    by_split: Dict[str, List[Tuple[str, Dict[str, object], Dict[str, bool]]]] = {
        "dev": [], "eval": []}
    for family, row, flags in kept:
        token = scene_of[str(row.get("sample_token") or row.get("token"))]
        by_split["eval" if token in eval_scenes else "dev"].append(
            (family, row, flags))

    items: List[ManifestItem] = []
    for split_name, target in (("dev", n_dev), ("eval", n_eval)):
        pool = by_split[split_name]
        buckets: Dict[str, List] = {f: [] for f in FAMILIES}
        for entry in pool:
            buckets[entry[0]].append(entry)
        for bucket in buckets.values():
            # Sort before shuffling: dict iteration order over the input file
            # is stable, but sorting makes the seed the only thing that
            # decides, so the manifest is reproducible from the seed alone.
            bucket.sort(key=lambda e: _question_id(
                str(e[1].get("sample_token") or e[1].get("token")),
                str(e[1].get("question"))))
            rng.shuffle(bucket)

        share = max(target // len(FAMILIES), 1)
        chosen: List = []
        for family in FAMILIES:
            chosen.extend(buckets[family][:share])
        # Any shortfall is made up from the families that still have examples,
        # in a fixed order, so the mix is reported rather than silently uneven.
        if len(chosen) < target:
            leftovers = [e for family in FAMILIES for e in buckets[family][share:]]
            rng.shuffle(leftovers)
            chosen.extend(leftovers[:target - len(chosen)])

        for family, row, flags in chosen:
            sample_token = str(row.get("sample_token") or row.get("token"))
            question = str(row.get("question"))
            items.append(ManifestItem(
                question_id=_question_id(sample_token, question),
                question=question,
                sample_token=sample_token,
                scene_token=scene_of[sample_token],
                split=split_name,
                family=family,
                template=str(row.get("template_type") or row.get("template") or ""),
                num_hop=int(row.get("num_hop") or 0),
                answer=str(row.get("answer")).strip().lower(),
                **flags,
            ))

    manifest = Manifest(version=MANIFEST_VERSION, items=items, excluded=excluded,
                        config=dict(config or {}))
    overlap = ({it.scene_token for it in manifest.split("dev")}
               & {it.scene_token for it in manifest.split("eval")})
    if overlap:
        raise RuntimeError(f"dev and eval share scenes: {sorted(overlap)}")
    return manifest


# --------------------------------------------------------------------------- #
# The object list both systems see
# --------------------------------------------------------------------------- #

def scene_fingerprint(scene: Scene) -> str:
    """A hash of the object list, so a result can be tied to what it saw.

    Recorded per question. If a run is repeated after the object filter or the
    adapter changes, the fingerprints differ and the two runs are not
    comparable -- which is better found by a mismatched hash than by an
    unexplained change in accuracy.
    """
    rows = sorted(
        f"{o.category}|{o.status}|{o.center[0]:.3f}|{o.center[1]:.3f}|"
        f"{o.center[2]:.3f}|{o.yaw:.4f}"
        for o in scene.objects)
    digest = hashlib.sha1("\n".join(rows).encode("utf-8"))
    return digest.hexdigest()[:16]


def describe_objects(
    scene: Scene, precision: int = 1, include_status: bool = True
) -> List[Dict[str, object]]:
    """The object list, in the one form both systems receive.

    Identical rounding for both. Giving the executor full precision while
    handing the direct model one decimal would make the comparison a test of
    how much resolution each was given, not of what each does with it.
    """
    rows: List[Dict[str, object]] = []
    for index, obj in enumerate(scene.objects):
        row: Dict[str, object] = {
            "id": _object_id(index),
            "category": obj.category,
            "x": round(float(obj.center[0]), precision),
            "y": round(float(obj.center[1]), precision),
        }
        if include_status:
            row["status"] = obj.status if obj.status is not None else "unknown"
        rows.append(row)
    return rows


def _object_id(index: int) -> str:
    """Short stable labels: a, b, ... z, aa, ab, ..."""
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return letters
