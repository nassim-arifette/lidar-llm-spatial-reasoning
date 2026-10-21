"""Question -> program, with a small frozen language model.

The rule-based parser of ``question_parser.py`` covers 31% of real
nuScenes-QA questions. That is a property of the grammar, not of the approach:
the questions it declines are phrased in ways a hand-written grammar cannot
anticipate, not computed in ways the executor cannot handle.

This module replaces the grammar with a frozen instruct model that emits a
program in the DSL of ``dsl.py``. The model never sees the scene and never
produces an answer -- it chooses *what to compute*, the executor computes it.
That division is the one KLDrive uses, and it is what this experiment is
testing.

Nothing is fine-tuned. The model is loaded once, used with greedy decoding for
reproducibility, and its output is validated against the DSL: anything that
does not parse into a well-formed :class:`~spatial_reasoning.dsl.Query` is a
refusal, not a guess. Parses are cached by question text, because nuScenes-QA
reuses a small number of templates across thousands of samples.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .dsl import Op, Query, STATUSES, Selector
from .objects import CANONICAL_CATEGORIES
from .regions import DIRECTIONS, Region

# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You translate questions about a 3D driving scene into a JSON program.
You never answer the question. You only describe what must be computed.

Output EXACTLY one JSON object, no prose, no markdown fence.

Schema:
{
  "op": one of ["count","exists","identify","status","same_status","compare_count"],
  "target": SELECTOR,
  "second": SELECTOR or null,       // only for same_status and compare_count
  "comparison": "more"|"fewer"|null // only for compare_count
}

SELECTOR = {
  "category": one of [CATEGORIES] or "object",
  "status": one of [STATUSES] or null,
  "negate_status": true or false,
  "direction": one of [DIRECTIONS] or null,
  "anchor": SELECTOR or null
}

Rules:
- "thing", "things", "object" -> category "object".
- "what is it" / "what is the ..." -> op "identify".
- "in what status" / "what is the status of" -> op "status".
- "do X and Y have the same status" -> op "same_status" with target X and second Y.
- "are there more A than B" -> op "compare_count" with comparison "more".
- "how many" / "what number of" -> op "count".
- "are there any" / "is there a" / "are ... visible" -> op "exists".
- A direction is relative to the ego vehicle unless the question says it is
  relative to another object ("to the left of the truck", "to the back of it").
  In that case put that object in "anchor" and keep the direction on the
  selector that carries it.
- "it" refers to the object introduced earlier in the same question; make that
  object the "anchor".
- "not standing", "not moving" -> set the status and negate_status true.
- If the question cannot be expressed in this schema, output {"op": null}.
"""

FEW_SHOT = [
    ("Are there any trailers?",
     {"op": "exists",
      "target": {"category": "trailer", "status": None, "negate_status": False,
                 "direction": None, "anchor": None},
      "second": None, "comparison": None}),
    ("How many cars are to the front left of the ego vehicle?",
     {"op": "count",
      "target": {"category": "car", "status": None, "negate_status": False,
                 "direction": "front_left", "anchor": None},
      "second": None, "comparison": None}),
    ("There is a parked thing; what is it?",
     {"op": "identify",
      "target": {"category": "object", "status": "parked", "negate_status": False,
                 "direction": None, "anchor": None},
      "second": None, "comparison": None}),
    ("The construction vehicle is in what status?",
     {"op": "status",
      "target": {"category": "construction_vehicle", "status": None,
                 "negate_status": False, "direction": None, "anchor": None},
      "second": None, "comparison": None}),
    ("There is a trailer; what number of traffic cones are to the back of it?",
     {"op": "count",
      "target": {"category": "traffic_cone", "status": None, "negate_status": False,
                 "direction": "back",
                 "anchor": {"category": "trailer", "status": None,
                            "negate_status": False, "direction": None, "anchor": None}},
      "second": None, "comparison": None}),
    ("Do the trailer and the truck have the same status?",
     {"op": "same_status",
      "target": {"category": "trailer", "status": None, "negate_status": False,
                 "direction": None, "anchor": None},
      "second": {"category": "truck", "status": None, "negate_status": False,
                 "direction": None, "anchor": None},
      "comparison": None}),
    ("Are there more cars than trucks?",
     {"op": "compare_count",
      "target": {"category": "car", "status": None, "negate_status": False,
                 "direction": None, "anchor": None},
      "second": {"category": "truck", "status": None, "negate_status": False,
                 "direction": None, "anchor": None},
      "comparison": "more"}),
    ("Are there any other pedestrians of the same colour as the bus?",
     {"op": None}),
]


def build_system_prompt(directions: Sequence[str] = DIRECTIONS) -> str:
    """Render the prompt for a given set of legal directions.

    The direction list is part of the prompt, so changing the sector layout
    changes the prompt and therefore the parser. Two caches built under
    different layouts are not interchangeable, which is why
    :class:`QwenQuestionParser` fingerprints the prompt.
    """
    return (SYSTEM_PROMPT
            .replace("[CATEGORIES]", json.dumps(list(CANONICAL_CATEGORIES)))
            .replace("[STATUSES]", json.dumps(list(STATUSES)))
            .replace("[DIRECTIONS]", json.dumps(list(directions))))


def build_messages(question: str, directions: Sequence[str] = DIRECTIONS) -> List[dict]:
    messages = [{"role": "system", "content": build_system_prompt(directions)}]
    for q, program in FEW_SHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": json.dumps(program)})
    messages.append({"role": "user", "content": question})
    return messages


# --------------------------------------------------------------------------- #
# JSON -> Query
# --------------------------------------------------------------------------- #

_VALID_CATEGORIES = set(CANONICAL_CATEGORIES) | {"object", "thing", "vehicle"}
_VALID_DIRECTIONS = frozenset(DIRECTIONS)
_VALID_STATUSES = set(STATUSES)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

#: Exactly the fields the schema defines. Anything else is malformed output,
#: not a field to ignore.
_SELECTOR_FIELDS = frozenset(
    {"category", "status", "negate_status", "direction", "anchor"})
_PROGRAM_FIELDS = frozenset({"op", "target", "second", "comparison"})


class InvalidProgram(ValueError):
    """The model produced something that is not a program in the DSL."""


def _selector_from_json(
    blob: Optional[dict],
    depth: int = 0,
    valid_directions: Optional[frozenset] = None,
) -> Optional[Selector]:
    """Validate one selector.

    ``valid_directions`` defaults to every direction any layout defines, but
    callers that know which layout is active must pass its own set: the prompt,
    the validator and the executor have to agree on the legal values, or the
    validator will wave through a program the executor must then refuse.
    """
    directions = _VALID_DIRECTIONS if valid_directions is None else valid_directions
    if blob is None:
        return None
    if not isinstance(blob, dict):
        raise InvalidProgram("selector is not an object")
    if depth > 2:
        raise InvalidProgram("anchor nested too deep")

    unknown = set(blob) - _SELECTOR_FIELDS
    if unknown:
        # Ignoring an unknown field drops a constraint the model meant to
        # express -- "radius": 10 silently became "anywhere". Rejecting makes
        # that a counted parse failure instead of a wrong answer.
        raise InvalidProgram(f"unknown selector field(s): {sorted(unknown)}")

    raw_category = blob.get("category")
    if raw_category is None:
        category = "object"
    elif isinstance(raw_category, str):
        category = raw_category.strip().lower().replace(" ", "_")
    else:
        # `str(x or "object")` turned 0, false, [] and {} into "object", so a
        # program with a malformed category counted every object in the scene
        # instead of being rejected. Absence is a documented default; a present
        # value of the wrong type is broken output.
        raise InvalidProgram(
            f"category must be a string or null, got {type(raw_category).__name__}")
    if category not in _VALID_CATEGORIES:
        raise InvalidProgram(f"unknown category {category!r}")

    status = blob.get("status")
    if status is not None:
        if not isinstance(status, str):
            raise InvalidProgram(f"status must be a string or null, got {type(status).__name__}")
        status = status.strip().lower()
        if status not in _VALID_STATUSES:
            raise InvalidProgram(f"unknown status {status!r}")

    negate = blob.get("negate_status", False)
    if not isinstance(negate, bool):
        # bool("false") is True. Coercing here turned a model that correctly
        # emitted the string "false" into a negated selector.
        raise InvalidProgram(
            f"negate_status must be a boolean, got {negate!r}")

    direction = blob.get("direction")
    region = Region(kind="any")
    if direction is not None:
        if not isinstance(direction, str):
            raise InvalidProgram(
                f"direction must be a string or null, got {type(direction).__name__}")
        direction = direction.strip().lower().replace(" ", "_")
        if direction not in directions:
            raise InvalidProgram(f"unknown direction {direction!r}")
        region = Region(kind="direction", direction=direction)

    return Selector(
        category=category,
        status=status,
        negate_status=negate,
        region=region,
        anchor=_selector_from_json(blob.get("anchor"), depth + 1, directions),
    )


def query_from_json(
    text: str, valid_directions: Optional[frozenset] = None
) -> Optional[Query]:
    """Validate the model's output. Returns ``None`` for an explicit refusal.

    Raises :class:`InvalidProgram` when the output is malformed, which is
    counted separately from a refusal: the first is a model failure, the second
    is the model correctly recognising a question outside the language.
    """
    match = _JSON_RE.search(text or "")
    if not match:
        raise InvalidProgram("no JSON object in the output")
    try:
        blob = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InvalidProgram(f"invalid JSON: {exc}") from exc

    unknown = set(blob) - _PROGRAM_FIELDS
    if unknown:
        raise InvalidProgram(f"unknown program field(s): {sorted(unknown)}")

    if "op" not in blob:
        # An explicit {"op": null} is the model saying "not expressible" --
        # a refusal, and a correct one. A bare {} is malformed output. Treating
        # them alike credited the model with a judgement it never made.
        raise InvalidProgram("missing required field 'op'")

    op_name = blob["op"]
    if op_name is None:
        return None
    if not isinstance(op_name, str):
        raise InvalidProgram(f"op must be a string or null, got {type(op_name).__name__}")
    try:
        op = Op(str(op_name).strip().lower())
    except ValueError as exc:
        raise InvalidProgram(f"unknown op {op_name!r}") from exc

    target = _selector_from_json(blob.get("target"), 0, valid_directions)
    if target is None:
        raise InvalidProgram("missing target selector")
    second = _selector_from_json(blob.get("second"), 0, valid_directions)

    if op in (Op.SAME_STATUS, Op.COMPARE_COUNT) and second is None:
        raise InvalidProgram(f"{op.value} needs a second selector")

    comparison = blob.get("comparison")
    if op is Op.COMPARE_COUNT:
        # The last surviving `str(x or default)`. Any falsy value of any type
        # collapsed to "more", so a model that emitted `false` for a "fewer"
        # question had its program silently inverted -- and counted as a
        # successful parse, so the error appeared as reasoning, not parsing.
        if comparison is None:
            raise InvalidProgram("compare_count requires an explicit comparison")
        if not isinstance(comparison, str):
            raise InvalidProgram(
                f"comparison must be a string, got {type(comparison).__name__}")
        comparison = comparison.strip().lower()
        if comparison not in ("more", "fewer"):
            raise InvalidProgram(f"unknown comparison {comparison!r}")
    else:
        comparison = None

    return Query(op, target, second, comparison)


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #

@dataclass
class ProgramAttempt:
    """One question, the model's raw output, and what came of it."""

    question: str
    raw: str
    query: Optional[Query] = None
    status: str = "ok"          # ok | refused | invalid
    reason: str = ""
    from_cache: bool = False
    # Cost, cached alongside the generation. A cached entry that reported zero
    # tokens would make the executed branch look free in the comparison table,
    # which is the opposite of the truth: it pays for a program AND the
    # executor's time.
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.query is not None


@dataclass
class ParserStats:
    """Where the parses went -- reported next to accuracy."""

    total: int = 0
    parsed: int = 0
    refused: int = 0          # the model said "not expressible"
    invalid: int = 0          # the model produced malformed output
    cache_hits: int = 0
    foreign_used: int = 0     # reused from another configuration
    truncated: int = 0        # generation stopped on the token budget
    unique_questions: int = 0
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        t = max(self.total, 1)
        return {
            "total": self.total,
            "unique_questions": self.unique_questions,
            "coverage": self.parsed / t,
            "refused": self.refused / t,
            "invalid": self.invalid / t,
            "cache_hit_rate": self.cache_hits / t,
            "truncated_generations": self.truncated,
            "foreign_cache_used": self.foreign_used / t,
            "seconds": round(self.seconds, 1),
        }


class QwenQuestionParser:
    """Frozen instruct model mapping a question onto a DSL program.

    Parameters
    ----------
    model_name:
        Any chat-tuned causal LM. Defaults to a small Qwen that fits a T4.
    cache_path:
        JSON file of ``{question: raw model output}``. nuScenes-QA repeats its
        templates heavily, so caching by question text turns thousands of
        generations into a few hundred.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        cache_path: Optional[str] = None,
        max_new_tokens: int = 96,
        device: Optional[str] = None,
        batch_size: int = 16,
        checkpoint_every: int = 5,
        directions: Optional[Sequence[str]] = None,
        schema_version: str = "dsl-1",
        allow_foreign: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every   # save every N batches
        self.directions = tuple(directions) if directions else DIRECTIONS
        self.schema_version = schema_version
        self.allow_foreign = allow_foreign
        self.config_registry: Dict[str, object] = {}
        self.stats = ParserStats()
        self._cache: Dict[str, Dict[str, str]] = {}
        self.cache_meta: Dict[str, object] = {}
        self.foreign_entries = 0
        self._runtime = None
        self._device = device
        self._load_cache()

    # -- cache identity ---------------------------------------------------- #

    def config_fingerprint(self) -> Dict[str, object]:
        """What a cached generation depends on.

        Keying the cache on the question text alone silently mixes outputs
        from different models, prompts and token budgets. Two entries that
        disagree are then indistinguishable from two entries that agree.

        The hash covers the system prompt *and* the few-shot exchanges: they
        are part of the model's input, so changing an example changes the
        parser just as surely as changing the instructions.
        """
        prompt = build_system_prompt(self.directions)
        shots = json.dumps(FEW_SHOT, sort_keys=True)
        digest = hashlib.sha256((prompt + "\x00" + shots).encode("utf-8"))
        return {
            "model": self.model_name,
            "prompt_sha256": digest.hexdigest()[:16],
            "directions": list(self.directions),
            "max_new_tokens": self.max_new_tokens,
            "schema_version": self.schema_version,
            "decoding": "greedy",
        }

    def config_id(self) -> str:
        """Short stable id for this configuration, used to tag entries."""
        blob = json.dumps(self.config_fingerprint(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def _load_cache(self) -> None:
        """Load the cache, keeping every entry's own provenance.

        Provenance belongs to the entry, not to the file. A single ``meta``
        block at the top meant that saving after reading a foreign cache
        stamped the current configuration onto generations it did not produce:
        one load/save cycle and the evidence was gone. Entries now carry the id
        of the configuration that generated them, and saving never rewrites it.
        """
        self.config_registry[self.config_id()] = self.config_fingerprint()
        if not (self.cache_path and os.path.exists(self.cache_path)):
            return

        with open(self.cache_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        mine = self.config_id()

        if isinstance(payload, dict) and "entries" in payload:
            raw_entries = payload.get("entries", {})
            registry = payload.get("configs", {})
            legacy_meta = payload.get("meta")
            for cid, fp in registry.items():
                self.config_registry.setdefault(cid, fp)
        else:
            # Oldest format: a bare {question: output} map, no provenance at
            # all. Keep the generations -- they cost GPU hours -- but never
            # claim they came from this configuration.
            raw_entries, legacy_meta = payload, None

        for question, value in raw_entries.items():
            if isinstance(value, dict):
                raw, cid = value.get("raw", ""), value.get("config")
            else:
                # Flat value: it came either from a single-meta file (use that
                # file's fingerprint) or from the oldest format (unknown).
                raw = value
                cid = self._register_legacy(legacy_meta) if legacy_meta else "unknown"
            self._cache[question] = {"raw": raw, "config": cid}
            if cid != mine:
                self.foreign_entries += 1

        if self.foreign_entries:
            others = sorted({e["config"] for e in self._cache.values()} - {mine})
            print(f"[cache] {self.foreign_entries}/{len(self._cache)} entries come "
                  f"from other configurations ({', '.join(others)}); "
                  f"{'reused and counted separately' if self.allow_foreign else 'ignored, they will be regenerated'}",
                  flush=True)

    def _register_legacy(self, meta: Dict[str, object]) -> str:
        """Give a legacy file-level fingerprint its own id."""
        blob = json.dumps(meta, sort_keys=True)
        cid = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
        self.config_registry.setdefault(cid, meta)
        return cid

    # -- model ------------------------------------------------------------ #

    def _ensure_model(self) -> None:
        """Load through the shared runtime, so both modes load identically.

        The comparison is between two uses of one model. A second loader here
        would be a second set of dtype and decoding choices that have to be
        kept in step by hand -- which is exactly how the status filter was lost
        between the two executors in this package.
        """
        if self._runtime is not None:
            return
        from .qwen_runtime import QwenRuntime, RuntimeConfig

        self._runtime = QwenRuntime(RuntimeConfig(
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            device=self._device))
        self._runtime.ensure_loaded()
        self._device = self._runtime.device

    def _generate(self, questions: Sequence[str]) -> List[str]:
        return [g.text for g in self._generate_full(questions)]

    def _generate_full(self, questions: Sequence[str]):
        """Generate, keeping the token counts the comparison has to report."""
        self._ensure_model()
        assert self._runtime is not None
        generations = self._runtime.generate(
            [build_messages(q, self.directions) for q in questions])
        for generation in generations:
            if generation.hit_token_budget:
                # A program cut off by the budget is unparseable for a reason
                # that has nothing to do with the model's understanding. An
                # earlier run lost 26% of its programs this way, 97% of them to
                # truncation, and the number was read as a model failure.
                self.stats.truncated += 1
        return generations

    # -- public ------------------------------------------------------------ #

    def warm_cache(self, questions: Sequence[str], verbose: bool = True) -> None:
        """Generate programs for every question not already cached."""
        missing = sorted({q for q in questions if self._usable(q) is None})
        self.stats.unique_questions = len({*questions})
        if not missing:
            return
        start = time.perf_counter()
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i:i + self.batch_size]
            for q, generation in zip(chunk, self._generate_full(chunk)):
                self._cache[q] = {
                    "raw": generation.text, "config": self.config_id(),
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                    "seconds": generation.seconds}
            done = min(i + self.batch_size, len(missing))
            # Checkpoint as we go. Generation is the expensive part of the
            # whole pipeline, and a saved cache makes a lost VM cost minutes
            # instead of the entire run.
            if done % (self.batch_size * self.checkpoint_every) == 0:
                self.save_cache()
            if verbose:
                print(f"  parsed {done}/{len(missing)} unique questions "
                      f"({time.perf_counter() - start:.0f}s)", flush=True)
        self.stats.seconds += time.perf_counter() - start
        self.save_cache()

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".",
                    exist_ok=True)
        # Entries keep the config id they were generated under; this never
        # relabels anything it did not produce.
        payload = {
            "current_config": self.config_id(),
            "configs": self.config_registry,
            "entries": self._cache,
        }
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=0)

    def _usable(self, question: str) -> Optional[str]:
        """The cached generation for ``question``, if we may use it.

        An entry from another configuration is only reused when the caller
        asked for it. Reusing it by default would quietly report generations
        from one prompt as the output of another.

        This is a pure lookup: it counts nothing. It is called both to test
        availability (``warm_cache``) and to fetch (``parse``), so counting
        here charged the same entry twice and could report a reuse rate above
        100%. The counter lives in :meth:`parse`, where a generation is
        actually consumed.
        """
        entry = self._cache.get(question)
        if entry is None:
            return None
        if entry.get("config") == self.config_id():
            return entry["raw"]
        return entry["raw"] if self.allow_foreign else None

    def _is_foreign(self, question: str) -> bool:
        entry = self._cache.get(question)
        return entry is not None and entry.get("config") != self.config_id()

    def parse(self, question: str) -> Optional[Query]:
        """Return a program, or ``None`` when the question is not expressible."""
        return self.parse_detailed(question).query

    def parse_detailed(self, question: str) -> "ProgramAttempt":
        """The same parse, keeping the model's text and why it was rejected.

        A per-question comparison has to separate "the model declined" from
        "the model produced something unreadable": they are different failures
        with different fixes, and an accuracy column alone hides which one
        happened. :meth:`parse` delegates here so the two cannot drift.
        """
        self.stats.total += 1
        raw = self._usable(question)
        from_cache = raw is not None
        if raw is None:
            generation = self._generate_full([question])[0]
            raw = generation.text
            self._cache[question] = {
                "raw": raw, "config": self.config_id(),
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "seconds": generation.seconds}
        else:
            self.stats.cache_hits += 1
            if self._is_foreign(question):
                self.stats.foreign_used += 1

        entry = self._cache.get(question, {})
        cost = {"input_tokens": int(entry.get("input_tokens", 0) or 0),
                "output_tokens": int(entry.get("output_tokens", 0) or 0),
                "seconds": float(entry.get("seconds", 0.0) or 0.0)}

        try:
            query = query_from_json(raw, frozenset(self.directions))
        except InvalidProgram as exc:
            self.stats.invalid += 1
            return ProgramAttempt(question, raw, None, "invalid", str(exc),
                                  from_cache, **cost)
        if query is None:
            self.stats.refused += 1
            return ProgramAttempt(question, raw, None, "refused",
                                  "the model declined to express the question",
                                  from_cache, **cost)
        self.stats.parsed += 1
        return ProgramAttempt(question, raw, query, "ok", "", from_cache, **cost)
