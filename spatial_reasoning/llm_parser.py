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


def build_system_prompt() -> str:
    return (SYSTEM_PROMPT
            .replace("[CATEGORIES]", json.dumps(list(CANONICAL_CATEGORIES)))
            .replace("[STATUSES]", json.dumps(list(STATUSES)))
            .replace("[DIRECTIONS]", json.dumps(list(DIRECTIONS))))


def build_messages(question: str) -> List[dict]:
    messages = [{"role": "system", "content": build_system_prompt()}]
    for q, program in FEW_SHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": json.dumps(program)})
    messages.append({"role": "user", "content": question})
    return messages


# --------------------------------------------------------------------------- #
# JSON -> Query
# --------------------------------------------------------------------------- #

_VALID_CATEGORIES = set(CANONICAL_CATEGORIES) | {"object", "thing", "vehicle"}
_VALID_DIRECTIONS = set(DIRECTIONS)
_VALID_STATUSES = set(STATUSES)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class InvalidProgram(ValueError):
    """The model produced something that is not a program in the DSL."""


def _selector_from_json(blob: Optional[dict], depth: int = 0) -> Optional[Selector]:
    if blob is None:
        return None
    if not isinstance(blob, dict):
        raise InvalidProgram("selector is not an object")
    if depth > 2:
        raise InvalidProgram("anchor nested too deep")

    category = str(blob.get("category") or "object").strip().lower().replace(" ", "_")
    if category not in _VALID_CATEGORIES:
        raise InvalidProgram(f"unknown category {category!r}")

    status = blob.get("status")
    if status is not None:
        status = str(status).strip().lower()
        if status not in _VALID_STATUSES:
            raise InvalidProgram(f"unknown status {status!r}")

    direction = blob.get("direction")
    region = Region(kind="any")
    if direction is not None:
        direction = str(direction).strip().lower().replace(" ", "_")
        if direction not in _VALID_DIRECTIONS:
            raise InvalidProgram(f"unknown direction {direction!r}")
        region = Region(kind="direction", direction=direction)

    return Selector(
        category=category,
        status=status,
        negate_status=bool(blob.get("negate_status", False)),
        region=region,
        anchor=_selector_from_json(blob.get("anchor"), depth + 1),
    )


def query_from_json(text: str) -> Optional[Query]:
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

    op_name = blob.get("op")
    if op_name is None:
        return None
    try:
        op = Op(str(op_name).strip().lower())
    except ValueError as exc:
        raise InvalidProgram(f"unknown op {op_name!r}") from exc

    target = _selector_from_json(blob.get("target"))
    if target is None:
        raise InvalidProgram("missing target selector")
    second = _selector_from_json(blob.get("second"))

    if op in (Op.SAME_STATUS, Op.COMPARE_COUNT) and second is None:
        raise InvalidProgram(f"{op.value} needs a second selector")

    comparison = blob.get("comparison")
    if op is Op.COMPARE_COUNT:
        comparison = str(comparison or "more").strip().lower()
        if comparison not in ("more", "fewer"):
            raise InvalidProgram(f"unknown comparison {comparison!r}")
    else:
        comparison = None

    return Query(op, target, second, comparison)


# --------------------------------------------------------------------------- #
# The parser
# --------------------------------------------------------------------------- #

@dataclass
class ParserStats:
    """Where the parses went -- reported next to accuracy."""

    total: int = 0
    parsed: int = 0
    refused: int = 0          # the model said "not expressible"
    invalid: int = 0          # the model produced malformed output
    cache_hits: int = 0
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
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every   # save every N batches
        self.stats = ParserStats()
        self._cache: Dict[str, str] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                self._cache = json.load(fh)

        self._tokenizer = None
        self._model = None
        self._device = device

    # -- model ------------------------------------------------------------ #

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"   # required for batched generation
        # transformers renamed `torch_dtype` to `dtype` in v5; support both so
        # the run is not tied to one release.
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=dtype)
        except TypeError:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype)
        self._model = self._model.to(self._device)
        self._model.eval()

    def _generate(self, questions: Sequence[str]) -> List[str]:
        import torch

        self._ensure_model()
        prompts = [
            self._tokenizer.apply_chat_template(
                build_messages(q), tokenize=False, add_generation_prompt=True
            )
            for q in questions
        ]
        batch = self._tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=4096).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,              # greedy: the run is reproducible
                pad_token_id=self._tokenizer.pad_token_id,
            )
        generated = out[:, batch["input_ids"].shape[1]:]
        return self._tokenizer.batch_decode(generated, skip_special_tokens=True)

    # -- public ------------------------------------------------------------ #

    def warm_cache(self, questions: Sequence[str], verbose: bool = True) -> None:
        """Generate programs for every question not already cached."""
        missing = sorted({q for q in questions if q not in self._cache})
        self.stats.unique_questions = len({*questions})
        if not missing:
            return
        start = time.perf_counter()
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i:i + self.batch_size]
            for q, raw in zip(chunk, self._generate(chunk)):
                self._cache[q] = raw
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
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump(self._cache, fh, indent=0)

    def parse(self, question: str) -> Optional[Query]:
        """Return a program, or ``None`` when the question is not expressible."""
        self.stats.total += 1
        raw = self._cache.get(question)
        if raw is None:
            raw = self._generate([question])[0]
            self._cache[question] = raw
        else:
            self.stats.cache_hits += 1

        try:
            query = query_from_json(raw)
        except InvalidProgram:
            self.stats.invalid += 1
            return None
        if query is None:
            self.stats.refused += 1
            return None
        self.stats.parsed += 1
        return query
