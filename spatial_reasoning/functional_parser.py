"""Question -> a typed functional program, written by a frozen language model.

This is the second target language, and a separate experiment. The first one
(``llm_parser`` into ``dsl``) is a flat record: one operation, one or two
selectors, each with at most a category, a status, a region and an anchor. It
covers the counting and existence templates of nuScenes-QA and nothing else,
because the shape itself cannot hold ``query the attribute of the one object
that ...`` or ``the other objects sharing its attribute``.

Here the model writes CLEVR's representation instead — a list of typed nodes,
each naming a function, its constant argument and the indices of its inputs,
in topological order — and ``functional.execute`` runs it. Expressiveness is
the only thing that changes: the model, the decoding, the object lists and the
conventions are the same.

Two consequences worth stating before any number is read.

**More coverage is not automatically better.** A wider language can express
more questions and can also express more wrong programs. Nothing here
guarantees that the model writes the right one, and the comparison against the
flat language is exactly the measurement of whether the extra room helps.

**The failure modes move.** The flat parser rejects a malformed record; this
one type-checks a graph, so a program can now fail for reasons that did not
exist before — an input of the wrong type, an index pointing nowhere, a final
node that returns a set rather than an answer. Those are reported separately
rather than folded into ``invalid``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .functional import (
    CATEGORY_VOCABULARY,
    STATUSES,
    Failure,
    Program,
    ProgramError,
    Signature,
    catalogue,
    program_from_json,
)
from .qwen_runtime import QwenRuntime, RuntimeConfig
from .regions import DEFAULT_REGION_CONFIG, RegionConfig

#: Bumped when the prompt below changes. Recorded with every result: two runs
#: under different prompts are two experiments.
SCHEMA_VERSION = "functional-1"


def _render_catalogue(cat: Dict[str, Signature]) -> str:
    lines = []
    for sig in cat.values():
        inputs = ", ".join(t.value for t in sig.inputs)
        constant = f"  constant: one of {list(sig.vocabulary)}" if sig.value_input else ""
        lines.append(f"  {sig.name}({inputs}) -> {sig.output.value}"
                     f"   # {sig.doc}{constant}")
    return "\n".join(lines)


def build_system_prompt(config: RegionConfig = DEFAULT_REGION_CONFIG) -> str:
    cat = catalogue(config)
    return f"""You translate a question about a 3D driving scene into a functional program.
You never answer the question. You only describe what must be computed.

Output EXACTLY one JSON object, no prose, no markdown fence:

{{"program": [ {{"function": NAME, "value_inputs": [CONSTANT] or [], "inputs": [INDEX, ...]}}, ... ]}}

The list is in topological order. "inputs" holds the indices of EARLIER nodes
in this same list. The LAST node produces the answer, and must return a
number, a bool or a value.

Functions, with their input and output types:

{_render_catalogue(cat)}

Rules:
- Every program starts with scene(), which is node 0.
- "thing", "things", "object" -> filter_category with "object".
- "how many" / "what number of" -> end with count.
- "are there any" / "is there" / "are ... visible" -> end with exist.
- "what is it" / "what is the <thing>" -> end with query_category, whose input
  must be a unique(...).
- "in what status" / "what is the status of" -> end with query_status.
- "do X and Y have the same status" -> query_status each, then equal_value.
- "are there more A than B" -> count each, then greater_than.
- "the other things in the same status as X" -> same_status(unique(... X ...)).
- A direction about the ego vehicle uses relate_ego on a SET.
  A direction about another object uses relate on a UNIQUE object.
- "it" refers to the object introduced earlier in the same question.
- "not standing" -> filter_not_status with "standing".
- An object whose status is unknown is neither the status nor its negation.
- If the question cannot be expressed with these functions, output {{"program": []}}."""


#: Worked examples, one per nuScenes-QA template plus the composite forms the
#: flat language could not reach. They are part of the prompt, so changing one
#: changes the parser and therefore the cache identity.
FEW_SHOT: Sequence[tuple] = (
    ("How many parked cars are there?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_category", "value_inputs": ["car"], "inputs": [0]},
      {"function": "filter_status", "value_inputs": ["parked"], "inputs": [1]},
      {"function": "count", "value_inputs": [], "inputs": [2]}]),
    ("Are there any pedestrians to the front left of the ego vehicle?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "relate_ego", "value_inputs": ["front_left"], "inputs": [0]},
      {"function": "filter_category", "value_inputs": ["pedestrian"], "inputs": [1]},
      {"function": "exist", "value_inputs": [], "inputs": [2]}]),
    ("There is a trailer; what number of traffic cones are to the back of it?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_category", "value_inputs": ["trailer"], "inputs": [0]},
      {"function": "unique", "value_inputs": [], "inputs": [1]},
      {"function": "relate", "value_inputs": ["back"], "inputs": [2]},
      {"function": "filter_category", "value_inputs": ["traffic_cone"], "inputs": [3]},
      {"function": "count", "value_inputs": [], "inputs": [4]}]),
    ("There is a parked thing; what is it?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_status", "value_inputs": ["parked"], "inputs": [0]},
      {"function": "unique", "value_inputs": [], "inputs": [1]},
      {"function": "query_category", "value_inputs": [], "inputs": [2]}]),
    ("The construction vehicle is in what status?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_category", "value_inputs": ["construction_vehicle"], "inputs": [0]},
      {"function": "unique", "value_inputs": [], "inputs": [1]},
      {"function": "query_status", "value_inputs": [], "inputs": [2]}]),
    ("Do the trailer and the truck have the same status?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_category", "value_inputs": ["trailer"], "inputs": [0]},
      {"function": "unique", "value_inputs": [], "inputs": [1]},
      {"function": "query_status", "value_inputs": [], "inputs": [2]},
      {"function": "filter_category", "value_inputs": ["truck"], "inputs": [0]},
      {"function": "unique", "value_inputs": [], "inputs": [4]},
      {"function": "query_status", "value_inputs": [], "inputs": [5]},
      {"function": "equal_value", "value_inputs": [], "inputs": [3, 6]}]),
    ("How many other things are in the same status as the bicycle?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_category", "value_inputs": ["bicycle"], "inputs": [0]},
      {"function": "unique", "value_inputs": [], "inputs": [1]},
      {"function": "same_status", "value_inputs": [], "inputs": [2]},
      {"function": "count", "value_inputs": [], "inputs": [3]}]),
    ("Are there more cars than trucks?",
     [{"function": "scene", "value_inputs": [], "inputs": []},
      {"function": "filter_category", "value_inputs": ["car"], "inputs": [0]},
      {"function": "count", "value_inputs": [], "inputs": [1]},
      {"function": "filter_category", "value_inputs": ["truck"], "inputs": [0]},
      {"function": "count", "value_inputs": [], "inputs": [3]},
      {"function": "greater_than", "value_inputs": [], "inputs": [2, 4]}]),
)


def build_messages(question: str, config: RegionConfig) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt(config)}]
    for q, program in FEW_SHOT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant",
                         "content": json.dumps({"program": program})})
    messages.append({"role": "user", "content": question})
    return messages


def _extract_json(raw: str) -> str:
    """Pull the first balanced JSON object out of the model's text."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


@dataclass
class FunctionalAttempt:
    """One question, the model's raw output, and what came of it."""

    question: str
    raw: str
    program: Optional[Program] = None
    status: str = "ok"          # ok | refused | malformed | ill_typed
    reason: str = ""
    from_cache: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.program is not None


@dataclass
class FunctionalParserStats:
    total: int = 0
    parsed: int = 0
    refused: int = 0
    malformed: int = 0
    ill_typed: int = 0
    truncated: int = 0
    cache_hits: int = 0
    seconds: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        t = max(self.total, 1)
        return {"total": self.total, "coverage": self.parsed / t,
                "refused": self.refused / t, "malformed": self.malformed / t,
                "ill_typed": self.ill_typed / t,
                "truncated_generations": self.truncated,
                "cache_hit_rate": self.cache_hits / t,
                "seconds": round(self.seconds, 1)}


class FunctionalQwenParser:
    """A frozen chat model writing typed functional programs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_path: Optional[str] = None,
        max_new_tokens: int = 384,
        device: Optional[str] = None,
        batch_size: int = 8,
        region_config: Optional[RegionConfig] = None,
        precision: str = "fp16",
        device_map: Optional[str] = None,
    ) -> None:
        self.region_config = region_config or DEFAULT_REGION_CONFIG
        self.catalogue = catalogue(self.region_config)
        self.runtime = QwenRuntime(RuntimeConfig(
            max_new_tokens=max_new_tokens, batch_size=batch_size,
            device=device, precision=precision, device_map=device_map,
            **({"model_name": model_name} if model_name else {})))
        self.cache_path = cache_path
        self.stats = FunctionalParserStats()
        self._cache: Dict[str, Dict[str, object]] = {}
        self._load_cache()

    # -- identity --------------------------------------------------------- #

    def config_fingerprint(self) -> Dict[str, object]:
        prompt = build_system_prompt(self.region_config)
        shots = json.dumps(FEW_SHOT, sort_keys=True, default=str)
        digest = hashlib.sha256((prompt + "\x00" + shots).encode("utf-8"))
        out = dict(self.runtime.config.fingerprint())
        out.update(prompt_sha256=digest.hexdigest()[:16],
                   schema_version=SCHEMA_VERSION,
                   layout=self.region_config.layout)
        return out

    def config_id(self) -> str:
        blob = json.dumps(self.config_fingerprint(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    # -- cache ------------------------------------------------------------ #

    def _load_cache(self) -> None:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                self._cache = json.load(fh).get("entries", {})
        except (OSError, json.JSONDecodeError):
            self._cache = {}

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".",
                    exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump({"config": self.config_fingerprint(),
                       "entries": self._cache}, fh)

    def _usable(self, question: str) -> Optional[Dict[str, object]]:
        entry = self._cache.get(question)
        if entry is None or entry.get("config") != self.config_id():
            return None
        return entry

    # -- generation ------------------------------------------------------- #

    def warm_cache(self, questions: Sequence[str], verbose: bool = True) -> None:
        missing = sorted({q for q in questions if self._usable(q) is None})
        if not missing:
            return
        start = time.perf_counter()
        size = max(self.runtime.config.batch_size, 1)
        for i in range(0, len(missing), size):
            chunk = missing[i:i + size]
            generations = self.runtime.generate(
                [build_messages(q, self.region_config) for q in chunk])
            for q, g in zip(chunk, generations):
                if g.hit_token_budget:
                    self.stats.truncated += 1
                self._cache[q] = {
                    "raw": g.text, "config": self.config_id(),
                    "input_tokens": g.input_tokens,
                    "output_tokens": g.output_tokens, "seconds": g.seconds}
            if verbose:
                print(f"  programs {min(i + size, len(missing))}/{len(missing)}"
                      f" ({time.perf_counter() - start:.0f}s)", flush=True)
            self.save_cache()
        self.stats.seconds += time.perf_counter() - start
        self.save_cache()

    def parse_detailed(self, question: str) -> FunctionalAttempt:
        self.stats.total += 1
        entry = self._usable(question)
        from_cache = entry is not None
        if entry is None:
            g = self.runtime.generate([build_messages(question, self.region_config)])[0]
            if g.hit_token_budget:
                self.stats.truncated += 1
            entry = {"raw": g.text, "config": self.config_id(),
                     "input_tokens": g.input_tokens,
                     "output_tokens": g.output_tokens, "seconds": g.seconds}
            self._cache[question] = entry
        else:
            self.stats.cache_hits += 1

        cost = {"input_tokens": int(entry.get("input_tokens", 0) or 0),
                "output_tokens": int(entry.get("output_tokens", 0) or 0),
                "seconds": float(entry.get("seconds", 0.0) or 0.0)}
        raw = str(entry["raw"])

        payload = _extract_json(raw)
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            self.stats.malformed += 1
            return FunctionalAttempt(question, raw, None, "malformed",
                                     f"not JSON: {exc}", from_cache, **cost)

        nodes = parsed.get("program") if isinstance(parsed, dict) else parsed
        if isinstance(nodes, list) and not nodes:
            # The model said the question is outside the language. That is an
            # answer about the language, not a failure to produce JSON, and it
            # is counted apart.
            self.stats.refused += 1
            return FunctionalAttempt(question, raw, None, "refused",
                                     "the model declined to express the question",
                                     from_cache, **cost)

        try:
            program = program_from_json(parsed, self.catalogue)
        except ProgramError as exc:
            bucket = ("malformed" if exc.failure is Failure.MALFORMED
                      else "ill_typed")
            setattr(self.stats, bucket, getattr(self.stats, bucket) + 1)
            return FunctionalAttempt(question, raw, None, bucket, exc.message,
                                     from_cache, **cost)

        self.stats.parsed += 1
        return FunctionalAttempt(question, raw, program, "ok", "", from_cache,
                                 **cost)

    def parse(self, question: str) -> Optional[Program]:
        return self.parse_detailed(question).program
