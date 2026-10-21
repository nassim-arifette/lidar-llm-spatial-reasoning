"""System A: the language model answers from the objects, with no program.

This is the condition the project has been missing. Everything it receives, the
executed system also receives: the same question, the same object list, the
same statement of the spatial conventions. What it does not receive is an
executor -- it has to do the selecting and counting itself, in its own
activations.

Three ways this comparison could be rigged, and what stops each:

**Giving it a filtered scene.** Handing it only the cars when the question asks
about cars, or a precomputed count, would be handing it the answer. It gets
every object in the sample, in a fixed order, with no field that depends on the
question.

**Giving it coarser numbers.** The executor computes on full precision, so if
the direct model saw one-decimal coordinates the table would partly be
measuring resolution. Both modes are built from
:func:`manifest.describe_objects`, with the same rounding.

**Letting it fail on the format and calling that reasoning.** A model that
answers "There are 3 cars" when the reference is "3" is right about the scene
and wrong about the format, and scoring those together measures neither. The
answer is normalised the same way for both modes, and anything the
normalisation cannot resolve is recorded as ``unparseable`` rather than counted
as a wrong answer -- it is reported separately in the table.

The conventions in the prompt are the executor's conventions, stated in prose:
the ego frame, the six sectors, and that an unknown status is not a different
status. Withholding them would make the comparison about who guessed the
convention, not about who applies it.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .manifest import ManifestItem, describe_objects, scene_fingerprint
from .objects import Scene
from .qwen_runtime import Generation, QwenRuntime, RuntimeConfig
from .regions import RegionConfig

#: Bumped whenever the wording below changes, and recorded with every answer.
#: Two runs under different prompts are two experiments.
PROMPT_VERSION = "direct-1"


def build_system_prompt(config: RegionConfig) -> str:
    """The conventions, stated exactly as the executor implements them."""
    # The rear sector wraps through 180, and printing it as "(150, -150]" reads
    # as an empty interval. Unwrapping it keeps the prose a correct statement
    # of what the executor computes, which is the whole point of giving the
    # direct model the conventions at all.
    sectors = config.sectors
    lines = ", ".join(
        f"{name.replace('_', ' ')} = bearing in "
        f"({low:g}, {high if high > low else high + 360:g}] degrees"
        for name, (low, high) in sorted(sectors.items(), key=lambda kv: kv[1][0]))
    return f"""You answer questions about a 3D driving scene from a list of objects.

Coordinate frame: the ego vehicle is at the origin. x points forward, y points
left, both in metres. A bearing of 0 degrees is straight ahead and bearings
increase to the left.

The spatial relations are bearing sectors around the ego vehicle: {lines}.
"To the <relation> of object Z" means the same sectors measured from Z's
position instead of the origin, keeping the ego vehicle's heading.

An object's status is one of moving, parked, stopped, standing, sitting lying
down, with rider, without rider, or unknown. An unknown status is not a
different status: an object whose status is unknown is neither "parked" nor
"not parked", so it must not be counted under either.

Answer with the answer alone and nothing else:
- a bare integer for a question that asks how many;
- yes or no for a question that asks whether something exists;
- an object class or a status, in lower case, for anything else.
Do not explain. Do not restate the question. Do not show your working."""


def build_user_prompt(question: str, objects: Sequence[Dict[str, object]]) -> str:
    """The scene and the question, in a fixed order that ignores the question.

    The object rows are emitted in the order the adapter produced them, which
    depends on the annotation file and not on what is being asked. Sorting them
    by relevance, or by distance, would be a filtering step done on the model's
    behalf.
    """
    header = "id, category, x, y, status"
    rows = "\n".join(
        f"{o['id']}, {o['category']}, {o['x']}, {o['y']}, {o['status']}"
        for o in objects)
    return (f"Objects ({len(objects)}):\n{header}\n{rows}\n\n"
            f"Question:\n{question}\n\nAnswer:")


def build_messages(
    question: str, objects: Sequence[Dict[str, object]], config: RegionConfig
) -> List[Dict[str, str]]:
    return [{"role": "system", "content": build_system_prompt(config)},
            {"role": "user", "content": build_user_prompt(question, objects)}]


# --------------------------------------------------------------------------- #
# Reading the answer back
# --------------------------------------------------------------------------- #

_NUMBER_WORDS = {
    "zero": "0", "none": "0", "no": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

_LEADING_INT = re.compile(r"^\s*(\d+)\b")
_ANY_INT = re.compile(r"\b(\d+)\b")


@dataclass
class DirectAnswer:
    """What the direct model said, and how much of it could be read."""

    question_id: str
    raw: str
    answer: Optional[str] = None
    status: str = "ok"            # ok | unparseable | truncated | overlong_prompt
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    n_objects: int = 0
    scene_fingerprint: str = ""
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalise_answer(raw: str, expects_count: bool) -> Tuple[Optional[str], str]:
    """Read a bare answer out of the model's text.

    Deliberately shallow. A generous reader that digs an integer out of a
    paragraph would be doing part of the task for the model, and the same
    reader is not applied to the executed branch, which emits a bare value by
    construction. So: strip the obvious wrappers, and give up visibly rather
    than guess.
    """
    text = (raw or "").strip().strip("`").strip()
    if not text:
        return None, "unparseable"
    first = text.splitlines()[0].strip().rstrip(".").strip()
    lowered = first.lower()

    if expects_count:
        match = _LEADING_INT.match(lowered)
        if match:
            return str(int(match.group(1))), "ok"
        if lowered in _NUMBER_WORDS:
            return _NUMBER_WORDS[lowered], "ok"
        # "There are 3 cars." -- one integer in a short sentence is readable;
        # several is ambiguous and is not guessed at.
        numbers = _ANY_INT.findall(first)
        if len(numbers) == 1:
            return str(int(numbers[0])), "ok"
        return None, "unparseable"

    # A leading yes or no is the answer, however much follows it. "No, there
    # are none." is a correct answer to an existence question, and discarding
    # it as unreadable would charge the direct mode for the executed mode's
    # structural advantage rather than for anything about the scene.
    verdict = re.match(r"^(yes|no)\b", lowered)
    if verdict:
        return verdict.group(1), "ok"
    # A class or a status, in the spelling the dataset uses.
    cleaned = re.sub(r"[^a-z ]", "", lowered).strip()
    if cleaned and len(cleaned.split()) <= 3:
        return cleaned, "ok"
    return None, "unparseable"


def asks_for_a_count(question: str) -> bool:
    """Whether the question wants a number, read from the question only.

    Never from the reference answer. Choosing the output format from the gold
    answer's type is a label leak that this project has already made once, in
    the learned baselines.
    """
    text = question.strip().lower()
    return bool(re.search(r"\bhow many\b|\bwhat (?:is the )?number of\b|"
                          r"\bcount\b", text))


# --------------------------------------------------------------------------- #
# The system
# --------------------------------------------------------------------------- #

class DirectQwenQA:
    """Question + objects -> answer, with no intermediate representation."""

    def __init__(
        self,
        runtime: Optional[QwenRuntime] = None,
        region_config: Optional[RegionConfig] = None,
        cache_path: Optional[str] = None,
        coordinate_precision: int = 1,
    ) -> None:
        self.runtime = runtime or QwenRuntime(RuntimeConfig())
        self.region_config = region_config or RegionConfig(layout="nuscenes_qa")
        self.coordinate_precision = coordinate_precision
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, object]] = {}
        self._load_cache()

    # -- cache ------------------------------------------------------------ #

    def cache_key(self, question: str, scene: Scene) -> str:
        """Question *and* scene.

        The program parser may cache on the question alone, because a program
        does not depend on the objects. A direct answer does. Sharing one cache
        between the two modes, or keying this one on the question, would serve
        the same answer for the same question asked of a different scene --
        which is precisely the control the comparison uses to show a system
        looks at the scene at all.
        """
        return json.dumps({
            "q": question,
            "scene": scene_fingerprint(scene),
            "prompt": PROMPT_VERSION,
            "layout": self.region_config.layout,
            "precision": self.coordinate_precision,
            "runtime": self.runtime.config.identity(),
        }, sort_keys=True)

    def _load_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                self._cache = json.load(fh).get("entries", {})
        except (OSError, json.JSONDecodeError):
            self._cache = {}

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        import os
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)) or ".",
                    exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as fh:
            json.dump({"prompt_version": PROMPT_VERSION,
                       "runtime": self.runtime.config.fingerprint(),
                       "entries": self._cache}, fh)

    # -- inference -------------------------------------------------------- #

    def answer_batch(
        self,
        items: Sequence[ManifestItem],
        scenes: Dict[str, Scene],
        verbose: bool = False,
    ) -> List[DirectAnswer]:
        """Answer every item. ``items`` carry answers; this never reads them."""
        pending: List[int] = []
        conversations: List[List[Dict[str, str]]] = []
        results: List[Optional[DirectAnswer]] = [None] * len(items)
        payloads: List[Dict[str, object]] = []

        for index, item in enumerate(items):
            scene = scenes[item.sample_token]
            objects = describe_objects(scene, precision=self.coordinate_precision)
            payloads.append({"objects": objects, "scene": scene})
            key = self.cache_key(item.question, scene)
            cached = self._cache.get(key)
            if cached is not None:
                results[index] = self._finish(item, cached["raw"], objects, scene,
                                              Generation(text=cached["raw"],
                                                         input_tokens=cached.get("input_tokens", 0),
                                                         output_tokens=cached.get("output_tokens", 0),
                                                         hit_token_budget=cached.get("hit_token_budget", False),
                                                         seconds=0.0))
                continue
            pending.append(index)
            conversations.append(build_messages(
                item.question, objects, self.region_config))

        if conversations:
            if verbose:
                print(f"direct: generating {len(conversations)} of {len(items)} "
                      f"({len(items) - len(conversations)} cached)")
            generations = self.runtime.generate_all(conversations, verbose=verbose)
            for index, generation in zip(pending, generations):
                item = items[index]
                objects = payloads[index]["objects"]           # type: ignore[index]
                scene = payloads[index]["scene"]               # type: ignore[index]
                self._cache[self.cache_key(item.question, scene)] = {
                    "raw": generation.text,
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                    "hit_token_budget": generation.hit_token_budget,
                }
                results[index] = self._finish(item, generation.text, objects,
                                              scene, generation)
            self.save_cache()

        return [r for r in results if r is not None]

    def _finish(self, item: ManifestItem, raw: str, objects, scene: Scene,
                generation: Generation) -> DirectAnswer:
        answer, status = normalise_answer(raw, asks_for_a_count(item.question))
        if generation.hit_token_budget and answer is None:
            status = "truncated"
        return DirectAnswer(
            question_id=item.question_id,
            raw=raw,
            answer=answer,
            status=status,
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            seconds=generation.seconds,
            n_objects=len(objects),
            scene_fingerprint=scene_fingerprint(scene),
        )
