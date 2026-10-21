"""The central comparison: one model, two ways of using it.

**A, direct**    question + objects -> Qwen -> answer.
**B, executed**  question -> the same Qwen -> program -> executor on the same
objects -> answer.

Same checkpoint, same decoding, same object lists, same stated conventions,
same example ids. The only thing that differs is whether the answer is produced
by the model or computed from a program the model wrote.

The question is not "can a language model reason about space". It is narrower
and answerable:

    at a fixed model and fixed scene data, is producing an executable program
    more reliable than producing the answer, and at what cost?

B brings code that knows the geometry. A result in its favour shows that
*this* division of labour helps on *these* questions; it does not show that
language models cannot do spatial reasoning, and the write-up should not say
so.

What the script refuses to do
-----------------------------

**Average away the failures.** A refusal, an invalid program and an unreadable
answer are all wrong answers in the headline accuracy, and all reported
separately underneath. A system that answers a third of the questions
perfectly is not ahead of one that answers all of them adequately, and only
the coverage column shows which is which.

**Share a cache between the modes.** A program depends on the question; a
direct answer depends on the question *and* the scene. Keying both on the
question would serve the same answer for the same question asked of a
different scene -- destroying the one control that shows a system looks at the
objects at all.

**Report only means.** Every question gets a row: both answers, the program,
the trace, the tokens, the latency, and the hashes that say what produced it.
A paired table -- A right and B wrong, B right and A wrong -- is what says
whether either is actually better, and a difference of means cannot.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .direct_qa import PROMPT_VERSION, DirectQwenQA, asks_for_a_count
from .dsl import ExecStatus, execute
from .llm_parser import QwenQuestionParser
from .manifest import Manifest, ManifestItem, describe_objects, scene_fingerprint
from .nuscenes_adapter import load_nuscenes_scenes
from .objects import Scene
from .qwen_runtime import QwenRuntime, RuntimeConfig
from .regions import RegionConfig

#: The target language B writes in. Recorded on every row: a change to the DSL
#: makes a new experiment, not a new number under the same name.
DSL_VERSION = "dsl-1"


@dataclass
class Row:
    """One question, both systems, everything needed to audit either answer."""

    question_id: str
    sample_token: str
    scene_token: str
    family: str
    template: str
    question: str
    reference_answer: str

    # A
    direct_answer: Optional[str] = None
    direct_status: str = ""
    direct_raw: str = ""
    direct_correct: bool = False
    direct_input_tokens: int = 0
    direct_output_tokens: int = 0
    direct_seconds: float = 0.0

    # B
    program_raw: str = ""
    program_valid: bool = False
    program_status: str = ""          # ok | refused | invalid
    program: str = ""
    executed_answer: Optional[str] = None
    execution_status: str = ""
    execution_reason: str = ""
    executed_correct: bool = False
    selected: List[str] = field(default_factory=list)
    program_input_tokens: int = 0
    program_output_tokens: int = 0
    program_seconds: float = 0.0

    # provenance
    n_objects: int = 0
    scene_hash: str = ""
    model: str = ""
    model_revision: Optional[str] = None
    prompt_version: str = PROMPT_VERSION
    dsl_version: str = DSL_VERSION

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _correct(answer: Optional[str], reference: str) -> bool:
    if answer is None:
        return False
    return str(answer).strip().lower() == reference.strip().lower()


def _as_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def count_errors(rows: Sequence[Row], field_name: str) -> Dict[str, object]:
    """Under- and over-counting, over the questions that asked for a number.

    Both halves are reported because they have opposite causes: under-counting
    is objects the system did not select, over-counting is a filter it did not
    apply. A single mean absolute error adds them together and hides which one
    is happening.

    ``n`` is the number of questions where the system produced an integer at
    all. A lower error over far fewer answers is not an improvement, so it is
    never reported without that count.
    """
    under = over = 0.0
    n_under = n_over = n = 0
    worst: List[Tuple[int, str]] = []
    for row in rows:
        if not asks_for_a_count(row.question):
            continue
        gold = _as_int(row.reference_answer)
        got = _as_int(getattr(row, field_name))
        if gold is None or got is None:
            continue
        n += 1
        delta = got - gold
        if delta < 0:
            under += -delta
            n_under += 1
        elif delta > 0:
            over += delta
            n_over += 1
        worst.append((abs(delta), row.question_id))
    asked = sum(1 for r in rows if asks_for_a_count(r.question))
    worst.sort(reverse=True)
    return {
        "n_asked": asked,
        "n_numeric": n,
        "numeric_coverage": n / max(asked, 1),
        "under": under / max(n, 1),
        "over": over / max(n, 1),
        "mae": (under + over) / max(n, 1),
        "n_undercounted": n_under,
        "n_overcounted": n_over,
        "worst": [{"error": e, "question_id": q} for e, q in worst[:5]],
    }


def paired(rows: Sequence[Row]) -> Dict[str, int]:
    """The only table that says whether one mode is better than the other."""
    table = collections.Counter()
    for row in rows:
        key = ("both" if row.direct_correct and row.executed_correct else
               "direct_only" if row.direct_correct else
               "executed_only" if row.executed_correct else "neither")
        table[key] += 1
    return {k: table.get(k, 0) for k in
            ("both", "direct_only", "executed_only", "neither")}


def summarise(rows: Sequence[Row]) -> Dict[str, object]:
    n = max(len(rows), 1)
    direct_answered = sum(1 for r in rows if r.direct_answer is not None)
    executed_answered = sum(1 for r in rows if r.executed_answer is not None)
    return {
        "n": len(rows),
        "direct": {
            "accuracy": sum(r.direct_correct for r in rows) / n,
            "answered_fraction": direct_answered / n,
            "accuracy_when_answered":
                sum(r.direct_correct for r in rows) / max(direct_answered, 1),
            "statuses": dict(collections.Counter(r.direct_status for r in rows)),
            "count_errors": count_errors(rows, "direct_answer"),
            "input_tokens": sum(r.direct_input_tokens for r in rows),
            "output_tokens": sum(r.direct_output_tokens for r in rows),
            "seconds": round(sum(r.direct_seconds for r in rows), 1),
        },
        "executed": {
            "accuracy": sum(r.executed_correct for r in rows) / n,
            "answered_fraction": executed_answered / n,
            "accuracy_when_answered":
                sum(r.executed_correct for r in rows) / max(executed_answered, 1),
            "program_statuses": dict(collections.Counter(r.program_status for r in rows)),
            "execution_statuses": dict(collections.Counter(
                r.execution_status for r in rows if r.program_valid)),
            "count_errors": count_errors(rows, "executed_answer"),
            "input_tokens": sum(r.program_input_tokens for r in rows),
            "output_tokens": sum(r.program_output_tokens for r in rows),
            "seconds": round(sum(r.program_seconds for r in rows), 1),
        },
        "paired": paired(rows),
        "by_family": _grouped(rows, lambda r: r.family),
        "by_scene": _grouped(rows, lambda r: r.scene_token),
    }


def _grouped(rows: Sequence[Row], key) -> Dict[str, Dict[str, object]]:
    buckets: Dict[str, List[Row]] = collections.defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return {
        name: {
            "n": len(group),
            "direct": sum(r.direct_correct for r in group) / len(group),
            "executed": sum(r.executed_correct for r in group) / len(group),
            "paired": paired(group),
        }
        for name, group in sorted(buckets.items())
    }


def scene_dependence(rows: Sequence[Row]) -> Dict[str, object]:
    """Questions asked verbatim of two scenes with different answers.

    A system that does not read the objects gives the same answer to both, and
    scores on whichever scene happens to match the prior. This control says
    more than another aggregate: it is the difference between answering the
    question and recalling the template's usual answer.
    """
    by_question: Dict[str, List[Row]] = collections.defaultdict(list)
    for row in rows:
        by_question[row.question.strip().lower()].append(row)

    pairs = 0
    direct_tracks = executed_tracks = 0
    direct_same = executed_same = 0
    for group in by_question.values():
        answers = {r.reference_answer for r in group}
        if len(group) < 2 or len(answers) < 2:
            continue
        pairs += 1
        if len({r.direct_answer for r in group}) == 1:
            direct_same += 1
        if all(r.direct_correct for r in group):
            direct_tracks += 1
        if len({r.executed_answer for r in group}) == 1:
            executed_same += 1
        if all(r.executed_correct for r in group):
            executed_tracks += 1
    return {
        "n_question_groups": pairs,
        "direct_gave_one_answer_to_all": direct_same,
        "direct_correct_on_every_scene": direct_tracks,
        "executed_gave_one_answer_to_all": executed_same,
        "executed_correct_on_every_scene": executed_tracks,
    }


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

def run(
    items: Sequence[ManifestItem],
    scenes: Dict[str, Scene],
    direct: DirectQwenQA,
    parser: QwenQuestionParser,
    config: RegionConfig,
    verbose: bool = True,
) -> List[Row]:
    rows: List[Row] = []

    direct_answers = {a.question_id: a
                      for a in direct.answer_batch(items, scenes, verbose=verbose)}

    # Generate the programs in batches too. Asking for them one at a time
    # inside the loop below measured the batch size, not the mode: the first
    # run put the direct branch at 47 s and the executed branch at 228 s, and
    # the whole of that gap was that one batched eight prompts and the other
    # batched one.
    if verbose:
        print(f"executed: generating programs for {len(items)} questions")
    parser.warm_cache([item.question for item in items], verbose=verbose)

    if verbose:
        print(f"executed: executing {len(items)} programs")
    for index, item in enumerate(items):
        scene = scenes[item.sample_token]
        row = Row(
            question_id=item.question_id,
            sample_token=item.sample_token,
            scene_token=item.scene_token,
            family=item.family,
            template=item.template,
            question=item.question,
            reference_answer=item.answer,
            n_objects=len(scene.objects),
            scene_hash=scene_fingerprint(scene),
            model=direct.runtime.config.model_name,
            model_revision=direct.runtime.model_revision,
        )

        answer = direct_answers.get(item.question_id)
        if answer is not None:
            row.direct_answer = answer.answer
            row.direct_status = answer.status
            row.direct_raw = answer.raw
            row.direct_correct = _correct(answer.answer, item.answer)
            row.direct_input_tokens = answer.input_tokens
            row.direct_output_tokens = answer.output_tokens
            row.direct_seconds = answer.seconds

        attempt = parser.parse_detailed(item.question)
        # The generation's own time, not this loop's: by here the program has
        # already been produced in a batch, and timing the cache lookup would
        # report the executed branch as free.
        row.program_seconds = attempt.seconds
        row.program_raw = attempt.raw
        row.program_status = attempt.status
        row.program_valid = attempt.ok
        row.program_input_tokens = attempt.input_tokens
        row.program_output_tokens = attempt.output_tokens

        if attempt.ok and attempt.query is not None:
            row.program = attempt.query.describe()
            result = execute(attempt.query, scene, config)
            row.execution_status = result.status.value
            row.execution_reason = result.reason
            if result.ok:
                row.executed_answer = str(result.answer)
                row.executed_correct = _correct(result.answer, item.answer)
                row.selected = [o.token for o in result.selected if o.token][:40]
        else:
            row.execution_status = "not_attempted"
            row.execution_reason = attempt.reason

        rows.append(row)
        if verbose and (index + 1) % 50 == 0:
            print(f"  executed {index + 1}/{len(items)}", flush=True)

    return rows


def render(summary: Dict[str, object]) -> str:
    direct = summary["direct"]          # type: ignore[index]
    executed = summary["executed"]      # type: ignore[index]
    pair = summary["paired"]            # type: ignore[index]
    lines = [
        f"n = {summary['n']}",
        "",
        f"{'':<22}{'direct':>10}{'executed':>10}",
        f"{'accuracy (all)':<22}{direct['accuracy']:>10.3f}{executed['accuracy']:>10.3f}",
        f"{'answers':<22}{direct['answered_fraction']:>10.3f}"
        f"{executed['answered_fraction']:>10.3f}",
        f"{'accuracy when answered':<22}{direct['accuracy_when_answered']:>10.3f}"
        f"{executed['accuracy_when_answered']:>10.3f}",
        "",
        f"counting  {'':<12}{'direct':>10}{'executed':>10}",
        f"{'  numeric coverage':<22}{direct['count_errors']['numeric_coverage']:>10.3f}"
        f"{executed['count_errors']['numeric_coverage']:>10.3f}",
        f"{'  under':<22}{direct['count_errors']['under']:>10.2f}"
        f"{executed['count_errors']['under']:>10.2f}",
        f"{'  over':<22}{direct['count_errors']['over']:>10.2f}"
        f"{executed['count_errors']['over']:>10.2f}",
        f"{'  MAE':<22}{direct['count_errors']['mae']:>10.2f}"
        f"{executed['count_errors']['mae']:>10.2f}",
        f"{'  n numeric':<22}{direct['count_errors']['n_numeric']:>10d}"
        f"{executed['count_errors']['n_numeric']:>10d}",
        "",
        "paired:  both right %d | direct only %d | executed only %d | neither %d"
        % (pair["both"], pair["direct_only"], pair["executed_only"], pair["neither"]),
        "",
        f"direct statuses   : {direct['statuses']}",
        f"program statuses  : {executed['program_statuses']}",
        f"execution statuses: {executed['execution_statuses']}",
        "",
        f"cost: direct {direct['input_tokens']}in/{direct['output_tokens']}out "
        f"{direct['seconds']}s | executed {executed['input_tokens']}in/"
        f"{executed['output_tokens']}out {executed['seconds']}s",
        "",
        f"{'family':<28}{'n':>5}{'direct':>9}{'executed':>10}",
    ]
    for name, row in summary["by_family"].items():      # type: ignore[index]
        lines.append(f"{name:<28}{row['n']:>5}{row['direct']:>9.3f}{row['executed']:>10.3f}")
    lines.append("")
    lines.append(f"{'scene':<28}{'n':>5}{'direct':>9}{'executed':>10}")
    for name, row in summary["by_scene"].items():       # type: ignore[index]
        lines.append(f"{name[:26]:<28}{row['n']:>5}{row['direct']:>9.3f}{row['executed']:>10.3f}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--metadata-root", required=True)
    p.add_argument("--split", default="dev", choices=("dev", "eval"))
    p.add_argument("--model", default=None,
                   help="defaults to the frozen checkpoint in qwen_runtime")
    p.add_argument("--layout", default="nuscenes_qa")
    p.add_argument("--min-lidar-points", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-input-tokens", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="spatial_reasoning/results/llm_modes")
    p.add_argument("--direct-cache", default=None)
    p.add_argument("--program-cache", default=None)
    p.add_argument("--limit", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = Manifest.load(args.manifest)
    items = manifest.split(args.split)
    if args.limit:
        items = items[:args.limit]

    min_lidar = args.min_lidar_points
    if min_lidar is None:
        min_lidar = int(manifest.config.get("min_lidar_points", 0))
    scenes, report = load_nuscenes_scenes(args.metadata_root, min_lidar_points=min_lidar)
    config = RegionConfig(layout=args.layout)

    runtime_config = RuntimeConfig(
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        batch_size=args.batch_size,
        device=args.device,
        **({"model_name": args.model} if args.model else {}))
    runtime = QwenRuntime(runtime_config)

    os.makedirs(args.out, exist_ok=True)
    # Separate files, and separate keys. A shared cache would make the two
    # modes answer from each other's generations.
    direct_cache = args.direct_cache or os.path.join(args.out, "cache_direct.json")
    program_cache = args.program_cache or os.path.join(args.out, "cache_program.json")

    direct = DirectQwenQA(runtime=runtime, region_config=config,
                          cache_path=direct_cache,
                          coordinate_precision=int(
                              manifest.config.get("coordinate_precision", 1)))
    parser = QwenQuestionParser(
        model_name=runtime_config.model_name,
        cache_path=program_cache,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        batch_size=args.batch_size,
        directions=config.directions())

    print(f"{args.split}: {len(items)} examples over "
          f"{len({i.scene_token for i in items})} scenes")
    print(f"model: {runtime_config.model_name}  ({runtime_config.identity()})")
    print(f"objects: min_lidar_points={min_lidar} -> {report.annotations_kept} "
          f"annotations")
    print(f"layout: {args.layout} -> {config.directions()}\n")

    started = time.time()
    rows = run(items, scenes, direct, parser, config)
    summary = summarise(rows)
    summary["scene_dependence"] = scene_dependence(rows)

    print("\n" + render(summary))
    print("\nscene-dependence control:", json.dumps(summary["scene_dependence"]))

    records_path = os.path.join(args.out, f"rows_{args.split}.jsonl")
    with open(records_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict()) + "\n")
    summary_path = os.path.join(args.out, f"summary_{args.split}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "manifest": args.manifest,
            "split": args.split,
            "runtime": runtime_config.fingerprint(),
            "prompt_version": PROMPT_VERSION,
            "dsl_version": DSL_VERSION,
            "layout": args.layout,
            "min_lidar_points": min_lidar,
            "parser_stats": parser.stats.as_dict(),
            "wall_seconds": round(time.time() - started, 1),
            "summary": summary,
        }, fh, indent=2, default=str)
    print(f"\nwrote {records_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
