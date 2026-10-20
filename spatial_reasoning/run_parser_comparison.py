"""Rule-based grammar versus a frozen language model, same executor.

The real-data run showed that the narrow grammar answers 31% of nuScenes-QA
and that its failures are refusals, not wrong computations. This experiment
changes one thing -- how the question becomes a program -- and holds everything
else fixed: the same object lists, the same DSL, the same parameter-free
executor, the same questions.

    question --[ parser ]--> program --[ executor ]--> answer
                 ^^^^^^
                 the only variable

Two learned references are kept for scale: the majority answer, and the
question-only prior measured in the main experiment.

Usage
-----
    python -m spatial_reasoning.run_parser_comparison \\
        --metadata-root /content/nuscenes/v1.0-mini \\
        --qa filtered_nuscenes_qa_train.json filtered_nuscenes_qa_validation.json \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --cache /content/qwen_cache.json --out /content/parser_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

from .datasets import QAItem
from .dsl import Query, execute, lift
from .nuscenes_adapter import load_nuscenes_scenes
from .objects import Scene
from .question_parser import RuleBasedQuestionParser


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_filtered_qa(paths: Sequence[str], scenes: Dict[str, Scene]) -> List[QAItem]:
    """Load the official nuScenes-QA files filtered to the mini split."""
    items: List[QAItem] = []
    missing = 0
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        records = payload.get("questions", payload) if isinstance(payload, dict) else payload
        for r in records:
            token = r.get("sample_token")
            if token is None:
                continue
            if token not in scenes:
                missing += 1
                continue
            items.append(QAItem(
                question=r["question"],
                answer=r["answer"],
                sample_token=token,
                scene_token=scenes[token].scene_token,
                template=f"{r.get('template_type', '?')}/hop{r.get('num_hop', 0)}",
            ))
    if missing:
        print(f"[qa] dropped {missing} questions with no matching sample")
    return items


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_parser(
    name: str,
    parse_fn,
    items: Sequence[QAItem],
    scenes: Dict[str, Scene],
    verbose: bool = True,
) -> Dict:
    """Run one parser end to end and record where every item ended up."""
    start = time.perf_counter()

    outcomes = Counter()
    per_template = defaultdict(lambda: {"n": 0, "correct": 0, "answered": 0})
    exec_failures = Counter()
    wrong_examples: List[str] = []
    correct = answered = 0

    for item in items:
        bucket = per_template[item.template]
        bucket["n"] += 1

        query: Optional[Query] = parse_fn(item.question)
        if query is None:
            outcomes["declined_by_parser"] += 1
            continue

        result = execute(query, scenes[item.sample_token])
        if not result.ok:
            outcomes["declined_by_executor"] += 1
            exec_failures[result.status.value] += 1
            continue

        answered += 1
        bucket["answered"] += 1
        if str(result.answer).strip().lower() == item.answer:
            correct += 1
            bucket["correct"] += 1
            outcomes["correct"] += 1
        else:
            outcomes["wrong"] += 1
            if len(wrong_examples) < 15:
                wrong_examples.append(
                    f"{item.question} | program={query.describe()} | "
                    f"got={result.answer} want={item.answer}"
                )

    n = max(len(items), 1)
    report = {
        "name": name,
        "n": len(items),
        "accuracy": correct / n,
        "answered_fraction": answered / n,
        "accuracy_when_answered": (correct / answered) if answered else 0.0,
        "outcomes": dict(outcomes),
        "executor_failures": dict(exec_failures),
        "seconds": round(time.perf_counter() - start, 1),
        "by_template": {
            k: {
                "n": v["n"],
                "answered": v["answered"] / max(v["n"], 1),
                "accuracy": v["correct"] / max(v["n"], 1),
            }
            for k, v in sorted(per_template.items())
        },
        "wrong_examples": wrong_examples,
    }
    if verbose:
        print(f"{name:>22}: acc={report['accuracy']:.3f} "
              f"answered={report['answered_fraction']:.3f} "
              f"acc|answered={report['accuracy_when_answered']:.3f} "
              f"({report['seconds']}s)")
    return report


def majority_accuracy(items: Sequence[QAItem]) -> float:
    counts = Counter(it.answer for it in items)
    if not counts:
        return 0.0
    best = counts.most_common(1)[0][0]
    return sum(1 for it in items if it.answer == best) / len(items)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-root", required=True)
    p.add_argument("--qa", nargs="+", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--cache", default=None, help="parse cache (JSON)")
    p.add_argument("--out", default="outputs/parser_comparison")
    p.add_argument("--limit", type=int, default=0, help="debug: cap the item count")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--skip-llm", action="store_true")
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)

    scenes, conversion = load_nuscenes_scenes(args.metadata_root)
    print("nuScenes:", json.dumps(conversion.as_dict()))
    with_status = sum(1 for s in scenes.values() for o in s.objects if o.status)
    total_objects = sum(len(s.objects) for s in scenes.values())
    print(f"objects carrying a status attribute: {with_status}/{total_objects}")

    items = load_filtered_qa(args.qa, scenes)
    if args.limit:
        items = items[:args.limit]
    print(f"questions: {len(items)} over "
          f"{len({i.sample_token for i in items})} samples\n")

    results = {
        "data": {
            "questions": len(items),
            "samples": len({i.sample_token for i in items}),
            "conversion": conversion.as_dict(),
            "objects_with_status": with_status,
            "objects_total": total_objects,
            "templates": dict(Counter(i.template for i in items)),
        },
        "majority_accuracy": majority_accuracy(items),
        "parsers": [],
    }
    print(f"majority answer: {results['majority_accuracy']:.3f}\n")

    # -- rule-based grammar, lifted into the shared DSL -------------------- #
    rule_parser = RuleBasedQuestionParser(allow_relative_regions=True)

    def parse_with_rules(question: str) -> Optional[Query]:
        parsed = rule_parser.parse(question)
        return lift(parsed.query) if parsed.ok else None

    results["parsers"].append(
        evaluate_parser("rule_grammar", parse_with_rules, items, scenes))

    # -- frozen language model --------------------------------------------- #
    if not args.skip_llm:
        from .llm_parser import QwenQuestionParser

        llm = QwenQuestionParser(model_name=args.model, cache_path=args.cache,
                                 batch_size=args.batch_size,
                                 max_new_tokens=args.max_new_tokens)
        print(f"\nwarming the parse cache with {args.model} ...")
        llm.warm_cache([it.question for it in items])

        report = evaluate_parser(f"llm::{args.model}", llm.parse, items, scenes)
        report["parser_stats"] = llm.stats.as_dict()
        results["parsers"].append(report)
        print("parser stats:", json.dumps(report["parser_stats"]))

    path = os.path.join(args.out, "parser_comparison.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
