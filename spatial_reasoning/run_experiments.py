"""Run the full comparison and write the numbers the report quotes.

The experiment answers one question:

    At fixed perception and comparable budget, is it better to learn the
    answer to a spatial question directly, or to parse the question into an
    explicit program and compute the answer on the detected objects?

Four systems, one protocol, one set of object lists:

    1. question-only       -- language prior, no scene
    2. deep sets           -- learn the answer from question + object set
    3. deep sets + counter -- same, plus an explicit differentiable count
    4. symbolic program    -- parse the question, then compute

Nothing is trained on the test split, splits are by scene, and every system
sees exactly the same detections.

Usage
-----
    python -m spatial_reasoning.run_experiments --source synthetic
    python -m spatial_reasoning.run_experiments \\
        --source files --detections det.json --qa questions.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .baselines import LearnedSystem, TrainConfig, build_system
from .consistency import (
    build_contrast_pairs,
    counterfactual_accuracy,
    default_perturbation_sweep,
    make_counterfactuals,
    pair_accuracy,
    robustness_curve,
)
from .datasets import (
    AnswerVocabulary,
    HashingQuestionEncoder,
    QAItem,
    load_qa_json,
    split_by_scene,
    truncation_rate,
)
from .evaluate import (
    decompose_errors,
    evaluate_predictions,
    majority_baseline,
    measure_cost,
)
from .objects import Scene, load_nuscenes_detections, load_simple_scenes
from .programs import SymbolicSpatialQA
from .question_parser import Query, RuleBasedQuestionParser, coverage
from .regions import SECTOR_LAYOUTS, RegionConfig
from .synthetic import default_detector, generate_corpus

#: A system, seen by the evaluation harness: questions + scenes -> answers.
PredictFn = Callable[[Sequence[QAItem], Dict[str, Scene]], Sequence[Optional[str]]]

LEARNED_KINDS = ("question_only", "deepsets", "counter")


# --------------------------------------------------------------------------- #
# Adapters: give every system the same calling convention
# --------------------------------------------------------------------------- #

def symbolic_predictor(system: SymbolicSpatialQA) -> PredictFn:
    def predict(items: Sequence[QAItem], scenes: Dict[str, Scene]) -> List[Optional[str]]:
        return [
            system.answer(
                it.question,
                scenes.get(it.sample_token) or Scene(it.sample_token, []),
            ).answer
            for it in items
        ]
    return predict


def learned_predictor(system: LearnedSystem) -> PredictFn:
    def predict(items: Sequence[QAItem], scenes: Dict[str, Scene]) -> List[Optional[str]]:
        return list(system.predict(items, scenes))
    return predict


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_from_files(
    detections_path: str,
    qa_path: str,
    oracle_path: Optional[str] = None,
    min_score: float = 0.3,
) -> Tuple[Dict[str, Scene], Optional[Dict[str, Scene]], List[QAItem], Dict[str, Query]]:
    """Load real data: detections (perception, fixed) plus a QA file."""
    with open(detections_path, "r", encoding="utf-8") as fh:
        head = fh.read(2048)
    # The nuScenes submission format is a dict with a "results" key; our own
    # compact format is a bare list. Sniffing the first bytes avoids loading a
    # large file twice.
    if '"results"' in head:
        detected = load_nuscenes_detections(detections_path, min_score=min_score)
    else:
        detected = load_simple_scenes(detections_path)

    oracle = load_simple_scenes(oracle_path) if oracle_path else None
    scene_lookup = {t: s.scene_token for t, s in detected.items() if s.scene_token}
    items = load_qa_json(qa_path, scene_lookup)
    return detected, oracle, items, {}


# --------------------------------------------------------------------------- #
# Experiment stages
# --------------------------------------------------------------------------- #

def _evaluate_learned_systems(
    train: Sequence[QAItem],
    val: Sequence[QAItem],
    test: Sequence[QAItem],
    detected: Dict[str, Scene],
    vocab: AnswerVocabulary,
    question_encoder: HashingQuestionEncoder,
    config: TrainConfig,
    verbose: bool,
) -> Tuple[List[Dict], Dict[str, PredictFn]]:
    entries: List[Dict] = []
    predictors: Dict[str, PredictFn] = {}

    for kind in LEARNED_KINDS:
        system = build_system(kind, vocab, question_encoder, config)
        report = system.fit(train, detected, val or None)
        preds, cost = measure_cost(lambda s=system: s.predict(test, detected), len(test))
        entry = evaluate_predictions(
            kind, preds, test, cost,
            extra={
                "num_parameters": report.num_parameters,
                "train_seconds": round(report.train_seconds, 2),
                "epochs_run": report.epochs_run,
                "best_val_accuracy": round(report.best_val_accuracy, 4),
            },
        )
        entries.append(entry.as_dict())
        predictors[kind] = learned_predictor(system)
        if verbose:
            print(f"{kind:>18}: acc={entry.accuracy:.3f} "
                  f"mae={entry.count_mae} params={report.num_parameters}")

    return entries, predictors


def _evaluate_symbolic_systems(
    test: Sequence[QAItem],
    detected: Dict[str, Scene],
    parser: RuleBasedQuestionParser,
    majority_answer: str,
    region_config: RegionConfig,
    verbose: bool,
) -> Tuple[List[Dict], Dict[str, PredictFn]]:
    """Two variants: one that abstains, one that always answers.

    Reporting both keeps coverage and accuracy separate while still giving a
    number that is directly comparable to a model which never abstains.
    """
    entries: List[Dict] = []
    predictors: Dict[str, PredictFn] = {}

    for label, fallback in (("symbolic_abstain", None),
                            ("symbolic_fallback", majority_answer)):
        system = SymbolicSpatialQA(parser=parser, config=region_config,
                                   fallback_answer=fallback)
        predict = symbolic_predictor(system)
        preds, cost = measure_cost(lambda p=predict: p(test, detected), len(test))
        entry = evaluate_predictions(
            label, preds, test, cost,
            extra={"num_parameters": 0, "train_seconds": 0.0,
                   "fallback_answer": fallback},
        )
        entries.append(entry.as_dict())
        predictors[label] = predict
        if verbose:
            print(f"{label:>18}: acc={entry.accuracy:.3f} "
                  f"answered={entry.answered_fraction:.3f} mae={entry.count_mae}")

    return entries, predictors


def _run_probes(
    predictors: Dict[str, PredictFn],
    test: Sequence[QAItem],
    detected: Dict[str, Scene],
    parser: RuleBasedQuestionParser,
    seed: int,
    region_config: RegionConfig,
) -> Dict[str, object]:
    """Contrast pairs, counterfactual moves and the perturbation sweep."""
    names = [n for n in predictors if n != "symbolic_abstain"]

    pairs = build_contrast_pairs(test, seed=seed)
    counterfactuals = make_counterfactuals(test, detected, parser,
                                          config=region_config, seed=seed)
    sweep = default_perturbation_sweep(seed=seed)

    return {
        "contrast_pairs": {
            name: pair_accuracy(predictors[name], pairs, detected) for name in names
        },
        "counterfactuals": {
            "n": len(counterfactuals),
            "note": ("reference answers come from the same executor the symbolic "
                     "system uses, so its score here is a sanity check, not evidence"),
            "systems": {
                name: counterfactual_accuracy(predictors[name], counterfactuals,
                                              original_scenes=detected)
                for name in names
            },
        },
        "robustness": {
            name: robustness_curve(predictors[name], test, detected, sweep)
            for name in names
        },
    }


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #

def run(
    detected: Dict[str, Scene],
    oracle: Optional[Dict[str, Scene]],
    items: List[QAItem],
    gold_queries: Dict[str, Query],
    output_dir: str = "outputs/spatial_reasoning",
    seed: int = 0,
    epochs: int = 40,
    device: str = "cpu",
    allow_relative: bool = False,
    layout: str = "octant",
    verbose: bool = True,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    splits = split_by_scene(items, detected, seed=seed)
    warnings: List[str] = list(splits.pop("warnings", []))  # type: ignore[arg-type]
    train, val, test = splits["train"], splits["val"], splits["test"]
    if verbose:
        print(f"split sizes: train={len(train)} val={len(val)} test={len(test)}")
    for w in warnings:
        print(f"WARNING: {w}")
    if not test:
        raise RuntimeError("empty test split; check the scene grouping")

    vocab = AnswerVocabulary([it.answer for it in train])
    question_encoder = HashingQuestionEncoder(dim=256)
    train_config = TrainConfig(epochs=epochs, seed=seed, device=device, verbose=False)
    parser = RuleBasedQuestionParser(allow_relative_regions=allow_relative)
    region_config = RegionConfig(layout=layout)
    if verbose:
        print(f"sector layout: {layout} -> {region_config.directions()}")

    results: Dict[str, object] = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "data": {
            "n_items": len(items),
            "n_samples": len(detected),
            "train": len(train), "val": len(val), "test": len(test),
            "answer_vocabulary": len(vocab),
            # Descriptive only: the encoder no longer truncates. Kept to show
            # how many scenes would have been cut by the old cap of 64.
            "scenes_over_64_objects": truncation_rate(detected.values(), 64),
            "object_truncation_rate": 0.0,
            "warnings": warnings,
        },
        "seed": seed,
        "sector_layout": layout,
        "majority_baseline_accuracy": majority_baseline(train, test),
        # Share of test answers meaning "nothing is there". When this is high
        # the benchmark rewards a system that never looks at the scene, and the
        # headline accuracies below must be read against it.
        "trivial_answer_fraction": sum(
            1 for it in test if it.answer in ("0", "no")
        ) / len(test),
        "parser_coverage": coverage([it.question for it in test], parser),
    }

    learned_entries, predictors = _evaluate_learned_systems(
        train, val, test, detected, vocab, question_encoder, train_config, verbose
    )
    symbolic_entries, symbolic_predictors = _evaluate_symbolic_systems(
        test, detected, parser, vocab.majority(train), region_config, verbose
    )
    predictors.update(symbolic_predictors)
    results["systems"] = learned_entries + symbolic_entries

    results.update(_run_probes(predictors, test, detected, parser, seed, region_config))

    analysis = decompose_errors(
        SymbolicSpatialQA(parser=parser, config=region_config, fallback_answer=None),
        test, detected, oracle, gold_queries or None,
    )
    results["error_decomposition"] = {
        "counts": analysis.counts,
        "fractions": {k: round(v, 4) for k, v in analysis.fractions().items()},
        "examples": analysis.examples,
    }

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    md_path = os.path.join(output_dir, "results.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(results))

    if verbose:
        print(f"\nwrote {json_path}\nwrote {md_path}")
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def render_markdown(results: Dict) -> str:
    lines: List[str] = ["# Spatial reasoning at fixed perception", ""]

    data = results["data"]
    lines += [
        f"- items: {data['n_items']} over {data['n_samples']} samples "
        f"(train {data['train']} / val {data['val']} / test {data['test']})",
        f"- answer vocabulary: {data['answer_vocabulary']}",
        f"- majority-answer accuracy on test: {results['majority_baseline_accuracy']:.3f}",
        f"- answers meaning \"nothing there\" (0 / no): "
        f"{results['trivial_answer_fraction']:.3f}",
        f"- parser coverage on test: {results['parser_coverage']['coverage']:.3f}",
        "",
    ]
    for w in data.get("warnings", []):
        lines.append(f"> warning: {w}")
    if data.get("warnings"):
        lines.append("")

    lines += [
        "## Main comparison",
        "",
        "| system | accuracy | answered | acc. when answered | count MAE | params | ms/item |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in results["systems"]:
        mae = "-" if s["count_mae"] is None else f"{s['count_mae']:.3f}"
        lines.append(
            f"| {s['name']} | {s['accuracy']:.3f} | {s['answered_fraction']:.3f} | "
            f"{s['accuracy_when_answered']:.3f} | {mae} | "
            f"{s['extra'].get('num_parameters', 0)} | {s['cost']['ms_per_item']:.2f} |"
        )

    lines += ["", "## Contrast pairs (same question, different answers)", "",
              "| system | pair accuracy | item accuracy |", "|---|---|---|"]
    for name, v in results["contrast_pairs"].items():
        lines.append(f"| {name} | {v['pair_accuracy']:.3f} | {v['item_accuracy']:.3f} |")

    cf = results["counterfactuals"]
    lines += ["", f"## Counterfactual moves (n = {cf['n']})", "",
              f"> {cf['note']}", "",
              "| system | accuracy | answer changed |", "|---|---|---|"]
    for name, v in cf["systems"].items():
        lines.append(f"| {name} | {v['accuracy']:.3f} | {v['changed_fraction']:.3f} |")

    lines += ["", "## Robustness to perception failures", "",
              "| system | setting | accuracy |", "|---|---|---|"]
    for name, curve in results["robustness"].items():
        for point in curve:
            lines.append(f"| {name} | {point['setting']} | {point['accuracy']:.3f} |")

    ed = results["error_decomposition"]
    lines += ["", "## Where the symbolic system's errors come from", "",
              "| kind | count | fraction |", "|---|---|---|"]
    for kind, count in ed["counts"].items():
        lines.append(f"| {kind} | {count} | {ed['fractions'].get(kind, 0.0):.3f} |")

    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "files", "nuscenes"],
                   default="synthetic")
    p.add_argument("--metadata-root", type=str, default=None,
                   help="nuscenes source: directory holding sample.json (v1.0-mini)")
    p.add_argument("--min-lidar-points", type=int, default=1,
                   help="nuscenes source: drop annotations with fewer LiDAR returns")
    p.add_argument("--simulate-detector", action="store_true",
                   help="nuscenes source: pass the ground-truth boxes through the "
                        "detector-failure model instead of using them directly")
    p.add_argument("--detections", type=str, default=None,
                   help="detections JSON (nuScenes result format or simple format)")
    p.add_argument("--oracle", type=str, default=None,
                   help="ground-truth boxes in simple format, for error decomposition")
    p.add_argument("--qa", type=str, default=None, help="question/answer JSON")
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--output-dir", type=str, default="outputs/spatial_reasoning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--scenes", type=int, default=8, help="synthetic scenes")
    p.add_argument("--samples-per-scene", type=int, default=12)
    p.add_argument("--questions-per-sample", type=int, default=6)
    p.add_argument("--perfect-perception", action="store_true",
                   help="synthetic only: skip the detector noise model")
    p.add_argument("--max-trivial-fraction", type=float, default=0.35,
                   help="synthetic only: cap the share of '0'/'no' answers "
                        "(1.0 disables the balancing)")
    p.add_argument("--layout", default=None, choices=sorted(SECTOR_LAYOUTS),
                   help="sector convention; defaults to nuscenes_qa for "
                        "--source nuscenes and octant otherwise")
    p.add_argument("--allow-relative", action="store_true",
                   help="also parse questions anchored on another object")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.source == "synthetic":
        corpus = generate_corpus(
            n_scenes=args.scenes,
            samples_per_scene=args.samples_per_scene,
            questions_per_sample=args.questions_per_sample,
            seed=args.seed,
            detector=None if args.perfect_perception else default_detector(args.seed),
            max_trivial_fraction=args.max_trivial_fraction,
        )
        print("synthetic corpus:", json.dumps(corpus.summary(), indent=2))
        detected, oracle = corpus.detected_scenes, corpus.oracle_scenes
        items, gold = corpus.items, corpus.gold_queries
    elif args.source == "nuscenes":
        if not (args.metadata_root and args.qa):
            print("--metadata-root and --qa are required with --source nuscenes",
                  file=sys.stderr)
            return 2
        from .nuscenes_adapter import load_nuscenes_qa, load_nuscenes_scenes

        oracle, conversion = load_nuscenes_scenes(args.metadata_root,
                                                  args.min_lidar_points)
        print("nuScenes conversion:", json.dumps(conversion.as_dict(), indent=2))

        if args.simulate_detector:
            from .consistency import perturb_scenes
            detected = perturb_scenes(oracle, default_detector(args.seed))
            print("perception: ground truth passed through the detector-failure model")
        else:
            detected = oracle
            print("perception: ground-truth boxes (oracle condition)")

        items = load_nuscenes_qa(args.qa, oracle)
        gold = {}   # real questions have no gold program
        print(f"nuScenes-QA: {len(items)} questions over "
              f"{len({it.sample_token for it in items})} samples")
    else:
        if not (args.detections and args.qa):
            print("--detections and --qa are required with --source files",
                  file=sys.stderr)
            return 2
        detected, oracle, items, gold = load_from_files(
            args.detections, args.qa, args.oracle, args.min_score
        )

    layout = args.layout or ("nuscenes_qa" if args.source == "nuscenes" else "octant")
    run(detected, oracle, items, gold,
        output_dir=args.output_dir, seed=args.seed, epochs=args.epochs,
        device=args.device, allow_relative=args.allow_relative, layout=layout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
