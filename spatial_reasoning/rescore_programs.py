"""Re-execute archived programs under another sector layout, on CPU.

A program names a relation ("front_left"); what that relation selects is the
executor's convention, not the model's. So when the convention changes, the
program mode does not need a GPU: the saved programs are parsed again by the
same validator and executed again by the same executor, under the new layout.

The direct mode cannot be treated this way. Its prompt *stated* the convention,
so a direct answer given under ``direct-1`` is an answer to a differently
specified task. Those answers are carried over unchanged and still compared
with the same references -- which is the published direct number -- and the
summary says so. A direct number under the new convention needs a new run.

Nothing in ``results/<run>/`` is modified. The output goes to
``results/rescored_<layout>/<run>/``, with the hashes of what it was made from.

Before trusting a replay under a new layout, the script replays every run under
the layout it was archived with and requires the result to be identical to the
archive, field for field. If that check fails, nothing is written.

    python -m spatial_reasoning.rescore_programs \\
        --metadata-root data/data/nuscenes/v1.0-mini
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .compare_llm_modes import (Row, fill_executed, fill_functional,
                                scene_dependence, summarise)
from .functional_parser import FunctionalQwenParser
from .llm_parser import QwenQuestionParser
from .nuscenes_adapter import load_nuscenes_scenes
from .objects import Scene
from .regions import RegionConfig

#: Every archived evaluation run. The development run is included: the layout
#: was confirmed on it, and it is the only split a choice may be read from.
RUNS = (
    "llm_modes/summary_dev.json", "llm_modes/summary_eval.json",
    "llm_modes_7b/summary_eval.json", "llm_modes_32b/summary_eval.json",
    "wide_1p5b/summary_eval.json", "gw_qwen3_32b/summary_eval.json",
    "gw_qwen3_next_80b/summary_eval.json", "gw_qwen3p8_27b/summary_eval.json",
)

EXECUTED_FIELDS = ("program_status", "program_valid", "program",
                   "executed_answer", "execution_status", "executed_correct",
                   "selected")
FUNCTIONAL_FIELDS = ("functional_status", "functional_valid",
                     "functional_program", "functional_answer",
                     "functional_execution_status", "functional_correct")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(raw_rows: Sequence[Dict[str, object]], scenes: Dict[str, Scene],
           config: RegionConfig) -> List[Row]:
    """Parse and execute every saved program again under ``config``.

    The parsers are the ones the live run uses, fed from the archive instead
    of a model: their caches are filled with the saved generations, so
    validation, refusal handling and execution cannot drift from the original.
    """
    parser = QwenQuestionParser(directions=config.directions(), allow_foreign=True)
    functional: Optional[FunctionalQwenParser] = None
    if any(r.get("functional_status") for r in raw_rows):
        functional = FunctionalQwenParser(region_config=config)

    for r in raw_rows:
        parser._cache[str(r["question"])] = {
            "raw": r["program_raw"], "config": "archived",
            "input_tokens": r["program_input_tokens"],
            "output_tokens": r["program_output_tokens"],
            "seconds": r["program_seconds"]}
        if functional is not None:
            functional._cache[str(r["question"])] = {
                "raw": r["functional_raw"], "config": functional.config_id(),
                "input_tokens": r["functional_input_tokens"],
                "output_tokens": r["functional_output_tokens"],
                "seconds": r["functional_seconds"]}

    out: List[Row] = []
    for r in raw_rows:
        row = Row(**copy.deepcopy(dict(r)))
        scene = scenes[row.sample_token]
        fill_executed(row, parser.parse_detailed(row.question), scene, config)
        if functional is not None:
            fill_functional(row, functional.parse_detailed(row.question),
                            scene, config)
        out.append(row)
    return out


def differences(raw_rows: Sequence[Dict[str, object]], rows: Sequence[Row],
                fields: Sequence[str]) -> List[Tuple[str, str, object, object]]:
    found = []
    for saved, row in zip(raw_rows, rows):
        for name in fields:
            if saved[name] != getattr(row, name):
                found.append((row.question_id, name, saved[name], getattr(row, name)))
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-root", required=True)
    p.add_argument("--layout", default="nuscenes_qa_as_labelled")
    p.add_argument("--results", type=Path,
                   default=Path(__file__).resolve().parent / "results")
    p.add_argument("--out", type=Path, default=None,
                   help="defaults to <results>/rescored_<layout suffix>")
    args = p.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    out_root = args.out or args.results / (
        "rescored_" + args.layout.replace("nuscenes_qa_", ""))
    config = RegionConfig(layout=args.layout)
    scene_cache: Dict[int, Dict[str, Scene]] = {}

    print(f"{'run':<34}{'n':>5}  {'direct-1':>9}{'flat old':>10}{'flat new':>10}"
          f"{'typed old':>11}{'typed new':>11}")
    for relative in RUNS:
        summary_path = args.results / relative
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        split = saved["split"]
        rows_path = summary_path.with_name(f"rows_{split}.jsonl")
        raw_rows = [json.loads(line) for line in
                    rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        min_lidar = int(saved.get("min_lidar_points", 0))
        if min_lidar not in scene_cache:
            scene_cache[min_lidar], _ = load_nuscenes_scenes(
                args.metadata_root, min_lidar_points=min_lidar)
        scenes = scene_cache[min_lidar]

        archived_layout = saved.get("layout", "nuscenes_qa")
        has_functional = any(r.get("functional_status") for r in raw_rows)
        fields = EXECUTED_FIELDS + (FUNCTIONAL_FIELDS if has_functional else ())
        check = differences(
            raw_rows, replay(raw_rows, scenes, RegionConfig(layout=archived_layout)),
            fields)
        if check:
            raise SystemExit(
                f"{relative}: replay under the archived layout {archived_layout!r} "
                f"differs from the archive in {len(check)} field(s), e.g. {check[:3]}. "
                f"Nothing written.")

        rows = replay(raw_rows, scenes, config)
        summary = summarise(rows)
        summary["scene_dependence"] = scene_dependence(rows)

        target = out_root / Path(relative).parent
        target.mkdir(parents=True, exist_ok=True)
        new_rows = target / f"rows_{split}.jsonl"
        new_rows.write_text(
            "".join(json.dumps(r.to_dict()) + "\n" for r in rows), encoding="utf-8")
        meta = {k: v for k, v in saved.items() if k != "summary"}
        meta.update({
            "layout": args.layout,
            "rescored_from": {
                "summary": str(summary_path.relative_to(root)).replace("\\", "/"),
                "sha256": {"summary": _sha(summary_path), "rows": _sha(rows_path)},
                "layout": archived_layout,
                "replay_check": f"identical to the archive under {archived_layout} "
                                f"({len(raw_rows)} rows, fields: {', '.join(fields)})",
            },
            "direct_note": (
                "direct_* fields are the archived answers, unchanged. They were "
                f"given under prompt {saved.get('prompt_version', 'direct-1')}, "
                f"which states the {archived_layout} convention; a direct result "
                "under this layout needs a new run."),
            "changed_rows": {
                "executed": sum(a["executed_answer"] != b.executed_answer
                                for a, b in zip(raw_rows, rows)),
                **({"functional": sum(a["functional_answer"] != b.functional_answer
                                      for a, b in zip(raw_rows, rows))}
                   if has_functional else {}),
            },
            "summary": summary,
        })
        (target / f"summary_{split}.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")

        old = saved["summary"]
        typed_old = f"{old['functional']['accuracy']:>11.3f}" if has_functional else f"{'':>11}"
        typed_new = f"{summary['functional']['accuracy']:>11.3f}" if has_functional else f"{'':>11}"
        print(f"{str(Path(relative).parent) + ' ' + split:<34}{len(rows):>5}  "
              f"{summary['direct']['accuracy']:>9.3f}"
              f"{old['executed']['accuracy']:>10.3f}"
              f"{summary['executed']['accuracy']:>10.3f}{typed_old}{typed_new}")
    print(f"\nwrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
