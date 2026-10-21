"""Check the executor on correct programs, before any model is involved.

If a question is answered wrongly by the executed system, there are three
places the fault can lie: the model wrote the wrong program, the executor
computed the wrong value, or the reference answer disagrees with the
annotation for a reason neither of them controls. Running the full pipeline
cannot tell them apart, and attributing an executor bug to the model would
make the comparison a measurement of the wrong thing.

So this script removes the model. For every development example it takes a
*reference program* -- one produced by the deterministic rule parser and read
by hand, or written by hand in the override file where the parser's version is
wrong -- executes it on the frozen object list, and compares the result with
nuScenes-QA's answer.

The passing criterion is not a score. Tuning until the number looks good is
exactly the failure this step exists to prevent, and dropping the examples
that disagree is the same thing with extra steps. The criterion is:

    the operations are right on the controlled tests, and every disagreement
    on a real example has been examined and placed in a category.

So a disagreement is never silently dropped. It is written to the log with its
program, its trace and the objects it selected, and it has to be read. An
ambiguous reference or an annotation mismatch stays in the log and counts
against nothing -- but the reference answer is never edited to make the
executor succeed.

Run:

    python -m spatial_reasoning.verify_executor \\
        --manifest spatial_reasoning/manifests/first_comparison.json \\
        --metadata-root data/data/nuscenes/v1.0-mini
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from .dsl import ExecStatus, execute, lift
from .manifest import Manifest, ManifestItem, describe_objects, scene_fingerprint
from .nuscenes_adapter import load_nuscenes_scenes
from .objects import Scene
from .question_parser import DEFAULT_PARSER, RuleBasedQuestionParser
from .regions import RegionConfig


@dataclass
class Verification:
    """One example, its reference program, and what the executor made of it."""

    question_id: str
    question: str
    family: str
    sample_token: str
    scene_fingerprint: str
    gold: str
    program: Optional[str] = None
    program_source: str = "rule_parser"   # or "override"
    answer: Optional[str] = None
    status: str = "ok"
    reason: str = ""
    selected: List[str] = field(default_factory=list)
    n_selected: int = 0
    n_objects: int = 0
    agrees: bool = False
    category: str = ""        # filled in for disagreements

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Reference programs
# --------------------------------------------------------------------------- #

def load_overrides(path: Optional[str]) -> Dict[str, Dict[str, object]]:
    """Hand-written programs, keyed by ``question_id``.

    The rule parser is a convenience for producing candidates, not an oracle.
    Where reading its program shows it says something other than the question,
    the correct program goes here and this file is what the executor is
    checked on.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("programs", payload)


def verify(
    items: Sequence[ManifestItem],
    scenes: Dict[str, Scene],
    config: RegionConfig,
    parser: Optional[RuleBasedQuestionParser] = None,
    overrides: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Verification]:
    parser = parser or DEFAULT_PARSER
    overrides = overrides or {}
    out: List[Verification] = []

    for item in items:
        scene = scenes[item.sample_token]
        record = Verification(
            question_id=item.question_id,
            question=item.question,
            family=item.family,
            sample_token=item.sample_token,
            scene_fingerprint=scene_fingerprint(scene),
            gold=item.answer,
            n_objects=len(scene.objects),
        )

        override = overrides.get(item.question_id)
        if override is not None:
            record.program_source = "override"
            query = _query_from_override(override)
        else:
            parsed = parser.parse(item.question)
            if not parsed.ok:
                record.status = "no_reference_program"
                record.reason = parsed.reason
                record.category = "needs_hand_written_program"
                out.append(record)
                continue
            query = lift(parsed.query)

        record.program = query.describe()
        result = execute(query, scene, config)
        if result.status is not ExecStatus.OK or result.answer is None:
            record.status = result.status.value
            record.reason = result.reason
            out.append(record)
            continue

        record.answer = str(result.answer)
        record.n_selected = len(result.selected)
        record.selected = [o.token for o in result.selected if o.token][:40]
        record.agrees = record.answer.strip().lower() == item.answer
        out.append(record)

    return out


def _query_from_override(spec: Dict[str, object]):
    """Build a DSL query from a hand-written specification.

    Deliberately strict: an unknown key is an error rather than a default, so
    a typo in the override file cannot quietly become a different program and
    then be reported as a verified reference.
    """
    from .dsl import ANY, Op, Query, Selector
    from .regions import Region

    allowed = {"operation", "category", "status", "negate_status", "direction",
               "anchor", "second", "comparison"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"unknown keys in override: {sorted(unknown)}")

    def selector(block: Dict[str, object], anchored: bool = False) -> "Selector":
        region = ANY
        direction = block.get("direction")
        anchor_block = block.get("anchor")
        if direction:
            if anchor_block:
                region = Region(kind="relative", direction=str(direction),
                                reference_category=str(anchor_block["category"]),
                                frame="ego")
            else:
                region = Region(kind="direction", direction=str(direction))
        return Selector(
            category=str(block.get("category", "object")),
            status=block.get("status"),            # type: ignore[arg-type]
            negate_status=bool(block.get("negate_status", False)),
            region=region,
            anchor=(Selector(category=str(anchor_block["category"]),
                             status=anchor_block.get("status"))  # type: ignore[arg-type]
                    if anchor_block else None),
        )

    operation = str(spec["operation"])
    target = selector(spec)
    if operation == "count":
        return Query(Op.COUNT, target)
    if operation == "exists":
        return Query(Op.EXISTS, target)
    if operation == "compare":
        return Query(Op.COMPARE_COUNT, target,
                     selector(dict(spec["second"])),  # type: ignore[arg-type]
                     comparison=str(spec.get("comparison", "more")))
    raise ValueError(f"unsupported operation {operation!r}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def summarise(records: Sequence[Verification]) -> Dict[str, object]:
    total = len(records) or 1
    answered = [r for r in records if r.answer is not None]
    agreed = [r for r in answered if r.agrees]
    by_family = collections.defaultdict(lambda: [0, 0])
    for r in records:
        by_family[r.family][1] += 1
        if r.agrees:
            by_family[r.family][0] += 1
    return {
        "n": len(records),
        "answered": len(answered),
        "answered_fraction": len(answered) / total,
        "agreed": len(agreed),
        "agreement_over_all": len(agreed) / total,
        "agreement_when_answered": len(agreed) / (len(answered) or 1),
        "by_family": {k: {"agreed": v[0], "n": v[1]} for k, v in sorted(by_family.items())},
        "statuses": dict(collections.Counter(r.status for r in records).most_common()),
        "disagreement_categories": dict(collections.Counter(
            r.category for r in records if r.category).most_common()),
    }


def disagreements(records: Sequence[Verification]) -> List[Verification]:
    """Everything that did not agree, for reading. Nothing is filtered out."""
    return [r for r in records if not r.agrees]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--metadata-root", required=True)
    p.add_argument("--split", default="dev", choices=("dev", "eval"))
    p.add_argument("--overrides", default="spatial_reasoning/manifests/reference_programs.json")
    p.add_argument("--categories",
                   default="spatial_reasoning/manifests/disagreement_categories.json",
                   help="the examined categories; an uncategorised disagreement "
                        "is an error, not a footnote")
    p.add_argument("--layout", default="nuscenes_qa")
    p.add_argument("--min-lidar-points", type=int, default=None,
                   help="defaults to the value recorded in the manifest")
    p.add_argument("--out", default="spatial_reasoning/results/executor_verification")
    p.add_argument("--limit", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = Manifest.load(args.manifest)

    min_lidar = args.min_lidar_points
    if min_lidar is None:
        min_lidar = int(manifest.config.get("min_lidar_points", 0))
    scenes, report = load_nuscenes_scenes(args.metadata_root, min_lidar_points=min_lidar)
    config = RegionConfig(layout=args.layout)

    items = manifest.split(args.split)
    if args.limit:
        items = items[:args.limit]
    print(f"{args.split}: {len(items)} examples over "
          f"{len({i.scene_token for i in items})} scenes")
    print(f"objects: min_lidar_points={min_lidar} -> {report.annotations_kept} "
          f"annotations over {report.samples} samples")
    print(f"layout: {args.layout} -> {config.directions()}\n")

    overrides = load_overrides(args.overrides)
    if overrides:
        print(f"hand-written programs: {len(overrides)}\n")

    records = verify(items, scenes, config, overrides=overrides)

    # Attach the examined categories. The passing criterion for this step is
    # that every disagreement has been read and placed, so an unexplained one
    # is reported loudly rather than averaged away.
    categories = {}
    if os.path.exists(args.categories):
        with open(args.categories, "r", encoding="utf-8") as fh:
            categories = json.load(fh).get("categories", {})
    for r in records:
        if not r.agrees and not r.category:
            entry = categories.get(r.question_id)
            r.category = entry["category"] if entry else "UNEXAMINED"

    stats = summarise(records)

    print(f"agreement: {stats['agreed']}/{stats['n']} = "
          f"{stats['agreement_over_all']:.1%} over all, "
          f"{stats['agreement_when_answered']:.1%} over the "
          f"{stats['answered']} answered")
    print("by family:")
    for family, row in stats["by_family"].items():
        print(f"   {family:<26} {row['agreed']:>3}/{row['n']:<3} "
              f"{row['agreed']/max(row['n'],1):6.1%}")
    print("outcomes:", stats["statuses"])

    bad = disagreements(records)
    unexamined = [r for r in bad if r.category == "UNEXAMINED"]
    print(f"\n{len(bad)} disagreement(s); {len(bad) - len(unexamined)} examined "
          f"and categorised, {len(unexamined)} not. None is dropped.")
    for category, n in stats["disagreement_categories"].items():
        print(f"   {category:<42} {n}")
    if unexamined:
        print("\nUNEXAMINED -- this step does not pass until each is read:")
        for r in unexamined:
            print(f"   {r.question_id}  {r.question}")
    for r in bad[:15]:
        print(f"\n  [{r.family}] {r.question}")
        print(f"    program : {r.program or '<none>'}  ({r.program_source})")
        print(f"    executed: {r.answer!r}   gold: {r.gold!r}   "
              f"status: {r.status}  {r.reason[:60]}")
        print(f"    objects : {r.n_selected} selected of {r.n_objects}")

    os.makedirs(args.out, exist_ok=True)
    records_path = os.path.join(args.out, f"records_{args.split}.jsonl")
    with open(records_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict()) + "\n")
    summary_path = os.path.join(args.out, f"summary_{args.split}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({"manifest": args.manifest, "split": args.split,
                   "min_lidar_points": min_lidar, "layout": args.layout,
                   "conversion": report.as_dict(), "summary": stats},
                  fh, indent=2, default=str)
    print(f"\nwrote {records_path}\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
