"""Two properties of the nuScenes-QA answers, tested on the official release.

Neither test executes a program, and the first does not even read a coordinate.

**Laterality.** For every counting or existence question that names exactly
one spatial relation, tally how often the answer is ``0`` or ``no``, by
relation. If the labels used signed sectors, "front right" would hold objects
about as often as "front left". In the release it never does.

**Things.** For every "How many things are there?", compare the answer with
the number of annotations of the sample, counted three ways. The answer is the
number of annotations of *every* category, zero-LiDAR ones included, plus one:
the ego vehicle is a thing.

    python -m spatial_reasoning.audit_labels \\
        --questions data/data/nuscenes_qa/NuScenes_train_questions.json \\
                    data/data/nuscenes_qa/NuScenes_val_questions.json \\
        --metadata-root data/data/nuscenes/v1.0-trainval
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .objects import canonical_category

RELATIONS = ("front left", "front right", "back left", "back right", "front", "back")
_ONE_RELATION = re.compile(r"to the (?:front|back)(?: left| right)? of")
_COUNT = re.compile(r"how many|what number")
_EXIST = re.compile(r"^(there is [^;]+; )?(are there any|are any)")
_THINGS = re.compile(r"(how many|what number of) things are there\?")


def read_questions(path: str) -> List[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["questions"] if isinstance(payload, dict) else payload
    return [{"sample_token": r.get("sample_token") or r.get("token"),
             "question": r["question"], "answer": str(r["answer"])} for r in rows]


def laterality(rows: Iterable[dict]) -> Dict[str, Dict[str, int]]:
    table = {r: {"n": 0, "empty": 0} for r in RELATIONS}
    for row in rows:
        q = row["question"].lower()
        if len(_ONE_RELATION.findall(q)) != 1:
            continue
        if not (_COUNT.search(q) or _EXIST.match(q)):
            continue
        relation = next(r for r in RELATIONS if re.search(rf"to the {r} of", q))
        table[relation]["n"] += 1
        table[relation]["empty"] += row["answer"] in ("0", "no")
    return table


def things(rows: Iterable[dict], metadata_root: str) -> Dict[str, object]:
    root = Path(metadata_root)
    category = {c["token"]: c["name"] for c in json.loads((root / "category.json").read_text())}
    instance = {i["token"]: category[i["category_token"]]
                for i in json.loads((root / "instance.json").read_text())}
    every = collections.Counter()
    named = collections.Counter()
    seen_by_lidar = collections.Counter()
    for ann in json.loads((root / "sample_annotation.json").read_text()):
        sample = ann["sample_token"]
        every[sample] += 1
        named[sample] += canonical_category(instance[ann["instance_token"]]) is not None
        seen_by_lidar[sample] += ann["num_lidar_pts"] > 0
    asked = [r for r in rows if _THINGS.fullmatch(r["question"].lower())
             and r["sample_token"] in every]
    hypotheses = {
        "every annotation": lambda s: every[s],
        "every annotation + ego": lambda s: every[s] + 1,
        "ten named classes": lambda s: named[s],
        "ten named classes + ego": lambda s: named[s] + 1,
        "annotations with a LiDAR return + ego": lambda s: seen_by_lidar[s] + 1,
    }
    return {"n": len(asked),
            "answer_equals": {name: sum(int(r["answer"]) == f(r["sample_token"])
                                        for r in asked)
                              for name, f in hypotheses.items()}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--questions", nargs="+", required=True)
    p.add_argument("--metadata-root", default=None,
                   help="needed for the 'things' test only")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    result: Dict[str, object] = {"files": {}}
    for path in args.questions:
        rows = read_questions(path)
        entry: Dict[str, object] = {
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "n_questions": len(rows),
            "n_samples": len({r["sample_token"] for r in rows}),
            "laterality": laterality(rows),
        }
        if args.metadata_root:
            entry["things"] = things(rows, args.metadata_root)
        result["files"][Path(path).name] = entry          # type: ignore[index]

        print(f"\n{Path(path).name}: {len(rows)} questions, {entry['n_samples']} samples")
        print(f"  {'relation':<12}{'questions':>10}{'answer 0/no':>13}{'share':>8}")
        for relation, cell in entry["laterality"].items():   # type: ignore[union-attr]
            share = cell["empty"] / cell["n"] if cell["n"] else float("nan")
            print(f"  {relation:<12}{cell['n']:>10}{cell['empty']:>13}{share:>8.3f}")
        if "things" in entry:
            t = entry["things"]
            print(f"  'How many things are there?': {t['n']} questions")   # type: ignore[index]
            for name, hits in t["answer_equals"].items():                   # type: ignore[index]
                print(f"    answer = {name:<40}{hits:>6}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
