"""Tables that compare the archived runs under the two layouts, offline.

Reads only saved rows: the archive (``results/<run>``) and its re-execution
under the as-labelled layout (``results/rescored_as_labelled/<run>``). No
model, no nuScenes metadata.

1. **Headline** -- direct, flat and typed accuracy under each layout.
2. **Fallback** -- answer with the program when it executes, otherwise with the
   direct answer (typed -> flat -> direct on the wide scope). Every question
   is answered by exactly one mode, chosen without looking at the reference.
3. **Ceiling** -- questions the program mode gets wrong for *every* model of
   the narrow comparison, by family, under each layout.
4. **Relations** -- program accuracy by the relation the question names.

The direct column is the archived ``direct-1`` answers in both halves; see
``rescore_programs`` for why that mode cannot be rescored.

    python -m spatial_reasoning.analyse_rescored
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

NARROW = ("llm_modes", "llm_modes_7b", "llm_modes_32b", "gw_qwen3_32b",
          "gw_qwen3_next_80b", "gw_qwen3p8_27b")
WIDE = ("wide_1p5b",)
LABELS = {
    "llm_modes": "Qwen2.5 1.5B", "llm_modes_7b": "Qwen2.5 7B",
    "llm_modes_32b": "Qwen2.5 32B (int4)", "gw_qwen3_32b": "Qwen3 32B",
    "gw_qwen3_next_80b": "Qwen3-Next 80B", "gw_qwen3p8_27b": "Qwen3.8 27B",
    "wide_1p5b": "Qwen2.5 1.5B, wide scope",
}
RELATIONS = ("front left", "front right", "back left", "back right", "front", "back")


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def relation_of(question: str) -> Optional[str]:
    """The relation a question names, when it names exactly one."""
    q = question.lower()
    if len(re.findall(r"to the (?:front|back)(?: left| right)? of", q)) != 1:
        return None
    return next(r for r in RELATIONS if re.search(rf"to the {r} of", q))


def correct(answer, reference) -> bool:
    return answer is not None and str(answer).strip().lower() == str(reference).strip().lower()


def fallback(rows: Sequence[dict], order: Sequence[str]) -> Dict[str, object]:
    """First mode in ``order`` that produced an answer; the last always counts.

    The choice reads whether a mode answered, never whether it was right.
    """
    used = collections.Counter()
    right = 0
    for r in rows:
        for mode in order:
            answer = r.get(f"{mode}_answer")
            if answer is not None or mode == order[-1]:
                used[mode] += 1
                right += correct(answer, r["reference_answer"])
                break
    return {"order": list(order), "correct": right, "n": len(rows),
            "accuracy": right / len(rows), "answered_by": dict(used)}


def accuracy(rows: Sequence[dict], mode: str) -> float:
    return sum(bool(r[f"{mode}_correct"]) for r in rows) / len(rows)


def analyse(results: Path, rescored: Path) -> Dict[str, object]:
    out: Dict[str, object] = {"layouts": {"old": "nuscenes_qa",
                                          "new": "nuscenes_qa_as_labelled"}}
    data = {}
    for run in NARROW + WIDE:
        data[run] = {"old": load(results / run / "rows_eval.jsonl"),
                     "new": load(rescored / run / "rows_eval.jsonl")}
        ids = [r["question_id"] for r in data[run]["old"]]
        assert ids == [r["question_id"] for r in data[run]["new"]], run

    headline = {}
    for run, both in data.items():
        entry: Dict[str, object] = {"label": LABELS[run], "n": len(both["old"]),
                                    "direct_1": accuracy(both["old"], "direct")}
        for layout, rows in both.items():
            entry[f"flat_{layout}"] = accuracy(rows, "executed")
            entry[f"flat_answered_{layout}"] = sum(
                r["executed_answer"] is not None for r in rows) / len(rows)
            if run in WIDE:
                entry[f"typed_{layout}"] = accuracy(rows, "functional")
        headline[run] = entry
    out["headline"] = headline

    fall = {}
    for run, both in data.items():
        chains = ([("executed", "direct")] if run in NARROW else
                  [("functional", "direct"), ("executed", "direct"),
                   ("functional", "executed", "direct")])
        fall[run] = {layout: {" -> ".join(c): fallback(rows, c) for c in chains}
                     for layout, rows in both.items()}
    out["fallback"] = fall

    ceiling = {}
    for layout in ("old", "new"):
        wrong_everywhere = None
        for run in NARROW:
            wrong = {r["question_id"] for r in data[run][layout]
                     if not r["executed_correct"]}
            wrong_everywhere = wrong if wrong_everywhere is None else wrong_everywhere & wrong
        index = {r["question_id"]: r for r in data[NARROW[0]][layout]}
        families = collections.Counter(index[q]["family"] for q in wrong_everywhere)
        ceiling[layout] = {
            "n_wrong_for_all_six": len(wrong_everywhere),
            "ceiling_accuracy": 1 - len(wrong_everywhere) / len(index),
            "by_family": dict(sorted(families.items())),
            "questions": [
                {"question_id": q, "family": index[q]["family"],
                 "question": index[q]["question"],
                 "reference": index[q]["reference_answer"],
                 "executed": {run: next(r["executed_answer"] for r in data[run][layout]
                                        if r["question_id"] == q) for run in NARROW}}
                for q in sorted(wrong_everywhere)],
        }
    family_n = collections.Counter(r["family"] for r in data[NARROW[0]]["old"])
    out["ceiling"] = {"family_sizes": dict(sorted(family_n.items())), **ceiling}

    relations = {}
    for run in NARROW:
        table = {}
        for layout in ("old", "new"):
            for r in data[run][layout]:
                name = relation_of(r["question"])
                if name is None:
                    continue
                cell = table.setdefault(name, {"n": 0, "old": 0, "new": 0})
                cell[layout] += bool(r["executed_correct"])
                if layout == "old":
                    cell["n"] += 1
        relations[run] = table
    out["relations"] = relations
    return out


def render(a: Dict[str, object]) -> str:
    lines = ["# Archived programs, re-executed under the as-labelled layout", "",
             "Generated by `python -m spatial_reasoning.analyse_rescored`. "
             "Evaluation split. The direct column is the archived `direct-1` "
             "answers, which stated the published convention.", "",
             "## Headline", "",
             "| run | n | direct-1 | flat, published layout | flat, as labelled | "
             "typed, published | typed, as labelled |", "|---|---:|---:|---:|---:|---:|---:|"]
    for run, e in a["headline"].items():          # type: ignore[union-attr]
        typed = (f"{e['typed_old']:.3f} | {e['typed_new']:.3f}"
                 if "typed_old" in e else " | ")
        lines.append(f"| {e['label']} | {e['n']} | {e['direct_1']:.3f} | "
                     f"{e['flat_old']:.3f} | {e['flat_new']:.3f} | {typed} |")
    lines += ["", "## Fallback: the program when it executes, otherwise the next mode", "",
              "| run | chain | published layout | as labelled | answered by (as labelled) |",
              "|---|---|---:|---:|---|"]
    for run, both in a["fallback"].items():       # type: ignore[union-attr]
        for chain in both["old"]:
            o, n = both["old"][chain], both["new"][chain]
            lines.append(f"| {LABELS[run]} | {chain} | {o['correct']}/{o['n']} = "
                         f"{o['accuracy']:.3f} | {n['correct']}/{n['n']} = "
                         f"{n['accuracy']:.3f} | {n['answered_by']} |")
    c = a["ceiling"]                               # type: ignore[index]
    lines += ["", "## Ceiling: wrong in program mode for all six models", "",
              "| family | n | published layout | as labelled |", "|---|---:|---:|---:|"]
    for family, n in c["family_sizes"].items():
        lines.append(f"| {family} | {n} | {c['old']['by_family'].get(family, 0)} | "
                     f"{c['new']['by_family'].get(family, 0)} |")
    lines.append(f"| **all** | {sum(c['family_sizes'].values())} | "
                 f"**{c['old']['n_wrong_for_all_six']}** | "
                 f"**{c['new']['n_wrong_for_all_six']}** |")
    lines += ["", f"Best reachable accuracy: {c['old']['ceiling_accuracy']:.3f} under "
              f"the published layout, {c['new']['ceiling_accuracy']:.3f} as labelled.", "",
              "Questions still wrong for all six, as labelled:", ""]
    for q in c["new"]["questions"]:
        lines.append(f"- `{q['question_id']}` [{q['family']}] {q['question']} "
                     f"(reference {q['reference']!r}; executed "
                     f"{sorted(set(map(str, q['executed'].values())))})")
    lines += ["", "## Program accuracy by the relation the question names", "",
              "| run | relation | n | published layout | as labelled |",
              "|---|---|---:|---:|---:|"]
    for run, table in a["relations"].items():     # type: ignore[union-attr]
        for name in RELATIONS:
            if name in table:
                cell = table[name]
                lines.append(f"| {LABELS[run]} | {name} | {cell['n']} | "
                             f"{cell['old']} | {cell['new']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent / "results"
    p.add_argument("--results", type=Path, default=here)
    p.add_argument("--rescored", type=Path, default=here / "rescored_as_labelled")
    args = p.parse_args(argv)
    result = analyse(args.results, args.rescored)
    (args.rescored / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    text = render(result)
    (args.rescored / "analysis.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
