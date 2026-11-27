"""Export the label audit and the November 2026 runs for the website.

Reads only saved results (no nuScenes download, no GPU) and writes
``web/public/data/audit.json``:

* the geometry-free counts of ``results/label_audit/official_release.json``;
* the held-out executor checks and the zero-LiDAR diagnostic;
* ``results/rescored_as_labelled/analysis.json`` (archived programs under both layouts);
* every table of ``results/runs_2026_11/README.md``, recomputed from the saved
  rows with the functions of ``spatial_reasoning.summarise_runs`` and
  ``spatial_reasoning.drive_bootstrap``. A run that is absent, has no summary
  yet or does not hold the expected number of questions is left out, so the
  page never shows an empty cell;
* ``stats``: every number quoted in the prose of the page, already formatted.
  This script writes them into the ``data-stat`` elements of ``web/index.html``,
  ``web/src/main.ts`` does the same in the browser, and ``tools/check_site.py``
  checks that the built HTML and the built ``audit.json`` agree.

It also refreshes the copies of the documents the page links to
(``web/public/files/``).

    python tools/build_web_audit.py
"""
from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_reasoning.drive_bootstrap import across, analyse  # noqa: E402
from spatial_reasoning.summarise_runs import ARCHIVE, MODELS, acc, rows_of  # noqa: E402

RESULTS = ROOT / "spatial_reasoning/results"
RUNS = RESULTS / "runs_2026_11"
OUT = ROOT / "web/public/data/audit.json"
HTML = ROOT / "web/index.html"
FILES = ROOT / "web/public/files"
#: Documents the page links to, copied so that the built site stands alone.
COPIES = (("docs/LABEL_AUDIT.md", "label_audit.md"), ("docs/REPRODUCIBILITY.md", "reproduction.md"),
          ("spatial_reasoning/results/runs_2026_11/README.md", "runs_2026_11.md"))
STAT = re.compile(r'(<(span|strong) data-stat="([^"]+)"( data-optional)?>)[^<]*(</\2>)')


def sync_html(stats):
    """Write each number into the element of web/index.html that quotes it.

    The page is readable without JavaScript and ``check_site.py`` can compare the
    built HTML with ``audit.json``. An element marked ``data-optional`` belongs to
    a passage that stays hidden until its run exists.
    """
    missing = []

    def fill(m):
        key = m.group(3)
        if key not in stats:
            if not m.group(4):
                missing.append(key)
            return f"{m.group(1)}…{m.group(5)}"
        return f"{m.group(1)}{stats[key]}{m.group(5)}"

    text = HTML.read_text(encoding="utf-8")
    new, n = STAT.subn(fill, text)
    if missing:
        raise SystemExit(f"web/index.html quotes numbers the results do not provide: {sorted(set(missing))}")
    if new != text:
        HTML.write_text(new, encoding="utf-8")
    return n
RELATIONS = ["front left", "front right", "back left", "back right", "front", "back"]
ARCHIVED = ["llm_modes", "llm_modes_7b", "llm_modes_32b", "gw_qwen3_32b", "gw_qwen3_next_80b", "gw_qwen3p8_27b"]
EXPECTED = {"dev": 100, "eval": 200}
FAMILIES = ["count_exist_category", "category_status", "category_ego_region", "category_anchor_relation"]


def pct(v, digits=1):
    return f"{v * 100:.{digits}f} %"


def num(v):
    return f"{v:,}"


def pts(v, digits=1):
    return f"{v * 100:+.{digits}f}"


def main():
    sources = []

    def read(rel):
        raw = (RESULTS / rel).read_bytes()
        sources.append({"path": f"spatial_reasoning/results/{rel}", "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw.decode("utf-8"))

    found = []

    def run(name, split="eval", n=None):
        """Rows of a November run, or None unless the run is complete."""
        folder = RUNS / name
        if not (folder / f"summary_{split}.json").exists():
            return None
        rows = rows_of(folder, split)
        if not rows or len(rows) != (n or EXPECTED[split]):
            return None
        raw = (folder / f"rows_{split}.jsonl").read_bytes()
        sources.append({"path": f"spatial_reasoning/results/runs_2026_11/{name}/rows_{split}.jsonl",
                        "sha256": hashlib.sha256(raw).hexdigest()})
        found.append(name)
        return rows

    stats = {}

    # ------------------------------------------------------------------ labels
    release = read("label_audit/official_release.json")["files"]
    mini = read("label_audit/mini.json")["files"]["nuscenes_qa_mini.json"]
    train, val = release["NuScenes_train_questions.json"], release["NuScenes_val_questions.json"]
    laterality = {"relations": RELATIONS, "sets": [
        {"id": key, "label": label, "n_questions": f["n_questions"], "n_samples": f["n_samples"],
         "rows": {r: f["laterality"][r] for r in RELATIONS}}
        for key, label, f in (("train", "train", train), ("val", "val", val), ("mini", "v1.0-mini", mini))]}
    never = ("front right", "back left")
    for r in never:
        for f in (train, val, mini):
            assert f["laterality"][r]["empty"] == f["laterality"][r]["n"], f"{r} is not always empty"
    stats["lat-never-n"] = num(sum(f["laterality"][r]["n"] for r in never for f in (train, val)))
    stats["lat-keyframes"] = num(train["n_samples"] + val["n_samples"])
    others = [f["laterality"][r]["empty"] / f["laterality"][r]["n"]
              for r in RELATIONS if r not in never for f in (train, val)]
    stats["lat-other-min"] = pct(min(others))
    stats["lat-other-max"] = pct(max(others))
    for r in never:
        k = r.replace(" ", "-")
        stats[f"lat-mini-{k}"] = f"{mini['laterality'][r]['empty']} of {mini['laterality'][r]['n']}"

    things = {key: f["things"] for key, f in (("train", train), ("val", val))}
    for key, t in things.items():
        e = t["answer_equals"]
        assert e["every annotation + ego"] == t["n"]
        stats[f"things-{key}-n"] = num(t["n"])
        stats[f"things-{key}-named"] = num(e["ten named classes + ego"])
        stats[f"things-{key}-lidar"] = num(e["annotations with a LiDAR return + ego"])
    stats["things-n"] = num(things["train"]["n"] + things["val"]["n"])

    # ------------------------------------------------------------------ held-out checks
    def verification(rel):
        s = read(rel)["summary"]
        return {"agreed": s["agreed"], "n": s["n"]}

    checks = []
    dev_s, dev_l = verification("executor_verification/summary_dev.json"), verification("executor_verification_as_labelled/summary_dev.json")
    eval_l = verification("executor_verification_as_labelled/summary_eval.json")
    checks.append({"label": "v1.0-mini, development split, hand-verified reference programs", "n": dev_s["n"],
                   "signed": dev_s["agreed"] / dev_s["n"], "as_labelled": dev_l["agreed"] / dev_l["n"]})
    parser = read("label_audit/val_rule_parser_by_relation.json")
    ps, pl = parser["nuscenes_qa"], parser["nuscenes_qa_as_labelled"]
    total = lambda t: (sum(v["correct"] for v in t.values()), sum(v["n"] for v in t.values()))  # noqa: E731
    (cs, ns), (cl, nl) = total(ps), total(pl)
    assert ns == nl
    checks.append({"label": "Official validation set, rule parser, every question it answers", "n": ns,
                   "signed": cs / ns, "as_labelled": cl / nl})
    for r in ("front left", "front right", "back left", "back right"):
        checks.append({"label": f"… of which {r}", "n": ps[r]["n"], "indent": True,
                       "signed": ps[r]["correct"] / ps[r]["n"], "as_labelled": pl[r]["correct"] / pl[r]["n"]})
    fb = lambda t: (t["front"]["correct"] + t["back"]["correct"], t["front"]["n"] + t["back"]["n"])  # noqa: E731
    checks.append({"label": "… front, back", "n": fb(ps)[1], "indent": True,
                   "signed": fb(ps)[0] / fb(ps)[1], "as_labelled": fb(pl)[0] / fb(pl)[1]})
    vd = {f"{t}__{l}": verification(f"executor_verification_val_drives/{t}__{l}/summary_eval.json")
          for t in ("named", "all") for l in ("nuscenes_qa", "nuscenes_qa_as_labelled")}
    for t, label in (("named", "146 validation drives, ten named classes"), ("all", "146 validation drives, every annotation and the ego vehicle")):
        a, b = vd[f"{t}__nuscenes_qa"], vd[f"{t}__nuscenes_qa_as_labelled"]
        checks.append({"label": label, "n": a["n"], "signed": a["agreed"] / a["n"], "as_labelled": b["agreed"] / b["n"]})
    stats.update({
        "check-dev-signed": str(dev_s["agreed"]), "check-dev-labelled": str(dev_l["agreed"]), "check-dev-n": str(dev_s["n"]),
        "check-eval-labelled": str(eval_l["agreed"]), "check-eval-n": str(eval_l["n"]),
        "check-val-n": num(ns), "check-val-signed": pct(cs / ns), "check-val-labelled": pct(cl / nl),
        "check-vd-n": num(vd["all__nuscenes_qa"]["n"]),
        "check-vd-named-signed": pct(vd["named__nuscenes_qa"]["agreed"] / vd["named__nuscenes_qa"]["n"]),
        "check-vd-all-labelled": pct(vd["all__nuscenes_qa_as_labelled"]["agreed"] / vd["all__nuscenes_qa_as_labelled"]["n"]),
        "check-vd-left": str(vd["all__nuscenes_qa_as_labelled"]["n"] - vd["all__nuscenes_qa_as_labelled"]["agreed"]),
    })

    # ------------------------------------------------------------------ zero-LiDAR objects
    lidar = []
    conv = None
    for label, files in (("v1.0-mini", ("symbolic_diagnostic_as_labelled/diagnostic_published_layout_rerun.json",
                                         "symbolic_diagnostic_as_labelled/diagnostic.json")),
                         ("Official validation set", ("label_audit/val_symbolic_diagnostic__nuscenes_qa.json",
                                                      "label_audit/val_symbolic_diagnostic__nuscenes_qa_as_labelled.json"))):
        for rel in files:
            d = read(rel)
            if label == "v1.0-mini":
                conv = d["conversions"]["1"]
            p = d["paired_numeric"]["B_filter_at_expressive_parser"]
            a, b = p["expressive_parser__min_lidar_1"], p["expressive_parser__min_lidar_0"]
            lidar.append({"set": label, "n": a["n_paired"] if "n_paired" in a else a["count_errors"]["n"],
                          "layout": d["layout"], "with_return": a["accuracy"], "every_annotation": b["accuracy"]})
    stats["lidar-zero"] = num(conv["dropped_no_lidar_points"])
    stats["lidar-total"] = num(conv["annotations_total"])
    for row in lidar:
        k = ("mini" if row["set"] == "v1.0-mini" else "val") + ("-labelled" if row["layout"].endswith("as_labelled") else "-signed")
        stats[f"lidar-{k}-return"] = pct(row["with_return"])
        stats[f"lidar-{k}-all"] = pct(row["every_annotation"])
        stats[f"lidar-{'mini' if row['set'] == 'v1.0-mini' else 'val'}-n"] = num(row["n"])

    # ------------------------------------------------------------------ archived programs, both layouts
    analysis = read("rescored_as_labelled/analysis.json")
    archive = []
    for run_id in ARCHIVED:
        h = analysis["headline"][run_id]
        fbk = analysis["fallback"][run_id]
        entry = {"id": run_id, "label": h["label"], "n": h["n"], "direct_1": h["direct_1"],
                 "program_published": h["flat_old"], "program_as_labelled": h["flat_new"],
                 "fallback_published": fbk["old"]["executed -> direct"]["accuracy"],
                 "fallback_as_labelled": fbk["new"]["executed -> direct"]["accuracy"],
                 "relations": {r: analysis["relations"][run_id][r] for r in RELATIONS}}
        archive.append(entry)
        stats[f"{run_id}-d1"] = pct(h["direct_1"])
        stats[f"{run_id}-pub"] = pct(h["flat_old"])
        stats[f"{run_id}-lab"] = pct(h["flat_new"])
    ceiling = analysis["ceiling"]
    stats.update({
        "ceil-old-n": str(ceiling["old"]["n_wrong_for_all_six"]), "ceil-new-n": str(ceiling["new"]["n_wrong_for_all_six"]),
        "ceil-old": pct(ceiling["old"]["ceiling_accuracy"], 0), "ceil-new": pct(ceiling["new"]["ceiling_accuracy"], 0),
        "ceil-old-directional": str(sum(ceiling["old"]["by_family"].get(f, 0) for f in ("category_ego_region", "category_anchor_relation"))),
    })
    big = analysis["relations"]["llm_modes_32b"]
    for r in ("front left", "front right", "back left", "back right"):
        stats[f"rel-32b-{r.replace(' ', '-')}"] = f"{big[r]['old']} to {big[r]['new']} of {big[r]['n']}"
    wide = analysis["headline"]["wide_1p5b"]
    wide_summary = read("wide_1p5b/summary_eval.json")["summary"]
    stats.update({
        "wide-n": str(wide["n"]), "wide-direct": pct(wide["direct_1"]),
        "wide-typed-pub": pct(wide["typed_old"]), "wide-typed-lab": pct(wide["typed_new"]),
        "wide-typed-when-run": pct(wide_summary["functional"]["accuracy_when_answered"]),
        "wide-typed-runs": pct(wide_summary["functional"]["answered_fraction"]),
    })

    # the first comparison as archived (published conventions), for the 1.5B model
    first = read("llm_modes/summary_eval.json")["summary"]
    stats.update({
        "pub-answered": pct(first["executed"]["answered_fraction"], 0),
        "pub-exec-only": str(first["paired"]["executed_only"]), "pub-direct-only": str(first["paired"]["direct_only"]),
        "pub-mae-d": f"{first['direct']['count_errors']['mae']:.2f}", "pub-mae-e": f"{first['executed']['count_errors']['mae']:.2f}",
        "pub-drives": ", ".join(pts(v["executed"] - v["direct"]) for v in first["by_scene"].values()),
    })
    hosted = rows_of(RESULTS / "gw_qwen3p8_27b", "eval")
    stats["q38-direct-tokens"] = f"{sum(r['direct_output_tokens'] for r in hosted) / len(hosted):.1f}"
    stats["q38-program-tokens"] = f"{sum(r['program_output_tokens'] for r in hosted) / len(hosted):.1f}"

    # ------------------------------------------------------------------ November runs
    prompts, reasoned, constrained, development = [], [], [], []
    bad = lambda rows, status: sum(r[status] not in ("ok", "refused") for r in rows)  # noqa: E731
    tokens = lambda rows: sum(r["direct_output_tokens"] for r in rows) / len(rows)  # noqa: E731
    for key, label in MODELS:
        archived = rows_of(RESULTS / ARCHIVE[key], "eval")
        relabelled = rows_of(RESULTS / "rescored_as_labelled" / ARCHIVE[key], "eval")
        d2 = run(f"d2_bare_{key}_eval")
        cells = {"direct_1": acc(archived, "direct"), "direct_2": acc(d2, "direct"),
                 "direct_1r": acc(run(f"d1r_{key}_eval"), "direct"), "direct_2r": acc(run(f"d2r_{key}_eval"), "direct"),
                 "program_published": acc(archived, "executed"), "program_as_labelled": acc(relabelled, "executed"),
                 "program_constrained": acc(run(f"c_{key}_eval"), "executed")}
        entry = {"key": key, "label": label, "archive": ARCHIVE[key], "n": len(archived),
                 "cells": {k: v for k, v in cells.items() if v is not None}}
        if d2 and relabelled:
            entry["programs_identical_to_archive"] = sum(a["program_raw"] == b["program_raw"] for a, b in zip(d2, relabelled))
            entry["new_run_program_as_labelled"] = acc(d2, "executed")
        prompts.append(entry)
        for k, v in entry["cells"].items():
            stats[f"p-{key}-{k.replace('_', '-')}"] = pct(v)
        for prompt, name, split in (("direct-1r", f"d1r_{key}_dev_v3", "dev"), ("direct-2r", f"d2r_{key}_dev_v3", "dev"),
                                    ("direct-1r", f"d1r_{key}_eval", "eval"), ("direct-2r", f"d2r_{key}_eval", "eval")):
            rows = run(name, split)
            if rows:
                reasoned.append({"key": key, "label": label, "prompt": prompt, "split": split, "accuracy": acc(rows, "direct"),
                                 "mean_output_tokens": tokens(rows), "unreadable": sum(r["direct_answer"] is None for r in rows), "n": len(rows)})
                if split == "eval":
                    stats[f"r-{key}-{prompt}-tokens"] = f"{tokens(rows):.0f}"
                    stats[f"r-{key}-{prompt}-unread"] = str(reasoned[-1]["unreadable"])
        for split in ("dev", "eval"):
            free, con = run(f"d2_bare_{key}_{split}", split), run(f"c_{key}_{split}", split)
            if free and con:
                constrained.append({"key": key, "label": label, "split": split, "n": len(con), "free": acc(free, "executed"),
                                    "constrained": acc(con, "executed"), "invalid_free": bad(free, "program_status"),
                                    "invalid_constrained": bad(con, "program_status")})
                stats[f"c-{key}-{split}-invalid"] = f"{bad(free, 'program_status')} to {bad(con, 'program_status')}"
        rows = run(f"d2_bare_{key}_dev", "dev")
        if rows:
            archived_dev = rows_of(RESULTS / ARCHIVE[key], "dev")
            cells = {"direct_1": acc(archived_dev, "direct"), "direct_2": acc(rows, "direct"), "program_as_labelled": acc(rows, "executed")}
            development.append({"key": key, "label": label, "n": len(rows), "cells": {k: v for k, v in cells.items() if v is not None}})

    # the 1.5B first comparison once the convention is right: direct-2 against the re-executed programs
    d2 = run("d2_bare_1p5b_eval")
    relabelled = rows_of(RESULTS / "rescored_as_labelled/llm_modes", "eval")
    assert d2 and [r["question_id"] for r in d2] == [r["question_id"] for r in relabelled]
    pairs = [(bool(a["direct_correct"]), bool(b["executed_correct"])) for a, b in zip(d2, relabelled)]
    stats["lab-exec-only"] = str(sum(e and not d for d, e in pairs))
    stats["lab-direct-only"] = str(sum(d and not e for d, e in pairs))
    drives = {}
    for a, (d, e) in zip(d2, pairs):
        drives.setdefault(a["scene_token"], []).append(e - d)
    stats["lab-drives"] = ", ".join(pts(sum(v) / len(v)) for v in drives.values())
    d2_summary = json.loads((RUNS / "d2_bare_1p5b_eval/summary_eval.json").read_text(encoding="utf-8"))["summary"]
    re_summary = read("rescored_as_labelled/llm_modes/summary_eval.json")["summary"]
    stats.update({
        "lab-mae-d": f"{d2_summary['direct']['count_errors']['mae']:.2f}", "lab-under": f"{d2_summary['direct']['count_errors']['under']:.2f}",
        "lab-mae-e": f"{re_summary['executed']['count_errors']['mae']:.2f}",
    })
    # questions asked verbatim of several keyframes whose reference answers differ
    groups = {}
    for a, b in zip(d2, relabelled):
        groups.setdefault(a["question"], []).append((a, b))
    kept = [g for g in groups.values() if len({a["reference_answer"] for a, _ in g}) > 1]
    stats.update({
        "dep-groups": str(len(kept)),
        "dep-exec-all": str(sum(all(b["executed_correct"] for _, b in g) for g in kept)),
        "dep-direct-all": str(sum(all(a["direct_correct"] for a, _ in g) for g in kept)),
        "dep-direct-constant": str(sum(len({a["direct_answer"] for a, _ in g}) == 1 for g in kept)),
    })

    # validation drives
    val_runs, effect, by_family, val_rows = [], [], [], {}
    for key, label in MODELS:
        for tag, name in (("published", f"val_old_{key}"), ("as_labelled", f"val_new_{key}")):
            rows = run(name, n=1460)
            if not rows:
                continue
            val_rows.setdefault(key, {})[tag] = rows
            a = analyse(rows)
            d = a["paired_difference"]["executed_minus_direct"]
            keep = lambda e: {k: e[k] for k in ("estimate", "low", "high")}  # noqa: E731
            val_runs.append({"key": key, "label": label, "conventions": tag, "n": len(rows),
                             "n_drives": a["accuracy"]["direct"]["n_drives"],
                             "direct": keep(a["accuracy"]["direct"]), "program": keep(a["accuracy"]["executed"]), "difference": keep(d),
                             "drives_ahead": d["drives_where_second_is_ahead"], "drives_behind": d["drives_where_first_is_ahead"],
                             "drives_tied": d["drives_tied"]})
            t = "lab" if tag == "as_labelled" else "pub"
            stats.update({
                f"val-{key}-{t}-direct": pct(a["accuracy"]["direct"]["estimate"]),
                f"val-{key}-{t}-program": pct(a["accuracy"]["executed"]["estimate"]),
                f"val-{key}-{t}-diff": f"{pts(d['estimate'])} points [{pts(d['low'])}, {pts(d['high'])}]",
                f"val-{key}-{t}-ahead": str(d["drives_where_second_is_ahead"]),
                f"val-{key}-{t}-behind": str(d["drives_where_first_is_ahead"]),
            })
            stats["val-n"], stats["val-drives"] = num(len(rows)), str(a["accuracy"]["direct"]["n_drives"])
        pair = val_rows.get(key, {})
        if len(pair) == 2:
            d = across(pair["as_labelled"], pair["published"])
            effect.append({"key": key, "label": label, **{m: {k: d[s][k] for k in ("estimate", "low", "high")}
                                                         for m, s in (("direct", "direct"), ("program", "executed"))}})
        if "as_labelled" in pair:
            for family in FAMILIES:
                group = [r for r in pair["as_labelled"] if r["family"] == family]
                by_family.append({"key": key, "label": label, "family": family, "n": len(group),
                                  "direct": acc(group, "direct"), "program": acc(group, "executed")})
                stats[f"val-{key}-{family}-direct"] = pct(acc(group, "direct"))
                stats[f"val-{key}-{family}-program"] = pct(acc(group, "executed"))

    free, con = run("free_wide_1p5b_eval", n=280), run("c_wide_1p5b_eval", n=280)
    if free and con:
        for mode, status, name in (("executed", "program_status", "flat"), ("functional", "functional_status", "typed")):
            constrained.append({"key": "1p5b", "label": "Qwen2.5 1.5B, wide scope", "split": "eval", "mode": name, "n": len(con),
                                "free": acc(free, mode), "constrained": acc(con, mode),
                                "invalid_free": bad(free, status), "invalid_constrained": bad(con, status)})

    thinking = []
    for label, name, split in (("off", "q3_8b_think_off_dev_b4", "dev"), ("on", "q3_8b_think_on_dev_b4", "dev"),
                               ("off", "q3_8b_think_off_eval", "eval"), ("on", "q3_8b_think_on_eval_b4", "eval")):
        rows = run(name, split)
        if rows:
            thinking.append({"thinking": label, "split": split, "n": len(rows), "direct": acc(rows, "direct"),
                             "mean_direct_output_tokens": tokens(rows), "unreadable": sum(r["direct_answer"] is None for r in rows),
                             "program": acc(rows, "executed"), "invalid_programs": bad(rows, "program_status")})
            if split == "eval":
                stats[f"q3-8b-{label}-direct"] = pct(acc(rows, "direct"))
                stats[f"q3-8b-{label}-program"] = pct(acc(rows, "executed"))

    # The README is generated from the same rows; say so if it has fallen behind.
    readme = (RUNS / "README.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^- (\S+) \(", readme, flags=re.M))
    for name in sorted(set(found) - listed):
        print(f"note: {name} is complete but not yet in runs_2026_11/README.md (rerun spatial_reasoning.summarise_runs)", file=sys.stderr)
    for name in sorted(listed - set(found)):
        print(f"note: {name} is listed in runs_2026_11/README.md but was not exported (no summary, or unexpected size)", file=sys.stderr)

    data = {"schema_version": 1,
            "evidence": "saved results only: label audit, archived programs re-executed, runs of November 2026",
            "laterality": laterality, "things": things, "checks": checks, "lidar": lidar,
            "archive": archive, "ceiling": {"family_sizes": ceiling["family_sizes"],
                                            "published": ceiling["old"], "as_labelled": ceiling["new"]},
            "prompts": prompts, "reasoned": reasoned, "constrained": constrained, "development": development,
            "validation": {"runs": val_runs, "effect": effect, "by_family": by_family},
            "thinking": thinking, "runs_read": sorted(set(found)), "stats": stats, "sources": sources}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    quoted = sync_html(stats)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for source, name in COPIES:
        (FILES / name).write_bytes((ROOT / source).read_bytes())
    print(f"{len(set(found))} November runs, {len(stats)} formatted numbers ({quoted} quoted in web/index.html) -> {OUT}")


if __name__ == "__main__":
    main()
