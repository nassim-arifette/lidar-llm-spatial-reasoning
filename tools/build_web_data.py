"""Export archived answers and the matching annotation geometry for the website.

Scenes are rebuilt from the local nuScenes v1.0-mini metadata with the study's
own adapter (ego frame, oracle annotations, same status mapping). Before
writing anything, every saved program of every exported comparison is executed
again with ``spatial_reasoning.dsl`` on the rebuilt scene; the export refuses to
proceed unless the answer and the selected annotation tokens match the archive.
The archive stores at most 40 selected tokens; the export stores the full,
re-executed list after checking that its first 40 match. The website re-runs the same programs in TypeScript and is tested against this
file.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_reasoning.dsl import execute  # noqa: E402
from spatial_reasoning.llm_parser import InvalidProgram, query_from_json  # noqa: E402
from spatial_reasoning.manifest import describe_objects  # noqa: E402
from spatial_reasoning.nuscenes_adapter import load_nuscenes_scenes  # noqa: E402
from spatial_reasoning.regions import NUSCENES_QA_REGION_CONFIG  # noqa: E402

META = ROOT / "data/data/nuscenes/v1.0-mini"
OUT = ROOT / "web/public/data"
RUNS = [
    ("llm_modes", "Qwen2.5 1.5B", "local, fp16, 256 output tokens"),
    ("llm_modes_7b", "Qwen2.5 7B", "local, fp16, 256 output tokens"),
    ("llm_modes_32b", "Qwen2.5 32B", "local, int4, 256 output tokens"),
    ("gw_qwen3_32b", "Qwen3 32B", "hosted endpoint, output cap requested"),
    ("gw_qwen3_next_80b", "Qwen3-Next 80B (3B active)", "hosted endpoint, mixture of experts"),
    ("gw_qwen3p8_27b", "Qwen3.8 27B", "hosted endpoint, output cap not respected"),
]
ROW_FIELDS = ["question_id", "sample_token", "family", "question", "reference_answer",
              "direct_answer", "direct_raw", "direct_correct", "direct_output_tokens",
              "program_raw", "program", "program_status", "executed_answer", "execution_status",
              "execution_reason", "executed_correct", "selected", "program_output_tokens"]


def r1(x):
    return round(float(x), 3)


def main():
    sources = []

    def read(rel, lines=False):
        raw = (ROOT / rel).read_bytes()
        sources.append({"path": rel, "sha256": hashlib.sha256(raw).hexdigest()})
        text = raw.decode("utf-8")
        return [json.loads(x) for x in text.splitlines() if x.strip()] if lines else json.loads(text)

    scenes, report = load_nuscenes_scenes(str(META), min_lidar_points=0)
    lidar_pts = {a["token"]: a["num_lidar_pts"]
                 for a in json.loads((META / "sample_annotation.json").read_text(encoding="utf-8"))}
    config = NUSCENES_QA_REGION_CONFIG

    runs, needed = [], set()
    checked = 0
    for folder, label, runtime in RUNS:
        base = f"spatial_reasoning/results/{folder}"
        summary = read(f"{base}/summary_eval.json")
        rows = read(f"{base}/rows_eval.jsonl", lines=True)
        assert len(rows) == summary["summary"]["n"] == 200
        for row in rows:
            scene = scenes[row["sample_token"]]
            needed.add(row["sample_token"])
            if row["program_status"] != "ok":
                continue
            try:
                query = query_from_json(row["program_raw"], frozenset(config.directions()))
            except InvalidProgram as exc:
                raise SystemExit(f"{folder}/{row['question_id']}: archived ok program no longer parses: {exc}")
            if query is None:
                continue
            res = execute(query, scene, config)
            answer = res.answer if res.ok else None
            if answer != row["executed_answer"]:
                raise SystemExit(f"{folder}/{row['question_id']}: re-executed {answer!r} != archived {row['executed_answer']!r}")
            tokens = [o.token for o in res.selected if o.token] if res.ok else []
            # the archive keeps the first 40 selected tokens, in execution order
            if tokens[:40] != (row.get("selected") or []):
                raise SystemExit(f"{folder}/{row['question_id']}: selected objects differ from the archive")
            row["selected"] = tokens
            checked += 1
        s = summary["summary"]
        runs.append({"id": folder, "label": label, "runtime": runtime, "model": summary["runtime"].get("model"),
                     "summary": {"direct": s["direct"], "executed": s["executed"], "paired": s.get("paired"),
                                 "by_family": s.get("by_family"), "scene_dependence": s.get("scene_dependence")},
                     "rows": [{k: row.get(k) for k in ROW_FIELDS} for row in rows]})

    out_scenes = {}
    for token in sorted(needed):
        scene = scenes[token]
        ids = describe_objects(scene)
        out_scenes[token] = {
            "scene_token": scene.scene_token,
            "objects": [{
                "id": d["id"], "token": o.token, "category": o.category, "status": o.status,
                # exact float32 values: the executor's sector tests run on them
                "x": float(o.center[0]), "y": float(o.center[1]), "z": r1(o.center[2]),
                "w": r1(o.size[0]), "l": r1(o.size[1]), "h": r1(o.size[2]), "yaw": round(o.yaw, 4),
                "lidar_pts": lidar_pts.get(o.token),
                "px": d["x"], "py": d["y"],
            } for o, d in zip(scene.objects, ids)],
        }

    manifest = read("spatial_reasoning/manifests/first_comparison.json")
    data = {"schema_version": 1, "evidence": "archived model outputs; oracle nuScenes annotations",
            "manifest_config": manifest["config"], "layout": {k: list(v) for k, v in config.sectors.items()},
            "runs": runs, "scenes": out_scenes, "sources": sources,
            "reexecuted_programs": checked}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "study.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(runs)} runs, {len(out_scenes)} samples, {checked} archived programs re-executed identically -> {OUT}")


if __name__ == "__main__":
    main()
