# Spatial reasoning at fixed perception

The current study compares a frozen Qwen model answering from a 3D object
list against the **same model** writing a program executed on that list.
See the [project README](../README.md), [report](../report/rapport_npm3d.pdf)
and [result ledger](results/state_of_results.md).

```text
nuScenes ground-truth objects ---------> direct model -> answer A
                 |                           ^
                 |                           |
                 +-> deterministic executor <- program <- same model <- question
                               |
                            answer B
```

Objects include positions, categories and annotated statuses. This is an
oracle-perception experiment. It does not read raw point clouds or run a
detector. The original GPT-2 training pipeline is separate.

## Entry points

| Module | Purpose |
|---|---|
| `audit_results` | Recompute the saved comparisons offline; no GPU or model |
| `verify_executor` | Check reference programs against development labels |
| `compare_llm_modes` | Run direct and executed modes, optionally a typed third mode |
| `analyse_failures` | Categorize archived model/execution failures |
| `run_symbolic_diagnostic` | Measure grammar and LiDAR-return-filter effects |
| `run_experiments` | Earlier synthetic/learned-baseline experiment, not the main study |

Run from the repository root:

```bash
python -m spatial_reasoning.audit_results
python -m pytest tests/ -q
```

GPU commands and dependencies are in [REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).
The narrow comparison has 100 development and 200 evaluation questions,
separated by drive. It uses a flat target language. `--functional` enables
the compositional typed catalogue for the **separate wide-scope comparison**
and gives all three modes the same output cap of at least 384 tokens.

## Conventions and failure behavior

- Ego coordinates are x forward, y left, z up.
- The `nuscenes_qa` layout uses the six 60-degree intervals published in
  NuScenes-QA Equation 2, with half-open boundaries. `octant` is the earlier
  eight-sector synthetic layout; results from the two are not interchangeable.
- Real-data comparisons retain ground-truth objects with zero LiDAR returns,
  as the benchmark labels may count them. `min_lidar_points` is recorded.
- Programs cannot access reference answers. Unsupported or malformed output
  is rejected; ambiguous or missing object references may be refused.
- Direct caches depend on question **and scene**; program caches on question.
  Overlong prompts are refused, not silently trimmed.
- The main exact-accuracy denominator includes every requested question,
  including refusals. Answered accuracy and numeric coverage are secondary.

## Archived early study

`results/all_results.json` and `summary.md` describe the balanced synthetic
corpus with question-only, DeepSets, neural-counter and rule-program systems.
Those small learned models run on CPU. Their best symbolic result (0.916)
uses generated questions and synthetic perception noise; it is **not** the
headline result on real nuScenes-QA. The current frozen-LLM study has separate
manifests and per-question records. `parser_comparison_v1_BAD_GEOMETRY.json`
is retained only as evidence of an earlier convention error.
