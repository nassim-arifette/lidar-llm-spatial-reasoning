# Answer it, or compute it?

Spatial questions about LiDAR driving scenes: should a language model answer directly, or write a program that an executor runs?

[Open the interactive article →](https://nassim-arifette.github.io/lidar-llm-spatial-reasoning/)

[![The website's hero: a question about a nuScenes scene, the direct answer (wrong) and the executed program (right), over the real LiDAR sweep](docs/assets/site-hero.png)](https://nassim-arifette.github.io/lidar-llm-spatial-reasoning/)

[LiDAR-LLM](https://arxiv.org/abs/2312.14074) turns 3D scene understanding into text generation. Here I hold perception fixed and ask something narrower. Given the list of objects in a nuScenes scene, is the same frozen model more reliable when it produces the answer itself, or when it writes a small program that computes it? Answering that turned up a problem in the benchmark first.

## 1. The NuScenes-QA answers do not tell left from right

The six spatial relations are published as signed 60° sectors. The released answers follow the unsigned angle `a` between the ego heading and the direction to the object: "front left" holds for 30° < a ≤ 90° on either side, "back right" for 90° < a ≤ 150° on either side, and "front right" and "back left" never hold.

| Counting or existence questions about | train | answered 0 or no | val | answered 0 or no |
|---|---:|---:|---:|---:|
| front left | 19,243 | 22.8 % | 4,156 | 22.0 % |
| **front right** | 5,070 | **100 %** | 1,134 | **100 %** |
| **back left** | 5,137 | **100 %** | 1,051 | **100 %** |
| back right | 19,373 | 23.4 % | 4,103 | 23.9 % |

Two more conventions sit beside it. "Things" are every annotation, unnamed classes included, plus the ego vehicle: "How many things are there?" equals the number of annotations plus one in 2,728 questions out of 2,728. And boxes with no LiDAR return are counted. On 1,460 questions from 146 validation drives, a deterministic executor agrees with the released answers on 85.1 % under the published reading and 97.5 % under these conventions. I inferred the rules from the answers and did not read the generation code. Details and limits: [`docs/LABEL_AUDIT.md`](docs/LABEL_AUDIT.md).

## 2. Direct answer or program, once the conventions are right

Same checkpoint for both modes, greedy decoding, ground-truth object lists, both modes told the conventions the answers follow. 1,460 questions from 146 validation drives that no prompt was developed on; 95 % intervals resample drives.

| Model | Direct answer | Program + executor | Difference | Drives where the program is ahead / behind / tied |
|---|---:|---:|---:|---:|
| Qwen2.5 1.5B (fp16) | 58.2 % | 74.7 % | +16.6 [+12.5, +20.5] | 108 / 29 / 9 |
| Qwen2.5 7B (fp16) | 59.5 % | 96.1 % | +36.6 [+33.7, +39.5] | 143 / 0 / 3 |
| Qwen2.5 32B (int4) | 67.9 % | 97.8 % | +29.9 [+27.4, +32.3] | 142 / 0 / 4 |

Refusals, invalid programs and unreadable answers count as wrong. At 1.5B the program mode is worse than the direct answer on one family, relations anchored on another object (33.1 % against 47.9 %).

The first comparison (200 questions, three drives of v1.0-mini) told both modes the published convention. Its saved programs can be executed again under the other one without a GPU:

| Model | Direct, published prompt | Direct, corrected prompt | Program, published layout | Program, as labelled |
|---|---:|---:|---:|---:|
| Qwen2.5 1.5B | 50.5 % | 54.0 % | 68.5 % | 76.5 % |
| Qwen2.5 7B | 50.5 % | 54.5 % | 82.5 % | 95.0 % |
| Qwen2.5 32B (int4) | 58.0 % | 64.0 % | 84.5 % | 98.0 % |
| Qwen3 32B (hosted) | 56.0 % | not rerun | 84.0 % | 97.5 % |
| Qwen3-Next 80B, 3B active (hosted) | 61.0 % | not rerun | 84.5 % | 98.0 % |
| Qwen3.8 27B (hosted) | 84.0 % | not rerun | 84.5 % | 98.0 % |

The plateau near 84.5 % was the convention mismatch: 28 questions were wrong for all six models under the published layout, 4 as labelled. The last row used to read as parity between the two modes; that model's direct answers were given under the old prompt and its endpoint was not rerun, so neither parity nor a gap is established for it.

Two more conditions, on the same 200 questions. Asking the direct mode to write out its working before a final `Answer:` line gave 34.5 % at 1.5B and 62.5 % at 7B with the corrected prompt, against 76.5 % and 95.0 % for their programs. Decoding programs under a JSON Schema with a closed vocabulary removed every invalid program and moved the 1.5B model from 76.5 % to 80.5 %. All tables: [`spatial_reasoning/results/runs_2026_11/README.md`](spatial_reasoning/results/runs_2026_11/README.md).

## What you can do on the site

| Figure | Try this |
| --- | --- |
| What the model is given | Browse the 102 evaluation keyframes in bird's-eye view or 3D, over the real LiDAR sweep, next to the exact prompt rows. Switch to "LiDAR returns" to see annotated objects the sensor never hit. |
| The answers do not tell left from right | Click any object to measure the six relations from it, as published or as the answers use them. |
| Writing a program | Build a program in the study's JSON schema and run it live on any scene, refusals included. |
| 200 questions, two answers | Every archived answer of all six models, filtered by which method was right, with the scene of each question. |
| Counting, scene dependence, scale | Where the gain comes from and where it goes away. |

The site re-runs the archived programs with a TypeScript port of the executor, and a test checks that the port reproduces every saved answer and every selected object for the six models.

## Verify the results without a GPU

```bash
python -m pip install -r requirements-audit.txt
python -m spatial_reasoning.audit_results
```

The audit recomputes correctness and every summary field from the per-question records: the eight archived comparisons (1,580 rows), their re-execution under the as-labelled layout, and the runs of November 2026 with their provenance. It also checks manifest membership and scene separation, and it fails on any disagreement. File hashes are in the audit receipts ([archive](spatial_reasoning/results/artifact_audit.json), [November 2026](spatial_reasoning/results/artifact_audit_2026_11.json)).

The label tests need only the official question files; see [`docs/LABEL_AUDIT.md`](docs/LABEL_AUDIT.md). The test suite runs with the command below. The last run passed all 283 tests, PyTorch modules included.

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

For GPU reruns, the metadata download, cache handling and the two unfinished extensions, see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Scope and limits

- Perception is an oracle: object lists and statuses are nuScenes ground-truth annotations, and no detector was run.
- The first comparison rests on three drives. The validation-drive comparison uses 146, and its intervals resample drives, not questions.
- The label conventions were found while explaining executor disagreements, then checked on data that played no part in finding them. The "things" rule was found on the validation drives and confirmed on the train set.
- Prompts, the layout and the reasoning budget were chosen on v1.0-mini. The 32B model is quantised to 4 bits.
- Program execution is not new; CLEVR and Neural-Symbolic VQA did it first. What this project adds is an auditable, same-checkpoint comparison on driving questions, with scene-disjoint splits and failure traces.
- A wider program language can do worse. On 280 broader questions, a typed CLEVR-style language covers 98.5 % of the benchmark. At 1.5B, though, it answers only 34.6 % of questions and reaches 29.6 % accuracy, against 52.1 % for the direct answer.
- The comparison is at fixed oracle perception on four question families. It does not show that language models cannot reason about space.

There is more in [`docs/LITERATURE.md`](docs/LITERATURE.md) and the [report](report/rapport_npm3d.pdf).

## Repository layout

| Path | Role |
| --- | --- |
| `spatial_reasoning/` | Frozen-model comparison, parsing, geometry and program execution |
| `spatial_reasoning/manifests/` | Frozen question sets and exclusions |
| `spatial_reasoning/results/` | Saved answers, programs, summaries and diagnostics |
| `tests/` | Protocol, parser, executor and runtime checks |
| `report/` | LaTeX report and compiled PDF |
| `web/` | Source of the website (Vite + TypeScript, D3, Three.js). See [`docs/SITE.md`](docs/SITE.md) |
| `tools/` | Data and LiDAR export for the site, link checks |
| `models/`, `training/`, `distillation/`, `data/`, `evaluation/`, `inference/` | Original 2025 course prototype |

To work on the site: `cd web && npm ci && npm run dev`. On every push to `main`, GitHub Actions tests the site, builds it and publishes it.

## History and credits

The original two-person NPM3D course project was forked from [`oomcth/npm-3D`](https://github.com/oomcth/npm-3D). It wired together a reduced PointNet/VAT/GPT-2 pipeline but never completed a real training run. That code is still here, kept apart from the revised study. Scenes come from [nuScenes](https://www.nuscenes.org/) v1.0-mini (CC BY-NC-SA 4.0) and questions from [NuScenes-QA](https://arxiv.org/abs/2305.14836).

Nassim Arifette, PhD student at École polytechnique. Master MVA, ENS Paris-Saclay.
