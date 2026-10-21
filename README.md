# LiDAR spatial reasoning: answering a question versus computing its answer

Course project for **NPM3D** (Nuages de Points et Modélisation 3D), Master MVA,
built around *LiDAR-LLM: Exploring the Potential of Large Language Models for
3D LiDAR Understanding*.

The question the repository now tests, in one sentence:

> Given **one** frozen language model and **one** object list, is it more
> reliable to produce the answer to a spatial question, or to produce a small
> program and let a deterministic executor compute the answer?

Posed between two *uses of the same model*, it is answerable without
confounds. A learned answer head and a program-writing model differ in
architecture, training and inputs; the same checkpoint, the same greedy
decoding and the same objects differ in exactly one respect.

## The result

Qwen2.5-1.5B-Instruct on both sides, 200 held-out questions from three
nuScenes drives that never appear in development:

| | A — direct | B — program + executor |
|---|---:|---:|
| Accuracy | 0.505 | **0.685** |
| Answers | 100 % | 91 % |
| Counting MAE | 2.16 | **0.68** |
| Numeric coverage | 1.000 | 0.988 |

Paired question by question: **B alone right on 67, A alone on 31**, both on
70, neither on 32 — sign test *p* = 3.5 × 10⁻⁴. Refusals and invalid programs
count as wrong answers throughout.

The sharper instrument is a **scene-dependence control**. Thirteen evaluation
questions appear verbatim on two scenes with different reference answers. The
executed system never gives one answer to a whole group and is right on all
thirteen; the direct model gives one answer to both in four groups and is right
across a group once.

**What this does not show.** B brings hand-written code that implements the
benchmark's geometry, verified separately at 91/100. Its ceiling is high by
construction, so the real question is whether the *model* can write the program
— it does validly 92–96 % of the time. This is not evidence that language
models cannot reason about space, and the evaluation rests on three drives.

## Two findings about the data

Both are independent of the comparison and likely more reusable than it.

**The six nuScenes-QA relations are uniform 60° bearing sectors**, recovered by
fitting the reference answers rather than assumed. Over 68 ego-anchored
directional questions: 76.5 % exact, against 72.1 % for a mirrored *y* axis,
67.6 % for the six nuScenes camera frusta (computed from the real calibration
and intrinsics), 61.8 % for 45° octants and 51.5 % for overlapping quadrants.
Every maximum-range cut from 25 m to 60 m makes the fit worse. The appealing
six-relations-six-cameras reading is wrong.

**23 % of nuScenes annotations carry zero LiDAR returns**, and nuScenes-QA
counts them anyway. Keeping only annotations the sensor hit — the obvious move
when building a scene from a point cloud — drops 4 278 of 18 538 objects and
costs **49 accuracy points** on otherwise correct programs (0.824 → 0.336,
paired over 125 counting questions).

## Where the executed system fails

Triaged over its 63 evaluation failures (`spatial_reasoning/analyse_failures.py`):

| Failure | n | share | what it asks for |
|---|---:|---:|---|
| Program says something else | 27 | 43 % | prompt, examples, bigger model |
| Program right, answer wrong | 18 | 29 % | the executor and the data |
| Refused at execution | 10 | 16 % | ambiguous or missing reference |
| Malformed output | 8 | 13 % | a budget, or constrained decoding |

71 % of failures happen *before* the executor. Widening the target language —
the obvious next move, and one deliberately not made — would address almost
none of them. The single largest failure is a return-type mistake: in 22 cases
the model writes `count` for a question asking `exists`, so the answer is an
integer that can never match a yes/no reference.

## How the comparison is kept honest

- **The example set is frozen before either side runs**
  (`spatial_reasoning/manifests/first_comparison.json`): 100 development and
  200 evaluation questions, **split by scene** so no drive appears on both
  sides. Consecutive samples of one drive show the same vehicles.
- **Families are assigned from the question's surface structure**, never by
  checking which questions a parser copes with — that would select the
  examples favouring the executed branch. Of 5 916 questions, 1 626 are in
  scope; the 4 290 excluded are recorded with their reason.
- **The reference answer is not reachable by either inference function**
  (`ManifestItem.payload()`).
- **The executor was verified before any model ran**: 91/100 on development
  with reference programs and no model involved, every disagreement read and
  categorised in `manifests/disagreement_categories.json`.
- **Sector boundaries were deliberately not fitted** to close the residual. An
  executor tuned on the reference answers would help the branch that computes
  with those boundaries far more than the branch that only reads them in prose.
- **Caches are separate and keyed differently.** A program depends on the
  question; a direct answer depends on the question *and* the scene. Sharing a
  key would destroy the scene-dependence control.
- **No silent truncation.** An over-long prompt is refused, not trimmed, and
  every generation records whether it stopped on its own. An earlier run lost
  26 % of its programs to a 96-token budget, 97 % of them to truncation — a
  configuration choice that had been read as a model failure.

Nothing is trained. The whole comparison is about three minutes on one free T4.

## Quick start

```bash
pip install numpy torch transformers pytest
```

Metadata only, no login, about a minute:

```bash
mkdir -p data/data/nuscenes && curl -sL https://www.nuscenes.org/data/v1.0-mini.tgz | tar -xz -C data/data/nuscenes v1.0-mini/
```

Check the executor with reference programs — no GPU, no model:

```bash
python -m spatial_reasoning.verify_executor --manifest spatial_reasoning/manifests/first_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split dev
```

Run both modes on the same questions (needs a GPU for the model):

```bash
python -m spatial_reasoning.compare_llm_modes --manifest spatial_reasoning/manifests/first_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split eval
```

Triage the failures:

```bash
python -m spatial_reasoning.analyse_failures --rows spatial_reasoning/results/llm_modes/rows_eval.jsonl
```

Tests:

```bash
python -m pytest tests/ -q
```

## Results and report

Every figure is traceable to a file:

- [`spatial_reasoning/results/state_of_results.md`](spatial_reasoning/results/state_of_results.md)
  — all numbers with their sources, including the ones that withdraw an
  earlier claim.
- `spatial_reasoning/results/llm_modes/` — per-question rows, summaries and
  the failure triage.
- [`report/rapport_npm3d.tex`](report/rapport_npm3d.tex) — the written report.

**Earlier results are kept and labelled, not deleted.** The synthetic campaign
in `results/all_results.json` used 45° octants and is a preliminary study, not
a result on nuScenes-QA. Its "detector" condition **simulates** detector
failures on ground-truth boxes; no 3D detector was run anywhere in this
project, and no claim about detection quality follows from it. On real
nuScenes-QA the learned baselines do **not** reliably beat a scene-blind prior
(paired per seed: −0.021, +0.032, +0.025) and are barely affected by replacing
ground-truth objects with simulated detections (+0.019, −0.030, +0.010).

## Deliberately not done

- **A larger checkpoint.** It is the one experiment that would settle the open
  question — it separates "the model cannot write the program" from "the model
  cannot count from a table" — but it changes nothing about the protocol and
  was out of scope here.
- **A composable, CLEVR-style typed operation catalogue**
  (`spatial_reasoning/functional.py`) — implemented, tested by hand, and
  deliberately **not wired into any experiment**. It would widen coverage from
  four families to the whole benchmark, but the failure triage says the current
  failures are overwhelmingly not expressiveness. Building a new language and
  changing model size at once would leave neither attributable.

## The two layers of this repository

### 1. The original re-implementation (2025)

A reduced LiDAR-LLM pipeline: mini-PointNet encoder → view-aware transformer
with six view-position embeddings → MLP adapter → GPT-2 fine-tuned with LoRA,
on nuScenes-mini.

It was developed on laptops with ~6 GB of RAM and no GPU, and never completed
a real training run. What it demonstrated is that the architecture can be wired
up and that gradients flow — not that the model understands a scene. That
limitation is stated plainly rather than worked around.

> **Origin.** The 2025 version was a two-person course project, forked from
> [`oomcth/npm-3D`](https://github.com/oomcth/npm-3D), which is the upstream of
> this repository and carries its commit history. My own share of that first
> version was the evaluation metrics, the data collection and formatting, and
> parts of the datasets, models and inference scripts.

### 2. The revision: `spatial_reasoning/`

Self-contained, documented in
[`spatial_reasoning/README.md`](spatial_reasoning/README.md). This revision is
my own work.

```
spatial_reasoning/
  manifest.py            the frozen example set, split by scene
  objects.py             object lists in the ego frame
  regions.py             bearing sectors and the layouts
  question_parser.py     the deterministic rule grammar
  llm_parser.py          question -> program, with a frozen Qwen
  direct_qa.py           question + objects -> answer, same Qwen
  qwen_runtime.py        one loader and decoder, shared by both modes
  dsl.py, programs.py    the executors
  functional.py          the CLEVR-style catalogue (not wired in)
  compare_llm_modes.py   the paired comparison
  verify_executor.py     the executor check, no model involved
  analyse_failures.py    the failure triage
report/                  LaTeX report
tests/                   the test suite for the above
models/ training/ distillation/ data/ evaluation/ inference/ utils/
                         the original 2025 pipeline
```
