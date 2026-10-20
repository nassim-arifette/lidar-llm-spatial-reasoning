# LiDAR-LLM: spatial reasoning at fixed perception

Course project for **NPM3D** (Nuages de Points et Modélisation 3D), Master MVA,
built around *LiDAR-LLM: Exploring the Potential of Large Language Models for
3D LiDAR Understanding*.

The repository has two layers.

## 1. The original re-implementation (2025)

A reduced LiDAR-LLM pipeline: mini-PointNet encoder → view-aware transformer
with six view-position embeddings → MLP adapter → GPT-2 fine-tuned with LoRA.
Data is nuScenes-mini filtered to the samples covered by nu-Caption,
nu-Grounding and nuScenes-QA.

It was developed on laptops with ~6 GB of RAM and no GPU, and never completed
a real training run. What it demonstrated is that the architecture can be
wired up and that gradients flow — not that the model understands a scene.
That limitation is stated plainly rather than worked around.

> **Origin.** The 2025 version was a two-person course project, forked from
> [`oomcth/npm-3D`](https://github.com/oomcth/npm-3D), which is the upstream
> of this repository and carries its commit history. My own share of that
> first version was the evaluation metrics, the data collection and
> formatting, and parts of the datasets, models and inference scripts.

## 2. The 2026 revision: `spatial_reasoning/`

Rather than retry the same architecture with a smaller language model, the
revision changes the claim being tested:

> At fixed perception and comparable budget, is it better to **learn** the
> answer to a spatial question, or to parse the question into an explicit
> program and **compute** the answer on the detected objects?

Detections are consumed, never trained. Four systems are compared on identical
object lists: a question-only language prior, a deep-sets answer classifier, a
differentiable counter, and a parameter-free symbolic program. Splits are by
scene; the protocol adds contrast pairs, counterfactual object moves, a
detector-degradation sweep, and a per-stage error decomposition.

No GPT-4, no GPT-2, no LLaMA. It runs on a CPU in minutes.

See [`spatial_reasoning/README.md`](spatial_reasoning/README.md) for the
design, and [`report/rapport_npm3d.tex`](report/rapport_npm3d.tex) for the
written report. This revision is my own work.

### Quick start

```bash
pip install numpy torch pytest          # the revision needs no more than this
python -m spatial_reasoning.run_experiments --source synthetic
python -m pytest tests/ -q
```

Running the original LiDAR-LLM pipeline additionally requires
`transformers`, `peft` and `torch_geometric`:

```bash
python main.py --mode train --data_path <dir>
```

## Layout

```
spatial_reasoning/   the 2026 revision (self-contained, CPU-only)
report/              LaTeX report
tests/               unit tests for spatial_reasoning
models/              original: PointNet encoder, VAT, GPT-2 + LoRA head
training/            original: training loop and callbacks
distillation/        original: distillation of the fine-tuned LLM
data/                original: dataset loaders, preprocessing, augmentation
evaluation/          original: BLEU, BERTScore, BEV mIoU, top-1
inference/           original: predictor
utils/               original: logging and visualisation
```
