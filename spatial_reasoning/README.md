# `spatial_reasoning` — spatial QA at fixed perception

This package is the revised contribution of the NPM3D LiDAR-LLM project. It
replaces the GPT-2 answer head for one restricted family of questions
(counting, existence, comparison) and makes the alternative *measurable*.

**The question it answers.** At fixed perception and comparable budget, is it
better to learn the answer to a spatial question directly, or to parse the
question into an explicit program and compute the answer on the detected
objects?

No GPT-4, no GPT-2, no LLaMA. Everything here runs on a CPU.

## Pipeline

```
pre-extracted 3D detections        (perception, computed once, never trained)
        |
        v
object list: class, box, position  objects.py
        |
question --> structured query      question_parser.py   (rule-based, declines
        |                                                what it cannot parse)
        v
select objects + geometric ops     programs.py / regions.py
        |
        v
answer
```

## The four systems compared

| system | sees the scene? | trained? | where the answer comes from |
|---|---|---|---|
| `question_only` | no | yes | the language prior alone |
| `deepsets` | yes | yes | a classifier over the answer vocabulary |
| `counter` | yes | yes | a differentiable sum of per-object relevances |
| `symbolic` | yes | **no** | parsing, then exact computation |

`question_only` is the control that matters: any system not clearly above it
has not been shown to use the 3D input.

## Headline result

Balanced synthetic corpus, 1600 questions over 200 samples, scene-level
splits, mean over 3 seeds. Full tables in [`results/summary.md`](results/summary.md).

| system | accuracy | count MAE | contrast pairs | params |
|---|---|---|---|---|
| question only | 0.656 | 0.722 | 0.000 | 35 863 |
| deep sets | 0.655 | 0.654 | 0.022 | 120 855 |
| deep sets + counter | 0.644 | 0.534 | 0.076 | 129 176 |
| **symbolic** | **0.916** | **0.132** | **0.643** | **0** |

Neither learned model beats the question-only control. Every symbolic error
traces to perception; none to question understanding or geometry. Under a
30% detection drop rate the symbolic system falls to 0.750 — still above
every learned system at every noise level tested.

The same corpus *without* answer balancing (75% of answers mean "nothing
there") gives 0.757 / 0.766 / 0.775 / 0.950: the differences compress and
the learned models appear to use the scene when contrast pairs show they do
not. Run both with `--max-trivial-fraction 0.35` and `1.0`.

## Running it

```bash
# fully synthetic, no download, a few minutes on a CPU
python -m spatial_reasoning.run_experiments --source synthetic

# real data: CenterPoint detections + a nuScenes-QA style question file
python -m spatial_reasoning.run_experiments --source files \
    --detections detections.json --qa questions.json --oracle gt_boxes.json
```

Also reachable from the project entry point:

```bash
python main.py --mode spatial --spatial_args -- --source synthetic
```

Outputs land in `outputs/spatial_reasoning/` as `results.json` and a
ready-to-read `results.md`.

Tests: `python -m pytest tests/test_spatial_reasoning.py -q`

## Conventions worth knowing before reading the code

* **Ego frame**: `x` forward, `y` left, `z` up. Defined in `objects.py`, used
  nowhere else implicitly.
* **Directions are 45° bearing sectors**, not half-planes, so an object is in
  at most one of front / front-left / left / … (`regions.py`).
* **Splits are by nuScenes scene**, never by question: a question-level split
  lets a model memorise a scene's object counts (`datasets.split_by_scene`).
* **The parser refuses rather than guesses.** Coverage and accuracy are
  reported separately, and both symbolic variants (abstain / majority
  fallback) appear in the results table.
* **Ambiguous references are refused.** "Left of the truck" with three trucks
  returns `AMBIGUOUS_REFERENCE` instead of silently picking the first.

## Honest limits

* The synthetic corpus has no occlusion, no range-dependent class confusion
  and a uniform object layout. It bounds what is worth trying on nuScenes; it
  does not stand in for it.
* The counterfactual probe computes its reference answers with the same
  executor the symbolic system uses, so that system passes it by construction.
  It is informative for the learned baselines only.
* `tracemalloc` measures Python allocations, so memory for the torch models is
  under-reported. Time is comparable across all systems; memory is comparable
  only between the two torch models.
