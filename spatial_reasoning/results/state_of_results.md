# Where the numbers stand

Every figure here is traceable to a file in this directory or to a run named
below.

The comparison the project exists to make — the same language model answering
directly against the same model writing a program — **has been run**; it is
section 0. The tables after it are the earlier work, kept because they are
what the report's argument was built on, and because two of them withdraw a
claim.

All object lists are **nuScenes ground-truth annotations**, not a detector.
That is an oracle condition: it fixes perception so that the comparison is
about what each system does with a scene, not about who sees it better.

---

## 0. The central comparison — one model, two ways of using it

Source: `llm_modes/summary_{dev,eval}.json` and `rows_{dev,eval}.jsonl`
(Colab, T4). Qwen2.5-1.5B-Instruct on **both** sides: same checkpoint, same
loader, same greedy decoding, same 256-token budget, same object lists at the
same rounding, same conventions stated. The example set
(`manifests/first_comparison.json`) was fixed before either side ran, and the
evaluation split shares no scene with development.

Nothing was changed between the two splits. The development result was
measured twice — once before a fix to the cost accounting and batching, once
after — and every accuracy figure was identical, so the batching change is not
a confound.

| | dev A | dev B | **eval A** | **eval B** |
|---|---:|---:|---:|---:|
| n | 100 | 100 | 200 | 200 |
| Accuracy (all) | 0.510 | **0.740** | 0.505 | **0.685** |
| Answers | 100 % | 86 % | 100 % | 91 % |
| Accuracy when answering | 0.510 | 0.860 | 0.505 | 0.753 |
| Numeric coverage | 1.000 | 0.946 | 1.000 | 0.988 |
| Under-counting | 2.08 | 0.54 | 1.84 | 0.34 |
| Over-counting | 0.35 | 0.00 | 0.31 | 0.34 |
| MAE | 2.43 | **0.54** | 2.16 | **0.68** |
| Tokens in / out | 124 311 / 200 | 112 095 / 5 756 | 217 272 / 400 | 178 402 / 9 731 |
| Generation time | 47.3 s | 59.3 s | 58.8 s | 101.0 s |

**Paired, which is the measurement that matters:**

| | A only | B only | both | neither | sign test |
|---|---:|---:|---:|---:|---:|
| dev | 15 | **38** | 36 | 11 | p = 0.0022 |
| eval | 31 | **67** | 70 | 32 | **p = 0.00035** |

Not sampling noise, and it survives the move to unseen scenes. The counting
gain is not abstention either — both modes produce an integer on practically
the same questions (1.000 against 0.988 on eval).

| Family | dev A | dev B | eval A | eval B |
|---|---:|---:|---:|---:|
| `category_status` | 0.640 | **1.000** | 0.547 | **0.962** |
| `count_exist_category` | **0.760** | 0.720 | 0.680 | **0.840** |
| `category_anchor_relation` | 0.320 | 0.440 | 0.352 | 0.444 |
| `category_ego_region` | 0.320 | **0.800** | 0.442 | 0.465 |

**Only one family is stable across the two splits.** The attribute family is a
large, consistent win (+0.36, +0.42). The other three move a lot between dev
and eval — `category_ego_region` goes from +0.48 to +0.02, and
`count_exist_category` changes sign. At 25–54 questions a family, these
per-family numbers are too noisy to carry an argument; the aggregate and the
paired table are what the result rests on.

**Scene-dependence control** — questions appearing verbatim on two scenes with
different reference answers:

| | groups | A gave one answer to all | A right on every scene | B gave one answer to all | B right on every scene |
|---|---:|---:|---:|---:|---:|
| dev | 5 | 2 | 0 | 1 | 4 |
| eval | 13 | 4 | **1** | **0** | **13** |

On the evaluation split the executed system answers the same question
differently for every scene that calls for a different answer, and is right in
all thirteen groups. The direct model is right across a group once in
thirteen. No aggregate accuracy reveals that.

What each side's failures look like:

* **A does not scan the list.** Where the answer is 7, 9 or 10 it answers 1, 3
  and 2; on preamble questions it answers 1 almost regardless of the scene.
  Its under-counting (1.84–2.08) dwarfs its over-counting (0.31–0.35) — the
  signature of not looking rather than of looking badly. Its answer format was
  never the problem: all 300 direct outputs were readable.
* **B failed on 14 % of dev and 9 % of eval**, all counted as wrong above:
  invalid programs (12 and 8) plus refusals for an ambiguous or missing
  reference (2 and 10). Every one of the twelve invalid dev programs has the
  same cause — for "things" the model emits `"category": "*"`, outside the
  vocabulary, although the prompt lists the ten classes *and* states the rule
  mapping "thing" to `object`. That is the model failing at this size, not a
  gap in the instructions, and it is the clearest single thing to re-test with
  a larger model.

**What it does not show.** B brings code that already knows the geometry. This
says that *this* division of labour helps on *these* questions at *this* model
size. It is not evidence that language models cannot reason about space.

---

## 1. No training, all 5916 questions

Same questions, same objects (`min_lidar_points=0`), same six 60° sectors,
nothing fitted. Source: `parser_comparison_v3` (Colab, T4, 2889 s) and a local
re-scoring of the rule grammars.

| System | Sees | Accuracy | Answers | Accuracy when it answers |
|---|---|---:|---:|---:|
| Majority answer | nothing | 0.255 | 100 % | 0.255 |
| Rule grammar `LEGACY` → executor | question | 0.156 | 24.7 % | 0.632 |
| Rule grammar `STRICT` → executor | question | 0.059 | 6.5 % | 0.904 |
| Rule grammar `EXPRESSIVE` → executor | question | 0.231 | 26.9 % | 0.856 |
| **Qwen2.5-1.5B → program → executor** | question | **0.427** | 56.3 % | **0.759** |

The direct mode is not in this table: it is in section 0, on the frozen
manifest, which is the only place the two are run on the same examples.

Reading it:

* The three rule grammars are the same parser under three policies for what it
  cannot express. `LEGACY` drops the part it does not understand and answers
  anyway; `STRICT` refuses; `EXPRESSIVE` represents it. At almost identical
  coverage, `LEGACY` is right 63.2 % of the time it answers and `EXPRESSIVE`
  85.6 % — the 22-point gap is the cost of answering a question other than the
  one asked.
* Qwen writing programs answers 56.3 % of the questions at 0.759, against a
  blind prior of 0.255. Its 0.427 overall is held down by abstention, not by
  wrong answers: 5.2 % refusals and 11.0 % unparseable output.
* **The empty cell is the experiment.** Nothing above shows that writing a
  program beats answering directly, because the direct condition has not been
  run.

## 2. Trained baselines — a different protocol, kept as reference

Source: `all_results_nuscenes.json`. These were trained on a per-seed split of
3068 items and scored on 723–981 held-out items, so they are **not comparable
row-for-row** with the table above, which trains nothing and scores all 5916.

| System | Accuracy (3 seeds) | Answers |
|---|---:|---:|
| `question_only` (blind to the scene) | 0.443 ± 0.029 | 100 % |
| `deepsets` | 0.455 ± 0.040 | 100 % |
| `counter` | 0.456 ± 0.039 | 100 % |
| `symbolic_abstain` (pre-change grammar) | 0.131 ± 0.005 | 31.3 % |
| `symbolic_fallback` (pre-change grammar) | 0.277 ± 0.001 | 100 % |

Two things in this block deserve to be stated rather than averaged:

* **`deepsets` does not reliably beat the blind baseline.** Paired by seed the
  deltas are −0.021, +0.032, +0.025. It loses on one of the three. An earlier
  note in this project claimed it won on all three; that claim does not hold on
  these files and is withdrawn.
* **Swapping ground-truth objects for detections changes almost nothing**:
  `deepsets` moves +0.019, −0.030, +0.010 across the seeds. A model whose score
  is unchanged when the 3D input is replaced has not been shown to use it.

The symbolic rows here used the grammar before the status and anchoring work,
which is why they are far below the `EXPRESSIVE` row in the first table.

## 3. The executor on its own, with correct programs

Source: `executor_verification/summary_dev.json`, 100 development examples of
`manifests/first_comparison.json`, no model involved.

| Family | Agreement with nuScenes-QA |
|---|---:|
| `count_exist_category` | 25/25 |
| `category_status` | 25/25 |
| `category_ego_region` | 22/25 |
| `category_anchor_relation` | 19/25 |
| **Total** | **91/100** |

This is the ceiling of the executed branch on the four in-scope families: what
it would score if the model always wrote the right program. All nine
disagreements are read and categorised in `manifests/disagreement_categories.json`
— eight are boundary disagreements on the diagonal sectors, one is a question
whose definite reference matches 22 objects.

The sector convention was checked against the gold answers rather than assumed:
over the 68 ego-anchored directional questions of the manifest, uniform 60°
sectors with no range limit score 76.5 % exact, against 72.1 % for a mirrored
*y* axis, 67.6 % for the six nuScenes camera fields of view, 61.8 % for 45°
octants and 51.5 % for overlapping quadrants; every maximum-range cut from
25 m to 60 m makes it worse.

## 4. What has to happen next

1. `direct_qa.py` — the same Qwen2.5-1.5B-Instruct, same checkpoint and
   decoding, answering from the question and the full object list.
2. `compare_llm_modes.py` — both modes on the same question ids, one row per
   question, separate caches, no silent truncation.
3. Read the result before widening anything: a bigger model and a composable
   DSL fix different failures, and running both at once says which did neither.
