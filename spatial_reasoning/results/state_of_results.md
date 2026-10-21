# Where the numbers stand

Every figure here is traceable to a file in this directory or to a run named
below. The point of the table is as much what is missing as what is in it: the
comparison the project exists to make — the same language model answering
directly against the same model writing a program — has one of its two cells
empty.

All object lists are **nuScenes ground-truth annotations**, not a detector.
That is an oracle condition: it fixes perception so that the comparison is
about what each system does with a scene, not about who sees it better.

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
| **Qwen2.5-1.5B direct** | question + objects | *not run* | — | — |

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
