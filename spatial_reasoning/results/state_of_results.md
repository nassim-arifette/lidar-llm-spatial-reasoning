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

The gain is positive on each of the three evaluation drives (+0.212, +0.188,
+0.138). The question-level sign test assumes independence that repeated
questions within drives do not satisfy; it is descriptive evidence, not a
population-level significance claim. The counting
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

* **A systematically under-counts.** Where the answer is 7, 9 or 10 it answers 1, 3
  and 2; on preamble questions it answers 1 almost regardless of the scene.
  Its under-counting (1.84–2.08) dwarfs its over-counting (0.31–0.35) — the
  measurable error pattern, not a measurement of internal attention. Its answer format was
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

## 0b. Does scale close the gap?

Same 200 held-out questions, same three drives, same prompts and decoding.
Sources: `llm_modes/`, `llm_modes_7b/`, `llm_modes_32b/`.

| Model | precision | A direct | B executed | A MAE | B MAE | B answers | invalid | A only / B only | sign test |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | fp16 | 0.505 | 0.685 | 2.16 | 0.68 | 0.910 | 8 | 31 / 67 | 3.5e-04 |
| Qwen2.5-7B | fp16 | **0.505** | 0.825 | 1.84 | 0.44 | 0.985 | 1 | 15 / 79 | 1.2e-11 |
| Qwen2.5-32B | int4 | **0.580** | 0.845 | 1.65 | 0.60 | 0.995 | **0** | 13 / 66 | 1.1e-09 |

* The direct mode gains **0.075 across a 21× increase in parameters**, and
  nothing between 1.5B and 7B. Not a cache artefact: the 7B changes its answer
  on 89 of 200 and flips correctness on 52, 26 in each direction.
* What scale fixes is **program validity**: 8 → 1 → 0 invalid programs,
  answered fraction 0.910 → 0.995.
* The 32B row is 4-bit (65 GB does not fit on a 40 GB card), so size and precision change together. Without a matched full-precision
  run, the direction and magnitude of the quantization effect are unknown.
* Scene-dependence, correct across every scene of a group: direct 1/13, 1/13,
  4/13; executed 13/13, 12/13, 13/13.

### Hosted checkpoints: a boundary to the observed advantage

Through the hosted gateway, same 200 questions, all three producing 200/200
valid programs:

| Model | answers | A | B | A out tok | B out tok | A scene | B scene | paired p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen-3-32b` | one step | 0.560 | **0.840** | 2.0 | 57.9 | 3/13 | 13/13 | 2.3e-10 |
| `qwen3-next-80b-a3b-instruct` | one step | 0.610 | **0.845** | 2.0 | 57.6 | 3/13 | 13/13 | 2.3e-08 |
| `qwen3.8-27b` | reasoning | **0.840** | 0.845 | 523.9 | 156.5 | 13/13 | 13/13 | 1.0 |

`qwen-3-32b` lands within 0.005 of Qwen2.5-32B. The near parity for
`qwen3.8-27b` shows that the executed advantage is not universal across the
saved endpoints. It does **not** isolate reasoning as its cause: model family,
training, architecture and effective output budget all change. Qwen3-Next-80B
is a mixture of experts with 80B total but 3B active parameters, so nominal
size is not a compute control.

The reported output-token counts (about 58, 157 or 524 at accuracy around
0.84) are descriptive provider usage, not measured FLOPs, latency or price.
The older claim of a nine-fold inference-cost saving is withdrawn.

Caveat on gateway rows: `temperature=0` is *requested*, not the exact greedy
loop the local runtime performs; token counts are the provider's; and the
output cap is not enforced — `qwen3.8-27b` exceeded 256 up to 8982 on one
question. These rows are comparable in accuracy, not in budget.

---

## 0c. Does a wider target language help?

`functional.py` is CLEVR's representation adapted to nuScenes: typed nodes, 19
functions, indices into earlier nodes, type-checked before execution. A second
manifest built by the same rules on the same drives covers **5830 of 5916
questions (98.5 %)** against 1626 (27.5 %) for the flat scope.
Source: `wide_1p5b/`.

| Qwen2.5-1.5B, wide scope (n = 280) | A direct | B flat | C typed |
|---|---:|---:|---:|
| Accuracy (all) | **0.521** | 0.457 | 0.296 |
| Answers | 1.000 | 0.571 | 0.346 |
| Accuracy when answering | 0.521 | 0.800 | **0.856** |

Paired: direct beats typed 102 to 39 (p = 1.1e-07); flat beats typed 65 to 20
(p = 1.0e-06). Direct beats flat 72 to 54, which is **not** significant
(p = 0.13) — on the wider question set the flat language's advantage
disappears.

The typed language is the most accurate of the three *when it produces a
program*, and it produces one for barely a third of the questions.

**Where it fails.** Of 280 typed programs 165 do not type-check: 70 use a
constant outside the vocabulary, 85 wire the wrong type, and **51 of those 85
are a missing `unique`** — a set passed where `relate`, `query_status` or
`same_status` needs one object. On the same questions the flat language
produces 240 valid programs and loses 80 at execution: 42 ambiguous
references, 38 missing.

**The same difficulty, moved.** Both languages are defeated by resolving a
definite reference in a real scene. The flat one does uniqueness implicitly
inside the executor and fails there; the typed one makes it an explicit node
and fails at the type check. Making the step visible did not make the model
take it.

Checked before claiming it: the flat language rejects `"trailers"` exactly as
the typed one does, so this is not an artefact of normalisation.

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
  program beats answering directly, because no direct condition was run on all 5,916 questions. The
  completed paired comparison on frozen IDs is in section 0.

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

This is a development-set reference-program diagnostic, not a formal ceiling
on evaluation accuracy. Different programs can agree with an answer for
different reasons. All nine
disagreements are read and categorised in `manifests/disagreement_categories.json`
— eight are boundary disagreements on the diagonal sectors, one is a question
whose definite reference matches 22 objects.

The six 60-degree sectors are specified by NuScenes-QA Equation 2
(https://arxiv.org/html/2305.14836v2). An archived diagnostic over 68
ego-anchored directional questions gave 76.5% exact agreement for this layout,
72.1% for a mirrored y axis, 67.6% for camera frusta, 61.8% for octants and
51.5% for quadrants. That diagnostic includes evaluation examples and is not
an independent held-out confirmation or a discovery of the published rule.

## 4. Completion and reproduction status

The direct-answer and paired-comparison implementations, local size curve,
hosted rows above and 1.5B typed-language experiment are complete and archived.
`python -m spatial_reasoning.audit_results` verifies all eight comparisons
(1,580 rows) against their frozen manifests and saved summaries without a GPU.
See `artifact_audit.json` for input hashes and the scope of that verification.

Two optional jobs mentioned at the interruption have no complete local
row/summary archive: hosted Qwen3-14B at 4,096 tokens, and the wide-scope
Qwen2.5-32B run. Neither is a result in this document. Their exact restart
commands and interpretation limits are in `../../../docs/REPRODUCIBILITY.md`
(from the repository root: `docs/REPRODUCIBILITY.md`).

The current study uses ground-truth boxes and statuses, a selected subset of
question families, and only three evaluation drives. It evaluates reasoning
at fixed perception; it does not measure end-to-end LiDAR performance.
