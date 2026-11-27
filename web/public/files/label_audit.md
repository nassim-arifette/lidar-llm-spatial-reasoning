# What the nuScenes-QA answers actually encode

Three conventions of the released answers differ from what the paper's text and
the question wording suggest. Each was found while explaining executor
disagreements, then tested on data that played no part in finding it. None of
them is a claim about the generation code, which was not read: they are
descriptions of the released labels.

All commands run from the repository root. `data/` is not versioned; see
`docs/REPRODUCIBILITY.md` for where the files come from.

## 1. The spatial relations do not distinguish left from right

With `a` the unsigned angle between the ego heading and the direction to the
object (0 to 180 degrees):

| relation in the question | true in the labels when |
|---|---|
| front | a ≤ 30 |
| front left | 30 < a ≤ 90, on either side |
| front right | never |
| back right | 90 < a ≤ 150, on either side |
| back left | never |
| back | a > 150 |

This is what an `arccos` of a normalised dot product gives (equation 2 of the
nuScenes-QA paper) when the sign of the angle is never recovered. The layout
`nuscenes_qa_as_labelled` in `spatial_reasoning/regions.py` implements it;
`nuscenes_qa` stays the signed reading.

**Without geometry.** Take every counting or existence question that names
exactly one relation and count how often the answer is `0` or `no`
(`python -m spatial_reasoning.audit_labels`, saved in
`results/label_audit/official_release.json`):

| relation | train: questions | answer 0/no | val: questions | answer 0/no |
|---|---:|---:|---:|---:|
| front left | 19,243 | 22.8% | 4,156 | 22.0% |
| **front right** | 5,070 | **100%** | 1,134 | **100%** |
| **back left** | 5,137 | **100%** | 1,051 | **100%** |
| back right | 19,373 | 23.4% | 4,103 | 23.9% |
| front | 19,345 | 20.8% | 4,427 | 20.2% |
| back | 19,778 | 20.3% | 4,284 | 21.2% |

12,392 questions about "front right" or "back left" across 34,117 keyframes,
and not one has an object there. The same holds on v1.0-mini (37/37 and 41/41).

**With geometry, on data not used to find the rule.** The rule was read off
evaluation failures of the first comparison, so it is checked elsewhere:

| check | signed sectors | as labelled |
|---|---:|---:|
| mini development split, hand-verified reference programs (100) | 91 | 99 |
| mini evaluation split, same procedure (200) | 168 | 195 |
| official validation set, rule parser, all it answers (25,973) | 82.8% | 91.9% |
| … of which front left (2,638) | 53.4% | 91.4% |
| … front right (1,092) | 73.7% | 100% |
| … back left (1,018) | 75.0% | 100% |
| … back right (2,535) | 57.2% | 90.3% |
| … front, back (5,804) | 87.7% | 87.7% |

The one development disagreement left is a question whose reference object is
not unique. Sources: `results/executor_verification_as_labelled/`,
`results/label_audit/val_rule_parser_by_relation.json`. The earlier
categorisation of the development disagreements as "diagonal sector boundary"
effects (`manifests/disagreement_categories.json`) is superseded: eight of the
nine were this.

An earlier pass over the directional evaluation questions, re-executing the
32B model's programs, found that no rotation or mirror of the frame explains
the disagreements: the best of them reached 71/97, the unsigned rule 94/97.

## 2. "Things" are every annotation, and the ego vehicle is one of them

For "How many things are there?" the answer equals the number of annotations
of the keyframe, of every category, plus one:

| the answer equals | train (2,080 questions) | val (648) |
|---|---:|---:|
| every annotation | 0 | 0 |
| **every annotation + 1** | **2,080** | **648** |
| the ten named classes | 0 | 0 |
| the ten named classes + 1 | 1,610 | 495 |
| annotations with a LiDAR return + 1 | 973 | 304 |

So a "thing" includes the classes no question ever names (pushable objects,
debris, bicycle racks, animals), includes boxes with no LiDAR return, and
includes the ego vehicle. The ego is also a thing in relations between objects
("how many things are to the back of the bus" can count the ego), though never
to the front or back *of me*. This was found on the validation drives and
confirmed on train, which was not looked at before.

The object-list option `things="all"` (`nuscenes_adapter.load_nuscenes_scenes`,
`--things all`) keeps unnamed classes as `other` and appends the ego at the
origin; both match only "thing"/"object". The default, `named`, is what every
archived run used. On v1.0-mini the difference is one question in 300.

Executor agreement with hand-free reference programs on the 1,460 questions of
`manifests/val_drives.json` (146 drives, none in v1.0-mini):

| object list | signed sectors | as labelled |
|---|---:|---:|
| ten named classes | 85.1% | 92.8% |
| every annotation + ego | 89.6% | **97.5%** |

Source: `results/executor_verification_val_drives/`.

## 3. Objects with no LiDAR return are counted

Already reported, and unchanged by the two findings above: 4,278 of the 18,538
annotations of v1.0-mini have zero LiDAR points, and the answers count them.
Holding the programs fixed and changing only the object list, on the counting
questions both conditions answer:

| counting questions | layout | only objects with a return | every annotation |
|---|---|---:|---:|
| v1.0-mini, 249 | signed | 0.373 | 0.799 |
| v1.0-mini, 249 | as labelled | 0.442 | 0.940 |
| official val, 10,087 | signed | 0.370 | 0.725 |
| official val, 10,087 | as labelled | 0.420 | 0.827 |

Sources: `results/symbolic_diagnostic_as_labelled/`,
`results/label_audit/val_symbolic_diagnostic__*.json`. The table in the report
(125 questions, 0.336 against 0.824) was produced by an earlier version of the
rule parser and cannot be regenerated exactly from this checkout; the rows
above are the current code.

## What this means for a benchmark user

* A model can score well on the directional questions without any notion of
  left and right, and a model that has one is penalised: a correct count of the
  cars to the front right is marked wrong whenever there is one.
* A LiDAR-only system is asked to count objects its sensor did not return, and
  objects of classes its detector does not have.
* Results reported on nuScenes-QA are comparable with each other, since every
  system faces the same labels. What changes is what a number means: it
  measures agreement with these conventions, not left/right spatial reasoning.

## Limits

* The rules are inferred from answers. Reading the generation code could
  confirm or refine them.
* The validation-set check with geometry uses the rule parser unverified, and
  its programs are right less often than the hand-verified ones: 91.9% is a
  floor on label agreement, not a measurement of label noise.
* Some questions still disagree under all three conventions: 4 of 200 on the
  mini evaluation split and 36 of 1,460 on the validation drives. On the
  validation drives these are 19 directional questions, mostly counts off by
  one or two, 8 questions whose reference object is not unique or not found
  ("the not standing pedestrian" when there are several; the executor
  refuses, the dataset answers), 7 status questions and 2 plain counts. They were not investigated
  further.
