# Results on real nuScenes-QA (mini)

3068 questions from nuScenes-QA over 276 samples of nuScenes v1.0-mini,
ground-truth boxes from the official annotations (14129 objects over 404 samples,
4278 annotations with zero LiDAR returns excluded), scene-level splits over 10 scenes,
mean over seeds 0-2, test n=981.

## Perception: oracle

- majority-answer accuracy: 0.246
- **parser coverage: 0.313**

| system | accuracy | answered | acc. when answered | count MAE |
|---|---|---|---|---|
| question_only | 0.443 | 1.000 | 0.443 | 3.78 |
| deepsets | 0.455 | 1.000 | 0.455 | 3.72 |
| counter | 0.456 | 1.000 | 0.456 | 3.52 |
| symbolic_abstain | 0.131 | 0.313 | 0.418 | 15.21 |
| symbolic_fallback | 0.277 | 1.000 | 0.277 | 12.92 |

| system | contrast pairs | counterfactual |
|---|---|---|
| question_only | 0.000 | 0.498 |
| deepsets | 0.069 | 0.460 |
| counter | 0.016 | 0.406 |
| symbolic_fallback | 0.005 | 1.000 |

| error kind | count | fraction |
|---|---|---|
| correct | 117.7 | 0.131 |
| declined | 612.3 | 0.687 |
| perception | 0.0 | 0.000 |
| comprehension | 0.0 | 0.000 |
| geometry | 0.0 | 0.000 |
| unattributed | 165.0 | 0.183 |

## Perception: detector

- majority-answer accuracy: 0.246
- **parser coverage: 0.313**

| system | accuracy | answered | acc. when answered | count MAE |
|---|---|---|---|---|
| question_only | 0.441 | 1.000 | 0.441 | 3.56 |
| deepsets | 0.455 | 1.000 | 0.455 | 3.57 |
| counter | 0.454 | 1.000 | 0.454 | 3.68 |
| symbolic_abstain | 0.131 | 0.313 | 0.418 | 13.83 |
| symbolic_fallback | 0.277 | 1.000 | 0.277 | 11.81 |

| system | contrast pairs | counterfactual |
|---|---|---|
| question_only | 0.000 | 0.430 |
| deepsets | 0.005 | 0.458 |
| counter | 0.033 | 0.391 |
| symbolic_fallback | 0.002 | 1.000 |

| error kind | count | fraction |
|---|---|---|
| correct | 117.7 | 0.131 |
| declined | 612.3 | 0.687 |
| perception | 5.3 | 0.006 |
| comprehension | 0.0 | 0.000 |
| geometry | 0.0 | 0.000 |
| unattributed | 159.7 | 0.177 |

## Why the parser declines (share of test questions)

- 0.574 -- no supported operation (expected a counting, existence or comparison question)
- 0.024 -- question is anchored on another object (car); reference resolution is out of scope
- 0.019 -- question is anchored on another object (truck); reference resolution is out of scope
- 0.013 -- question is anchored on another object (construction_vehicle); reference resolution is out of scope
- 0.013 -- question is anchored on another object (bicycle); reference resolution is out of scope
- 0.011 -- no known category mentioned
- 0.008 -- question is anchored on another object (object); reference resolution is out of scope
- 0.007 -- question is anchored on another object (pedestrian); reference resolution is out of scope
- 0.003 -- question is anchored on another object (motorcycle); reference resolution is out of scope

## Declined examples

- Does the thing to the back right of the construction vehicle have the same status as the construction vehicle?
- There is a moving thing to the back of the construction vehicle; what is it?
- Are there any other pedestrians of the same status as the thing that is to the back of the not standing pedestrian?
- The construction vehicle is in what status?
- Do the truck to the back of the bus and the car that is to the back right of the parked construction vehicle have the same status?
