# Results: spatial reasoning at fixed perception

Mean +/- population std over 3 seeds. 10 synthetic scenes, 20 samples each,
8 questions per sample; detector model: drop 12%, duplicate 5%, jitter 0.6 m.

## Setting: balanced (3 seeds, scene-level splits)

- questions: 1600 over 200 samples (train 960 / val 320 / test 320)
- answers meaning "nothing there": 0.371
- majority-answer accuracy: 0.296
- parser coverage: 1.000

| system | accuracy | count MAE | ms/item | params |
|---|---|---|---|---|
| question_only | 0.656 ± 0.012 | 0.722 ± 0.093 | 0.68 | 35863 |
| deepsets | 0.655 ± 0.020 | 0.654 ± 0.088 | 0.88 | 120855 |
| counter | 0.644 ± 0.018 | 0.534 ± 0.035 | 1.09 | 129176 |
| symbolic_abstain | 0.916 ± 0.027 | 0.132 ± 0.053 | 0.23 | 0 |
| symbolic_fallback | 0.916 ± 0.027 | 0.132 ± 0.053 | 0.23 | 0 |

| system | pair accuracy | counterfactual acc. | answer changed |
|---|---|---|---|
| question_only | 0.000 | 0.433 | 0.531 |
| deepsets | 0.022 | 0.443 | 0.558 |
| counter | 0.076 | 0.406 | 0.521 |
| symbolic_fallback | 0.643 | 1.000 | 1.000 |

| setting | question_only | deepsets | counter | symbolic_fallback |
|---|---|---|---|---|
| drop=0 dup=0 jitter=0m | 0.656 | 0.655 | 0.644 | 0.916 |
| drop=0.1 dup=0 jitter=0m | 0.656 | 0.652 | 0.641 | 0.867 |
| drop=0.2 dup=0 jitter=0m | 0.656 | 0.646 | 0.636 | 0.811 |
| drop=0.3 dup=0 jitter=0m | 0.656 | 0.648 | 0.620 | 0.750 |
| drop=0 dup=0.1 jitter=0m | 0.656 | 0.661 | 0.639 | 0.878 |
| drop=0 dup=0.2 jitter=0m | 0.656 | 0.654 | 0.617 | 0.832 |
| drop=0 dup=0 jitter=0.5m | 0.656 | 0.655 | 0.644 | 0.915 |
| drop=0 dup=0 jitter=1.5m | 0.656 | 0.656 | 0.646 | 0.907 |
| drop=0 dup=0 jitter=3m | 0.656 | 0.657 | 0.646 | 0.899 |

| error kind | mean count | fraction |
|---|---|---|
| correct | 293.0 | 0.916 |
| declined | 0.0 | 0.000 |
| perception | 27.0 | 0.084 |
| comprehension | 0.0 | 0.000 |
| geometry | 0.0 | 0.000 |
| unattributed | 0.0 | 0.000 |

## Setting: degenerate (3 seeds, scene-level splits)

- questions: 1600 over 200 samples (train 960 / val 320 / test 320)
- answers meaning "nothing there": 0.750
- majority-answer accuracy: 0.412
- parser coverage: 1.000

| system | accuracy | count MAE | ms/item | params |
|---|---|---|---|---|
| question_only | 0.757 ± 0.015 | 0.502 ± 0.036 | 0.67 | 35863 |
| deepsets | 0.766 ± 0.032 | 0.443 ± 0.098 | 0.86 | 120855 |
| counter | 0.775 ± 0.007 | 0.310 ± 0.027 | 1.12 | 129176 |
| symbolic_abstain | 0.950 ± 0.003 | 0.075 ± 0.012 | 0.25 | 0 |
| symbolic_fallback | 0.950 ± 0.003 | 0.075 ± 0.012 | 0.24 | 0 |

| system | pair accuracy | counterfactual acc. | answer changed |
|---|---|---|---|
| question_only | 0.000 | 0.124 | 0.200 |
| deepsets | 0.045 | 0.192 | 0.263 |
| counter | 0.063 | 0.182 | 0.253 |
| symbolic_fallback | 0.616 | 1.000 | 1.000 |

| setting | question_only | deepsets | counter | symbolic_fallback |
|---|---|---|---|---|
| drop=0 dup=0 jitter=0m | 0.757 | 0.766 | 0.775 | 0.950 |
| drop=0.1 dup=0 jitter=0m | 0.757 | 0.762 | 0.770 | 0.933 |
| drop=0.2 dup=0 jitter=0m | 0.757 | 0.766 | 0.773 | 0.909 |
| drop=0.3 dup=0 jitter=0m | 0.757 | 0.771 | 0.770 | 0.892 |
| drop=0 dup=0.1 jitter=0m | 0.757 | 0.759 | 0.772 | 0.945 |
| drop=0 dup=0.2 jitter=0m | 0.757 | 0.761 | 0.769 | 0.923 |
| drop=0 dup=0 jitter=0.5m | 0.757 | 0.766 | 0.774 | 0.949 |
| drop=0 dup=0 jitter=1.5m | 0.757 | 0.766 | 0.772 | 0.943 |
| drop=0 dup=0 jitter=3m | 0.757 | 0.766 | 0.771 | 0.934 |

| error kind | mean count | fraction |
|---|---|---|
| correct | 304.0 | 0.950 |
| declined | 0.0 | 0.000 |
| perception | 16.0 | 0.050 |
| comprehension | 0.0 | 0.000 |
| geometry | 0.0 | 0.000 |
| unattributed | 0.0 | 0.000 |
