# Literature, limitation and scope

The project is a small empirical study, not a new neuro-symbolic architecture.
The primary sources below define what the experiment can reasonably claim.

## What limitation is addressed?

[LiDAR-LLM (Yang et al., arXiv:2312.14074)](https://arxiv.org/html/2312.14074v1)
aligns LiDAR features with a language model through staged training. Its
conclusion names lightweight operation as future work; Appendix B documents
unsupported semantic descriptions, including color. The revised project
tests a lightweight and inspectable *reasoning stage*: use a frozen small
model to translate questions into computations over explicit 3D objects.
This addresses reliable factual answering given available facts. It cannot
recover unobserved color or establish raw-LiDAR perception quality, since the
present input is ground-truth boxes and human-annotated statuses.

## What is already established?

[CLEVR (Johnson et al.)](https://arxiv.org/abs/1612.06890) and
[Neural-Symbolic VQA (Yi et al.)](https://arxiv.org/abs/1810.02338) already
use functional programs over structured scenes. Program execution is not
claimed as novel here. The contribution is an auditable same-checkpoint
comparison on driving questions with scene-disjoint splits and failure traces.

[Real-3DQA (Ma et al.)](https://arxiv.org/abs/2603.23523) demonstrates that
text-only shortcuts can undermine claims of 3D understanding. This motivates
the scene-dependence probe and the archived question-only controls. It does
not make our three-drive sample representative of broader driving conditions.

[KLDrive (Tian et al.)](https://arxiv.org/abs/2603.21029) combines a scene
knowledge graph with constrained LLM reasoning. Its perception and evaluation
protocol differ; its numbers are context for the idea, not baselines in our
tables.

## Two corrections to the earlier interpretation

1. **Geometry is a specification, not a discovered result.**
   [NuScenes-QA, Equation 2](https://arxiv.org/html/2305.14836v2) explicitly
   defines six 60-degree angular sectors. Our earlier eight-octant version
   was wrong for this benchmark. Testing candidate layouts against archived
   labels is useful as a diagnostic, but does not rediscover an unknown
   benchmark rule or provide an independent held-out validation.
2. **The hosted-model table is not a causal reasoning ablation.**
   The [official Qwen3-Next-80B-A3B model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
   specifies 80B total and 3B activated parameters and a non-thinking mode.
   Comparing its nominal size with a different dense/provider model is not
   an equal-compute comparison. The archive's `qwen3.8-27b` alias reaches near
   parity between modes with a larger observed output-token count, but the
   endpoint changes several factors at once and did not honor the requested
   output cap. We retain the measured row without claiming that it isolates
   the effect of reasoning, proves a universal scaling law, or measures a
   nine-fold cost saving.

## Defensible conclusion and useful next experiment

For the saved small-model configurations and selected questions, explicit
execution improves exact answers and scene dependence. A broader typed
language at 1.5B reduces coverage enough to lose overall. One hosted endpoint
nearly closes the gap. These findings bound the scope of the method.

The next experiment should vary one factor within a fixed checkpoint: for
example, direct answering with/without an explicit reasoning budget, or
typed generation with/without constrained decoding. Freeze the prompts on
development data, use more unseen drives, and report accuracy, abstention,
latency and actual token usage. This is a future experiment, not an outcome
claimed by the archived runs.
