# Reproduction and experiment status

Run all commands from the repository root. The saved result audit is offline;
model inference requires a suitable GPU and downloads model weights. The
requirement files give dependency ranges, **not a historical lockfile**.
The original runs did not preserve a complete package/hardware snapshot, so
bit-for-bit reproduction of those runs is not claimed.

## Offline verification

```bash
python -m pip install -r requirements-audit.txt
python -m spatial_reasoning.audit_results
```

The eight complete comparisons contain 1,580 saved rows. The audit checks
membership in the frozen manifests, duplicate IDs, scene separation,
reference answers, correctness flags, every stored summary field and hashes.
To write a fresh receipt:

```bash
python -m spatial_reasoning.audit_results --out outputs/artifact_audit.json
```

The audit does not contact a model or verify the identity of a hosted provider.
Historical aliases and missing immutable model revisions remain explicit.

## CPU tests

```bash
python -m pip install -r requirements-test.txt
python -m pytest tests/ -q
```

Torch is needed by the older learned-baseline tests even though the main
artifact audit and most deterministic checks do not need it. GPU-only or
optional integration checks may skip according to their recorded conditions.

The local verification on 1 November 2026 used Python 3.11.0, with NumPy and
pytest available but no PyTorch. The complete suite could not be collected in
that environment. The following subset passed 203 tests, with three optional
skips and one explicit deselection:

```bash
python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_config_propagation.py --ignore=tests/test_silent_acceptance.py -k 'not test_count_routing_reads_the_question_not_the_label'
```

The excluded modules check the original Torch-based pipeline; the deselected
test checks its learned counting route. Install `requirements-test.txt` before
claiming a full-suite pass. The archived-result audit passed independently;
no GPU inference was rerun during this verification.

## New GPU runs on Colab

Use a separate output directory for each configuration. Never overwrite the
archived `spatial_reasoning/results/` files with an incomplete rerun. Mount
Drive through the authorized Colab session and put outputs and model caches
on durable storage if the runtime may be interrupted.

Install the inference requirements in the runtime:

```bash
python -m pip install -r requirements-inference.txt
```

Download **metadata only**, without the large raw sensor dataset:

```bash
mkdir -p data/data/nuscenes
curl -fL https://www.nuscenes.org/data/v1.0-mini.tgz -o /tmp/nuscenes-mini.tgz
tar -xzf /tmp/nuscenes-mini.tgz -C data/data/nuscenes v1.0-mini/
```

Check the executor first; this command needs no LLM:

```bash
python -m spatial_reasoning.verify_executor --manifest spatial_reasoning/manifests/first_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split dev
```

Then reproduce the core comparison (T4 is sufficient for this 1.5B setup):

```bash
python -m spatial_reasoning.compare_llm_modes --manifest spatial_reasoning/manifests/first_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split eval --model Qwen/Qwen2.5-1.5B-Instruct --precision fp16 --max-new-tokens 256 --batch-size 8 --out outputs/reproduce_1p5b
```

Before publishing a new run, preserve `pip freeze`, GPU model/driver,
repository commit, exact command, both row and summary files, and caches.
A summary existing is not sufficient: require the complete expected IDs and
parse the saved rows. The archived audit checks the published configurations;
new extensions must be inspected before being added to that inventory.

## The two historical runs that did not become results

The last conversation on 22 October 2026 mentioned these running jobs. No
complete row/summary pair for either was found in this checkout during the
recovery. They are **optional extensions**, not prerequisites for reproducing
the already complete core experiment. Neither contributes a number to the
report. A past runtime name is not evidence that a job is still running.

| Experiment | Last known configuration | Current evidence |
|---|---|---|
| Hosted Qwen3-14B | Narrow eval, output cap raised to 4,096 after truncation | No completed archive recovered |
| Qwen2.5-32B, wide scope | A100, int4, three modes on 280 eval questions | No completed archive recovered |

To rerun the wide extension on a GPU with enough memory (the historical
configuration used an A100):

```bash
python -m spatial_reasoning.compare_llm_modes --manifest spatial_reasoning/manifests/wide_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split eval --model Qwen/Qwen2.5-32B-Instruct --precision int4 --device-map auto --batch-size 1 --max-new-tokens 384 --functional --out outputs/wide_32b
```

For the hosted extension, first verify that the recorded provider model ID is
still offered. Supply `AI_GATEWAY_API_KEY` through the environment, never a
command-line literal, notebook output or committed file:

```bash
python -m spatial_reasoning.compare_llm_modes --manifest spatial_reasoning/manifests/first_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split eval --backend gateway --model alibaba/qwen-3-14b --max-new-tokens 4096 --out outputs/gw_qwen3_14b_4096
```

Changing a token cap after looking at evaluation failures makes this an
exploratory extension. A confirmatory follow-up needs a development-selected
budget and new held-out questions. If an endpoint returns empty or truncated
content, preserve the failures; do not substitute accuracy-on-answered for
accuracy on every requested question.

## Build the report

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error rapport_npm3d.tex
pdflatex -interaction=nonstopmode -halt-on-error rapport_npm3d.tex
```

Review the compiled PDF, including tables and references. The source and PDF
are both deliverables. Generated LaTeX auxiliaries are ignored.

The current PDF was rebuilt from the current source on 1 November 2026 with
MiKTeX pdfTeX 1.40.29. All 14 pages were rendered for visual review. The final
log contains no undefined references or overfull boxes.
