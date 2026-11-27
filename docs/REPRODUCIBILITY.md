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

## November 2026: the as-labelled conventions

`docs/LABEL_AUDIT.md` says what was found; this section says how to rerun it.
Nothing below modifies `spatial_reasoning/results/<archived run>/`.

### Data

```bash
mkdir -p data/data/nuscenes data/data/nuscenes_qa
# v1.0-mini metadata: as above.
# v1.0-trainval metadata, 460 MB, no sensor data:
curl -fL https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz -o /tmp/trainval_meta.tgz
tar -xzf /tmp/trainval_meta.tgz -C data/data/nuscenes v1.0-trainval/
# Official nuScenes-QA questions (the Drive folder linked from github.com/qiantianwen/NuScenes-QA):
python -m pip install gdown
python -m gdown --folder https://drive.google.com/drive/folders/1jIkICT23wZWZYPrWCa0x-ubjpClSzOuU -O /tmp/nuscenes_qa
cp /tmp/nuscenes_qa/questions/NuScenes_{train,val}_questions.json data/data/nuscenes_qa/
```

### CPU

```bash
# The labels themselves: laterality without geometry, and what a "thing" is.
python -m spatial_reasoning.audit_labels --questions data/data/nuscenes_qa/NuScenes_train_questions.json data/data/nuscenes_qa/NuScenes_val_questions.json --metadata-root data/data/nuscenes/v1.0-trainval --out outputs/label_audit.json

# The executor under each layout, on the hand-verified reference programs.
python -m spatial_reasoning.verify_executor --manifest spatial_reasoning/manifests/first_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split dev --layout nuscenes_qa --out outputs/ev_signed
python -m spatial_reasoning.verify_executor --manifest spatial_reasoning/manifests/first_comparison.json --metadata-root data/data/nuscenes/v1.0-mini --split dev --layout nuscenes_qa_as_labelled --out outputs/ev_as_labelled

# Every archived program re-executed under the as-labelled layout.
python -m spatial_reasoning.rescore_programs --metadata-root data/data/nuscenes/v1.0-mini --out outputs/rescored_as_labelled
python -m spatial_reasoning.analyse_rescored --rescored outputs/rescored_as_labelled

# The held-out manifest over the validation drives and its metadata slice.
python -m spatial_reasoning.build_val_manifest --metadata-root data/data/nuscenes/v1.0-trainval --mini-root data/data/nuscenes/v1.0-mini --questions data/data/nuscenes_qa/NuScenes_val_questions.json --out outputs/val_drives.json
python -m spatial_reasoning.verify_executor --manifest spatial_reasoning/manifests/val_drives.json --metadata-root data/data/nuscenes/v1.0-trainval-val-drives --split eval --layout nuscenes_qa_as_labelled --things all --out outputs/ev_val

# Tables and drive-level intervals from the saved rows (no data needed).
python -m spatial_reasoning.summarise_runs
python -m spatial_reasoning.drive_bootstrap spatial_reasoning/results/runs_2026_11/val_new_7b/rows_eval.jsonl --against spatial_reasoning/results/runs_2026_11/val_old_7b/rows_eval.jsonl
```

`rescore_programs` first replays each run under the layout it was archived
with and stops if anything differs from the archive.

### GPU, on Modal

`tools/modal_runs.py` builds an image with the repository and the metadata
found under `data/data/nuscenes/`, runs one entry point of the package, and
writes rows, summary, log and `provenance.json` (commit, exact command, GPU
and driver, `pip freeze`) to the volume `lidar-llm-outputs`. A name that
already holds a summary is refused.

```bash
python -m pip install "modal[api-proxy-support]"     # the extra only matters behind an HTTPS proxy
M="--manifest spatial_reasoning/manifests/first_comparison.json --split eval --max-new-tokens 256 --batch-size 8"

# direct-2: the convention as labelled, bare answer.
modal run tools/modal_runs.py --name d2_bare_1p5b_eval --gpu T4 --args "$M --model Qwen/Qwen2.5-1.5B-Instruct --precision fp16 --layout nuscenes_qa_as_labelled"
# direct-2r: working asked for, final 'Answer:' line read, budget chosen on dev.
modal run tools/modal_runs.py --name d2r_1p5b_eval --gpu T4 --args "$M --model Qwen/Qwen2.5-1.5B-Instruct --precision fp16 --layout nuscenes_qa_as_labelled --direct-style reasoned --direct-max-new-tokens 2048"
# Programs decoded under a JSON Schema (a separate condition).
modal run tools/modal_runs.py --name c_1p5b_eval --gpu T4 --args "$M --model Qwen/Qwen2.5-1.5B-Instruct --precision fp16 --layout nuscenes_qa_as_labelled --constrained"
# The validation drives, as labelled.
modal run tools/modal_runs.py --name val_new_7b --gpu A10G --metadata val-drives --args "--manifest spatial_reasoning/manifests/val_drives.json --split eval --max-new-tokens 256 --batch-size 8 --model Qwen/Qwen2.5-7B-Instruct --precision fp16 --layout nuscenes_qa_as_labelled --things all"

tools/fetch_modal_runs.sh d2_bare_1p5b_eval d2r_1p5b_eval c_1p5b_eval val_new_7b
```

The 7B model ran on an A10G, the 32B (int4, `--device-map auto --batch-size 4`)
on an L40S. The package versions of each run are in its `provenance.json`;
the runs of this section used torch 2.15.0 and transformers 5.24.1.

What was chosen on the development split, and only there: the wording of the
reasoned prompt (three versions; the first two left the 1.5B model answering
in four tokens), its reader's acceptance of a bare one-line answer, and the
2,048-token budget (at 1,536 the 7B model was cut off on 4 to 5 of 100
development questions).

### Verification of this section

On 27 November 2026, Python 3.13, NumPy 2.5, torch 2.15 (CPU):
`python -m pytest tests/ -q` passed 283 tests with none skipped;
`python -m spatial_reasoning.audit_results` verified the 8 archived
comparisons and their 8 rescored counterparts; `npx vitest run` in `web/`
passed 14 tests.

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

The current PDF was rebuilt from the current source on 27 November 2026 with
pdfTeX from TeX Live (Debian). It has 14 pages; the final log contains no
undefined references and no overfull boxes.
