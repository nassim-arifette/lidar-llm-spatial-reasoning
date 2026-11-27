#!/bin/bash
# Copy finished runs from the Modal volume into the results folder.
# A run already present is left alone: results are never overwritten.
#   tools/fetch_modal_runs.sh name1 name2 ...
set -u
cd "$(dirname "$0")/.."
DEST=spatial_reasoning/results/runs_2026_11
mkdir -p "$DEST" outputs/modal
for name in "$@"; do
  if ls "$DEST/$name"/summary_*.json >/dev/null 2>&1; then echo "kept    $name"; continue; fi
  rm -rf "outputs/modal/$name"
  python -m modal volume get lidar-llm-outputs "outputs/$name" outputs/modal >/dev/null 2>&1
  if ls "outputs/modal/$name"/summary_*.json >/dev/null 2>&1; then
    mkdir -p "$DEST/$name"
    cp "outputs/modal/$name"/{rows_*.jsonl,summary_*.json,provenance.json,log.txt} "$DEST/$name/"
    echo "fetched $name"
  else
    echo "missing $name (no summary yet)"
  fi
done
