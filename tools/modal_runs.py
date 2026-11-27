"""Run the comparison on Modal, with the provenance REPRODUCIBILITY.md asks for.

    modal run --detach tools/modal_runs.py --name direct2_1p5b_eval --gpu T4 \\
        --args "--manifest spatial_reasoning/manifests/first_comparison.json \\
                --split eval --model Qwen/Qwen2.5-1.5B-Instruct --precision fp16 \\
                --max-new-tokens 256 --batch-size 8 --layout nuscenes_qa_as_labelled"

``--metadata-root`` and ``--out`` are filled in. Outputs land in the
``lidar-llm-outputs`` volume under ``<name>/``, next to ``provenance.json``
(pip freeze, GPU and driver, commit, exact command). Fetch them with

    modal volume get lidar-llm-outputs <name> outputs/

A name that already holds a summary is refused: a rerun is a new name, never an
overwrite. ``--module`` selects another entry point of the package.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time

import modal

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA = {
    "v1.0-mini": "data/data/nuscenes/v1.0-mini",
    # The slice build_val_manifest writes: only what val_drives.json needs.
    "val-drives": "data/data/nuscenes/v1.0-trainval-val-drives",
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return ""


def _ignore(path) -> bool:
    parts = str(path).replace("\\", "/").split("/")
    return (".git" in parts or "node_modules" in parts or "__pycache__" in parts
            or parts[:1] in (["web"], ["report"], ["outputs"], ["data"]))


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4,<3", "numpy>=1.26,<3", "transformers>=4.51,<6",
        "accelerate>=1,<2", "bitsandbytes>=0.45,<1", "lm-format-enforcer>=0.10",
        "sentencepiece", "protobuf")
    .env({"HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(REPO, "/repo", ignore=_ignore)
)
for _version, _path in METADATA.items():
    if os.path.isdir(os.path.join(REPO, _path)):
        image = image.add_local_dir(os.path.join(REPO, _path), f"/repo/{_path}")

app = modal.App("lidar-llm-spatial-reasoning", image=image)
cache = modal.Volume.from_name("lidar-llm-hf-cache", create_if_missing=True)
outputs = modal.Volume.from_name("lidar-llm-outputs", create_if_missing=True)


@app.function(gpu="T4", timeout=6 * 3600,
              volumes={"/cache": cache, "/vol": outputs})
def job(name: str, module: str, argv: list, commit: str, dirty: bool,
        metadata: str = "v1.0-mini") -> dict:
    out = f"/vol/outputs/{name}"
    if any(f.startswith("summary_") for f in
           (os.listdir(out) if os.path.isdir(out) else [])):
        raise RuntimeError(f"{out} already holds a summary; use a new name")
    os.makedirs(out, exist_ok=True)

    command = ["python", "-m", f"spatial_reasoning.{module}", *argv,
               "--metadata-root", f"/repo/{METADATA[metadata]}", "--out", out]

    def sh(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    provenance = {
        "name": name,
        "commit": commit,
        "worktree_dirty": dirty,
        "command": " ".join(shlex.quote(c) for c in command),
        "gpu": sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                  "--format=csv,noheader"),
        "python": sh("python", "--version"),
        "pip_freeze": sh("python", "-m", "pip", "freeze").splitlines(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(f"{out}/provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)

    with open(f"{out}/log.txt", "w") as log:
        proc = subprocess.Popen(command, cwd="/repo", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:           # type: ignore[union-attr]
            print(line, end="")
            log.write(line)
        code = proc.wait()

    provenance["exit_code"] = code
    provenance["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"{out}/provenance.json", "w") as fh:
        json.dump(provenance, fh, indent=2)
    outputs.commit()
    cache.commit()
    if code:
        raise RuntimeError(f"{name}: exit code {code}")
    return {"name": name, "exit_code": code}


@app.local_entrypoint()
def main(name: str, args: str, gpu: str = "T4",
         module: str = "compare_llm_modes", metadata: str = "v1.0-mini"):
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    if dirty:
        print("WARNING: uncommitted changes; the recorded commit is not the code that runs")
    runner = job.with_options(gpu=gpu)
    print(runner.remote(name, module, shlex.split(args), commit, dirty, metadata))
