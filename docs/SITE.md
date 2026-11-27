# Research website

A single English page in an interactive-article style. Sources live in `web/` (Vite + TypeScript, D3, Three.js); `npm run build` writes `site/`, which is not tracked.

The figures re-run the archived programs with `web/src/executor.ts`, a TypeScript port of `spatial_reasoning/dsl.py` and of both nuScenes-QA layouts of `regions.py` (`nuscenes_qa`, the signed sectors as published, and `nuscenes_qa_as_labelled`, the rule the released answers follow). `web/test/executor.test.ts` checks, for all six exported comparisons, that every archived valid program gives the same answer, the same selected annotations in the same order and the same program text under the published layout, and the answer of the Python re-execution under the as-labelled one.

The page reads two data files:

- `web/public/data/study.json` (`tools/build_web_data.py`): the six archived runs question by question, with each program's outcome under both layouts and, where a November 2026 run exists, the `direct-2` answer; plus the annotation geometry.
- `web/public/data/audit.json` (`tools/build_web_audit.py`): the label audit, the re-executed archive and the November 2026 runs, recomputed from the saved rows. A run enters only when its summary exists and it holds the expected number of questions, so a table row or a chart mark appears when its data does. Its `stats` entry holds every number quoted in the prose, already formatted.

Numbers in the text are `data-stat` elements. `build_web_audit.py` writes their text into `web/index.html`, `main.ts` writes it again in the browser, and `tools/check_site.py` fails if the built HTML and the built `audit.json` disagree. A passage marked `data-if` stays hidden until its number exists (`data-optional` on the number). To quote a new number, add a key to `stats` and an element with that key.

## Regenerating the data

The export needs the nuScenes v1.0-mini metadata in `data/data/nuscenes/v1.0-mini` (see `docs/REPRODUCIBILITY.md`). It rebuilds every keyframe with the study's adapter and refuses to write unless the Python executor reproduces each archived answer and selection, and each re-executed answer and selection of `results/rescored_as_labelled/`.

```bash
python tools/build_web_data.py        # -> web/public/data/study.json (needs the v1.0-mini metadata)
python tools/build_web_audit.py       # -> web/public/data/audit.json, numbers in web/index.html, web/public/files/*.md (saved results only)
cd web
npm ci
npm test
npm run build                         # -> ../site
cd ..
python tools/check_site.py
```

The LiDAR layer needs the keyframe sweeps in `data/data/nuscenes/samples/LIDAR_TOP/` (the public `https://www.nuscenes.org/data/v1.0-mini.tgz` contains them). `python tools/build_lidar_points.py` moves each sweep into the ego frame of its boxes, keeps 16,000 points within 70 m and writes `web/public/lidar/<sample_token>.bin` (12 MB for the 102 keyframes, CC BY-NC-SA 4.0 like nuScenes).

`study.json` and the sweeps are committed, so the GitHub Pages build does not need the nuScenes download.

## Local preview

The page loads its data with `fetch`, so serve it over HTTP: `cd web && npm run dev`, or `python -m http.server 8000 --directory site` after a build.

## Publishing

`.github/workflows/pages.yml` installs, tests, builds, checks and deploys `site/`. In the repository settings choose **Pages → Source → GitHub Actions**.
