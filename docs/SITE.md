# Research website

A single English page in an interactive-article style. Sources live in `web/` (Vite + TypeScript, D3, Three.js); `npm run build` writes `site/`, which is not tracked.

The figures re-run the archived programs with `web/src/executor.ts`, a TypeScript port of `spatial_reasoning/dsl.py` and the nuScenes-QA sectors of `regions.py`. `web/test/executor.test.ts` checks, for all six exported comparisons, that every archived valid program gives the same answer, the same selected annotations in the same order and the same program text.

## Regenerating the data

The export needs the nuScenes v1.0-mini metadata in `data/data/nuscenes/v1.0-mini` (see `docs/REPRODUCIBILITY.md`). It rebuilds every keyframe with the study's adapter and refuses to write unless the Python executor reproduces each archived answer and selection.

```bash
python tools/build_web_data.py        # -> web/public/data/study.json
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
