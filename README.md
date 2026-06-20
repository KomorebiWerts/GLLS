# KRagAD

KRagAD is a Python research codebase for industrial anomaly question answering on MMAD-style datasets. This repository has been cleaned into a standard `src` layout and intentionally keeps one primary batch pipeline plus one visualization UI.

## Repository layout

The active package lives in `src/kragad/`.

- `kragad.cli.run` is the main batch entrypoint.
- `kragad.cli.visualize` is the Gradio inspection UI.
- `kragad.data` contains dataset loaders for annotated MMAD-style samples.
- `kragad.rag` contains graph construction and retrieval logic for text-plus-image inspection knowledge.
- `kragad.models` contains the anomaly localizer, logic processor, and MCTS-based inspection agent.
- `kragad.seg` contains the SAM3 wrapper and segmentation helpers.

Large resources are expected outside git under a local data root.

## Environment

By default, helper scripts use `~/data/kragad` as the external data root. Override any path with environment variables before activation:

- `KRAGAD_DATA_ROOT`
- `KRAGAD_DATASET_ROOT`
- `KRAGAD_DATABASE_ROOT`
- `KRAGAD_SAM3_PATH`
- `KRAGAD_ADAPTCLIP_ROOT`
- `KRAGAD_VENV`

Activate the environment with:

```bash
source scripts/dev/activate_kragad.sh
```

That script exports the expected `KRAGAD_*` variables, activates `KRAGAD_VENV` when present, and sets `PYTHONPATH` for the package plus the external SAM3 checkout.

## Install dependencies

```bash
pip install -r requirements-kragad.txt
```

Torch is managed separately on the target machine; see `requirements-kragad.txt` for the expected CUDA wheel notes.

## Run the main pipeline

```bash
python -m kragad.cli.run --help
bash scripts/run/run_main.sh --help
```

Typical options are defined directly in `src/kragad/cli/run.py`, including dataset selection, subclass filtering, GPU selection, output directory, graph cache root, SAM3 checkpoint path, and ABounD checkpoint paths.

## Run the visualization UI

```bash
python -m kragad.cli.visualize
bash scripts/run/run_visualize.sh
```

The UI reuses the same core pipeline components and supports interactive inspection, manual reruns, and export of ZIP or DOCX run bundles.

## Tests and checks

Run the test suite with:

```bash
python -m pytest tests
```

Run a single test with:

```bash
python -m pytest tests/<file>.py::<test_name>
```

Run a syntax/import sanity check with:

```bash
python -m compileall src/kragad
```

## Data and graph expectations

The batch pipeline expects MMAD-organized QA data under the shared data root, especially:

- `DS-MVTec/*/QA.json`
- `VisA/*/QA.json`
- graph caches under `.../datasets/KRagAD/databases/graph_index`

Graph building logic lives in `python -m kragad.rag.build_graph` and uses text knowledge plus reference images to produce `{category}_graph.pkl` files consumed by the RAG agent.

## Notes on current scope

This repository no longer keeps the old ablation, timing, heatmap-sensitivity, and baseline top-level entrypoints as active workflows. If you are extending the method, prefer modifying the retained `kragad` package modules and the two supported CLIs instead of reintroducing parallel script trees.

## Upgrade notes

Environment-specific paths and private machine notes should stay outside the repository. When adding new entrypoints, prefer the shared `KRAGAD_*` variables and `src/kragad/paths.py` defaults instead of hard-coded absolute paths.
