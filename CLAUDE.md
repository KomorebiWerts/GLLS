# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

GLLS is a Python research codebase for agentic industrial anomaly QA on MMAD-style benchmarks. This cleaned repository keeps one primary batch pipeline and one visualization UI, rather than preserving all historical ablation entrypoints.

## Current working layout

The active Python package still lives under `src/kragad/` for import compatibility, with two entrypoints:

- `python -m kragad.cli.run` for the main multi-GPU batch pipeline.
- `python -m kragad.cli.visualize` for the Gradio-based visualization and inspection UI.

Shell wrappers:

- `bash scripts/run/run_main.sh`
- `bash scripts/run/run_visualize.sh`
- `source scripts/dev/activate_glls.sh`

The repository root is intentionally lightweight. Large datasets, checkpoints, caches, external repos, and outputs are expected under a local data root, not committed into git.

## Environment and data paths

The default external data root is `~/data/GLLS`. On machines with an existing `~/data/kragad` layout, the activation script will keep using it when no GLLS data root exists, so no data migration is required. Override machine-specific paths with environment variables before activation:

- `GLLS_DATA_ROOT`
- `GLLS_DATASET_ROOT`
- `GLLS_QA_ROOT`
- `GLLS_DATABASE_ROOT`
- `GLLS_SAM3_PATH`
- `GLLS_ADAPTCLIP_ROOT`
- `GLLS_VENV`

`scripts/dev/activate_glls.sh` exports the standard `GLLS_*` environment variables, also sets legacy `KRAGAD_*` compatibility aliases, activates `GLLS_VENV` when present, and wires `PYTHONPATH` to the package source plus the external SAM3 checkout.

## Common commands

Activate the environment first:

```bash
source scripts/dev/activate_glls.sh
```

Install runtime dependencies into the active environment:

```bash
pip install -r requirements-glls.txt
```

Run the main pipeline:

```bash
python -m kragad.cli.run --help
bash scripts/run/run_main.sh --help
```

Run the visualization UI:

```bash
python -m kragad.cli.visualize
bash scripts/run/run_visualize.sh
```

Run tests:

```bash
python -m pytest tests
python -m pytest tests/<file>.py::<test_name>
```

Run a quick syntax/import sanity check:

```bash
python -m compileall src/kragad
```

## High-level architecture

The main pipeline in `kragad.cli.run` orchestrates dataset loading, RAG retrieval, anomaly localization, SAM3-guided inspection, and final VLM verification across one or more GPUs. It is the only retained batch execution path.

Key subsystem boundaries:

- `kragad.data.dataset_loader` loads annotated MMAD-style question-answer samples for DS-MVTec and VisA.
- `kragad.rag.build_graph` and `kragad.rag.GraphRag` build and load region-aware graph knowledge bases from text knowledge plus reference images.
- `kragad.rag.agent.SimInspecAgent` turns graph knowledge into prompt-ready multimodal evidence blocks.
- `kragad.models.localizer.ABounD_Localizer` generates anomaly heatmaps and category-level thresholds using ABounD / AdaptCLIP checkpoints.
- `kragad.seg.sam3_engine.Sam3Engine` wraps SAM3 image prompting and mask composition.
- `kragad.models.logical.VisualLogicProcessor` adds category-specific structural checks, especially for cases where spatial relations matter.
- `kragad.models.mcts_sam.MCTSQuestionSample` is the core agentic inspection loop that combines global heatmaps, local crop exploration, logic reports, and VLM verification.
- `kragad.cli.visualize` exposes the same pipeline components interactively through Gradio, with exports for ZIP bundles and optional DOCX run reports.

When changing behavior, prefer improving these shared modules instead of adding new top-level scripts.

## Repository cleanup assumptions

The repository was intentionally reduced to the main path plus visualization. Historical ablation, timing, heatmap-sensitivity, and baseline entrypoints were deleted from the active tree. Future work should improve the retained path rather than restore every prior experiment script.

Some legacy helper scripts and notes still exist inside package subdirectories, especially under `kragad/seg` and `kragad/rag`, but they are not the primary development interface unless a task explicitly targets them.

## Upgrade preparation

Keep environment-specific setup notes outside git. When adding or changing entrypoints, use `src/kragad/paths.py` and the shared `GLLS_*` variables instead of hard-coded absolute paths.
