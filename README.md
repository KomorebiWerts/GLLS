# GLLS

Global Logic and Local Semantic inspection for industrial anomaly question answering.

GLLS is a method-focused research implementation for MMAD-style industrial visual
QA. The released code keeps one clean path:

- `glls.cli.run`: batch QA evaluation on DS-MVTec and VisA.
- `glls.cli.binary_ad`: binary anomaly detection on MPDD, DTD-Synthetic, and DAGM.
- `glls.cli.visualize`: one Gradio frontend for per-question online inspection.

The repository does not publish benchmark images, model checkpoints, experiment
outputs, tests, old ablations, or static result viewers. Users only need to
configure dataset/model paths and use the curated QA annotations in
`qa_collection/`.

## Method Overview

GLLS combines four components:

1. **Global inspection**: a VLM first reasons over the full target image and
   the question/options.
2. **Local evidence search**: AdaptCLIP produces anomaly heatmaps;
   MCTS expands and ranks local region proposals instead of passing every crop
   blindly to the VLM.
3. **SAM3 refinement**: SAM3 is used when the task and category benefit from
   structural/local segmentation. Text and box prompts are selected from
   conservative category profiles in `src/glls/seg/sam3_profiles.py`.
4. **PVLA/RAG knowledge**: source-backed graph knowledge is retrieved from
   normal references and text knowledge, then passed to the VLM as hierarchical
   visual/text evidence.

This keeps the code aligned with the paper idea: global logic, local semantic
evidence, graph-structured normal knowledge, and traceable local search.

## Why This Release Includes a Curated QA Collection

The original MMAD QA annotations are noisy enough to affect evaluation:

- Some questions have the wrong answer label.
- Some questions have multiple valid options.
- Some questions have no valid option.
- Some options are inconsistent with the image or the declared task type.
- DS-MVTec `pill` contains a known contamination problem where QA content was
  mixed with content from another dataset/category; the curated collection uses
  the corresponding corrected MVTec `pill` annotation.

`qa_collection/` fixes annotation issues only. It does not change any MMAD image.
The collection is still a curated research annotation set, not a claim that every
remaining QA row is perfect.

The activation script uses this bundled collection by default. To override it,
set:

```bash
export GLLS_QA_ROOT=/path/to/GLLS/qa_collection
```

Expected layout:

```text
qa_collection/
  DS-MVTec/<category>/QA.json
  VisA/<category>/QA.json
```

## Repository Layout

```text
GLLS/
  qa_collection/                 # curated MMAD QA annotations only
  scripts/
    dev/activate_glls.sh         # local environment and path setup
    run/run_main.sh              # DS-MVTec/VisA QA wrapper
    run/run_binary_ad.sh         # MPDD/DTD/DAGM wrapper
    run/run_visualize.sh         # Gradio frontend wrapper
    data/                        # binary AD metadata/offline PVLA preparation
  src/glls/
    cli/                         # run, binary_ad, visualize
    data/                        # MMAD QA loaders
    eval/                        # MPDD/DTD/DAGM binary AD pipeline
    models/                      # localizer, MCTS, logic, policies
    rag/                         # PVLA graph/RAG construction and retrieval
    seg/                         # SAM3 engine and prompt profiles
```

## Environment Setup

Create a Python environment at the default location used by the helper scripts:

```bash
mkdir -p ~/data/GLLS/envs
python3 -m venv ~/data/GLLS/envs/GLLS
source ~/data/GLLS/envs/GLLS/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-glls.txt
```

Torch is intentionally not pinned inside `requirements-glls.txt`; install the
CUDA build that matches your machine first.

Activate GLLS paths:

```bash
source scripts/dev/activate_glls.sh
```

Default external root:

```text
~/data/GLLS
```

The activation script sets:

```text
GLLS_DATA_ROOT
GLLS_DATASET_ROOT
GLLS_QA_ROOT
GLLS_DATABASE_ROOT
GLLS_GRAPH_CACHE_ROOT
GLLS_VLM_MODEL_PATH
GLLS_SAM3_PATH
GLLS_ADAPTCLIP_ROOT
GLLS_ABOUND_MODEL_PATH
GLLS_ABOUND_SAVE_PATH
GLLS_EMBEDDING_MODEL_PATH
GLLS_MPDD_ROOT
GLLS_DTD_ROOT
GLLS_DAGM_ROOT
GLLS_VENV
GLLS_PYTHON
```

Override any of them before activation if your paths differ.

## Dataset Layout

For MMAD-style QA:

```text
$GLLS_DATASET_ROOT/
  DS-MVTec/<category>/...
  VisA/<category>/...

$GLLS_QA_ROOT/
  DS-MVTec/<category>/QA.json
  VisA/<category>/QA.json
```

For binary AD:

```text
$GLLS_MPDD_ROOT
$GLLS_DTD_ROOT          # DTD-Synthetic, not raw Oxford DTD classification
$GLLS_DAGM_ROOT
```

The binary AD CLI supports MVTec-style `train/good` and `test/<defect>` folders,
DAGM `Train/Test/Label` layouts, and generated `meta.json` split files.

## Model and Checkpoint Resources

The repository does not redistribute model weights.

Recommended resources and expected local paths:

| Component | Download source | Expected path |
| --- | --- | --- |
| VLM | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) or another compatible Qwen/LLaVA local model | `$GLLS_VLM_MODEL_PATH` |
| SAM3 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) checkpoint links | `$GLLS_SAM3_PATH` |
| CLIP backbone | [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) or OpenCLIP-compatible ViT-L/14@336px weights | downloaded/cached by the localizer |
| AdaptCLIP | [gaobb/AdaptCLIP](https://github.com/gaobb/AdaptCLIP) adapter checkpoints | `$GLLS_ADAPTCLIP_ROOT/checkpoints/<domain>_epoch_15.pth` |
| ABounD (optional) | author-provided or locally trained ABounD checkpoint | `$GLLS_ABOUND_MODEL_PATH/<k>shot.pt` and `$GLLS_ABOUND_SAVE_PATH/<dataset>/...` |
| Text embedding model | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | `$GLLS_EMBEDDING_MODEL_PATH` |

The public default is AdaptCLIP 1-shot local evidence plus MCTS/SAM3/PVLA.
ABounD source code is retained for explicit local runs, but ABounD checkpoints
are not redistributed.

SAM3 also needs the official SAM3 Python package/repository on `PYTHONPATH`; the
activation script expects it at:

```text
$GLLS_DATA_ROOT/external/sam3
```

## Build PVLA Graph Knowledge

Graph caches are read from:

```text
$GLLS_GRAPH_CACHE_ROOT
```

Build or rebuild graph caches from text knowledge and visual references with:

```bash
python -m glls.rag.build_graph
```

The generated graph files are local resources and should stay outside git.

## Run DS-MVTec / VisA QA

Quick help:

```bash
python -m glls.cli.run --help
bash scripts/run/run_main.sh --help
```

Example:

```bash
source scripts/dev/activate_glls.sh
python -m glls.cli.run \
  --dataset mvtec \
  --subclass bottle \
  --dataset_root "$GLLS_DATASET_ROOT" \
  --qa_root "$GLLS_QA_ROOT" \
  --graph_cache_root "$GLLS_GRAPH_CACHE_ROOT" \
  --model_path "$GLLS_VLM_MODEL_PATH" \
  --sam_path "$GLLS_SAM3_PATH" \
  --localizer adaptclip \
  --k_shot 1 \
  --adaptclip_checkpoint_domain mvtec \
  --output_dir outputs/mvtec_bottle
```

To switch to VisA, use the VisA dataset/category and the VisA AdaptCLIP
checkpoint domain:

```bash
python -m glls.cli.run \
  --dataset visa \
  --subclass candle \
  --localizer adaptclip \
  --k_shot 1 \
  --adaptclip_checkpoint_domain visa \
  --dataset_root "$GLLS_DATASET_ROOT" \
  --qa_root "$GLLS_QA_ROOT" \
  --output_dir outputs/visa_candle
```

Outputs include per-question predictions and compact method traces showing
heatmap evidence, MCTS actions, SAM3 refinement, PVLA/RAG provenance, and final
answer parsing.

## Run MPDD / DTD / DAGM Binary AD

Prepare dataset metadata and offline normal-reference PVLA/SAM3 assets:

```bash
source scripts/dev/activate_glls.sh
python scripts/data/prepare_binary_ad_datasets.py
python scripts/data/prepare_binary_ad_offline.py --dataset all --max_refs 4 --with_sam3
```

Run the evaluator:

```bash
python -m glls.cli.binary_ad --dataset all --output_dir outputs/binary_ad
```

or use:

```bash
bash scripts/run/run_binary_ad.sh
```

The binary AD path keeps the same method structure: localizer heatmap scoring,
MCTS-style region selection, SAM3 region refinement, and PVLA normal-reference
knowledge.

## Run the Online Frontend

```bash
source scripts/dev/activate_glls.sh
bash scripts/run/run_visualize.sh --host 127.0.0.1 --port 7861
```

Then open:

```text
http://127.0.0.1:7861
```

The frontend exposes the same path configuration fields as the CLI. After
loading a dataset/category, each QA row can be run online and inspected with:

- original image, question, options, prediction, and ground truth;
- global heatmap;
- selected local crops;
- SAM3 refinement artifacts;
- PVLA/RAG source-backed knowledge blocks;
- MCTS/SAM3/RAG method trace;
- exportable run bundle.

## Minimal Release Policy

The GitHub release intentionally excludes:

- benchmark images;
- model checkpoints;
- generated graph caches;
- `outputs/`;
- `tests/`;
- old ablation/sweep/audit scripts;
- static HTML result dumps.

Keep new work inside the retained `src/glls` modules and the three supported
entrypoints unless the method itself needs a new shared component.
