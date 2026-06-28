# GLLS

Global Logic and Local Semantic inspection for industrial anomaly QA.

GLLS runs MMAD-style visual QA with a global VLM pass, local anomaly evidence,
SAM3 cutouts, and PVLA/RAG graph knowledge. The repo also includes a binary
anomaly-detection path for MPDD, DTD-Synthetic, and DAGM.

Main entrypoints:

- `glls.cli.run`: DS-MVTec and VisA QA evaluation.
- `glls.cli.binary_ad`: MPDD, DTD-Synthetic, and DAGM binary AD.
- `glls.cli.visualize`: Gradio frontend for single-question inspection.

## Motivation

Industrial visual QA is brittle when the model only sees the whole image, or
when it receives crops without context. GLLS keeps the whole-image answer, local
heatmap evidence, SAM3 masks, and retrieved normal references in one trace.

<p align="center">
  <img src="docs/assets/motivation.png" alt="GLLS motivation and benchmark summary" width="900">
</p>

## Method Overview

GLLS combines four components:

1. **Global inspection**: a VLM reads the full image, question, and options.
2. **Local evidence search**: the localizer produces heatmaps, and MCTS ranks
   candidate regions instead of sending every crop to the VLM.
3. **SAM3 refinement**: SAM3 refines selected regions with conservative category
   prompts from `src/glls/seg/sam3_profiles.py`.
4. **PVLA/RAG knowledge**: the VLM receives retrieved normal-reference and graph
   evidence alongside the target image.

<p align="center">
  <img src="docs/assets/framework.png" alt="GLLS framework" width="900">
</p>

## Dataset Notes

Use the MMAD images, but use this repo's QA files:

```bash
export GLLS_QA_ROOT=/path/to/GLLS/qa_collection
```

The raw MMAD `QA.json` files contain known label and option errors. The
`qa_collection/` directory contains only replacement `QA.json` files; it does
not contain or edit benchmark images.

Before running DS-MVTec `pill`, fix the image folder if your MMAD copy has this
issue. In affected MMAD copies, `DS-MVTec/pill/image/good/000.png` through
`021.png` are `metal_nut` images. Replace `DS-MVTec/pill/image/good/` with the
official MVTec-AD
`pill/test/good/` images (`000.png` through `025.png`).

## Quick Start From Zero

The commands below assume Linux, CUDA, and Python 3.10 or newer. The default
workspace is `~/data/GLLS`; change it once in `scripts/dev/local_paths.sh` if
your datasets or checkpoints live elsewhere.

```bash
git clone https://github.com/KomorebiWerts/GLLS.git
cd GLLS

mkdir -p ~/data/GLLS/envs
python3 -m venv ~/data/GLLS/envs/GLLS
source ~/data/GLLS/envs/GLLS/bin/activate
python -m pip install -U pip

# Install the CUDA build that matches your machine first.
# Example for CUDA 12.8:
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirements.txt
```

Create your local path config:

```bash
cp scripts/dev/local_paths.example.sh scripts/dev/local_paths.sh
${EDITOR:-nano} scripts/dev/local_paths.sh
source scripts/dev/activate_glls.sh
```

`activate_glls.sh` loads `scripts/dev/local_paths.sh` automatically and then
sets `PYTHONPATH`, model paths, dataset paths, graph-cache paths, and the Python
environment used by the helper scripts.

Download the localizer artifacts used by the published runtime routes:

```bash
# AdaptCLIP is used for 0-shot MVTec/VisA runs, MPDD/DTD/DAGM runs,
# and as the fallback localizer.
mkdir -p "$GLLS_ADAPTCLIP_ROOT/checkpoints"
# Download the upstream AdaptCLIP MVTec and VisA adapter checkpoints, then place:
#   $GLLS_ADAPTCLIP_ROOT/checkpoints/mvtec_epoch_15.pth
#   $GLLS_ADAPTCLIP_ROOT/checkpoints/visa_epoch_15.pth

# ABounD 1-shot artifact for the published MVTec/VisA path.
hf download komorebi01/glls-abound-1shot \
  --local-dir "$GLLS_ABOUND_SAVE_PATH"
```

The ABounD download should keep this layout:

```text
$GLLS_ABOUND_SAVE_PATH/
  model_config.json
  model/ViT-L-14-336px.pt
  mvtec/final_vvclip_model_state_mvtec.pth
  mvtec/final_soft_prompt_state_mvtec.pth
  mvtec/final_memory_bank_mvtec.pt
  visa/final_vvclip_model_state_visa.pth
  visa/final_soft_prompt_state_visa.pth
  visa/final_memory_bank_visa.pt
```

`scripts/dev/local_paths.example.sh` already points ABounD to the clean artifact
path:

```bash
export GLLS_ABOUND_MODEL_PATH="$GLLS_DATA_ROOT/models/glls-abound-1shot/model"
export GLLS_ABOUND_SAVE_PATH="$GLLS_DATA_ROOT/models/glls-abound-1shot"
```

Check the Hugging Face model card for the artifact license before redistributing
the ABounD files. If no license is declared there, treat the artifact as
research-use until the license is clarified.

Verify that the published runtime route resolves as expected:

```bash
PYTHONPATH=src python scripts/dev/check_runtime_weight_config.py
```

## Repository Layout

```text
GLLS/
  qa_collection/                 # curated MMAD QA annotations
  requirements.txt               # public install entrypoint
  requirements-glls.txt          # pinned GLLS runtime dependencies
  scripts/
    dev/activate_glls.sh         # environment and path setup
    dev/local_paths.example.sh   # copy to local_paths.sh and edit once
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

## Data and Checkpoint Setup

Download MMAD from the upstream project
[`jam-cc/mmad`](https://github.com/jam-cc/mmad) or the Hugging Face mirror
[`jiang-cc/MMAD`](https://huggingface.co/datasets/jiang-cc/MMAD). The paths
should look like this after setup:

```text
$GLLS_DATASET_ROOT/
  DS-MVTec/<category>/...
  VisA/<category>/...

$GLLS_QA_ROOT/
  DS-MVTec/<category>/QA.json
  VisA/<category>/QA.json
```

Quick check:

```bash
test -f "$GLLS_DATASET_ROOT/DS-MVTec/pill/image/good/000.png"
test -f "$GLLS_DATASET_ROOT/DS-MVTec/pill/image/good/025.png"
test -d "$GLLS_DATASET_ROOT/VisA/candle/test/good"
test -f "$GLLS_QA_ROOT/DS-MVTec/pill/QA.json"
test -f "$GLLS_QA_ROOT/VisA/candle/QA.json"
```

For binary anomaly detection, configure these optional paths:

```text
$GLLS_MPDD_ROOT
$GLLS_DTD_ROOT          # DTD-Synthetic, not raw Oxford DTD classification
$GLLS_DAGM_ROOT
```

Recommended model resources:

| Component | Source | Configure as |
| --- | --- | --- |
| VLM | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) or another compatible Qwen/LLaVA local model | `GLLS_VLM_MODEL_PATH` |
| SAM3 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) checkpoint links | `GLLS_SAM3_PATH` |
| SAM3 code | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | `GLLS_DATA_ROOT/external/sam3` on `PYTHONPATH` |
| CLIP backbone | [openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336) or OpenCLIP-compatible ViT-L/14@336px weights | downloaded/cached by localizer |
| AdaptCLIP | [gaobb/AdaptCLIP](https://github.com/gaobb/AdaptCLIP) adapter checkpoints | `GLLS_ADAPTCLIP_ROOT/checkpoints/<domain>_epoch_15.pth` |
| Text embedding | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | `GLLS_EMBEDDING_MODEL_PATH` |
| ABounD 1-shot | [komorebi01/glls-abound-1shot](https://huggingface.co/komorebi01/glls-abound-1shot), including `model_config.json` | `GLLS_ABOUND_MODEL_PATH` / `GLLS_ABOUND_SAVE_PATH` |

The published localizer route is shared by the batch CLI and frontend:

| Dataset route | Shot | Localizer | Decision parameter source |
| --- | ---: | --- | --- |
| DS-MVTec / VisA QA | 1 | ABounD | ABounD `model_config.json` `decision_parameters.heatmap`, normal-only source |
| DS-MVTec / VisA QA | 0 | AdaptCLIP | bundled AdaptCLIP `decision_parameters.heatmap_by_shot` |
| MPDD / DTD-Synthetic / DAGM binary AD | 0 or 1 | AdaptCLIP | bundled AdaptCLIP `decision_parameters.binary_ad` |

Use `--localizer auto` for DS-MVTec/VisA QA to get this route. Explicit
`--localizer adaptclip` and `--localizer abound` remain available for ablations.

## Build PVLA Graph Knowledge

PVLA graph caches are read from `GLLS_GRAPH_CACHE_ROOT`.

```bash
source scripts/dev/activate_glls.sh
python -m glls.rag.build_graph
```

Run this after your text knowledge and normal-reference assets are placed under
the configured database root. The QA and binary-AD entrypoints will then retrieve
source-backed graph blocks from `GLLS_GRAPH_CACHE_ROOT`.

## Run DS-MVTec / VisA QA

Check the available options:

```bash
python -m glls.cli.run --help
bash scripts/run/run_main.sh --help
```

Run one DS-MVTec category:

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
  --localizer auto \
  --k_shot 1 \
  --output_dir outputs/mvtec_bottle
```

Run one VisA category:

```bash
source scripts/dev/activate_glls.sh
python -m glls.cli.run \
  --dataset visa \
  --subclass candle \
  --dataset_root "$GLLS_DATASET_ROOT" \
  --qa_root "$GLLS_QA_ROOT" \
  --graph_cache_root "$GLLS_GRAPH_CACHE_ROOT" \
  --model_path "$GLLS_VLM_MODEL_PATH" \
  --sam_path "$GLLS_SAM3_PATH" \
  --localizer auto \
  --k_shot 1 \
  --output_dir outputs/visa_candle
```

For the 0-shot QA setting, keep the same command and set `--k_shot 0`.
`--localizer auto` will switch to AdaptCLIP and load the matching MVTec or VisA
checkpoint.

Each result row includes prediction, answer parsing, heatmap evidence, MCTS
action trace, SAM3 refinement audit, PVLA/RAG provenance, and final prompt
evidence.

## Run the Online Frontend

```bash
source scripts/dev/activate_glls.sh
bash scripts/run/run_visualize.sh --host 127.0.0.1 --port 7861
```

Open:

```text
http://127.0.0.1:7861
```

The frontend is designed for method playback rather than raw path debugging:

1. Initialize the runtime from your local path config.
2. Choose dataset, shot count, category, task, and QA row.
3. Run GLLS and inspect the two-stage evidence flow: Phase-1 global report,
   small-model heatmap proposals, MCTS crop search, SAM3 structural cut/gate,
   source-backed PVLA/RAG recall, and Phase-2 answer fusion.

For the published ABounD artifact path, choose `mvtec` or `visa` with `1-shot`
in the frontend. The frontend will load ABounD from `GLLS_ABOUND_MODEL_PATH` and
`GLLS_ABOUND_SAVE_PATH`. Choose `0-shot` for MVTec/VisA to use AdaptCLIP; other
dataset/shot combinations also use AdaptCLIP.

Representative frontend captures:

<p align="center">
  <img src="docs/assets/frontend/frontend_pvla_cable.png" alt="GLLS frontend cable sample selection and PVLA graph-shaped knowledge" width="900">
</p>

<p align="center">
  <img src="docs/assets/frontend/frontend_evidence_streams.png" alt="GLLS frontend global stream local search stream and fusion summary" width="900">
</p>

<p align="center">
  <img src="docs/assets/frontend/frontend_run_visuals_cable.png" alt="GLLS frontend cable logic view anomaly heatmap and local evidence crops" width="900">
</p>

<p align="center">
  <img src="docs/assets/frontend/frontend_pvla_pcb.png" alt="GLLS frontend PCB sample selection and PVLA graph-shaped knowledge" width="900">
</p>

<p align="center">
  <img src="docs/assets/frontend/frontend_run_visuals_pcb.png" alt="GLLS frontend PCB anomaly heatmap red-box trace and focus crop views" width="900">
</p>

## Run MPDD / DTD / DAGM Binary AD

Prepare dataset metadata and offline normal-reference PVLA/SAM3 assets:

```bash
source scripts/dev/activate_glls.sh
python scripts/data/prepare_binary_ad_datasets.py
python scripts/data/prepare_binary_ad_offline.py --dataset all --max_refs 1 --with_sam3
```

Run the paper-comparable full binary-AD route:

```bash
python scripts/eval/run_binary_ad_qwen3_sweep.py \
  --shot 0 \
  --output_root outputs/binary_ad/qwen3_full_0shot \
  --gpus 0,2,3 \
  --model_path /home/dataset_model/model/qwen3-vl-8B

python scripts/eval/run_binary_ad_qwen3_sweep.py \
  --shot 1 \
  --output_root outputs/binary_ad/qwen3_full_1shot \
  --gpus 0,2,3 \
  --model_path /home/dataset_model/model/qwen3-vl-8B
```

or:

```bash
bash scripts/run/run_binary_ad.sh
```

Use `--k_shot 0` for the zero-shot wrapper run. MPDD, DTD-Synthetic, and DAGM
use AdaptCLIP in both 0-shot and 1-shot settings. The paper-comparable route
uses:

- `--threshold_policy table`, with MPDD/DTD/DAGM decision cutoffs loaded from
  `src/glls/models/AdaptCLIP/adaptcliplib/model_config.json`
- `--binary_score_source localizer_image`
- `--final_verifier qwen3 --final_verifier_policy anomaly_or`
- scaled MCTS/SAM3 crop evidence and offline PVLA normal references

The older train-normal `normal_robust` threshold route is only a lightweight
ablation; it is not the paper-comparable binary split and can substantially
underestimate MPDD/DTD. To run that ablation intentionally through the wrapper,
set `GLLS_BINARY_AD_LIGHTWEIGHT=1`.

The binary AD path keeps the same method structure: localizer heatmap scoring,
MCTS-style region selection, SAM3 region refinement, and PVLA normal-reference
knowledge.
