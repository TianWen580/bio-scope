# BioScope Studio

BioScope Studio is an open-source bio-identification application for biological image and video understanding.
It combines BioCLIP retrieval priors, hierarchical taxonomy constraints, multimodal reasoning, and annotation write-back.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](.github/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Designed for hard field conditions (large background, tiny targets, occlusion, blur, color cast), the project focuses on explainability, controllability, and iterative improvement.

## Why BioScope Studio

- Retrieval-grounded reasoning pipeline, not free-form guessing
- Explicit Observe -> Induce -> Deduce -> Converge methodology
- Taxonomy-constrained final classification to reduce species drift
- Small-target optimization path for challenging wildlife scenes
- Human-in-the-loop annotation write-back for continuous improvement

## Quick links

- Chinese documentation: `README.zh-CN.md`
- Release notes: `RELEASE_v0.1.0.md`
- Contribution guide: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
# set DASHSCOPE_API_KEY in .env
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501`.

## Overview

The reasoning flow follows a first-principles routine: observe facts first, induce patterns from evidence, validate hypotheses deductively with BioCLIP priors and taxonomy constraints, then converge to the safest current conclusion.

This is a critical transformation rather than a reset: the legacy recognition stack is preserved, but it is now constrained into the new methodology as an evidence layer, constraint layer, and risk-check layer.

## Open-source readiness

This repository now includes baseline open-source governance and collaboration assets:

- `LICENSE` for reuse terms
- `CONTRIBUTING.md` for contributor workflow
- `CODE_OF_CONDUCT.md` for community behavior expectations
- `SECURITY.md` for vulnerability reporting policy
- `.github/ISSUE_TEMPLATE/` and `.github/pull_request_template.md` for standardized collaboration
- `.github/workflows/ci.yml` for automated test checks on pull requests

## Why this prototype matters

- Uses **retrieval-grounded reasoning** rather than pure free-form VLM guessing
- Makes **inductive reasoning explicit** by forming species hypotheses from repeated visual patterns before committing to a conclusion
- Makes **deductive reasoning explicit** by checking those hypotheses against BioCLIP priors, taxonomy constraints, and interference analysis
- Enforces **taxonomy-scoped final classification** (confidence-gated rank constraints)
- Supports **small-target workflows** through localization + crop-based retrieval
- Adds an **independent interference-analysis agent** before final report generation
- Keeps a full **annotation loop** so field feedback can continuously improve local retrieval behavior

In short: this is not just a demo response generator; it is a controllable decision pipeline suitable for high-value biodiversity and ecological monitoring scenarios.

## First-principles reasoning method

- `Observe`: separate directly visible facts from guesses
- `Induce`: summarize recurring morphology, texture, color, pose, and context patterns into candidate hypotheses
- `Deduce`: validate or eliminate hypotheses using retrieval evidence, taxonomy constraints, and interference risks
- `Converge`: keep the strongest current conclusion while surfacing uncertainty and the next verification step

## Core capabilities (latest status)

1. BioCLIP embedding + local FAISS retrieval
2. TreeOfLife-backed species priors (ToL classifier -> ToL species list -> metadata fallback)
3. Full taxonomy reference on priors (kingdom/phylum/class/order/family/genus/species + common name if available)
4. Confidence-gated taxonomy constraints for final Qwen classification
   - Degradation logic: family -> order -> class -> phylum -> kingdom (threshold default 0.9)
   - If kingdom confidence < threshold, constraints are disabled for this run
5. Independent interference-analysis agent between BioCLIP priors and final report
   - Route rule: if no species-like abstract Qwen box OR target boxes > limit, use full-image analysis; otherwise per-box analysis
   - Interference factors include rare pose, occlusion, color cast, low resolution, defocus blur, motion blur, exposure issues, tiny target, background clutter, truncation, taxonomy conflict
6. Large-scene small-target optimization
   - Qwen two-stage localization (reasoning -> JSON targets)
   - Optional YOLO assistant proposals
   - Box fusion + crop generation for retrieval
7. Stage consistency guardrails
   - Stage-1 is explicitly treated as localization hypothesis (not final taxonomy assertion)
   - Interference agent now consumes BioCLIP priors + taxonomy constraints and records taxonomy conflict risk rather than giving contradictory final taxonomic conclusions
8. Alias-aware species handling
   - Catalog search supports scientific name + common name + alias dictionary (e.g. 华南兔 -> Lepus sinensis)
   - Final report alias/common-name normalization to canonical scientific names
   - Annotation write-back stores canonical species names
9. Formal video reasoning support
   - keyframe sampling from uploaded video files
   - frame-level constrained reasoning reuse
   - video-level temporal summary synthesis
10. BioCLIP2/BioCLIP parallel model candidates (default BioCLIP2)
    - runtime fallback to BioCLIP1 when OOM/load failures occur
    - model-specific vector index paths to avoid dimension conflicts (768 vs 512)
11. Bilingual UI (Chinese/English), Chinese default
12. Configurable long-timeout and thinking budget fallback for robust remote inference

## Typical pipeline

1. User uploads image or video
2. Optional localization and crop preparation
3. BioCLIP retrieval and ToL prior generation
4. Observation pass on visible facts and target positions
5. Inductive pass on recurring evidence patterns
6. Deductive pass with taxonomy constraints and interference checks
7. Converged report generation with uncertainty and next verification steps
8. Optional user annotation write-back to vector store

## Repository layout

- `app.py`: Streamlit application and orchestration
- `small_target_optimizer.py`: localization, fusion, crop generation, interference-analysis agent
- `bioclip_model.py`: BioCLIP loaders, ToL species helpers, taxonomy constraints, taxonomy enrichment
- `vector_store.py`: local FAISS wrapper
- `build_index.py`: build retrieval index from sample images
- `export_tol_species_list.py`: export ToL taxa CSV + species TXT
- `prepare_bioclip_local.py`: warm local cache for BioCLIP assets
- `compare_small_target.py`: baseline vs optimized comparison utility
- `run_demo.sh`: startup script (Conda by default, supports env overrides)
- `data/`: retrieval data, ToL files, alias dictionary

## Environment variables

Copy `.env.example` to `.env` in project root, then edit values as needed:

```bash
cp .env.example .env
```

Reference configuration:

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
DASHSCOPE_MODEL=qwen3.5-plus
DASHSCOPE_TIMEOUT_SECONDS=1800
DASHSCOPE_ENABLE_THINKING=1
DASHSCOPE_THINKING_BUDGET=8192

APP_DEFAULT_LANGUAGE=zh

SMALL_TARGET_OPTIMIZATION=1
SMALL_TARGET_USE_QWEN=1
SMALL_TARGET_USE_YOLO=1
SMALL_TARGET_MAX_CROPS=4
YOLO_ASSIST_MODEL_PATH=./models/ultralytics/yolov12/best_yolo12_s_动物_1024_randcopybg.pt

BIOCLIP_MODEL_CANDIDATES=hf-hub:imageomics/bioclip-2,hf-hub:imageomics/bioclip
BIOCLIP_MODEL_ID=hf-hub:imageomics/bioclip-2
BIOCLIP_EMBEDDING_DIM=
BIOCLIP_TOL_MODEL_ID=hf-hub:imageomics/bioclip
BIOCLIP_USE_TOL_CLASSIFIER=1
BIOCLIP_AUTO_EXPORT_TOL_SPECIES=1
BIOCLIP_SPECIES_LIST_PATH=./data/bioclip_tol_species.txt
BIOCLIP_SPECIES_CSV_PATH=./data/bioclip_tol_taxa.csv
BIOCLIP_SPECIES_ALIAS_PATH=./data/species_aliases.json
BIOCLIP_SPECIES_LIST_MAX_LABELS=0
BIOCLIP_TAXONOMY_CONSTRAINT_THRESHOLD=0.9

BIOCLIP_INDEX_PATH=./data/faiss_index.bin
BIOCLIP_METADATA_PATH=./data/faiss_metadata.pkl
BIOCLIP2_INDEX_PATH=./data/faiss_index_bioclip2.bin
BIOCLIP2_METADATA_PATH=./data/faiss_metadata_bioclip2.pkl

INTERFERENCE_BOX_LIMIT=10
INTERFERENCE_MAX_TARGETS=10

VIDEO_FRAME_INTERVAL_SECONDS=2.0
VIDEO_MAX_FRAMES=10

# Keyframe strategy: mechanical (default), qwen_video, bioclip2_consistency
VIDEO_KEYFRAME_STRATEGY=mechanical

# Strategy-3 (qwen_video) parameters
VIDEO_QWEN_KEYFRAME_FPS=1.0
VIDEO_QWEN_MAX_CANDIDATE_FRAMES=64

# Strategy-2 (bioclip2_consistency) parameters
VIDEO_BIOCLIP_TEMPORAL_WEIGHT=0.35
VIDEO_BIOCLIP_DIVERSITY_WEIGHT=0.65

# Multi-GPU configuration (optional)
# BIOCLIP_DEVICE=cuda:1    # Force BioCLIP to use specific GPU
# YOLO_DEVICE=cuda:0       # Force YOLO to use specific GPU
# If not set, system auto-selects GPU with most free memory

HF_HOME=./models/hf_cache
BIOCLIP_OFFLINE=0
```

## Keyframe extraction strategies

The video reasoning pipeline supports three keyframe selection strategies:

1. **mechanical** (default): Interval-based sampling using `VIDEO_FRAME_INTERVAL_SECONDS` and `VIDEO_MAX_FRAMES`. Simple and deterministic.

2. **bioclip2_consistency**: Embedding-aware selection that balances visual diversity and temporal coverage. Uses BioCLIP2 embeddings with configurable weights:
   - `VIDEO_BIOCLIP_TEMPORAL_WEIGHT`: Temporal gap importance (default 0.35)
   - `VIDEO_BIOCLIP_DIVERSITY_WEIGHT`: Visual diversity importance (default 0.65)

3. **qwen_video**: Qwen3.5-plus guided selection using `json_object` response format.
   - Calls Qwen with `response_format={"type":"json_object"}` (NOT `json_schema`)
   - Output contract: `{"frame_positions":[{"frame_id":<integer>}]}`
   - Frame-position-only: no timestamps, no prose descriptions
   - Parameters:
     - `VIDEO_QWEN_KEYFRAME_FPS`: Candidate frame sampling rate (default 1.0)
     - `VIDEO_QWEN_MAX_CANDIDATE_FRAMES`: Maximum frames for Qwen to evaluate (default 64)
   - Qwen acts as formatter: selects frame IDs from candidate set

**Fallback behavior**: If a non-default strategy (qwen_video or bioclip2_consistency) fails due to provider rejection, malformed output, or runtime error, the pipeline automatically falls back to mechanical extraction with a warning. This ensures video processing continues even when advanced strategies are unavailable.

**API constraint for qwen3.5-plus**: This flow uses `json_object` response format only. The model does NOT support `json_schema` constrained decoding in this prototype. Invalid JSON or non-frame-position outputs trigger automatic mechanical fallback.


## Setup and run

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Configure environment variables

```bash
cp .env.example .env
```

Set at least `DASHSCOPE_API_KEY` in `.env` before starting the app.

### 3) Prepare BioCLIP local cache (recommended)

```bash
python prepare_bioclip_local.py
```

Optional offline mode:

```bash
export BIOCLIP_OFFLINE=1
```

### 4) Export TreeOfLife species assets (recommended)

```bash
python export_tol_species_list.py \
  --species-txt ./data/bioclip_tol_species.txt \
  --taxa-csv ./data/bioclip_tol_taxa.csv
```

### 5) Build retrieval index (if needed)

```bash
# Prepare images in ./sample_images before running this command
python build_index.py --sample-dir ./sample_images
```

### 6) Run app

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Or use the helper script:

```bash
./run_demo.sh
```

Open: `http://localhost:8501`

## Development and quality checks

```bash
python -m pip install -r requirements-test.txt
python -m pytest
```

## Alias dictionary example

`data/species_aliases.json`

```json
{
  "Lepus sinensis": ["华南兔", "中国野兔", "Chinese hare", "South China hare"]
}
```

## Notes on BioCLIP2 upgrade

- BioCLIP2 (ViT-L/14) is larger than BioCLIP1 (ViT-B/16) and may trigger OOM on constrained GPUs.
- The app defaults to BioCLIP2 but keeps BioCLIP1 as fallback candidate.
- Do not reuse the same FAISS index across different embedding dimensions; use model-specific index files.

## Practical application advantages

- **Ecology field surveys**: supports noisy captures and incomplete morphology
- **Biodiversity monitoring**: controllable taxonomy constraints reduce hallucinated species drift
- **Education and extension**: bilingual explainable output with evidence and risk factors
- **Expert-in-the-loop workflows**: annotation feedback continuously strengthens local retrieval memory
- **Risk-managed deployment**: intermediate stage hypotheses are separated from final constrained conclusions

## Upstream references

- BioCLIP official site: https://imageomics.github.io/bioclip/
- Bailian model doc entry: https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=3005961
- OpenAI-compatible Qwen API: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
- Vision guide: https://help.aliyun.com/zh/model-studio/vision
