# llm_biofactory

Bio demo for **BioCLIP + Qwen** with retrieval, multimodal reasoning, and correction write-back.

## Upstream References

- BioCLIP official site: https://imageomics.github.io/bioclip/
- Bailian doc entry: https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=3005961
- OpenAI-compatible API reference: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions
- Vision guide (localization/visual understanding): https://help.aliyun.com/zh/model-studio/vision

## Key Features

- BioCLIP embedding + local FAISS retrieval
- Qwen (`qwen3.5-plus`) reasoning with `enable_thinking`
- Chinese/English UI switch (default Chinese)
- **Large-scene small-target optimization**:
  - Qwen two-stage localization (reasoning -> structured JSON boxes)
  - optional YOLO assistant proposals
  - fused candidate boxes + crop preprocessing before BioCLIP retrieval
  - frontend explicitly shows recognition path (full-image vs localized-box)
  - frontend renders the exact boxes used for crop-based recognition
  - clue-guided final prompt for analysis
- Expert correction form writes back to vector store

## Project Layout

- `app.py`: Streamlit demo app
- `small_target_optimizer.py`: two-stage localization, YOLO assist, box fusion, crop generation
- `compare_small_target.py`: baseline vs optimized comparison script
- `build_index.py`: batch image-to-index builder
- `prepare_bioclip_local.py`: pre-download BioCLIP for local deployment
- `run_demo.sh`: startup script (Conda `torch1`)
- `vector_store.py`: local FAISS abstraction
- `bioclip_model.py`: model loading and embedding helpers
- `assets/`: test images (e.g. `白鹭2.jpg`)

## Environment

Create `.env` in project root:

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
DASHSCOPE_MODEL=qwen3.5-plus
DASHSCOPE_ENABLE_THINKING=1
APP_DEFAULT_LANGUAGE=zh
SMALL_TARGET_OPTIMIZATION=1
SMALL_TARGET_USE_QWEN=1
SMALL_TARGET_USE_YOLO=1
SMALL_TARGET_MAX_CROPS=4
YOLO_ASSIST_MODEL_PATH=./models/ultralytics/yolov12/best_yolo12_s_动物_1024_randcopybg.pt
BIOCLIP_MODEL_ID=hf-hub:imageomics/bioclip
HF_HOME=/home/buluwasior/Works/llm_biofactory/models/hf_cache
BIOCLIP_OFFLINE=0
```

## Install Dependencies (Conda torch1)

```bash
cd /home/buluwasior/Works/llm_biofactory
~/anaconda3/bin/conda run -n torch1 python -m pip install -r requirements.txt
```

## Prepare BioCLIP Local Cache (Recommended)

```bash
cd /home/buluwasior/Works/llm_biofactory
~/anaconda3/bin/conda run -n torch1 python prepare_bioclip_local.py
```

Optional offline mode:

```bash
export BIOCLIP_OFFLINE=1
```

## Build Retrieval Data

```bash
~/anaconda3/bin/conda run -n torch1 python build_index.py --sample-dir ./sample_images
```

## Run Demo

```bash
cd /home/buluwasior/Works/llm_biofactory
./run_demo.sh
```

Open `http://<server-ip>:8501`.

## Compare Baseline vs Optimized on Test Image

```bash
cd /home/buluwasior/Works/llm_biofactory
~/anaconda3/bin/conda run --no-capture-output -n torch1   python compare_small_target.py --image ./assets/白鹭2.jpg --top-k 3 --max-crops 4
```

The report is saved by your command pipeline if redirected, e.g. `data/compare_egret2.json`.
