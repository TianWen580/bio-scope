# llm_biofactory

Bio demo implementation for **BioCLIP + Qwen** with retrieval, reasoning, and write-back loop.

## Upstream References

- BioCLIP official site: https://imageomics.github.io/bioclip/
- Bailian doc entry: https://bailian.console.aliyun.com/cn-beijing/?tab=doc#/doc/?type=model&url=3005961
- OpenAI-compatible API reference: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions

## Features

- BioCLIP image embeddings (local cached model)
- Local FAISS retrieval with metadata evidence
- Qwen call via OpenAI-compatible endpoint
- `qwen3.5-plus` with `enable_thinking` on by default
- Chinese/English UI switch (default: Chinese)
- Model output language follows selected UI language
- Expert correction form with vector write-back
- Batch index builder for cold-start dataset import

## Project Layout

- `app.py`: Streamlit demo app
- `build_index.py`: batch image-to-index builder
- `prepare_bioclip_local.py`: pre-download BioCLIP for local deployment
- `run_demo.sh`: one-command startup script (Conda `torch1`)
- `vector_store.py`: local FAISS abstraction
- `bioclip_model.py`: model loading and embedding helpers
- `data/`: generated index and metadata files
- `sample_images/`: initial image dataset directory
- `models/hf_cache/`: local BioCLIP cache

## Environment

Create `.env` in project root:

```bash
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
DASHSCOPE_MODEL=qwen3.5-plus
DASHSCOPE_ENABLE_THINKING=1
APP_DEFAULT_LANGUAGE=zh
BIOCLIP_MODEL_ID=hf-hub:imageomics/bioclip
HF_HOME=/home/buluwasior/Works/llm_biofactory/models/hf_cache
BIOCLIP_OFFLINE=0
```

## Install Dependencies (Conda torch1)

```bash
cd /home/buluwasior/Works/llm_biofactory
~/anaconda3/bin/conda run -n torch1 python -m pip install -r requirements.txt
```

## Deploy BioCLIP Locally (Recommended)

Run once with internet to cache model files locally:

```bash
cd /home/buluwasior/Works/llm_biofactory
~/anaconda3/bin/conda run -n torch1 python prepare_bioclip_local.py
```

Then force offline loading:

```bash
export BIOCLIP_OFFLINE=1
```

## Prepare Retrieval Data

Put sample images under `sample_images/`, then build index:

```bash
~/anaconda3/bin/conda run -n torch1 python build_index.py --sample-dir ./sample_images
```

## Run Demo

```bash
cd /home/buluwasior/Works/llm_biofactory
./run_demo.sh
```

Open `http://<server-ip>:8501`.
