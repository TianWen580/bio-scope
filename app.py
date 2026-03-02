from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from typing import Any

from PIL import Image
import requests
import streamlit as st

from bioclip_model import encode_image, load_bioclip_model
from vector_store import LocalFAISSStore


INDEX_PATH = './data/faiss_index.bin'
METADATA_PATH = './data/faiss_metadata.pkl'


LANGUAGE_PACK = {
    'zh': {
        'page_title': 'BioCLIP + Qwen 生物助手 v4.0',
        'title': 'BioCLIP + Qwen 多模态生物 Demo v4.0',
        'caption': '专业视觉表征 + 证据检索 + 多模态推理 + 修正回写闭环',
        'language_label': '语言 / Language',
        'sidebar_config': '配置',
        'api_key_label': 'DashScope API Key',
        'model_label': '模型 ID',
        'base_url_label': 'API Base URL',
        'thinking_label': '开启深度思考',
        'topk_label': 'Top-K 检索',
        'index_path_label': '索引路径',
        'metadata_path_label': '元数据路径',
        'bio_loaded': 'BioCLIP 已加载（设备: {device}）',
        'bio_failed': 'BioCLIP 加载失败: {error}',
        'upload_label': '上传图片',
        'upload_info': '请先上传图片再开始分析。',
        'uploaded_caption': '已上传图片',
        'run_analysis_btn': '开始分析',
        'missing_key': '请先填写 DashScope API Key。',
        'no_samples': '本地样本库为空，将仅基于通用生物知识分析。',
        'analysis_report': '分析报告',
        'retrieved_evidence': '检索证据',
        'feedback_title': '专家修正与回写',
        'correct_name': '修正物种名',
        'location': '采集地点',
        'confidence': '修正置信度 (%)',
        'notes': '修正备注',
        'save_btn': '保存修正到向量库',
        'no_embedding': '没有可用向量，请先完成一次分析。',
        'save_success': '修正已保存，当前样本数: {count}',
        'save_failed': '保存失败: {error}',
        'footer': '首次使用请先运行 build_index.py 初始化检索库。',
        'request_failed': '请求失败: {error}',
    },
    'en': {
        'page_title': 'BioCLIP + Qwen Bio Assistant v4.0',
        'title': 'BioCLIP + Qwen Multimodal Bio Demo v4.0',
        'caption': 'Professional visual embedding + evidence retrieval + multimodal reasoning + correction loop',
        'language_label': 'Language / 语言',
        'sidebar_config': 'Configuration',
        'api_key_label': 'DashScope API Key',
        'model_label': 'Model ID',
        'base_url_label': 'API Base URL',
        'thinking_label': 'Enable thinking',
        'topk_label': 'Top-K retrieval',
        'index_path_label': 'Index path',
        'metadata_path_label': 'Metadata path',
        'bio_loaded': 'BioCLIP loaded on {device}',
        'bio_failed': 'Failed to load BioCLIP: {error}',
        'upload_label': 'Upload image',
        'upload_info': 'Upload an image to start analysis.',
        'uploaded_caption': 'Uploaded image',
        'run_analysis_btn': 'Run analysis',
        'missing_key': 'Please provide DashScope API Key.',
        'no_samples': 'No local samples found. Proceed with generic biological reasoning only.',
        'analysis_report': 'Analysis Report',
        'retrieved_evidence': 'Retrieved Evidence',
        'feedback_title': 'Expert Correction and Write-back',
        'correct_name': 'Correct species name',
        'location': 'Collection location',
        'confidence': 'Correct confidence (%)',
        'notes': 'Correction notes',
        'save_btn': 'Save correction into vector store',
        'no_embedding': 'No embedding available. Run analysis first.',
        'save_success': 'Correction saved. Current sample count: {count}',
        'save_failed': 'Failed to write correction: {error}',
        'footer': 'Run build_index.py first to initialize local retrieval data.',
        'request_failed': 'Request failed: {error}',
    },
}


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def response_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                if 'text' in item and isinstance(item['text'], str):
                    chunks.append(item['text'])
                elif 'content' in item and isinstance(item['content'], str):
                    chunks.append(item['content'])
            elif isinstance(item, str):
                chunks.append(item)
        return '\n'.join(chunks).strip()
    return str(content)


def build_prompt(evidence_text: str, language: str) -> str:
    if language == 'zh':
        return f'''你是一位专业生物学家。请结合以下检索证据分析上传的样本图片。

[检索证据]
{evidence_text}

[输出要求]
1. 给出 Top-3 候选物种，并提供置信度与依据。
2. 描述关键形态学特征。
3. 如果 Top-1 置信度低于 80%，请明确列出你需要用户补充的关键信息。
4. 给出下一步采样或验证建议。

必须使用简体中文输出，使用清晰分段。'''

    return f'''You are a professional biologist. Analyze the uploaded specimen image using the retrieval evidence below.

[Retrieval Evidence]
{evidence_text}

[Output Requirements]
1. Top-3 candidate species with confidence and rationale.
2. Key morphology observations.
3. If top-1 confidence is below 80%, list exact follow-up questions for the user.
4. Next sampling or validation recommendation.

You must answer in English and use clear section headers.'''


def call_openai_compatible(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    image_base64: str,
    image_mime: str,
    enable_thinking: bool = True,
) -> tuple[bool, str]:
    url = base_url.rstrip('/') + '/chat/completions'
    payload = {
        'model': model_name,
        'enable_thinking': enable_thinking,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:{image_mime};base64,{image_base64}'}},
                    {'type': 'text', 'text': prompt},
                ],
            }
        ],
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
    except Exception as exc:
        return False, f'HTTP request failed: {exc}'

    if response.status_code != 200:
        return False, f'HTTP {response.status_code}: {response.text[:500]}'

    try:
        data = response.json()
        content = data['choices'][0]['message'].get('content', '')
        return True, response_to_text(content)
    except Exception as exc:
        return False, f'Invalid API response format: {exc}'


def get_default_language() -> str:
    default_language = os.getenv('APP_DEFAULT_LANGUAGE', 'zh').strip().lower()
    if default_language not in {'zh', 'en'}:
        return 'zh'
    return default_language


if 'app_lang' not in st.session_state:
    st.session_state.app_lang = get_default_language()

if st.session_state.app_lang not in {'zh', 'en'}:
    st.session_state.app_lang = 'zh'

st.set_page_config(page_title=LANGUAGE_PACK[st.session_state.app_lang]['page_title'], layout='wide')

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'current_embedding' not in st.session_state:
    st.session_state.current_embedding = None
if 'evidence_rows' not in st.session_state:
    st.session_state.evidence_rows = []

with st.sidebar:
    selected_lang = st.selectbox(
        LANGUAGE_PACK[st.session_state.app_lang]['language_label'],
        options=['zh', 'en'],
        index=0 if st.session_state.app_lang == 'zh' else 1,
        format_func=lambda x: '中文' if x == 'zh' else 'English',
    )
    st.session_state.app_lang = selected_lang

lang = st.session_state.app_lang
text = LANGUAGE_PACK[lang]

st.title(text['title'])
st.caption(text['caption'])

with st.sidebar:
    st.subheader(text['sidebar_config'])
    env_key = os.getenv('DASHSCOPE_API_KEY', '')
    env_model = os.getenv('DASHSCOPE_MODEL', 'qwen3.5-plus')
    env_base = os.getenv('DASHSCOPE_BASE_URL', 'https://coding.dashscope.aliyuncs.com/v1')
    env_thinking = os.getenv('DASHSCOPE_ENABLE_THINKING', '1').lower() in {'1', 'true', 'yes', 'on'}
    api_key = st.text_input(text['api_key_label'], value=env_key, type='password')
    model_name = st.text_input(text['model_label'], value=env_model)
    base_url = st.text_input(text['base_url_label'], value=env_base)
    enable_thinking = st.checkbox(text['thinking_label'], value=env_thinking)
    top_k = st.slider(text['topk_label'], min_value=1, max_value=10, value=3)
    st.write(f"{text['index_path_label']}:", INDEX_PATH)
    st.write(f"{text['metadata_path_label']}:", METADATA_PATH)

try:
    model, preprocess, device = load_bioclip_model()
    st.sidebar.success(text['bio_loaded'].format(device=device))
except Exception as exc:
    st.error(text['bio_failed'].format(error=exc))
    st.stop()

store = LocalFAISSStore(INDEX_PATH, METADATA_PATH, dimension=512)

uploaded_file = st.file_uploader(text['upload_label'], type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])
if uploaded_file is None:
    st.info(text['upload_info'])
    st.stop()

image = Image.open(uploaded_file).convert('RGB')
image_mime = uploaded_file.type or 'image/jpeg'
left_col, right_col = st.columns([1, 1])

with left_col:
    st.image(image, caption=text['uploaded_caption'], use_container_width=True)

with right_col:
    run_analysis = st.button(text['run_analysis_btn'], type='primary', use_container_width=True)

if run_analysis:
    if not api_key:
        st.warning(text['missing_key'])
    else:
        embedding = encode_image(image, model, preprocess, device)
        st.session_state.current_embedding = embedding

        similar_results = store.search(embedding, top_k=top_k)
        if similar_results:
            evidence_lines = []
            evidence_rows = []
            for i, item in enumerate(similar_results, start=1):
                meta = item['metadata']
                score = item['similarity']
                line = (
                    f"{i}. species={meta.get('name', 'unknown')}, "
                    f"similarity={score:.4f}, "
                    f"location={meta.get('location', 'unknown')}, "
                    f"notes={meta.get('notes', '')}"
                )
                evidence_lines.append(line)
                evidence_rows.append(
                    {
                        'species': meta.get('name', 'unknown'),
                        'similarity': f'{score:.4f}',
                        'location': meta.get('location', 'unknown'),
                        'notes': meta.get('notes', ''),
                    }
                )
            evidence_text = '\n'.join(evidence_lines)
            st.session_state.evidence_rows = evidence_rows
        else:
            evidence_text = text['no_samples']
            st.session_state.evidence_rows = []

        base64_image = image_to_base64(image)
        prompt = build_prompt(evidence_text, language=lang)
        ok, result = call_openai_compatible(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            image_base64=base64_image,
            image_mime=image_mime,
            enable_thinking=enable_thinking,
        )
        if ok:
            st.session_state.analysis_result = result
        else:
            st.error(text['request_failed'].format(error=result))

if st.session_state.analysis_result:
    st.markdown('---')
    st.subheader(text['analysis_report'])
    st.markdown(st.session_state.analysis_result)

    if st.session_state.evidence_rows:
        st.subheader(text['retrieved_evidence'])
        st.table(st.session_state.evidence_rows)

    st.markdown('---')
    st.subheader(text['feedback_title'])
    with st.form('feedback_form'):
        corrected_name = st.text_input(text['correct_name'], value='unknown_species')
        location = st.text_input(text['location'], value='unknown')
        confidence = st.slider(text['confidence'], min_value=0, max_value=100, value=50)
        notes = st.text_area(text['notes'], value='')
        submit = st.form_submit_button(text['save_btn'])

    if submit:
        if st.session_state.current_embedding is None:
            st.warning(text['no_embedding'])
        else:
            new_metadata = {
                'name': corrected_name.strip() or 'unknown_species',
                'location': location.strip() or 'unknown',
                'notes': f'user_correction(confidence={confidence}%): {notes.strip()}',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'user_correction',
            }
            try:
                store.add(st.session_state.current_embedding, new_metadata)
                store.save()
                st.success(text['save_success'].format(count=store.count()))
                st.session_state.analysis_result = None
                st.session_state.current_embedding = None
                st.session_state.evidence_rows = []
            except Exception as exc:
                st.error(text['save_failed'].format(error=exc))

st.markdown('---')
st.caption(text['footer'])
