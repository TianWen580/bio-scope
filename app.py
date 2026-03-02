from __future__ import annotations

import base64
import io
import os
from datetime import datetime
from typing import Any

from PIL import Image, ImageDraw
import requests
import streamlit as st

from bioclip_model import encode_image, load_bioclip_model
from small_target_optimizer import detect_and_prepare_crops
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
        'optimize_enable': '大场景小目标优化',
        'use_qwen_locator': '使用 Qwen 二阶段定位',
        'use_yolo_assist': '使用 YOLO 辅助候选框',
        'yolo_model_path': 'YOLO 权重路径',
        'max_crops': '最大裁切数量',
        'localization_title': '定位与裁切调试信息',
        'localization_stage1': '阶段一推理输出',
        'localization_boxes': '融合候选框',
        'localization_errors': '定位告警',
        'mode_status': '识别路径',
        'mode_full': '全图直接识别（未使用定位框）',
        'mode_localized': '定位框增强识别（先定位再裁切识别）',
        'overlay_caption': '定位框渲染图（用于裁切识别）',
        'rendered_boxes': '本次渲染定位框',
        'species_col': '物种',
        'similarity_col': '相似度',
        'crop_col': '裁切ID',
        'source_col': '来源',
        'bbox_col': '框坐标',
        'clues_col': '线索',
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
        'optimize_enable': 'Optimize for small targets in large scenes',
        'use_qwen_locator': 'Use Qwen two-stage localization',
        'use_yolo_assist': 'Use YOLO assistant proposals',
        'yolo_model_path': 'YOLO weight path',
        'max_crops': 'Max crop count',
        'localization_title': 'Localization and Crop Debug Info',
        'localization_stage1': 'Stage-1 reasoning output',
        'localization_boxes': 'Fused candidate boxes',
        'localization_errors': 'Localization warnings',
        'mode_status': 'Recognition path',
        'mode_full': 'Full-image direct recognition (no localization boxes used)',
        'mode_localized': 'Localization-enhanced recognition (boxes -> crops -> recognition)',
        'overlay_caption': 'Rendered localization boxes (used for crop recognition)',
        'rendered_boxes': 'Rendered boxes in this run',
        'species_col': 'Species',
        'similarity_col': 'Similarity',
        'crop_col': 'Crop ID',
        'source_col': 'Source',
        'bbox_col': 'BBox',
        'clues_col': 'Clues',
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
        chunks: list[str] = []
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


def build_prompt(evidence_text: str, language: str, clue_text: str) -> str:
    if language == 'zh':
        return f'''你是一位专业生物学家。请结合以下检索证据和定位线索分析上传的样本图片。

[定位线索]
{clue_text}

[检索证据]
{evidence_text}

[输出要求]
1. 给出 Top-3 候选物种，并提供置信度与依据。
2. 描述关键形态学特征。
3. 如果 Top-1 置信度低于 80%，请明确列出你需要用户补充的关键信息。
4. 给出下一步采样或验证建议。

必须使用简体中文输出，使用清晰分段。'''

    return f'''You are a professional biologist. Analyze the uploaded specimen image using the retrieval evidence and localization clues below.

[Localization Clues]
{clue_text}

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
    return default_language if default_language in {'zh', 'en'} else 'zh'


def fmt_bbox(vals: list[float] | tuple[float, float, float, float]) -> str:
    return ','.join([f'{float(v):.3f}' for v in vals])


def get_render_boxes(localization_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not localization_info:
        return []
    crops = localization_info.get('crops', [])
    out: list[dict[str, Any]] = []
    if isinstance(crops, list):
        for c in crops:
            if not isinstance(c, dict):
                continue
            if c.get('id') == 'full':
                continue
            bbox = c.get('bbox_norm', [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            out.append(c)
    return out


def draw_overlay_boxes(image: Image.Image, boxes: list[dict[str, Any]]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    palette = ['#ff3b30', '#34c759', '#007aff', '#ff9500', '#af52de', '#00c7be']
    line_w = max(2, int(round(min(w, h) * 0.003)))

    for i, box in enumerate(boxes, start=1):
        vals = box.get('bbox_norm', [0.0, 0.0, 1.0, 1.0])
        if not isinstance(vals, list) or len(vals) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in vals]
        except Exception:
            continue

        px1 = int(round(max(0.0, min(1.0, x1)) * w))
        py1 = int(round(max(0.0, min(1.0, y1)) * h))
        px2 = int(round(max(0.0, min(1.0, x2)) * w))
        py2 = int(round(max(0.0, min(1.0, y2)) * h))
        if px2 <= px1 or py2 <= py1:
            continue

        color = palette[(i - 1) % len(palette)]
        draw.rectangle((px1, py1, px2, py2), outline=color, width=line_w)

        label = f"{i}:{box.get('source', 'box')}/{box.get('label', '')}"
        tx = px1 + 4
        ty = max(0, py1 - 14)
        draw.text((tx, ty), label, fill=color)

    return out


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
if 'localization_info' not in st.session_state:
    st.session_state.localization_info = None

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
    env_opt = os.getenv('SMALL_TARGET_OPTIMIZATION', '1').lower() in {'1', 'true', 'yes', 'on'}
    env_qwen = os.getenv('SMALL_TARGET_USE_QWEN', '1').lower() in {'1', 'true', 'yes', 'on'}
    env_yolo = os.getenv('SMALL_TARGET_USE_YOLO', '1').lower() in {'1', 'true', 'yes', 'on'}
    env_yolo_path = os.getenv(
        'YOLO_ASSIST_MODEL_PATH',
        './models/ultralytics/yolov12/best_yolo12_s_动物_1024_randcopybg.pt',
    )
    env_max_crops = int(os.getenv('SMALL_TARGET_MAX_CROPS', '4'))

    api_key = st.text_input(text['api_key_label'], value=env_key, type='password')
    model_name = st.text_input(text['model_label'], value=env_model)
    base_url = st.text_input(text['base_url_label'], value=env_base)
    enable_thinking = st.checkbox(text['thinking_label'], value=env_thinking)

    enable_small_target_opt = st.checkbox(text['optimize_enable'], value=env_opt)
    use_qwen_locator = st.checkbox(text['use_qwen_locator'], value=env_qwen)
    use_yolo_assist = st.checkbox(text['use_yolo_assist'], value=env_yolo)
    yolo_model_path = st.text_input(text['yolo_model_path'], value=env_yolo_path)
    max_crops = st.slider(text['max_crops'], min_value=1, max_value=8, value=max(1, min(8, env_max_crops)))

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
        base64_image = image_to_base64(image)
        localization_info: dict[str, Any] | None = None
        crop_entries = [
            {
                'id': 'full',
                'source': 'full',
                'score': 1.0,
                'label': 'full_scene',
                'clues': [],
                'bbox_norm': (0.0, 0.0, 1.0, 1.0),
                'bbox_px': (0, 0, image.size[0], image.size[1]),
                'image': image,
            }
        ]

        if enable_small_target_opt:
            localization_info = detect_and_prepare_crops(
                image=image,
                image_base64=base64_image,
                image_mime=image_mime,
                language=lang,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                use_qwen_locator=use_qwen_locator,
                use_yolo_assist=use_yolo_assist,
                yolo_model_path=yolo_model_path,
                max_crops=max_crops,
            )
            crop_entries = localization_info.get('crops', crop_entries)

            loc_view = {k: v for k, v in localization_info.items() if k != 'crops'}
            loc_view['crops'] = [
                {
                    'id': c['id'],
                    'source': c['source'],
                    'score': c['score'],
                    'label': c['label'],
                    'bbox_norm': list(c['bbox_norm']),
                    'bbox_px': list(c['bbox_px']),
                    'clues': c['clues'],
                }
                for c in crop_entries
            ]
            loc_view['render_boxes'] = get_render_boxes(loc_view)
            loc_view['recognition_mode'] = 'localized' if loc_view['render_boxes'] else 'full'
            st.session_state.localization_info = loc_view
        else:
            st.session_state.localization_info = {'recognition_mode': 'full', 'render_boxes': []}

        all_hits: list[dict[str, Any]] = []
        best_embedding = None
        best_score = -1.0

        for crop in crop_entries:
            embedding = encode_image(crop['image'], model, preprocess, device)
            search_results = store.search(embedding, top_k=top_k)

            if best_embedding is None:
                best_embedding = embedding
            if search_results and float(search_results[0]['similarity']) > best_score:
                best_score = float(search_results[0]['similarity'])
                best_embedding = embedding

            for item in search_results:
                all_hits.append(
                    {
                        'similarity': float(item['similarity']),
                        'metadata': item['metadata'],
                        'crop': crop,
                    }
                )

        st.session_state.current_embedding = best_embedding

        deduped: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for hit in sorted(all_hits, key=lambda x: x['similarity'], reverse=True):
            meta = hit['metadata']
            key = (str(meta.get('path', '')), str(meta.get('name', '')))
            if key in seen_keys:
                continue
            deduped.append(hit)
            seen_keys.add(key)
            if len(deduped) >= top_k:
                break

        clue_text = 'none'
        if st.session_state.localization_info and st.session_state.localization_info.get('fused_boxes'):
            clue_lines = []
            for idx, box in enumerate(st.session_state.localization_info['fused_boxes'], start=1):
                clue_lines.append(
                    f"{idx}. source={box.get('source','')}, label={box.get('label','')}, "
                    f"bbox={fmt_bbox(box.get('bbox_norm',[0,0,1,1]))}, clues={'|'.join(box.get('clues', []))}"
                )
            clue_text = '\n'.join(clue_lines)

        if deduped:
            evidence_lines = []
            evidence_rows = []
            for i, item in enumerate(deduped, start=1):
                meta = item['metadata']
                score = item['similarity']
                crop = item['crop']
                line = (
                    f"{i}. species={meta.get('name', 'unknown')}, similarity={score:.4f}, "
                    f"crop={crop.get('id', 'full')}, source={crop.get('source', 'full')}, "
                    f"bbox={fmt_bbox(crop.get('bbox_norm', (0, 0, 1, 1)))}, clues={'|'.join(crop.get('clues', []))}, "
                    f"location={meta.get('location', 'unknown')}, notes={meta.get('notes', '')}"
                )
                evidence_lines.append(line)
                evidence_rows.append(
                    {
                        text['species_col']: meta.get('name', 'unknown'),
                        text['similarity_col']: f'{score:.4f}',
                        text['crop_col']: crop.get('id', 'full'),
                        text['source_col']: crop.get('source', 'full'),
                        text['bbox_col']: fmt_bbox(crop.get('bbox_norm', (0, 0, 1, 1))),
                        text['clues_col']: '|'.join(crop.get('clues', [])),
                    }
                )
            evidence_text = '\n'.join(evidence_lines)
            st.session_state.evidence_rows = evidence_rows
        else:
            evidence_text = text['no_samples']
            st.session_state.evidence_rows = []

        prompt = build_prompt(evidence_text, language=lang, clue_text=clue_text)
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

    mode = 'full'
    render_boxes = []
    if st.session_state.localization_info:
        mode = st.session_state.localization_info.get('recognition_mode', 'full')
        render_boxes = st.session_state.localization_info.get('render_boxes', [])

    if mode == 'localized' and render_boxes:
        st.info(f"{text['mode_status']}: {text['mode_localized']}")
        st.image(draw_overlay_boxes(image, render_boxes), caption=text['overlay_caption'], use_container_width=True)
    else:
        st.info(f"{text['mode_status']}: {text['mode_full']}")

    st.subheader(text['analysis_report'])
    st.markdown(st.session_state.analysis_result)

    if st.session_state.evidence_rows:
        st.subheader(text['retrieved_evidence'])
        st.table(st.session_state.evidence_rows)

    if st.session_state.localization_info and st.session_state.localization_info.get('recognition_mode') == 'localized':
        st.subheader(text['localization_title'])
        if st.session_state.localization_info.get('stage1_output'):
            st.text_area(
                text['localization_stage1'],
                value=st.session_state.localization_info['stage1_output'],
                height=180,
            )

        if st.session_state.localization_info.get('fused_boxes'):
            st.write(text['localization_boxes'])
            st.table(st.session_state.localization_info['fused_boxes'])

        if st.session_state.localization_info.get('render_boxes'):
            st.write(text['rendered_boxes'])
            st.table(st.session_state.localization_info['render_boxes'])

        errs = []
        if st.session_state.localization_info.get('qwen_error'):
            errs.append(f"qwen: {st.session_state.localization_info['qwen_error']}")
        if st.session_state.localization_info.get('yolo_error'):
            errs.append(f"yolo: {st.session_state.localization_info['yolo_error']}")
        if errs:
            st.warning(text['localization_errors'] + ': ' + '; '.join(errs))

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
                st.session_state.localization_info = None
            except Exception as exc:
                st.error(text['save_failed'].format(error=exc))

st.markdown('---')
st.caption(text['footer'])
