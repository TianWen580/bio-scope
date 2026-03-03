from __future__ import annotations

import base64
import io
import json
import os
import re
from datetime import datetime
from typing import Any

from PIL import Image, ImageDraw
import requests
import streamlit as st

from bioclip_model import (
    attach_taxonomy_to_species_suggestions,
    encode_image,
    get_tol_taxonomy_constraints,
    load_bioclip_model,
    load_or_export_tol_species_list,
    load_species_taxonomy_map,
    suggest_species_from_embedding,
    suggest_species_with_tol_classifier,
)
from small_target_optimizer import detect_and_prepare_crops, run_interference_analysis_agent
from vector_store import LocalFAISSStore


INDEX_PATH = './data/faiss_index.bin'
METADATA_PATH = './data/faiss_metadata.pkl'
DEFAULT_TOL_SPECIES_TXT = './data/bioclip_tol_species.txt'
DEFAULT_TOL_SPECIES_CSV = './data/bioclip_tol_taxa.csv'
DEFAULT_SPECIES_ALIAS_PATH = './data/species_aliases.json'
TAXONOMY_ORDER = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
CONSTRAINT_ORDER = ['kingdom', 'phylum', 'class', 'order', 'family']

_SPECIES_ALIAS_CACHE: dict[tuple[str, float], dict[str, list[str]]] = {}
DEFAULT_SPECIES_ALIAS_MAP: dict[str, list[str]] = {
    'Lepus sinensis': ['华南兔', '中国野兔', 'Chinese hare', 'South China hare'],
}


LANGUAGE_PACK = {
    'zh': {
        'page_title': 'BioScope Studio - 生物助手 v4.0',
        'title': 'BioScope Studio - BioCLIP + Qwen 多模态生物 Demo v4.0',
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
        'species_list_path_label': 'BioCLIP 物种列表',
        'species_alias_path_label': 'BioCLIP 物种别名',
        'bio_loaded': 'BioCLIP 已加载（设备: {device}）',
        'bio_failed': 'BioCLIP 加载失败: {error}',
        'upload_label': '上传图片',
        'upload_info': '请先上传图片再开始分析。',
        'uploaded_caption': '已上传图片',
        'run_analysis_btn': '开始分析',
        'missing_key': '请先填写 DashScope API Key。',
        'no_samples': '本地样本库为空，将仅基于通用生物知识分析。',
        'analysis_report': '分析报告',
        'analysis_alias_normalized': '报告中的别名/俗名已归一化为标准学名。',
        'retrieved_evidence': '检索证据',
        'bioclip_prior_title': 'BioCLIP 预置分类建议',
        'bioclip_prior_species': '候选物种',
        'bioclip_prior_score': '匹配分数',
        'taxonomy_kingdom': '界',
        'taxonomy_phylum': '门',
        'taxonomy_class': '纲',
        'taxonomy_order': '目',
        'taxonomy_family': '科',
        'taxonomy_genus': '属',
        'taxonomy_species': '种',
        'taxonomy_common_name': '常见名',
        'taxonomy_constraint_title': 'BioCLIP 层级约束',
        'taxonomy_constraint_enabled': '约束已启用（最高约束层级: {rank}）',
        'taxonomy_constraint_disabled': '约束未启用（界置信度低于 {threshold}%）',
        'taxonomy_constraint_scope': '约束范围',
        'taxonomy_constraint_confidence': '层级置信度',
        'taxonomy_constraint_rank': '层级',
        'taxonomy_constraint_label': '标签',
        'taxonomy_constraint_score': '置信度',
        'taxonomy_constraint_error': '层级约束计算失败: {error}',
        'interference_title': '识别干扰因素分析（独立Agent）',
        'interference_route': '分析路线',
        'interference_route_full': '全图分析',
        'interference_route_per_box': '逐框分析',
        'interference_summary': '干扰摘要',
        'interference_recommendations': '改进建议',
        'interference_error': '干扰分析失败: {error}',
        'interference_risk_col': '风险分',
        'interference_factor_col': '干扰因素',
        'interference_severity_col': '严重度',
        'interference_evidence_col': '证据',
        'annotation_title': '结果标注入库',
        'annotation_input_mode_label': '物种输入方式',
        'annotation_input_mode_catalog': '最大名录搜索选择',
        'annotation_input_mode_custom': '自定义录入',
        'annotation_search_label': '搜索名录物种',
        'annotation_search_help': '输入学名片段进行检索，优先前缀匹配。',
        'annotation_catalog_candidates': '名录候选物种',
        'annotation_catalog_none': '名录中未匹配到候选，请修改关键词或切换自定义录入。',
        'annotation_catalog_load_failed': '名录加载失败，已降级为自定义录入: {error}',
        'annotation_catalog_data_warning': '名录扩展数据加载告警: {error}',
        'species_name': '标注物种名',
        'annotation_type_label': '标注类型',
        'annotation_type_confirm': '确认模型结果',
        'annotation_type_corrected': '人工修正后入库',
        'annotator': '标注人',
        'location': '采集地点',
        'confidence': '标注置信度 (%)',
        'notes': '标注备注',
        'save_annotation_btn': '保存标注到向量库',
        'no_embedding': '没有可用向量，请先完成一次分析。',
        'save_success': '标注已保存（{annotation_type}），当前样本数: {count}',
        'save_failed': '保存失败: {error}',
        'footer': '首次使用请先运行 build_index.py 初始化检索库。',
        'request_failed': '请求失败: {error}',
        'optimize_enable': '大场景小目标优化',
        'use_qwen_locator': '使用 Qwen 二阶段定位',
        'use_yolo_assist': '使用 YOLO 辅助候选框',
        'yolo_model_path': 'YOLO 权重路径',
        'max_crops': '最大裁切数量',
        'localization_title': '定位与裁切调试信息',
        'localization_stage1': '阶段一定位假设（未施加分类约束）',
        'localization_stage1_note': '说明：阶段一仅用于定位候选目标，可能与最终约束分类不同。',
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
        'localization_index_col': '序号',
        'localization_label_col': '候选标签',
        'localization_score_col': '置信度',
        'localization_source_col': '来源',
        'localization_bbox_col': '框坐标',
        'localization_clues_col': '线索',
    },
    'en': {
        'page_title': 'BioScope Studio - Bio Assistant v4.0',
        'title': 'BioScope Studio - BioCLIP + Qwen Multimodal Bio Demo v4.0',
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
        'species_list_path_label': 'BioCLIP species list',
        'species_alias_path_label': 'BioCLIP species aliases',
        'bio_loaded': 'BioCLIP loaded on {device}',
        'bio_failed': 'Failed to load BioCLIP: {error}',
        'upload_label': 'Upload image',
        'upload_info': 'Upload an image to start analysis.',
        'uploaded_caption': 'Uploaded image',
        'run_analysis_btn': 'Run analysis',
        'missing_key': 'Please provide DashScope API Key.',
        'no_samples': 'No local samples found. Proceed with generic biological reasoning only.',
        'analysis_report': 'Analysis Report',
        'analysis_alias_normalized': 'Aliases/common names in report were normalized to canonical scientific names.',
        'retrieved_evidence': 'Retrieved Evidence',
        'bioclip_prior_title': 'BioCLIP Prior Suggestions',
        'bioclip_prior_species': 'Candidate species',
        'bioclip_prior_score': 'Match score',
        'taxonomy_kingdom': 'Kingdom',
        'taxonomy_phylum': 'Phylum',
        'taxonomy_class': 'Class',
        'taxonomy_order': 'Order',
        'taxonomy_family': 'Family',
        'taxonomy_genus': 'Genus',
        'taxonomy_species': 'Species',
        'taxonomy_common_name': 'Common name',
        'taxonomy_constraint_title': 'BioCLIP Taxonomy Constraint',
        'taxonomy_constraint_enabled': 'Constraint enabled (deepest enforced rank: {rank})',
        'taxonomy_constraint_disabled': 'Constraint disabled (kingdom confidence below {threshold}%)',
        'taxonomy_constraint_scope': 'Constraint scope',
        'taxonomy_constraint_confidence': 'Rank confidences',
        'taxonomy_constraint_rank': 'Rank',
        'taxonomy_constraint_label': 'Label',
        'taxonomy_constraint_score': 'Confidence',
        'taxonomy_constraint_error': 'Failed to compute taxonomy constraint: {error}',
        'interference_title': 'Recognition Interference Analysis (Independent Agent)',
        'interference_route': 'Analysis route',
        'interference_route_full': 'Full-image analysis',
        'interference_route_per_box': 'Per-box analysis',
        'interference_summary': 'Interference summary',
        'interference_recommendations': 'Mitigation recommendations',
        'interference_error': 'Interference analysis failed: {error}',
        'interference_risk_col': 'Risk',
        'interference_factor_col': 'Factor',
        'interference_severity_col': 'Severity',
        'interference_evidence_col': 'Evidence',
        'annotation_title': 'Annotation to Vector Store',
        'annotation_input_mode_label': 'Species input mode',
        'annotation_input_mode_catalog': 'Search and pick from catalog',
        'annotation_input_mode_custom': 'Custom input',
        'annotation_search_label': 'Search species in catalog',
        'annotation_search_help': 'Type scientific name keywords. Prefix matches are prioritized.',
        'annotation_catalog_candidates': 'Catalog candidates',
        'annotation_catalog_none': 'No catalog candidates found. Refine query or switch to custom input.',
        'annotation_catalog_load_failed': 'Catalog unavailable, fallback to custom input: {error}',
        'annotation_catalog_data_warning': 'Catalog enrichment warning: {error}',
        'species_name': 'Annotated species name',
        'annotation_type_label': 'Annotation type',
        'annotation_type_confirm': 'Confirm model result',
        'annotation_type_corrected': 'Correct and save',
        'annotator': 'Annotator',
        'location': 'Collection location',
        'confidence': 'Annotation confidence (%)',
        'notes': 'Annotation notes',
        'save_annotation_btn': 'Save annotation into vector store',
        'no_embedding': 'No embedding available. Run analysis first.',
        'save_success': 'Annotation saved ({annotation_type}). Current sample count: {count}',
        'save_failed': 'Failed to save annotation: {error}',
        'footer': 'Run build_index.py first to initialize local retrieval data.',
        'request_failed': 'Request failed: {error}',
        'optimize_enable': 'Optimize for small targets in large scenes',
        'use_qwen_locator': 'Use Qwen two-stage localization',
        'use_yolo_assist': 'Use YOLO assistant proposals',
        'yolo_model_path': 'YOLO weight path',
        'max_crops': 'Max crop count',
        'localization_title': 'Localization and Crop Debug Info',
        'localization_stage1': 'Stage-1 localization hypothesis (pre-constraint)',
        'localization_stage1_note': 'Note: Stage-1 is only for candidate localization and may differ from final constrained classification.',
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
        'localization_index_col': 'Index',
        'localization_label_col': 'Candidate label',
        'localization_score_col': 'Confidence',
        'localization_source_col': 'Source',
        'localization_bbox_col': 'BBox',
        'localization_clues_col': 'Clues',
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


def build_prompt(
    evidence_text: str,
    language: str,
    clue_text: str,
    bioclip_prior_text: str,
    taxonomy_constraint_text: str,
    interference_text: str,
) -> str:
    if language == 'zh':
        return f'''你是一位专业生物学家。请结合以下检索证据和定位线索分析上传的样本图片。

[定位线索]
{clue_text}

[BioCLIP 预置分类建议]
{bioclip_prior_text}

[BioCLIP 层级约束（必须遵守）]
{taxonomy_constraint_text}

[识别干扰因素分析]
{interference_text}

[检索证据]
{evidence_text}

[输出要求]
1. 给出 Top-3 候选物种，并提供置信度与依据。
2. 你的 Top-3 必须严格满足上面的 BioCLIP 层级约束；若约束启用则不允许超出约束范围。
3. 描述关键形态学特征。
4. 如果 Top-1 置信度低于 80%，请明确列出你需要用户补充的关键信息。
5. 给出下一步采样或验证建议。

必须使用简体中文输出，使用清晰分段。'''

    return f'''You are a professional biologist. Analyze the uploaded specimen image using the retrieval evidence and localization clues below.

[Localization Clues]
{clue_text}

[BioCLIP Prior Suggestions]
{bioclip_prior_text}

[BioCLIP Taxonomy Constraints (MUST FOLLOW)]
{taxonomy_constraint_text}

[Recognition Interference Analysis]
{interference_text}

[Retrieval Evidence]
{evidence_text}

[Output Requirements]
1. Top-3 candidate species with confidence and rationale.
2. Your Top-3 must strictly satisfy the BioCLIP taxonomy constraints above; if constraints are enabled, do not go outside the allowed scope.
3. Key morphology observations.
4. If top-1 confidence is below 80%, list exact follow-up questions for the user.
5. Next sampling or validation recommendation.

You must answer in English and use clear section headers.'''


def _is_timeout_like_error(message: str) -> bool:
    lowered = message.lower()
    return (
        'http 504' in lowered
        or 'gateway timeout' in lowered
        or 'stream timeout' in lowered
        or 'read timed out' in lowered
        or 'timeout' in lowered
    )


def call_openai_compatible(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    image_base64: str,
    image_mime: str,
    enable_thinking: bool = True,
    request_timeout: int = 1800,
    thinking_budget: int | None = None,
) -> tuple[bool, str]:
    url = base_url.rstrip('/') + '/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    attempts: list[tuple[bool, int | None]] = []
    if enable_thinking:
        attempts.append((True, thinking_budget))
        if thinking_budget is not None and thinking_budget > 2048:
            attempts.append((True, max(2048, thinking_budget // 2)))
        attempts.append((False, None))
    else:
        attempts.append((False, None))

    last_error = 'Unknown request failure'

    for idx, (thinking_flag, budget) in enumerate(attempts):
        payload: dict[str, Any] = {
            'model': model_name,
            'enable_thinking': thinking_flag,
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
        if budget is not None:
            payload['thinking_budget'] = int(budget)

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
        except Exception as exc:
            last_error = f'HTTP request failed: {exc}'
            if idx < len(attempts) - 1 and _is_timeout_like_error(last_error):
                continue
            return False, last_error

        if response.status_code != 200:
            last_error = f'HTTP {response.status_code}: {response.text[:500]}'
            if idx < len(attempts) - 1 and _is_timeout_like_error(last_error):
                continue
            return False, last_error

        try:
            data = response.json()
            content = data['choices'][0]['message'].get('content', '')
            return True, response_to_text(content)
        except Exception as exc:
            last_error = f'Invalid API response format: {exc}'
            return False, last_error

    return False, last_error


def get_default_language() -> str:
    default_language = os.getenv('APP_DEFAULT_LANGUAGE', 'zh').strip().lower()
    return default_language if default_language in {'zh', 'en'} else 'zh'


def get_request_timeout_seconds() -> int:
    raw = os.getenv('DASHSCOPE_TIMEOUT_SECONDS', '1800').strip()
    try:
        timeout = int(raw)
    except ValueError:
        timeout = 1800
    return max(1800, timeout)


def get_thinking_budget_tokens() -> int:
    raw = os.getenv('DASHSCOPE_THINKING_BUDGET', '8192').strip()
    try:
        budget = int(raw)
    except ValueError:
        budget = 8192
    return max(1024, budget)


def _env_flag(name: str, default: bool) -> bool:
    fallback = '1' if default else '0'
    return os.getenv(name, fallback).strip().lower() in {'1', 'true', 'yes', 'on'}


def get_species_list_path() -> str:
    value = os.getenv('BIOCLIP_SPECIES_LIST_PATH', DEFAULT_TOL_SPECIES_TXT).strip()
    return value or DEFAULT_TOL_SPECIES_TXT


def get_species_csv_path() -> str:
    value = os.getenv('BIOCLIP_SPECIES_CSV_PATH', DEFAULT_TOL_SPECIES_CSV).strip()
    return value or DEFAULT_TOL_SPECIES_CSV


def get_species_alias_path() -> str:
    value = os.getenv('BIOCLIP_SPECIES_ALIAS_PATH', DEFAULT_SPECIES_ALIAS_PATH).strip()
    return value or DEFAULT_SPECIES_ALIAS_PATH


def get_species_list_max_labels() -> int:
    raw = os.getenv('BIOCLIP_SPECIES_LIST_MAX_LABELS', '0').strip()
    try:
        value = int(raw)
    except ValueError:
        value = 0
    return max(0, value)


def get_use_tol_classifier() -> bool:
    return _env_flag('BIOCLIP_USE_TOL_CLASSIFIER', True)


def get_auto_export_tol_species() -> bool:
    return _env_flag('BIOCLIP_AUTO_EXPORT_TOL_SPECIES', True)


def get_taxonomy_constraint_threshold() -> float:
    raw = os.getenv('BIOCLIP_TAXONOMY_CONSTRAINT_THRESHOLD', '0.6').strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.6
    return max(0.0, min(1.0, value))


def prior_source_label(source: str, language: str) -> str:
    if language == 'zh':
        if source == 'tol_classifier':
            return 'TreeOfLife 官方分类器'
        if source == 'tol_species_list':
            return 'TreeOfLife 物种列表'
        if source == 'metadata_fallback':
            return '本地样本元数据（降级）'
        return '未命中'

    if source == 'tol_classifier':
        return 'TreeOfLife official classifier'
    if source == 'tol_species_list':
        return 'TreeOfLife species list'
    if source == 'metadata_fallback':
        return 'Local metadata fallback'
    return 'none'


def interference_route_label(route: str, language: str) -> str:
    if language == 'zh':
        return '逐框分析' if route == 'per_box' else '全图分析'
    return 'Per-box analysis' if route == 'per_box' else 'Full-image analysis'


def taxonomy_rank_label(rank: str, language: str) -> str:
    if language == 'zh':
        mapping = {
            'kingdom': '界',
            'phylum': '门',
            'class': '纲',
            'order': '目',
            'family': '科',
            'genus': '属',
            'species': '种',
        }
    else:
        mapping = {
            'kingdom': 'Kingdom',
            'phylum': 'Phylum',
            'class': 'Class',
            'order': 'Order',
            'family': 'Family',
            'genus': 'Genus',
            'species': 'Species',
        }
    return mapping.get(rank, rank)


def interference_factor_label(name: str, language: str) -> str:
    if language != 'zh':
        return name

    mapping = {
        'rare_pose': '少见姿态',
        'occlusion': '遮挡',
        'color_cast': '色度偏差',
        'low_resolution': '低分辨率',
        'defocus_blur': '失焦模糊',
        'motion_blur': '运动模糊',
        'exposure_issue': '曝光异常',
        'tiny_target': '目标过小',
        'background_clutter': '背景干扰',
        'truncation': '目标截断',
        'taxonomy_conflict': '分类约束冲突',
    }
    return mapping.get(name, name)


def localization_source_label(source: str, language: str) -> str:
    if language != 'zh':
        return source
    mapping = {
        'qwen': 'Qwen 定位',
        'yolo': 'YOLO 辅助',
        'fusion': '融合结果',
        'full': '全图',
    }
    return mapping.get(source, source)


def severity_label(severity: str, language: str) -> str:
    level = str(severity).strip().lower()
    if language != 'zh':
        return level or 'unknown'
    mapping = {
        'low': '低',
        'medium': '中',
        'high': '高',
    }
    return mapping.get(level, level or '未知')


def runtime_warning(label_zh: str, label_en: str, error: str, language: str) -> str:
    prefix = label_zh if language == 'zh' else label_en
    return f"{prefix}: {error}"


def candidate_label_text(label: str, language: str) -> str:
    raw = str(label).strip()
    if language != 'zh':
        return raw
    if raw.startswith('candidate_'):
        suffix = raw.split('candidate_', 1)[1]
        if suffix:
            return f'候选目标{suffix}'
        return '候选目标'
    if raw == 'full_scene':
        return '全图目标'
    return raw


def fmt_bbox(vals: list[float] | tuple[float, float, float, float]) -> str:
    return ','.join([f'{float(v):.3f}' for v in vals])


def collect_species_labels(metadata_rows: list[dict[str, Any]], max_labels: int = 2048) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for meta in metadata_rows:
        name = str(meta.get('name', '')).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        labels.append(name)
        if len(labels) >= max_labels:
            break
    return labels


def _clean_alias_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ''
    lowered = text.lower()
    if lowered in {'none', 'nan', 'null'}:
        return ''
    return text


def load_species_alias_map(alias_path: str) -> tuple[dict[str, list[str]], str | None]:
    merged: dict[str, list[str]] = {k: list(v) for k, v in DEFAULT_SPECIES_ALIAS_MAP.items()}
    alias_file = os.path.abspath(alias_path)
    if not os.path.exists(alias_file):
        return merged, None

    try:
        mtime = os.path.getmtime(alias_file)
        cache_key = (alias_file, mtime)
        if cache_key in _SPECIES_ALIAS_CACHE:
            cached = _SPECIES_ALIAS_CACHE[cache_key]
            out = {k: list(v) for k, v in merged.items()}
            for species, aliases in cached.items():
                out.setdefault(species, [])
                out[species].extend([a for a in aliases if a not in out[species]])
            return out, None

        with open(alias_file, 'r', encoding='utf-8') as f:
            payload = json.load(f)
    except Exception as exc:
        return merged, f'alias file load failed: {exc}'

    loaded: dict[str, list[str]] = {}

    if isinstance(payload, dict):
        for species, aliases in payload.items():
            species_key = _clean_alias_text(species)
            if not species_key:
                continue
            alias_values: list[str] = []
            if isinstance(aliases, str):
                val = _clean_alias_text(aliases)
                if val:
                    alias_values.append(val)
            elif isinstance(aliases, list):
                for item in aliases:
                    val = _clean_alias_text(item)
                    if val and val not in alias_values:
                        alias_values.append(val)
            if alias_values:
                loaded[species_key] = alias_values

    for species, aliases in loaded.items():
        merged.setdefault(species, [])
        for alias in aliases:
            if alias not in merged[species]:
                merged[species].append(alias)

    _SPECIES_ALIAS_CACHE[cache_key] = loaded
    return merged, None


def build_species_search_records(
    species_labels: list[str],
    species_csv_path: str,
    alias_path: str,
) -> tuple[list[dict[str, Any]], str | None]:
    taxonomy_map, taxonomy_error = load_species_taxonomy_map(species_csv_path)
    alias_map, alias_error = load_species_alias_map(alias_path)

    warnings: list[str] = []
    if taxonomy_error:
        warnings.append(taxonomy_error)
    if alias_error:
        warnings.append(alias_error)

    out: list[dict[str, Any]] = []
    for species in species_labels:
        clean_species = _clean_alias_text(species)
        if not clean_species:
            continue
        taxonomy = taxonomy_map.get(clean_species, {}) if isinstance(taxonomy_map, dict) else {}
        common_name = _clean_alias_text(taxonomy.get('common_name', ''))
        aliases = [x for x in alias_map.get(clean_species, []) if _clean_alias_text(x)]

        search_terms: list[str] = [clean_species]
        if common_name:
            search_terms.append(common_name)
        search_terms.extend(aliases)

        searchable = ' '.join([x.lower() for x in search_terms if x]).strip()
        display = clean_species if not common_name else f'{clean_species} ({common_name})'
        out.append(
            {
                'species': clean_species,
                'common_name': common_name,
                'aliases': aliases,
                'display': display,
                'searchable': searchable,
            }
        )

    warning_text = '; '.join(warnings[:2]) if warnings else None
    return out, warning_text


def search_species_candidates(records: list[dict[str, Any]], query: str, limit: int = 30) -> list[dict[str, Any]]:
    if not records:
        return []

    max_items = max(1, min(200, int(limit)))
    q = query.strip().lower()
    if not q:
        return records[: max_items]

    scored: list[tuple[int, dict[str, Any]]] = []
    for rec in records:
        species = str(rec.get('species', '')).lower()
        common_name = str(rec.get('common_name', '')).lower()
        aliases = [str(x).lower() for x in rec.get('aliases', []) if str(x).strip()]
        searchable = str(rec.get('searchable', '')).lower()

        score = -1
        if species.startswith(q):
            score = 100
        elif common_name.startswith(q):
            score = 92
        elif any(alias.startswith(q) for alias in aliases):
            score = 90
        elif len(q) >= 2 and q in species:
            score = 80
        elif len(q) >= 2 and q in common_name:
            score = 74
        elif len(q) >= 2 and any(q in alias for alias in aliases):
            score = 72
        elif len(q) >= 3 and q in searchable:
            score = 65

        if score >= 0:
            scored.append((score, rec))

    scored.sort(key=lambda x: (-x[0], str(x[1].get('species', ''))))
    return [rec for _, rec in scored[:max_items]]


def build_species_alias_lookup(
    records: list[dict[str, Any]],
    alias_map: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    out: dict[str, str] = {}

    for rec in records:
        species = _clean_alias_text(rec.get('species', ''))
        if not species:
            continue

        variants: list[str] = [species]
        common_name = _clean_alias_text(rec.get('common_name', ''))
        if common_name:
            variants.append(common_name)

        for alias in rec.get('aliases', []):
            clean_alias = _clean_alias_text(alias)
            if clean_alias:
                variants.append(clean_alias)

        for variant in variants:
            out.setdefault(variant.lower(), species)

    if isinstance(alias_map, dict):
        for species, aliases in alias_map.items():
            clean_species = _clean_alias_text(species)
            if not clean_species:
                continue
            out.setdefault(clean_species.lower(), clean_species)
            if not isinstance(aliases, list):
                continue
            for alias in aliases:
                clean_alias = _clean_alias_text(alias)
                if clean_alias:
                    out.setdefault(clean_alias.lower(), clean_species)

    return out


def canonicalize_species_name(name: str, alias_lookup: dict[str, str]) -> str:
    clean_name = _clean_alias_text(name)
    if not clean_name:
        return ''
    return alias_lookup.get(clean_name.lower(), clean_name)


def build_analysis_alias_replacements(
    bioclip_suggestions: list[dict[str, Any]],
    alias_map: dict[str, list[str]],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()

    for item in bioclip_suggestions:
        species = _clean_alias_text(item.get('species', ''))
        if not species:
            continue

        common_name = _clean_alias_text(item.get('common_name', ''))
        if common_name and common_name.lower() not in seen and common_name.lower() != species.lower():
            pairs.append((common_name, species))
            seen.add(common_name.lower())

        for alias in alias_map.get(species, []):
            clean_alias = _clean_alias_text(alias)
            if not clean_alias:
                continue
            lowered = clean_alias.lower()
            if lowered in seen or lowered == species.lower():
                continue
            pairs.append((clean_alias, species))
            seen.add(lowered)

    pairs.sort(key=lambda x: len(x[0]), reverse=True)
    return pairs


def normalize_analysis_text_species_names(
    report_text: str,
    replacements: list[tuple[str, str]],
) -> tuple[str, bool]:
    if not report_text or not replacements:
        return report_text, False

    normalized = report_text
    changed = False
    for alias, species in replacements:
        if not alias or not species:
            continue

        if alias.isascii():
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b', flags=re.IGNORECASE)
            new_text, count = pattern.subn(species, normalized)
            if count > 0:
                normalized = new_text
                changed = True
        else:
            if alias in normalized:
                normalized = normalized.replace(alias, species)
                changed = True

    return normalized, changed


def format_bioclip_prior_text(priors: list[dict[str, Any]], language: str) -> str:
    if not priors:
        return 'none'
    lines: list[str] = []
    for idx, item in enumerate(priors, start=1):
        species = str(item.get('species', 'unknown'))
        score = float(item.get('score', 0.0))
        taxonomy_bits: list[str] = []
        for rank in TAXONOMY_ORDER[:-1]:
            val = str(item.get(rank, '')).strip()
            if not val:
                continue
            taxonomy_bits.append(f"{taxonomy_rank_label(rank, language)}={val}")
        common_name = str(item.get('common_name', '')).strip()
        if common_name:
            if language == 'zh':
                taxonomy_bits.append(f"常见名={common_name}")
            else:
                taxonomy_bits.append(f"common_name={common_name}")
        taxonomy_text = '; '.join(taxonomy_bits) if taxonomy_bits else 'none'
        if language == 'zh':
            lines.append(f"{idx}. 种={species}, 匹配分数={score:.4f}, 层级参考={taxonomy_text}")
        else:
            lines.append(f"{idx}. species={species}, match_score={score:.4f}, taxonomy={taxonomy_text}")
    return '\n'.join(lines)


def format_taxonomy_constraint_text(constraint_info: dict[str, Any] | None, language: str) -> str:
    if not constraint_info:
        return 'none'

    threshold = float(constraint_info.get('threshold', 0.6))
    threshold_pct = threshold * 100.0
    rank_scores = constraint_info.get('rank_scores', {})
    score_parts: list[str] = []
    if isinstance(rank_scores, dict):
        for rank in CONSTRAINT_ORDER:
            if rank in rank_scores:
                rank_name = taxonomy_rank_label(rank, language)
                score_parts.append(f"{rank_name}={float(rank_scores[rank]):.3f}")

    if not constraint_info.get('enabled'):
        if language == 'zh':
            return (
                f"约束启用=否\n"
                f"原因=界层级置信度低于 {threshold_pct:.0f}%\n"
                f"层级置信度={'；'.join(score_parts)}\n"
                "规则=本轮不施加 BioCLIP 层级限制。"
            )
        return (
            f"constraint_enabled=0\n"
            f"reason=kingdom_score_below_{threshold_pct:.0f}%\n"
            f"rank_scores={' ; '.join(score_parts)}\n"
            "rule=No BioCLIP taxonomy constraint is enforced in this run."
        )

    active_rank = str(constraint_info.get('active_rank', 'kingdom'))
    active_taxonomy = constraint_info.get('active_taxonomy', {})
    scope_parts: list[str] = []
    if isinstance(active_taxonomy, dict):
        for rank in CONSTRAINT_ORDER:
            if rank not in active_taxonomy:
                continue
            val = str(active_taxonomy.get(rank, '')).strip()
            if val:
                rank_name = taxonomy_rank_label(rank, language)
                scope_parts.append(f"{rank_name}={val}")

    if language == 'zh':
        return (
            f"约束启用=是\n"
            f"阈值={threshold_pct:.0f}%\n"
            f"最高约束层级={taxonomy_rank_label(active_rank, language)}\n"
            f"允许范围={'；'.join(scope_parts)}\n"
            f"层级置信度={'；'.join(score_parts)}\n"
            "硬约束=Top-1/Top-2/Top-3 必须落在允许范围的下级范围内，禁止超出。"
        )

    return (
        f"constraint_enabled=1\n"
        f"threshold={threshold_pct:.0f}%\n"
        f"active_rank={active_rank}\n"
        f"allowed_scope={' ; '.join(scope_parts)}\n"
        f"rank_scores={' ; '.join(score_parts)}\n"
        "hard_rule=Top-1/Top-2/Top-3 must stay within the descendants of allowed_scope and must not go outside it."
    )


def format_interference_text(interference_info: dict[str, Any] | None, language: str) -> str:
    if not interference_info:
        return 'none'

    if interference_info.get('error'):
        return f"unavailable: {interference_info.get('error')}"

    route = str(interference_info.get('route', 'full_image'))
    analysis_json = interference_info.get('analysis_json')
    if not isinstance(analysis_json, dict):
        return 'none'

    out: list[str] = []
    if language == 'zh':
        out.append(f"分析路线={interference_route_label(route, language)}")
    else:
        out.append(f"route={route}")

    summary = str(analysis_json.get('global_summary', '')).strip()
    if summary:
        if language == 'zh':
            out.append(f"摘要={summary}")
        else:
            out.append(f"summary={summary}")

    targets = analysis_json.get('targets', [])
    if isinstance(targets, list):
        for idx, item in enumerate(targets, start=1):
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', 'unknown'))
            risk_score = item.get('risk_score', 'n/a')
            factors = item.get('factors', [])
            factor_parts: list[str] = []
            if isinstance(factors, list):
                for f in factors:
                    if not isinstance(f, dict):
                        continue
                    name = str(f.get('name', 'unknown'))
                    sev = str(f.get('severity', 'low'))
                    factor_name = interference_factor_label(name, language)
                    factor_parts.append(f"{factor_name}:{severity_label(sev, language)}")
            joined = ','.join(factor_parts)
            if language == 'zh':
                out.append(f"目标{idx}={label}，风险分={risk_score}，干扰因素={joined}")
            else:
                out.append(f"target_{idx}={label},risk={risk_score},factors={joined}")

    recs = analysis_json.get('recommendations', [])
    if isinstance(recs, list) and recs:
        rec_text = ' | '.join([str(x) for x in recs[:5]])
        if language == 'zh':
            out.append(f"建议={rec_text}")
        else:
            out.append(f"recommendations={rec_text}")

    return '\n'.join(out) if out else 'none'


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
if 'analysis_alias_normalized' not in st.session_state:
    st.session_state.analysis_alias_normalized = False
if 'current_embedding' not in st.session_state:
    st.session_state.current_embedding = None
if 'evidence_rows' not in st.session_state:
    st.session_state.evidence_rows = []
if 'localization_info' not in st.session_state:
    st.session_state.localization_info = None
if 'bioclip_suggestions' not in st.session_state:
    st.session_state.bioclip_suggestions = []
if 'bioclip_prior_source' not in st.session_state:
    st.session_state.bioclip_prior_source = 'none'
if 'bioclip_prior_warning' not in st.session_state:
    st.session_state.bioclip_prior_warning = ''
if 'taxonomy_constraints' not in st.session_state:
    st.session_state.taxonomy_constraints = None
if 'taxonomy_constraint_warning' not in st.session_state:
    st.session_state.taxonomy_constraint_warning = ''
if 'interference_info' not in st.session_state:
    st.session_state.interference_info = None

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
    request_timeout = get_request_timeout_seconds()
    thinking_budget = get_thinking_budget_tokens()
    species_list_path = get_species_list_path()
    species_csv_path = get_species_csv_path()
    species_alias_path = get_species_alias_path()
    species_list_max_labels = get_species_list_max_labels()
    use_tol_classifier = get_use_tol_classifier()
    auto_export_tol_species = get_auto_export_tol_species()
    taxonomy_constraint_threshold = get_taxonomy_constraint_threshold()

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
    st.write(f"{text['species_list_path_label']}:", species_list_path)
    st.write(f"{text['species_alias_path_label']}:", species_alias_path)

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
    st.session_state.analysis_alias_normalized = False
    st.session_state.interference_info = None
    st.session_state.taxonomy_constraints = None
    st.session_state.taxonomy_constraint_warning = ''
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
        best_crop_image: Image.Image | None = None
        best_score = -1.0

        for crop in crop_entries:
            embedding = encode_image(crop['image'], model, preprocess, device)
            search_results = store.search(embedding, top_k=top_k)

            if best_embedding is None:
                best_embedding = embedding
                best_crop_image = crop['image']
            if search_results and float(search_results[0]['similarity']) > best_score:
                best_score = float(search_results[0]['similarity'])
                best_embedding = embedding
                best_crop_image = crop['image']

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

        bioclip_suggestions: list[dict[str, Any]] = []
        bioclip_prior_source = 'none'
        prior_warnings: list[str] = []

        if best_embedding is not None:
            if use_tol_classifier and best_crop_image is not None:
                tol_suggestions, tol_error = suggest_species_with_tol_classifier(
                    image=best_crop_image,
                    top_n=5,
                    device=device,
                )
                if tol_suggestions:
                    bioclip_suggestions = tol_suggestions
                    bioclip_prior_source = 'tol_classifier'
                elif tol_error:
                    prior_warnings.append(runtime_warning('ToL 分类器', 'ToL classifier', tol_error, lang))

            if not bioclip_suggestions:
                species_labels, species_error = load_or_export_tol_species_list(
                    species_txt_path=species_list_path,
                    species_csv_path=species_csv_path,
                    max_labels=species_list_max_labels,
                    auto_export=auto_export_tol_species,
                    device=device,
                )
                if species_labels:
                    try:
                        bioclip_suggestions = suggest_species_from_embedding(
                            image_embedding=best_embedding,
                            species_labels=species_labels,
                            model=model,
                            device=device,
                            top_n=min(5, len(species_labels)),
                        )
                        if bioclip_suggestions:
                            bioclip_prior_source = 'tol_species_list'
                    except Exception as exc:
                        prior_warnings.append(
                            runtime_warning('物种列表评分失败', 'Species-list scoring failed', str(exc), lang)
                        )
                elif species_error:
                    prior_warnings.append(runtime_warning('物种列表加载失败', 'Species-list load failed', species_error, lang))

            if not bioclip_suggestions:
                species_labels = collect_species_labels(store.metadata)
                if species_labels:
                    try:
                        bioclip_suggestions = suggest_species_from_embedding(
                            image_embedding=best_embedding,
                            species_labels=species_labels,
                            model=model,
                            device=device,
                            top_n=min(5, len(species_labels)),
                        )
                        if bioclip_suggestions:
                            bioclip_prior_source = 'metadata_fallback'
                    except Exception as exc:
                        prior_warnings.append(
                            runtime_warning('元数据回退失败', 'Metadata fallback failed', str(exc), lang)
                        )

        if bioclip_suggestions:
            bioclip_suggestions, taxonomy_attach_error = attach_taxonomy_to_species_suggestions(
                bioclip_suggestions,
                species_csv_path=species_csv_path,
            )
            if taxonomy_attach_error:
                prior_warnings.append(
                    runtime_warning('层级补全失败', 'Taxonomy attach failed', taxonomy_attach_error, lang)
                )

        taxonomy_constraint_info: dict[str, Any] | None = None
        taxonomy_constraint_error: str | None = None
        if best_crop_image is not None:
            taxonomy_constraint_info, taxonomy_constraint_error = get_tol_taxonomy_constraints(
                image=best_crop_image,
                threshold=taxonomy_constraint_threshold,
                device=device,
            )
            if taxonomy_constraint_error:
                prior_warnings.append(
                    runtime_warning('层级约束计算失败', 'Taxonomy constraint failed', taxonomy_constraint_error, lang)
                )
        else:
            taxonomy_constraint_error = (
                '层级约束跳过：缺少可用目标裁切图像'
                if lang == 'zh'
                else 'taxonomy constraint skipped: missing best crop image'
            )

        st.session_state.bioclip_suggestions = bioclip_suggestions
        st.session_state.bioclip_prior_source = bioclip_prior_source
        st.session_state.taxonomy_constraints = taxonomy_constraint_info
        st.session_state.taxonomy_constraint_warning = taxonomy_constraint_error or ''
        st.session_state.bioclip_prior_warning = '; '.join(prior_warnings[:2]) if prior_warnings else ''
        bioclip_prior_text = format_bioclip_prior_text(bioclip_suggestions, lang)
        taxonomy_constraint_text = format_taxonomy_constraint_text(taxonomy_constraint_info, lang)

        runtime_alias_map, runtime_alias_error = load_species_alias_map(species_alias_path)
        if runtime_alias_error:
            prior_warnings.append(
                runtime_warning('分析别名映射失败', 'Analysis alias map failed', runtime_alias_error, lang)
            )
        analysis_alias_replacements = build_analysis_alias_replacements(
            bioclip_suggestions,
            runtime_alias_map,
        )
        st.session_state.bioclip_prior_warning = '; '.join(prior_warnings[:2]) if prior_warnings else ''

        interference_info = run_interference_analysis_agent(
            image_base64=base64_image,
            image_mime=image_mime,
            language=lang,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            localization_info=st.session_state.localization_info,
            bioclip_prior_text=bioclip_prior_text,
            taxonomy_constraint_text=taxonomy_constraint_text,
            enable_thinking=enable_thinking,
        )
        st.session_state.interference_info = interference_info
        interference_text = format_interference_text(interference_info, lang)

        clue_text = 'none'
        if st.session_state.localization_info and st.session_state.localization_info.get('fused_boxes'):
            clue_lines = []
            for idx, box in enumerate(st.session_state.localization_info['fused_boxes'], start=1):
                if lang == 'zh':
                    clue_lines.append(
                        f"{idx}. 来源={localization_source_label(str(box.get('source','')), lang)}，"
                        f"候选标签={candidate_label_text(str(box.get('label','')), lang)}，"
                        f"框坐标={fmt_bbox(box.get('bbox_norm',[0,0,1,1]))}，"
                        f"线索={'|'.join(box.get('clues', []))}"
                    )
                else:
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
                        text['source_col']: localization_source_label(crop.get('source', 'full'), lang),
                        text['bbox_col']: fmt_bbox(crop.get('bbox_norm', (0, 0, 1, 1))),
                        text['clues_col']: '|'.join(crop.get('clues', [])),
                    }
                )
            evidence_text = '\n'.join(evidence_lines)
            st.session_state.evidence_rows = evidence_rows
        else:
            evidence_text = text['no_samples']
            st.session_state.evidence_rows = []

        prompt = build_prompt(
            evidence_text,
            language=lang,
            clue_text=clue_text,
            bioclip_prior_text=bioclip_prior_text,
            taxonomy_constraint_text=taxonomy_constraint_text,
            interference_text=interference_text,
        )
        ok, result = call_openai_compatible(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            prompt=prompt,
            image_base64=base64_image,
            image_mime=image_mime,
            enable_thinking=enable_thinking,
            request_timeout=request_timeout,
            thinking_budget=thinking_budget,
        )
        if ok:
            normalized_result, normalized_changed = normalize_analysis_text_species_names(
                result,
                analysis_alias_replacements,
            )
            st.session_state.analysis_result = normalized_result
            st.session_state.analysis_alias_normalized = normalized_changed
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
    if st.session_state.analysis_alias_normalized:
        st.caption(text['analysis_alias_normalized'])
    st.markdown(st.session_state.analysis_result)

    if st.session_state.evidence_rows:
        st.subheader(text['retrieved_evidence'])
        st.table(st.session_state.evidence_rows)

    if st.session_state.bioclip_suggestions:
        st.subheader(text['bioclip_prior_title'])
        source_text = prior_source_label(st.session_state.bioclip_prior_source, lang)
        prior_rows: list[dict[str, str]] = []
        for item in st.session_state.bioclip_suggestions:
            row = {
                text['bioclip_prior_species']: str(item.get('species', 'unknown')),
                text['bioclip_prior_score']: f"{float(item.get('score', 0.0)):.4f}",
                text['taxonomy_kingdom']: str(item.get('kingdom', '')),
                text['taxonomy_phylum']: str(item.get('phylum', '')),
                text['taxonomy_class']: str(item.get('class', '')),
                text['taxonomy_order']: str(item.get('order', '')),
                text['taxonomy_family']: str(item.get('family', '')),
                text['taxonomy_genus']: str(item.get('genus', '')),
                text['source_col']: source_text,
            }
            common_name = str(item.get('common_name', '')).strip()
            if common_name:
                row[text['taxonomy_common_name']] = common_name
            prior_rows.append(row)
        st.table(prior_rows)

    if st.session_state.taxonomy_constraints:
        st.subheader(text['taxonomy_constraint_title'])
        constraint_info = st.session_state.taxonomy_constraints
        threshold = float(constraint_info.get('threshold', taxonomy_constraint_threshold))
        threshold_pct = threshold * 100.0

        if constraint_info.get('enabled'):
            active_rank = str(constraint_info.get('active_rank', 'kingdom'))
            st.info(text['taxonomy_constraint_enabled'].format(rank=taxonomy_rank_label(active_rank, lang)))

            active_taxonomy = constraint_info.get('active_taxonomy', {})
            if isinstance(active_taxonomy, dict) and active_taxonomy:
                scope_rows: list[dict[str, str]] = []
                for rank in CONSTRAINT_ORDER:
                    val = str(active_taxonomy.get(rank, '')).strip()
                    if not val:
                        continue
                    scope_rows.append(
                        {
                            text['taxonomy_constraint_rank']: taxonomy_rank_label(rank, lang),
                            text['taxonomy_constraint_label']: val,
                        }
                    )
                if scope_rows:
                    st.write(text['taxonomy_constraint_scope'])
                    st.table(scope_rows)
        else:
            st.info(text['taxonomy_constraint_disabled'].format(threshold=f'{threshold_pct:.0f}'))

        rank_rows: list[dict[str, str]] = []
        rank_predictions = constraint_info.get('rank_predictions', [])
        if isinstance(rank_predictions, list):
            for item in rank_predictions:
                if not isinstance(item, dict):
                    continue
                rank = str(item.get('rank', ''))
                label = str(item.get('label', ''))
                score = float(item.get('score', 0.0))
                rank_rows.append(
                    {
                        text['taxonomy_constraint_rank']: taxonomy_rank_label(rank, lang),
                        text['taxonomy_constraint_label']: label,
                        text['taxonomy_constraint_score']: f'{score:.4f}',
                    }
                )
        if rank_rows:
            st.write(text['taxonomy_constraint_confidence'])
            st.table(rank_rows)

    if st.session_state.taxonomy_constraint_warning:
        st.warning(text['taxonomy_constraint_error'].format(error=st.session_state.taxonomy_constraint_warning))

    if st.session_state.interference_info:
        st.subheader(text['interference_title'])
        interference_info = st.session_state.interference_info
        route = str(interference_info.get('route', 'full_image'))
        st.write(f"{text['interference_route']}: {interference_route_label(route, lang)}")

        if interference_info.get('error'):
            st.warning(text['interference_error'].format(error=interference_info['error']))
        else:
            analysis_json = interference_info.get('analysis_json')
            if isinstance(analysis_json, dict):
                summary = str(analysis_json.get('global_summary', '')).strip()
                if summary:
                    st.write(f"{text['interference_summary']}: {summary}")

                recommendations = analysis_json.get('recommendations', [])
                if isinstance(recommendations, list) and recommendations:
                    st.write(text['interference_recommendations'])
                    st.table([{'-': str(x)} for x in recommendations])

                targets = analysis_json.get('targets', [])
                factor_rows: list[dict[str, str]] = []
                if isinstance(targets, list):
                    for t in targets:
                        if not isinstance(t, dict):
                            continue
                        t_label = str(t.get('label', 'unknown'))
                        t_risk = str(t.get('risk_score', 'n/a'))
                        factors = t.get('factors', [])
                        if isinstance(factors, list):
                            for f in factors:
                                if not isinstance(f, dict):
                                    continue
                                factor_rows.append(
                                    {
                                        text['species_col']: candidate_label_text(t_label, lang),
                                        text['interference_risk_col']: t_risk,
                                        text['interference_factor_col']: interference_factor_label(
                                            str(f.get('name', 'unknown')),
                                            lang,
                                        ),
                                        text['interference_severity_col']: severity_label(
                                            str(f.get('severity', 'low')),
                                            lang,
                                        ),
                                        text['interference_evidence_col']: str(f.get('evidence', '')),
                                    }
                                )
                if factor_rows:
                    st.table(factor_rows)

    if st.session_state.bioclip_prior_warning:
        st.warning(st.session_state.bioclip_prior_warning)

    if st.session_state.localization_info and st.session_state.localization_info.get('recognition_mode') == 'localized':
        st.subheader(text['localization_title'])
        if st.session_state.localization_info.get('stage1_output'):
            st.caption(text['localization_stage1_note'])
            st.text_area(
                text['localization_stage1'],
                value=st.session_state.localization_info['stage1_output'],
                height=180,
            )

        if st.session_state.localization_info.get('fused_boxes'):
            st.write(text['localization_boxes'])
            fused_rows: list[dict[str, str]] = []
            for idx, box in enumerate(st.session_state.localization_info['fused_boxes'], start=1):
                if not isinstance(box, dict):
                    continue
                fused_rows.append(
                    {
                        text['localization_index_col']: str(idx),
                        text['localization_label_col']: candidate_label_text(str(box.get('label', 'candidate')), lang),
                        text['localization_score_col']: f"{float(box.get('score', 0.0)):.4f}",
                        text['localization_source_col']: localization_source_label(str(box.get('source', 'fusion')), lang),
                        text['localization_bbox_col']: fmt_bbox(box.get('bbox_norm', [0, 0, 1, 1])),
                        text['localization_clues_col']: '|'.join([str(x) for x in box.get('clues', [])]),
                    }
                )
            if fused_rows:
                st.table(fused_rows)

        if st.session_state.localization_info.get('render_boxes'):
            st.write(text['rendered_boxes'])
            render_rows: list[dict[str, str]] = []
            for idx, box in enumerate(st.session_state.localization_info['render_boxes'], start=1):
                if not isinstance(box, dict):
                    continue
                render_rows.append(
                    {
                        text['localization_index_col']: str(idx),
                        text['localization_label_col']: candidate_label_text(str(box.get('label', 'candidate')), lang),
                        text['localization_score_col']: f"{float(box.get('score', 0.0)):.4f}",
                        text['localization_source_col']: localization_source_label(str(box.get('source', 'fusion')), lang),
                        text['localization_bbox_col']: fmt_bbox(box.get('bbox_norm', [0, 0, 1, 1])),
                        text['localization_clues_col']: '|'.join([str(x) for x in box.get('clues', [])]),
                    }
                )
            if render_rows:
                st.table(render_rows)

        errs = []
        if st.session_state.localization_info.get('qwen_error'):
            prefix = 'Qwen' if lang == 'en' else 'Qwen 定位'
            errs.append(f"{prefix}: {st.session_state.localization_info['qwen_error']}")
        if st.session_state.localization_info.get('yolo_error'):
            prefix = 'YOLO' if lang == 'en' else 'YOLO 辅助'
            errs.append(f"{prefix}: {st.session_state.localization_info['yolo_error']}")
        if errs:
            st.warning(text['localization_errors'] + ': ' + '; '.join(errs))

    st.markdown('---')
    st.subheader(text['annotation_title'])

    suggested_species = 'unknown_species'
    if st.session_state.bioclip_suggestions:
        suggested_species = str(st.session_state.bioclip_suggestions[0].get('species', 'unknown_species'))
    elif st.session_state.evidence_rows:
        suggested_species = str(st.session_state.evidence_rows[0].get(text['species_col'], 'unknown_species'))

    annotation_catalog_labels, annotation_catalog_error = load_or_export_tol_species_list(
        species_txt_path=species_list_path,
        species_csv_path=species_csv_path,
        max_labels=species_list_max_labels,
        auto_export=auto_export_tol_species,
        device=device,
    )
    catalog_available = bool(annotation_catalog_labels)
    search_records: list[dict[str, Any]] = []
    catalog_data_warning: str | None = None

    if catalog_available:
        search_records, catalog_data_warning = build_species_search_records(
            annotation_catalog_labels,
            species_csv_path=species_csv_path,
            alias_path=species_alias_path,
        )
        if not search_records:
            catalog_available = False

    if annotation_catalog_error and not catalog_available:
        st.warning(text['annotation_catalog_load_failed'].format(error=annotation_catalog_error))
    elif catalog_data_warning:
        st.caption(text['annotation_catalog_data_warning'].format(error=catalog_data_warning))

    annotation_alias_map, annotation_alias_error = load_species_alias_map(species_alias_path)
    if annotation_alias_error and not catalog_data_warning:
        st.caption(text['annotation_catalog_data_warning'].format(error=annotation_alias_error))
    annotation_alias_lookup = build_species_alias_lookup(search_records, annotation_alias_map)

    with st.form('annotation_form'):
        mode_options = [
            text['annotation_input_mode_catalog'],
            text['annotation_input_mode_custom'],
        ]
        default_mode_index = 0 if catalog_available else 1
        input_mode = st.radio(
            text['annotation_input_mode_label'],
            options=mode_options,
            index=default_mode_index,
            horizontal=True,
        )

        if input_mode == text['annotation_input_mode_catalog'] and catalog_available:
            search_seed = suggested_species if suggested_species != 'unknown_species' else ''
            search_query = st.text_input(
                text['annotation_search_label'],
                value=search_seed,
                help=text['annotation_search_help'],
            )
            candidates = search_species_candidates(search_records, search_query, limit=30)

            if suggested_species in annotation_catalog_labels and not any(
                str(x.get('species', '')) == suggested_species for x in candidates
            ):
                suggest_row = next(
                    (x for x in search_records if str(x.get('species', '')) == suggested_species),
                    {
                        'species': suggested_species,
                        'common_name': '',
                        'aliases': [],
                        'display': suggested_species,
                        'searchable': suggested_species.lower(),
                    },
                )
                candidates = [suggest_row] + candidates

            if candidates:
                default_candidate_index = 0
                for idx, rec in enumerate(candidates):
                    if str(rec.get('species', '')) == suggested_species:
                        default_candidate_index = idx
                        break

                option_labels = [str(rec.get('display', rec.get('species', 'unknown'))) for rec in candidates]
                annotated_name = st.selectbox(
                    text['annotation_catalog_candidates'],
                    options=option_labels,
                    index=default_candidate_index,
                )
                display_to_species = {
                    str(rec.get('display', rec.get('species', 'unknown'))): str(rec.get('species', 'unknown'))
                    for rec in candidates
                }
                annotated_name = display_to_species.get(annotated_name, suggested_species)
            else:
                st.caption(text['annotation_catalog_none'])
                annotated_name = st.text_input(text['species_name'], value=suggested_species)
        else:
            annotated_name = st.text_input(text['species_name'], value=suggested_species)

        annotation_type_label = st.selectbox(
            text['annotation_type_label'],
            options=[text['annotation_type_confirm'], text['annotation_type_corrected']],
            index=0,
        )
        annotator = st.text_input(text['annotator'], value='')
        location = st.text_input(text['location'], value='unknown')
        confidence = st.slider(text['confidence'], min_value=0, max_value=100, value=50)
        notes = st.text_area(text['notes'], value='')
        submit = st.form_submit_button(text['save_annotation_btn'])

    if submit:
        if st.session_state.current_embedding is None:
            st.warning(text['no_embedding'])
        else:
            annotation_type = 'confirmed' if annotation_type_label == text['annotation_type_confirm'] else 'corrected'
            canonical_annotated_name = canonicalize_species_name(annotated_name, annotation_alias_lookup)
            canonical_suggested_name = canonicalize_species_name(suggested_species, annotation_alias_lookup)
            new_metadata = {
                'name': canonical_annotated_name or 'unknown_species',
                'location': location.strip() or 'unknown',
                'annotator': annotator.strip() or 'unknown',
                'annotation_type': annotation_type,
                'model_suggested_name': canonical_suggested_name or 'unknown_species',
                'notes': f'annotation(type={annotation_type},confidence={confidence}%): {notes.strip()}',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'annotation',
            }
            try:
                store.add(st.session_state.current_embedding, new_metadata)
                store.save()
                st.success(text['save_success'].format(annotation_type=annotation_type, count=store.count()))
                st.session_state.analysis_result = None
                st.session_state.analysis_alias_normalized = False
                st.session_state.current_embedding = None
                st.session_state.evidence_rows = []
                st.session_state.localization_info = None
                st.session_state.bioclip_suggestions = []
                st.session_state.bioclip_prior_source = 'none'
                st.session_state.bioclip_prior_warning = ''
                st.session_state.taxonomy_constraints = None
                st.session_state.taxonomy_constraint_warning = ''
                st.session_state.interference_info = None
            except Exception as exc:
                st.error(text['save_failed'].format(error=exc))

st.markdown('---')
st.caption(text['footer'])
