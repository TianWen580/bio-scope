from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime
from typing import Any

from PIL import Image, ImageDraw
import numpy as np
import requests
import streamlit as st  # type: ignore

from bioclip_model import (
    attach_taxonomy_to_species_suggestions,
    encode_image,
    get_tol_taxonomy_constraints,
    get_embedding_dimension,
    get_model_candidates,
    load_bioclip_model,
    load_or_export_tol_species_list,
    load_species_taxonomy_map,
    model_display_name,
    suggest_species_from_embedding,
    suggest_species_with_tol_classifier,
)
from services.video_processing.keyframe_extraction.strategy_contract import (
    DEFAULT_KEYFRAME_STRATEGY,
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    KEYFRAME_STRATEGY_QWEN_VIDEO,
    extract_keyframes_with_strategy,
    resolve_keyframe_strategy,
)
from services.video_processing.keyframe_extraction.qwen_video_extractor import extract_qwen_video_keyframes  # pyright: ignore[reportMissingImports]
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
        'caption': '第一性原理驱动的观察、归纳、演绎与自敛生物识别闭环',
        'language_label': '语言 / Language',
        'sidebar_config': '配置',
        'api_key_label': 'DashScope API Key',
        'model_label': '模型 ID',
        'base_url_label': 'API Base URL',
        'bioclip_model_option_label': 'BioCLIP 模型候选',
        'bioclip_model_active': '当前 BioCLIP 模型',
        'bioclip_model_fallback_warning': 'BioCLIP2 加载失败，已降级为 BioCLIP1: {error}',
        'thinking_label': '开启深度思考',
        'topk_label': 'Top-K 检索',
        'index_path_label': '索引路径',
        'metadata_path_label': '元数据路径',
        'species_list_path_label': 'BioCLIP 物种列表',
        'species_alias_path_label': 'BioCLIP 物种别名',
        'bio_loaded': 'BioCLIP 已加载（设备: {device}）',
        'bio_failed': 'BioCLIP 加载失败: {error}',
        'upload_label': '上传图片',
        'upload_mode_label': '输入类型',
        'upload_mode_image': '图像',
        'upload_mode_video': '视频',
        'upload_video_label': '上传视频',
        'upload_info': '请先上传图片再开始分析。',
        'upload_video_info': '请先上传视频再开始分析。',
        'uploaded_caption': '已上传图片',
        'uploaded_video_caption': '已上传视频',
        'video_sampling_seconds': '视频抽帧间隔（秒）',
        'video_max_frames': '视频最大分析帧数',
        'video_keyframe_strategy': '视频关键帧策略',
        'video_keyframe_strategy_mechanical': '机械抽帧（固定间隔）',
        'video_keyframe_strategy_bioclip2_consistency': 'BioCLIP2 一致性（预留）',
        'video_keyframe_strategy_qwen_video': 'Qwen 视频策略（预留）',
        'video_qwen_keyframe_fps': 'Qwen 候选抽帧频率（FPS）',
        'video_qwen_max_candidate_frames': 'Qwen 候选帧上限',
        'video_bioclip_temporal_weight': 'BioCLIP 时间一致性权重',
        'video_bioclip_diversity_weight': 'BioCLIP 多样性权重',
        'video_keyframe_strategy_mechanical_desc': '按固定时间间隔抽取帧，不分析内容。适合快速处理。',
        'video_keyframe_strategy_bioclip2_desc': '基于 BioCLIP2 特征一致性选择代表性帧（实现中）。',
        'video_keyframe_strategy_qwen_video_desc': '使用 Qwen 多模态模型直接分析视频并输出帧位置。',
        'video_keyframe_fallback_warning': '警告：{strategy} 策略执行失败，已降级为机械抽帧。',
        'video_keyframe_formatter_validation_error': '格式验证失败：Qwen 输出必须为帧位置列表（整数），不能包含时间戳或抽象描述。',
        'video_keyframe_strategy3_output_contract': '策略 3 输出必须严格符合帧位置格式：{"frame_positions":[{"frame_id": 整数}]}。时间戳（如 00:12、12s）和抽象描述将被拒绝。',
        'video_keyframe_invalid_output_rejected': '无效输出已拒绝：检测到时间戳、非整数帧 ID 或超出范围的帧位置。',
        'video_no_frames': '视频未提取到可用帧，请检查视频编码或增加时长。',
        'video_extract_failed': '视频解析失败: {error}',
        'video_summary_title': '视频推理汇总结果',
        'video_frame_reports_title': '关键帧分析结果',
        'video_frame_report': '关键帧 {index}（{timestamp}s）',
        'video_aggregate_failed': '视频汇总失败，已返回逐帧结果: {error}',
        'video_frame_failed': '关键帧 {index} 分析失败: {error}',
        'run_analysis_btn': '开始分析',
        'methodology_title': '方法论框架',
        'methodology_summary': '默认采用第一性原理：先观察事实，再归纳模式，再演绎验证，最后自敛为当前最稳妥结论；旧识别链路被保留为证据层与约束层，做批判性转换而非全盘否定。',
        'methodology_habit': '思考习惯：观察 -> 归纳 -> 演绎 -> 自敛',
        'methodology_inductive': '归纳推理：从形态、纹理、颜色、姿态、环境和多帧一致性中总结重复模式，提出候选假设。',
        'methodology_deductive': '演绎推理：用 BioCLIP 检索先验、层级约束和干扰分析逐项验证或排除假设。',
        'methodology_converge': '自敛结论：保留当前最可信候选，同时明确不确定性、冲突点和下一步验证动作。',
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
        'annotation_video_disabled': '视频模式下暂不支持直接标注入库，请先基于关键帧单图分析后再标注。',
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
        'caption': 'A first-principles bio-identification loop for observation, induction, deduction, and convergence',
        'language_label': 'Language / 语言',
        'sidebar_config': 'Configuration',
        'api_key_label': 'DashScope API Key',
        'model_label': 'Model ID',
        'base_url_label': 'API Base URL',
        'bioclip_model_option_label': 'BioCLIP model candidates',
        'bioclip_model_active': 'Active BioCLIP model',
        'bioclip_model_fallback_warning': 'BioCLIP2 load failed; fallback to BioCLIP1: {error}',
        'thinking_label': 'Enable thinking',
        'topk_label': 'Top-K retrieval',
        'index_path_label': 'Index path',
        'metadata_path_label': 'Metadata path',
        'species_list_path_label': 'BioCLIP species list',
        'species_alias_path_label': 'BioCLIP species aliases',
        'bio_loaded': 'BioCLIP loaded on {device}',
        'bio_failed': 'Failed to load BioCLIP: {error}',
        'upload_label': 'Upload image',
        'upload_mode_label': 'Input type',
        'upload_mode_image': 'Image',
        'upload_mode_video': 'Video',
        'upload_video_label': 'Upload video',
        'upload_info': 'Upload an image to start analysis.',
        'upload_video_info': 'Upload a video to start analysis.',
        'uploaded_caption': 'Uploaded image',
        'uploaded_video_caption': 'Uploaded video',
        'video_sampling_seconds': 'Video frame interval (seconds)',
        'video_max_frames': 'Maximum analyzed frames',
        'video_keyframe_strategy': 'Video keyframe strategy',
        'video_keyframe_strategy_mechanical': 'Mechanical sampling (fixed interval)',
        'video_keyframe_strategy_bioclip2_consistency': 'BioCLIP2 consistency (reserved)',
        'video_keyframe_strategy_qwen_video': 'Qwen video strategy (reserved)',
        'video_qwen_keyframe_fps': 'Qwen candidate sampling FPS',
        'video_qwen_max_candidate_frames': 'Qwen max candidate frames',
        'video_bioclip_temporal_weight': 'BioCLIP temporal consistency weight',
        'video_bioclip_diversity_weight': 'BioCLIP diversity weight',
        'video_keyframe_strategy_mechanical_desc': 'Extract frames at fixed time intervals without content analysis. Suitable for fast processing.',
        'video_keyframe_strategy_bioclip2_desc': 'Select representative frames based on BioCLIP2 feature consistency (in progress).',
        'video_keyframe_strategy_qwen_video_desc': 'Use Qwen multimodal model to directly analyze video and output frame positions.',
        'video_keyframe_fallback_warning': 'Warning: {strategy} strategy execution failed. Downgraded to mechanical sampling.',
        'video_keyframe_formatter_validation_error': 'Format validation failed: Qwen output must be a list of frame positions (integers). Timestamps or abstract descriptions are not allowed.',
        'video_keyframe_strategy3_output_contract': 'Strategy 3 output must strictly follow frame position format: {"frame_positions":[{"frame_id": integer}]}. Timestamps (e.g. 00:12, 12s) and abstract descriptions will be rejected.',
        'video_keyframe_invalid_output_rejected': 'Invalid output rejected: detected timestamps, non-integer frame IDs, or out-of-range frame positions.',
        'video_no_frames': 'No valid frames extracted from video. Check codec or duration.',
        'video_extract_failed': 'Video parsing failed: {error}',
        'video_summary_title': 'Video reasoning summary',
        'video_frame_reports_title': 'Keyframe analysis reports',
        'video_frame_report': 'Frame {index} ({timestamp}s)',
        'video_aggregate_failed': 'Video aggregation failed; showing frame-level results: {error}',
        'video_frame_failed': 'Frame {index} analysis failed: {error}',
        'run_analysis_btn': 'Run analysis',
        'methodology_title': 'Methodology Framework',
        'methodology_summary': 'The demo now follows first principles by default: observe facts first, induce patterns next, validate them deductively, then converge to the safest current conclusion; the legacy recognition stack is retained as the evidence and constraint layer through critical transformation rather than wholesale rejection.',
        'methodology_habit': 'Thinking habit: Observe -> Induce -> Deduce -> Converge',
        'methodology_inductive': 'Inductive reasoning: extract repeated patterns from morphology, texture, color, pose, context, and frame consistency to form candidate hypotheses.',
        'methodology_deductive': 'Deductive reasoning: validate or eliminate those hypotheses with BioCLIP priors, taxonomy constraints, and interference analysis.',
        'methodology_converge': 'Converged Conclusion: keep the strongest current candidate while exposing uncertainty, conflict points, and next verification actions.',
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
        'annotation_video_disabled': 'Annotation write-back is disabled for video mode. Please annotate from single-frame image analysis.',
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
    temporal_context: dict | None = None,
    enable_multi_species: bool = True,
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

[第一性原理工作法]
1. 先陈述可直接观察的事实，不得把猜测写成事实。
2. 先做归纳推理：从形态、纹理、颜色、姿态、环境和多目标/多帧一致性中总结模式，形成候选假设。
3. 再做演绎推理：用 BioCLIP 先验、层级约束和干扰分析验证、保留或排除假设。
4. 最后做自敛：给出当前最稳妥结论，并明确剩余不确定性与下一步验证动作。

[多物种检测要求]
图片中可能存在多个不同物种。请识别并列出图中所有明显不同的生物目标，对每个目标分别分析候选物种。如果多个目标属于同一物种，请合并说明。

[输出要求]
1. 使用以下固定分段输出：观察事实、归纳推理、演绎推理、自敛结论、后续验证。
2. 识别图中所有不同生物目标，对每个目标给出 Top-3 候选物种。
3. 标注每个目标的位置（使用坐标或相对位置描述）。
4. 若上面的 BioCLIP 层级约束已启用，你的 Top-3 必须严格满足约束范围；若未启用，请在演绎推理中说明本轮未启用约束。
5. 在归纳推理中描述关键形态学特征与证据模式，在演绎推理中写明哪些约束或证据支持/排除了候选。
6. 在自敛结论中给出当前最可信 Top-1、置信度、主要不确定性；如果 Top-1 置信度低于 80%，明确列出需要补充的关键信息。
7. 给出下一步采样或验证建议。

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

[First-Principles Method]
1. State directly observable facts first and do not present guesses as facts.
2. Run inductive reasoning first: summarize recurring patterns from morphology, texture, color, pose, context, and multi-target or multi-frame consistency to form hypotheses.
3. Then run deductive reasoning: validate, keep, or eliminate those hypotheses using BioCLIP priors, taxonomy constraints, and interference analysis.
4. Finish with convergence: provide the safest current conclusion, remaining uncertainty, and the next best verification action.

[Output Requirements]
1. Use these exact sections: Observed Facts, Inductive Reasoning, Deductive Reasoning, Converged Conclusion, Next Verification.
2. Provide Top-3 candidate species with confidence and rationale for each distinct biological target.
3. Mark the location of each target using coordinates or relative position descriptions.
4. Your Top-3 must strictly satisfy the BioCLIP taxonomy constraints above; if constraints are enabled, do not go outside the allowed scope.
5. In Inductive Reasoning, explain the pattern-based evidence. In Deductive Reasoning, explain which priors, constraints, or interference findings support or eliminate candidates.
6. In Converged Conclusion, provide the strongest current Top-1, confidence, and key uncertainty; if top-1 confidence is below 80%, list exact follow-up questions.
7. Provide the next sampling or validation recommendation.

You must answer in English and use clear section headers.'''


def build_methodology_markdown(language: str) -> str:
    if language == 'zh':
        return (
            '- ' + LANGUAGE_PACK['zh']['methodology_summary'] + '\n'
            '- ' + LANGUAGE_PACK['zh']['methodology_habit'] + '\n'
            '- ' + LANGUAGE_PACK['zh']['methodology_inductive'] + '\n'
            '- ' + LANGUAGE_PACK['zh']['methodology_deductive'] + '\n'
            '- ' + LANGUAGE_PACK['zh']['methodology_converge']
        )

    return (
        '- ' + LANGUAGE_PACK['en']['methodology_summary'] + '\n'
        '- ' + LANGUAGE_PACK['en']['methodology_habit'] + '\n'
        '- ' + LANGUAGE_PACK['en']['methodology_inductive'] + '\n'
        '- ' + LANGUAGE_PACK['en']['methodology_deductive'] + '\n'
        '- ' + LANGUAGE_PACK['en']['methodology_converge']
    )


def _is_timeout_like_error(message: str) -> bool:
    lowered = message.lower()
    return (
        'http 504' in lowered
        or 'gateway timeout' in lowered
        or 'stream timeout' in lowered
        or 'read timed out' in lowered
        or 'timeout' in lowered
    )


def _is_oom_like_error(message: str) -> bool:
    lowered = message.lower()
    return ('out of memory' in lowered) or ('cuda oom' in lowered) or ('cublas_status_alloc_failed' in lowered)


def load_bioclip_with_fallback(
    preferred_model_id: str,
    model_candidates: list[str],
    device: str | None = None,
) -> tuple[Any, Any, str, str, str | None]:
    ordered = [preferred_model_id] + [x for x in model_candidates if x != preferred_model_id]
    unique: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        key = item.strip()
        if not key or key in seen:
            continue
        unique.append(key)
        seen.add(key)

    last_error = 'unknown'
    for idx, model_id in enumerate(unique):
        try:
            model, preprocess, target_device = load_bioclip_model(device=device, model_id=model_id)
            warning = None
            if idx > 0:
                warning = f'fallback_from={preferred_model_id};active={model_id};reason={last_error}'
            return model, preprocess, target_device, model_id, warning
        except Exception as exc:
            last_error = str(exc)
            if idx < len(unique) - 1 and _is_oom_like_error(last_error):
                continue
            if idx < len(unique) - 1:
                continue
            raise

    raise RuntimeError(f'Failed to load all BioCLIP candidates: {last_error}')


def call_openai_compatible(
    base_url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    image_base64: str | None,
    image_mime: str | None,
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
            'messages': [],
        }

        user_content: list[dict[str, Any]] = []
        if image_base64 and image_mime:
            user_content.append({'type': 'image_url', 'image_url': {'url': f'data:{image_mime};base64,{image_base64}'}})
        user_content.append({'type': 'text', 'text': prompt})
        payload['messages'] = [{'role': 'user', 'content': user_content}]
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
    raw = os.getenv('BIOCLIP_TAXONOMY_CONSTRAINT_THRESHOLD', '0.9').strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.9
    return max(0.0, min(1.0, value))


def get_video_frame_interval_seconds() -> float:
    raw = os.getenv('VIDEO_FRAME_INTERVAL_SECONDS', '2.0').strip()
    try:
        value = float(raw)
    except ValueError:
        value = 2.0
    return max(0.5, min(30.0, value))


def get_video_max_frames() -> int:
    raw = os.getenv('VIDEO_MAX_FRAMES', '10').strip()
    try:
        value = int(raw)
    except ValueError:
        value = 10
    return max(1, min(64, value))


def get_video_keyframe_strategy() -> str:
    raw = os.getenv('VIDEO_KEYFRAME_STRATEGY', DEFAULT_KEYFRAME_STRATEGY)
    return resolve_keyframe_strategy(raw)


def get_video_qwen_keyframe_fps() -> float:
    raw = os.getenv('VIDEO_QWEN_KEYFRAME_FPS', '1.0').strip()
    try:
        value = float(raw)
    except ValueError:
        value = 1.0
    return max(0.1, min(12.0, value))


def get_video_qwen_max_candidate_frames() -> int:
    raw = os.getenv('VIDEO_QWEN_MAX_CANDIDATE_FRAMES', '64').strip()
    try:
        value = int(raw)
    except ValueError:
        value = 64
    return max(1, min(512, value))


def get_video_bioclip_temporal_weight() -> float:
    raw = os.getenv('VIDEO_BIOCLIP_TEMPORAL_WEIGHT', '0.35').strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.35
    return max(0.0, min(1.0, value))


def get_video_bioclip_diversity_weight() -> float:
    raw = os.getenv('VIDEO_BIOCLIP_DIVERSITY_WEIGHT', '0.65').strip()
    try:
        value = float(raw)
    except ValueError:
        value = 0.65
    return max(0.0, min(1.0, value))


def model_slug(model_id: str) -> str:
    key = model_id.strip().lower()
    if 'bioclip-2' in key:
        return 'bioclip2'
    if 'imageomics/bioclip' in key:
        return 'bioclip1'
    return re.sub(r'[^a-z0-9]+', '_', key).strip('_') or 'bioclip_custom'


def get_store_paths_for_model(model_id: str) -> tuple[str, str]:
    base_index = os.getenv('BIOCLIP_INDEX_PATH', './data/faiss_index.bin').strip() or './data/faiss_index.bin'
    base_meta = os.getenv('BIOCLIP_METADATA_PATH', './data/faiss_metadata.pkl').strip() or './data/faiss_metadata.pkl'

    slug = model_slug(model_id)
    if slug == 'bioclip1':
        return base_index, base_meta

    if slug == 'bioclip2':
        index_path = os.getenv('BIOCLIP2_INDEX_PATH', '').strip()
        metadata_path = os.getenv('BIOCLIP2_METADATA_PATH', '').strip()
        if index_path and metadata_path:
            return index_path, metadata_path

    index_suffix = f'_{slug}.bin'
    meta_suffix = f'_{slug}.pkl'
    if base_index.endswith('.bin'):
        index_path = base_index[:-4] + index_suffix
    else:
        index_path = base_index + index_suffix
    if base_meta.endswith('.pkl'):
        metadata_path = base_meta[:-4] + meta_suffix
    else:
        metadata_path = base_meta + meta_suffix
    return index_path, metadata_path


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
    zh_map = {
        'animal': '动物',
        'mammal': '哺乳动物',
        'bird': '鸟类',
        'reptile': '爬行动物',
        'amphibian': '两栖动物',
        'insect': '昆虫',
    }
    lowered = raw.lower()
    if lowered in zh_map:
        return zh_map[lowered]
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


def _contains_cjk_text(text: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff]', text or ''))


def _pick_zh_name(common_name: str, aliases: list[str]) -> str:
    for alias in aliases:
        if _contains_cjk_text(alias):
            return alias
    if _contains_cjk_text(common_name):
        return common_name
    return ''


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
    language: str,
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
        if language == 'zh':
            zh_name = _pick_zh_name(common_name, aliases)
            if zh_name:
                display = f'{zh_name}（{clean_species}）'
            else:
                display = clean_species if not common_name else f'{clean_species} ({common_name})'
        else:
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

    threshold = float(constraint_info.get('threshold', 0.9))
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
        return '无' if language == 'zh' else 'none'

    if interference_info.get('error'):
        if language == 'zh':
            return f"不可用: {interference_info.get('error')}"
        return f"unavailable: {interference_info.get('error')}"

    route = str(interference_info.get('route', 'full_image'))
    analysis_json = interference_info.get('analysis_json')
    if not isinstance(analysis_json, dict):
        return '无' if language == 'zh' else 'none'

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
            label = str(item.get('label', '未知' if language == 'zh' else 'unknown'))
            risk_score = item.get('risk_score', '无' if language == 'zh' else 'n/a')
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

    if out:
        return '\n'.join(out)
    return '无' if language == 'zh' else 'none'


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


def _extract_video_keyframes_mechanical(
    uploaded_file,
    interval_seconds: float,
    max_frames: int,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return [], f'OpenCV unavailable: {exc}'

    suffix = Path(uploaded_file.name or 'upload.mp4').suffix or '.mp4'
    tmp_path = ''
    frames: list[dict[str, Any]] = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return [], 'failed to open video stream'

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0
        step = max(1, int(round(fps * interval_seconds)))

        frame_idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                timestamp = frame_idx / fps
                frames.append(
                    {
                        'index': len(frames) + 1,
                        'frame_id': frame_idx,
                        'timestamp_sec': float(timestamp),
                        'image': image,
                    }
                )
            frame_idx += 1

        cap.release()
        return frames, None
    except Exception as exc:
        return [], str(exc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _extract_video_keyframes_bioclip2_consistency(
    uploaded_file,
    interval_seconds: float,
    max_frames: int,
    temporal_weight: float,
    diversity_weight: float,
) -> tuple[list[dict[str, Any]], str | None]:
    from services.video_processing.keyframe_extraction.bioclip2_consistency_extractor import (
        extract_bioclip2_consistency_keyframes,
    )

    return extract_bioclip2_consistency_keyframes(
        uploaded_file=uploaded_file,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
        temporal_weight=temporal_weight,
        diversity_weight=diversity_weight,
        encode_image_fn=encode_image,
        load_bioclip_model_fn=load_bioclip_model,
    )


def _extract_video_keyframes_qwen_video(
    uploaded_file,
    interval_seconds: float,
    max_frames: int,
    keyframe_fps: float,
    max_candidate_frames: int,
    base_url: str,
    api_key: str,
    model_name: str,
    request_timeout: int,
    language: str = 'zh',
) -> tuple[list[dict[str, Any]], str | None]:
    _ = interval_seconds
    return extract_qwen_video_keyframes(
        uploaded_file=uploaded_file,
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        request_timeout=request_timeout,
        keyframe_fps=keyframe_fps,
        max_candidate_frames=max_candidate_frames,
        max_frames=max_frames,
        language=language,
    )


def extract_video_keyframes(
    uploaded_file,
    interval_seconds: float,
    max_frames: int,
    strategy: str | None = None,
    qwen_keyframe_fps: float | None = None,
    qwen_max_candidate_frames: int | None = None,
    bioclip_temporal_weight: float | None = None,
    bioclip_diversity_weight: float | None = None,
    base_url: str = '',
    api_key: str = '',
    model_name: str = '',
    request_timeout: int = 1800,
    language: str = 'zh',
    include_dispatch_metadata: bool = False,
) -> tuple[list[dict[str, Any]], str | None] | tuple[list[dict[str, Any]], str | None, str | None]:
    resolved_qwen_keyframe_fps = get_video_qwen_keyframe_fps() if qwen_keyframe_fps is None else float(qwen_keyframe_fps)
    resolved_qwen_keyframe_fps = max(0.1, min(12.0, resolved_qwen_keyframe_fps))
    hard_fps_cap = 1.0 / max(interval_seconds, 1e-6)
    resolved_qwen_keyframe_fps = min(resolved_qwen_keyframe_fps, hard_fps_cap)

    resolved_qwen_max_candidate_frames = (
        get_video_qwen_max_candidate_frames() if qwen_max_candidate_frames is None else int(qwen_max_candidate_frames)
    )
    resolved_qwen_max_candidate_frames = max(1, min(512, resolved_qwen_max_candidate_frames))
    resolved_qwen_max_candidate_frames = min(resolved_qwen_max_candidate_frames, max_frames)

    resolved_bioclip_temporal_weight = (
        get_video_bioclip_temporal_weight() if bioclip_temporal_weight is None else float(bioclip_temporal_weight)
    )
    resolved_bioclip_temporal_weight = max(0.0, min(1.0, resolved_bioclip_temporal_weight))

    resolved_bioclip_diversity_weight = (
        get_video_bioclip_diversity_weight() if bioclip_diversity_weight is None else float(bioclip_diversity_weight)
    )
    resolved_bioclip_diversity_weight = max(0.0, min(1.0, resolved_bioclip_diversity_weight))

    selected_strategy = strategy.strip().lower() if isinstance(strategy, str) else 'mechanical'
    selected_non_default = selected_strategy in {
        KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        KEYFRAME_STRATEGY_QWEN_VIDEO,
    }

    frames, err, used_strategy = extract_keyframes_with_strategy(
        strategy=strategy,
        mechanical_extractor=lambda: _extract_video_keyframes_mechanical(
            uploaded_file,
            interval_seconds=interval_seconds,
            max_frames=max_frames,
        ),
        non_default_extractors={
            KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY: lambda: _extract_video_keyframes_bioclip2_consistency(
                uploaded_file,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
                temporal_weight=resolved_bioclip_temporal_weight,
                diversity_weight=resolved_bioclip_diversity_weight,
            ),
            KEYFRAME_STRATEGY_QWEN_VIDEO: lambda: _extract_video_keyframes_qwen_video(
                uploaded_file,
                interval_seconds=interval_seconds,
                max_frames=max_frames,
                keyframe_fps=resolved_qwen_keyframe_fps,
                max_candidate_frames=resolved_qwen_max_candidate_frames,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                request_timeout=request_timeout,
                language=language,
            ),
        },
    )
    warning_key: str | None = None
    if selected_non_default and used_strategy == 'mechanical':
        warning_key = 'video_keyframe_fallback_warning'

    if include_dispatch_metadata:
        return frames, err, warning_key

    return frames, err


def build_video_summary_prompt(frame_reports: list[dict[str, Any]], language: str) -> str:
    lines: list[str] = []
    for frame in frame_reports:
        lines.append(
            f"frame_index={frame['index']}, timestamp={frame['timestamp_sec']:.2f}s\n"
            f"report=\n{frame['analysis']}\n"
        )
    frame_blob = '\n'.join(lines)

    if language == 'zh':
        return (
            '你是一名专业生物学家。以下是同一段视频多个关键帧的识别报告。\n'
        '请遵循第一性原理，对视频做“观察 -> 归纳 -> 演绎 -> 自敛”聚合推理。\n'
        '必须使用以下固定分段输出：观察事实、归纳推理、演绎推理、自敛结论、后续验证。\n'
        '1) 观察事实：关键帧共同出现的稳定事实与显著变化\n'
        '2) 归纳推理：跨帧模式、重复形态线索、时间一致性\n'
        '3) 演绎推理：哪些候选被帧间一致性、BioCLIP 证据或约束支持/排除；若本轮约束未启用，需要明确说明\n'
        '4) 自敛结论：视频级 Top-3 物种候选（含置信度与依据）\n'
        '5) 后续验证：继续采样、补充信息或复核建议\n\n'
        f'{frame_blob}\n'
        '必须使用简体中文，并清晰分段。'
    )

    return (
        'You are a professional biologist. The following are frame-level reports from the same video.\n'
        'Use first principles to produce a video-level synthesis through Observe -> Induce -> Deduce -> Converge.\n'
        'You must use these exact sections: Observed Facts, Inductive Reasoning, Deductive Reasoning, Converged Conclusion, Next Verification.\n'
        '1) Observed Facts: stable facts and meaningful changes across frames\n'
        '2) Inductive Reasoning: repeated patterns, morphology cues, and temporal consistency\n'
        '3) Deductive Reasoning: which candidates are supported or eliminated by consistency, BioCLIP evidence, or constraints; if constraints are disabled for this run, say so explicitly\n'
        '4) Converged Conclusion: video-level Top-3 species candidates with confidence and evidence\n'
        '5) Next Verification: next sampling, added evidence, or review recommendations\n\n'
        f'{frame_blob}\n'
        'Use clear English sections.'
    )


def run_single_image_pipeline(
    *,
    image: Image.Image,
    image_mime: str,
    lang: str,
    model,
    preprocess,
    device: str,
    selected_bioclip_model_id: str,
    store: LocalFAISSStore,
    top_k: int,
    enable_small_target_opt: bool,
    use_qwen_locator: bool,
    use_yolo_assist: bool,
    yolo_model_path: str,
    max_crops: int,
    base_url: str,
    api_key: str,
    model_name: str,
    enable_thinking: bool,
    request_timeout: int,
    thinking_budget: int,
    species_list_path: str,
    species_csv_path: str,
    species_alias_path: str,
    species_list_max_labels: int,
    use_tol_classifier: bool,
    auto_export_tol_species: bool,
    taxonomy_constraint_threshold: float,
    temporal_context: dict | None = None,
    enable_multi_species: bool = True,
) -> dict[str, Any]:
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
    else:
        loc_view = {'recognition_mode': 'full', 'render_boxes': []}

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
                        model_id=selected_bioclip_model_id,
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
                        model_id=selected_bioclip_model_id,
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

    interference_info = run_interference_analysis_agent(
        image_base64=base64_image,
        image_mime=image_mime,
        language=lang,
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        localization_info=loc_view,
        bioclip_prior_text=bioclip_prior_text,
        taxonomy_constraint_text=taxonomy_constraint_text,
        enable_thinking=enable_thinking,
    )
    interference_text = format_interference_text(interference_info, lang)

    clue_text = 'none'
    if loc_view and loc_view.get('fused_boxes'):
        clue_lines = []
        for idx, box in enumerate(loc_view['fused_boxes'], start=1):
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

    evidence_text = ''
    evidence_rows: list[dict[str, str]] = []
    if deduped:
        evidence_lines = []
        for rank, item in enumerate(deduped, start=1):
            score = item['similarity']
            meta = item['metadata']
            crop = item['crop']
            if lang == 'zh':
                line = (
                    f"{rank}. 物种={meta.get('name', '未知')}, "
                    f"相似度={score:.4f}, 裁切={crop.get('id', 'full')}, 来源={localization_source_label(crop.get('source', 'full'), lang)}, "
                    f"框坐标={fmt_bbox(crop.get('bbox_norm', (0, 0, 1, 1)))}, 线索={'|'.join(crop.get('clues', []))}, "
                    f"地点={meta.get('location', '未知')}, 备注={meta.get('notes', '')}"
                )
            else:
                line = (
                    f"{rank}. species={meta.get('name', 'unknown')}, "
                    f"similarity={score:.4f}, crop={crop.get('id', 'full')}, source={crop.get('source', 'full')}, "
                    f"bbox={fmt_bbox(crop.get('bbox_norm', (0, 0, 1, 1)))}, clues={'|'.join(crop.get('clues', []))}, "
                    f"location={meta.get('location', 'unknown')}, note={meta.get('notes', '')}"
                )
            evidence_lines.append(line)
            evidence_rows.append(
                {
                    LANGUAGE_PACK[lang]['species_col']: meta.get('name', 'unknown'),
                    LANGUAGE_PACK[lang]['similarity_col']: f'{score:.4f}',
                    LANGUAGE_PACK[lang]['crop_col']: crop.get('id', 'full'),
                    LANGUAGE_PACK[lang]['source_col']: localization_source_label(crop.get('source', 'full'), lang),
                    LANGUAGE_PACK[lang]['bbox_col']: fmt_bbox(crop.get('bbox_norm', (0, 0, 1, 1))),
                    LANGUAGE_PACK[lang]['clues_col']: '|'.join(crop.get('clues', [])),
                }
            )
        evidence_text = '\n'.join(evidence_lines)
    else:
        evidence_text = LANGUAGE_PACK[lang]['no_samples']

    prompt = build_prompt(
        evidence_text,
        language=lang,
        clue_text=clue_text,
        bioclip_prior_text=bioclip_prior_text,
        taxonomy_constraint_text=taxonomy_constraint_text,
        interference_text=interference_text,
        temporal_context=temporal_context,
        enable_multi_species=enable_multi_species,
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
    if not ok:
        return {
            'error': result,
            'localization_info': loc_view,
            'interference_info': interference_info,
            'taxonomy_constraints': taxonomy_constraint_info,
            'taxonomy_constraint_warning': taxonomy_constraint_error or '',
            'bioclip_suggestions': bioclip_suggestions,
            'bioclip_prior_source': bioclip_prior_source,
            'bioclip_prior_warning': '; '.join(prior_warnings[:2]) if prior_warnings else '',
            'evidence_rows': evidence_rows,
            'current_embedding': best_embedding,
            'analysis_alias_normalized': False,
            'analysis_result': None,
            'best_crop_image': best_crop_image,
        }

    normalized_result, normalized_changed = normalize_analysis_text_species_names(
        result,
        analysis_alias_replacements,
    )
    return {
        'error': None,
        'localization_info': loc_view,
        'interference_info': interference_info,
        'taxonomy_constraints': taxonomy_constraint_info,
        'taxonomy_constraint_warning': taxonomy_constraint_error or '',
        'bioclip_suggestions': bioclip_suggestions,
        'bioclip_prior_source': bioclip_prior_source,
        'bioclip_prior_warning': '; '.join(prior_warnings[:2]) if prior_warnings else '',
        'evidence_rows': evidence_rows,
        'current_embedding': best_embedding,
        'analysis_alias_normalized': normalized_changed,
        'analysis_result': normalized_result,
        'best_crop_image': best_crop_image,
    }


if 'app_lang' not in st.session_state:
    st.session_state.app_lang = get_default_language()
if st.session_state.app_lang not in {'zh', 'en'}:
    st.session_state.app_lang = 'zh'

st.set_page_config(page_title=LANGUAGE_PACK[st.session_state.app_lang]['page_title'], layout='wide')

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_input_mode' not in st.session_state:
    st.session_state.analysis_input_mode = 'image'
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
with st.expander(text['methodology_title'], expanded=True):
    st.markdown(build_methodology_markdown(lang))

with st.sidebar:
    st.subheader(text['sidebar_config'])
    env_key = os.getenv('DASHSCOPE_API_KEY', '')
    env_model = os.getenv('DASHSCOPE_MODEL', 'qwen3.5-plus')
    env_base = os.getenv('DASHSCOPE_BASE_URL', 'https://coding.dashscope.aliyuncs.com/v1')
    model_candidates = get_model_candidates()
    env_bioclip_model = os.getenv('BIOCLIP_MODEL_ID', model_candidates[0] if model_candidates else 'hf-hub:imageomics/bioclip-2').strip()
    if env_bioclip_model and env_bioclip_model not in model_candidates:
        model_candidates = [env_bioclip_model] + model_candidates
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
    video_interval_default = get_video_frame_interval_seconds()
    video_max_frames_default = get_video_max_frames()
    video_keyframe_strategy_default = get_video_keyframe_strategy()
    video_qwen_keyframe_fps_default = get_video_qwen_keyframe_fps()
    video_qwen_max_candidate_frames_default = get_video_qwen_max_candidate_frames()
    video_bioclip_temporal_weight_default = get_video_bioclip_temporal_weight()
    video_bioclip_diversity_weight_default = get_video_bioclip_diversity_weight()

    api_key = st.text_input(text['api_key_label'], value=env_key, type='password')
    model_name = st.text_input(text['model_label'], value=env_model)
    base_url = st.text_input(text['base_url_label'], value=env_base)
    selected_bioclip_model_id = st.selectbox(
        text['bioclip_model_option_label'],
        options=model_candidates,
        index=model_candidates.index(env_bioclip_model) if env_bioclip_model in model_candidates else 0,
        format_func=model_display_name,
    )
    enable_thinking = st.checkbox(text['thinking_label'], value=env_thinking)

    enable_small_target_opt = st.checkbox(text['optimize_enable'], value=env_opt)
    use_qwen_locator = st.checkbox(text['use_qwen_locator'], value=env_qwen)
    use_yolo_assist = st.checkbox(text['use_yolo_assist'], value=env_yolo)
    yolo_model_path = st.text_input(text['yolo_model_path'], value=env_yolo_path)
    max_crops = st.slider(text['max_crops'], min_value=1, max_value=8, value=max(1, min(8, env_max_crops)))

    top_k = st.slider(text['topk_label'], min_value=1, max_value=10, value=3)
    video_frame_interval = st.slider(text['video_sampling_seconds'], min_value=0.5, max_value=10.0, value=float(video_interval_default), step=0.5)
    video_max_frames = st.slider(text['video_max_frames'], min_value=1, max_value=30, value=int(video_max_frames_default), step=1)
    keyframe_strategy_options = [
        DEFAULT_KEYFRAME_STRATEGY,
        KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        KEYFRAME_STRATEGY_QWEN_VIDEO,
    ]
    keyframe_strategy_labels = {
        DEFAULT_KEYFRAME_STRATEGY: text['video_keyframe_strategy_mechanical'],
        KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY: text['video_keyframe_strategy_bioclip2_consistency'],
        KEYFRAME_STRATEGY_QWEN_VIDEO: text['video_keyframe_strategy_qwen_video'],
    }
    video_keyframe_strategy = st.selectbox(
        text['video_keyframe_strategy'],
        options=keyframe_strategy_options,
        index=keyframe_strategy_options.index(video_keyframe_strategy_default),
        format_func=lambda value: keyframe_strategy_labels.get(value, value),
    )

    video_qwen_keyframe_fps = video_qwen_keyframe_fps_default
    video_qwen_max_candidate_frames = video_qwen_max_candidate_frames_default
    video_bioclip_temporal_weight = video_bioclip_temporal_weight_default
    video_bioclip_diversity_weight = video_bioclip_diversity_weight_default
    if video_keyframe_strategy == KEYFRAME_STRATEGY_QWEN_VIDEO:
        max_fps = max(0.1, min(12.0, 1.0 / max(video_frame_interval, 1e-6)))
        video_qwen_keyframe_fps = st.slider(
            text['video_qwen_keyframe_fps'],
            min_value=0.1,
            max_value=max_fps,
            value=float(min(video_qwen_keyframe_fps_default, max_fps)),
            step=0.1,
        )
        video_qwen_max_candidate_frames = st.slider(
            text['video_qwen_max_candidate_frames'],
            min_value=1,
            max_value=video_max_frames,
            value=int(min(video_qwen_max_candidate_frames_default, video_max_frames)),
            step=1,
        )
    elif video_keyframe_strategy == KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY:
        video_bioclip_temporal_weight = st.slider(
            text['video_bioclip_temporal_weight'],
            min_value=0.0,
            max_value=1.0,
            value=float(video_bioclip_temporal_weight_default),
            step=0.05,
        )
        video_bioclip_diversity_weight = st.slider(
            text['video_bioclip_diversity_weight'],
            min_value=0.0,
            max_value=1.0,
            value=float(video_bioclip_diversity_weight_default),
            step=0.05,
        )

    active_index_path, active_metadata_path = get_store_paths_for_model(selected_bioclip_model_id)
    st.write(f"{text['bioclip_model_active']}:", model_display_name(selected_bioclip_model_id))
    st.write(f"{text['index_path_label']}:", active_index_path)
    st.write(f"{text['metadata_path_label']}:", active_metadata_path)
    st.write(f"{text['species_list_path_label']}:", species_list_path)
    st.write(f"{text['species_alias_path_label']}:", species_alias_path)

model = None
preprocess = None
device = 'cpu'
active_model_id = selected_bioclip_model_id
model_load_warning: str | None = None

try:
    model, preprocess, device, active_model_id, model_load_warning = load_bioclip_with_fallback(
        preferred_model_id=selected_bioclip_model_id,
        model_candidates=model_candidates,
    )
    st.sidebar.success(text['bio_loaded'].format(device=device))
    if model_load_warning:
        st.sidebar.warning(text['bioclip_model_fallback_warning'].format(error=model_load_warning))
except Exception as exc:
    st.error(text['bio_failed'].format(error=exc))
    st.stop()

active_embedding_dim = get_embedding_dimension(active_model_id)
store = LocalFAISSStore(active_index_path, active_metadata_path, dimension=active_embedding_dim)

upload_mode = st.radio(
    text['upload_mode_label'],
    options=[text['upload_mode_image'], text['upload_mode_video']],
    horizontal=True,
)

uploaded_image = None
uploaded_video = None

if upload_mode == text['upload_mode_image']:
    uploaded_image = st.file_uploader(text['upload_label'], type=['jpg', 'jpeg', 'png', 'bmp', 'webp'])
    if uploaded_image is None:
        st.info(text['upload_info'])
        st.stop()
    image = Image.open(uploaded_image).convert('RGB')
    image_mime = uploaded_image.type or 'image/jpeg'
else:
    uploaded_video = st.file_uploader(text['upload_video_label'], type=['mp4', 'mov', 'avi', 'mkv', 'webm'])
    if uploaded_video is None:
        st.info(text['upload_video_info'])
        st.stop()
    image = None
    image_mime = None

left_col, right_col = st.columns([1, 1])

with left_col:
    if upload_mode == text['upload_mode_image'] and image is not None:
        st.image(image, caption=text['uploaded_caption'], use_container_width=True)
    elif uploaded_video is not None:
        st.video(uploaded_video)
        st.caption(text['uploaded_video_caption'])

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
        if upload_mode == text['upload_mode_image'] and image is not None:
            frame_result = run_single_image_pipeline(
                image=image,
                image_mime=image_mime or 'image/jpeg',
                lang=lang,
                model=model,
                preprocess=preprocess,
                device=device,
                selected_bioclip_model_id=active_model_id,
                store=store,
                top_k=top_k,
                enable_small_target_opt=enable_small_target_opt,
                use_qwen_locator=use_qwen_locator,
                use_yolo_assist=use_yolo_assist,
                yolo_model_path=yolo_model_path,
                max_crops=max_crops,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                enable_thinking=enable_thinking,
                request_timeout=request_timeout,
                thinking_budget=thinking_budget,
                species_list_path=species_list_path,
                species_csv_path=species_csv_path,
                species_alias_path=species_alias_path,
                species_list_max_labels=species_list_max_labels,
                use_tol_classifier=use_tol_classifier,
                auto_export_tol_species=auto_export_tol_species,
                taxonomy_constraint_threshold=taxonomy_constraint_threshold,
            )
            if frame_result['error']:
                st.error(text['request_failed'].format(error=frame_result['error']))
            else:
                st.session_state.analysis_input_mode = 'image'
                st.session_state.analysis_result = frame_result['analysis_result']
                st.session_state.analysis_alias_normalized = frame_result['analysis_alias_normalized']
                st.session_state.current_embedding = frame_result['current_embedding']
                st.session_state.evidence_rows = frame_result['evidence_rows']
                st.session_state.localization_info = frame_result['localization_info']
                st.session_state.bioclip_suggestions = frame_result['bioclip_suggestions']
                st.session_state.bioclip_prior_source = frame_result['bioclip_prior_source']
                st.session_state.bioclip_prior_warning = frame_result['bioclip_prior_warning']
                st.session_state.taxonomy_constraints = frame_result['taxonomy_constraints']
                st.session_state.taxonomy_constraint_warning = frame_result['taxonomy_constraint_warning']
                st.session_state.interference_info = frame_result['interference_info']
        else:
            dispatch_result = extract_video_keyframes(
                uploaded_video,
                interval_seconds=video_frame_interval,
                max_frames=video_max_frames,
                strategy=video_keyframe_strategy,
                qwen_keyframe_fps=video_qwen_keyframe_fps,
                qwen_max_candidate_frames=video_qwen_max_candidate_frames,
                bioclip_temporal_weight=video_bioclip_temporal_weight,
                bioclip_diversity_weight=video_bioclip_diversity_weight,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                request_timeout=request_timeout,
                language=lang,
                include_dispatch_metadata=True,
            )
            if len(dispatch_result) == 3:
                frames, video_error, keyframe_warning_key = dispatch_result
            else:
                frames, video_error = dispatch_result
                keyframe_warning_key = None
            if keyframe_warning_key:
                selected_strategy_label = keyframe_strategy_labels.get(video_keyframe_strategy, video_keyframe_strategy)
                st.warning(text[keyframe_warning_key].format(strategy=selected_strategy_label))
            if video_error:
                st.error(text['video_extract_failed'].format(error=video_error))
            elif not frames:
                st.warning(text['video_no_frames'])
            else:
                frame_reports: list[dict[str, Any]] = []
                frame_errors: list[str] = []
                prev_context: dict | None = None

                for frame in frames:
                    frame_output = run_single_image_pipeline(
                        image=frame['image'],
                        image_mime='image/jpeg',
                        lang=lang,
                        model=model,
                        preprocess=preprocess,
                        device=device,
                        selected_bioclip_model_id=active_model_id,
                        store=store,
                        top_k=top_k,
                        enable_small_target_opt=enable_small_target_opt,
                        use_qwen_locator=use_qwen_locator,
                        use_yolo_assist=use_yolo_assist,
                        yolo_model_path=yolo_model_path,
                        max_crops=max_crops,
                        base_url=base_url,
                        api_key=api_key,
                        model_name=model_name,
                        enable_thinking=enable_thinking,
                        request_timeout=request_timeout,
                        thinking_budget=thinking_budget,
                        species_list_path=species_list_path,
                        species_csv_path=species_csv_path,
                        species_alias_path=species_alias_path,
                        species_list_max_labels=species_list_max_labels,
                        use_tol_classifier=use_tol_classifier,
                        auto_export_tol_species=auto_export_tol_species,
                        taxonomy_constraint_threshold=taxonomy_constraint_threshold,
                        temporal_context=prev_context,
                        enable_multi_species=True,
                    )

                    if frame_output['error']:
                        frame_errors.append(text['video_frame_failed'].format(index=frame['index'], error=frame_output['error']))
                        continue

                    frame_reports.append(
                        {
                            'index': frame['index'],
                            'timestamp_sec': frame['timestamp_sec'],
                            'analysis': frame_output['analysis_result'],
                        }
                    )
                    # Update context for next frame with detected species
                    prev_context = {
                        'prev_species': frame_output.get('detected_species', []),
                        'prev_frame_idx': frame['index'],
                    }

                if not frame_reports:
                    first_error = frame_errors[0] if frame_errors else text['video_no_frames']
                    st.error(first_error)
                else:
                    if frame_errors:
                        st.warning('; '.join(frame_errors[:2]))

                    summary_prompt = build_video_summary_prompt(frame_reports, lang)
                    ok, summary_text = call_openai_compatible(
                        base_url=base_url,
                        api_key=api_key,
                        model_name=model_name,
                        prompt=summary_prompt,
                        image_base64=None,
                        image_mime=None,
                        enable_thinking=enable_thinking,
                        request_timeout=request_timeout,
                        thinking_budget=thinking_budget,
                    )
                    if not ok:
                        st.warning(text['video_aggregate_failed'].format(error=summary_text))
                        fallback_sections: list[str] = []
                        for x in frame_reports:
                            ts = f"{x['timestamp_sec']:.1f}"
                            fallback_sections.append(
                                f"{text['video_frame_report'].format(index=x['index'], timestamp=ts)}\n{x['analysis']}"
                            )
                        summary_text = '\n\n'.join(fallback_sections)

                    frame_sections: list[str] = []
                    for x in frame_reports:
                        ts = f"{x['timestamp_sec']:.1f}"
                        frame_sections.append(
                            f"#### {text['video_frame_report'].format(index=x['index'], timestamp=ts)}\n{x['analysis']}"
                        )

                    composed = (
                        f"## {text['video_summary_title']}\n\n{summary_text}\n\n"
                        f"## {text['video_frame_reports_title']}\n\n" + '\n\n'.join(frame_sections)
                    )

                    st.session_state.analysis_input_mode = 'video'
                    st.session_state.analysis_result = composed
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

if st.session_state.analysis_result:
    st.markdown('---')

    if st.session_state.analysis_input_mode == 'image':
        mode = 'full'
        render_boxes = []
        if st.session_state.localization_info:
            mode = st.session_state.localization_info.get('recognition_mode', 'full')
            render_boxes = st.session_state.localization_info.get('render_boxes', [])

        if mode == 'localized' and render_boxes and image is not None:
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
    if st.session_state.analysis_input_mode == 'video':
        st.subheader(text['annotation_title'])
        st.info(text['annotation_video_disabled'])
    else:
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
                language=lang,
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
                    st.session_state.analysis_input_mode = 'image'
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
