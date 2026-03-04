from __future__ import annotations

from dataclasses import dataclass
import os
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import requests


@dataclass
class CandidateBox:
    source: str
    score: float
    label: str
    bbox_norm: tuple[float, float, float, float]
    clues: list[str]


def _get_request_timeout_seconds() -> int:
    raw = os.getenv('DASHSCOPE_TIMEOUT_SECONDS', '1800').strip()
    try:
        timeout = int(raw)
    except ValueError:
        timeout = 1800
    return max(1800, timeout)


def _get_thinking_budget_tokens() -> int:
    raw = os.getenv('DASHSCOPE_THINKING_BUDGET', '8192').strip()
    try:
        budget = int(raw)
    except ValueError:
        budget = 8192
    return max(1024, budget)


def _is_timeout_like_error(message: str) -> bool:
    lowered = message.lower()
    return (
        'http 504' in lowered
        or 'gateway timeout' in lowered
        or 'stream timeout' in lowered
        or 'read timed out' in lowered
        or 'timeout' in lowered
    )


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize_bbox(values: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in values]
    x1, x2 = sorted((_clamp01(x1), _clamp01(x2)))
    y1, y2 = sorted((_clamp01(y1), _clamp01(y2)))
    if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
        return None
    return (x1, y1, x2, y2)


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get('text'), str):
                    parts.append(item['text'])
                elif isinstance(item.get('content'), str):
                    parts.append(item['content'])
            elif isinstance(item, str):
                parts.append(item)
        return '\n'.join(parts).strip()
    return str(content)


def _extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    stripped = re.sub(r'<think>.*?</think>', '', stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    stripped = re.sub(r'<thinking>.*?</thinking>', '', stripped, flags=re.IGNORECASE | re.DOTALL).strip()

    if '```json' in stripped:
        try:
            stripped = stripped.split('```json', 1)[1].split('```', 1)[0].strip()
        except Exception:
            pass
    elif '```' in stripped:
        parts = stripped.split('```')
        if len(parts) >= 3:
            stripped = parts[1].strip()

    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    first = stripped.find('{')
    last = stripped.rfind('}')
    if first < 0 or last <= first:
        return None
    try:
        data = json.loads(stripped[first:last + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _post_chat(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
    response_format: dict[str, Any] | None = None,
    thinking_budget: int | None = None,
    timeout: int | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    url = base_url.rstrip('/') + '/chat/completions'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    if timeout is None:
        timeout = _get_request_timeout_seconds()

    base_budget = thinking_budget if thinking_budget is not None else _get_thinking_budget_tokens()

    attempts: list[tuple[bool, int | None]] = []
    if enable_thinking:
        attempts.append((True, base_budget))
        if base_budget > 2048:
            attempts.append((True, max(2048, base_budget // 2)))
        attempts.append((False, None))
    else:
        attempts.append((False, None))

    last_error = 'Unknown request failure'

    for idx, (thinking_flag, budget) in enumerate(attempts):
        payload: dict[str, Any] = {
            'model': model_name,
            'messages': messages,
            'enable_thinking': thinking_flag,
        }
        if response_format is not None:
            payload['response_format'] = response_format
        if budget is not None:
            payload['thinking_budget'] = int(budget)

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except Exception as exc:
            last_error = f'HTTP request failed: {exc}'
            if idx < len(attempts) - 1 and _is_timeout_like_error(last_error):
                continue
            return False, last_error, None

        if response.status_code != 200:
            last_error = f'HTTP {response.status_code}: {response.text[:1000]}'
            if idx < len(attempts) - 1 and _is_timeout_like_error(last_error):
                continue
            return False, last_error, None

        try:
            data = response.json()
            message = data['choices'][0]['message']
            text = _response_content_to_text(message.get('content', ''))
            return True, text, data
        except Exception as exc:
            last_error = f'Invalid response format: {exc}'
            return False, last_error, None

    return False, last_error, None


def run_qwen_two_stage_localization(
    *,
    image_base64: str,
    image_mime: str,
    language: str,
    base_url: str,
    api_key: str,
    model_name: str,
) -> tuple[str, list[CandidateBox], dict[str, Any] | None, str | None]:
    if language == 'zh':
        stage1_text = (
            '请观察这张大场景图，重点寻找可能是生物个体（尤其是小目标）的区域。'
            '先给出你的推理：包含可疑目标、外观线索（颜色/形状/姿态）和大致位置描述。'
            '这是定位任务，不是分类任务。禁止下结论到科/属/种；只能使用“候选目标A/B/...”这类中性指代。'
            '如果不确定，请明确不确定性。'
        )
        stage2_text = (
            '请根据上一阶段推理和同一张图，输出严格 JSON。'
            'JSON schema: {targets:[{label:string,confidence:number,bbox_norm:[x1,y1,x2,y2],clues:[string]}]}。'
            '其中 bbox_norm 使用 0-1 归一化坐标。'
            'label 必须使用中性命名，如 candidate_1/candidate_2，不得使用猫科、兔属等分类词。'
            '只输出 JSON，不要输出其他文本。JSON keyword required.'
        )
    else:
        stage1_text = (
            'Analyze this large-scene image and find possible biological individuals, especially small targets. '
            'First provide reasoning with candidate cues (color/shape/posture) and approximate locations. '
            'This is a localization task, not a taxonomy classification task. Do not assert family/genus/species names. '
            'Use neutral target references only, such as candidate_A/B. '
            'State uncertainty explicitly when needed.'
        )
        stage2_text = (
            'Based on the stage-1 reasoning and the same image, output strict JSON only. '
            'Schema: {targets:[{label:string,confidence:number,bbox_norm:[x1,y1,x2,y2],clues:[string]}]}. '
            'bbox_norm must use normalized coordinates in [0,1]. '
            'label must be neutral names like candidate_1/candidate_2, and must not contain taxonomy words. '
            'Output JSON only and no extra text. JSON keyword required.'
        )

    stage1_messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{image_mime};base64,{image_base64}'}},
                {'type': 'text', 'text': stage1_text},
            ],
        }
    ]

    ok1, stage1_output, _ = _post_chat(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        messages=stage1_messages,
        enable_thinking=True,
    )
    if not ok1:
        return '', [], None, stage1_output

    stage2_messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{image_mime};base64,{image_base64}'}},
                {
                    'type': 'text',
                    'text': f'Stage-1 reasoning:\n{stage1_output}\n\n{stage2_text}',
                },
            ],
        }
    ]
    ok2, stage2_output, _ = _post_chat(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        messages=stage2_messages,
        enable_thinking=False,
        response_format={'type': 'json_object'},
    )
    if not ok2:
        return stage1_output, [], None, stage2_output

    data = _extract_json(stage2_output)
    if data is None:
        return stage1_output, [], None, 'Stage-2 JSON parse failed'

    targets = data.get('targets', [])
    boxes: list[CandidateBox] = []
    if isinstance(targets, list):
        for item in targets:
            if not isinstance(item, dict):
                continue
            bbox = item.get('bbox_norm')
            if not isinstance(bbox, (list, tuple)):
                continue
            normalized = _normalize_bbox(list(bbox))
            if normalized is None:
                continue
            score = float(item.get('confidence', 0.5))
            label = str(item.get('label', 'unknown'))
            clues_raw = item.get('clues', [])
            clues = [str(x) for x in clues_raw] if isinstance(clues_raw, list) else []
            boxes.append(
                CandidateBox(
                    source='qwen',
                    score=max(0.0, min(1.0, score)),
                    label=label,
                    bbox_norm=normalized,
                    clues=clues,
                )
            )

    return stage1_output, boxes, data, None


def run_yolo_detector(
    *,
    image: Image.Image,
    model_path: str,
    conf: float = 0.12,
    iou: float = 0.55,
    imgsz: int = 1536,
    max_det: int = 25,
) -> tuple[list[CandidateBox], str | None]:
    try:
        # Fix for PyTorch 2.6+ weights_only default change
        import torch
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass  # Ignore if already patched or not needed

    try:
        from ultralytics import YOLO
    except Exception as exc:
        return [], f'ultralytics not available: {exc}'

    path = Path(model_path)
    if not path.exists():
        return [], f'YOLO weight not found: {model_path}'

    try:
        model = YOLO(str(path))
        arr = np.array(image)
        results = model.predict(source=arr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, verbose=False)
    except Exception as exc:
        return [], f'YOLO inference failed: {exc}'

    if not results:
        return [], None

    h, w = arr.shape[0], arr.shape[1]
    boxes_out: list[CandidateBox] = []
    boxes = results[0].boxes
    if boxes is None:
        return boxes_out, None

    names = getattr(model, 'names', {})
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].detach().cpu().numpy().tolist()
        score = float(boxes.conf[i].detach().cpu().item())
        cls_idx = int(boxes.cls[i].detach().cpu().item())
        label = str(names.get(cls_idx, 'animal'))

        normalized = _normalize_bbox([xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h])
        if normalized is None:
            continue

        boxes_out.append(
            CandidateBox(
                source='yolo',
                score=max(0.0, min(1.0, score)),
                label=label,
                bbox_norm=normalized,
                clues=['detector:animal'],
            )
        )

    return boxes_out, None


def _fallback_tile_boxes() -> list[CandidateBox]:
    tile_defs = [
        (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 0.62, 0.62),
        (0.38, 0.0, 1.0, 0.62),
        (0.0, 0.38, 0.62, 1.0),
        (0.38, 0.38, 1.0, 1.0),
        (0.2, 0.2, 0.8, 0.8),
    ]
    return [
        CandidateBox(source='tile', score=0.2, label='tile', bbox_norm=t, clues=['fallback-tile'])
        for t in tile_defs
    ]


def merge_candidate_boxes(
    qwen_boxes: list[CandidateBox],
    yolo_boxes: list[CandidateBox],
    max_boxes: int = 8,
    iou_threshold: float = 0.5,
) -> list[CandidateBox]:
    merged: list[CandidateBox] = []

    for cand in sorted(qwen_boxes + yolo_boxes, key=lambda x: x.score, reverse=True):
        duplicated = False
        for idx, exist in enumerate(merged):
            if _box_iou(cand.bbox_norm, exist.bbox_norm) >= iou_threshold:
                duplicated = True
                if cand.score > exist.score:
                    merged[idx] = CandidateBox(
                        source=f'{exist.source}+{cand.source}',
                        score=cand.score,
                        label=cand.label,
                        bbox_norm=cand.bbox_norm,
                        clues=list(dict.fromkeys(exist.clues + cand.clues)),
                    )
                break
        if not duplicated:
            merged.append(cand)
        if len(merged) >= max_boxes:
            break

    if not merged:
        merged = _fallback_tile_boxes()[:max_boxes]

    return merged


def crop_from_bbox(
    image: Image.Image,
    bbox_norm: tuple[float, float, float, float],
    expand_ratio: float = 0.18,
    min_size: int = 96,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    w, h = image.size
    x1, y1, x2, y2 = bbox_norm

    px1, py1 = int(round(x1 * w)), int(round(y1 * h))
    px2, py2 = int(round(x2 * w)), int(round(y2 * h))

    bw = max(1, px2 - px1)
    bh = max(1, py2 - py1)
    ex = int(round(bw * expand_ratio))
    ey = int(round(bh * expand_ratio))

    nx1 = max(0, px1 - ex)
    ny1 = max(0, py1 - ey)
    nx2 = min(w, px2 + ex)
    ny2 = min(h, py2 + ey)

    if nx2 - nx1 < min_size:
        cx = (nx1 + nx2) // 2
        half = min_size // 2
        nx1 = max(0, cx - half)
        nx2 = min(w, nx1 + min_size)
        nx1 = max(0, nx2 - min_size)

    if ny2 - ny1 < min_size:
        cy = (ny1 + ny2) // 2
        half = min_size // 2
        ny1 = max(0, cy - half)
        ny2 = min(h, ny1 + min_size)
        ny1 = max(0, ny2 - min_size)

    return image.crop((nx1, ny1, nx2, ny2)), (nx1, ny1, nx2, ny2)


def build_crops(
    image: Image.Image,
    fused_boxes: list[CandidateBox],
    max_crops: int = 6,
) -> list[dict[str, Any]]:
    crops: list[dict[str, Any]] = []

    crops.append(
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
    )

    for idx, cand in enumerate(fused_boxes[: max(0, max_crops - 1)], start=1):
        crop_img, bbox_px = crop_from_bbox(image, cand.bbox_norm)
        crops.append(
            {
                'id': f'crop_{idx}',
                'source': cand.source,
                'score': cand.score,
                'label': cand.label,
                'clues': cand.clues,
                'bbox_norm': cand.bbox_norm,
                'bbox_px': bbox_px,
                'image': crop_img,
            }
        )

    return crops


def candidate_to_dict(c: CandidateBox) -> dict[str, Any]:
    return {
        'source': c.source,
        'score': round(float(c.score), 4),
        'label': c.label,
        'bbox_norm': [round(float(v), 6) for v in c.bbox_norm],
        'clues': c.clues,
    }


def _box_from_dict(item: dict[str, Any]) -> CandidateBox | None:
    bbox = item.get('bbox_norm')
    if not isinstance(bbox, (list, tuple)):
        return None
    normalized = _normalize_bbox(list(bbox))
    if normalized is None:
        return None

    score = float(item.get('score', 0.5))
    label = str(item.get('label', 'unknown'))
    source = str(item.get('source', 'unknown'))
    clues_raw = item.get('clues', [])
    clues = [str(x) for x in clues_raw] if isinstance(clues_raw, list) else []
    return CandidateBox(
        source=source,
        score=max(0.0, min(1.0, score)),
        label=label,
        bbox_norm=normalized,
        clues=clues,
    )


def _get_interference_box_limit() -> int:
    raw = os.getenv('INTERFERENCE_BOX_LIMIT', '10').strip()
    try:
        limit = int(raw)
    except ValueError:
        limit = 10
    return max(1, limit)


def _get_interference_max_targets() -> int:
    raw = os.getenv('INTERFERENCE_MAX_TARGETS', '10').strip()
    try:
        limit = int(raw)
    except ValueError:
        limit = 10
    return max(1, min(20, limit))


def _trim_prompt_text(text: str, max_chars: int = 2000) -> str:
    content = str(text or '').strip()
    if not content:
        return 'none'
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + '...'


def run_interference_analysis_agent(
    *,
    image_base64: str,
    image_mime: str,
    language: str,
    base_url: str,
    api_key: str,
    model_name: str,
    localization_info: dict[str, Any] | None,
    bioclip_prior_text: str,
    taxonomy_constraint_text: str,
    enable_thinking: bool,
) -> dict[str, Any]:
    data = localization_info if isinstance(localization_info, dict) else {}

    fused_raw = data.get('fused_boxes', [])
    qwen_raw = data.get('qwen_boxes', [])
    yolo_raw = data.get('yolo_boxes', [])

    fused_boxes: list[CandidateBox] = []
    if isinstance(fused_raw, list):
        for item in fused_raw:
            if not isinstance(item, dict):
                continue
            box = _box_from_dict(item)
            if box is not None:
                fused_boxes.append(box)

    raw_candidates: list[CandidateBox] = []
    for source_list in (qwen_raw, yolo_raw):
        if not isinstance(source_list, list):
            continue
        for item in source_list:
            if not isinstance(item, dict):
                continue
            box = _box_from_dict(item)
            if box is not None:
                raw_candidates.append(box)

    def _is_species_candidate(c: CandidateBox) -> bool:
        src = c.source.lower()
        label = c.label.lower()
        return 'tile' not in src and label != 'tile'

    species_boxes = [x for x in fused_boxes if _is_species_candidate(x)]
    target_box_count = len(species_boxes)
    raw_species_count = len([x for x in raw_candidates if _is_species_candidate(x)])

    qwen_species_count = 0
    if isinstance(qwen_raw, list):
        for item in qwen_raw:
            if not isinstance(item, dict):
                continue
            box = _box_from_dict(item)
            if box is not None and _is_species_candidate(box):
                qwen_species_count += 1

    box_limit = _get_interference_box_limit()
    route = 'per_box'
    if qwen_species_count <= 0 or target_box_count > box_limit:
        route = 'full_image'

    targets: list[CandidateBox]
    if route == 'full_image':
        targets = [
            CandidateBox(
                source='full',
                score=1.0,
                label='full_scene',
                bbox_norm=(0.0, 0.0, 1.0, 1.0),
                clues=['route:full_image'],
            )
        ]
    else:
        targets = species_boxes[: _get_interference_max_targets()]

    target_text_lines: list[str] = []
    for idx, target in enumerate(targets, start=1):
        target_text_lines.append(
            f"{idx}. id=box_{idx}, source={target.source}, label={target.label}, "
            f"score={target.score:.3f}, bbox_norm={[round(v, 6) for v in target.bbox_norm]}, "
            f"clues={target.clues}"
        )
    target_text = '\n'.join(target_text_lines)
    prior_text = _trim_prompt_text(bioclip_prior_text)
    constraint_text = _trim_prompt_text(taxonomy_constraint_text)

    if language == 'zh':
        prompt = (
            '你是识别干扰因素分析Agent。目标：在最终分类前，分析识别风险。\n\n'
            f'路线: {route}\n'
            f'原始候选框数量(raw_species_count): {raw_species_count}\n'
            f'阈值(box_limit): {box_limit}\n'
            '[BioCLIP 预置分类建议]\n'
            f'{prior_text}\n\n'
            '[BioCLIP 层级约束]\n'
            f'{constraint_text}\n\n'
            '分析对象如下:\n'
            f'{target_text}\n\n'
            '请仅输出严格JSON，结构如下:\n'
            '{'
            '"route":"full_image|per_box",'
            '"global_summary":"string",'
            '"targets":['
            '{"id":"string","label":"string","bbox_norm":[x1,y1,x2,y2],"risk_score":0-100,'
            '"factors":['
            '{"name":"rare_pose|occlusion|color_cast|low_resolution|defocus_blur|motion_blur|exposure_issue|tiny_target|background_clutter|truncation|taxonomy_conflict",'
            '"severity":"low|medium|high","evidence":"string","impact":"string"}'
            '],"suggestion":"string"}'
            '],'
            '"recommendations":["string"]'
            '}\n'
            '要求:\n'
            '1) full_image路线时targets只保留一个full目标。\n'
            '2) per_box路线时对每个目标都给出factors。\n'
            '3) 覆盖以下干扰因素: 少见姿态、网状/面状遮挡、色度偏差、低分辨率、失焦模糊。\n'
            '4) 可补充其他实际干扰因素并给出证据。\n'
            '5) 本阶段不是最终分类，不得下结论到最终科/属/种；若涉及分类仅可表述为“可能冲突风险”。\n'
            '6) 如与 BioCLIP 层级约束存在冲突，请使用 taxonomy_conflict 因子记录，不要直接给出相反最终结论。\n'
            '7) 只输出JSON，不要附加文本。JSON keyword required.'
        )
    else:
        prompt = (
            'You are an interference-analysis agent before final species classification.\n\n'
            f'Route: {route}\n'
            f'Raw candidate count (species-like): {raw_species_count}\n'
            f'Box limit: {box_limit}\n'
            '[BioCLIP Prior Suggestions]\n'
            f'{prior_text}\n\n'
            '[BioCLIP Taxonomy Constraints]\n'
            f'{constraint_text}\n\n'
            'Targets:\n'
            f'{target_text}\n\n'
            'Output strict JSON only with schema:\n'
            '{'
            '"route":"full_image|per_box",'
            '"global_summary":"string",'
            '"targets":['
            '{"id":"string","label":"string","bbox_norm":[x1,y1,x2,y2],"risk_score":0-100,'
            '"factors":['
            '{"name":"rare_pose|occlusion|color_cast|low_resolution|defocus_blur|motion_blur|exposure_issue|tiny_target|background_clutter|truncation|taxonomy_conflict",'
            '"severity":"low|medium|high","evidence":"string","impact":"string"}'
            '],"suggestion":"string"}'
            '],'
            '"recommendations":["string"]'
            '}\n'
            'Rules:\n'
            '1) For full_image route, keep only one full target.\n'
            '2) For per_box route, analyze every target.\n'
            '3) Cover rare posture, net/planar occlusion, color cast, low resolution, defocus blur.\n'
            '4) Add other practical interference factors when needed.\n'
            '5) This phase is risk analysis, not final taxonomy classification; do not assert final family/genus/species.\n'
            '6) If there is conflict with BioCLIP taxonomy constraints, record it using taxonomy_conflict factor instead of asserting opposite final taxonomy.\n'
            '7) Output JSON only and no extra text. JSON keyword required.'
        )

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{image_mime};base64,{image_base64}'}},
                {'type': 'text', 'text': prompt},
            ],
        }
    ]

    ok, output_text, _ = _post_chat(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        messages=messages,
        enable_thinking=enable_thinking,
        response_format={'type': 'json_object'},
    )
    if not ok:
        return {
            'route': route,
            'raw_species_count': raw_species_count,
            'target_box_count': target_box_count,
            'qwen_species_count': qwen_species_count,
            'targets': [candidate_to_dict(x) for x in targets],
            'analysis_json': None,
            'error': output_text,
        }

    parsed = _extract_json(output_text)
    if parsed is None:
        return {
            'route': route,
            'raw_species_count': raw_species_count,
            'target_box_count': target_box_count,
            'qwen_species_count': qwen_species_count,
            'targets': [candidate_to_dict(x) for x in targets],
            'analysis_json': None,
            'error': 'Interference JSON parse failed',
            'raw_output': output_text,
        }

    return {
        'route': route,
        'raw_species_count': raw_species_count,
        'target_box_count': target_box_count,
        'qwen_species_count': qwen_species_count,
        'targets': [candidate_to_dict(x) for x in targets],
        'analysis_json': parsed,
        'error': None,
    }


def detect_and_prepare_crops(
    *,
    image: Image.Image,
    image_base64: str,
    image_mime: str,
    language: str,
    base_url: str,
    api_key: str,
    model_name: str,
    use_qwen_locator: bool,
    use_yolo_assist: bool,
    yolo_model_path: str,
    max_crops: int,
) -> dict[str, Any]:
    stage1_output = ''
    stage2_json: dict[str, Any] | None = None
    qwen_error: str | None = None
    yolo_error: str | None = None

    qwen_boxes: list[CandidateBox] = []
    if use_qwen_locator:
        stage1_output, qwen_boxes, stage2_json, qwen_error = run_qwen_two_stage_localization(
            image_base64=image_base64,
            image_mime=image_mime,
            language=language,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        )

    yolo_boxes: list[CandidateBox] = []
    if use_yolo_assist:
        yolo_boxes, yolo_error = run_yolo_detector(
            image=image,
            model_path=yolo_model_path,
        )

    fused = merge_candidate_boxes(qwen_boxes, yolo_boxes, max_boxes=max(1, max_crops))
    crops = build_crops(image, fused, max_crops=max(1, max_crops))

    return {
        'stage1_output': stage1_output,
        'stage2_json': stage2_json,
        'qwen_error': qwen_error,
        'yolo_error': yolo_error,
        'qwen_boxes': [candidate_to_dict(x) for x in qwen_boxes],
        'yolo_boxes': [candidate_to_dict(x) for x in yolo_boxes],
        'fused_boxes': [candidate_to_dict(x) for x in fused],
        'crops': crops,
    }
