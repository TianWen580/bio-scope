from __future__ import annotations

from dataclasses import dataclass
import json
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
    timeout: int = 120,
) -> tuple[bool, str, dict[str, Any] | None]:
    url = base_url.rstrip('/') + '/chat/completions'
    payload: dict[str, Any] = {
        'model': model_name,
        'messages': messages,
        'enable_thinking': enable_thinking,
    }
    if response_format is not None:
        payload['response_format'] = response_format
    if thinking_budget is not None:
        payload['thinking_budget'] = int(thinking_budget)

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:
        return False, f'HTTP request failed: {exc}', None

    if response.status_code != 200:
        return False, f'HTTP {response.status_code}: {response.text[:1000]}', None

    try:
        data = response.json()
        message = data['choices'][0]['message']
        text = _response_content_to_text(message.get('content', ''))
        return True, text, data
    except Exception as exc:
        return False, f'Invalid response format: {exc}', None


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
            '如果不确定，请明确不确定性。'
        )
        stage2_text = (
            '请根据上一阶段推理和同一张图，输出严格 JSON。'
            'JSON schema: {targets:[{label:string,confidence:number,bbox_norm:[x1,y1,x2,y2],clues:[string]}]}。'
            '其中 bbox_norm 使用 0-1 归一化坐标。'
            '只输出 JSON，不要输出其他文本。JSON keyword required.'
        )
    else:
        stage1_text = (
            'Analyze this large-scene image and find possible biological individuals, especially small targets. '
            'First provide reasoning with candidate cues (color/shape/posture) and approximate locations. '
            'State uncertainty explicitly when needed.'
        )
        stage2_text = (
            'Based on the stage-1 reasoning and the same image, output strict JSON only. '
            'Schema: {targets:[{label:string,confidence:number,bbox_norm:[x1,y1,x2,y2],clues:[string]}]}. '
            'bbox_norm must use normalized coordinates in [0,1]. '
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
