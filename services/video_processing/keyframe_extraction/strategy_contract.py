from __future__ import annotations
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

import json
import re
from collections.abc import Mapping
from typing import Callable

KEYFRAME_STRATEGY_MECHANICAL = 'mechanical'
KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY = 'bioclip2_consistency'
KEYFRAME_STRATEGY_QWEN_VIDEO = 'qwen_video'

KEYFRAME_STRATEGY_IDS = (
    KEYFRAME_STRATEGY_MECHANICAL,
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    KEYFRAME_STRATEGY_QWEN_VIDEO,
)
DEFAULT_KEYFRAME_STRATEGY = KEYFRAME_STRATEGY_MECHANICAL

KEYFRAME_STRATEGY3_OUTPUT_CONTRACT = '{"frame_positions":[{"frame_id":int}]}'

KEYFRAME_WARNING_FORMATTER_VALIDATION = 'video_keyframe_formatter_validation_error'
KEYFRAME_WARNING_INVALID_OUTPUT_REJECTED = 'video_keyframe_invalid_output_rejected'

_TIMESTAMP_TOKEN_RE = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')
_DURATION_TOKEN_RE = re.compile(
    r'\b\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|second|seconds|min|mins|minute|minutes|h|hr|hrs|hour|hours)\b',
    flags=re.IGNORECASE,
)


def resolve_keyframe_strategy(value: str | None) -> str:
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in KEYFRAME_STRATEGY_IDS:
            return candidate
    return DEFAULT_KEYFRAME_STRATEGY


def _response_content_to_text(content: object) -> str:
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


def _extract_json_object(text: str) -> dict[str, object] | None:
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
        parsed: object = json.loads(stripped)  # pyright: ignore[reportAny]
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
        return None
    except Exception:
        pass

    first = stripped.find('{')
    last = stripped.rfind('}')
    if first < 0 or last <= first:
        return None

    try:
        parsed_tail: object = json.loads(stripped[first:last + 1])  # pyright: ignore[reportAny]
        if isinstance(parsed_tail, dict):
            return {str(k): v for k, v in parsed_tail.items()}
        return None
    except Exception:
        return None


def _contains_time_like_token(raw: object) -> bool:
    text = _response_content_to_text(raw)
    if not text:
        return False
    return bool(_TIMESTAMP_TOKEN_RE.search(text) or _DURATION_TOKEN_RE.search(text))


def parse_qwen_video_frame_positions(
    payload: object,
    *,
    total_frames: int | None = None,
    video_max_frames: int | None = None,
) -> list[int] | None:
    if _contains_time_like_token(payload):
        return None

    if isinstance(payload, str):
        parsed_payload = _extract_json_object(payload)
        if parsed_payload is None:
            return None
        payload = parsed_payload

    if not isinstance(payload, dict):
        return None
    frame_positions = payload.get('frame_positions')
    if not isinstance(frame_positions, list):
        return None

    out: list[int] = []
    for item in frame_positions:
        frame_id: object
        if isinstance(item, dict):
            if set(item.keys()) != {'frame_id'}:
                return None
            frame_id = item.get('frame_id')
        else:
            frame_id = item

        if not isinstance(frame_id, int) or isinstance(frame_id, bool):
            return None
        if isinstance(total_frames, int) and frame_id >= total_frames:
            return None
        if frame_id < 0:
            return None
        out.append(frame_id)

    if not out:
        return None

    normalized = sorted(set(out))
    if isinstance(video_max_frames, int):
        normalized = normalized[: max(0, video_max_frames)]
    return normalized


def normalize_qwen_role2_frame_positions(
    payload: object,
    *,
    total_frames: int,
    video_max_frames: int,
) -> tuple[dict[str, list[dict[str, int]]] | None, str | None]:
    frame_ids = parse_qwen_video_frame_positions(
        payload,
        total_frames=total_frames,
        video_max_frames=video_max_frames,
    )
    if frame_ids is None:
        return None, KEYFRAME_WARNING_INVALID_OUTPUT_REJECTED

    return ({'frame_positions': [{'frame_id': frame_id} for frame_id in frame_ids]}, None)


def run_qwen_role2_formatter_with_retry(
    *,
    formatter_call: Callable[[str, bool], object],
    role1_output: object,
    total_frames: int,
    video_max_frames: int,
) -> tuple[dict[str, list[dict[str, int]]] | None, str | None, bool]:
    source_text = _response_content_to_text(role1_output)
    validation_error: str | None = KEYFRAME_WARNING_FORMATTER_VALIDATION

    for strict_retry in (False, True):
        formatter_output = formatter_call(source_text, strict_retry)
        canonical, err = normalize_qwen_role2_frame_positions(
            formatter_output,
            total_frames=total_frames,
            video_max_frames=video_max_frames,
        )
        if canonical is not None:
            return canonical, None, False
        if err:
            validation_error = err

    return None, validation_error, True


def extract_keyframes_with_strategy(
    *,
    strategy: str | None,
    mechanical_extractor: Callable[[], tuple[list[dict[str, object]], str | None]],
    non_default_extractors: Mapping[str, Callable[[], tuple[list[dict[str, object]], str | None]]] | None = None,
) -> tuple[list[dict[str, object]], str | None, str]:
    resolved = resolve_keyframe_strategy(strategy)
    if resolved == DEFAULT_KEYFRAME_STRATEGY:
        frames, err = mechanical_extractor()
        return frames, err, DEFAULT_KEYFRAME_STRATEGY

    extractor = (non_default_extractors or {}).get(resolved)
    if extractor is None:
        frames, err = mechanical_extractor()
        return frames, err, DEFAULT_KEYFRAME_STRATEGY

    try:
        frames, err = extractor()
        if err:
            raise RuntimeError(err)
        return frames, None, resolved
    except Exception:
        frames, err = mechanical_extractor()
        return frames, err, DEFAULT_KEYFRAME_STRATEGY
