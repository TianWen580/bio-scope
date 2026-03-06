from __future__ import annotations

import base64
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, cast

from PIL import Image
import requests

from services.video_processing.keyframe_extraction.strategy_contract import run_qwen_role2_formatter_with_retry


def _response_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get('text'), str):
                    chunks.append(item['text'])
                elif isinstance(item.get('content'), str):
                    chunks.append(item['content'])
            elif isinstance(item, str):
                chunks.append(item)
        return '\n'.join(chunks).strip()
    return str(content)


def _guess_video_mime(filename: str | None) -> str:
    suffix = Path(filename or 'upload.mp4').suffix.lower()
    if suffix == '.mov':
        return 'video/quicktime'
    if suffix == '.webm':
        return 'video/webm'
    if suffix == '.mkv':
        return 'video/x-matroska'
    if suffix == '.avi':
        return 'video/x-msvideo'
    return 'video/mp4'


def _build_video_data_url(uploaded_file: Any) -> str | None:
    try:
        raw = bytes(uploaded_file.getbuffer())
    except Exception:
        return None
    if not raw:
        return None
    mime = _guess_video_mime(getattr(uploaded_file, 'name', None))
    encoded = base64.b64encode(raw).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def _build_fps_attempts(keyframe_fps: float) -> list[float]:
    primary = max(0.1, min(12.0, float(keyframe_fps)))
    reduced = max(0.1, round(primary * 0.5, 4))
    if reduced < primary:
        return [primary, reduced]
    return [primary]


def _build_messages(*, video_data_url: str, fps: float, max_candidate_frames: int) -> list[dict[str, Any]]:
    prompt = (
        'Analyze the uploaded video and return keyframe positions only. '
        f'Output strict JSON object with schema {{"frame_positions":[{{"frame_id":int}}]}} and no extra keys. '
        f'Return at most {max_candidate_frames} frame positions. '
        'frame_id must be a non-negative integer. '
        'Do not output timestamps (00:12, 12s), scene prose, markdown, or any text outside JSON.'
    )
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'video_url', 'video_url': {'url': video_data_url, 'fps': float(fps)}},
                {'type': 'text', 'text': prompt},
            ],
        }
    ]


def _build_formatter_messages(
    *,
    role1_raw_text: str,
    max_candidate_frames: int,
    strict_retry: bool,
) -> list[dict[str, Any]]:
    strict_clause = (
        'Reject any timestamps, durations, prose, scene descriptions, markdown, or abstract summaries. '
        'If uncertain, return {"frame_positions":[]}.'
        if strict_retry
        else 'Convert the source output to canonical frame positions.'
    )
    prompt = (
        'Role: formatter for keyframe selection. '
        'You receive raw role-1 output and must canonicalize to strict JSON schema '
        '{"frame_positions":[{"frame_id":int}]}. '
        f'Return at most {max_candidate_frames} items and no extra keys. '
        'frame_id must be a non-negative integer. '
        f'{strict_clause}'
    )
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'text', 'text': f'raw_role1_output:\n{role1_raw_text}'},
            ],
        }
    ]


def _resolve_total_frames(uploaded_file: Any) -> int | None:
    try:
        import cv2  # pyright: ignore[reportMissingImports]
    except Exception:
        return None

    suffix = Path(getattr(uploaded_file, 'name', 'upload.mp4') or 'upload.mp4').suffix or '.mp4'
    tmp_path = ''
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames if total_frames > 0 else None
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _post_chat(
    *,
    base_url: str,
    api_key: str,
    model_name: str,
    messages: list[dict[str, Any]],
    request_timeout: int,
    post_fn: Callable[..., Any] | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    url = base_url.rstrip('/') + '/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload: dict[str, Any] = {
        'model': model_name,
        'messages': messages,
        'enable_thinking': False,
        'response_format': {'type': 'json_object'},
    }

    sender = post_fn or requests.post
    try:
        response = sender(url, headers=headers, json=payload, timeout=request_timeout)
    except Exception as exc:
        return False, f'provider_request_failed:{exc}', None

    if response.status_code != 200:
        return False, f'provider_http_{response.status_code}', None

    try:
        data = response.json()
        return True, '', data
    except Exception as exc:
        return False, f'provider_invalid_json:{exc}', None


def _materialize_selected_frames(
    uploaded_file: Any,
    frame_ids: Sequence[int],
    max_frames: int,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        import cv2  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        return [], f'opencv_unavailable:{exc}'

    selected = sorted({int(x) for x in frame_ids if isinstance(x, int) and x >= 0})
    if not selected:
        return [], 'empty_frame_positions'
    selected = selected[: max(1, int(max_frames))]

    suffix = Path(getattr(uploaded_file, 'name', 'upload.mp4') or 'upload.mp4').suffix or '.mp4'
    tmp_path = ''
    frames: list[dict[str, Any]] = []

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return [], 'failed_to_open_video_stream'

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0

        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames_raw > 0:
            selected = [frame_id for frame_id in selected if frame_id < total_frames_raw]
            if not selected:
                cap.release()
                return [], 'frame_positions_out_of_range'

        target_set = set(selected)
        last_target = selected[-1]
        frame_idx = 0

        while frame_idx <= last_target and len(frames) < len(selected):
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx in target_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(
                    {
                        'index': len(frames) + 1,
                        'frame_id': frame_idx,
                        'timestamp_sec': float(frame_idx / fps),
                        'image': Image.fromarray(rgb),
                    }
                )
            frame_idx += 1

        cap.release()
        if not frames:
            return [], 'failed_to_materialize_selected_frames'
        return frames, None
    except Exception as exc:
        return [], f'frame_materialization_failed:{exc}'
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_qwen_video_keyframes(
    *,
    uploaded_file: Any,
    base_url: str,
    api_key: str,
    model_name: str,
    request_timeout: int,
    keyframe_fps: float,
    max_candidate_frames: int,
    max_frames: int,
    post_fn: Callable[..., Any] | None = None,
    frame_materializer: Callable[[Any, Sequence[int], int], tuple[list[dict[str, Any]], str | None]] | None = None,
    total_frames_resolver: Callable[[Any], int | None] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    video_data_url = _build_video_data_url(uploaded_file)
    if video_data_url is None:
        return [], 'qwen_video_payload_error:missing_video_bytes'

    candidate_cap = max(1, int(max_candidate_frames))
    selected_cap = max(1, int(max_frames))
    materializer = frame_materializer or _materialize_selected_frames
    total_frame_count = (total_frames_resolver or _resolve_total_frames)(uploaded_file)
    validator_total_frames = total_frame_count if isinstance(total_frame_count, int) and total_frame_count > 0 else 2**31 - 1

    attempts = _build_fps_attempts(keyframe_fps)
    last_error = 'qwen_video_provider_error:unknown'

    for fps in attempts:
        messages = _build_messages(
            video_data_url=video_data_url,
            fps=fps,
            max_candidate_frames=candidate_cap,
        )
        ok, req_error, data = _post_chat(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            messages=messages,
            request_timeout=request_timeout,
            post_fn=post_fn,
        )
        if not ok:
            last_error = f'qwen_video_provider_error:{req_error}'
            continue

        try:
            if not isinstance(data, dict):
                raise ValueError('response payload is not a JSON object')
            payload_data = cast(dict[str, Any], data)
            message = payload_data['choices'][0]['message']
            raw_text = _response_to_text(message.get('content', ''))
        except Exception as exc:
            last_error = f'qwen_video_provider_error:invalid_response_shape:{exc}'
            continue

        def _formatter_call(source_text: str, strict_retry: bool) -> object:
            formatter_messages = _build_formatter_messages(
                role1_raw_text=source_text,
                max_candidate_frames=candidate_cap,
                strict_retry=strict_retry,
            )
            formatter_ok, formatter_error, formatter_data = _post_chat(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                messages=formatter_messages,
                request_timeout=request_timeout,
                post_fn=post_fn,
            )
            if not formatter_ok:
                raise RuntimeError(formatter_error)
            if not isinstance(formatter_data, dict):
                return ''
            try:
                formatter_payload = cast(dict[str, Any], formatter_data)
                formatter_message = formatter_payload['choices'][0]['message']
                return _response_to_text(formatter_message.get('content', ''))
            except Exception:
                return ''

        try:
            canonical, formatter_error, should_fallback = run_qwen_role2_formatter_with_retry(
                formatter_call=_formatter_call,
                role1_output=raw_text,
                total_frames=validator_total_frames,
                video_max_frames=candidate_cap,
            )
        except Exception as exc:
            last_error = f'qwen_video_provider_error:{exc}'
            continue

        if should_fallback or canonical is None:
            normalized_formatter_error = formatter_error or 'invalid_frame_positions_schema'
            last_error = f'qwen_video_provider_error:{normalized_formatter_error}'
            continue

        normalized = [int(item['frame_id']) for item in canonical['frame_positions']][:candidate_cap]
        if not normalized:
            last_error = 'qwen_video_provider_error:empty_frame_positions'
            continue

        frames, materialize_error = materializer(uploaded_file, normalized, selected_cap)
        if materialize_error:
            last_error = f'qwen_video_provider_error:{materialize_error}'
            continue
        return frames, None

    return [], last_error
