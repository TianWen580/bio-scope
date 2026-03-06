from __future__ import annotations

# pyright: basic

import ast
from pathlib import Path
from typing import Any

from services.video_processing.keyframe_extraction.qwen_video_extractor import extract_qwen_video_keyframes
from services.video_processing.keyframe_extraction.strategy_contract import (
    DEFAULT_KEYFRAME_STRATEGY,
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    KEYFRAME_STRATEGY_QWEN_VIDEO,
    extract_keyframes_with_strategy,
    resolve_keyframe_strategy,
)


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


class _UploadedFileStub:
    def __init__(self, payload: bytes, name: str = 'sample.mp4') -> None:
        self._payload = payload
        self.name = name

    def getbuffer(self) -> memoryview:
        return memoryview(self._payload)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any], text: str = '') -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._payload


def _load_extract_video_keyframes(*, mechanical_impl, bioclip_impl, qwen_impl):
    source = APP_PATH.read_text(encoding='utf-8')
    parsed = ast.parse(source, filename=str(APP_PATH))
    target = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == 'extract_video_keyframes'
    )
    subset = ast.Module(body=[target], type_ignores=[])
    subset = ast.fix_missing_locations(subset)

    scope: dict[str, Any] = {
        'Any': Any,
        'DEFAULT_KEYFRAME_STRATEGY': DEFAULT_KEYFRAME_STRATEGY,
        'KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY': KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        'KEYFRAME_STRATEGY_QWEN_VIDEO': KEYFRAME_STRATEGY_QWEN_VIDEO,
        '_extract_video_keyframes_mechanical': mechanical_impl,
        '_extract_video_keyframes_bioclip2_consistency': bioclip_impl,
        '_extract_video_keyframes_qwen_video': qwen_impl,
        'extract_keyframes_with_strategy': extract_keyframes_with_strategy,
        'resolve_keyframe_strategy': resolve_keyframe_strategy,
        'get_video_qwen_keyframe_fps': lambda: 1.0,
        'get_video_qwen_max_candidate_frames': lambda: 16,
        'get_video_bioclip_temporal_weight': lambda: 0.35,
        'get_video_bioclip_diversity_weight': lambda: 0.65,
    }
    exec(compile(subset, str(APP_PATH), 'exec'), scope)
    return scope['extract_video_keyframes']


def test_qwen_chain_valid_output_materializes_frames() -> None:
    captured_payloads: list[dict[str, Any]] = []
    materialized_ids: list[int] = []

    def _fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = (url, headers, timeout)
        captured_payloads.append(json)
        has_video_payload = any(item.get('type') == 'video_url' for item in json['messages'][0]['content'])
        if has_video_payload:
            return _FakeResponse(
                200,
                {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":5},{"frame_id":1}]}'}}]},
            )
        return _FakeResponse(
            200,
            {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":3},{"frame_id":1},{"frame_id":3}]}'}}]},
        )

    def _fake_materializer(_uploaded_file: Any, frame_ids: Any, max_frames: int) -> tuple[list[dict[str, Any]], str | None]:
        _ = max_frames
        materialized_ids.extend(int(x) for x in frame_ids)
        return (
            [
                {
                    'index': i,
                    'frame_id': int(frame_id),
                    'timestamp_sec': float(frame_id),
                    'image': object(),
                }
                for i, frame_id in enumerate(frame_ids, start=1)
            ],
            None,
        )

    frames, err = extract_qwen_video_keyframes(
        uploaded_file=_UploadedFileStub(b'video-bytes'),
        base_url='https://dashscope.example/v1',
        api_key='test-key',
        model_name='qwen3.5-plus',
        request_timeout=30,
        keyframe_fps=2.0,
        max_candidate_frames=6,
        max_frames=4,
        post_fn=_fake_post,
        frame_materializer=_fake_materializer,
        total_frames_resolver=lambda _uploaded_file: 32,
    )

    assert err is None
    assert [frame['frame_id'] for frame in frames] == [1, 3]
    assert materialized_ids == [1, 3]
    assert len(captured_payloads) == 2
    assert captured_payloads[0]['messages'][0]['content'][0]['type'] == 'video_url'
    assert captured_payloads[0]['messages'][0]['content'][0]['video_url']['fps'] == 2.0
    assert 'raw_role1_output' in captured_payloads[1]['messages'][0]['content'][1]['text']
    assert set(frames[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}


def test_invalid_chain_retries_then_falls_back() -> None:
    captured_payloads: list[dict[str, Any]] = []

    def _fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = (url, headers, timeout)
        captured_payloads.append(json)
        has_video_payload = any(item.get('type') == 'video_url' for item in json['messages'][0]['content'])
        if has_video_payload:
            return _FakeResponse(200, {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":4}]}'}}]})
        return _FakeResponse(200, {'choices': [{'message': {'content': 'Highlights around 00:12 and 15s.'}}]})

    def _qwen_fail(uploaded_file, interval_seconds: float, max_frames: int, **kwargs):
        _ = interval_seconds
        return extract_qwen_video_keyframes(
            uploaded_file=uploaded_file,
            base_url=kwargs['base_url'],
            api_key=kwargs['api_key'],
            model_name=kwargs['model_name'],
            request_timeout=kwargs['request_timeout'],
            keyframe_fps=kwargs['keyframe_fps'],
            max_candidate_frames=kwargs['max_candidate_frames'],
            max_frames=max_frames,
            post_fn=_fake_post,
            frame_materializer=lambda _file, _frame_ids, _max_frames: ([], None),
            total_frames_resolver=lambda _file: 32,
        )

    def _mechanical(*_args, **_kwargs):
        return ([{'index': 1, 'frame_id': 8, 'timestamp_sec': 0.32, 'image': object()}], None)

    extract_video_keyframes = _load_extract_video_keyframes(
        mechanical_impl=_mechanical,
        bioclip_impl=lambda *_args, **_kwargs: ([], 'not-used'),
        qwen_impl=_qwen_fail,
    )

    frames, err, warning_key = extract_video_keyframes(
        _UploadedFileStub(b'video-bytes'),
        interval_seconds=0.5,
        max_frames=4,
        strategy=KEYFRAME_STRATEGY_QWEN_VIDEO,
        qwen_keyframe_fps=2.0,
        qwen_max_candidate_frames=6,
        base_url='https://dashscope.example/v1',
        api_key='test-key',
        model_name='qwen3.5-plus',
        request_timeout=30,
        include_dispatch_metadata=True,
    )

    strict_retry_prompts = [
        payload['messages'][0]['content'][0]['text']
        for payload in captured_payloads
        if not any(item.get('type') == 'video_url' for item in payload['messages'][0]['content'])
        and 'Reject any timestamps' in payload['messages'][0]['content'][0]['text']
    ]

    assert err is None
    assert warning_key == 'video_keyframe_fallback_warning'
    assert frames[0]['frame_id'] == 8
    assert set(frames[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}
    assert len(captured_payloads) == 6
    assert len(strict_retry_prompts) == 2
