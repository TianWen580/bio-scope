from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from services.video_processing.keyframe_extraction.qwen_video_extractor import extract_qwen_video_keyframes  # pyright: ignore[reportMissingImports]
from services.video_processing.keyframe_extraction.strategy_contract import (
    KEYFRAME_STRATEGY_MECHANICAL,
    KEYFRAME_STRATEGY_QWEN_VIDEO,
    extract_keyframes_with_strategy,
)


class _UploadedFileStub:
    def __init__(self, payload: bytes, name: str = 'sample.mp4') -> None:
        self._payload = payload
        self.name = name

    def getbuffer(self) -> memoryview:
        return memoryview(self._payload)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any], text: str | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


def test_builds_video_payload_and_parses_json(monkeypatch) -> None:
    captured_payloads: list[dict[str, Any]] = []
    materialized: dict[str, Any] = {}

    def _fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = (url, headers, timeout)
        captured_payloads.append({'url': url, 'headers': headers, 'json': json, 'timeout': timeout})
        has_video_payload = any(item.get('type') == 'video_url' for item in json['messages'][0]['content'])
        if has_video_payload:
            return _FakeResponse(
                200,
                {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":3},{"frame_id":1}]}'}}]},
            )
        return _FakeResponse(
            200,
            {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":1},{"frame_id":3}]}'}}]},
        )

    def _fake_materializer(
        _uploaded_file: Any,
        frame_ids: Sequence[int],
        max_frames: int,
    ) -> tuple[list[dict[str, Any]], str | None]:
        materialized['frame_ids'] = list(frame_ids)
        materialized['max_frames'] = max_frames
        frames = [
            {'index': idx + 1, 'frame_id': frame_id, 'timestamp_sec': float(frame_id), 'image': object()}
            for idx, frame_id in enumerate(frame_ids)
        ]
        return frames, None

    monkeypatch.setattr('services.video_processing.keyframe_extraction.qwen_video_extractor.requests.post', _fake_post)

    frames, err = extract_qwen_video_keyframes(
        uploaded_file=_UploadedFileStub(b'fake-video-bytes'),
        base_url='https://dashscope.example/v1',
        api_key='test-key',
        model_name='qwen3.5-plus',
        request_timeout=30,
        keyframe_fps=2.0,
        max_candidate_frames=4,
        max_frames=4,
        frame_materializer=_fake_materializer,
        total_frames_resolver=lambda _uploaded_file: 24,
    )

    assert err is None
    assert [frame['frame_id'] for frame in frames] == [1, 3]
    assert materialized['frame_ids'] == [1, 3]
    assert materialized['max_frames'] == 4

    assert len(captured_payloads) == 2
    role1_payload = captured_payloads[0]['json']
    assert role1_payload['enable_thinking'] is False
    assert role1_payload['response_format'] == {'type': 'json_object'}
    assert role1_payload['messages'][0]['role'] == 'user'

    content = role1_payload['messages'][0]['content']
    video_item = next(item for item in content if item.get('type') == 'video_url')
    assert video_item['video_url']['url'].startswith('data:video/mp4;base64,')
    assert video_item['video_url']['fps'] == 2.0

    formatter_payload = captured_payloads[1]['json']
    formatter_content = formatter_payload['messages'][0]['content']
    assert all(item.get('type') == 'text' for item in formatter_content)
    assert 'raw_role1_output' in formatter_content[1]['text']


def test_formatter_invalid_output_uses_strict_retry_and_then_succeeds(monkeypatch) -> None:
    captured_payloads: list[dict[str, Any]] = []

    def _fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = (url, headers, timeout)
        captured_payloads.append(json)
        has_video_payload = any(item.get('type') == 'video_url' for item in json['messages'][0]['content'])
        if has_video_payload:
            return _FakeResponse(200, {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":4}]}'}}]})
        prompt = json['messages'][0]['content'][0]['text']
        if 'Reject any timestamps' in prompt:
            return _FakeResponse(200, {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":2}]}'}}]})
        return _FakeResponse(200, {'choices': [{'message': {'content': 'This video peaks around 00:12.'}}]})

    monkeypatch.setattr('services.video_processing.keyframe_extraction.qwen_video_extractor.requests.post', _fake_post)

    frames, err = extract_qwen_video_keyframes(
        uploaded_file=_UploadedFileStub(b'fake-video-bytes'),
        base_url='https://dashscope.example/v1',
        api_key='test-key',
        model_name='qwen3.5-plus',
        request_timeout=30,
        keyframe_fps=2.0,
        max_candidate_frames=4,
        max_frames=2,
        total_frames_resolver=lambda _uploaded_file: 32,
        frame_materializer=lambda _uploaded_file, frame_ids, _max_frames: (
            [
                {'index': 1, 'frame_id': int(frame_ids[0]), 'timestamp_sec': 0.1, 'image': object()},
            ],
            None,
        ),
    )

    assert err is None
    assert [frame['frame_id'] for frame in frames] == [2]
    assert len(captured_payloads) == 3
    assert 'Reject any timestamps' in captured_payloads[2]['messages'][0]['content'][0]['text']


def test_formatter_invalid_output_returns_normalized_error() -> None:
    call_count = {'count': 0}

    def _fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = (url, headers, timeout)
        call_count['count'] += 1
        has_video_payload = any(item.get('type') == 'video_url' for item in json['messages'][0]['content'])
        if has_video_payload:
            return _FakeResponse(200, {'choices': [{'message': {'content': '{"frame_positions":[{"frame_id":4}]}'}}]})
        return _FakeResponse(200, {'choices': [{'message': {'content': 'No exact frame ids available; sampled moments only.'}}]})

    frames, err = extract_qwen_video_keyframes(
        uploaded_file=_UploadedFileStub(b'fake-video-bytes'),
        base_url='https://dashscope.example/v1',
        api_key='test-key',
        model_name='qwen3.5-plus',
        request_timeout=30,
        keyframe_fps=2.0,
        max_candidate_frames=4,
        max_frames=2,
        total_frames_resolver=lambda _uploaded_file: 32,
        post_fn=_fake_post,
        frame_materializer=lambda _uploaded_file, _frame_ids, _max_frames: ([], None),
    )

    assert frames == []
    assert err == 'qwen_video_provider_error:video_keyframe_invalid_output_rejected'
    assert call_count['count'] == 6


def test_provider_error_falls_back_to_mechanical(monkeypatch) -> None:
    captured_payloads: list[dict[str, Any]] = []
    calls = {'mechanical': 0}

    def _fake_post(url: str, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        _ = (url, headers, timeout)
        captured_payloads.append(json)
        return _FakeResponse(500, {'error': 'provider failed'}, text='provider failed')

    monkeypatch.setattr('services.video_processing.keyframe_extraction.qwen_video_extractor.requests.post', _fake_post)

    def _mechanical_extractor():
        calls['mechanical'] += 1
        return [{'index': 1, 'frame_id': 8, 'timestamp_sec': 0.32, 'image': object()}], None

    def _qwen_extractor():
        return extract_qwen_video_keyframes(
            uploaded_file=_UploadedFileStub(b'fake-video-bytes'),
            base_url='https://dashscope.example/v1',
            api_key='test-key',
            model_name='qwen3.5-plus',
            request_timeout=30,
            keyframe_fps=2.0,
            max_candidate_frames=4,
            max_frames=4,
        )

    frames, err, used_strategy = extract_keyframes_with_strategy(
        strategy=KEYFRAME_STRATEGY_QWEN_VIDEO,
        mechanical_extractor=_mechanical_extractor,
        non_default_extractors={KEYFRAME_STRATEGY_QWEN_VIDEO: _qwen_extractor},
    )

    assert err is None
    assert used_strategy == KEYFRAME_STRATEGY_MECHANICAL
    assert calls['mechanical'] == 1
    assert len(frames) == 1

    fps_attempts = [payload['messages'][0]['content'][0]['video_url']['fps'] for payload in captured_payloads]
    assert fps_attempts == [2.0, 1.0]
