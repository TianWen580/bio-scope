from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from services.video_processing.keyframe_extraction.strategy_contract import (
    DEFAULT_KEYFRAME_STRATEGY,
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    KEYFRAME_STRATEGY_QWEN_VIDEO,
    extract_keyframes_with_strategy,
    resolve_keyframe_strategy,
)


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def _load_extract_video_keyframes(
    *,
    mechanical_impl,
    bioclip_impl,
    qwen_impl,
):
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


def _make_frame(frame_id: int) -> dict[str, Any]:
    return {
        'index': 1,
        'frame_id': frame_id,
        'timestamp_sec': float(frame_id),
        'image': object(),
    }


def test_dispatcher_routes_all_three_strategies() -> None:
    calls = {'mechanical': 0, 'bioclip2_consistency': 0, 'qwen_video': 0}

    def _mechanical(*_args, **_kwargs):
        calls['mechanical'] += 1
        return [_make_frame(10)], None

    def _bioclip(*_args, **_kwargs):
        calls['bioclip2_consistency'] += 1
        return [_make_frame(20)], None

    def _qwen(*_args, **_kwargs):
        calls['qwen_video'] += 1
        return [_make_frame(30)], None

    extract_video_keyframes = _load_extract_video_keyframes(
        mechanical_impl=_mechanical,
        bioclip_impl=_bioclip,
        qwen_impl=_qwen,
    )

    selected_mechanical, mechanical_err, mechanical_warning = extract_video_keyframes(
        object(),
        interval_seconds=2.0,
        max_frames=5,
        strategy=DEFAULT_KEYFRAME_STRATEGY,
        include_dispatch_metadata=True,
    )
    selected_bioclip, bioclip_err, bioclip_warning = extract_video_keyframes(
        object(),
        interval_seconds=2.0,
        max_frames=5,
        strategy=KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        include_dispatch_metadata=True,
    )
    selected_qwen, qwen_err, qwen_warning = extract_video_keyframes(
        object(),
        interval_seconds=2.0,
        max_frames=5,
        strategy=KEYFRAME_STRATEGY_QWEN_VIDEO,
        include_dispatch_metadata=True,
    )

    assert calls == {'mechanical': 1, 'bioclip2_consistency': 1, 'qwen_video': 1}
    assert mechanical_err is None and bioclip_err is None and qwen_err is None
    assert mechanical_warning is None and bioclip_warning is None and qwen_warning is None
    assert selected_mechanical[0]['frame_id'] == 10
    assert selected_bioclip[0]['frame_id'] == 20
    assert selected_qwen[0]['frame_id'] == 30
    assert set(selected_mechanical[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}
    assert set(selected_bioclip[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}
    assert set(selected_qwen[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}


def test_dispatcher_surfaces_fallback_warning() -> None:
    calls = {'mechanical': 0, 'qwen_video': 0}

    def _mechanical(*_args, **_kwargs):
        calls['mechanical'] += 1
        return [_make_frame(8)], None

    def _qwen_fail(*_args, **_kwargs):
        calls['qwen_video'] += 1
        return [], 'provider failure'

    extract_video_keyframes = _load_extract_video_keyframes(
        mechanical_impl=_mechanical,
        bioclip_impl=lambda *_args, **_kwargs: ([_make_frame(2)], None),
        qwen_impl=_qwen_fail,
    )

    frames, err, warning_key = extract_video_keyframes(
        object(),
        interval_seconds=1.0,
        max_frames=4,
        strategy=KEYFRAME_STRATEGY_QWEN_VIDEO,
        include_dispatch_metadata=True,
    )

    assert err is None
    assert warning_key == 'video_keyframe_fallback_warning'
    assert calls == {'mechanical': 1, 'qwen_video': 1}
    assert frames[0]['frame_id'] == 8
    assert set(frames[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}


def test_invalid_strategy_is_downgraded() -> None:
    calls = {'mechanical': 0, 'bioclip2_consistency': 0, 'qwen_video': 0}

    def _mechanical(*_args, **_kwargs):
        calls['mechanical'] += 1
        return [_make_frame(3)], None

    def _bioclip(*_args, **_kwargs):
        calls['bioclip2_consistency'] += 1
        return [_make_frame(4)], None

    def _qwen(*_args, **_kwargs):
        calls['qwen_video'] += 1
        return [_make_frame(5)], None

    extract_video_keyframes = _load_extract_video_keyframes(
        mechanical_impl=_mechanical,
        bioclip_impl=_bioclip,
        qwen_impl=_qwen,
    )

    frames, err, warning_key = extract_video_keyframes(
        object(),
        interval_seconds=1.5,
        max_frames=6,
        strategy='unknown_strategy',
        include_dispatch_metadata=True,
    )

    assert err is None
    assert warning_key is None
    assert calls == {'mechanical': 1, 'bioclip2_consistency': 0, 'qwen_video': 0}
    assert frames[0]['frame_id'] == 3
    assert set(frames[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}
