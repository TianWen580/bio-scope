from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Callable, cast

from services.video_processing.keyframe_extraction.strategy_contract import (
    DEFAULT_KEYFRAME_STRATEGY,
    KEYFRAME_STRATEGY_MECHANICAL,
    resolve_keyframe_strategy,
)


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'
_GETTER_NAMES = (
    'get_video_keyframe_strategy',
    'get_video_qwen_keyframe_fps',
    'get_video_qwen_max_candidate_frames',
    'get_video_bioclip_temporal_weight',
    'get_video_bioclip_diversity_weight',
)


def _load_keyframe_getters() -> dict[str, Callable[[], object]]:
    source = APP_PATH.read_text(encoding='utf-8')
    parsed = ast.parse(source, filename=str(APP_PATH))
    function_nodes = {
        node.name: node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in _GETTER_NAMES
    }
    missing = [name for name in _GETTER_NAMES if name not in function_nodes]
    if missing:
        raise AssertionError(f'Missing getter(s) in app.py: {missing}')

    subset = ast.Module(body=[function_nodes[name] for name in _GETTER_NAMES], type_ignores=[])
    subset = ast.fix_missing_locations(subset)
    scope: dict[str, object] = {
        'os': os,
        'DEFAULT_KEYFRAME_STRATEGY': DEFAULT_KEYFRAME_STRATEGY,
        'resolve_keyframe_strategy': resolve_keyframe_strategy,
    }
    exec(compile(subset, str(APP_PATH), 'exec'), scope)
    return {name: cast(Callable[[], object], scope[name]) for name in _GETTER_NAMES}


def test_env_getters_clamp_and_defaults(monkeypatch) -> None:
    for key in (
        'VIDEO_QWEN_KEYFRAME_FPS',
        'VIDEO_QWEN_MAX_CANDIDATE_FRAMES',
        'VIDEO_BIOCLIP_TEMPORAL_WEIGHT',
        'VIDEO_BIOCLIP_DIVERSITY_WEIGHT',
    ):
        monkeypatch.delenv(key, raising=False)

    getters = _load_keyframe_getters()
    assert getters['get_video_qwen_keyframe_fps']() == 1.0
    assert getters['get_video_qwen_max_candidate_frames']() == 64
    assert getters['get_video_bioclip_temporal_weight']() == 0.35
    assert getters['get_video_bioclip_diversity_weight']() == 0.65

    monkeypatch.setenv('VIDEO_QWEN_KEYFRAME_FPS', '-9')
    assert getters['get_video_qwen_keyframe_fps']() == 0.1
    monkeypatch.setenv('VIDEO_QWEN_KEYFRAME_FPS', '999')
    assert getters['get_video_qwen_keyframe_fps']() == 12.0
    monkeypatch.setenv('VIDEO_QWEN_KEYFRAME_FPS', 'bad')
    assert getters['get_video_qwen_keyframe_fps']() == 1.0

    monkeypatch.setenv('VIDEO_QWEN_MAX_CANDIDATE_FRAMES', '-3')
    assert getters['get_video_qwen_max_candidate_frames']() == 1
    monkeypatch.setenv('VIDEO_QWEN_MAX_CANDIDATE_FRAMES', '9999')
    assert getters['get_video_qwen_max_candidate_frames']() == 512
    monkeypatch.setenv('VIDEO_QWEN_MAX_CANDIDATE_FRAMES', 'bad')
    assert getters['get_video_qwen_max_candidate_frames']() == 64

    monkeypatch.setenv('VIDEO_BIOCLIP_TEMPORAL_WEIGHT', '-0.5')
    assert getters['get_video_bioclip_temporal_weight']() == 0.0
    monkeypatch.setenv('VIDEO_BIOCLIP_TEMPORAL_WEIGHT', '1.7')
    assert getters['get_video_bioclip_temporal_weight']() == 1.0
    monkeypatch.setenv('VIDEO_BIOCLIP_TEMPORAL_WEIGHT', 'bad')
    assert getters['get_video_bioclip_temporal_weight']() == 0.35

    monkeypatch.setenv('VIDEO_BIOCLIP_DIVERSITY_WEIGHT', '-0.8')
    assert getters['get_video_bioclip_diversity_weight']() == 0.0
    monkeypatch.setenv('VIDEO_BIOCLIP_DIVERSITY_WEIGHT', '2')
    assert getters['get_video_bioclip_diversity_weight']() == 1.0
    monkeypatch.setenv('VIDEO_BIOCLIP_DIVERSITY_WEIGHT', 'bad')
    assert getters['get_video_bioclip_diversity_weight']() == 0.65


def test_invalid_strategy_value_defaults_to_mechanical(monkeypatch) -> None:
    monkeypatch.setenv('VIDEO_KEYFRAME_STRATEGY', 'not_a_real_strategy')
    getter = _load_keyframe_getters()['get_video_keyframe_strategy']
    assert getter() == KEYFRAME_STRATEGY_MECHANICAL
