from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from services.video_processing.keyframe_extraction.strategy_contract import (
    DEFAULT_KEYFRAME_STRATEGY,
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    KEYFRAME_STRATEGY_QWEN_VIDEO,
    extract_keyframes_with_strategy,
)


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'
_FUNCTION_NAMES = (
    '_extract_video_keyframes_mechanical',
    'extract_video_keyframes',
)


def _load_mechanical_functions():
    source = APP_PATH.read_text(encoding='utf-8')
    parsed = ast.parse(source, filename=str(APP_PATH))
    function_nodes = {
        node.name: node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCTION_NAMES
    }
    subset = ast.Module(body=[function_nodes[name] for name in _FUNCTION_NAMES], type_ignores=[])
    subset = ast.fix_missing_locations(subset)

    scope: dict[str, Any] = {
        'Any': Any,
        'Image': Image,
        'Path': Path,
        'os': os,
        'tempfile': tempfile,
        'DEFAULT_KEYFRAME_STRATEGY': DEFAULT_KEYFRAME_STRATEGY,
        'KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY': KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        'KEYFRAME_STRATEGY_QWEN_VIDEO': KEYFRAME_STRATEGY_QWEN_VIDEO,
        'extract_keyframes_with_strategy': extract_keyframes_with_strategy,
        'get_video_qwen_keyframe_fps': lambda: 1.0,
        'get_video_qwen_max_candidate_frames': lambda: 16,
        'get_video_bioclip_temporal_weight': lambda: 0.35,
        'get_video_bioclip_diversity_weight': lambda: 0.65,
        '_extract_video_keyframes_bioclip2_consistency': lambda *_args, **_kwargs: ([], 'unused'),
        '_extract_video_keyframes_qwen_video': lambda *_args, **_kwargs: ([], 'unused'),
    }
    exec(compile(subset, str(APP_PATH), 'exec'), scope)
    return {name: scope[name] for name in _FUNCTION_NAMES}


def _install_fake_cv2(monkeypatch, frames: list[Any], fps: float) -> None:
    class FakeCapture:
        def __init__(self, _path: str):
            self._frames = [frame.copy() for frame in frames]
            self._idx = 0

        def isOpened(self) -> bool:
            return True

        def get(self, _prop):
            return fps

        def read(self):
            if self._idx >= len(self._frames):
                return False, None
            frame = self._frames[self._idx]
            self._idx += 1
            return True, frame

        def release(self) -> None:
            return None

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=5,
        COLOR_BGR2RGB=1,
        VideoCapture=FakeCapture,
        cvtColor=lambda frame, _code: frame,
    )
    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)


class _UploadedFile:
    name = 'test.mp4'

    def getbuffer(self):
        return b'fake-video-content'


def _legacy_interval_positions(*, total_frames: int, fps: float, interval_seconds: float, max_frames: int) -> list[int]:
    step = max(1, int(round(fps * interval_seconds)))
    selected: list[int] = []
    frame_idx = 0
    while frame_idx < total_frames and len(selected) < max_frames:
        if frame_idx % step == 0:
            selected.append(frame_idx)
        frame_idx += 1
    return selected


def test_default_mechanical_matches_legacy_interval(monkeypatch) -> None:
    fps = 2.4
    interval_seconds = 1.5
    max_frames = 5
    frames = [np.full((8, 8, 3), fill_value=i * 9, dtype=np.uint8) for i in range(18)]
    _install_fake_cv2(monkeypatch, frames=frames, fps=fps)

    functions = _load_mechanical_functions()
    direct_mechanical = functions['_extract_video_keyframes_mechanical']
    extract_video_keyframes = functions['extract_video_keyframes']

    expected_positions = _legacy_interval_positions(
        total_frames=len(frames),
        fps=fps,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )

    mechanical_frames, mechanical_err = direct_mechanical(
        _UploadedFile(),
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )
    selected_frames, selected_err = extract_video_keyframes(
        _UploadedFile(),
        interval_seconds=interval_seconds,
        max_frames=max_frames,
        strategy=DEFAULT_KEYFRAME_STRATEGY,
    )

    assert mechanical_err is None
    assert selected_err is None

    mechanical_positions = [int(item['frame_id']) for item in mechanical_frames]
    selected_positions = [int(item['frame_id']) for item in selected_frames]
    assert mechanical_positions == expected_positions
    assert selected_positions == expected_positions
    assert [int(item['index']) for item in selected_frames] == [1, 2, 3, 4, 5]
    assert np.allclose(
        [float(item['timestamp_sec']) for item in selected_frames],
        [frame_id / fps for frame_id in expected_positions],
    )
