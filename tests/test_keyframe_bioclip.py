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
from services.video_processing.keyframe_extraction.qwen_video_extractor import _materialize_selected_frames

from services.video_processing.keyframe_extraction.strategy_contract import (
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    extract_keyframes_with_strategy,
)


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'
_FUNCTION_NAMES = (
    '_extract_video_keyframes_mechanical',
    '_extract_video_keyframes_bioclip2_consistency',
    '_extract_video_keyframes_qwen_video',
    'extract_video_keyframes',
)


def _load_video_keyframe_functions(encode_image_impl, load_bioclip_model_impl):
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
        'np': np,
        'os': os,
        'tempfile': tempfile,
        'encode_image': encode_image_impl,
        'extract_keyframes_with_strategy': extract_keyframes_with_strategy,
        'get_video_bioclip_diversity_weight': lambda: 0.65,
        'get_video_bioclip_temporal_weight': lambda: 0.35,
        'get_video_qwen_keyframe_fps': lambda: 1.0,
        'get_video_qwen_max_candidate_frames': lambda: 32,
        'KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY': KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        'KEYFRAME_STRATEGY_QWEN_VIDEO': 'qwen_video',
        'load_bioclip_model': load_bioclip_model_impl,
    }
    exec(compile(subset, str(APP_PATH), 'exec'), scope)
    return {name: scope[name] for name in _FUNCTION_NAMES}


def _install_fake_cv2(monkeypatch, frames, fps: float = 1.0, read_counter: dict[str, int] | None = None) -> None:
    class FakeCapture:
        def __init__(self, _path: str):
            self._frames = [f.copy() for f in frames]
            self._idx = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop):
            if prop == 5:
                return fps
            if prop == 7:
                return len(self._frames)
            return 0.0

        def read(self):
            if read_counter is not None:
                read_counter['count'] = read_counter.get('count', 0) + 1
            if self._idx >= len(self._frames):
                return False, None
            frame = self._frames[self._idx]
            self._idx += 1
            return True, frame

        def release(self) -> None:
            return None

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        COLOR_BGR2RGB=1,
        VideoCapture=FakeCapture,
        cvtColor=lambda frame, _code: frame,
    )
    monkeypatch.setitem(sys.modules, 'cv2', fake_cv2)


class _UploadedFile:
    name = 'test.mp4'

    def getbuffer(self):
        return b'fake-video-content'


def test_temporal_diversity_returns_sorted_unique_positions(monkeypatch) -> None:
    frames = [np.full((8, 8, 3), fill_value=i * 15, dtype=np.uint8) for i in range(12)]
    _install_fake_cv2(monkeypatch, frames, fps=1.0)

    def encode_image_stub(image: Image.Image, *_args):
        value = float(np.asarray(image, dtype='float32').mean())
        return np.array([[value + 1.0, (value * 0.3) + 2.0, (value % 11.0) + 0.5]], dtype='float32')

    functions = _load_video_keyframe_functions(
        encode_image_impl=encode_image_stub,
        load_bioclip_model_impl=lambda **_kwargs: (object(), object(), 'cpu'),
    )
    extract_video_keyframes = functions['extract_video_keyframes']

    selected, err = extract_video_keyframes(
        _UploadedFile(),
        interval_seconds=1.0,
        max_frames=5,
        strategy=KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    )

    assert err is None
    assert len(selected) <= 5
    frame_positions = [int(item['frame_id']) for item in selected]
    assert frame_positions == sorted(set(frame_positions))
    assert all(0 <= pos < len(frames) for pos in frame_positions)
    assert [int(item['index']) for item in selected] == [1, 2, 3, 4, 5]


def test_degenerate_embeddings_fallback_to_mechanical(monkeypatch) -> None:
    frames = [np.full((8, 8, 3), fill_value=i * 10, dtype=np.uint8) for i in range(10)]
    _install_fake_cv2(monkeypatch, frames, fps=1.0)

    functions = _load_video_keyframe_functions(
        encode_image_impl=lambda *_args, **_kwargs: np.array([[np.nan, 1.0, 2.0]], dtype='float32'),
        load_bioclip_model_impl=lambda **_kwargs: (object(), object(), 'cpu'),
    )

    extract_video_keyframes = functions['extract_video_keyframes']
    mechanical = functions['_extract_video_keyframes_mechanical']

    expected_frames, expected_err = mechanical(_UploadedFile(), interval_seconds=1.0, max_frames=4)
    assert expected_err is None

    selected, err = extract_video_keyframes(
        _UploadedFile(),
        interval_seconds=1.0,
        max_frames=4,
        strategy=KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    )

    assert err is None
    assert [int(item['frame_id']) for item in selected] == [int(item['frame_id']) for item in expected_frames]
    assert [int(item['index']) for item in selected] == [1, 2, 3, 4]


def test_out_of_range_positions_are_filtered(monkeypatch) -> None:
    frames = [np.full((6, 6, 3), fill_value=i * 8, dtype=np.uint8) for i in range(8)]
    read_counter = {'count': 0}
    _install_fake_cv2(monkeypatch, frames, fps=2.0, read_counter=read_counter)

    selected, err = _materialize_selected_frames(
        _UploadedFile(),
        frame_ids=[2, 50, 4],
        max_frames=5,
    )

    assert err is None
    frame_positions = [int(item['frame_id']) for item in selected]
    assert frame_positions == [2, 4]
    assert all(0 <= pos < len(frames) for pos in frame_positions)
    assert read_counter['count'] == 5
