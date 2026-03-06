from __future__ import annotations

from services.video_processing.keyframe_extraction.strategy_contract import (
    DEFAULT_KEYFRAME_STRATEGY,
    KEYFRAME_STRATEGY3_OUTPUT_CONTRACT,
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    KEYFRAME_STRATEGY_IDS,
    KEYFRAME_STRATEGY_MECHANICAL,
    KEYFRAME_STRATEGY_QWEN_VIDEO,
    extract_keyframes_with_strategy,
    parse_qwen_video_frame_positions,
    resolve_keyframe_strategy,
)


def test_strategy_ids_and_default() -> None:
    assert KEYFRAME_STRATEGY_IDS == (
        KEYFRAME_STRATEGY_MECHANICAL,
        KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        KEYFRAME_STRATEGY_QWEN_VIDEO,
    )
    assert DEFAULT_KEYFRAME_STRATEGY == KEYFRAME_STRATEGY_MECHANICAL
    assert KEYFRAME_STRATEGY3_OUTPUT_CONTRACT == '{"frame_positions":[{"frame_id":int}]}'

    assert resolve_keyframe_strategy(None) == KEYFRAME_STRATEGY_MECHANICAL
    assert resolve_keyframe_strategy('') == KEYFRAME_STRATEGY_MECHANICAL
    assert resolve_keyframe_strategy('qwen_video') == KEYFRAME_STRATEGY_QWEN_VIDEO
    assert resolve_keyframe_strategy('BIOCLIP2_CONSISTENCY') == KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY
    assert resolve_keyframe_strategy('unexpected') == KEYFRAME_STRATEGY_MECHANICAL

    assert parse_qwen_video_frame_positions({'frame_positions': [{'frame_id': 0}, {'frame_id': 9}]}) == [0, 9]
    assert parse_qwen_video_frame_positions({'frame_positions': [{'frame_id': '9'}]}) is None
    assert parse_qwen_video_frame_positions({'frame_positions': [{'frame_id': 9, 'x': 1}]}) is None


def test_non_default_failures_fallback_to_mechanical() -> None:
    mechanical_frame = {'index': 1, 'frame_id': 8, 'timestamp_sec': 0.32, 'image': object()}
    calls = {'mechanical': 0}

    def mechanical_extractor():
        calls['mechanical'] += 1
        return [mechanical_frame], None

    def qwen_failure_extractor():
        return [], 'qwen_video failed'

    frames, err, used_strategy = extract_keyframes_with_strategy(
        strategy=KEYFRAME_STRATEGY_QWEN_VIDEO,
        mechanical_extractor=mechanical_extractor,
        non_default_extractors={KEYFRAME_STRATEGY_QWEN_VIDEO: qwen_failure_extractor},
    )

    assert err is None
    assert used_strategy == KEYFRAME_STRATEGY_MECHANICAL
    assert calls['mechanical'] == 1
    assert len(frames) == 1
    assert set(frames[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}
