from __future__ import annotations

from services.video_processing.keyframe_extraction.strategy_contract import (
    KEYFRAME_WARNING_INVALID_OUTPUT_REJECTED,
    normalize_qwen_role2_frame_positions,
    run_qwen_role2_formatter_with_retry,
)


def test_formatter_normalizes_to_frame_id_schema() -> None:
    calls: list[bool] = []

    def formatter_call(_source_text: str, strict_retry: bool):
        calls.append(strict_retry)
        return {'frame_positions': [{'frame_id': 9}, {'frame_id': 1}, {'frame_id': 9}, {'frame_id': 4}]}

    canonical, err, should_fallback = run_qwen_role2_formatter_with_retry(
        formatter_call=formatter_call,
        role1_output='{"raw": true}',
        total_frames=16,
        video_max_frames=3,
    )

    assert err is None
    assert should_fallback is False
    assert calls == [False]
    assert canonical == {'frame_positions': [{'frame_id': 1}, {'frame_id': 4}, {'frame_id': 9}]}


def test_rejects_timestamp_and_abstract_only_outputs() -> None:
    calls: list[bool] = []

    def formatter_call(_source_text: str, strict_retry: bool):
        calls.append(strict_retry)
        if not strict_retry:
            return '{"frame_positions":[{"frame_id":"00:12"}]}'
        return 'These scenes are representative of the overall temporal behavior.'

    canonical, err, should_fallback = run_qwen_role2_formatter_with_retry(
        formatter_call=formatter_call,
        role1_output='raw extractor output',
        total_frames=120,
        video_max_frames=8,
    )

    assert canonical is None
    assert err == KEYFRAME_WARNING_INVALID_OUTPUT_REJECTED
    assert should_fallback is True
    assert calls == [False, True]

    rejected, rejected_err = normalize_qwen_role2_frame_positions(
        'The animal appears intermittently and no exact frame ids are available.',
        total_frames=120,
        video_max_frames=8,
    )
    assert rejected is None
    assert rejected_err == KEYFRAME_WARNING_INVALID_OUTPUT_REJECTED
