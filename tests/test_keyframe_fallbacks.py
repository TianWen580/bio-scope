from __future__ import annotations

from services.video_processing.keyframe_extraction.strategy_contract import (
    KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
    KEYFRAME_STRATEGY_MECHANICAL,
    extract_keyframes_with_strategy,
)


def _legacy_frame(frame_id: int) -> dict[str, object]:
    return {
        'index': 1,
        'frame_id': frame_id,
        'timestamp_sec': float(frame_id),
        'image': object(),
    }


def test_invalid_strategy_downgrades_to_mechanical_with_legacy_shape() -> None:
    calls = {'mechanical': 0}

    def mechanical_extractor() -> tuple[list[dict[str, object]], str | None]:
        calls['mechanical'] += 1
        return [_legacy_frame(3)], None

    frames, err, used_strategy = extract_keyframes_with_strategy(
        strategy='unknown_strategy',
        mechanical_extractor=mechanical_extractor,
        non_default_extractors={
            KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY: lambda: ([_legacy_frame(99)], None),
        },
    )

    assert err is None
    assert used_strategy == KEYFRAME_STRATEGY_MECHANICAL
    assert calls['mechanical'] == 1
    assert frames[0]['frame_id'] == 3
    assert set(frames[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}


def test_non_default_extractor_error_falls_back_to_mechanical_with_legacy_shape() -> None:
    calls = {'mechanical': 0, 'bioclip2_consistency': 0}

    def mechanical_extractor() -> tuple[list[dict[str, object]], str | None]:
        calls['mechanical'] += 1
        return [_legacy_frame(8)], None

    def failing_non_default_extractor() -> tuple[list[dict[str, object]], str | None]:
        calls['bioclip2_consistency'] += 1
        return [], 'embedding failure'

    frames, err, used_strategy = extract_keyframes_with_strategy(
        strategy=KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY,
        mechanical_extractor=mechanical_extractor,
        non_default_extractors={
            KEYFRAME_STRATEGY_BIOCLIP2_CONSISTENCY: failing_non_default_extractor,
        },
    )

    assert err is None
    assert used_strategy == KEYFRAME_STRATEGY_MECHANICAL
    assert calls == {'mechanical': 1, 'bioclip2_consistency': 1}
    assert frames[0]['frame_id'] == 8
    assert set(frames[0].keys()) == {'index', 'frame_id', 'timestamp_sec', 'image'}
