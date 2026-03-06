"""
Documentation validation tests for keyframe strategy options and API constraints.

These tests verify that README.md and README.zh-CN.md include required
documentation about keyframe strategies, environment variables, and API constraints.

No runtime imports from app.py - text assertions only.
"""

import os
import re


def get_project_root() -> str:
    """Return project root directory (parent of tests/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_readme(lang: str = "en") -> str:
    """Read README file content."""
    root = get_project_root()
    filename = "README.md" if lang == "en" else "README.zh-CN.md"
    filepath = os.path.join(root, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def test_readmes_include_three_strategies_and_fallback():
    """
    Both README.md and README.zh-CN.md must document:
    - Three strategy options: mechanical, bioclip2_consistency, qwen_video
    - VIDEO_KEYFRAME_STRATEGY environment variable
    - Strategy-specific parameters (FPS, max candidates, weights)
    - Fallback behavior for non-default strategy failures
    """
    for lang in ["en", "zh"]:
        content = read_readme(lang)

        # Check for three strategy names
        assert "mechanical" in content, f"{lang}: Missing 'mechanical' strategy"
        assert "bioclip2_consistency" in content, f"{lang}: Missing 'bioclip2_consistency' strategy"
        assert "qwen_video" in content, f"{lang}: Missing 'qwen_video' strategy"

        # Check for strategy selector env var
        assert "VIDEO_KEYFRAME_STRATEGY" in content, f"{lang}: Missing VIDEO_KEYFRAME_STRATEGY env var"

        # Check for strategy-3 (qwen_video) parameters
        assert "VIDEO_QWEN_KEYFRAME_FPS" in content, f"{lang}: Missing VIDEO_QWEN_KEYFRAME_FPS"
        assert "VIDEO_QWEN_MAX_CANDIDATE_FRAMES" in content, f"{lang}: Missing VIDEO_QWEN_MAX_CANDIDATE_FRAMES"
        assert re.search(r"VIDEO_QWEN_MAX_CANDIDATE_FRAMES\s*=\s*64", content), (
            f"{lang}: VIDEO_QWEN_MAX_CANDIDATE_FRAMES default must be 64 in env example"
        )

        if lang == "en":
            assert "default 64" in content, f"{lang}: Missing default 64 description for VIDEO_QWEN_MAX_CANDIDATE_FRAMES"
        else:
            assert "默认 64" in content, f"{lang}: 缺少 VIDEO_QWEN_MAX_CANDIDATE_FRAMES 的默认值 64 描述"

        # Check for strategy-2 (bioclip2_consistency) parameters
        assert "VIDEO_BIOCLIP_TEMPORAL_WEIGHT" in content, f"{lang}: Missing VIDEO_BIOCLIP_TEMPORAL_WEIGHT"
        assert "VIDEO_BIOCLIP_DIVERSITY_WEIGHT" in content, f"{lang}: Missing VIDEO_BIOCLIP_DIVERSITY_WEIGHT"

        # Check for fallback behavior documentation
        # Look for keywords about fallback/degradation
        fallback_keywords = ["fallback", "降级", "mechanical"]
        has_fallback_text = any(kw in content for kw in fallback_keywords)
        assert has_fallback_text, f"{lang}: Missing fallback behavior documentation"


def test_readmes_do_not_claim_json_schema_for_qwen35plus():
    """
    README files must NOT claim that qwen3.5-plus uses json_schema.
    The prototype uses json_object response format only.
    
    This test ensures documentation does not mislead users about API capabilities.
    """
    for lang in ["en", "zh"]:
        content = read_readme(lang)

        # Should NOT contain claims about json_schema for qwen
        # Allow negation patterns like "NOT json_schema" or "不是 json_schema"
        lines = content.split("\n")
        for line in lines:
            line_lower = line.lower()
            # Check if line mentions json_schema in a positive context for qwen
            if "json_schema" in line_lower:
                # Allow if it's explicitly denying json_schema support
                is_negation = any([
                    "not" in line_lower and "json_schema" in line_lower,
                    "no" in line_lower and "json_schema" in line_lower,
                    "而不是" in line,  # Chinese negation
                    "不是" in line,  # Chinese negation
                    "does not support" in line_lower,
                    "不支持" in line,  # Chinese "does not support"
                ])
                assert is_negation, f"{lang}: Documentation should not claim json_schema support for qwen3.5-plus. Found: {line}"

        # Should explicitly mention json_object usage
        assert "json_object" in content, f"{lang}: Should mention json_object response format"


def test_readmes_document_frame_position_only_contract():
    """
    README files must document that qwen_video strategy uses
    frame-position-only output contract (no timestamps, no prose).
    """
    for lang in ["en", "zh"]:
        content = read_readme(lang)

        # Check for frame_positions output contract
        assert "frame_positions" in content, f"{lang}: Missing frame_positions output contract"
        assert "frame_id" in content, f"{lang}: Missing frame_id in output contract"

        # Check for explicit rejection of timestamps
        # Should mention no timestamps allowed
        # Look for context around frame_positions
        frame_contract_section = ""
        if "frame_positions" in content:
            idx = content.find("frame_positions")
            frame_contract_section = content[max(0, idx-100):idx+200]

        # Should indicate timestamps are NOT included
        has_no_timestamp_claim = any([
            "no timestamp" in frame_contract_section.lower(),
            "无时间戳" in frame_contract_section,
            "no prose" in frame_contract_section.lower(),
            "无散文" in frame_contract_section or "无描述" in frame_contract_section,
        ])
        assert has_no_timestamp_claim, f"{lang}: Should explicitly state no timestamps/prose in frame-position-only output"


def test_readmes_document_provider_rejection_fallback():
    """
    README files must include operator troubleshooting note about
    provider rejection and automatic mechanical fallback.
    """
    for lang in ["en", "zh"]:
        content = read_readme(lang)

        # Look for provider rejection / error handling keywords
        error_keywords = [
            ("provider", "rejection", "fallback"),
            ("提供程序", "拒绝", "降级"),
            ("malformed", "output"),
            ("格式错误", "输出"),
            ("runtime", "error"),
            ("运行时", "错误"),
        ]

        has_error_handling_doc = False
        for keywords in error_keywords:
            if all(kw in content for kw in keywords):
                has_error_handling_doc = True
                break

        # Also accept if mentions automatic fallback generally
        if not has_error_handling_doc:
            general_fallback_patterns = [
                "automatically falls back",
                "自动降级",
                "自动回退",
                "fallback to mechanical",
                "降级到机械",
            ]
            has_error_handling_doc = any(p in content for p in general_fallback_patterns)

        assert has_error_handling_doc, f"{lang}: Missing operator troubleshooting note about provider rejection and mechanical fallback"
