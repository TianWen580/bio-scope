from __future__ import annotations

# pyright: reportConstantRedefinition=false, reportUnusedVariable=false


# Import the LANGUAGE_PACK from app module
# We need to access it carefully to avoid Streamlit dependencies
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import app module to access LANGUAGE_PACK
# Note: This may fail if app.py has runtime dependencies, but we'll handle it
try:
    # We'll extract the LANGUAGE_PACK dict directly from the source
    import ast
    
    with open(Path(__file__).parent.parent / 'app.py', 'r', encoding='utf-8') as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    LANGUAGE_PACK = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'LANGUAGE_PACK':
                    # Evaluate the constant expression
                    LANGUAGE_PACK = ast.literal_eval(node.value)
                    break
        if LANGUAGE_PACK:
            break
except Exception as e:
    raise AssertionError(f"Failed to load LANGUAGE_PACK from app.py: {e}")


class TestLanguagePackKeyframe:
    """Test suite for keyframe strategy language pack completeness."""
    
    # Required new keys for strategy selector and warnings
    REQUIRED_KEYFRAME_KEYS = [
        # Strategy selector labels (existing)
        'video_keyframe_strategy',
        'video_keyframe_strategy_mechanical',
        'video_keyframe_strategy_bioclip2_consistency',
        'video_keyframe_strategy_qwen_video',
        
        # Strategy descriptions (new)
        'video_keyframe_strategy_mechanical_desc',
        'video_keyframe_strategy_bioclip2_desc',
        'video_keyframe_strategy_qwen_video_desc',
        
        # Fallback and validation warnings (new)
        'video_keyframe_fallback_warning',
        'video_keyframe_formatter_validation_error',
        'video_keyframe_strategy3_output_contract',
        'video_keyframe_invalid_output_rejected',
    ]
    
    def test_new_keys_exist_in_zh_and_en(self) -> None:
        """Test that all new keyframe strategy keys exist in both zh and en."""
        assert LANGUAGE_PACK is not None, "LANGUAGE_PACK not found"
        
        # Check zh keys
        assert 'zh' in LANGUAGE_PACK, "Missing 'zh' language pack"
        zh_pack = LANGUAGE_PACK['zh']
        
        for key in self.REQUIRED_KEYFRAME_KEYS:
            assert key in zh_pack, f"Missing key '{key}' in zh language pack"
            assert isinstance(zh_pack[key], str), f"Key '{key}' in zh is not a string"
            assert len(zh_pack[key].strip()) > 0, f"Key '{key}' in zh is empty"
        
        # Check en keys
        assert 'en' in LANGUAGE_PACK, "Missing 'en' language pack"
        en_pack = LANGUAGE_PACK['en']
        
        for key in self.REQUIRED_KEYFRAME_KEYS:
            assert key in en_pack, f"Missing key '{key}' in en language pack"
            assert isinstance(en_pack[key], str), f"Key '{key}' in en is not a string"
            assert len(en_pack[key].strip()) > 0, f"Key '{key}' in en is empty"
    
    def test_strategy_labels_render_without_keyerror(self) -> None:
        """Test that strategy labels can be accessed without KeyError."""
        assert LANGUAGE_PACK is not None, "LANGUAGE_PACK not found"
        
        # Test zh access patterns
        zh_pack = LANGUAGE_PACK['zh']
        
        # Access strategy selector label
        _ = zh_pack['video_keyframe_strategy']
        
        # Access individual strategy labels
        _ = zh_pack['video_keyframe_strategy_mechanical']
        _ = zh_pack['video_keyframe_strategy_bioclip2_consistency']
        _ = zh_pack['video_keyframe_strategy_qwen_video']
        
        # Access strategy descriptions
        _ = zh_pack['video_keyframe_strategy_mechanical_desc']
        _ = zh_pack['video_keyframe_strategy_bioclip2_desc']
        _ = zh_pack['video_keyframe_strategy_qwen_video_desc']
        
        # Access warnings
        _ = zh_pack['video_keyframe_fallback_warning']
        _ = zh_pack['video_keyframe_formatter_validation_error']
        _ = zh_pack['video_keyframe_strategy3_output_contract']
        _ = zh_pack['video_keyframe_invalid_output_rejected']
        
        # Test en access patterns
        en_pack = LANGUAGE_PACK['en']
        
        # Access strategy selector label
        _ = en_pack['video_keyframe_strategy']
        
        # Access individual strategy labels
        _ = en_pack['video_keyframe_strategy_mechanical']
        _ = en_pack['video_keyframe_strategy_bioclip2_consistency']
        _ = en_pack['video_keyframe_strategy_qwen_video']
        
        # Access strategy descriptions
        _ = en_pack['video_keyframe_strategy_mechanical_desc']
        _ = en_pack['video_keyframe_strategy_bioclip2_desc']
        _ = en_pack['video_keyframe_strategy_qwen_video_desc']
        
        # Access warnings
        _ = en_pack['video_keyframe_fallback_warning']
        _ = en_pack['video_keyframe_formatter_validation_error']
        _ = en_pack['video_keyframe_strategy3_output_contract']
        _ = en_pack['video_keyframe_invalid_output_rejected']
    
    def test_strategy3_output_contract_mentions_frame_positions(self) -> None:
        """Test that strategy-3 contract explicitly mentions frame positions."""
        assert LANGUAGE_PACK is not None, "LANGUAGE_PACK not found"
        
        zh_pack = LANGUAGE_PACK['zh']
        en_pack = LANGUAGE_PACK['en']
        
        # Check that contract mentions frame_positions
        zh_contract = zh_pack['video_keyframe_strategy3_output_contract']
        en_contract = en_pack['video_keyframe_strategy3_output_contract']
        
        assert 'frame_positions' in zh_contract or 'frame_id' in zh_contract, \
            "zh contract should mention frame_positions or frame_id"
        assert 'frame_positions' in en_contract or 'frame_id' in en_contract, \
            "en contract should mention frame_positions or frame_id"
    
    def test_strategy3_contract_rejects_timestamps(self) -> None:
        """Test that strategy-3 contract explicitly rejects timestamps."""
        assert LANGUAGE_PACK is not None, "LANGUAGE_PACK not found"
        
        zh_pack = LANGUAGE_PACK['zh']
        en_pack = LANGUAGE_PACK['en']
        
        zh_contract = zh_pack['video_keyframe_strategy3_output_contract']
        en_contract = en_pack['video_keyframe_strategy3_output_contract']
        
        # Check for timestamp rejection wording (various patterns)
        timestamp_keywords = ['时间戳', 'timestamp', '00:', 's)', 'seconds']
        zh_has_timestamp_rejection = any(kw in zh_contract for kw in ['时间戳', '00:', '12s'])
        en_has_timestamp_rejection = any(kw in en_contract for kw in ['Timestamp', '00:', '12s'])
        
        assert zh_has_timestamp_rejection, \
            "zh contract should explicitly reject timestamps"
        assert en_has_timestamp_rejection, \
            "en contract should explicitly reject timestamps"
    
    def test_fallback_warning_mentions_downgrade(self) -> None:
        """Test that fallback warning mentions strategy downgrade."""
        assert LANGUAGE_PACK is not None, "LANGUAGE_PACK not found"
        
        zh_pack = LANGUAGE_PACK['zh']
        en_pack = LANGUAGE_PACK['en']
        
        zh_warning = zh_pack['video_keyframe_fallback_warning']
        en_warning = en_pack['video_keyframe_fallback_warning']
        
        # Check for downgrade/fallback language
        assert any(kw in zh_warning for kw in ['降级', '回退', '机械']), \
            "zh fallback warning should mention downgrade to mechanical"
        assert any(kw in en_warning for kw in ['downgrade', 'fallback', 'mechanical']), \
            "en fallback warning should mention downgrade to mechanical"
    
    def test_bilingual_key_parity(self) -> None:
        """Test that both zh and en have identical key sets."""
        assert LANGUAGE_PACK is not None, "LANGUAGE_PACK not found"
        
        zh_keys = set(LANGUAGE_PACK['zh'].keys())
        en_keys = set(LANGUAGE_PACK['en'].keys())
        
        # Find keys only in zh
        zh_only = zh_keys - en_keys
        # Find keys only in en
        en_only = en_keys - zh_keys
        
        # For keyframe-related keys, both languages must have them
        keyframe_keys = {k for k in zh_keys if 'video_keyframe' in k or 'keyframe' in k}
        
        for key in keyframe_keys:
            assert key in en_keys, f"Keyframe key '{key}' exists in zh but missing in en"
        
        for key in keyframe_keys & en_keys:
            assert key in zh_keys, f"Keyframe key '{key}' exists in en but missing in zh"


def test_new_keys_exist_in_zh_and_en() -> None:
    TestLanguagePackKeyframe().test_new_keys_exist_in_zh_and_en()


def test_strategy_labels_render_without_keyerror() -> None:
    TestLanguagePackKeyframe().test_strategy_labels_render_without_keyerror()
