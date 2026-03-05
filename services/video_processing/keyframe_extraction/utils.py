"""
便捷使用接口和示例
"""

from typing import List, Optional, Union
from pathlib import Path

from . import (
    KeyframeConfig, ExtractionMethod, KeyframeResult,
    get_keyframe_extractor, KeyframeExtractorFactory
)


def extract_keyframes(
    video_path: str,
    method: Union[str, ExtractionMethod] = "clustering",
    n_keyframes: int = 20,
    **kwargs
) -> List[KeyframeResult]:
    """
    便捷函数：提取关键帧
    
    Args:
        video_path: 视频文件路径
        method: 提取方法 ('mechanical', 'scene_boundary', 'clustering', 'bioclip', 'qwen_multimodal')
        n_keyframes: 目标关键帧数量
        **kwargs: 其他配置参数
        
    Returns:
        关键帧列表
        
    Examples:
        >>> # 使用传统CV聚类
        >>> frames = extract_keyframes("video.mp4", method="clustering", n_keyframes=20)
        
        >>> # 使用BioCLIP
        >>> frames = extract_keyframes("wildlife.mp4", method="bioclip", n_keyframes=15)
        
        >>> # 使用Qwen多模态
        >>> frames = extract_keyframes("video.mp4", method="qwen_multimodal", n_keyframes=10)
    """
    # 解析方法
    if isinstance(method, str):
        method = ExtractionMethod(method)
    
    # 创建配置
    config = KeyframeConfig(
        method=method,
        target_frames=n_keyframes,
        **kwargs
    )
    
    # 提取
    extractor = get_keyframe_extractor(config)
    keyframes, metrics = extractor.extract(video_path)
    
    print(f"提取完成: {len(keyframes)} 关键帧, 耗时 {metrics.total_time_seconds:.2f}s")
    
    return keyframes


def compare_methods(
    video_path: str,
    methods: Optional[List[str]] = None,
    n_keyframes: int = 20
) -> dict:
    """
    对比不同方法的结果
    
    Args:
        video_path: 视频路径
        methods: 要对比的方法列表（None则对比所有可用方法）
        n_keyframes: 每种方法的关键帧数
        
    Returns:
        对比结果字典
    """
    if methods is None:
        methods = ['mechanical', 'clustering', 'bioclip', 'qwen_multimodal']
    
    results = {}
    
    for method_name in methods:
        try:
            print(f"\n测试方法: {method_name}")
            config = KeyframeConfig(
                method=ExtractionMethod(method_name),
                target_frames=n_keyframes
            )
            
            extractor = get_keyframe_extractor(config)
            keyframes, metrics = extractor.extract(video_path)
            
            results[method_name] = {
                'keyframes': keyframes,
                'metrics': metrics,
                'frame_indices': [k.frame_index for k in keyframes],
                'timestamps': [k.timestamp_seconds for k in keyframes],
            }
            
            print(f"  成功: {len(keyframes)} 关键帧, {metrics.total_time_seconds:.2f}s")
            
        except Exception as e:
            print(f"  失败: {e}")
            results[method_name] = {'error': str(e)}
    
    return results


# 导出所有关键类
__all__ = [
    # 核心类
    'KeyframeConfig',
    'KeyframeResult',
    'ExtractionMethod',
    'ExtractionMetrics',
    'BaseKeyframeExtractor',
    'KeyframeExtractorFactory',
    
    # 便捷函数
    'get_keyframe_extractor',
    'extract_keyframes',
    'compare_methods',
    
    # 具体提取器
    'MechanicalKeyframeExtractor',
    'TraditionalCVExtractor',
    'BioCLIPExtractor',
    'QwenMultimodalExtractor',
    
    # 测试
    'KeyframeBenchmark',
]

# 延迟导入具体类（避免循环导入）
def __getattr__(name):
    if name == 'MechanicalKeyframeExtractor':
        from .mechanical_extractor import MechanicalKeyframeExtractor
        return MechanicalKeyframeExtractor
    elif name == 'TraditionalCVExtractor':
        from .traditional_cv_extractor import TraditionalCVExtractor
        return TraditionalCVExtractor
    elif name == 'BioCLIPExtractor':
        from .bioclip_extractor import BioCLIPExtractor
        return BioCLIPExtractor
    elif name == 'QwenMultimodalExtractor':
        from .qwen_extractor import QwenMultimodalExtractor
        return QwenMultimodalExtractor
    elif name == 'KeyframeBenchmark':
        from .benchmark import KeyframeBenchmark
        return KeyframeBenchmark
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
