"""
BioScope Studio - 关键帧提取模块
支持多种提取策略：Qwen多模态、BioCLIP生物评分、传统CV
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import os
import json
import time
import logging
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionMethod(Enum):
    """关键帧提取方法枚举"""
    MECHANICAL = "mechanical"           # 原有机械采样
    SCENE_BOUNDARY = "scene_boundary"   # PySceneDetect场景边界
    CLUSTERING = "clustering"           # 聚类法（视觉特征）
    BIOCLIP = "bioclip"                 # BioCLIP生物评分
    QWEN_MULTIMODAL = "qwen_multimodal" # Qwen多模态理解


@dataclass
class KeyframeConfig:
    """关键帧提取配置"""
    method: ExtractionMethod = ExtractionMethod.CLUSTERING
    target_frames: int = 20
    min_frames: int = 5
    max_frames: int = 50
    
    # 机械采样参数
    interval_seconds: float = 2.0
    
    # 场景边界检测参数 (PySceneDetect)
    scene_threshold: float = 3.0
    min_scene_len: int = 15
    
    # 聚类参数
    clustering_feature: str = "hog"  # hog | lbp | color_hist
    use_elbow: bool = True
    max_clusters: int = 15
    sample_fps: float = 1.0
    
    # BioCLIP参数
    bioclip_version: str = "bioclip2"
    bioclip_device: str = "cuda"
    min_bio_score: float = 0.3
    diversity_method: str = "maxmin"  # maxmin | kcenter
    
    # Qwen参数
    qwen_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_device: str = "auto"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    fps_for_qwen: float = 1.0  # 提供给Qwen的采样fps
    max_frames_for_qwen: int = 128  # 限制输入Qwen的最大帧数
    
    # 性能参数
    batch_size: int = 32
    use_cache: bool = True
    cache_dir: str = ".keyframe_cache"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'target_frames': self.target_frames,
            'min_frames': self.min_frames,
            'max_frames': self.max_frames,
            'interval_seconds': self.interval_seconds,
            'scene_threshold': self.scene_threshold,
            'min_scene_len': self.min_scene_len,
            'clustering_feature': self.clustering_feature,
            'use_elbow': self.use_elbow,
            'max_clusters': self.max_clusters,
            'sample_fps': self.sample_fps,
            'bioclip_version': self.bioclip_version,
            'bioclip_device': self.bioclip_device,
            'min_bio_score': self.min_bio_score,
            'diversity_method': self.diversity_method,
            'qwen_model': self.qwen_model,
            'qwen_device': self.qwen_device,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'fps_for_qwen': self.fps_for_qwen,
            'max_frames_for_qwen': self.max_frames_for_qwen,
            'batch_size': self.batch_size,
            'use_cache': self.use_cache,
            'cache_dir': self.cache_dir,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyframeConfig':
        """从字典创建配置"""
        config = cls()
        if 'method' in data:
            config.method = ExtractionMethod(data['method'])
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    @classmethod
    def from_env(cls) -> 'KeyframeConfig':
        """从环境变量加载配置"""
        config = cls()
        
        # 方法选择
        method_str = os.getenv('KEYFRAME_EXTRACTION_METHOD', 'clustering')
        try:
            config.method = ExtractionMethod(method_str)
        except ValueError:
            logger.warning(f"Unknown method: {method_str}, using clustering")
            config.method = ExtractionMethod.CLUSTERING
        
        # 数值参数
        config.target_frames = int(os.getenv('KEYFRAME_TARGET_FRAMES', '20'))
        config.min_frames = int(os.getenv('KEYFRAME_MIN_FRAMES', '5'))
        config.max_frames = int(os.getenv('KEYFRAME_MAX_FRAMES', '50'))
        config.interval_seconds = float(os.getenv('VIDEO_FRAME_INTERVAL_SECONDS', '2.0'))
        
        # BioCLIP参数
        config.bioclip_version = os.getenv('BIOCLIP_VERSION', 'bioclip2')
        config.bioclip_device = os.getenv('BIOCLIP_DEVICE', 'cuda')
        config.min_bio_score = float(os.getenv('BIOCLIP_MIN_SCORE', '0.3'))
        
        # Qwen参数
        config.qwen_model = os.getenv('QWEN_MODEL', 'Qwen/Qwen2.5-VL-7B-Instruct')
        config.fps_for_qwen = float(os.getenv('QWEN_FPS', '1.0'))
        config.max_frames_for_qwen = int(os.getenv('QWEN_MAX_FRAMES', '128'))
        
        return config


@dataclass
class KeyframeResult:
    """关键帧结果"""
    frame_index: int              # 帧编号（在原始视频中的位置）
    timestamp_seconds: float      # 时间戳（秒）
    method: str                   # 使用的提取方法
    score: float = 0.0            # 相关性评分（0-1）
    description: str = ""         # 描述（可选）
    cluster_id: int = -1          # 聚类ID（-1表示未聚类）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'frame_index': self.frame_index,
            'timestamp_seconds': self.timestamp_seconds,
            'method': self.method,
            'score': self.score,
            'description': self.description,
            'cluster_id': self.cluster_id,
            'metadata': self.metadata,
        }


@dataclass
class ExtractionMetrics:
    """提取性能指标"""
    method: str
    total_time_seconds: float
    frames_processed: int
    keyframes_selected: int
    fps_processing: float  # 处理速度（帧/秒）
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'method': self.method,
            'total_time_seconds': self.total_time_seconds,
            'frames_processed': self.frames_processed,
            'keyframes_selected': self.keyframes_selected,
            'fps_processing': self.fps_processing,
            'memory_mb': self.memory_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
        }


class BaseKeyframeExtractor(ABC):
    """关键帧提取器基类"""
    
    def __init__(self, config: KeyframeConfig):
        self.config = config
        self.method_name = "base"
    
    @abstractmethod
    def extract(self, video_path: str) -> Tuple[List[KeyframeResult], ExtractionMetrics]:
        """
        提取关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            (关键帧列表, 性能指标)
        """
        pass
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """获取视频信息"""
        import cv2
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        }
        
        cap.release()
        return info


class KeyframeExtractorFactory:
    """关键帧提取器工厂"""
    
    _extractors: Dict[ExtractionMethod, type] = {}
    
    @classmethod
    def register(cls, method: ExtractionMethod, extractor_class: type):
        """注册提取器"""
        cls._extractors[method] = extractor_class
    
    @classmethod
    def create(cls, config: KeyframeConfig) -> BaseKeyframeExtractor:
        """创建提取器实例"""
        if config.method not in cls._extractors:
            raise ValueError(f"Unknown method: {config.method}")
        
        return cls._extractors[config.method](config)
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """列出可用方法"""
        return [method.value for method in cls._extractors.keys()]


def get_keyframe_extractor(config: Optional[KeyframeConfig] = None) -> BaseKeyframeExtractor:
    """
    获取关键帧提取器（便捷函数）
    
    使用环境变量或传入的配置
    """
    if config is None:
        config = KeyframeConfig.from_env()
    
    return KeyframeExtractorFactory.create(config)
