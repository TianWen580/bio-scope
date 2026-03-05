"""
机械采样提取器（向后兼容原有实现）
"""

import cv2
import numpy as np
from typing import List, Tuple, Any
from PIL import Image
import time
import os

from . import (
    BaseKeyframeExtractor, KeyframeConfig, KeyframeResult, 
    ExtractionMetrics, ExtractionMethod, KeyframeExtractorFactory
)


class MechanicalKeyframeExtractor(BaseKeyframeExtractor):
    """机械采样关键帧提取器 - 向后兼容原有实现"""
    
    def __init__(self, config: KeyframeConfig):
        super().__init__(config)
        self.method_name = "mechanical"
    
    def extract(self, video_path: str) -> Tuple[List[KeyframeResult], ExtractionMetrics]:
        """提取关键帧 - 固定时间间隔采样"""
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        interval_frames = max(1, int(fps * self.config.interval_seconds))
        
        keyframes = []
        frame_idx = 0
        frames_processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % interval_frames == 0:
                timestamp = frame_idx / fps
                keyframes.append(KeyframeResult(
                    frame_index=frame_idx,
                    timestamp_seconds=timestamp,
                    method=self.method_name,
                    score=1.0,  # 机械采样无评分
                    description=f"机械采样帧 (间隔 {self.config.interval_seconds}s)",
                ))
            
            frame_idx += 1
            frames_processed += 1
            
            # 限制最大帧数
            if len(keyframes) >= self.config.max_frames:
                break
        
        cap.release()
        
        elapsed = time.time() - start_time
        metrics = ExtractionMetrics(
            method=self.method_name,
            total_time_seconds=elapsed,
            frames_processed=frames_processed,
            keyframes_selected=len(keyframes),
            fps_processing=frames_processed / elapsed if elapsed > 0 else 0,
        )
        
        return keyframes, metrics


# 注册提取器
KeyframeExtractorFactory.register(ExtractionMethod.MECHANICAL, MechanicalKeyframeExtractor)
