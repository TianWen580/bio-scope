"""
Qwen多模态提取器 - 基于LLM理解的关键帧选择

重要约束: 输出必须是帧位置(frame_index/timestamp)，不能依赖图像中的时间显示
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import time
import json
import re

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from qwen_vl_utils import process_vision_info
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from . import (
    BaseKeyframeExtractor, KeyframeConfig, KeyframeResult, 
    ExtractionMetrics, ExtractionMethod, KeyframeExtractorFactory
)


class QwenMultimodalExtractor(BaseKeyframeExtractor):
    """
    Qwen多模态关键帧提取器
    
    使用Qwen2.5-VL或Qwen3-VL进行视频理解，提取生物事件关键帧。
    
    关键约束: 
    - 输出必须是frame_index或timestamp_seconds
    - 不能使用OCR读取图像中的时间显示
    """
    
    def __init__(self, config: KeyframeConfig):
        super().__init__(config)
        self.method_name = "qwen_multimodal"
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers未安装。请运行: pip install transformers qwen-vl-utils")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载Qwen模型"""
        logger.info(f"加载Qwen模型: {self.config.qwen_model}")
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.qwen_model,
            torch_dtype="auto",
            device_map=self.config.qwen_device,
            attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(self.config.qwen_model)
        
        logger.info("Qwen模型加载完成")
    
    def extract(self, video_path: str) -> Tuple[List[KeyframeResult], ExtractionMetrics]:
        """提取关键帧"""
        start_time = time.time()
        
        # 获取视频信息
        video_info = self._get_video_info(video_path)
        fps = video_info['fps']
        total_frames = video_info['total_frames']
        
        # 阶段1: 采样帧（均匀采样用于输入Qwen）
        logger.info("阶段1: 采样视频帧...")
        sampled_frames = self._sample_frames_for_qwen(video_path, fps, total_frames)
        
        if len(sampled_frames) == 0:
            return [], ExtractionMetrics(
                method=self.method_name,
                total_time_seconds=time.time() - start_time,
                frames_processed=0,
                keyframes_selected=0,
                fps_processing=0,
            )
        
        # 阶段2: Qwen分析 - 使用约束性提示词确保输出帧位置
        logger.info("阶段2: Qwen多模态分析...")
        keyframe_data = self._analyze_with_qwen(video_path, sampled_frames, fps)
        
        # 阶段3: 映射回原始帧
        logger.info("阶段3: 映射关键帧...")
        keyframes = self._map_to_original_frames(keyframe_data, sampled_frames, fps)
        
        elapsed = time.time() - start_time
        metrics = ExtractionMetrics(
            method=self.method_name,
            total_time_seconds=elapsed,
            frames_processed=len(sampled_frames),
            keyframes_selected=len(keyframes),
            fps_processing=len(sampled_frames) / elapsed if elapsed > 0 else 0,
        )
        
        return keyframes, metrics
    
    def _sample_frames_for_qwen(
        self, 
        video_path: str, 
        fps: float, 
        total_frames: int
    ) -> List[Dict[str, Any]]:
        """
        为Qwen采样帧
        
        记录原始frame_index，供后续映射使用
        """
        cap = cv2.VideoCapture(video_path)
        
        # 计算采样间隔
        sample_interval = max(1, int(fps / self.config.fps_for_qwen))
        max_samples = self.config.max_frames_for_qwen
        
        frames = []
        frame_idx = 0
        sample_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                frames.append({
                    'sample_index': sample_count,  # 在采样序列中的索引
                    'frame_index': frame_idx,       # 原始视频中的帧编号
                    'timestamp': frame_idx / fps,   # 时间戳（秒）
                    'frame': frame,
                })
                sample_count += 1
            
            frame_idx += 1
            
            # 限制最大采样数
            if sample_count >= max_samples:
                break
        
        cap.release()
        return frames
    
    def _analyze_with_qwen(
        self, 
        video_path: str,
        sampled_frames: List[Dict],
        fps: float
    ) -> List[Dict[str, Any]]:
        """
        使用Qwen分析视频，提取关键帧
        
        提示词明确要求输出frame_index或timestamp_seconds
        """
        # 构建提示词 - 强调输出必须是帧位置
        prompt = f"""
分析这个生物视频，识别所有重要的生物事件（如动物出现、行为变化、互动等）。

视频信息:
- 总帧数: {len(sampled_frames)} 帧（已采样）
- 帧率: {fps:.2f} fps
- 采样间隔: 每 {self.config.fps_for_qwen:.1f} 秒一帧

任务:
1. 识别所有重要的生物事件
2. 对每个事件，输出其在采样序列中的位置

重要约束:
- 必须输出 frame_index（采样序列中的索引，从0开始）或 timestamp_seconds（秒）
- 不要依赖图像中的时间显示（如监控画面上的时间戳）
- 基于视频内容的变化来判断事件

输出格式（JSON）:
{{
    "events": [
        {{
            "frame_index": 15,           // 在采样序列中的索引（从0开始）
            "timestamp_seconds": 15.0,    // 对应的时间戳（秒）
            "event_type": "animal_appearance",
            "description": "鹿出现在画面左侧",
            "importance": 9               // 重要性评分（1-10）
        }},
        {{
            "frame_index": 45,
            "timestamp_seconds": 45.0,
            "event_type": "feeding_behavior",
            "description": "鸟开始觅食",
            "importance": 8
        }}
    ],
    "total_events": 2,
    "summary": "视频展示了鹿和鸟的活动，主要行为包括觅食和移动"
}}

请确保frame_index对应于采样序列中的实际位置。
"""
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_frames": self.config.max_frames_for_qwen,
                        "fps": self.config.fps_for_qwen,
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 应用聊天模板
        text = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        text = {k: v.to(self.model.device) for k, v in text.items()}
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **text,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=0.8
            )
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(text['input_ids'], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Qwen输出:\n{output_text[:500]}...")
        
        # 解析JSON
        return self._parse_qwen_output(output_text)
    
    def _parse_qwen_output(self, output_text: str) -> List[Dict[str, Any]]:
        """解析Qwen的输出"""
        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get('events', [])
        except json.JSONDecodeError:
            logger.warning("无法解析JSON，尝试提取关键帧信息")
        
        # 回退：尝试提取数字（假设为frame_index）
        numbers = re.findall(r'\bframe_index["\']?\s*[:=]\s*(\d+)', output_text)
        if numbers:
            return [{'frame_index': int(n), 'importance': 5} for n in numbers]
        
        return []
    
    def _map_to_original_frames(
        self,
        keyframe_data: List[Dict],
        sampled_frames: List[Dict],
        fps: float
    ) -> List[KeyframeResult]:
        """
        将Qwen输出的位置映射回原始帧
        """
        keyframes = []
        
        for event in keyframe_data:
            # 获取frame_index（采样序列中的索引）
            sample_idx = event.get('frame_index', -1)
            
            # 如果没有frame_index，尝试从timestamp计算
            if sample_idx < 0:
                timestamp = event.get('timestamp_seconds', 0)
                sample_idx = int(timestamp * self.config.fps_for_qwen)
            
            # 检查边界
            if sample_idx < 0 or sample_idx >= len(sampled_frames):
                logger.warning(f"无效的sample_idx: {sample_idx}")
                continue
            
            # 获取原始帧信息
            original_frame = sampled_frames[sample_idx]
            
            keyframes.append(KeyframeResult(
                frame_index=original_frame['frame_index'],
                timestamp_seconds=original_frame['timestamp'],
                method=self.method_name,
                score=event.get('importance', 5) / 10.0,
                description=event.get('description', ''),
                metadata={
                    'event_type': event.get('event_type', 'unknown'),
                    'sample_index': sample_idx,
                }
            ))
        
        # 按时间排序
        keyframes.sort(key=lambda x: x.frame_index)
        
        # 限制数量
        return keyframes[:self.config.max_frames]


# 注册提取器
KeyframeExtractorFactory.register(ExtractionMethod.QWEN_MULTIMODAL, QwenMultimodalExtractor)
