"""
BioCLIP提取器 - 基于生物学相关性的关键帧选择
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import time
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

try:
    from bioclip import TreeOfLifeClassifier, CustomLabelsClassifier, Rank
    BIOCLIP_AVAILABLE = True
except ImportError:
    BIOCLIP_AVAILABLE = False

try:
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from . import (
    BaseKeyframeExtractor, KeyframeConfig, KeyframeResult, 
    ExtractionMetrics, ExtractionMethod, KeyframeExtractorFactory
)


class BioCLIPExtractor(BaseKeyframeExtractor):
    """
    BioCLIP关键帧提取器
    
    流程:
    1. 采样视频帧
    2. BioCLIP评分（生物学相关性）
    3. 过滤低分帧
    4. BioCLIP嵌入聚类
    5. 每簇选择最高分帧
    """
    
    def __init__(self, config: KeyframeConfig):
        super().__init__(config)
        self.method_name = "bioclip"
        
        if not BIOCLIP_AVAILABLE:
            raise ImportError("BioCLIP未安装。请运行: pip install pybioclip")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装。请运行: pip install scikit-learn")
        
        # 初始化BioCLIP分类器
        self.classifier = TreeOfLifeClassifier(
            device=config.bioclip_device
        )
        
        # 二进制分类器（生物 vs 非生物）
        self.binary_classifier = CustomLabelsClassifier(
            cls_ary=[
                'animal', 'bird', 'fish', 'insect', 'reptile', 'amphibian',
                'plant', 'tree', 'flower', 'fungus',
                'empty scene', 'landscape', 'sky', 'water', 'ground'
            ],
            device=config.bioclip_device
        )
        
        # 缓存模型引用
        self.model = self.classifier.model
        self.preprocess = self.classifier.preprocess
    
    def extract(self, video_path: str) -> Tuple[List[KeyframeResult], ExtractionMetrics]:
        """提取关键帧"""
        start_time = time.time()
        
        # 阶段1: 采样帧
        logger.info("阶段1: 采样视频帧...")
        frames_data = self._sample_frames(video_path)
        
        if len(frames_data) == 0:
            return [], ExtractionMetrics(
                method=self.method_name,
                total_time_seconds=time.time() - start_time,
                frames_processed=0,
                keyframes_selected=0,
                fps_processing=0,
            )
        
        # 阶段2: BioCLIP评分
        logger.info("阶段2: BioCLIP生物学评分...")
        scores = self._score_frames(frames_data)
        
        # 阶段3: 过滤低分帧
        logger.info(f"阶段3: 过滤（阈值 {self.config.min_bio_score}）...")
        bio_frames = self._filter_by_score(frames_data, scores)
        
        if len(bio_frames) == 0:
            # 如果没有生物帧，使用最高分的前N个
            bio_frames = sorted(
                zip(frames_data, scores),
                key=lambda x: x[1]['combined_score'],
                reverse=True
            )[:max(10, self.config.target_frames)]
            bio_frames = [(f, s) for f, s in bio_frames]
        
        # 阶段4: 提取嵌入
        logger.info("阶段4: 提取BioCLIP嵌入...")
        embeddings = self._extract_embeddings([f for f, s in bio_frames])
        
        # 阶段5: 聚类
        logger.info("阶段5: 聚类...")
        labels = self._cluster_embeddings(embeddings)
        
        # 阶段6: 选择代表性帧
        logger.info("阶段6: 选择代表性帧...")
        keyframes = self._select_representatives(bio_frames, scores, labels, embeddings)
        
        elapsed = time.time() - start_time
        metrics = ExtractionMetrics(
            method=self.method_name,
            total_time_seconds=elapsed,
            frames_processed=len(frames_data),
            keyframes_selected=len(keyframes),
            fps_processing=len(frames_data) / elapsed if elapsed > 0 else 0,
        )
        
        return keyframes, metrics
    
    def _sample_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """均匀采样帧"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算采样间隔
        sample_interval = max(1, int(fps / self.config.sample_fps))
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                frames.append({
                    'frame_index': frame_idx,
                    'timestamp': frame_idx / fps,
                    'frame': frame,
                })
            
            frame_idx += 1
            
            # 限制最大采样数
            if len(frames) >= self.config.max_frames * 3:
                break
        
        cap.release()
        return frames
    
    def _score_frames(self, frames_data: List[Dict]) -> List[Dict[str, Any]]:
        """使用BioCLIP评分帧"""
        scores = []
        
        for frame_data in frames_data:
            frame = frame_data['frame']
            
            # 转换为PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 临时保存（BioCLIP需要路径）
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                pil_image.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                # 1. 分类置信度评分
                preds = self.classifier.predict(tmp_path, Rank.SPECIES, k=5)
                
                if preds:
                    top_score = preds[0]['score']
                    sum_top5 = sum(p['score'] for p in preds[:5])
                    top_species = preds[0]['species']
                    top_common = preds[0].get('common_name', '')
                else:
                    top_score = 0
                    sum_top5 = 0
                    top_species = 'unknown'
                    top_common = ''
                
                # 2. 二进制评分
                binary_preds = self.binary_classifier.predict(tmp_path, k=None)
                bio_categories = ['animal', 'bird', 'fish', 'insect', 'reptile', 
                                'amphibian', 'plant', 'tree', 'flower', 'fungus']
                bio_score = sum(p['score'] for p in binary_preds 
                              if p['classification'] in bio_categories)
                
                # 3. 综合评分
                combined_score = 0.4 * top_score + 0.6 * bio_score
                
                scores.append({
                    'combined_score': combined_score,
                    'top_score': top_score,
                    'bio_score': bio_score,
                    'top_species': top_species,
                    'top_common': top_common,
                    'is_biological': combined_score > self.config.min_bio_score,
                })
            finally:
                # 清理临时文件
                try:
                    import os
                    os.unlink(tmp_path)
                except:
                    pass
        
        return scores
    
    def _filter_by_score(
        self, 
        frames_data: List[Dict], 
        scores: List[Dict]
    ) -> List[Tuple[Dict, Dict]]:
        """过滤低分帧"""
        filtered = []
        for frame, score in zip(frames_data, scores):
            if score['is_biological']:
                filtered.append((frame, score))
        return filtered
    
    def _extract_embeddings(self, frames_data: List[Dict]) -> np.ndarray:
        """提取BioCLIP嵌入"""
        embeddings = []
        device = next(self.model.parameters()).device
        
        for frame_data in frames_data:
            frame = frame_data['frame']
            
            # 预处理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(device)
            
            # 提取嵌入
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = F.normalize(image_features, dim=-1)
            
            embeddings.append(image_features.cpu().numpy()[0])
        
        return np.array(embeddings)
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """聚类嵌入"""
        n_samples = len(embeddings)
        
        if n_samples < 3:
            return np.zeros(n_samples, dtype=int)
        
        min_cluster_size = min(3, n_samples // 2)
        
        try:
            # 尝试HDBSCAN
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric='cosine',
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embeddings)
        except:
            # 回退到K-means
            k = min(self.config.target_frames, n_samples)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        
        return labels
    
    def _select_representatives(
        self,
        bio_frames: List[Tuple[Dict, Dict]],
        scores: List[Dict],
        labels: np.ndarray,
        embeddings: np.ndarray
    ) -> List[KeyframeResult]:
        """选择代表性帧"""
        keyframes = []
        unique_labels = set(labels[labels != -1])  # -1是噪声
        
        # 每个聚类选择最高分帧
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_indices = [i for i in range(len(bio_frames)) if cluster_mask[i]]
            
            if not cluster_indices:
                continue
            
            # 找到最高分帧
            best_idx_in_cluster = max(
                cluster_indices,
                key=lambda i: bio_frames[i][1]['combined_score']
            )
            
            frame_data, score_data = bio_frames[best_idx_in_cluster]
            
            keyframes.append(KeyframeResult(
                frame_index=frame_data['frame_index'],
                timestamp_seconds=frame_data['timestamp'],
                method=self.method_name,
                score=score_data['combined_score'],
                description=f"{score_data['top_species']} ({score_data['top_common']})",
                cluster_id=int(cluster_id),
                metadata={
                    'top_species': score_data['top_species'],
                    'top_common': score_data['top_common'],
                    'bio_score': score_data['bio_score'],
                }
            ))
        
        # 按时间排序
        keyframes.sort(key=lambda x: x.frame_index)
        
        # 限制数量
        return keyframes[:self.config.max_frames]


# 注册提取器
KeyframeExtractorFactory.register(ExtractionMethod.BIOCLIP, BioCLIPExtractor)
