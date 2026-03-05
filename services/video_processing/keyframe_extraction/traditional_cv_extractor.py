"""
传统CV提取器 - PySceneDetect + 聚类
"""

import cv2
import numpy as np
from typing import List, Tuple, Any, Optional
from PIL import Image
import time
import tempfile
from pathlib import Path

try:
    from scenedetect import detect, ContentDetector, AdaptiveDetector, ThresholdDetector
    from scenedetect.backends import VideoStreamCv2
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from . import (
    BaseKeyframeExtractor, KeyframeConfig, KeyframeResult, 
    ExtractionMetrics, ExtractionMethod, KeyframeExtractorFactory
)


class TraditionalCVExtractor(BaseKeyframeExtractor):
    """
    传统CV关键帧提取器
    
    结合:
    1. PySceneDetect场景边界检测
    2. 视觉特征聚类 (HOG/LBP/颜色直方图)
    3. 每簇选择代表性帧
    """
    
    def __init__(self, config: KeyframeConfig):
        super().__init__(config)
        self.method_name = "traditional_cv"
        
        if not SCENEDETECT_AVAILABLE:
            raise ImportError("PySceneDetect未安装。请运行: pip install scenedetect[opencv]")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn未安装。请运行: pip install scikit-learn")
    
    def extract(self, video_path: str) -> Tuple[List[KeyframeResult], ExtractionMetrics]:
        """提取关键帧"""
        start_time = time.time()
        
        # 阶段1: 场景边界检测
        logger.info("阶段1: 检测场景边界...")
        scene_frames = self._detect_scenes(video_path)
        
        # 阶段2: 特征提取与聚类
        logger.info(f"阶段2: 提取特征 ({self.config.clustering_feature})...")
        features = self._extract_features(video_path, scene_frames)
        
        # 阶段3: 聚类
        logger.info("阶段3: 聚类...")
        labels = self._cluster_features(features)
        
        # 阶段4: 选择代表性帧
        logger.info("阶段4: 选择代表性帧...")
        keyframes = self._select_representatives(
            video_path, scene_frames, features, labels
        )
        
        elapsed = time.time() - start_time
        metrics = ExtractionMetrics(
            method=self.method_name,
            total_time_seconds=elapsed,
            frames_processed=len(scene_frames),
            keyframes_selected=len(keyframes),
            fps_processing=len(scene_frames) / elapsed if elapsed > 0 else 0,
        )
        
        return keyframes, metrics
    
    def _detect_scenes(self, video_path: str) -> List[int]:
        """使用PySceneDetect检测场景边界"""
        # 使用AdaptiveDetector（推荐）
        detector = AdaptiveDetector(
            adaptive_threshold=self.config.scene_threshold,
            min_scene_len=self.config.min_scene_len
        )
        
        scene_list = detect(video_path, detector)
        
        # 提取场景起始帧
        scene_frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        for scene in scene_list:
            start_frame = int(scene[0].get_frames())
            scene_frames.append(start_frame)
        
        # 如果没有检测到场景，使用均匀采样
        if len(scene_frames) < self.config.min_frames:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            step = max(1, total_frames // self.config.target_frames)
            scene_frames = list(range(0, total_frames, step))
        
        return scene_frames[:self.config.max_frames * 2]  # 限制数量
    
    def _extract_features(self, video_path: str, frame_indices: List[int]) -> np.ndarray:
        """提取视觉特征"""
        cap = cv2.VideoCapture(video_path)
        features = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            if self.config.clustering_feature == 'hog':
                feat = self._extract_hog(frame)
            elif self.config.clustering_feature == 'lbp':
                feat = self._extract_lbp(frame)
            elif self.config.clustering_feature == 'color_hist':
                feat = self._extract_color_hist(frame)
            else:
                feat = self._extract_hog(frame)
            
            features.append(feat)
        
        cap.release()
        return np.array(features)
    
    def _extract_hog(self, frame: np.ndarray) -> np.ndarray:
        """提取HOG特征"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 72))
        
        hog = cv2.HOGDescriptor(
            win_size=(128, 72),
            block_size=(16, 16),
            block_stride=(8, 8),
            cell_size=(8, 8),
            nbins=9
        )
        
        return hog.compute(resized).flatten()
    
    def _extract_lbp(self, frame: np.ndarray) -> np.ndarray:
        """提取LBP特征"""
        try:
            from skimage.feature import local_binary_pattern
        except ImportError:
            # 如果skimage不可用，使用简化版本
            return self._extract_hog(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        hist, _ = np.histogram(lbp.ravel(), bins=n_points+2, 
                               range=(0, n_points+2), density=True)
        return hist
    
    def _extract_color_hist(self, frame: np.ndarray) -> np.ndarray:
        """提取颜色直方图"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # H和S通道
        hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [30], [0, 256])
        
        cv2.normalize(hist_h, hist_h)
        cv2.normalize(hist_s, hist_s)
        
        return np.concatenate([hist_h.flatten(), hist_s.flatten()])
    
    def _cluster_features(self, features: np.ndarray) -> np.ndarray:
        """K-means聚类"""
        n_samples = len(features)
        
        # 确定聚类数
        if self.config.use_elbow:
            k = self._find_optimal_k(features)
        else:
            k = min(self.config.target_frames, n_samples)
        
        k = min(k, n_samples)
        
        if k < 2:
            return np.zeros(n_samples, dtype=int)
        
        # 降维（如果特征维度太高）
        if features.shape[1] > 50:
            pca = PCA(n_components=min(50, n_samples-1))
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
        
        # K-means聚类
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_reduced)
        
        return labels
    
    def _find_optimal_k(self, features: np.ndarray, max_k: int = 15) -> int:
        """使用肘部法则确定最优k"""
        from sklearn.metrics import silhouette_score
        
        n_samples = len(features)
        max_k = min(max_k, n_samples - 1)
        
        if max_k < 2:
            return 2
        
        best_score = -1
        best_k = 2
        
        # 降维以加速
        if features.shape[1] > 50:
            pca = PCA(n_components=min(50, n_samples-1))
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_reduced)
            score = silhouette_score(features_reduced, labels, metric='cosine')
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def _select_representatives(
        self, video_path: str, frame_indices: List[int], 
        features: np.ndarray, labels: np.ndarray
    ) -> List[KeyframeResult]:
        """从每个聚类中选择代表性帧"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        keyframes = []
        unique_labels = set(labels)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_indices = [frame_indices[i] for i in range(len(frame_indices)) if cluster_mask[i]]
            cluster_features = features[cluster_mask]
            
            if len(cluster_indices) == 0:
                continue
            
            # 计算聚类中心
            centroid = np.mean(cluster_features, axis=0)
            
            # 找到最接近质心的帧
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            best_local_idx = np.argmin(distances)
            best_frame_idx = cluster_indices[best_local_idx]
            
            timestamp = best_frame_idx / fps
            
            keyframes.append(KeyframeResult(
                frame_index=int(best_frame_idx),
                timestamp_seconds=float(timestamp),
                method=self.method_name,
                score=1.0 - (distances[best_local_idx] / np.max(distances)) if len(distances) > 0 else 1.0,
                description=f"聚类 {cluster_id} 的代表帧",
                cluster_id=int(cluster_id),
            ))
        
        # 按时间排序
        keyframes.sort(key=lambda x: x.frame_index)
        
        return keyframes[:self.config.max_frames]


# 注册提取器
KeyframeExtractorFactory.register(ExtractionMethod.CLUSTERING, TraditionalCVExtractor)
KeyframeExtractorFactory.register(ExtractionMethod.SCENE_BOUNDARY, TraditionalCVExtractor)
