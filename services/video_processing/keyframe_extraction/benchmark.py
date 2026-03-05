"""
性能测试与基准对比模块

测试所有关键帧提取方法的推理速度
"""

import time
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import tempfile
import cv2
import numpy as np
from tqdm import tqdm

from . import (
    KeyframeConfig, ExtractionMethod, 
    get_keyframe_extractor, KeyframeExtractorFactory
)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    method: str
    video_duration_seconds: float
    video_resolution: str
    total_time_seconds: float
    frames_processed: int
    keyframes_selected: int
    fps_processing: float
    time_per_frame_ms: float
    memory_peak_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KeyframeBenchmark:
    """关键帧提取性能测试器"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def create_test_video(
        self, 
        duration_seconds: float = 60.0,
        fps: float = 30.0,
        resolution: Tuple[int, int] = (1280, 720),
        output_path: Optional[str] = None
    ) -> str:
        """
        创建测试视频
        
        Args:
            duration_seconds: 视频时长（秒）
            fps: 帧率
            resolution: 分辨率 (宽, 高)
            output_path: 输出路径（None则创建临时文件）
            
        Returns:
            视频文件路径
        """
        if output_path is None:
            output_path = str(self.output_dir / f"test_video_{duration_seconds}s_{fps}fps.mp4")
        
        width, height = resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        total_frames = int(duration_seconds * fps)
        
        print(f"创建测试视频: {duration_seconds}s, {fps}fps, {resolution}")
        for i in tqdm(range(total_frames), desc="生成测试视频"):
            # 创建渐变帧以模拟场景变化
            t = i / total_frames
            
            # 颜色随时间变化
            r = int(255 * (0.5 + 0.5 * np.sin(t * 2 * np.pi)))
            g = int(255 * (0.5 + 0.5 * np.sin(t * 4 * np.pi)))
            b = int(255 * (0.5 + 0.5 * np.sin(t * 6 * np.pi)))
            
            frame = np.full((height, width, 3), (b, g, r), dtype=np.uint8)
            
            # 添加时间戳文字
            timestamp = f"Frame {i} / {total_frames}"
            cv2.putText(frame, timestamp, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"测试视频已保存: {output_path}")
        
        return output_path
    
    def benchmark_method(
        self,
        video_path: str,
        method: ExtractionMethod,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        测试单个方法
        
        Args:
            video_path: 视频路径
            method: 提取方法
            config_overrides: 配置覆盖
            
        Returns:
            测试结果
        """
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        cap.release()
        
        print(f"\n{'='*60}")
        print(f"测试方法: {method.value}")
        print(f"视频: {duration:.1f}s, {total_frames} frames, {width}x{height}")
        print(f"{'='*60}")
        
        # 创建配置
        config = KeyframeConfig.from_env()
        config.method = method
        
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 记录内存使用
        import psutil
        import gc
        
        gc.collect()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 运行测试
        start_time = time.time()
        
        try:
            extractor = get_keyframe_extractor(config)
            keyframes, metrics = extractor.extract(video_path)
            success = True
        except Exception as e:
            print(f"错误: {e}")
            keyframes = []
            metrics = None
            success = False
        
        elapsed = time.time() - start_time
        
        # 记录内存使用
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_peak = mem_after - mem_before
        
        if success and metrics:
            result = BenchmarkResult(
                method=method.value,
                video_duration_seconds=duration,
                video_resolution=f"{width}x{height}",
                total_time_seconds=metrics.total_time_seconds,
                frames_processed=metrics.frames_processed,
                keyframes_selected=metrics.keyframes_selected,
                fps_processing=metrics.fps_processing,
                time_per_frame_ms=(metrics.total_time_seconds / max(metrics.frames_processed, 1)) * 1000,
                memory_peak_mb=mem_peak,
            )
        else:
            result = BenchmarkResult(
                method=method.value,
                video_duration_seconds=duration,
                video_resolution=f"{width}x{height}",
                total_time_seconds=elapsed,
                frames_processed=0,
                keyframes_selected=0,
                fps_processing=0,
                time_per_frame_ms=0,
                memory_peak_mb=mem_peak,
            )
        
        self.results.append(result)
        
        # 打印结果
        print(f"\n结果:")
        print(f"  处理帧数: {result.frames_processed}")
        print(f"  关键帧数: {result.keyframes_selected}")
        print(f"  总时间: {result.total_time_seconds:.2f}s")
        print(f"  处理速度: {result.fps_processing:.2f} fps")
        print(f"  每帧耗时: {result.time_per_frame_ms:.2f} ms")
        print(f"  内存峰值: {result.memory_peak_mb:.1f} MB")
        
        return result
    
    def run_full_benchmark(
        self,
        video_duration: float = 60.0,
        video_fps: float = 30.0,
        video_resolution: Tuple[int, int] = (1280, 720),
        methods: Optional[List[ExtractionMethod]] = None
    ) -> Dict[str, Any]:
        """
        运行完整基准测试
        
        Args:
            video_duration: 测试视频时长（秒）
            video_fps: 测试视频帧率
            video_resolution: 测试视频分辨率
            methods: 要测试的方法列表（None则测试所有）
            
        Returns:
            完整测试结果
        """
        if methods is None:
            methods = [
                ExtractionMethod.MECHANICAL,
                ExtractionMethod.CLUSTERING,
            ]
            
            # 如果依赖可用，添加高级方法
            try:
                import bioclip
                methods.append(ExtractionMethod.BIOCLIP)
            except ImportError:
                print("BioCLIP不可用，跳过")
            
            try:
                import transformers
                methods.append(ExtractionMethod.QWEN_MULTIMODAL)
            except ImportError:
                print("Transformers不可用，跳过")
        
        # 创建测试视频
        video_path = self.create_test_video(
            duration_seconds=video_duration,
            fps=video_fps,
            resolution=video_resolution
        )
        
        # 测试每个方法
        print(f"\n{'#'*60}")
        print(f"开始完整基准测试")
        print(f"方法数量: {len(methods)}")
        print(f"{'#'*60}\n")
        
        for method in methods:
            try:
                self.benchmark_method(video_path, method)
            except Exception as e:
                print(f"测试 {method.value} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存结果
        results_data = {
            'test_config': {
                'video_duration_seconds': video_duration,
                'video_fps': video_fps,
                'video_resolution': f"{video_resolution[0]}x{video_resolution[1]}",
                'methods_tested': [m.value for m in methods],
            },
            'results': [r.to_dict() for r in self.results],
        }
        
        # 保存JSON
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # 生成报告
        self._generate_report(results_data)
        
        return results_data
    
    def _generate_report(self, results_data: Dict[str, Any]):
        """生成测试报告"""
        report_path = self.output_dir / "benchmark_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# 关键帧提取性能测试报告\n\n")
            
            # 测试配置
            f.write("## 测试配置\n\n")
            config = results_data['test_config']
            f.write(f"- 视频时长: {config['video_duration_seconds']}s\n")
            f.write(f"- 视频帧率: {config['video_fps']}fps\n")
            f.write(f"- 视频分辨率: {config['video_resolution']}\n")
            f.write(f"- 测试方法: {', '.join(config['methods_tested'])}\n\n")
            
            # 结果表格
            f.write("## 性能对比\n\n")
            f.write("| 方法 | 处理帧数 | 关键帧数 | 总时间(s) | 速度(fps) | 每帧耗时(ms) | 内存(MB) |\n")
            f.write("|------|----------|----------|-----------|-----------|--------------|----------|\n")
            
            for result in results_data['results']:
                f.write(f"| {result['method']} | "
                       f"{result['frames_processed']} | "
                       f"{result['keyframes_selected']} | "
                       f"{result['total_time_seconds']:.2f} | "
                       f"{result['fps_processing']:.2f} | "
                       f"{result['time_per_frame_ms']:.2f} | "
                       f"{result['memory_peak_mb']:.1f} |\n")
            
            f.write("\n## 结论\n\n")
            
            # 找出最快的方法
            if results_data['results']:
                fastest = min(results_data['results'], key=lambda x: x['total_time_seconds'])
                f.write(f"- **最快方法**: {fastest['method']} ({fastest['total_time_seconds']:.2f}s)\n")
                
                # 找出质量最好的方法（关键帧数适中）
                good_quality = [r for r in results_data['results'] 
                              if 10 <= r['keyframes_selected'] <= 30]
                if good_quality:
                    best_quality = min(good_quality, key=lambda x: x['total_time_seconds'])
                    f.write(f"- **推荐方法**: {best_quality['method']} (质量与时间平衡)\n")
        
        print(f"\n报告已保存: {report_path}")


def run_benchmark_cli():
    """命令行运行基准测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description="关键帧提取性能测试")
    parser.add_argument("--duration", type=float, default=60.0, 
                       help="测试视频时长（秒）")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="测试视频帧率")
    parser.add_argument("--width", type=int, default=1280,
                       help="测试视频宽度")
    parser.add_argument("--height", type=int, default=720,
                       help="测试视频高度")
    parser.add_argument("--output", type=str, default="./benchmark_results",
                       help="输出目录")
    parser.add_argument("--methods", type=str, nargs="+",
                       choices=['mechanical', 'scene_boundary', 'clustering', 'bioclip', 'qwen_multimodal'],
                       help="要测试的方法（不指定则测试所有可用方法）")
    
    args = parser.parse_args()
    
    benchmark = KeyframeBenchmark(output_dir=args.output)
    
    methods = None
    if args.methods:
        methods = [ExtractionMethod(m) for m in args.methods]
    
    results = benchmark.run_full_benchmark(
        video_duration=args.duration,
        video_fps=args.fps,
        video_resolution=(args.width, args.height),
        methods=methods
    )
    
    print("\n" + "="*60)
    print("测试完成！")
    print(f"结果保存在: {args.output}/")
    print("="*60)


if __name__ == "__main__":
    run_benchmark_cli()
