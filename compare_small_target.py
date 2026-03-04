from __future__ import annotations

import argparse
import base64
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from PIL import Image

from bioclip_model import encode_image, get_embedding_dimension, get_model_id, load_bioclip_model
from small_target_optimizer import detect_and_prepare_crops
from vector_store import LocalFAISSStore


INDEX_PATH = './data/faiss_index.bin'
METADATA_PATH = './data/faiss_metadata.pkl'


def _model_slug(model_id: str) -> str:
    key = model_id.strip().lower()
    if 'bioclip-2' in key:
        return 'bioclip2'
    if 'imageomics/bioclip' in key:
        return 'bioclip1'
    return re.sub(r'[^a-z0-9]+', '_', key).strip('_') or 'bioclip_custom'


def resolve_store_paths(model_id: str) -> tuple[str, str]:
    base_index = os.getenv('BIOCLIP_INDEX_PATH', INDEX_PATH).strip() or INDEX_PATH
    base_meta = os.getenv('BIOCLIP_METADATA_PATH', METADATA_PATH).strip() or METADATA_PATH
    slug = _model_slug(model_id)

    if slug == 'bioclip1':
        return base_index, base_meta

    if slug == 'bioclip2':
        env_index = os.getenv('BIOCLIP2_INDEX_PATH', '').strip()
        env_meta = os.getenv('BIOCLIP2_METADATA_PATH', '').strip()
        if env_index and env_meta:
            return env_index, env_meta

    index_suffix = f'_{slug}.bin'
    meta_suffix = f'_{slug}.pkl'
    resolved_index = (base_index[:-4] + index_suffix) if base_index.endswith('.bin') else (base_index + index_suffix)
    resolved_meta = (base_meta[:-4] + meta_suffix) if base_meta.endswith('.pkl') else (base_meta + meta_suffix)
    return resolved_index, resolved_meta


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def top_search(store: LocalFAISSStore, embedding, top_k: int) -> list[dict[str, Any]]:
    return store.search(embedding, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare baseline vs small-target optimized retrieval')
    parser.add_argument('--image', default='./assets/白鹭2.jpg', help='Test image path')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--language', default='zh', choices=['zh', 'en'])
    parser.add_argument('--max-crops', type=int, default=4)
    parser.add_argument('--use-yolo', action='store_true', default=True)
    parser.add_argument('--use-qwen', action='store_true', default=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    load_env_file(project_root / '.env')

    image_path = (project_root / args.image).resolve() if not Path(args.image).is_absolute() else Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f'image not found: {image_path}')

    api_key = os.getenv('DASHSCOPE_API_KEY', '')
    base_url = os.getenv('DASHSCOPE_BASE_URL', 'https://coding.dashscope.aliyuncs.com/v1')
    model_name = os.getenv('DASHSCOPE_MODEL', 'qwen3.5-plus')
    bioclip_model_id = os.getenv('BIOCLIP_MODEL_ID', get_model_id())
    yolo_model_path = os.getenv('YOLO_ASSIST_MODEL_PATH', './models/ultralytics/yolov12/best_yolo12_s_动物_1024_randcopybg.pt')

    image = Image.open(image_path).convert('RGB')
    image_base64 = image_to_base64(image)

    embedding_dim = get_embedding_dimension(bioclip_model_id)
    index_path, metadata_path = resolve_store_paths(bioclip_model_id)

    model, preprocess, device = load_bioclip_model(model_id=bioclip_model_id)
    store = LocalFAISSStore(index_path, metadata_path, dimension=embedding_dim)

    t0 = time.time()
    emb_full = encode_image(image, model, preprocess, device)
    baseline_hits = top_search(store, emb_full, top_k=args.top_k)
    baseline_ms = int((time.time() - t0) * 1000)

    t1 = time.time()
    optimization_error = None
    loc_info = None
    optimized_hits: list[dict[str, Any]] = []

    try:
        loc_info = detect_and_prepare_crops(
            image=image,
            image_base64=image_base64,
            image_mime='image/jpeg',
            language=args.language,
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            use_qwen_locator=args.use_qwen,
            use_yolo_assist=args.use_yolo,
            yolo_model_path=str((project_root / yolo_model_path).resolve()) if not Path(yolo_model_path).is_absolute() else yolo_model_path,
            max_crops=args.max_crops,
        )

        all_hits = []
        for crop in loc_info['crops']:
            emb = encode_image(crop['image'], model, preprocess, device)
            hits = top_search(store, emb, top_k=args.top_k)
            for item in hits:
                all_hits.append(
                    {
                        'similarity': float(item['similarity']),
                        'metadata': item['metadata'],
                        'crop_id': crop['id'],
                        'crop_source': crop['source'],
                        'bbox_norm': [float(v) for v in crop['bbox_norm']],
                    }
                )

        all_hits.sort(key=lambda x: x['similarity'], reverse=True)
        dedup = []
        seen = set()
        for hit in all_hits:
            key = (str(hit['metadata'].get('path', '')), str(hit['metadata'].get('name', '')))
            if key in seen:
                continue
            dedup.append(hit)
            seen.add(key)
            if len(dedup) >= args.top_k:
                break
        optimized_hits = dedup
    except Exception as exc:
        optimization_error = str(exc)

    optimized_ms = int((time.time() - t1) * 1000)

    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image': str(image_path),
        'image_size': image.size,
        'model': model_name,
        'baseline': {
            'elapsed_ms': baseline_ms,
            'top_hits': baseline_hits,
        },
        'optimized': {
            'elapsed_ms': optimized_ms,
            'error': optimization_error,
            'top_hits': optimized_hits,
            'qwen_error': None if loc_info is None else loc_info.get('qwen_error'),
            'yolo_error': None if loc_info is None else loc_info.get('yolo_error'),
            'qwen_boxes': [] if loc_info is None else loc_info.get('qwen_boxes', []),
            'yolo_boxes': [] if loc_info is None else loc_info.get('yolo_boxes', []),
            'fused_boxes': [] if loc_info is None else loc_info.get('fused_boxes', []),
            'crop_count': 0 if loc_info is None else len(loc_info.get('crops', [])),
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
