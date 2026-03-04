from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import os
import re

import faiss
from PIL import Image

from bioclip_model import encode_image, get_embedding_dimension, get_model_id, load_bioclip_model
from vector_store import LocalFAISSStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build FAISS index from local sample images')
    parser.add_argument('--sample-dir', default='./sample_images', help='Directory with image files')
    parser.add_argument('--model-id', default=get_model_id(), help='BioCLIP model id used for embedding')
    parser.add_argument('--index-path', default='', help='Output index path (auto-derived by model when empty)')
    parser.add_argument('--metadata-path', default='', help='Output metadata path (auto-derived by model when empty)')
    parser.add_argument('--append', action='store_true', help='Append to existing index instead of overwrite')
    parser.add_argument('--default-note', default='Auto-imported', help='Default metadata note')
    return parser.parse_args()


def _model_slug(model_id: str) -> str:
    key = model_id.strip().lower()
    if 'bioclip-2' in key:
        return 'bioclip2'
    if 'imageomics/bioclip' in key:
        return 'bioclip1'
    return re.sub(r'[^a-z0-9]+', '_', key).strip('_') or 'bioclip_custom'


def resolve_store_paths(model_id: str, index_path: str, metadata_path: str) -> tuple[str, str]:
    if index_path.strip() and metadata_path.strip():
        return index_path.strip(), metadata_path.strip()

    base_index = os.getenv('BIOCLIP_INDEX_PATH', './data/faiss_index.bin').strip() or './data/faiss_index.bin'
    base_meta = os.getenv('BIOCLIP_METADATA_PATH', './data/faiss_metadata.pkl').strip() or './data/faiss_metadata.pkl'
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


def collect_images(sample_dir: Path) -> list[Path]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return sorted([p for p in sample_dir.rglob('*') if p.suffix.lower() in exts])


def reset_store(store: LocalFAISSStore) -> None:
    store.index = faiss.IndexFlatIP(store.dimension)
    store.metadata = []


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        raise FileNotFoundError(f'Sample directory not found: {sample_dir}')

    image_paths = collect_images(sample_dir)
    if not image_paths:
        raise RuntimeError(f'No images found in {sample_dir}')

    embedding_dim = get_embedding_dimension(args.model_id)
    index_path, metadata_path = resolve_store_paths(args.model_id, args.index_path, args.metadata_path)

    model, preprocess, device = load_bioclip_model(model_id=args.model_id)
    store = LocalFAISSStore(index_path, metadata_path, dimension=embedding_dim)
    if not args.append:
        reset_store(store)

    added = 0
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            embedding = encode_image(image, model, preprocess, device)
            metadata = {
                'name': image_path.stem,
                'path': str(image_path),
                'location': 'unknown',
                'notes': args.default_note,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'bulk_import',
            }
            store.add(embedding, metadata)
            added += 1
        except Exception as exc:
            print(f'Skipped {image_path}: {exc}')

    if added == 0:
        raise RuntimeError('No image embeddings were added. Index not saved.')

    store.save()
    print(f'Index build complete. model={args.model_id}, dim={embedding_dim}, index={index_path}')
    print(f'Added {added} images. Total samples: {store.count()}')


if __name__ == '__main__':
    main()
