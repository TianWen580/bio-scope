from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import faiss
from PIL import Image

from bioclip_model import encode_image, load_bioclip_model
from vector_store import LocalFAISSStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build FAISS index from local sample images')
    parser.add_argument('--sample-dir', default='./sample_images', help='Directory with image files')
    parser.add_argument('--index-path', default='./data/faiss_index.bin', help='Output index path')
    parser.add_argument('--metadata-path', default='./data/faiss_metadata.pkl', help='Output metadata path')
    parser.add_argument('--append', action='store_true', help='Append to existing index instead of overwrite')
    parser.add_argument('--default-note', default='Auto-imported', help='Default metadata note')
    return parser.parse_args()


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

    model, preprocess, device = load_bioclip_model()
    store = LocalFAISSStore(args.index_path, args.metadata_path, dimension=512)
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
    print(f'Index build complete. Added {added} images. Total samples: {store.count()}')


if __name__ == '__main__':
    main()
