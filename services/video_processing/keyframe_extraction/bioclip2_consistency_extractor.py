from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bioclip_model import encode_image, load_bioclip_model


def extract_bioclip2_consistency_keyframes(
    uploaded_file,
    interval_seconds: float,
    max_frames: int,
    temporal_weight: float,
    diversity_weight: float,
    *,
    encode_image_fn: Callable[..., Any] = encode_image,
    load_bioclip_model_fn: Callable[..., Any] = load_bioclip_model,
) -> tuple[list[dict[str, Any]], str | None]:
    def _normalize_embedding_rows_for_cosine(vectors):
        arr = np.asarray(vectors, dtype='float32')
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            return None
        if not np.isfinite(arr).all():
            return None
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms <= 1e-8):
            return None
        normed = arr / norms
        spread = np.linalg.norm(normed - normed.mean(axis=0, keepdims=True), axis=1)
        if float(np.max(spread)) <= 1e-4:
            return None
        return normed

    def _select_bioclip2_temporal_diversity_positions(
        frame_positions: list[int],
        normalized_embeddings,
        max_selected: int,
        *,
        wd: float = 0.65,
        wt: float = 0.35,
        lambda_value: float = 0.4,
        min_gap_frames: int = 2,
    ) -> list[int]:
        total = len(frame_positions)
        if total == 0 or max_selected <= 0:
            return []
        if total == 1 or max_selected == 1:
            return [int(frame_positions[0])]

        target = min(total, max_selected)
        selected_idx: list[int] = [0]
        remaining = set(range(1, total))
        min_gap_frames = max(1, int(min_gap_frames))
        lambda_value = max(1e-6, float(lambda_value))

        while remaining and len(selected_idx) < target:
            best_idx: int | None = None
            best_score = -1.0
            best_gap = -1.0
            fallback_idx: int | None = None
            fallback_score = -1.0
            fallback_gap = -1.0

            for idx in remaining:
                min_gap = min(abs(int(frame_positions[idx]) - int(frame_positions[s])) for s in selected_idx)
                max_cosine = max(float(np.dot(normalized_embeddings[idx], normalized_embeddings[s])) for s in selected_idx)
                temporal_term = 1.0 - float(np.exp(-lambda_value * float(min_gap)))
                score = (wd * (1.0 - max_cosine)) + (wt * temporal_term)

                if score > fallback_score or (np.isclose(score, fallback_score) and min_gap > fallback_gap):
                    fallback_idx = idx
                    fallback_score = score
                    fallback_gap = float(min_gap)

                if min_gap >= min_gap_frames:
                    if score > best_score or (np.isclose(score, best_score) and min_gap > best_gap):
                        best_idx = idx
                        best_score = score
                        best_gap = float(min_gap)

            chosen = best_idx if best_idx is not None else fallback_idx
            if chosen is None:
                break
            selected_idx.append(chosen)
            remaining.remove(chosen)

        selected_positions = sorted({int(frame_positions[idx]) for idx in selected_idx})
        return selected_positions[:target]

    try:
        import cv2  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        return [], f'OpenCV unavailable: {exc}'

    suffix = Path(uploaded_file.name or 'upload.mp4').suffix or '.mp4'
    tmp_path = ''

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return [], 'failed to open video stream'

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0
        step = max(1, int(round(fps * interval_seconds)))

        sampled_frames: list[dict[str, Any]] = []
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(
                    {
                        'frame_id': int(frame_idx),
                        'timestamp_sec': float(frame_idx / fps),
                        'image': Image.fromarray(rgb),
                    }
                )
            frame_idx += 1

        cap.release()
        if not sampled_frames:
            return [], None

        if len(sampled_frames) <= max_frames:
            out: list[dict[str, Any]] = []
            for i, item in enumerate(sampled_frames[:max_frames], start=1):
                out.append(
                    {
                        'index': i,
                        'frame_id': item['frame_id'],
                        'timestamp_sec': item['timestamp_sec'],
                        'image': item['image'],
                    }
                )
            return out, None

        try:
            model, preprocess, target_device = load_bioclip_model_fn(model_id='hf-hub:imageomics/bioclip-2')
        except Exception as exc:
            return [], f'bioclip2 model load failed: {exc}'

        embeddings = []
        for item in sampled_frames:
            try:
                emb = np.asarray(encode_image_fn(item['image'], model, preprocess, target_device), dtype='float32').reshape(-1)
            except Exception as exc:
                return [], f'bioclip2 embedding failed: {exc}'
            if emb.size == 0:
                return [], 'bioclip2 embedding failed: empty embedding'
            embeddings.append(emb)

        normalized = _normalize_embedding_rows_for_cosine(embeddings)
        if normalized is None:
            return [], 'bioclip2 embedding failed: degenerate embeddings'

        wd = float(diversity_weight)
        wt = float(temporal_weight)
        if not np.isfinite(wd):
            wd = 0.65
        if not np.isfinite(wt):
            wt = 0.35
        wd = max(0.0, min(1.0, wd))
        wt = max(0.0, min(1.0, wt))
        if wd + wt <= 1e-8:
            wd, wt = 0.65, 0.35
        else:
            total_w = wd + wt
            wd /= total_w
            wt /= total_w

        selected_positions = _select_bioclip2_temporal_diversity_positions(
            [int(x['frame_id']) for x in sampled_frames],
            normalized,
            max_frames,
            wd=wd,
            wt=wt,
            lambda_value=0.4,
            min_gap_frames=2,
        )
        if not selected_positions:
            return [], 'bioclip2 selection failed: no valid frame positions'

        selected_set = set(selected_positions)
        selected_items: list[dict[str, Any]] = []
        seen_positions: set[int] = set()
        for item in sampled_frames:
            frame_id = int(item['frame_id'])
            if frame_id in selected_set and frame_id not in seen_positions:
                selected_items.append(item)
                seen_positions.add(frame_id)

        selected_items.sort(key=lambda x: int(x['frame_id']))
        selected_items = selected_items[:max_frames]

        out: list[dict[str, Any]] = []
        for i, item in enumerate(selected_items, start=1):
            out.append(
                {
                    'index': i,
                    'frame_id': int(item['frame_id']),
                    'timestamp_sec': float(item['timestamp_sec']),
                    'image': item['image'],
                }
            )
        return out, None
    except Exception as exc:
        return [], str(exc)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
