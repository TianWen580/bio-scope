from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Any

import faiss
import numpy as np
from filelock import FileLock


class BaseVectorStore(ABC):
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def add(self, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


class LocalFAISSStore(BaseVectorStore):
    def __init__(self, index_path: str, metadata_path: str, dimension: int = 512):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.dimension = dimension
        self.lock_path = self.metadata_path.with_suffix(self.metadata_path.suffix + '.lock')

        self.index = self._load_index()
        self.metadata = self._load_metadata()

        if self.index.d != self.dimension:
            raise ValueError(
                f'FAISS dimension mismatch: index={self.index.d}, expected={self.dimension}'
            )

    def _load_index(self):
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatIP(self.dimension)

    def _load_metadata(self) -> list[dict[str, Any]]:
        if self.metadata_path.exists():
            with self.metadata_path.open('rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                return data
        return []

    @staticmethod
    def _to_float32_2d(array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype='float32')
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f'Embedding must be 2D, got shape {arr.shape}')
        return np.ascontiguousarray(arr)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index.ntotal == 0 or not self.metadata:
            return []

        query = self._to_float32_2d(query_embedding)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f'Query dimension mismatch: query={query.shape[1]}, expected={self.dimension}'
            )

        faiss.normalize_L2(query)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append(
                    {
                        'similarity': float(distances[0][rank]),
                        'metadata': self.metadata[idx],
                    }
                )
        return results

    def add(self, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        vector = self._to_float32_2d(embedding)
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f'Embedding dimension mismatch: embedding={vector.shape[1]}, expected={self.dimension}'
            )

        faiss.normalize_L2(vector)
        self.index.add(vector)
        self.metadata.append(metadata)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        lock = FileLock(str(self.lock_path))
        with lock:
            faiss.write_index(self.index, str(self.index_path))
            with self.metadata_path.open('wb') as f:
                pickle.dump(self.metadata, f)

    def count(self) -> int:
        return len(self.metadata)
