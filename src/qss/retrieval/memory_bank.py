"""Classical candidate recall followed by pluggable quantum re-ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qss.quantum.interfaces import SimilarityBackend


@dataclass(frozen=True)
class RetrievalResult:
    indices: np.ndarray
    scores: np.ndarray


class HierarchicalMemoryBank:
    """Stores patch embeddings with a lightweight partition label.

    Candidate recall uses cosine similarity; only the bounded candidate set reaches
    the quantum backend, preserving the intended hierarchical design.
    """

    def __init__(self, embeddings: np.ndarray, partitions: np.ndarray | None = None) -> None:
        array = np.asarray(embeddings, dtype=float)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("embeddings must be a non-empty 2D array")
        self.embeddings = array
        self.partitions = partitions

    def recall(self, query: np.ndarray, candidate_count: int) -> np.ndarray:
        if not 1 <= candidate_count <= len(self.embeddings):
            raise ValueError("candidate_count must be between 1 and memory-bank size")
        query_vector = np.asarray(query, dtype=float).reshape(-1)
        if query_vector.size != self.embeddings.shape[1]:
            raise ValueError("query and memory embeddings must have the same dimension")
        denom = np.linalg.norm(self.embeddings, axis=1) * max(np.linalg.norm(query_vector), 1e-12)
        cosine = (self.embeddings @ query_vector) / np.maximum(denom, 1e-12)
        return np.argsort(-cosine)[:candidate_count]

    def rerank(
        self, query: np.ndarray, backend: SimilarityBackend, candidate_count: int, top_k: int
    ) -> RetrievalResult:
        if not 1 <= top_k <= candidate_count:
            raise ValueError("top_k must be between 1 and candidate_count")
        candidates = self.recall(query, candidate_count)
        scores = backend.score_candidates(query, self.embeddings[candidates])
        order = np.argsort(-scores)[:top_k]
        return RetrievalResult(indices=candidates[order], scores=scores[order])

