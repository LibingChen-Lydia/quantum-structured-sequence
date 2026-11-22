"""Task-aware historical memory retrieval for Module C.

The retrieval representation is learned from past-window features while future
trajectories supervise which historical windows should become neighbours.  Future
values are stored only as memory values and are never part of a query input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from qss.quantum.fidelity import AngleFidelityKernel


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def future_positive_mask(futures: torch.Tensor, positives_per_query: int = 1) -> torch.Tensor:
    """Mark nearest future trajectories as contrastive positives.

    ``futures`` are labels available only while training.  The diagonal is removed
    so a window cannot solve the task merely by retrieving itself.
    """
    if futures.ndim < 2 or len(futures) < 2:
        raise ValueError("futures must contain at least two trajectories")
    if not 1 <= positives_per_query < len(futures):
        raise ValueError("positives_per_query must be in [1, n_windows - 1]")
    flat = futures.reshape(len(futures), -1)
    flat = (flat - flat.mean(dim=1, keepdim=True)) / flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    distance = torch.cdist(flat, flat, p=2)
    distance.fill_diagonal_(float("inf"))
    nearest = distance.topk(positives_per_query, largest=False).indices
    mask = torch.zeros_like(distance, dtype=torch.bool)
    mask.scatter_(1, nearest, True)
    return mask


class TaskAwareRetriever(nn.Module):
    """Learns query/key and quantum-projection spaces from structured embeddings."""

    def __init__(self, input_dim: int, embedding_dim: int = 32, quantum_dim: int = 4) -> None:
        super().__init__()
        if min(input_dim, embedding_dim, quantum_dim) < 1:
            raise ValueError("all dimensions must be positive")
        self.query_encoder = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.Tanh())
        self.key_encoder = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.Tanh())
        self.quantum_projection = nn.Linear(embedding_dim, quantum_dim)
        self.embedding_dim = embedding_dim
        self.quantum_dim = quantum_dim

    def encode_query(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_encoder(features), dim=-1)

    def encode_key(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.key_encoder(features), dim=-1)

    def encode_quantum(self, embedding: torch.Tensor) -> torch.Tensor:
        # Bounded angles preserve a stable angle-encoding range.
        return torch.pi * torch.tanh(self.quantum_projection(embedding))

    def contrastive_loss(
        self, features: torch.Tensor, futures: torch.Tensor, temperature: float = 0.1,
        positives_per_query: int = 1,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        query, key = self.encode_query(features), self.encode_key(features)
        logits = query @ key.T / temperature
        positive_mask = future_positive_mask(futures, positives_per_query)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        return -(log_prob * positive_mask).sum(dim=1).mean() / positives_per_query


class TemporalANNIndex:
    """Candidate index with HNSW when available and an exact temporal-safe fallback.

    HNSW is used for unrestricted historical retrieval.  A per-query ``before``
    cutoff invokes exact cosine search on the allowed subset, guaranteeing that a
    backtest never returns future memory.  Time-sharded HNSW can replace this
    fallback when the memory bank grows large.
    """

    def __init__(self, keys: np.ndarray, end_times: np.ndarray) -> None:
        keys = np.asarray(keys, dtype=np.float32)
        end_times = np.asarray(end_times)
        if keys.ndim != 2 or len(keys) != len(end_times):
            raise ValueError("keys must be [n, d] and end_times must be [n]")
        self.keys = _l2_normalize(keys)
        self.end_times = end_times
        self._hnsw = None
        try:
            import hnswlib  # type: ignore

            index = hnswlib.Index(space="cosine", dim=keys.shape[1])
            index.init_index(max_elements=len(keys), ef_construction=100, M=16)
            index.add_items(self.keys, np.arange(len(keys)))
            index.set_ef(64)
            self._hnsw = index
        except ImportError:
            pass

    def query(self, query: np.ndarray, candidate_count: int, before: int | float | None = None) -> tuple[np.ndarray, np.ndarray]:
        if not 1 <= candidate_count <= len(self.keys):
            raise ValueError("candidate_count must be between 1 and memory size")
        query = _l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        if query.size != self.keys.shape[1]:
            raise ValueError("query dimension must match memory keys")
        if before is None and self._hnsw is not None:
            labels, distances = self._hnsw.knn_query(query, k=candidate_count)
            return labels[0], 1.0 - distances[0]
        allowed = np.arange(len(self.keys)) if before is None else np.flatnonzero(self.end_times < before)
        if len(allowed) < candidate_count:
            raise ValueError("not enough past memory items for candidate_count")
        # ``einsum`` avoids spurious BLAS floating-point warnings on some CPU
        # builds when the admissible history size crosses an implementation tile.
        scores = np.einsum("nd,d->n", self.keys[allowed], query)
        order = np.argsort(-scores)[:candidate_count]
        return allowed[order], scores[order]


@dataclass(frozen=True)
class RetrievalBatch:
    indices: np.ndarray
    candidate_indices: np.ndarray
    classical_scores: np.ndarray
    quantum_scores: np.ndarray
    futures: np.ndarray
    candidate_futures: np.ndarray


class QuantumMemoryBank:
    """Task-aware keys, temporally valid candidate search, and quantum reranking."""

    def __init__(
        self, retriever: TaskAwareRetriever, features: np.ndarray, futures: np.ndarray, end_times: np.ndarray
    ) -> None:
        self.retriever = retriever.eval()
        self.features = np.asarray(features, dtype=np.float32)
        self.futures = np.asarray(futures, dtype=np.float32)
        if len(self.features) != len(self.futures):
            raise ValueError("features and futures must have the same number of windows")
        with torch.no_grad():
            feature_tensor = torch.as_tensor(self.features)
            keys = retriever.encode_key(feature_tensor)
            quantum_keys = retriever.encode_quantum(keys)
        self.keys = keys.numpy()
        self.raw_keys = _l2_normalize(self.features)
        self.quantum_keys = quantum_keys.numpy()
        self.index = TemporalANNIndex(self.keys, end_times)
        self.kernel = AngleFidelityKernel(n_qubits=retriever.quantum_dim)

    def retrieve(
        self, query_features: np.ndarray, before: int | float, candidate_count: int = 20, top_k: int = 3,
        reranker: str = "quantum_fidelity",
    ) -> RetrievalBatch:
        if not 1 <= top_k <= candidate_count:
            raise ValueError("top_k must be between 1 and candidate_count")
        with torch.no_grad():
            query = torch.as_tensor(np.asarray(query_features, dtype=np.float32).reshape(1, -1))
            query_key = self.retriever.encode_query(query)
            query_quantum = self.retriever.encode_quantum(query_key)
        candidate_indices, classical_scores = self.index.query(
            query_key.numpy()[0], candidate_count, before=before
        )
        candidate_keys = self.keys[candidate_indices]
        raw_query = _l2_normalize(np.asarray(query_features, dtype=np.float32).reshape(1, -1))[0]
        raw_scores = self.raw_keys[candidate_indices] @ raw_query
        if reranker == "cosine":
            rerank_scores = raw_scores
        elif reranker == "learned_metric":
            rerank_scores = classical_scores
        elif reranker == "rbf":
            squared_distance = np.sum((self.raw_keys[candidate_indices] - raw_query) ** 2, axis=1)
            rerank_scores = np.exp(-squared_distance)
        elif reranker == "quantum_fidelity":
            rerank_scores = self.kernel.score_candidates(
                query_quantum.numpy()[0], self.quantum_keys[candidate_indices]
            )
        else:
            raise ValueError("reranker must be cosine, rbf, learned_metric, or quantum_fidelity")
        order = np.argsort(-rerank_scores)[:top_k]
        selected = candidate_indices[order]
        return RetrievalBatch(
            indices=selected,
            candidate_indices=candidate_indices,
            classical_scores=classical_scores[order],
            quantum_scores=rerank_scores[order],
            futures=self.futures[selected],
            candidate_futures=self.futures[candidate_indices],
        )


class RetrievedFutureAttention(nn.Module):
    """Fuses Top-K future trajectories into one context vector for a forecast head."""

    def forward(
        self, query: torch.Tensor, candidate_keys: torch.Tensor, candidate_futures: torch.Tensor,
        quantum_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.einsum("bd,bkd->bk", query, candidate_keys)
        if quantum_scores is not None:
            logits = logits + quantum_scores
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bk,bkv->bv", weights, candidate_futures.flatten(start_dim=2))
        return context, weights


class TaskAwareForecastHead(nn.Module):
    """Forecast head that conditions a query on retrieved future trajectories."""

    def __init__(self, embedding_dim: int, future_dim: int) -> None:
        super().__init__()
        self.attention = RetrievedFutureAttention()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + future_dim, embedding_dim), nn.GELU(), nn.Linear(embedding_dim, future_dim)
        )

    def forward(
        self, query: torch.Tensor, candidate_keys: torch.Tensor, candidate_futures: torch.Tensor,
        quantum_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context, weights = self.attention(query, candidate_keys, candidate_futures, quantum_scores)
        return self.head(torch.cat((query, context), dim=-1)), weights


def task_aware_loss(
    retriever: TaskAwareRetriever, forecast_head: TaskAwareForecastHead, features: torch.Tensor,
    futures: torch.Tensor, candidate_keys: torch.Tensor, candidate_futures: torch.Tensor,
    contrastive_weight: float = 0.2, temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, forecasting, and contrastive losses for one training batch."""
    query = retriever.encode_query(features)
    prediction, _ = forecast_head(query, candidate_keys, candidate_futures)
    forecast = F.mse_loss(prediction, futures.flatten(start_dim=1))
    contrastive = retriever.contrastive_loss(features, futures, temperature=temperature)
    return forecast + contrastive_weight * contrastive, forecast, contrastive
