"""Final forecasting model that combines Module A periodic interaction and Module C retrieval.

The shared encoder produces one periodic amplitude/phase pair per ``(patch,
channel)``.  Running the current state-vector Module A circuit over every
pair is impractical for multivariate 96-step windows, so this module pools the
channels *within each patch* into a four-token periodic representation.  The
resulting Module A circuit models interactions across period patches, while
Module C still receives the complete, unpooled encoder embedding for retrieval.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from qss.classical.encoder import ScaleEncoding
from qss.quantum.interaction import QuantumInteractionLayer
from qss.retrieval.task_aware import RetrievedFutureAttention, TaskAwareRetriever


def pooled_periodic_tokens(scale: ScaleEncoding) -> tuple[np.ndarray, np.ndarray]:
    """Return RMS amplitude and amplitude-weighted circular phase per patch.

    Input values are shaped ``[batch, patches, channels]``.  Pooling only
    reduces channels: patch order is preserved so a four-patch, 96-step input
    remains a four-item Module A circuit.  A zero resultant phase is defined as
    zero to avoid an arbitrary angle from numerical noise.
    """
    amplitude = np.asarray(scale.amplitude, dtype=np.float32)
    phase = np.asarray(scale.phase, dtype=np.float32)
    if amplitude.ndim != 3 or phase.shape != amplitude.shape:
        raise ValueError("scale amplitude and phase must have shape [batch, patches, channels]")
    pooled_amplitude = np.sqrt(np.mean(amplitude**2, axis=-1, dtype=np.float32))
    resultant = np.mean(amplitude * np.exp(1j * phase), axis=-1)
    pooled_phase = np.angle(resultant).astype(np.float32)
    pooled_phase[np.abs(resultant) < 1e-7] = 0.0
    return pooled_amplitude, pooled_phase


class PeriodicRetrievalForecastHead(nn.Module):
    """Gate Module A measurement features into Module C's retrieval forecast head."""

    def __init__(self, embedding_dim: int, periodic_dim: int, future_dim: int) -> None:
        super().__init__()
        self.periodic_projection = nn.Linear(periodic_dim, embedding_dim)
        self.periodic_gate = nn.Linear(embedding_dim + periodic_dim, embedding_dim)
        self.attention = RetrievedFutureAttention()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + future_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, future_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        periodic_features: torch.Tensor,
        candidate_keys: torch.Tensor,
        candidate_futures: torch.Tensor,
        quantum_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape[:-1] != periodic_features.shape[:-1]:
            raise ValueError("query and periodic_features must share batch dimensions")
        gate = torch.sigmoid(self.periodic_gate(torch.cat((query, periodic_features), dim=-1)))
        fused_query = query + gate * self.periodic_projection(periodic_features)
        context, weights = self.attention(fused_query, candidate_keys, candidate_futures, quantum_scores)
        return self.head(torch.cat((fused_query, context), dim=-1)), weights


class StructuredRetrievalForecaster(nn.Module):
    """The final model: shared-encoder periodic circuit + task-aware retrieval.

    Candidate selection stays outside this module because Module C's temporal
    memory bank is intentionally non-differentiable and must enforce its
    per-query historical cutoff.  Gradients flow through the retriever, Module
    A circuit, and fusion head once temporally valid candidates are supplied.
    """

    def __init__(
        self,
        retriever: TaskAwareRetriever,
        periodic_layer: QuantumInteractionLayer,
        amp_max: float,
        future_dim: int,
    ) -> None:
        super().__init__()
        if amp_max <= 0:
            raise ValueError("amp_max must be positive")
        self.retriever = retriever
        self.periodic_layer = periodic_layer
        self.amp_max = float(amp_max)
        self.forecast_head = PeriodicRetrievalForecastHead(
            retriever.embedding_dim, periodic_layer.n_out, future_dim
        )

    def retrieval_input(
        self, retrieval_features: torch.Tensor, amplitude: torch.Tensor, phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the Module-C query/key input enriched with Module-A features."""
        if amplitude.shape[-1] != self.periodic_layer.spec.n_items:
            raise ValueError("periodic token count must match the Module A circuit specification")
        circuit_dtype = self.periodic_layer.weights.dtype
        circuit_device = self.periodic_layer.weights.device
        theta = torch.arcsin(torch.clamp(
            amplitude.to(device=circuit_device, dtype=circuit_dtype) / self.amp_max, 0.0, 1.0
        ))
        periodic = self.periodic_layer(
            theta, phase.to(device=circuit_device, dtype=circuit_dtype)
        ).to(device=retrieval_features.device, dtype=retrieval_features.dtype)
        return torch.cat((retrieval_features, periodic), dim=-1), periodic

    def forward(
        self,
        retrieval_features: torch.Tensor,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        candidate_keys: torch.Tensor,
        candidate_futures: torch.Tensor,
        quantum_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Module A enriches the representation before Module C creates its
        # query/key; its features are also retained for prediction-time gating.
        retrieval_input, periodic_features = self.retrieval_input(retrieval_features, amplitude, phase)
        query = self.retriever.encode_query(retrieval_input)
        return self.forecast_head(query, periodic_features, candidate_keys, candidate_futures, quantum_scores)
