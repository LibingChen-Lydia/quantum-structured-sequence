"""Small deterministic state-vector kernel for retrieval prototyping.

This reference implementation has no device dependency. Production experiments can
replace it with a PennyLane/Qiskit backend implementing ``SimilarityBackend``.
"""

from __future__ import annotations

import numpy as np


class AngleFidelityKernel:
    """Product-state angle encoding with an exact fidelity score."""

    def __init__(self, n_qubits: int = 4) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits

    def _angles(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("features cannot be empty")
        return np.resize(values, self.n_qubits)

    def score_candidates(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        candidate_array = np.asarray(candidates, dtype=float)
        if candidate_array.ndim != 2:
            raise ValueError("candidates must have shape (n_candidates, n_features)")
        query_angles = self._angles(query)
        candidate_angles = np.vstack([self._angles(row) for row in candidate_array])
        # |<psi(a)|psi(b)>|^2 for independently Ry-encoded qubits.
        return np.prod(np.cos((candidate_angles - query_angles) / 2.0) ** 2, axis=1)

