"""Backend-independent contracts for the quantum modules."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class InteractionBackend(Protocol):
    """Produces a measurement feature vector from one periodic patch."""

    def run_interaction(self, features: np.ndarray) -> np.ndarray: ...


class SimilarityBackend(Protocol):
    """Scores a query against candidate feature vectors."""

    def score_candidates(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray: ...

