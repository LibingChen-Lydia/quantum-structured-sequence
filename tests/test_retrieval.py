import numpy as np

from qss.quantum.fidelity import AngleFidelityKernel
from qss.retrieval.memory_bank import HierarchicalMemoryBank


def test_fidelity_is_one_for_identical_angle_features() -> None:
    kernel = AngleFidelityKernel(n_qubits=4)
    scores = kernel.score_candidates(np.array([0.2, -0.1]), np.array([[0.2, -0.1]]))
    assert np.allclose(scores, [1.0])


def test_rerank_returns_requested_top_k() -> None:
    memory = HierarchicalMemoryBank(np.array([[0.0, 0.0], [0.1, 0.1], [1.0, 1.0]]))
    result = memory.rerank(
        np.array([0.0, 0.0]), AngleFidelityKernel(n_qubits=2), candidate_count=3, top_k=2
    )
    assert result.indices.shape == (2,)
    assert result.indices[0] == 0

