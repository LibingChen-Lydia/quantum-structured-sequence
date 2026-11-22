import numpy as np
import torch

from qss.classical.encoder import StructuredEncoderConfig, StructuredSequenceEncoder
from qss.fusion.final_model import StructuredRetrievalForecaster, pooled_periodic_tokens
from qss.quantum.circuits import CircuitSpec
from qss.quantum.interaction import QuantumInteractionLayer
from qss.retrieval import TaskAwareRetriever


def test_periodic_pooling_preserves_patch_order_and_circular_phase() -> None:
    windows = np.zeros((2, 96, 3), dtype=np.float32)
    encoder = StructuredSequenceEncoder(StructuredEncoderConfig(patch_lengths=(24,), period=24)).fit(windows)
    scale = encoder.transform(windows).scales[24]
    amp, phase = pooled_periodic_tokens(scale)
    assert amp.shape == phase.shape == (2, 4)
    assert np.all(np.isfinite(amp)) and np.all(np.isfinite(phase))


def test_final_model_backpropagates_through_both_modules() -> None:
    torch.manual_seed(0)
    # Module A's four Z expectations are appended to the 12 base retrieval features.
    retriever = TaskAwareRetriever(input_dim=16, embedding_dim=5, quantum_dim=2)
    layer = QuantumInteractionLayer(CircuitSpec(4, "hybrid", "cross_period", 1, "z"), seed=0)
    model = StructuredRetrievalForecaster(retriever, layer, amp_max=1.0, future_dim=6).double()
    prediction, weights = model(
        torch.rand(3, 12, dtype=torch.float64),
        torch.rand(3, 4, dtype=torch.float64),
        torch.rand(3, 4, dtype=torch.float64),
        torch.rand(3, 2, 5, dtype=torch.float64),
        torch.rand(3, 2, 6, dtype=torch.float64),
    )
    prediction.square().mean().backward()
    assert prediction.shape == (3, 6) and weights.shape == (3, 2)
    assert model.periodic_layer.weights.grad is not None
    assert model.retriever.query_encoder[0].weight.grad is not None
