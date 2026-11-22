import numpy as np
import torch

from qss.retrieval.task_aware import (
    QuantumMemoryBank,
    RetrievedFutureAttention,
    TaskAwareForecastHead,
    TaskAwareRetriever,
    future_positive_mask,
    task_aware_loss,
)


def test_future_positives_exclude_the_same_window() -> None:
    futures = torch.tensor([[[0.0], [0.1]], [[0.0], [0.2]], [[3.0], [3.0]]])
    mask = future_positive_mask(futures)
    assert not torch.diagonal(mask).any()
    assert mask.sum(dim=1).tolist() == [1, 1, 1]


def test_memory_retrieval_never_uses_future_windows() -> None:
    torch.manual_seed(1)
    model = TaskAwareRetriever(input_dim=3, embedding_dim=4, quantum_dim=2)
    features = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)
    futures = np.arange(10, dtype=np.float32).reshape(5, 2, 1)
    bank = QuantumMemoryBank(model, features, futures, end_times=np.array([10, 20, 30, 40, 50]))
    result = bank.retrieve(features[4], before=35, candidate_count=3, top_k=2)
    assert set(result.indices).issubset({0, 1, 2})
    assert result.futures.shape == (2, 2, 1)


def test_attention_returns_a_weighted_future_context() -> None:
    module = RetrievedFutureAttention()
    context, weights = module(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),
    )
    assert context.shape == (1, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1))


def test_forecast_head_and_joint_loss_are_differentiable() -> None:
    torch.manual_seed(2)
    retriever = TaskAwareRetriever(input_dim=3, embedding_dim=4, quantum_dim=2)
    head = TaskAwareForecastHead(embedding_dim=4, future_dim=2)
    features = torch.rand(4, 3)
    futures = torch.rand(4, 2, 1)
    keys = torch.rand(4, 2, 4)
    values = torch.rand(4, 2, 2, 1)
    total, forecast, contrastive = task_aware_loss(retriever, head, features, futures, keys, values)
    total.backward()
    assert total.item() > 0 and forecast.item() > 0 and contrastive.item() > 0
    assert retriever.query_encoder[0].weight.grad is not None
