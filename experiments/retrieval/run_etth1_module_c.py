"""ETTh1 Module C benchmark using the Time-Series-Library long-forecast protocol.

Example:
  python experiments/retrieval/run_etth1_module_c.py --data data/raw/ETTh1.csv --pred-len 96 --quick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qss.classical import StructuredEncoderConfig, StructuredSequenceEncoder
from qss.data import load_etth1
from qss.retrieval import QuantumMemoryBank, TaskAwareForecastHead, TaskAwareRetriever


ROOT = Path(__file__).resolve().parents[2]


def window_features(encoder: StructuredSequenceEncoder, windows: np.ndarray, patch_length: int) -> np.ndarray:
    """Pool Patch embeddings into one task-aware retrieval feature per input window."""
    return encoder.transform(windows).scales[patch_length].embedding.mean(axis=1).astype(np.float32)


def train_retriever(model: TaskAwareRetriever, features: np.ndarray, futures: np.ndarray, epochs: int, batch_size: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(2021)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(features))
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = order[start : start + batch_size]
            loss = model.contrastive_loss(torch.as_tensor(features[batch]), torch.as_tensor(futures[batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def candidates(bank: QuantumMemoryBank, features: np.ndarray, end_times: np.ndarray, candidate_count: int):
    """Generate temporally valid fixed candidate sets for forecast-head training."""
    start = candidate_count
    for row in range(start, len(features)):
        indices, _ = bank.index.query(bank.keys[row], candidate_count, before=end_times[row])
        yield row, indices


def train_forecast_head(
    retriever: TaskAwareRetriever, bank: QuantumMemoryBank, features: np.ndarray, futures: np.ndarray,
    end_times: np.ndarray, epochs: int, candidate_count: int,
) -> TaskAwareForecastHead:
    for parameter in retriever.parameters():
        parameter.requires_grad_(False)
    future_dim = int(np.prod(futures.shape[1:]))
    head = TaskAwareForecastHead(retriever.embedding_dim, future_dim)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    rows = list(candidates(bank, features, end_times, candidate_count))
    rng = np.random.default_rng(2021)
    head.train()
    for _ in range(epochs):
        rng.shuffle(rows)
        for start in range(0, len(rows), 32):
            batch = rows[start : start + 32]
            query_rows = np.asarray([row for row, _ in batch])
            indices = np.asarray([index for _, index in batch])
            query = retriever.encode_query(torch.as_tensor(features[query_rows]))
            keys = torch.as_tensor(bank.keys[indices])
            values = torch.as_tensor(bank.futures[indices])
            prediction, _ = head(query, keys, values)
            loss = torch.nn.functional.mse_loss(prediction, torch.as_tensor(futures[query_rows]).flatten(start_dim=1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
    return head.eval()


def evaluate(
    retriever: TaskAwareRetriever, head: TaskAwareForecastHead, bank: QuantumMemoryBank,
    features: np.ndarray, futures: np.ndarray, end_times: np.ndarray, reranker: str, candidate_count: int, top_k: int,
) -> dict[str, float]:
    predictions, targets, recalls, elapsed = [], [], [], 0.0
    for feature, future, end_time in zip(features, futures, end_times):
        t0 = time.perf_counter()
        result = bank.retrieve(feature, end_time, candidate_count, top_k, reranker)
        elapsed += time.perf_counter() - t0
        with torch.no_grad():
            query = retriever.encode_query(torch.as_tensor(feature[None]))
            prediction, _ = head(query, torch.as_tensor(bank.keys[result.indices][None]), torch.as_tensor(result.futures[None]))
        predictions.append(prediction.numpy()[0])
        targets.append(future.reshape(-1))
        # Candidate-relative relevance: closest future trajectory among the fixed classical candidate set.
        candidate_error = ((result.candidate_futures - future[None]) ** 2).mean(
            axis=tuple(range(1, result.candidate_futures.ndim))
        )
        relevant = result.candidate_indices[np.argmin(candidate_error)]
        recalls.append(float(relevant in result.indices))
    pred, target = np.asarray(predictions), np.asarray(targets)
    return {
        "mse": float(np.mean((pred - target) ** 2)),
        "mae": float(np.mean(np.abs(pred - target))),
        "conditional_recall_at_1": float(np.mean(recalls)),
        "rerank_ms_per_query": 1000 * elapsed / len(features),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/ETTh1.csv")
    parser.add_argument("--pred-len", type=int, choices=(96, 192, 336, 720), required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/module_c")
    args = parser.parse_args()
    torch.manual_seed(2021)
    train = load_etth1(args.data, "train", pred_len=args.pred_len)
    val = load_etth1(args.data, "val", pred_len=args.pred_len)
    test = load_etth1(args.data, "test", pred_len=args.pred_len)
    train_limit = 512 if args.quick else len(train.inputs)
    eval_limit = 128 if args.quick else len(test.inputs)
    train_inputs, train_targets, train_end = train.inputs[:train_limit], train.targets[:train_limit], train.end_times[:train_limit]

    encoder = StructuredSequenceEncoder(StructuredEncoderConfig(patch_lengths=(24, 48, 96), period=24)).fit(train_inputs)
    train_features = window_features(encoder, train_inputs, 24)
    test_features = window_features(encoder, test.inputs[:eval_limit], 24)
    retriever = TaskAwareRetriever(train_features.shape[1], embedding_dim=32, quantum_dim=4)
    epochs = 1 if args.quick else 5
    train_retriever(retriever, train_features, train_targets, epochs, batch_size=64)
    bank = QuantumMemoryBank(retriever, train_features, train_targets, train_end)
    head = train_forecast_head(retriever, bank, train_features, train_targets, train_end, epochs, candidate_count=20)
    report = {
        "dataset": "ETTh1", "seq_len": 96, "pred_len": args.pred_len, "features": "M",
        "protocol": "Time-Series-Library Dataset_ETT_hour 12/4/4-month split", "quick": args.quick,
        "validation_windows": len(val.inputs), "test_windows_evaluated": eval_limit,
        "results": {
            reranker: evaluate(retriever, head, bank, test_features, test.targets[:eval_limit], test.end_times[:eval_limit], reranker, 20, 3)
            for reranker in ("cosine", "rbf", "learned_metric", "quantum_fidelity")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"etth1_96_{args.pred_len}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
