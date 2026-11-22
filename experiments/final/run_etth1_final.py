"""ETTh1 benchmark for the early-fused Module A + Module C final model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from qss.classical import StructuredEncoderConfig, StructuredSequenceEncoder
from qss.data.tsl_adapter import load_tsl
from qss.fusion import StructuredRetrievalForecaster, pooled_periodic_tokens
from qss.quantum.circuits import CircuitSpec
from qss.quantum.interaction import QuantumInteractionLayer
from qss.retrieval import QuantumMemoryBank, TaskAwareRetriever

ROOT = Path(__file__).resolve().parents[2]


def features(encoder: StructuredSequenceEncoder, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = encoder.transform(windows).scales[24]
    amp, phase = pooled_periodic_tokens(scale)
    return scale.embedding.reshape(len(windows), -1).astype(np.float32), amp, phase


def enhanced(model: StructuredRetrievalForecaster, base: np.ndarray, amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        value, _ = model.retrieval_input(torch.as_tensor(base), torch.as_tensor(amp), torch.as_tensor(phase))
    return value.numpy()


def train(model: StructuredRetrievalForecaster, base: np.ndarray, amp: np.ndarray, phase: np.ndarray,
          futures: np.ndarray, end_times: np.ndarray, epochs: int, seed: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    base_t, amp_t, phase_t, future_t = (torch.as_tensor(x) for x in (base, amp, phase, futures))
    for _ in range(epochs):
        cached_features = enhanced(model, base, amp, phase)
        bank = QuantumMemoryBank(model.retriever, cached_features, futures, end_times)
        rows = []
        for row in range(20, len(base)):
            with torch.no_grad():
                query_key = model.retriever.encode_query(torch.as_tensor(cached_features[row:row + 1])).numpy()[0]
            indices, _ = bank.index.query(query_key, 20, before=end_times[row])
            rows.append((row, indices))
        rng.shuffle(rows)
        model.train()
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            query_rows = np.asarray([row for row, _ in batch])
            indices = np.asarray([items for _, items in batch])
            retrieval_input, _ = model.retrieval_input(base_t[query_rows], amp_t[query_rows], phase_t[query_rows])
            prediction, _ = model(
                base_t[query_rows], amp_t[query_rows], phase_t[query_rows],
                torch.as_tensor(bank.keys[indices]), future_t[indices],
            )
            loss = F.mse_loss(prediction, future_t[query_rows].flatten(start_dim=1))
            loss = loss + 0.2 * model.retriever.contrastive_loss(retrieval_input, future_t[query_rows])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


def evaluate(model: StructuredRetrievalForecaster, bank: QuantumMemoryBank, base: np.ndarray, amp: np.ndarray,
             phase: np.ndarray, targets: np.ndarray, end_times: np.ndarray) -> dict[str, float]:
    encoded = enhanced(model, base, amp, phase)
    predicted = []
    for row, (feature, target, end_time) in enumerate(zip(encoded, targets, end_times)):
        result = bank.retrieve(feature, end_time, candidate_count=20, top_k=3, reranker="quantum_fidelity")
        with torch.no_grad():
            p, _ = model(torch.as_tensor(base[row:row + 1]), torch.as_tensor(amp[row:row + 1]),
                         torch.as_tensor(phase[row:row + 1]), torch.as_tensor(bank.keys[result.indices][None]),
                         torch.as_tensor(result.futures[None]))
        predicted.append(p.numpy()[0])
    pred, target = np.asarray(predicted), targets.reshape(len(targets), -1)
    return {"mse": float(np.mean((pred - target) ** 2)), "mae": float(np.mean(np.abs(pred - target)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl-root", type=Path, default=ROOT / "third_party/Time-Series-Library")
    parser.add_argument("--pred-len", type=int, choices=(96, 192, 336, 720), required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final_model")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_data, test_data = (
        load_tsl("ETTh1", split, seq_len=96, pred_len=args.pred_len, root=args.tsl_root)
        for split in ("train", "test")
    )
    n_train, n_test = (512, 128) if args.quick else (len(train_data.inputs), len(test_data.inputs))
    x, y, t = train_data.inputs[:n_train], train_data.targets[:n_train], train_data.end_times[:n_train]
    encoder = StructuredSequenceEncoder(StructuredEncoderConfig(patch_lengths=(24, 48, 96), period=24)).fit(x)
    base, amp, phase = features(encoder, x)
    test_base, test_amp, test_phase = features(encoder, test_data.inputs[:n_test])
    circuit = QuantumInteractionLayer(CircuitSpec(4, "hybrid", "cross_period", 1, "z"), seed=args.seed)
    retriever = TaskAwareRetriever(base.shape[1] + circuit.n_out, embedding_dim=32, quantum_dim=4)
    model = StructuredRetrievalForecaster(retriever, circuit, 1.05 * float(amp.max()), int(np.prod(y.shape[1:])))
    train(model, base, amp, phase, y, t, 1 if args.quick else 5, args.seed)
    bank = QuantumMemoryBank(model.retriever, enhanced(model, base, amp, phase), y, t)
    metrics = evaluate(model, bank, test_base, test_amp, test_phase, test_data.targets[:n_test], test_data.end_times[:n_test])
    report = {"model": "ModuleA-periodic-features -> ModuleC-query/key -> retrieval forecast", "dataset": "ETTh1",
              "pred_len": args.pred_len, "seed": args.seed, "quick": args.quick, "metrics": metrics}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"etth1_final_{args.pred_len}_s{args.seed}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
