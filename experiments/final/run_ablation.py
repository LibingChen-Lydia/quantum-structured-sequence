"""Minimal Module A / Module C ablations for the final forecasting model.

Examples:
  PYTHONPATH=src python experiments/final/run_ablation.py --dataset ETTh1 --pred-len 96 --variant full
  PYTHONPATH=src python experiments/final/run_ablation.py --dataset weather --pred-len 192 --variant no_module_a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from qss.classical import StructuredEncoderConfig, StructuredSequenceEncoder
from qss.data.tsl_adapter import load_tsl
from qss.quantum.fidelity import AngleFidelityKernel
from qss.retrieval import TaskAwareForecastHead, TaskAwareRetriever, TemporalANNIndex

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_COUNT = 20
TOP_K = 3


class AcceleratorMemoryBank:
    """CPU temporal ANN with accelerator-resident query/key encoders.

    The index and the bounded Top-M reranking interface remain NumPy because
    their work is small.  Training and inference encoders stay on ``device``.
    """

    def __init__(self, retriever: TaskAwareRetriever, features: np.ndarray, futures: np.ndarray,
                 end_times: np.ndarray, device: torch.device) -> None:
        self.retriever = retriever.eval()
        self.features = np.asarray(features, dtype=np.float32)
        self.futures = np.asarray(futures, dtype=np.float32)
        self.device = device
        with torch.no_grad():
            feature_tensor = torch.as_tensor(self.features, device=device)
            keys = retriever.encode_key(feature_tensor)
            quantum_keys = retriever.encode_quantum(keys)
        self.keys = keys.detach().cpu().numpy()
        self.quantum_keys = quantum_keys.detach().cpu().numpy()
        self.raw_keys = self.features / np.maximum(np.linalg.norm(self.features, axis=-1, keepdims=True), 1e-12)
        self.index = TemporalANNIndex(self.keys, end_times)
        self.kernel = AngleFidelityKernel(n_qubits=retriever.quantum_dim)

    def retrieve(self, query_features: np.ndarray, before: float, reranker: str) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            query = torch.as_tensor(query_features[None], device=self.device)
            key = self.retriever.encode_query(query)
            quantum = self.retriever.encode_quantum(key)
        candidate_indices, learned_scores = self.index.query(key.detach().cpu().numpy()[0], CANDIDATE_COUNT, before)
        raw_query = query_features / max(float(np.linalg.norm(query_features)), 1e-12)
        raw_scores = np.einsum("nd,d->n", self.raw_keys[candidate_indices], raw_query)
        if reranker == "learned_metric":
            scores = learned_scores
        else:
            scores = self.kernel.score_candidates(quantum.detach().cpu().numpy()[0], self.quantum_keys[candidate_indices])
        selected = candidate_indices[np.argsort(-scores)[:TOP_K]]
        return selected, self.futures[selected]


def structured_features(
    encoder: StructuredSequenceEncoder, windows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from qss.fusion import pooled_periodic_tokens

    scale = encoder.transform(windows).scales[24]
    amplitude, phase = pooled_periodic_tokens(scale)
    return scale.embedding.reshape(len(windows), -1).astype(np.float32), amplitude, phase


def retrieval_features(
    variant: str, model: Any,
    retriever: TaskAwareRetriever, base: np.ndarray, amplitude: np.ndarray | None, phase: np.ndarray | None,
    device: torch.device,
) -> np.ndarray:
    """Create the exact retrieval input used by a variant at inference."""
    if variant == "no_module_a":
        return base
    assert amplitude is not None and phase is not None
    model.eval()
    with torch.no_grad():
        value, _ = model.retrieval_input(
            torch.as_tensor(base, device=device), torch.as_tensor(amplitude, device=device),
            torch.as_tensor(phase, device=device),
        )
    return value.detach().cpu().numpy()


def predict(
    variant: str, model: Any,
    retriever: TaskAwareRetriever, base: torch.Tensor, amplitude: torch.Tensor | None, phase: torch.Tensor | None,
    candidate_keys: torch.Tensor, candidate_futures: torch.Tensor,
) -> torch.Tensor:
    if variant == "no_module_a":
        query = retriever.encode_query(base)
        prediction, _ = model(query, candidate_keys, candidate_futures)  # type: ignore[operator]
        return prediction
    assert amplitude is not None and phase is not None
    prediction, _ = model(base, amplitude, phase, candidate_keys, candidate_futures)
    return prediction


def train(
    variant: str, model: Any, retriever: TaskAwareRetriever,
    base: np.ndarray, amplitude: np.ndarray | None, phase: np.ndarray | None, futures: np.ndarray,
    end_times: np.ndarray, epochs: int, seed: int, device: torch.device,
) -> None:
    parameters = list(model.parameters())
    if variant == "no_module_a":
        parameters += list(retriever.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3)
    rng = np.random.default_rng(seed)
    base_t, future_t = torch.as_tensor(base, device=device), torch.as_tensor(futures, device=device)
    amplitude_t = None if amplitude is None else torch.as_tensor(amplitude, device=device)
    phase_t = None if phase is None else torch.as_tensor(phase, device=device)
    for _ in range(epochs):
        cached = retrieval_features(variant, model, retriever, base, amplitude, phase, device)
        bank = AcceleratorMemoryBank(retriever, cached, futures, end_times, device)
        rows: list[tuple[int, np.ndarray]] = []
        for row in range(CANDIDATE_COUNT, len(base)):
            with torch.no_grad():
                query = retriever.encode_query(torch.as_tensor(cached[row : row + 1], device=device)).detach().cpu().numpy()[0]
            indices, _ = bank.index.query(query, CANDIDATE_COUNT, before=end_times[row])
            rows.append((row, indices))
        rng.shuffle(rows)
        model.train()
        retriever.train()
        for start in range(0, len(rows), 32):
            batch = rows[start : start + 32]
            query_rows = np.asarray([row for row, _ in batch])
            indices = np.asarray([items for _, items in batch])
            input_features = torch.as_tensor(cached[query_rows], device=device)
            prediction = predict(
                variant, model, retriever, base_t[query_rows],
                None if amplitude_t is None else amplitude_t[query_rows],
                None if phase_t is None else phase_t[query_rows],
                retriever.encode_key(torch.as_tensor(cached[indices], device=device)), future_t[indices],
            )
            loss = F.mse_loss(prediction, future_t[query_rows].flatten(start_dim=1))
            loss = loss + 0.2 * retriever.contrastive_loss(input_features, future_t[query_rows])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()


def evaluate(
    variant: str, model: Any, retriever: TaskAwareRetriever,
    bank: AcceleratorMemoryBank, base: np.ndarray, amplitude: np.ndarray | None, phase: np.ndarray | None,
    targets: np.ndarray, end_times: np.ndarray, device: torch.device,
) -> dict[str, float]:
    cached = retrieval_features(variant, model, retriever, base, amplitude, phase, device)
    reranker = "learned_metric" if variant == "no_module_c" else "quantum_fidelity"
    predicted: list[np.ndarray] = []
    for row, (feature, end_time) in enumerate(zip(cached, end_times)):
        indices, candidate_futures = bank.retrieve(feature, end_time, reranker)
        with torch.no_grad():
            prediction = predict(
                variant, model, retriever, torch.as_tensor(base[row : row + 1], device=device),
                None if amplitude is None else torch.as_tensor(amplitude[row : row + 1], device=device),
                None if phase is None else torch.as_tensor(phase[row : row + 1], device=device),
                torch.as_tensor(bank.keys[indices][None], device=device), torch.as_tensor(candidate_futures[None], device=device),
            )
        predicted.append(prediction.detach().cpu().numpy()[0])
    pred = np.asarray(predicted)
    target = targets.reshape(len(targets), -1)
    return {"mse": float(np.mean((pred - target) ** 2)), "mae": float(np.mean(np.abs(pred - target)))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("ETTh1", "weather"), required=True)
    parser.add_argument("--tsl-root", type=Path, default=ROOT / "third_party/Time-Series-Library")
    parser.add_argument("--pred-len", type=int, choices=(96, 192, 336, 720), required=True)
    parser.add_argument("--seed", type=int, choices=(2021, 2022, 2023), required=True)
    parser.add_argument("--variant", choices=("full", "no_module_a", "no_module_c"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--quantum-device", default="lightning.gpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ablation")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires an available CUDA device")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    train_data, test_data = (
        load_tsl(args.dataset, split, seq_len=96, pred_len=args.pred_len, root=args.tsl_root)
        for split in ("train", "test")
    )
    n_train, n_test = (512, 128) if args.quick else (len(train_data.inputs), len(test_data.inputs))
    x, y, end_times = train_data.inputs[:n_train], train_data.targets[:n_train], train_data.end_times[:n_train]
    if args.variant == "no_module_a":
        base, test_base = x.reshape(len(x), -1).astype(np.float32), test_data.inputs[:n_test].reshape(n_test, -1).astype(np.float32)
        amplitude = phase = test_amplitude = test_phase = None
        retriever = TaskAwareRetriever(base.shape[1], embedding_dim=32, quantum_dim=4)
        model: Any = TaskAwareForecastHead(
            retriever.embedding_dim, int(np.prod(y.shape[1:]))
        )
    else:
        from qss.fusion import StructuredRetrievalForecaster
        from qss.quantum.circuits import CircuitSpec
        from qss.quantum.interaction import QuantumInteractionLayer

        encoder = StructuredSequenceEncoder(StructuredEncoderConfig(patch_lengths=(24, 48, 96), period=24)).fit(x)
        base, amplitude, phase = structured_features(encoder, x)
        test_base, test_amplitude, test_phase = structured_features(encoder, test_data.inputs[:n_test])
        circuit = QuantumInteractionLayer(
            CircuitSpec(4, "hybrid", "cross_period", 1, "z"), seed=args.seed,
            device=args.quantum_device if args.device == "cuda" else "default.qubit",
        )
        retriever = TaskAwareRetriever(base.shape[1] + circuit.n_out, embedding_dim=32, quantum_dim=4)
        model = StructuredRetrievalForecaster(retriever, circuit, 1.05 * float(amplitude.max()), int(np.prod(y.shape[1:])))
    model.to(device)
    retriever.to(device)
    train(args.variant, model, retriever, base, amplitude, phase, y, end_times, 1 if args.quick else 5, args.seed, device)
    train_cached = retrieval_features(args.variant, model, retriever, base, amplitude, phase, device)
    bank = AcceleratorMemoryBank(retriever, train_cached, y, end_times, device)
    metrics = evaluate(
        args.variant, model, retriever, bank, test_base, test_amplitude, test_phase,
        test_data.targets[:n_test], test_data.end_times[:n_test], device,
    )
    report = {
        "dataset": args.dataset, "seq_len": 96, "pred_len": args.pred_len, "seed": args.seed,
        "variant": args.variant, "quick": args.quick, "device": str(device), "quantum_device": args.quantum_device,
        "candidate_count": CANDIDATE_COUNT, "top_k": TOP_K,
        "metrics": metrics,
    }
    destination = args.output / ("module_a" if args.variant == "no_module_a" else "module_c" if args.variant == "no_module_c" else "full")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{args.dataset}_96_{args.pred_len}_s{args.seed}_{args.variant}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
