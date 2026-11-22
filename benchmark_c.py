"""Run the hierarchical retrieval model on the four TSL forecasting datasets.

The final numeric column is the target (OT for the two ETT datasets). All
scalers and retrieval memories are fitted from the training split only.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from c import HierarchicalPatchRetriever
from src.qss.classical.encoder import StructuredEncoderConfig, StructuredSequenceEncoder


DATA = {
    "ETTh1": ("ETT-small/ETTh1.csv", "hour"),
    "ETTm1": ("ETT-small/ETTm1.csv", "minute"),
    "weather": ("weather/weather.csv", "custom"),
    "traffic": ("traffic/traffic.csv", "custom"),
}
HORIZONS = (96, 192, 336, 420)
SEQ_LEN = 96
PATCH_LEN = 24


def _split_sizes(name: str, n: int) -> tuple[int, int, int]:
    if name == "ETTh1":
        train, val, test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    elif name == "ETTm1":
        train, val, test = 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
    else:
        train = int(n * 0.7)
        val = int(n * 0.1)
        test = n - train - val
    if train + val + test > n:
        raise ValueError(f"{name} has {n} rows, fewer than the TSL split requires")
    return train, val, test


def _embed_windows(encoder: StructuredSequenceEncoder, series: np.ndarray,
                   origins: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    chunks = []
    for start in range(0, len(origins), batch_size):
        batch_origins = origins[start : start + batch_size]
        windows = np.stack([series[o - SEQ_LEN : o] for o in batch_origins])
        encoded = encoder.transform(windows).scales[PATCH_LEN].embedding.mean(axis=1)
        chunks.append(encoded)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0))


def run_dataset(name: str, dataset_root: Path, result_dir: Path,
                quantum: bool, query_stride: int) -> list[dict]:
    relative, _ = DATA[name]
    frame = pd.read_csv(dataset_root / relative)
    if len(frame.columns) < 2:
        raise ValueError(f"Expected date plus at least one value column in {relative}")
    raw = frame.iloc[:, 1:].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError(f"{relative} contains missing or non-finite measurements")
    train_end, val_len, test_len = _split_sizes(name, len(raw))
    test_start = train_end + val_len
    mean = raw[:train_end].mean(axis=0)
    scale = raw[:train_end].std(axis=0)
    scale[scale < 1e-12] = 1.0
    series = (raw - mean) / scale
    target_index = raw.shape[1] - 1

    # Fit the encoder's feature normalization exclusively on training values.
    encoder = StructuredSequenceEncoder(
        StructuredEncoderConfig(patch_lengths=(PATCH_LEN,), period=24, stride=PATCH_LEN)
    ).fit(series[:train_end][None, :, :])
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"qss_{name}.csv"
    rows: list[dict] = []

    for horizon in HORIZONS:
        train_origins = np.arange(SEQ_LEN, train_end - horizon + 1)
        query_origins = np.arange(test_start, train_end + val_len + test_len - horizon + 1,
                                  query_stride)
        memory_embeddings = _embed_windows(encoder, series, train_origins)
        query_embeddings = _embed_windows(encoder, series, query_origins)
        payloads = [
            {"start": int(o - SEQ_LEN), "future": series[o : o + horizon, target_index].copy()}
            for o in train_origins
        ]
        retriever = HierarchicalPatchRetriever(
            n_clusters=min(8, len(train_origins)), top_r_clusters=2,
            candidate_pool=20, top_k=3, use_quantum=quantum,
        ).fit(memory_embeddings, payloads=payloads)

        predictions, actuals = [], []
        for origin, embedding in zip(query_origins, query_embeddings):
            hits = retriever.retrieve(embedding)
            if not hits:
                continue
            predictions.append(np.stack([h.payload["future"] for h in hits]).mean(axis=0))
            actuals.append(series[origin : origin + horizon, target_index])
        if not predictions:
            raise RuntimeError(f"No forecasts produced for {name} pred_len={horizon}")
        pred = np.concatenate(predictions)
        true = np.concatenate(actuals)
        pred_raw = pred * scale[target_index] + mean[target_index]
        true_raw = true * scale[target_index] + mean[target_index]
        row = {
            "dataset": name, "model": "QSS_HierarchicalRetriever",
            "target": str(frame.columns[-1]), "features": "MS",
            "seq_len": SEQ_LEN, "pred_len": horizon,
            "train_windows": len(train_origins), "test_windows": len(predictions),
            "query_stride": query_stride, "quantum_reranking": quantum,
            "mae": float(np.mean(np.abs(pred - true))),
            "mse": float(np.mean((pred - true) ** 2)),
            "mae_raw": float(np.mean(np.abs(pred_raw - true_raw))),
            "mse_raw": float(np.mean((pred_raw - true_raw) ** 2)),
        }
        rows.append(row)
        print(f"[{name} QSS] pred_len={horizon} MAE={row['mae']:.6f} MSE={row['mse']:.6f}", flush=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATA), default=list(DATA))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--classical-only", action="store_true",
                        help="Skip PennyLane reranking (faster; not the full quantum model).")
    parser.add_argument("--query-stride", type=int, default=1)
    args = parser.parse_args()
    if args.query_stride < 1:
        parser.error("--query-stride must be positive")
    rows = []
    for name in args.datasets:
        rows.extend(run_dataset(name, args.dataset_root, args.results,
                                not args.classical_only, args.query_stride))
    print(f"QSS CSVs saved under {args.results.resolve()}")


if __name__ == "__main__":
    main()
