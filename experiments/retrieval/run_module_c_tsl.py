"""Module C on the four TSL long-term-forecast datasets (M mode, seq_len 96).

Data, split, train-only scaling and windows come from Time-Series-Library's own loaders through
``qss.data.tsl_adapter``, i.e. exactly what the TSL baselines see. The model and training are the
ETTh1 runner's (``run_etth1_module_c.py``), parameterised by dataset and seed.

    python experiments/retrieval/run_module_c_tsl.py --dataset ETTh1 --pred-len 96 --seed 2021
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_etth1_module_c import evaluate, train_forecast_head, train_retriever, window_features  # noqa: E402

from qss.classical import StructuredEncoderConfig, StructuredSequenceEncoder  # noqa: E402
from qss.data.tsl_adapter import DATASETS, load_tsl  # noqa: E402
from qss.retrieval import QuantumMemoryBank, TaskAwareRetriever  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RERANKERS = ("cosine", "learned_metric", "quantum_fidelity")


def package_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("numpy", "torch", "pennylane", "scikit-learn", "pandas", "hnswlib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--pred-len", type=int, choices=(96, 192, 336, 720), required=True)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--memory-stride", type=int, default=1, help="keep every k-th training window in memory")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rerankers", nargs="+", default=list(RERANKERS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/benchmark/module_c")
    a = ap.parse_args()
    out = a.output / f"{a.dataset}_96_{a.pred_len}_s{a.seed}.json"
    if out.exists() and not a.quick:
        print(f"skip {out.name}")
        return
    t_start = time.time()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    train = load_tsl(a.dataset, "train", 96, a.pred_len, stride=a.memory_stride)
    test = load_tsl(a.dataset, "test", 96, a.pred_len)
    n_tr = 512 if a.quick else len(train.inputs)
    n_te = 128 if a.quick else len(test.inputs)
    tr_x, tr_y, tr_t = train.inputs[:n_tr], train.targets[:n_tr], train.end_times[:n_tr]

    cfg = StructuredEncoderConfig(patch_lengths=(24, 48, 96), period=24)
    encoder = StructuredSequenceEncoder(cfg).fit(tr_x)
    tr_f = window_features(encoder, tr_x, 24)
    te_f = window_features(encoder, test.inputs[:n_te], 24)
    retriever = TaskAwareRetriever(tr_f.shape[1], embedding_dim=32, quantum_dim=4)
    epochs = 1 if a.quick else a.epochs
    t0 = time.time()
    train_retriever(retriever, tr_f, tr_y, epochs, batch_size=64, seed=a.seed)
    bank = QuantumMemoryBank(retriever, tr_f, tr_y, tr_t)
    head = train_forecast_head(retriever, bank, tr_f, tr_y, tr_t, epochs, candidate_count=20, seed=a.seed)
    train_secs = time.time() - t0
    results = {}
    for r in a.rerankers:
        t1 = time.time()
        results[r] = evaluate(retriever, head, bank, te_f, test.targets[:n_te], test.end_times[:n_te], r, 20, 3)
        results[r]["eval_seconds"] = round(time.time() - t1, 1)
        print(f"[{a.dataset} H={a.pred_len} s={a.seed}] {r}: mse={results[r]['mse']:.4f} mae={results[r]['mae']:.4f}",
              flush=True)
    rec = {
        "model": "ModuleC_TaskAwareRetrieval", "dataset": a.dataset, "features": "M", "seq_len": 96,
        "pred_len": a.pred_len, "seed": a.seed, "status": "OK", "quick": a.quick,
        "data_source": "Time-Series-Library data_provider (qss.data.tsl_adapter)",
        "tsl_loader": DATASETS[a.dataset][0], "channels": int(train.inputs.shape[-1]),
        "train_windows_in_memory": int(n_tr), "memory_stride": a.memory_stride, "test_windows": int(n_te),
        "encoder": {"config": asdict(cfg), "feature_scale": 24},
        "normalization": {"mean": train.mean.tolist(), "std": train.std.tolist(), "fit_on": "TSL train split"},
        "candidates": 20, "top_k": 3, "epochs": epochs, "train_seconds": round(train_secs, 1),
        "results": results, "wall_seconds": round(time.time() - t_start, 1),
        "device": "cpu", "host": platform.node(), "versions": package_versions(),
        "command": shlex.join([sys.executable, *sys.argv]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    if not a.quick:
        out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}" if not a.quick else json.dumps({k: v["mse"] for k, v in results.items()}))


if __name__ == "__main__":
    main()
