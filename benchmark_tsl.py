"""Train/evaluate the four requested TSL baselines and record normalized MAE/MSE."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


DATASETS = {
    "ETTh1": ("ETT-small", "ETTh1.csv", "ETTh1", "h"),
    "ETTm1": ("ETT-small", "ETTm1.csv", "ETTm1", "t"),
    "weather": ("weather", "weather.csv", "custom", "t"),
    "traffic": ("traffic", "traffic.csv", "custom", "h"),
}
MODELS = ("Transformer", "iTransformer", "PatchTST", "DLinear")
HORIZONS = (96, 192, 336, 420)
METRIC_RE = re.compile(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)")


def _model_args(model: str) -> list[str]:
    # TSL model-specific defaults follow its official long-term forecast scripts.
    if model == "Transformer":
        return ["--e_layers", "2", "--d_layers", "1", "--factor", "3"]
    if model == "iTransformer":
        return ["--e_layers", "2", "--d_layers", "1", "--d_model", "512",
                "--d_ff", "512", "--factor", "3"]
    if model == "PatchTST":
        return ["--e_layers", "1", "--n_heads", "2", "--patch_len", "16",
                "--d_model", "128", "--d_ff", "256"]
    return ["--e_layers", "2", "--d_layers", "1", "--factor", "3"]


def run_one(tsl_root: Path, dataset_root: Path, results_dir: Path,
            dataset: str, model: str, horizon: int, epochs: int,
            batch_size: int, learning_rate: float) -> dict:
    folder, filename, data_kind, freq = DATASETS[dataset]
    file_path = dataset_root / folder / filename
    frame = pd.read_csv(file_path, nrows=1)
    enc_in = len(frame.columns) - 1
    target = str(frame.columns[-1])
    model_id = f"{dataset}_seq96_pred{horizon}_{model}_MS"
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{model_id}.log"
    command = [
        sys.executable, "-u", "run.py",
        "--task_name", "long_term_forecast", "--is_training", "1",
        "--model_id", model_id, "--model", model,
        "--data", data_kind, "--root_path", str((dataset_root / folder).resolve()) + "/",
        "--data_path", filename, "--features", "MS", "--target", target,
        "--freq", freq, "--seq_len", "96", "--label_len", "48",
        "--pred_len", str(horizon), "--enc_in", str(enc_in),
        "--dec_in", str(enc_in), "--c_out", str(enc_in),
        "--train_epochs", str(epochs), "--patience", "3",
        "--batch_size", str(batch_size), "--learning_rate", str(learning_rate),
        "--num_workers", "0", "--itr", "1", "--seed", "2021",
        "--checkpoints", str((results_dir / "checkpoints").resolve()) + "/",
        "--des", "C1-benchmark", "--no_use_gpu",
    ] + _model_args(model)
    start = time.time()
    completed = subprocess.run(command, cwd=tsl_root, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    log_path.write_text(completed.stdout, encoding="utf-8")
    matches = METRIC_RE.findall(completed.stdout)
    if completed.returncode != 0 or not matches:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        return {
            "dataset": dataset, "model": model, "target": target,
            "features": "MS", "seq_len": 96, "pred_len": horizon,
            "mae": "", "mse": "", "status": "FAILED",
            "elapsed_seconds": round(elapsed, 3), "log_file": str(log_path),
            "error": tail,
        }
    mse, mae = map(float, matches[-1])
    return {
        "dataset": dataset, "model": model, "target": target,
        "features": "MS", "seq_len": 96, "pred_len": horizon,
        "mae": mae, "mse": mse, "status": "OK",
        "elapsed_seconds": round(elapsed, 3), "log_file": str(log_path),
        "error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsl-root", type=Path, default=Path("Time-Series-Library"))
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true",
                        help="Skip successful rows already present in baseline_metrics.csv")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    output_path = args.results / "baseline_metrics.csv"
    rows = []
    if args.resume and output_path.exists():
        with output_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    done = {(r["dataset"], r["model"], int(r["pred_len"])) for r in rows if r["status"] == "OK"}
    fields = ["dataset", "model", "target", "features", "seq_len", "pred_len",
              "mae", "mse", "status", "elapsed_seconds", "log_file", "error"]
    for dataset in args.datasets:
        for model in args.models:
            for horizon in args.horizons:
                key = (dataset, model, horizon)
                if key in done:
                    print(f"[skip existing] {key}", flush=True)
                    continue
                # TSL's published Transformer scripts use 3 epochs on Weather/Traffic.
                epochs = min(args.epochs, 3) if dataset in {"weather", "traffic"} and model == "Transformer" else args.epochs
                batch = min(args.batch_size, 16) if dataset == "traffic" else args.batch_size
                row = run_one(args.tsl_root.resolve(), args.dataset_root.resolve(),
                              args.results.resolve(), dataset, model, horizon,
                              epochs, batch, args.learning_rate)
                rows = [r for r in rows if not (r["dataset"] == dataset and r["model"] == model and int(r["pred_len"]) == horizon)]
                rows.append(row)
                with output_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"[{row['status']}] {dataset} {model} pred_len={horizon} "
                      f"MAE={row['mae']} MSE={row['mse']} elapsed={row['elapsed_seconds']}s",
                      flush=True)
    print(f"Baseline results saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
