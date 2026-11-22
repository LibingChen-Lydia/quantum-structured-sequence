"""Run Transformer / iTransformer / PatchTST with Time-Series-Library's official scripts.

Hyper-parameters are parsed from TSL's own ``scripts/long_term_forecast/*.sh`` (one command block
per horizon), so nothing is re-tuned here. Only run-time settings are overridden: seed (via the
``TSL_SEED`` patch from setup.sh), device, data path, checkpoint dir and ``num_workers``.
TSL ships no iTransformer script for ETTh1/ETTm1; its ETTh2 script is used with the data swapped
(recorded as ``script_source`` in every result).

Jobs are ordered seed-first (every cell gets seed 2021 before any cell gets 2022), resumable,
and can be sharded over parallel processes:

    python benchmarks/tsl/run_baselines.py --device cuda --shard 0/4 &
    python benchmarks/tsl/run_baselines.py --device cuda --shard 1/4 &  ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TSL = Path(os.environ.get("TSL_ROOT", ROOT / "third_party" / "Time-Series-Library")).resolve()
OUT = ROOT / "outputs" / "benchmark" / "tsl"

MODELS = ("Transformer", "iTransformer", "PatchTST")
DATASETS = ("ETTh1", "ETTm1", "weather", "traffic")
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023, 2024, 2025)
SCRIPT = {
    ("Transformer", "ETTh1"): "ETT_script/Transformer_ETTh1.sh",
    ("Transformer", "ETTm1"): "ETT_script/Transformer_ETTm1.sh",
    ("Transformer", "weather"): "Weather_script/Transformer.sh",
    ("Transformer", "traffic"): "Traffic_script/Transformer.sh",
    ("iTransformer", "ETTh1"): "ETT_script/iTransformer_ETTh2.sh",
    ("iTransformer", "ETTm1"): "ETT_script/iTransformer_ETTh2.sh",
    ("iTransformer", "weather"): "Weather_script/iTransformer.sh",
    ("iTransformer", "traffic"): "Traffic_script/iTransformer.sh",
    ("PatchTST", "ETTh1"): "ETT_script/PatchTST_ETTh1.sh",
    ("PatchTST", "ETTm1"): "ETT_script/PatchTST_ETTm1.sh",
    ("PatchTST", "weather"): "Weather_script/PatchTST.sh",
    ("PatchTST", "traffic"): "Traffic_script/PatchTST.sh",
}
DATA_SWAP = {"ETTh1": ("ETTh1.csv", "ETTh1"), "ETTm1": ("ETTm1.csv", "ETTm1")}
METRIC_RE = re.compile(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)")


def parse_script(path: Path, model: str) -> dict[int, list[str]]:
    """Return {pred_len: run.py argument list} from an official TSL shell script."""
    text = path.read_text().replace("\\\n", " ").replace("$model_name", model)
    blocks = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("python") or "run.py" not in line:
            continue
        args = shlex.split(line)[3:]  # drop "python -u run.py"
        pred = int(args[args.index("--pred_len") + 1])
        blocks[pred] = args
    return blocks


def set_arg(args: list[str], key: str, value: str | None) -> list[str]:
    args = list(args)
    if key in args:
        i = args.index(key)
        if value is None:
            del args[i]
        else:
            args[i + 1] = value
    elif value is not None:
        args += [key, value]
    return args


def build(model: str, dataset: str, horizon: int, seed: int, device: str, epochs: int | None) -> list[str]:
    rel = SCRIPT[(model, dataset)]
    args = parse_script(TSL / "scripts" / "long_term_forecast" / rel, model)[horizon]
    if dataset in DATA_SWAP:
        csv, data = DATA_SWAP[dataset]
        args = set_arg(set_arg(args, "--data_path", csv), "--data", data)
    folder = {"ETTh1": "ETT-small", "ETTm1": "ETT-small", "weather": "weather", "traffic": "traffic"}[dataset]
    args = set_arg(args, "--root_path", str(TSL / "dataset" / folder) + "/")
    args = set_arg(args, "--model_id", f"{dataset}_96_{horizon}_s{seed}")
    args = set_arg(args, "--checkpoints", str(OUT / "checkpoints") + "/")
    args = set_arg(args, "--num_workers", "2")
    args = set_arg(args, "--des", "qss-benchmark")
    if epochs is not None:
        args = set_arg(args, "--train_epochs", str(epochs))
    if device == "cpu":
        args += ["--no_use_gpu"]
    else:
        args = set_arg(args, "--gpu_type", device)
    return [sys.executable, "-u", "run.py", *args]


def jobs(a) -> list[tuple]:
    out = []
    for seed in a.seeds:  # seed-first ordering
        for dataset in a.datasets:
            for horizon in a.horizons:
                for model in a.models:
                    out.append((model, dataset, horizon, seed))
    i, n = map(int, a.shard.split("/"))
    return out[i::n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--device", default="cuda", choices=("cuda", "mps", "cpu"))
    ap.add_argument("--epochs", type=int, default=None, help="override train_epochs (smoke tests only)")
    ap.add_argument("--shard", default="0/1", help="i/N: run every N-th job starting at i")
    ap.add_argument("--max-hours", type=float, default=None, help="stop starting new jobs after this")
    ap.add_argument("--tag", default="", help="output sub-folder, e.g. 'smoke'")
    a = ap.parse_args()
    out_dir = OUT / (a.tag or "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    todo = jobs(a)
    print(f"{len(todo)} jobs in shard {a.shard} -> {out_dir}", flush=True)
    for k, (model, dataset, horizon, seed) in enumerate(todo, 1):
        path = out_dir / f"{model}_{dataset}_96_{horizon}_s{seed}.json"
        if path.exists() and json.loads(path.read_text()).get("status") == "OK":
            print(f"[skip] {path.name}", flush=True)
            continue
        if a.max_hours and time.time() - t_start > a.max_hours * 3600:
            print("max-hours reached, stopping", flush=True)
            break
        cmd = build(model, dataset, horizon, seed, a.device, a.epochs)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=TSL, env={**os.environ, "TSL_SEED": str(seed)}, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        secs = time.time() - t0
        log = out_dir / "logs" / f"{path.stem}.log"
        log.parent.mkdir(exist_ok=True)
        log.write_text(proc.stdout)
        found = METRIC_RE.findall(proc.stdout)
        ok = proc.returncode == 0 and bool(found)
        rec = {"model": model, "dataset": dataset, "features": "M", "seq_len": 96, "pred_len": horizon,
               "seed": seed, "status": "OK" if ok else "FAILED",
               "mse": float(found[-1][0]) if ok else None, "mae": float(found[-1][1]) if ok else None,
               "wall_seconds": round(secs, 1), "device": a.device, "host": platform.node(),
               "script_source": f"TSL scripts/long_term_forecast/{SCRIPT[(model, dataset)]}",
               "tsl_commit": "4e938a1767106324dd753b2a44832bf870a0252e",
               "epochs_override": a.epochs, "command": shlex.join(cmd[1:]), "log": str(log.relative_to(ROOT)),
               "error": "" if ok else "\n".join(proc.stdout.splitlines()[-25:])}
        try:
            import torch
            rec["torch"] = torch.__version__
            rec["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            pass
        path.write_text(json.dumps(rec, indent=2))
        print(f"[{k}/{len(todo)} {rec['status']}] {model} {dataset} H={horizon} s={seed} "
              f"mse={rec['mse']} mae={rec['mae']} {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
