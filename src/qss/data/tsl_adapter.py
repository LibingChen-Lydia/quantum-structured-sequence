"""Windows for Module C taken directly from Time-Series-Library's own data loaders.

The split, the train-only ``StandardScaler`` and the windowing all come from THUML's
``data_provider.data_loader`` (``Dataset_ETT_hour``, ``Dataset_ETT_minute``, ``Dataset_Custom``),
so Module C sees exactly the arrays the TSL baselines are trained and tested on. This module only
copies those arrays into the ``ETTForecastDataset`` layout used by the retrieval code.

``TSL_ROOT`` (env var or argument) must point to a Time-Series-Library checkout; see
``benchmarks/tsl/setup.sh``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qss.data.ett import ETTForecastDataset

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TSL_ROOT = ROOT / "third_party" / "Time-Series-Library"

# name -> (loader class, root sub-folder, csv file, TSL --data value, freq)
DATASETS = {
    "ETTh1": ("Dataset_ETT_hour", "ETT-small", "ETTh1.csv", "ETTh1", "h"),
    "ETTm1": ("Dataset_ETT_minute", "ETT-small", "ETTm1.csv", "ETTm1", "t"),
    "weather": ("Dataset_Custom", "weather", "weather.csv", "custom", "t"),
    "traffic": ("Dataset_Custom", "traffic", "traffic.csv", "custom", "h"),
}
SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


def tsl_root(path: str | Path | None = None) -> Path:
    root = Path(path or os.environ.get("TSL_ROOT", DEFAULT_TSL_ROOT)).resolve()
    if not (root / "data_provider" / "data_loader.py").exists():
        raise FileNotFoundError(f"Time-Series-Library not found at {root}; run benchmarks/tsl/setup.sh")
    return root


def tsl_dataset(name: str, split: str, seq_len: int = 96, label_len: int = 48, pred_len: int = 96,
                root: str | Path | None = None):
    """Instantiate the TSL dataset object itself (features='M', scale=True)."""
    loader, folder, filename, _, freq = DATASETS[name]
    troot = tsl_root(root)
    if str(troot) not in sys.path:
        sys.path.insert(0, str(troot))
    from data_provider import data_loader  # noqa: PLC0415  (TSL's own module)

    cls = getattr(data_loader, loader)
    args = SimpleNamespace(augmentation_ratio=0)
    return cls(args, str(troot / "dataset" / folder) + "/", flag=split, size=[seq_len, label_len, pred_len],
               features="M", data_path=filename, target="OT", scale=True, timeenc=1, freq=freq)


def _border1(name: str, split: str, n_rows: int, seq_len: int) -> int:
    """Global row of data_x[0], used only for the temporal-validity index (checked below)."""
    i = SPLIT_INDEX[split]
    if name == "ETTh1":
        unit = 30 * 24
    elif name == "ETTm1":
        unit = 30 * 24 * 4
    else:
        num_train, num_test = int(n_rows * 0.7), int(n_rows * 0.2)
        return [0, num_train - seq_len, n_rows - num_test - seq_len][i]
    return [0, 12 * unit - seq_len, 16 * unit - seq_len][i]


def load_tsl(name: str, split: str, seq_len: int = 96, pred_len: int = 96,
             root: str | Path | None = None, stride: int = 1) -> ETTForecastDataset:
    """TSL windows as float32 arrays. ``stride`` > 1 keeps every stride-th window (memory budget)."""
    ds = tsl_dataset(name, split, seq_len, 48, pred_len, root)
    n_rows = _n_rows(ds)
    b1 = _border1(name, split, n_rows, seq_len)
    starts = np.arange(0, len(ds), stride)
    x = ds.data_x
    inputs = np.stack([x[s:s + seq_len] for s in starts]).astype(np.float32)
    targets = np.stack([ds.data_y[s + seq_len:s + seq_len + pred_len] for s in starts]).astype(np.float32)
    names = tuple(_columns(ds))
    return ETTForecastDataset(inputs, targets, b1 + starts + seq_len, names,
                              ds.scaler.mean_.astype(np.float64), ds.scaler.scale_.astype(np.float64))


def _csv_path(ds) -> Path:
    return Path(ds.root_path) / ds.data_path


def _n_rows(ds) -> int:
    with open(_csv_path(ds), "rb") as handle:
        return sum(1 for _ in handle) - 1


def _columns(ds) -> list[str]:
    with open(_csv_path(ds)) as handle:
        header = handle.readline().strip().split(",")
    return [c for c in header if c != "date"]
