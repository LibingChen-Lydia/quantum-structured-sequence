"""ETT hourly long-term-forecast protocol compatible with Time-Series-Library."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ETTH1_TRAIN_END = 12 * 30 * 24
ETTH1_VAL_END = 16 * 30 * 24
ETTH1_TEST_END = 20 * 30 * 24


@dataclass(frozen=True)
class ETTForecastDataset:
    """Sliding windows with the exact Dataset_ETT_hour split boundaries.

    ``inputs`` and ``targets`` are normalized with train-only statistics.  A
    sample's ``end_time`` is the first forecast timestamp and defines what
    historical memory is legal for that query.
    """

    inputs: np.ndarray
    targets: np.ndarray
    end_times: np.ndarray
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray


def _read_numeric_csv(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError("ETTh1 CSV must contain date plus numeric feature columns")
        names = tuple(name for name in reader.fieldnames if name != "date")
        values = [[float(row[name]) for name in names] for row in reader]
    array = np.asarray(values, dtype=np.float32)
    if len(array) < ETTH1_TEST_END:
        raise ValueError(f"ETTh1 must contain at least {ETTH1_TEST_END} hourly observations")
    return array, names


def _windows(data: np.ndarray, border1: int, border2: int, seq_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(border1, border2 - seq_len - pred_len + 1)
    inputs = np.stack([data[start : start + seq_len] for start in starts])
    targets = np.stack([data[start + seq_len : start + seq_len + pred_len] for start in starts])
    return inputs, targets, starts + seq_len


def load_etth1(
    path: str | Path, split: str, seq_len: int = 96, pred_len: int = 96
) -> ETTForecastDataset:
    """Load ETTh1 using Time-Series-Library's 12/4/4-month borders.

    The validation and test input borders include ``seq_len`` observations from
    the preceding split, exactly as ``Dataset_ETT_hour`` does.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if seq_len != 96:
        raise ValueError("this benchmark protocol fixes seq_len=96")
    raw, names = _read_numeric_csv(Path(path))
    # Accumulate statistics in float64: a float32 column-wise (axis=0) reduction over 8640 rows
    # drifts (e.g. mean 4319.335 instead of 4319.5). Time-Series-Library's StandardScaler is float64.
    train = raw[:ETTH1_TRAIN_END].astype(np.float64)
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), 1e-6)
    normalized = ((raw - mean) / std).astype(np.float32)
    borders = {
        "train": (0, ETTH1_TRAIN_END),
        "val": (ETTH1_TRAIN_END - seq_len, ETTH1_VAL_END),
        "test": (ETTH1_VAL_END - seq_len, ETTH1_TEST_END),
    }
    inputs, targets, end_times = _windows(normalized, *borders[split], seq_len, pred_len)
    return ETTForecastDataset(inputs, targets, end_times, names, mean, std)
