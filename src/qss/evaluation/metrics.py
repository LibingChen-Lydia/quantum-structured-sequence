from __future__ import annotations

import numpy as np


def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
