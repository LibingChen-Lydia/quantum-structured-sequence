from __future__ import annotations

import numpy as np


def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def classification_metrics(y: np.ndarray, logits: np.ndarray) -> dict:
    """Accuracy, macro-F1 and BCE for 0/1 labels from logits (threshold at 0)."""
    from sklearn.metrics import accuracy_score, f1_score

    y = np.asarray(y).astype(int)
    z = np.asarray(logits, dtype=np.float64)
    pred = (z > 0).astype(int)
    bce = np.mean(np.logaddexp(0.0, z) - y * z)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")), "bce": float(bce)}
