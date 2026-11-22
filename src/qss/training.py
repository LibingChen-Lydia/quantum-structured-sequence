"""Shared training loop (Adam, MSE or BCE-with-logits, early stopping on validation loss)."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn


def _t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def predict(model: nn.Module, amp: np.ndarray, phase: np.ndarray, batch: int = 2048) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(amp), batch):
            outs.append(model(_t(amp[i:i + batch]), _t(phase[i:i + batch])).numpy())
    return np.concatenate(outs)


def _loss(out: torch.Tensor, y: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "bce":
        return nn.functional.binary_cross_entropy_with_logits(out, y)
    return torch.mean((out - y) ** 2)


def train(model: nn.Module, train_set, val_set, *, seed: int, lr: float = 0.01, epochs: int = 300,
          batch: int = 128, patience: int = 30, l1: float = 0.0, loss: str = "mse") -> dict:
    """``loss="bce"`` treats the model output as a logit for 0/1 labels (classification)."""
    if loss not in ("mse", "bce"):
        raise ValueError(f"unknown loss {loss!r}")
    torch.manual_seed(seed)
    amp, phase, y = (_t(a) for a in train_set)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=patience // 3)
    best, best_state, bad, history = np.inf, copy.deepcopy(model.state_dict()), 0, []
    t0 = time.perf_counter()
    n = len(y)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            obj = _loss(model(amp[idx], phase[idx]), y[idx], loss)
            if l1 and hasattr(model, "layer"):
                obj = obj + model.layer.regularization(l1)
            obj.backward()
            opt.step()
        with torch.no_grad():
            val = float(_loss(_t(predict(model, val_set[0], val_set[1])), _t(val_set[2]), loss))
        sched.step(val)
        history.append(val)
        if val < best - 1e-7:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_loss": best, "loss": loss, "epochs": epoch + 1, "train_seconds": time.perf_counter() - t0,
            "val_curve": history}
