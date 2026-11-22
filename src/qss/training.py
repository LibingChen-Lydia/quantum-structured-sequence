"""Shared training loop (Adam, MSE, early stopping on validation)."""

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


def train(model: nn.Module, train_set, val_set, *, seed: int, lr: float = 0.01, epochs: int = 300,
          batch: int = 128, patience: int = 30) -> dict:
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
            loss = torch.mean((model(amp[idx], phase[idx]) - y[idx]) ** 2)
            loss.backward()
            opt.step()
        val = float(np.mean((predict(model, val_set[0], val_set[1]) - val_set[2]) ** 2))
        sched.step(val)
        history.append(val)
        if val < best - 1e-7:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_mse": best, "epochs": epoch + 1, "train_seconds": time.perf_counter() - t0,
            "val_curve": history}
