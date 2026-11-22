"""Classical controls for Module A. Every model maps (amp, phase) of the items to y."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from torch import nn


def mlp(n_in: int, hidden: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    width = n_in
    for _ in range(layers):
        mods += [nn.Linear(width, hidden), nn.Tanh()]
        width = hidden
    mods.append(nn.Linear(width, 1))
    return nn.Sequential(*mods).double()


class RawMLP(nn.Module):
    """MLP on raw (A, phi). Has to learn the phase wrap-around at +-pi itself."""

    def __init__(self, n_items: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = mlp(2 * n_items, hidden)

    def forward(self, amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([amp, phase], -1)).squeeze(-1)


class FourierMLP(nn.Module):
    """MLP on Fourier features (A, cos phi, sin phi): the strongest generic classical control."""

    def __init__(self, n_items: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = mlp(3 * n_items, hidden)

    def forward(self, amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([amp, torch.cos(phase), torch.sin(phase)], -1)).squeeze(-1)


def matched_hidden(n_in: int, n_params: int) -> int:
    """Hidden width h of a 1-hidden-layer MLP with about n_params parameters: (n_in + 2) h + 1."""
    return max(1, round((n_params - 1) / (n_in + 2)))


class MatchedMLP(nn.Module):
    """One-hidden-layer MLP on Fourier features with ~ the quantum model's parameter count."""

    def __init__(self, n_items: int, n_params: int) -> None:
        super().__init__()
        self.net = mlp(3 * n_items, matched_hidden(3 * n_items, n_params), layers=1)

    def forward(self, amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([amp, torch.cos(phase), torch.sin(phase)], -1)).squeeze(-1)


class ExplicitInteraction:
    """Ridge regression on z = [A cos phi, A sin phi] and all pairwise products z_a z_b.

    The S1/S2 targets lie exactly in this feature space, so this is an oracle-structured
    classical control: it shows what knowing the right interaction form buys.
    """

    def _features(self, amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
        z = np.concatenate([amp * np.cos(phase), amp * np.sin(phase)], axis=1)
        prods = [z[:, a] * z[:, b] for a, b in itertools.combinations_with_replacement(range(z.shape[1]), 2)]
        return np.column_stack([z, *prods])

    def fit(self, amp: np.ndarray, phase: np.ndarray, y: np.ndarray) -> ExplicitInteraction:
        self.model = RidgeCV(alphas=np.logspace(-6, 2, 17)).fit(self._features(amp, phase), y)
        self.n_params = self.model.coef_.size + 1
        return self

    def predict(self, amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
        return self.model.predict(self._features(amp, phase))
