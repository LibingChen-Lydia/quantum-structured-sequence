"""Quantum-classical fusion (proposal section 8.2):  h~ = h + g * W_q q,  y = head(h~)."""

from __future__ import annotations

import torch
from torch import nn

from qss.quantum.interaction import QuantumInteractionLayer


class GatedFusion(nn.Module):
    def __init__(self, d_hidden: int, d_quantum: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_quantum, d_hidden).double()
        self.gate = nn.Linear(d_hidden + d_quantum, d_hidden).double()

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([h, q], -1)))
        return h + g * self.proj(q)


def phase_head_features(q: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
    """Phase-only circuits never see A, so the head combines them classically:
    [q, A, q * A_i A_j for every item pair]."""
    n = amp.shape[-1]
    pair = torch.stack([amp[..., i] * amp[..., j] for i in range(n) for j in range(i + 1, n)], -1)
    cross = (q[..., :, None] * pair[..., None, :]).flatten(-2)
    return torch.cat([q, amp, cross], -1)


def phase_head_dim(n_out: int, n_items: int) -> int:
    return n_out + n_items + n_out * (n_items * (n_items - 1) // 2)


class QuantumRegressor(nn.Module):
    """Pure quantum feature model: y = w . q(theta, phi) + b  (phase encoding also sees A)."""

    def __init__(self, layer: QuantumInteractionLayer, amp_max: float) -> None:
        super().__init__()
        self.layer = layer
        self.amp_max = amp_max
        n_in = (phase_head_dim(layer.n_out, layer.spec.n_items) if layer.spec.encoding == "phase"
                else layer.n_out)
        self.head = nn.Linear(n_in, 1).double()

    def forward(self, amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        theta = torch.arcsin(torch.clamp(amp / self.amp_max, 0.0, 1.0))
        q = self.layer(theta, phase)
        if self.layer.spec.encoding == "phase":
            q = phase_head_features(q, amp)
        return self.head(q).squeeze(-1)


class FusedForecaster(nn.Module):
    """Classical MLP backbone on raw features plus gated quantum features."""

    def __init__(self, layer: QuantumInteractionLayer, amp_max: float, hidden: int = 32) -> None:
        super().__init__()
        n = layer.spec.n_items
        self.layer = layer
        self.amp_max = amp_max
        self.backbone = nn.Sequential(nn.Linear(2 * n, hidden), nn.Tanh()).double()
        self.fuse = GatedFusion(hidden, layer.n_out)
        self.head = nn.Linear(hidden, 1).double()

    def forward(self, amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        h = self.backbone(torch.cat([amp, phase], -1))
        theta = torch.arcsin(torch.clamp(amp / self.amp_max, 0.0, 1.0))
        return self.head(self.fuse(h, self.layer(theta, phase))).squeeze(-1)
