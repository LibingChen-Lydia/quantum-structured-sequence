"""Quantum interaction layer (trainable, torch) and InteractionBackend implementation."""

from __future__ import annotations

import time
from functools import partial

import numpy as np
import pennylane as qml
import torch
from torch import nn

from qss.quantum.circuits import CircuitSpec, circuit


def make_qnode(spec: CircuitSpec, device: str = "default.qubit", shots: int | None = None):
    dev = qml.device(device, wires=spec.n_wires)
    fn = partial(circuit, spec)
    qnode = qml.QNode(
        fn, dev, interface="torch", diff_method="backprop" if shots is None else "parameter-shift"
    )
    if shots is not None:
        qnode = qml.set_shots(qnode, shots=shots)
    return qnode


class QuantumInteractionLayer(nn.Module):
    """(theta, phi) -> vector of measured expectations q, differentiable via backprop."""

    def __init__(self, spec: CircuitSpec, seed: int = 0, device: str = "default.qubit") -> None:
        super().__init__()
        self.spec = spec
        g = torch.Generator().manual_seed(seed)
        init = torch.rand(spec.weight_shape, generator=g, dtype=torch.float64) * 2 * np.pi
        self.weights = nn.Parameter(init)
        # learned topology: start with weak entanglers so training must switch edges on
        ent = (torch.rand(spec.ent_shape, generator=g, dtype=torch.float64) - 0.5) * 0.6
        self.ent = nn.Parameter(ent) if spec.learned else None
        self.qnode = make_qnode(spec, device)
        self.n_out = len(spec.observables())
        self.calls = 0  # circuit executions (one per sample per measurement setting)

    def forward(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        out = self.qnode(theta, phi, self.weights, self.ent)
        self.calls += theta.shape[0]
        q = torch.stack(list(out), dim=-1) if isinstance(out, (list, tuple)) else out[..., None]
        return q.reshape(theta.shape[0], self.n_out)

    def edge_strength(self) -> np.ndarray | None:
        """|sin(gamma)| per (layer, edge) for the learned topology."""
        if self.ent is None:
            return None
        return torch.abs(torch.sin(self.ent)).detach().numpy()

    def regularization(self, l1: float) -> torch.Tensor:
        """L1 penalty on entangling strength: pushes unneeded edges to zero."""
        if self.ent is None or l1 == 0:
            return torch.zeros((), dtype=torch.float64)
        return l1 * torch.abs(torch.sin(self.ent)).sum()


class PennyLaneInteractionBackend:
    """Implements ``qss.quantum.interfaces.InteractionBackend`` for trained weights.

    ``features`` is (n_items, 2) = [theta, phi] per item, or a batch (B, n_items, 2).
    Supports finite shots and any PennyLane device (e.g. ``lightning.qubit``, ``qiskit.aer``).
    """

    def __init__(self, spec: CircuitSpec, weights: np.ndarray, device: str = "default.qubit",
                 shots: int | None = None, ent: np.ndarray | None = None) -> None:
        self.spec = spec
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.float64)
        self.ent = None if ent is None else torch.as_tensor(np.asarray(ent), dtype=torch.float64)
        self.shots = shots
        self.qnode = make_qnode(spec, device, shots)
        self.log: list[dict] = []

    def run_interaction(self, features: np.ndarray) -> np.ndarray:
        feats = np.asarray(features, dtype=float)
        single = feats.ndim == 2
        if single:
            feats = feats[None]
        theta = torch.as_tensor(feats[..., 0])
        phi = torch.as_tensor(feats[..., 1])
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.qnode(theta, phi, self.weights, self.ent)
        q = torch.stack([torch.as_tensor(o) for o in out], dim=-1).reshape(len(feats), -1).numpy()
        self.log.append({"batch": len(feats), "shots": self.shots,
                         "seconds": time.perf_counter() - t0, **resources(self.spec)})
        return q[0] if single else q


def resources(spec: CircuitSpec) -> dict:
    """Gate counts, depth and measurement settings for one execution of the circuit."""
    theta = np.zeros(spec.n_items)
    tape = qml.workflow.construct_tape(make_qnode(spec))(
        torch.as_tensor(theta), torch.as_tensor(theta), torch.zeros(spec.weight_shape, dtype=torch.float64),
        torch.zeros(spec.ent_shape, dtype=torch.float64),
    )
    level = [0] * spec.n_wires
    n_two = 0
    for op in tape.operations:
        wires = list(op.wires)
        layer = max(level[w] for w in wires) + 1
        for w in wires:
            level[w] = layer
        n_two += len(wires) == 2
    obs = spec.observables()
    groups = qml.pauli.group_observables(obs, grouping_type="qwc")
    return {
        "n_qubits": spec.n_wires,
        "depth": max(level),
        "n_gates": len(tape.operations),
        "n_two_qubit_gates": n_two,
        "n_trainable_params": int(np.prod(spec.weight_shape)) + int(np.prod(spec.ent_shape)),
        "n_observables": len(obs),
        "measurement_settings": len(groups),
    }
