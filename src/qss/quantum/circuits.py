"""Module A circuits: encodings, structured entanglement topologies, re-uploading, readouts.

Layer d (d = 0..D-1):  E(z)  ->  Rot(w_d) on every wire  ->  CZ on topology edges
Final:                 Rot(w_D) on every wire  ->  measure readout observables

The final rotation matters: CZ commutes with Z, so without it a Z readout would not see
any entanglement. All topologies use fixed CZ gates, so the trainable parameter count
(3 * n_wires * (D + 1)) is identical across topologies.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import pennylane as qml

ENCODINGS = ("angle", "phase", "hybrid")
TOPOLOGIES = ("none", "local", "cross_period", "all_to_all", "mismatched")
READOUTS = ("z", "zz", "pauli2")


def n_wires_for(encoding: str, n_items: int) -> int:
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}")
    return 2 * n_items if encoding == "angle" else n_items


def topology_edges(topology: str, n_wires: int) -> list[tuple[int, int]]:
    """Q0 none, Q1 local chain, Q2 cross-period (k <-> k + n/2), Q3 all-to-all, + mismatched control."""
    if topology == "none":
        return []
    if topology == "local":
        return [(k, k + 1) for k in range(n_wires - 1)]
    if topology == "cross_period":
        s = n_wires // 2
        return [(k, k + s) for k in range(s)]
    if topology == "mismatched":
        # control: same CZ count as cross_period but on the wrong pairing (2k, 2k+1)
        return [(2 * k, 2 * k + 1) for k in range(n_wires // 2)]
    if topology == "all_to_all":
        return list(itertools.combinations(range(n_wires), 2))
    raise ValueError(f"unknown topology {topology!r}")


def readout_observables(readout: str, n_wires: int) -> list:
    wires = range(n_wires)
    if readout == "z":
        return [qml.PauliZ(i) for i in wires]
    if readout == "zz":
        return [qml.PauliZ(i) for i in wires] + [
            qml.PauliZ(i) @ qml.PauliZ(j) for i, j in itertools.combinations(wires, 2)
        ]
    if readout == "pauli2":
        paulis = (qml.PauliX, qml.PauliY, qml.PauliZ)
        single = [p(i) for i in wires for p in paulis]
        pair = [
            p(i) @ q(j)
            for i, j in itertools.combinations(wires, 2)
            for p in paulis
            for q in paulis
        ]
        return single + pair
    raise ValueError(f"unknown readout {readout!r}")


@dataclass(frozen=True)
class CircuitSpec:
    n_items: int
    encoding: str = "hybrid"
    topology: str = "cross_period"
    depth: int = 2
    readout: str = "z"

    @property
    def n_wires(self) -> int:
        return n_wires_for(self.encoding, self.n_items)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return topology_edges(self.topology, self.n_wires)

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.depth + 1, self.n_wires, 3)

    def observables(self) -> list:
        return readout_observables(self.readout, self.n_wires)


def encode(spec: CircuitSpec, theta, phi) -> None:
    """theta, phi: (..., n_items). theta = arcsin(A / A_max), phi = phase."""
    if spec.encoding == "hybrid":
        for k in range(spec.n_items):
            qml.RY(theta[..., k], wires=k)
            qml.RZ(phi[..., k], wires=k)
    elif spec.encoding == "phase":
        for k in range(spec.n_items):
            qml.Hadamard(wires=k)
            qml.RZ(phi[..., k], wires=k)
    else:  # angle: one scalar per wire, RY only
        for k in range(spec.n_items):
            qml.RY(2 * theta[..., k], wires=2 * k)
            qml.RY(phi[..., k], wires=2 * k + 1)


def circuit(spec: CircuitSpec, theta, phi, weights):
    for d in range(spec.depth):
        encode(spec, theta, phi)
        for q in range(spec.n_wires):
            qml.Rot(weights[d, q, 0], weights[d, q, 1], weights[d, q, 2], wires=q)
        for i, j in spec.edges:
            qml.CZ(wires=[i, j])
    for q in range(spec.n_wires):
        qml.Rot(weights[spec.depth, q, 0], weights[spec.depth, q, 1], weights[spec.depth, q, 2], wires=q)
    return [qml.expval(o) for o in spec.observables()]
