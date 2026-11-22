import numpy as np
import torch

from qss.classical.baselines import ExplicitInteraction
from qss.classical.patch_features import amp_phase_features, amp_to_theta
from qss.classical.synthetic import make_s1, make_s2
from qss.quantum.circuits import CircuitSpec, topology_edges
from qss.quantum.interaction import PennyLaneInteractionBackend, QuantumInteractionLayer, resources


def test_feature_extractor_recovers_amp_phase_without_noise() -> None:
    for data in (make_s1(64, 0, noise_x=0.0), make_s2(64, 0, noise_x=0.0)):
        amp, phase = amp_phase_features(data.series, data.period, data.patch_len)
        assert np.allclose(amp, data.amp, atol=1e-3)
        assert np.allclose(np.angle(np.exp(1j * (phase - data.phase))), 0, atol=1e-3)


def test_s2_target_couples_cross_period_pairs() -> None:
    data = make_s2(8, 1, n_patches=6)
    assert data.pairs == ((0, 3), (1, 4), (2, 5))


def test_topology_edges() -> None:
    assert topology_edges("none", 4) == []
    assert topology_edges("local", 4) == [(0, 1), (1, 2), (2, 3)]
    assert topology_edges("cross_period", 4) == [(0, 2), (1, 3)]
    assert len(topology_edges("all_to_all", 6)) == 15
    assert topology_edges("mismatched", 4) == [(0, 1), (2, 3)]


def test_hybrid_encoding_gives_target_analytically() -> None:
    """No entanglement, no training: <X0X1> + <Y0Y1> = sin t0 sin t1 cos(phi0 - phi1)."""
    spec = CircuitSpec(2, "hybrid", "none", depth=1, readout="pauli2")
    weights = np.zeros(spec.weight_shape)  # Rot(0, 0, 0) = identity
    backend = PennyLaneInteractionBackend(spec, weights)
    rng = np.random.default_rng(0)
    amp = rng.uniform(0.2, 1.0, size=(16, 2))
    phase = rng.uniform(-np.pi, np.pi, size=(16, 2))
    theta = amp_to_theta(amp, 1.0)
    q = backend.run_interaction(np.stack([theta, phase], -1))
    xx, yy = q[:, 6], q[:, 10]  # pair block starts after 6 single-qubit observables
    target = amp[:, 0] * amp[:, 1] * np.cos(phase[:, 0] - phase[:, 1])
    assert np.allclose(xx + yy, target, atol=1e-8)
    assert backend.log[-1]["n_two_qubit_gates"] == 0


def test_layer_shapes_and_gradients() -> None:
    for enc in ("angle", "phase", "hybrid"):
        spec = CircuitSpec(2, enc, "all_to_all", depth=2, readout="zz")
        layer = QuantumInteractionLayer(spec)
        theta = torch.rand(5, 2, dtype=torch.float64)
        q = layer(theta, theta)
        assert q.shape == (5, layer.n_out)
        q.sum().backward()
        assert layer.weights.grad is not None


def test_parameter_count_independent_of_topology() -> None:
    counts = {t: resources(CircuitSpec(4, "hybrid", t, 2, "z"))["n_trainable_params"]
              for t in ("none", "local", "cross_period", "all_to_all")}
    assert len(set(counts.values())) == 1
    assert resources(CircuitSpec(4, "hybrid", "cross_period", 2, "z"))["n_two_qubit_gates"] == 4


def test_explicit_interaction_solves_s1_noise_free() -> None:
    data = make_s1(400, 0, noise_x=0.0, noise_y=0.0)
    est = ExplicitInteraction().fit(data.amp, data.phase, data.y)
    assert np.mean((est.predict(data.amp, data.phase) - data.y) ** 2) < 1e-6


def test_layer_is_seed_deterministic() -> None:
    spec = CircuitSpec(4, "hybrid", "cross_period", 2, "z")
    a, b = QuantumInteractionLayer(spec, seed=3), QuantumInteractionLayer(spec, seed=3)
    x = torch.rand(4, 4, dtype=torch.float64)
    assert torch.allclose(a(x, x), b(x, x))
