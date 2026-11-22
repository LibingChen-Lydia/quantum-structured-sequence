import numpy as np
import torch

from qss.classical.baselines import ExplicitInteraction, ExplicitInteractionClassifier
from qss.classical.patch_features import amp_to_theta, encode_items
from qss.classical.synthetic import make_s1, make_s2, make_s3
from qss.evaluation.metrics import classification_metrics
from qss.quantum.circuits import CircuitSpec, topology_edges
from qss.quantum.interaction import PennyLaneInteractionBackend, QuantumInteractionLayer, resources
from qss.training import train


def test_shared_encoder_recovers_amp_phase_without_noise() -> None:
    """Module A reads amplitude/phase from the shared StructuredSequenceEncoder."""
    for data in (make_s1(64, 0, noise_x=0.0), make_s2(64, 0, noise_x=0.0)):
        amp, phase, encoder = encode_items(data.series, data.series, data.period, data.patch_len)
        # the encoder normalizes by the training input std; undo that per channel
        std = encoder.transform(data.series[:1]).input_std
        channels = data.series.shape[2]
        scale = np.tile(std, amp.shape[1] // channels)
        assert np.allclose(amp * scale, data.amp, atol=1e-3)
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


def test_learned_topology_trains_edge_angles_and_backend_matches() -> None:
    spec = CircuitSpec(4, "hybrid", "learned", depth=1, readout="z")
    layer = QuantumInteractionLayer(spec, seed=1)
    assert layer.ent.shape == (1, 6)
    x = torch.rand(3, 4, dtype=torch.float64)
    q = layer(x, x)
    (q.sum() + layer.regularization(0.1)).backward()
    assert layer.ent.grad is not None and torch.any(layer.ent.grad != 0)
    backend = PennyLaneInteractionBackend(spec, layer.weights.detach().numpy(),
                                          ent=layer.ent.detach().numpy())
    out = backend.run_interaction(np.stack([x.numpy(), x.numpy()], -1))
    assert np.allclose(out, q.detach().numpy(), atol=1e-10)
    assert resources(spec)["n_trainable_params"] == 4 * 3 * 2 + 6


def test_s3_labels_are_sign_of_s2_and_balanced() -> None:
    d2, d3 = make_s2(4000, seed=3), make_s3(4000, seed=3)
    np.testing.assert_array_equal(d3.series, d2.series)
    np.testing.assert_array_equal(d3.y, (d2.y > 0).astype(float))
    assert 0.45 < d3.y.mean() < 0.55


def test_explicit_classifier_solves_s3_noise_free() -> None:
    d = make_s3(3000, seed=0, noise_x=0.0, noise_y=0.0)
    est = ExplicitInteractionClassifier().fit(d.amp[:2000], d.phase[:2000], d.y[:2000])
    m = classification_metrics(d.y[2000:], est.predict(d.amp[2000:], d.phase[2000:]))
    assert m["accuracy"] > 0.95 and m["macro_f1"] > 0.95


def test_classification_metrics_and_bce_training() -> None:
    m = classification_metrics(np.array([0, 1, 1, 0]), np.array([-2.0, 3.0, -1.0, -0.5]))
    assert m["accuracy"] == 0.75 and 0 < m["bce"] < 1
    d = make_s3(600, seed=1)
    layer = QuantumInteractionLayer(CircuitSpec(4, "hybrid", "cross_period", 1, "z"), seed=0)
    from qss.fusion.gated import QuantumRegressor
    model = QuantumRegressor(layer, amp_max=1.05)
    tr = (d.amp[:400], d.phase[:400], d.y[:400])
    va = (d.amp[400:], d.phase[400:], d.y[400:])
    info = train(model, tr, va, seed=0, lr=0.05, epochs=3, loss="bce")
    assert info["loss"] == "bce" and np.isfinite(info["best_val_loss"])
