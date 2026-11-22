# Module A — key findings (hackathon MVP)

All numbers are test MSE (mean over 3 seeds unless noted). They come from
[results_module_a.md](results_module_a.md), which `make_plots.py` regenerates from the run
records. Simulator: PennyLane `default.qubit`, exact expectations unless shots are stated.

## What is built (M1)

- An end-to-end pipeline: noisy series → DFT amplitude/phase patch features → quantum
  interaction backend → prediction.
- `PennyLaneInteractionBackend` implements the shared `InteractionBackend` protocol. It supports
  finite shots and any PennyLane device.
- Every run logs qubits, circuit depth, two-qubit gates, measurement settings, circuit executions
  and wall-clock time.

## S1: amplitude–phase interaction, `y = A1·A2·cos(φ1 − φ2)`

- **The hybrid encoding RY(arcsin A)·RZ(φ) has the right inductive bias.**
  - 2 qubits, one re-uploading layer and a Pauli-2 readout reach the explicit-interaction oracle
    at every training size: 0.0008 at n=64 and 0.0006 at n≥256.
  - Generic MLPs need about 4096 samples to get there. At n=64 they score 0.015 (raw) and
    0.022 (Fourier).
- **The gain comes from the feature map, not from training or from anything uniquely quantum.**
  - An untrained random hybrid circuit with a ridge readout is just as good.
  - The reason is analytic: ⟨X1X2⟩ + ⟨Y1Y2⟩ = sinθ1·sinθ2·cos(φ1 − φ2) exactly (unit test
    `test_hybrid_encoding_gives_target_analytically`).
  - A classical model given the same features ([A·cos φ, A·sin φ] and their products) matches it.
- **Angle encoding works but costs more.** It needs 4 qubits and 6–12 CZ gates, and reaches
  0.0008–0.0012 at n≥256.
- **More re-uploading is not better here.**
  - With hybrid encoding, D=2 gives 0.019–0.024, against 0.0006 for D=1.
  - A two-layer circuit does not contain the one-layer function. The second data upload rotates
    a state that is already data-dependent.
- **Z-readout + CZ variants are seed-sensitive.** Some seeds get stuck in local minima, so the
  right-hand panel of the figure shows medians.
- **The phase-only encoding row is not an encoding comparison.** Its head is given classical
  A_i·A_j products, so it is labelled separately.
- **Gated fusion with an MLP backbone adds nothing measurable on S1.** It scores 0.0008 at
  n=4096, the same as the MLPs.

## S2: cross-period interaction, `y = Σ A_k·A_{k+s}·cos(φ_k − φ_{k+s})`

Setup: one qubit per period patch, readout restricted to single-qubit ⟨Z⟩, and the same number
of trainable parameters for every topology.

- **The cross-period topology is the only entangled circuit that learns the task.**
  - 6 qubits, D=1: 0.0024, equal to the explicit oracle (0.0025) and 10–18× better than the MLPs
    (0.026 raw, 0.043 Fourier).
  - 4 qubits, D=1: two of three seeds reach 0.0018 (oracle 0.0018). One seed stopped early at
    0.048.
- **Structure matters, not gate count.**
  - The mismatched control has the same number of CZ gates as cross-period, but on the wrong
    pairs (0,1), (2,3), and so on.
  - It scores the same as no entanglement: 0.17 at 4 qubits and 0.26 at 6 qubits.
  - Local-chain and all-to-all topologies also fail at both depths.
- **Entanglement and measurements can substitute for each other.** Without any CZ, a full
  two-body Pauli readout also solves S2 (0.0018 at 4 qubits, 0.0026 at 6 qubits). The price is
  21 or 33 measurement settings per prediction instead of 1.

## Finite shots (S1, trained at n=4096)

| model | exact | 10 000 shots | 1000 shots | 100 shots |
|---|---|---|---|---|
| angle, Z readout + CZ | 0.0008 | 0.0011 | 0.0033 | 0.025 |

Results for the other models are in the results file.

## Honest limits

- The circuits are small (2–6 qubits) and fully classically simulable, so no computational
  advantage is claimed. The claim is about the inductive bias of the encoding and topology,
  measured against strong classical controls.
- The cross-period edges are chosen from the known period structure, which is the point of
  hypothesis H-A2. Because they coincide with the target pairs by construction, the S2 result
  shows that a structure-matched topology helps. It does not show that the circuit discovers the
  structure on its own.
- The explicit-interaction ridge model always matches the best quantum model. Only generic
  MLPs are beaten.
- Real-data benchmarks (ETTh1) are not run yet.
