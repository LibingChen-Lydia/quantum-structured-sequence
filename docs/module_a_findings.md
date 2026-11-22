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

Each trained quantum model keeps the best of 5 random restarts, selected on validation MSE.

- **The hybrid encoding RY(arcsin A)·RZ(φ) has the right inductive bias.**
  - With 2 qubits and one re-uploading layer, it reaches the explicit-interaction oracle at every
    training size: 0.0006–0.0007, including n=64.
  - This holds for both readouts: Z readout with one CZ, and Pauli-2 readout with no CZ.
  - Generic MLPs need about 4096 samples to get there. At n=64 they score 0.015 (raw) and
    0.022 (Fourier), 20–30× worse.
- **The gain comes from the feature map, not from training or from anything uniquely quantum.**
  - An untrained random hybrid circuit with a ridge readout is just as good.
  - The reason is analytic: ⟨X1X2⟩ + ⟨Y1Y2⟩ = sinθ1·sinθ2·cos(φ1 − φ2) exactly (unit test
    `test_hybrid_encoding_gives_target_analytically`).
  - A classical model given the same features ([A·cos φ, A·sin φ] and their products) matches it.
- **Angle encoding works but costs more.** It needs 4 qubits and 12 CZ gates (D=2), and is
  slower to learn: 0.0019 at n=64 and 0.0006 only at n=4096.
- **More re-uploading is not better here.**
  - With hybrid encoding, D=2 gives 0.017–0.030, against 0.0006 for D=1, even with restarts.
  - A two-layer circuit does not contain the one-layer function. The second data upload rotates
    a state that is already data-dependent.
- **Restarts remove the seed sensitivity.**
  - Without restarts, the Z-readout + CZ models sometimes got stuck (0.015–0.03 on some seeds).
  - With 5 restarts, the standard deviation across seeds is at most 0.0001 for every good
    configuration.
- **The phase-only encoding row is not an encoding comparison.** Its head is given classical
  A_i·A_j products, so it is labelled separately.
- **Gated fusion with an MLP backbone helps the MLP only slightly.** At n=64 it scores 0.015
  against 0.022 for the Fourier MLP. It stays far from the pure hybrid circuit (0.0006).

## S2: cross-period interaction, `y = Σ A_k·A_{k+s}·cos(φ_k − φ_{k+s})`

Setup: one qubit per period patch, readout restricted to single-qubit ⟨Z⟩, and the same number
of trainable parameters for every fixed topology. Each quantum model keeps the best of 5 random
restarts, selected on validation MSE.

- **The cross-period topology learns the task on every seed.**
  - 4 qubits, D=1: 0.0017–0.0018 on all 3 seeds, equal to the explicit oracle (0.0018).
  - 6 qubits, D=1: 0.0024, equal to the oracle (0.0025) and 10–18× better than the MLPs
    (0.026 raw, 0.043 Fourier).
  - Restarts fixed the seed that got stuck in the first run (0.048 → 0.0018).
- **Structure matters, not gate count.**
  - The mismatched control has the same number of CZ gates as cross-period, but on the wrong
    pairs (0,1), (2,3), and so on.
  - It scores the same as no entanglement: 0.17 at 4 qubits and 0.26 at 6 qubits.
  - Local-chain and all-to-all topologies also fail at both depths.
- **Entanglement and measurements can substitute for each other.** Without any CZ, a full
  two-body Pauli readout also solves S2 (0.0018 at 4 qubits, 0.0026 at 6 qubits). The price is
  21 or 33 measurement settings per prediction instead of 1.

## Learned topology: the circuit finds the cross-period edges by itself

Setup: every qubit pair gets a trainable RZZ(γ). Its entangling strength is |sin γ|. All
strengths start small (|γ| ≤ 0.3), and the circuit is never told which pairs matter.

- **Without any sparsity penalty (L1 = 0), every seed switches on exactly the target pairs.**
  - 4 qubits: edges 0–2 and 1–3 go to strength 1.00. Every other edge stays at 0.03 or below.
    Test MSE is 0.0017–0.0018 on all 3 seeds, equal to the hand-designed cross-period circuit.
  - 6 qubits: out of 15 candidate edges, 0–3, 1–4 and 2–5 go to 1.00. All others stay at
    0.12 or below, and 87–93% of the total strength sits on the target pairs.
  - At 6 qubits, 2 of 3 seeds reach the oracle (0.0024–0.0026). The third picks the right edges
    but its other parameters do not converge (0.047).
- **An L1 penalty of 0.003 is too strong at 6 qubits.** It switches off one or two of the needed
  edges (0.087–0.18). At 4 qubits it is harmless. Use L1 = 0.
- **What this means:** the cross-period structure is *discovered from data*, not imposed. This
  answers the main objection to the S2 result.

## Finite shots (S1, trained at n=4096)

| model | exact | 10 000 shots | 1000 shots | 100 shots |
|---|---|---|---|---|
| hybrid, D=1, Z readout + 1 CZ (2 qubits) | 0.0006 | 0.0007 | 0.0016 | 0.012 |
| hybrid, D=1, Pauli-2, no CZ (9 settings) | 0.0006 | 0.0009 | 0.0045 | 0.039 |
| angle, D=2, Z readout + 12 CZ (4 qubits) | 0.0006 | 0.0011 | 0.0046 | 0.042 |

- The single-setting Z-readout circuit is the most shot-efficient.
- The Pauli-2 readout runs each of its 9 settings with the full shot count, so it uses 9× the
  shots. It is still about 3× noisier at 1000 shots, because the head sums 15 noisy estimates.

## Honest limits

- The circuits are small (2–6 qubits) and fully classically simulable, so no computational
  advantage is claimed. The claim is about the inductive bias of the encoding and topology,
  measured against strong classical controls.
- The hand-designed cross-period edges coincide with the target pairs by construction. The
  learned-topology experiment shows the circuit finds the same edges without being told, but
  only on synthetic data where the true structure is known.
- The explicit-interaction ridge model always matches the best quantum model. Only generic
  MLPs are beaten.
- Real-data benchmarks (ETTh1) are not run yet.
