# Module A — key findings (hackathon MVP)

All numbers are test MSE (mean over 3 seeds unless noted). They come from
[results_module_a.md](results_module_a.md), which `make_plots.py` regenerates from the run
records. Simulator: PennyLane `default.qubit`, exact expectations unless shots are stated.
Features come from the shared `StructuredSequenceEncoder` (`qss.classical.encoder`), fitted on
the training split only.

## What is built (M1)

- An end-to-end pipeline: noisy series → shared `StructuredSequenceEncoder` (amplitude/phase per
  patch) → quantum interaction backend → prediction.
- `PennyLaneInteractionBackend` implements the shared `InteractionBackend` protocol. It supports
  finite shots and any PennyLane device.
- Every run logs qubits, circuit depth, two-qubit gates, measurement settings, circuit executions
  and wall-clock time.
- Every run also records the shared-encoder configuration (patch lengths, period, stride), the
  normalization mean and std fitted on the training split, the data settings and the split
  boundaries, as `docs/benchmark_protocol.md` asks. Older S1/S2 records got this field through
  `backfill_encoder_record.py`, which recomputes it deterministically.

## S1: amplitude–phase interaction, `y = A1·A2·cos(φ1 − φ2)`

Each trained quantum model keeps the best of 5 random restarts, selected on validation MSE.

- **The hybrid encoding RY(arcsin A)·RZ(φ) has the right inductive bias.**
  - With 2 qubits and one re-uploading layer, it reaches the explicit-interaction oracle at every
    training size: 0.0006–0.0007, including n=64.
  - This holds for both readouts: Z readout with one CZ, and Pauli-2 readout with no CZ.
  - Generic MLPs need about 4096 samples to get there. At n=64 they score 0.016 (raw) and
    0.015 (Fourier), about 25× worse.
- **The gain comes from the feature map, not from training or from anything uniquely quantum.**
  - An untrained random hybrid circuit with a ridge readout is just as good.
  - The reason is analytic: ⟨X1X2⟩ + ⟨Y1Y2⟩ = sinθ1·sinθ2·cos(φ1 − φ2) exactly (unit test
    `test_hybrid_encoding_gives_target_analytically`).
  - A classical model given the same features ([A·cos φ, A·sin φ] and their products) matches it.
- **Angle encoding works but costs more.** It needs 4 qubits and 12 CZ gates (D=2), and is
  slower to learn: 0.0027 at n=64 and 0.0006–0.0011 at n≥256.
- **More re-uploading is not better here.**
  - With hybrid encoding, D=2 gives 0.020–0.033, against 0.0006 for D=1, even with restarts.
  - A two-layer circuit does not contain the one-layer function. The second data upload rotates
    a state that is already data-dependent.
- **Restarts remove the seed sensitivity.**
  - Without restarts, the Z-readout + CZ models sometimes got stuck (0.015–0.03 on some seeds).
  - With 5 restarts, the standard deviation across seeds is at most 0.0001 for every hybrid
    D=1 configuration.
- **The phase-only encoding row is not an encoding comparison.** Its head is given classical
  A_i·A_j products, so it is labelled separately.
- **Gated fusion with an MLP backbone helps the MLP only slightly.** At n=64 it scores 0.014
  against 0.015–0.016 for the MLPs. It stays far from the pure hybrid circuit (0.0006).

## S2: cross-period interaction, `y = Σ A_k·A_{k+s}·cos(φ_k − φ_{k+s})`

Setup: one qubit per period patch, readout restricted to single-qubit ⟨Z⟩, and the same number
of trainable parameters for every fixed topology. Each quantum model keeps the best of 5 random
restarts, selected on validation MSE.

- **The cross-period topology learns the task on every seed.**
  - 4 qubits, D=1: 0.0017–0.0018 on all 3 seeds, equal to the explicit oracle (0.0018).
  - 6 qubits, D=1: 0.0024, equal to the oracle (0.0025) and 12–19× better than the MLPs
    (0.029 raw, 0.046 Fourier).
  - Restarts fixed the seed that gets stuck without them (0.090 → 0.0018).
- **Structure matters, not gate count.**
  - The mismatched control has the same number of CZ gates as cross-period, but on the wrong
    pairs (0,1), (2,3), and so on.
  - It scores the same as no entanglement: 0.17 at 4 qubits and 0.26 at 6 qubits.
  - Local-chain and all-to-all topologies also fail at both depths.
- **Entanglement and measurements can substitute for each other.** Without any CZ, a full
  two-body Pauli readout also solves S2 (0.0018 at 4 qubits, 0.0027 at 6 qubits). The price is
  21 or 33 measurement settings per prediction instead of 1.

## Learned topology: the circuit finds the cross-period edges by itself

Setup: every qubit pair gets a trainable RZZ(γ). Its entangling strength is |sin γ|. All
strengths start small (|γ| ≤ 0.3), and the circuit is never told which pairs matter.

- **Without any sparsity penalty (L1 = 0), every seed switches on exactly the target pairs.**
  - 4 qubits: edges 0–2 and 1–3 go to strength 1.00. Every other edge stays at 0.04 or below.
    Test MSE is 0.0017–0.0018 on all 3 seeds, equal to the hand-designed cross-period circuit.
  - 6 qubits: out of 15 candidate edges, 0–3, 1–4 and 2–5 go to 1.00. All others stay at
    0.03 or below, and 94–96% of the total strength sits on the target pairs.
  - At 6 qubits, all 3 seeds reach the oracle (0.0024–0.0026), the same as the hand-designed
    cross-period circuit (0.0024).
- **An L1 penalty of 0.003 does not help.**
  - 6 qubits: 2 of 3 seeds keep exactly the 3 target edges and reach the oracle (0.0023–0.0026).
    The third drops one needed edge (0.087).
  - 4 qubits: every seed keeps the right edges, but 2 of 3 do not converge (0.044–0.048).
  - Use L1 = 0.
- **What this means:** the cross-period structure is *discovered from data*, not imposed. This
  answers the main objection to the S2 result.

## S3: structured classification (periodic pattern labels)

This is the "structured classification" row of `docs/experiment_plan.md`.

- **Setup.** The series are the S2 series with 4 period patches. The label is 1 if the noisy S2
  score is positive, meaning patches one super-period apart are, on balance, in phase. The two
  classes are balanced, so chance is 0.5.
- **Training and metrics.** Models are trained with BCE-with-logits and evaluated by test
  accuracy and macro-F1 (3 seeds).
- **Readout.** The quantum models use a ⟨Z⟩ readout only.

| model | acc n=256 | acc n=2048 |
|---|---|---|
| circuit, cross-period (2 CZ) | 0.961 | 0.974 |
| circuit, learned RZZ | 0.952 | 0.971 |
| explicit interaction (logistic) | 0.927 | 0.970 |
| MLP on Fourier features | 0.784 | 0.920 |
| MLP on raw (A, φ) | 0.720 | 0.906 |
| circuit, local chain / all-to-all | 0.74 | 0.74 |
| circuit, mismatched (2 CZ) / none | ≈ 0.50 | ≈ 0.50 |

Macro-F1 equals accuracy to within 0.001 for every model above chance.

- **The S2 conclusion carries over to classification.** The structure-matched circuit is best, and
  it is already near its ceiling at 256 samples.
- **The circuit beats the generic MLPs, and beats the explicit logistic oracle at small n.** At
  2048 samples the oracle catches up. The gap at small n most likely reflects the oracle's larger
  number of features (44), not any advantage of the circuit.
- **Wrong structure gives chance.** The mismatched control has the same CZ count as cross-period
  but sits at chance. Local and all-to-all partly mix the right pairs and plateau at 0.74.
- **The learned topology again picks (0,2) and (1,3)** in all 6 runs, with 91–98% of the
  entangling strength.

## Finite shots (S1, trained at n=4096)

| model | exact | 10 000 shots | 1000 shots | 100 shots |
|---|---|---|---|---|
| hybrid, D=1, Z readout + 1 CZ (2 qubits) | 0.0006 | 0.0006 | 0.0015 | 0.0098 |
| hybrid, D=1, Pauli-2, no CZ (9 settings) | 0.0006 | 0.0008 | 0.0032 | 0.028 |
| angle, D=2, Z readout + 12 CZ (4 qubits) | 0.0008 | 0.0011 | 0.0039 | 0.030 |

- The single-setting Z-readout circuit is the most shot-efficient.
- The Pauli-2 readout runs each of its 9 settings with the full shot count, so it uses 9× the
  shots. It is still about 2× noisier at 1000 shots, because the head sums 15 noisy estimates.

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
