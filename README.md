# Quantum-Structured Sequence

Hybrid quantum–classical sequence modelling for the Infleqtion Hackathon, built around two independently testable modules:

- **Module A — Structured Quantum Periodic Interaction:** encode amplitude, phase, frequency, and trend features from time-series patches; compare structured entanglement topologies.
- **Module C — Quantum-Assisted Hierarchical Retrieval:** retrieve candidates classically from a history bank, then re-rank them with a quantum fidelity kernel.

Together, the modules provide clear baselines, controlled ablations, and an end-to-end demonstrator.

## Architecture

```text
sequence → patch/period feature extractor ─┬→ Module A quantum interaction → fusion → prediction
                                           └→ classical candidate recall → Module C quantum re-rank → retrieved context
```

## Repository layout

```text
configs/                 Experiment configurations
data/                    Local datasets (ignored by Git)
docs/                    Design notes and experiment protocol
experiments/
  synthetic_periodic/    Module A controlled experiments
  retrieval/             Module C controlled experiments
src/qss/
  classical/             Patch features and classical baselines
  quantum/               Circuit definitions and backend interface
  retrieval/             Memory bank, candidate recall, kernel re-ranking
  fusion/                Prediction/context fusion
  evaluation/            Metrics, resource and latency logging
tests/                   Unit tests
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Experiment plan

| Module | Primary synthetic task | Classical controls | Main metrics |
|---|---|---|---|
| A | Amplitude–phase and cross-period interactions | MLP, Fourier features, explicit interactions | MSE/MAE, accuracy, circuit resources |
| C | Delayed historical-patch retrieval | cosine, RBF, learned metric | Recall@K, MRR, re-ranking latency |

Every run must record seed, backend, qubit count, circuit depth, shots, quantum calls, and wall-clock time. See [docs/experiment_protocol.md](docs/experiment_protocol.md).

## Module A — how to run

```bash
pytest                                                                     # incl. analytic encoding check
python experiments/synthetic_periodic/run_module_a.py --config configs/module_a_synthetic.yaml --quick  # 1-min smoke test
python experiments/synthetic_periodic/run_module_a.py --config configs/module_a_s1.yaml     # S1 encoding comparison
python experiments/synthetic_periodic/run_module_a.py --config configs/module_a_s2_4q.yaml  # S2 topology, 4 qubits
python experiments/synthetic_periodic/run_module_a.py --config configs/module_a_s2_6q.yaml  # S2 topology, 6 qubits
python experiments/synthetic_periodic/make_plots.py                                         # docs/results_module_a.md
```

Everything runs on CPU (PennyLane `default.qubit`, backprop through the exact state vector); no GPU
is needed at 2–6 qubits. Runs are resumable: finished `(model, n_train, seed)` records are skipped.

| piece | file |
|---|---|
| synthetic S1/S2 generators | `src/qss/classical/synthetic.py` |
| DFT amplitude/phase patch features | `src/qss/classical/patch_features.py` |
| classical controls (raw MLP, Fourier MLP, param-matched MLP, explicit interaction) | `src/qss/classical/baselines.py` |
| encodings (angle / phase / hybrid), topologies (none / local / cross-period / all-to-all), re-uploading, readouts | `src/qss/quantum/circuits.py` |
| trainable layer, `InteractionBackend` implementation (shots, any PennyLane device), resource counter | `src/qss/quantum/interaction.py` |
| gated fusion `h + g ⊙ W_q q` | `src/qss/fusion/gated.py` |

Results: [docs/module_a_findings.md](docs/module_a_findings.md) (summary) and [docs/results_module_a.md](docs/results_module_a.md) (all numbers).

## Status

Scaffolded for the Module A + C hackathon MVP. The next implementation milestones are:

1. Patch feature extractor and classical baseline.
2. Four-qubit, two-layer data re-uploading circuit with topology ablations.
3. Hierarchical memory bank and quantum-fidelity re-ranker.
4. Synthetic experiments, reproducible configuration files, and results table.

