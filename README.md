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

## Shared structured encoder

`StructuredSequenceEncoder` is the shared upstream contract for both modules. It accepts
`[batch, length, channels]` time series and returns scale-indexed Patch features, normalized
retrieval embeddings, periodic amplitude/phase/frequency tensors, and Patch start indices.

```python
from qss.classical import StructuredEncoderConfig, StructuredSequenceEncoder

encoder = StructuredSequenceEncoder(
    StructuredEncoderConfig(patch_lengths=(24, 48, 96), period=24)
)
encoded = encoder.fit_transform(train_series)
module_a_input = encoded.scales[48].amplitude, encoded.scales[48].phase
module_c_input = encoded.scales[48].embedding
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

## Status

Scaffolded for the Module A + C hackathon MVP. The next implementation milestones are:

1. Patch feature extractor and classical baseline.
2. Four-qubit, two-layer data re-uploading circuit with topology ablations.
3. Hierarchical memory bank and quantum-fidelity re-ranker.
4. Synthetic experiments, reproducible configuration files, and results table.

