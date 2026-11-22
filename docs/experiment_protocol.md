# A+C experiment protocol

## Module A: structured periodic interaction

**Initial configuration:** four qubits, two data re-uploading layers, exact simulator expectations.

Compare `none`, `local`, `cross_period`, and `all_to_all` entanglement topologies, keeping parameter count and input features as comparable as practical. Run angle, phase, and hybrid amplitude–phase encodings. Report performance alongside circuit depth, number of circuit executions, shots, and latency.

## Module C: quantum-assisted retrieval

Use a task-aware continuous memory index to retrieve a fixed-size candidate set
from historical windows. Apply the shared preprocessing and temporal cutoff to
every re-ranker. Compare cosine, RBF, learned metric, and quantum fidelity
scores. Report Recall@K, MRR, retrieval and re-ranking latency, quantum calls,
and downstream forecasting error when context is fused. The shared data
contract is specified in [benchmark_protocol.md](benchmark_protocol.md).

## Reproducibility

Each experiment configuration must capture dataset/version, split, random seed, preprocessing, model parameters, backend/device, shots, and package versions. Run at least three seeds for summary results.

