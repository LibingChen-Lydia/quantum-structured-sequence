# A+C experiment plan

## Goal

This plan coordinates the experiments for the two project modules:

- **Module A:** structured quantum periodic interaction.
- **Module C:** quantum-assisted historical retrieval for forecasting.

Both modules use the shared sequence-data contract in
[benchmark_protocol.md](benchmark_protocol.md). That contract supplies the
sequence tensor, temporal splits, normalization, timestamps, and Patch
construction used by the common structured encoder.

## Shared experiment settings

| Setting | Value |
|---|---|
| Input length | 96 |
| Forecast horizons | 96, 192, 336, 720 |
| Real-world datasets | ETTh1, Weather |
| Feature setting | Multivariate (`M`) |
| Random seeds | 2021, 2022, 2023 |
| Shared encoder | `StructuredSequenceEncoder` with Patch lengths 24, 48, 96 |
| Result summary | Mean and standard deviation over three seeds |

## Module A experiments

Module A evaluates whether structured quantum interactions capture periodic
relations that classical features and unstructured circuits represent less
directly.

### Synthetic periodic tasks

| Task | Inputs | Compared interaction structures | Metrics |
|---|---|---|---|
| Amplitude–phase interaction | Periodic patches with controlled amplitude and phase | None, local, cross-period, all-to-all | MSE, MAE |
| Cross-period interaction | Patches with coupled short and long periods | None, local, cross-period, all-to-all | MSE, MAE |
| Structured classification | Periodic pattern labels | Classical MLP, Fourier features, structured circuit | Accuracy, macro-F1 |

### Module A run record

Each run saves the encoder configuration, circuit topology, qubit count,
re-uploading layers, feature map, trainable parameter count, backend, shots,
circuit depth, circuit executions, latency, seed, and metrics.

## Module C experiments

Module C uses the current window to retrieve useful historical contexts and
fuses them with the current query representation in a final MLP.

```text
current sequence -> ordered Patch tokens -> task-aware query/key
                -> ANN candidates -> quantum re-rank -> historical context
                -> [query; context] -> MLP forecast
```

### Core comparison

| Variant | Historical context ranking | Forecast fusion |
|---|---|---|
| Classical retrieval | Learned key/query similarity | Query and context MLP |
| Quantum retrieval | Learned key/query candidates plus quantum fidelity re-ranking | Query and context MLP |

### Quantum-focused ablations

| Factor | Values |
|---|---|
| Re-ranking score | Learned metric, quantum fidelity |
| Quantum feature map | Structured angle projections |
| Qubit count | 2, 4, 6 |
| Kernel evaluation budget | Exact expectation, 100, 1,000, 10,000 shots |
| Alignment objective | Horizon-conditioned quantum alignment at 96, 192, 336, 720 |

### Module C run record

Each run saves dataset, split boundaries, candidate count, returned-context
count, query/key dimensions, forecasting-path configuration, feature-map
configuration, qubits, depth, shots, quantum calls, latency, seed, MSE, MAE,
Recall@K, and MRR.

## Forecasting comparison

The real-data comparison uses the same dataset split, feature setting, input
length, prediction horizon, and seed set for every method.

| Method | Role |
|---|---|
| Module C with quantum retrieval | Project method |
| Module C with classical retrieval | Retrieval control |
| Transformer | Forecasting baseline |
| iTransformer | Forecasting baseline |
| PatchTST | Forecasting baseline |

## Result artifacts

Each completed experiment produces:

1. A machine-readable JSON record under `outputs/`.
2. A run log with command, package versions, device, and elapsed time.
3. A row in the relevant results table with mean and standard deviation over
   the shared seeds.
4. A concise interpretation covering accuracy, retrieval quality, and resource
   use.

## Collaboration handoff

| Area | Owner | Shared interface |
|---|---|---|
| Shared sequence data and encoder | A + C | `src/qss/classical/encoder.py` and `docs/benchmark_protocol.md` |
| Module A circuits and synthetic experiments | Module A teammate | `src/qss/quantum/` and `experiments/synthetic_periodic/` |
| Module C retrieval, fusion, and real-data experiments | Module C owner | `src/qss/retrieval/` and `experiments/retrieval/` |
| Final comparison tables | A + C | `docs/` result tables and `outputs/` JSON records |

Each contribution uses a focused branch and pull request. A pull request
includes the experiment configuration, reproducibility record, tests, and the
result artifact for its completed scope.
