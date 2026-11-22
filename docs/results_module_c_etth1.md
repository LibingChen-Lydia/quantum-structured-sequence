# Module C — ETTh1 results

## Protocol

ETTh1 multivariate (`M`) long-term forecasting, following Time-Series-Library `Dataset_ETT_hour`:
input length 96; prediction horizons 96, 192, 336, and 720; first 12 months for training, next
4 months for validation, and final 4 months for test. The shared structured encoder fits its input
and feature normalization on training windows only. Every test query retrieves only memory windows
whose forecast start precedes that query's forecast start.

The reported run trains the task-aware key/query retrieval representation for five epochs, forms a
training-memory bank, retrieves 20 candidates, re-ranks to Top-3, and predicts with attention over
the retrieved future trajectories. MSE and MAE are computed on the normalized values, consistently
with the benchmark loader. `conditional Recall@1` measures whether the first returned item is the
closest future trajectory among the fixed ANN candidate set.

| Horizon | Reranker | MSE | MAE | Conditional Recall@1 | Rerank ms/query |
|---:|---|---:|---:|---:|---:|
| 96 | Cosine | 1.4311 | 0.9464 | 0.239 | 0.865 |
| 96 | Learned metric | 1.4550 | 0.9541 | 0.115 | 0.859 |
| 96 | Quantum fidelity | **1.4168** | **0.9421** | 0.205 | 0.952 |
| 192 | Cosine | 1.6270 | 0.9937 | 0.217 | 0.855 |
| 192 | Learned metric | 1.6026 | 0.9850 | 0.072 | 0.853 |
| 192 | Quantum fidelity | **1.5678** | **0.9730** | 0.196 | 0.946 |
| 336 | Cosine | 1.3005 | 0.8924 | **0.213** | 0.843 |
| 336 | Learned metric | **1.2877** | **0.8882** | 0.079 | 0.842 |
| 336 | Quantum fidelity | 1.3235 | 0.8976 | 0.130 | 0.937 |
| 720 | Cosine | **1.2270** | **0.8964** | **0.193** | 0.807 |
| 720 | Learned metric | 1.2323 | 0.8969 | 0.088 | 0.807 |
| 720 | Quantum fidelity | 1.2394 | 0.8980 | 0.143 | 0.901 |

RBF gives the same ordering as cosine for these L2-normalized raw embeddings and is omitted from the
table. Quantum fidelity adds about 0.09 ms/query to the current simulator-style kernel calculation.
It improves the 96- and 192-step forecasts, while longer horizons require better kernel/representation
design before any improvement claim is made.
