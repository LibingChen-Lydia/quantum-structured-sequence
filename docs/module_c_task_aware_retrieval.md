# Module C — task-aware historical memory retrieval

## Retrieval contract

The shared structured encoder produces one embedding for every historical Patch. A task-aware
retriever maps that embedding to a memory key \(k_i\). The paired future trajectory \(v_i\) is stored
as the memory value. A query uses only its current and past observations.

```text
past window → shared encoder → query/key embedding → ANN candidate retrieval
                                                   → quantum fidelity re-ranking → Top-K futures → attention fusion
```

## Training objective

The forecast loss is combined with a contrastive retrieval loss. Positive historical windows have
similar future trajectories; negatives have dissimilar future trajectories.

\[
\mathcal L = \mathcal L_{forecast} + \lambda_{ctr}\mathcal L_{contrastive}.
\]

The future trajectory is supervision during training and a stored value during retrieval. It is never
included in a query feature vector.

The forecast head uses attention over the quantum-reranked Top-K future trajectories and predicts the
target horizon from the concatenated query and retrieved context.

## Temporal validity

Every memory item records its ending timestamp. A query at time \(t\) can retrieve only items ending
before \(t\). Train, validation, and test splits fit their encoders and retrieval representations using
training data only. The initial implementation uses exact cosine filtering when a time cutoff is applied;
this guarantees temporal validity. HNSW is used for unrestricted candidate search and can be time-sharded
for larger memory banks.

## Evaluation

Each reranker receives the same ANN candidate set. Compare cosine, RBF, learned metric, and quantum
fidelity re-ranking with Recall@K, MRR, MSE, MAE, latency, quantum calls, and shots. Relevance labels
for Recall@K and MRR are defined from the held-out future-trajectory similarity relation.
