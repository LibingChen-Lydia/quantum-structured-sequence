"""C: Encoder-backed historical Patch retrieval with optional quantum reranking.

Pipeline:
1. Encode historical sequence patches with the shared structured encoder.
2. Fit latent categories (K-means) on historical Patch embeddings.
3. Route a query to its nearest category/categories.
4. Use cosine similarity to shortlist candidates, then optionally rerank them
   with a PennyLane fidelity kernel.

Install: pip install numpy scikit-learn pennylane
"""
from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.qss.classical.encoder import StructuredEncoderConfig, StructuredSequenceEncoder

try:
    import pennylane as qml
except ImportError:
    qml = None



def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


@dataclass
class RetrievedPatch:
    memory_id: Any
    cluster_id: int
    payload: Any
    cosine_score: float
    quantum_score: Optional[float]


class QuantumFidelityReranker:
    """Small phase-aware fidelity kernel; larger scores mean more similar."""

    def __init__(self, n_qubits: int = 4, shots: Optional[int] = None):
        if qml is None:
            raise ImportError("Install PennyLane to use quantum reranking: pip install pennylane")
        if n_qubits < 2:
            raise ValueError("n_qubits must be at least 2")
        self.n_qubits = n_qubits
        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        def phase_map(x):
            # Put data-dependent phases on a superposition, then entangle neighbors.
            for wire in range(n_qubits):
                qml.Hadamard(wires=wire)
                qml.RZ(x[wire], wires=wire)
            for wire in range(n_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
                qml.RZ(x[wire] * x[wire + 1], wires=wire + 1)
                qml.CNOT(wires=[wire, wire + 1])

        @qml.qnode(dev)
        def fidelity_circuit(x1, x2):
            phase_map(x1)
            qml.adjoint(phase_map)(x2)
            return qml.probs(wires=range(n_qubits))

        self._circuit = fidelity_circuit

    def score(self, angles_a: np.ndarray, angles_b: np.ndarray) -> float:
        # Probability of measuring all zeros after U(a) followed by U(b)^dagger.
        return float(self._circuit(angles_a, angles_b)[0])


class HierarchicalPatchRetriever:
    """Automatic category routing, classical shortlist, optional quantum rerank.

    Pass one embedding per historical Patch. Payloads can contain the original
    Patch and its known following segment, for example:
    {"patch": patch, "future": following_segment}.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        top_r_clusters: int = 2,
        candidate_pool: int = 20,
        top_k: int = 3,
        n_qubits: int = 4,
        shots: Optional[int] = None,
        use_quantum: bool = True,
        random_state: int = 7,
    ):
        self.n_clusters = n_clusters
        self.top_r_clusters = top_r_clusters
        self.candidate_pool = candidate_pool
        self.top_k = top_k
        self.random_state = random_state
        self.use_quantum = use_quantum
        self.reranker = QuantumFidelityReranker(n_qubits, shots) if use_quantum else None
        self._is_fitted = False

    def fit(
        self,
        historical_embeddings: np.ndarray,
        payloads: Optional[Sequence[Any]] = None,
        memory_ids: Optional[Sequence[Any]] = None,
    ):
        """Fit categories from training-history embeddings only."""
        x = np.asarray(historical_embeddings, dtype=float)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("historical_embeddings must be a non-empty 2D array")
        if not np.isfinite(x).all():
            raise ValueError("Embeddings must contain only finite values")
        n, dim = x.shape
        if self.n_clusters < 1 or self.n_clusters > n:
            raise ValueError("n_clusters must be between 1 and number of memories")
        if payloads is not None and len(payloads) != n:
            raise ValueError("payloads length must match embeddings")
        if memory_ids is not None and len(memory_ids) != n:
            raise ValueError("memory_ids length must match embeddings")

        # Standardize dimensions before clustering so large-scale features do
        # not dominate. Fit only on past/training data to avoid time leakage.
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        self.X_scaled_ = (x - self.mean_) / self.scale_
        self.X_unit_ = _normalize_rows(self.X_scaled_)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit(self.X_scaled_)
        self.labels_ = self.kmeans_.labels_
        self.centroids_unit_ = _normalize_rows(self.kmeans_.cluster_centers_)

        # Compress to the circuit width, then map standardized values to angles.
        self.n_qubits_ = self.reranker.n_qubits if self.reranker else 0
        if self.use_quantum:
            n_components = min(self.n_qubits_, dim, n)
            self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
            projected = self.pca_.fit_transform(self.X_scaled_)
            self.q_mean_ = projected.mean(axis=0)
            self.q_scale_ = projected.std(axis=0)
            self.q_scale_[self.q_scale_ < 1e-12] = 1.0
            self.quantum_angles_ = self._to_angles(projected)

        self.payloads_ = list(payloads) if payloads is not None else [None] * n
        self.memory_ids_ = list(memory_ids) if memory_ids is not None else list(range(n))
        self._is_fitted = True
        return self

    def _to_angles(self, projected: np.ndarray) -> np.ndarray:
        scaled = (projected - self.q_mean_) / self.q_scale_
        bounded = np.pi * np.tanh(scaled)
        if bounded.ndim == 1:
            bounded = bounded[None, :]
        # Pad if there are fewer input dimensions than qubits.
        if bounded.shape[1] < self.n_qubits_:
            bounded = np.pad(bounded, ((0, 0), (0, self.n_qubits_ - bounded.shape[1])))
        return bounded

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        candidate_pool: Optional[int] = None,
        top_r_clusters: Optional[int] = None,
    ) -> list[RetrievedPatch]:
        """Route -> shortlist by cosine -> quantum rerank -> return Top-K."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before retrieve()")
        q = np.asarray(query_embedding, dtype=float).reshape(1, -1)
        if q.shape[1] != self.X_scaled_.shape[1]:
            raise ValueError("Query embedding dimension does not match memory")
        if not np.isfinite(q).all():
            raise ValueError("Query embedding must contain only finite values")

        k = top_k or self.top_k
        m = candidate_pool or self.candidate_pool
        r = top_r_clusters or self.top_r_clusters
        q_scaled = (q - self.mean_) / self.scale_
        q_unit = _normalize_rows(q_scaled)[0]

        # Stage 1: route to the closest learned categories.
        route_scores = self.centroids_unit_ @ q_unit
        selected_clusters = np.argsort(-route_scores)[: min(r, self.n_clusters)]
        candidate_indices = np.flatnonzero(np.isin(self.labels_, selected_clusters))
        if candidate_indices.size == 0:
            return []

        # Stage 2: inexpensive classical similarity retrieves a small pool.
        cosine_scores = self.X_unit_[candidate_indices] @ q_unit
        order = np.argsort(-cosine_scores)[: min(m, candidate_indices.size)]
        shortlist = candidate_indices[order]
        shortlist_cosine = self.X_unit_[shortlist] @ q_unit

        # Stage 3: quantum kernel only scores the shortlist.
        quantum_scores = None
        if self.use_quantum:
            q_projected = self.pca_.transform(q_scaled)
            q_angles = self._to_angles(q_projected)[0]
            quantum_scores = np.array([
                self.reranker.score(q_angles, self.quantum_angles_[i])
                for i in shortlist
            ])
            final_order = np.argsort(-quantum_scores)[: min(k, len(shortlist))]
        else:
            final_order = np.argsort(-shortlist_cosine)[: min(k, len(shortlist))]

        hits = []
        for position in final_order:
            i = int(shortlist[position])
            hits.append(RetrievedPatch(
                memory_id=self.memory_ids_[i],
                cluster_id=int(self.labels_[i]),
                payload=self.payloads_[i],
                cosine_score=float(shortlist_cosine[position]),
                quantum_score=(float(quantum_scores[position]) if quantum_scores is not None else None),
            ))
        return hits


if __name__ == "__main__":
    # Match Time-Series-Library ETTh1 forecasting: seq_len=96, the standard
    # 12/4/4-month split, and stride-1 sliding windows for every pred_len.
    input_length = 96
    prediction_lengths = (96, 192, 336, 420)
    patch_length = 24
    data_dir = Path(__file__).resolve().parent / "data"
    ett = np.loadtxt(
        data_dir / "ETTh1.csv",
        delimiter=",",
        skiprows=1,
        usecols=(1, 2, 3, 4, 5, 6, 7),
    )
    ett = np.atleast_2d(ett).astype(float)
    target_index = 6  # OT

    train_end = 12 * 30 * 24
    validation_end = 16 * 30 * 24
    test_end = 20 * 30 * 24
    if ett.shape[0] < test_end:
        raise ValueError("ETTh1 does not contain enough hourly rows for the TSL split")

    # Same train-only StandardScaler convention used by TSL.
    feature_mean = ett[:train_end].mean(axis=0)
    feature_std = ett[:train_end].std(axis=0)
    feature_std[feature_std < 1e-12] = 1.0
    scaled_ett = (ett - feature_mean) / feature_std
    train_data = scaled_ett[:train_end]

    # Fit encoder normalization only on the training partition. Each 96-hour
    # sample contains four 24-hour patches, whose embeddings are mean pooled.
    encoder = StructuredSequenceEncoder(
        StructuredEncoderConfig(patch_lengths=(patch_length,), period=24, stride=patch_length)
    )
    encoder.fit(train_data[None, :, :])

    # Cache all stride-1 training windows valid for the shortest forecast.
    train_origins_all = np.arange(input_length, train_end - min(prediction_lengths) + 1)
    train_windows_all = np.stack([
        train_data[origin - input_length : origin] for origin in train_origins_all
    ])
    train_embeddings_all = encoder.transform(train_windows_all).scales[
        patch_length
    ].embedding.mean(axis=1)

    # TSL test samples begin at validation_end, use the prior 96 hours as input,
    # and slide one hour at a time. Shorter horizons have more test windows.
    test_start = validation_end
    query_origins_all = np.arange(
        test_start,
        test_end - min(prediction_lengths) + 1,
    )
    query_windows_all = np.stack([
        scaled_ett[origin - input_length : origin] for origin in query_origins_all
    ])
    query_embeddings_all = encoder.transform(query_windows_all).scales[
        patch_length
    ].embedding.mean(axis=1)

    print(f"Input length: {input_length} hours")
    print(f"Prediction lengths: {prediction_lengths} hours")
    print("Input variables: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT")
    print("Target: OT")
    print(f"Split hours: train={train_end}, validation={validation_end - train_end}, test={test_end - validation_end}")
    print("Window stride: 1 hour")
    metrics_rows = []
    use_quantum = qml is not None

    for horizon in prediction_lengths:
        # Match TSL's Dataset_ETT_hour count separately for each pred_len.
        train_window_count = train_end - input_length - horizon + 1
        train_origins = train_origins_all[:train_window_count]
        memory_embeddings = train_embeddings_all[:train_window_count]
        memories = [
            {
                "window_start": int(origin - input_length),
                "forecast_origin": int(origin),
                "future": scaled_ett[origin : origin + horizon, target_index].copy(),
            }
            for origin in train_origins
        ]
        retriever = HierarchicalPatchRetriever(
            n_clusters=min(8, len(memory_embeddings)),
            top_r_clusters=2,
            candidate_pool=20,
            top_k=3,
            use_quantum=use_quantum,
        ).fit(np.asarray(memory_embeddings), payloads=memories)

        test_window_count = test_end - test_start - horizon + 1
        predictions = []
        actuals = []
        first_hits = None
        for origin, query_embedding in zip(
            query_origins_all[:test_window_count],
            query_embeddings_all[:test_window_count],
        ):
            hits = retriever.retrieve(query_embedding)
            if not hits:
                continue
            retrieved_futures = np.stack([hit.payload["future"] for hit in hits])
            predictions.append(retrieved_futures.mean(axis=0))
            actuals.append(scaled_ett[origin : origin + horizon, target_index])
            if first_hits is None:
                first_hits = hits

        if not predictions:
            print(f"pred_len={horizon}: no test windows")
            continue

        actual_scaled = np.concatenate(actuals)
        prediction_scaled = np.concatenate(predictions)
        mae_scaled = float(np.mean(np.abs(prediction_scaled - actual_scaled)))
        mse_scaled = float(np.mean((prediction_scaled - actual_scaled) ** 2))
        actual_raw = actual_scaled * feature_std[target_index] + feature_mean[target_index]
        prediction_raw = prediction_scaled * feature_std[target_index] + feature_mean[target_index]
        mae_raw = float(np.mean(np.abs(prediction_raw - actual_raw)))
        mse_raw = float(np.mean((prediction_raw - actual_raw) ** 2))
        validation_window_count = validation_end - train_end - horizon + 1

        print(
            f"pred_len={horizon}: train_windows={train_window_count}, "
            f"validation_windows={validation_window_count}, "
            f"test_windows={len(predictions)}"
        )
        print(
            f"  normalized MAE={mae_scaled:.3f}, normalized MSE={mse_scaled:.3f}; "
            f"raw OT MAE={mae_raw:.3f}, raw OT MSE={mse_raw:.3f}"
        )
        metrics_rows.append({
            "dataset": "ETTh1",
            "model": "QSS_HierarchicalRetriever",
            "target": "OT",
            "seq_len": input_length,
            "pred_len": horizon,
            "window_stride": 1,
            "train_windows": train_window_count,
            "validation_windows": validation_window_count,
            "test_windows": len(predictions),
            "quantum_reranking": use_quantum,
            "mae_normalized": mae_scaled,
            "mse_normalized": mse_scaled,
            "mae_raw_ot_units": mae_raw,
            "mse_raw_ot_units_squared": mse_raw,
        })
        if first_hits:
            print("  first test window retrievals:")
            for hit in first_hits:
                print(
                    f"    window_start={hit.payload['window_start']:5d}, "
                    f"cosine={hit.cosine_score:.3f}, quantum={hit.quantum_score}"
                )

    if metrics_rows:
        results_dir = Path(__file__).resolve().parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "c_retrieval_etth1.csv"
        with results_path.open("w", newline="", encoding="utf-8") as result_file:
            writer = csv.DictWriter(result_file, fieldnames=list(metrics_rows[0]))
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"Metrics saved to: {results_path}")
