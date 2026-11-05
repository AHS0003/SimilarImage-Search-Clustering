"""K-Means diagnostics and explicit selection using gallery vectors only."""
import warnings
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def _feature_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite, two-dimensional feature matrix")
    return X


def _cluster_count(k, n_samples):
    if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= n_samples:
        raise ValueError(f"k must be an integer between 1 and the gallery size ({n_samples})")
    return int(k)


def _labels(labels, n_samples, name):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n_samples:
        raise ValueError(f"{name} must contain one label per gallery image")
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError(f"{name} must not contain NaN or infinity")
    return labels


def _finite_score(function, *args, **kwargs):
    try:
        value = float(function(*args, **kwargs))
    except (ValueError, FloatingPointError):
        return None
    return value if np.isfinite(value) else None


def clustering_metrics(X, labels, truth=None) -> dict[str, float | None]:
    """Three internal measures plus ARI/NMI when semantic truth is available.

    Internal measures require 2 <= distinct predicted clusters < n_samples.
    Undefined measures are None, never invented zeros. ARI and NMI remain
    available for one-cluster cases because their definitions permit them.
    """
    X = _feature_matrix(X)
    labels = _labels(labels, len(X), "labels")
    metrics = {name: None for name in ("silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi")}
    if 2 <= len(np.unique(labels)) < len(X):
        metrics["silhouette"] = _finite_score(silhouette_score, X, labels, metric="euclidean")
        metrics["davies_bouldin"] = _finite_score(davies_bouldin_score, X, labels)
        metrics["calinski_harabasz"] = _finite_score(calinski_harabasz_score, X, labels)
    if truth is not None:
        truth = _labels(truth, len(X), "truth")
        metrics["ari"] = _finite_score(adjusted_rand_score, truth, labels)
        metrics["nmi"] = _finite_score(normalized_mutual_info_score, truth, labels)
    return metrics


@dataclass
class ClusteringEvaluation:
    records: list[dict]
    selected_k: int
    model: KMeans
    labels: np.ndarray
    selection_reason: str


def evaluate_clustering(X, k_values, true_labels=None, random_state=42, manual_k=None) -> ClusteringEvaluation:
    """Fit candidate K-Means models on gallery X and select without queries.

    Automatic selection maximizes finite gallery silhouette, with lower k
    breaking ties (absolute tolerance 1e-12). A collapsed candidate requesting
    more clusters than it actually produces is not eligible for selection.
    If no eligible candidate exists, k=1 is fitted and the fallback is explicit.
    Supplying manual_k bypasses score-based selection. Candidate k values must
    already lie in [1, len(X)]; the driver must bound its default grid.
    """
    X = _feature_matrix(X)
    truth = None if true_labels is None else _labels(true_labels, len(X), "true_labels")
    candidates = sorted({_cluster_count(k, len(X)) for k in k_values})
    if manual_k is not None:
        manual_k = _cluster_count(manual_k, len(X))
        candidates = sorted(set(candidates) | {manual_k})
    records, models = [], {}

    def fit_candidate(k):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10, algorithm="lloyd")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            labels = model.fit_predict(X)
        actual = len(np.unique(labels))
        row = {
            "k": k,
            "n_clusters": actual,
            "inertia": float(model.inertia_),
            **clustering_metrics(X, labels, truth),
            "warning": "; ".join(str(item.message) for item in caught),
        }
        records.append(row)
        models[k] = model
        return row

    for k in candidates:
        fit_candidate(k)
    if manual_k is not None:
        selected_k = manual_k
        reason = "Manual k supplied; no score-based selection."
    else:
        eligible = [row for row in records if row["silhouette"] is not None
                    and row["n_clusters"] == row["k"]]
        if eligible:
            best = max(row["silhouette"] for row in eligible)
            selected_k = min(row["k"] for row in eligible if abs(row["silhouette"] - best) <= 1e-12)
            reason = "Maximum finite gallery silhouette; equal scores choose lower k."
        else:
            selected_k = 1
            if selected_k not in models:
                fit_candidate(selected_k)
            reason = "No eligible finite gallery silhouette; fallback to k=1."
    model = models[selected_k]
    return ClusteringEvaluation(records=sorted(records, key=lambda row: row["k"]),
                                selected_k=selected_k, model=model,
                                labels=model.labels_.copy(), selection_reason=reason)
