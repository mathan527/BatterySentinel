"""
BatterySentinel — src/clustering.py
======================================
K-Means unsupervised clustering of battery discharge cycles.

Unit 1 Concepts Covered
------------------------
  • Unsupervised Learning — finding structure without labels
  • K-Means algorithm — partition cycles into k groups
  • Elbow method — selecting optimal k via inertia
  • Cluster interpretation — mapping clusters to degradation states
  • Cluster statistics — mean, variance per cluster per feature
  • Silhouette score — cluster quality metric

Mathematical Background
-----------------------
K-Means minimises the within-cluster sum of squares (WCSS):

    J = sum_{k=1}^{K} sum_{x in C_k} ||x - mu_k||^2

where mu_k = (1/|C_k|) * sum_{x in C_k} x   (centroid of cluster k)

Algorithm (Lloyd's):
  1. Initialise K centroids (k-means++ by default)
  2. Assign each point to the nearest centroid:
       c_i = argmin_k ||x_i - mu_k||^2
  3. Recompute centroids:
       mu_k = mean of all x_i assigned to cluster k
  4. Repeat 2–3 until convergence

Elbow Method:
  Plot WCSS vs K. The "elbow" is the K beyond which adding more clusters
  gives diminishing returns on WCSS reduction.

Silhouette Score:
  s(i) = (b_i - a_i) / max(a_i, b_i)
  a_i = mean intra-cluster distance
  b_i = mean nearest-cluster distance
  s in [-1, 1]; higher = better cluster cohesion + separation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CLUSTER_FEATURES = [
    "voltage_mean", "voltage_std",
    "current_mean", "current_std",
    "temperature_mean", "temperature_std",
    "capacity", "soh",
    "Re_Ohm", "discharge_duration",
    "cycle_norm",
]


def prepare_cluster_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, list[str], pd.Index]:
    """
    Prepare and standardise features for K-Means clustering.

    K-Means uses Euclidean distance, so features must be on the same scale.
    Standardisation: z_i = (x_i - mu) / sigma

    Parameters
    ----------
    df       : cycle-level DataFrame
    features : list of feature columns to use (None = defaults)

    Returns
    -------
    X_scaled    : np.ndarray — standardised features, shape (n_valid, n_features)
    feat_names  : list[str]  — feature names used
    valid_index : pd.Index   — original DataFrame index of valid rows
    """
    if features is None:
        features = DEFAULT_CLUSTER_FEATURES
    feat_names = [f for f in features if f in df.columns]
    log.info(f"  Clustering features: {feat_names}")

    sub = df[feat_names].dropna()
    valid_index = sub.index

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub.values)

    log.info(f"  Valid rows for clustering: {len(sub)} / {len(df)}")
    return X_scaled, feat_names, valid_index


# ─────────────────────────────────────────────────────────────────────────────
# Elbow method
# ─────────────────────────────────────────────────────────────────────────────

def elbow_analysis(
    X: np.ndarray,
    k_range: range | None = None,
    random_state: int = 42,
) -> dict[int, dict[str, float]]:
    """
    Run K-Means for multiple values of K and record inertia + silhouette.

    Elbow Formula:
        WCSS(K) = sum_{k=1}^{K} sum_{x in C_k} ||x - mu_k||^2

    The elbow is where the rate of WCSS decrease sharply slows down —
    indicating that additional clusters add little structural information.

    Parameters
    ----------
    X         : standardised feature matrix
    k_range   : range of K values to try (default: 2–10)
    random_state : seed for reproducibility

    Returns
    -------
    dict: {k: {"inertia": float, "silhouette": float}}
    """
    if k_range is None:
        k_range = range(2, 11)

    results: dict[int, dict[str, float]] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X, labels, sample_size=min(2000, len(X)),
                                     random_state=random_state))
        results[k] = {"inertia": round(inertia, 4), "silhouette": round(sil, 4)}
        log.debug(f"    K={k}: WCSS={inertia:.2f}  Silhouette={sil:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Fit K-Means
# ─────────────────────────────────────────────────────────────────────────────

def fit_kmeans(
    X: np.ndarray,
    k: int = 3,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit K-Means with the specified number of clusters.

    Parameters
    ----------
    X            : standardised feature matrix
    k            : number of clusters
    random_state : seed

    Returns
    -------
    (kmeans_model, cluster_labels)
    """
    km = KMeans(n_clusters=k, n_init=20, max_iter=500,
                random_state=random_state, algorithm="lloyd")
    labels = km.fit_predict(X)

    inertia = float(km.inertia_)
    sil = float(silhouette_score(X, labels, sample_size=min(2000, len(X)),
                                  random_state=random_state))
    log.info(f"  K-Means (K={k}): WCSS={inertia:.4f}  Silhouette={sil:.4f}")

    return km, labels


# ─────────────────────────────────────────────────────────────────────────────
# Cluster interpretation
# ─────────────────────────────────────────────────────────────────────────────

def interpret_clusters(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    valid_index: pd.Index,
    feat_names: list[str],
) -> pd.DataFrame:
    """
    Annotate the DataFrame with cluster labels and compute cluster statistics.

    Returns a DataFrame with one row per cluster containing:
        cluster_id, n_members, mean SOH, mean capacity,
        dominant_degradation_label, and per-feature mean/std.
    """
    df_c = df.copy()
    df_c["cluster"] = -1
    df_c.loc[valid_index, "cluster"] = cluster_labels

    records: list[dict] = []
    for cid in sorted(np.unique(cluster_labels)):
        members = df_c[df_c["cluster"] == cid]
        row: dict[str, Any] = {
            "cluster_id": int(cid),
            "n_members": int(len(members)),
        }
        for feat in feat_names + ["soh", "capacity"]:
            if feat in members.columns:
                valid = members[feat].dropna()
                row[f"{feat}_mean"] = round(float(valid.mean()), 5) if len(valid) else np.nan
                row[f"{feat}_std"] = round(float(valid.std()), 5) if len(valid) > 1 else np.nan

        # Dominant label
        if "degradation_label" in members.columns:
            label_counts = members["degradation_label"].value_counts()
            row["dominant_label"] = str(label_counts.index[0]) if not label_counts.empty else "Unknown"
            row["label_purity"] = round(float(label_counts.iloc[0] / len(members)), 4) if len(members) > 0 else 0.0
        records.append(row)

    return pd.DataFrame(records)


def assign_cluster_names(
    cluster_stats: pd.DataFrame,
    soh_col: str = "soh_mean",
) -> dict[int, str]:
    """
    Assign human-readable names to clusters based on mean SOH ordering.

    Rule: sort clusters by mean SOH (descending).
        Highest SOH cluster → "Cluster: Healthy-like"
        Middle SOH cluster  → "Cluster: Degrading"
        Lowest SOH cluster  → "Cluster: Degraded-like"

    Note: Cluster names are INTERPRETATIONS, not ground-truth labels.
    They describe the DOMINANT characteristic of the cluster members.
    """
    if soh_col not in cluster_stats.columns:
        return {int(row["cluster_id"]): f"Cluster_{int(row['cluster_id'])}"
                for _, row in cluster_stats.iterrows()}

    sorted_clusters = cluster_stats.sort_values(soh_col, ascending=False)
    n = len(sorted_clusters)
    name_map = ["Healthy-like", "Degrading", "Degraded-like",
                "Cluster-4", "Cluster-5"]  # extend if needed

    names: dict[int, str] = {}
    for i, (_, row) in enumerate(sorted_clusters.iterrows()):
        cid = int(row["cluster_id"])
        names[cid] = name_map[i] if i < len(name_map) else f"Cluster-{i+1}"

    return names
