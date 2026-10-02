"""
BatterySentinel — Experiment 04: K-Means Clustering
====================================================
Unsupervised learning: cluster discharge cycles by feature similarity.
Compare clusters to ground-truth degradation labels.
"""

from __future__ import annotations
import json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("experiment_04")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config
from src.clustering import (
    prepare_cluster_features, elbow_analysis,
    fit_kmeans, interpret_clusters, assign_cluster_names,
    DEFAULT_CLUSTER_FEATURES,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
RES_DIR = PROJECT_ROOT / "reports" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="darkgrid", font_scale=1.1)

CLUSTER_COLORS = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#3498db"]
LABEL_COLORS   = {"Healthy": "#2ecc71", "Early Degradation": "#f39c12",
                  "Degraded": "#e74c3c", "Unknown": "#95a5a6"}


def plot_elbow(elbow_res, save=True):
    ks = sorted(elbow_res.keys())
    inertias   = [elbow_res[k]["inertia"] for k in ks]
    silhouettes = [elbow_res[k]["silhouette"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(ks, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters K", fontsize=12)
    ax1.set_ylabel("WCSS (Inertia)", fontsize=12)
    ax1.set_title("Elbow Method\nWCSS = sum||x - mu_k||^2 per cluster", fontsize=11)
    ax1.grid(True, alpha=0.4)

    ax2.plot(ks, silhouettes, "rs-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters K", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Analysis\nHigher = better cluster separation", fontsize=11)
    ax2.grid(True, alpha=0.4)

    fig.suptitle("K-Means Cluster Count Selection", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "16_kmeans_elbow.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_pca_clusters(X_scaled, labels, df_sub, cluster_names, save=True):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: coloured by K-Means cluster
    unique_labels = sorted(set(labels))
    for i, cid in enumerate(unique_labels):
        mask = labels == cid
        name = cluster_names.get(cid, f"Cluster {cid}")
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], s=12, alpha=0.5,
                    label=f"Cluster {cid}: {name}")
    ax1.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
    ax1.set_title("K-Means Clusters in PCA Space\n"
                  "(Unsupervised — no labels used)", fontsize=11)
    ax1.legend(fontsize=9)

    # Right: coloured by true degradation label
    true_labels = df_sub["degradation_label"].fillna("Unknown").values
    for lbl, col in LABEL_COLORS.items():
        mask = true_labels == lbl
        if mask.sum() == 0:
            continue
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    c=col, s=12, alpha=0.5, label=lbl)
    ax2.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
    ax2.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
    ax2.set_title("True Degradation Labels in PCA Space\n"
                  "(Ground truth for comparison)", fontsize=11)
    ax2.legend(fontsize=9)

    fig.suptitle("K-Means vs True Labels: PCA Projection", fontsize=13)
    fig.tight_layout()
    path = FIG_DIR / "17_kmeans_pca.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_cluster_soh(df_with_cluster, cluster_names, save=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    clusters = sorted(df_with_cluster["cluster"].unique())
    data = [df_with_cluster[df_with_cluster["cluster"] == c]["soh"].dropna()
            for c in clusters]
    labels_plot = [f"C{c}: {cluster_names.get(c, '')}\n(n={len(d)})"
                   for c, d in zip(clusters, data)]
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(clusters))]
    bp = ax.boxplot(data, labels=labels_plot, patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.set_ylabel("SOH (%)", fontsize=12)
    ax.set_title("SOH Distribution per K-Means Cluster\n"
                 "Validates cluster-to-degradation-state alignment", fontsize=11)
    ax.grid(axis="y", alpha=0.4)

    path = FIG_DIR / "18_cluster_soh_boxplot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def main():
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["processed_dir"] / "cycle_data.csv")
    k_opt = cfg.get("clustering", {}).get("k", 3)

    log.info("[1] Preparing cluster features + standardisation")
    features = [f for f in DEFAULT_CLUSTER_FEATURES if f in df.columns]
    X_scaled, feat_names, valid_idx = prepare_cluster_features(df, features=features)

    log.info("\n[2] Elbow analysis (K = 2 to 9)")
    elbow_res = elbow_analysis(X_scaled, k_range=range(2, 10))
    for k, metrics in sorted(elbow_res.items()):
        log.info(f"  K={k}: WCSS={metrics['inertia']:.2f}  Silhouette={metrics['silhouette']:.4f}")
    plot_elbow(elbow_res)

    log.info(f"\n[3] Fitting K-Means with K={k_opt}")
    km_model, labels = fit_kmeans(X_scaled, k=k_opt)

    log.info("\n[4] Interpreting clusters")
    df_valid = df.loc[valid_idx].copy()
    cluster_stats = interpret_clusters(df_valid, labels, valid_idx, feat_names)
    cluster_names = assign_cluster_names(cluster_stats)

    log.info("\n  Cluster summary:")
    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster_id"])
        soh_m = row.get("soh_mean", "N/A")
        n = row["n_members"]
        dom = row.get("dominant_label", "?")
        purity = row.get("label_purity", 0.0)
        name = cluster_names.get(cid, "?")
        log.info(f"  Cluster {cid} ({name:<18}): n={n:4d}  mean_SOH={soh_m:.2f}%  "
                 f"dominant={dom}  purity={purity:.2%}")

    log.info("\n[5] Visualisations")
    plot_pca_clusters(X_scaled, labels, df_valid, cluster_names)
    df_valid["cluster"] = labels
    plot_cluster_soh(df_valid, cluster_names)

    # Save results
    cluster_stats.to_csv(RES_DIR / "clustering_stats.csv", index=False)
    with open(RES_DIR / "clustering_results.json", "w", encoding="utf-8") as fh:
        json.dump({
            "k_optimal": k_opt,
            "elbow_analysis": {str(k): v for k, v in elbow_res.items()},
            "cluster_names": {str(k): v for k, v in cluster_names.items()},
            "cluster_summary": cluster_stats.to_dict(orient="records"),
        }, fh, indent=2, default=str)

    log.info("\n" + "=" * 60)
    log.info("  EXPERIMENT 04: K-Means Clustering COMPLETE")
    log.info("=" * 60)
    log.info(f"  Optimal K     : {k_opt}")
    log.info(f"  Best silhouette: {max(v['silhouette'] for v in elbow_res.values()):.4f}")


if __name__ == "__main__":
    main()
