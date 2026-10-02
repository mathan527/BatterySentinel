"""
BatterySentinel — Experiment 02: Bayesian Risk Engine
======================================================
Trains and evaluates the Gaussian Naive Bayes degradation risk predictor.

Steps:
  1. Battery-level train/test split (no data leakage)
  2. Demonstrate Bayes' Rule step-by-step
  3. Fit GaussianNaiveBayesRiskEngine on training batteries
  4. Predict posteriors on test batteries
  5. Evaluate accuracy, precision, recall
  6. Plot: risk score vs cycle, posterior distributions
  7. Save model and results
"""

from __future__ import annotations
import json, logging, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("experiment_02")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config
from src.bayesian_risk import (
    GaussianNaiveBayesRiskEngine, battery_level_train_test_split,
    demonstrate_bayes_rule, CLASSES,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
RES_DIR = PROJECT_ROOT / "reports" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", font_scale=1.1)
LABEL_COLORS = {"Healthy": "#2ecc71", "Early Degradation": "#f39c12",
                "Degraded": "#e74c3c", "Unknown": "#95a5a6"}


def plot_risk_score_vs_cycle(df_pred: pd.DataFrame, df_orig: pd.DataFrame,
                              batteries: list[str], save: bool = True) -> None:
    n = len(batteries)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4 * n))
    if n == 1:
        axes = [axes]
    for ax, bid in zip(axes, batteries):
        mask = df_orig["battery_id"] == bid
        cyc = df_orig.loc[mask, "cycle_number"].values
        risk = df_pred.loc[mask, "risk_score"].values
        soh  = df_orig.loc[mask, "soh"].values

        color_pts = [LABEL_COLORS.get(df_pred.loc[i, "predicted_class"], "#7f8c8d")
                     for i in df_orig[mask].index]

        ax2 = ax.twinx()
        ax2.plot(cyc, soh, color="#3498db", linewidth=1.2, alpha=0.5, label="SOH (%)")
        ax2.set_ylabel("SOH (%)", color="#3498db", fontsize=10)
        ax2.tick_params(axis="y", labelcolor="#3498db")

        ax.scatter(cyc, risk, c=color_pts, s=20, zorder=4, alpha=0.85)
        ax.plot(cyc, risk, color="#2c3e50", linewidth=0.8, alpha=0.5)
        ax.axhline(0.5, color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7,
                   label="Risk=0.5 boundary")
        ax.set_ylabel("Risk Score P(Degraded|E)", fontsize=10)
        ax.set_xlabel("Cycle Number", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Battery {bid} — Bayesian Risk Score vs Cycle\n"
                     "Colour = predicted class  |  Blue line = actual SOH", fontsize=11)
        ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    path = FIG_DIR / "11_bayesian_risk_vs_cycle.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_posterior_heatmap(df_pred: pd.DataFrame, df_orig: pd.DataFrame,
                            battery_id: str, save: bool = True) -> None:
    mask = df_orig["battery_id"] == battery_id
    cyc  = df_orig.loc[mask, "cycle_number"].values
    ph   = df_pred.loc[mask, "P_Healthy"].values
    ped  = df_pred.loc[mask, "P_Early_Degradation"].values
    pd_  = df_pred.loc[mask, "P_Degraded"].values

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.stackplot(cyc, ph, ped, pd_,
                 labels=["P(Healthy)", "P(Early Degradation)", "P(Degraded)"],
                 colors=["#2ecc71", "#f39c12", "#e74c3c"], alpha=0.75)
    ax.set_xlabel("Cycle Number", fontsize=11)
    ax.set_ylabel("Posterior Probability", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"Battery {battery_id} — Posterior Probabilities (Bayes Rule)\n"
                 "P(class|Evidence) updates as new cycles are observed", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

    path = FIG_DIR / f"12_posterior_heatmap_{battery_id}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, save: bool = True) -> None:
    labels = ["Healthy", "Early Degradation", "Degraded"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_title("Gaussian Naive Bayes — Confusion Matrix\n"
                 "Rows = True labels | Columns = Predicted labels", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "13_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def main() -> None:
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["processed_dir"] / "cycle_data.csv")
    log.info(f"Loaded: {df.shape}")

    # ── 1. Bayes' Rule Step-by-Step Demonstration ────────────────────────────
    log.info("\n[1] BAYES RULE DEMONSTRATION")
    # Use real statistics from the dataset
    soh_healthy  = df[df["degradation_label"] == "Healthy"]["capacity"].dropna()
    soh_degraded = df[df["degradation_label"] == "Degraded"]["capacity"].dropna()

    prior_deg = len(df[df["degradation_label"] == "Degraded"]) / len(df.dropna(subset=["degradation_label"]))
    # P(capacity < 1.0 | class)
    thresh = 1.0
    like_deg = float((soh_degraded < thresh).mean())
    like_hlt = float((soh_healthy  < thresh).mean())

    bayes_demo = demonstrate_bayes_rule(prior_deg, like_deg, like_hlt,
                                        f"capacity < {thresh:.1f} Ah")
    log.info("\n  MANUAL BAYES CALCULATION:")
    for k, v in bayes_demo.items():
        log.info(f"    {k:<45}: {v}")

    with open(RES_DIR / "bayes_rule_demo.json", "w", encoding="utf-8") as fh:
        json.dump(bayes_demo, fh, indent=2, default=str)

    # ── 2. Battery-level split ───────────────────────────────────────────────
    log.info("\n[2] BATTERY-LEVEL TRAIN/TEST SPLIT")
    test_batteries = cfg["training"]["test_batteries"]
    train_df, test_df = battery_level_train_test_split(df, test_batteries=test_batteries)
    log.info(f"  Train: {train_df['battery_id'].nunique()} batteries, {len(train_df)} cycles")
    log.info(f"  Test : {test_df['battery_id'].nunique()} batteries, {len(test_df)} cycles")

    # ── 3. Fit model ─────────────────────────────────────────────────────────
    log.info("\n[3] FITTING GAUSSIAN NAIVE BAYES")
    evidence_feats = [f for f in cfg.get("bayesian", {}).get("evidence_features", [
        "capacity", "soh", "voltage_mean", "voltage_std",
        "temperature_mean", "temperature_std", "Re_Ohm",
    ]) if f in df.columns]

    engine = GaussianNaiveBayesRiskEngine(evidence_features=evidence_feats)
    engine.fit(train_df.dropna(subset=["degradation_label"]))

    # ── 4. Predict on test set ───────────────────────────────────────────────
    log.info("\n[4] PREDICTING ON TEST SET")
    test_clean = test_df.dropna(subset=["degradation_label"]).copy()
    preds = engine.predict_df(test_clean)
    test_clean = test_clean.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    combined = pd.concat([test_clean[["battery_id","cycle_number","soh",
                                       "degradation_label"]], preds], axis=1)

    # ── 5. Evaluation ────────────────────────────────────────────────────────
    log.info("\n[5] EVALUATION")
    y_true = test_clean["degradation_label"].values
    y_pred = preds["predicted_class"].values

    # Filter to known classes
    mask_known = np.isin(y_true, CLASSES) & np.isin(y_pred, CLASSES)
    y_true_k = y_true[mask_known]
    y_pred_k = y_pred[mask_known]

    report = classification_report(y_true_k, y_pred_k, labels=CLASSES, zero_division=0)
    log.info(f"\n  Classification Report:\n{report}")

    # Per-class accuracy
    for cls in CLASSES:
        mask = y_true_k == cls
        if mask.sum() > 0:
            acc = (y_pred_k[mask] == cls).mean()
            log.info(f"  Accuracy for {cls:<22}: {acc:.3f}  (n={mask.sum()})")

    # Overall accuracy
    overall_acc = (y_true_k == y_pred_k).mean()
    log.info(f"\n  Overall Accuracy: {overall_acc:.4f}")

    # ── 6. Save results ──────────────────────────────────────────────────────
    combined.to_csv(RES_DIR / "bayesian_predictions.csv", index=False)
    engine.save(RES_DIR / "bayesian_model.json")

    results_summary = {
        "generated_at": datetime.now().isoformat(),
        "train_batteries": train_df["battery_id"].unique().tolist(),
        "test_batteries": test_df["battery_id"].unique().tolist(),
        "n_train_cycles": int(len(train_df)),
        "n_test_cycles": int(len(test_df)),
        "evidence_features": evidence_feats,
        "overall_accuracy": round(float(overall_acc), 4),
        "bayes_demo": bayes_demo,
        "class_priors": engine.class_priors,
    }
    with open(RES_DIR / "bayesian_results.json", "w", encoding="utf-8") as fh:
        json.dump(results_summary, fh, indent=2, default=str)

    # ── 7. Visualisations ────────────────────────────────────────────────────
    log.info("\n[6] VISUALISATIONS")
    test_bids = sorted(test_df["battery_id"].unique().tolist())[:3]
    plot_risk_score_vs_cycle(preds, test_clean, test_bids[:min(3, len(test_bids))])
    for bid in test_bids[:2]:
        plot_posterior_heatmap(preds, test_clean, bid)
    plot_confusion_matrix(y_true_k, y_pred_k)

    log.info("\n" + "=" * 60)
    log.info("  EXPERIMENT 02: Bayesian Risk Engine COMPLETE")
    log.info("=" * 60)
    log.info(f"  Overall accuracy : {overall_acc:.4f}")
    log.info(f"  Model saved      : {RES_DIR / 'bayesian_model.json'}")
    log.info(f"  Predictions saved: {RES_DIR / 'bayesian_predictions.csv'}")


if __name__ == "__main__":
    main()
