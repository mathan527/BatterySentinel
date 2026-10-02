"""
BatterySentinel — Experiment 01: Statistical Analysis
======================================================
Runs the complete Unit 1 descriptive statistics pipeline.

Outputs
-------
  reports/results/statistical_summary.csv
  reports/results/covariance_matrix.csv
  reports/results/correlation_matrix.csv
  reports/results/independence_tests.json
  reports/results/conditional_stats.json
  reports/figures/01_soh_vs_cycle.png
  reports/figures/02_capacity_vs_cycle.png
  reports/figures/03_feature_distributions.png
  reports/figures/05_covariance_matrix.png
  reports/figures/06_correlation_pearson.png
  reports/figures/07_boxplot_soh.png
  reports/figures/08_anomaly_B0005_soh.png
  reports/figures/09_soh_distribution.png
  reports/figures/10_resistance_vs_cycle.png
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("experiment_01")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config
from src.statistics_engine import (
    describe_all_features, compute_covariance_matrix,
    compute_correlation_matrix, independence_test,
    conditional_statistics, full_statistical_summary,
    iqr_anomaly_flags, describe_feature,
)
from src.probability_engine import (
    fit_kde, compute_cdf, compare_class_distributions,
    fit_normal_distribution,
)
from src.visualization import (
    plot_soh_vs_cycle, plot_capacity_vs_cycle,
    plot_feature_distributions, plot_pdf_comparison,
    plot_covariance_heatmap, plot_correlation_heatmap,
    plot_quantile_boxplots, plot_anomaly_flags,
    plot_soh_distribution, plot_resistance_vs_cycle,
)


def main() -> None:
    cfg = load_config()
    results_dir = PROJECT_ROOT / cfg["reporting"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load processed dataset
    csv_path = PROJECT_ROOT / cfg["data"]["processed_dir"] / "cycle_data.csv"
    if not csv_path.exists():
        log.error(f"cycle_data.csv not found at {csv_path}. Run preprocess_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    log.info(f"Loaded cycle_data.csv: {df.shape[0]} rows x {df.shape[1]} cols")

    eol_pct = cfg["soh"]["eol_threshold"] * 100

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Descriptive Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[1] Descriptive Statistics (Mean, Variance, Quantiles, Skewness ...)")

    key_features = [
        "voltage_mean", "voltage_std", "current_mean", "current_std",
        "temperature_mean", "temperature_std", "capacity", "soh",
        "Re_Ohm", "Rct_Ohm", "cycle_number", "discharge_duration",
        "voltage_range", "current_range", "temperature_range",
    ]
    key_features = [f for f in key_features if f in df.columns]

    stats_df = describe_all_features(df, key_features)

    log.info("\n  Key statistics for SOH:")
    soh_stats = describe_feature(df["soh"], "soh")
    for k, v in soh_stats.items():
        if k not in ("feature",):
            log.info(f"    {k:<25}: {v}")

    log.info("\n  Key statistics for capacity:")
    cap_stats = describe_feature(df["capacity"].dropna(), "capacity")
    for k, v in cap_stats.items():
        if k not in ("feature",):
            log.info(f"    {k:<25}: {v}")

    stats_df.to_csv(results_dir / "statistical_summary.csv")
    log.info(f"  Saved: {results_dir / 'statistical_summary.csv'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Normal distribution fit check
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[2] Normal Distribution Fitting")
    normality_results = {}
    for feat in ["soh", "capacity", "voltage_mean", "temperature_mean", "Re_Ohm"]:
        if feat in df.columns:
            result = fit_normal_distribution(df[feat])
            normality_results[feat] = result
            log.info(
                f"  {feat:<25}: mu={result.get('mu', 'N/A'):.4f}  "
                f"sigma={result.get('sigma', 'N/A'):.4f}  "
                f"normal={result.get('is_approximately_normal', 'N/A')}"
            )

    with open(results_dir / "normality_tests.json", "w", encoding="utf-8") as fh:
        json.dump(normality_results, fh, indent=2, default=str)
    log.info(f"  Saved: {results_dir / 'normality_tests.json'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: IQR Anomaly Detection
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[3] IQR Anomaly Detection (Quantile-based)")
    for feat in ["soh", "voltage_mean", "temperature_mean", "Re_Ohm"]:
        if feat in df.columns:
            flags = iqr_anomaly_flags(df[feat], iqr_multiplier=cfg["anomaly"]["iqr_multiplier"])
            n_anom = (flags == "STATISTICAL ANOMALY").sum()
            log.info(f"  {feat:<25}: {n_anom} anomalies ({n_anom/len(df)*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Covariance Analysis
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[4] Covariance Matrix")
    cov_features = ["cycle_number", "capacity", "soh", "voltage_mean",
                    "temperature_mean", "Re_Ohm", "Rct_Ohm"]
    cov_features = [f for f in cov_features if f in df.columns]
    cov_matrix = compute_covariance_matrix(df, cov_features)
    cov_matrix.to_csv(results_dir / "covariance_matrix.csv")
    log.info(f"  Saved: {results_dir / 'covariance_matrix.csv'}")

    log.info("\n  Key covariances:")
    pairs = [("cycle_number", "soh"), ("cycle_number", "capacity"),
             ("Re_Ohm", "soh"), ("temperature_mean", "capacity")]
    for a, b in pairs:
        if a in cov_matrix.columns and b in cov_matrix.columns:
            log.info(f"    Cov({a}, {b}) = {cov_matrix.loc[a, b]:.6f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: Correlation Matrix
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[5] Correlation Matrix (Pearson)")
    corr_matrix = compute_correlation_matrix(df, cov_features, method="pearson")
    corr_matrix.to_csv(results_dir / "correlation_matrix.csv")
    log.info(f"  Saved: {results_dir / 'correlation_matrix.csv'}")

    log.info("\n  Key correlations:")
    for a, b in pairs:
        if a in corr_matrix.columns and b in corr_matrix.columns:
            log.info(f"    Corr({a}, {b}) = {corr_matrix.loc[a, b]:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Independence Testing
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[6] Statistical Independence Tests")
    independence_pairs = [
        ("cycle_number", "capacity"), ("cycle_number", "soh"),
        ("temperature_mean", "capacity"), ("Re_Ohm", "soh"),
        ("voltage_std", "soh"), ("cycle_number", "Re_Ohm"),
    ]
    independence_results = {}
    for xa, ya in independence_pairs:
        if xa in df.columns and ya in df.columns:
            result = independence_test(df[xa], df[ya])
            key = f"{xa}_vs_{ya}"
            independence_results[key] = result
            sig = "DEPENDENT" if (result.get("pearson_significant") or result.get("spearman_significant")) else "insufficient evidence"
            log.info(f"  {key:<40}: {sig}  (Pearson r={result.get('pearson_r', 'N/A'):.4f}, p={result.get('pearson_p', 'N/A'):.4f})")

    with open(results_dir / "independence_tests.json", "w", encoding="utf-8") as fh:
        json.dump(independence_results, fh, indent=2, default=str)
    log.info(f"  Saved: {results_dir / 'independence_tests.json'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7: Conditional Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[7] Conditional Statistics  P(X | degradation_label)")
    cond_stats: dict = {}
    for feat in ["capacity", "soh", "temperature_mean", "Re_Ohm", "voltage_std"]:
        if feat in df.columns:
            cond_stats[feat] = conditional_statistics(
                df, feat, "degradation_label",
                condition_values=["Healthy", "Early Degradation", "Degraded"],
            )
            log.info(f"\n  {feat}:")
            for cls, stats_d in cond_stats[feat].items():
                log.info(
                    f"    [{cls:<20}] mean={stats_d['mean']:.4f}  "
                    f"std={stats_d['std']:.4f}  "
                    f"median={stats_d['median']:.4f}"
                )

    with open(results_dir / "conditional_stats.json", "w", encoding="utf-8") as fh:
        json.dump(cond_stats, fh, indent=2, default=str)
    log.info(f"\n  Saved: {results_dir / 'conditional_stats.json'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: Visualisations
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n[8] Generating Visualisations ...")

    log.info("  01 SOH vs Cycle ...")
    plot_soh_vs_cycle(df, eol_threshold=eol_pct)

    log.info("  02 Capacity vs Cycle ...")
    plot_capacity_vs_cycle(df)

    log.info("  03 Feature Distributions ...")
    plot_feature_distributions(df, features=[
        "voltage_mean", "current_mean", "temperature_mean",
        "capacity", "soh", "Re_Ohm",
    ])

    log.info("  04 PDF Comparisons ...")
    for feat in ["soh", "capacity", "temperature_mean", "Re_Ohm", "voltage_std"]:
        if feat in df.columns:
            plot_pdf_comparison(df, feature=feat)

    log.info("  05 Covariance Heatmap ...")
    plot_covariance_heatmap(df, features=cov_features)

    log.info("  06 Correlation Heatmap ...")
    plot_correlation_heatmap(df, features=cov_features)

    log.info("  07 Box Plots ...")
    for feat in ["soh", "capacity", "voltage_mean", "temperature_mean"]:
        if feat in df.columns:
            plot_quantile_boxplots(df, feature=feat)

    log.info("  08 Anomaly Flags ...")
    for bid in ["B0005", "B0006"]:
        if bid in df["battery_id"].values:
            plot_anomaly_flags(df, battery_id=bid, feature="soh")

    log.info("  09 SOH Distribution ...")
    plot_soh_distribution(df)

    log.info("  10 Resistance vs Cycle ...")
    plot_resistance_vs_cycle(df)

    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("  EXPERIMENT 01: Statistical Analysis COMPLETE")
    log.info("=" * 60)
    log.info(f"  Results : {results_dir}")
    log.info(f"  Figures : {PROJECT_ROOT / 'reports' / 'figures'}")

    # Print a mini-report
    log.info("\n  KEY FINDINGS:")
    for a, b in [("cycle_number", "soh"), ("Re_Ohm", "soh")]:
        if a in corr_matrix.columns and b in corr_matrix.columns:
            r = corr_matrix.loc[a, b]
            direction = "negative (degradation signal)" if r < 0 else "positive"
            log.info(f"  Corr({a}, {b}) = {r:.4f} — {direction}")

    log.info("\n  SOH Statistics across fleet:")
    soh = df["soh"].dropna()
    log.info(f"    E[SOH]    = {soh.mean():.2f}%  (mean / expectation)")
    log.info(f"    Var(SOH)  = {soh.var():.2f}   (variance)")
    log.info(f"    std(SOH)  = {soh.std():.2f}%  (standard deviation)")
    log.info(f"    Q25(SOH)  = {soh.quantile(0.25):.2f}%  (25th quantile)")
    log.info(f"    Q50(SOH)  = {soh.quantile(0.50):.2f}%  (median)")
    log.info(f"    Q75(SOH)  = {soh.quantile(0.75):.2f}%  (75th quantile)")
    log.info(f"    IQR(SOH)  = {soh.quantile(0.75) - soh.quantile(0.25):.2f}%  (interquartile range)")


if __name__ == "__main__":
    main()
