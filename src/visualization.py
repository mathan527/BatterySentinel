"""
BatterySentinel — src/visualization.py
=======================================
Reusable, publication-quality plotting functions.

All plots saved to reports/figures/.
Supports both file output (PNG) and inline Streamlit (Plotly).

Plot catalogue
--------------
  plot_soh_vs_cycle()          — per-battery degradation curves
  plot_capacity_vs_cycle()     — raw capacity trajectories
  plot_feature_distributions() — histogram + KDE for each feature
  plot_pdf_comparison()        — healthy vs degraded density overlay
  plot_covariance_heatmap()    — covariance matrix
  plot_correlation_heatmap()   — correlation matrix
  plot_quantile_boxplots()     — per-class box plots
  plot_anomaly_flags()         — IQR anomalies highlighted on SOH curve
  plot_soh_distribution()      — SOH histogram by degradation class
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

# ── Style defaults ────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
LABEL_COLORS = {
    "Healthy": "#2ecc71",
    "Early Degradation": "#f39c12",
    "Degraded": "#e74c3c",
    "Unknown": "#95a5a6",
}
BATTERY_CMAP = cm.get_cmap("tab20")

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save figure to reports/figures/ and close."""
    out = FIG_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved figure: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SOH / Capacity curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_soh_vs_cycle(
    df: pd.DataFrame,
    highlight_batteries: list[str] | None = None,
    eol_threshold: float = 70.0,
    save: bool = True,
) -> plt.Figure:
    """
    Plot SOH vs cycle number for all batteries.

    Shows the degradation trajectory of each battery over its lifetime.
    The horizontal dashed line marks the configurable EOL threshold.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    batteries = sorted(df["battery_id"].unique())

    for i, bid in enumerate(batteries):
        bdf = df[df["battery_id"] == bid].sort_values("cycle_number")
        soh = bdf["soh"].dropna()
        cyc = bdf.loc[soh.index, "cycle_number"]
        color = BATTERY_CMAP(i / max(len(batteries), 1))
        alpha = 0.9 if (highlight_batteries and bid in highlight_batteries) else 0.45
        lw = 2.0 if (highlight_batteries and bid in highlight_batteries) else 0.8
        ax.plot(cyc, soh, color=color, linewidth=lw, alpha=alpha, label=bid)

    ax.axhline(eol_threshold, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"EOL threshold ({eol_threshold:.0f}%)\n[Configurable research parameter]")
    ax.axhspan(0, eol_threshold, alpha=0.05, color="#e74c3c")

    ax.set_xlabel("Discharge Cycle Number", fontsize=12)
    ax.set_ylabel("State of Health — SOH (%)", fontsize=12)
    ax.set_title("SOH vs Cycle Number — All Batteries\n"
                 "SOH = capacity / peak_capacity × 100%", fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", ncol=4, fontsize=7, framealpha=0.6)

    if save:
        _save(fig, "01_soh_vs_cycle.png")
    return fig


def plot_capacity_vs_cycle(
    df: pd.DataFrame,
    eol_threshold: float = 70.0,
    save: bool = True,
) -> plt.Figure:
    """Plot raw capacity (Ah) vs cycle number for all batteries."""
    fig, ax = plt.subplots(figsize=(14, 7))
    batteries = sorted(df["battery_id"].unique())

    for i, bid in enumerate(batteries):
        bdf = df[df["battery_id"] == bid].sort_values("cycle_number")
        cap = bdf["capacity"].dropna()
        cyc = bdf.loc[cap.index, "cycle_number"]
        color = BATTERY_CMAP(i / max(len(batteries), 1))
        ax.plot(cyc, cap, color=color, linewidth=0.9, alpha=0.6, label=bid)

    ax.set_xlabel("Discharge Cycle Number", fontsize=12)
    ax.set_ylabel("Discharge Capacity (Ah)", fontsize=12)
    ax.set_title("Capacity vs Cycle Number — All Batteries", fontsize=13)
    ax.legend(loc="upper right", ncol=4, fontsize=7, framealpha=0.6)

    if save:
        _save(fig, "02_capacity_vs_cycle.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Feature distributions
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str] | None = None,
    save: bool = True,
) -> plt.Figure:
    """
    Plot histograms + KDE for key continuous random variables.

    Unit 1: Treats each feature as a continuous random variable and
    visualises its empirical distribution.
    """
    if features is None:
        features = ["voltage_mean", "current_mean", "temperature_mean",
                    "capacity", "soh", "Re_Ohm"]
    features = [f for f in features if f in df.columns]
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = df[feat].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        ax.hist(data, bins=40, density=True, color="#3498db", alpha=0.5,
                edgecolor="white", linewidth=0.5, label="Histogram")
        # KDE overlay
        if len(data) >= 5:
            x = np.linspace(data.min(), data.max(), 200)
            kde = gaussian_kde(data, bw_method="scott")
            ax.plot(x, kde(x), color="#2c3e50", linewidth=2, label="KDE")
        ax.axvline(data.mean(), color="#e74c3c", linestyle="--", linewidth=1.2,
                   label=f"Mean={data.mean():.3f}")
        ax.axvline(data.median(), color="#f39c12", linestyle=":", linewidth=1.2,
                   label=f"Median={data.median():.3f}")
        ax.set_title(f"{feat}\nVar={data.var():.5f}  std={data.std():.4f}", fontsize=10)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Continuous Random Variable Distributions\n"
                 "Histogram + KDE | Mean (red) | Median (orange)", fontsize=13, y=1.01)
    fig.tight_layout()

    if save:
        _save(fig, "03_feature_distributions.png")
    return fig


def plot_pdf_comparison(
    df: pd.DataFrame,
    feature: str = "soh",
    class_col: str = "degradation_label",
    save: bool = True,
) -> plt.Figure:
    """
    Overlay PDFs for each degradation class for a given feature.

    Unit 1: Probability Density + Conditional Distributions
    Shows P(X | class) for Healthy, Early Degradation, Degraded.
    The SEPARATION between these curves is the signal used by the
    Bayesian classifier.
    """
    classes = ["Healthy", "Early Degradation", "Degraded"]
    classes = [c for c in classes if c in df[class_col].values]

    fig, ax = plt.subplots(figsize=(11, 6))

    for cls in classes:
        data = df[df[class_col] == cls][feature].dropna()
        if len(data) < 5:
            continue
        x = np.linspace(data.min() - 0.1 * data.std(),
                        data.max() + 0.1 * data.std(), 300)
        kde = gaussian_kde(data, bw_method="scott")
        pdf = kde(x)
        color = LABEL_COLORS.get(cls, "#7f8c8d")
        ax.plot(x, pdf, linewidth=2.5, color=color, label=cls)
        ax.fill_between(x, pdf, alpha=0.12, color=color)
        ax.axvline(data.mean(), color=color, linestyle="--", linewidth=1,
                   alpha=0.7)

    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel("Probability Density  f(x)", fontsize=12)
    ax.set_title(
        f"Probability Density: P({feature} | degradation class)\n"
        "Separation between classes = discriminative power of this feature",
        fontsize=12,
    )
    ax.legend(fontsize=10)

    if save:
        _save(fig, f"04_pdf_comparison_{feature}.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Covariance / Correlation
# ─────────────────────────────────────────────────────────────────────────────

def plot_covariance_heatmap(
    df: pd.DataFrame,
    features: list[str] | None = None,
    save: bool = True,
) -> plt.Figure:
    """
    Visualise the covariance matrix.

    Unit 1: Cov(X,Y) = E[(X-E[X])(Y-E[Y])]
    Captures direction and magnitude of joint variation.
    """
    if features is None:
        features = ["cycle_number", "capacity", "soh", "voltage_mean", "voltage_std",
                    "temperature_mean", "Re_Ohm", "Rct_Ohm", "discharge_duration"]
    features = [f for f in features if f in df.columns]
    cov = df[features].cov()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.zeros_like(cov, dtype=bool)
    np.fill_diagonal(mask, False)  # show diagonal
    sns.heatmap(
        cov, annot=True, fmt=".4f", cmap="RdBu_r", center=0,
        linewidths=0.5, ax=ax, annot_kws={"size": 8},
        mask=None,
    )
    ax.set_title("Sample Covariance Matrix\n"
                 "Cov(X,Y) = E[(X - E[X])(Y - E[Y])]\n"
                 "Blue = negative (inverse), Red = positive (joint increase)", fontsize=12)
    fig.tight_layout()

    if save:
        _save(fig, "05_covariance_matrix.png")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str] | None = None,
    method: str = "pearson",
    save: bool = True,
) -> plt.Figure:
    """
    Visualise the correlation matrix (Pearson or Spearman).

    Unit 1: Corr(X,Y) = Cov(X,Y) / (std(X) * std(Y))  in [-1, +1]
    Normalised — comparable across different feature pairs.
    """
    if features is None:
        features = ["cycle_number", "capacity", "soh", "voltage_mean", "voltage_std",
                    "temperature_mean", "Re_Ohm", "Rct_Ohm", "discharge_duration"]
    features = [f for f in features if f in df.columns]
    corr = df[features].corr(method=method)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 9},
    )
    ax.set_title(
        f"{method.capitalize()} Correlation Matrix\n"
        "Corr(X,Y) = Cov(X,Y) / (std(X) * std(Y))  in [-1, +1]\n"
        "-1 = perfect inverse  |  0 = no linear relation  |  +1 = perfect positive",
        fontsize=12,
    )
    fig.tight_layout()

    if save:
        _save(fig, f"06_correlation_{method}.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Quantile / Box plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_quantile_boxplots(
    df: pd.DataFrame,
    feature: str = "soh",
    group_col: str = "degradation_label",
    save: bool = True,
) -> plt.Figure:
    """
    Box plots showing Q1, median, Q3, IQR, and outliers per class.

    Unit 1: Quantiles
    Box shows: [Q1, Q3]  (IQR = Q3 - Q1)
    Whiskers: Q1 - 1.5*IQR  to  Q3 + 1.5*IQR
    Points beyond whiskers: statistical outliers (Tukey fences)
    """
    order = ["Healthy", "Early Degradation", "Degraded"]
    order = [o for o in order if o in df[group_col].values]
    colors = [LABEL_COLORS.get(o, "#95a5a6") for o in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_group = [df[df[group_col] == cls][feature].dropna() for cls in order]
    bp = ax.boxplot(data_by_group, labels=order, patch_artist=True,
                    notch=False, vert=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(feature, fontsize=12)
    ax.set_title(
        f"Quantile Analysis: {feature} by Degradation Class\n"
        "Box = [Q1, Q3]  |  IQR = Q3 - Q1  |  Whiskers = Tukey 1.5*IQR fences\n"
        "Points beyond whiskers = Statistical Anomalies",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.4)

    if save:
        _save(fig, f"07_boxplot_{feature}.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly flags
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomaly_flags(
    df: pd.DataFrame,
    battery_id: str,
    feature: str = "soh",
    save: bool = True,
) -> plt.Figure:
    """
    Plot feature over cycles for one battery, highlighting IQR anomalies.

    Unit 1: Quantile-based anomaly detection
    """
    bdf = df[df["battery_id"] == battery_id].sort_values("cycle_number")
    data = bdf[feature].dropna()
    cyc = bdf.loc[data.index, "cycle_number"]

    q1, q3 = float(data.quantile(0.25)), float(data.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    is_anomaly = (data < lower) | (data > upper)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(cyc, data, color="#2980b9", linewidth=1.5, label=feature)
    ax.scatter(cyc[is_anomaly], data[is_anomaly], color="#e74c3c", zorder=5,
               s=50, label=f"IQR anomaly ({is_anomaly.sum()})")
    ax.axhline(upper, color="#f39c12", linestyle="--", linewidth=1, label=f"Upper fence={upper:.2f}")
    ax.axhline(lower, color="#f39c12", linestyle="--", linewidth=1, label=f"Lower fence={lower:.2f}")
    ax.axhspan(lower, upper, alpha=0.06, color="#27ae60", label="IQR band [Q1-1.5*IQR, Q3+1.5*IQR]")

    ax.set_xlabel("Cycle Number", fontsize=11)
    ax.set_ylabel(feature, fontsize=11)
    ax.set_title(f"Battery {battery_id} — {feature} with IQR Anomaly Detection\n"
                 "Red points = statistical anomalies (Tukey 1.5*IQR fence)\n"
                 "NOTE: anomaly != failure; context required for interpretation",
                 fontsize=11)
    ax.legend(fontsize=9)

    if save:
        _save(fig, f"08_anomaly_{battery_id}_{feature}.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SOH distribution overview
# ─────────────────────────────────────────────────────────────────────────────

def plot_soh_distribution(
    df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Stacked histogram showing SOH distribution coloured by degradation class.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked histogram
    ax = axes[0]
    classes = ["Healthy", "Early Degradation", "Degraded"]
    for cls in classes:
        data = df[df["degradation_label"] == cls]["soh"].dropna()
        ax.hist(data, bins=30, alpha=0.6, color=LABEL_COLORS.get(cls, "#95a5a6"),
                label=f"{cls} (n={len(data)})", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("SOH (%)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("SOH Distribution by Degradation Class", fontsize=12)
    ax.legend(fontsize=9)

    # Right: PDF overlay per class
    ax2 = axes[1]
    for cls in classes:
        data = df[df["degradation_label"] == cls]["soh"].dropna()
        if len(data) < 5:
            continue
        x = np.linspace(max(0, data.min() - 5), min(105, data.max() + 5), 200)
        kde = gaussian_kde(data, bw_method="scott")
        color = LABEL_COLORS.get(cls, "#95a5a6")
        ax2.plot(x, kde(x), linewidth=2.5, color=color, label=cls)
        ax2.fill_between(x, kde(x), alpha=0.1, color=color)
    ax2.set_xlabel("SOH (%)", fontsize=12)
    ax2.set_ylabel("Probability Density  f(SOH)", fontsize=12)
    ax2.set_title("PDF of SOH per Class\n"
                  "f(SOH | class) — Bayesian likelihood distributions", fontsize=11)
    ax2.legend(fontsize=9)

    fig.suptitle("State of Health (SOH) Distribution Analysis\n"
                 "SOH = Capacity / Peak_Capacity x 100%", fontsize=13)
    fig.tight_layout()

    if save:
        _save(fig, "09_soh_distribution.png")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Resistance evolution
# ─────────────────────────────────────────────────────────────────────────────

def plot_resistance_vs_cycle(
    df: pd.DataFrame,
    batteries: list[str] | None = None,
    save: bool = True,
) -> plt.Figure:
    """Plot Re (electrolyte resistance) vs cycle for selected batteries."""
    if batteries is None:
        # Pick batteries with the most Re data
        re_counts = df.groupby("battery_id")["Re_Ohm"].count()
        batteries = re_counts.nlargest(8).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, feat, title in zip(axes, ["Re_Ohm", "Rct_Ohm"],
                                ["Re (Electrolyte Resistance)", "Rct (Charge Transfer Resistance)"]):
        for i, bid in enumerate(batteries):
            bdf = df[df["battery_id"] == bid].sort_values("cycle_number")
            rdata = bdf[feat].dropna()
            cyc = bdf.loc[rdata.index, "cycle_number"]
            color = BATTERY_CMAP(i / max(len(batteries), 1))
            ax.plot(cyc, rdata, color=color, linewidth=1.3, alpha=0.8, label=bid)
        ax.set_xlabel("Cycle Number", fontsize=11)
        ax.set_ylabel(f"{feat} (Ohm)", fontsize=11)
        ax.set_title(f"{title} vs Cycle\nRising = electrolyte aging signal", fontsize=11)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Internal Resistance Evolution over Lifetime\n"
                 "Increasing resistance is a key degradation indicator", fontsize=13)
    fig.tight_layout()

    if save:
        _save(fig, "10_resistance_vs_cycle.png")
    return fig
