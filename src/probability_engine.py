"""
BatterySentinel — src/probability_engine.py
============================================
Continuous Random Variable analysis, KDE/PDF estimation,
CDF computation, and probability-density visualizations.

Unit 1 Concepts Covered
------------------------
  • Continuous Random Variables
  • Probability Density Function (PDF)
  • Kernel Density Estimation (KDE)
  • Cumulative Distribution Function (CDF)
  • Quantiles
  • Expectation  E[X]
  • Variance     Var(X)
  • Standard Deviation
  • Skewness / Kurtosis
  • Conditional distributions  P(X | class)

Public API
----------
    fit_kde(series, bandwidth)       -> (x_grid, pdf_values)
    compute_cdf(series)              -> (x_sorted, cdf_values)
    compare_class_distributions(...) -> dict
    plot_pdf_comparison(...)         -> plt.Figure
    plot_cdf_comparison(...)         -> plt.Figure
    anomaly_score(value, series)     -> float
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# KDE / PDF
# ─────────────────────────────────────────────────────────────────────────────

def fit_kde(
    series: pd.Series,
    n_grid: int = 200,
    bandwidth: str | float = "scott",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Probability Density Function (PDF) using Kernel Density
    Estimation (KDE).

    Unit 1 Concept: Probability Density
    ------------------------------------
    For a continuous random variable X, the PDF f(x) satisfies:

        P(a <= X <= b) = integral from a to b of f(x) dx

    The PDF is NOT a probability itself (it can exceed 1), but the
    AREA UNDER THE CURVE between two points gives the probability.

    KDE Formula:
        f_hat(x) = (1 / (n * h)) * sum_i K((x - x_i) / h)

    where:
        n  = number of observations
        h  = bandwidth (smoothing parameter)
        K  = kernel function (default: Gaussian)

    The Gaussian kernel is:
        K(u) = (1 / sqrt(2*pi)) * exp(-u^2 / 2)

    Parameters
    ----------
    series    : pd.Series of numeric values
    n_grid    : number of points in the evaluation grid
    bandwidth : "scott" | "silverman" | float

    Returns
    -------
    (x_grid, pdf_values) : numpy arrays of shape (n_grid,)
    """
    clean = series.dropna().values.astype(float)
    if len(clean) < 5:
        log.warning(f"KDE: too few samples ({len(clean)}) for reliable estimate.")
        x_grid = np.linspace(clean.min(), clean.max(), n_grid) if len(clean) > 0 else np.array([])
        return x_grid, np.full(n_grid, np.nan)

    kde = gaussian_kde(clean, bw_method=bandwidth)
    x_min, x_max = float(clean.min()), float(clean.max())
    margin = (x_max - x_min) * 0.1
    x_grid = np.linspace(x_min - margin, x_max + margin, n_grid)
    pdf_values = kde(x_grid)
    return x_grid, pdf_values


def compute_cdf(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical Cumulative Distribution Function (CDF).

    Formula:
        F_n(x) = (1/n) * sum_i I(x_i <= x)

    where I(.) is the indicator function.

    Interpretation:
        F(x) = P(X <= x)

    The CDF is always:
        - Non-decreasing
        - F(-inf) = 0
        - F(+inf) = 1

    For battery SOH:
        F(80) = 0.45 means 45% of cycles have SOH <= 80%.

    Returns
    -------
    (x_sorted, cdf_values) : x values and corresponding CDF probabilities.
    """
    clean = series.dropna().values.astype(float)
    x_sorted = np.sort(clean)
    n = len(x_sorted)
    cdf_values = np.arange(1, n + 1) / n
    return x_sorted, cdf_values


# ─────────────────────────────────────────────────────────────────────────────
# Class-conditional distribution comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_class_distributions(
    df: pd.DataFrame,
    feature: str,
    class_col: str = "degradation_label",
    classes: list[str] | None = None,
) -> dict[str, dict]:
    """
    Compare the distribution of `feature` across degradation classes.

    Unit 1 Concept: Conditional Distributions
    ------------------------------------------
    P(X | Y = class) is the conditional distribution of feature X
    given that the battery is in a particular degradation state.

    This is the FOUNDATION of the Bayesian likelihood term:
        P(Evidence | Degradation) vs P(Evidence | Healthy)

    For each class we estimate:
        - KDE / PDF
        - CDF
        - Summary statistics (mean, std, quantiles)

    Parameters
    ----------
    df        : cycle-level DataFrame
    feature   : column name of the continuous feature
    class_col : column name of the class label
    classes   : which classes to include (None = all)

    Returns
    -------
    dict: {class_name: {"kde": (x, pdf), "cdf": (x, cdf), "stats": dict}}
    """
    if classes is None:
        classes = sorted(df[class_col].dropna().unique().tolist())

    results: dict[str, dict] = {}
    for cls in classes:
        subset = df[df[class_col] == cls][feature].dropna()
        if subset.empty:
            continue
        x_kde, pdf = fit_kde(subset)
        x_cdf, cdf = compute_cdf(subset)
        results[cls] = {
            "kde": {"x": x_kde.tolist(), "pdf": pdf.tolist()},
            "cdf": {"x": x_cdf.tolist(), "cdf": x_cdf.tolist()},
            "stats": {
                "n": int(len(subset)),
                "mean": float(subset.mean()),
                "std": float(subset.std()),
                "median": float(subset.median()),
                "q25": float(subset.quantile(0.25)),
                "q75": float(subset.quantile(0.75)),
                "min": float(subset.min()),
                "max": float(subset.max()),
            },
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly scoring
# ─────────────────────────────────────────────────────────────────────────────

def anomaly_score(value: float, series: pd.Series) -> dict[str, float]:
    """
    Compute a statistical anomaly score for a single observed value
    relative to a reference distribution.

    Two scores are computed:

    1. Z-score:
           z = (x - mu) / sigma
       Interpretation: how many standard deviations x is from the mean.
       |z| > 2 is commonly considered unusual;
       |z| > 3 is considered a strong outlier.

    2. Percentile (quantile):
           p = F(x) = empirical probability that X <= x
       A value at the 99th percentile is more extreme than 99% of
       the reference distribution.

    Parameters
    ----------
    value  : float — the observed value to score
    series : pd.Series — the reference distribution

    Returns
    -------
    dict with z_score, percentile, is_outlier
    """
    clean = series.dropna()
    if clean.empty:
        return {"z_score": np.nan, "percentile": np.nan, "is_outlier": False}

    mu = float(clean.mean())
    sigma = float(clean.std())
    z = (value - mu) / sigma if sigma > 0 else 0.0
    pct = float(stats.percentileofscore(clean, value, kind="rank")) / 100.0

    # Tukey IQR fence
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    is_outlier = (value < q1 - 1.5 * iqr) or (value > q3 + 1.5 * iqr)

    return {
        "z_score": round(z, 4),
        "percentile": round(pct, 4),
        "is_outlier_iqr": bool(is_outlier),
        "is_outlier_zscore": bool(abs(z) > 2.5),
        "reference_mean": round(mu, 4),
        "reference_std": round(sigma, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Distribution fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_normal_distribution(series: pd.Series) -> dict[str, float]:
    """
    Fit a Gaussian (normal) distribution to a series.

    The Normal distribution is the most important distribution in
    statistics, defined by:

        f(x; mu, sigma^2) = (1 / (sigma * sqrt(2*pi))) * exp(-(x-mu)^2 / (2*sigma^2))

    Parameters:
        mu    = E[X] = sample mean
        sigma = sqrt(Var(X)) = sample standard deviation

    Goodness-of-fit: Shapiro-Wilk test
        H0: the data comes from a normal distribution
        p < 0.05 suggests non-normality

    Returns
    -------
    dict: {mu, sigma, shapiro_stat, shapiro_p, is_approximately_normal}
    """
    clean = series.dropna().values.astype(float)
    if len(clean) < 5:
        return {"error": "Too few samples"}

    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))

    if len(clean) <= 5000:
        sw_stat, sw_p = stats.shapiro(clean[:5000])
    else:
        sw_stat, sw_p = float("nan"), float("nan")

    return {
        "mu": round(mu, 6),
        "sigma": round(sigma, 6),
        "shapiro_stat": round(float(sw_stat), 6) if not np.isnan(sw_stat) else None,
        "shapiro_p": round(float(sw_p), 6) if not np.isnan(sw_p) else None,
        "is_approximately_normal": bool(sw_p > 0.05) if not np.isnan(sw_p) else None,
        "note": (
            "Shapiro-Wilk p > 0.05: insufficient evidence to reject normality. "
            "p <= 0.05: evidence against normality."
        ),
    }
