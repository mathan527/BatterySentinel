"""
BatterySentinel — src/statistics_engine.py
==========================================
Comprehensive descriptive statistics module.

Implements all Unit 1 statistical concepts:
  - Mean (E[X])
  - Median
  - Variance (Var(X) = E[(X - mu)^2])
  - Standard Deviation
  - Min, Max
  - Quantiles (Q1, Q2, Q3, IQR)
  - Skewness
  - Kurtosis
  - Covariance matrix
  - Correlation matrix
  - Independence testing (Pearson + Spearman)
  - Conditional statistics

All formulae are documented with:
  • Mathematical definition
  • Python implementation
  • Physical interpretation for battery data

Public API
----------
    describe_feature(series)         -> dict
    describe_all_features(df, cols)  -> pd.DataFrame
    compute_covariance_matrix(df)    -> pd.DataFrame
    compute_correlation_matrix(df)   -> pd.DataFrame
    compute_quantiles(series, q)     -> dict
    iqr_anomaly_flags(series)        -> pd.Series
    conditional_statistics(df, ...)  -> dict
    independence_test(x, y)          -> dict
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Single-feature descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────

def describe_feature(series: pd.Series, feature_name: str = "") -> dict[str, Any]:
    """
    Compute comprehensive descriptive statistics for a single continuous
    random variable (feature).

    Unit 1 Concepts Demonstrated
    ----------------------------
    Mean:
        mu = E[X] = (1/n) * sum(x_i)
        Interpretation: The expected value of the feature.
        For SOH: average state of health across all observed cycles.

    Variance:
        Var(X) = E[(X - mu)^2] = (1/(n-1)) * sum((x_i - mu)^2)
        Interpretation: Average squared deviation from the mean.
        For SOH: how much health varies across batteries/cycles.

    Standard Deviation:
        std(X) = sqrt(Var(X))
        Same units as X — more interpretable than variance.

    Quantiles:
        Q_p = value below which p% of observations fall.
        Q1 = 25th percentile, Q2 = 50th (median), Q3 = 75th.
        IQR = Q3 - Q1 — the spread of the middle 50% of data.

    Skewness:
        g1 = E[(X-mu)^3] / std(X)^3
        Positive: right tail; Negative: left tail.
        For capacity: often left-skewed as batteries accumulate degradation.

    Kurtosis (excess):
        g2 = E[(X-mu)^4] / std(X)^4 - 3
        > 0: heavier tails than Gaussian (leptokurtic).
        = 0: Gaussian (mesokurtic).
        < 0: lighter tails (platykurtic).

    Parameters
    ----------
    series       : pd.Series of numeric values
    feature_name : str label for the result dict

    Returns
    -------
    dict with all computed statistics.
    """
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return {"feature": feature_name, "n": 0, "error": "No valid data"}

    x = clean.values.astype(float)
    mu = float(np.mean(x))
    sigma2 = float(np.var(x, ddof=1)) if n > 1 else 0.0
    sigma = float(np.sqrt(sigma2))

    q = np.quantile(x, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    q1, q3 = float(q[2]), float(q[4])
    iqr = q3 - q1

    skewness = float(stats.skew(x)) if n >= 3 else np.nan
    kurt = float(stats.kurtosis(x, fisher=True)) if n >= 4 else np.nan  # excess kurtosis

    # Tukey fences (IQR-based anomaly bounds)
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    n_outliers = int(((x < lower_fence) | (x > upper_fence)).sum())

    return {
        "feature": feature_name,
        "n": n,
        # Central tendency
        "mean": round(mu, 6),
        "median": round(float(q[3]), 6),
        # Dispersion
        "variance": round(sigma2, 6),
        "std": round(sigma, 6),
        "min": round(float(x.min()), 6),
        "max": round(float(x.max()), 6),
        "range": round(float(x.max() - x.min()), 6),
        # Quantiles
        "q05": round(float(q[0]), 6),
        "q10": round(float(q[1]), 6),
        "q25": round(q1, 6),
        "q50": round(float(q[3]), 6),
        "q75": round(q3, 6),
        "q90": round(float(q[5]), 6),
        "q95": round(float(q[6]), 6),
        "iqr": round(iqr, 6),
        # Shape
        "skewness": round(skewness, 6) if not np.isnan(skewness) else None,
        "kurtosis_excess": round(kurt, 6) if not np.isnan(kurt) else None,
        # Anomaly bounds (Tukey fences)
        "lower_fence": round(lower_fence, 6),
        "upper_fence": round(upper_fence, 6),
        "n_outliers": n_outliers,
        "outlier_pct": round(n_outliers / n * 100, 2),
    }


def describe_all_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Apply describe_feature() to multiple columns.

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list of column names to analyse (None = all numeric)

    Returns
    -------
    pd.DataFrame — one row per feature, columns = statistic names.
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    records = [describe_feature(df[col], col) for col in columns if col in df.columns]
    return pd.DataFrame(records).set_index("feature")


# ─────────────────────────────────────────────────────────────────────────────
# Quantile analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_quantiles(
    series: pd.Series,
    quantiles: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute arbitrary quantiles for a series.

    Formula:
        Q_p = F^{-1}(p) = inf{x : F(x) >= p}
    where F is the empirical CDF.

    A quantile Q_p is the value below which a fraction p of the
    observations fall.

    Parameters
    ----------
    series    : pd.Series
    quantiles : list of probabilities in [0, 1]
                Default: [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    Returns
    -------
    dict : {f"q{int(p*100)}": value}
    """
    if quantiles is None:
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    clean = series.dropna()
    if clean.empty:
        return {}
    result = {}
    for p in quantiles:
        key = f"q{int(round(p * 100))}"
        result[key] = float(np.quantile(clean.values, p))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# IQR-based anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

def iqr_anomaly_flags(
    series: pd.Series,
    iqr_multiplier: float = 1.5,
    label_col: bool = True,
) -> pd.Series:
    """
    Flag statistical anomalies using the Tukey IQR fence method.

    Formula:
        IQR = Q3 - Q1
        Lower Bound = Q1 - k * IQR
        Upper Bound = Q3 + k * IQR

    where k = iqr_multiplier (default 1.5, Tukey's classic choice).

    Classification:
        NORMAL            — within the fences
        STATISTICAL ANOMALY — outside the fences

    Physical meaning for battery data:
        An IQR anomaly does NOT automatically mean battery failure.
        It means the observation is unusually far from the central
        tendency compared to the bulk of observations.
        Context is required to interpret whether the anomaly is a:
          (a) Measurement artifact
          (b) Degradation signal
          (c) Operating condition change

    Parameters
    ----------
    series         : pd.Series of numeric values
    iqr_multiplier : float — k in the fence formula (default 1.5)
    label_col      : bool  — return string labels (True) or bool (False)

    Returns
    -------
    pd.Series of anomaly flags (same index as input).
    """
    clean = series.dropna()
    if clean.empty:
        return pd.Series("UNKNOWN", index=series.index)

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    if label_col:
        flags = pd.Series("NORMAL", index=series.index)
        flags[(series < lower) | (series > upper)] = "STATISTICAL ANOMALY"
        flags[series.isna()] = "MISSING"
        return flags
    else:
        return (series < lower) | (series > upper)


# ─────────────────────────────────────────────────────────────────────────────
# Covariance and correlation
# ─────────────────────────────────────────────────────────────────────────────

def compute_covariance_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the sample covariance matrix.

    Formula:
        Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
                  = (1/(n-1)) * sum((x_i - mu_x)(y_i - mu_y))

    Interpretation:
        Cov(X, Y) > 0 : X and Y tend to increase together
        Cov(X, Y) < 0 : when X increases, Y tends to decrease
        Cov(X, Y) = 0 : no linear relationship (but may be nonlinear)

    Note: Covariance has the units of X * Y, which makes comparing
    across different feature pairs difficult.  Use correlation for
    normalised comparison.

    Physical meaning for battery data:
        Cov(cycle, capacity) < 0 :
            As cycle count increases, capacity tends to decrease.
            This is the fundamental degradation signal.
        Cov(temperature, Re) > 0 :
            Higher temperature may coincide with higher resistance
            (or vice versa depending on test conditions).

    Parameters
    ----------
    df      : pd.DataFrame
    columns : list of numeric columns (None = all numeric)

    Returns
    -------
    pd.DataFrame — symmetric covariance matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    cols_present = [c for c in columns if c in df.columns]
    return df[cols_present].cov()


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute the correlation matrix.

    Formula (Pearson):
        Corr(X, Y) = Cov(X, Y) / (std(X) * std(Y))
                   = E[(X - mu_x)(Y - mu_y)] / (sigma_x * sigma_y)

    Range: [-1, +1]
        +1 : perfect positive linear relationship
         0 : no linear relationship
        -1 : perfect negative linear relationship

    Difference from covariance:
        Covariance captures the DIRECTION and MAGNITUDE of joint variation,
        but its scale depends on the units of the variables.
        Correlation is DIMENSIONLESS — it captures the STRENGTH and
        DIRECTION, normalised to [-1, 1].

    Parameters
    ----------
    df     : pd.DataFrame
    columns: list of numeric columns (None = all numeric)
    method : "pearson" | "spearman" | "kendall"

    Returns
    -------
    pd.DataFrame — symmetric correlation matrix
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
    cols_present = [c for c in columns if c in df.columns]
    return df[cols_present].corr(method=method)


# ─────────────────────────────────────────────────────────────────────────────
# Independence testing
# ─────────────────────────────────────────────────────────────────────────────

def independence_test(
    x: pd.Series,
    y: pd.Series,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Test for statistical independence between two continuous variables.

    Tests performed:
    1. Pearson correlation + p-value
       H0: no linear correlation (rho = 0)
    2. Spearman rank correlation + p-value
       H0: no monotonic relationship

    Note:
        Failing to reject H0 does NOT prove independence.
        It means there is insufficient evidence to conclude dependence
        under the linear/monotonic model.
        Nonlinear dependencies may exist even when linear correlation is zero.

    Important distinction:
        Correlation = linear/monotonic DEPENDENCE
        Independence = no relationship of ANY kind
        These are related but not equivalent.

    Parameters
    ----------
    x, y  : pd.Series of float
    alpha : float — significance level (default 0.05)

    Returns
    -------
    dict with:
        pearson_r, pearson_p, pearson_significant
        spearman_r, spearman_p, spearman_significant
        interpretation
    """
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 5:
        return {"error": "Insufficient data for independence test (n < 5)"}

    xv, yv = clean["x"].values, clean["y"].values

    p_r, p_p = stats.pearsonr(xv, yv)
    s_r, s_p = stats.spearmanr(xv, yv)

    pearson_sig = bool(p_p < alpha)
    spearman_sig = bool(s_p < alpha)

    if pearson_sig or spearman_sig:
        interp = (
            f"Evidence of DEPENDENCE (Pearson p={p_p:.4f}, Spearman p={s_p:.4f}). "
            "Note: correlation implies linear/monotonic dependence, not causation."
        )
    else:
        interp = (
            f"Insufficient evidence to reject independence at alpha={alpha}. "
            f"(Pearson p={p_p:.4f}, Spearman p={s_p:.4f}). "
            "Nonlinear dependencies may still exist."
        )

    return {
        "n": int(len(clean)),
        "pearson_r": round(float(p_r), 6),
        "pearson_p": round(float(p_p), 6),
        "pearson_significant": pearson_sig,
        "spearman_r": round(float(s_r), 6),
        "spearman_p": round(float(s_p), 6),
        "spearman_significant": spearman_sig,
        "alpha": alpha,
        "interpretation": interp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Conditional statistics
# ─────────────────────────────────────────────────────────────────────────────

def conditional_statistics(
    df: pd.DataFrame,
    feature: str,
    condition_col: str,
    condition_values: list[Any] | None = None,
) -> dict[str, dict]:
    """
    Compute statistics of `feature` conditional on groups of `condition_col`.

    Example:
        conditional_statistics(df, "capacity", "degradation_label")
        => P(capacity | label == "Healthy")
        => P(capacity | label == "Degraded")

    This allows us to compare:
        - The distribution of capacity for healthy batteries vs degraded ones.
        - Whether the distribution shifts significantly — the foundation of
          the Bayesian likelihood P(X | class).

    Unit 1 concept: Conditional Probability
        P(X in A | Y = y) = P(X in A, Y = y) / P(Y = y)

    Parameters
    ----------
    df               : pd.DataFrame
    feature          : column name of the feature to analyse
    condition_col    : column name to condition on
    condition_values : which values of condition_col to use (None = all)

    Returns
    -------
    dict : {condition_value: describe_feature_result}
    """
    if condition_values is None:
        condition_values = sorted(df[condition_col].dropna().unique().tolist())

    results: dict[str, dict] = {}
    for val in condition_values:
        subset = df[df[condition_col] == val][feature]
        results[str(val)] = describe_feature(subset, f"{feature} | {condition_col}={val}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Expectation with weighting
# ─────────────────────────────────────────────────────────────────────────────

def compute_expectation(
    series: pd.Series,
    weights: pd.Series | None = None,
) -> float:
    """
    Compute the expectation (mean) of a random variable, optionally weighted.

    Formula (unweighted):
        E[X] = (1/n) * sum(x_i)

    Formula (weighted):
        E[X] = sum(w_i * x_i) / sum(w_i)

    Parameters
    ----------
    series  : pd.Series of values
    weights : pd.Series of non-negative weights (None = uniform)

    Returns
    -------
    float : the (weighted) expectation
    """
    clean = series.dropna()
    if clean.empty:
        return np.nan
    if weights is None:
        return float(clean.mean())
    w = weights.loc[clean.index].fillna(0)
    total_w = w.sum()
    if total_w == 0:
        return float(clean.mean())
    return float((clean * w).sum() / total_w)


# ─────────────────────────────────────────────────────────────────────────────
# Complete statistical summary
# ─────────────────────────────────────────────────────────────────────────────

def full_statistical_summary(df: pd.DataFrame, cfg: dict) -> dict[str, Any]:
    """
    Run the complete Unit 1 statistical analysis on the cycle-level dataset.

    Returns a structured dict with:
        - descriptive stats for all key features
        - covariance matrix
        - correlation matrix
        - independence tests between key pairs
        - conditional stats by degradation label

    Parameters
    ----------
    df  : cycle-level DataFrame (from preprocessing)
    cfg : config dict

    Returns
    -------
    dict of results
    """
    key_features = [
        "voltage_mean", "voltage_std",
        "current_mean", "current_std",
        "temperature_mean", "temperature_std",
        "capacity", "soh",
        "Re_Ohm", "Rct_Ohm",
        "cycle_number", "discharge_duration",
    ]
    key_features = [f for f in key_features if f in df.columns]

    log.info("  Computing descriptive statistics ...")
    desc = describe_all_features(df, key_features)

    log.info("  Computing covariance matrix ...")
    cov_matrix = compute_covariance_matrix(df, key_features)

    log.info("  Computing correlation matrix ...")
    corr_matrix = compute_correlation_matrix(df, key_features)

    log.info("  Running independence tests ...")
    independence_pairs = [
        ("cycle_number", "capacity"),
        ("cycle_number", "soh"),
        ("temperature_mean", "capacity"),
        ("Re_Ohm", "soh"),
        ("voltage_std", "soh"),
        ("cycle_number", "Re_Ohm"),
    ]
    independence_results = {}
    for xa, ya in independence_pairs:
        if xa in df.columns and ya in df.columns:
            independence_results[f"{xa}_vs_{ya}"] = independence_test(df[xa], df[ya])

    log.info("  Computing conditional statistics ...")
    conditional_cap = conditional_statistics(df, "capacity", "degradation_label")
    conditional_soh = conditional_statistics(df, "soh", "degradation_label")
    conditional_temp = conditional_statistics(df, "temperature_mean", "degradation_label")

    return {
        "descriptive_stats": desc.reset_index().to_dict(orient="records"),
        "covariance_matrix": cov_matrix.round(8).to_dict(),
        "correlation_matrix": corr_matrix.round(6).to_dict(),
        "independence_tests": independence_results,
        "conditional_capacity_by_label": conditional_cap,
        "conditional_soh_by_label": conditional_soh,
        "conditional_temperature_by_label": conditional_temp,
    }
