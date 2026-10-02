"""
BatterySentinel — src/feature_engineering.py
=============================================
Derives additional features from the cycle-level dataset produced by
preprocessing.py.

All derived features are computed from existing columns only — no data
is invented or hallucinated.  Each feature is documented with:
  • Mathematical formula
  • Physical interpretation
  • Source columns
  • Validity conditions

Features added by this module
------------------------------
  voltage_range          : voltage_max - voltage_min  (V)
  current_range          : current_max - current_min  (A)
  temperature_range      : temperature_max - temperature_min  (°C)
  voltage_cv             : voltage_std / |voltage_mean|  (coefficient of variation)
  current_cv             : current_std / |current_mean|  (coefficient of variation)
  temperature_cv         : temperature_std / |temperature_mean|
  capacity_fade          : initial_capacity - capacity  (cumulative Ah lost)
  capacity_fade_rate     : (Δcapacity) / (Δcycle)  per-cycle fade rate
  soh_change             : Δ SOH between consecutive cycles  (%)
  log_Re                 : log(Re_Ohm + ε)  — stabilises skewed distribution
  log_Rct                : log(Rct_Ohm + ε)
  cycle_norm             : cycle_number / max_cycle  per battery  (0–1)
  degradation_bin        : ordinal label  0=Healthy, 1=Early Degradation, 2=Degraded
  risk_features          : composite risk indicator (Re change + soh decline)

Public API
----------
    engineer_features(df, cfg) -> pd.DataFrame
    get_feature_columns()      -> dict[str, list[str]]
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering functions
# ─────────────────────────────────────────────────────────────────────────────

def _add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add range features: max - min for voltage, current, temperature.

    Formula:  X_range = X_max - X_min

    Physical meaning:
      - voltage_range: spread of voltage during a discharge cycle.
        A wider range can indicate higher internal resistance or
        deeper discharge events.
      - temperature_range: thermal excursion within a cycle.
        Larger ranges may indicate faster degradation.
    """
    for col in ("voltage", "current", "temperature"):
        max_col = f"{col}_max"
        min_col = f"{col}_min"
        if max_col in df.columns and min_col in df.columns:
            df[f"{col}_range"] = df[max_col] - df[min_col]
    return df


def _add_coefficient_of_variation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add coefficient of variation (CV) for voltage, current, temperature.

    Formula:  CV(X) = std(X) / |mean(X)|

    Physical meaning:
      - CV is a normalised measure of dispersion.
      - Rising voltage CV across cycles may indicate increasing
        internal resistance causing voltage to vary more widely.
      - CV is dimensionless, so it can be compared across features.

    Note: CV is undefined when mean = 0; we return NaN in that case.
    """
    for col in ("voltage", "current", "temperature"):
        std_col = f"{col}_std"
        mean_col = f"{col}_mean"
        if std_col in df.columns and mean_col in df.columns:
            mean_abs = df[mean_col].abs()
            df[f"{col}_cv"] = np.where(
                mean_abs > 1e-9, df[std_col] / mean_abs, np.nan
            )
    return df


def _add_capacity_fade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add capacity fade and fade rate per battery.

    capacity_fade:
        Formula:  fade(t) = C_max - C(t)
        The cumulative capacity loss from the battery's peak state.
        Physically: how many Ah of capacity has been permanently lost.

    capacity_fade_rate:
        Formula:  dC/dn  (finite difference, Ah per cycle)
        Negative values mean capacity is decreasing (normal aging).
        Positive values may indicate formation cycling or measurement noise.
    """
    result_dfs: list[pd.DataFrame] = []
    for bid, group in df.groupby("battery_id"):
        g = group.sort_values("cycle_number").copy()
        c_max = g["capacity"].max()
        g["capacity_fade"] = c_max - g["capacity"]
        g["capacity_fade_rate"] = g["capacity"].diff() / g["cycle_number"].diff()
        result_dfs.append(g)
    return pd.concat(result_dfs, ignore_index=True)


def _add_soh_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SOH change between consecutive discharge cycles.

    Formula:  DSOH(t) = SOH(t) - SOH(t-1)

    Physical meaning:
      - Negative DSOH indicates health decline between cycles.
      - Sudden large negative values may be degradation acceleration signals.
    """
    result_dfs: list[pd.DataFrame] = []
    for bid, group in df.groupby("battery_id"):
        g = group.sort_values("cycle_number").copy()
        g["soh_change"] = g["soh"].diff()
        result_dfs.append(g)
    return pd.concat(result_dfs, ignore_index=True)


def _add_log_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed internal resistance features.

    Formula:  log_Re = log(Re_Ohm + epsilon)
              log_Rct = log(Rct_Ohm + epsilon)

    where epsilon = 1e-6 to handle zero values.

    Physical meaning:
      - Internal resistance distributions are typically right-skewed.
      - Log-transformation brings the distribution closer to Gaussian,
        which improves Bayesian Gaussian likelihood estimates.
      - Rising log_Re over cycles indicates electrolyte degradation.
    """
    eps = 1e-6
    if "Re_Ohm" in df.columns:
        df["log_Re"] = np.log(df["Re_Ohm"].clip(lower=eps))
    if "Rct_Ohm" in df.columns:
        df["log_Rct"] = np.log(df["Rct_Ohm"].clip(lower=eps))
    return df


def _add_cycle_normalised(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cycle_norm: cycle number normalised to [0, 1] per battery.

    Formula:  cycle_norm(t) = cycle_number(t) / max_cycle_number

    Physical meaning:
      - Provides a relative age measure independent of total lifespan.
      - Useful for comparing batteries with different total cycle counts.
    """
    result_dfs: list[pd.DataFrame] = []
    for bid, group in df.groupby("battery_id"):
        g = group.copy()
        max_cyc = g["cycle_number"].max()
        g["cycle_norm"] = g["cycle_number"] / max_cyc if max_cyc > 0 else 0.0
        result_dfs.append(g)
    return pd.concat(result_dfs, ignore_index=True)


def _add_degradation_bin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ordinal degradation_bin column.

    Encoding:
        0 = Healthy
        1 = Early Degradation
        2 = Degraded

    Used by:
      - K-Means clustering (as reference for interpretation)
      - Supervised learning (as ordinal target)
    """
    label_map = {"Healthy": 0, "Early Degradation": 1, "Degraded": 2, "Unknown": -1}
    if "degradation_label" in df.columns:
        df["degradation_bin"] = df["degradation_label"].map(label_map).fillna(-1).astype(int)
    return df


def _add_rolling_stats(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling-window statistics for capacity and voltage_mean.

    Formula:  X_rolling_mean(t) = mean(X[t-w+1 : t])
              X_rolling_std(t)  = std(X[t-w+1 : t])

    Physical meaning:
      - Rolling mean smooths short-term noise, revealing the degradation trend.
      - Rolling std captures local variability — increasing std may signal
        onset of degradation instability.

    Parameters
    ----------
    window : int — number of cycles in the rolling window (default 5)
    """
    result_dfs: list[pd.DataFrame] = []
    for bid, group in df.groupby("battery_id"):
        g = group.sort_values("cycle_number").copy()
        for col in ("capacity", "soh"):
            if col in g.columns:
                g[f"{col}_rolling_mean"] = (
                    g[col].rolling(window=window, min_periods=1).mean()
                )
                g[f"{col}_rolling_std"] = (
                    g[col].rolling(window=window, min_periods=1).std()
                )
        result_dfs.append(g)
    return pd.concat(result_dfs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature column registry
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_columns() -> dict[str, list[str]]:
    """
    Return the canonical grouping of feature columns.

    Returns a dict with keys:
        "raw"       : directly measured / extracted
        "derived"   : computed from raw
        "statistical": rolling / change features
        "bayesian"  : features used as evidence in the Bayesian risk engine
        "target"    : label / target columns
    """
    return {
        "raw": [
            "voltage_mean", "voltage_std", "voltage_min", "voltage_max",
            "current_mean", "current_std", "current_min", "current_max",
            "temperature_mean", "temperature_std", "temperature_min", "temperature_max",
            "discharge_duration", "capacity", "Re_Ohm", "Rct_Ohm",
            "ambient_temperature",
        ],
        "derived": [
            "voltage_range", "current_range", "temperature_range",
            "voltage_cv", "current_cv", "temperature_cv",
            "capacity_fade", "capacity_fade_rate",
            "log_Re", "log_Rct",
            "cycle_norm",
        ],
        "statistical": [
            "soh_change",
            "capacity_rolling_mean", "capacity_rolling_std",
            "soh_rolling_mean", "soh_rolling_std",
        ],
        "bayesian": [
            "capacity", "soh", "voltage_mean", "voltage_std",
            "temperature_mean", "temperature_std",
            "Re_Ohm", "Rct_Ohm",
            "current_mean", "current_std",
        ],
        "target": [
            "soh", "eol_flag", "degradation_label", "degradation_bin",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to the cycle-level DataFrame.

    Steps applied in order:
      1. Range features (voltage_range, current_range, temperature_range)
      2. Coefficient of variation
      3. Capacity fade + fade rate
      4. SOH change (per-cycle delta)
      5. Log-transformed resistance
      6. Normalised cycle number
      7. Degradation ordinal bin
      8. Rolling statistics (capacity, SOH)

    Parameters
    ----------
    df  : pd.DataFrame — output of preprocessing.build_cycle_level_dataset()
    cfg : dict         — config.yaml contents

    Returns
    -------
    pd.DataFrame with all original columns plus derived features.
    """
    log.info("  Applying feature engineering ...")
    df = df.copy()
    df = _add_range_features(df)
    df = _add_coefficient_of_variation(df)
    df = _add_capacity_fade(df)
    df = _add_soh_change(df)
    df = _add_log_resistance(df)
    df = _add_cycle_normalised(df)
    df = _add_degradation_bin(df)
    df = _add_rolling_stats(df, window=5)

    n_new = len(df.columns)
    log.info(f"  Feature engineering complete: {n_new} total columns")
    log.info(f"  New derived features: voltage_range, current_range, temperature_range, "
             f"voltage_cv, current_cv, temperature_cv, capacity_fade, capacity_fade_rate, "
             f"soh_change, log_Re, log_Rct, cycle_norm, degradation_bin, "
             f"capacity_rolling_mean, capacity_rolling_std, soh_rolling_mean, soh_rolling_std")
    return df
