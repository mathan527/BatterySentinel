"""
BatterySentinel — src/preprocessing.py
=========================================
Converts raw NASA battery cycle records into a clean, cycle-level DataFrame.

Key design decisions (documented):
  - SOH denominator: max(capacity) per battery, NOT first capacity.
    Reason: some batteries have formation cycles where capacity ramps up
    from near-zero before stabilising. Using the first capacity would
    produce SOH values far above 100%.
  - Only DISCHARGE cycles are included in the cycle-level SOH table.
    Charge and impedance cycles contribute features (duration, Re/Rct)
    but are not counted as SOH-bearing observations.
  - Zero-capacity discharge readings are flagged as DATA_QUALITY anomalies,
    not silently dropped — they are retained with a flag column.
  - Internal resistance (Re, Rct) from impedance cycles is forward-filled
    onto the nearest preceding discharge cycle.

Output schema (cycle_data.csv):
  battery_id            — string   e.g. "B0005"
  cycle_number          — int      sequential discharge index (1-based)
  uid                   — int      original dataset uid
  ambient_temperature   — float    °C at cycle start
  voltage_mean          — float    mean V during discharge
  voltage_std           — float    std  V during discharge
  voltage_min           — float    min  V during discharge
  voltage_max           — float    max  V during discharge
  current_mean          — float    mean A during discharge (negative = discharging)
  current_std           — float    std  A
  current_min           — float    min  A
  current_max           — float    max  A
  temperature_mean      — float    mean °C during discharge
  temperature_std       — float    std  °C
  temperature_min       — float    min  °C
  temperature_max       — float    max  °C
  discharge_duration    — float    seconds (max Time in discharge CSV)
  capacity              — float    Ah discharged
  soh                   — float    capacity / max_capacity × 100  (%)
  Re_Ohm                — float    electrolyte resistance (from impedance, forward-filled)
  Rct_Ohm               — float    charge-transfer resistance  (from impedance, forward-filled)
  eol_flag              — int      1 if SOH <= eol_threshold, else 0
  degradation_label     — str      "Healthy" / "Early Degradation" / "Degraded"
  data_quality_flag     — str      "" / "zero_capacity" / "formation_cycle"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_all_batteries, load_config, load_cycle_csv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent


# ─────────────────────────────────────────────────────────────────────────────
# Per-cycle feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_cycle_stats(df: pd.DataFrame, col: str) -> dict[str, float]:
    """
    Compute descriptive statistics for a single column of a cycle CSV.

    Returns a dict: {col_mean, col_std, col_min, col_max}
    Returns NaN values if the column is missing or the series is empty.
    """
    if df is None or col not in df.columns or df[col].dropna().empty:
        return {
            f"{col}_mean": np.nan,
            f"{col}_std": np.nan,
            f"{col}_min": np.nan,
            f"{col}_max": np.nan,
        }
    s = df[col].dropna()
    return {
        f"{col}_mean": float(s.mean()),
        f"{col}_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        f"{col}_min": float(s.min()),
        f"{col}_max": float(s.max()),
    }


def _extract_discharge_features(
    meta_row: pd.Series,
    data_dir: Path,
) -> dict[str, Any]:
    """
    Extract per-cycle features from a single discharge cycle.

    Parameters
    ----------
    meta_row : pd.Series  — one row of the discharge metadata
    data_dir : Path       — cleaned_dataset/ root

    Returns
    -------
    dict with all feature columns for this discharge cycle.
    """
    features: dict[str, Any] = {
        "battery_id": meta_row["battery_id"],
        "uid": int(meta_row["uid"]) if pd.notna(meta_row["uid"]) else np.nan,
        "ambient_temperature": meta_row["ambient_temperature"],
        "capacity": meta_row["capacity_Ah"],
    }

    df = load_cycle_csv(data_dir, meta_row["filename"])

    if df is not None and not df.empty:
        # Voltage statistics
        features.update(_extract_cycle_stats(df, "Voltage_measured"))
        # Current statistics
        features.update(_extract_cycle_stats(df, "Current_measured"))
        # Temperature statistics
        features.update(_extract_cycle_stats(df, "Temperature_measured"))
        # Discharge duration
        if "Time" in df.columns and not df["Time"].dropna().empty:
            features["discharge_duration"] = float(df["Time"].max())
        else:
            features["discharge_duration"] = np.nan
    else:
        # File missing or unreadable — fill with NaN
        for col in ("Voltage_measured", "Current_measured", "Temperature_measured"):
            features.update({
                f"{col}_mean": np.nan, f"{col}_std": np.nan,
                f"{col}_min": np.nan, f"{col}_max": np.nan,
            })
        features["discharge_duration"] = np.nan

    return features


# ─────────────────────────────────────────────────────────────────────────────
# SOH calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_soh(
    capacity_series: pd.Series,
    reference: str = "max",
) -> pd.Series:
    """
    Compute State of Health (SOH) for a sequence of discharge capacities.

    Mathematical definition:
        SOH(t) = C(t) / C_ref × 100  (%)

    where C_ref is the reference (nominal) capacity.

    Parameters
    ----------
    capacity_series : pd.Series of float (Ah)
    reference : "max" (default) | "first"
        "max"   — use the peak capacity observed across all cycles.
                  RECOMMENDED: avoids formation-cycle distortion.
        "first" — use the very first capacity reading.
                  Suitable only when the first cycle is a known full charge.

    Returns
    -------
    pd.Series of SOH values (%), same index as input.

    Notes
    -----
    For the NASA dataset, many batteries undergo *formation cycling* where
    capacity increases for the first several cycles before stabilising.
    Using "first" as the reference yields SOH > 100% for later cycles.
    Using "max" anchors SOH at the peak healthy state.

    Formula:
        SOH = (C_t / C_ref) × 100
    """
    finite = capacity_series.replace(0, np.nan).dropna()
    if finite.empty:
        return pd.Series(np.nan, index=capacity_series.index)

    if reference == "max":
        c_ref = float(finite.max())
    elif reference == "first":
        c_ref = float(finite.iloc[0])
    else:
        raise ValueError(f"reference must be 'max' or 'first', got '{reference}'")

    if c_ref == 0:
        log.warning("Reference capacity is 0 — SOH undefined.")
        return pd.Series(np.nan, index=capacity_series.index)

    soh = (capacity_series / c_ref) * 100.0
    return soh


# ─────────────────────────────────────────────────────────────────────────────
# Degradation labelling
# ─────────────────────────────────────────────────────────────────────────────

def assign_degradation_label(
    soh: float,
    eol_threshold: float = 0.70,
    early_threshold: float = 0.85,
) -> str:
    """
    Assign a human-readable degradation label based on SOH.

    Labels:
        "Healthy"           — SOH > early_threshold × 100%
        "Early Degradation" — eol_threshold × 100% < SOH <= early_threshold × 100%
        "Degraded"          — SOH <= eol_threshold × 100%

    Parameters
    ----------
    soh            : float — SOH percentage (0–100)
    eol_threshold  : float — configurable EOL threshold (default 0.70 = 70%)
    early_threshold: float — early-warning boundary (default 0.85 = 85%)

    Notes
    -----
    These thresholds are CONFIGURABLE RESEARCH PARAMETERS, not universal
    safety standards.
    """
    if pd.isna(soh):
        return "Unknown"
    eol_pct = eol_threshold * 100.0
    early_pct = early_threshold * 100.0
    if soh > early_pct:
        return "Healthy"
    elif soh > eol_pct:
        return "Early Degradation"
    else:
        return "Degraded"


# ─────────────────────────────────────────────────────────────────────────────
# Impedance mapping
# ─────────────────────────────────────────────────────────────────────────────

def _map_impedance_to_discharge(
    discharge_df: pd.DataFrame,
    impedance_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward-fill Re and Rct from impedance cycles onto discharge cycles.

    Strategy: for each discharge cycle (identified by uid / test_id),
    find the most recent preceding impedance measurement and assign its
    Re and Rct values.

    Parameters
    ----------
    discharge_df    : DataFrame of discharge cycles (with 'uid' column)
    impedance_meta  : DataFrame of impedance rows from metadata

    Returns
    -------
    discharge_df with 'Re_Ohm' and 'Rct_Ohm' columns added.
    """
    if impedance_meta.empty:
        discharge_df = discharge_df.copy()
        discharge_df["Re_Ohm"] = np.nan
        discharge_df["Rct_Ohm"] = np.nan
        return discharge_df

    # Build a uid-indexed impedance lookup
    imp = impedance_meta[["uid", "Re_Ohm", "Rct_Ohm"]].dropna(
        subset=["Re_Ohm"]
    ).sort_values("uid")

    # Merge using merge_asof: for each discharge uid, find last impedance uid <= it
    dis_sorted = discharge_df.sort_values("uid").copy()
    merged = pd.merge_asof(
        dis_sorted,
        imp.rename(columns={"uid": "imp_uid"}),
        left_on="uid",
        right_on="imp_uid",
        direction="backward",
    )
    # Restore original order
    merged = merged.sort_values("cycle_number").reset_index(drop=True)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline: build cycle-level dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_cycle_level_dataset(
    data_dir: Path,
    cfg: dict,
    batteries_to_include: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build the clean, cycle-level dataset from raw NASA battery data.

    This is the main Phase 2 function. It:
    1. Loads metadata for all batteries.
    2. For each battery, iterates over discharge cycles.
    3. Loads the per-cycle CSV and extracts voltage/current/temperature stats.
    4. Computes SOH using max-capacity as the reference.
    5. Maps impedance Re/Rct to discharge cycles via forward-fill.
    6. Assigns EOL flag and degradation label.
    7. Returns a unified, clean DataFrame.

    Parameters
    ----------
    data_dir             : Path to cleaned_dataset/
    cfg                  : Config dict from config.yaml
    batteries_to_include : list of battery IDs to process (None = all)

    Returns
    -------
    pd.DataFrame — one row per discharge cycle
    """
    eol_threshold = cfg["soh"]["eol_threshold"]
    early_threshold = cfg["labels"]["early_degradation_soh"]

    log.info("=" * 60)
    log.info("  Building cycle-level dataset ...")
    log.info("=" * 60)

    batteries = load_all_batteries(data_dir)
    if batteries_to_include:
        batteries = {k: v for k, v in batteries.items() if k in batteries_to_include}

    all_records: list[pd.DataFrame] = []

    for bid, bdata in batteries.items():
        dis_meta = bdata["discharge"]
        imp_meta = bdata["impedance"]

        if dis_meta.empty:
            log.warning(f"  Battery {bid}: no discharge cycles — skipping.")
            continue

        log.info(f"  Battery {bid}: processing {len(dis_meta)} discharge cycles ...")

        # ── Extract per-cycle features ─────────────────────────────────────────
        records: list[dict] = []
        for cycle_num, (_, row) in enumerate(dis_meta.iterrows(), start=1):
            feats = _extract_discharge_features(row, data_dir)
            feats["cycle_number"] = cycle_num
            records.append(feats)

        if not records:
            continue

        df = pd.DataFrame(records)

        # ── Rename columns to clean names ──────────────────────────────────────
        rename_map = {
            "Voltage_measured_mean": "voltage_mean",
            "Voltage_measured_std": "voltage_std",
            "Voltage_measured_min": "voltage_min",
            "Voltage_measured_max": "voltage_max",
            "Current_measured_mean": "current_mean",
            "Current_measured_std": "current_std",
            "Current_measured_min": "current_min",
            "Current_measured_max": "current_max",
            "Temperature_measured_mean": "temperature_mean",
            "Temperature_measured_std": "temperature_std",
            "Temperature_measured_min": "temperature_min",
            "Temperature_measured_max": "temperature_max",
        }
        df = df.rename(columns=rename_map)

        # ── SOH calculation ────────────────────────────────────────────────────
        df["soh"] = compute_soh(df["capacity"], reference="max")

        # ── Data quality flags ─────────────────────────────────────────────────
        df["data_quality_flag"] = ""
        # Formation cycles: capacity starts unusually low
        cap_max = df["capacity"].max()
        is_formation = df["capacity"] < (cap_max * 0.30)
        df.loc[is_formation & (df["cycle_number"] <= 10), "data_quality_flag"] = "formation_cycle"
        # Zero or near-zero capacity
        df.loc[df["capacity"].fillna(0) < 0.01, "data_quality_flag"] = "zero_capacity"

        # ── Map impedance ──────────────────────────────────────────────────────
        if "Re_Ohm" not in df.columns:
            df["Re_Ohm"] = np.nan
            df["Rct_Ohm"] = np.nan

        # Merge impedance Re/Rct by forward-fill on uid
        imp_lookup = imp_meta[["uid", "Re_Ohm", "Rct_Ohm"]].copy()
        imp_lookup = imp_lookup.dropna(subset=["Re_Ohm"]).sort_values("uid")
        if not imp_lookup.empty:
            df_sorted = df.sort_values("uid").copy()
            merged = pd.merge_asof(
                df_sorted,
                imp_lookup.rename(columns={"uid": "imp_uid"}),
                left_on="uid",
                right_on="imp_uid",
                direction="backward",
                suffixes=("", "_imp"),
            )
            # Use the impedance-sourced columns
            if "Re_Ohm_imp" in merged.columns:
                merged["Re_Ohm"] = merged["Re_Ohm_imp"]
                merged["Rct_Ohm"] = merged["Rct_Ohm_imp"]
                merged = merged.drop(
                    columns=["Re_Ohm_imp", "Rct_Ohm_imp", "imp_uid"], errors="ignore"
                )
            df = merged.sort_values("cycle_number").reset_index(drop=True)

        # ── EOL flag and degradation label ────────────────────────────────────
        df["eol_flag"] = (df["soh"] <= eol_threshold * 100.0).astype(int)
        df["degradation_label"] = df["soh"].apply(
            lambda s: assign_degradation_label(s, eol_threshold, early_threshold)
        )
        # Fill any NaN data_quality_flags with empty string
        df["data_quality_flag"] = df["data_quality_flag"].fillna("")

        all_records.append(df)

    if not all_records:
        raise ValueError("No discharge cycle records were extracted. Check data directory.")

    # ── Concatenate and tidy ──────────────────────────────────────────────────
    full_df = pd.concat(all_records, ignore_index=True)

    # Reorder columns to the canonical schema
    col_order = [
        "battery_id", "cycle_number", "uid", "ambient_temperature",
        "voltage_mean", "voltage_std", "voltage_min", "voltage_max",
        "current_mean", "current_std", "current_min", "current_max",
        "temperature_mean", "temperature_std", "temperature_min", "temperature_max",
        "discharge_duration",
        "capacity", "soh",
        "Re_Ohm", "Rct_Ohm",
        "eol_flag", "degradation_label", "data_quality_flag",
    ]
    # Add any extra columns that appear
    extra = [c for c in full_df.columns if c not in col_order]
    full_df = full_df[[c for c in col_order if c in full_df.columns] + extra]

    log.info(f"\nCycle-level dataset built: {full_df.shape[0]} rows x {full_df.shape[1]} cols")
    log.info(f"  Batteries        : {full_df['battery_id'].nunique()}")
    log.info(f"  Discharge cycles : {len(full_df)}")
    log.info(f"  EOL cycles       : {full_df['eol_flag'].sum()}")
    log.info(f"  Degradation labels: {full_df['degradation_label'].value_counts().to_dict()}")
    log.info(f"  Missing SOH      : {full_df['soh'].isna().sum()}")
    log.info(f"  Formation flags  : {(full_df['data_quality_flag']=='formation_cycle').sum()}")
    log.info(f"  Zero-cap flags   : {(full_df['data_quality_flag']=='zero_capacity').sum()}")

    return full_df


# ─────────────────────────────────────────────────────────────────────────────
# Missing-value report
# ─────────────────────────────────────────────────────────────────────────────

def report_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missing values per column.

    Returns
    -------
    pd.DataFrame with columns: feature, n_missing, pct_missing
    """
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({
        "feature": missing.index,
        "n_missing": missing.values,
        "pct_missing": pct.values,
    })
    report = report[report["n_missing"] > 0].sort_values("n_missing", ascending=False)
    return report.reset_index(drop=True)
