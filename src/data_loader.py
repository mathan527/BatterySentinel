"""
BatterySentinel — src/data_loader.py
=====================================
Loads raw NASA battery data from the cleaned CSV dataset.

Dataset layout (patrickfleith/nasa-battery-dataset on Kaggle):
  cleaned_dataset/
    metadata.csv                 — cycle index (7565 rows)
    data/00001.csv ... 07565.csv — per-cycle time-series

Public API
----------
    load_metadata(data_dir)       -> pd.DataFrame
    load_cycle_csv(data_dir, fn)  -> pd.DataFrame
    load_all_batteries(data_dir)  -> dict[battery_id, dict]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ── UTF-8 safe logging on Windows ────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SRC_DIR.parent


# ─────────────────────────────────────────────────────────────────────────────
# Config helper
# ─────────────────────────────────────────────────────────────────────────────

def load_config(cfg_path: Path | None = None) -> dict:
    """Load project YAML configuration."""
    if cfg_path is None:
        cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata loader
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load metadata.csv from the cleaned_dataset directory.

    The metadata file is the central index of all 7565 cycle records.
    It contains one row per cycle with columns:
        type, start_time, ambient_temperature, battery_id,
        test_id, uid, filename, Capacity, Re, Rct

    Parameters
    ----------
    data_dir : Path
        Path to the cleaned_dataset/ directory.

    Returns
    -------
    pd.DataFrame with additional parsed columns:
        capacity_Ah  : float or NaN  (discharge cycles only)
        Re_Ohm       : float or NaN  (impedance cycles only)
        Rct_Ohm      : float or NaN  (impedance cycles only)
    """
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {meta_path}.\n"
            "Ensure the dataset is downloaded into data/raw/cleaned_dataset/"
        )

    meta = pd.read_csv(meta_path, dtype=str)  # load as str first — Capacity is mixed
    log.debug(f"Loaded metadata: {len(meta)} rows, columns={list(meta.columns)}")

    # ── Parse numeric columns safely ──────────────────────────────────────────
    meta["ambient_temperature"] = pd.to_numeric(meta["ambient_temperature"], errors="coerce")
    meta["test_id"] = pd.to_numeric(meta["test_id"], errors="coerce")
    meta["uid"] = pd.to_numeric(meta["uid"], errors="coerce")

    # Capacity, Re, Rct stored as strings; convert to float
    meta["capacity_Ah"] = pd.to_numeric(meta["Capacity"], errors="coerce")
    meta["Re_Ohm"] = pd.to_numeric(meta["Re"], errors="coerce")
    meta["Rct_Ohm"] = pd.to_numeric(meta["Rct"], errors="coerce")

    # ── Physical validity filtering for impedance values ──────────────────────
    # Li-Ion internal resistance (Re) is physically in the range ~0.01–1.0 Ohm.
    # Values outside (0, 10) are measurement artifacts (documented NASA anomalies
    # in B0050, B0052 from experiment software crashes). Replace with NaN.
    RE_MIN, RE_MAX = 1e-4, 10.0
    n_bad_re = ((meta["Re_Ohm"] <= RE_MIN) | (meta["Re_Ohm"] > RE_MAX)).sum()
    n_bad_rct = ((meta["Rct_Ohm"] <= 0) | (meta["Rct_Ohm"] > 50.0)).sum()
    if n_bad_re > 0:
        log.warning(
            f"  Replacing {n_bad_re} physically implausible Re_Ohm values with NaN "
            f"(outside 0–{RE_MAX} Ohm — NASA experiment anomalies)."
        )
        meta.loc[(meta["Re_Ohm"] <= RE_MIN) | (meta["Re_Ohm"] > RE_MAX), "Re_Ohm"] = float("nan")
    if n_bad_rct > 0:
        log.warning(
            f"  Replacing {n_bad_rct} physically implausible Rct_Ohm values with NaN."
        )
        meta.loc[(meta["Rct_Ohm"] <= 0) | (meta["Rct_Ohm"] > 50.0), "Rct_Ohm"] = float("nan")

    # ── Validate required columns ─────────────────────────────────────────────
    required = ["type", "battery_id", "filename", "ambient_temperature"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    log.info(
        f"Metadata loaded: {len(meta)} cycles | "
        f"batteries={meta['battery_id'].nunique()} | "
        f"types={meta['type'].value_counts().to_dict()}"
    )
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Per-cycle CSV loader
# ─────────────────────────────────────────────────────────────────────────────

def load_cycle_csv(data_dir: Path, filename: str) -> pd.DataFrame | None:
    """
    Load a single per-cycle time-series CSV from the data/ subdirectory.

    Parameters
    ----------
    data_dir : Path  — path to cleaned_dataset/
    filename : str   — e.g. "00001.csv"

    Returns
    -------
    pd.DataFrame or None if the file cannot be read.
    """
    fpath = data_dir / "data" / filename
    if not fpath.exists():
        log.warning(f"Cycle CSV not found: {fpath}")
        return None
    try:
        df = pd.read_csv(fpath, dtype=float)
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not read {filename}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Full battery data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_batteries(data_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Load metadata grouped by battery ID.

    Returns a dictionary::

        {
            "B0005": {
                "metadata": pd.DataFrame,       # all cycles for this battery
                "discharge": pd.DataFrame,      # discharge rows only
                "charge": pd.DataFrame,         # charge rows only
                "impedance": pd.DataFrame,      # impedance rows only
            },
            ...
        }

    The actual per-cycle CSV data is NOT loaded here to save memory.
    Use ``load_cycle_csv()`` to load individual cycle files on demand,
    or ``build_cycle_level_dataset()`` in preprocessing.py for the full
    aggregated dataset.

    Parameters
    ----------
    data_dir : Path — path to cleaned_dataset/

    Returns
    -------
    dict[str, dict]
    """
    meta = load_metadata(data_dir)
    batteries: dict[str, dict[str, Any]] = {}

    for bid in sorted(meta["battery_id"].unique()):
        b_meta = meta[meta["battery_id"] == bid].copy().reset_index(drop=True)
        batteries[bid] = {
            "metadata": b_meta,
            "discharge": b_meta[b_meta["type"] == "discharge"].reset_index(drop=True),
            "charge": b_meta[b_meta["type"] == "charge"].reset_index(drop=True),
            "impedance": b_meta[b_meta["type"] == "impedance"].reset_index(drop=True),
        }

    log.info(f"Loaded {len(batteries)} batteries from metadata.")
    return batteries
