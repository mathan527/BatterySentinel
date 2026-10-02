"""
BatterySentinel — Phase 2 & 3: Preprocessing Pipeline Script
==============================================================
Converts raw NASA battery CSV data into a clean, cycle-level dataset.

Steps:
  1. Load metadata and group by battery
  2. Extract per-cycle features (voltage, current, temperature stats)
  3. Compute SOH using max-capacity reference
  4. Map impedance Re/Rct to discharge cycles
  5. Assign EOL flag and degradation label
  6. Apply feature engineering (derived features)
  7. Save cycle_data.csv to data/processed/

Usage:
    python scripts/preprocess_dataset.py

Output:
    data/processed/cycle_data.csv
    data/processed/preprocessing_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("preprocess_dataset")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config
from src.preprocessing import build_cycle_level_dataset, report_missing_values
from src.feature_engineering import engineer_features, get_feature_columns


def main() -> None:
    cfg = load_config()
    data_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "cleaned_dataset"
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  BATTERYSENTINEL — Phase 2/3: Preprocessing Pipeline")
    log.info("=" * 60)

    # ── Phase 2: Build cycle-level dataset ─────────────────────────────────────
    cycle_df = build_cycle_level_dataset(data_dir, cfg)

    # ── Phase 3: Feature engineering ──────────────────────────────────────────
    cycle_df = engineer_features(cycle_df, cfg)

    # ── Missing value report ───────────────────────────────────────────────────
    missing_report = report_missing_values(cycle_df)
    log.info("\n  Missing values:")
    if missing_report.empty:
        log.info("    None.")
    else:
        for _, row in missing_report.iterrows():
            log.info(f"    {row['feature']:<30}: {row['n_missing']:4d} ({row['pct_missing']:.1f}%)")

    # ── Descriptive summary ────────────────────────────────────────────────────
    log.info("\n  Dataset shape: %s", cycle_df.shape)
    log.info("  Columns: %s", list(cycle_df.columns))
    log.info("\n  Degradation label counts:")
    for label, cnt in cycle_df["degradation_label"].value_counts().items():
        log.info(f"    {label:<25}: {cnt:4d}  ({cnt/len(cycle_df)*100:.1f}%)")
    log.info("\n  SOH statistics:")
    soh_stats = cycle_df["soh"].describe()
    for stat, val in soh_stats.items():
        log.info(f"    {stat:<8}: {val:.2f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_csv = processed_dir / "cycle_data.csv"
    cycle_df.to_csv(out_csv, index=False)
    log.info(f"\n  Saved: {out_csv}  ({out_csv.stat().st_size:,} bytes)")

    # ── Save preprocessing report ──────────────────────────────────────────────
    feature_cols = get_feature_columns()
    report = {
        "generated_at": datetime.now().isoformat(),
        "n_batteries": int(cycle_df["battery_id"].nunique()),
        "n_discharge_cycles": int(len(cycle_df)),
        "batteries": sorted(cycle_df["battery_id"].unique().tolist()),
        "columns": list(cycle_df.columns),
        "feature_groups": feature_cols,
        "degradation_counts": cycle_df["degradation_label"].value_counts().to_dict(),
        "eol_flag_count": int(cycle_df["eol_flag"].sum()),
        "data_quality_flags": cycle_df["data_quality_flag"].value_counts().to_dict(),
        "soh_statistics": cycle_df["soh"].describe().round(4).to_dict(),
        "missing_values": {
            row["feature"]: {"n_missing": int(row["n_missing"]), "pct": float(row["pct_missing"])}
            for _, row in missing_report.iterrows()
        },
        "per_battery_summary": [],
    }

    for bid, grp in cycle_df.groupby("battery_id"):
        report["per_battery_summary"].append({
            "battery_id": bid,
            "n_cycles": int(len(grp)),
            "soh_initial": float(grp.sort_values("cycle_number")["soh"].iloc[0]),
            "soh_final": float(grp.sort_values("cycle_number")["soh"].iloc[-1]),
            "soh_min": float(grp["soh"].min()),
            "reached_eol": bool(grp["eol_flag"].any()),
            "ambient_temperatures": sorted(grp["ambient_temperature"].dropna().unique().tolist()),
            "degradation_labels": grp["degradation_label"].value_counts().to_dict(),
        })

    report_path = processed_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"  Saved: {report_path}")

    log.info("\n" + "=" * 60)
    log.info("  Phase 2/3 COMPLETE")
    log.info("=" * 60)
    log.info(f"  Cycle-level dataset : {out_csv}")
    log.info(f"  Preprocessing report: {report_path}")
    log.info("  Ready for Phase 4 — Statistical Analysis")


if __name__ == "__main__":
    main()
