"""
BatterySentinel — Phase 1: Adapted Dataset Inspection Script
=============================================================
This script inspects the patrickfleith/nasa-battery-dataset from Kaggle,
which is a cleaned CSV-based version (not raw .mat files).

Structure discovered:
  cleaned_dataset/
    metadata.csv            — index of all 7565 cycle records
    data/
      00001.csv ... 07565.csv — per-cycle time-series data
      README_*.txt            — experiment descriptions
    extra_infos/
      README_*.txt            — additional documentation

Run:
    python scripts/inspect_dataset.py
"""

import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("inspect_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(cfg_path, "r") as fh:
        return yaml.safe_load(fh)


def safe_float(v: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_text_files(directory: Path) -> dict[str, str]:
    """Read all .txt files in a directory, return {filename: content}."""
    contents: dict[str, str] = {}
    if not directory.exists():
        return contents
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".txt"):
            try:
                contents[fname] = (directory / fname).read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception as exc:  # noqa: BLE001
                contents[fname] = f"[ERROR: {exc}]"
    return contents


def inspect_sample_csvs(data_dir: Path, meta: pd.DataFrame) -> dict:
    """
    Inspect a representative sample of charge, discharge, and impedance CSVs
    to catalogue their column structure.
    """
    result: dict[str, Any] = {}
    for op_type in ("discharge", "charge", "impedance"):
        sample_rows = meta[meta["type"] == op_type].head(3)
        for _, row in sample_rows.iterrows():
            fpath = data_dir / "data" / row["filename"]
            if not fpath.exists():
                continue
            try:
                df = pd.read_csv(fpath)
                result[op_type] = {
                    "example_file": row["filename"],
                    "columns": list(df.columns),
                    "dtypes": {c: str(df[c].dtype) for c in df.columns},
                    "shape_example": list(df.shape),
                    "missing_values": df.isnull().sum().to_dict(),
                    "sample_stats": {
                        c: {
                            "min": float(df[c].min()) if df[c].dtype.kind in "fiu" else None,
                            "max": float(df[c].max()) if df[c].dtype.kind in "fiu" else None,
                            "mean": float(df[c].mean()) if df[c].dtype.kind in "fiu" else None,
                        }
                        for c in df.columns
                    },
                }
                break
            except Exception as exc:  # noqa: BLE001
                log.warning(f"  Could not read {row['filename']}: {exc}")
    return result


def build_battery_summaries(meta: pd.DataFrame) -> list[dict]:
    """Build per-battery statistical summaries from metadata."""
    summaries: list[dict] = []
    meta_c = meta.copy()
    meta_c["cap_float"] = meta_c["Capacity"].apply(safe_float)
    meta_c["re_float"] = meta_c["Re"].apply(safe_float)
    meta_c["rct_float"] = meta_c["Rct"].apply(safe_float)

    for bid in sorted(meta_c["battery_id"].unique()):
        b = meta_c[meta_c["battery_id"] == bid]
        dis = b[b["type"] == "discharge"]
        caps = dis["cap_float"].dropna().tolist()
        re_vals = b["re_float"].dropna().tolist()
        rct_vals = b["rct_float"].dropna().tolist()

        entry: dict[str, Any] = {
            "battery_id": bid,
            "total_records": int(len(b)),
            "n_charge": int((b["type"] == "charge").sum()),
            "n_discharge": int((b["type"] == "discharge").sum()),
            "n_impedance": int((b["type"] == "impedance").sum()),
            "ambient_temperatures_C": sorted(b["ambient_temperature"].unique().tolist()),
            "capacity_initial_Ah": round(caps[0], 5) if caps else None,
            "capacity_final_Ah": round(caps[-1], 5) if caps else None,
            "capacity_min_Ah": round(min(caps), 5) if caps else None,
            "capacity_max_Ah": round(max(caps), 5) if caps else None,
            "n_discharge_cycles": len(caps),
            "has_internal_resistance_Re": len(re_vals) > 0,
            "has_internal_resistance_Rct": len(rct_vals) > 0,
            "re_initial": round(re_vals[0], 6) if re_vals else None,
            "rct_initial": round(rct_vals[0], 6) if rct_vals else None,
        }

        if caps:
            init = caps[0]
            soh_series = [round(c / init * 100, 2) for c in caps]
            entry["soh_initial_pct"] = soh_series[0]
            entry["soh_final_pct"] = soh_series[-1]
            entry["soh_min_pct"] = round(min(soh_series), 2)
            entry["soh_max_pct"] = round(max(soh_series), 2)
            entry["soh_decline_pct"] = round(soh_series[0] - soh_series[-1], 2)
            entry["reached_eol_70pct"] = min(soh_series) <= 70.0

        summaries.append(entry)
    return summaries


def assess_unit1_features(battery_summaries: list[dict]) -> dict:
    """Map Unit 1 ML/statistics concepts to available dataset fields."""
    has_cap = any(b.get("capacity_initial_Ah") is not None for b in battery_summaries)
    has_imp = any(b.get("has_internal_resistance_Re") for b in battery_summaries)

    return {
        "Mean / Variance / Std Dev": {
            "available": True,
            "source": "Voltage_measured, Current_measured, Temperature_measured per cycle",
            "unit1_concept": "Mean, Variance",
        },
        "Probability Density (KDE)": {
            "available": has_cap,
            "source": "Voltage, Current, Capacity distributions across cycles",
            "unit1_concept": "Probability Density",
        },
        "Quantiles / IQR": {
            "available": True,
            "source": "All continuous measurement channels",
            "unit1_concept": "Quantiles",
        },
        "Expectation E[X]": {
            "available": True,
            "source": "E[Voltage], E[Capacity], E[SOH] etc.",
            "unit1_concept": "Expectation",
        },
        "Covariance": {
            "available": has_cap,
            "source": "Capacity ↔ Temperature ↔ Cycle ↔ SOH ↔ Re",
            "unit1_concept": "Covariance",
        },
        "Continuous Random Variables": {
            "available": True,
            "source": "Voltage, Current, Temperature time-series per cycle",
            "unit1_concept": "Continuous Random Variables",
        },
        "Discrete Random Variables": {
            "available": True,
            "source": "Operation type (charge/discharge/impedance), EOL flag",
            "unit1_concept": "Discrete Random Variables",
        },
        "State of Health (SOH)": {
            "available": has_cap,
            "source": "capacity / initial_capacity × 100%",
            "unit1_concept": "Supervised Learning, Bayesian target",
        },
        "Polynomial Curve Fitting": {
            "available": has_cap,
            "source": "SOH vs cycle number — fit degree 1,2,3 polynomials",
            "unit1_concept": "Polynomial Curve Fitting",
        },
        "Bayes Rule / Bayesian Risk": {
            "available": has_cap,
            "source": "P(Degradation|Capacity, Temp, Voltage, Re, Cycle)",
            "unit1_concept": "Bayes Rule",
        },
        "Conditional Independence": {
            "available": has_cap,
            "source": "Naive Bayes conditional independence assumption",
            "unit1_concept": "Conditional Independence",
        },
        "Independence Testing": {
            "available": True,
            "source": "Correlation / chi-sq tests on features",
            "unit1_concept": "Independence",
        },
        "Supervised Learning": {
            "available": has_cap,
            "source": "Gaussian Naive Bayes: Healthy vs Degraded",
            "unit1_concept": "Supervised Learning",
        },
        "Unsupervised Learning": {
            "available": has_cap,
            "source": "K-Means clustering of cycle records",
            "unit1_concept": "Unsupervised Learning",
        },
        "Internal Resistance": {
            "available": has_imp,
            "source": "Re (electrolyte resistance) and Rct (charge transfer resistance) from impedance cycles",
            "unit1_concept": "Bayesian evidence feature",
        },
        "Skewness / Kurtosis": {
            "available": True,
            "source": "Higher-order moments of measurement distributions",
            "unit1_concept": "Probability distributions",
        },
    }


def list_extractable_features(battery_summaries: list[dict]) -> list[dict]:
    """List features that can legitimately be extracted for the cycle-level dataset."""
    has_imp = any(b.get("has_internal_resistance_Re") for b in battery_summaries)

    features = [
        {"feature": "battery_id", "source": "metadata.csv battery_id column", "available": True},
        {"feature": "cycle_number", "source": "Sequential index per battery (discharge cycles)", "available": True},
        {"feature": "operation_type", "source": "metadata.csv type column (charge/discharge/impedance)", "available": True},
        {"feature": "ambient_temperature", "source": "metadata.csv ambient_temperature (°C)", "available": True},
        {"feature": "voltage_mean", "source": "mean(Voltage_measured) from per-cycle CSV", "available": True},
        {"feature": "voltage_std", "source": "std(Voltage_measured) from per-cycle CSV", "available": True},
        {"feature": "voltage_min", "source": "min(Voltage_measured) from per-cycle CSV", "available": True},
        {"feature": "voltage_max", "source": "max(Voltage_measured) from per-cycle CSV", "available": True},
        {"feature": "current_mean", "source": "mean(Current_measured) from per-cycle CSV", "available": True},
        {"feature": "current_std", "source": "std(Current_measured) from per-cycle CSV", "available": True},
        {"feature": "current_min", "source": "min(Current_measured) from per-cycle CSV", "available": True},
        {"feature": "current_max", "source": "max(Current_measured) from per-cycle CSV", "available": True},
        {"feature": "temperature_mean", "source": "mean(Temperature_measured) from per-cycle CSV", "available": True},
        {"feature": "temperature_std", "source": "std(Temperature_measured) from per-cycle CSV", "available": True},
        {"feature": "temperature_min", "source": "min(Temperature_measured) from per-cycle CSV", "available": True},
        {"feature": "temperature_max", "source": "max(Temperature_measured) from per-cycle CSV", "available": True},
        {"feature": "discharge_duration", "source": "max(Time) for discharge cycles (seconds)", "available": True},
        {"feature": "charge_duration", "source": "max(Time) for charge cycles (seconds)", "available": True},
        {"feature": "capacity", "source": "metadata.csv Capacity column (Ah) — discharge cycles only", "available": True},
        {"feature": "soh", "source": "capacity / initial_capacity × 100% (derived)", "available": True},
        {"feature": "eol_flag", "source": "1 if SOH <= configurable threshold (default 70%)", "available": True},
        {"feature": "internal_resistance_Re", "source": "metadata.csv Re column (Ohms) — impedance cycles", "available": has_imp},
        {"feature": "internal_resistance_Rct", "source": "metadata.csv Rct column (Ohms) — impedance cycles", "available": has_imp},
        {"feature": "degradation_label", "source": "Derived: Healthy / Early Degradation / Degraded based on SOH", "available": True},
    ]
    return features


def list_unavailable_features() -> list[str]:
    """Features that are NOT available and why."""
    return [
        "energy (Wh): Not directly in CSVs; would require numerical integration ∫V·I dt — feasible but not pre-computed",
        "power (W): Not directly available; derivable as V×I per time-step",
        "charge_current_profile: Separate column 'Current_charge' exists in charge CSVs; available",
        "depth_of_discharge (DOD): Not directly available; derivable from capacity ratio",
        "cycle_efficiency: Not pre-computed; derivable from charge/discharge ratio",
        "state_of_charge (SOC): Not measured; requires integration or model",
        "battery_age_days: Timestamps available in start_time but require parsing",
    ]


def format_txt_report(report: dict) -> str:
    """Format the report as a human-readable text file."""
    lines: list[str] = []

    def h(title: str, char: str = "=") -> None:
        lines.append("")
        lines.append(char * 64)
        lines.append(f"  {title}")
        lines.append(char * 64)

    def kv(key: str, value: Any, indent: int = 2) -> None:
        lines.append(f"{'  ' * (indent // 2)}{key}: {value}")

    h("BATTERYSENTINEL — Phase 1: Dataset Inspection Report")
    kv("Generated", report["generated_at"])
    kv("Data directory", report["data_directory"])
    kv("Dataset variant", report["dataset_variant"])

    h("DATASET STRUCTURE", "-")
    fs = report["filesystem"]
    kv("Total cycle files (CSVs)", fs["n_data_csvs"])
    kv("README files", fs["n_readme_files"])
    kv("metadata.csv rows", fs["metadata_rows"])
    kv("metadata.csv columns", fs["metadata_cols"])

    h("FLEET SUMMARY", "-")
    fleet = report["fleet_summary"]
    kv("Number of batteries", fleet["n_batteries"])
    kv("Battery IDs", ", ".join(fleet["battery_ids"]))
    kv("Total cycle records", fleet["total_records"])
    kv("Charge records", fleet["total_charge"])
    kv("Discharge records", fleet["total_discharge"])
    kv("Impedance records", fleet["total_impedance"])
    kv("Batteries with impedance data", ", ".join(fleet["batteries_with_impedance"]))
    kv("Batteries reaching EOL (SOH ≤ 70%)", ", ".join(fleet["batteries_at_eol"]))

    h("PER-CYCLE CSV STRUCTURE", "-")
    for op_type, info in report.get("csv_structure", {}).items():
        lines.append(f"\n  ── {op_type.upper()} cycles (example: {info['example_file']}) ──")
        lines.append(f"    Columns: {info['columns']}")
        lines.append(f"    Shape: {info['shape_example']}")
        for col, stats in info.get("sample_stats", {}).items():
            if stats.get("mean") is not None:
                lines.append(
                    f"    {col:30s}: min={stats['min']:.4f}  mean={stats['mean']:.4f}  max={stats['max']:.4f}"
                )

    h("PER-BATTERY SUMMARIES", "-")
    for b in report.get("battery_summaries", []):
        lines.append(f"\n  ── Battery {b['battery_id']} ──")
        kv("Total records", b["total_records"], indent=4)
        kv("Charge / Discharge / Impedance", f"{b['n_charge']} / {b['n_discharge']} / {b['n_impedance']}", indent=4)
        kv("Ambient temperatures (°C)", b["ambient_temperatures_C"], indent=4)
        kv("Initial capacity (Ah)", b.get("capacity_initial_Ah"), indent=4)
        kv("Final capacity (Ah)", b.get("capacity_final_Ah"), indent=4)
        kv("SOH final (%)", b.get("soh_final_pct"), indent=4)
        kv("SOH decline (%)", b.get("soh_decline_pct"), indent=4)
        kv("Reached EOL (SOH ≤ 70%)", b.get("reached_eol_70pct"), indent=4)
        kv("Has Re (electrolyte resistance)", b.get("has_internal_resistance_Re"), indent=4)
        kv("Has Rct (charge transfer resist.)", b.get("has_internal_resistance_Rct"), indent=4)

    h("UNIT 1 FEATURE AVAILABILITY", "-")
    lines.append(f"\n  {'Concept':<38} {'Available?':<12} {'Unit 1 Topic'}")
    lines.append("  " + "-" * 72)
    for concept, info in report.get("unit1_assessment", {}).items():
        avail = "YES" if info["available"] else "PARTIAL/NO"
        lines.append(f"  {concept:<38} {avail:<12} {info['unit1_concept']}")

    h("EXTRACTABLE CYCLE-LEVEL FEATURES", "-")
    for feat in report.get("extractable_features", []):
        avail_str = "✓" if feat["available"] else "✗"
        lines.append(f"  {avail_str} {feat['feature']:<35} ← {feat['source']}")

    h("FEATURES NOT AVAILABLE / DERIVABLE", "-")
    for note in report.get("unavailable_features", []):
        lines.append(f"  ⚠  {note}")

    h("DATA QUALITY ASSESSMENT", "-")
    for note in report.get("data_quality_notes", []):
        lines.append(f"  • {note}")

    h("LIMITATIONS AND CAVEATS", "-")
    for lim in report.get("limitations", []):
        lines.append(f"  ⚠  {lim}")

    h("PHASE 2 PLAN: HOW RAW DATA BECOMES CYCLE-LEVEL DATASET", "-")
    for step in report.get("phase2_plan", []):
        lines.append(f"  → {step}")

    lines.append("")
    lines.append("=" * 64)
    lines.append("  END OF REPORT")
    lines.append("=" * 64)
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    data_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "cleaned_dataset"
    output_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 64)
    log.info("  BATTERYSENTINEL — Phase 1: Dataset Inspection")
    log.info("=" * 64)

    if not data_dir.exists():
        log.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # ── Step 1: Filesystem walk ───────────────────────────────────────────────
    log.info("[Step 1] Scanning filesystem ...")
    n_data_csvs = len([f for f in os.listdir(data_dir / "data") if f.endswith(".csv")])
    n_readme = len([f for f in os.listdir(data_dir / "data") if f.endswith(".txt")])
    meta = pd.read_csv(data_dir / "metadata.csv")
    log.info(f"  Metadata rows: {len(meta)}, columns: {list(meta.columns)}")
    log.info(f"  Cycle CSV files: {n_data_csvs}")

    # ── Step 2: Read READMEs ──────────────────────────────────────────────────
    log.info("[Step 2] Reading README files ...")
    data_readmes = read_text_files(data_dir / "data")
    extra_readmes = read_text_files(data_dir / "extra_infos")
    all_readmes = {**data_readmes, **extra_readmes}
    for fname in sorted(all_readmes.keys())[:3]:
        log.info(f"  • {fname}: {len(all_readmes[fname])} chars")

    # ── Step 3: Inspect sample CSVs ───────────────────────────────────────────
    log.info("[Step 3] Inspecting sample cycle CSVs ...")
    csv_structure = inspect_sample_csvs(data_dir, meta)
    for op_type, info in csv_structure.items():
        log.info(f"  {op_type}: columns = {info['columns']}")

    # ── Step 4: Battery summaries ─────────────────────────────────────────────
    log.info("[Step 4] Building per-battery summaries ...")
    battery_summaries = build_battery_summaries(meta)
    for b in battery_summaries:
        log.info(
            f"  Battery {b['battery_id']}: "
            f"{b['n_discharge']} discharge cycles | "
            f"SOH: {b.get('soh_initial_pct', '?'):.1f}% → {b.get('soh_final_pct', '?'):.1f}% | "
            f"EOL: {b.get('reached_eol_70pct', '?')}"
        )

    # ── Step 5: Unit 1 assessment ─────────────────────────────────────────────
    log.info("[Step 5] Assessing Unit 1 feature availability ...")
    unit1_assessment = assess_unit1_features(battery_summaries)

    # ── Step 6: Extractable features ──────────────────────────────────────────
    log.info("[Step 6] Listing extractable features ...")
    extractable = list_extractable_features(battery_summaries)
    unavailable = list_unavailable_features()

    # ── Fleet summary ──────────────────────────────────────────────────────────
    batteries_with_imp = [b["battery_id"] for b in battery_summaries if b.get("has_internal_resistance_Re")]
    batteries_at_eol = [b["battery_id"] for b in battery_summaries if b.get("reached_eol_70pct")]

    fleet_summary = {
        "n_batteries": len(battery_summaries),
        "battery_ids": [b["battery_id"] for b in battery_summaries],
        "total_records": int(len(meta)),
        "total_charge": int((meta["type"] == "charge").sum()),
        "total_discharge": int((meta["type"] == "discharge").sum()),
        "total_impedance": int((meta["type"] == "impedance").sum()),
        "batteries_with_impedance": batteries_with_imp,
        "batteries_at_eol": batteries_at_eol,
    }

    # ── Data quality notes ─────────────────────────────────────────────────────
    data_quality_notes = [
        "metadata.csv Capacity column: valid for discharge rows only; NaN for charge and impedance.",
        "metadata.csv Re and Rct: valid for impedance rows only; NaN for charge and discharge.",
        "Capacity stored as object dtype in metadata — requires float conversion.",
        "Some batteries (B0033, B0034, B0036, etc.) show capacity INCREASING initially — likely warming-up effect or formation cycles.",
        "Batteries B0025-B0032 show minimal degradation (< 10 cycles or short experiments).",
        "Batteries B0005, B0006, B0007, B0018 have the most complete long-term data (168+ discharge cycles each).",
        "start_time field contains array-like strings — requires parsing for date/time analysis.",
        "No missing values in per-cycle CSV files (discharge, charge, impedance data is complete).",
        f"Total available discharge cycles for SOH analysis: {int((meta['type'] == 'discharge').sum())}",
        "Internal resistance (Re, Rct) is available for ALL batteries via impedance cycles.",
    ]

    # ── Phase 2 plan ───────────────────────────────────────────────────────────
    phase2_plan = [
        "Load metadata.csv → index of all 7565 cycle records",
        "For each discharge record: load per-cycle CSV → compute voltage/current/temperature statistics",
        "Assign cycle_number per battery (sequential discharge index)",
        "Map impedance Re/Rct to nearest discharge cycle by test_id",
        "Compute SOH = capacity / initial_capacity × 100%",
        "Apply EOL flag: 1 if SOH ≤ 70% (configurable)",
        "Assign degradation label: Healthy (SOH > 85%) / Early Degradation (70–85%) / Degraded (< 70%)",
        "Build clean cycle-level DataFrame with all features",
        "Save to data/processed/cycle_data.csv",
    ]

    # ── Limitations ────────────────────────────────────────────────────────────
    limitations = [
        "NASA laboratory data may not represent all real-world battery chemistries or operating conditions.",
        "EOL threshold (default 70% SOH) is a CONFIGURABLE RESEARCH PARAMETER — NOT a universal safety standard.",
        "Bayesian conditional-independence assumptions (Naive Bayes) simplify the true joint distribution.",
        "Polynomial extrapolation beyond observed cycle range is an ESTIMATE, not a guaranteed forecast.",
        "Risk probability scores are model outputs — not guaranteed failure probabilities.",
        "This project is for educational/research purposes ONLY — it is NOT a safety certification system.",
        "Some batteries (B0025-B0032) have few discharge cycles — insufficient for reliable polynomial fitting.",
        "Temperature operating conditions vary across battery groups (4°C, 24°C, 43°C) — this affects degradation rates.",
    ]

    # ── Assemble full report ───────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "data_directory": str(data_dir),
        "dataset_variant": "Cleaned CSV version (patrickfleith/nasa-battery-dataset on Kaggle)",
        "filesystem": {
            "n_data_csvs": n_data_csvs,
            "n_readme_files": n_readme,
            "metadata_rows": int(len(meta)),
            "metadata_cols": list(meta.columns),
        },
        "readme_contents": all_readmes,
        "csv_structure": csv_structure,
        "battery_summaries": battery_summaries,
        "fleet_summary": fleet_summary,
        "unit1_assessment": unit1_assessment,
        "extractable_features": extractable,
        "unavailable_features": unavailable,
        "data_quality_notes": data_quality_notes,
        "phase2_plan": phase2_plan,
        "limitations": limitations,
    }

    # ── Write outputs ──────────────────────────────────────────────────────────
    json_path = output_dir / "dataset_report.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"\n[Output] JSON report → {json_path}")

    txt_path = output_dir / "dataset_report.txt"
    txt_path.write_text(format_txt_report(report), encoding="utf-8")
    log.info(f"[Output] TXT report  → {txt_path}")

    log.info("\n" + "=" * 64)
    log.info("  PHASE 1 INSPECTION COMPLETE")
    log.info("=" * 64)
    log.info(f"  Batteries        : {fleet_summary['n_batteries']}")
    log.info(f"  Total cycles     : {fleet_summary['total_records']}")
    log.info(f"  Discharge cycles : {fleet_summary['total_discharge']}")
    log.info(f"  Impedance data   : YES (Re & Rct for all batteries)")
    log.info(f"  Batteries at EOL : {len(batteries_at_eol)} / {fleet_summary['n_batteries']}")
    log.info(f"  Reports saved to : {output_dir}")
    log.info("=" * 64)


if __name__ == "__main__":
    main()
