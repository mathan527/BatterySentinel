"""
BatterySentinel — Experiment 03: Polynomial Degradation Model
=============================================================
Fits polynomial SOH curves for all batteries.
Compares degrees 1–4 by R² and RMSE.
Predicts EOL cycle for each battery.
"""

from __future__ import annotations
import json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("experiment_03")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config
from src.polynomial_model import (
    fit_polynomial, compare_polynomial_degrees,
    predict_eol_cycle, fit_fleet_polynomials,
)

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
RES_DIR = PROJECT_ROOT / "reports" / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
CMAP = cm.get_cmap("tab10")


def plot_degree_comparison(cycles, soh, bid, results, eol_threshold, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: fit curves
    ax = axes[0]
    ax.scatter(cycles, soh, color="#bdc3c7", s=15, alpha=0.7, zorder=2, label="Observed SOH")
    x_ext = np.linspace(cycles.min(), cycles.max() * 1.3, 400)
    colors = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
    for (deg, res), col in zip(sorted(results.items()), colors):
        poly_fn = res["poly_fn"]
        y_fit = poly_fn(x_ext)
        ax.plot(x_ext, y_fit, linewidth=2, color=col, alpha=0.85,
                label=f"deg={deg}  R²={res['r_squared']:.3f}")
    ax.axhline(eol_threshold, color="#e74c3c", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"EOL {eol_threshold:.0f}%")
    ax.set_xlabel("Cycle Number", fontsize=11)
    ax.set_ylabel("SOH (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_title(f"Battery {bid} — Polynomial Fits (degrees 1–4)", fontsize=11)
    ax.legend(fontsize=9)

    # Right: RMSE vs degree
    ax2 = axes[1]
    degrees = sorted(results.keys())
    r2_vals = [results[d]["r_squared"] for d in degrees]
    rmse_vals = [results[d]["rmse"] for d in degrees]
    ax2.bar(degrees, r2_vals, color="#3498db", alpha=0.7, label="R²")
    ax2r = ax2.twinx()
    ax2r.plot(degrees, rmse_vals, "ro--", linewidth=1.5, markersize=7, label="RMSE")
    ax2.set_xlabel("Polynomial Degree", fontsize=11)
    ax2.set_ylabel("R² (higher = better)", fontsize=11, color="#3498db")
    ax2r.set_ylabel("RMSE (lower = better)", fontsize=11, color="#e74c3c")
    ax2.set_title(f"Battery {bid} — Degree Selection\n"
                  "R² vs RMSE trade-off (bias-variance)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2r.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    path = FIG_DIR / f"14_polynomial_{bid}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def plot_fleet_summary(fleet_df, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² distribution
    ax = axes[0]
    ax.hist(fleet_df["r_squared"].dropna(), bins=20, color="#3498db",
            alpha=0.7, edgecolor="white")
    ax.axvline(fleet_df["r_squared"].mean(), color="#e74c3c", linestyle="--",
               linewidth=2, label=f"Mean R²={fleet_df['r_squared'].mean():.3f}")
    ax.set_xlabel("R² (degree-2 polynomial)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Fleet Polynomial Fit Quality Distribution", fontsize=11)
    ax.legend()

    # Cycles remaining
    ax2 = axes[1]
    eol_df = fleet_df.dropna(subset=["cycles_remaining"]).sort_values("final_soh")
    colors = ["#e74c3c" if s < 70 else "#f39c12" if s < 85 else "#2ecc71"
              for s in eol_df["final_soh"]]
    bars = ax2.barh(eol_df["battery_id"], eol_df["cycles_remaining"],
                    color=colors, alpha=0.8)
    ax2.set_xlabel("Estimated Cycles Remaining to EOL", fontsize=11)
    ax2.set_title("EOL Prediction — Cycles Remaining\n"
                  "(Polynomial extrapolation — research estimate only)", fontsize=11)
    ax2.grid(axis="x", alpha=0.4)

    fig.tight_layout()
    path = FIG_DIR / "15_fleet_polynomial_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path}")


def main():
    cfg = load_config()
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["processed_dir"] / "cycle_data.csv")
    eol_pct = cfg["soh"]["eol_threshold"] * 100

    log.info("[1] Polynomial degree comparison for key batteries")
    focus_bids = ["B0005", "B0006", "B0007", "B0018"]
    degree_results_all = {}

    for bid in focus_bids:
        bdf = df[df["battery_id"] == bid].sort_values("cycle_number")
        soh = bdf["soh"].dropna().values
        cyc = bdf.loc[bdf["soh"].notna(), "cycle_number"].values.astype(float)
        if len(soh) < 5:
            continue

        results = compare_polynomial_degrees(cyc, soh, degrees=[1, 2, 3, 4])
        degree_results_all[bid] = {
            d: {k: v for k, v in res.items() if k != "poly_fn"}
            for d, res in results.items()
        }
        log.info(f"  {bid}: " + " | ".join(
            f"deg={d} R²={r['r_squared']:.3f} RMSE={r['rmse']:.3f}"
            for d, r in sorted(results.items())
        ))
        plot_degree_comparison(cyc, soh, bid, results, eol_pct)

    log.info("\n[2] Fleet polynomial fits (degree=2)")
    fleet_df = fit_fleet_polynomials(df, eol_threshold=eol_pct, best_degree=2, min_cycles=20)
    fleet_df.to_csv(RES_DIR / "polynomial_fleet_summary.csv", index=False)
    log.info(f"  Batteries fitted: {len(fleet_df)}")
    log.info(f"  Mean R²: {fleet_df['r_squared'].mean():.4f}")
    log.info(f"  Batteries with EOL prediction: {fleet_df['estimated_eol_cycle'].notna().sum()}")

    for _, row in fleet_df.iterrows():
        cr = row["cycles_remaining"]
        log.info(f"  {row['battery_id']}: R²={row['r_squared']:.3f}  "
                 f"final_SOH={row['final_soh']:.1f}%  "
                 f"cycles_remaining={cr if cr is not None else 'N/A'}")

    plot_fleet_summary(fleet_df)

    log.info("\n[3] Detailed EOL prediction for focus batteries")
    for bid in focus_bids:
        bdf = df[df["battery_id"] == bid].sort_values("cycle_number")
        soh = bdf["soh"].dropna().values
        cyc = bdf.loc[bdf["soh"].notna(), "cycle_number"].values.astype(float)
        if len(soh) < 5:
            continue
        fit = fit_polynomial(cyc, soh, degree=2)
        if "error" in fit:
            continue
        eol = predict_eol_cycle(fit["poly_fn"], int(cyc[-1]), eol_threshold=eol_pct)
        log.info(f"  {bid}: {eol['note']}")

    with open(RES_DIR / "polynomial_results.json", "w", encoding="utf-8") as fh:
        json.dump({
            "degree_comparison": degree_results_all,
            "fleet_summary_path": "polynomial_fleet_summary.csv",
        }, fh, indent=2, default=str)

    log.info("\n" + "=" * 60)
    log.info("  EXPERIMENT 03: Polynomial Model COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
