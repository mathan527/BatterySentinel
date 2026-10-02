"""
BatterySentinel — src/polynomial_model.py
==========================================
Polynomial degradation curve fitting for SOH vs cycle.

Unit 1 Concepts Covered
------------------------
  • Polynomial Curve Fitting
  • Least Squares Estimation
  • Bias–Variance Tradeoff (underfitting vs overfitting)
  • Residual Analysis
  • Model Selection (compare degrees 1, 2, 3, 4)
  • End-of-Life Cycle Prediction
  • Extrapolation uncertainty

Mathematical Background
-----------------------
Fit a polynomial of degree d:

    SOH(n) = w_0 + w_1*n + w_2*n^2 + ... + w_d*n^d

Equivalently in matrix form (Vandermonde):

    y = X_vand @ w

where X_vand[i,j] = n_i^j  for j=0,...,d

Solved by least squares:
    w* = (X^T X)^{-1} X^T y   (normal equations)
    or via np.polyfit / np.polynomial.polynomial.Polynomial.fit

Quality metrics:
    R^2 = 1 - SS_res / SS_tot         (coefficient of determination)
    RMSE = sqrt(mean((y - y_hat)^2))   (root mean squared error)
    SS_res = sum((y_i - y_hat_i)^2)
    SS_tot = sum((y_i - mean(y))^2)

Degree Selection:
    Degree 1 — linear trend (simplest, may underfit)
    Degree 2 — quadratic (captures acceleration of degradation)
    Degree 3 — cubic (more flexible; risk of overfitting short series)
    Compare by R^2, RMSE, and visual residual inspection.

EOL Prediction:
    Extrapolate polynomial to find n* where SOH(n*) = EOL threshold (70%)
    Report as ESTIMATED cycle number — NOT a guaranteed date.
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
# Polynomial fitting for a single battery
# ─────────────────────────────────────────────────────────────────────────────

def fit_polynomial(
    cycles: np.ndarray,
    soh: np.ndarray,
    degree: int = 2,
) -> dict[str, Any]:
    """
    Fit a polynomial of given degree to SOH vs cycle data.

    Formula:
        SOH(n) = sum_{j=0}^{d} w_j * n^j

    Parameters
    ----------
    cycles : np.ndarray of cycle numbers
    soh    : np.ndarray of SOH values (%)
    degree : polynomial degree (1=linear, 2=quadratic, 3=cubic)

    Returns
    -------
    dict with:
        coefficients : array, highest power first (numpy convention)
        degree       : int
        r_squared    : float
        rmse         : float
        predictions  : np.ndarray of fitted values at training cycles
        poly_fn      : callable — polynomial function p(n)
    """
    # Remove NaNs
    mask = ~(np.isnan(cycles) | np.isnan(soh))
    x, y = cycles[mask], soh[mask]

    if len(x) < degree + 1:
        log.warning(f"  Not enough points ({len(x)}) to fit degree-{degree} polynomial.")
        return {"error": f"Insufficient data: {len(x)} points for degree {degree}"}

    # Fit polynomial (coefficients: highest power first)
    coeffs = np.polyfit(x, y, degree)
    poly_fn = np.poly1d(coeffs)

    y_hat = poly_fn(x)
    residuals = y - y_hat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "degree": degree,
        "coefficients": coeffs.tolist(),
        "r_squared": round(r2, 6),
        "rmse": round(rmse, 6),
        "predictions": y_hat.tolist(),
        "residuals": residuals.tolist(),
        "cycles_used": x.tolist(),
        "soh_used": y.tolist(),
        "poly_fn": poly_fn,  # callable — not serialised to JSON
    }


def compare_polynomial_degrees(
    cycles: np.ndarray,
    soh: np.ndarray,
    degrees: list[int] | None = None,
) -> dict[int, dict]:
    """
    Fit polynomials of multiple degrees and compare metrics.

    Returns a dict {degree: fit_result} for each degree tested.

    Use R^2 and RMSE to select the best degree:
      - Higher R^2 = better fit (max 1.0)
      - Lower RMSE = smaller average error
      - Higher degree does not always win — may overfit on small datasets
    """
    if degrees is None:
        degrees = [1, 2, 3, 4]

    results: dict[int, dict] = {}
    for d in degrees:
        result = fit_polynomial(cycles, soh, degree=d)
        if "error" not in result:
            results[d] = result
            log.debug(f"    degree={d}: R2={result['r_squared']:.4f}  RMSE={result['rmse']:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EOL cycle prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_eol_cycle(
    poly_fn,
    current_cycle: int,
    eol_threshold: float = 70.0,
    max_extrapolate: int = 500,
) -> dict[str, Any]:
    """
    Predict the cycle at which SOH is estimated to reach the EOL threshold.

    Method: Evaluate the fitted polynomial at future cycle numbers until
    SOH drops to or below eol_threshold.

    Important Caveats:
      1. Extrapolation beyond observed data range has HIGH uncertainty.
      2. Polynomial fits may curve back up or produce nonsensical values
         far beyond training data — inspect visually.
      3. This is a TREND ESTIMATE, not a guaranteed failure date.
      4. EOL threshold is a configurable research parameter.

    Parameters
    ----------
    poly_fn       : callable np.poly1d object from fit_polynomial
    current_cycle : int — last observed cycle
    eol_threshold : float — SOH % at EOL (default 70%)
    max_extrapolate : int — how many future cycles to search

    Returns
    -------
    dict with estimated_eol_cycle and cycles_remaining.
    """
    future_cycles = np.arange(current_cycle, current_cycle + max_extrapolate + 1)
    future_soh = poly_fn(future_cycles)

    below_eol = np.where(future_soh <= eol_threshold)[0]

    if len(below_eol) == 0:
        return {
            "estimated_eol_cycle": None,
            "cycles_remaining": None,
            "note": (
                f"SOH did not reach {eol_threshold}% within {max_extrapolate} "
                f"extrapolated cycles. Battery may outlast the prediction window."
            ),
        }

    eol_idx = int(below_eol[0])
    eol_cycle = int(future_cycles[eol_idx])
    cycles_remaining = eol_cycle - current_cycle

    return {
        "estimated_eol_cycle": eol_cycle,
        "cycles_remaining": cycles_remaining,
        "soh_at_eol": round(float(future_soh[eol_idx]), 4),
        "note": (
            f"ESTIMATED: SOH reaches {eol_threshold}% at cycle {eol_cycle} "
            f"({cycles_remaining} cycles from now). "
            "This is a polynomial extrapolation — verify with physical inspection."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fleet-level analysis
# ─────────────────────────────────────────────────────────────────────────────

def fit_fleet_polynomials(
    df: pd.DataFrame,
    eol_threshold: float = 70.0,
    best_degree: int = 2,
    min_cycles: int = 20,
) -> pd.DataFrame:
    """
    Fit polynomial degradation curves for all batteries with enough data.

    Parameters
    ----------
    df            : cycle-level DataFrame
    eol_threshold : SOH % threshold for EOL prediction
    best_degree   : polynomial degree to use for fleet summary
    min_cycles    : minimum discharge cycles required to fit

    Returns
    -------
    pd.DataFrame — one row per battery with columns:
        battery_id, n_cycles, best_degree, r_squared, rmse,
        estimated_eol_cycle, cycles_remaining, initial_soh, final_soh
    """
    records: list[dict] = []

    for bid, group in df.groupby("battery_id"):
        group_sorted = group.sort_values("cycle_number")
        cycles = group_sorted["cycle_number"].values.astype(float)
        soh = group_sorted["soh"].dropna().values.astype(float)
        cyc_clean = group_sorted.loc[group_sorted["soh"].notna(), "cycle_number"].values.astype(float)

        if len(soh) < min_cycles:
            log.warning(f"  Battery {bid}: only {len(soh)} valid SOH points — skipping.")
            continue

        fit = fit_polynomial(cyc_clean, soh, degree=best_degree)
        if "error" in fit:
            continue

        eol_pred = predict_eol_cycle(
            fit["poly_fn"],
            current_cycle=int(cyc_clean[-1]),
            eol_threshold=eol_threshold,
        )

        records.append({
            "battery_id": bid,
            "n_cycles": int(len(soh)),
            "degree": best_degree,
            "r_squared": fit["r_squared"],
            "rmse": fit["rmse"],
            "initial_soh": round(float(soh[0]), 2),
            "final_soh": round(float(soh[-1]), 2),
            "estimated_eol_cycle": eol_pred.get("estimated_eol_cycle"),
            "cycles_remaining": eol_pred.get("cycles_remaining"),
            "eol_note": eol_pred.get("note", ""),
        })

    return pd.DataFrame(records)
