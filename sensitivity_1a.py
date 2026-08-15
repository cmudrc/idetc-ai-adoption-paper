"""
Layer 1 full-grid local sensitivity analysis for steady-state total adoption.

This script computes local sensitivity of:
    A_inf = steady_Q + steady_L

with respect to the four surrogate inputs:
    - teams_num
    - teams_size
    - agents_average_initial_opinion
    - technology_success_rate

across the full unique design grid from config.py.

Sensitivity metrics
-------------------
For each scenario and each input, this script computes:

1. Raw derivative:
       dA/dx

2. Elasticity (dimensionless, local, point-based):
       E_x = (x / A) * dA/dx
   Computed only when meaningful:
       - |A| > eps
       - |x| > eps

3. Range-normalized sensitivity (dimensionless-ish practical influence metric):
       R_x = (dA/dx) * (x_max - x_min)

Why both?
---------
- Elasticity tells you proportional local influence.
- Range-normalized sensitivity tells you practical influence across the feasible range.

Important note
--------------
We do NOT include seed in this layer, because the surrogate model is deterministic
in the four inputs above and does not take seed as an input.

Outputs
-------
Writes:
    results/layer1_fullgrid/layer1_fullgrid_adoption_sensitivity.csv
    results/layer1_fullgrid/layer1_fullgrid_adoption_sensitivity_summary_by_input.csv
    results/layer1_fullgrid/layer1_fullgrid_adoption_sensitivity_summary_by_input_and_regime.csv
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd

from config import (
    teams_num_list,
    teams_size_list,
    agents_average_initial_opinion_list,
    technology_success_rate_list,
)

try:
    from config import BASE_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent

from predict_surrogate import predict_steady_state_ratios


# ============================================================================
# User settings
# ============================================================================

INPUT_KEYS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]

RESULTS_DIR = BASE_DIR / "sensitivity" / "layer1_fullgrid"

# Finite-difference steps.
# You can later change these if you want a step-size robustness check.
STEPS = {
    "teams_num": 1.0,
    "teams_size": 1.0,
    "agents_average_initial_opinion": 0.02,
    "technology_success_rate": 0.02,
}

# Bounds inferred from the full design grid.
BOUNDS = {
    "teams_num": (
        float(min(teams_num_list)),
        float(max(teams_num_list)),
    ),
    "teams_size": (
        float(min(teams_size_list)),
        float(max(teams_size_list)),
    ),
    "agents_average_initial_opinion": (
        float(min(agents_average_initial_opinion_list)),
        float(max(agents_average_initial_opinion_list)),
    ),
    "technology_success_rate": (
        float(min(technology_success_rate_list)),
        float(max(technology_success_rate_list)),
    ),
}

# Full feasible ranges used for range-normalized sensitivity.
INPUT_RANGES = {
    key: float(BOUNDS[key][1] - BOUNDS[key][0])
    for key in INPUT_KEYS
}

# Small threshold for numerical validity checks
EPS = 1e-12


# ============================================================================
# Data container
# ============================================================================

@dataclass
class AdoptionSensitivityRecord:
    teams_num: float
    teams_size: float
    agents_average_initial_opinion: float
    technology_success_rate: float

    input_key: str
    base_input_value: float
    input_min: float
    input_max: float
    input_range: float
    step: float

    x_minus: float
    x_plus: float

    steady_adoption: float
    adoption_minus: float
    adoption_plus: float

    derivative_dA_dx: float
    elasticity: float
    elasticity_defined: bool

    range_normalized_sensitivity: float

    used_asymmetric_stencil: bool
    note: str
    regime: str


# ============================================================================
# Helper functions
# ============================================================================

def clip_to_bounds(value: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return float(np.clip(value, lo, hi))


def choose_perturbation_pair(
    x0: float,
    h: float,
    bounds: Tuple[float, float],
) -> Tuple[float, float, bool, str]:
    """
    Prefer centered difference. If the perturbation hits a boundary,
    use the resulting asymmetric stencil.
    """
    x_minus = clip_to_bounds(x0 - h, bounds)
    x_plus = clip_to_bounds(x0 + h, bounds)

    if math.isclose(x_minus, x_plus, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(
            f"Cannot perturb around x0={x0} with h={h} inside bounds={bounds}"
        )

    used_asymmetric = (
        not math.isclose(x_minus, x0 - h, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(x_plus, x0 + h, rel_tol=0.0, abs_tol=1e-15)
    )
    note = "asymmetric due to bounds" if used_asymmetric else "centered"

    return x_minus, x_plus, used_asymmetric, note


def compute_derivative(
    y_minus: float,
    y_plus: float,
    x_minus: float,
    x_plus: float,
) -> float:
    dx = x_plus - x_minus
    if abs(dx) < 1e-15:
        raise ValueError("Cannot compute derivative because x_plus == x_minus")
    return (y_plus - y_minus) / dx


def compute_elasticity(
    x0: float,
    A0: float,
    dA_dx: float,
    eps: float = EPS,
) -> Tuple[float, bool]:
    """
    Elasticity:
        E = (x / A) * dA/dx

    Only defined when |x| and |A| are both sufficiently away from zero.
    """
    if abs(x0) <= eps or abs(A0) <= eps:
        return float("nan"), False
    return (x0 / A0) * dA_dx, True


def compute_range_normalized_sensitivity(
    dA_dx: float,
    input_range: float,
) -> float:
    """
    Range-normalized sensitivity:
        R = (dA/dx) * (x_max - x_min)
    """
    return dA_dx * input_range


def classify_regime(adoption: float) -> str:
    """
    Simple regime labels based on steady adoption.
    """
    if adoption <= 0.2:
        return "low_adoption"
    if adoption >= 0.8:
        return "high_adoption"
    return "mixed"


def evaluate_adoption(
    point: Dict[str, float],
    surrogate_path: Optional[Path] = None,
) -> float:
    """
    Evaluate steady-state total adoption:
        A_inf = steady_Q + steady_L
    """
    raw = predict_steady_state_ratios(
        teams_num=point["teams_num"],
        teams_size=point["teams_size"],
        agents_average_initial_opinion=point["agents_average_initial_opinion"],
        technology_success_rate=point["technology_success_rate"],
        surrogate_path=surrogate_path,
    )
    return float(raw["Q"] + raw["L"])


def build_full_unique_grid() -> List[Dict[str, float]]:
    grid: List[Dict[str, float]] = []

    for teams_num in teams_num_list:
        for teams_size in teams_size_list:
            for opinion in agents_average_initial_opinion_list:
                for success in technology_success_rate_list:
                    grid.append(
                        {
                            "teams_num": float(teams_num),
                            "teams_size": float(teams_size),
                            "agents_average_initial_opinion": float(opinion),
                            "technology_success_rate": float(success),
                        }
                    )

    return grid


# ============================================================================
# Core analysis
# ============================================================================

def compute_local_adoption_sensitivity_at_point(
    base_point: Dict[str, float],
    surrogate_path: Optional[Path] = None,
) -> List[AdoptionSensitivityRecord]:
    """
    Compute local sensitivity of steady adoption with respect to all 4 inputs
    at one point in the design space.
    """
    base_adoption = evaluate_adoption(base_point, surrogate_path=surrogate_path)
    regime = classify_regime(base_adoption)

    records: List[AdoptionSensitivityRecord] = []

    for input_key in INPUT_KEYS:
        x0 = float(base_point[input_key])
        h = float(STEPS[input_key])

        x_min, x_max = BOUNDS[input_key]
        input_range = INPUT_RANGES[input_key]

        x_minus, x_plus, used_asymmetric, note = choose_perturbation_pair(
            x0=x0,
            h=h,
            bounds=BOUNDS[input_key],
        )

        point_minus = dict(base_point)
        point_plus = dict(base_point)
        point_minus[input_key] = x_minus
        point_plus[input_key] = x_plus

        adoption_minus = evaluate_adoption(point_minus, surrogate_path=surrogate_path)
        adoption_plus = evaluate_adoption(point_plus, surrogate_path=surrogate_path)

        dA_dx = compute_derivative(
            y_minus=adoption_minus,
            y_plus=adoption_plus,
            x_minus=x_minus,
            x_plus=x_plus,
        )

        elasticity, elasticity_defined = compute_elasticity(
            x0=x0,
            A0=base_adoption,
            dA_dx=dA_dx,
        )

        range_norm = compute_range_normalized_sensitivity(
            dA_dx=dA_dx,
            input_range=input_range,
        )

        records.append(
            AdoptionSensitivityRecord(
                teams_num=float(base_point["teams_num"]),
                teams_size=float(base_point["teams_size"]),
                agents_average_initial_opinion=float(base_point["agents_average_initial_opinion"]),
                technology_success_rate=float(base_point["technology_success_rate"]),
                input_key=input_key,
                base_input_value=x0,
                input_min=float(x_min),
                input_max=float(x_max),
                input_range=float(input_range),
                step=h,
                x_minus=float(x_minus),
                x_plus=float(x_plus),
                steady_adoption=float(base_adoption),
                adoption_minus=float(adoption_minus),
                adoption_plus=float(adoption_plus),
                derivative_dA_dx=float(dA_dx),
                elasticity=float(elasticity),
                elasticity_defined=bool(elasticity_defined),
                range_normalized_sensitivity=float(range_norm),
                used_asymmetric_stencil=bool(used_asymmetric),
                note=note,
                regime=regime,
            )
        )

    return records


# ============================================================================
# Summaries
# ============================================================================

def summarize_by_input(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["abs_derivative_dA_dx"] = work["derivative_dA_dx"].abs()
    work["abs_range_normalized_sensitivity"] = work["range_normalized_sensitivity"].abs()

    # Only use valid elasticities
    elast = work[work["elasticity_defined"]].copy()
    elast["abs_elasticity"] = elast["elasticity"].abs()

    deriv_summary = (
        work.groupby("input_key", dropna=False)
        .agg(
            n_total=("input_key", "size"),
            mean_abs_derivative=("abs_derivative_dA_dx", "mean"),
            median_abs_derivative=("abs_derivative_dA_dx", "median"),
            max_abs_derivative=("abs_derivative_dA_dx", "max"),
            mean_abs_range_normalized=("abs_range_normalized_sensitivity", "mean"),
            median_abs_range_normalized=("abs_range_normalized_sensitivity", "median"),
            max_abs_range_normalized=("abs_range_normalized_sensitivity", "max"),
        )
        .reset_index()
    )

    if len(elast) > 0:
        elast_summary = (
            elast.groupby("input_key", dropna=False)
            .agg(
                n_elasticity_defined=("input_key", "size"),
                mean_abs_elasticity=("abs_elasticity", "mean"),
                median_abs_elasticity=("abs_elasticity", "median"),
                max_abs_elasticity=("abs_elasticity", "max"),
            )
            .reset_index()
        )
        summary = deriv_summary.merge(elast_summary, on="input_key", how="left")
    else:
        summary = deriv_summary.copy()
        summary["n_elasticity_defined"] = 0
        summary["mean_abs_elasticity"] = np.nan
        summary["median_abs_elasticity"] = np.nan
        summary["max_abs_elasticity"] = np.nan

    summary = summary.sort_values("mean_abs_range_normalized", ascending=False)
    return summary


def summarize_by_input_and_regime(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["abs_derivative_dA_dx"] = work["derivative_dA_dx"].abs()
    work["abs_range_normalized_sensitivity"] = work["range_normalized_sensitivity"].abs()

    elast = work[work["elasticity_defined"]].copy()
    elast["abs_elasticity"] = elast["elasticity"].abs()

    deriv_summary = (
        work.groupby(["regime", "input_key"], dropna=False)
        .agg(
            n_total=("input_key", "size"),
            mean_abs_derivative=("abs_derivative_dA_dx", "mean"),
            median_abs_derivative=("abs_derivative_dA_dx", "median"),
            max_abs_derivative=("abs_derivative_dA_dx", "max"),
            mean_abs_range_normalized=("abs_range_normalized_sensitivity", "mean"),
            median_abs_range_normalized=("abs_range_normalized_sensitivity", "median"),
            max_abs_range_normalized=("abs_range_normalized_sensitivity", "max"),
        )
        .reset_index()
    )

    if len(elast) > 0:
        elast_summary = (
            elast.groupby(["regime", "input_key"], dropna=False)
            .agg(
                n_elasticity_defined=("input_key", "size"),
                mean_abs_elasticity=("abs_elasticity", "mean"),
                median_abs_elasticity=("abs_elasticity", "median"),
                max_abs_elasticity=("abs_elasticity", "max"),
            )
            .reset_index()
        )
        summary = deriv_summary.merge(
            elast_summary,
            on=["regime", "input_key"],
            how="left",
        )
    else:
        summary = deriv_summary.copy()
        summary["n_elasticity_defined"] = 0
        summary["mean_abs_elasticity"] = np.nan
        summary["median_abs_elasticity"] = np.nan
        summary["max_abs_elasticity"] = np.nan

    summary = summary.sort_values(
        ["regime", "mean_abs_range_normalized"],
        ascending=[True, False],
    )
    return summary


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    surrogate_path = None
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    grid = build_full_unique_grid()
    print(f"Unique surrogate scenarios: {len(grid)}")

    all_records: List[AdoptionSensitivityRecord] = []

    for i, point in enumerate(grid, start=1):
        if i == 1 or i % 250 == 0 or i == len(grid):
            print(f"Running scenario {i} / {len(grid)}")

        records = compute_local_adoption_sensitivity_at_point(
            base_point=point,
            surrogate_path=surrogate_path,
        )
        all_records.extend(records)

    df = pd.DataFrame(asdict(r) for r in all_records)

    full_results_path = RESULTS_DIR / "layer1_fullgrid_adoption_sensitivity.csv"
    df.to_csv(full_results_path, index=False)

    summary_by_input = summarize_by_input(df)
    summary_by_input_path = RESULTS_DIR / "layer1_fullgrid_adoption_sensitivity_summary_by_input.csv"
    summary_by_input.to_csv(summary_by_input_path, index=False)

    summary_by_input_and_regime = summarize_by_input_and_regime(df)
    summary_by_input_and_regime_path = RESULTS_DIR / "layer1_fullgrid_adoption_sensitivity_summary_by_input_and_regime.csv"
    summary_by_input_and_regime.to_csv(summary_by_input_and_regime_path, index=False)

    print(f"\nSaved full results to: {full_results_path}")
    print(f"Saved summary by input to: {summary_by_input_path}")
    print(f"Saved summary by input and regime to: {summary_by_input_and_regime_path}")

    print("\n=== Mean absolute RANGE-normalized sensitivity ranking ===")
    print(summary_by_input.to_string(index=False))

    print("\n=== Mean absolute ELASTICITY ranking (only where defined) ===")
    elast_view = summary_by_input[
        ["input_key", "n_elasticity_defined", "mean_abs_elasticity", "median_abs_elasticity", "max_abs_elasticity"]
    ].sort_values("mean_abs_elasticity", ascending=False)
    print(elast_view.to_string(index=False))