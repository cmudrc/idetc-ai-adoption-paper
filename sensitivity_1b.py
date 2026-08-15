"""
Layer 1B: Full-grid local sensitivity of finite-horizon adoption to
initial SQLB fractions.

This script is analogous to Layer 1A, but for initial-state composition.

Target output:
    A(T) = Q(T) + L(T)

Dynamics:
    x(T) = x0 expm(Q T)

where:
    - Q is predicted by the surrogate from the scenario inputs
    - x0 = [S0, Q0, L0, B0] is the initial SQLB composition
    - T is the finite horizon (default: config.steps)

Why this exists
---------------
Initial fractions S0, Q0, L0, B0 are compositional:
    S0 + Q0 + L0 + B0 = 1

So they cannot be perturbed independently in the ordinary sense.
This script uses a simplex-preserving one-at-a-time perturbation rule:

To perturb one component upward or downward, the other three are adjusted
proportionally to their current masses so the vector remains feasible.

Outputs
-------
Writes:
    results/layer1b/layer1b_initial_fraction_sensitivity.csv
    results/layer1b/layer1b_initial_fraction_sensitivity_summary_by_input.csv
    results/layer1b/layer1b_initial_fraction_sensitivity_summary_by_input_and_regime.csv
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

try:
    from config import steps as HORIZON_T
except ImportError:
    HORIZON_T = 100

from predict_surrogate import load_surrogate_model, predict_generator, STATE_ORDER


# =============================================================================
# User settings
# =============================================================================

RESULTS_DIR = BASE_DIR / "results" / "layer1b"

SCENARIO_KEYS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]

INITIAL_KEYS = [
    "initial_S",
    "initial_Q",
    "initial_L",
    "initial_B",
]

STATE_INDEX = {name: i for i, name in enumerate(STATE_ORDER)}

# Use an interior baseline for meaningful sensitivities of all four fractions.
# Replace with your best-estimate organizational baseline if you have one.
BASE_INITIAL_STATE = np.array([0.70, 0.10, 0.10, 0.10], dtype=float)

# Finite-difference step for the initial fractions.
DELTA_H = 0.02

EPS = 1e-12


# =============================================================================
# Matrix exponential helper
# =============================================================================

def matrix_exponential(M: np.ndarray) -> np.ndarray:
    try:
        from scipy.linalg import expm
        return expm(M)
    except Exception:
        vals, vecs = np.linalg.eig(M)
        inv_vecs = np.linalg.inv(vecs)
        exp_vals = np.diag(np.exp(vals))
        out = vecs @ exp_vals @ inv_vecs
        out = np.real_if_close(out, tol=1000)
        return np.asarray(out, dtype=float)


# =============================================================================
# Data container
# =============================================================================

@dataclass
class InitialFractionSensitivityRecord:
    teams_num: float
    teams_size: float
    agents_average_initial_opinion: float
    technology_success_rate: float

    horizon_T: float

    base_S0: float
    base_Q0: float
    base_L0: float
    base_B0: float

    input_key: str
    base_input_value: float
    input_min: float
    input_max: float
    input_range: float
    step: float

    h_minus: float
    h_plus: float

    steady_adoption_T: float
    adoption_minus: float
    adoption_plus: float

    derivative_dA_dx0: float
    elasticity: float
    elasticity_defined: bool

    range_normalized_sensitivity: float

    used_asymmetric_stencil: bool
    note: str
    regime: str


# =============================================================================
# Helpers
# =============================================================================

def normalize_initial_state(x0: np.ndarray) -> np.ndarray:
    x = np.asarray(x0, dtype=float).copy()
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= 0:
        raise ValueError("Initial state must have positive sum.")
    x /= s
    return x


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


def compute_state_at_time(Q: np.ndarray, x0: np.ndarray, T: float) -> np.ndarray:
    """
    Row-vector evolution:
        x(T) = x0 expm(Q T)
    """
    P = matrix_exponential(Q * float(T))
    xT = np.asarray(x0, dtype=float) @ P
    xT = np.real_if_close(xT, tol=1000)
    xT = np.asarray(xT, dtype=float)
    xT = np.clip(xT, 0.0, None)
    s = xT.sum()
    if s > 0:
        xT /= s
    return xT


def evaluate_adoption_at_horizon(
    model,
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    initial_state: np.ndarray,
    horizon_T: float,
) -> float:
    Q, _ = predict_generator(
        model,
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
    )

    xT = compute_state_at_time(Q, initial_state, horizon_T)
    idx_Q = STATE_INDEX["Q"]
    idx_L = STATE_INDEX["L"]
    return float(xT[idx_Q] + xT[idx_L])


def classify_regime(adoption: float) -> str:
    if adoption <= 0.2:
        return "low_adoption"
    if adoption >= 0.8:
        return "high_adoption"
    return "mixed"


def compute_elasticity(
    x0_value: float,
    A0: float,
    dA_dx: float,
    eps: float = EPS,
) -> Tuple[float, bool]:
    if abs(x0_value) <= eps or abs(A0) <= eps:
        return float("nan"), False
    return (x0_value / A0) * dA_dx, True


def compute_derivative(
    y_minus: float,
    y_plus: float,
    x_minus: float,
    x_plus: float,
) -> float:
    dx = x_plus - x_minus
    if abs(dx) < 1e-15:
        raise ValueError("Cannot compute derivative because denominator is zero.")
    return (y_plus - y_minus) / dx


def perturb_component_proportionally(
    x0: np.ndarray,
    *,
    target_idx: int,
    delta: float,
) -> np.ndarray:
    """
    Increase/decrease one component by delta and redistribute the opposite amount
    proportionally across the other components, preserving simplex feasibility.

    If delta > 0:
        x_target' = x_target + delta
        other components lose total delta proportionally to their current masses.

    If delta < 0:
        x_target' = x_target + delta
        other components gain total -delta proportionally to their current masses.

    If the three "other" components sum to zero, redistribution is done equally
    among them.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = len(x)

    if target_idx < 0 or target_idx >= n:
        raise IndexError("Invalid target index.")

    x_target = x[target_idx]

    # Feasibility for target component itself
    if x_target + delta < -1e-12 or x_target + delta > 1 + 1e-12:
        raise ValueError("Requested perturbation violates simplex feasibility.")

    others = [i for i in range(n) if i != target_idx]
    other_mass = float(x[others].sum())

    x[target_idx] = x_target + delta

    if abs(delta) <= 1e-15:
        return normalize_initial_state(x)

    if delta > 0:
        # Need to subtract delta from the others
        if other_mass <= 1e-15:
            raise ValueError("Cannot increase target component; no mass available in others.")
        for i in others:
            share = x0[i] / other_mass
            x[i] = x0[i] - delta * share
    else:
        # Need to add -delta to the others
        increment = -delta
        if other_mass > 1e-15:
            for i in others:
                share = x0[i] / other_mass
                x[i] = x0[i] + increment * share
        else:
            # Only possible if target was 1 and others 0; split equally
            equal_share = increment / len(others)
            for i in others:
                x[i] = x0[i] + equal_share

    if np.any(x < -1e-10):
        raise ValueError("Perturbation produced negative mass.")
    x = np.clip(x, 0.0, None)
    return normalize_initial_state(x)


def choose_feasible_step_interval(
    x0: np.ndarray,
    *,
    target_idx: int,
    requested_h: float,
) -> Tuple[float, float, float, bool, str]:
    """
    Determine a feasible interval [h_minus, h_plus] around 0 for perturbing
    a target component while preserving simplex feasibility.

    Since each component is a fraction in [0,1], a simple conservative interval is:
        h in [-x_target, 1 - x_target]
    """
    x_target = float(x0[target_idx])
    h_min = -x_target
    h_max = 1.0 - x_target

    h_minus = max(-requested_h, h_min)
    h_plus = min(requested_h, h_max)

    if math.isclose(h_minus, h_plus, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("No feasible perturbation interval at this point.")

    used_asymmetric = (
        not math.isclose(h_minus, -requested_h, rel_tol=0.0, abs_tol=1e-15)
        or not math.isclose(h_plus, requested_h, rel_tol=0.0, abs_tol=1e-15)
    )
    note = "asymmetric due to simplex boundary" if used_asymmetric else "centered"

    # Global range of each initial fraction is [0,1]
    input_range = 1.0

    return h_minus, h_plus, input_range, used_asymmetric, note


# =============================================================================
# Core analysis
# =============================================================================

def compute_initial_fraction_sensitivity_at_point(
    model,
    scenario: Dict[str, float],
    *,
    base_initial_state: np.ndarray,
    horizon_T: float,
    requested_h: float,
) -> List[InitialFractionSensitivityRecord]:
    x0 = normalize_initial_state(base_initial_state)

    base_adoption = evaluate_adoption_at_horizon(
        model,
        teams_num=scenario["teams_num"],
        teams_size=scenario["teams_size"],
        agents_average_initial_opinion=scenario["agents_average_initial_opinion"],
        technology_success_rate=scenario["technology_success_rate"],
        initial_state=x0,
        horizon_T=horizon_T,
    )

    regime = classify_regime(base_adoption)

    records: List[InitialFractionSensitivityRecord] = []

    for state_name in STATE_ORDER:
        target_idx = STATE_INDEX[state_name]
        input_key = f"initial_{state_name}"
        base_value = float(x0[target_idx])

        h_minus, h_plus, input_range, used_asymmetric, note = choose_feasible_step_interval(
            x0,
            target_idx=target_idx,
            requested_h=requested_h,
        )

        x_minus = perturb_component_proportionally(
            x0,
            target_idx=target_idx,
            delta=h_minus,
        )
        x_plus = perturb_component_proportionally(
            x0,
            target_idx=target_idx,
            delta=h_plus,
        )

        adoption_minus = evaluate_adoption_at_horizon(
            model,
            teams_num=scenario["teams_num"],
            teams_size=scenario["teams_size"],
            agents_average_initial_opinion=scenario["agents_average_initial_opinion"],
            technology_success_rate=scenario["technology_success_rate"],
            initial_state=x_minus,
            horizon_T=horizon_T,
        )

        adoption_plus = evaluate_adoption_at_horizon(
            model,
            teams_num=scenario["teams_num"],
            teams_size=scenario["teams_size"],
            agents_average_initial_opinion=scenario["agents_average_initial_opinion"],
            technology_success_rate=scenario["technology_success_rate"],
            initial_state=x_plus,
            horizon_T=horizon_T,
        )

        dA_dx0 = compute_derivative(
            y_minus=adoption_minus,
            y_plus=adoption_plus,
            x_minus=h_minus,
            x_plus=h_plus,
        )

        elasticity, elasticity_defined = compute_elasticity(
            x0_value=base_value,
            A0=base_adoption,
            dA_dx=dA_dx0,
        )

        # Since the global range of each initial fraction is [0,1], this equals dA_dx0 numerically.
        range_norm = dA_dx0 * input_range

        records.append(
            InitialFractionSensitivityRecord(
                teams_num=float(scenario["teams_num"]),
                teams_size=float(scenario["teams_size"]),
                agents_average_initial_opinion=float(scenario["agents_average_initial_opinion"]),
                technology_success_rate=float(scenario["technology_success_rate"]),
                horizon_T=float(horizon_T),
                base_S0=float(x0[STATE_INDEX["S"]]),
                base_Q0=float(x0[STATE_INDEX["Q"]]),
                base_L0=float(x0[STATE_INDEX["L"]]),
                base_B0=float(x0[STATE_INDEX["B"]]),
                input_key=input_key,
                base_input_value=base_value,
                input_min=0.0,
                input_max=1.0,
                input_range=float(input_range),
                step=float(requested_h),
                h_minus=float(h_minus),
                h_plus=float(h_plus),
                steady_adoption_T=float(base_adoption),
                adoption_minus=float(adoption_minus),
                adoption_plus=float(adoption_plus),
                derivative_dA_dx0=float(dA_dx0),
                elasticity=float(elasticity),
                elasticity_defined=bool(elasticity_defined),
                range_normalized_sensitivity=float(range_norm),
                used_asymmetric_stencil=bool(used_asymmetric),
                note=note,
                regime=regime,
            )
        )

    return records


# =============================================================================
# Summaries
# =============================================================================

def summarize_by_input(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["abs_derivative_dA_dx0"] = work["derivative_dA_dx0"].abs()
    work["abs_range_normalized_sensitivity"] = work["range_normalized_sensitivity"].abs()

    elast = work[work["elasticity_defined"]].copy()
    elast["abs_elasticity"] = elast["elasticity"].abs()

    deriv_summary = (
        work.groupby("input_key", dropna=False)
        .agg(
            n_total=("input_key", "size"),
            mean_abs_derivative=("abs_derivative_dA_dx0", "mean"),
            median_abs_derivative=("abs_derivative_dA_dx0", "median"),
            max_abs_derivative=("abs_derivative_dA_dx0", "max"),
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

    work["abs_derivative_dA_dx0"] = work["derivative_dA_dx0"].abs()
    work["abs_range_normalized_sensitivity"] = work["range_normalized_sensitivity"].abs()

    elast = work[work["elasticity_defined"]].copy()
    elast["abs_elasticity"] = elast["elasticity"].abs()

    deriv_summary = (
        work.groupby(["regime", "input_key"], dropna=False)
        .agg(
            n_total=("input_key", "size"),
            mean_abs_derivative=("abs_derivative_dA_dx0", "mean"),
            median_abs_derivative=("abs_derivative_dA_dx0", "median"),
            max_abs_derivative=("abs_derivative_dA_dx0", "max"),
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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_surrogate_model(None)
    grid = build_full_unique_grid()

    print(f"Unique surrogate scenarios: {len(grid)}")
    print(f"Horizon T: {HORIZON_T}")
    print(f"Base initial state [S,Q,L,B]: {BASE_INITIAL_STATE.tolist()}")

    all_records: List[InitialFractionSensitivityRecord] = []

    for i, scenario in enumerate(grid, start=1):
        if i == 1 or i % 250 == 0 or i == len(grid):
            print(f"Running scenario {i} / {len(grid)}")

        recs = compute_initial_fraction_sensitivity_at_point(
            model,
            scenario,
            base_initial_state=BASE_INITIAL_STATE,
            horizon_T=HORIZON_T,
            requested_h=DELTA_H,
        )
        all_records.extend(recs)

    df = pd.DataFrame(asdict(r) for r in all_records)

    full_path = RESULTS_DIR / "layer1b_initial_fraction_sensitivity.csv"
    df.to_csv(full_path, index=False)

    summary_df = summarize_by_input(df)
    summary_path = RESULTS_DIR / "layer1b_initial_fraction_sensitivity_summary_by_input.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_regime_df = summarize_by_input_and_regime(df)
    summary_regime_path = RESULTS_DIR / "layer1b_initial_fraction_sensitivity_summary_by_input_and_regime.csv"
    summary_regime_df.to_csv(summary_regime_path, index=False)

    print(f"\nSaved full results to: {full_path}")
    print(f"Saved summary by input to: {summary_path}")
    print(f"Saved summary by input and regime to: {summary_regime_path}")

    print("\n=== Mean absolute RANGE-normalized sensitivity ranking ===")
    print(summary_df.to_string(index=False))

    print("\n=== Mean absolute ELASTICITY ranking (only where defined) ===")
    elast_view = summary_df[
        ["input_key", "n_elasticity_defined", "mean_abs_elasticity", "median_abs_elasticity", "max_abs_elasticity"]
    ].sort_values("mean_abs_elasticity", ascending=False)
    print(elast_view.to_string(index=False))