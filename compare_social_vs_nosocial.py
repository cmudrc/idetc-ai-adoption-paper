"""
Compare the social model with the no-social-layer baseline.

What this file does:
1. Loads `settings.csv` and, for every setting, reads the SQLB trajectory
   with the social layer (`states/{name}.csv`, from `state.py`) and without it
   (`states_nosocial/{name}.csv`, from `run_nosocial.py`).
2. Takes adoption = Q + L at the final recorded step for both versions and
   the paired difference (social - no social) for each run.
3. Averages over seeds (pairs share the same setting name, so the same
   network, initial opinions, initial trust, and technology seed) and saves
   tables to `results/nosocial/`:
       runs.csv          one row per run: both versions and the difference
       summary.csv       seed mean / min / max per configuration
       min_accuracy.csv  lowest AI accuracy reaching TARGET_ADOPTION
       structure_average.csv  results averaged over the 16 structures
4. Prints sanity checks of the baseline.
5. Saves two figures to `figures/`:
       figure_social_effect_map.pdf
           change in adoption from adding the social layer, by initial
           opinion x AI accuracy (averaged over structures and seeds)
       figure_social_vs_nosocial_curves.pdf
           adoption vs AI accuracy with and without the social layer for
           negative, neutral, and positive initial opinion, with the
           most that can be adopted without the social layer

Notes:
- The comparison uses ABM trajectories directly, not the CTMC/surrogate
  steady state. Without the social layer the burned state is absorbing
  (burned agents stop using the AI and nothing else can change their opinion),
  so a fitted steady state would be zero adoption.
- In social runs the stored t = 0 opinions already include one round of
  social mixing (the first DeGroot step overwrites them); in the baseline they
  are the raw initial opinions. The comparison therefore uses the final step.

How it is used in the pipeline:
- Run after `run.py` + `state.py` (social) and `run_nosocial.py` (baseline).
- Supplies the "contribution of the social layer" result for the paper.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from utils import BASE_DIR, load_settings

STATES_DIR = BASE_DIR / "states"
STATES_NOSOCIAL_DIR = BASE_DIR / "states_nosocial"
RESULTS_DIR = BASE_DIR / "results" / "nosocial"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Analysis settings
# =========================
TARGET_ADOPTION = 0.5

# Figure 1 (effect map): accuracies shown (below 0.5 adoption is zero in both versions).
MAP_MIN_ACCURACY = 0.5

# Figure 2 (curves): one panel per initial opinion. Values must be on the ABM grid.
CURVE_INITIAL_OPINIONS = [-0.3, 0.0, 0.3]
CURVE_PANEL_TITLES = ["Negative initial opinion", "Neutral initial opinion", "Positive initial opinion"]
CURVE_TITLE = "Adoption with and without peer influence"

# =========================
# Figure style settings
# =========================
MAP_FIG_WIDTH = 5.75  # was 7.5; heatmap columns about 30% narrower
MAP_FIG_HEIGHT = 8.5
CURVE_FIG_WIDTH = 7.0   # inches; drawn at its printed size (1.0\textwidth), so CURVE_FONT_* are printed sizes
CURVE_FIG_HEIGHT = 2.8

FONT_TITLE = 15
FONT_SUBTITLE = 13
FONT_AXIS_LABEL = 13
FONT_TICK = 11
FONT_LEGEND = 12
FONT_CELL = 8.5

# Curves figure (printed sizes, matching the vehicle figures in the paper)
CURVE_FONT_TITLE = 9.5
CURVE_FONT_SUBTITLE = 8.5
CURVE_FONT_AXIS_LABEL = 8
CURVE_FONT_TICK = 7.5
CURVE_FONT_LEGEND = 7.5
CURVE_LINEWIDTH = 1.5
CURVE_MARKER_SIZE = 3.5

LINEWIDTH_MAIN = 2.0
MARKER_SIZE = 5
BAND_ALPHA = 0.18
GRID_ALPHA = 0.3

COLOR_SOCIAL = "#0072B2"     # blue
COLOR_NOSOCIAL = "#D55E00"   # vermillion
COLOR_REFERENCE = "0.45"     # gray reference lines
COLOR_INK = "#1a1a1a"
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "social_effect", [COLOR_NOSOCIAL, "#f0efec", COLOR_SOCIAL]
)


# =========================
# Loading
# =========================
def read_states(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_runs() -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per setting with final-step states of both versions."""
    rows = []
    checks = {"missing_social": 0, "missing_nosocial": 0, "burned_decreased": 0}

    for st in load_settings():
        name = st["name"]
        social_path = STATES_DIR / f"{name}.csv"
        nosocial_path = STATES_NOSOCIAL_DIR / f"{name}.csv"
        if not social_path.exists():
            checks["missing_social"] += 1
            continue
        if not nosocial_path.exists():
            checks["missing_nosocial"] += 1
            continue

        social = read_states(social_path)
        nosocial = read_states(nosocial_path)

        # Check: without the social layer, burned agents can never recover.
        b = np.array([r["ratio_B"] for r in nosocial])
        if np.any(np.diff(b) < -1e-12):
            checks["burned_decreased"] += 1

        s_end, n_end, n_start = social[-1], nosocial[-1], nosocial[0]
        rows.append(
            {
                "name": name,
                "seed_index": int(name.split("_")[0]),
                "agents_average_initial_opinion": round(float(st["agents_average_initial_opinion"]), 1),
                "technology_success_rate": round(float(st["technology_success_rate"]), 1),
                "teams_num": int(st["teams_num"]),
                "teams_size": int(st["teams_size"]),
                "social_S": s_end["ratio_S"],
                "social_Q": s_end["ratio_Q"],
                "social_L": s_end["ratio_L"],
                "social_B": s_end["ratio_B"],
                "nosocial_S": n_end["ratio_S"],
                "nosocial_Q": n_end["ratio_Q"],
                "nosocial_L": n_end["ratio_L"],
                "nosocial_B": n_end["ratio_B"],
                "initial_not_burned": 1.0 - n_start["ratio_B"],                 # opinion >= -0.2
                "initial_adoption": n_start["ratio_Q"] + n_start["ratio_L"],    # opinion >= 0
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["adoption_social"] = df["social_Q"] + df["social_L"]
        df["adoption_nosocial"] = df["nosocial_Q"] + df["nosocial_L"]
        df["adoption_difference"] = df["adoption_social"] - df["adoption_nosocial"]
    return df, checks


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["agents_average_initial_opinion", "technology_success_rate", "teams_num", "teams_size"]
    values = ["adoption_social", "adoption_nosocial", "adoption_difference", "initial_not_burned", "initial_adoption"]
    g = runs.groupby(keys)[values].agg(["mean", "min", "max"])
    g.columns = [f"{v}_{stat}" for v, stat in g.columns]
    g["n_seeds"] = runs.groupby(keys).size()
    return g.reset_index()


def min_accuracy_for_target(acc: np.ndarray, adoption: np.ndarray, target: float) -> float:
    """First accuracy where the seed-mean curve reaches `target` (linear interpolation)."""
    order = np.argsort(acc)
    acc, adoption = acc[order], adoption[order]
    for i in range(len(acc)):
        if adoption[i] >= target:
            if i == 0:
                return float(acc[0])
            a0, a1 = adoption[i - 1], adoption[i]
            return float(acc[i - 1] + (target - a0) * (acc[i] - acc[i - 1]) / (a1 - a0))
    return float("nan")


def min_accuracy_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["agents_average_initial_opinion", "teams_num", "teams_size"]
    for (opinion, tn, ts), grp in summary.groupby(keys):
        acc = grp["technology_success_rate"].to_numpy(dtype=float)
        rows.append(
            {
                "agents_average_initial_opinion": opinion,
                "teams_num": tn,
                "teams_size": ts,
                "min_accuracy_social": min_accuracy_for_target(
                    acc, grp["adoption_social_mean"].to_numpy(dtype=float), TARGET_ADOPTION
                ),
                "min_accuracy_nosocial": min_accuracy_for_target(
                    acc, grp["adoption_nosocial_mean"].to_numpy(dtype=float), TARGET_ADOPTION
                ),
                "max_adoption_nosocial": float(grp["adoption_nosocial_mean"].max()),
                "initial_not_burned_mean": float(grp["initial_not_burned_mean"].mean()),
                "initial_adoption_mean": float(grp["initial_adoption_mean"].mean()),
            }
        )
    return pd.DataFrame(rows)


# =========================
# Sanity checks
# =========================
def print_checks(runs: pd.DataFrame, summary: pd.DataFrame, checks: dict[str, int]) -> None:
    print("\n=== Sanity checks ===")
    print(f"Runs compared: {len(runs)}")
    print(f"Missing social states: {checks['missing_social']}  |  missing baseline states: {checks['missing_nosocial']}")
    print(f"[1] Baseline runs where the burned share ever decreased (expected 0): {checks['burned_decreased']}")

    perfect = runs[np.isclose(runs["technology_success_rate"], 1.0)]
    if not perfect.empty:
        # At accuracy 1.0 every agent still using the AI ends as an adopter, so final
        # baseline adoption equals the initial share of agents above the usage cutoff.
        gap_02 = (perfect["adoption_nosocial"] - perfect["initial_not_burned"]).abs().max()
        gap_00 = (perfect["adoption_nosocial"] - perfect["initial_adoption"]).abs().max()
        print("[2] At accuracy 1.0, baseline adoption vs initial share above the usage cutoff:")
        print(f"      cutoff -0.2 (opinion >= -0.2): max |gap| = {gap_02:.4f}")
        print(f"      cutoff  0.0 (opinion >=  0.0): max |gap| = {gap_00:.4f}")
        if gap_00 < 1e-9 <= gap_02:
            print("      -> agents stop using the AI below 0.0, not -0.2 (see the trustdynamics cutoff note).")
        elif gap_02 < 1e-9:
            print("      -> agents stop using the AI below -0.2, as configured.")

    keys = ["agents_average_initial_opinion", "technology_success_rate"]
    spread = summary.groupby(keys)["adoption_nosocial_mean"].agg(lambda x: x.max() - x.min())
    large = summary[summary["teams_num"] * summary["teams_size"] >= 100]
    spread_large = large.groupby(keys)["adoption_nosocial_mean"].agg(lambda x: x.max() - x.min())
    print(
        "[3] Baseline adoption should not depend on structure (only sampling noise): "
        f"max spread across structures = {spread.max():.3f} (all), "
        f"{spread_large.max():.3f} (organizations with >= 100 agents)"
    )


# =========================
# Figures
# =========================
def structure_average(summary: pd.DataFrame) -> pd.DataFrame:
    """Average seed-mean results over the 16 organizational structures (equal weight)."""
    keys = ["agents_average_initial_opinion", "technology_success_rate"]
    cols = ["adoption_social_mean", "adoption_nosocial_mean", "adoption_difference_mean", "initial_not_burned_mean"]
    mean = summary.groupby(keys)[cols].mean()
    low = summary.groupby(keys)[["adoption_social_mean", "adoption_nosocial_mean"]].min().add_suffix("_lowest_structure")
    high = summary.groupby(keys)[["adoption_social_mean", "adoption_nosocial_mean"]].max().add_suffix("_highest_structure")
    return pd.concat([mean, low, high], axis=1).reset_index()


def plot_effect_map(avg: pd.DataFrame, save_name: str) -> None:
    """
    Figure 1: change in adoption from adding the social layer, by initial opinion
    and AI accuracy (averaged over structures and seeds).
    Blue: the social layer raises adoption. Vermillion: it lowers adoption.
    """
    d = avg[avg["technology_success_rate"] >= MAP_MIN_ACCURACY - 1e-9]
    grid = d.pivot(index="agents_average_initial_opinion", columns="technology_success_rate",
                   values="adoption_difference_mean").sort_index()
    opinions = grid.index.to_numpy(dtype=float)
    accuracies = grid.columns.to_numpy(dtype=float)
    Z = grid.to_numpy(dtype=float)
    vmax = max(0.05, float(np.ceil(np.nanmax(np.abs(Z)) * 10) / 10))

    fig, ax = plt.subplots(figsize=(MAP_FIG_WIDTH, MAP_FIG_HEIGHT))
    im = ax.imshow(Z, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z[i, j]
            text = "0" if abs(v) < 0.005 else f"{v:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=FONT_CELL,
                    color="white" if abs(v) > 0.6 * vmax else COLOR_INK)

    ax.set_xticks(range(len(accuracies)))
    ax.set_xticklabels([f"{a:.1f}" for a in accuracies], fontsize=FONT_TICK)
    ax.set_yticks(range(len(opinions)))
    ax.set_yticklabels([f"{o:+.1f}" if abs(o) > 1e-9 else "0.0" for o in opinions], fontsize=FONT_TICK)
    ax.set_xlabel("AI accuracy", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Mean initial opinion toward AI", fontsize=FONT_AXIS_LABEL)
    ax.set_title("Change in adoption from peer influence\n(with − without, averaged over structures)",
                 fontsize=FONT_TITLE, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("Adoption with − without peer influence (Q + L)", fontsize=FONT_AXIS_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    plt.tight_layout()
    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {save_path}")


def plot_adoption_curves(avg: pd.DataFrame, save_name: str) -> None:
    """
    Figure 2: adoption vs AI accuracy with and without the social layer.
    Lines: average over structures. Bands: lowest to highest structure.
    Dashed gray: the most the organization can adopt without the social layer
    (the share of engineers who start above the usage cutoff).
    """
    n = len(CURVE_INITIAL_OPINIONS)
    fig, axes = plt.subplots(1, n, figsize=(CURVE_FIG_WIDTH, CURVE_FIG_HEIGHT), sharey=True, squeeze=False)
    axes = axes[0]

    for k, opinion in enumerate(CURVE_INITIAL_OPINIONS):
        ax = axes[k]
        d = avg[np.isclose(avg["agents_average_initial_opinion"], opinion)].sort_values("technology_success_rate")
        if d.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", color=COLOR_INK, transform=ax.transAxes)
            continue
        x = d["technology_success_rate"].to_numpy(dtype=float)

        for prefix, color, style, label in [
            ("adoption_social", COLOR_SOCIAL, "-", "With peer influence"),
            ("adoption_nosocial", COLOR_NOSOCIAL, "--", "Without peer influence"),
        ]:
            ax.fill_between(x, d[f"{prefix}_mean_lowest_structure"], d[f"{prefix}_mean_highest_structure"],
                            color=color, alpha=BAND_ALPHA, linewidth=0)
            ax.plot(x, d[f"{prefix}_mean"], linestyle=style, color=color, linewidth=CURVE_LINEWIDTH,
                    marker="o", markersize=CURVE_MARKER_SIZE, label=label)

        ceiling = float(d["initial_not_burned_mean"].mean())
        ax.axhline(ceiling, color=COLOR_REFERENCE, linestyle=(0, (5, 3)), linewidth=1.0,
                   label="Maximum without peer influence")

        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True, alpha=GRID_ALPHA)
        ax.tick_params(axis="both", labelsize=CURVE_FONT_TICK)
        ax.set_title(CURVE_PANEL_TITLES[k], fontsize=CURVE_FONT_SUBTITLE)
        ax.set_xlabel("AI accuracy", fontsize=CURVE_FONT_AXIS_LABEL)
        if k == 0:
            ax.set_ylabel("Adoption (Q + L) at final step", fontsize=CURVE_FONT_AXIS_LABEL)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=CURVE_FONT_LEGEND, handlelength=2.2, columnspacing=1.5)
    fig.suptitle(CURVE_TITLE, fontsize=CURVE_FONT_TITLE, y=0.995)
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    fig.subplots_adjust(top=0.84)  # keep the title close to the panel titles

    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {save_path}")


def main() -> None:
    runs, checks = load_runs()
    if runs.empty:
        print("No paired runs found. Run run_nosocial.py first (and state.py for the social runs).")
        return

    summary = summarize(runs)
    min_acc = min_accuracy_table(summary)
    avg = structure_average(summary)

    runs.to_csv(RESULTS_DIR / "runs.csv", index=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    min_acc.to_csv(RESULTS_DIR / "min_accuracy.csv", index=False)
    avg.to_csv(RESULTS_DIR / "structure_average.csv", index=False)
    print(f"Saved tables to: {RESULTS_DIR}")

    print_checks(runs, summary, checks)

    print(f"\n=== Lowest accuracy reaching {int(TARGET_ADOPTION * 100)}% adoption (NaN = never reached) ===")
    show = min_acc[min_acc["agents_average_initial_opinion"].isin(CURVE_INITIAL_OPINIONS)]
    with pd.option_context("display.max_rows", 200, "display.width", 140):
        print(show.round(3).to_string(index=False))

    plot_effect_map(avg, "figure_social_effect_map.pdf")
    plot_adoption_curves(avg, "figure_social_vs_nosocial_curves.pdf")


if __name__ == "__main__":
    main()
