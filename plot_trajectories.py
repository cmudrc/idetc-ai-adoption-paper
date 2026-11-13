"""
Plot SQLB trajectories from ABM against reduced-order ODE models.

What this module does:
- Loads ABM state-ratio trajectories from `states/{model_name}.csv`.
- Loads fitted ODE generators from `odes/{model_name}.npz`.
- Optionally predicts surrogate ODE generators from `settings.csv` + surrogate model.
- Simulates ODE trajectories with explicit Euler from the same ABM initial state.
- Produces overlays for:
  1) ABM only
  2) ABM vs fitted ODE
  3) ABM vs surrogate ODE

How it is used in the pipeline:
- Provides qualitative/visual validation figures to compare reduced models against ABM
  trajectories before reporting aggregate validation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from utils import BASE_DIR
from predict_surrogate import load_surrogate_model, predict_generator


def _load_states_df(states_dir: Path, model_name: str) -> pd.DataFrame:
    csv_path = states_dir / f"{model_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"State CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"t", "ratio_S", "ratio_Q", "ratio_L", "ratio_B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

    # Keep a stable state order
    df = df.sort_values("t").reset_index(drop=True)
    return df


def _load_Q_npz(odes_dir: Path, model_name: str):
    npz_path = odes_dir / f"{model_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"ODE fit file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    Q = data["Q"].astype(float)
    dt = float(data["dt"][0]) if "dt" in data else 1.0
    state_order = data["state_order"].tolist() if "state_order" in data else ["S", "Q", "L", "B"]

    return Q, dt, state_order


def _simulate_ode_fractions(
    Q: np.ndarray,
    x0: np.ndarray,
    T: int,
    *,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Forward simulate fractions with explicit Euler:
        x_{t+1} = x_t + dt * x_t Q
    """
    x = np.zeros((T, 4), dtype=float)
    x[0] = x0

    for t in range(T - 1):
        x[t + 1] = x[t] + dt * (x[t] @ Q)

        # numerical safety: clip and renormalize
        x[t + 1] = np.clip(x[t + 1], 0.0, 1.0)
        s = x[t + 1].sum()
        if s > 0:
            x[t + 1] /= s

    return x


def plot_sqlb_states(base_dir: Path, model_name: str, *, show: bool = True, save_path: str | Path | None = None):
    df = _load_states_df(base_dir / "states", model_name)

    plt.figure()
    plt.plot(df["t"], df["ratio_S"], label="S")
    plt.plot(df["t"], df["ratio_Q"], label="Q")
    plt.plot(df["t"], df["ratio_L"], label="L")
    plt.plot(df["t"], df["ratio_B"], label="B")

    plt.xlabel("Time step (t)")
    plt.ylabel("Population ratio")
    plt.title(f"State Ratios")
    ax = plt.gca()
    ax.set_xticks([0.0])
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return df


def plot_sqlb_states_vs_ode(
    base_dir: Path,
    model_name: str,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Overlay ABM state ratios (solid) vs ODE simulated ratios (dashed),
    with consistent colors per state.
    """
    states_dir = base_dir / "states"
    odes_dir = base_dir / "odes"
    df = _load_states_df(states_dir, model_name)
    Q, dt_fit, state_order = _load_Q_npz(odes_dir, model_name)

    # Ensure ordering matches (expects S,Q,L,B)
    desired = ["S", "Q", "L", "B"]
    if list(state_order) != desired:
        # reorder Q into desired order
        idx = {s: i for i, s in enumerate(state_order)}
        perm = [idx[s] for s in desired]
        Q = Q[np.ix_(perm, perm)]
        state_order = desired

    T = len(df)
    t = df["t"].to_numpy()

    x0 = np.array([df.loc[0, "ratio_S"], df.loc[0, "ratio_Q"], df.loc[0, "ratio_L"], df.loc[0, "ratio_B"]], dtype=float)
    ode = _simulate_ode_fractions(Q, x0, T, dt=dt_fit)

    # --- consistent colors: plot ABM first, reuse those colors for ODE ---
    fig = plt.figure()
    ax = plt.gca()

    line_S = ax.plot(t, df["ratio_S"], label="S (ABM)")[0]
    line_Q = ax.plot(t, df["ratio_Q"], label="Q (ABM)")[0]
    line_L = ax.plot(t, df["ratio_L"], label="L (ABM)")[0]
    line_B = ax.plot(t, df["ratio_B"], label="B (ABM)")[0]

    colors = {
        "S": line_S.get_color(),
        "Q": line_Q.get_color(),
        "L": line_L.get_color(),
        "B": line_B.get_color(),
    }

    ax.plot(t, ode[:, 0], linestyle="--", color=colors["S"], label="S (ODE)")
    ax.plot(t, ode[:, 1], linestyle="--", color=colors["Q"], label="Q (ODE)")
    ax.plot(t, ode[:, 2], linestyle="--", color=colors["L"], label="L (ODE)")
    ax.plot(t, ode[:, 3], linestyle="--", color=colors["B"], label="B (ODE)")

    ax.set_xlabel("Time step (t)")
    ax.set_ylabel("Agent ratio")
    ax.set_title(f"SQLB — ABM vs. ODE")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return df, ode, Q, dt_fit


def _load_settings_row(settings_path: Path, model_name: str) -> pd.Series:
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.csv not found: {settings_path}")

    df = pd.read_csv(settings_path)
    if "name" not in df.columns:
        raise ValueError("settings.csv must contain a 'name' column.")

    match = df.loc[df["name"].astype(str) == str(model_name)]
    if match.empty:
        raise ValueError(f"Model name '{model_name}' not found in settings.csv")

    return match.iloc[0]


def plot_sqlb_states_vs_surrogate(
    base_dir: Path,
    model_name: str,
    *,
    surrogate_path: str | Path | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Overlay ABM state ratios (solid) vs surrogate-predicted ODE state ratios (dashed),
    using the model inputs from settings.csv.

    Workflow:
        1. Load ABM state fractions from states/{model_name}.csv
        2. Load model inputs from settings.csv
        3. Load saved surrogate model
        4. Predict CTMC generator Q from surrogate
        5. Simulate x_{t+1} = x_t + dt * x_t Q from the same x0 as the ABM
        6. Plot ABM vs surrogate

    Returns
    -------
    df : pd.DataFrame
        ABM state fractions
    surrogate_ode : np.ndarray
        Simulated surrogate state fractions, shape (T,4)
    Q_surrogate : np.ndarray
        Predicted surrogate generator matrix
    settings_row : pd.Series
        Matching row from settings.csv
    """
    states_dir = base_dir / "states"
    settings_path = base_dir / "settings.csv"

    df = _load_states_df(states_dir, model_name)
    settings_row = _load_settings_row(settings_path, model_name)

    model = load_surrogate_model(
        Path(surrogate_path) if surrogate_path is not None else None
    )

    Q_surrogate, rates_surrogate = predict_generator(
        model,
        teams_num=float(settings_row["teams_num"]),
        teams_size=float(settings_row["teams_size"]),
        agents_average_initial_opinion=float(settings_row["agents_average_initial_opinion"]),
        technology_success_rate=float(settings_row["technology_success_rate"]),
    )

    T = len(df)
    t = df["t"].to_numpy()

    x0 = np.array(
        [
            df.loc[0, "ratio_S"],
            df.loc[0, "ratio_Q"],
            df.loc[0, "ratio_L"],
            df.loc[0, "ratio_B"],
        ],
        dtype=float,
    )

    surrogate_ode = _simulate_ode_fractions(Q_surrogate, x0, T, dt=1.0)

    fig = plt.figure()
    ax = plt.gca()

    # ABM first, to get stable colors
    line_S = ax.plot(t, df["ratio_S"], label="S (ABM)")[0]
    line_Q = ax.plot(t, df["ratio_Q"], label="Q (ABM)")[0]
    line_L = ax.plot(t, df["ratio_L"], label="L (ABM)")[0]
    line_B = ax.plot(t, df["ratio_B"], label="B (ABM)")[0]

    colors = {
        "S": line_S.get_color(),
        "Q": line_Q.get_color(),
        "L": line_L.get_color(),
        "B": line_B.get_color(),
    }

    ax.plot(t, surrogate_ode[:, 0], linestyle="--", color=colors["S"], label="S (Surrogate)")
    ax.plot(t, surrogate_ode[:, 1], linestyle="--", color=colors["Q"], label="Q (Surrogate)")
    ax.plot(t, surrogate_ode[:, 2], linestyle="--", color=colors["L"], label="L (Surrogate)")
    ax.plot(t, surrogate_ode[:, 3], linestyle="--", color=colors["B"], label="B (Surrogate)")

    ax.set_xlabel("Time step (t)")
    ax.set_ylabel("Agent ratio")
    ax.set_title("SQLB — ABM vs. Surrogate")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return df, surrogate_ode, Q_surrogate, settings_row

def plot_sqlb_states_vs_ode_two_examples(
    base_dir: Path,
    model_names: list[str] | tuple[str, str],
    *,
    subplot_titles: list[str] | tuple[str, str] = ("Failed adoption", "Successful adoption"),
    figure_title: str = "ABM vs. fitted ODE examples",
    show: bool = True,
    save_path: str | Path | None = None,
    fig_width: float = 12.0,
    fig_height: float = 4.8,
    title_fontsize: float = 16,
    subplot_title_fontsize: float = 14,
    axis_label_fontsize: float = 13,
    tick_fontsize: float = 11,
    legend_fontsize: float = 10,
    linewidth_abm: float = 2.0,
    linewidth_ode: float = 2.0,
    legend_ncol: int = 2,
):
    """
    Plot two ABM-vs-fitted-ODE examples in a 1x2 grid.

    Parameters
    ----------
    model_names
        Exactly two model names, e.g. ["2_9_10_2_3", "2_12_6_3_2"].
    subplot_titles
        Titles for left and right subplots.
    figure_title
        Global figure title.
    Font/size parameters are exposed for iterative tuning.
    """
    if len(model_names) != 2:
        raise ValueError("model_names must contain exactly two model names.")

    if len(subplot_titles) != 2:
        raise ValueError("subplot_titles must contain exactly two titles.")

    states_dir = base_dir / "states"
    odes_dir = base_dir / "odes"

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True)

    desired = ["S", "Q", "L", "B"]

    for ax, model_name, subtitle in zip(axes, model_names, subplot_titles):
        df = _load_states_df(states_dir, model_name)
        Q, dt_fit, state_order = _load_Q_npz(odes_dir, model_name)

        if list(state_order) != desired:
            idx = {s: i for i, s in enumerate(state_order)}
            perm = [idx[s] for s in desired]
            Q = Q[np.ix_(perm, perm)]

        T = len(df)
        t = df["t"].to_numpy()

        x0 = np.array(
            [
                df.loc[0, "ratio_S"],
                df.loc[0, "ratio_Q"],
                df.loc[0, "ratio_L"],
                df.loc[0, "ratio_B"],
            ],
            dtype=float,
        )
        ode = _simulate_ode_fractions(Q, x0, T, dt=dt_fit)

        # Plot ABM first so ODE can reuse colors
        line_S = ax.plot(t, df["ratio_S"], linewidth=linewidth_abm, label="S (ABM)")[0]
        line_Q = ax.plot(t, df["ratio_Q"], linewidth=linewidth_abm, label="Q (ABM)")[0]
        line_L = ax.plot(t, df["ratio_L"], linewidth=linewidth_abm, label="L (ABM)")[0]
        line_B = ax.plot(t, df["ratio_B"], linewidth=linewidth_abm, label="B (ABM)")[0]

        colors = {
            "S": line_S.get_color(),
            "Q": line_Q.get_color(),
            "L": line_L.get_color(),
            "B": line_B.get_color(),
        }

        ax.plot(
            t, ode[:, 0],
            linestyle="--",
            linewidth=linewidth_ode,
            color=colors["S"],
            label="S (ODE)",
        )
        ax.plot(
            t, ode[:, 1],
            linestyle="--",
            linewidth=linewidth_ode,
            color=colors["Q"],
            label="Q (ODE)",
        )
        ax.plot(
            t, ode[:, 2],
            linestyle="--",
            linewidth=linewidth_ode,
            color=colors["L"],
            label="L (ODE)",
        )
        ax.plot(
            t, ode[:, 3],
            linestyle="--",
            linewidth=linewidth_ode,
            color=colors["B"],
            label="B (ODE)",
        )

        # Make only the second subplot title bold
        title_weight = "bold"

        ax.set_title(
            subtitle,
            fontsize=subplot_title_fontsize,
            fontweight=title_weight,
        )
        ax.set_xlabel("Time step (t)", fontsize=axis_label_fontsize)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    handles, labels = axes[1].get_legend_handles_labels()

    axes[1].legend(
        handles,
        labels,
        fontsize=legend_fontsize,
        ncol=legend_ncol,
    )

    axes[0].set_ylabel("Agent ratio", fontsize=axis_label_fontsize)

    fig.suptitle(figure_title, fontsize=title_fontsize, y=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=400)

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    # Example model list for quick figure generation.
    #model_names = [
        #"2_12_6_3_2",
        #"2_17_9_3_1",
        #"2_9_10_2_3",
        #"4_15_7_5_0",
        #"1_13_8_4_0"
    #]

    #for i, model_name in enumerate(model_names):
        #plot_sqlb_states(BASE_DIR, model_name, show=False, save_path=BASE_DIR / "figures" / f"abm_example_{i+1}.pdf")
        #plot_sqlb_states_vs_ode(BASE_DIR, model_name, show=False, save_path=BASE_DIR / "figures" / f"abm_ode_example_{i+1}.pdf")
        #plot_sqlb_states_vs_surrogate(BASE_DIR, model_name, show=False, save_path=BASE_DIR / "figures" / f"abm_surrogate_example_{i+1}.pdf")

    plot_sqlb_states_vs_ode_two_examples(
        BASE_DIR,
        model_names=["2_9_10_2_3", "2_12_6_3_2"],
        subplot_titles=["Successful adoption", "Failed adoption"],
        figure_title="ABM vs. fitted CTMC/ODE examples",
        show=False,
        save_path=BASE_DIR / "figures" / "abm_ode_two_examples.pdf",
        fig_width=8.0,
        fig_height=4.8,
        title_fontsize=17,
        subplot_title_fontsize=15,
        axis_label_fontsize=14,
        tick_fontsize=12,
        legend_fontsize=11,
    )

    # The two largest trajectory-RMSE cases in the fitted-ODE validation.
    plot_sqlb_states_vs_ode_two_examples(
        BASE_DIR,
        model_names=["2_15_7_0_0", "2_17_7_0_0"],
        subplot_titles=["Poor fit: 2_15_7_0_0", "Poor fit: 2_17_7_0_0"],
        figure_title="ABM vs. fitted CTMC/ODE examples with poor agreement",
        show=False,
        save_path=BASE_DIR / "figures" / "abm_ode_two_examples_bad.pdf",
        fig_width=8.0,
        fig_height=4.8,
        title_fontsize=17,
        subplot_title_fontsize=15,
        axis_label_fontsize=14,
        tick_fontsize=12,
        legend_fontsize=11,
    )
