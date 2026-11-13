"""
CTMC steady-state utilities used across the SQLB reduction pipeline.

What this module does:
1. Validates a fitted/predicted CTMC generator matrix Q.
2. Finds closed communicating classes and transient states.
3. Computes class-level stationary distributions and absorption weights.
4. Returns the long-run state composition:
       x_inf = lim_{t->infinity} x0 expm(Q t)

How it is used in the pipeline:
- `steady_state_fitted_ode.py`:
  Converts each fitted ODE generator in `odes/*.npz` + model-specific x0
  into steady-state SQLB ratios and writes `steady_states.csv`.
- `steady_state_surrogate_ode.py`:
  Uses surrogate-predicted generators + x0 to estimate steady-state SQLB ratios.
- Validation scripts (`validate_surrogate_ode.py`, `validate_fitted_ode.py`):
  Compare surrogate steady-state reconstruction against fitted-ODE steady states.

This module is the core Markov-chain long-run solver for all steady-state results
reported downstream (plots, CSV summaries, and paper metrics).
"""

from __future__ import annotations

from typing import List

import numpy as np


def validate_generator(Q: np.ndarray, tol: float = 1e-10) -> None:
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")

    offdiag = Q.copy()
    np.fill_diagonal(offdiag, 0.0)
    if np.any(offdiag < -tol):
        raise ValueError(f"Invalid generator: negative off-diagonal entries found\n{Q}")

    diag = np.diag(Q)
    if np.any(diag > tol):
        raise ValueError(f"Invalid generator: positive diagonal entries found\n{Q}")

    row_sums = Q.sum(axis=1)
    if not np.allclose(row_sums, 0.0, atol=tol):
        raise ValueError(f"Invalid generator: row sums not zero\nrow_sums={row_sums}")


def adjacency_from_Q(Q: np.ndarray, tol: float = 1e-14) -> List[List[int]]:
    n = Q.shape[0]
    graph: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and Q[i, j] > tol:
                graph[i].append(j)
    return graph


def transpose_graph(graph: List[List[int]]) -> List[List[int]]:
    n = len(graph)
    gt: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in graph[i]:
            gt[j].append(i)
    return gt


def strongly_connected_components(graph: List[List[int]]) -> List[List[int]]:
    n = len(graph)
    visited = [False] * n
    order: List[int] = []

    def dfs1(v: int) -> None:
        visited[v] = True
        for w in graph[v]:
            if not visited[w]:
                dfs1(w)
        order.append(v)

    for v in range(n):
        if not visited[v]:
            dfs1(v)

    gt = transpose_graph(graph)
    visited = [False] * n
    sccs: List[List[int]] = []

    def dfs2(v: int, comp: List[int]) -> None:
        visited[v] = True
        comp.append(v)
        for w in gt[v]:
            if not visited[w]:
                dfs2(w, comp)

    for v in reversed(order):
        if not visited[v]:
            comp: List[int] = []
            dfs2(v, comp)
            sccs.append(sorted(comp))

    return sccs


def closed_classes_from_Q(Q: np.ndarray, tol: float = 1e-14) -> List[List[int]]:
    graph = adjacency_from_Q(Q, tol=tol)
    sccs = strongly_connected_components(graph)

    closed: List[List[int]] = []
    for comp in sccs:
        comp_set = set(comp)
        is_closed = True
        for i in comp:
            for j in graph[i]:
                if j not in comp_set:
                    is_closed = False
                    break
            if not is_closed:
                break
        if is_closed:
            closed.append(comp)

    return closed


def stationary_distribution_closed_class(Qc: np.ndarray) -> np.ndarray:
    m = Qc.shape[0]
    A = np.vstack([Qc.T, np.ones(m, dtype=float)])
    b = np.zeros(m + 1, dtype=float)
    b[-1] = 1.0

    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.real_if_close(pi).astype(float)
    pi[np.abs(pi) < 1e-12] = 0.0

    if np.any(pi < -1e-8):
        raise ValueError(f"Closed-class stationary distribution has negative entries: {pi}")

    pi = np.clip(pi, 0.0, None)
    s = pi.sum()
    if s <= 0:
        raise ValueError("Could not normalize class stationary distribution")

    return pi / s


def class_absorption_probabilities(
    Q: np.ndarray,
    transient_idx: List[int],
    closed_classes: List[List[int]],
) -> np.ndarray:
    nt = len(transient_idx)
    nc = len(closed_classes)

    if nt == 0:
        return np.zeros((0, nc), dtype=float)

    QTT = Q[np.ix_(transient_idx, transient_idx)]
    H = np.zeros((nt, nc), dtype=float)

    for c, cls in enumerate(closed_classes):
        rhs = -Q[np.ix_(transient_idx, cls)] @ np.ones(len(cls), dtype=float)
        h, *_ = np.linalg.lstsq(QTT, rhs, rcond=None)
        h = np.real_if_close(h).astype(float)
        h[np.abs(h) < 1e-12] = 0.0
        h = np.clip(h, 0.0, 1.0)
        H[:, c] = h

    row_sums = H.sum(axis=1)
    for i in range(nt):
        if row_sums[i] > 0:
            H[i, :] /= row_sums[i]

    return H


def compute_long_run_state(Q: np.ndarray, x0: np.ndarray) -> np.ndarray:
    validate_generator(Q)

    n = Q.shape[0]
    closed_classes = closed_classes_from_Q(Q)

    if len(closed_classes) == 0:
        raise ValueError("No closed classes found; invalid finite CTMC structure")

    closed_flat = sorted(i for cls in closed_classes for i in cls)
    transient_idx = [i for i in range(n) if i not in closed_flat]

    class_stationaries = []
    for cls in closed_classes:
        Qc = Q[np.ix_(cls, cls)]
        pi_c = stationary_distribution_closed_class(Qc)
        class_stationaries.append(pi_c)

    H = class_absorption_probabilities(Q, transient_idx, closed_classes)

    alpha = np.zeros(len(closed_classes), dtype=float)

    if transient_idx:
        x0_T = x0[transient_idx]
        alpha += x0_T @ H

    for c, cls in enumerate(closed_classes):
        alpha[c] += float(x0[cls].sum())

    x_inf = np.zeros(n, dtype=float)
    for c, cls in enumerate(closed_classes):
        x_inf[cls] += alpha[c] * class_stationaries[c]

    x_inf[np.abs(x_inf) < 1e-12] = 0.0
    x_inf = np.clip(x_inf, 0.0, None)

    s = x_inf.sum()
    if s <= 0:
        raise ValueError("Computed long-run state has nonpositive total mass")

    return x_inf / s
