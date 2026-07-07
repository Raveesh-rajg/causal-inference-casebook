"""Case 2 — Synthetic Control.

Scenario: ONE state changes policy (n_treated = 1, so DiD's averaging has
nothing to average). Build a synthetic twin as a convex combination of
donor states that matches the treated state's PRE-period path, then read
the effect as the post-period gap. Inference by placebo-in-space: run the
same procedure on every donor and ask whether the treated state's gap is
extreme in that distribution (permutation logic, no parametric SEs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls

TRUE_EFFECT = -12.0     # policy reduces the outcome by 12 units post-launch


def generate(n_donors: int = 20, periods: int = 40, launch: int = 24,
             seed: int = 20260715) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # latent factors give states correlated-but-distinct trajectories
    factors = np.vstack([np.linspace(0, 8, periods),
                         np.sin(np.linspace(0, 6 * np.pi, periods)) * 3,
                         rng.normal(0, 1, periods).cumsum()])
    rows = []
    loadings = rng.uniform(0.2, 1.4, (n_donors + 1, 3))
    levels = rng.normal(80, 10, n_donors + 1)
    for u in range(n_donors + 1):
        y = levels[u] + loadings[u] @ factors + rng.normal(0, 1.5, periods)
        if u == 0:  # treated state
            y[launch:] += TRUE_EFFECT
        for t in range(periods):
            rows.append({"unit": u, "period": t, "treated": u == 0,
                         "outcome": round(float(y[t]), 3)})
    return pd.DataFrame(rows)


def fit_weights(df: pd.DataFrame, launch: int = 24) -> np.ndarray:
    """Donor weights: constrained least squares (w >= 0, sum(w) = 1).

    Implementation note that cost a failing test: solving unconstrained
    NNLS and renormalizing afterwards destroys the fit whenever the raw
    weights don't already sum near 1. The sum-to-one constraint must live
    INSIDE the solver — done here by augmenting the system with a heavily
    weighted row of ones (K=1000), a standard trick that keeps plain NNLS
    while enforcing the constraint to numerical precision."""
    pre = df[df.period < launch]
    Y0 = pre[~pre.treated].pivot(index="period", columns="unit", values="outcome").values
    y1 = pre[pre.treated].sort_values("period").outcome.values
    K = 1000.0
    A = np.vstack([Y0, K * np.ones(Y0.shape[1])])
    b = np.append(y1, K)
    w, _ = nnls(A, b)
    return w


def effect_path(df: pd.DataFrame, weights: np.ndarray, launch: int = 24) -> pd.DataFrame:
    donors = df[~df.treated].pivot(index="period", columns="unit", values="outcome")
    synth = donors.values @ weights
    actual = df[df.treated].sort_values("period").outcome.values
    out = pd.DataFrame({"period": donors.index, "actual": actual,
                        "synthetic": synth.round(3),
                        "gap": (actual - synth).round(3)})
    out["post"] = out.period >= launch
    return out


def estimate(df: pd.DataFrame, launch: int = 24) -> dict:
    w = fit_weights(df, launch)
    path = effect_path(df, w, launch)
    pre_rmse = float(np.sqrt((path[~path.post].gap ** 2).mean()))
    att = float(path[path.post].gap.mean())
    return {"att": round(att, 3), "pre_rmse": round(pre_rmse, 3),
            "n_donors_used": int((w > 0.01).sum())}


def placebo_in_space(df: pd.DataFrame, launch: int = 24) -> dict:
    """Assign treatment to each donor in turn; the treated unit's post/pre
    RMSE ratio should sit in the tail of the placebo distribution."""
    def ratio_for(unit: int) -> float:
        d = df.copy()
        d["treated"] = d.unit == unit
        w = fit_weights(d, launch)
        path = effect_path(d, w, launch)
        pre = np.sqrt((path[~path.post].gap ** 2).mean())
        post = np.sqrt((path[path.post].gap ** 2).mean())
        return float(post / max(pre, 1e-9))

    units = sorted(df.unit.unique())
    ratios = {u: ratio_for(u) for u in units}
    treated_ratio = ratios[0]
    rank = sum(1 for u, r in ratios.items() if r >= treated_ratio)
    return {"treated_ratio": round(treated_ratio, 2),
            "p_value_permutation": round(rank / len(units), 3)}
