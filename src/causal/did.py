"""Case 1 — Difference-in-Differences.

Scenario: a retailer rolls a loyalty program to 40 of 100 stores in July.
No experiment was possible (rollout chosen by ops), and treated stores were
already HIGHER-revenue — so the naive treated-vs-control comparison is
badly confounded. DiD identifies the effect off CHANGES, not levels, under
parallel trends.

Everything is generated with a known true effect, so the case proves:
naive comparison wrong -> DiD recovers truth -> event study shows no
pre-trend (the assumption made visible) -> placebo-in-time comes up null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_EFFECT = 8.0        # program adds $8k/month to treated stores post-launch
SELECTION_GAP = 25.0     # treated stores were already $25k/month bigger (confound)


def generate(n_stores: int = 100, n_treated: int = 40, months: int = 24,
             launch: int = 12, seed: int = 20260714) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    store_fe = rng.normal(100, 15, n_stores)          # store level differences
    treated = np.zeros(n_stores, dtype=bool)
    # selection on levels: biggest stores got the program (the confound)
    treated[np.argsort(store_fe)[-n_treated:]] = True
    store_fe[treated] += SELECTION_GAP - (store_fe[treated].mean() - store_fe.mean())

    rows = []
    for m in range(months):
        season = 6 * np.sin(2 * np.pi * m / 12)
        common_shock = rng.normal(0, 2)
        for s in range(n_stores):
            y = (store_fe[s] + season + common_shock + 0.4 * m   # shared trend
                 + rng.normal(0, 4))
            if treated[s] and m >= launch:
                y += TRUE_EFFECT
            rows.append({"store_id": s, "month": m, "treated": bool(treated[s]),
                         "post": m >= launch, "revenue_k": round(y, 2)})
    return pd.DataFrame(rows)


def naive_estimate(df: pd.DataFrame) -> float:
    """Post-period treated-vs-control mean gap — what a dashboard shows,
    and it's off by the entire selection gap."""
    post = df[df.post]
    return float(post[post.treated].revenue_k.mean()
                 - post[~post.treated].revenue_k.mean())


def did_estimate(df: pd.DataFrame) -> dict:
    """2x2 DiD plus the regression version with standard errors."""
    g = df.groupby(["treated", "post"]).revenue_k.mean()
    two_by_two = (g[True, True] - g[True, False]) - (g[False, True] - g[False, False])

    import statsmodels.formula.api as smf
    df = df.copy()
    df["treat_post"] = df.treated & df.post
    m = smf.ols("revenue_k ~ treated + post + treat_post", df).fit(
        cov_type="cluster", cov_kwds={"groups": df.store_id})
    return {"did_2x2": round(float(two_by_two), 3),
            "did_ols": round(float(m.params["treat_post[T.True]"]), 3),
            "se_clustered": round(float(m.bse["treat_post[T.True]"]), 3)}


def event_study(df: pd.DataFrame, launch: int = 12) -> pd.DataFrame:
    """Per-month treated-control gap relative to launch-1 — the parallel
    trends check. Pre-launch coefficients should hover at zero."""
    gap = (df[df.treated].groupby("month").revenue_k.mean()
           - df[~df.treated].groupby("month").revenue_k.mean())
    base = gap[launch - 1]
    return pd.DataFrame({"month_rel": gap.index - launch,
                         "gap_vs_baseline": (gap - base).round(3)})


def placebo_in_time(df: pd.DataFrame, fake_launch: int = 6) -> float:
    """Rerun DiD pretending launch happened at month 6, using only true
    pre-period data. A real design must return ~0 here."""
    pre = df[df.month < 12].copy()
    pre["post"] = pre.month >= fake_launch
    g = pre.groupby(["treated", "post"]).revenue_k.mean()
    return round(float((g[True, True] - g[True, False])
                       - (g[False, True] - g[False, False])), 3)
