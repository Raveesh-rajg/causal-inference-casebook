"""Case 3 — Matching / propensity stratification on observational data.

Scenario: did users who adopted a premium feature spend more BECAUSE of it?
Adoption is confounded: heavy users adopt more AND spend more regardless.
The naive adopter-vs-non gap dramatically overstates the effect; stratifying
on the propensity score (estimated from pre-adoption covariates) recovers
the planted truth.

The honest boundary, stated: matching only handles OBSERVED confounding.
The generator has no unobserved confounder by construction; real data
offers no such guarantee — that caveat belongs in every readout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TRUE_EFFECT = 15.0    # feature genuinely adds $15/month


def generate(n: int = 20_000, seed: int = 20260716) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.exponential(14, n).clip(1, 60)
    pre_activity = rng.gamma(2, 6, n) + tenure * 0.4      # confounder
    org_size = rng.lognormal(3, 1, n)
    # adoption driven by the SAME variables that drive spend
    logit = -3.2 + 0.10 * pre_activity + 0.02 * tenure + 0.15 * np.log(org_size)
    adopted = rng.random(n) < 1 / (1 + np.exp(-logit))
    spend = (20 + 2.2 * pre_activity + 0.5 * tenure + 4 * np.log(org_size)
             + rng.normal(0, 12, n))
    spend = spend + TRUE_EFFECT * adopted
    return pd.DataFrame({"tenure": tenure.round(1),
                         "pre_activity": pre_activity.round(2),
                         "org_size": org_size.round(1),
                         "adopted": adopted, "spend": spend.round(2)})


def naive_estimate(df: pd.DataFrame) -> float:
    return float(df[df.adopted].spend.mean() - df[~df.adopted].spend.mean())


def propensity_stratified(df: pd.DataFrame, n_strata: int = 10) -> dict:
    X = np.column_stack([df.pre_activity, df.tenure, np.log(df.org_size)])
    ps = LogisticRegression(max_iter=1000).fit(X, df.adopted).predict_proba(X)[:, 1]
    d = df.assign(ps=ps)
    d["stratum"] = pd.qcut(d.ps, n_strata, labels=False, duplicates="drop")
    effs, weights = [], []
    for _, s in d.groupby("stratum"):
        if s.adopted.nunique() == 2:
            effs.append(s[s.adopted].spend.mean() - s[~s.adopted].spend.mean())
            weights.append(len(s))
    att = float(np.average(effs, weights=weights))
    # covariate balance check: standardized mean difference before/after
    smd_before = _smd(df, "pre_activity")
    within = [abs(_smd(s, "pre_activity")) for _, s in d.groupby("stratum")
              if s.adopted.nunique() == 2]
    return {"att_stratified": round(att, 3),
            "smd_pre_activity_before": round(smd_before, 3),
            "smd_pre_activity_within_max": round(max(within), 3)}


def _smd(d: pd.DataFrame, col: str) -> float:
    a, b = d[d.adopted][col], d[~d.adopted][col]
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / pooled)
