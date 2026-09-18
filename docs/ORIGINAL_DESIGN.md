# Original project narrative (historical)

This preserves the earlier design and example results. The root README and current tests control implementation status. Planned integrations and old test counts below are not completion claims.

# Causal Impact Measurement | When an experiment is not possible

Three quasi-experimental methods, each built as a complete case with a
planted true effect: the naive estimate is computed first and shown wrong,
the method recovers the truth, and the method's key ASSUMPTION is made
visible and tested. 9 pytest tests pin every claim.

This is the senior-analyst skill big-tech loops probe hardest: experiments
are the gold standard (see the sibling ab-testing-framework project), but
half of real decisions can't be randomized — and knowing what replaces
randomization, and what it costs, is the difference between an analyst and
a dashboard operator.

## The three cases (all numbers measured, seeded)

**1. Difference-in-Differences** — loyalty program rolled to the *biggest*
stores (selection on levels, no experiment possible):
```
naive post-period gap : +44.2   (truth: +8.0 — off by the entire selection gap)
DiD (2x2 and OLS)     : +7.97  ± 0.35 clustered SE
event study           : pre-launch gaps flat (parallel trends, visualized)
placebo-in-time       : -0.8 (fake launch date finds nothing — as it must)
```

**2. Synthetic Control** — ONE treated state (nothing to average over):
```
pre-period fit        : RMSE 1.9 on outcome levels ~80-110
estimated ATT         : -9.1   (truth: -12.0)
placebo-in-space      : treated post/pre RMSE ratio 5.0, permutation p = 0.095
                        (the honest limit of 21-unit inference — stated, not hidden)
```
Implementation note kept in the code: solving NNLS then renormalizing
weights destroys the fit; the sum-to-one constraint must live inside the
solver. Found by a failing test.

**3. Propensity Stratification** — feature adoption confounded by usage:
```
naive adopter gap     : +$40.5  (truth: +$15 — 2.7x overstated)
stratified ATT        : +$15.9
balance check         : pre-activity SMD 1.00 -> 0.49 max within strata
boundary condition    : matching handles OBSERVED confounding only; the
                        generator guarantees no unobserved confounder,
                        real data never does — the caveat every readout carries
```

## Run
```bash
pip install pandas numpy scipy scikit-learn statsmodels pytest
PYTHONPATH=src pytest tests/ -q     # 9 tests
```
`src/causal/{did,synth,matching}.py` — each file is one self-contained case
with its scenario, estimator, diagnostics, and honesty notes in docstrings.
