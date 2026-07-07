import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from causal import did, synth, matching


class TestDiD:
    @pytest.fixture(scope="class")
    def df(self):
        return did.generate()

    def test_naive_is_badly_confounded(self, df):
        naive = did.naive_estimate(df)
        assert abs(naive - did.TRUE_EFFECT) > 10  # off by ~the selection gap

    def test_did_recovers_truth(self, df):
        est = did.did_estimate(df)
        assert est["did_2x2"] == pytest.approx(did.TRUE_EFFECT, abs=1.5)
        assert est["did_ols"] == pytest.approx(did.TRUE_EFFECT, abs=1.5)
        # truth inside ~2 clustered SEs
        assert abs(est["did_ols"] - did.TRUE_EFFECT) < 2.5 * est["se_clustered"]

    def test_no_pretrend(self, df):
        ev = did.event_study(df)
        pre = ev[ev.month_rel < 0].gap_vs_baseline
        assert pre.abs().max() < 3.0

    def test_placebo_in_time_null(self, df):
        assert abs(did.placebo_in_time(df)) < 2.0


class TestSyntheticControl:
    @pytest.fixture(scope="class")
    def df(self):
        return synth.generate()

    def test_pre_fit_tight_and_effect_recovered(self, df):
        est = synth.estimate(df)
        assert est["pre_rmse"] < 2.5
        assert est["att"] == pytest.approx(synth.TRUE_EFFECT, abs=3.0)

    def test_placebo_in_space_significant(self, df):
        p = synth.placebo_in_space(df)
        assert p["p_value_permutation"] <= 0.1   # 1/21 ideally


class TestMatching:
    @pytest.fixture(scope="class")
    def df(self):
        return matching.generate()

    def test_naive_overstates(self, df):
        naive = matching.naive_estimate(df)
        assert naive > matching.TRUE_EFFECT * 1.8

    def test_stratification_recovers_truth(self, df):
        est = matching.propensity_stratified(df)
        assert est["att_stratified"] == pytest.approx(matching.TRUE_EFFECT, rel=0.15)

    def test_balance_improves_within_strata(self, df):
        est = matching.propensity_stratified(df)
        assert est["smd_pre_activity_within_max"] < abs(est["smd_pre_activity_before"]) / 2
