"""Fast checks on data integrity and the key analytic invariants."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gbsg2 import causal, data, ml, survival  # noqa: E402
from gbsg2.config import CONFOUNDERS, EVENT, TIME, TREATMENT  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return data.load()


def test_shape_and_no_missing(df):
    assert df.shape[0] == 686
    assert df.isna().sum().sum() == 0


def test_outcome_and_treatment_coding(df):
    assert set(df[EVENT].unique()) == {0, 1}
    assert set(df[TREATMENT].unique()) == {0, 1}
    assert (df[TIME] > 0).all()


def test_quality_checks_clean(df):
    checks = data.quality_report(df).attrs["checks"]
    assert all(v == 0 for v in checks.values())


def test_propensity_in_unit_interval(df):
    ps = causal.propensity_scores(df, CONFOUNDERS)
    assert ps.between(0, 1, inclusive="neither").all()


def test_iptw_achieves_balance(df):
    ps = causal.propensity_scores(df, CONFOUNDERS)
    w = causal.iptw(df, ps)
    assert (w > 0).all()
    bal = causal.balance_table(df, CONFOUNDERS, w)
    assert bal["smd_weighted"].max() < 0.1
    assert bal["smd_weighted"].max() < bal["smd_unweighted"].max()


def test_cox_returns_plausible_hr(df):
    _, tab = survival.cox_model(df, CONFOUNDERS)
    hr = tab.loc[TREATMENT, "HR"]
    assert 0.2 < hr < 2.0
    assert tab.loc[TREATMENT, "HR_lcl"] < hr < tab.loc[TREATMENT, "HR_ucl"]


def test_e_value_null_ci_is_one():
    assert causal.e_value_hr(0.8, 1.1)["e_value_ci"] == 1.0


def test_models_beat_chance(df):
    X_tr, X_te, y_tr, y_te = ml.split(df, CONFOUNDERS)
    perf = ml.evaluate(ml.fit_models(X_tr, y_tr), X_tr, y_tr, X_te, y_te)
    assert (perf["harrell_c"] > 0.55).all()
