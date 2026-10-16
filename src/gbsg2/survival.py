"""Descriptive survival (Kaplan–Meier, log-rank) and multivariable Cox regression."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

from .config import EVENT, HORIZON_DAYS, TIME, TREATMENT


def km_by_group(df: pd.DataFrame, path=None, weights: pd.Series | None = None,
                title: str = "Recurrence-free survival by hormone therapy") -> pd.DataFrame:
    """Fit KM per arm; return median and 5-year RFS (95% CI). Optionally plot."""
    rows, ax = [], None
    if path is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    for g, label in [(0, "No hormone therapy"), (1, "Hormone therapy")]:
        sub = df[df[TREATMENT] == g]
        w = None if weights is None else weights.loc[sub.index]
        km = KaplanMeierFitter(label=label).fit(sub[TIME], sub[EVENT], weights=w)
        s = km.survival_function_at_times(HORIZON_DAYS).iloc[0]
        ci = km.confidence_interval_survival_function_
        ci_at = ci[ci.index <= HORIZON_DAYS].iloc[-1]
        rows.append({"group": label, "n": len(sub), "events": int(sub[EVENT].sum()),
                     "median_rfs_days": km.median_survival_time_,
                     "rfs_5y": round(float(s), 3),
                     "rfs_5y_lcl": round(float(ci_at.iloc[0]), 3),
                     "rfs_5y_ucl": round(float(ci_at.iloc[1]), 3)})
        if ax is not None:
            km.plot_survival_function(ax=ax, ci_show=True)
    if ax is not None:
        ax.set_xlabel("Days since study entry"); ax.set_ylabel("Recurrence-free survival")
        ax.set_title(title); ax.axvline(HORIZON_DAYS, ls=":", c="grey", lw=1)
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return pd.DataFrame(rows)


def logrank(df: pd.DataFrame) -> float:
    a, b = df[df[TREATMENT] == 1], df[df[TREATMENT] == 0]
    return float(logrank_test(a[TIME], b[TIME], a[EVENT], b[EVENT]).p_value)


def cox_model(df: pd.DataFrame, covariates: list[str]) -> tuple[CoxPHFitter, pd.DataFrame]:
    """Multivariable Cox model: treatment plus confounders."""
    cols = [TREATMENT] + covariates + [TIME, EVENT]
    cph = CoxPHFitter().fit(df[cols], duration_col=TIME, event_col=EVENT)
    summ = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    summ.columns = ["HR", "HR_lcl", "HR_ucl", "p"]
    return cph, summ.round(3)


def ph_check(cph: CoxPHFitter, df: pd.DataFrame) -> pd.DataFrame:
    """Schoenfeld-residual test of proportional hazards (rank-transformed time)."""
    cols = list(cph.params_.index) + [TIME, EVENT]
    res = proportional_hazard_test(cph, df[cols], time_transform="rank")
    out = res.summary[["test_statistic", "p"]].round(3)
    out["ph_violation_flag_p<0.05"] = out["p"] < 0.05
    return out


def cox_stratified(df: pd.DataFrame, covariates: list[str], strata: list[str]) -> pd.DataFrame:
    """Sensitivity model: stratify on covariates flagged by the PH test so they get
    their own baseline hazards; report the treatment HR only."""
    d = df[[TREATMENT] + covariates + [TIME, EVENT]].copy()
    if "grade_2" in strata or "grade_3" in strata:
        d["grade"] = d["grade_2"] + 2 * d["grade_3"]
        d = d.drop(columns=["grade_2", "grade_3"])
        strata = [s for s in strata if s not in ("grade_2", "grade_3")] + ["grade"]
    if "age" in strata:
        d["age_band"] = pd.cut(d["age"], [0, 45, 55, 65, 120], labels=False)
        d = d.drop(columns=["age"])
        strata = [s for s in strata if s != "age"] + ["age_band"]
    cph = CoxPHFitter().fit(d, duration_col=TIME, event_col=EVENT, strata=strata)
    s = cph.summary.loc[TREATMENT]
    return pd.DataFrame([{"model": f"Cox stratified by {', '.join(strata)}",
                          "HR": round(s["exp(coef)"], 3),
                          "HR_lcl": round(s["exp(coef) lower 95%"], 3),
                          "HR_ucl": round(s["exp(coef) upper 95%"], 3),
                          "p": round(s["p"], 4)}])
