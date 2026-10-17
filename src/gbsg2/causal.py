"""Propensity-score (IPTW) estimate of the hormone-therapy effect on RFS.

Treatment here was not assigned at random for everyone (allocation differs strongly
by menopausal status in these data), so it is analysed as an observational exposure.
Assumptions: no unmeasured confounding, positivity, correct PS model, no interference.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import warnings

from lifelines import KaplanMeierFitter
from sklearn.linear_model import LogisticRegression

from .config import EVENT, TIME, TREATMENT
from .descriptives import smd


def propensity_scores(df: pd.DataFrame, covariates: list[str]) -> pd.Series:
    model = LogisticRegression(C=np.inf, max_iter=5000)
    model.fit(df[covariates], df[TREATMENT])
    return pd.Series(model.predict_proba(df[covariates])[:, 1], index=df.index, name="ps")


def iptw(df: pd.DataFrame, ps: pd.Series, trim: tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Stabilized ATE weights, truncated at the given percentiles."""
    t = df[TREATMENT]
    p_t = t.mean()
    w = np.where(t == 1, p_t / ps, (1 - p_t) / (1 - ps))
    lo, hi = np.quantile(w, trim)
    return pd.Series(np.clip(w, lo, hi), index=df.index, name="w")


def effective_sample_size(w: pd.Series) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum())


def balance_table(df: pd.DataFrame, covariates: list[str], w: pd.Series) -> pd.DataFrame:
    t = df[TREATMENT].to_numpy()
    rows = [{"variable": c,
             "smd_unweighted": round(smd(df[c].to_numpy(float), t), 3),
             "smd_weighted": round(smd(df[c].to_numpy(float), t, w.to_numpy()), 3)}
            for c in covariates]
    return pd.DataFrame(rows)


def weighted_cox(df: pd.DataFrame, w: pd.Series) -> pd.DataFrame:
    """Marginal HR from a weighted Cox model with robust (sandwich) SEs."""
    d = df[[TREATMENT, TIME, EVENT]].assign(w=w)
    cph = CoxPHFitter().fit(d, duration_col=TIME, event_col=EVENT, weights_col="w", robust=True)
    s = cph.summary.loc[TREATMENT]
    return pd.DataFrame([{"estimand": "Marginal HR (IPTW, ATE)",
                          "HR": round(s["exp(coef)"], 3),
                          "HR_lcl": round(s["exp(coef) lower 95%"], 3),
                          "HR_ucl": round(s["exp(coef) upper 95%"], 3),
                          "p": round(s["p"], 4)}])


def e_value_hr(hr: float, ci_limit: float) -> dict:
    """VanderWeele & Ding E-value, using the common-outcome HR→RR approximation."""
    def to_rr(h):
        return (1 - 0.5 ** np.sqrt(h)) / (1 - 0.5 ** np.sqrt(1 / h))
    def ev(rr):
        rr = 1 / rr if rr < 1 else rr
        return float(rr + np.sqrt(rr * (rr - 1)))
    rr_ci = to_rr(ci_limit)
    ci_ev = 1.0 if (hr < 1 <= ci_limit or hr > 1 >= ci_limit) else ev(rr_ci)
    return {"e_value_point": round(ev(to_rr(hr)), 2), "e_value_ci": round(ci_ev, 2)}


def plot_overlap(df: pd.DataFrame, ps: pd.Series, path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for g, label in [(0, "No hormone therapy"), (1, "Hormone therapy")]:
        ax.hist(ps[df[TREATMENT] == g], bins=30, alpha=0.55, density=True, label=label)
    ax.set_xlabel("Propensity score"); ax.set_ylabel("Density")
    ax.set_title("Propensity score overlap (positivity check)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_love(bal: pd.DataFrame, path) -> None:
    b = bal.sort_values("smd_unweighted")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(b["smd_unweighted"], b["variable"], label="Unweighted", marker="o")
    ax.scatter(b["smd_weighted"], b["variable"], label="IPTW-weighted", marker="s")
    ax.axvline(0.1, ls="--", c="grey", lw=1)
    ax.set_xlabel("Absolute standardized mean difference"); ax.set_title("Covariate balance")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _marginal_estimates(df: pd.DataFrame, covariates: list[str], horizon: float) -> tuple[float, float]:
    """Re-estimate PS and weights, return (marginal HR, 5-y RFS difference treated − untreated)."""
    ps = propensity_scores(df, covariates)
    w = iptw(df, ps)
    hr = weighted_cox(df, w).loc[0, "HR"]
    s = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in (0, 1):
            sub = df[df[TREATMENT] == g]
            km = KaplanMeierFitter().fit(sub[TIME], sub[EVENT], weights=w.loc[sub.index])
            s[g] = float(km.survival_function_at_times(horizon).iloc[0])
    return float(hr), s[1] - s[0]


def bootstrap(df: pd.DataFrame, covariates: list[str], horizon: float,
              n_boot: int = 300, seed: int = 2026) -> pd.DataFrame:
    """Nonparametric bootstrap re-fitting the PS model in every replicate, so CIs
    reflect uncertainty in the weights (naive weighted-KM CIs ignore it)."""
    rng = np.random.default_rng(seed)
    point = _marginal_estimates(df, covariates, horizon)
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        b = df.iloc[idx].reset_index(drop=True)
        reps.append(_marginal_estimates(b, covariates, horizon))
    reps = np.array(reps)
    rows = []
    for j, name in enumerate(["Marginal HR (IPTW)", "5-year RFS difference (IPTW)"]):
        lo, hi = np.percentile(reps[:, j], [2.5, 97.5])
        rows.append({"estimand": name, "estimate": round(point[j], 3),
                     "boot_lcl": round(lo, 3), "boot_ucl": round(hi, 3), "n_boot": n_boot})
    return pd.DataFrame(rows)
