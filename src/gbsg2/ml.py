"""Prognostic modelling: Cox PH vs random survival forest on a held-out test set."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score
from sksurv.util import Surv

from .config import EVENT, SEED, TIME, TREATMENT

FEATURES_FN = lambda covs: [TREATMENT] + covs


def split(df: pd.DataFrame, covariates: list[str], test_size: float = 0.3):
    X = df[FEATURES_FN(covariates)].astype(float)
    y = Surv.from_arrays(event=df[EVENT].astype(bool), time=df[TIME])
    return train_test_split(X, y, test_size=test_size, random_state=SEED, stratify=df[EVENT])


def fit_models(X_tr, y_tr) -> dict:
    cox = CoxPHSurvivalAnalysis(alpha=1e-4).fit(X_tr, y_tr)
    rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=15, max_features="sqrt",
                               n_jobs=-1, random_state=SEED).fit(X_tr, y_tr)
    return {"Cox PH": cox, "Random survival forest": rsf}


def evaluate(models: dict, X_tr, y_tr, X_te, y_te) -> pd.DataFrame:
    # Evaluation grid kept inside test follow-up range for IPCW/Brier validity
    lo, hi = np.percentile(y_te["time"][y_te["event"]], [10, 80])
    grid = np.linspace(lo, hi, 25)
    tau = hi
    rows = []
    for name, m in models.items():
        risk = m.predict(X_te)
        surv = np.vstack([fn(grid) for fn in m.predict_survival_function(X_te)])
        rows.append({
            "model": name,
            "harrell_c": round(concordance_index_censored(y_te["event"], y_te["time"], risk)[0], 3),
            "uno_c_ipcw": round(concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0], 3),
            "integrated_brier": round(integrated_brier_score(y_tr, y_te, surv, grid), 3),
        })
    return pd.DataFrame(rows)


def rsf_importance(model, X_te, y_te, path=None) -> pd.DataFrame:
    r = permutation_importance(model, X_te, y_te, n_repeats=20, random_state=SEED)
    imp = (pd.DataFrame({"feature": X_te.columns, "mean_drop_in_c": r.importances_mean,
                         "sd": r.importances_std})
           .sort_values("mean_drop_in_c", ascending=False).round(4))
    if path is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        s = imp.sort_values("mean_drop_in_c")
        ax.barh(s["feature"], s["mean_drop_in_c"], xerr=s["sd"])
        ax.set_xlabel("Mean decrease in C-index (permutation)")
        ax.set_title("Random survival forest: permutation importance")
        fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return imp
