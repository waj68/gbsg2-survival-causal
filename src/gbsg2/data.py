"""Load GBSG2, derive analysis variables, and run data-quality checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sksurv.datasets import load_gbsg2

from .config import EVENT, TIME, TREATMENT


def load_raw() -> pd.DataFrame:
    """Return GBSG2 as one tidy frame (686 women, node-positive breast cancer)."""
    X, y = load_gbsg2()
    df = X.copy()
    df[TIME] = y["time"]
    df[EVENT] = y["cens"].astype(int)  # sksurv's 'cens' is the event indicator
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready variables. Log1p for right-skewed counts/receptor levels."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float)
    out["postmenopausal"] = (df["menostat"] == "Post").astype(int)
    out["tumour_size_mm"] = df["tsize"].astype(float)
    out["grade_2"] = (df["tgrade"] == "II").astype(int)
    out["grade_3"] = (df["tgrade"] == "III").astype(int)
    out["pos_nodes"] = df["pnodes"].astype(float)
    out["log_pos_nodes"] = np.log1p(df["pnodes"].astype(float))
    out["log_prog_rec"] = np.log1p(df["progrec"].astype(float))
    out["log_estr_rec"] = np.log1p(df["estrec"].astype(float))
    out[TREATMENT] = (df["horTh"] == "yes").astype(int)
    out[TIME] = df[TIME].astype(float)
    out[EVENT] = df[EVENT].astype(int)
    return out


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness, ranges, and plausibility flags per variable."""
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "variable": c,
            "n_missing": int(s.isna().sum()),
            "min": float(s.min()),
            "median": float(s.median()),
            "max": float(s.max()),
        })
    rep = pd.DataFrame(rows)
    checks = {
        "non_positive_follow_up": int((df[TIME] <= 0).sum()),
        "age_outside_18_100": int(((df["age"] < 18) | (df["age"] > 100)).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    rep.attrs["checks"] = checks
    return rep


def load() -> pd.DataFrame:
    return prepare(load_raw())
