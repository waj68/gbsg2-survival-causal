"""Table 1 stratified by treatment, with standardized mean differences (SMD)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    m = np.average(x, weights=w)
    v = np.average((x - m) ** 2, weights=w)
    return float(m), float(v)


def smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> float:
    """Absolute SMD; pooled SD from unweighted groups so before/after are comparable."""
    w = np.ones_like(x, dtype=float) if w is None else w
    m1, _ = weighted_mean_var(x[t == 1], w[t == 1])
    m0, _ = weighted_mean_var(x[t == 0], w[t == 0])
    sd = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
    return float(abs(m1 - m0) / sd) if sd > 0 else 0.0


def table_one(df: pd.DataFrame, treatment: str, covariates: list[str]) -> pd.DataFrame:
    t = df[treatment].to_numpy()
    rows = []
    for c in covariates:
        x = df[c].to_numpy(dtype=float)
        binary = set(np.unique(x)) <= {0.0, 1.0}
        def fmt(v):
            if binary:
                return f"{int(v.sum())} ({100 * v.mean():.1f}%)"
            return f"{np.median(v):.1f} [{np.percentile(v, 25):.1f}–{np.percentile(v, 75):.1f}]"
        rows.append({
            "variable": c,
            "no_hormone_tx": fmt(x[t == 0]),
            "hormone_tx": fmt(x[t == 1]),
            "smd": round(smd(x, t), 3),
        })
    tab = pd.DataFrame(rows)
    tab.loc[len(tab)] = ["n", int((t == 0).sum()), int((t == 1).sum()), np.nan]
    return tab
