"""Run the full GBSG2 analysis end to end: `python run_pipeline.py`."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np

from gbsg2 import causal, data, descriptives, ml, survival
import os

from gbsg2.config import CONFOUNDERS, FIG, HORIZON_DAYS, OUT, SEED, TAB, TREATMENT

N_BOOT = int(os.environ.get("N_BOOT", 300))


def main() -> dict:
    np.random.seed(SEED)
    for p in (FIG, TAB):
        p.mkdir(parents=True, exist_ok=True)

    # 1. Data and quality checks
    df = data.load()
    qc = data.quality_report(df)
    qc.to_csv(TAB / "01_data_quality.csv", index=False)

    # 2. Descriptives
    t1 = descriptives.table_one(df, TREATMENT, CONFOUNDERS + ["pos_nodes"])
    t1.to_csv(TAB / "02_table1.csv", index=False)

    # 3. Unadjusted survival
    km = survival.km_by_group(df, FIG / "01_km_unweighted.png")
    km.to_csv(TAB / "03_km_unweighted.csv", index=False)
    p_lr = survival.logrank(df)

    # 4. Conditional (covariate-adjusted) Cox model + PH diagnostics
    cph, cox_tab = survival.cox_model(df, CONFOUNDERS)
    cox_tab.to_csv(TAB / "04_cox_multivariable.csv")
    ph = survival.ph_check(cph, df)
    ph.to_csv(TAB / "05_ph_test.csv")
    flagged = [v for v in ph.index[ph["ph_violation_flag_p<0.05"]] if v != TREATMENT]
    strat = survival.cox_stratified(df, CONFOUNDERS, flagged) if flagged else None
    if strat is not None:
        strat.to_csv(TAB / "05b_cox_stratified_sensitivity.csv", index=False)

    # 5. Propensity score / IPTW (marginal effect)
    ps = causal.propensity_scores(df, CONFOUNDERS)
    w = causal.iptw(df, ps)
    bal = causal.balance_table(df, CONFOUNDERS, w)
    bal.to_csv(TAB / "06_balance.csv", index=False)
    causal.plot_overlap(df, ps, FIG / "02_ps_overlap.png")
    causal.plot_love(bal, FIG / "03_love_plot.png")
    km_w = survival.km_by_group(df, FIG / "04_km_iptw.png", weights=w,
                                title="IPTW-weighted recurrence-free survival")
    km_w.to_csv(TAB / "07_km_iptw.csv", index=False)
    marg = causal.weighted_cox(df, w)
    marg.to_csv(TAB / "08_iptw_cox.csv", index=False)
    hr, ucl = marg.loc[0, "HR"], marg.loc[0, "HR_ucl"]
    boot = causal.bootstrap(df, CONFOUNDERS, HORIZON_DAYS, n_boot=N_BOOT)
    boot.to_csv(TAB / "08b_iptw_bootstrap.csv", index=False)
    ev = causal.e_value_hr(hr, ucl if hr < 1 else marg.loc[0, "HR_lcl"])

    # 6. Prognostic ML comparison
    X_tr, X_te, y_tr, y_te = ml.split(df, CONFOUNDERS)
    models = ml.fit_models(X_tr, y_tr)
    perf = ml.evaluate(models, X_tr, y_tr, X_te, y_te)
    perf.to_csv(TAB / "09_model_performance.csv", index=False)
    imp = ml.rsf_importance(models["Random survival forest"], X_te, y_te, FIG / "05_rsf_importance.png")
    imp.to_csv(TAB / "10_rsf_importance.csv", index=False)

    summary = {
        "n": int(len(df)), "events": int(df["event"].sum()),
        "qc_checks": qc.attrs["checks"],
        "logrank_p": round(p_lr, 4),
        "cox_adjusted_hr_hormone": cox_tab.loc[TREATMENT].to_dict(),
        "ph_global_violation_any": bool(ph["ph_violation_flag_p<0.05"].any()),
        "iptw_marginal_hr": marg.iloc[0].to_dict(),
        "ph_sensitivity_stratified": None if strat is None else strat.iloc[0].to_dict(),
        "iptw_bootstrap": boot.to_dict(orient="records"),
        "e_value": ev,
        "ess_iptw": round(causal.effective_sample_size(w), 1),
        "max_smd_unweighted": float(bal["smd_unweighted"].max()),
        "max_smd_weighted": float(bal["smd_weighted"].max()),
        "model_performance": perf.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
