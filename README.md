# Hormone therapy and recurrence-free survival in node-positive breast cancer
### A reproducible survival, causal-inference and machine-learning pipeline in Python (GBSG2)

**Author:** Abdul Wajid, MBBS, MSc, MS, PhD — physician-epidemiologist

**Status:** v0.1 (October 2026)

## Question
Among women with node-positive breast cancer, what is the effect of hormone therapy (tamoxifen) on recurrence-free survival (RFS)? How well do a Cox model and a random survival forest predict individual risk?

## Data
GBSG2 (German Breast Cancer Study Group 2) comprises 686 women with primary node-positive breast cancer, with 299 recurrence-or-death events over a median follow-up of about 3 years. The data are loaded from `scikit-survival` (`load_gbsg2`), so nothing needs to be downloaded or shared. The original trial is described in Schumacher et al., *J Clin Oncol* 1994.

**Why a causal approach on trial data?** Hormone-therapy allocation is strongly unbalanced in this dataset. Of those treated, 76% are postmenopausal, compared with 48% of the untreated, and treated women are older (Table 1, SMD > 0.5). The comparison is therefore analysed as an observational exposure, the way it would be in administrative or registry data.

## Pipeline
| Step | Module | What it does |
|---|---|---|
| 1. Data & QC | `data.py` | Derives variables (log-transformed nodes and receptors, grade dummies) and checks missingness, ranges, duplicates and follow-up |
| 2. Descriptives | `descriptives.py` | Table 1 by treatment with standardized mean differences |
| 3. Survival | `survival.py` | Kaplan–Meier curves, log-rank test, 5-year RFS, multivariable Cox model, Schoenfeld PH test, stratified sensitivity model |
| 4. Causal | `causal.py` | Propensity score, stabilized and truncated IPTW, balance diagnostics, weighted KM, marginal HR with robust SE, bootstrap CIs (PS model re-fitted in each replicate), E-value |
| 5. Prediction | `ml.py` | 70/30 split; Cox PH vs random survival forest; Harrell's C, Uno's IPCW C, integrated Brier score; permutation importance |

Outputs are written to `outputs/tables/` (CSV), `outputs/figures/` (PNG) and `outputs/summary.json`.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # 8 checks: data integrity, balance, plausibility
python run_pipeline.py    # full analysis (~1–2 min; N_BOOT=300 by default)
```

## Results
**Confounding and balance.** Before weighting, age (SMD 0.57) and menopausal status (0.61) were badly imbalanced. After IPTW, every covariate had an SMD below 0.02. The effective sample size was 620 of 686, and propensity scores overlapped well.

**Effect of hormone therapy on RFS**

| Estimate | HR (95% CI) |
|---|---|
| Multivariable Cox (conditional) | 0.68 (0.53–0.88) |
| Cox stratified by grade and age band (PH sensitivity) | 0.67 (0.51–0.86) |
| IPTW marginal HR, robust SE | 0.67 (0.51–0.88) |
| IPTW marginal HR, bootstrap (300 reps) | 0.67 (0.53–0.84) |

The IPTW-weighted 5-year RFS was 59.6% with hormone therapy and 42.3% without. The difference is **17.3 percentage points** (bootstrap 95% CI 8.8–25.4).

**Robustness.** The PH test flagged age and grade 3 (p ≈ 0.04) but not treatment (p = 0.60). Stratifying on both left the treatment HR unchanged. The E-value is 1.97 for the point estimate and 1.42 for the CI bound. An unmeasured confounder would therefore need to be associated with both treatment and RFS by a risk ratio of about 2 to explain away the estimate, and about 1.4 to move the CI to include the null.

**Prediction (held-out 30%)**

| Model | Harrell's C | Uno's C (IPCW) | Integrated Brier |
|---|---|---|---|
| Cox PH | 0.705 | 0.702 | 0.145 |
| Random survival forest | 0.695 | 0.688 | 0.148 |

The forest did not outperform Cox, which is a common finding with low-dimensional clinical data. Permutation importance ranked positive nodes first, then progesterone receptor level and age, which is consistent with established prognostic factors.

## Interpretation
After addressing confounding by age and menopausal status, hormone therapy was associated with roughly a one-third lower hazard of recurrence or death, a finding consistent across conditional, marginal and stratified models. Machine learning added no discrimination over a well-specified Cox model here. In practice that argues for the simpler, interpretable model unless richer data (treatment sequences, comorbidity, imaging, genomics) are available.

## Limitations
- The analysis relies on no unmeasured confounding and a correctly specified PS model. The E-value is reported to quantify this.
- The marginal HR is a weighted average over follow-up and is non-collapsible, so it is not directly comparable to the conditional HR. The 5-year RFS difference is the more interpretable estimand.
- Naive CIs on weighted KM curves ignore weight estimation; the bootstrap intervals are the ones to use.
- The single train/test split is small. Repeated cross-validation is a planned extension.

## Planned extensions
- Repeated cross-validation and calibration plots
- Restricted mean survival time (RMST) difference
- Doubly robust estimation (AIPW)
- A synthetic administrative-data version using Synthea cancer modules

## Repository layout
```
├── run_pipeline.py          # entry point
├── src/gbsg2/               # config, data, descriptives, survival, causal, ml
├── tests/test_pipeline.py   # pytest checks
├── outputs/                 # generated tables, figures, summary.json
├── requirements.txt         # pinned versions
└── Makefile                 # make install | test | run
```
