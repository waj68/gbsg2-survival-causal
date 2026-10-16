"""Central configuration: variable roles, seeds, and output paths."""
from pathlib import Path

SEED = 2026
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"

TIME = "time_days"
EVENT = "event"            # 1 = recurrence or death (recurrence-free survival)
TREATMENT = "hormone_tx"   # 1 = tamoxifen (horTh == "yes")

# Pre-treatment covariates used for adjustment and propensity modelling
CONFOUNDERS = [
    "age", "postmenopausal", "tumour_size_mm", "grade_2", "grade_3",
    "log_pos_nodes", "log_prog_rec", "log_estr_rec",
]
HORIZON_DAYS = 5 * 365.25  # 5-year landmark
