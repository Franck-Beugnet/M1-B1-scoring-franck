"""Generate the historical 2017 Lending Club subset used to train pyrenex_risk_v1.

Reproducible (``random_state=2017``). ~12k rows, target ~14% default.
Compared to the 2025 dataset (M1-B1 brief):
- Smaller volume (12k vs 30k)
- Cleaner data (no injected NaN — assumed better data quality in 2017)
- Slightly different grade distribution (more A/B, less E/F/G — pre-crisis credit mix)
- Same schema (columns identical to 2025) — so the v1 model can be re-evaluated
  on the new data with no schema adapter.

Run from this folder::

    python generate_2017.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE: int = 2017
N_ROWS: int = 12_000
OUTPUT_PATH: Path = Path(__file__).parent / "lending_club_2017_subset.csv"

GRADES: list[str] = ["A", "B", "C", "D", "E", "F", "G"]
# Slightly safer 2017 mix vs 2025
GRADE_PROBS: list[float] = [0.22, 0.34, 0.26, 0.11, 0.05, 0.015, 0.005]

GRADE_RATE_MEAN: dict[str, float] = {
    "A": 7.0, "B": 10.5, "C": 13.5, "D": 16.5, "E": 20.0, "F": 24.0, "G": 28.0,
}
GRADE_RATE_STD: dict[str, float] = {
    "A": 0.8, "B": 1.0, "C": 1.2, "D": 1.5, "E": 1.8, "F": 2.0, "G": 2.5,
}
GRADE_FICO_MEAN: dict[str, int] = {
    "A": 745, "B": 715, "C": 700, "D": 685, "E": 670, "F": 660, "G": 650,
}
GRADE_FICO_STD: int = 12

GRADE_DEFAULT_PROB: dict[str, float] = {
    "A": 0.05, "B": 0.10, "C": 0.16, "D": 0.24, "E": 0.32, "F": 0.40, "G": 0.45,
}

TERMS: list[str] = ["36 months", "60 months"]
TERM_PROBS: list[float] = [0.76, 0.24]  # shorter terms more common pre-2018

HOME_OWNERSHIP: list[str] = ["MORTGAGE", "RENT", "OWN", "OTHER"]
HOME_OWNERSHIP_PROBS: list[float] = [0.50, 0.39, 0.10, 0.01]

PURPOSES: list[str] = [
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "other",
]
PURPOSE_PROBS: list[float] = [0.55, 0.24, 0.07, 0.04, 0.03, 0.025, 0.02, 0.025]

VERIFICATION_STATUS: list[str] = ["Verified", "Source Verified", "Not Verified"]
VERIFICATION_PROBS: list[float] = [0.36, 0.28, 0.36]

EMP_LENGTHS: list[str] = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
    "6 years", "7 years", "8 years", "9 years", "10+ years",
]
EMP_LENGTH_PROBS: list[float] = [
    0.07, 0.08, 0.09, 0.08, 0.07, 0.07, 0.05, 0.05, 0.05, 0.04, 0.35,
]


def generate() -> pd.DataFrame:
    """Generate the 2017 subset."""
    rng = np.random.default_rng(seed=RANDOM_STATE)
    n = N_ROWS

    grade = rng.choice(GRADES, size=n, p=GRADE_PROBS)
    int_rate = np.array(
        [rng.normal(GRADE_RATE_MEAN[g], GRADE_RATE_STD[g]) for g in grade]
    ).clip(5.0, 30.0).round(2)
    fico_range_low = (
        np.array([rng.normal(GRADE_FICO_MEAN[g], GRADE_FICO_STD) for g in grade])
        .clip(610, 845).round().astype(int)
    )
    loan_amnt = (
        rng.lognormal(mean=9.3, sigma=0.55, size=n)
        .clip(500, 40_000).round(-2).astype(int)
    )
    annual_inc = (
        rng.lognormal(mean=11.05, sigma=0.55, size=n)
        .clip(12_000, 500_000).round(-2).astype(int)
    )
    term = rng.choice(TERMS, size=n, p=TERM_PROBS)
    home_ownership = rng.choice(HOME_OWNERSHIP, size=n, p=HOME_OWNERSHIP_PROBS)
    purpose = rng.choice(PURPOSES, size=n, p=PURPOSE_PROBS)
    verification_status = rng.choice(VERIFICATION_STATUS, size=n, p=VERIFICATION_PROBS)
    emp_length = rng.choice(EMP_LENGTHS, size=n, p=EMP_LENGTH_PROBS)
    dti = rng.gamma(shape=4.0, scale=4.0, size=n).clip(0.0, 50.0).round(2)
    delinq_2yrs = rng.poisson(lam=0.25, size=n).clip(0, 8)
    revol_util = (rng.beta(2.5, 2.5, size=n) * 100).round(1)

    n_months = np.where(term == "36 months", 36, 60)
    monthly_rate = int_rate / 100.0 / 12.0
    installment = (
        loan_amnt * monthly_rate * (1 + monthly_rate) ** n_months
        / ((1 + monthly_rate) ** n_months - 1)
    ).round(2)

    # Cleaner 2017 data → no NaN injection (in contrast to 2025)
    base_prob = np.array([GRADE_DEFAULT_PROB[g] for g in grade])
    dti_adj = (dti - 17.0) / 100.0
    fico_adj = (700.0 - fico_range_low) / 1500.0
    revol_adj = (revol_util - 50.0) / 400.0
    delinq_adj = delinq_2yrs * 0.03
    term_adj = np.where(term == "60 months", 0.04, 0.0)

    prob = (base_prob + dti_adj + fico_adj + revol_adj + delinq_adj + term_adj).clip(
        0.01, 0.95
    )
    target = rng.binomial(1, prob)
    loan_status = np.where(target == 1, "Charged Off", "Fully Paid")

    return pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "installment": installment,
        "grade": grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "fico_range_low": fico_range_low,
        "revol_util": revol_util,
        "loan_status": loan_status,
    })


def main() -> None:
    """Generate and write the CSV."""
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    default_rate = (df["loan_status"] == "Charged Off").mean()
    print(f"Generated {len(df):,} rows → {OUTPUT_PATH.name}")
    print(f"  Default rate: {default_rate:.2%}")


if __name__ == "__main__":
    main()