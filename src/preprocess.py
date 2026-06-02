"""Reproducible preprocessing pipeline for Pyrenex Crédit scoring.

Build a scikit-learn Pipeline + ColumnTransformer that handles:
- Numeric features: imputation + scaling
- Categorical features: imputation + one-hot encoding
- Target: binary mapping

This module is imported by `train.py` and `evaluate.py`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# TODO — fill these lists from your EDA
NUMERIC_FEATURES: list[str] = [
    # e.g. "loan_amnt", "int_rate", "annual_inc", ...
]
CATEGORICAL_FEATURES: list[str] = [
    # e.g. "term", "grade", "home_ownership", "purpose", ...
]
TARGET_COLUMN: str = "loan_status"
TARGET_MAPPING: dict[str, int] = {"Fully Paid": 0, "Charged Off": 1}


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load Lending Club CSV and split features/target.

    Args:
        path: CSV path.

    Returns:
        (X, y) where X is the feature DataFrame and y the target Series.
    """
    df = pd.read_csv(path)
    y = df[TARGET_COLUMN].map(TARGET_MAPPING)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer applying numeric + categorical pipelines."""
    if not NUMERIC_FEATURES or not CATEGORICAL_FEATURES:
        raise ValueError(
            "NUMERIC_FEATURES and CATEGORICAL_FEATURES are empty in "
            "src/preprocess.py. Fill them from your EDA before training. "
            "See ressources/01_Pandas_Sklearn_split_essentiel.md."
        )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
