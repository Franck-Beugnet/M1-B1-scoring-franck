"""Train pyrenex_risk_v1 on the 2017 Lending Club subset.

Historical script — kept as-is for reproducibility of the v1 baseline.
Démarche 2017, basique, avec quelques angles morts assumés :
- Pas de Pipeline scikit-learn (preprocessing dispersé dans le script)
- Split aléatoire **sans** stratify (la cible étant 80/20, on espérait que ça
  passe — on a vu plus tard que non, cf. métriques par classe)
- Hyperparamètres RandomForest **par défaut** (n_estimators=100, max_depth=None)
- Pas de class_weight (le déséquilibre n'était pas adressé)
- Métriques rapportées : **accuracy** (trompeuse en déséquilibre) + matrice
  de confusion. F1 macro recalculé a posteriori pour comparaison équitable
  avec les futurs modèles.

Run::

    cd src && python train_v1.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE: int = 2017
DATA_PATH: Path = Path(__file__).parent.parent / "data" / "lending_club_2017_subset.csv"
MODELS_DIR: Path = Path(__file__).parent.parent / "models"

NUMERIC_FEATURES: list[str] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "revol_util",
]
CATEGORICAL_FEATURES: list[str] = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "emp_length",
]
TARGET_COLUMN: str = "loan_status"
TARGET_MAPPING: dict[str, int] = {"Fully Paid": 0, "Charged Off": 1}


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Train the v1 baseline and persist it."""
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COLUMN].map(TARGET_MAPPING).astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    X_num = X[NUMERIC_FEATURES]
    X_cat = X[CATEGORICAL_FEATURES]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    # /!\ Démarche 2017 : on fit le preprocessing sur TOUT le dataset
    # (avant le split). Aujourd'hui on sait que c'est une fuite légère.
    # On a laissé tel quel pour fidélité historique.
    X_num_imp = num_imputer.fit_transform(X_num)
    X_num_scaled = scaler.fit_transform(X_num_imp)
    X_cat_imp = cat_imputer.fit_transform(X_cat)
    X_cat_encoded = encoder.fit_transform(X_cat_imp)
    X_prepared = np.hstack([X_num_scaled, X_cat_encoded])

    # /!\ Pas de stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X_prepared, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Hyperparamètres par défaut
    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Métriques sur le test split (20% de 12k) ===\n")
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 macro  : {f1_macro:.4f}   (recalculé a posteriori — non rapporté en 2017)")
    print(f"ROC-AUC   : {roc_auc:.4f}   (recalculé a posteriori)")
    print("\nMatrice de confusion :")
    print(cm)
    print("\nClassification report :")
    print(classification_report(y_test, y_pred, target_names=["Fully Paid", "Charged Off"]))

    # Persistance — bundle preprocessor + model dans un seul .joblib
    bundle = {
        "num_imputer": num_imputer,
        "scaler": scaler,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "model": model,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_mapping": TARGET_MAPPING,
    }
    model_path = MODELS_DIR / "pyrenex_risk_v1.joblib"
    joblib.dump(bundle, model_path, compress=3)

    metadata = {
        "model_name": "pyrenex_risk_v1",
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "training_dataset": DATA_PATH.name,
        "dataset_sha256": sha256_of(DATA_PATH),
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "default_rate_overall": float((df[TARGET_COLUMN] == "Charged Off").mean()),
        "metrics_test_split": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
        },
        "notes": [
            "Démarche 2017 — pas de stratify, pas de class_weight, hyperparams default.",
            "F1 macro et ROC-AUC recalculés a posteriori (non rapportés à l'époque).",
            "Preprocessing fit sur tout le dataset avant split (fuite légère assumée).",
        ],
    }
    meta_path = MODELS_DIR / "pyrenex_risk_v1.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Modèle sauvegardé    : {model_path}")
    print(f"✓ Métadonnées sauvegardées : {meta_path}")


if __name__ == "__main__":
    main()
