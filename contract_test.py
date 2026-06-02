"""Contract test du modèle Pyrenex-risk-v2 — à compléter.

Ce script valide que le `.joblib` packagé respecte la signature attendue
par l'API M1-B2. Plus exigeant qu'un simple `print OK` : on vérifie
shapes, classes, bornes des probas, et stabilité prédictive vs notebook.

Mini-cours d'appui : `ressources/05_Persistance_modele_joblib_essentiel.md`
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "models" / "pyrenex_risk_v2.joblib"


def contract_test_model(
    model_path: Path,
    x_sample: pd.DataFrame,
    expected_classes: set[int] | None = None,
    expected_first_proba: list[float] | None = None,
) -> None:
    """Valide schéma + stabilité prédictive d'un modèle rechargé.

    Args:
        model_path: chemin vers le `.joblib` à valider.
        x_sample: DataFrame d'au moins 3 lignes aligné sur `feature_columns`.
        expected_classes: classes attendues (ex. ``{0, 1}``).
        expected_first_proba: probabilités attendues pour la 1re ligne,
            issues d'une référence notebook.
    """
    pipeline = joblib.load(model_path)
    prediction = pipeline.predict(x_sample.head(3))
    proba = pipeline.predict_proba(x_sample.head(3))

    assert prediction.shape == (3,), f"shape predict={prediction.shape}, attendu (3,)"
    assert proba.shape == (3, 2), f"shape predict_proba={proba.shape}, attendu (3, 2)"
    assert (proba >= 0).all() and (proba <= 1).all(), "probabilités hors [0, 1]"

    if expected_classes is not None:
        observed = set(prediction.tolist())
        assert observed <= expected_classes, (
            f"classes inattendues : {observed - expected_classes}"
        )

    if expected_first_proba is not None:
        observed = proba[0].round(6).tolist()
        reference = [round(p, 6) for p in expected_first_proba]
        assert observed == reference, (
            f"dérive prédictive — observé {observed}, référence notebook {reference}"
        )

    print("Contract test OK — shapes valides, probas dans [0,1], stabilité confirmée.")


if __name__ == "__main__":
    # 1. Capture la référence proba depuis ton notebook (à coller en cellule) :
    #
    #    import joblib, pandas as pd
    #    pipeline = joblib.load("../models/pyrenex_risk_v2.joblib")
    #    x_hold = pd.read_csv("../data/lending_club_holdout.csv").drop(columns=["loan_status"])
    #    print(pipeline.predict_proba(x_hold.head(3))[0].tolist())
    #
    # 2. Colle les deux floats affichés dans `expected_first_proba` ci-dessous,
    #    puis lance `python contract_test.py` depuis la racine du repo.
    #
    # ⚠️ Le `.drop(columns=["loan_status"])` est essentiel : le pipeline
    #    attend les features uniquement, pas la cible.

    expected_first_proba: list[float] | None = None  # TODO — colle ici les 2 floats du print

    if expected_first_proba is None:
        raise NotImplementedError(
            "Renseigne `expected_first_proba` à partir du snippet du notebook "
            "ci-dessus, puis relance ce script."
        )

    x_holdout = pd.read_csv(
        Path(__file__).parent / "data" / "lending_club_holdout.csv"
    ).drop(columns=["loan_status"])

    contract_test_model(
        MODEL_PATH,
        x_sample=x_holdout,
        expected_classes={0, 1},
        expected_first_proba=expected_first_proba,
    )