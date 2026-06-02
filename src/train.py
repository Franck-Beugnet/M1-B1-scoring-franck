"""Train Pyrenex Crédit risk model — M1-B1.

Usage:
    python src/train.py --config default
    python src/train.py --config balanced
    python src/train.py --config trial_01
    python src/train.py --config all

Each run writes:
    models/pyrenex_risk_v2_<config>.joblib   (full Pipeline)
    models/pyrenex_risk_v2_<config>.json     (metadata, no holdout metric yet)

Once you have chosen which configuration to retain, promote it to the
canonical name expected by `evaluate.py` and `contract_test.py`:

    cp models/pyrenex_risk_v2_<chosen>.joblib models/pyrenex_risk_v2.joblib
    cp models/pyrenex_risk_v2_<chosen>.json   models/pyrenex_risk_v2.json

Then `python src/evaluate.py --update-meta` fills in `metrics_holdout`.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    build_preprocessor,
    load_dataset,
)

BASE_CONFIGS: dict[str, dict] = {
    "default": {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    },
    "balanced": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    },
    "fbt": {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_leaf": 10,
        "min_samples_split": 25,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1,
    },
    "gb_proto": {
        "model_type": "gradient_boosting",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 15,
        "min_samples_split": 30,
        "subsample": 0.8,
        "max_features": "sqrt",
        "random_state": 42,
    },
}

# Espace de recherche pour les essais supplémentaires.
# n_estimators : plus d'arbres améliore souvent la stabilité/ROC-AUC, mais augmente le temps d'entraînement.
# max_depth : une profondeur faible régularise (moins d'overfitting), une profondeur élevée capte des motifs complexes.
# min_samples_leaf : une valeur plus haute lisse les feuilles et améliore souvent la généralisation sur la classe minoritaire.
# min_samples_split : une valeur plus haute évite des splits trop agressifs et réduit la variance.
# class_weight : augmente le poids de la classe défaut (1) pour améliorer recall_default, au prix de la précision.
# max_features : utiliser moins de features par split décorrèle les arbres et peut améliorer la robustesse.
SWEEP_SPACE = {
    "n_estimators": [200, 350, 500],
    "max_depth": [10, 14],
    "min_samples_leaf": [5, 10],
    "min_samples_split": [10, 25],
    "class_weight": ["balanced", {0: 1.0, 1: 2.0}],
    "max_features": ["sqrt", 0.6],
}


def build_sweep_configs(max_trials: int = 10) -> dict[str, dict]:
    """Génère des configs d'essai depuis une grille d'hyperparamètres compacte et déterministe."""
    trials: dict[str, dict] = {}
    all_combinations = product(
        SWEEP_SPACE["n_estimators"],
        SWEEP_SPACE["max_depth"],
        SWEEP_SPACE["min_samples_leaf"],
        SWEEP_SPACE["min_samples_split"],
        SWEEP_SPACE["class_weight"],
        SWEEP_SPACE["max_features"],
    )

    for idx, combo in enumerate(all_combinations, start=1):
        if idx > max_trials:
            break
        (
            n_estimators,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            class_weight,
            max_features,
        ) = combo

        trials[f"trial_{idx:02d}"] = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "min_samples_split": min_samples_split,
            "class_weight": class_weight,
            "max_features": max_features,
            "random_state": 42,
            "n_jobs": -1,
        }

    return trials


CONFIGS: dict[str, dict] = {**BASE_CONFIGS, **build_sweep_configs(max_trials=10)}


def train(config_name: str, data_path: Path, output_dir: Path) -> dict:
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Available: {list(CONFIGS)}")
    params = CONFIGS[config_name]
    model_type = params.get("model_type", "random_forest")

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_samples_split=params["min_samples_split"],
                    subsample=params["subsample"],
                    max_features=params["max_features"],
                    random_state=params["random_state"],
                )
                if model_type == "gradient_boosting"
                else RandomForestClassifier(**params),
            ),
        ]
    )

    if model_type == "gradient_boosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_default": f1_score(y_test, y_pred, pos_label=1),
        "precision_default": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall_default": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"pyrenex_risk_v2_{config_name}.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    meta = {
        "model_name": "pyrenex_risk_v2",
        "model_version": "v2.0.0",
        "config_name": config_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "dataset_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "hyperparameters": params,
        "model_type": model_type,
        "metrics_test_internal": {
            k: round(v, 4) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
        },
        "feature_columns": {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "target": {"column": TARGET_COLUMN, "mapping": TARGET_MAPPING},
    }
    meta_path = output_dir / f"pyrenex_risk_v2_{config_name}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"model_path": model_path, "meta_path": meta_path, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pyrenex risk model")
    parser.add_argument(
        "--config",
        default="default",
        help="Config to run. Use one of CONFIGS keys or 'all' to run the full sweep.",
    )
    
    # Compute robust default paths relative to project root, not current directory
    project_root = Path(__file__).parent.parent
    default_data = project_root / "data" / "lending_club_train.csv"
    default_output = project_root / "models"
    
    parser.add_argument("--data", default=str(default_data), type=Path)
    parser.add_argument("--output", default=str(default_output), type=Path)
    args = parser.parse_args()

    if args.config == "all":
        print("Running full sweep on all configurations...")
        results: dict[str, dict] = {}
        for cfg_name in CONFIGS:
            result = train(cfg_name, args.data, args.output)
            results[cfg_name] = result
            print(f"[{cfg_name}] metrics: {result['metrics']}")

        best_cfg = max(results, key=lambda k: results[k]["metrics"]["f1_default"])
        best_result = results[best_cfg]
        print("\nSweep finished.")
        print(f"Best config by f1_default: {best_cfg}")
        print(f"Model saved to {best_result['model_path']}")
        print(f"Metadata saved to {best_result['meta_path']}")
    else:
        if args.config not in CONFIGS:
            raise ValueError(
                f"Unknown config '{args.config}'. Available: {list(CONFIGS)} or 'all'."
            )
        result = train(args.config, args.data, args.output)
        print(f"Model saved to {result['model_path']}")
        print(f"Metadata saved to {result['meta_path']}")
        print(f"Metrics (test internal): {result['metrics']}")
        print(
            "\nNext step: once you have chosen your retained config, promote it:\n"
            f"  cp {result['model_path']} {args.output}/pyrenex_risk_v2.joblib\n"
            f"  cp {result['meta_path']} {args.output}/pyrenex_risk_v2.json\n"
            "  python src/evaluate.py --update-meta"
        )


if __name__ == "__main__":
    main()