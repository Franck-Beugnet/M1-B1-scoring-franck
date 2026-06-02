"""Train Pyrenex Crédit risk model — M1-B1.

Usage:
    python src/train.py --config default
    python src/train.py --config balanced

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
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TARGET_MAPPING,
    build_preprocessor,
    load_dataset,
)

CONFIGS: dict[str, dict] = {
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
    # TODO — ajoute ton propre jeu d'hyperparamètres ici
}


def train(config_name: str, data_path: Path, output_dir: Path) -> dict:
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Available: {list(CONFIGS)}")
    params = CONFIGS[config_name]

    X, y = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", RandomForestClassifier(**params)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "f1_macro": f1_score(y_test, pipeline.predict(X_test), average="macro"),
        "roc_auc": roc_auc_score(y_test, y_proba),
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
        "metrics_test_internal": {k: round(v, 4) for k, v in metrics.items()},
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
    parser.add_argument("--config", default="default", choices=list(CONFIGS))
    parser.add_argument("--data", default="data/lending_club_train.csv", type=Path)
    parser.add_argument("--output", default="models/", type=Path)
    args = parser.parse_args()

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