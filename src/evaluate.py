"""Evaluate Pyrenex Crédit risk model on the holdout dataset.

Usage:
    python src/evaluate.py --model models/pyrenex_risk_v2.joblib \\
                           --data data/lending_club_holdout.csv
    python src/evaluate.py --update-meta            # patch the .json with metrics_holdout

The `--update-meta` flag writes `metrics_holdout` into the JSON adjacent to
the model (same stem). This is the **only** moment the holdout score is
persisted into the metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from preprocess import load_dataset


def evaluate(model_path: Path, data_path: Path) -> dict:
    pipeline = joblib.load(model_path)
    X_holdout, y_holdout = load_dataset(data_path)

    y_pred = pipeline.predict(X_holdout)
    y_proba = pipeline.predict_proba(X_holdout)[:, 1]

    return {
        "f1_macro": round(f1_score(y_holdout, y_pred, average="macro"), 4),
        "f1_default": round(f1_score(y_holdout, y_pred, pos_label=1), 4),
        "roc_auc": round(roc_auc_score(y_holdout, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_holdout, y_pred).tolist(),
        "classification_report": classification_report(
            y_holdout, y_pred, target_names=["Remboursé", "Défaut"], output_dict=True
        ),
    }


def update_metadata(model_path: Path, metrics: dict) -> Path:
    """Write `metrics_holdout` into the JSON file adjacent to `model_path`.

    The JSON keeps the same stem as the .joblib (e.g. `pyrenex_risk_v2.json`
    for `pyrenex_risk_v2.joblib`). Raises FileNotFoundError if missing.
    """
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata file at {meta_path}. Did you forget to copy the "
            f".json of your retained config alongside the .joblib?"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    recall_default = metrics["classification_report"]["Défaut"]["recall"]
    meta["metrics_holdout"] = {
        "f1_macro": metrics["f1_macro"],
        "f1_default": metrics["f1_default"],
        "roc_auc": metrics["roc_auc"],
        "recall_default": round(recall_default, 4),
        "confusion_matrix": metrics["confusion_matrix"],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pyrenex risk model on holdout")
    parser.add_argument("--model", default="models/pyrenex_risk_v2.joblib", type=Path)
    parser.add_argument("--data", default="data/lending_club_holdout.csv", type=Path)
    parser.add_argument(
        "--update-meta",
        action="store_true",
        help="Patch metrics_holdout into the JSON next to --model.",
    )
    args = parser.parse_args()

    metrics = evaluate(args.model, args.data)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.update_meta:
        meta_path = update_metadata(args.model, metrics)
        print(f"\nUpdated {meta_path} with metrics_holdout.")


if __name__ == "__main__":
    main()