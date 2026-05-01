from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from music_genre.config import ensure_parent, load_config
from music_genre.features import feature_columns, run_feature_pipeline
from music_genre.metrics import classification_metrics

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _predict_proba_if_available(model: Any, x_data: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(x_data)
        except Exception:  # noqa: BLE001
            return None
    return None


def _classes_if_available(model: Any) -> np.ndarray | None:
    return getattr(model, "classes_", None)


def evaluate_split(model: Any, data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    target = config["data"]["target"]
    features = feature_columns(config)
    if data.empty:
        return {"rows": 0, "metrics": {}}
    predictions = model.predict(data[features])
    probabilities = _predict_proba_if_available(model, data[features])
    labels = sorted(pd.Series(data[target]).astype(str).unique().tolist())
    return {
        "rows": int(len(data)),
        "metrics": classification_metrics(
            data[target],
            predictions,
            probabilities,
            _classes_if_available(model),
        ),
        "classification_report": classification_report(
            data[target],
            predictions,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(data[target], predictions, labels=labels).tolist(),
        },
    }


def evaluate_model(config_path: str | Path = "configs/project.yaml") -> dict[str, Any]:
    config = load_config(config_path)
    split_dir = Path(config["paths"]["split_dir"])
    if not split_dir.exists():
        run_feature_pipeline(config_path)

    model_path = Path(config["paths"]["model_dir"]) / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}. Run `make train` first.")
    model = joblib.load(model_path)

    results: dict[str, Any] = {}
    for split_name in ["test", "external"]:
        split_path = split_dir / f"{split_name}.csv"
        data = pd.read_csv(split_path) if split_path.exists() else pd.DataFrame()
        results[split_name] = evaluate_split(model, data, config)

    output_path = ensure_parent(config["paths"]["metrics_path"])
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Wrote evaluation metrics to %s.", output_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best music genre model.")
    parser.add_argument("--config", default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    evaluate_model(args.config)


if __name__ == "__main__":
    main()
