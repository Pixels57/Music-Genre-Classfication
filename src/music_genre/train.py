from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from music_genre.config import ensure_dir, ensure_parent, load_config
from music_genre.features import feature_columns, run_feature_pipeline
from music_genre.metrics import classification_metrics
from music_genre.models import build_model_pipeline

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _param_grid(grid: dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combination, strict=False)) for combination in itertools.product(*values)]


def _predict_proba_if_available(model: Any, x_data: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(x_data)
        except Exception:  # noqa: BLE001
            return None
    return None


def _classes_if_available(model: Any) -> np.ndarray | None:
    return getattr(model, "classes_", None)


def _mlflow_tracking_uri(uri: str | Path) -> str:
    text = str(uri)
    if "://" in text or text in {"databricks", "databricks-uc", "uc"}:
        return text
    return Path(text).resolve().as_uri()


def train_models(config_path: str | Path = "configs/project.yaml") -> dict[str, Any]:
    import mlflow

    config = load_config(config_path)
    split_dir = Path(config["paths"]["split_dir"])
    run_feature_pipeline(config_path)

    train = pd.read_csv(split_dir / "train.csv")
    validation = pd.read_csv(split_dir / "validation.csv")
    target = config["data"]["target"]
    features = feature_columns(config)
    x_train = train[features]
    y_train = train[target]
    x_validation = validation[features]
    y_validation = validation[target]

    mlflow.set_tracking_uri(_mlflow_tracking_uri(config["paths"]["mlflow_tracking_uri"]))
    mlflow.set_experiment(config["project"]["experiment_name"])
    model_dir = ensure_dir(config["paths"]["model_dir"])

    results: list[dict[str, Any]] = []
    for model_name in config["models"]["train"]:
        grids = _param_grid(config["models"].get("grids", {}).get(model_name))
        for run_index, params in enumerate(grids, start=1):
            run_name = f"{model_name}_{run_index}"
            pipeline = build_model_pipeline(model_name, config, params)
            LOGGER.info("Training %s with params %s.", model_name, params)
            with mlflow.start_run(run_name=run_name):
                pipeline.fit(x_train, y_train)
                predictions = pipeline.predict(x_validation)
                probabilities = _predict_proba_if_available(pipeline, x_validation)
                metrics = classification_metrics(
                    y_validation,
                    predictions,
                    probabilities,
                    _classes_if_available(pipeline),
                )

                mlflow.log_param("model_name", model_name)
                mlflow.log_params(params)
                mlflow.log_param("features", ",".join(features))
                for metric_name, value in metrics.items():
                    if value == value:
                        mlflow.log_metric(metric_name, value)

                report = classification_report(
                    y_validation,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                )
                labels = sorted(pd.Series(y_validation).astype(str).unique().tolist())
                matrix = confusion_matrix(y_validation, predictions, labels=labels)
                artifacts = {
                    "classification_report.json": report,
                    "confusion_matrix.json": {"labels": labels, "matrix": matrix.tolist()},
                    "feature_columns.json": features,
                }
                for artifact_name, artifact_content in artifacts.items():
                    artifact_path = ensure_parent(Path("reports") / artifact_name)
                    artifact_path.write_text(json.dumps(artifact_content, indent=2), encoding="utf-8")
                    mlflow.log_artifact(str(artifact_path))

                mlflow.sklearn.log_model(pipeline, artifact_path="model")
                model_path = model_dir / f"{run_name}.joblib"
                joblib.dump(pipeline, model_path)
                results.append(
                    {
                        "run_name": run_name,
                        "model_name": model_name,
                        "params": params,
                        "metrics": metrics,
                        "model_path": str(model_path),
                    }
                )

    best = max(results, key=lambda item: item["metrics"]["macro_f1"])
    best_path = Path(config["paths"]["model_dir"]) / "best_model.joblib"
    joblib.dump(joblib.load(best["model_path"]), best_path)
    summary_path = ensure_parent("reports/training_summary.json")
    summary = {"best_run": best, "runs": results}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Best validation run: %s.", best["run_name"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train music genre classifiers.")
    parser.add_argument("--config", default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    train_models(args.config)


if __name__ == "__main__":
    main()
