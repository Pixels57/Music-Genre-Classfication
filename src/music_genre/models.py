from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from music_genre.features import build_preprocessor


def estimator_registry(random_state: int) -> dict[str, Any]:
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            n_jobs=None,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=random_state),
    }


def build_model_pipeline(name: str, config: dict[str, Any], params: dict[str, Any] | None = None) -> Pipeline:
    registry = estimator_registry(config["project"]["random_state"])
    if name not in registry:
        raise KeyError(f"Unknown model '{name}'. Available models: {sorted(registry)}")
    estimator = registry[name]
    if params:
        estimator.set_params(**params)
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("classifier", estimator),
        ]
    )
