from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from music_genre.config import ensure_dir, ensure_parent, load_config


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["energy_acoustic_ratio"] = frame["energy"] / (frame["acousticness"] + 0.001)
    frame["dance_valence_score"] = frame["danceability"] * frame["valence"]

    loudness = pd.to_numeric(frame["loudness"], errors="coerce")
    denominator = loudness.max() - loudness.min()
    frame["loudness_normalized"] = (
        (loudness - loudness.min()) / denominator if pd.notna(denominator) and denominator != 0 else 0.5
    )

    frame["duration_bucket"] = pd.cut(
        frame["duration_ms"],
        bins=[-np.inf, 150000, 300000, np.inf],
        labels=["short", "mid", "long"],
    ).astype("string")
    frame["speechiness_tier"] = pd.cut(
        frame["speechiness"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["music", "mixed", "speech"],
    ).astype("string")
    frame["major_minor_flag"] = frame["mode"].map({1: "major", 0: "minor"}).astype("string")
    return frame


def build_preprocessor(config: dict[str, Any]) -> ColumnTransformer:
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
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
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def feature_columns(config: dict[str, Any]) -> list[str]:
    return config["features"]["numeric"] + config["features"]["categorical"]


def split_spotify_data(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    source_column = config["data"]["source_column"]
    spotify_name = config["data"]["spotify_source_name"]
    target = config["data"]["target"]
    spotify = data[data[source_column] == spotify_name].copy()
    if spotify.empty:
        raise ValueError("No Spotify rows are available for main train/validation/test split.")

    test_size = config["data"]["test_size"]
    validation_size = config["data"]["validation_size"]
    random_state = config["project"]["random_state"]

    stratify = spotify[target] if spotify[target].value_counts().min() >= 2 else None
    train_valid, test = train_test_split(
        spotify,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    relative_validation = validation_size / (1 - test_size)
    stratify_train_valid = (
        train_valid[target] if train_valid[target].value_counts().min() >= 2 else None
    )
    train, validation = train_test_split(
        train_valid,
        test_size=relative_validation,
        random_state=random_state,
        stratify=stratify_train_valid,
    )
    external = data[data[source_column] != spotify_name].copy()
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "external": external.reset_index(drop=True),
    }


def run_feature_pipeline(config_path: str | Path = "configs/project.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    source = Path(config["paths"]["integrated_data"])
    if not source.exists():
        raise FileNotFoundError(f"Integrated data not found: {source}. Run `make data` first.")
    data = pd.read_csv(source)
    data = add_engineered_features(data)
    output_path = ensure_parent(config["paths"]["processed_data"])
    data.to_csv(output_path, index=False)

    split_dir = ensure_dir(config["paths"]["split_dir"])
    for name, split in split_spotify_data(data, config).items():
        split.to_csv(split_dir / f"{name}.csv", index=False)
    return data
