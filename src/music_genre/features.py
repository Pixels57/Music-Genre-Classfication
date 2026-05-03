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


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _minmax(series: pd.Series, default: float = 0.5) -> pd.Series:
    denominator = series.max() - series.min()
    if pd.isna(denominator) or denominator == 0:
        return pd.Series(default, index=series.index, dtype="float64")
    return (series - series.min()) / denominator


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    popularity = _numeric(frame, "popularity")
    duration = _numeric(frame, "duration_ms")
    danceability = _numeric(frame, "danceability")
    energy = _numeric(frame, "energy")
    loudness = _numeric(frame, "loudness")
    speechiness = _numeric(frame, "speechiness")
    acousticness = _numeric(frame, "acousticness")
    instrumentalness = _numeric(frame, "instrumentalness")
    liveness = _numeric(frame, "liveness")
    valence = _numeric(frame, "valence")
    tempo = _numeric(frame, "tempo")

    frame["energy_acoustic_ratio"] = energy / (acousticness + 0.001)
    frame["dance_valence_score"] = danceability * valence
    frame["loudness_normalized"] = _minmax(loudness)
    frame["tempo_normalized"] = _minmax(tempo)
    frame["energy_danceability_score"] = energy * danceability
    frame["acoustic_instrumental_score"] = acousticness * instrumentalness
    frame["speech_energy_score"] = speechiness * energy
    frame["valence_energy_score"] = valence * energy
    frame["liveness_energy_score"] = liveness * energy
    frame["acoustic_energy_balance"] = acousticness - energy
    frame["tempo_energy_score"] = frame["tempo_normalized"] * energy
    frame["popularity_energy_score"] = popularity.fillna(popularity.median()) * energy
    frame["duration_minutes"] = duration / 60000

    frame["duration_bucket"] = pd.cut(
        duration,
        bins=[-np.inf, 150000, 300000, np.inf],
        labels=["short", "mid", "long"],
    ).astype("string")
    frame["speechiness_tier"] = pd.cut(
        speechiness,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["music", "mixed", "speech"],
    ).astype("string")
    frame["major_minor_flag"] = _numeric(frame, "mode").map({1: "major", 0: "minor"}).astype("string")
    frame["tempo_bucket"] = pd.cut(
        tempo,
        bins=[-np.inf, 90, 120, 150, np.inf],
        labels=["slow", "medium", "fast", "very_fast"],
    ).astype("string")
    frame["popularity_bucket"] = pd.cut(
        popularity,
        bins=[-np.inf, 25, 50, 75, np.inf],
        labels=["niche", "emerging", "popular", "mainstream"],
    ).astype("string")
    frame["energy_bucket"] = pd.cut(
        energy,
        bins=[-np.inf, 0.35, 0.7, np.inf],
        labels=["low", "mid", "high"],
    ).astype("string")
    frame["acousticness_bucket"] = pd.cut(
        acousticness,
        bins=[-np.inf, 0.25, 0.65, np.inf],
        labels=["electric", "mixed", "acoustic"],
    ).astype("string")
    frame["instrumentalness_flag"] = np.where(instrumentalness.fillna(0) >= 0.5, "instrumental", "vocal")
    frame["valence_bucket"] = pd.cut(
        valence,
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["dark", "neutral", "bright"],
    ).astype("string")
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
