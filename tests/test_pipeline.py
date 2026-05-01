from __future__ import annotations

import pandas as pd

from music_genre.features import add_engineered_features, feature_columns
from music_genre.models import build_model_pipeline


def test_small_training_pipeline_runs() -> None:
    config = {
        "project": {"random_state": 42},
        "data": {"target": "genre_family"},
        "features": {
            "numeric": [
                "popularity",
                "duration_ms",
                "danceability",
                "energy",
                "key",
                "loudness",
                "mode",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "liveness",
                "valence",
                "tempo",
                "time_signature",
                "energy_acoustic_ratio",
                "dance_valence_score",
                "loudness_normalized",
            ],
            "categorical": [
                "explicit",
                "duration_bucket",
                "speechiness_tier",
                "major_minor_flag",
                "source_dataset",
            ],
        },
    }
    data = pd.DataFrame(
        {
            "popularity": [50, 60, 10, 15],
            "duration_ms": [180000, 200000, 220000, 240000],
            "explicit": [0, 0, 1, 1],
            "danceability": [0.8, 0.75, 0.25, 0.3],
            "energy": [0.9, 0.85, 0.2, 0.25],
            "key": [1, 2, 3, 4],
            "loudness": [-4, -5, -20, -18],
            "mode": [1, 1, 0, 0],
            "speechiness": [0.1, 0.12, 0.05, 0.08],
            "acousticness": [0.1, 0.15, 0.8, 0.75],
            "instrumentalness": [0.0, 0.0, 0.2, 0.3],
            "liveness": [0.1, 0.2, 0.1, 0.2],
            "valence": [0.8, 0.7, 0.3, 0.35],
            "tempo": [120, 125, 80, 85],
            "time_signature": [4, 4, 4, 4],
            "source_dataset": ["spotify", "spotify", "spotify", "spotify"],
            "genre_family": ["pop", "pop", "jazz_blues_classical", "jazz_blues_classical"],
        }
    )
    data = add_engineered_features(data)
    pipeline = build_model_pipeline("dummy", config)
    pipeline.fit(data[feature_columns(config)], data["genre_family"])
    predictions = pipeline.predict(data[feature_columns(config)])
    assert len(predictions) == len(data)
