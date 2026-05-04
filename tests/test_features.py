from __future__ import annotations

import pandas as pd

from music_genre.features import add_engineered_features, build_preprocessor


def config() -> dict:
    return {
        "features": {
            "numeric": ["energy", "acousticness", "energy_acoustic_ratio"],
            "categorical": ["duration_bucket", "speechiness_tier", "major_minor_flag"],
        }
    }


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "energy": [0.8, 0.2],
            "acousticness": [0.2, 0.6],
            "danceability": [0.5, 0.3],
            "valence": [0.7, 0.4],
            "loudness": [-5.0, -20.0],
            "duration_ms": [120000, 400000],
            "speechiness": [0.1, 0.7],
            "mode": [1, 0],
            "source_dataset": ["spotify", "fma"],
        }
    )


def test_add_engineered_features() -> None:
    engineered = add_engineered_features(frame())
    assert "energy_acoustic_ratio" in engineered
    assert engineered.loc[0, "duration_bucket"] == "short"
    assert engineered.loc[1, "speechiness_tier"] == "speech"
    assert engineered.loc[0, "major_minor_flag"] == "major"


def test_preprocessor_output_shape() -> None:
    engineered = add_engineered_features(frame())
    transformer = build_preprocessor(config())
    output = transformer.fit_transform(engineered)
    assert output.shape[0] == 2
    assert output.shape[1] >= 4
