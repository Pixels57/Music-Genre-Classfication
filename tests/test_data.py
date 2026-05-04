from __future__ import annotations

import pandas as pd
import pytest

from music_genre.data import clean_tracks, read_spotify, validate_required_columns


def sample_config() -> dict:
    return {
        "data": {
            "duration_min_ms": 30000,
            "duration_max_ms": 1200000,
            "class_min_count": 1,
            "min_rows_after_cleaning": 3,
            "min_features_after_cleaning": 7,
        },
        "features": {"numeric": ["energy"], "categorical": ["duration_bucket"]},
        "genre_mapping": {"rock": ["rock"], "pop": ["pop"]},
    }


def spotify_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": ["1", "1", "2", "3"],
            "track_name": ["a", "a duplicate", "b", "c"],
            "artists": ["artist", "artist", "artist", "artist"],
            "album_name": ["album", "album", "album", "album"],
            "popularity": [50, 50, 40, 30],
            "duration_ms": [180000, 180000, 20000, 200000],
            "explicit": [False, False, True, False],
            "danceability": [0.5, 0.5, 0.6, 0.7],
            "energy": [0.8, 0.8, 0.9, 0.3],
            "key": [1, 1, 3, 5],
            "loudness": [-5.0, -5.0, -7.0, -8.0],
            "mode": [1, 1, 0, 1],
            "speechiness": [0.1, 0.1, 0.2, 0.3],
            "acousticness": [0.2, 0.2, 0.1, 0.7],
            "instrumentalness": [0.0, 0.0, 0.0, 0.2],
            "liveness": [0.1, 0.1, 0.2, 0.2],
            "valence": [0.6, 0.6, 0.4, 0.5],
            "tempo": [120, 120, 0, 90],
            "time_signature": [4, 4, 4, 4],
            "track_genre": ["rock", "rock", "rock", "pop"],
            "source_dataset": ["spotify", "spotify", "spotify", "spotify"],
        }
    )


def test_validate_required_columns_fails_on_missing_column() -> None:
    with pytest.raises(ValueError):
        validate_required_columns(pd.DataFrame({"track_id": ["1"]}), ["track_id", "track_genre"], "test")


def test_clean_tracks_removes_duplicates_and_invalid_duration() -> None:
    cleaned = clean_tracks(spotify_frame(), sample_config())
    assert len(cleaned) == 2
    assert cleaned["track_id"].tolist() == ["1", "3"]
    assert cleaned["genre_family"].tolist() == ["rock", "pop"]


def test_read_spotify_sets_source_and_canonical_columns(tmp_path) -> None:
    path = tmp_path / "spotify.csv"
    spotify_frame().drop(columns=["source_dataset"]).to_csv(path, index=False)
    loaded = read_spotify(path)
    assert "source_dataset" in loaded.columns
    assert loaded["source_dataset"].eq("spotify").all()
