from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from music_genre.config import ensure_parent, load_config
from music_genre.labels import map_genre_family
from music_genre.schema import (
    CANONICAL_COLUMNS,
    NUMERIC_COLUMNS,
    PROBABILITY_COLUMNS,
    REQUIRED_SPOTIFY_COLUMNS,
)

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def read_spotify(path: str | Path, source_name: str = "spotify") -> pd.DataFrame:
    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(
            "Spotify dataset is required. Place the CSV at "
            f"{raw_path} or update paths.spotify_raw in configs/project.yaml."
        )
    data = pd.read_csv(raw_path)
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    validate_required_columns(data, REQUIRED_SPOTIFY_COLUMNS, dataset_name="spotify")
    data = data.copy()
    data["source_dataset"] = source_name
    return canonicalize_columns(data)


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [
            "_".join(str(part) for part in col if str(part) != "nan").strip("_")
            for col in frame.columns
        ]
    return frame


def read_fma(tracks_path: str | Path, echonest_path: str | Path) -> pd.DataFrame:
    tracks_file = Path(tracks_path)
    echonest_file = Path(echonest_path)
    if not tracks_file.exists() or not echonest_file.exists():
        LOGGER.warning("Skipping FMA integration; expected %s and %s.", tracks_file, echonest_file)
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    try:
        tracks = pd.read_csv(tracks_file, header=[0, 1], index_col=0)
        tracks = _flatten_columns(tracks).reset_index(names="track_id")
    except ValueError:
        tracks = pd.read_csv(tracks_file)

    try:
        echonest = pd.read_csv(echonest_file, header=[0, 1, 2], index_col=0)
        echonest = _flatten_columns(echonest).reset_index(names="track_id")
    except ValueError:
        echonest = pd.read_csv(echonest_file)

    genre_column = _first_existing(tracks, ["track_genre_top", "track_genre", "genre_top", "genre"])
    if genre_column is None:
        LOGGER.warning("Skipping FMA; no genre column found in %s.", tracks_file)
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    audio_map = {
        "acousticness": ["echonest_audio_features_acousticness", "audio_features_acousticness", "acousticness"],
        "danceability": ["echonest_audio_features_danceability", "audio_features_danceability", "danceability"],
        "energy": ["echonest_audio_features_energy", "audio_features_energy", "energy"],
        "instrumentalness": [
            "echonest_audio_features_instrumentalness",
            "audio_features_instrumentalness",
            "instrumentalness",
        ],
        "liveness": ["echonest_audio_features_liveness", "audio_features_liveness", "liveness"],
        "speechiness": ["echonest_audio_features_speechiness", "audio_features_speechiness", "speechiness"],
        "tempo": ["echonest_audio_features_tempo", "audio_features_tempo", "tempo"],
        "valence": ["echonest_audio_features_valence", "audio_features_valence", "valence"],
    }
    selected = pd.DataFrame({"track_id": echonest["track_id"] if "track_id" in echonest else echonest.index})
    for target, candidates in audio_map.items():
        source = _first_existing(echonest, candidates)
        selected[target] = echonest[source] if source else np.nan

    metadata_cols = ["track_id", genre_column]
    for optional in ["track_title", "title", "artist_name", "album_title"]:
        if optional in tracks.columns:
            metadata_cols.append(optional)
    merged = selected.merge(tracks[metadata_cols], on="track_id", how="left")
    merged["track_genre"] = merged[genre_column]
    merged["track_name"] = merged.get("track_title", merged.get("title", ""))
    merged["artists"] = merged.get("artist_name", "")
    merged["album_name"] = merged.get("album_title", "")
    merged["source_dataset"] = "fma"
    return canonicalize_columns(merged)


def read_gtzan(path: str | Path) -> pd.DataFrame:
    gtzan_file = Path(path)
    if not gtzan_file.exists():
        LOGGER.warning("Skipping GTZAN integration; expected %s.", gtzan_file)
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    data = pd.read_csv(gtzan_file)
    genre_column = _first_existing(data, ["track_genre", "genre", "label", "class"])
    if genre_column is None:
        LOGGER.warning("Skipping GTZAN; no genre/label column found in %s.", gtzan_file)
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    data = data.copy()
    data["track_genre"] = data[genre_column]
    data["source_dataset"] = "gtzan"
    if "track_id" not in data:
        id_source = data["filename"] if "filename" in data else pd.Series(range(len(data)))
        data["track_id"] = "gtzan_" + id_source.astype(str)
    data["track_name"] = data.get("filename", data["track_id"])
    data["artists"] = data.get("artist", "unknown")
    data["album_name"] = data.get("album", "GTZAN")
    return canonicalize_columns(data)


def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def canonicalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for column in CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[CANONICAL_COLUMNS]
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["explicit"] = frame["explicit"].map(_normalize_bool).astype("float")
    for column in ["track_id", "track_name", "artists", "album_name", "track_genre", "source_dataset"]:
        frame[column] = frame[column].astype("string")
    return frame


def _normalize_bool(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"true", "1", "yes", "y"} else 0.0
    return float(bool(value))


def validate_required_columns(data: pd.DataFrame, required: list[str], dataset_name: str) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def clean_tracks(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    settings = config["data"]
    frame = data.copy()
    before = len(frame)
    frame = frame.dropna(subset=["track_genre"])
    frame = frame[frame["track_genre"].astype(str).str.strip().ne("")]
    frame = frame.drop_duplicates(subset=["source_dataset", "track_id"], keep="first")

    min_duration = settings["duration_min_ms"]
    max_duration = settings["duration_max_ms"]
    known_duration = frame["duration_ms"].notna()
    frame = frame[(~known_duration) | frame["duration_ms"].between(min_duration, max_duration)]

    for column in PROBABILITY_COLUMNS:
        frame.loc[~frame[column].between(0, 1) & frame[column].notna(), column] = np.nan

    frame.loc[frame["tempo"].le(0), "tempo"] = np.nan
    frame.loc[~frame["key"].between(0, 11) & frame["key"].notna(), "key"] = np.nan
    frame.loc[~frame["mode"].isin([0, 1]) & frame["mode"].notna(), "mode"] = np.nan
    frame.loc[~frame["time_signature"].between(3, 7) & frame["time_signature"].notna(), "time_signature"] = np.nan

    frame["genre_family"] = map_genre_family(frame["track_genre"], config["genre_mapping"])
    counts = frame["genre_family"].value_counts()
    keep_classes = counts[counts >= settings["class_min_count"]].index
    frame = frame[frame["genre_family"].isin(keep_classes)].copy()
    LOGGER.info("Cleaned tracks from %s to %s rows.", before, len(frame))
    return frame.reset_index(drop=True)


def build_validation_report(raw: pd.DataFrame, cleaned: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    from music_genre.features import add_engineered_features

    feature_columns = config["features"]["numeric"] + config["features"]["categorical"]
    model_ready_preview = add_engineered_features(cleaned)
    present_features = [column for column in feature_columns if column in model_ready_preview.columns]
    report = {
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "raw_columns": int(raw.shape[1]),
        "cleaned_columns": int(cleaned.shape[1]),
        "usable_feature_count": int(len(present_features)),
        "missing_values": cleaned.isna().sum().astype(int).to_dict(),
        "source_distribution": cleaned["source_dataset"].value_counts().astype(int).to_dict(),
        "target_distribution": cleaned["genre_family"].value_counts().astype(int).to_dict(),
        "duplicate_track_source_pairs": int(cleaned.duplicated(["source_dataset", "track_id"]).sum()),
    }
    report["meets_minimum_rows"] = report["cleaned_rows"] >= config["data"]["min_rows_after_cleaning"]
    report["meets_minimum_features"] = (
        report["usable_feature_count"] >= config["data"]["min_features_after_cleaning"]
    )
    return report


def integrate_datasets(config: dict[str, Any]) -> pd.DataFrame:
    paths = config["paths"]
    spotify = read_spotify(paths["spotify_raw"], config["data"]["spotify_source_name"])
    fma = read_fma(paths["fma_tracks_raw"], paths["fma_echonest_raw"])
    gtzan = read_gtzan(paths["gtzan_raw"])
    datasets = [dataset for dataset in [spotify, fma, gtzan] if not dataset.empty]
    integrated = pd.concat(datasets, ignore_index=True)
    return integrated


def run_data_pipeline(config_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = load_config(config_path)
    raw = integrate_datasets(config)
    cleaned = clean_tracks(raw, config)
    report = build_validation_report(raw, cleaned, config)

    integrated_path = ensure_parent(config["paths"]["integrated_data"])
    report_path = ensure_parent(config["paths"]["validation_report"])
    cleaned.to_csv(integrated_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote integrated data to %s.", integrated_path)
    LOGGER.info("Wrote validation report to %s.", report_path)
    return cleaned, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and validate integrated music genre data.")
    parser.add_argument("--config", default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    run_data_pipeline(args.config)


if __name__ == "__main__":
    main()
