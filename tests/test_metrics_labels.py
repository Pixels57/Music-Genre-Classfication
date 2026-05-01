from __future__ import annotations

import numpy as np
import pandas as pd

from music_genre.labels import map_genre_family, normalize_label
from music_genre.metrics import classification_metrics, costly_misclassification_rate


def test_normalize_and_map_genre_family() -> None:
    mapping = {"hip_hop_rnb": ["hip-hop", "r-n-b"], "latin_world": ["world-music"]}
    genres = pd.Series(["Hip Hop", "R&B", "world music", "unknown style"])
    mapped = map_genre_family(genres, mapping)
    assert normalize_label("R&B") == "r-n-b"
    assert mapped.tolist() == ["hip_hop_rnb", "hip_hop_rnb", "latin_world", "unknown-style"]


def test_business_metrics_include_costly_and_top_k_values() -> None:
    y_true = ["jazz_blues_classical", "pop", "rock"]
    y_pred = ["electronic", "pop", "rock"]
    probabilities = np.array(
        [
            [0.1, 0.7, 0.2],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )
    classes = ["jazz_blues_classical", "pop", "rock"]
    metrics = classification_metrics(y_true, y_pred, probabilities, classes)
    assert costly_misclassification_rate(y_true, y_pred) == 1 / 3
    assert metrics["accuracy"] == 2 / 3
    assert metrics["high_confidence_accuracy"] == 2 / 3
    assert metrics["top_3_accuracy"] == 1.0
