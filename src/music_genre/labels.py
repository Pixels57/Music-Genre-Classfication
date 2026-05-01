from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd


def normalize_label(value: object) -> str:
    text = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", text.replace("&", "n").replace("and", "n"))
    if compact == "rnb":
        return "r-n-b"
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def build_reverse_mapping(mapping: Mapping[str, list[str]]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for family, labels in mapping.items():
        reverse[normalize_label(family)] = family
        for label in labels:
            reverse[normalize_label(label)] = family
    return reverse


def map_genre_family(genres: pd.Series, mapping: Mapping[str, list[str]]) -> pd.Series:
    reverse = build_reverse_mapping(mapping)
    normalized = genres.map(normalize_label)
    return normalized.map(reverse).fillna(normalized)


def broad_family(value: object, mapping: Mapping[str, list[str]]) -> str:
    return build_reverse_mapping(mapping).get(normalize_label(value), normalize_label(value))
