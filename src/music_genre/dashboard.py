from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from music_genre.config import load_config
from music_genre.features import add_engineered_features


def load_dashboard_data(config_path: str | Path) -> tuple[dict, pd.DataFrame, dict]:
    config = load_config(config_path)
    data_path = Path(config["paths"]["processed_data"])
    if not data_path.exists():
        data_path = Path(config["paths"]["integrated_data"])
    data = pd.read_csv(data_path) if data_path.exists() else pd.DataFrame()
    if not data.empty and "energy_acoustic_ratio" not in data:
        data = add_engineered_features(data)
    metrics_path = Path(config["paths"]["metrics_path"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return config, data, metrics


def render_dashboard(config_path: str | Path = "configs/project.yaml") -> None:
    import plotly.express as px
    import streamlit as st

    config, data, metrics = load_dashboard_data(config_path)
    st.set_page_config(page_title="Music Genre Classification", layout="wide")
    st.title("Music Genre Classification")

    if data.empty:
        st.warning("No processed data found. Run `make data` and `make train` first.")
        return

    target = config["data"]["target"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Tracks", f"{len(data):,}")
    col2.metric("Genre Families", data[target].nunique())
    col3.metric("Sources", data["source_dataset"].nunique())

    source_filter = st.multiselect(
        "Source",
        sorted(data["source_dataset"].dropna().unique()),
        default=sorted(data["source_dataset"].dropna().unique()),
    )
    filtered = data[data["source_dataset"].isin(source_filter)]

    left, right = st.columns(2)
    with left:
        counts = filtered[target].value_counts().reset_index()
        counts.columns = [target, "tracks"]
        st.plotly_chart(px.bar(counts, x=target, y="tracks", title="Class Distribution"), use_container_width=True)
    with right:
        st.plotly_chart(
            px.scatter(
                filtered,
                x="energy",
                y="acousticness",
                color=target,
                hover_data=["track_name", "artists"],
                title="Energy vs Acousticness",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        px.box(filtered, x=target, y="danceability", color=target, title="Danceability by Genre Family"),
        use_container_width=True,
    )

    if metrics:
        st.subheader("Best Model Evaluation")
        metric_rows = []
        for split_name, payload in metrics.items():
            for metric_name, value in payload.get("metrics", {}).items():
                metric_rows.append({"split": split_name, "metric": metric_name, "value": value})
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Streamlit dashboard.")
    parser.add_argument("--config", default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_dashboard(args.config)


if __name__ == "__main__":
    main()
