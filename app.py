"""Streamlit Community Cloud entry point (set Main file to `app.py` in Cloud settings)."""

from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from music_genre.dashboard import render_dashboard  # noqa: E402

render_dashboard()
