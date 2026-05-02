PYTHON ?= python
POETRY := $(PYTHON) -m poetry
CONFIG ?= configs/project.yaml
STREAMLIT_PORT ?= 8501

.PHONY: install data validate train evaluate pipeline dashboard serve pipeline-serve test lint clean

install:
	$(POETRY) install

data:
	$(POETRY) run music-genre-data --config $(CONFIG)

validate:
	$(POETRY) run music-genre-validate --config $(CONFIG)

train:
	$(POETRY) run music-genre-train --config $(CONFIG)

evaluate:
	$(POETRY) run music-genre-evaluate --config $(CONFIG)

pipeline: data train evaluate

dashboard:
	$(POETRY) run streamlit run src/music_genre/dashboard.py

serve: dashboard

pipeline-serve: pipeline dashboard

test:
	$(POETRY) run pytest

lint:
	$(POETRY) run ruff check src tests

clean:
	powershell -NoProfile -Command "Remove-Item -Recurse -Force data/interim,data/processed,reports/*.json,models/*.joblib -ErrorAction SilentlyContinue"
