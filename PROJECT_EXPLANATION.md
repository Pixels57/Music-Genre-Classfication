# Music Genre Classification Project Explanation

This document explains the project in detail so every team member can understand what the code does, why it exists, how to run it, and how to explain it during discussion.

## 1. Project Goal

The project is a supervised machine learning system that predicts a song's genre from audio-related features.

The main question is:

> Given numerical audio features for a track, can we automatically classify the track into the correct genre family?

This is a classification problem because the target is a discrete label, not a continuous number. The model predicts categories such as:

- `electronic`
- `rock`
- `latin_world`
- `pop`
- `jazz_blues_classical`
- `hip_hop_rnb`
- `acoustic_folk_country`
- `mood_context`

The business motivation is that music platforms need automatic genre tagging for:

- search
- recommendations
- playlist generation
- catalog organization
- music discovery
- faster tagging of newly uploaded tracks

Manual tagging is slow, inconsistent, and hard to scale. A supervised model can provide a first-pass automated label.

## 2. Course Requirements Covered

The course project document requires several things. This project was implemented to satisfy them as much as possible.

### Data Requirements

The course asks for:

- at least 3,000 rows
- at least 7 features
- real-world data
- a clear classification target
- multiple reliable data sources

The current cleaned dataset has:

```text
100,008 cleaned rows
16 usable model features
3 real-world sources
```

The classification target is:

```text
genre_family
```

### Modeling Requirements

The course asks for at least five classification models, including a baseline.

The project trains:

1. Dummy baseline
2. Logistic Regression
3. Random Forest
4. Gradient Boosting
5. Linear SVM

### Experiment Tracking

The course asks for MLflow tracking.

The project logs every model run to:

```text
mlruns/
```

Each run logs:

- model name
- hyperparameters
- metrics
- feature list
- classification report
- confusion matrix
- trained model artifact

### Testing and Automation

The project includes:

- `pytest` tests
- `ruff` linting
- a `Makefile`
- GitHub Actions CI
- Poetry dependency management

Useful commands:

```powershell
make data
make train
make evaluate
make pipeline
make dashboard
make pipeline-serve
make test
make lint
```

## 3. Data Sources

The project currently uses three sources.

### 3.1 Spotify Tracks Dataset

File:

```text
data/raw/spotify_tracks.csv
```

This is the main dataset. It contains Spotify-style audio features and genre labels.

Important columns include:

- `track_id`
- `track_name`
- `artists`
- `album_name`
- `popularity`
- `duration_ms`
- `explicit`
- `danceability`
- `energy`
- `key`
- `loudness`
- `mode`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `time_signature`
- `track_genre`

Spotify is used for the main machine learning split:

```text
train
validation
test
```

This means the model learns from Spotify rows and is tested on unseen Spotify rows.

### 3.2 FMA Dataset

Files:

```text
data/raw/fma_tracks.csv
data/raw/fma_echonest.csv
```

FMA stands for Free Music Archive. It provides music metadata and audio features.

In this project, FMA is used as an external source. It is not mixed into the training split. Instead, it is kept for external evaluation.

The reason is that FMA has a different structure and comes from a different music collection, so it helps test whether the model generalizes outside Spotify.

### 3.3 GTZAN Dataset

File:

```text
data/raw/gtzan_features.csv
```

GTZAN is a classic music genre recognition dataset. It usually contains genres such as:

- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

In this project, GTZAN is also used as an external source.

### 3.4 Why External Sources Are Separate

The project does not train directly on FMA and GTZAN by default.

Instead:

```text
Spotify -> train, validation, test
FMA + GTZAN -> external evaluation
```

This is intentional.

Spotify has the full set of Spotify-style features. FMA and GTZAN have different available features and different collection biases. Keeping them external makes the evaluation more honest because it measures dataset shift.

## 4. Current Dataset Counts

After running the data pipeline, the current cleaned data is:

```text
spotify: 89,653 rows
fma:      9,355 rows
gtzan:    1,000 rows
total:  100,008 rows
```

The current split is:

```text
train:      62,757 Spotify rows
validation: 13,448 Spotify rows
test:       13,448 Spotify rows
external:   10,355 FMA + GTZAN rows
```

The external split is important because it tells us how the model behaves on data from sources it was not trained on.

## 5. Project Folder Structure

Important folders and files:

```text
Music-Genre-Classfication/
  configs/
    project.yaml

  data/
    raw/
      spotify_tracks.csv
      fma_tracks.csv
      fma_echonest.csv
      gtzan_features.csv

    interim/
      integrated_tracks.csv

    processed/
      model_ready.csv
      splits/
        train.csv
        validation.csv
        test.csv
        external.csv

  models/
    best_model.joblib
    dummy_1.joblib
    logistic_regression_1.joblib
    random_forest_1.joblib
    gradient_boosting_1.joblib
    linear_svm_1.joblib

  reports/
    validation_report.json
    training_summary.json
    evaluation_metrics.json
    classification_report.json
    confusion_matrix.json
    feature_columns.json

  src/
    music_genre/
      data.py
      features.py
      labels.py
      metrics.py
      models.py
      train.py
      evaluate.py
      dashboard.py

  tests/

  Makefile
  pyproject.toml
  poetry.lock
  README.md
```

## 6. Configuration File

The main configuration file is:

```text
configs/project.yaml
```

This file controls:

- raw file paths
- output file paths
- target column
- feature lists
- split sizes
- model list
- hyperparameters
- genre mapping
- MLflow experiment name

The reason we use a config file is reproducibility. Instead of hardcoding paths and model settings in many Python files, they are centralized in one place.

## 7. Data Pipeline

The data pipeline is implemented in:

```text
src/music_genre/data.py
```

Run it with:

```powershell
make data
```

or:

```powershell
python -m poetry run music-genre-data --config configs/project.yaml
```

### 7.1 What The Data Pipeline Does

The data pipeline:

1. Reads Spotify data.
2. Reads FMA metadata and Echo Nest features.
3. Reads GTZAN features.
4. Converts all sources into a common format.
5. Cleans invalid rows.
6. Maps raw genre names into common genre families.
7. Saves the integrated dataset.
8. Saves a validation report.

### 7.2 Data Cleaning

The cleaning stage handles:

- duplicate track/source pairs
- missing target labels
- invalid durations
- invalid tempo values
- invalid feature ranges
- invalid keys, modes, and time signatures
- rare classes below the configured minimum count

Important duration rules:

```text
minimum duration: 30 seconds
maximum duration: 20 minutes
```

These rules remove tracks that are probably intros, skits, corrupted rows, or very long recordings that would distort duration features.

### 7.3 Validation Report

The validation report is saved to:

```text
reports/validation_report.json
```

It includes:

- raw row count
- cleaned row count
- source distribution
- target distribution
- missing values
- duplicate count
- whether minimum row/feature requirements are met

This report is useful for the final PDF report.

## 8. Genre Mapping

The raw Spotify dataset contains many detailed genres. Some examples:

- `alt-rock`
- `deep-house`
- `black-metal`
- `indie-pop`
- `salsa`
- `j-pop`
- `classical`
- `hip-hop`

Some are broad genres, while others are subgenres, countries, moods, or contexts.

The project maps many raw genres into broader families.

Example:

```text
rock family:
  rock
  alt-rock
  alternative
  hard-rock
  punk
  grunge
  metal
  heavy-metal
```

Example:

```text
electronic family:
  electronic
  edm
  house
  deep-house
  techno
  trance
  dubstep
  ambient
```

This helps reduce the complexity of the classification problem and makes the results easier to interpret.

The genre mapping lives in:

```text
configs/project.yaml
```

The code that applies the mapping lives in:

```text
src/music_genre/labels.py
```

## 9. Feature Engineering

Feature engineering is implemented in:

```text
src/music_genre/features.py
```

Run indirectly through:

```powershell
make train
```

or:

```powershell
make pipeline
```

### 9.1 Original Features

Important original numerical features:

- `popularity`
- `duration_ms`
- `danceability`
- `energy`
- `key`
- `loudness`
- `mode`
- `speechiness`
- `acousticness`
- `instrumentalness`
- `liveness`
- `valence`
- `tempo`
- `time_signature`

Important categorical features:

- `explicit`
- `source_dataset`

### 9.2 Engineered Features

The project creates these engineered features:

#### energy_acoustic_ratio

Formula:

```text
energy / (acousticness + 0.001)
```

Purpose:

This separates high-energy electronic or rock tracks from acoustic tracks.

#### dance_valence_score

Formula:

```text
danceability * valence
```

Purpose:

This identifies tracks that are both danceable and positive/happy.

#### loudness_normalized

Formula:

```text
(loudness - min_loudness) / (max_loudness - min_loudness)
```

Purpose:

Loudness is measured in decibels and usually has negative values. Normalization makes it easier for models to use.

#### duration_bucket

Categories:

```text
short: less than 2.5 minutes
mid:   2.5 to 5 minutes
long:  more than 5 minutes
```

Purpose:

Different genres often have different duration patterns.

#### speechiness_tier

Categories:

```text
music
mixed
speech
```

Purpose:

This helps separate spoken-word-heavy genres from instrumental or vocal music.

#### major_minor_flag

Based on:

```text
mode
```

Values:

```text
major
minor
```

Purpose:

Major/minor mode can relate to mood and genre style.

### 9.3 Preprocessing

The project uses scikit-learn preprocessing pipelines.

Numerical features:

- missing values are filled with median
- values are standardized

Categorical features:

- missing values are filled with most frequent value
- categories are one-hot encoded

This is done inside a `ColumnTransformer`.

## 10. Splitting Strategy

The project uses Spotify for the main supervised learning workflow.

Spotify rows are split into:

```text
train
validation
test
```

FMA and GTZAN rows are saved into:

```text
external
```

Current split:

```text
train:      62,757 Spotify rows
validation: 13,448 Spotify rows
test:       13,448 Spotify rows
external:   10,355 FMA + GTZAN rows
```

Why this matters:

- validation is used to compare models
- test is used for final held-out Spotify performance
- external is used to evaluate generalization to different datasets

## 11. Models

Models are defined in:

```text
src/music_genre/models.py
```

Training is implemented in:

```text
src/music_genre/train.py
```

Run training with:

```powershell
make train
```

or the full pipeline:

```powershell
make pipeline
```

### 11.1 Dummy Baseline

The dummy model predicts the most frequent class.

Purpose:

It gives a minimum baseline. Any serious model should outperform it.

### 11.2 Logistic Regression

Logistic Regression is a linear model.

Purpose:

It is simple, explainable, and works well as a classical baseline for classification.

### 11.3 Random Forest

Random Forest combines many decision trees.

Purpose:

It can capture nonlinear relationships and feature interactions.

### 11.4 Gradient Boosting

Gradient Boosting builds trees sequentially, where each tree tries to fix previous errors.

Purpose:

It often performs well on tabular data.

Current best model:

```text
gradient_boosting_1
```

### 11.5 Linear SVM

Linear SVM tries to find separating hyperplanes between classes.

Purpose:

It is useful for high-dimensional feature spaces after one-hot encoding.

## 12. Experiment Tracking With MLflow

MLflow stores model runs in:

```text
mlruns/
```

To open the MLflow UI:

```powershell
python -m poetry run mlflow ui --backend-store-uri mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

For the final report, you should include a screenshot showing the five model runs side by side.

Each run logs:

- model name
- hyperparameters
- accuracy
- macro F1
- weighted F1
- top-3 accuracy
- high-confidence accuracy
- costly misclassification rate
- classification report
- confusion matrix
- model artifact

## 13. Metrics

Metrics are implemented in:

```text
src/music_genre/metrics.py
```

### 13.1 Accuracy

Accuracy measures the percentage of correct predictions.

Formula:

```text
correct predictions / total predictions
```

Accuracy is easy to understand but can be misleading when classes are imbalanced.

### 13.2 Macro F1

Macro F1 calculates F1 for each class and averages all classes equally.

This is important because it gives rare classes equal importance.

### 13.3 Weighted F1

Weighted F1 averages F1 scores while considering class sizes.

This is useful when classes have very different numbers of examples.

### 13.4 Top-3 Accuracy

Top-3 accuracy checks whether the correct class appears in the model's top three predicted probabilities.

This is useful for music recommendation systems because a platform might show multiple possible tags or suggestions.

### 13.5 High-Confidence Accuracy

High-confidence accuracy measures accuracy only when the model is confident enough.

In this project, confidence is based on the highest predicted class probability.

Business interpretation:

If the model is highly confident, we can trust it more for automatic tagging. If not, the track can be flagged for manual review.

### 13.6 Costly Misclassification Rate

This metric counts serious mistakes between very different genre families.

Example:

Misclassifying classical music as electronic is more damaging than confusing two similar rock subgenres.

Business interpretation:

This metric helps estimate how often the model makes mistakes that could badly affect playlist quality or user experience.

## 14. Current Results

The current best model is:

```text
gradient_boosting_1
```

### 14.1 Validation Performance

From `reports/training_summary.json`:

```text
accuracy:                      0.4284
macro_f1:                      0.3190
weighted_f1:                   0.4046
costly_misclassification_rate: 0.0473
high_confidence_accuracy:      0.8617
top_3_accuracy:                0.6950
```

### 14.2 Test Performance

From `reports/evaluation_metrics.json`:

```text
test rows: 13,448

accuracy:                      0.4268
macro_f1:                      0.3267
weighted_f1:                   0.4037
costly_misclassification_rate: 0.0474
high_confidence_accuracy:      0.8713
top_3_accuracy:                0.6902
```

### 14.3 External Source Performance

External data is FMA + GTZAN.

```text
external rows: 10,355

accuracy:                      0.2168
macro_f1:                      0.0408
weighted_f1:                   0.2225
costly_misclassification_rate: 0.0773
high_confidence_accuracy:      0.6829
top_3_accuracy:                0.3553
```

### 14.4 How To Interpret These Results

The Spotify test result is moderate. The model performs much better than the dummy baseline, but it is not perfect.

This is expected because:

- music genres overlap heavily
- many Spotify labels are subjective
- some labels are moods or contexts, not true genres
- audio features alone cannot capture lyrics, culture, artist identity, or production context
- many genres are very similar in feature space

The external result is lower. This is also expected because FMA and GTZAN are different datasets with different feature distributions. This is called dataset shift.

This is not necessarily a failure. It is a useful limitation:

> A model trained on Spotify-style tabular audio features generalizes less well to external datasets with different feature distributions.

## 15. Dashboard

The dashboard is implemented in:

```text
src/music_genre/dashboard.py
```

Run it with:

```powershell
make dashboard
```

or:

```powershell
python -m poetry run streamlit run src/music_genre/dashboard.py
```

Open:

```text
http://localhost:8501
```

The dashboard shows:

- total tracks
- number of genre families
- number of sources
- class distribution
- feature relationships
- model evaluation metrics

For the final report, take screenshots from the dashboard.

## 16. Complete Run Sequence

From the project folder:

```powershell
cd "D:\CUFE\Spring 2026\Data Science\Project\Music-Genre-Classfication"
```

Install dependencies:

```powershell
python -m poetry install
```

Run everything:

```powershell
make pipeline-serve
```

If `make` is not available, run manually:

```powershell
python -m poetry run music-genre-data --config configs/project.yaml
python -m poetry run music-genre-train --config configs/project.yaml
python -m poetry run music-genre-evaluate --config configs/project.yaml
python -m poetry run streamlit run src/music_genre/dashboard.py
```

Run tests:

```powershell
make test
```

Run linting:

```powershell
make lint
```

## 17. Output Files To Mention In The Report

### validation_report.json

Path:

```text
reports/validation_report.json
```

Use it for:

- data validation section
- source distribution
- missing values
- class distribution
- row counts

### training_summary.json

Path:

```text
reports/training_summary.json
```

Use it for:

- model comparison
- best model selection
- validation metrics

### evaluation_metrics.json

Path:

```text
reports/evaluation_metrics.json
```

Use it for:

- final test metrics
- external evaluation metrics
- classification report
- confusion matrix

### MLflow

Path:

```text
mlruns/
```

Use it for:

- experiment tracking screenshot
- run comparison
- model artifacts

## 18. What To Say In The Final Report

### Problem Definition

We are solving automatic music genre classification for a streaming platform. The stakeholder is a recommendation/content engineering team. The model supports automated tagging, playlist creation, and catalog organization.

### Data Acquisition

We use three real-world sources:

1. Spotify Tracks Dataset
2. FMA metadata and Echo Nest audio features
3. GTZAN feature dataset

Spotify is used as the primary training dataset. FMA and GTZAN are integrated as external evaluation sources.

### Data Cleaning

We remove duplicates, invalid durations, invalid tempos, invalid feature ranges, and rare classes below the configured threshold. We map raw labels into genre families.

### Feature Engineering

We create interaction and transformed features such as energy/acoustic ratio, dance-valence score, loudness normalization, duration buckets, speechiness tiers, and major/minor flag.

### Modeling

We train five classifiers and track them with MLflow. Gradient Boosting currently performs best on the validation split.

### Evaluation

The model achieves moderate Spotify test performance and lower external performance. The lower external score shows dataset shift across music sources.

### Business Interpretation

The model is useful as a first-pass genre tagging tool, especially when it is highly confident. Low-confidence tracks should be sent to manual review or enriched with more data such as lyrics, artist metadata, or audio embeddings.

### Limitations

Important limitations:

- genre labels can be subjective
- some labels are moods or contexts, not strict genres
- Spotify, FMA, and GTZAN have different feature distributions
- audio features alone are not enough for perfect genre prediction
- external generalization is weak
- some rare classes have low support

### Future Work

Possible improvements:

- better genre taxonomy
- stronger hyperparameter tuning
- more advanced models such as XGBoost or LightGBM
- lyrics features
- artist metadata
- audio embeddings
- better GTZAN feature alignment
- manual review workflow for low-confidence predictions

## 19. What Each Team Member Should Understand

Every team member should be able to explain:

1. What the classification target is.
2. Why this is a classification problem.
3. What the three data sources are.
4. Why Spotify is the main training source.
5. Why FMA and GTZAN are external sources.
6. What cleaning steps are applied.
7. What engineered features were added.
8. What five models were trained.
9. What MLflow is used for.
10. Which model performed best.
11. What accuracy, F1, top-3 accuracy, and high-confidence accuracy mean.
12. Why external-source performance is lower.
13. What the main limitations are.
14. How to run the project.

## 20. Common Discussion Questions And Answers

### Q: What is the target variable?

The raw target is `track_genre`. The model uses `genre_family`, which maps raw genre labels into broader categories.

### Q: Why not use only the original Spotify genres?

The original labels are very detailed and sometimes inconsistent. Mapping them into broader families makes the classification problem more stable and interpretable.

### Q: Why use FMA and GTZAN?

The course requires multiple reliable data sources. FMA and GTZAN provide external music datasets that help test whether the model generalizes beyond Spotify.

### Q: Why is external accuracy lower?

Because FMA and GTZAN come from different distributions. They do not perfectly match Spotify features and labels. This dataset shift makes prediction harder.

### Q: What is the best model?

The current best validation model is Gradient Boosting.

### Q: Is 42% accuracy bad?

Not necessarily. There are many genre families, genres overlap, and labels are subjective. The model is much better than the dummy baseline. Top-3 accuracy is around 69%, which is useful for recommendation/tag suggestion workflows.

### Q: What is the business use?

The model can automatically suggest genre tags. High-confidence predictions can be accepted automatically, while low-confidence predictions can be reviewed manually.

### Q: What is MLflow?

MLflow tracks machine learning experiments. It stores model parameters, metrics, artifacts, and trained models so runs can be compared.

### Q: What is the dashboard for?

The dashboard presents dataset summaries, visualizations, and model metrics in a stakeholder-friendly way.

## 21. Submission Checklist

Code side:

- Poetry project exists.
- `poetry.lock` exists.
- Data pipeline works.
- Feature pipeline works.
- Five models train.
- MLflow logs runs.
- Evaluation runs.
- Dashboard runs.
- Tests pass.
- Lint passes.
- Makefile automation exists.
- GitHub Actions CI exists.

Report side still needed:

- final PDF report
- source citations
- screenshots of MLflow
- screenshots of dashboard
- EDA visualizations and explanations
- model comparison table
- error analysis
- limitations and future work
- team contribution table

## 22. Short Summary

This project predicts music genre families from audio features. Spotify is used as the main supervised training dataset. FMA and GTZAN are integrated as external sources. The pipeline cleans and validates data, engineers features, trains five classifiers, tracks experiments with MLflow, evaluates the best model, and presents results in a Streamlit dashboard. The best current model is Gradient Boosting, with moderate Spotify test performance and weaker external performance due to dataset shift.
