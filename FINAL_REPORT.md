# Music Genre Classification Final Report

CMPS344 Applied Data Science, Spring 2026

Team:

- Mario Emad Saleh Fouad - 1210158
- Mohamed ElSayed Shaaban - 1210025
- Moaz Mohamed ElSayed - 1210162
- Seif Wael - 1210243

Team number: TBD

Repository link: TBD

Date: May 2026

## Executive Summary

This project builds a supervised machine learning pipeline for automatic music genre classification. The business goal is to support a music platform or catalog management team by automatically suggesting genre-family labels for songs using tabular audio features. The project uses Spotify tracks as the main training source and integrates FMA and GTZAN as external real-world benchmark sources.

The final cleaned dataset contains 100,008 rows from three sources and 37 usable model features after feature engineering. Spotify rows are used for the main train, validation, and test workflow. FMA and GTZAN rows are held out as external evaluation data to measure how well the trained model generalizes to different music collections.

Five classifiers were trained and tracked with MLflow: Dummy baseline, Logistic Regression, Random Forest, Gradient Boosting, and Linear SVM. The best validation model is Random Forest. On the held-out Spotify test split, it achieves 60.28% accuracy, 56.48% macro F1, 59.63% weighted F1, and 85.09% top-3 accuracy. External performance is lower at 24.23% accuracy, which shows dataset shift between Spotify, FMA, and GTZAN.

The system is useful as a first-pass tagging tool. High-confidence predictions are especially valuable: the best model reaches 91.92% high-confidence accuracy on the Spotify test split. Low-confidence predictions should be reviewed manually or enriched with additional features such as lyrics, artist metadata, or audio embeddings.

## 1. Problem Definition And Business Context

### Business Problem

Music platforms contain large catalogs of tracks that need genre labels for search, recommendation, playlist generation, user discovery, and catalog organization. Manual genre labeling is slow, subjective, and difficult to scale. A machine learning model can automate part of this workflow by suggesting genre labels from available audio descriptors.

### Stakeholder

The main stakeholder is a music streaming or digital music catalog team. This includes recommendation engineers, content operations teams, playlist curators, and metadata quality teams.

### Why This Is A Classification Problem

The target variable is categorical. For each song, the model predicts one genre family, such as `rock`, `electronic`, `pop`, or `latin_world`. Because the output is a discrete class rather than a continuous value, the project is a supervised classification problem.

### Decision-Making Impact

Predictions can support:

- automatic first-pass genre tagging for new tracks
- routing low-confidence tracks to human review
- improving search filters and playlist organization
- helping recommendation systems group similar music
- detecting metadata inconsistencies in large catalogs

The model is not intended to replace human curation completely. It is more appropriate as an assistive classification tool.

## 2. Data Acquisition And Integration

### Data Sources

| Source | Local File | Role In Project | Reference |
| --- | --- | --- | --- |
| Spotify Tracks Dataset | `data/raw/spotify_tracks.csv` | Main supervised training, validation, and test source | https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs |
| FMA metadata | `data/raw/fma_tracks.csv` | External real-world benchmark source | https://github.com/mdeff/fma |
| FMA Echo Nest features | `data/raw/fma_echonest.csv` | FMA audio-feature support | https://archive.ics.uci.edu/ml/datasets/FMA |
| GTZAN derived features | `data/raw/gtzan_features.csv` | External benchmark source | https://huggingface.co/datasets/Lehrig/GTZAN-Collection |

Spotify is the primary source because it contains the most complete Spotify-style audio-feature schema. FMA and GTZAN are integrated to satisfy the multiple-source requirement and to test external generalization.

### Acquisition Pipeline Snapshot

The data acquisition and integration logic is implemented in:

```text
src/music_genre/data.py
```

It is run with:

```powershell
make data
```

or:

```powershell
python -m poetry run music-genre-data --config configs/project.yaml
```

The pipeline reads raw files from `data/raw`, standardizes compatible columns, adds a `source_dataset` column, maps raw genres into genre families, cleans invalid values, and writes the integrated dataset to:

```text
data/interim/integrated_tracks.csv
```

### Integration Strategy

The project uses a common canonical schema. Spotify already contains most target columns. FMA and GTZAN have different formats, so the pipeline maps available fields into the shared structure where possible. Columns that do not exist in external sources are left missing and later handled by imputation.

The project does not force FMA and GTZAN into the Spotify training split. Instead:

```text
Spotify -> train, validation, test
FMA + GTZAN -> external evaluation
```

This strategy keeps the main supervised task consistent while still testing real-world generalization across sources.

### Counts After Integration And Cleaning

| Source | Cleaned Rows |
| --- | ---: |
| Spotify | 89,653 |
| FMA | 9,355 |
| GTZAN | 1,000 |
| Total | 100,008 |

The raw input contained 128,129 rows. After validation, cleaning, genre mapping, and rare-class filtering, 100,008 rows remained.

## 3. Data Validation And Documentation

### Validation Pipeline Snapshot

The validation report is generated by the data pipeline and saved to:

```text
reports/validation_report.json
```

The latest report shows:

| Item | Value |
| --- | ---: |
| Raw rows | 128,129 |
| Cleaned rows | 100,008 |
| Raw columns | 21 |
| Cleaned columns | 22 |
| Usable features after engineering | 37 |
| Duplicate track/source pairs | 0 |
| Minimum row requirement met | Yes |
| Minimum feature requirement met | Yes |

### Target Variable

The raw target is:

```text
track_genre
```

The modeling target is:

```text
genre_family
```

Raw genres are mapped into broader genre families to reduce label noise and make the problem more stable. For example, raw labels such as `alt-rock`, `hard-rock`, `punk`, and `metal` are mapped into the broader `rock` family.

### Final Target Distribution

| Genre Family | Rows |
| --- | ---: |
| latin_world | 20,974 |
| electronic | 19,819 |
| rock | 19,255 |
| pop | 10,413 |
| mood_context | 9,179 |
| acoustic_folk_country | 6,262 |
| ambient_chill | 4,930 |
| jazz_blues_classical | 4,914 |
| hip_hop_rnb | 4,262 |

The target distribution is imbalanced. This is expected in real music catalogs because some genres have many more songs than others. The project handles this by reporting macro F1 in addition to accuracy and weighted F1. Macro F1 gives each class equal importance.

### Missing Values

Missing values remain mainly because FMA and GTZAN do not contain every Spotify-style feature. Important examples from the validation report:

| Column | Missing Values |
| --- | ---: |
| popularity | 10,355 |
| duration_ms | 10,355 |
| explicit | 10,355 |
| key | 10,355 |
| loudness | 10,355 |
| mode | 10,355 |
| time_signature | 11,356 |
| danceability | 1,000 |
| energy | 1,000 |
| speechiness | 1,000 |
| acousticness | 1,000 |
| instrumentalness | 1,000 |
| liveness | 1,000 |
| valence | 1,000 |

The project does not drop every row with missing values. Dropping all rows with missing external features would remove most external data. Instead, invalid values are cleaned and missing values are imputed during preprocessing.

### Quality Issues And Removal Justification

The cleaning step handles:

- missing or blank target labels
- duplicate `(source_dataset, track_id)` pairs
- known durations shorter than 30 seconds
- known durations longer than 20 minutes
- zero or negative tempo values
- probability-like audio features outside the 0 to 1 range
- keys outside 0 to 11
- modes outside 0 or 1
- unreasonable time signatures
- unmapped genre labels
- classes below the configured minimum count

These rules remove corrupted or unhelpful rows while preserving external benchmark data that is legitimately missing some Spotify-specific fields.

## 4. Preprocessing, Transformation, And Feature Engineering

### Preprocessing Pipeline Snapshot

Feature engineering and preprocessing are implemented in:

```text
src/music_genre/features.py
```

The training command runs the feature pipeline automatically:

```powershell
make train
```

Processed outputs are written to:

```text
data/processed/model_ready.csv
data/processed/splits/train.csv
data/processed/splits/validation.csv
data/processed/splits/test.csv
data/processed/splits/external.csv
```

### Cleaning Steps With Justification

| Step | Reason |
| --- | --- |
| Remove rows without target labels | Supervised classification requires a known label. |
| Remove duplicate track/source pairs | Prevents duplicated songs from biasing the model. |
| Validate duration range | Removes intros, corrupted tracks, and extremely long recordings. |
| Treat invalid tempo as missing | A tempo of 0 BPM is not meaningful for this task. |
| Validate 0 to 1 audio features | Spotify-style probability features must remain in valid ranges. |
| Validate key, mode, and time signature | Ensures musical metadata is physically meaningful. |
| Map raw genres to genre families | Reduces noisy subgenre labels into a stable taxonomy. |
| Drop very rare mapped classes | Avoids unreliable classes with too few examples. |

### Outlier Detection And Handling

The main outlier handling is rule-based because the audio features have known valid ranges:

- duration must be between 30 seconds and 20 minutes when present
- tempo must be positive
- probability-like descriptors must be between 0 and 1
- key must be 0 to 11
- mode must be 0 or 1
- time signature must be in a reasonable range

Rule-based validation is preferable here because the feature semantics are known. For example, `danceability = 1.5` is not an extreme but valid value; it is invalid by definition.

### Original Features

Important original numeric features include:

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

Important categorical features include:

- `explicit`

### Engineered Features

The project engineers additional features to expose genre-relevant interactions:

| Engineered Feature | Purpose |
| --- | --- |
| `energy_acoustic_ratio` | Separates high-energy/electronic/rock tracks from acoustic tracks. |
| `dance_valence_score` | Captures tracks that are both danceable and positive. |
| `loudness_normalized` | Converts loudness into a normalized scale. |
| `tempo_normalized` | Makes tempo easier for models to compare. |
| `energy_danceability_score` | Captures energetic and danceable tracks. |
| `acoustic_instrumental_score` | Captures acoustic instrumental tracks. |
| `speech_energy_score` | Helps identify speech-heavy energetic tracks. |
| `valence_energy_score` | Captures positive and energetic tracks. |
| `liveness_energy_score` | Captures energetic live-performance signals. |
| `acoustic_energy_balance` | Measures acoustic versus energetic balance. |
| `tempo_energy_score` | Combines speed and energy. |
| `popularity_energy_score` | Combines popularity and energy. |
| `duration_minutes` | Makes duration more interpretable than milliseconds. |
| `duration_bucket` | Groups tracks into short, mid, and long. |
| `speechiness_tier` | Groups tracks into music, mixed, and speech-like. |
| `major_minor_flag` | Converts mode into major/minor categories. |
| `tempo_bucket` | Groups tracks by tempo range. |
| `popularity_bucket` | Groups tracks by popularity level. |
| `energy_bucket` | Groups tracks by energy level. |
| `acousticness_bucket` | Groups tracks by acousticness level. |
| `instrumentalness_flag` | Marks likely instrumental tracks. |
| `valence_bucket` | Groups tracks by mood/positivity. |

These features helped improve model performance because genre is usually represented by combinations of audio properties, not by one feature alone.

### Transformation And Encoding

The project uses scikit-learn pipelines:

- numeric features: median imputation and standard scaling
- categorical features: most-frequent imputation and one-hot encoding

Using a pipeline prevents leakage because preprocessing is fitted on training data and then applied consistently to validation, test, and external data.

### Feature Selection Rationale

The model uses 37 configured features after engineering. Features were selected because they are available in the Spotify-style schema, interpretable, relevant to music style, or derived from relevant audio descriptors. The `source_dataset` column is not used as a model input. It is kept for reporting and splitting only, because including it could let the model learn dataset identity instead of music properties.

### Train, Validation, Test, And External Split

| Split | Rows | Purpose |
| --- | ---: | --- |
| Train | 62,757 | Fit models. |
| Validation | 13,448 | Compare models and select the best run. |
| Test | 13,448 | Final held-out Spotify evaluation. |
| External | 10,355 | FMA and GTZAN generalization evaluation. |

The split uses a fixed random seed of 42 for reproducibility.

### Data Balancing Strategy

No oversampling or undersampling is applied in the current final pipeline. The project keeps the natural class distribution and reports macro F1, weighted F1, and per-class results to account for imbalance. This is appropriate because the business setting is a real catalog where class imbalance is expected.

## 5. Exploratory Data Analysis

The Streamlit dashboard provides the main interactive EDA interface:

```text
src/music_genre/dashboard.py
```

Run it with:

```powershell
make dashboard
```

or run the full pipeline and dashboard with:

```powershell
make serve
```

### Recommended Figures For The PDF Report

The final PDF should include screenshots from the Streamlit dashboard and MLflow. Use the following figure list.

#### Figure 1: Class Distribution

Source: dashboard bar chart.

Interpretation: the dataset is imbalanced. `latin_world`, `electronic`, and `rock` are the largest genre families, while `hip_hop_rnb`, `jazz_blues_classical`, and `ambient_chill` have fewer rows. This justifies reporting macro F1 alongside accuracy.

#### Figure 2: Source Distribution

Source: validation report or dashboard source filter.

Interpretation: Spotify provides most rows, while FMA and GTZAN provide smaller external benchmark sets. This supports the choice to train on Spotify and externally evaluate on FMA/GTZAN.

#### Figure 3: Energy Versus Acousticness

Source: dashboard scatter plot.

Interpretation: genres separate partly by the relationship between energy and acousticness. Electronic and rock tracks often appear in more energetic regions, while acoustic/folk/country and classical-related tracks tend to show stronger acoustic patterns.

#### Figure 4: Danceability By Genre Family

Source: dashboard box plot.

Interpretation: danceability varies by genre family and helps distinguish dance-oriented categories from less dance-oriented ones. However, distributions overlap, which explains why the classification task is not trivial.

#### Figure 5: Model Comparison

Source: MLflow comparison table or `reports/training_summary.json`.

Interpretation: Random Forest performs best on validation accuracy and macro F1. The Dummy baseline is much lower, proving that the trained models learn meaningful patterns.

#### Figure 6: Confusion Matrix For Best Model

Source: `reports/evaluation_metrics.json` or dashboard extension if visualized.

Interpretation: many mistakes occur between musically overlapping classes, such as `latin_world`, `pop`, `rock`, and `electronic`. This supports the limitation that audio features alone cannot perfectly represent human genre labels.

### EDA Insights That Influenced Modeling

- Genre families are imbalanced, so macro F1 is needed.
- Audio features overlap across genres, so a nonlinear model is likely useful.
- Energy and acousticness interact strongly, motivating `energy_acoustic_ratio`.
- Danceability and valence interact, motivating `dance_valence_score`.
- Duration, speechiness, tempo, acousticness, energy, and valence show useful bucketed patterns, motivating categorical bucket features.

## 6. Model Development

### Models Trained

The project trains five classifiers:

| Model | Purpose |
| --- | --- |
| Dummy baseline | Minimum baseline that predicts the most frequent class. |
| Logistic Regression | Simple linear model and interpretable classical baseline. |
| Random Forest | Nonlinear tree ensemble for feature interactions and threshold patterns. |
| Gradient Boosting | Sequential tree ensemble that corrects previous errors. |
| Linear SVM | Linear margin-based classifier useful for high-dimensional encoded data. |

### Hyperparameters

The current hyperparameter grid is configured in:

```text
configs/project.yaml
```

| Model | Hyperparameters Used |
| --- | --- |
| Dummy | Default baseline strategy. |
| Logistic Regression | `C = 1.0` |
| Random Forest | `n_estimators = 100`, `max_depth = None` |
| Gradient Boosting | `n_estimators = 50`, `learning_rate = 0.08`, `max_depth = 3` |
| Linear SVM | `C = 1.0` |

### Validation Strategy

The model selection workflow is:

1. Fit each model on the Spotify train split.
2. Evaluate each model on the Spotify validation split.
3. Select the best model by validation performance.
4. Save the best model to `models/best_model.joblib`.
5. Evaluate the best model on the Spotify test split.
6. Evaluate the same best model on the external FMA/GTZAN split.

### Why Random Forest Was Selected

Random Forest performed best because the task depends on nonlinear interactions. For example, high energy alone does not determine genre, but high energy plus low acousticness plus high danceability can indicate electronic or dance-oriented music. Decision trees naturally learn threshold combinations, and Random Forest averages many trees to reduce instability.

## 7. Experiment Tracking With MLflow

MLflow tracking is stored in:

```text
mlruns/
```

Open the MLflow UI with:

```powershell
python -m poetry run mlflow ui --backend-store-uri mlruns
```

Then open:

```text
http://127.0.0.1:5000
```

Each model run logs:

- model name
- hyperparameters
- accuracy
- macro F1
- weighted F1
- top-3 accuracy when probabilities are available
- high-confidence accuracy when probabilities are available
- costly misclassification rate
- classification report
- confusion matrix
- model artifact
- feature list

Before submitting the final PDF, insert a screenshot of the MLflow experiment comparison interface showing all five runs side by side.

## 8. Testing

### Test Suite

Tests are implemented under:

```text
tests/
```

The test suite covers:

- schema validation behavior
- duplicate removal and invalid value handling
- genre-label mapping
- engineered feature creation
- preprocessing output shape
- metric functions
- small end-to-end pipeline behavior

### Latest Test Results

The latest local test command was:

```powershell
python -m poetry run pytest
```

Result:

```text
8 passed, 2 warnings
Coverage: 32%
```

The warnings come from tiny synthetic test frames used in feature tests and do not affect the production pipeline.

### Latest Lint Results

The latest lint command was:

```powershell
python -m poetry run ruff check src tests
```

Result:

```text
All checks passed.
```

### Coverage Interpretation

Current coverage is 32%. This is enough to show that important data, feature, metric, and pipeline logic is tested, but it is not complete. The lowest-coverage modules are dashboard, training, and evaluation entry points. Future work should add tests for CLI behavior, dashboard data loading, MLflow logging, and evaluation artifact generation.

## 9. Results And Evaluation

### Validation Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 | Costly Misclassification Rate | High-Confidence Accuracy | Top-3 Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0.2313 | 0.0417 | 0.0869 | 0.1692 | 0.2313 | 0.5117 |
| Logistic Regression | 0.4133 | 0.3857 | 0.4283 | 0.0561 | 0.7633 | 0.7075 |
| Random Forest | 0.6065 | 0.5738 | 0.6003 | 0.0558 | 0.9160 | 0.8572 |
| Gradient Boosting | 0.5023 | 0.4330 | 0.4850 | 0.0717 | 0.8624 | 0.7975 |
| Linear SVM | 0.4363 | 0.3818 | 0.4359 | 0.0605 | N/A | N/A |

Random Forest has the best validation accuracy, macro F1, weighted F1, high-confidence accuracy, and top-3 accuracy.

### Best Model Performance Across Splits

| Split | Rows | Accuracy | Macro F1 | Weighted F1 | Costly Misclassification Rate | High-Confidence Accuracy | Top-3 Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 62,757 | 0.9996 | 0.9995 | 0.9996 | 0.0000 | 0.9998 | 1.0000 |
| Validation | 13,448 | 0.6065 | 0.5738 | 0.6003 | 0.0558 | 0.9160 | 0.8572 |
| Test | 13,448 | 0.6028 | 0.5648 | 0.5963 | 0.0568 | 0.9192 | 0.8509 |
| External | 10,355 | 0.2423 | 0.1253 | 0.2594 | 0.1000 | 0.7401 | 0.5652 |

The training score is almost perfect, which means the Random Forest learns the training set very strongly. The validation and test scores are much lower but close to each other, so the held-out Spotify performance estimate is stable at about 60%. The external score is much lower because FMA and GTZAN are collected differently and do not fully match Spotify-style features.

### Test Classification Report Summary

| Genre Family | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| acoustic_folk_country | 0.5737 | 0.4515 | 0.5053 | 793 |
| ambient_chill | 0.6101 | 0.6662 | 0.6370 | 740 |
| electronic | 0.6551 | 0.7143 | 0.6834 | 2,632 |
| hip_hop_rnb | 0.7500 | 0.2459 | 0.3704 | 488 |
| jazz_blues_classical | 0.6963 | 0.5135 | 0.5911 | 594 |
| latin_world | 0.5251 | 0.7223 | 0.6081 | 3,111 |
| mood_context | 0.7201 | 0.4322 | 0.5402 | 1,321 |
| pop | 0.5573 | 0.4749 | 0.5128 | 1,495 |
| rock | 0.6458 | 0.6253 | 0.6354 | 2,274 |

### Error Analysis

The best model performs strongest on `electronic`, `ambient_chill`, `rock`, and `latin_world`. It struggles more with `hip_hop_rnb`, `pop`, and some acoustic or mood/context labels.

Important failure patterns:

- `latin_world` absorbs many false positives from pop, rock, electronic, and acoustic categories.
- `hip_hop_rnb` has high precision but low recall, meaning that when the model predicts it, it is often correct, but it misses many true hip-hop/R&B tracks.
- `pop` is difficult because it overlaps with many other genres in audio space.
- `mood_context` is difficult because labels such as happy, sad, party, and comedy describe context or mood rather than strict genre.
- External performance is weak because FMA and GTZAN have different feature distributions and missing Spotify-specific features.

### Business-Oriented Metrics

Top-3 accuracy is important because a music platform may suggest several possible genre tags to a curator instead of forcing one final answer. The best model reaches 85.09% top-3 accuracy on Spotify test data.

High-confidence accuracy is important because automatic tagging should be used only when the model is confident. The best model reaches 91.92% high-confidence accuracy on Spotify test data. This supports a workflow where high-confidence predictions are accepted automatically and low-confidence predictions are reviewed manually.

Costly misclassification rate measures serious mistakes between distant genre families. The test rate is 5.68%, which indicates that the model makes fewer severe cross-family mistakes than ordinary classification errors.

### Unsuccessful Or Weaker Approaches

Earlier runs had test accuracy around 42%. The main improvements came from better genre-family mapping and expanded feature engineering. Gradient Boosting was previously strongest, but after those changes Random Forest became the best model.

Linear models were weaker because the task depends on nonlinear feature interactions. Gradient Boosting was strong but did not outperform Random Forest under the current conservative hyperparameter settings.

## 10. Code Automation

The project uses a Makefile to automate common commands:

| Command | Purpose |
| --- | --- |
| `make install` | Install dependencies with Poetry. |
| `make data` | Ingest, clean, integrate, and validate data. |
| `make train` | Engineer features, split data, train models, and log MLflow runs. |
| `make evaluate` | Evaluate the best model on test and external splits. |
| `make pipeline` | Run data, train, and evaluate. |
| `make dashboard` | Start the Streamlit dashboard. |
| `make serve` | Run the full pipeline and then start the dashboard. |
| `make test` | Run pytest with coverage. |
| `make lint` | Run Ruff linting. |
| `make clean` | Remove generated outputs. |

The main reproducibility command is:

```powershell
make serve
```

This runs the full pipeline and starts the dashboard.

Configuration is centralized in:

```text
configs/project.yaml
```

The project also includes:

```text
.env.example
```

This file documents where environment-specific settings would go if the project later needed credentials or environment variables.

## 11. Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The CI workflow runs on pushes and pull requests to `main`. It:

1. checks out the repository
2. installs Python 3.11
3. installs Poetry
4. installs project dependencies
5. runs Ruff linting
6. runs pytest

This supports reproducibility and prevents broken code from being merged without running quality checks.

## 12. Limitations And Future Work

### Limitations

- Genre labels are subjective and sometimes inconsistent.
- Some labels describe mood or context rather than strict musical genre.
- Spotify, FMA, and GTZAN have different schemas and feature distributions.
- External performance is weak because of dataset shift.
- Audio features alone cannot capture lyrics, language, artist identity, culture, release era, or listener context.
- The Random Forest has a large train-test gap, so stronger regularization or tuning may improve generalization.
- Current test coverage is useful but not complete.
- The current dashboard is useful for presentation but could include more static export-ready visualizations.

### Future Work

- Tune Random Forest depth, leaf size, and number of estimators to reduce overfitting.
- Try XGBoost or LightGBM for stronger tabular modeling.
- Add lyrics features and language detection.
- Add artist-level and album-level metadata.
- Add audio embeddings from a pretrained music model.
- Improve FMA and GTZAN feature alignment.
- Add calibration for model probabilities.
- Add threshold-based human review workflow.
- Add more tests for CLI entry points, dashboard loading, MLflow logging, and evaluation outputs.

## 13. Team Contributions

Replace or adjust this table before final submission if your team wants more specific wording.

| Team Member | Contribution |
| --- | --- |
| Mario Emad Saleh Fouad | Project setup, pipeline implementation support, dashboard/report preparation, review and discussion preparation. |
| Mohamed ElSayed Shaaban | Data understanding, data validation review, model evaluation review, report review. |
| Moaz Mohamed ElSayed | Feature engineering review, testing support, experiment tracking review, report review. |
| Seif Wael | Dataset preparation, EDA/dashboard review, reproducibility checks, report review. |

## 14. Reproducibility Instructions

From the project folder:

```powershell
cd "D:\CUFE\Spring 2026\Data Science\Project\Music-Genre-Classfication"
```

Install dependencies:

```powershell
python -m poetry install
```

Run the full project:

```powershell
make serve
```

Run only the pipeline:

```powershell
make pipeline
```

Run tests:

```powershell
make test
```

Run linting:

```powershell
make lint
```

Expected important outputs:

```text
data/interim/integrated_tracks.csv
data/processed/model_ready.csv
data/processed/splits/
reports/validation_report.json
reports/training_summary.json
reports/evaluation_metrics.json
models/best_model.joblib
mlruns/
```

## 15. Required Screenshots Before PDF Submission

Before exporting this Markdown report to PDF, add these screenshots:

1. Streamlit dashboard class distribution.
2. Streamlit dashboard energy versus acousticness scatter plot.
3. Streamlit dashboard danceability by genre family box plot.
4. Streamlit dashboard best model evaluation table.
5. Test-set song prediction example from the dashboard.
6. MLflow experiment comparison showing all five model runs.
7. Confusion matrix visualization, if added or exported from `reports/evaluation_metrics.json`.
8. Terminal output or CI screenshot showing tests and lint passing.

## 16. References

1. Priyam Choksi, Spotify Dataset 114k Songs, Kaggle: https://www.kaggle.com/datasets/priyamchoksi/spotify-dataset-114k-songs
2. Defferrard et al., FMA: A Dataset For Music Analysis, GitHub: https://github.com/mdeff/fma
3. UCI Machine Learning Repository, FMA dataset page: https://archive.ics.uci.edu/ml/datasets/FMA
4. GTZAN Collection, Hugging Face: https://huggingface.co/datasets/Lehrig/GTZAN-Collection
5. scikit-learn documentation: https://scikit-learn.org/
6. MLflow documentation: https://mlflow.org/
7. Streamlit documentation: https://streamlit.io/

## 17. Conclusion

The project satisfies the main requirements of the CMPS344 final project: it defines a clear supervised classification problem, uses three real-world data sources, performs validation and cleaning, engineers meaningful features, trains five classifiers, tracks experiments with MLflow, evaluates models using standard and business-oriented metrics, provides tests and automation, includes CI, and exposes results through a Streamlit dashboard.

The best current model is Random Forest. It achieves about 60% accuracy on held-out Spotify test data and about 85% top-3 accuracy, making it useful for assisted music tagging. Its lower external performance on FMA and GTZAN is an important finding: the model generalizes less well when the source distribution changes. This suggests that future improvements should focus on stronger feature alignment, richer metadata, more regularized models, and additional sources of musical information.
