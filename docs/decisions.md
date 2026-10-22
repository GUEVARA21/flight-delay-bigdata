# Technical Decisions

## 1. Purpose

This document records the main technical decisions made during the Flight Delay Prediction project.

Each decision includes its justification, alternatives considered and consequences.

The objective is to explain how the team designed the data pipeline, selected the modeling approach and addressed practical limitations.

## Decision 1. Use Databricks and PySpark

### Decision

Use Databricks as the main processing environment and PySpark for large-scale transformations.

### Reason

The original BTS dataset contains approximately 27.7 million records.

PySpark supports distributed processing, while Databricks provides an integrated environment for executing notebooks, managing tables and organizing the pipeline.

### Alternatives considered

- pandas on a local computer.
- Google Colab.
- Local processing with Python.

### Consequences

Large datasets can be processed without loading all records into local memory.

However, the workflow depends on Databricks access, compute availability and permissions between team members.

## Decision 2. Adopt a Medallion Architecture

### Decision

Organize the pipeline into Bronze, Silver and Gold layers using Delta tables.

### Reason

Separating raw, cleaned and analytical datasets makes the workflow easier to understand, validate and reproduce.

Bronze preserves source data.

Silver applies cleaning and standardization.

Gold prepares features and aggregated datasets for analysis and modeling.

### Alternatives considered

- Process raw files directly in each notebook.
- Maintain a single transformed dataset.
- Use only local CSV files.

### Consequences

The architecture improves traceability and prevents repeated cleaning.

It also requires managing multiple tables and their dependencies.

## Decision 3. Use BTS as the Flight Data Source

### Decision

Use historical flight data from the Bureau of Transportation Statistics.

### Reason

BTS provides flight-level operational information, including scheduled departure times and actual departure delays.

These variables allow the team to construct a measurable binary prediction target.

### Alternatives considered

- Commercial aviation APIs.
- Smaller flight datasets from public repositories.
- Simulated flight data.

### Consequences

The project uses a large public historical dataset.

However, reporting conventions, cancellations and missing values require careful processing.

## Decision 4. Use Open-Meteo Historical Forecasts

### Decision

Use the Open-Meteo Previous Runs API with the GFS Global model and `_previous_day1` variables.

### Reason

The project aims to predict delays using information available before departure.

Historical forecast values are more appropriate for this objective than observed weather recorded after the flight.

The `_previous_day1` variables represent forecasts generated approximately 24 hours before the corresponding meteorological valid time.

### Alternatives considered

- Historical observed weather.
- Weather forecasts generated closer to departure.
- Training without meteorological information.

### Consequences

The weather model can be compared with a baseline without weather.

The analysis is limited by historical forecast availability, temporal coverage and the selected weather model.

The forecast valid hour is matched to the scheduled departure hour in UTC.

## Decision 5. Focus on 30 Origin Airports

### Decision

Limit the initial weather integration to the 30 selected origin airports.

### Reason

Collecting historical hourly forecasts for every BTS airport would require substantially more API requests, storage and processing time.

The selected airports represent a substantial share of flight activity during the study period.

### Alternatives considered

- Include all available airports.
- Use only a small number of major hubs.
- Select airports randomly.

### Consequences

The project remains manageable within the available computing resources.

However, findings from the weather-integrated dataset should not automatically be generalized to airports outside the selected group.

## Decision 6. Use February 2024 to December 2025

### Decision

Use February 2024 through December 2025 for the weather-integrated modeling dataset.

### Reason

The selected period was supported by the historical weather forecast coverage used during ingestion.

Earlier BTS records remain available in Bronze, but are not included in the initial weather-model comparison.

### Alternatives considered

- Use all BTS records from 2022 to 2025.
- Use only 2025.
- Use a shorter recent period.

### Consequences

Both XGBoost models can be evaluated on the same historical period.

The model comparison does not cover all years present in the original BTS dataset.

## Decision 7. Convert Local Departure Times to UTC

### Decision

Convert scheduled airport departure times into UTC before matching flight and weather records.

### Reason

BTS scheduled departure times use local airport time.

Open-Meteo forecast valid times are represented in UTC.

Matching the raw local clock hours directly could associate a flight with the wrong weather forecast.

### Alternatives considered

- Join using local timestamps without conversion.
- Apply a fixed UTC offset for all airports.
- Join using only the flight date.

### Consequences

The integration respects different airport time zones.

Daylight-saving-time transitions require additional edge-case checks.

## Decision 8. Treat Out-of-Range Cloud Cover in Gold

### Decision

Clip weather cloud-cover values to the valid interval of 0% to 100% when preparing Gold.

### Reason

Quality validation identified:

- 120 records with a cloud-cover value of -1.
- 251 records with a cloud-cover value of 101.

These values fall outside the standard percentage range.

### Alternatives considered

- Remove complete weather records.
- Replace invalid cloud-cover values with null.
- Impute missing values using a statistical method.
- Preserve the original out-of-range values as model inputs.

### Consequences

The other meteorological variables and flight matches are preserved.

The adjusted cloud-cover values should not be interpreted as exact original provider forecasts.

The unadjusted data remains available in earlier layers.

## Decision 9. Use a Deterministic Modeling Sample

### Decision

Select the modeling sample using a deterministic hash-based rule.

### Reason

Training XGBoost on the entire weather-integrated dataset could be expensive for the available environment.

A reproducible selection allows both models to use the same flights.

### Alternatives considered

- Train on all 8.65 million matched flights.
- Use a nonreproducible random sample.
- Use the original Colab sample of 120,000 flights.

### Consequences

The initial experiment is computationally manageable and directly comparable.

The selected modeling sample contains 258,844 flights.

Sampling may reduce representation of less common operational patterns.

## Decision 10. Use a Chronological Train/Validation/Test Split

### Decision

Divide the data by flight date rather than randomly mixing records across the study period.

### Reason

The project predicts future flight outcomes based on historical examples.

Using earlier records for training and later records for evaluation better reflects this prediction setting.

### Alternatives considered

- Random train/test split.
- Cross-validation with randomly distributed flights.
- A single training and test period without validation.

### Consequences

The evaluation better reflects changes over time.

Class proportions differ between training, validation and test periods, which may affect model performance.

## Decision 11. Compare XGBoost With and Without Weather

### Decision

Train two XGBoost classifiers using the same flights, temporal split and initial hyperparameters.

### Reason

The comparison isolates the practical contribution of six historical weather forecast variables under a common evaluation design.

The baseline uses operational and calendar features.

The second model adds forecasted weather information.

### Alternatives considered

- Train only a weather-based model.
- Use logistic regression.
- Use a random forest classifier.
- Use different datasets for each model.

### Consequences

The two models can be compared using the same evaluation metrics.

The comparison does not establish causality and does not prove that XGBoost is the optimal model family.

## Decision 12. Select Classification Thresholds Using Validation

### Decision

Choose the classification threshold for each XGBoost model by maximizing F1 on the validation set.

### Reason

The target classes are imbalanced, and a default threshold of 0.50 did not identify enough delayed flights in the baseline.

Validation provides a separate dataset for threshold selection.

### Alternatives considered

- Always use the default 0.50 threshold.
- Select thresholds using the test dataset.
- Optimize accuracy instead of F1.

### Consequences

The baseline used a threshold of 0.225.

The weather model used a threshold of 0.25.

The test set was used only for final evaluation after threshold selection.

The choice of F1 reflects a balance between precision and recall, but other operational objectives could require a different threshold.

## Decision 13. Evaluate Multiple Classification Metrics

### Decision

Report accuracy, precision, recall, F1 and ROC-AUC.

### Reason

A single metric can hide important weaknesses, especially when most flights are not delayed.

Recall indicates the proportion of actual delayed flights detected.

Precision indicates the reliability of predicted delay alerts.

F1 balances precision and recall.

ROC-AUC evaluates discrimination across classification thresholds.

### Alternatives considered

- Report accuracy only.
- Report F1 only.
- Evaluate solely through the confusion matrix.

### Consequences

The comparison reveals trade-offs instead of declaring a model better based on a single score.

The weather model improved accuracy, precision and ROC-AUC, while recall and F1 decreased.

## Decision 14. Compare Probability Distributions for Significant Delays

### Decision

Fit exponential, gamma and lognormal distributions to the excess delay duration above the 15-minute threshold.

### Reason

The XGBoost models predict whether a flight will be delayed.

The statistical analysis provides a complementary description of how long significant delays tend to last.

### Alternatives considered

- Describe the data using only summary statistics.
- Fit a single exponential distribution.
- Analyze time-varying volatility using GARCH.

### Consequences

Gamma achieved the lowest AIC and BIC among the three evaluated distributions.

This conclusion is limited to the selected sample and candidate distributions.

GARCH was considered conceptually but was not implemented as part of the completed statistical comparison.

## Decision 15. Keep Large Data Outside GitHub

### Decision

Store large datasets in Databricks and use GitHub for code, documentation and small reproducible samples.

### Reason

The original flight records and weather-integrated datasets are too large for practical version control in a regular source repository.

### Alternatives considered

- Commit all raw CSV and Parquet files directly to GitHub.
- Store processed data only on a personal computer.
- Avoid retaining data references.

### Consequences

The repository remains manageable.

Reproducing the complete workflow requires access to the external sources and an appropriate processing environment.

The small evaluation sample and its access instructions should be documented separately.

## 16. Future Improvements

The following improvements remain relevant:

- Validate daylight-saving-time edge cases.
- Investigate potential duplicate flight records using stronger identifiers.
- Evaluate additional airports and forecast horizons.
- Add rolling operational features available before departure.
- Evaluate more model configurations.
- Add automated transformation tests and pipeline execution scripts.
- Benchmark processing time and memory requirements.
- Investigate volatility models such as GARCH if the time-series diagnostics justify them.

These are future improvements, not features claimed as implemented in the current project.
