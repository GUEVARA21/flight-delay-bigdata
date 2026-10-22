# Flight Delay Prediction & Propagation

Big Data project focused on analyzing and predicting flight delays in U.S. domestic aviation.

## Project objectives

- Analyze flight delay patterns using historical BTS data.
- Predict whether a flight will be delayed by 15 minutes or more.
- Compare models with and without weather information.
- Study delay distributions, seasonality, and operational factors.
- Explore delay propagation between consecutive flights.

## Data sources

- Bureau of Transportation Statistics (BTS): U.S. domestic flight records.
- Historical weather data: to be integrated and validated.

## Data architecture

We use Databricks and Delta Lake with a Medallion Architecture:

- **Bronze:** Raw data ingestion.
- **Silver:** Data cleaning, normalization, and validation.
- **Gold:** Analytical datasets and engineered features.

## Feature engineering

Three initial engineered features:

1. `is_peak_delay_season`
2. `origin_hour_delay_rate_28d`
3. `airline_delay_rate_28d`

Historical features must use only information available before prediction time.

## Modeling

Initial approaches:

- XGBoost baseline without weather.
- XGBoost with historical weather information.
- Statistical analysis of positive delays using probability distributions.

Prediction target: departure delay >= 15 minutes.

Prediction horizon: 3 hours before scheduled departure.

## Repository structure

- `src/ingestion/`: Data ingestion.
- `src/processing/`: Bronze, Silver, and Gold transformations.
- `src/feature_engineering/`: Feature creation.
- `src/modeling/`: Model training and evaluation.
- `src/validation/`: Data quality checks.
- `notebooks/`: EDA and modeling notebooks.
- `tests/`: Automated tests.
- `docs/`: Technical documentation.
- `reports/`: Results and deliverables.
- `data/sample/`: Small reproducible data samples.

## Collaboration

All three team members contribute through individual GitHub accounts, feature branches, Pull Requests, and code reviews.

## Project status

In development.
