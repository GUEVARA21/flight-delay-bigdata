# Data Architecture

## 1. Overview

Flight Delay Prediction is a Big Data project developed in Databricks using PySpark, Delta Lake and a Medallion Architecture.

The goal is to predict whether a flight will have a departure delay of at least 15 minutes and evaluate the contribution of historical weather forecasts.

The architecture has three main layers:

- Bronze: raw data acquisition and storage.
- Silver: cleaning, standardization and validation.
- Gold: analytical features, statistics and datasets for modeling.

GitHub stores the source code and documentation. Large datasets remain in Databricks.

## 2. Architecture Diagram

```text
Bureau of Transportation Statistics (BTS)
           |
           v
    BTS Data Ingestion
           |
           v
  Bronze: bts_flights_raw
           |
           v
  Silver: flights_clean
           |
           v
  Gold: flights_features
           |                  Open-Meteo Previous-Day Forecasts
           |                                |
           |                                v
           |                      Weather API Ingestion
           |                                |
           |                                v
           |                   Bronze: weather_forecasts_raw
           |                                |
           |                                v
           |                    Silver: weather_clean
           |                                |
           +---------------+----------------+
                           |
                           v
              Gold: flights_weather_features
                           |
                   +-------+-------+
                   |               |
                   v               v
             XGBoost         XGBoost
            without          with
             weather         weather
                   |               |
                   +-------+-------+
                           |
                           v
                 Model Comparison


Gold: flights_features
           |
           v
Gold: delay_distribution
Gold: delay_daily_timeseries
Gold: delay_hourly_summary
           |
           v
Exploratory and Statistical Analysis
           |
           v
Exponential / Gamma / Lognormal
```

## 3. Data Sources

### BTS Flight Data

Source: Bureau of Transportation Statistics, United States Department of Transportation.

Historical period ingested: January 2022 to December 2025.

The Bronze flight dataset contains approximately 27.7 million records.

The source includes dates, origin and destination airports, airlines, scheduled departure times, cancellations and actual departure delays.

### Open-Meteo Weather Data

Source: Open-Meteo Previous Runs API.

Forecasting model: GFS Global.

Period: February 2024 to December 2025.

Coverage: 30 selected origin airports.

The weather dataset contains 504,000 airport-hour forecast records.

The six meteorological features are:

- Temperature at 2 meters.
- Precipitation.
- Wind speed at 10 meters.
- Relative humidity.
- Cloud cover.
- Mean sea-level pressure.

The `_previous_day1` variables represent forecasts generated approximately 24 hours before the corresponding weather valid time.

These are historical forecasts, not weather measurements observed after flight departure.

## 4. Bronze Layer

Schema: `workspace.bronze_flights`

Main tables:

| Table | Purpose |
|---|---|
| bts_flights_raw | Preserve original BTS flight records |
| weather_forecasts_raw | Store historical hourly weather forecasts |
| airport_coordinates | Airport locations and time zones |
| weather_airports_top30 | Selected airports for weather integration |

Bronze includes ingestion metadata, allowing the source periods and files to be traced.

Original records are preserved without applying the final model transformations.

## 5. Silver Layer

Schema: `workspace.silver_flights`

### flights_clean

Processing includes:

- Standardizing column names and data types.
- Excluding canceled flights.
- Removing records without a valid delay target.
- Selecting the study period and origin airports.
- Creating the binary target `delayed_15`.
- Extracting calendar and scheduled departure features.

Output: 8,655,359 flights.

### weather_clean

Processing includes:

- Converting weather valid times into UTC timestamps.
- Standardizing meteorological variable names.
- Checking missing values.
- Removing duplicate airport-hour forecast keys.
- Validating meteorological ranges.

Output: 504,000 weather records.

Quality checks identified 371 out-of-range cloud cover values and 179 groups of potentially duplicated flight records.

The potentially duplicated flight groups were retained pending verification.

## 6. Gold Layer

Schema: `workspace.gold_flights`

### flights_features

This table contains operational features derived from Silver flights.

The scheduled departure is converted from airport local time into UTC using the airport time zone.

`weather_hour_utc` represents the scheduled departure timestamp rounded down to the hour.

Additional indicators include:

- is_weekend
- is_summer
- is_peak_hour

Output: 8,655,359 flights.

### flights_weather_features

Flights are joined to historical weather forecasts using:

1. Origin airport.
2. Scheduled departure hour in UTC.

The original join preserved the total flight count.

Weather forecasts were matched to 8,653,172 flights.

Coverage: approximately 99.97%.

The 2,187 unmatched flights were excluded from the shared modeling dataset.

Cloud cover values below zero or above 100 were clipped to the valid range during Gold integration.

### Statistical Gold Tables

| Table | Purpose |
|---|---|
| delay_distribution | Distribution across six departure delay categories |
| delay_daily_timeseries | Daily delay averages, dispersion and delay rates |
| delay_hourly_summary | Delay behavior by scheduled departure hour |
| baseline_model_metrics | Evaluation metrics for XGBoost without weather |
| weather_model_metrics | Evaluation metrics for XGBoost with weather |

## 7. Machine Learning

### Prediction Target

The classification target is:

`delayed_15 = 1` when `dep_delay >= 15`.

Otherwise, `delayed_15 = 0`.

Actual departure delay is not included in model predictors.

### Sampling

Both XGBoost models use the same deterministic sampling rule based on flight date, origin, destination, airline and departure hour.

The modeling sample contains 258,844 flights.

### Temporal Split

| Dataset | Period |
|---|---|
| Training | February 2024 - May 2025 |
| Validation | June 2025 - August 2025 |
| Testing | September 2025 - December 2025 |

The test period is later than both training and validation.

### Models

Baseline XGBoost uses scheduled operational and calendar information.

Weather XGBoost uses the same operational information plus six historical forecast variables.

Each model selects its classification threshold using validation data.

Both models are evaluated on the same test period using accuracy, precision, recall, F1 and ROC-AUC.

## 8. Statistical Analysis

A complementary analysis studies flights with departure delays of at least 15 minutes.

A sample of 156,413 delayed flights was analyzed.

Exponential, gamma and lognormal probability distributions were fitted and compared using AIC and BIC.

Gamma had the lowest AIC and BIC among the evaluated distributions.

This analysis describes delay duration rather than predicting a binary delay outcome.

## 9. Technologies

| Technology | Role |
|---|---|
| Python | Data processing, analysis and modeling |
| PySpark | Distributed processing of large datasets |
| Databricks | Execution and data platform |
| Delta Lake | Storage for Bronze, Silver and Gold tables |
| Open-Meteo API | Historical weather forecast acquisition |
| pandas | Local analysis of sampled and aggregated data |
| Matplotlib | Data visualization |
| XGBoost | Binary delay classification |
| scikit-learn | Encoding and model evaluation |
| SciPy | Statistical distribution fitting |
| Git and GitHub | Version control and collaboration |

## 10. Reproducibility and Limitations

The project uses notebooks stored in GitHub, while the large datasets and Delta tables are stored in Databricks.

To reproduce the workflow, the evaluator needs access to an appropriately configured Databricks environment or must recreate the required tables using the ingestion and processing notebooks.

Important limitations include:

- Only 30 origin airports are included in the weather-integrated analysis.
- Historical forecasts are used instead of observed weather.
- Some flights could not be matched to a forecast.
- Potential duplicate flight groups require additional verification.
- The current XGBoost models use scheduled features and previous-day forecasts, not continuously updated operational information.
- Local-to-UTC timestamp conversions around daylight-saving transitions require particular care.
- A fully automated command-line pipeline and independent execution tests are still needed for complete reproducibility.
