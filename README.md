# Flight Delay Prediction Using Big Data

**Predicting U.S. flight departure delays using historical flight data and weather forecasts.**

## 1. Project Overview

This project investigates whether scheduled flight characteristics and historical weather forecasts can help predict significant flight departure delays.

A flight is classified as delayed when its actual departure delay is **15 minutes or more**.

We developed a Big Data pipeline using Databricks, PySpark and Delta Lake, followed by exploratory analysis and a comparison of two XGBoost classification models.

The project also includes a statistical analysis of significant delay durations.

## 2. Research Question

Can historical scheduled flight characteristics and previous-day weather forecasts help predict whether a U.S. flight will depart at least 15 minutes late?

## 3. Project Objectives

- Acquire and process historical U.S. flight records.
- Integrate historical hourly weather forecasts.
- Build Bronze, Silver and Gold data layers.
- Validate data quality and temporal consistency.
- Analyze flight delay patterns.
- Compare XGBoost with and without weather features.
- Investigate statistical distributions of significant delays.
- Document the methodology, decisions and limitations.

## 4. Team

| GitHub User | Main Responsibilities |
|---|---|
| [GUEVARA21](https://github.com/GUEVARA21) | Flight ingestion, operational features, seasonality EDA and baseline XGBoost |
| [diegogargo](https://github.com/diegogargo) | Weather ingestion, weather features, airport and weather EDA, and weather XGBoost |
| [Skraeder](https://github.com/Skraeder) | Data validation, delay distributions, time series and statistical analysis |

All three members collaborated through Git branches, issues, commits and pull requests.

## 5. Data Sources

### Bureau of Transportation Statistics (BTS)

Historical flight information from the U.S. Department of Transportation.

Original period: **2022-2025**.

Approximate raw volume: **27.7 million flight records**.

Relevant variables include scheduled departure time, origin, destination, airline, cancellation status and actual departure delay.

Source: https://www.transtats.bts.gov/

### Open-Meteo

Historical forecasts from the Open-Meteo Previous Runs API using the GFS Global model.

Period used: **February 2024 - December 2025**.

Meteorological variables:

- Temperature
- Precipitation
- Wind speed
- Relative humidity
- Cloud cover
- Sea-level pressure

The project uses `_previous_day1` forecast variables, representing weather predictions generated approximately one day before the forecast valid hour.

Source: https://open-meteo.com/

### Geographic Scope

The weather-integrated analysis focuses on **30 selected origin airports**.

## 6. Big Data Architecture

The project follows a Medallion Architecture:

```text
BTS Flights                 Open-Meteo Forecasts
     |                               |
     v                               v
 Bronze Flights                Bronze Weather
     |                               |
     v                               v
 Silver Flights                Silver Weather
     |                               |
     +---------------+---------------+
                     |
                     v
               Gold Features
                     |
          +----------+----------+
          |                     |
          v                     v
    XGBoost Baseline      XGBoost + Weather
          |                     |
          +----------+----------+
                     |
                     v
              Model Comparison

Gold Data
    |
    v
EDA and Delay Time Series
    |
    v
Statistical Distribution Analysis
```

### Bronze

Preserves source records and ingestion metadata.

### Silver

Cleans records, standardizes data types, validates targets and prepares weather timestamps.

### Gold

Builds analytical features, integrates flights and weather forecasts, and produces modeling and statistical datasets.

See [Data Architecture](docs/architecture.md) for the technical details.

## 7. Dataset Summary

| Dataset | Records |
|---|---:|
| BTS Bronze | Approximately 27.7 million |
| Silver cleaned flights | 8,655,359 |
| Weather airport-hour forecasts | 504,000 |
| Gold flights with weather | 8,653,172 |
| XGBoost modeling sample | 258,844 |

Weather forecasts were successfully matched to approximately **99.97%** of the cleaned flights.

## 8. Exploratory Data Analysis

The team analyzed:

- Departure delay rates by month.
- Delay rates by day of the week.
- Delay patterns by scheduled departure hour.
- Differences across origin airports and airlines.
- Associations between forecasted precipitation and delays.
- Daily average delays and changes in variability.
- Distribution of departure delay durations.

### Weather Finding

Flights without forecasted rain had a delay rate of **21.52%**.

Flights with forecasted precipitation of at least 10 mm had a delay rate of **54.67%**.

This is an observed association, not proof that rainfall alone causes delays.

## 9. Machine Learning

### Prediction Target

`delayed_15 = 1` when departure delay is at least 15 minutes.

Otherwise, `delayed_15 = 0`.

The actual departure delay is not used as a predictor.

### Models

**Baseline XGBoost**

Uses scheduled operational information, including airports, airline, month, weekday and departure hour.

**Weather XGBoost**

Uses the same features plus six previous-day weather forecasts.

### Dataset Split

| Split | Period | Flights |
|---|---|---:|
| Training | February 2024 - May 2025 | 179,783 |
| Validation | June - August 2025 | 34,571 |
| Testing | September - December 2025 | 44,490 |

Both models used the same reproducible sample and chronological split.

Classification thresholds were selected using validation data only.

### Test Results

| Metric | Without Weather | With Weather |
|---|---:|---:|
| Accuracy | 72.13% | 74.57% |
| Precision | 38.18% | 41.98% |
| Recall | 39.04% | 34.77% |
| F1 | 0.3861 | 0.3804 |
| ROC-AUC | 0.6731 | 0.6873 |

Adding weather forecasts improved accuracy, precision and ROC-AUC.

However, recall and F1 decreased.

Therefore, neither model was superior on every metric.

## 10. Statistical Delay Analysis

A complementary analysis examined **156,413 flights** with departure delays of at least 15 minutes.

We compared:

- Exponential distribution.
- Gamma distribution.
- Lognormal distribution.

The **Gamma distribution** achieved the lowest AIC and BIC among the three evaluated candidates.

This analysis describes delay durations; it is not a replacement for the binary classification models.

GARCH was considered as a possible future extension but was not implemented.

## 11. Technologies

- Python
- PySpark
- Databricks
- Delta Lake
- pandas
- Matplotlib
- NumPy
- SciPy
- scikit-learn
- XGBoost
- Git and GitHub

## 12. Repository Organization

```text
flight-delay-bigdata/
|
|-- data/
|   |-- raw/
|   |-- interim/
|   |-- processed/
|   `-- sample/
|
|-- docs/
|   |-- project_scope.md
|   |-- data_dictionary.md
|   |-- architecture.md
|   |-- decisions.md
|   |-- validation.md
|   `-- final_report.md
|
|-- src/
|   |-- ingestion/
|   |-- processing/
|   |-- validation/
|   |-- eda/
|   `-- modeling/
|
|-- README.md
`-- requirements.txt
```

This structure summarizes the main project locations. Some additional files and folders may also be present in the repository.

## 13. Reproducing the Project

### Requirements

The main processing workflow uses Databricks with access to PySpark and Delta Lake.

Machine learning and statistical analysis require Python packages such as XGBoost, scikit-learn, pandas, NumPy, SciPy and Matplotlib.

### Recommended Execution Order

1. Configure the Databricks workspace and required permissions.
2. Run the flight and weather ingestion notebooks in `src/ingestion/`.
3. Run the cleaning notebooks in `src/processing/silver/`.
4. Run the Gold feature preparation notebooks in `src/processing/gold/`.
5. Run the validation and EDA notebooks.
6. Run the modeling notebooks in `src/modeling/`.
7. Review the metrics and results in the Gold tables.

The repository contains the notebooks, but the current workflow is not yet packaged as a fully automated command-line pipeline.

A clean environment must recreate or obtain access to the required Delta tables.

### Data Availability

Large flight and weather datasets are stored outside GitHub.

The initial exploratory sample contains 120,000 flight records, selected as 5,000 flights per month for 2024 and 2025.

A reproducible sample of **120,000 BTS flight records** is included in this repository:

`data/sample/muestra_historica_bts.parquet`

The sample contains 5,000 flights per month from January 2024 through December 2025 (24 months).

The Parquet file was verified to contain 120,000 records and is approximately 5.81 MB.

This sample was used for the initial exploratory analysis. It is separate from the 258,844-flight modeling sample selected from Gold in Databricks.

The complete historical datasets remain stored outside GitHub.

To inspect the sample locally:

```python
import pandas as pd

df = pd.read_parquet("data/sample/muestra_historica_bts.parquet")
print(df.shape)
```

## 14. Documentation

| Document | Description |
|---|---|
| [Project Scope](docs/project_scope.md) | Problem, questions, objectives and limitations |
| [Data Dictionary](docs/data_dictionary.md) | Variables, types, units and known missing values |
| [Architecture](docs/architecture.md) | Data flow and processing layers |
| [Technical Decisions](docs/decisions.md) | Reasons behind methodological choices |
| [Validation](docs/validation.md) | Quality controls and findings |
| [Final Report](docs/final_report.md) | EDA, models, metrics, interpretations and conclusions |

## 15. Limitations

- The weather study includes 30 origin airports.
- Forecasts are historical predictions, not observed weather.
- Canceled flights are excluded.
- Modeling uses a reproducible sample rather than every processed flight.
- Potential duplicate records require further verification.
- Some cloud-cover values required clipping.
- Daylight-saving-time transitions need additional testing.
- Real-time operational data and delay propagation are not included.
- The current pipeline still requires additional automation, tests and performance benchmarks.
- The results show statistical associations, not causal effects.

## 16. Project Status

**Completed**

- Bronze ingestion.
- Silver cleaning and validation.
- Gold feature engineering.
- Exploratory analysis.
- Baseline XGBoost.
- Weather XGBoost.
- Statistical distribution comparison.

**Remaining improvements**

- Publish and document a small reproducible data sample.
- Complete dependency verification and automated testing.
- Benchmark processing performance.
- Improve end-to-end execution instructions.
- Add individual reflections and presentation materials.

## 17. References

- Bureau of Transportation Statistics: https://www.transtats.bts.gov/
- Open-Meteo: https://open-meteo.com/
- Databricks Documentation: https://docs.databricks.com/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- scikit-learn Documentation: https://scikit-learn.org/

