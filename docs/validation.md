# Data Quality and Validation

## 1. Objective

This document describes the data quality checks performed during the ingestion, cleaning, transformation and integration of flight and weather data.

The project uses three layers in Databricks: Bronze, Silver and Gold.

The purpose of validation is to identify missing values, unexpected duplicates, invalid ranges, inconsistent records and possible issues affecting the machine learning models.

## 2. Dataset Overview

| Dataset | Table | Records |
|---|---|---:|
| Raw BTS flights | workspace.bronze_flights.bts_flights_raw | Approximately 27.7 million |
| Clean flights | workspace.silver_flights.flights_clean | 8,655,359 |
| Raw weather forecasts | workspace.bronze_flights.weather_forecasts_raw | 504,000 |
| Clean weather forecasts | workspace.silver_flights.weather_clean | 504,000 |
| Gold operational flights | workspace.gold_flights.flights_features | 8,655,359 |
| Gold flights with weather | workspace.gold_flights.flights_weather_features | 8,653,172 |

The datasets cover different scopes. Raw BTS data includes 2022-2025, while the weather-integrated analysis focuses on February 2024 through December 2025 and 30 selected origin airports.

## 3. Bronze Flight Validation

The following checks were performed on the BTS flight data:

- Reviewed ingestion volume and historical coverage.
- Examined missing departure delay values.
- Checked cancellation records.
- Preserved source metadata for traceability.

### Findings

A total of 453,438 records had missing departure delay values in the raw flight dataset.

The missing departure delay values corresponded to canceled flights.

These canceled flights were excluded from the Silver dataset because the project predicts departure delays for non-canceled flights.

The raw records were preserved in Bronze.

## 4. Silver Flight Validation

Table: `workspace.silver_flights.flights_clean`

Total records: 8,655,359.

### Required Fields

The following fields were checked for missing values:

| Field | Missing values |
|---|---:|
| flight_date | 0 |
| origin | 0 |
| dest | 0 |
| airline | 0 |
| crs_dep_time | 0 |
| dep_delay | 0 |
| delayed_15 | 0 |

### Business Rules

| Validation | Result |
|---|---|
| Canceled flights remaining | 0 |
| Incorrect delayed_15 labels | 0 |
| Missing required fields | 0 |
| Potential duplicate flight groups | 179 |

The target was validated using the rule:

`delayed_15 = 1 if dep_delay >= 15, otherwise 0`

### Potential Duplicates

The quality check identified 179 groups of potentially duplicated flights based on:

- Flight date.
- Origin.
- Destination.
- Airline.
- Scheduled departure time.
- Actual departure delay.

These groups were not automatically deleted.

The selected fields do not establish a guaranteed unique flight identifier. Different flight records may share the same values, so additional verification is necessary before treating them as true duplicates.

## 5. Silver Weather Validation

Table: `workspace.silver_flights.weather_clean`

Total records: 504,000.

### Missing Values

| Weather variable | Missing values |
|---|---:|
| airport | 0 |
| weather_time_utc | 0 |
| temperature_2m | 0 |
| precipitation | 0 |
| wind_speed_10m | 0 |
| relative_humidity_2m | 0 |
| cloud_cover | 0 |
| pressure_msl | 0 |

### Duplicate Checks

No duplicate weather records were found using the airport, UTC forecast hour, source model and forecast lead time as keys.

### Range Validation

The following conditions were checked:

- Precipitation must not be negative.
- Wind speed must not be negative.
- Relative humidity must remain between 0% and 100%.
- Cloud cover must remain between 0% and 100%.
- Mean sea-level pressure must be greater than zero.

### Findings

| Problem | Records |
|---|---:|
| Negative precipitation | 0 |
| Negative wind speed | 0 |
| Humidity outside valid range | 0 |
| Cloud cover outside valid range | 371 |
| Nonpositive pressure | 0 |

The 371 invalid cloud cover observations had only two values:

| Original value | Records |
|---|---:|
| -1 | 120 |
| 101 | 251 |
| Total | 371 |

These records represent approximately 0.074% of the weather dataset.

### Treatment

The values were retained in Bronze and Silver for traceability.

During Gold weather integration, cloud cover values were adjusted to the valid range:

- Values below 0 were set to 0.
- Values above 100 were set to 100.

This operation is known as clipping.

The adjusted values should not be interpreted as corrected observations from the original forecast provider. They are a documented preprocessing decision.

## 6. Temporal Consistency

BTS scheduled departure times are reported in the local time of each origin airport.

Open-Meteo forecast valid times are recorded in UTC.

The Gold pipeline:

1. Identifies the origin airport time zone.
2. Constructs the scheduled local departure timestamp.
3. Converts the scheduled departure time to UTC.
4. Rounds the scheduled UTC departure down to its hour.
5. Joins the flight with the corresponding airport-hour weather forecast.

All Gold flight records had an identified origin airport time zone.

The conversion was reviewed through example output records.

Daylight-saving-time transitions remain a known area requiring additional edge-case testing.

## 7. Gold Integration Validation

Source flight records: 8,655,359.

Records after the initial weather join: 8,655,359.

Flights with matching weather forecasts: 8,653,172.

Flights without matching forecasts: 2,187.

Weather coverage: approximately 99.97%.

The join preserved the original flight count, indicating that it did not introduce extra rows or remove flights during the initial integration.

Only flights with weather information were retained in the shared dataset for the two XGBoost models.

This ensures both models can be evaluated on the same modeling population.

## 8. Modeling Data Validation

The two XGBoost models used the same deterministic sampling rule.

### Chronological Splits

| Dataset | Flights | Delayed flights (%) |
|---|---:|---:|
| Training | 179,783 | 21.56 |
| Validation | 34,571 | 27.98 |
| Testing | 44,490 | 22.45 |
| Total | 258,844 | - |

The split was chronological to avoid random mixing of earlier and later flights.

### Leakage Prevention

The actual delay variable `dep_delay` was excluded from the XGBoost predictors.

The binary target `delayed_15` was used only as the supervised learning label.

The baseline used scheduled flight information.

The weather model used the same flight information plus six previous-day weather forecast features.

The classification thresholds were selected using the validation dataset, not the test dataset.

## 9. Model Evaluation Results

| Metric | Without weather | With weather |
|---|---:|---:|
| Accuracy | 0.7213 | 0.7457 |
| Precision | 0.3818 | 0.4198 |
| Recall | 0.3904 | 0.3477 |
| F1 | 0.3861 | 0.3804 |
| ROC-AUC | 0.6731 | 0.6873 |

Adding forecasted weather improved accuracy, precision and ROC-AUC.

However, recall and F1 decreased.

The weather model therefore did not outperform the baseline on every metric.

## 10. Remaining Validation Work

The following items remain open for future improvement:

- Verify the 179 groups of potentially duplicated flight records using stronger flight identifiers.
- Add automated schema and transformation tests.
- Test time zone conversions around daylight-saving transitions.
- Verify the acquisition and processing workflow from a clean environment.
- Benchmark processing performance and document execution times.
- Evaluate model reliability across airports, airlines and time periods.
- Consider alternative treatments for out-of-range cloud cover values.

## 11. Conclusion

The validation process confirmed that the main Silver and Gold datasets are suitable for the initial exploratory analysis and model comparison.

The project achieved high weather-join coverage and identified data quality problems before training.

The results do not imply that every possible quality issue has been eliminated. Remaining limitations are documented to support reproducibility and further improvement.
