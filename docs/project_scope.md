# Project Scope

## 1. Project Title

Flight Delay Prediction Using Big Data and Historical Weather Forecasts

## 2. Problem Context

Flight delays affect passengers, airlines and airport operations. Although delays cannot always be prevented, identifying flights with a higher probability of departing late could support better planning and decision-making.

Flight punctuality is influenced by several factors, including departure schedules, airport operations, airline characteristics, weather conditions and disruptions affecting the aviation network.

This project investigates whether historical flight information and weather forecasts can help identify flights likely to experience significant departure delays.

The project also addresses a data engineering challenge: integrating millions of flight records with historical hourly weather forecasts while preserving data quality and temporal consistency.

## 3. Main Research Question

Can historical flight characteristics and previous-day weather forecasts help predict whether a U.S. flight will depart at least 15 minutes late?

## 4. Secondary Research Questions

1. How do departure delay rates vary across airports, airlines, months, weekdays and scheduled departure hours?

2. Is forecasted precipitation associated with a higher frequency of departure delays?

3. Does including historical weather forecast variables improve XGBoost classification performance compared with a model using only scheduled flight characteristics?

4. What probability distribution best describes the duration of departure delays of at least 15 minutes among exponential, gamma and lognormal alternatives?

5. How can a Bronze, Silver and Gold data architecture support the acquisition, cleaning, integration and analysis of large flight datasets?

## 5. Project Objectives

### General Objective

Develop a reproducible Big Data workflow to process historical U.S. flight records and weather forecasts, identify patterns associated with departure delays and compare classification models with and without meteorological features.

### Specific Objectives

- Acquire historical flight data from the Bureau of Transportation Statistics.
- Acquire historical hourly weather forecasts from Open-Meteo.
- Organize data into Bronze, Silver and Gold layers.
- Clean, standardize and validate flight and weather records.
- Integrate forecasts with flights using origin airport and scheduled departure time in UTC.
- Explore delay patterns by time, airport, airline and weather conditions.
- Train an XGBoost baseline without weather features.
- Train a comparable XGBoost model with six weather forecast variables.
- Evaluate both models using a chronological train, validation and test split.
- Analyze significant delay durations using exponential, gamma and lognormal probability distributions.
- Document technical decisions, quality checks, limitations and results.

## 6. Unit of Analysis

### What Does One Row Represent?

The meaning of a row depends on the dataset.

| Dataset | Unit of analysis |
|---|---|
| BTS Bronze | One flight record reported by BTS |
| Weather Bronze/Silver | One forecast for an airport at a specific UTC hour |
| Flights Silver | One non-canceled flight record with a valid departure delay |
| Gold operational flights | One flight with engineered operational and temporal features |
| Gold flights with weather | One flight matched to a historical forecast for its origin airport and scheduled departure hour |
| Daily delay time series | One calendar day containing aggregated flight delay statistics |
| Hourly delay summary | One scheduled departure-hour group containing aggregated statistics |
| Delay distribution | One delay-duration category with its frequency |

The primary modeling unit is an individual flight record.

The target variable indicates whether that flight departed at least 15 minutes later than its scheduled departure time.

## 7. Data Sources

### Flight Data

Provider: Bureau of Transportation Statistics (BTS), U.S. Department of Transportation.

Original ingestion period: January 2022 through December 2025.

Approximate original volume: 27.7 million records.

Relevant fields include flight date, origin airport, destination airport, reporting airline, scheduled departure time, cancellation status and actual departure delay.

### Weather Data

Provider: Open-Meteo Previous Runs API.

Weather model: GFS Global.

Forecast variables: temperature, precipitation, wind speed, relative humidity, cloud cover and sea-level pressure.

Forecast type: `_previous_day1`.

The forecast variables represent predictions generated approximately one day before their weather valid hour.

The project does not use actual weather observations recorded after the flight.

## 8. Study Scope

### Geographic Scope

The original flight dataset covers U.S. flight operations reported by BTS.

The weather-integrated study focuses on 30 selected origin airports.

The destination airport can be outside this group because the weather integration uses the origin airport.

### Temporal Scope

The weather-integrated analysis covers February 2024 through December 2025.

BTS data from 2022 and 2023 remains available in the original Bronze layer but is not part of the current weather-model comparison.

### Flight Scope

The classification analysis includes non-canceled flights with a valid departure delay target.

Canceled flights are excluded because the binary objective concerns actual departure delays.

### Weather Scope

Forecasts are matched using the flight origin airport and scheduled departure hour converted to UTC.

The modeling dataset includes only flights with a matching forecast.

## 9. Modeling Approach

### Target Definition

A significant departure delay is defined as:

`delayed_15 = 1` if actual departure delay is at least 15 minutes.

Otherwise:

`delayed_15 = 0`.

### Baseline Model

The baseline XGBoost uses information available from the flight schedule:

- Origin airport.
- Destination airport.
- Reporting airline.
- Month.
- Day of the week.
- Scheduled departure hour.
- Weekend indicator.
- Summer indicator.
- Peak-hour indicator.

### Weather Model

The weather XGBoost uses the same baseline features plus six previous-day weather forecast variables.

Both models use the same sampled flights and chronological evaluation periods.

### Prediction Timing

The implemented models use scheduled flight characteristics and historical weather forecasts that are available before departure.

The current experiment does not use live operational updates or recent actual flight delays.

Although an earlier project plan considered a three-hour prediction horizon, the implemented comparison does not use dynamic information specifically collected three hours before departure.

Therefore, the current results should be described as advance predictions using scheduled information and previous-day forecasts, rather than as evidence of a continuously updated three-hour-ahead system.

## 10. Expected Users

Potential users of the results include:

- Airline operations and planning teams.
- Airport analysts.
- Aviation researchers.
- Data science teams studying flight reliability.
- Passengers interested in understanding factors associated with delays.

The current project is an academic prototype and is not a production flight-alert service.

## 11. Expected Value

The project aims to provide value in three areas.

### Data Engineering

Demonstrate how millions of flight records can be acquired, processed and integrated with another historical data source using a layered architecture.

### Predictive Analytics

Measure whether previous-day weather forecasts contribute useful information to departure delay classification.

### Statistical Analysis

Describe the distribution of significant delay durations and identify temporal patterns requiring further investigation.

## 12. Results Obtained

The project processed:

- Approximately 27.7 million original BTS flight records in Bronze.
- 504,000 weather forecast records.
- 8,655,359 cleaned flights in Silver.
- 8,653,172 flights with matching weather forecasts.
- 258,844 sampled flights for the XGBoost comparison.

### Model Comparison

| Metric | XGBoost without weather | XGBoost with weather |
|---|---:|---:|
| Accuracy | 0.7213 | 0.7457 |
| Precision | 0.3818 | 0.4198 |
| Recall | 0.3904 | 0.3477 |
| F1 | 0.3861 | 0.3804 |
| ROC-AUC | 0.6731 | 0.6873 |

Weather features improved accuracy, precision and ROC-AUC, but recall and F1 were lower.

The results do not identify a single model that is superior on every metric.

### Statistical Distribution Analysis

A sample of 156,413 flights delayed by at least 15 minutes was analyzed.

Gamma achieved the lowest AIC and BIC among the exponential, gamma and lognormal candidates.

This result describes significant delay durations and does not replace the binary classification models.

## 13. Limitations

The main limitations are:

- The weather-model comparison covers 30 origin airports rather than all airports.
- Weather features come from historical forecasts, not real-time forecast updates.
- Some flights lack matching weather forecasts.
- The project excludes canceled flights.
- The initial modeling experiment uses a sampled subset of the full dataset.
- The models do not incorporate live airport conditions or delay propagation.
- Potentially duplicated flight records require additional investigation.
- Some weather values required documented adjustments.
- Additional reproducibility tests and processing benchmarks are still required.
- The statistical analysis compares only three candidate probability distributions.
- Model associations do not establish causal relationships.

## 14. Out of Scope

The current implementation does not include:

- Real-time delay monitoring.
- A passenger-facing prediction application.
- Automatic model retraining.
- Live flight tracking.
- Delay propagation modeling across aircraft rotations.
- Production deployment.
- A completed GARCH volatility model.
- Guaranteed predictions for airports outside the selected group.

These may be considered for future project iterations.

## 15. Conclusion

The project demonstrates an end-to-end academic Big Data workflow, from historical flight and weather acquisition to data validation, exploratory analysis and predictive modeling.

Its main contribution is a reproducible comparison of flight delay classifiers with and without previous-day weather forecasts.

The results show that weather information provides some additional discriminatory value, while also introducing trade-offs between precision and recall.

Further improvements should focus on automated reproducibility, additional airports, updated operational information and more extensive model evaluation.
