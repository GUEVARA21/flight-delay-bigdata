# Final Project Report
## Flight Delay Prediction Using Big Data and Historical Weather Forecasts

## 1. Executive Summary

This project investigates whether historical flight information and previous-day weather forecasts can help predict significant departure delays in U.S. aviation.

The team developed a data engineering pipeline in Databricks using PySpark, Delta Lake and a Bronze, Silver and Gold architecture.

Approximately 27.7 million historical flight records were ingested from the Bureau of Transportation Statistics (BTS). The weather-integrated study focused on 30 origin airports between February 2024 and December 2025.

The final modeling dataset included 8,653,172 flights matched with historical weather forecasts from Open-Meteo.

Two XGBoost classifiers were trained on the same reproducible sample: one using scheduled operational information and another adding six meteorological variables.

The weather model achieved a higher ROC-AUC (0.6873 versus 0.6731), accuracy and precision. However, the baseline performed better in recall and F1.

A complementary statistical analysis compared exponential, gamma and lognormal distributions for significant departure delay durations. Gamma produced the lowest AIC and BIC among the three distributions.

The results demonstrate both the potential and limitations of incorporating historical forecasts into flight delay prediction.

## 2. Problem Statement

Flight delays affect passengers, airport operations and airline planning.

Predicting which flights are more likely to depart late could support earlier decisions and improved operational awareness.

However, departure delays are influenced by multiple factors, and useful prediction requires both reliable historical data and information available before the flight.

This project combines a Big Data engineering challenge with a supervised learning problem.

## 3. Research Question

Can historical scheduled flight characteristics and previous-day weather forecasts help predict whether a U.S. flight will depart at least 15 minutes late?

### Secondary Questions

- How do delay rates vary by airport, airline, month, weekday and scheduled departure hour?
- How are forecasted precipitation levels associated with departure delays?
- Does including weather information improve XGBoost classification performance?
- Which of three candidate statistical distributions best describes significant delay duration?
- How can a layered data architecture support processing and validation at scale?

## 4. Data Sources

### 4.1 Bureau of Transportation Statistics

Historical BTS flight records were used as the main source.

Original ingestion period: 2022-2025.

Approximate Bronze volume: 27.7 million records.

Relevant variables include flight date, origin, destination, reporting airline, scheduled departure time, cancellation status and actual departure delay.

### 4.2 Open-Meteo

Historical hourly forecasts were obtained through the Open-Meteo Previous Runs API using the GFS Global model.

The selected variables were:

- Temperature at 2 meters.
- Precipitation.
- Wind speed at 10 meters.
- Relative humidity.
- Cloud cover.
- Mean sea-level pressure.

The `_previous_day1` forecasts represent meteorological predictions generated approximately one day before their corresponding weather valid hour.

This distinction matters because the models should not use weather observations collected after the flight.

## 5. Data Architecture

The system follows a Medallion Architecture:

```text
BTS Flights                Open-Meteo Forecasts
     |                              |
     v                              v
 Bronze Flights               Bronze Weather
     |                              |
     v                              v
 Silver Flights               Silver Weather
     |                              |
     v                              |
 Gold Operational Features         |
     |                              |
     +---------------+--------------+
                     |
                     v
           Gold Flights + Weather
                     |
              +------+------+
              |             |
              v             v
         XGBoost         XGBoost
        Without           With
         Weather         Weather
              |             |
              +------+------+
                     |
                     v
               Comparison

Gold Flight Data
       |
       v
Delay Distributions and Daily Time Series
       |
       v
Statistical Distribution Analysis
```

### Bronze

Preserves the original flight and weather information together with ingestion metadata.

### Silver

Standardizes data types and names, removes canceled flights from the modeling population, validates target labels and prepares UTC weather timestamps.

### Gold

Creates operational features, integrates hourly forecasts and prepares datasets for EDA, machine learning and statistical analysis.

The complete architecture and table names are documented in `docs/architecture.md`.

## 6. Data Processing

### 6.1 Flight Preparation

The original BTS records were filtered to retain non-canceled flights with valid departure delay information.

The modeling scope was restricted to the selected 30 origin airports and the historical forecast coverage period.

The cleaned Silver flight table contains 8,655,359 records.

The target was defined as:

`delayed_15 = 1` when `dep_delay >= 15`.

Otherwise:

`delayed_15 = 0`.

### 6.2 Weather Preparation

The weather dataset contains 504,000 airport-hour forecasts.

Weather valid times were standardized to UTC.

The six forecast variables were preserved for the weather-based model.

### 6.3 Flight and Weather Integration

Scheduled flight departure times were converted from airport local time to UTC.

Each flight was matched to the forecast corresponding to its origin airport and scheduled departure hour.

The initial left join preserved all 8,655,359 flight rows.

Matching forecasts were available for 8,653,172 flights, providing approximately 99.97% coverage.

The common modeling dataset excludes the 2,187 unmatched records.

## 7. Data Quality and Validation

### Flight Validation

The Silver flight dataset contained:

- No null values in the essential modeling fields.
- No canceled flights.
- No incorrect binary delay labels.
- 179 groups of potentially duplicated records requiring further verification.

Potential duplicate groups were not removed automatically because the checked fields do not guarantee a unique flight identifier.

### Weather Validation

The six weather variables had no missing values in the cleaned weather dataset.

However, 371 cloud-cover values were outside the expected 0%-100% range:

- 120 records had -1.
- 251 records had 101.

These values were clipped to 0 and 100 when preparing Gold, while the source values remained preserved in earlier layers.

Additional details are available in `docs/validation.md`.

## 8. Exploratory Data Analysis

### 8.1 Seasonality

Delay rates varied across months.

Summer months, particularly June and July, showed relatively high proportions of delayed departures.

July reached approximately 31% in the exploratory visualization.

This is a descriptive association and does not establish that summer itself causes delays.

### 8.2 Day of the Week

Sunday showed one of the highest observed delay rates, approximately 26%.

The analysis indicates that the frequency of delays is not uniform throughout the week.

### 8.3 Scheduled Departure Hour

Delay rates generally increased toward the afternoon and evening.

Hours around 19:00-20:00 showed relatively high delay rates.

One apparent 100% delay rate at 02:00 was traced to only two flights.

This result highlights the importance of considering sample size before interpreting percentages.

### 8.4 Airports and Airlines

Origin airports and airlines displayed different historical delay rates.

The results suggest that operational and geographic characteristics may contain useful predictive information.

These comparisons are descriptive and are not adjusted causal effects.

### 8.5 Forecasted Precipitation

The following association was found between forecasted precipitation and significant departure delays:

| Forecasted precipitation | Flights | Delay rate |
|---|---:|---:|
| No rain | 7,741,389 | 21.52% |
| Greater than 0 and less than 1 mm | 692,463 | 30.33% |
| 1 to less than 5 mm | 199,779 | 36.97% |
| 5 to less than 10 mm | 14,487 | 42.85% |
| 10 mm or more | 5,054 | 54.67% |

Higher forecasted precipitation was associated with higher observed delay rates.

However, this association does not prove that precipitation alone caused the delays.

### 8.6 Daily Time Series

Daily average delay and dispersion varied substantially over time.

Some days showed unusually high mean delays and standard deviations.

These observations support further investigation of time-series behavior, but do not by themselves establish the suitability of a GARCH model.

## 9. Machine Learning Methodology

### 9.1 Dataset

Both XGBoost models used the same deterministic sample of 258,844 flights.

The dataset was split chronologically:

| Dataset | Period | Flights |
|---|---|---:|
| Training | February 2024 - May 2025 | 179,783 |
| Validation | June 2025 - August 2025 | 34,571 |
| Testing | September 2025 - December 2025 | 44,490 |

The observed delay rates were:

- Training: 21.56%.
- Validation: 27.98%.
- Testing: 22.45%.

### 9.2 Baseline XGBoost

The baseline model used:

- Origin airport.
- Destination airport.
- Airline.
- Month.
- Day of week.
- Scheduled departure hour.
- Weekend indicator.
- Summer indicator.
- Peak-hour indicator.

The actual delay value was excluded from model features.

### 9.3 Weather XGBoost

The weather model used the same operational features plus:

- Temperature.
- Precipitation.
- Wind speed.
- Relative humidity.
- Cloud cover.
- Sea-level pressure.

Both models used the same initial XGBoost hyperparameters.

### 9.4 Classification Threshold

The baseline initially performed poorly in recall when using the default 0.50 classification threshold.

Each model subsequently selected its threshold by maximizing F1 using validation data only.

Selected thresholds:

- Baseline: 0.225.
- Weather model: 0.250.

These thresholds were then applied to the independent test period.

## 10. Machine Learning Results

### 10.1 Test-Set Performance

| Metric | Without weather | With weather | Difference |
|---|---:|---:|---:|
| Accuracy | 72.13% | 74.57% | +2.44 percentage points |
| Precision | 38.18% | 41.98% | +3.80 percentage points |
| Recall | 39.04% | 34.77% | -4.27 percentage points |
| F1 | 0.3861 | 0.3804 | -0.0057 |
| ROC-AUC | 0.6731 | 0.6873 | +0.0142 |

### 10.2 Interpretation

The weather model improved accuracy, precision and ROC-AUC.

The higher ROC-AUC indicates a modest improvement in the model's ability to distinguish flights that experienced significant delays from those that did not.

The higher precision means a greater proportion of the weather model's positive delay predictions were correct at its selected threshold.

However, recall decreased from 39.04% to 34.77%.

This means the weather model identified a smaller proportion of actual delayed flights at the selected operating threshold.

F1 also decreased slightly.

Therefore, adding meteorological predictors provided some measurable benefit, but did not improve every evaluation metric.

### 10.3 Main Finding

The experiment supports the conclusion that previous-day weather forecasts add some discriminatory information to the available scheduled flight features.

Nevertheless, the improvement was modest and involved trade-offs.

If the operational priority is to detect more delayed flights, the baseline achieved higher recall in this experiment.

If the priority is to obtain more reliable delay alerts, the weather model achieved higher precision.

The best model therefore depends on the operational objective and the cost of false alarms compared with missed delays.

## 11. Statistical Distribution Analysis

A complementary analysis investigated the duration of departure delays of at least 15 minutes.

A reproducible sample of 156,413 delayed flights was used.

Descriptive results:

- Mean delay: 70.12 minutes.
- Median delay: 43 minutes.
- Maximum observed delay: 2,521 minutes.

Three candidate probability distributions were fitted to the excess delay duration above 15 minutes.

| Distribution | AIC | BIC |
|---|---:|---:|
| Gamma | 1,528,592.88 | 1,528,612.80 |
| Exponential | 1,567,104.56 | 1,567,114.52 |
| Lognormal | 1,611,343.64 | 1,611,363.56 |

Gamma achieved the lowest AIC and BIC of the three candidates.

This indicates the best relative statistical fit under the fitting procedure used.

It does not prove that gamma perfectly describes all delays or their extreme tail.

This analysis is descriptive and complementary to the binary XGBoost classifiers.

GARCH was discussed as a possible future time-series extension but was not implemented in this analysis.

## 12. Technical Challenges

### 12.1 Data Volume

Processing millions of flight records required an approach more scalable than working exclusively with local pandas DataFrames.

PySpark and Delta tables allowed the team to prepare the larger datasets before extracting smaller samples for modeling.

### 12.2 Weather Coverage

Collecting and integrating weather forecasts required limiting the initial geographic scope and verifying temporal coverage.

### 12.3 Time Zone Conversion

BTS scheduled departure times were local to the origin airport, while the weather forecasts used UTC.

The team implemented a time zone conversion before the weather join.

### 12.4 Data Quality

Canceled flights, potentially duplicated records and invalid cloud-cover values required different handling decisions.

### 12.5 Model Evaluation

Class imbalance and temporal changes affected classification performance.

The baseline showed that accuracy alone was insufficient to evaluate delayed-flight detection.

## 13. Limitations

The main limitations are:

1. The weather-integrated analysis includes 30 origin airports.
2. Only February 2024 through December 2025 is included in the model comparison.
3. The models use a sample of the full weather-integrated dataset.
4. The weather values are historical forecasts rather than observations.
5. Canceled flights are excluded from the target population.
6. The models do not include real-time operational updates or delay propagation.
7. Potential duplicate flight groups remain under investigation.
8. Additional validation around daylight-saving-time transitions is recommended.
9. Classification results were obtained from one primary chronological holdout design.
10. The statistical distribution comparison does not guarantee accurate modeling of extreme events.
11. The complete workflow still requires additional automation and clean-environment reproducibility tests.
12. The differences in model performance are descriptive results from this evaluation, not proof of causal effects.

## 14. Conclusions

The project successfully implemented a layered data engineering workflow that integrated large historical flight records with hourly weather forecasts.

The Bronze, Silver and Gold architecture supported data acquisition, cleaning, validation, feature engineering and exploratory analysis.

The XGBoost comparison showed that previous-day weather information modestly improved ROC-AUC, accuracy and precision.

At the same time, recall and F1 decreased, showing that adding more variables does not necessarily improve all aspects of predictive performance.

The statistical analysis showed that gamma was the best-fitting distribution among the three candidate models for significant departure delay duration.

Overall, the project demonstrates the importance of combining data engineering, data quality and careful evaluation when developing predictive models.

## 15. Future Work

Recommended future improvements include:

- Expand weather coverage to additional airports.
- Add historical operational features available before departure.
- Investigate aircraft rotations and delay propagation.
- Test more forecast horizons and updated weather products.
- Evaluate alternative model architectures and hyperparameter settings.
- Compare calibrated probabilities and threshold strategies.
- Test robustness across airports, airlines and seasons.
- Investigate stronger identifiers for possible duplicate flights.
- Add automated quality tests and command-line execution.
- Benchmark processing performance and scalability.
- Evaluate volatility models such as GARCH after appropriate time-series diagnostics.

## 16. Supporting Documentation

- `README.md`: project overview and execution guidance.
- `docs/project_scope.md`: research question and scope.
- `docs/data_dictionary.md`: variables and definitions.
- `docs/architecture.md`: data architecture and transformations.
- `docs/validation.md`: quality controls and findings.
- `docs/decisions.md`: technical decisions and alternatives.

Source notebooks and analyses are stored in the repository under `src/`.
