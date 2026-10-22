# Data Dictionary

## 1. Project Overview

This project predicts whether a U.S. domestic flight will experience a departure delay of 15 minutes or more.

Flight information comes from the Bureau of Transportation Statistics (BTS), while historical weather forecasts come from Open-Meteo.

The data pipeline is organized into Bronze, Silver, and Gold layers in Databricks.

## 2. Data Sources

| Source | Description | Period |
|---|---|---|
| BTS | Historical flight operations and departure delays | 2022-2025 |
| Open-Meteo | Historical GFS weather forecasts using previous_day1 | February 2024 - December 2025 |

The final weather-integrated dataset focuses on 30 origin airports.

## 3. Bronze Layer

### 3.1 BTS Flights

Table: `workspace.bronze_flights.bts_flights_raw`

Each record represents a flight reported in the BTS dataset.

| Variable | Type | Description | Unit | Missing values | Observations |
|---|---|---|---|---|---|
| FlightDate | Date/String | Flight operating date | Local date | Not quantified | Converted to DATE in Silver |
| Origin | String | Origin airport IATA code | - | Not quantified | Used for weather matching |
| Dest | String | Destination airport IATA code | - | Not quantified | Categorical predictor |
| Reporting_Airline | String | Airline reporting code | - | Not quantified | Renamed to airline |
| CRSDepTime | Integer | Scheduled departure time in HHMM format | Local time | Not quantified | Not a UTC timestamp |
| DepDelay | Numeric | Actual departure delay | Minutes | 453,438 nulls in Bronze | Missing values corresponded to canceled flights |
| Cancelled | Numeric/Boolean | Flight cancellation indicator | 0 or 1 | Not quantified | Canceled flights excluded from Silver |
| Tail_Number | String | Aircraft registration identifier | - | Not quantified | May contain missing values |
| _source_year | Integer | Source file year | Year | Not quantified | Ingestion traceability |
| _source_month | Integer | Source file month | Month | Not quantified | Ingestion traceability |
| _source_file | String | Original file identifier | - | Not quantified | Ingestion traceability |
| _ingestion_timestamp | Timestamp | Time when data was ingested | Timestamp | Not quantified | Pipeline metadata |

Bronze preserves the original data and its ingestion metadata.

### 3.2 Open-Meteo Weather Forecasts

Table: `workspace.bronze_flights.weather_forecasts_raw`

Each record represents a forecast for one airport and one UTC weather hour.

| Variable | Type | Description | Unit | Missing values | Observations |
|---|---|---|---|---|---|
| airport | String | Airport IATA code | - | 0 in cleaned dataset | Weather join key |
| forecast_valid_time_utc | String | Hour represented by the forecast | UTC | 0 in cleaned dataset | Converted to timestamp in Silver |
| temperature_2m_previous_day1 | Double | Forecasted temperature at 2 meters | Celsius | 0 in cleaned dataset | Forecast generated approximately 24 hours before valid time |
| precipitation_previous_day1 | Double | Forecasted hourly precipitation | mm | 0 in cleaned dataset | Used in weather analysis |
| wind_speed_10m_previous_day1 | Double | Forecasted wind speed at 10 meters | km/h | 0 in cleaned dataset | Meteorological predictor |
| relative_humidity_2m_previous_day1 | Double | Forecasted relative humidity | % | 0 in cleaned dataset | Meteorological predictor |
| cloud_cover_previous_day1 | Double | Forecasted cloud coverage | % | 0 in cleaned dataset | Some values outside 0-100 |
| pressure_msl_previous_day1 | Double | Forecasted pressure at mean sea level | hPa | 0 in cleaned dataset | Meteorological predictor |
| forecast_lead_hours | Integer | Forecast lead time | Hours | Not quantified | Expected value: 24 |
| source_model | String | Weather forecasting model | - | Not quantified | GFS Global |
| source_year | Integer | Source year | Year | Not quantified | Ingestion metadata |
| source_month | Integer | Source month | Month | Not quantified | Ingestion metadata |

The weather dataset contains 504,000 airport-hour records for 30 airports.

## 4. Silver Layer

### 4.1 Clean Flights

Table: `workspace.silver_flights.flights_clean`

Each row represents a non-canceled flight with a valid departure delay label.

| Variable | Type | Description | Unit | Missing values | Observations |
|---|---|---|---|---|---|
| flight_date | Date | Flight date | Local date | 0 | Used for chronological splitting |
| origin | String | Origin airport | IATA code | 0 | Categorical predictor |
| dest | String | Destination airport | IATA code | 0 | Categorical predictor |
| airline | String | Reporting airline | Airline code | 0 | Categorical predictor |
| tail_number | String | Aircraft identifier | - | Not quantified | Not used in initial XGBoost |
| crs_dep_time | Integer | Scheduled departure HHMM | Local time | 0 | Source for departure hour |
| departure_hour | Integer | Scheduled departure hour | 0-23 | 0 | Operational predictor |
| departure_minute | Integer | Scheduled departure minute | 0-59 | Not quantified | Used for timestamp construction |
| dep_delay | Double | Actual departure delay | Minutes | 0 | Target information; NEVER a predictor |
| cancelled | Numeric | Cancellation indicator | 0 or 1 | 0 | All retained flights are non-canceled |
| delayed_15 | Integer | Binary departure delay label | 0 or 1 | 0 | 1 if dep_delay >= 15 |
| year | Integer | Operating year | Year | Not quantified | Calendar feature |
| month | Integer | Operating month | 1-12 | Not quantified | Operational predictor |
| day_of_week | Integer | Day of week | 1-7 | Not quantified | Spark convention: Sunday = 1 |

Silver flights contains 8,655,359 records covering February 2024 through December 2025.

The binary target is defined as:

`delayed_15 = 1 if dep_delay >= 15, otherwise 0`

### 4.2 Clean Weather

Table: `workspace.silver_flights.weather_clean`

Weather records have UTC timestamps and standardized variable names.

| Variable | Type | Description | Unit | Missing values | Observations |
|---|---|---|---|---|---|
| airport | String | Airport code | IATA code | 0 | Weather join key |
| weather_time_utc | Timestamp | Forecast valid hour | UTC | 0 | Weather join key |
| temperature_2m | Double | Forecasted temperature | Celsius | 0 | Predictor |
| precipitation | Double | Forecasted precipitation | mm | 0 | Predictor |
| wind_speed_10m | Double | Forecasted wind speed | km/h | 0 | Predictor |
| relative_humidity_2m | Double | Forecasted relative humidity | % | 0 | Predictor |
| cloud_cover | Double | Forecasted cloud coverage | % | 0 | Predictor; range issues identified |
| pressure_msl | Double | Forecasted sea-level pressure | hPa | 0 | Predictor |
| forecast_lead_hours | Integer | Forecast horizon | Hours | Not quantified | Expected value: 24 |

The Silver weather table contains 504,000 records.

Quality validation identified 371 out-of-range cloud cover values:
- 120 observations with cloud_cover = -1.
- 251 observations with cloud_cover = 101.

These values were clipped to 0 and 100, respectively, in the Gold weather integration. The original Bronze values were preserved.

## 5. Gold Layer

### 5.1 Operational Flight Features

Table: `workspace.gold_flights.flights_features`

| Variable | Type | Description | Unit | Missing values | Observations |
|---|---|---|---|---|---|
| origin_timezone | String | IANA time zone of origin airport | - | 0 | Used for UTC conversion |
| scheduled_departure_local | Timestamp | Scheduled departure in local airport time | Local time | Not quantified | Constructed from BTS schedule |
| scheduled_departure_utc | Timestamp | Scheduled departure converted to UTC | UTC | Not quantified | Uses airport time zone |
| weather_hour_utc | Timestamp | Scheduled departure rounded down to the hour | UTC | Not quantified | Used to join weather |
| is_weekend | Integer | Weekend indicator | 0 or 1 | Not quantified | Saturday or Sunday |
| is_summer | Integer | Summer month indicator | 0 or 1 | Not quantified | June, July or August |
| is_peak_hour | Integer | Evening peak-hour indicator | 0 or 1 | Not quantified | Scheduled departures from 16:00 to 21:59 |

Gold retains the relevant Silver variables and adds features for model training.

### 5.2 Flights with Weather

Table: `workspace.gold_flights.flights_weather_features`

Each row represents a flight matched with the forecast for its origin airport and scheduled departure hour.

| Field group | Description | Observations |
|---|---|---|
| Operational features | Origin, destination, airline and scheduled calendar features | Shared by both XGBoost models |
| Meteorological features | Six previous_day1 forecast variables | Used only by the weather model |
| delayed_15 | Binary classification target | Used for training and evaluation |
| dep_delay | Actual delay in minutes | Excluded from model predictors |

The initial Gold weather join preserved all 8,655,359 flights.

A total of 8,653,172 flights had matching weather forecasts, representing approximately 99.97% coverage.

The 2,187 flights without matched forecasts were excluded from the shared modeling dataset.

## 6. Machine Learning Features

### Baseline XGBoost

Predictors:
- origin
- dest
- airline
- month
- day_of_week
- departure_hour
- is_weekend
- is_summer
- is_peak_hour

### Weather XGBoost

Uses all baseline predictors plus:
- temperature_2m
- precipitation
- wind_speed_10m
- relative_humidity_2m
- cloud_cover
- pressure_msl

### Target Variable

`delayed_15`: binary variable indicating whether departure delay is 15 minutes or more.

Excluded from predictive features:
- dep_delay
- delayed_15 itself
- actual post-departure information

Both models use the same deterministic sample and chronological train, validation and test periods.

## 7. Statistical Analysis

The complementary statistical study uses `dep_delay` for flights with departure delays of at least 15 minutes.

The analysis compares exponential, gamma and lognormal distributions based on AIC and BIC.

The gamma distribution achieved the best relative fit among the three tested alternatives.

## 8. Known Limitations

- Only 30 origin airports were included in the weather-integrated modeling dataset.
- Weather data represents historical forecasts, not actual observed weather.
- Some weather hours were not matched to flights.
- 179 groups of potentially duplicated flight records were identified during Silver validation; they were not automatically removed.
- Cloud cover values outside the valid range were corrected in Gold.
- Some variable-level missing-value percentages have not been measured and are reported as not quantified rather than assumed to be zero.
- The statistical distribution models describe delays of at least 15 minutes and do not represent all departures.
