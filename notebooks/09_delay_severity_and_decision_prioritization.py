# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "xgboost==3.0.5",
# ]
# ///
# MAGIC %md
# MAGIC # 09 · Delay severity & decision prioritization
# MAGIC **Flight Delay Intelligence | reproducible extension of notebooks 05 and 08**
# MAGIC
# MAGIC **Decision question:** If only 5%, 10%, or 20% of flights can receive extra attention, does ranking them by **expected significant-delay minutes** identify more observed delay burden than ranking by **delay probability alone**?
# MAGIC
# MAGIC **Key safeguards**
# MAGIC 1. The source is `flights_weather_features`, exactly as in Notebook 05. Same deterministic sample and four chronological splits; *never* join the persisted Notebook 05 predictions to source flights because that output lacks a unique physical flight ID.
# MAGIC 2. We rebuild the two classifiers from the **explicit Notebook 05 configuration** and compare the newly measured test AUCs with the persisted 05 results. This is a reproduction check, **not** a loaded production model artifact.
# MAGIC 3. A second XGBoost model learns the conditional capped departure-delay duration **using only late flights in the training period**. It never uses `dep_delay` as a feature.
# MAGIC 4. Operational candidate = **schedule-only baseline**. Weather version is **retrospective sensitivity only**, until forecast availability at the decision timestamp is proven; observed weather must not masquerade as 24h forecasts.
# MAGIC 5. A constant hypothetical USD/min rate **does not change the order of the flights**. Ranking expected cost and expected minutes would be identical in our current setup. This notebook measures *retrospective identification* of costly delays, **not realized savings, intervention efficacy, or causal benefit**.
# MAGIC 6. The sample is from the 30-airport analytical subset with weather data (not all US flights). 2024–2025 only. No cancellation costs or passenger counts. Row-level physical-flight uniqueness remains to be audited.
# MAGIC
# MAGIC **Execution:** start with `SAVE_TABLES = False`; run and inspect ALL checks first. Enabling saving writes only new `part3_decision_*` Delta tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Configuration and source checks
# MAGIC If import fails on a new compute session, use a separate Databricks cell with `%pip install xgboost==3.0.5` and restart Python if requested.

# COMMAND ----------

# MAGIC %pip install xgboost==3.0.5

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %pip install catboost==1.2.10

# COMMAND ----------

import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, mean_absolute_error,
                             mean_squared_error, brier_score_loss)
from xgboost import XGBClassifier, XGBRegressor

GOLD_TABLE = "workspace.gold_flights.flights_weather_features"
METRICS_05_TABLE = "workspace.gold_flights.part3_model_metrics"
PREDICTIONS_05_TABLE = "workspace.gold_flights.part3_test_predictions"
FINANCIAL_08_TABLE = "workspace.gold_flights.part3_financial_assumptions"
OUTPUT_SCHEMA = "workspace.gold_flights"
SAVE_TABLES = False  # first pass: do not overwrite/create any tables
SAMPLE_PERCENT = 3.0
RANDOM_SEED = 2026
LATE_THRESHOLD_MIN = 15
COST_CAP_MIN = 300
REFERENCE_USD_PER_MIN = 22.173  # illustrative gate delay proxy from Notebook 08, NOT a realized loss
CAPACITIES = [0.05, 0.10, 0.20]
BOOTSTRAP_ITERATIONS = 200
BOOTSTRAP_CAPACITY = 0.10

CATEGORICAL = ["origin", "dest", "airline"]
OPERATIONAL_NUMERIC = ["month", "day_of_week", "departure_hour", "is_weekend", "is_summer", "is_peak_hour"]
WEATHER_NUMERIC = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "cloud_cover", "pressure_msl"]
FEATURES_ALL = CATEGORICAL + OPERATIONAL_NUMERIC + WEATHER_NUMERIC
REQUIRED_05 = ["flight_date", "delayed_15", "crs_dep_time"] + FEATURES_ALL
REQUIRED_09 = REQUIRED_05 + ["dep_delay"]
assert 0 < SAMPLE_PERCENT <= 100 and COST_CAP_MIN >= LATE_THRESHOLD_MIN
assert BOOTSTRAP_ITERATIONS >= 50
assert spark.catalog.tableExists(GOLD_TABLE), f"Missing source: {GOLD_TABLE}"
assert spark.catalog.tableExists(METRICS_05_TABLE), f"Missing Notebook 05 metrics: {METRICS_05_TABLE}"
assert spark.catalog.tableExists(PREDICTIONS_05_TABLE), f"Missing Notebook 05 test predictions: {PREDICTIONS_05_TABLE}"
assert spark.catalog.tableExists(FINANCIAL_08_TABLE), f"Missing Notebook 08 assumptions: {FINANCIAL_08_TABLE}"
source = spark.table(GOLD_TABLE)
missing = sorted(set(REQUIRED_09) - set(source.columns))
if missing:
    raise ValueError(f"Required source columns unavailable: {missing}")
print("Source:", GOLD_TABLE)
print("Target:", f"dep_delay >= {LATE_THRESHOLD_MIN} minutes; cap={COST_CAP_MIN} min")
print("Cost rate =", REFERENCE_USD_PER_MIN, "illustrative USD/min; no realized savings claims")
print("SAVE_TABLES:", SAVE_TABLES)
print("NOTE: Weather is not verified as a forecast available at the pre-departure decision time.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Reproduce Notebook 05's deterministic sample and temporal split
# MAGIC Original Notebook 05 does NOT persist a unique flight key in `part3_test_predictions`. To avoid a many-to-many or arbitrary row join, reproduce its sampling and the classifiers directly here. The `dep_delay` column is added strictly as a **target** for severity and evaluation.

# COMMAND ----------

working = (source.select(*REQUIRED_09)
           .withColumn("flight_date", F.to_date("flight_date"))
           .filter((F.col("flight_date") >= F.lit("2024-02-01")) &
                   (F.col("flight_date") < F.lit("2026-01-01")))
           .filter(F.col("delayed_15").isin(0, 1)))
for c in CATEGORICAL:
    working = working.filter(F.col(c).isNotNull())
working = working.filter(F.col("crs_dep_time").isNotNull())
keys = [F.coalesce(F.col(x).cast("string"), F.lit("<NULL>"))
        for x in ["flight_date", "origin", "dest", "airline", "crs_dep_time"]]
sample = working.filter(F.pmod(F.xxhash64(*keys), F.lit(10000)) < F.lit(round(SAMPLE_PERCENT * 100)))
pdf = sample.toPandas()  # ~3% only; never collect the 8.65M source
if len(pdf) < 10000:
    raise RuntimeError("Sample too small; review source and sampling before proceeding.")
pdf["flight_date"] = pd.to_datetime(pdf["flight_date"])
pdf["delayed_15"] = pd.to_numeric(pdf["delayed_15"], errors="raise").astype(int)
pdf["dep_delay"] = pd.to_numeric(pdf["dep_delay"], errors="coerce")
for c in CATEGORICAL:
    pdf[c] = pdf[c].astype(str)
for c in OPERATIONAL_NUMERIC + WEATHER_NUMERIC:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
if pdf.dep_delay.isna().any():
    raise ValueError("Null dep_delay: explicit treatment is required before financial modeling")
if not ((pdf["dep_delay"] >= LATE_THRESHOLD_MIN).astype(int) == pdf["delayed_15"]).all():
    raise ValueError("Delayed_15 label disagrees with dep_delay >= 15. Stop before training.")

cut_train, cut_calib, cut_valid = (pd.Timestamp("2025-06-01"),
                                   pd.Timestamp("2025-08-01"),
                                   pd.Timestamp("2025-09-01"))
train = pdf.loc[pdf.flight_date < cut_train].copy()
calib = pdf.loc[(pdf.flight_date >= cut_train) & (pdf.flight_date < cut_calib)].copy()
valid = pdf.loc[(pdf.flight_date >= cut_calib) & (pdf.flight_date < cut_valid)].copy()
test = pdf.loc[pdf.flight_date >= cut_valid].copy()
split_summary = pd.DataFrame([{"split": name, "flights": len(frame),
                               "delay_rate": float(frame.delayed_15.mean()),
                               "capped_observed_minutes": float(np.where(frame.delayed_15 == 1,
                                  np.minimum(frame.dep_delay, COST_CAP_MIN), 0).sum())}
                              for name, frame in [("train", train), ("calibration", calib),
                                                  ("threshold_validation", valid), ("test", test)]])
print(split_summary.to_string(index=False))
for name, frame in [("train", train), ("calibration", calib), ("validation", valid), ("test", test)]:
    if len(frame) < 200 or frame.delayed_15.nunique() != 2:
        raise ValueError(f"Bad split {name}: insufficient rows or missing class")
assert train.flight_date.max() < calib.flight_date.min()
assert calib.flight_date.max() < valid.flight_date.min()
assert valid.flight_date.max() < test.flight_date.min()
print("Sample total:", len(pdf), "| test:", len(test))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Rebuild Notebook 05's calibrated classifiers and train conditional severity
# MAGIC Classifier hyperparameters and preprocessing below mirror Notebook 05. Severity training uses only **train-period delayed flights**; the regression target is `min(dep_delay, 300)`. The conditional estimate is clipped to [15, 300] to respect the target definition. No post-departure predictors are used.

# COMMAND ----------

def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def make_prep(numeric):
    return ColumnTransformer([
        ("cat", Pipeline([("encoder", make_encoder())]), CATEGORICAL),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=False))]), numeric)
    ], remainder="drop")

def make_classifier(use_weather):
    num = OPERATIONAL_NUMERIC + (WEATHER_NUMERIC if use_weather else [])
    return Pipeline([("prep", make_prep(num)),
        ("model", XGBClassifier(n_estimators=260, max_depth=5, learning_rate=0.055,
               subsample=0.85, colsample_bytree=0.85,
               objective="binary:logistic", eval_metric="logloss",
               tree_method="hist", random_state=RANDOM_SEED, n_jobs=4))])

def make_severity_regressor(use_weather):
    num = OPERATIONAL_NUMERIC + (WEATHER_NUMERIC if use_weather else [])
    return Pipeline([("prep", make_prep(num)),
        ("model", XGBRegressor(n_estimators=260, max_depth=4, learning_rate=0.05,
               subsample=0.85, colsample_bytree=0.85, reg_lambda=3.0,
               objective="reg:squarederror", tree_method="hist",
               random_state=RANDOM_SEED, n_jobs=4))])

all_predictions = {}
model_objects = {}
train_late = train.loc[train.delayed_15 == 1].copy()
if len(train_late) < 1000:
    raise RuntimeError("Too few late flights to estimate conditional duration")
y_train_minutes = np.minimum(train_late.dep_delay.to_numpy(dtype=float), COST_CAP_MIN)
train_constant_minutes = float(y_train_minutes.mean())
print("Train-period delayed flights:", len(train_late))
print("Train-only constant severity benchmark (capped min):", round(train_constant_minutes, 2))

for name, add_weather in [("baseline", False), ("weather", True)]:
    cols = CATEGORICAL + OPERATIONAL_NUMERIC + (WEATHER_NUMERIC if add_weather else [])
    assert not ({"dep_delay", "delayed_15", "cancelled", "scheduled_departure_utc"} & set(cols))
    classifier = make_classifier(add_weather)
    classifier.fit(train[cols], train.delayed_15)
    raw_calib = classifier.predict_proba(calib[cols])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(raw_calib, calib.delayed_15.to_numpy())
    raw_test = classifier.predict_proba(test[cols])[:, 1]
    calibrated_test = calibrator.predict(raw_test)

    severity = make_severity_regressor(add_weather)
    severity.fit(train_late[cols], y_train_minutes)
    severity_test = np.clip(severity.predict(test[cols]), LATE_THRESHOLD_MIN, COST_CAP_MIN)
    severity_late_test = severity_test[test.delayed_15.to_numpy() == 1]

    all_predictions[name] = {
        "raw": raw_test, "prob": np.asarray(calibrated_test, dtype=float),
        "severity": np.asarray(severity_test, dtype=float),
        "expected_minutes": np.asarray(calibrated_test * severity_test, dtype=float)
    }
    model_objects[name] = (classifier, calibrator, severity)
    print(f"Trained {name}: classifier + calibrated probabilities + conditional severity regressor")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Validate 05 reproduction + severity metrics on untouched test
# MAGIC Test data are NOT used to tune model hyperparameters. Any mismatch against Notebook 05 must be investigated **before** Streamlit or saving outputs. If the persisted test population differs, there is no safe per-flight join to fall back on.

# COMMAND ----------

y_test = test.delayed_15.to_numpy(dtype=int)
d_test = test.dep_delay.to_numpy(dtype=float)
observed_capped = np.where(y_test == 1, np.minimum(d_test, COST_CAP_MIN), 0.0)
check_05_count = int(spark.table(PREDICTIONS_05_TABLE).count())
check_05_metrics = spark.table(METRICS_05_TABLE).select("model", "n_test", "roc_auc_raw", "pr_auc_raw").toPandas()
model_metrics_rows = []
parity_rows = []
for name in ("baseline", "weather"):
    scores = all_predictions[name]
    raw_roc = float(roc_auc_score(y_test, scores["raw"]))
    raw_pr = float(average_precision_score(y_test, scores["raw"]))
    late_mask = (y_test == 1)
    sev_mae = float(mean_absolute_error(observed_capped[late_mask], scores["severity"][late_mask]))
    sev_rmse = float(np.sqrt(mean_squared_error(observed_capped[late_mask], scores["severity"][late_mask])))
    constant_mae = float(mean_absolute_error(observed_capped[late_mask],
                            np.full(int(late_mask.sum()), train_constant_minutes)))
    expected_mae = float(mean_absolute_error(observed_capped, scores["expected_minutes"]))
    model_metrics_rows.append({
        "model": name, "n_test": int(len(test)), "late_test": int(late_mask.sum()),
        "roc_auc_raw": raw_roc, "pr_auc_raw": raw_pr,
        "brier_calibrated": float(brier_score_loss(y_test, scores["prob"])),
        "severity_mae_late_minutes": sev_mae,
        "severity_rmse_late_minutes": sev_rmse,
        "train_mean_severity_baseline_mae": constant_mae,
        "expected_minutes_mae_all_flights": expected_mae,
        "severity_beats_train_mean_mae": bool(sev_mae < constant_mae),
        "weather_is_retrospective_only": bool(name == "weather")
    })
    prior = check_05_metrics.loc[check_05_metrics.model == name]
    if len(prior) != 1:
        raise RuntimeError(f"Missing or duplicate 05 metrics for {name}")
    old = prior.iloc[0]
    parity_rows.append({
        "model": name, "test_count_05": int(old.n_test), "test_count_09": int(len(test)),
        "stored_test_rows_05": check_05_count,
        "roc_auc_05": float(old.roc_auc_raw), "roc_auc_09": raw_roc,
        "pr_auc_05": float(old.pr_auc_raw), "pr_auc_09": raw_pr,
        "count_pass": bool(len(test) == int(old.n_test) == check_05_count),
        "auc_pass": bool(abs(float(old.roc_auc_raw) - raw_roc) < 0.001 and
                         abs(float(old.pr_auc_raw) - raw_pr) < 0.001)
    })
model_metrics = pd.DataFrame(model_metrics_rows)
parity = pd.DataFrame(parity_rows)
print("=== Notebook 05 version/reproduction check ===")
print(parity.round(5).to_string(index=False))
print("=== Model 09 out-of-time evaluation ===")
print(model_metrics.round(4).to_string(index=False))
PARITY_05_OK = bool(parity["count_pass"].all() and parity["auc_pass"].all())
if not PARITY_05_OK:
    print("STOP BEFORE PUBLICATION: Notebook 05 parity failed. Do not join by row order or publish output.")
else:
    print("PASS: Notebook 05 sample count and predictive metrics reproduced within tolerance")

# COMMAND ----------


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np
import pandas as pd

# Solo vuelos del TEST con retraso >= 15 minutos
late_mask = (y_test == 1)
y_real = observed_capped[late_mask]

# Referencia: predecir siempre la media del entrenamiento
predictions = {
    "Media entrenamiento": np.full(
        len(y_real), train_constant_minutes
    ),
    "XGBoost Baseline": all_predictions["baseline"]["severity"][late_mask],
    "XGBoost Weather": all_predictions["weather"]["severity"][late_mask]
}

results = []

for name, y_pred in predictions.items():
    mse = mean_squared_error(y_real, y_pred)

    results.append({
        "Modelo": name,
        "MAE": mean_absolute_error(y_real, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_real, y_pred)
    })

metrics_df = pd.DataFrame(results)

display(metrics_df.round(4))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Retrospective decision comparison at fixed alert capacity
# MAGIC **Fair metric:** same flights, same alert capacity, held-out test; compare how many *actual capped delayed-departure minutes* are associated with the prioritized flights. A ranking can identify minutes without proving an intervention can prevent them.
# MAGIC
# MAGIC Strategies: `probability_only`, `severity_only`, `expected_minutes`. Since USD/min is constant, `expected_cost_usd` has exactly the **same rank** as `expected_minutes` and is NOT repeated as a fourth independent strategy.

# COMMAND ----------

def selected_indices(values, fraction):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite ranking scores")
    n = max(1, int(round(len(values) * fraction)))
    # Stable order for ties. Never use actual delay or observed cost to choose flights.
    return np.argsort(-values, kind="stable")[:n]

capacity_rows = []
for name in ("baseline", "weather"):
    scores = all_predictions[name]
    for strategy, rank_values in [
        ("probability_only", scores["prob"]),
        ("severity_only", scores["severity"]),
        ("expected_minutes", scores["expected_minutes"])
    ]:
        for fraction in CAPACITIES:
            ix = selected_indices(rank_values, fraction)
            capture_minutes = float(observed_capped[ix].sum())
            captured_late = int(y_test[ix].sum())
            capacity_rows.append({
                "model": name, "strategy": strategy, "capacity_fraction": float(fraction),
                "flights_selected": int(len(ix)), "late_flights_selected": captured_late,
                "late_precision_selected": float(captured_late / len(ix)),
                "observed_capped_minutes_selected": capture_minutes,
                "observed_capped_minutes_total": float(observed_capped.sum()),
                "captured_minutes_share": float(capture_minutes / observed_capped.sum()),
                "historical_exposure_proxy_in_selected_usd": float(capture_minutes * REFERENCE_USD_PER_MIN),
                "outcome_intervention_effect": "NOT MEASURED",
                "weather_available_at_decision_time": ("UNVERIFIED" if name == "weather" else "SCHEDULE_ONLY")
            })
capacity_eval = pd.DataFrame(capacity_rows)
# Calculate comparison against probability strategy at the same model and capacity.
base_lookup = {(r.model, float(r.capacity_fraction)): float(r.observed_capped_minutes_selected)
               for r in capacity_eval.itertuples() if r.strategy == "probability_only"}
capacity_eval["minutes_vs_probability_only"] = capacity_eval.apply(
    lambda r: float(r.observed_capped_minutes_selected - base_lookup[(r.model, float(r.capacity_fraction))]), axis=1)
capacity_eval["capture_lift_vs_probability_pct"] = capacity_eval.apply(
    lambda r: (100 * r.minutes_vs_probability_only / base_lookup[(r.model, float(r.capacity_fraction))]
               if base_lookup[(r.model, float(r.capacity_fraction))] > 0 else np.nan), axis=1)
print("=== Decision comparison: fixed alert capacity, out-of-time TEST ===")
print(capacity_eval.loc[capacity_eval.model == "baseline", [
    "strategy", "capacity_fraction", "flights_selected", "observed_capped_minutes_selected",
    "captured_minutes_share", "minutes_vs_probability_only", "capture_lift_vs_probability_pct"
]].round(4).to_string(index=False))
print("=== Weather sensitivity (RETROSPECTIVE only) ===")
print(capacity_eval.loc[capacity_eval.model == "weather", [
    "strategy", "capacity_fraction", "observed_capped_minutes_selected", "minutes_vs_probability_only"
]].round(4).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Day-block bootstrap on ranking improvement at 10% alert capacity
# MAGIC Resample test *days*, not individual flights; re-rank **within each bootstrap draw** at fixed 10% capacity. Confidence intervals reflect the evaluation sample and fixed fitted models, not uncertainty in future operational effects.

# COMMAND ----------

unique_days, day_codes = np.unique(test.flight_date.dt.strftime("%Y-%m-%d"), return_inverse=True)
index_by_day = [np.flatnonzero(day_codes == j) for j in range(len(unique_days))]
rng = np.random.default_rng(RANDOM_SEED + 9)
bootstrap_rows = []
for name in ("baseline", "weather"):
    sc = all_predictions[name]
    deltas = []
    total_deltas_fraction = []
    for iteration in range(BOOTSTRAP_ITERATIONS):
        days = rng.integers(0, len(index_by_day), size=len(index_by_day))
        sampled_ix = np.concatenate([index_by_day[j] for j in days])
        ix_prob_local = selected_indices(sc["prob"][sampled_ix], BOOTSTRAP_CAPACITY)
        ix_ev_local = selected_indices(sc["expected_minutes"][sampled_ix], BOOTSTRAP_CAPACITY)
        min_prob = float(observed_capped[sampled_ix[ix_prob_local]].sum())
        min_ev = float(observed_capped[sampled_ix[ix_ev_local]].sum())
        total = float(observed_capped[sampled_ix].sum())
        deltas.append(min_ev - min_prob)
        total_deltas_fraction.append((min_ev - min_prob) / total if total > 0 else np.nan)
    series = pd.Series(deltas, dtype="float64")
    fraction_series = pd.Series(total_deltas_fraction, dtype="float64").dropna()
    bootstrap_rows.append({
        "model": name, "capacity_fraction": float(BOOTSTRAP_CAPACITY),
        "iterations": int(BOOTSTRAP_ITERATIONS),
        "delta_selected_minutes_median": float(series.median()),
        "delta_selected_minutes_ci_low": float(series.quantile(0.025)),
        "delta_selected_minutes_ci_high": float(series.quantile(0.975)),
        "delta_capture_share_median": float(fraction_series.median()),
        "delta_capture_share_ci_low": float(fraction_series.quantile(0.025)),
        "delta_capture_share_ci_high": float(fraction_series.quantile(0.975)),
        "comparison": "expected_minutes minus probability_only; test day-block bootstrap",
        "causal_savings": False
    })
bootstrap_eval = pd.DataFrame(bootstrap_rows)
print("=== Bootstrap evidence for decision ranking ===")
print(bootstrap_eval.round(5).to_string(index=False))
print("Positive CI lower bound supports retrospective capture improvement; never call this realized savings.")

# COMMAND ----------


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

resultados = []

test_late = test.loc[test["delayed_15"] == 1].copy()

for nombre in ["baseline", "weather"]:
    regresor = model_objects[nombre][2]

    columnas = CATEGORICAL + OPERATIONAL_NUMERIC
    if nombre == "weather":
        columnas = columnas + WEATHER_NUMERIC

    for periodo, datos in [
        ("TRAIN", train_late),
        ("TEST", test_late)
    ]:
        y_real = np.clip(
            datos["dep_delay"].to_numpy(dtype=float),
            15, COST_CAP_MIN
        )

        y_pred = np.clip(
            regresor.predict(datos[columnas]),
            15, COST_CAP_MIN
        )

        resultados.append({
            "modelo": nombre,
            "periodo": periodo,
            "vuelos": len(datos),
            "MAE": mean_absolute_error(y_real, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
            "R2": r2_score(y_real, y_pred),
            "mediana_real": np.median(y_real),
            "p90_real": np.percentile(y_real, 90),
            "pct_180plus": 100 * np.mean(y_real >= 180)
        })

diagnostico = pd.DataFrame(resultados)

display(diagnostico.round(3))


# COMMAND ----------


from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

# Duraciones reales: solo retrasos >=15 min
y_train = np.clip(
    train_late["dep_delay"].to_numpy(dtype=float),
    15, COST_CAP_MIN
)

y_test = np.clip(
    test.loc[test["delayed_15"] == 1, "dep_delay"]
        .to_numpy(dtype=float),
    15, COST_CAP_MIN
)

# Mediana calculada SOLO con entrenamiento
mediana_train = np.median(y_train)
pred_mediana = np.full(len(y_test), mediana_train)

print("Mediana del entrenamiento:", mediana_train)

print("=== REFERENCIA CONSTANTE: MEDIANA ===")
print("MAE:", round(mean_absolute_error(y_test, pred_mediana), 4))
print("MSE:", round(mean_squared_error(y_test, pred_mediana), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred_mediana)), 4))
print("R2:", round(r2_score(y_test, pred_mediana), 4))

print("\n=== XGBOOST BASELINE ===")
print("MAE: 40.1756")
print("RMSE: 59.4723")

print("\n=== XGBOOST WEATHER ===")
print("MAE: 40.0712")
print("RMSE: 59.3547")


# COMMAND ----------


# ==========================================================
# GRID SEARCH — XGBOOST REGRESSOR
# Flight Delay Intelligence | Notebook 09
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# 1. Preparar datos: SOLO TRAIN y vuelos retrasados
assert "train" in globals(), (
    "Primero ejecuta las secciones 1 y 2 del Notebook 09"
)

datos = train.loc[
    train["delayed_15"] == 1
].copy()

datos["flight_date"] = pd.to_datetime(
    datos["flight_date"]
)

datos = datos.sort_values(
    "flight_date"
).reset_index(drop=True)

# Variables disponibles antes de la salida.
# No usamos dep_delay ni delayed_15 como predictores.
categoricas = ["origin", "dest", "airline"]

numericas = [
    "month",
    "day_of_week",
    "departure_hour",
    "is_weekend",
    "is_summer",
    "is_peak_hour"
]

columnas = categoricas + numericas

X = datos[columnas]

# Variable objetivo: duración limitada a 300 minutos
y = np.clip(
    datos["dep_delay"].to_numpy(dtype=float),
    15,
    300
)

print("Vuelos para entrenamiento:", len(datos))
print("Media de duración:", round(np.mean(y), 2))
print("Mediana de duración:", round(np.median(y), 2))

assert len(datos) == 38678, (
    "La muestra no coincide con el entrenamiento anterior"
)

# 2. Preprocesamiento
preprocesamiento = ColumnTransformer(
    transformers=[
        (
            "categoricas",
            OneHotEncoder(handle_unknown="ignore"),
            categoricas
        ),
        (
            "numericas",
            SimpleImputer(strategy="median"),
            numericas
        )
    ]
)

# 3. XGBoost Regressor
pipeline = Pipeline([
    ("prep", preprocesamiento),
    ("model", XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=2026,
        n_jobs=4
    ))
])

# 4. Validación temporal
# Cada modelo se entrena únicamente con fechas anteriores
# a su respectiva ventana de validación.

ventanas = [
    ("2024-09-01", "2024-11-01"),
    ("2025-01-01", "2025-03-01"),
    ("2025-04-01", "2025-06-01")
]

folds = []
mae_mediana = []

for inicio, fin in ventanas:

    inicio = pd.Timestamp(inicio)
    fin = pd.Timestamp(fin)

    idx_train = np.flatnonzero(
        datos["flight_date"] < inicio
    )

    idx_val = np.flatnonzero(
        (datos["flight_date"] >= inicio) &
        (datos["flight_date"] < fin)
    )

    assert len(idx_train) > 1000
    assert len(idx_val) > 100

    folds.append((idx_train, idx_val))

    # Referencia sencilla: mediana del entrenamiento
    mediana = np.median(y[idx_train])

    error_mediana = mean_absolute_error(
        y[idx_val],
        np.full(len(idx_val), mediana)
    )

    mae_mediana.append(error_mediana)

    print(
        f"Fold: train={len(idx_train)}, "
        f"validación={len(idx_val)}, "
        f"MAE mediana={error_mediana:.3f}"
    )

# 5. Hiperparámetros a evaluar
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.03, 0.08],
    "model__reg_lambda": [3.0, 12.0]
}

# 16 combinaciones x 3 folds = 48 evaluaciones
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=folds,
    refit=True,
    n_jobs=1,
    verbose=1,
    error_score="raise"
)

# 6. Ejecutar búsqueda
grid.fit(X, y)

# 7. Mostrar resultados
print("\n=== RESULTADOS GRID SEARCH ===")

print("\nMejores hiperparámetros:")
print(grid.best_params_)

print(
    "\nMejor MAE promedio en validación:",
    round(-grid.best_score_, 4)
)

print(
    "MAE promedio de mediana en validación:",
    round(np.mean(mae_mediana), 4)
)

# 8. Top 5 configuraciones
resultados = pd.DataFrame(grid.cv_results_)

resultados["MAE"] = -resultados["mean_test_score"]

top5 = resultados.sort_values(
    "MAE"
)[[
    "param_model__n_estimators",
    "param_model__max_depth",
    "param_model__learning_rate",
    "param_model__reg_lambda",
    "MAE"
]].head(5)

print("\n=== TOP 5 CONFIGURACIONES ===")
display(top5)


# COMMAND ----------


# ============================================================
# NOTEBOOK 09 — MODEL COMPARISON
# XGBoost vs Random Forest vs CatBoost vs Ensemble
# ============================================================

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ------------------------------------------------------------
# 1. Verificar datos y preparar muestras
# ------------------------------------------------------------

assert "train" in globals(), "Ejecuta la sección 2 del Notebook 09"
assert "valid" in globals(), "Ejecuta la sección 2 del Notebook 09"

assert train["flight_date"].max() < valid["flight_date"].min()

train_late_cmp = train.loc[
    train["delayed_15"] == 1
].copy()

valid_late_cmp = valid.loc[
    valid["delayed_15"] == 1
].copy()

categoricas = ["origin", "dest", "airline"]

numericas = [
    "month",
    "day_of_week",
    "departure_hour",
    "is_weekend",
    "is_summer",
    "is_peak_hour"
]

columnas = categoricas + numericas

X_train_cmp = train_late_cmp[columnas].copy()
X_valid_cmp = valid_late_cmp[columnas].copy()

for df_cmp in [X_train_cmp, X_valid_cmp]:
    for c in categoricas:
        df_cmp[c] = df_cmp[c].fillna("missing").astype(str)
    for c in numericas:
        df_cmp[c] = pd.to_numeric(df_cmp[c], errors="coerce")

y_train_cmp = np.clip(
    train_late_cmp["dep_delay"].to_numpy(dtype=float),
    15, 300
)

y_valid_cmp = np.clip(
    valid_late_cmp["dep_delay"].to_numpy(dtype=float),
    15, 300
)

print("TRAIN:", len(y_train_cmp))
print("VALIDATION:", len(y_valid_cmp))

# ------------------------------------------------------------
# 2. Preprocesamiento para XGBoost y Random Forest
# ------------------------------------------------------------

try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True
    )
except TypeError:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=True
    )

def crear_preprocesamiento():
    return ColumnTransformer([
        ("categoricas", OneHotEncoder(
            handle_unknown="ignore"
        ), categoricas),
        ("numericas", SimpleImputer(
            strategy="median"
        ), numericas)
    ])

# ------------------------------------------------------------
# 3. XGBoost optimizado con GridSearch anterior
# ------------------------------------------------------------

modelo_xgb = Pipeline([
    ("prep", crear_preprocesamiento()),
    ("model", XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.08,
        reg_lambda=12.0,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=2026,
        n_jobs=4
    ))
])

# ------------------------------------------------------------
# 4. Random Forest
# ------------------------------------------------------------

modelo_rf = Pipeline([
    ("prep", crear_preprocesamiento()),
    ("model", RandomForestRegressor(
        n_estimators=250,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=8,
        max_features=0.75,
        random_state=2026,
        n_jobs=4
    ))
])

# ------------------------------------------------------------
# 5. CatBoost
# Maneja variables categóricas directamente.
# ------------------------------------------------------------

modelo_cat = CatBoostRegressor(
    iterations=450,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=5,
    loss_function="RMSE",
    cat_features=categoricas,
    random_seed=2026,
    thread_count=4,
    verbose=False,
    allow_writing_files=False
)

# ------------------------------------------------------------
# 6. Entrenar
# ------------------------------------------------------------

print("\nEntrenando XGBoost...")
modelo_xgb.fit(X_train_cmp, y_train_cmp)

print("Entrenando Random Forest...")
modelo_rf.fit(X_train_cmp, y_train_cmp)

print("Entrenando CatBoost...")
modelo_cat.fit(X_train_cmp, y_train_cmp)

print("Entrenamiento completado.")

# ------------------------------------------------------------
# 7. Predicciones sobre validación temporal
# ------------------------------------------------------------

predicciones_cmp = {
    "Mediana constante": np.full(
        len(y_valid_cmp),
        np.median(y_train_cmp)
    ),
    "XGBoost optimizado": modelo_xgb.predict(X_valid_cmp),
    "Random Forest": modelo_rf.predict(X_valid_cmp),
    "CatBoost": modelo_cat.predict(X_valid_cmp)
}

# Limitar a la definición de retraso significativo
for nombre in predicciones_cmp:
    predicciones_cmp[nombre] = np.clip(
        predicciones_cmp[nombre],
        15, 300
    )

# ------------------------------------------------------------
# 8. Ensemble: promedio de los tres regresores
# Sin ajustar pesos con el TEST.
# ------------------------------------------------------------

predicciones_cmp["Ensemble"] = (
    predicciones_cmp["XGBoost optimizado"] +
    predicciones_cmp["Random Forest"] +
    predicciones_cmp["CatBoost"]
) / 3

# ------------------------------------------------------------
# 9. Comparación de métricas
# ------------------------------------------------------------

resultados_cmp = []

for nombre, pred in predicciones_cmp.items():

    mae = mean_absolute_error(y_valid_cmp, pred)
    mse = mean_squared_error(y_valid_cmp, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_valid_cmp, pred)

    resultados_cmp.append({
        "Modelo": nombre,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

comparacion_modelos = pd.DataFrame(resultados_cmp)

print("\n=== COMPARACION DE MODELOS — VALIDACION ===")

display(
    comparacion_modelos
    .sort_values("MAE")
    .round(4)
)

print("\n=== MEJOR MODELO POR METRICA ===")

for metrica in ["MAE", "RMSE", "R2"]:
    if metrica == "R2":
        mejor = comparacion_modelos.loc[
            comparacion_modelos[metrica].idxmax()
        ]
    else:
        mejor = comparacion_modelos.loc[
            comparacion_modelos[metrica].idxmin()
        ]

    print(
        f"{metrica}: {mejor['Modelo']} "
        f"({mejor[metrica]:.4f})"
    )


# COMMAND ----------


# =====================================================
# DIAGNÓSTICO DE ERRORES POR GRAVEDAD DEL RETRASO
# XGBoost vs CatBoost vs Ensemble
# =====================================================

import numpy as np
import pandas as pd

resultados = []

modelos = [
    "XGBoost optimizado",
    "CatBoost",
    "Ensemble"
]

for nombre in modelos:

    df = pd.DataFrame({
        "real": y_valid_cmp,
        "predicho": predicciones_cmp[nombre]
    })

    df["grupo"] = pd.cut(
        df["real"],
        bins=[14, 30, 60, 180, 301],
        labels=[
            "15-29 min",
            "30-59 min",
            "60-179 min",
            "180-300 min"
        ],
        right=False
    )

    df["error_absoluto"] = abs(
        df["real"] - df["predicho"]
    )

    df["sesgo"] = df["predicho"] - df["real"]

    resumen = (
        df.groupby("grupo", observed=True)
        .agg(
            vuelos=("real", "size"),
            duracion_real=("real", "mean"),
            duracion_predicha=("predicho", "mean"),
            MAE=("error_absoluto", "mean"),
            sesgo=("sesgo", "mean")
        )
        .reset_index()
    )

    resumen.insert(0, "modelo", nombre)
    resultados.append(resumen)

diagnostico_gravedad = pd.concat(
    resultados,
    ignore_index=True
)

display(diagnostico_gravedad.round(2))


# COMMAND ----------


import pandas as pd

columnas_revision = [
    "month",
    "is_summer",
    "day_of_week",
    "airline",
    "origin",
    "dest"
]

resultado = pd.DataFrame({
    "variable": columnas_revision,
    "valores_unicos_train": [
        X_train_cmp[c].nunique(dropna=False)
        for c in columnas_revision
    ],
    "valores_unicos_validacion": [
        X_valid_cmp[c].nunique(dropna=False)
        for c in columnas_revision
    ]
})

display(resultado)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Priority audit by airport and airline (TEST only)
# MAGIC Airport/carrier groups are **observed slices** of the selected top 10%; this is a retrospective audit, not a recommendation that a named airline is at fault. We use the **schedule-only** model for operational prioritization.

# COMMAND ----------

scored = test[["flight_date", "origin", "dest", "airline", "crs_dep_time", "delayed_15", "dep_delay"]].copy().reset_index(drop=True)
scored["flight_date"] = scored["flight_date"].dt.strftime("%Y-%m-%d")
scored["capped_late_minutes_actual"] = observed_capped.astype(float)
for name in ("baseline", "weather"):
    for key, data in all_predictions[name].items():
        scored[f"{name}_{key}"] = data.astype(float)
    scored[f"{name}_expected_cost_proxy_usd"] = all_predictions[name]["expected_minutes"] * REFERENCE_USD_PER_MIN
selected_ten_pct = np.zeros(len(scored), dtype=bool)
selected_ten_pct[selected_indices(all_predictions["baseline"]["expected_minutes"], 0.10)] = True
scored["priority_top10_baseline"] = selected_ten_pct.astype(int)
# Fully aggregated flight segments. Avoid outputting a claim about flight identity uniqueness.
segment_rows = []
for segment_type in ("origin", "airline"):
    groups = scored.groupby(segment_type, dropna=False)
    for segment_name, part in groups:
        if len(part) < 100:
            continue
        chosen = part.loc[part.priority_top10_baseline == 1]
        segment_rows.append({
            "segment_type": segment_type, "segment_name": str(segment_name),
            "test_flights": int(len(part)), "test_late_rate": float(part.delayed_15.mean()),
            "selected_for_review": int(len(chosen)),
            "observed_capped_delay_minutes_in_selected": float(chosen.capped_late_minutes_actual.sum()),
            "predicted_expected_delay_minutes_in_selected": float(chosen.baseline_expected_minutes.sum()),
            "observed_capped_delay_minutes_all": float(part.capped_late_minutes_actual.sum()),
            "not_causal_or_proven_preventable": True
        })
segment_priority = pd.DataFrame(segment_rows)
print("=== Operational review candidates by origin (retrospective test) ===")
print(segment_priority.query("segment_type == 'origin'").sort_values(
    "observed_capped_delay_minutes_in_selected", ascending=False).head(12).round(3).to_string(index=False))
print("=== Operational review candidates by airline ===")
print(segment_priority.query("segment_type == 'airline'").sort_values(
    "observed_capped_delay_minutes_in_selected", ascending=False).head(12).round(3).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Invariants, recommendations and limitations
# MAGIC A severity model is justified only if it improves either late-only duration prediction or observed decision capture versus reasonable baselines; if not, report a negative finding. Operational use requires forecasts, a well-defined intervention, capacity constraints and measurement of causal effect.

# COMMAND ----------

assert len(scored) == len(test)
assert np.all((observed_capped >= 0) & (observed_capped <= COST_CAP_MIN))
assert (scored["priority_top10_baseline"].sum() == max(1, round(len(scored) * 0.10)))
assert len(capacity_eval) == 2 * 3 * len(CAPACITIES)
assert len(bootstrap_eval) == 2
assert all(np.isfinite(all_predictions[n]["expected_minutes"]).all() for n in ("baseline", "weather"))
assert all(np.all((all_predictions[n]["prob"] >= 0) & (all_predictions[n]["prob"] <= 1))
           for n in ("baseline", "weather"))
assert all(np.all((all_predictions[n]["severity"] >= LATE_THRESHOLD_MIN) &
                     (all_predictions[n]["severity"] <= COST_CAP_MIN))
           for n in ("baseline", "weather"))
assert all(np.all(all_predictions[n]["expected_minutes"] <= COST_CAP_MIN + 1e-9)
           for n in ("baseline", "weather"))
for name in ("baseline", "weather"):
    for cap in CAPACITIES:
        slices = capacity_eval[(capacity_eval.model == name) & (capacity_eval.capacity_fraction == cap)]
        assert slices.flights_selected.nunique() == 1
        assert slices.observed_capped_minutes_selected.max() <= observed_capped.sum() + 1e-7
print("PASS: delay labels, chronological split, score bounds, capacity, and financial units")
print("PASS: no observed departure delay used as a model feature")
print("Notebook 05 metric parity:", "PASS" if PARITY_05_OK else "FAIL - do not publish")
for name in ("baseline", "weather"):
    cap = capacity_eval[(capacity_eval.model == name) &
                        (capacity_eval.capacity_fraction == BOOTSTRAP_CAPACITY)]
    prob = cap.loc[cap.strategy == "probability_only"].iloc[0]
    expected = cap.loc[cap.strategy == "expected_minutes"].iloc[0]
    print(f"{name} @ {int(BOOTSTRAP_CAPACITY*100)}%: probability captures {prob.observed_capped_minutes_selected:,.0f} min; "
          f"expected-minutes captures {expected.observed_capped_minutes_selected:,.0f} min; "
          f"difference {expected.minutes_vs_probability_only:+,.0f} min (historical identification only)")
print("NO PROVEN SAVINGS. Conditional severity and monetary prioritization are model-assisted scenarios.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Output documentation and controlled Gold persistence
# MAGIC **New only:** `part3_decision_model_metrics`, `part3_decision_model_parity`, `part3_decision_capacity`, `part3_decision_bootstrap`, `part3_decision_test_predictions`, `part3_decision_segment_priority`, `part3_decision_assumptions`.
# MAGIC
# MAGIC Keep `SAVE_TABLES=False` first. Inspect metrics and bootstrap CI. Then explicitly change to True, run the configuration cell and **only this persistence cell**. This overwrites only the seven `part3_decision_*` tables, not Notebook 05 or 08 outputs.

# COMMAND ----------

assumption_rows = [
    ("source", GOLD_TABLE, "Same analytical population / deterministic sample rule as Notebook 05"),
    ("source_population", "selected_30_airport_gold_weather_subset", "Not all US airlines or flights"),
    ("source_uniqueness", "not_fully_audited", "Counts are source rows; no safe unique flight ID available"),
    ("sampling", "3_percent_xxhash64_05", "Selection reproduced, no row-order join to 05 predictions"),
    ("splits", "train_to_2025_05;calib_2025_06_07;valid_2025_08;test_2025_09_12", "Chronological; test used only to evaluate"),
    ("target", "dep_delay_ge_15", "Pre-departure 15+ minute binary risk"),
    ("severity_target", "min(dep_delay,300)_conditional_on_dep_delay_ge_15", "Only late training flights; target excluded from predictors"),
    ("primary_predictors", "schedule_only", "Origin, destination, airline, calendar, scheduled departure time variables"),
    ("weather_predictors", "retrospective_sensitivity_only", "Observed historical weather availability at decision timestamp unverified"),
    ("financial_rate", str(REFERENCE_USD_PER_MIN), "08 hypothetical at-gate USD/min proxy; uniform rate means identical cost/minute ranking"),
    ("financial_cap", str(COST_CAP_MIN), "Sensitivity scenario; not complete IATA methodology"),
    ("operational_intervention", "not_observed", "No real interventions, effectiveness estimates or operating costs in source"),
    ("causal_savings", "not_estimated", "Captured historical delay minutes are not avoidable or realized savings"),
    ("05_reproduction_pass", str(PARITY_05_OK), "Do not publish if persisted Notebook 05 metrics do not match"),
    ("model_artifacts", "not_persisted", "Notebook rebuilds models; deployment requires versioned model registry"),
    ("decision_timing", "pre_departure_horizon_unspecified", "Define horizon and verify all input availability before operational deployment"),
]
assumptions = pd.DataFrame(assumption_rows, columns=["parameter", "setting", "explanation"])

# Dataframe export only uses the 3% TEST sample; no source Gold overwrites.
def dataframe_for_spark(frame):
    out = frame.copy().replace([np.inf, -np.inf], np.nan)
    out = out.astype(object).where(pd.notna(out), None)
    return spark.createDataFrame(out)

def persist(frame, table_name):
    full_name = f"{OUTPUT_SCHEMA}.{table_name}"
    if not SAVE_TABLES:
        print(f"SKIPPED {full_name} (SAVE_TABLES=False)")
        return
    if frame.empty:
        raise RuntimeError(f"Refusing to write an empty table: {full_name}")
    (dataframe_for_spark(frame).write.format("delta")
       .mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(full_name))
    print(f"SAVED {full_name}: {len(frame):,} rows")

if SAVE_TABLES:
    if not PARITY_05_OK:
        raise RuntimeError("Cannot save: Notebook 05 reproduction failed. Investigate sample/model version.")
    if not spark.catalog.tableExists(FINANCIAL_08_TABLE):
        raise RuntimeError("Cannot save: Notebook 08 assumption table missing")

outputs = [
    (model_metrics, "part3_decision_model_metrics"),
    (parity, "part3_decision_model_parity"),
    (capacity_eval, "part3_decision_capacity"),
    (bootstrap_eval, "part3_decision_bootstrap"),
    (scored, "part3_decision_test_predictions"),
    (segment_priority, "part3_decision_segment_priority"),
    (assumptions, "part3_decision_assumptions"),
]
for frame, name in outputs:
    persist(frame, name)
print("Done. Results are retrospective analytics, NOT a validated live flight dispatch system.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Reporting checklist
# MAGIC - Compare **baseline schedule-only** to baseline schedule-only with severity ranking. Do not claim weather scores can be used 24h early until an appropriate forecast is available and time-aligned.
# MAGIC - Report severity MAE/RMSE against a **training-only mean** predictor and both PR/ROC parity checks against Notebook 05.
# MAGIC - Compare minutes *captured* at 5/10/20% alert capacity on the untouched test. An increase is a **better ranking of observed delay burden**, not airline savings.
# MAGIC - Include day-block bootstrap CI at 10% alert capacity. If it overlaps zero, the ranking improvement remains uncertain.
# MAGIC - Reference Notebook 08 for hypothetical monetary cost and capped minutes; **cost = expected minutes × same constant USD/min**, so cost ranking has the same flight order as expected-minute ranking.
# MAGIC - Do **not** join 05 test predictions row by row or claim calibrated classifier model artifacts were saved. Save and version model artifacts separately before a real app makes live inferences.
# MAGIC - Do not conflate Notebook 06 historical propagation associations with the separate Monte Carlo prototype. They require independent scopes, assumptions, and validation.