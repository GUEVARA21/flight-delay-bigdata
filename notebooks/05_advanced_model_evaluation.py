# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "xgboost==3.0.5",
# ]
# ///
# MAGIC %md
# MAGIC # 05 · Model reliability & decision intelligence
# MAGIC **Flight Delay Big Data — third deliverable extension.**
# MAGIC
# MAGIC Enhances the existing XGBoost comparison without changing the prediction target. Builds a *four-stage chronological evaluation*, isotonic calibration, validation-only decision thresholds, decision-cost scenarios, airport/airline robustness checks and paired day-block bootstrap confidence intervals.
# MAGIC
# MAGIC The notebook assumes Unity Catalog tables already built in the project's Gold layer. It does **not** claim results until executed on actual data. Do not include post-departure fields as predictors.

# COMMAND ----------

# MAGIC %pip install xgboost==3.0.5

# COMMAND ----------


import xgboost
print("XGBoost instalado:", xgboost.__version__)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Configuration and integrity checks

# COMMAND ----------

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier

GOLD_TABLE = "workspace.gold_flights.flights_weather_features"
OUTPUT_SCHEMA = "workspace.gold_flights"
SAVE_TABLES = False   # set False if permission to create Gold tables is unavailable
SAMPLE_PERCENT = 3.0  # target sample near the ~258k rows of the repository
RANDOM_SEED = 2026
BOOTSTRAP_ITERATIONS = 250
FALSE_ALERT_COST = 1.0     # illustrative units; NOT observed economic costs
MISSED_DELAY_COST = 5.0    # change to reflect a stakeholder-specific scenario

CATEGORICAL = ["origin", "dest", "airline"]
OPERATIONAL_NUMERIC = ["month", "day_of_week", "departure_hour", "is_weekend", "is_summer", "is_peak_hour"]
WEATHER_NUMERIC = ["temperature_2m", "precipitation", "wind_speed_10m", "relative_humidity_2m", "cloud_cover", "pressure_msl"]
FEATURES_ALL = CATEGORICAL + OPERATIONAL_NUMERIC + WEATHER_NUMERIC
REQUIRED_COLUMNS = ["flight_date", "delayed_15", "crs_dep_time"] + FEATURES_ALL

assert 0 < SAMPLE_PERCENT <= 100
assert MISSED_DELAY_COST > 0 and FALSE_ALERT_COST > 0
assert BOOTSTRAP_ITERATIONS >= 30
assert spark.catalog.tableExists(GOLD_TABLE), f"No existe {GOLD_TABLE}. Ejecuta antes Bronze/Silver/Gold."
source = spark.table(GOLD_TABLE)
missing = sorted(set(REQUIRED_COLUMNS) - set(source.columns))
if missing:
    raise ValueError(f"Faltan columnas en Gold: {missing}; revisar docs/data_dictionary.md")
print(f"Fuente: {GOLD_TABLE}")
print(f"Filas reportadas por el proyecto en Gold: ~8.65M. Muestra configurada: {SAMPLE_PERCENT}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Reproducible sample and time-based split
# MAGIC **Train:** Feb 2024–May 2025 | **Calibration:** June–July 2025 |
# MAGIC **Threshold selection:** August 2025 | **Untouched test:** Sept–Dec 2025.
# MAGIC
# MAGIC The sample rule is deterministic; it is NOT exactly the original authors' sample rule. All models below use the same selected records.

# COMMAND ----------

working = (
    source
    .select(*REQUIRED_COLUMNS)
    .withColumn("flight_date", F.to_date("flight_date"))
    .filter((F.col("flight_date") >= F.lit("2024-02-01")) & (F.col("flight_date") < F.lit("2026-01-01")))
    .filter(F.col("delayed_15").isin(0, 1))
)
for c in CATEGORICAL:
    working = working.filter(F.col(c).isNotNull())
working = working.filter(F.col("crs_dep_time").isNotNull())

# Stable hash: no random order, no full collect of the 8.65M rows.
keys = [F.coalesce(F.col(x).cast("string"), F.lit("<NULL>")) for x in
        ["flight_date", "origin", "dest", "airline", "crs_dep_time"]]
sample = working.filter(F.pmod(F.xxhash64(*keys), F.lit(10000)) < F.lit(round(SAMPLE_PERCENT * 100)))
pdf = sample.toPandas()  # only ~3%; do not collect the 8.65M-row Gold table
if len(pdf) < 10000:
    raise ValueError("La muestra es muy pequeña. Aumenta SAMPLE_PERCENT o revisa el origen.")
pdf["flight_date"] = pd.to_datetime(pdf["flight_date"])
pdf["delayed_15"] = pdf["delayed_15"].astype("int8")
for c in CATEGORICAL:
    pdf[c] = pdf[c].astype(str)
for c in OPERATIONAL_NUMERIC + WEATHER_NUMERIC:
    pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

cut_train = pd.Timestamp("2025-06-01")
cut_calib = pd.Timestamp("2025-08-01")
cut_valid = pd.Timestamp("2025-09-01")
train = pdf.loc[pdf.flight_date < cut_train].copy()
calib = pdf.loc[(pdf.flight_date >= cut_train) & (pdf.flight_date < cut_calib)].copy()
valid = pdf.loc[(pdf.flight_date >= cut_calib) & (pdf.flight_date < cut_valid)].copy()
test = pdf.loc[pdf.flight_date >= cut_valid].copy()
summary = pd.DataFrame([
    {"split": k, "rows": len(d), "delay_rate": d.delayed_15.mean()}
    for k, d in [("train", train), ("calibration", calib), ("threshold_valid", valid), ("test", test)]
])
print(summary.to_string(index=False))
for k, d in [("train",train), ("calibration",calib), ("threshold_valid",valid), ("test",test)]:
    if len(d) < 200 or d.delayed_15.nunique() != 2:
        raise ValueError(f"Split {k} no tiene suficientes filas/clases. Revisa fechas y muestreo.")
assert train.flight_date.max() < calib.flight_date.min()
assert calib.flight_date.max() < valid.flight_date.min()
assert valid.flight_date.max() < test.flight_date.min()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Fair experiment: operational vs. operational + weather
# MAGIC Fit transformations ONLY on train. Avoid actual delay, cancellations, live flight data or other post-departure information as inputs.

# COMMAND ----------

def build_pipeline(add_weather):
    numeric = OPERATIONAL_NUMERIC + (WEATHER_NUMERIC if add_weather else [])
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # compatible with older scikit-learn runtimes
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    prep = ColumnTransformer([
        ("cat", Pipeline([("encoder", encoder)]), CATEGORICAL),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=False))]), numeric)
    ], remainder="drop")
    classifier = XGBClassifier(
        n_estimators=260, max_depth=5, learning_rate=0.055,
        subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=RANDOM_SEED, n_jobs=4
    )
    return Pipeline([("prep", prep), ("model", classifier)])

models = {}
raw_scores = {}
cal_scores = {}
for key, use_weather in [("baseline", False), ("weather", True)]:
    columns = CATEGORICAL + OPERATIONAL_NUMERIC + (WEATHER_NUMERIC if use_weather else [])
    assert not ({"dep_delay", "delayed_15", "scheduled_departure_utc"} & set(columns))
    pipeline = build_pipeline(use_weather)
    pipeline.fit(train[columns], train.delayed_15)
    scores = {name: pipeline.predict_proba(part[columns])[:, 1]
              for name, part in [("calib",calib),("valid",valid),("test",test)]}
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    isotonic.fit(scores["calib"], calib.delayed_15.to_numpy())
    models[key] = (pipeline, isotonic)
    raw_scores[key] = scores
    cal_scores[key] = {name: isotonic.predict(v) for name, v in scores.items()}
    print(f"{key}: trained on {len(train):,} flights; calibrated on {len(calib):,} flights")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Thresholds chosen BEFORE looking at test labels
# MAGIC Optimization is evaluated on August 2025, once, and then frozen. Cost values are explicitly a hypothetical decision scenario.

# COMMAND ----------

def select_threshold(y, prob, missed_cost=5.0, false_alert_cost=1.0):
    thresholds = np.unique(np.r_[0.0, np.linspace(0.01, 0.99, 99), 1.0])
    rows = []
    y = np.asarray(y).astype(int)
    for t in thresholds:
        pred = np.asarray(prob) >= t
        fn = int(((y == 1) & (~pred)).sum())
        fp = int(((y == 0) & pred).sum())
        rows.append({"threshold": float(t), "cost": fn * missed_cost + fp * false_alert_cost,
                     "recall": recall_score(y, pred, zero_division=0),
                     "precision": precision_score(y, pred, zero_division=0),
                     "alert_share": float(pred.mean())})
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["cost", "threshold"], ascending=[True,False]).iloc[0]
    return float(best.threshold), frame

thresholds, tradeoff_tables = {}, []
for name in models:
    t, curve = select_threshold(valid.delayed_15, cal_scores[name]["valid"],
                                MISSED_DELAY_COST, FALSE_ALERT_COST)
    thresholds[name] = t
    curve["model"] = name
    tradeoff_tables.append(curve)
print("Decision thresholds chosen in August 2025:", thresholds)
tradeoffs = pd.concat(tradeoff_tables, ignore_index=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Independent test: ROC-AUC, PR-AUC, calibration, cost and operational coverage

# COMMAND ----------

def ece_binary(y, probs, n_bins=10):
    y, probs = np.asarray(y), np.asarray(probs)
    bins = np.linspace(0, 1, n_bins + 1)
    membership = np.minimum(np.digitize(probs, bins[1:-1]), n_bins - 1)
    ece = 0.0
    for j in range(n_bins):
        selected = membership == j
        if selected.any():
            ece += selected.mean() * abs(y[selected].mean() - probs[selected].mean())
    return float(ece)

def model_metrics(y, raw, calibrated, threshold):
    y = np.asarray(y).astype(int)
    pred = np.asarray(calibrated) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc_raw": float(roc_auc_score(y, raw)),
        "pr_auc_raw": float(average_precision_score(y, raw)),
        "brier_calibrated": float(brier_score_loss(y, calibrated)),
        "ece10_calibrated": ece_binary(y, calibrated),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alerts_pct": float(100 * pred.mean()),
        "cost_per_1000_flights": float((MISSED_DELAY_COST * fn + FALSE_ALERT_COST * fp) * 1000 / len(y)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": float(threshold), "n_test": len(y)
    }

y_test = test.delayed_15.to_numpy()
records = []
for name in models:
    record = model_metrics(y_test, raw_scores[name]["test"], cal_scores[name]["test"], thresholds[name])
    record["model"] = name
    records.append(record)
metrics = pd.DataFrame(records)
print(metrics.round(4).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Is the weather improvement robust? Paired block bootstrap by flight date
# MAGIC All resamples preserve the same flights for both models. Daily blocks reflect shared airport/day shocks better than independently resampling individual flights. The confidence intervals are descriptive, conditional on this trained-model/test-window design.

# COMMAND ----------

def paired_day_bootstrap(frame, scores_a, scores_b, iterations=250, seed=2026):
    dates = frame.flight_date.dt.strftime("%Y-%m-%d").to_numpy()
    labels = frame.delayed_15.to_numpy().astype(int)
    unique_days, day_codes = np.unique(dates, return_inverse=True)
    day_rows = [np.flatnonzero(day_codes == j) for j in range(len(unique_days))]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(iterations):
        sampled_days = rng.integers(0, len(unique_days), size=len(unique_days))
        inds = np.concatenate([day_rows[j] for j in sampled_days])
        if np.unique(labels[inds]).size < 2:
            continue
        roc_diff = roc_auc_score(labels[inds], scores_b[inds]) - roc_auc_score(labels[inds], scores_a[inds])
        pr_diff = average_precision_score(labels[inds], scores_b[inds]) - average_precision_score(labels[inds], scores_a[inds])
        rows.append({"delta_roc_auc": roc_diff, "delta_pr_auc": pr_diff})
    if len(rows) < 30:
        raise RuntimeError("No hay suficientes bootstrap samples válidos")
    return pd.DataFrame(rows)

boots = paired_day_bootstrap(test, raw_scores["baseline"]["test"], raw_scores["weather"]["test"],
                             BOOTSTRAP_ITERATIONS, RANDOM_SEED)
ci_summary = pd.DataFrame([
    {"metric": metric, "observed_delta": float(metrics.loc[metrics.model=="weather",metric].iloc[0] - metrics.loc[metrics.model=="baseline",metric].iloc[0]),
     "ci_low": float(boots[boot_col].quantile(0.025)), "ci_high": float(boots[boot_col].quantile(0.975)),
     "bootstrap_resamples": len(boots)}
    for metric, boot_col in [("roc_auc_raw","delta_roc_auc"),("pr_auc_raw","delta_pr_auc")]
])
print(ci_summary.to_string(index=False))
print("If a 95% interval crosses zero, the improvement is uncertain under this bootstrap design.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Reliability by airport and airline, monthly drift, and top-decile alert capture
# MAGIC Look for segments where overall average metrics hide performance degradation. We do not interpret these group differences as causal.

# COMMAND ----------

def subgroup_audit(frame, group_column, min_flights=150):
    rows = []
    y = frame.delayed_15.to_numpy()
    for group_name, part in frame.groupby(group_column, dropna=False):
        if len(part) < min_flights:
            continue
        pred = part.prob_weather.to_numpy() >= thresholds["weather"]
        result = {
            "group_type": group_column, "group_name": str(group_name),
            "flights": len(part), "actual_delay_rate": float(part.delayed_15.mean()),
            "mean_prediction": float(part.prob_weather.mean()),
            "recall": float(recall_score(part.delayed_15, pred, zero_division=0)),
            "precision": float(precision_score(part.delayed_15, pred, zero_division=0)),
            "brier": float(brier_score_loss(part.delayed_15, part.prob_weather))
        }
        if part.delayed_15.nunique() == 2:
            result["roc_auc"] = float(roc_auc_score(part.delayed_15, part.prob_weather_raw))
        else:
            result["roc_auc"] = np.nan
        rows.append(result)
    return pd.DataFrame(rows)

scored = test[["flight_date", "origin", "dest", "airline", "delayed_15"]].copy()
for name in models:
    scored[f"prob_{name}"] = cal_scores[name]["test"]
    scored[f"prob_{name}_raw"] = raw_scores[name]["test"]
scored["month"] = scored.flight_date.dt.to_period("M").astype(str)
airport_audit = subgroup_audit(scored, "origin", min_flights=150)
airline_audit = subgroup_audit(scored, "airline", min_flights=150)
monthly_audit = subgroup_audit(scored, "month", min_flights=150)
subgroups = pd.concat([airport_audit, airline_audit, monthly_audit], ignore_index=True)
print("Worst airport groups by Brier score (minimum group size applied):")
print(airport_audit.sort_values("brier", ascending=False).head(12).round(4).to_string(index=False))

# Prioritization within each complete test batch, no test label is used to rank flights.
# Thresholds here are percentiles of model scores IN EACH TEST BATCH, not fitted with test labels.
capacity_rows = []
for name in models:
    for fraction in [0.05, 0.10, 0.20, 0.30]:
        n_top = max(1, round(len(scored) * fraction))
        idx = np.argpartition(scored[f"prob_{name}"].to_numpy(), -n_top)[-n_top:]
        captured = int(scored.delayed_15.to_numpy()[idx].sum())
        total_late = int(scored.delayed_15.sum())
        capacity_rows.append({"model": name, "alert_capacity_fraction": fraction,
                              "flights_alerted": int(n_top), "delays_captured": captured,
                              "delay_recall_at_capacity": captured / total_late if total_late else np.nan,
                              "precision_at_capacity": captured / n_top})
capacity = pd.DataFrame(capacity_rows)
print("Retrospective ranked-capacity analysis:")
print(capacity.round(4).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Decision-quality figures

# COMMAND ----------

fig, axs = plt.subplots(1, 3, figsize=(17, 4))
for name in models:
    scores = scored[f"prob_{name}"].to_numpy()
    edges = np.linspace(0, 1, 11)
    bucket = np.minimum(np.digitize(scores, edges[1:-1]), 9)
    rows = [(scores[bucket==k].mean(), y_test[bucket==k].mean())
            for k in range(10) if (bucket==k).sum() >= 25]
    if rows:
        v = np.asarray(rows)
        axs[0].plot(v[:,0],v[:,1],marker="o",label=name)
axs[0].plot([0,1],[0,1],linestyle="--",color="gray",label="perfect")
axs[0].set(xlabel="Predicted delay probability",ylabel="Observed delay rate",title="Calibration | test 2025")
axs[0].legend()
for name in models:
    part = capacity.loc[capacity.model == name]
    axs[1].plot(part.alert_capacity_fraction * 100,part.delay_recall_at_capacity * 100,marker="o",label=name)
axs[1].set(xlabel="Flights alerted (%)",ylabel="Actual delayed flights captured (%)",title="Alert capacity tradeoff")
axs[1].legend()
if not monthly_audit.empty:
    axs[2].plot(monthly_audit.group_name,monthly_audit.actual_delay_rate * 100,marker="o",label="observed")
    axs[2].plot(monthly_audit.group_name,monthly_audit.mean_prediction * 100,marker="o",label="predicted")
    axs[2].set(xlabel="Test month",ylabel="Delay rate (%)",title="Monthly calibration drift")
    axs[2].legend()
plt.tight_layout()
plt.show()

# COMMAND ----------


monthly_report = monthly_audit[
    ["group_name", "flights", "actual_delay_rate",
     "mean_prediction", "brier"]
].copy()

monthly_report = monthly_report.sort_values("group_name")

monthly_report["actual_delay_rate"] *= 100
monthly_report["mean_prediction"] *= 100

monthly_report["gap_pp"] = (
    monthly_report["actual_delay_rate"]
    - monthly_report["mean_prediction"]
)

display(monthly_report.round(2))


# COMMAND ----------


tables_part3 = [
    "part3_model_metrics",
    "part3_bootstrap_ci",
    "part3_subgroup_reliability",
    "part3_alert_capacity",
    "part3_validation_thresholds",
    "part3_test_predictions"
]

for table in tables_part3:
    full_name = f"{OUTPUT_SCHEMA}.{table}"
    exists = spark.catalog.tableExists(full_name)
    print(f"{table}: {'YA EXISTE' if exists else 'Disponible'}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Persist audited results to Gold (optional)
# MAGIC Outputs are new tables; existing Gold source tables and model metrics are untouched.

# COMMAND ----------

def persist(df, table_name):
    if not SAVE_TABLES:
        print(f"WRITE SKIPPED: {OUTPUT_SCHEMA}.{table_name}")
        return
    if df is None or len(df) == 0:
        print(f"EMPTY: {table_name}")
        return
    writable = df.replace([np.inf,-np.inf],np.nan)
    writable = writable.astype(object).where(pd.notna(writable), None)
    spark.createDataFrame(writable).write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(f"{OUTPUT_SCHEMA}.{table_name}")
    print(f"Saved: {OUTPUT_SCHEMA}.{table_name} ({len(writable):,} rows)")

persist(metrics, "part3_model_metrics")
persist(ci_summary, "part3_bootstrap_ci")
persist(subgroups, "part3_subgroup_reliability")
persist(capacity, "part3_alert_capacity")
persist(tradeoffs, "part3_validation_thresholds")
persist(scored.assign(flight_date=scored.flight_date.dt.strftime("%Y-%m-%d")), "part3_test_predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Conclusions you may write *after running this notebook*
# MAGIC 1. Compare ROC-AUC and PR-AUC, not accuracy alone; examine both confidence intervals.
# MAGIC 2. Inspect calibration (Brier/ECE) and whether risks correspond to observed frequencies.
# MAGIC 3. Compare recall under fixed alert capacity and under validation-derived decision costs.
# MAGIC 4. Identify any weaker airports, airlines and months using adequate sample sizes.
# MAGIC 5. State clearly that false-alert and missed-delay costs are **illustrative assumptions**, not airline financial data.
# MAGIC 6. The current design uses one chronological training period and one test window. No causal claim is supported.