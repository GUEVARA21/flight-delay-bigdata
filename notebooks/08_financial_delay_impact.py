# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 08 · Financial delay impact | Flight Delay Intelligence
# MAGIC **Status: scenario-based exploratory financial analysis, not airline accounting.**
# MAGIC
# MAGIC This notebook counts each *source row once* from `flights_features`, NOT the aircraft-pair propagation table. Physical-flight uniqueness is still an unresolved source-data audit. It builds reproducible descriptive/financial summaries for the Streamlit dashboard, starting with a full reconciliation against independently recorded project totals.
# MAGIC
# MAGIC **Scope:** historical scheduled departure delays for the 30-origin-airport analytical subset, 2024–2025. Departure delay is not the same as incremental block time, arrival delay, disruption causality, or a proven financial loss. Cancellations excluded from existing Gold source. The source's row-level flight uniqueness has not yet been independently audited.
# MAGIC
# MAGIC **Reference methodology:**
# MAGIC - IATA *Air Traffic Control Delays in the US*, Appendix methodology: caps cost of individual delays at 300 min; more sophisticated nonlinear costing by aircraft, stage, duration and passengers; specific to FAA ATC delays, unlike our general departure-delay data. https://www.iata.org/en/iata-repository/publications/economic-reports/air-traffic-control-delays-in-the-us
# MAGIC - A4A, July 2026: 2025 **direct block-operation cost** USD 98.41/min (taxi + airborne) is *context only*, not the incremental cost of a gate departure delay. https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs/
# MAGIC - FAA *Airport Benefit-Cost Guidance* (2026), Appendix E, Exhibit E-6: phase-dependent marginal fuel, crew and maintenance cost factors. https://www.faa.gov/regulations_policies/policy_guidance/benefit_cost/FAA-Airport-Benefit-Cost-Guidance.pdf
# MAGIC - BTS definition: `dep_delay` is actual **gate departure** minus scheduled gate departure, not taxi/airborne delay. https://transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD
# MAGIC - EUROCONTROL standard inputs: stage-specific, indicative tactical costs from European airline research; NOT directly transferable to US airlines or to departure-delay minutes. https://permanent-release-v10-0-0--standard-inputs.netlify.app/chapters/cost_of_delay.html
# MAGIC
# MAGIC **Cost scenarios:** a transparent **FAA × A4A 2025 hypothetical marginal at-gate cost proxy**, then sensitivity at 0.5×, 1.0×, 2.0×. These multipliers are *analyst-defined*, NOT published ranges and not actual airline expense. This is a proxy for an avoidable minute truly incurred **at-gate**, not proof that every minute of departure lateness has that marginal cost. Streamlit must label all scenarios **hypothetical** and show the rate derivation.

# COMMAND ----------


print("=== ESQUEMAS DESDE EL NOTEBOOK ===")
spark.sql("SHOW SCHEMAS IN workspace").show(
    truncate=False
)

print("=== VERIFICAR TABLA GOLD ===")
print(
    "Existe:",
    spark.catalog.tableExists(
        "workspace.gold_flights.flights_features"
    )
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Configuration and source/columns

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from functools import reduce
from datetime import datetime, timezone

SOURCE_TABLE = "workspace.gold_flights.flights_features"
OUTPUT_SCHEMA = "workspace.gold_flights"
SAVE_TABLES = False              # DO NOT enable until audit cells all PASS
COST_CAP_MINUTES = 300           # comparable sensitivity treatment, NOT complete IATA costing
LATE_THRESHOLD = 15
# A4A 2025 USD per block minute component averages; not, by themselves, incremental delay costs.
A4A_2025_FUEL_USD_PER_BLOCK_MIN = 29.34
A4A_2025_CREW_USD_PER_BLOCK_MIN = 37.01
A4A_2025_MAINT_USD_PER_BLOCK_MIN = 18.35
# FAA Appendix E Exhibit E-6 marginal share of each component for AT-GATE delay.
FAA_GATE_FUEL_FACTOR = 0.10
FAA_GATE_CREW_FACTOR = 0.50
FAA_GATE_MAINT_FACTOR = 0.04
FAA_A4A_GATE_PROXY_USD_PER_MIN = (
    FAA_GATE_FUEL_FACTOR * A4A_2025_FUEL_USD_PER_BLOCK_MIN
    + FAA_GATE_CREW_FACTOR * A4A_2025_CREW_USD_PER_BLOCK_MIN
    + FAA_GATE_MAINT_FACTOR * A4A_2025_MAINT_USD_PER_BLOCK_MIN
)  # USD 22.173 / min; an illustrative synthesis, not an A4A or FAA published quote.
# Analyst-defined sensitivity multipliers; NOT official FAA or A4A confidence bounds.
SCENARIO_RATES_USD = {
    "low": round(FAA_A4A_GATE_PROXY_USD_PER_MIN * 0.50, 4),
    "central": round(FAA_A4A_GATE_PROXY_USD_PER_MIN, 4),
    "high": round(FAA_A4A_GATE_PROXY_USD_PER_MIN * 2.0, 4),
}
REDUCTION_ASSUMPTIONS = [0.05, 0.10, 0.20]
REQUIRE_EXPECTED_RECONCILIATION = True

OUTPUT_TABLES = {
    "overview": f"{OUTPUT_SCHEMA}.part3_financial_overview",
    "by_dimension": f"{OUTPUT_SCHEMA}.part3_financial_by_dimension",
    "duration": f"{OUTPUT_SCHEMA}.part3_financial_delay_bands",
    "sensitivity": f"{OUTPUT_SCHEMA}.part3_financial_scenarios",
    "assumptions": f"{OUTPUT_SCHEMA}.part3_financial_assumptions"
}

assert COST_CAP_MINUTES >= LATE_THRESHOLD
assert all(0 < r for r in SCENARIO_RATES_USD.values())
assert all(0 <= x <= 1 for x in REDUCTION_ASSUMPTIONS)
assert spark.catalog.tableExists(SOURCE_TABLE), f"Missing source table: {SOURCE_TABLE}"
source = spark.table(SOURCE_TABLE)
required = {"flight_date", "airline", "origin", "dest", "dep_delay"}
missing = sorted(required - set(source.columns))
if missing:
    raise ValueError(f"Missing required financial columns: {missing}")
print("Source", SOURCE_TABLE)
print("Input columns (inspect before publication):")
print(", ".join(sorted(source.columns)))
print("Reference at-gate marginal cost proxy (2025 USD/min):", round(FAA_A4A_GATE_PROXY_USD_PER_MIN, 4))
print("Scenario rates (USD/min; multipliers assumed):", SCENARIO_RATES_USD)
print("Proxy assumes hypothetical avoidable delay at gate; NOT actual documented expenses.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Derived minutes, thresholds, and a disjoint delay duration classification
# MAGIC Use *one Gold row per source record*, never join in the 4.42M aircraft-pair table.

# COMMAND ----------

d = F.col("dep_delay_num")
work = (
    source.select("flight_date", "airline", "origin", "dest", "dep_delay")
      .withColumn("flight_date", F.to_date("flight_date"))
      .withColumn("dep_delay_num", F.col("dep_delay").cast("double"))
      .drop("dep_delay")
      .withColumn("year", F.year("flight_date"))
      .withColumn("month", F.date_format("flight_date", "yyyy-MM"))
      .withColumn("positive_minutes", F.when(d > 0, d).otherwise(F.lit(0.0)))
      .withColumn("late_minutes", F.when(d >= LATE_THRESHOLD, d).otherwise(F.lit(0.0)))
      .withColumn("capped_late_minutes", F.when(d >= LATE_THRESHOLD, F.least(d, F.lit(float(COST_CAP_MINUTES)))).otherwise(F.lit(0.0)))
      .withColumn("delay_band", 
          F.when(d.isNull(), "NULL")
           .when(d < 0, "EARLY")
           .when(d == 0, "ZERO")
           .when(d < 15, "01_1-14")
           .when(d < 30, "02_15-29")
           .when(d < 60, "03_30-59")
           .when(d < 180, "04_60-179")
           .when(d <= 300, "05_180-300")
           .when(d <= 600, "06_301-600")
           .when(d <= 1440, "07_601-1440")
           .otherwise("08_over_1440")
      )
      .select("flight_date", "year", "month", "airline", "origin", "dest", "dep_delay_num", "positive_minutes", "late_minutes", "capped_late_minutes", "delay_band")
)
# Only the selected columns are persisted, avoiding caching the full flight record.
work = work.persist(StorageLevel.MEMORY_AND_DISK)
print("Working rows:", work.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Full-population audit and reconciliation to verified figures
# MAGIC This is an explicit guardrail: never publish financial projections if base counts changed silently.

# COMMAND ----------

def summary_aggs():
    return [
        F.count(F.lit(1)).alias("flights"),
        F.sum(F.when(d >= LATE_THRESHOLD, 1).otherwise(0)).alias("late_flights"),
        F.sum("positive_minutes").alias("positive_minutes"),
        F.sum("late_minutes").alias("late_minutes"),
        F.sum("capped_late_minutes").alias("capped_late_minutes"),
        F.sum(F.when(d > 300, 1).otherwise(0)).alias("flights_over_300"),
        F.sum(F.when(d > 600, 1).otherwise(0)).alias("flights_over_600"),
        F.sum(F.when(d > 1440, 1).otherwise(0)).alias("flights_over_1440"),
        F.sum(F.when(d > 600, d).otherwise(0)).alias("minutes_over_600_flights"),
        F.sum(F.when(d > 1440, d).otherwise(0)).alias("minutes_over_1440_flights"),
        F.sum(F.when(d.isNull(), 1).otherwise(0)).alias("missing_delay"),
        F.sum(F.when(F.col("flight_date").isNull(), 1).otherwise(0)).alias("missing_date"),
        F.avg("dep_delay_num").alias("mean_signed_departure_delay"),
        F.max("dep_delay_num").alias("maximum_departure_delay")
    ]

overview = work.agg(*summary_aggs())
record = overview.first().asDict()
print("=== Audited original population ===")
for k, v in record.items():
    print(f"{k}: {v}")

previously_verified = {
    "flights": 8655359,
    "late_flights": 1959157,
    "positive_minutes": 146664763,
    "late_minutes": 137438884,
    "flights_over_600": 11368,
    "flights_over_1440": 394,
    "minutes_over_600_flights": 9791435,
    "minutes_over_1440_flights": 696098,
}
checks = []
for metric, expected in previously_verified.items():
    actual = record[metric]
    passed = actual is not None and abs(float(actual) - float(expected)) < 0.00001
    checks.append((metric, expected, actual, "PASS" if passed else "FAIL"))
print("=== Verification against previously observed Databricks results ===")
for line in checks:
    print(line)
if REQUIRE_EXPECTED_RECONCILIATION and any(s != "PASS" for _,_,_,s in checks):
    raise AssertionError("Financial input no longer reconciles with previous figures. Investigate before saving.")
if record["missing_date"]:
    raise ValueError("Missing flight dates: annual/monthly financial grouping would omit rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Detailed historical operational findings by year, month, airline, origin, route, and duration
# MAGIC Rows at different granularities **must never be added together** in Streamlit.

# COMMAND ----------

def metric_aggs():
    return [
        F.count(F.lit(1)).alias("flights"),
        F.sum(F.when(d >= LATE_THRESHOLD, 1).otherwise(0)).alias("late_flights"),
        F.sum("positive_minutes").alias("positive_minutes"),
        F.sum("late_minutes").alias("late_minutes"),
        F.sum("capped_late_minutes").alias("capped_late_minutes")
    ]

def apply_costs(df):
    out = (df.withColumn("delay_rate_pct", F.round(F.col("late_flights") / F.col("flights") * 100, 4))
             .withColumn("late_minutes_per_flight", F.round(F.col("late_minutes") / F.col("flights"), 4))
             .withColumn("cap_removed_minutes", F.col("late_minutes") - F.col("capped_late_minutes")))
    for name, rate in SCENARIO_RATES_USD.items():
        out = (out.withColumn(f"cost_{name}_usd_scenario", F.round(F.col("capped_late_minutes") * F.lit(rate), 2))
                  .withColumn(f"cost_{name}_uncapped_usd_scenario", F.round(F.col("late_minutes") * F.lit(rate), 2)))
    return out

# Shared, consistently shaped dataset for Streamlit; source table scanned for each aggregation.
granularities = [
    ("year", ["year"]),
    ("month", ["month"]),
    ("airline", ["airline"]),
    ("airport", ["origin"]),
    ("year_airline", ["year", "airline"]),
    ("year_airport", ["year", "origin"]),
    ("route", ["origin", "dest"]),
]
financial_slices = []
for level, keys in granularities:
    block = apply_costs(work.groupBy(*keys).agg(*metric_aggs()))
    block = block.withColumn("level", F.lit(level))
    for field_name, dtype in [("year", "int"), ("month", "string"), ("airline", "string"), ("origin", "string"), ("dest", "string")]:
        if field_name not in block.columns:
            block = block.withColumn(field_name, F.lit(None).cast(dtype))
    financial_slices.append(block)
by_dimension = reduce(lambda a, b: a.unionByName(b), financial_slices).persist(StorageLevel.MEMORY_AND_DISK)
print("Financial summary rows:", by_dimension.count())

delay_bands = apply_costs(work.groupBy("delay_band").agg(*metric_aggs())).orderBy("delay_band").persist(StorageLevel.MEMORY_AND_DISK)
print("=== Duration distribution ===")
display(delay_bands)
print("=== Annual financial scenarios (USD amounts are hypothetical only) ===")
display(by_dimension.filter(F.col("level") == "year").orderBy("year"))
print("=== Highest airports by capped minutes (NOT a ranking of airport efficiency) ===")
display(by_dimension.filter(F.col("level") == "airport").orderBy(F.desc("capped_late_minutes")).limit(12))
print("=== Highest carriers by capped minutes (NOT a normalized risk ranking) ===")
display(by_dimension.filter(F.col("level") == "airline").orderBy(F.desc("capped_late_minutes")).limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Sensitivity analysis: original vs capped minutes, USD/min scenarios, and hypothetical avoidable fraction
# MAGIC Savings are **counterfactual scenario arithmetic**, not the realized benefit of XGBoost or demonstrated effects of any intervention. Gross savings exclude intervention costs and cannot be claimed as ROI.

# COMMAND ----------

scenario_rows = []
for rate_name, usd_per_min in SCENARIO_RATES_USD.items():
    for minutes_basis, mins in [("uncapped_15plus", float(record["late_minutes"])), ("capped_300min_15plus", float(record["capped_late_minutes"]))]:
        gross_cost = mins * usd_per_min
        for target in [0.0] + REDUCTION_ASSUMPTIONS:
            scenario_rows.append({
                "minutes_basis": minutes_basis,
                "scenario": rate_name,
                "hypothetical_usd_per_minute": float(usd_per_min),
                "eligible_departure_delay_minutes": float(mins),
                "gross_proxy_usd": float(gross_cost),
                "assumed_reduction_fraction": float(target),
                "gross_avoidable_proxy_usd": float(gross_cost * target),
                "remaining_proxy_usd": float(gross_cost * (1-target)),
                "cost_definition": "ILLUSTRATIVE cost exposure based on departure delay; NOT measured airline loss",
                "scope": "Selected US origin airports; 2024-2025; Gold flights_features"
            })
sensitivity = spark.createDataFrame(scenario_rows)
print("=== Scenario sensitivity (hypothetical USD): cap vs no cap ===")
display(sensitivity.orderBy("minutes_basis", "scenario", "assumed_reduction_fraction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Automated invariants and denominator checks

# COMMAND ----------

# Compare aggregates at independent grains with source to avoid double counting.
annual = by_dimension.filter(F.col("level") == "year")
by_airline = by_dimension.filter(F.col("level") == "airline")
by_airport = by_dimension.filter(F.col("level") == "airport")
by_month = by_dimension.filter(F.col("level") == "month")

def sum_col(frame, column):
    return float(frame.agg(F.sum(F.col(column)).alias("total")).first()["total"] or 0)

for tag, frame in [("year", annual), ("month", by_month), ("airline", by_airline), ("airport", by_airport), ("delay_band", delay_bands)]:
    for metric in ["flights", "late_flights", "positive_minutes", "late_minutes", "capped_late_minutes"]:
        actual = sum_col(frame, metric)
        expected = float(record[metric])
        assert abs(actual - expected) < 0.0001, (tag, metric, actual, expected)
    print(f"PASS additive reconciliation: {tag}")

assert float(record["capped_late_minutes"]) <= float(record["late_minutes"])
assert float(record["late_minutes"]) <= float(record["positive_minutes"])
assert float(record["late_flights"]) <= float(record["flights"])
assert record["missing_delay"] == 0, "Delay nulls require explicit treatment before final reporting"
print("PASS financial invariants; no tables written yet unless SAVE_TABLES=True in final cell.")

# COMMAND ----------


display(
    sensitivity.filter(
        (F.col("scenario") == "central") &
        (F.col("minutes_basis") == "capped_300min_15plus")
    )
    .select(
        "assumed_reduction_fraction",
        "gross_proxy_usd",
        "gross_avoidable_proxy_usd",
        "remaining_proxy_usd"
    )
    .orderBy("assumed_reduction_fraction")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Assumptions log and source/version transparency

# COMMAND ----------

assumption_rows = [
    ("financial_scope", "selected_US_origin_airports_2024_2025", "Project source Gold; 30-airport selected population; departures only"),
    ("source_table", SOURCE_TABLE, "Row counts audited; not validated as unique physical flight IDs"),
    ("delay_threshold_minutes", str(LATE_THRESHOLD), "Classification convention from project dep_delay >= 15"),
    ("per_flight_cap_minutes", str(COST_CAP_MINUTES), "IATA methodological sensitivity: NOT equivalent to IATA nonlinear cost model"),
    ("cost_model", "scenario_linear_cap_per_flight", "Illustrative USD cost proxy = min(dep_delay,cap) * user-chosen rate for dep_delay>=15"),
    ("rate_low_usd_per_min", str(SCENARIO_RATES_USD["low"]), "0.5 x FAA×A4A 2025 derived at-gate proxy; multiplier analyst assumption"),
    ("rate_central_usd_per_min", str(SCENARIO_RATES_USD["central"]), "1 x FAA×A4A 2025 derived at-gate proxy; not realized airline cost"),
    ("rate_high_usd_per_min", str(SCENARIO_RATES_USD["high"]), "2 x FAA×A4A 2025 derived at-gate proxy; multiplier analyst assumption"),
    ("proxy_faa_a4a_at_gate_usd_per_min", str(round(FAA_A4A_GATE_PROXY_USD_PER_MIN, 4)), "FAA 2026 factors × A4A 2025 fuel/crew/maintenance USD/block-minute"),
    ("rate_price_base_year", "2025_reference_USD", "2025 source component rates applied to 2024-2025 minutes without additional inflation adjustments"),
    ("external_reference_A4A", "https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs/", "2025 $98.41 per BLOCK minute; contextual, not directly applied"),
    ("external_reference_IATA", "https://www.iata.org/en/iata-repository/publications/economic-reports/air-traffic-control-delays-in-the-us", "ATC delay model; cost capped at 300 min; not directly reproduced"),
    ("external_reference_FAA", "https://www.faa.gov/regulations_policies/policy_guidance/benefit_cost/FAA-Airport-Benefit-Cost-Guidance.pdf", "2026 Appendix E, Exhibit E-6 marginal at-gate factors: fuel 0.1, crew 0.5, maintenance 0.04"),
    ("external_reference_BTS", "https://transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD", "Departure delay = scheduled to actual gate departure difference; not necessarily avoidable at-gate time"),
    ("external_reference_EUROCONTROL", "https://permanent-release-v10-0-0--standard-inputs.netlify.app/chapters/cost_of_delay.html", "European tactical delay cost; stage-specific, not applied to US flight minutes"),
    ("passenger_time_valuation", "excluded", "No passenger counts/occupancy verified at flight level"),
    ("causal_benefit", "not_estimated", "Prediction precision, delay propagation correlation and hypothetical savings do not prove interventions reduce delays"),
    ("cancellations", "excluded_from_Gold_scope", "Not included in historic cost proxy"),
    ("source_row_uniqueness", "not_audited", "Check flight ID uniqueness and effect of Gold selection before publication"),
]
assumptions = spark.createDataFrame(assumption_rows, ["parameter", "setting", "explanation"])
display(assumptions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Controlled Delta persistence AFTER all audit checks pass
# MAGIC Default SAFE mode previews data only. `SAVE_TABLES=True` intentionally overwrites only the five **financial** tables below, never the operational Gold or ML outputs.

# COMMAND ----------

overview_financial = apply_costs(overview)
if SAVE_TABLES:
    outputs = {
        "overview": overview_financial,
        "by_dimension": by_dimension,
        "duration": delay_bands,
        "sensitivity": sensitivity,
        "assumptions": assumptions,
    }
    for key, output_df in outputs.items():
        dest = OUTPUT_TABLES[key]
        (output_df.write.format("delta")
             .mode("overwrite").option("overwriteSchema", "true")
             .saveAsTable(dest))
        print(f"SAVED {dest}: {spark.table(dest).count()} rows")
else:
    print("SAFE MODE / SAVE_TABLES=False: All aggregates calculated and checked; no output tables written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Hand-off guidance (ONLY after end-to-end validation)
# MAGIC - Financial figures are *scenarios*, not observed carrier expense or recoverable cash.
# MAGIC - For Streamlit, query the five saved finance tables at their grain; never sum across distinct `level` values from `part3_financial_by_dimension`.
# MAGIC - Calculate potential reductions from explicit user assumptions, never from the ML model's test ROC-AUC/PR-AUC.
# MAGIC - Show USD/min knobs (2025 FAA×A4A at-gate proxy ≈ $22.17/min; 0.5×/1×/2× are analyst assumptions), source links, 300-minute cap switch and effect of extreme delays.
# MAGIC - Label and document selected 30-origin-airport scope, original 27.6M-to-8.65M shrinkage still unaudited, cancellations omitted, and flight-unique-key concern.
# MAGIC - Do NOT add propagation-pair rows to financial totals (they would double-count flights).
# MAGIC - Recommended follow-up: validate the source's flight-unique key and calibrate delay-phase-specific airline marginal costs using US data before claiming economic effects beyond scenario sensitivity.