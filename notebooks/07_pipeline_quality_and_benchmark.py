# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 07 · Data engineering audit and performance benchmark
# MAGIC Validates reproducibility assumptions across Silver and Gold and benchmarks a live aggregated query against an explicitly materialized Delta aggregate for the **same** business question.
# MAGIC
# MAGIC This is a measured case study, not a claim that Delta is inherently faster than Parquet, nor a comparison of equivalent end-to-end computation cost. Materialization has an upfront cost and must be refreshed when source tables change.

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import time

SILVER_FLIGHTS = "workspace.silver_flights.flights_clean"
GOLD_FEATURES = "workspace.gold_flights.flights_features"
GOLD_WEATHER = "workspace.gold_flights.flights_weather_features"
WEATHER_SILVER = "workspace.silver_flights.weather_clean"
AGG_TABLE = "workspace.gold_flights.part3_airport_materialized"
RESULT_TABLE = "workspace.gold_flights.part3_benchmark_results"
RUN_BENCHMARK = True
SAVE_TABLES = False
REPETITIONS = 5
for table in [SILVER_FLIGHTS, GOLD_FEATURES, GOLD_WEATHER, WEATHER_SILVER]:
    assert spark.catalog.tableExists(table), f"Missing: {table}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Automated checks with explicit pass/fail status

# COMMAND ----------

audit = []
def check(name, observed, condition, explanation):
    audit.append({"check":name,"value":float(observed),"status":"PASS" if condition else "FAIL","note":explanation})

silver = spark.table(SILVER_FLIGHTS)
gold = spark.table(GOLD_FEATURES)
weather_gold = spark.table(GOLD_WEATHER)
weather = spark.table(WEATHER_SILVER)

n_silver = silver.count()
n_gold = gold.count()
n_joined = weather_gold.count()
check("silver_equals_operational_gold_count", n_gold-n_silver, n_gold == n_silver,
      "Gold operational features should retain Silver flight row count")
check("weather_joined_not_larger_than_gold", n_joined-n_gold, n_joined <= n_gold,
      "Weather-matched modeling population should not exceed operational Gold")
check("weather_coverage_percent", n_joined / n_gold * 100, n_gold > 0 and n_joined/n_gold >= 0.95,
      "95% is an explicitly selected project threshold, not a universal standard")

label_mismatch = silver.filter(F.col("delayed_15") != F.when(F.col("dep_delay") >= 15, F.lit(1)).otherwise(F.lit(0))).count()
check("incorrect_binary_labels", label_mismatch, label_mismatch == 0,
      "Label must be dep_delay >= 15")
null_required = silver.filter(
    F.col("flight_date").isNull() | F.col("origin").isNull() |
    F.col("dest").isNull() | F.col("airline").isNull() | F.col("delayed_15").isNull()
).count()
check("null_required_fields", null_required, null_required == 0, "Key model variables must exist")

# Weather grain is airport and hour (in the existing workflow one GFS model / forecast horizon).
# If distinct lead times or models are added, include those fields in the uniqueness key.
weather_keys = ["airport", "weather_time_utc"]
key_multiplicity = weather.groupBy(*weather_keys).agg(F.count("*").alias("n"))
dup_weather_keys = key_multiplicity.filter(F.col("n") > 1).count()
check("duplicate_weather_airport_hours", dup_weather_keys, dup_weather_keys == 0,
      "Multiple weather sources require a richer key")

summary = pd.DataFrame(audit)
print(summary.to_string(index=False))
if (summary.status == "FAIL").any():
    print("WARNING: At least one integrity check failed; investigate before claiming final reproducibility.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Query plan and materialization
# MAGIC Compare an **uncached live aggregation** of ~8.65M flights against querying a materialized Delta summary (roughly 30 airport rows). The queries yield the **same answer**, but materialization must be included in the cost discussion.

# COMMAND ----------


benchmark_tables = [
    "part3_airport_materialized",
    "part3_benchmark_results"
]

for name in benchmark_tables:
    table = f"workspace.gold_flights.{name}"
    exists = spark.catalog.tableExists(table)
    print(
        f"{name}: "
        f"{'YA EXISTE' if exists else 'Disponible'}"
    )


# COMMAND ----------

source = weather_gold.select("origin", "delayed_15")
def make_live_query():
    return (source.groupBy("origin")
            .agg(F.count("*").alias("flights"), F.sum(F.col("delayed_15").cast("long")).alias("delays")))

if SAVE_TABLES:
    write_start = time.perf_counter()
    make_live_query().write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(AGG_TABLE)
    materialize_seconds = time.perf_counter() - write_start
    print("Materialization time (s):", round(materialize_seconds,2))
    print("Live query physical plan:")
    make_live_query().explain(mode="formatted")

    # Validate numerical equivalence before benchmarking.
    left = {r["origin"]:(r["flights"],r["delays"]) for r in make_live_query().collect()}
    right = {r["origin"]:(r["flights"],r["delays"]) for r in spark.table(AGG_TABLE).collect()}
    assert left == right, "Live and materialized results differ; benchmark is invalid."
else:
    print("SAVE_TABLES=False: materialized table / benchmark skipped; integrity checks still run.")
    materialize_seconds = None

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Repeat timings without pretending that caching and cluster size disappear

# COMMAND ----------

def timed_collect(query_builder, repetitions):
    times = []
    _ = query_builder().collect()   # explicitly warm up once
    for i in range(repetitions):
        started = time.perf_counter()
        _ = query_builder().collect()
        times.append(time.perf_counter()-started)
    return times

if RUN_BENCHMARK and SAVE_TABLES:
    live_times = timed_collect(make_live_query,REPETITIONS)
    materialized_times = timed_collect(lambda: spark.table(AGG_TABLE),REPETITIONS)
    timing = pd.DataFrame([
        {"method":"live_aggregate", "trial":j+1,"seconds":float(t)} for j,t in enumerate(live_times)
    ] + [
        {"method":"materialized_gold", "trial":j+1,"seconds":float(t)} for j,t in enumerate(materialized_times)
    ])
    print(timing.groupby("method").seconds.agg(["count","min","median","mean","max"]).round(3))
    print("One-off materialization cost (seconds):",round(materialize_seconds,2))
    print("Do NOT use this as a hardware-normalized Spark-vs-other-engine benchmark.")
    if SAVE_TABLES:
        spark.createDataFrame(timing).write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(RESULT_TABLE)
else:
    print("Benchmark skipped; enable both RUN_BENCHMARK and SAVE_TABLES to measure Delta materialization.")

# COMMAND ----------


tables = [
    "part3_airport_materialized",
    "part3_benchmark_results"
]

for name in tables:
    full_name = f"workspace.gold_flights.{name}"

    if spark.catalog.tableExists(full_name):
        rows = spark.table(full_name).count()
        print(f"OK: {name} - {rows} filas")
    else:
        print(f"FALTA: {name}")


# COMMAND ----------


# INSPECCIÓN DEL ESQUEMA FINANCIERO

tabla = spark.table(
    "workspace.gold_flights.flights_features"
)

print("=== COLUMNAS DISPONIBLES ===")

for columna in tabla.columns:
    print(columna)

print("\n=== TIPOS DE DATOS ===")
tabla.printSchema()

print("\n=== INSPECCIÓN COMPLETADA ===")


# COMMAND ----------


from pyspark.sql import functions as F

tabla = spark.table(
    "workspace.gold_flights.flights_features"
)

resumen = tabla.agg(
    F.count("*").alias("total_vuelos"),
    F.sum(
        F.when(F.col("dep_delay") >= 15, 1)
        .otherwise(0)
    ).alias("vuelos_retrasados_15min"),
    F.sum(
        F.greatest(
            F.col("dep_delay"),
            F.lit(0)
        )
    ).alias("minutos_positivos_totales"),
    F.avg("dep_delay").alias("retraso_promedio"),
    F.max("dep_delay").alias("retraso_maximo")
)

display(resumen)


# COMMAND ----------


from pyspark.sql import functions as F

df = spark.table(
    "workspace.gold_flights.flights_features"
)

resultado = df.agg(
    F.sum(
        F.when(F.col("dep_delay") >= 15,
               F.col("dep_delay")).otherwise(0)
    ).alias("minutos_retrasos_15plus"),

    F.sum(
        F.when(F.col("dep_delay") > 0,
               F.col("dep_delay")).otherwise(0)
    ).alias("minutos_positivos_total"),

    F.sum(
        F.when(F.col("dep_delay") > 600, 1)
         .otherwise(0)
    ).alias("vuelos_mas_10_horas"),

    F.sum(
        F.when(F.col("dep_delay") > 1440, 1)
         .otherwise(0)
    ).alias("vuelos_mas_24_horas")
)

display(resultado)


# COMMAND ----------


from pyspark.sql import functions as F

df = spark.table(
    "workspace.gold_flights.flights_features"
)

d = F.col("dep_delay")

resultado = df.agg(
    F.sum(
        F.when(d > 600, d).otherwise(0)
    ).alias("minutos_mas_10_horas"),

    F.sum(
        F.when(d > 1440, d).otherwise(0)
    ).alias("minutos_mas_24_horas"),

    F.sum(
        F.when((d >= 15) & (d <= 600), d)
        .otherwise(0)
    ).alias("minutos_15plus_sin_extremos_10h"),

    F.sum(
        F.when((d >= 15) & (d <= 1440), d)
        .otherwise(0)
    ).alias("minutos_15plus_sin_extremos_24h")
)

display(resultado)


# COMMAND ----------


from pyspark.sql import functions as F

df = spark.table(
    "workspace.gold_flights.flights_features"
)

d = F.col("dep_delay")

resultado = df.agg(
    F.sum(
        F.when(d >= 15, d).otherwise(0)
    ).alias("minutos_originales"),

    F.sum(
        F.when(d >= 15, F.least(d, F.lit(300)))
        .otherwise(0)
    ).alias("minutos_limitados_300"),

    F.sum(
        F.when(d > 300, 1).otherwise(0)
    ).alias("vuelos_mayores_5h")
)

resultado = resultado.withColumn(
    "minutos_excedentes",
    F.col("minutos_originales")
    - F.col("minutos_limitados_300")
)

display(resultado)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Reproducibility checklist (must be confirmed by the team)
# MAGIC - [ ] Document the exact Databricks Runtime and installed package versions.
# MAGIC - [ ] Record the Unity Catalog/catalog/schema permissions, without credentials.
# MAGIC - [ ] Confirm that notebook execution order succeeds on a clean compute environment.
# MAGIC - [ ] Verify training thresholds are frozen on validation (not selected from test).
# MAGIC - [ ] Save notebook outputs and actual counts/timings in the final report.
# MAGIC - [ ] Include current limitations: selected 30 airports, canceled flights excluded, unresolved possible duplicates and daylight-saving edge cases.
# MAGIC - [ ] Document materialized table refresh dependencies, source version and data period.