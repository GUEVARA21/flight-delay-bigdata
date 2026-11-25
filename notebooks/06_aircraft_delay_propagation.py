# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 06 · Delay propagation across aircraft rotations
# MAGIC **Advanced descriptive analysis — NOT part of the H-24 predictive model.**
# MAGIC
# MAGIC The existing repository has `tail_number` and scheduled UTC departure timestamps, but its report acknowledges that delay propagation was not yet implemented. This notebook explores whether delays on a *previous observed flight with the same aircraft* are associated with delays on the next observed flight.
# MAGIC
# MAGIC **Crucial caveat:** Gold only covers a selected set of 30 origin airports. Some legs may be missing. A match of aircraft tail number and geographic continuity is an **inferred sequence**, not a verified complete aircraft itinerary. Scheduled departure-to-departure gap is **not** true turnaround time. This analysis is associative, not causal, and the preceding flight's actual delay must not be used as a 24h-in-advance predictor.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Source and validation

# COMMAND ----------

from pyspark.sql import functions as F, Window
from pyspark.storagelevel import StorageLevel
import matplotlib.pyplot as plt

SOURCE_TABLE = "workspace.gold_flights.flights_features"
OUTPUT_SCHEMA = "workspace.gold_flights"
SAVE_TABLES = False
GAP_MIN_HOURS = 2.0
GAP_MAX_HOURS = 18.0
MIN_STRATUM_SIZE = 30

required = ["tail_number", "flight_date", "scheduled_departure_utc", "origin", "dest", "airline", "dep_delay", "delayed_15"]
assert spark.catalog.tableExists(SOURCE_TABLE), f"No existe {SOURCE_TABLE}"
source = spark.table(SOURCE_TABLE)
missing = sorted(set(required) - set(source.columns))
if missing:
    raise ValueError(f"Faltan columnas requeridas en Gold: {missing}")

base = (
    source.select(*required)
    .withColumn("flight_date", F.to_date("flight_date"))
    .withColumn("scheduled_departure_utc", F.col("scheduled_departure_utc").cast("timestamp"))
    .withColumn("tail_number", F.upper(F.trim(F.col("tail_number"))))
    .filter((F.col("flight_date") >= F.lit("2024-02-01")) & (F.col("flight_date") < F.lit("2026-01-01")))
    .filter(F.col("scheduled_departure_utc").isNotNull())
    .filter(F.col("tail_number").isNotNull() & (F.length("tail_number") >= 3))
    .filter(~F.col("tail_number").isin("UNKNOWN", "NONE", "NULL", "N/A"))
    .filter(F.col("dep_delay").isNotNull() & F.col("delayed_15").isin(0, 1))
)

# A tail number scheduled for two different flight records at the exact same UTC time
# is ambiguous for sequencing. We *exclude these keys* instead of guessing an order.
key_w = Window.partitionBy("tail_number", "scheduled_departure_utc")
with_key_quality = base.withColumn("same_aircraft_time_records", F.count(F.lit(1)).over(key_w))
ambiguous = with_key_quality.filter(F.col("same_aircraft_time_records") > 1)
unique = with_key_quality.filter(F.col("same_aircraft_time_records") == 1).drop("same_aircraft_time_records")
print("Candidate flight rows:", base.count())
print("Excluded ambiguous same-tail/scheduled-time rows:", ambiguous.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Reconstruct plausible successive observed legs
# MAGIC The matching rules are deliberately strict: same tail number, previous destination = current origin, same reporting airline, and 2–18 hours between *scheduled departures*.

# COMMAND ----------

w = Window.partitionBy("tail_number").orderBy("scheduled_departure_utc", "flight_date", "origin", "dest")
legs = (
    unique
    .withColumn("prev_origin", F.lag("origin").over(w))
    .withColumn("prev_dest", F.lag("dest").over(w))
    .withColumn("prev_airline", F.lag("airline").over(w))
    .withColumn("prev_delay", F.lag("dep_delay").over(w))
    .withColumn("prev_delayed_15", F.lag("delayed_15").over(w))
    .withColumn("prev_sched_utc", F.lag("scheduled_departure_utc").over(w))
    .withColumn("scheduled_gap_hours", (F.unix_timestamp("scheduled_departure_utc") - F.unix_timestamp("prev_sched_utc")) / F.lit(3600.0))
    .withColumn("matching_airport", F.col("prev_dest") == F.col("origin"))
    .withColumn("matching_airline", F.col("prev_airline") == F.col("airline"))
    .withColumn("plausible_consecutive", (
        F.col("matching_airport") & F.col("matching_airline") &
        F.col("scheduled_gap_hours").between(GAP_MIN_HOURS,GAP_MAX_HOURS)
    ))
)
legs.persist(StorageLevel.MEMORY_AND_DISK)
pairs = legs.filter(F.col("plausible_consecutive")).persist(StorageLevel.MEMORY_AND_DISK)
num_pairs = pairs.count()
print(f"Plausible successive observed legs: {num_pairs:,}")
if num_pairs < 100:
    raise ValueError("Muy pocos pares. Revisa tail_number, scheduled_departure_utc, geographic continuity y horarios.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Delay-transition matrix and relative risk
# MAGIC Compare flights following a late observed leg with flights following an on-time observed leg. Both groups are subject to observational confounding (airports, season, scheduled time, network and aircraft utilization).

# COMMAND ----------

transitions = (
    pairs.groupBy("prev_delayed_15", "delayed_15")
    .agg(F.count("*").alias("flights"))
    .orderBy("prev_delayed_15", "delayed_15")
)
display(transitions)

risk = (
    pairs.groupBy("prev_delayed_15")
    .agg(F.count("*").alias("n"), F.avg(F.col("delayed_15").cast("double")).alias("current_delay_rate"))
)
risk_rows = {int(r["prev_delayed_15"]): r for r in risk.collect()}
if 0 not in risk_rows or 1 not in risk_rows:
    raise ValueError("No se observaron suficientes pares en ambos estados del vuelo anterior")
p0 = float(risk_rows[0]["current_delay_rate"])
p1 = float(risk_rows[1]["current_delay_rate"])
print(f"P(current delayed | previous NOT delayed): {p0:.2%}, N={risk_rows[0]['n']:,}")
print(f"P(current delayed | previous delayed):     {p1:.2%}, N={risk_rows[1]['n']:,}")
print(f"Descriptive risk difference: {(p1 - p0) * 100:+.2f} percentage points")
print(f"Descriptive risk ratio: {p1/p0:.3f}" if p0 > 0 else "Risk ratio undefined (baseline 0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Sensitivity by severity of previous delay, and flight gap

# COMMAND ----------

pairs_enriched = (
    pairs
    .withColumn("previous_delay_band", F.when(F.col("prev_delay") < 0,"early")
                .when(F.col("prev_delay") < 15,"0_to_14")
                .when(F.col("prev_delay") < 60,"15_to_59")
                .otherwise("60_plus"))
    .withColumn("gap_band", F.when(F.col("scheduled_gap_hours") < 5,"2_to_5h")
                .when(F.col("scheduled_gap_hours") < 9,"5_to_9h")
                .otherwise("9_to_18h"))
)
severity = (
    pairs_enriched.groupBy("previous_delay_band")
    .agg(F.count("*").alias("n"),F.avg(F.col("delayed_15").cast("double")).alias("following_delay_rate"),
         F.avg("prev_delay").alias("avg_previous_delay_min"))
    .orderBy("previous_delay_band")
)
gap_sensitivity = (
    pairs_enriched.groupBy("gap_band","prev_delayed_15")
    .agg(F.count("*").alias("n"),F.avg(F.col("delayed_15").cast("double")).alias("following_delay_rate"))
    .orderBy("gap_band","prev_delayed_15")
)
display(severity)
display(gap_sensitivity)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Airport × month matched comparison
# MAGIC A more careful descriptive comparison: within each *current origin airport and operating month*, compare previous delayed vs previous on-time legs. Only retain strata with enough flights in both groups. This reduces some compositional differences but cannot eliminate confounding or prove causation.

# COMMAND ----------

strata = (
    pairs.withColumn("month", F.date_format("flight_date", "yyyy-MM"))
    .groupBy("origin", "month")
    .agg(
        F.sum(F.when(F.col("prev_delayed_15") == 1, 1).otherwise(0)).alias("n_prev_late"),
        F.sum(F.when(F.col("prev_delayed_15") == 0, 1).otherwise(0)).alias("n_prev_ontime"),
        F.sum(F.when((F.col("prev_delayed_15") == 1) & (F.col("delayed_15") == 1), 1).otherwise(0)).alias("late_after_late"),
        F.sum(F.when((F.col("prev_delayed_15") == 0) & (F.col("delayed_15") == 1), 1).otherwise(0)).alias("late_after_ontime")
    )
    .filter((F.col("n_prev_late") >= MIN_STRATUM_SIZE) & (F.col("n_prev_ontime") >= MIN_STRATUM_SIZE))
    .withColumn("p_after_late", F.col("late_after_late") / F.col("n_prev_late"))
    .withColumn("p_after_ontime", F.col("late_after_ontime") / F.col("n_prev_ontime"))
    .withColumn("risk_difference_pp", (F.col("p_after_late") - F.col("p_after_ontime")) * 100)
    .withColumn("total_flights", F.col("n_prev_late") + F.col("n_prev_ontime"))
)
if strata.limit(1).count():
    weighted = strata.agg((F.sum(F.col("risk_difference_pp") * F.col("total_flights")) /
                           F.sum("total_flights")).alias("weighted_risk_difference_pp"),
                           F.sum("total_flights").alias("covered_flights"),
                           F.count("*").alias("strata")).first()
    print(f"Airport×month weighted risk difference: {weighted['weighted_risk_difference_pp']:+.2f} pp")
    print(f"Covered flights: {weighted['covered_flights']:,}, comparable strata: {weighted['strata']:,}")
    display(strata.orderBy(F.desc("total_flights")).limit(20))
else:
    print("No strata reach the minimum size; reduce MIN_STRATUM_SIZE or examine coverage.")

# COMMAND ----------


from pyspark.sql import functions as F

summary_strata = strata.agg(
    F.count("*").alias("grupos_analizados"),

    F.sum("total_flights").alias("pares_analizados"),

    F.round(
        F.avg("risk_difference_pp"), 2
    ).alias("diferencia_promedio_pp"),

    F.round(
        F.sum(
            F.col("risk_difference_pp") *
            F.col("total_flights")
        ) / F.sum("total_flights"), 2
    ).alias("diferencia_ponderada_pp"),

    F.round(
        100 * F.avg(
            F.when(
                F.col("risk_difference_pp") > 0, 1
            ).otherwise(0)
        ), 2
    ).alias("porcentaje_grupos_positivos")
)

display(summary_strata)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Repeated-delay chains on plausible consecutive observed legs
# MAGIC This counts consecutive *observed* delayed flights; it does not recover unobserved flights outside the 30-airport sample. Chains break when aircraft/airport/airline/time continuity fails.

# COMMAND ----------

chain_w = Window.partitionBy("tail_number").orderBy("scheduled_departure_utc", "flight_date", "origin", "dest")
chain_cumulative = chain_w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
chains = (
    legs.withColumn("new_chain", F.when(
        F.col("plausible_consecutive") & (F.col("delayed_15") == 1) & (F.col("prev_delayed_15") == 1),
        F.lit(0)).otherwise(F.lit(1)))
    .withColumn("chain_id", F.sum("new_chain").over(chain_cumulative))
    .filter(F.col("delayed_15") == 1)
    .groupBy("tail_number", "chain_id").agg(F.count("*").alias("consecutive_delayed_observed_legs"))
)
chain_histogram = (
    chains.withColumn("streak_length_group", F.when(F.col("consecutive_delayed_observed_legs") >= 4,"4_plus")
                      .otherwise(F.col("consecutive_delayed_observed_legs").cast("string")))
    .groupBy("streak_length_group").agg(F.count("*").alias("observed_streaks"))
)
display(chain_histogram.orderBy("streak_length_group"))

# COMMAND ----------


display(
    chains
    .groupBy("consecutive_delayed_observed_legs")
    .count()
    .orderBy(
        F.desc("consecutive_delayed_observed_legs")
    )
    .limit(15)
)


# COMMAND ----------


display(
    chains
    .select(
        "tail_number",
        "chain_id",
        "consecutive_delayed_observed_legs"
    )
    .orderBy(
        F.desc("consecutive_delayed_observed_legs")
    )
    .limit(10)
)


# COMMAND ----------


from pyspark.sql import functions as F, Window

# Reconstruir los identificadores de cadenas
w_chain = Window.partitionBy("tail_number").orderBy(
    "scheduled_departure_utc",
    "flight_date",
    "origin",
    "dest"
)

w_cumulative = w_chain.rowsBetween(
    Window.unboundedPreceding,
    Window.currentRow
)

audited_legs = (
    legs
    .withColumn(
        "new_chain",
        F.when(
            F.col("plausible_consecutive")
            & (F.col("delayed_15") == 1)
            & (F.col("prev_delayed_15") == 1),
            F.lit(0)
        ).otherwise(F.lit(1))
    )
    .withColumn(
        "chain_id",
        F.sum("new_chain").over(w_cumulative)
    )
)

# Examinar los 23 vuelos
longest_chain = (
    audited_legs
    .filter(
        (F.col("tail_number") == "N641FR")
        & (F.col("chain_id") == 1205)
        & (F.col("delayed_15") == 1)
    )
    .select(
        "flight_date",
        "scheduled_departure_utc",
        "origin",
        "dest",
        "airline",
        "dep_delay",
        "scheduled_gap_hours",
        "plausible_consecutive"
    )
    .orderBy("scheduled_departure_utc")
)

display(longest_chain)
print("Vuelos en cadena:", longest_chain.count())


# COMMAND ----------


columnas_llegada = [
    c for c in spark.table(SOURCE_TABLE).columns
    if "arr" in c.lower()
    or "arrival" in c.lower()
    or "elapsed" in c.lower()
]

print("Columnas relacionadas con llegada:")
for columna in columnas_llegada:
    print("-", columna)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Top feeding routes and the network structure
# MAGIC The edge `prev_origin → origin` is the previous observed aircraft leg ending at the current departure airport (required by continuity). Ranking by delay rate requires a minimum sample to avoid tiny-cell artifacts.

# COMMAND ----------

network = (
    pairs.groupBy(F.col("prev_origin").alias("feeder_origin"), F.col("origin").alias("next_departure_airport"))
    .agg(F.count("*").alias("paired_legs"),
         F.avg(F.col("delayed_15").cast("double")).alias("following_delay_rate"),
         F.avg(F.col("prev_delayed_15").cast("double")).alias("previous_delay_rate"))
)
network_filtered = network.filter(F.col("paired_legs") >= 50)
display(network_filtered.orderBy(F.desc("following_delay_rate")).limit(15))

# COMMAND ----------


from pyspark.sql import functions as F

route_risk = (
    pairs
    .groupBy("prev_origin", "origin")
    .agg(
        F.count("*").alias("n"),
        F.sum(F.when(F.col("prev_delayed_15") == 1, 1)
              .otherwise(0)).alias("n_late"),
        F.sum(F.when(F.col("prev_delayed_15") == 0, 1)
              .otherwise(0)).alias("n_ontime"),
        F.avg(F.when(F.col("prev_delayed_15") == 1,
                     F.col("delayed_15").cast("double"))
             ).alias("p_after_late"),
        F.avg(F.when(F.col("prev_delayed_15") == 0,
                     F.col("delayed_15").cast("double"))
             ).alias("p_after_ontime")
    )
    .filter(
        (F.col("n") >= 1000) &
        (F.col("n_late") >= 100) &
        (F.col("n_ontime") >= 100)
    )
    .withColumn(
        "difference_pp",
        F.round(
            (F.col("p_after_late") -
             F.col("p_after_ontime")) * 100, 2
        )
    )
    .select(
        F.col("prev_origin").alias("feeder_origin"),
        F.col("origin").alias("next_airport"),
        "n",
        F.round(F.col("p_after_late") * 100, 2)
         .alias("after_late_pct"),
        F.round(F.col("p_after_ontime") * 100, 2)
         .alias("after_ontime_pct"),
        "difference_pp"
    )
    .orderBy(F.desc("difference_pp"))
)

display(route_risk.limit(15))


# COMMAND ----------


variables = [
    "legs",
    "pairs",
    "transitions",
    "severity",
    "gap_sensitivity",
    "strata",
    "chain_histogram",
    "network_filtered"
]

for variable in variables:
    estado = "OK" if variable in globals() else "FALTA"
    print(f"{variable}: {estado}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Save additional Gold analytical products (optional)
# MAGIC Existing source tables are not overwritten. Output tables can feed a Databricks SQL Dashboard.

# COMMAND ----------


tablas_rotation = [
    "part3_rotation_transitions",
    "part3_rotation_severity",
    "part3_rotation_gap_sensitivity",
    "part3_rotation_airport_month",
    "part3_rotation_streaks",
    "part3_rotation_network"
]

for nombre in tablas_rotation:
    tabla = f"workspace.gold_flights.{nombre}"
    existe = spark.catalog.tableExists(tabla)
    print(
        f"{nombre}: "
        f"{'YA EXISTE' if existe else 'Disponible'}"
    )


# COMMAND ----------

def write_analytical_table(df, name):
    if SAVE_TABLES:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{OUTPUT_SCHEMA}.{name}")
        print("Saved:", f"{OUTPUT_SCHEMA}.{name}")
    else:
        print("Skipped write:", name)

write_analytical_table(transitions, "part3_rotation_transitions")
write_analytical_table(severity, "part3_rotation_severity")
write_analytical_table(gap_sensitivity, "part3_rotation_gap_sensitivity")
write_analytical_table(strata, "part3_rotation_airport_month")
write_analytical_table(chain_histogram, "part3_rotation_streaks")
write_analytical_table(network_filtered, "part3_rotation_network")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Interpretation and limitations
# MAGIC - **Finding to report:** whether the current flight's delay probability is higher after a delayed *previous observed* flight of the same aircraft.
# MAGIC - **Strongest support:** a persistent airport×month adjusted descriptive association, a severity gradient and sensitivity across gap durations.
# MAGIC - **Cannot assert:** direct causal propagation, true turnaround time, 100% complete aircraft rotations or a 24-hour-ahead predictor based on observed preceding delays.
# MAGIC - **Selection issue:** only observed departures from the 30-airport sample participate, and canceled records were excluded.
# MAGIC - **Next research upgrade:** validated aircraft itinerary, arrival timestamps, scheduled/actual ground time, and event-time availability.