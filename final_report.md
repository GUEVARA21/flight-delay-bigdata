# Reporte final — Flight Delay Big Data / Flight Delay Intelligence

**Curso:** Datos Masivos  
**Repositorio:** `GUEVARA21/flight-delay-bigdata`  
**Periodo analizado:** 2024–2025  
**Equipo:** Santiago Nicolás Roldán Schroeder (@Skraeder), Diego García (@diegogargo) y José Miguel Guevara (@GUEVARA21)  
**Estado del documento:** reporte de resultados y contribuciones del equipo. Revisar evidencias de GitHub, instrucciones de ejecución y atribuciones individuales antes de la entrega definitiva.

## Resumen ejecutivo

El proyecto construye una solución de ingeniería de datos y analítica para estudiar retrasos de salida de vuelos en Estados Unidos. Integra el procesamiento de datos a gran escala con Spark y Delta Lake, modelos de clasificación, un análisis retrospectivo de secuencias de aeronaves, controles de calidad, un benchmark de consultas, escenarios económicos y un simulador probabilístico de itinerarios.

La población operativa Gold contiene **8,655,359 vuelos** de 2024 y 2025 en una selección de **30 aeropuertos de origen**. De ellos, **1,959,157 (22.64%)** tuvieron al menos 15 minutos de retraso de salida. Se compararon dos clasificadores XGBoost en una prueba temporal de 44,687 vuelos. El modelo con datos meteorológicos históricos obtuvo **ROC-AUC = 0.6917** y **PR-AUC = 0.3808**, frente a 0.6758 y 0.3512 del modelo operativo baseline. La disponibilidad anticipada de esas variables meteorológicas no está verificada: el resultado con clima debe leerse como **retrospectivo**.

En un análisis independiente se identificaron **4,422,412 pares plausibles de vuelos consecutivos** de una misma aeronave. La frecuencia de retraso del segundo vuelo fue 48.11% si el anterior se retrasó, frente a 14.50% si no se retrasó (riesgo relativo descriptivo aproximado de 3.32). Esto constituye una **asociación**, no evidencia causal.

El análisis financiero utiliza una tarifa hipotética central de **USD 22.173 por minuto** y un máximo de 300 minutos contabilizados por vuelo retrasado. Con **128,309,351 minutos elegibles**, la exposición *ilustrativa* resultó de aproximadamente **USD 2,845 millones** para toda la población estudiada. No son pérdidas verificadas de aerolíneas. Un simulador Monte Carlo separado permite explorar itinerarios con escalas utilizando retrasos históricos y supuestos editables, sin pretender emitir pronósticos operativos en tiempo real.

## Problema

Los retrasos de vuelos afectan la planificación aeroportuaria, la utilización de aeronaves, las conexiones y la experiencia de los pasajeros. El fenómeno presenta diferencias entre aerolíneas, aeropuertos, rutas, horarios y periodos, y es difícil de explicar utilizando una sola variable o un único indicador.

El reto del proyecto es convertir registros históricos heterogéneos en datos depurados, verificables y útiles para análisis descriptivos, modelos predictivos y escenarios de decisión, sin confundir estimaciones con hechos observados.

## Pregunta principal

**¿Qué patrones históricos, operativos y meteorológicos se relacionan con los retrasos de salida de al menos 15 minutos, y cómo pueden utilizarse para caracterizar riesgos, asociaciones entre vuelos consecutivos y escenarios de impacto operativo y económico?**

### Preguntas secundarias

1. ¿Qué aeropuertos, aerolíneas y periodos concentran mayor volumen o tasa de retrasos?
2. ¿Qué capacidad tienen los clasificadores XGBoost para distinguir vuelos con y sin retraso de al menos 15 minutos?
3. ¿Qué diferencia introduce el clima histórico frente a variables programadas operativas?
4. ¿Cómo cambia la frecuencia de retraso de un vuelo según el estado del vuelo anterior observado de la misma aeronave?
5. ¿Qué controles verifican la consistencia del pipeline y qué beneficio ofrece materializar agregados analíticos?
6. ¿Cómo cambia la exposición económica hipotética bajo diferentes tarifas y reducciones supuestas de minutos?
7. ¿Qué alcance y limitaciones tiene un simulador Monte Carlo de itinerarios con escalas construido a partir del historial?

## Fuentes de datos

- **Vuelos:** registros históricos del Bureau of Transportation Statistics (BTS) de Estados Unidos, con campos como fecha, aerolínea, origen, destino, horarios programados, matrícula de aeronave, retraso de salida y cancelación.
- **Meteorología:** variables históricas integradas al dataset Gold, como temperatura, precipitación, viento, humedad, nubosidad y presión atmosférica. Su carácter de información **disponible como pronóstico antes del vuelo no ha sido demostrado**.
- **Supuestos de costos:** parámetros económicos declarados en el Notebook 08; no corresponden a facturas ni gastos individuales observados.
- **Itinerarios simulados:** las duraciones de vuelo, tiempos de conexión y reglas de reacomodo se introducen como supuestos. No son itinerarios oficiales completos con llegadas verificadas.

**Unidad de análisis:** un registro de vuelo en las tablas operativas; en el Notebook 06, un par plausible de vuelos observados de una misma aeronave; en Monte Carlo, un escenario simulado de un itinerario.

**Alcance:** el conjunto Gold de origen cubre 30 aeropuertos seleccionados. No representa todos los vuelos de Estados Unidos. La conciliación detallada entre los aproximadamente 27.66 millones de registros originales y los 8.65 millones de Gold sigue pendiente de documentación integral.

## Arquitectura

Se utilizó una arquitectura **Bronze → Silver → Gold** sobre Databricks, PySpark, Delta Lake y Unity Catalog.

```text
Fuente BTS / fuente meteorológica
             ↓
Bronze: conservación de datos de origen
             ↓
Silver: limpieza y estandarización
             ↓
Gold: flights_features / flights_weather_features
             ↓
  ┌──────────┼────────────┬──────────┬────────────┐
  │          │            │          │            │
Modelo 05  Rotaciones 06 Calidad 07 Finanzas 08  Severidad 09
  │          │            │          │            │
  └──────────┴────────────┴──────────┴────────────┘
             ↓
Resultados analíticos en tablas Gold / reportes
             ↓
Consultas SQL y exploración de escenarios
```

Las tablas operativas principales son:

- `workspace.gold_flights.flights_features` — 8,655,359 filas.
- `workspace.gold_flights.flights_weather_features` — 8,653,172 filas.

El repositorio organiza las implementaciones en `notebooks/`, `src/`, `tests/`, `docs/` y `reports/`. El cálculo distribuido y la extracción de muestras se realizan en Databricks; no se almacenan en GitHub los datasets masivos.

## Metodología

### Preparación y población

Se prepararon variables temporales y operativas y se definió `delayed_15 = 1` cuando `dep_delay >= 15`. La información meteorológica se vinculó a una población Gold específica. El Notebook 07 auditó que Gold operativo y Silver coincidieran en conteo y verificó la cobertura del enlace meteorológico.

### Clasificación (Notebook 05)

Se utilizó una **muestra hash reproducible de 259,545 vuelos** de la población con clima y se entrenaron dos clasificadores XGBoost: **baseline** con variables operativas y **weather** con esas variables más seis meteorológicas.

El diseño temporal separó los registros en:

| Etapa | Registros |
|---|---:|
| Entrenamiento | 179,909 |
| Calibración | 23,384 |
| Selección de umbral | 11,565 |
| Prueba temporal | 44,687 |

Se empleó calibración de probabilidades, selección de umbrales en un periodo distinto del test y evaluación con ROC-AUC, PR-AUC, Brier, precision, recall y F1. Se aplicó bootstrap por bloques de día para estimar incertidumbre. Los umbrales de alerta fueron **0.17 para baseline** y **0.18 para weather**, basados en una función de costos relativos ilustrativos.

### Secuencias de aeronaves (Notebook 06)

Se ordenaron vuelos por `tail_number` y salida programada UTC. Mediante ventanas de PySpark y `lag()` se identificó el vuelo previo observado. Se retuvieron secuencias plausibles cuando había continuidad de aeropuerto, misma aerolínea y una diferencia de 2–18 horas entre **salidas programadas**. Este intervalo no es el tiempo real de escala. Se estudiaron transiciones, gravedad previa, intervalos entre vuelos, estratos aeropuerto–mes y rachas de vuelos retrasados.

### Calidad y rendimiento (Notebook 07)

Se ejecutaron controles automáticos entre Silver y Gold: conteos, cobertura meteorológica, consistencia de etiquetas, valores requeridos y unicidad de la clave meteorológica utilizada. También se comparó una agregación en vivo con la consulta a una tabla Delta agregada previamente, midiendo cinco ejecuciones de cada alternativa.

### Análisis financiero (Notebook 08)

Se distinguieron minutos positivos, minutos de vuelos retrasados 15+ y minutos limitados a 300 por vuelo. Se aplicaron tarifas hipotéticas por minuto de **USD 11.0865, 22.173 y 44.346** para los escenarios bajo, central y alto. Se realizaron agregaciones por año, mes, aerolínea, aeropuerto, ruta y banda de gravedad, y escenarios de reducción supuesta de 5%, 10% y 20%. Las sumas se reconciliaron para evitar inconsistencias.

### Duración y priorización experimental (Notebook 09)

Se entrenaron regresores para la duración condicional de retrasos de 15+ minutos y se contrastaron XGBoost, Random Forest, CatBoost y un ensamble en evaluaciones complementarias. Se investigó la priorización de vuelos mediante probabilidad de retraso, gravedad estimada y minutos esperados. El desempeño de regresión resultó bajo; esta etapa sigue siendo exploratoria.

### Monte Carlo para itinerarios

El simulador desarrollado aparte de los modelos 05–09 toma distribuciones empíricas de retrasos históricos por tramo, con filtros que pueden relajarse cuando no hay suficientes observaciones. Ejecuta **1,000–20,000 simulaciones** con retrasos de salida positivos limitados a 300 minutos, calcula escalas y aplica una regla **hipotética** de reacomodo cuando una conexión no se alcanza. Reporta demora esperada, P50, P90, P95, riesgo de conexiones perdidas y costos de pasajero configurables. **No integra actualmente el XGBoost entrenado ni una cadena completa Gamma–recuperación BTS.**

## Procesamiento

El procesamiento masivo se hizo mediante operaciones distribuidas de Spark, incluidas selección de columnas, filtros, agrupaciones, sumas y ventanas temporales. Los cálculos que requieren tratamiento local se realizan sobre muestras o tablas agregadas, no trasladando la totalidad de los millones de vuelos al proceso de Python de la aplicación.

Las salidas analíticas se guardaron en Unity Catalog. Algunos ejemplos:

- Notebook 05: `part3_model_metrics`, `part3_bootstrap_ci`, `part3_subgroup_reliability`, `part3_alert_capacity`, `part3_validation_thresholds`, `part3_test_predictions`.
- Notebook 06: `part3_rotation_transitions`, `part3_rotation_severity`, `part3_rotation_gap_sensitivity`, `part3_rotation_airport_month`, `part3_rotation_streaks`, `part3_rotation_network`.
- Notebook 07: `part3_airport_materialized`, `part3_benchmark_results` (si se activó y completó su guardado).
- Notebook 08: `part3_financial_overview`, `part3_financial_by_dimension`, `part3_financial_delay_bands`, `part3_financial_scenarios`, `part3_financial_assumptions`.

**Nota de reproducción:** indicar en `README.md` la versión de runtime y dependencias, el origen de cada tabla, permisos requeridos, orden de ejecución y las opciones `SAVE_TABLES` de cada notebook. El Notebook 09 no tiene todas sus salidas persistidas confirmadas.

## Validación

Los seis controles principales del Notebook 07 finalizaron en **PASS**. Entre ellos:

| Control | Resultado |
|---|---|
| Silver = Gold operativo en conteo | PASS |
| Gold con clima no supera población Gold operativa | PASS |
| Cobertura meteorológica | 99.9747% |
| Etiquetas binarias inconsistentes | 0 |
| Nulos en campos requeridos auditados | 0 |
| Duplicados en la clave de aeropuerto–hora meteorológica utilizada | 0 |

En auditorías adicionales de la población con clima se observaron **0 nulos en 12 predictores revisados**, **0 discrepancias de etiquetas** y **0 grupos duplicados bajo la clave aproximada utilizada**. Estos resultados no sustituyen una validación exhaustiva de la trazabilidad ni de disponibilidad temporal de variables.

La evaluación de modelos se hizo sobre periodos posteriores a los del entrenamiento. También se midieron diferencias por bootstrap y se revisaron errores en subgrupos y por gravedad.

## Resultados

### 1. Panorama histórico

| Indicador | Resultado |
|---|---:|
| Vuelos Gold operativos | 8,655,359 |
| Vuelos retrasados 15+ minutos | 1,959,157 |
| Tasa de retraso | 22.64% |
| Minutos positivos totales | 146,664,763 |
| Minutos de vuelos con retraso 15+ | 137,438,884 |
| Minutos 15+ con límite de 300 por vuelo | 128,309,351 |
| Vuelos con retraso superior a 10 h | 11,368 |
| Vuelos con retraso superior a 24 h | 394 |

### 2. Clasificación temporal (Notebook 05)

| Métrica | XGBoost baseline | XGBoost con clima histórico |
|---|---:|---:|
| ROC-AUC | 0.6758 | 0.6917 |
| PR-AUC | 0.3512 | 0.3808 |
| Brier calibrado | 0.1625 | 0.1595 |
| Precision | 32.94% | 35.53% |
| Recall | 58.11% | 55.19% |
| F1 | 0.4205 | 0.4323 |

El bootstrap por días produjo una diferencia de ROC-AUC a favor del modelo con clima de aproximadamente **0.0159**, con intervalo entre **0.0113 y 0.0206** en la evaluación realizada. La mejora es retrospectiva: la validez operacional de los datos de clima antes de la decisión no está confirmada.

### 3. Asociación entre vuelos sucesivos (Notebook 06)

| Indicador | Resultado |
|---|---:|
| Pares observados plausibles | 4,422,412 |
| Tasa siguiente con vuelo previo no retrasado | 14.50% |
| Tasa siguiente con vuelo previo retrasado | 48.11% |
| Diferencia absoluta | 33.61 puntos porcentuales |
| Riesgo relativo descriptivo | 3.32 veces |
| Estratos aeropuerto–mes | 690 |
| Diferencia ponderada dentro de estratos | 32.20 puntos porcentuales |
| Racha máxima observada de vuelos retrasados | 23 |

La cobertura de aeropuertos es parcial y el análisis es asociativo, no causal.

### 4. Rendimiento de consultas (Notebook 07)

| Método | Mediana en cinco ejecuciones |
|---|---:|
| Agregación directa sobre Gold | 0.597 s |
| Lectura de agregado materializado | 0.413 s |

La materialización inicial costó aproximadamente **3.74 s**. No es una comparación equivalente de motores ni una garantía universal de mejora.

### 5. Exposición económica hipotética (Notebook 08)

| Escenario | Exposición sobre 128,309,351 minutos |
|---|---:|
| Bajo: USD 11.0865/min | USD 1,422.5 millones |
| Central: USD 22.173/min | USD 2,845.0 millones |
| Alto: USD 44.346/min | USD 5,690.0 millones |

Con una **reducción hipotética de 10%** de los minutos elegibles, el escenario central generaría una diferencia aritmética de aproximadamente **USD 284.5 millones** en la exposición modelada. **No** se trata de un ahorro demostrado por una intervención real.

### 6. Estimación de duración (Notebook 09)

| Métrica temporal en test, solo vuelos retrasados | Baseline | Con clima histórico |
|---|---:|---:|
| MAE | 40.18 min | 40.07 min |
| RMSE | 59.47 min | 59.35 min |
| R² | 0.0177 | 0.0216 |

El bajo R² indica que las variables disponibles explican muy poco de la variación de la duración individual. Los ensayos adicionales con Random Forest, CatBoost y ensamble no resolvieron esta limitación. Por eso el regresor no se presenta como una estimación fiable de minutos exactos por vuelo.

### 7. Simulación probabilística

El motor Monte Carlo permite estudiar itinerarios de hasta cuatro tramos y tres escalas bajo reglas de duración programada, conexión mínima y reacomodo configurables. Produce percentiles y costos generalizados hipotéticos del pasajero. No pretende conocer las llegadas reales, la disponibilidad efectiva de vuelos alternativos ni las tarifas comerciales.

## Limitaciones

1. La población Gold está restringida a 30 aeropuertos de origen y no equivale a todos los vuelos de Estados Unidos.
2. La reducción entre datos BTS originales y Gold requiere una conciliación completa de exclusiones.
3. Las variables meteorológicas históricas aún no están verificadas como pronósticos disponibles en el instante de decisión.
4. El análisis de secuencia por matrícula tiene tramos omitidos y no demuestra causalidad de propagación.
5. La regresión de duración es débil en datos de prueba, especialmente para retrasos extremos.
6. Las tarifas por minuto y costos del pasajero son supuestos, no gastos auditados.
7. Monte Carlo utiliza distribuciones históricas y reglas simplificadas de conexiones; no integra un servicio predictivo online de XGBoost ni una cadena Gamma–BTS validada.
8. La conexión de la aplicación a Databricks consulta tablas actuales; **no hay una alimentación continua de vuelos en tiempo real**.
9. La reproducibilidad desde una instalación completamente limpia debe documentarse y probarse con datos de muestra, dependencias y permisos adecuados.

## Organización del equipo y contribuciones por entrega

### Integrantes y criterio de atribución

| Integrante | Identificador GitHub | Área principal acordada para la Parte 3 |
|---|---|---|
| José Miguel Guevara | `@GUEVARA21` | Evaluación predictiva: Notebook 05, comparación de XGBoost baseline/con clima, calibración y documentación de métricas |
| Santiago Nicolás Roldán Schroeder (Nico) | `@Skraeder` | Propagación y rotación de aeronaves: Notebook 06, continuidad de tramos, análisis de transiciones y discusión de límites |
| Diego García | `@diegogargo` | Calidad, rendimiento y consultas analíticas: Notebook 07, controles Silver/Gold, benchmark y SQL del dashboard |

**Nota sobre la forma real de colaboración:** la distribución anterior se definió en el plan de la Parte 3 como **responsabilidades de trabajo**, no como prueba de que cada notebook se ejecutó desde una cuenta distinta. Los tres colaboramos en el desarrollo, la revisión y la interpretación, y la ejecución conjunta de los notebooks avanzados y de las tablas Gold se concentró en la cuenta de Databricks de Santiago/Nico por motivos prácticos de acceso y continuidad. **La cuenta desde la que se ejecuta un notebook no acredita por sí sola quién escribió, revisó o explicó cada cambio.** De la misma manera, un commit de GitHub solamente prueba la identidad registrada en ese commit; no debe presentarse como evidencia de una revisión o ejecución inexistente. Para confirmar la autoría granular se deben contrastar archivos, historial real de GitHub, PRs e issues y los registros del equipo.

### Primer parcial — Definición, fuente y exploración (contribución conjunta)

Los tres integrantes participaron en la construcción de la base del proyecto: delimitación del problema de retrasos aéreos, selección y comprensión de la fuente BTS, discusión de la unidad de análisis por vuelo, exploración inicial de columnas y valores, organización del repositorio, documentación del alcance y planteamiento de la arquitectura. Esta etapa fue compartida: **no se atribuyen scripts o commits específicos a una persona sin verificarlos en el historial del repositorio**.

El resultado de esta etapa fue disponer de una pregunta medible, una fuente identificada, un esquema inicial de organización de datos y una ruta técnica para pasar de archivos históricos a análisis reproducibles.

### Segundo parcial — Construcción del pipeline (contribución conjunta)

El equipo trabajó en adquisición, limpieza y transformación de datos, estandarización de variables, separación por capas Bronze–Silver–Gold e integración de información meteorológica. Se desarrollaron datasets preparados para análisis con PySpark/Delta Lake y se discutieron reglas de calidad, alcance y escalabilidad. El trabajo de esta etapa corresponde a **la infraestructura compartida que consumen los Notebooks 05–09**.

En la documentación de las primeras dos entregas conviene vincular los commits, PRs e issues reales de cada integrante para precisar quién implementó cada archivo; esta versión del reporte **no inventa esa asignación detallada**.

### Tercer parcial — Distribución de responsabilidades y ejecución colaborativa

| Componente | Responsable de enfoque según el plan | Aporte y resultado del equipo | Estado/evidencia |
|---|---|---|---|
| Notebook 05 · clasificación XGBoost | José Miguel Guevara | Comparación baseline/weather, partición temporal, calibración, bootstrap, matriz de confusión, curvas y métricas; seis tablas de resultados | Resultados ejecutados y documentados; cotejar commits/PR con el plan |
| Notebook 06 · secuencias por matrícula | Santiago Nicolás Roldán | Identificación de 4,422,412 pares plausibles, transiciones, intervalos, estratos y rachas; advertencia de asociación, no causalidad | Resultados ejecutados y seis tablas Gold; cotejar commits/PR |
| Notebook 07 · auditoría y benchmark | Diego García | Controles de Silver/Gold, cobertura meteorológica, comparaciones de consulta directa y agregado materializado | Resultados del benchmark documentados; cotejar commits/PR |
| Notebook 08 · exposición financiera | Trabajo colaborativo de integración | Escenarios bajos/centrales/altos, tope de 300 min, agregaciones y cinco tablas Gold; costos ilustrativos | Cinco tablas guardadas y conciliaciones reportadas |
| Notebook 09 · duración y priorización | Trabajo colaborativo de experimentación | Reproducción de la clasificación, regresores y comparación con Random Forest/CatBoost/ensamble; análisis de debilidad de R² y priorización | Métricas ejecutadas; resultados experimentales no presentados como predicciones precisas |
| Simulador Monte Carlo e integración | Trabajo colaborativo de cierre | Escenarios históricos de rutas, reglas editables de escalas y costos del pasajero, visualización en aplicación | Prototipo funcional; **no** es integración validada de XGBoost–Gamma–BTS |
| Informe, interpretación y presentación | Los tres | Revisión de hallazgos, limitaciones, documentación y comunicación de resultados | Reporte y reflexiones; evidencia de GitHub por completar con enlaces reales |

En el plan se contemplaron también una validación independiente de resultados de rotación (`06b_rotation_results_validation.py`) asociada a José Miguel y la actualización de auditoría/SQL asociada a Diego. Se distinguen de los notebooks y análisis **efectivamente ejecutados**: su asignación no demuestra por sí sola que exista un PR o una revisión aprobada.

### Aportaciones individuales y aprendizaje

Las reflexiones individuales se entregan en:

- [`docs/reflections/nico-roldan.md`](reflections/nico-roldan.md)
- [`docs/reflections/diego-garcia.md`](reflections/diego-garcia.md)
- [`docs/reflections/jose-miguel-guevara.md`](reflections/jose-miguel-guevara.md)

Estos textos reflejan la división de responsabilidades acordada y el trabajo compartido. **Cada integrante debe revisar su reflexión en primera persona y corregir cualquier detalle que no represente su participación real.**

### Evidencia pendiente de registrar, sin fabricar historial

Antes de la entrega conviene añadir enlaces a los commits, PRs, revisiones e issues realmente existentes de las etapas 1, 2 y 3; la rúbrica los evalúa por integrante. El hecho de trabajar desde la misma cuenta de Databricks es compatible con colaboración real, pero no sustituye por sí solo la evidencia personal exigida en GitHub.

## Conclusiones

Se logró consolidar una población operativa de millones de vuelos en una arquitectura analítica, evaluar clasificadores con separación temporal y desarrollar análisis de asociaciones entre vuelos sucesivos, rendimiento de consultas y exposición económica hipotética.

El clasificador XGBoost con datos meteorológicos históricos mostró mejor discriminación que el baseline en la prueba realizada, aunque su uso operacional requeriría pronósticos auténticos disponibles antes del vuelo. El análisis por matrícula identificó asociaciones fuertes entre retrasos consecutivos, sin probar un mecanismo causal. La comparación entre consultas directas y agregados materializados mostró una mejora de latencia en el caso medido, considerando su costo de construcción.

La regresión de duración no logró precisión suficiente para una promesa de predicción individual. En vez de ocultar este resultado, se documenta como evidencia de que faltan variables operativas importantes. Los escenarios financieros y Monte Carlo son útiles para explorar sensibilidad, siempre que los supuestos sean transparentes.

En conjunto, el proyecto ilustra cómo **volumen, variedad, velocidad de procesamiento, veracidad y valor** se relacionan en una solución de Datos Masivos, con límites explícitos respecto a actualización en tiempo real y precisión predictiva.

### Conclusiones individuales del equipo (síntesis)

**Santiago Nicolás Roldán Schroeder (Nico).** El estudio de secuencias de aeronaves y los experimentos de integración mostraron que un análisis de datos masivos requiere interpretar los resultados más allá de sus cifras. Una asociación de retrasos no equivale a causalidad y una simulación sólo es útil si se distinguen datos observados de supuestos. El aprendizaje principal fue la importancia de verificar cada transformación y documentar las limitaciones para tomar decisiones con responsabilidad.

**Diego García.** La confiabilidad de los análisis depende de la preparación y validación de la base. El enfoque de auditoría Silver/Gold y benchmark permitió reconocer que calidad y eficiencia deben medirse con controles y pruebas reproducibles, no con afirmaciones generales sobre una tecnología. El aprendizaje central fue que una conclusión útil debe poder rastrearse hasta sus fuentes, sus reglas de procesamiento y sus consultas.

**José Miguel Guevara.** La evaluación de XGBoost permitió comprobar que la calidad de un modelo no se resume en su accuracy. Separar el conjunto de entrenamiento de la prueba, calibrar probabilidades y analizar precision, recall y ROC/PR-AUC resultó esencial para entender el alcance de las predicciones. El aprendizaje más importante fue comunicar tanto las mejoras observadas como las condiciones que impiden trasladarlas directamente a un sistema operativo en tiempo real.

Estas síntesis corresponden a los enfoques de contribución acordados por el equipo; cada integrante debe validarlas y complementarlas con su reflexión individual antes de publicar.

## Trabajo futuro

- Conciliar formalmente todos los registros desde la fuente BTS hasta Gold, con filtros y motivos de exclusión.
- Probar la ejecución completa desde una instalación limpia y registrar versiones, tiempos y resultados.
- Incorporar pronósticos meteorológicos realmente disponibles antes de la salida y evaluar otra vez la clasificación.
- Añadir características operativas válidas temporalmente: congestión conocida, historial de aerolínea–ruta y estado conocido de aeronaves.
- Recuperar y validar los datos de llegada y horarios reales antes de calibrar conexiones perdidas y recuperación durante vuelo.
- Validar el simulador mediante evaluación temporal y análisis de sensibilidad de la dependencia entre tramos.
- Definir un pipeline de actualización e ingestión incremental para futuras fuentes de vuelos.
- Documentar decisiones, revisiones de código, *issues* y contribuciones individuales en GitHub.

## Evidencia y trazabilidad en el repositorio

- **Código:** notebooks `05_advanced_model_evaluation`, `06_aircraft_delay_propagation`, `07_pipeline_quality_and_benchmark`, `08_financial_delay_impact` y `09_delay_severity_and_decision_prioritization`, además del pipeline original.
- **Datos analíticos:** tablas Unity Catalog bajo `workspace.gold_flights`.
- **Calidad:** controles de Notebook 07 y auditorías complementarias.
- **Resultados para entrega:** colocar exportaciones verificadas en `reports/figures/` y `reports/tables/` (no solo capturas del dashboard).
- **Reproducción:** describir en `README.md` configuración, fuentes, dependencias, comandos/notebooks, permisos y resultados esperados.
- **Colaboración:** vincular Pull Requests, commits, revisiones, issues y reflexiones individuales reales.

> **Antes de entregar:** completar integrantes, vínculos a evidencias, instrucciones reproducibles probadas y figuras/tablas exportadas. No publicar campos pendientes como si fueran validaciones concluidas.
