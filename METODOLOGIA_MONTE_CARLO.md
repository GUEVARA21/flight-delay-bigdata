# Nota metodológica — Itinerarios con Monte Carlo

## Pregunta

> Bajo los patrones históricos de retraso de salida por ruta y un conjunto de supuestos de tiempo y reacomodo, ¿cómo se distribuiría el atraso final de un pasajero y su costo económico generalizado en un itinerario con escalas?

**No** responde a la hora exacta de llegada ni al precio real de una reserva. Tampoco prueba una causa de propagación de retrasos.

## 1. Datos observados

Se consulta `workspace.gold_flights.flights_features` (2024–2025) usando las columnas `origin`, `dest`, `airline`, `month`, `departure_hour`, `flight_date`, `tail_number`, `scheduled_departure_utc` y `dep_delay`.

El algoritmo muestra un itinerario solo si cada par origen–destino existe en el catálogo con al menos 20 registros históricos. Por tramo intenta obtener una muestra con estas reglas, deteniéndose en el primer nivel con ≥40 registros o usando la ruta completa como último respaldo:

1. Ruta + aerolínea + mes + hora de salida.
2. Ruta + aerolínea + mes.
3. Ruta + aerolínea.
4. Ruta completa.

Se limita la muestra a 1,800 filas ordenadas con un hash determinístico de variables del vuelo. Se emplea una transformación de sensibilidad: `D = min(max(dep_delay, 0), 300)` minutos. Por lo tanto, se pierde la cola >300 y las salidas tempranas pasan a cero. **No** es la distribución completa e íntegra de retrasos originales.

La etiqueta «aerolínea Todas» desactiva ese filtro. Los filtros que se relajan quedan documentados en la tabla «Datos que utilizó la simulación».

## 2. Mecanismo simulado

Sea `i = 1,...,K` con 1 ≤ K ≤ 4.

- `S_i`: salida programada del tramo `i` en minutos relativos al primer despegue.
- `B_i`: duración programada del vuelo, **introducida por el usuario**.
- `L_i`: escala programada después del tramo `i`, **introducida por el usuario**.
- `M_i`: tiempo mínimo requerido para alcanzar la conexión, **introducido por el usuario**.
- `D_i`: retraso de salida simulado, tomado con reemplazo de observaciones reales del tramo.

Los horarios programados se construyen como `S_1 = 0` y `S_{i+1} = S_i + B_i + L_i`. Esto es un eje temporal relativo que **no intenta inferir husos horarios**.

El vuelo inicial llega en `A_1 = S_1 + D_1 + B_1`.

Para cada tramo posterior, el siguiente vuelo habría salido en `O_i = S_i + D_i`. La conexión se considera alcanzada si `A_{i−1} + M_{i−1} ≤ O_i`. Si no se alcanza, el pasajero toma una salida de reemplazo **hipotética**:

`O_i' = max(O_i, A_{i−1} + M_{i−1}) + R`

con `R` = espera de reacomodo editada por el usuario. Si se alcanza, `O_i' = O_i`. En ambos casos `A_i = O_i' + B_i`.

**No tenemos** registros de llegadas reales, asientos de reemplazo ni ventanas oficiales de conexión. Por eso esta regla es un escenario deliberadamente simplificado, no una predicción de conexiones perdidas.

La demora final relativa a la llegada programada es

`T = max(0, A_K − (S_K + B_K))`.

Se repite de 1,000 a 20,000 veces con semilla fija 2026. Se reportan `E[T]`, mediana, P90, P95, P(T≥15), P(T≥60), P(T≥180) y la fracción de simulaciones donde se perdió alguna conexión.

## 3. Dependencia entre tramos

Con parámetro 0, los percentiles de retraso por tramo se extraen de manera independiente. Con valores superiores, algunas iteraciones comparten un mismo percentil aleatorio para representar choques comunes. **Esto no equivale a calibrar una cópula ni a reproducir el riesgo relativo 3.32× del Notebook 06**. Su fin es permitir análisis de sensibilidad ante dependencias no identificadas.

## 4. Costos editables (MXN)

Sea `F` el costo hipotético de todo el boleto, `C` comida si `T≥60`, `V` transporte si se pierde alguna conexión, `H` hospedaje si `T≥360`, `J` cargo supuesto por conexión perdida, y `W` valor del tiempo por hora. Para cada simulación:

`Desembolso = F + C·1[T≥60] + V·1[conexión perdida] + H·1[T≥360] + J·(# conexiones perdidas)`.

`Tiempo económico = W·T/60`.

`Costo generalizado = Desembolso + Tiempo económico`.

Los indicadores que se muestran son **promedios de escenarios**; el valor del tiempo no es dinero necesariamente desembolsado. Las cifras no son cotizaciones de pasajes ni pagos efectivos. El costo hipotético en USD del Notebook 08 se mantiene separado del costo del pasajero en MXN.

## 5. Relación con los notebooks anteriores

- **05:** el XGBoost Classifier fue evaluado, pero sus resultados no son una inferencia desplegada para nuevos itinerarios. La V6 no lo finge.
- **06:** evidenció asociación histórica entre el retraso anterior de una matrícula y el siguiente; **no** utilizamos 3.32× como multiplicador causal en este simulador.
- **07:** auditoría de Silver/Gold y benchmark; explica la procedencia de la base y su consistencia parcial.
- **08:** escenarios de costo de aerolínea en USD; son distintos de los costos del pasajero.
- **09:** regresor de duración con R² cercano a 0.02; no es una base suficientemente validada para prometer el número de minutos que llegará tarde un vuelo específico.

## 6. Limitaciones y siguientes pasos técnicos

Para convertir este prototipo en una solución operativa sería necesario: itinerarios oficiales y horarios UTC de llegada/salida, mínimo de conexión por aeropuerto y terminal, registros de maletas/conexiones, disponibilidad de vuelos de reemplazo, calibración histórica de dependencias entre tramos y validación temporal con datos de llegadas. También harían falta modelos versionados e inferencia usando solo variables disponibles en el instante de decisión.
