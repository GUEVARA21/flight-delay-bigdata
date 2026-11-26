# Flight Delay Intelligence · V6 — Network Scenario Studio

Aplicación Streamlit con **consultas SQL en vivo** sobre `workspace.gold_flights`, análisis histórico y simulador Monte Carlo de itinerarios con hasta tres escalas (cuatro tramos). Es la evolución directa de la V5: conserva sus páginas y su conexión OAuth **On-Behalf-Of**, y sustituye **Route Simulator** por **Network Simulator**.

## Archivos

- **`app.py`**: aplicación completa. Incluye consultas SQL, el simulador y el mapa vectorial con avión animado. El mapa del simulador **no utiliza Mapbox, MapTiler, CARTO ni claves de API**. El dibujo de costas/estados está incorporado en el propio código.
- **`requirements.txt`**: dependencias de Databricks Apps, Python 3.11. Conserva `airportsdata==20250523`.
- **`app.yaml`**: configuración de referencia para ejecutar `streamlit run app.py` y vincular el recurso existente `sql-warehouse`.
- **`tests/test_monte_carlo.py`**: pruebas locales del cálculo y mapa offline.
- **`METODOLOGIA_MONTE_CARLO.md`**: explicación detallada del método y sus supuestos.

## Instalación sobre tu aplicación actual

1. En Databricks Apps, abre `flight-delay-intelligence-dm` y ubica **Source → View source**. Entra al archivo **`app.py` original (sin fecha)** dentro de `streamlit-hello-world-app`.
2. Guarda una copia de seguridad del `app.py` que ya funciona. Pega el contenido del `app.py` de este proyecto y guarda (`Ctrl+S`). **No uses `Run` del editor de Python**: ese botón ejecuta fuera del entorno de Streamlit.
3. Compara el `requirements.txt` original con el que viene aquí. Si ya contiene las librerías necesarias (`numpy`, `plotly`, `pydeck`, `airportsdata==20250523`, `streamlit`, conectores Databricks), puedes dejarlo igual; de lo contrario, sustituye el archivo original por éste.
4. **No modifiques** el `app.yaml`, la autorización `sql`, ni la conexión al **Serverless Starter Warehouse** si la V5 ya funcionaba.
5. Regresa a **Databricks Apps → Overview → Deploy**. Revisa la instalación de paquetes y la conexión. Abre el enlace de la aplicación y elige **Network Simulator**.
6. Prueba un itinerario **ATL → DFW → LAX** si aparece en el catálogo de rutas; elige dos tramos, ajusta duración programada y escala, y pulsa **Ejecutar análisis de itinerario**.

## Qué introduce la V6

1. **Itinerarios reales en el catálogo histórico Gold**: origen y destino se seleccionan mediante pares que existen en `flights_features` (al menos 20 vuelos por par); las escalas enlazan por aeropuerto y se evita repetir un aeropuerto en un mismo recorrido.
2. **Hasta cuatro tramos**: posibilidad de elegir aerolínea por tramo y un tiempo programado de vuelo, escala y mínimo de conexión. Los datos de duración programada de vuelo se introducen como *supuestos*, no se recuperan de una columna de llegada real (no disponible en la Gold documentada).
3. **Muestreo histórico por tramo**: selección de `dep_delay` real con filtros por ruta, aerolínea, mes y hora; si el subgrupo no llega a 40 registros, el sistema relaja filtros siguiendo una jerarquía visible. Cada tramo usa hasta 1,800 observaciones seleccionadas por hash reproducible. Todos los retrasos negativos se llevan a cero y la severidad máxima se topa en 300 minutos por vuelo.
4. **Monte Carlo**: de 1,000 a 20,000 escenarios. Simula demora de llegada bajo la duración programada indicada; verifica si se alcanza cada conexión usando el margen mínimo configurado. Si se pierde una conexión, calcula una salida alternativa con un tiempo de espera hipotético, no un horario de una aerolínea.
5. **Medición de resultados**: retraso medio, mediana, P90/P95, tasas de demoras de llegada ≥15/60/180 minutos y probabilidad de perder al menos una conexión. Son **resultados del escenario**, no tasas predichas/validadas contra vuelos actuales.
6. **Costos ilustrativos en MXN**: precio del itinerario, comida, transporte, hospedaje, posible cargo de reacomodo y valor económico del tiempo. Se distingue el **desembolso esperado** del **costo generalizado**.
7. **Animación sin claves**: fondo geográfico vectorial local; arco del trayecto con avión SVG animado. No hay solicitudes a proveedores de teselas ni marcas de agua `API KEY REQUIRED`. La animación no muestra un avión real en seguimiento.
8. **Exportación CSV**: demoras finales simuladas y número de conexiones perdidas por escenario, sin datos personales.

## ¿Usa el XGBoost entrenado?

**Todavía no como predictor de cada vuelo nuevo.** Los Notebooks 05 y 09 generaron métricas, tablas y predicciones de test, pero aquí no hemos verificado la disponibilidad de un artefacto de modelo serializado, versionado y desplegado para inferencia. La V6 utiliza **registros históricos reales** y filtra con variables que también usamos al entrenar (`origin`, `dest`, `airline`, `month`, `departure_hour`). El regresor del Notebook 09 sigue siendo experimental (R² ~0.02) y **no se presenta como una predicción fiable de duración individual**.

La correlación entre tramos, si el usuario activa su control, es un **supuesto de sensibilidad**, no un parámetro causal estimado a partir del Notebook 06. Con independencia (0.0), los tramos se muestrean por separado.

## Alcance y seguridad

- Datos históricos de **2024–2025** de una selección de **30 aeropuertos de origen**.
- Solo lectura mediante Databricks SQL Warehouse y el token OAuth `x-forwarded-access-token` administrado por Databricks Apps.
- No se guardan ni modifican tablas Gold. No se colocan contraseñas, tokens ni claves de mapas en los archivos.
- Los datos de horarios de llegada, tiempos mínimos oficiales, tarifas, reacomodos reales y disponibilidad de asientos **no están documentados** en nuestra Gold actual; se especifican o aproximan con supuestos del escenario.
- Nunca afirmar que las cifras de Monte Carlo son probabilidades reales de perder vuelos o gastos efectivos del pasajero. Son una evaluación hipotética de itinerario bajo un mecanismo definido.
- Las demás secciones de V5 (Overview, Airport Map, Operations, Predictive Models, Propagation, Financial Impact, Operational Priorities y Data Reliability) se conservan.

## Verificaciones ejecutadas

- Compilación de `app.py` mediante `python -m py_compile`.
- Ocho pruebas locales `pytest` del motor Monte Carlo y del mapa offline (incluyen valores extremos, conexión alcanzada/perdida, consistencia de costos, reproducibilidad y ausencia de solicitudes externas en el SVG).
- Se generó una vista previa estática del mapa SVG y se inspeccionó visualmente.
- **Pendiente:** prueba de integración de la V6 contra tu SQL Warehouse y despliegue real de Databricks Apps. No tenemos acceso directo a tu sesión para realizarlos desde aquí.

## Fuentes y documentación técnica

- [Databricks Apps: dependencias](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/dependencies)
- [Databricks Apps: autorización](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/auth)
- [Streamlit: componentes HTML](https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.html)
- Costa y contorno de estados: datos geográficos simplificados a partir de Basemap (GSHHS/US boundaries) preparados e incorporados en código, sin teselas remotas.
