# Reflexión individual — Santiago Nicolás Roldán Schroeder (Nico)

**Usuario GitHub:** `@Skraeder`  
**Proyecto:** Flight Delay Big Data — Datos Masivos

> **Nota de atribución:** En las entregas 1 y 2 hubo trabajo conjunto sobre el mismo repositorio y pipeline. Para la Parte 3 se acordaron responsabilidades por componente, pero las ejecuciones colaborativas en Databricks se realizaron desde la cuenta de Nico. Esta reflexión describe el enfoque de participación del integrante y debe ser validada por él antes de presentarse como testimonio personal. No atribuye commits, PRs o revisiones que no se hayan comprobado.


## ¿Cuál fue mi contribución principal?

### Entregas anteriores

Durante la primera entrega participé en el planteamiento del problema de retrasos aéreos, el análisis inicial de la información de BTS, la delimitación de la pregunta y la propuesta de arquitectura. Para la segunda entrega colaboré con el equipo en comprender cómo quedaría organizada la información dentro del proceso Bronze–Silver–Gold y qué controles y variables necesitaríamos para los análisis siguientes. Estas tareas fueron compartidas entre los tres integrantes.

### Parte 3

Mi área acordada fue el **Notebook 06: propagación y rotación de aeronaves**. Este módulo construyó pares de vuelos plausibles a partir de `tail_number`, continuidad entre destino y origen y una ventana entre salidas programadas. El análisis permitió estudiar 4,422,412 pares, comparar una frecuencia de retraso siguiente de 14.50% frente a 48.11% según el estado previo y examinar estratos aeropuerto–mes, intervalos y rachas. Mi responsabilidad de enfoque fue interpretar los resultados sin confundir una asociación descriptiva con una relación causal.

También participé en la ejecución y comprobación conjunta de la Parte 3 en mi entorno de Databricks, que fue el utilizado por el equipo para desarrollar, probar y guardar los resultados avanzados. Esto incluyó revisar los cálculos del módulo financiero (Notebook 08), investigar las limitaciones de la regresión de duración (Notebook 09) y apoyar la integración analítica en la aplicación. La cuenta compartida como entorno de ejecución no implica que haya realizado individualmente todas las aportaciones del equipo.

## ¿Qué decisión técnica tomé?

Trabajamos con una definición conservadora de conexión entre vuelos observados: no bastaba coincidir en la matrícula; exigimos continuidad de aeropuerto, aerolínea y separación programada plausible. También mantuvimos claramente separadas las métricas históricas, los supuestos financieros y los resultados de Monte Carlo. Preferimos describir lo que los datos sostienen antes que afirmar efectos causales o ahorros garantizados.

## ¿Qué problema enfrenté?

El principal problema fue que la base Gold no representa el itinerario completo de cada aeronave: contiene vuelos de un conjunto de aeropuertos seleccionado y carece de ciertas variables de llegada necesarias para reconstruir escalas reales. Además, en los experimentos de duración observamos un R² cercano a 0.02, insuficiente para estimar minutos exactos con confianza. Esto nos obligó a revisar el alcance de nuestras conclusiones.

## ¿Qué aprendí de Git y GitHub?

Aprendí la diferencia entre editar notebooks en Databricks, mantener una rama de desarrollo, hacer commits y preparar una integración mediante Pull Requests. También entendí que los cambios y revisiones deben dejar evidencia real en el repositorio: ejecutar algo desde una cuenta no equivale automáticamente a registrar quién colaboró o revisó cada parte.

## ¿Qué aprendí sobre trabajo colaborativo?

Compartir el mismo entorno facilitó dar continuidad a las pruebas de la Parte 3, pero también hizo más importante documentar responsabilidades y decisiones. Aprendí a distinguir entre el responsable de cada investigación y el usuario con el que se ejecutó el código. La revisión cruzada y una explicación compartida de los resultados fueron necesarias para conectar modelos, propagación, ingeniería y finanzas.

## ¿Qué parte mejoraría?

Me gustaría auditar de forma completa la reducción entre la fuente original y los registros Gold, incorporar datos de llegada y operación verdaderamente disponibles antes del vuelo y validar las simulaciones con itinerarios y dependencias temporales más completos. También mejoraría el registro del trabajo individual en GitHub y la reproducibilidad del flujo completo.

## ¿Qué contribución de otro integrante fue importante?

La investigación del Notebook 05 asignada a José Miguel aportó métricas predictivas y calibración que sirvieron para comprender qué es posible anticipar. El trabajo de calidad y benchmark asignado a Diego fue complementario porque las conclusiones pierden valor si las tablas no son consistentes o las consultas no son reproducibles.

## Conclusión personal

El proyecto me enseñó que el reto más difícil en datos masivos no es únicamente procesar millones de registros, sino convertirlos en resultados bien definidos y verificables. Nuestro análisis de propagación encontró patrones relevantes, pero también mostró que un número llamativo no sustituye una interpretación metodológica correcta. Me llevo el aprendizaje de construir, probar y reconocer los límites de cada herramienta antes de proponer decisiones basadas en ella.
