# Reflexión individual — José Miguel Guevara

**Usuario GitHub:** `@GUEVARA21`  
**Proyecto:** Flight Delay Big Data — Datos Masivos

> **Nota de atribución:** En las entregas 1 y 2 hubo trabajo conjunto sobre el mismo repositorio y pipeline. Para la Parte 3 se acordaron responsabilidades por componente, pero las ejecuciones colaborativas en Databricks se realizaron desde la cuenta de Nico. Esta reflexión describe el enfoque de participación del integrante y debe ser validada por él antes de presentarse como testimonio personal. No atribuye commits, PRs o revisiones que no se hayan comprobado.


## ¿Cuál fue mi contribución principal?

### Entregas anteriores

Durante la primera entrega participé junto con el equipo en definir la pregunta sobre retrasos aéreos, seleccionar la fuente BTS y establecer el alcance y la documentación inicial. En la segunda colaboré en discutir el pipeline de limpieza y transformación, la arquitectura Bronze–Silver–Gold y la preparación de variables necesarias para los modelos. La contribución de estas etapas fue compartida; los detalles de quién incorporó cada archivo se pueden respaldar con el historial real de GitHub.

### Parte 3

Mi responsabilidad principal según el plan fue **el Notebook 05: evaluación avanzada de XGBoost**. Este análisis comparó un modelo baseline de variables operativas con otro que incorporaba meteorología histórica; utilizó un conjunto temporal de entrenamiento, calibración, selección de umbral y prueba. También empleó métricas ROC-AUC, PR-AUC, precision, recall, Brier y bootstrap por bloques de día para interpretar la incertidumbre. El modelo con clima alcanzó ROC-AUC 0.6917 en test, pero su uso como predicción anticipada aún requiere validar la disponibilidad temporal de los datos meteorológicos.

Otro frente previsto en la división de trabajo fue la revisión independiente de resultados de rotación (archivo `06b_rotation_results_validation.py`). Distingo esa responsabilidad planificada de una revisión aprobada en GitHub: solo debemos registrar esta última si existe evidencia real de ejecución o PR. Los tres participamos en la evaluación de resultados de la Parte 3 desde el entorno compartido de Databricks de Nico y en el cierre del análisis y de su documentación.

## ¿Qué decisión técnica tomé?

Dentro de mi enfoque, una decisión central fue preferir una evaluación temporal en lugar de una división aleatoria que pudiera sobreestimar el desempeño. También fue importante calibrar las probabilidades antes de interpretarlas y elegir el umbral de alerta sin utilizar el conjunto final de prueba. Diferenciamos métricas de discriminación, como ROC-AUC, de métricas operativas de decisiones, como precision y recall.

## ¿Qué problema enfrenté?

El principal reto fue reconocer que, aunque el modelo con clima mejora varias métricas, el desempeño sigue siendo moderado. Además, usar clima observado después de los hechos no garantiza que esas variables estuvieran disponibles antes del vuelo; por eso sería incorrecto afirmar que tenemos un sistema predictivo en tiempo real completamente validado. Otra dificultad fue que predecir duración es mucho más complejo que clasificar retrasos, como mostraron los experimentos posteriores del equipo.

## ¿Qué aprendí de Git y GitHub?

Aprendí a organizar el trabajo por ramas y componentes, diferenciar el desarrollo de un notebook de su publicación en una rama estable y documentar resultados para que otra persona pueda revisarlos. También comprendí que trabajar sobre una misma cuenta de ejecución no debe ocultar qué responsabilidades tuvo cada integrante, y que las revisiones deben corresponder a acciones reales registradas en GitHub.

## ¿Qué aprendí sobre trabajo colaborativo?

El trabajo de modelos necesitaba datos limpios y variables consistentes, por lo que dependía de la arquitectura compartida de las primeras entregas y de los controles del área de Diego. A su vez, la interpretación del modelo se complementó con el análisis de secuencias de Nico: ambas investigaciones responden preguntas diferentes y no deben confundirse como si una demostrara la causalidad de la otra.

## ¿Qué parte mejoraría?

Validaría el modelo con pronósticos meteorológicos realmente disponibles antes de la salida, incorporaría variables operativas no filtradas del futuro y vigilaría su desempeño cuando cambien rutas, frecuencias y aerolíneas. También ampliaría la validación por subgrupos y documentaría una ejecución de entrenamiento completamente reproducible desde una base limpia.

## ¿Qué contribución de otro integrante fue importante?

El trabajo de Nico en propagación ayudó a contextualizar el riesgo de retraso dentro de secuencias de vuelos y a reconocer la diferencia entre asociación y causalidad. Las pruebas de calidad y rendimiento del enfoque de Diego aportaron controles necesarios para confiar en la población y en las consultas que soportan los modelos.

## Conclusión personal

Mi aprendizaje principal fue que la calidad de un modelo no puede resumirse en un solo porcentaje. Una evaluación sólida exige separar correctamente los periodos, calibrar probabilidades, comparar contra referencias y estudiar los errores. El proyecto mostró que Machine Learning aporta evidencia útil sobre el riesgo de retraso, pero también que debemos comunicar con claridad lo que todavía no puede anticiparse de forma confiable.
