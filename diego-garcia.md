# Reflexión individual — Diego García

**Usuario GitHub:** `@diegogargo`  
**Proyecto:** Flight Delay Big Data — Datos Masivos

> **Nota de atribución:** En las entregas 1 y 2 hubo trabajo conjunto sobre el mismo repositorio y pipeline. Para la Parte 3 se acordaron responsabilidades por componente, pero las ejecuciones colaborativas en Databricks se realizaron desde la cuenta de Nico. Esta reflexión describe el enfoque de participación del integrante y debe ser validada por él antes de presentarse como testimonio personal. No atribuye commits, PRs o revisiones que no se hayan comprobado.


## ¿Cuál fue mi contribución principal?

### Entregas anteriores

Durante la primera entrega participé con el equipo en delimitar el alcance, revisar las características de la fuente BTS y organizar la base conceptual del repositorio. En la segunda colaboré en la construcción y discusión del pipeline de ingestión, limpieza y transformación hacia las capas Bronze, Silver y Gold. Como el trabajo inicial fue conjunto, prefiero no atribuirme exclusivamente scripts específicos sin contrastarlos con el historial de GitHub.

### Parte 3

Mi responsabilidad acordada fue **el Notebook 07: auditoría de calidad y benchmarking**, así como la preparación de consultas analíticas para comunicar resultados. Este frente buscó comprobar que las tablas Silver y Gold conservaran recuentos consistentes, que las etiquetas de retraso fueran válidas, que los campos requeridos estuvieran completos y que la información meteorológica estuviera correctamente relacionada. También evaluamos el tiempo de una agregación directa frente a una tabla Delta materializada para entender el costo y la conveniencia de precalcular resultados recurrentes.

Participé con mis compañeros en la revisión general de la Parte 3 y en la interpretación de métricas y visualizaciones, trabajando sobre los resultados compartidos del entorno de Databricks utilizado por Nico. Las tareas de los Notebooks 08 y 09, así como la integración del simulador, se consideran colaborativas y no deben atribuirse a una sola persona basándose en la cuenta de ejecución.

## ¿Qué decisión técnica tomé?

Dentro de mi responsabilidad de trabajo, defendí que la confiabilidad del dataset debía comprobarse con reglas explícitas antes de interpretar los modelos. Para evaluar rendimiento, también era importante comparar operaciones concretas y registrar el costo de crear la tabla materializada, en lugar de afirmar de manera general que una tecnología siempre es más rápida que otra.

## ¿Qué problema enfrenté?

El reto fue asegurar que la base operativa y sus tablas derivadas representaran la misma población cuando correspondía, y detectar cuándo un join meteorológico podía generar duplicados o filas faltantes. También hubo dificultades de permisos y visibilidad entre entornos de Databricks. Aprendimos a investigar el acceso y la ubicación real de las tablas antes de modificar consultas o replicar datos.

## ¿Qué aprendí de Git y GitHub?

Aprendí que GitHub debe mostrar cambios identificables, revisiones y trazabilidad, no solamente versiones finales de archivos. En un proyecto donde las ejecuciones se concentran en una cuenta de Databricks, documentar la persona responsable de cada análisis y contrastar las aportaciones con los commits y PRs reales es especialmente importante.

## ¿Qué aprendí sobre trabajo colaborativo?

Entendí que el modelo predictivo, la investigación de propagación y el análisis financiero dependen de una base preparada y validada. Mi área se complementó con los otros dos enfoques: una métrica puede ser correcta matemáticamente y aun así no ser útil si la fuente tiene errores o no corresponde al periodo que se quiere estudiar.

## ¿Qué parte mejoraría?

Automatizaría más controles del pipeline, añadiría verificaciones de claves y conteos por etapa, documentaría completamente los filtros desde BTS hasta Gold y probaría un proceso de reejecución incremental. También dejaría comparativas de rendimiento reproducibles con condiciones de medición y actualización de tablas bien identificadas.

## ¿Qué contribución de otro integrante fue importante?

El Notebook 05 asignado a José Miguel dio un criterio cuantitativo para evaluar las predicciones y evitó valorar los modelos únicamente por accuracy. El Notebook 06 asignado a Nico mostró por qué la estructura y continuidad de los registros son esenciales para interpretar el comportamiento de una aeronave entre vuelos.

## Conclusión personal

El aprendizaje más importante fue que un sistema de datos masivos necesita una base verificable y una ruta clara de procesamiento, no solo modelos atractivos. Las validaciones y las mediciones de rendimiento ayudan a detectar errores y a asegurar que la información utilizada en las conclusiones sea consistente. Un resultado confiable es aquel cuyo origen y cálculo pueden explicarse y reproducirse.
