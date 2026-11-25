# Estado de implementación y límites de evidencia

Esta entrega es un versión 0.1 y separa dos rutas ejecutables:

1. **ModelArena**: núcleo genérico nuevo con contratos Pydantic, registro de modelos, datos tabulares, HPO explícito, CV, selección por entidad, fit/predict, curvas, guardado y publicación local/S3. Incluye TFT compacto de dimensiones de objetivos/features configurables y baseline de cuantiles históricos.
2. **MacProject**: adaptador de migración con etapas explícitas que conserva el motor contable y el dashboard Athena de 0.13.1. Es la ruta del notebook ADA para preservar el caso de uso existente. Su implementación todavía contiene código de compatibilidad aislado; no se afirma que el perfil ya esté reescrito íntegramente sobre el core nuevo.

El adaptador separa tune, select, fit, predict, postprocess y publish; el ganador de cada serie se conserva como configuración única en F1/F2/F3/FUTURO. Las curvas históricas usan los pesos de cada corte, no pesos finales retrospectivos. F1–F3 siguen siendo evidencia utilizada en selección.

El ajuste final es de 75 épocas. Se conserva TPESampler y el espacio configurado del perfil. Las nuevas configuraciones ADA usan 15 épocas máximas por ajuste HPO y FoldPatiencePruner con paciencia 4 sobre validación fija; el primer trial de cada input completa los folds como referencia. Configuraciones históricas sin pruner siguen con NopPruner. El monitor por época cubre todas las entrenables, con filas por cuenta/objetivo. Los budgets siguen siendo la segmentación por entropía existente. Las atípicas conservan zero-shot para el ajuste y se evalúan como entrenables en el contrato de alertamiento.

`obs_training_entities` añade la evidencia desglosada de aprendizaje sin cambiar los esquemas antiguos de entrenamiento. Total, budget y cuenta se consultan agregando sus filas en Athena. La evaluación completa por época aumenta tiempo, disco y volumen publicado; no limita las cuentas entrenadas.

Los modelos de compatibilidad están en un recurso interno y se extraen a un proyecto local autocontenido. La copia de código se verifica; la configuración y el calendario se identifican por hash. Los notebooks permanecen editables y no son parte de esa validación de integridad.

## Pendiente de acreditar en ADA

- Ejecución real con el rol y las tablas Athena de la instalación destino.
- Publicación real de los nuevos datasets y verificación de permisos.
- Rendimiento y memoria con el universo completo y la evaluación por época de todas las cuentas.
- Disponibilidad del wheel en Artifactory y compatibilidad con el runtime CUDA aprobado.

## Límites de la API inicial

- El core genérico selecciona finalistas por input global; la selección por budget/divisa del proyecto está conservada en MacProject.
- Los adaptadores nuevos de arquitecturas o fuentes requieren registro e implementación; no basta cambiar un nombre en YAML.
- Los checkpoints del core son artefactos de inferencia. No se anuncia reanudación exacta de optimizador a mitad de época.
- `MacProject.resume()` reutiliza etapas y checkpoints completos compatibles; un ajuste interrumpido sin checkpoint completo se reinicia desde cero. La publicación continúa siendo explícita.
- La publicación genérica S3 es por manifiestos. El perfil MAC mantiene los layouts y las tablas Athena específicas; no se afirma equivalencia entre ambos formatos.
