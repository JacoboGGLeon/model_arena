# ADR-0001 — Model Arena y Observatory: API y ciclo de vida

| Campo | Valor |
|---|---|
| Proyecto | OBSERVATORIO CUENTAS |
| Fecha | 23 de noviembre de 2026 |
| Estado | Decisión de diseño acordada; implementación y verificación pendientes |
| Versión del ADR | 1.0 |
| Base de migración inspeccionada | Observatorio SlowFast 0.13.1 — Loss Clásicas |
| Alcance | Librería, API, entrenamiento, resultados, publicación y aplicación |

Este documento establece el contrato para la siguiente implementación. Los ejemplos son normativos de diseño: su presencia en este ADR no acredita que el bundle actual ya los implemente. La versión del ADR se administra separadamente de la versión del software.

## 1. Contexto y problema

El proyecto necesita entrenar modelos globales de series temporales, comparar configuraciones e inputs, seleccionar un ganador por serie y producir pronósticos y alertamientos auditables. El usuario debe poder ejecutar el flujo completo y abrir el monitor por separado.

La implementación actual distribuye responsabilidades entre scripts, paquetes históricos y componentes del visor. La API pública todavía contiene operaciones como `compete`, `publish` y `evaluate_day`; no constituye la API definida aquí. Además, los resultados de entrenamiento y las métricas finales tienen distinta granularidad y no pueden intercambiarse como si fueran curvas equivalentes.

Necesitamos una interfaz familiar para quien utiliza librerías de forecasting, sin perder selección por serie, transferencia zero-shot, reglas del dominio ni trazabilidad. La interfaz debe servir para otros proyectos compatibles y permitir inspeccionar lo que efectivamente ocurrió en el entrenamiento.

## 2. Decisión principal

Se concentra el comportamiento funcional en una librería pública, `model_arena`. Observatory es una aplicación generada mediante scaffolding que importa esa librería y consume sus contratos de resultados.

| Componente | Responsabilidad |
|---|---|
| Model Arena | Preparación, HPO, entrenamiento, evaluación, selección, inferencia, postprocesamiento, persistencia y publicación |
| Observatory | Navegación, filtros, visualización, tablas, descargas y presentación de procedencia |
| Perfil del proyecto | Esquemas, poblaciones, inputs, segmentos, modelos, métricas, reglas y layouts del dominio |
| Adaptadores | Fuentes, almacenamiento, ejecución e integración con implementaciones de modelos |

El **slow system** construye una versión reproducible del conocimiento. El **fast system** explota resultados publicados y, cuando se solicita explícitamente, aplica inferencia o reglas con componentes congelados de Model Arena. Esa actualización produce una revisión identificable y no ejecuta entrenamiento implícito.

La lectura del monitor no modifica la publicación, los ganadores ni los pesos de los modelos.

## 3. Relación con decisiones anteriores

Este ADR complementa la especificación «Model Arena con Observatory» del 21 de noviembre y sustituye las siguientes decisiones cuando exista conflicto:

1. La API habitual adopta verbos como `fit`, `predict`, `tune` y `cross_validation`; los nombres de las etapas organizan la arquitectura.
2. El ajuste final pertenece a **processing** y se ejecuta mediante `fit`. Postprocessing recibe modelos o pronósticos ya disponibles.
3. La publicación es una operación explícita. Completar un entrenamiento no publica automáticamente.
4. Las dependencias avanzadas de ejecución se pueden inyectar, pero no son argumentos obligatorios en el uso básico.
5. La API de resultados distingue curvas por época, evaluaciones temporales y resultados operativos.

Se mantiene la composición mediante interfaces, configuración validada, almacenamiento por referencias y procedencia de extremo a extremo.

## 4. Convenciones adoptadas de Nixtla

Se adoptan el vocabulario familiar de operaciones, las colecciones de modelos y la separación entre `df`, `static_df` y `futr_df`. También se hace explícita la configuración de la evaluación temporal y del reentrenamiento entre cortes.

NeuralForecast documenta esos argumentos y métodos en su API core [R1]. Sus clases AutoModel explicitan la búsqueda de hiperparámetros [R2]. Model Arena toma esa claridad, pero decide exponer `tune` como operación propia para poder inspeccionar todos los candidatos y controlar después la selección y el ajuste final.

Esto no establece compatibilidad binaria con Nixtla ni exige utilizar sus modelos. Los adaptadores se implementarán cuando se necesiten y se verificarán por separado. `fit` devolverá `self` por decisión de Model Arena, sin afirmar que todas las librerías de referencia devuelven lo mismo.

## 5. Agnosticidad y estructura

El core es agnóstico de dominio, proveedor y modelo dentro de las capacidades implementadas. La primera API especializada es forecasting de panel con uno o varios objetivos. No se declara soporte de clasificación u otras tareas sin adaptadores, contratos y pruebas correspondientes.

Los nombres X0–X3, saldo, variación, cuentas atípicas, budgets por entropía y layouts MAC pertenecen al perfil contable. Las credenciales, rutas de usuarios, bases Athena y destinos S3 pertenecen a configuración de despliegue.

| Mecanismo | Uso |
|---|---|
| POO | Encapsular estado entrenado, ejecución y consulta |
| Composición | Integrar modelos, transformaciones, métricas y políticas |
| Polimorfismo | Sustituir componentes mediante interfaces pequeñas |
| Pydantic | Validar configuración, referencias y resultados serializables |

Los contratos serializables rechazan campos desconocidos y validan relaciones entre parámetros. Las redes, conexiones y funciones ejecutables no se almacenan como campos arbitrarios de configuración. Cada extensión declara nombre, versión, configuración, capacidades y fábrica.

Interfaces iniciales: `DataSource`, `Transformer`, `ForecastModel`, `Evaluator`, `SelectionPolicy`, `TransferPolicy`, `Postprocessor`, `Exporter`, `ArtifactStore` y `Runtime`.

La distribución `model_arena` incluye contratos, implementación y plantillas de Observatory. Las dependencias de entrenamiento y las del monitor se separan en extras instalables; consultar resultados no debe importar PyTorch, inicializar CUDA ni exigir Optuna.

## 6. API pública decidida

### 6.1 Construcción

Se soportan configuración declarativa y construcción mediante objetos equivalentes:

```python
arena = ModelArena.from_config("arena.yaml")

# Alternativa avanzada: componentes explícitos y validados.
arena = ModelArena(
    models=models,
    schema=schema,
    calendar=calendar,
    store=store,
    runtime=runtime,
)
```

Ambas rutas producen una especificación resuelta equivalente. `describe()` expone los valores efectivos; `plan()` muestra operaciones, dependencias y cantidades estimadas antes de iniciar trabajos costosos. La configuración de runtime es explícita y no selecciona entornos mediante búsquedas ambiguas en carpetas anteriores.

### 6.2 Operaciones y efectos

| Método | Entradas principales | Resultado | Efecto permitido |
|---|---|---|---|
| `prepare` | `df` o fuente, `static_df`, corte | `PreparedData` | Validar y materializar preparación |
| `tune` | Datos, protocolo CV, espacio y recursos de búsqueda | `StudyResult` | Entrenar y evaluar candidatos |
| `cross_validation` | Datos, configuraciones y protocolo CV | `BacktestResult` | Ajustar y evaluar según política explícita |
| `select` | Estudio/evaluación, unidad y política | `SelectionResult` | Resolver decisiones reproducibles |
| `fit` | Datos, configuraciones o selección, plan de entrenamiento | `self` | Crear un nuevo estado entrenado |
| `predict` | Historia/contexto, `static_df`, `futr_df`, horizonte | `ForecastResult` | Inferencia con estado congelado |
| `evaluate` | Pronósticos, observados y métricas | `EvaluationResult` | Calcular métricas |
| `postprocess` | Pronósticos y contexto del dominio | `Release` | Reglas, alertas y layouts |
| `save` | Destino y política de retención | `ArtifactRef` | Persistir estado y procedencia |
| `load` | Referencia y dependencias compatibles | `ModelArena` | Recuperar estado |
| `publish` | Release y destino | `Publication` | Publicar una revisión consistente |
| `status` / `resume` | Referencia de ejecución | Estado / resultado | Consultar o continuar trabajo compatible |

`arena.results` es una fachada de consulta. Permite acceder a estudios, candidatos, métricas, curvas, selecciones, predicciones y alertas mediante filtros tipados o argumentos de conveniencia. Puede abrirse desde una referencia publicada sin cargar los modelos.

Reglas obligatorias:

- `fit` no inicia HPO ni publica. Con configuraciones conocidas, basta `fit` seguido de `predict`.
- `tune` utiliza internamente el protocolo temporal; no se exige repetir `cross_validation` para recuperar la misma evidencia.
- `tune` y `cross_validation` conservan sus ajustes experimentales sin sustituir un estado operativo previamente entrenado.
- `select` no entrena. `evaluate` no cambia ganadores.
- `predict` no ajusta pesos ni recalcula parámetros aprendidos de transformaciones con datos futuros.
- `fit` comienza desde cero por defecto. Continuar desde un checkpoint requiere una solicitud explícita, compatible y registrada.
- `h` se valida frente a las capacidades del modelo entrenado. Un horizonte incompatible produce un error antes de inferir.
- Las operaciones devuelven referencias persistentes cuando su volumen hace inadecuado cargar todo en memoria.

Podrá existir `run()` como orquestador explícito de un plan configurado. Será equivalente a las operaciones detalladas y mostrará si incluye publicación; no se establece un comportamiento oculto adicional.

### 6.3 Etapas arquitectónicas

| Etapa | Operaciones |
|---|---|
| Preprocessing | Ingesta, validación, elegibilidad, segmentación, features y protocolo temporal |
| Processing | HPO, evaluación, selección y ajuste final |
| Postprocessing | Inferencia operativa, reglas, diagnósticos y layouts |

La persistencia atraviesa las etapas; la publicación expone únicamente una revisión completa y validada.

## 7. Contratos de datos y tiempo

La forma tabular predeterminada utiliza `unique_id` y `ds`. Los nombres originales se pueden mapear sin modificar los datos de origen:

```python
schema = PanelSchema(
    id_col="cross_key_id",
    time_col="cutoff_date",
    target_cols=["saldo", "variacion"],
)
```

| Entrada | Semántica |
|---|---|
| `df` | Historia de objetivos y covariables observadas o conocidas |
| `static_df` | Atributos de entidad declarados como estáticos |
| `futr_df` | Covariables de fechas futuras conocidas al emitir el pronóstico |
| Calendario | Fechas válidas, frecuencia, huecos y disponibilidad |

Saldo y variación son dos objetivos de una misma serie, modelados mediante dos heads. No se convierten implícitamente en entidades independientes. Las covariables declaradas como conocidas deben ser realmente disponibles en el origen de emisión.

Se conservan fecha del dato, fecha de disponibilidad y origen del pronóstico cuando corresponda. Transformaciones y segmentaciones aprendidas se ajustan únicamente con información admisible al corte. El calendario autoritativo resuelve las veinte fechas operativas; no se sustituyen por veinte días naturales sin validación.

La predicción canónica identifica ejecución, versión del modelo, candidato, entidad, origen, fecha objetivo, objetivo, cuantil, valor, divisa/unidad y procedencia. Las métricas monetarias se reportan por moneda; cualquier agregación entre monedas requiere una normalización o conversión declarada.

## 8. Perfil inicial de OBSERVATORIO CUENTAS

Esta migración preserva el problema de modelado y hace explícitos sus parámetros. Un cambio posterior de transformaciones, pérdida o arquitectura debe identificarse como experimento y versión distintos.

| Elemento | Configuración de partida |
|---|---|
| Universo | Todas las cuentas elegibles para entrenamiento; sin límite implícito de 5,000 |
| Inputs | X0, X1, X2 y X3; definiciones de features resueltas y versionadas |
| Arquitectura | TFT compacto |
| Diseño del encoder | Shared o independent |
| Decoder | Dos heads: saldo y variación |
| Capas encoder / decoder | {2, 3} / {1, 2} |
| Unidades | {32, 64, 128} |
| Activación | ReLU, Mish, GELU o Tanh |
| Dropout | 0.1 |
| Optimizador | AdamW |
| Learning rate | [1e-8, 1e-2], muestreo logarítmico |
| Weight decay | [1e-6, 1e-2], muestreo logarítmico |
| HPO | 15 trials por input; 20 épocas por ajuste |
| Sampler / pruner | TPESampler / NopPruner; configuración efectiva y versiones persistidas |
| Evaluación HPO | F1, F2 y F3, con cortes persistidos |
| Ajuste final | 75 épocas desde cero sobre la historia admisible al origen final |
| Horizonte | 20 fechas operativas |

Con cuatro inputs, quince trials y tres folds hay **60 configuraciones candidatas y 180 ajustes HPO**, antes de reintentos o ajustes finales. Cada ajuste tiene veinte épocas; no son 180 modelos operativos ni 180 épocas.

La semilla base es 42; se conservan las semillas efectivas por input y ajuste. Los defaults efectivos del sampler se registran con la versión utilizada. Este ADR no incorpora warmup, finetuning, pruning o early stopping adicionales. Las 75 épocas pertenecen al ajuste final.

### 8.1 Selección y referencia

La selección tiene dos niveles explícitos: obtener la mejor configuración admisible de cada input para el ámbito configurado —total o segmento— y elegir entre esas alternativas el ganador de cada serie. El ámbito, la métrica, los pesos entre objetivos, los criterios de cobertura y los desempates forman parte de `SelectionPolicy`.

Una misma configuración global puede atender muchas cuentas. Se entrena una vez por configuración y corte, y se reutiliza para las cuentas asignadas. Si distintos segmentos requieren configuraciones diferentes, el número de modelos finales puede superar cuatro; `plan()` debe mostrar el número real de configuraciones únicas.

Cada serie tiene un ganador operativo identificable. La elección entre ese Winner y Baseline como referencia de alertamiento es otra decisión y conserva su evidencia. No se construyen ensambles ni una referencia «Baseline + Winner».

La vista permite comparar siempre Baseline, Winner y Observado cuando existan datos. La disponibilidad de una familia visual no implica que sea la referencia activa.

### 8.2 Budgets y poblaciones

`budget` conserva la definición existente de segmentación por entropía del perfil contable. Su fórmula, bordes y fechas de cálculo se versionan; este ADR no los redefine. Se distingue de los recursos de HPO: trials, épocas, tiempo y memoria.

Las cuentas atípicas se atienden por zero-shot, sin incorporarlas al ajuste de pesos. La política de transferencia especifica el grupo de evidencia, mínimos y fallback. Su pronóstico conserva esa procedencia y pasa por el mismo contrato de alertamiento que el de una cuenta entrenable.

Se distingue entre población de entrenamiento, población de evaluación y población usada para monitorizar curvas. Una muestra de monitorización de 128 cuentas no limita el universo entrenado. Cualquier entidad sin asignación debe aparecer con un motivo; no se elimina silenciosamente.

## 9. Evaluación temporal y ausencia de fuga

Cada fold identifica corte de entrenamiento, intervalo de evaluación, calendario y política de ajuste. F1–F3 empleados en HPO y selección se etiquetan como evidencia de selección, no como test independiente.

Una estimación de desempeño independiente exige un bloque temporal reservado o un protocolo externo que no se haya utilizado para decidir hiperparámetros o políticas. El ajuste final usa la historia disponible al corte final y no tiene observados futuros para calcular validación en ese horizonte.

Las curvas históricas de un ganador requieren ajustes realizados con información admisible al origen de cada fold. Se puede mantener la misma configuración con diferentes pesos entrenados al corte; ambos identificadores deben distinguirse. El monitor no compone distintos candidatos como si fueran un único ganador ni aplica pesos finales retrospectivamente para aparentar un backtest causal.

Si un run antiguo carece de una curva o predicción, se marca como no disponible. Obtener nueva evidencia exige ejecutar y registrar la evaluación correspondiente.

## 10. Curvas de aprendizaje como producto obligatorio

Se registran tres clases de evidencia:

| Split | Qué mide | Uso |
|---|---|---|
| `train_sampled` | Loss de optimización en batches utilizados durante la época | Seguimiento del proceso estocástico |
| `train_fixed` | Evaluación del estado de fin de época en ventanas fijas de entrenamiento | Comparación entre épocas |
| `validation` | Evaluación temporal fuera del ajuste de pesos de ese fold | Selección y generalización en ese protocolo |

Las evaluaciones de seguimiento se ejecutan sin gradientes, sin pasos del optimizador y con el modo apropiado del modelo. Las ventanas fijas se mantienen comparables entre candidatos compatibles y folds; sus identificadores y su método de selección quedan persistidos. No se requiere entrenar siempre sobre las mismas ventanas para obtener curvas comparables.

Esquema mínimo:

```text
run_id, study_id, candidate_id, model_version, input, fold, phase,
epoch, target, split, metric, value, scope, group_id,
population_ref, window_set_ref, transform_ref, aggregation,
n_entities, n_observations
```

`scope` distingue total, segmento y entidad. Las curvas por budget o cuenta se obtienen calculando esa evaluación por época y guardando su denominador; no se crean replicando una curva global ni interpolando una métrica final. La retención de curvas por entidad debe estimarse en el plan y puede almacenarse particionada. Si se utiliza muestreo, Observatory indica su cobertura y no afirma contener todas las cuentas.

Para los nuevos runs del perfil, el objetivo es conservar la evaluación temporal por época a nivel de entidad y permitir agregaciones reproducibles por budget y total. Cualquier reducción de esa cobertura debe ser explícita en configuración y en el plan.

Se diferencian pinball de entrenamiento, pinball en escala normalizada y métricas monetarias nativas. Los ejes muestran métrica, escala y agregación. No se comparan numéricamente pérdidas de transformaciones diferentes sin una métrica común.

En Observatory se presentan saldo arriba y variación abajo; folds en paneles separados; filtros por input, candidato, budget y cuenta donde exista evidencia. Se permite resaltar un candidato, alternar gráfico/tabla y descargar lo filtrado. Se evitan leyendas de cientos de series. Los valores crudos permanecen accesibles y cualquier suavizado se identifica.

## 11. Persistencia y publicación

| Objeto | Información conservada |
|---|---|
| Preparación | Snapshot o referencias reproducibles, población, cortes y calendario |
| Estudio | Todos los candidatos, parámetros, trials, estados, semillas, métricas y curvas |
| Selección | Finalistas, asignación por serie, referencia activa, evidencia y fallback |
| Estado entrenado | Pesos, arquitectura, orden de features, transformaciones, corte y versiones |
| Release | Pronósticos, alertas, motivos, layouts y controles de calidad |
| Publicación | Identidad, revisión, manifiesto y referencias verificables |

Se conservan los resultados de todos los candidatos. Los checkpoints de experimentación y los modelos operativos tienen políticas de retención separadas. «Solo sobrevive el ganador» significa que la inferencia operativa utiliza las configuraciones ganadoras asignadas; no implica borrar evidencia experimental o modelos sin una política explícita.

`save/load` debe reproducir la inferencia dentro de tolerancias declaradas. Reanudar entrenamiento requiere además checkpoint del optimizador, contador de épocas y estado aleatorio cuando corresponda; un artefacto de inferencia no se presenta como checkpoint reanudable.

Publicar consiste en escribir artefactos completos, validarlos y después hacer visible su revisión. Una escritura parcial no se convierte en publicación activa. Se deben poder reabrir publicaciones anteriores y reintentar una publicación sin duplicar sus efectos.

En el despliegue ADA, los artefactos operativos se almacenan en S3 y se consultan mediante tablas/adaptadores Athena. Los esquemas de Engine 2 y Engine 3 se conservan como exportadores del perfil MAC, con versiones y validaciones de layout. Los nombres de buckets y tablas no pertenecen al core.

Se ofrecen descargas de los layouts publicados y su diccionario de campos. Los pesos no se cargan a través de tablas Athena: la inferencia recupera sus artefactos mediante `ArtifactStore`.

## 12. Observatory y experiencia de ejecución

El monitor abre una publicación explícita o una elección validada de publicaciones. Sus consultas son acotadas y utilizan paginación, filtros y carga diferida. No depende de conservar el directorio de entrenamiento ni de un entorno virtual ubicado en otro bundle.

Se mantienen las vistas Historia y budgets, Model Arena y Alertamientos. Comparten contratos y filtros; la aplicación no implementa de nuevo normalización, selección, pérdida o reglas contables.

En Alertamientos se preservan el layout de referencia, los controles con botones para Baseline/Winner/Observado y las miniaturas dentro del listado. Filas, conteos y miniaturas utilizan el mismo origen, tramo, objetivo, severidad y causa. Los ejes de montos conservan unidades monetarias también al comprimir la escala.

Se entregarán dos notebooks:

- **Slow System:** didáctico, muestra preparación, protocolo HPO, selección, ajuste final, inferencia, controles y publicación.
- **Fast System:** directo, valida configuración, abre la publicación y muestra un botón HTML para acceder al monitor.

La búsqueda de bundle debe ser determinista. Si existen varias copias, se informa la ruta elegida o se solicita la ruta concreta antes de ejecutar. Los notebooks pueden guardarse y ejecutarse sin invalidar la integridad del código por cambios de outputs o metadatos.

## 13. Ejemplo del flujo objetivo

```python
from model_arena import ModelArena

arena = ModelArena.from_config("arena.yaml")
plan = arena.plan(source=source, as_of=cutoff)
prepared = arena.prepare(source=source, as_of=cutoff)

study = arena.tune(
    prepared,
    cv=temporal_protocol,
    n_trials=15,
    epochs=20,
)
selection = arena.select(study, by="unique_id", policy=selection_policy)
arena.fit(prepared, selection=selection, epochs=75)

forecasts = arena.predict(
    df=prediction_history,
    static_df=accounts,
    futr_df=future_calendar,
    h=20,
    quantiles=[0.025, 0.5, 0.975],
)
release = arena.postprocess(forecasts, context=accounting_context)
artifact = arena.save(artifact_destination)
publication = arena.publish(release, destination=publication_destination)
```

`prepare` y `tune/fit` aceptan el resultado tipado o su referencia. La ruta tabular permite también `tune(df=history, static_df=accounts, ...)` y `fit(df=history, static_df=accounts, ...)`. No se aceptan simultáneamente datos preparados y un `df` alternativo.

La aplicación generada puede exponer:

```python
app = Observatory.from_config(
    "observatory.yaml",
    publication=publication_ref,
)
app.launch()
```

`Observatory` es el punto de entrada del scaffold, no una segunda librería de negocio. Las variables del ejemplo son dependencias del proyecto; no requieren valores contables incrustados en el core.

## 14. Alternativas consideradas

| Alternativa | Motivo para descartarla como interfaz principal |
|---|---|
| Conservar solo scripts y notebooks | Dificulta reutilizar, probar y distinguir efectos |
| Exponer únicamente las tres etapas | Hace menos familiar el uso de configuraciones conocidas |
| Incluir HPO y publicación implícitos en `fit` | Oculta costo, efectos y procedencia |
| Copiar íntegramente Nixtla | No resuelve por sí mismo selección por entidad, dos heads ni layouts MAC |
| Duplicar lógica en Observatory | Riesgo de discrepancias entre entrenamiento y presentación |
| Un modelo independiente por cuenta | Cambia el enfoque global y el costo sin ser requisito |
| Recalcular curvas antiguas en el visor | Presentaría evidencia que el entrenamiento no registró |

## 15. Consecuencias

La separación permite probar entrenamiento, selección, inferencia y presentación de manera independiente. También permite cambiar el frontend o el almacenamiento sin cambiar las reglas del dominio.

El costo adicional está en definir y mantener contratos versionados, registrar más evidencia y ejecutar evaluación por época. La evaluación completa por entidad puede aumentar tiempo y almacenamiento; el plan debe mostrar esa estimación y la política de retención antes de iniciar el run.

La agnosticidad requiere implementaciones y pruebas reales. Encapsular un subprocess existente no demuestra por sí solo que una fuente, una tarea o una arquitectura sean intercambiables.

## 16. Migración e invariantes

1. Introducir contratos, API y adaptadores sobre la base 0.13.1, conservando una comparación funcional reproducible.
2. Separar el motor de estudio, la selección y el ajuste final. Retirar efectos de publicación ocultos en operaciones de entrenamiento.
3. Unificar artefactos y curvas; migrar el monitor a consultas del mismo contrato.
4. Conservar lectores de publicaciones anteriores. Exponer campos ausentes como no disponibles.
5. Entregar notebooks Slow/Fast, documentación e instalación independiente del bundle anterior.
6. Cambiar la API pública únicamente después de verificar los criterios siguientes.

No se ejecuta un entrenamiento productivo ni se borran runs como consecuencia de aprobar este ADR. No se altera la función de pérdida, las transformaciones ni las reglas MAC por el hecho de adoptar la nueva API.

## 17. Criterios de aceptación

| ID | Comprobación requerida |
|---|---|
| A01 | Configuración inválida o incompatibilidad de horizonte/features falla antes de entrenar |
| A02 | Ruta declarativa y construcción por objetos resuelven la misma configuración |
| A03 | `fit` con parámetros conocidos no invoca HPO ni publicación |
| A04 | `predict` conserva pesos y estado aprendido de transformaciones |
| A05 | `tune` conserva todos los trials y sus tres folds sin duplicar el CV por consulta |
| A06 | Cada ganador identifica candidato, configuración, artefacto, corte y evidencia |
| A07 | Las cuentas zero-shot quedan fuera del ajuste y dentro de pronósticos/alertamientos según política |
| A08 | Las curvas tienen valores reales por época, objetivo, candidato y split; denominadores y scopes verificables |
| A09 | Activar el seguimiento no cambia los pesos en una prueba determinista compatible |
| A10 | FUTURO no publica una curva de validación con observados inexistentes |
| A11 | `save/load` reproduce predicciones; reanudar valida identidad y no mezcla estudios |
| A12 | Una publicación incompleta no sustituye la revisión activa |
| A13 | Observatory abre una publicación sin GPU, modelos cargados ni directorio del run |
| A14 | Filtros producen el mismo universo para conteos, filas, miniaturas y descargas |
| A15 | Curvas y tablas concuerdan; comparar folds no compone diferentes candidatos como un solo ganador |
| A16 | Layouts Engine 2/3 cumplen sus contratos MAC versionados |
| A17 | Una instalación limpia abre ambos notebooks y no depende de entornos de bundles anteriores |
| A18 | Un perfil sintético sin nombres contables demuestra sustitución de esquema, modelo y almacenamiento |

Las pruebas locales con datos sintéticos y servicios simulados se distinguen de la verificación real en ADA. Hasta ejecutar esta última, no se declara acreditado el flujo productivo Athena/S3/GPU.

## 18. Referencias y procedencia

- **[R1]** Nixtla, *Core — NeuralForecast*: https://nixtlaverse.nixtla.io/neuralforecast/core.html. Consultado durante esta revisión. Referencia para vocabulario y contratos de uso; las decisiones propias de este ADR no se atribuyen a Nixtla.
- **[R2]** Nixtla, *Automatic Forecasting*: https://nixtlaverse.nixtla.io/neuralforecast/models.html. Referencia para la separación explícita del HPO.
- **[R3]** *Model Arena con Observatory — Especificación objetivo*, 21 de noviembre de 2026. Documento previo del proyecto.
- **[R4]** Bundle local `Observatorio_SlowFast_0131`: `BUILD_MONITOR.json`, `config/entrenamiento_ada.json`, `entrenamiento/arena_estudio.py` y API histórica de `model_arena`. Inspeccionados para distinguir configuración existente y contrato objetivo.

Las rutas de ejecución, tablas y credenciales se resuelven en la implementación y en el perfil de despliegue; este ADR no las fija como requisitos del core.
