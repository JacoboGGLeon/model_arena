# ADR-0005 · HPO con paciencia por fold

Estado: implementado para 0.1.2. Fecha: 2026-11-24, America/Mexico_City.

## Contexto

El usuario solicita 15 épocas máximas por intento HPO y poda tras 4 steps sin mejora. El entrenamiento usa ventanas muestreadas distintas por época; esa loss no ofrece una comparación estable para decidir estancamiento. F1–F3 tienen distintos cortes y escalas. No deben compartir el contador de paciencia ni compararse como si fueran épocas consecutivas del mismo ajuste.

## Decisión

Un step es una época completada. Se reporta a Optuna la media de pinball normalizado en validación temporal fija sobre las entrenables y los objetivos declarados; en MAC son saldo y variación. Se reinicia la paciencia por fold y se usan rangos distintos de steps para evitar colisiones en Trial.report. Cuatro reportes consecutivos sin mejora estricta disparan TrialPruned. min_delta es configurable, no negativo y finito. Con min_delta=0 las igualdades cuentan como no mejora; primera poda posible en época 5.

FoldPatiencePruner implementa BasePruner. La implementación no depende de las diferencias históricas en el tratamiento de mesetas/steps de PatientPruner entre versiones de Optuna. No incorpora MedianPruner ni Hyperband. El primer trial completo de cada input queda exento de poda para garantizar una referencia comparable en todos los folds. No se promete que esta heurística conserve al candidato que habría ganado con todas las épocas.

La poda descarta el trial entero, conserva parámetros y curvas parciales, y excluye su evaluación incompleta del ranking. Los 15 intentos por input cuentan COMPLETE + PRUNED; un reinicio no añade intentos por contar sólo completos. Los fallos por disco y su reintento conservan la identidad lógica anterior. Las excepciones de cómputo siguen siendo errores, no podas ocultas.

La política es explícita en SearchSpec.pruner o curvas.pruner para MAC. Los contratos/configuraciones sin política conservan NopPruner. La configuración ADA nueva usa 15 épocas HPO, paciencia 4 y 75 épocas finales; los proyectos existentes no se modifican. El ajuste final no llama a la poda ni inventa validación futura.

Se publica hpo_trials como dataset adicional, con estados, parámetros y punto de poda. La última loss intermedia permanece separada del score final de un trial completo. Engine 2/3 y el criterio de selección entre candidatos completos conservan sus contratos.

## Verificación

Pruebas de meseta, reinicio de paciencia tras mejora y entre folds, min_delta, dirección, referencia completa, validación de parámetros y persistencia SQLite. Entrenamiento neuronal real en la API genérica y perfil MAC: un trial completa los folds y otro se poda en época 5. Se verifican curvas parciales, exclusión de scores/pesos, reanudación sin intentos extra y publicación del registro. La suite se ejecuta sobre fuentes y wheel instalado.

## Referencias

- https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.BasePruner.html
- https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PatientPruner.html
- https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
