# ADR-0003 · Bundle local y almacenamiento Parquet

Estado: implementado para prueba local, pendiente de validación completa en ADA.

Fecha: 2026-11-24.

## Contexto

El run del SDK 0.1 se interrumpió durante HPO al fallar una escritura de Optuna/SQLite. El usuario identificó que el disco se llena. El registro detallado por cuenta en CSV puede generar 83,534,400 filas para 5,801 cuentas, dos objetivos, train/validación, 20 épocas y 180 ajustes, además de predicciones y modelos.

El usuario pide probar un bundle autocontenido antes de consolidar una nueva versión de la librería y usar Parquet tanto como sea posible.

## Decisión

1. Entregar fuentes locales, notebooks, configuraciones privadas y tests en ZIP. No publicar GitHub/PyPI en esta fase.
2. Guardar detalle de loss por época en Parquet Zstandard y resúmenes por ajuste en Parquet. Conservar Float64. Mantener JSON para configuración/identidad, SQLite para el estudio y `.pt` para pesos.
3. Escribir atómicamente y leer detalle por lotes al preparar publicaciones.
4. Comprobar una reserva de disco configurable y bloquear escritores simultáneos sobre un proyecto. Detenerse con errores útiles y reencolar trials interrumpidos por falta de margen con los mismos parámetros.
5. Ofrecer una migración explícita y acotada del runtime SDK 0.1: respaldar metadatos y runtime, verificar SQLite, compactar sin cambiar valores, actualizar identidades de código y retirar CSV únicamente después de verificar. No cambiar modelos, transformaciones, hiperparámetros o selección.
6. Reutilizar originales inmutables mediante hardlinks en preparación cuando el sistema lo permita, con copia comprobada como alternativa.
7. Mantener un visor local de curvas separado del monitor operativo Athena.

## Alternativas

- Aumentar sólo el disco: recupera margen pero conserva el crecimiento innecesario del CSV.
- Bajar cuentas o registrar una muestra: pierde el desglose por cuenta solicitado.
- Subir y borrar automáticamente por cada ajuste: requiere nuevas garantías de carga, descarga/reanudación y permisos; se difiere para validar primero la ejecución local.
- Cambiar SQLite: no corrige el volumen de loss y predicciones.
- Único Parquet appendable por fit: más acoplado al cierre del proceso y recuperación. Las épocas independientes permiten commits acotados y lecturas durante entrenamiento.

## Consecuencias

Se conservan todos los candidatos y registros por cuenta; sigue creciendo el volumen con cada run. La reserva de disco es una comprobación puntual con estimaciones para algunas escrituras, no una garantía de capacidad para toda la corrida. La conversión requiere espacio temporal y no repara SQLite corrupto. No hay continuación del optimizador desde una época intermedia.

Los enlaces físicos exigen tratar los originales como inmutables; el flujo actual reutiliza checkpoints y reemplaza los demás archivos por nombre con escrituras atómicas. La publicación verifica hashes antes de subir datos.

## Validación y recuperación

La suite debe comprobar valores y orden en compactación, conservación de checkpoints y predicciones durante migración, reanudación del HPO, error por reserva de disco, bloqueo de escritores, compatibilidad de layouts y lectura local por filtros. El resultado ejecutado se registra en `VALIDATION.md`.

Si falla la instalación del runtime durante el upgrade se restaura el respaldo de código y metadatos; no se retiran los CSV. Tras una migración terminada, volver al código anterior exigiría reconstruir sus CSV desde Parquet: no se ofrece un rollback ciego de formatos. Los respaldos de runtime no sustituyen backups de datos/modelos.
