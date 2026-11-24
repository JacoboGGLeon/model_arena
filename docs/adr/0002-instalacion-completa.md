# ADR-0002 — Instalación completa con un solo comando

Fecha: 24 de noviembre de 2026.

Estado: decisión de implementación para atender la solicitud de instalación `python -m pip install "model-arena-sdk==0.1"`.

## Contexto

La versión candidata 0.1.0rc1 separaba dependencias de entrenamiento, AWS y monitor mediante extras. Instalar únicamente el paquete base no preparaba el flujo contable de ADA. El usuario solicita la instalación simple, sin extras, de la versión 0.1.

## Decisión

La versión 0.1 incluye por defecto las dependencias del flujo completo: núcleo, entrenamiento, AWS, perfil MAC y Observatory. Los extras anteriores se conservan como alias compatibles; no son necesarios ni reducen el conjunto de dependencias base.

Los imports siguen siendo diferidos. Consultar resultados no importa PyTorch ni inicializa CUDA. La configuración de índices, credenciales y la instalación CUDA aprobada por ADA siguen perteneciendo al entorno; el paquete no cambia esas configuraciones.

La versión de distribución y la versión utilizada en artefactos comparten una fuente única, `_version.py`. El tag publicado será `v0.1`. Los notebooks solicitan `model-arena-sdk==0.1`, sin corchetes.

## Alternativas

- Mantener el núcleo ligero y exigir extras: conserva la descarga reducida, pero no satisface la instalación completa solicitada.
- Instalar dependencias automáticamente al llamar `fit` o abrir el monitor: introduce efectos de instalación ocultos en ejecución; se descarta.
- Instalación completa mediante metadatos de paquete: permite que pip y Artifactory resuelvan las dependencias antes de ejecutar; es la alternativa elegida.

## Consecuencias y compatibilidad

La instalación predeterminada incorpora más dependencias y puede descargar PyTorch. El fast system sigue sin requerir GPU para ejecutarse. No se modifica el entrenamiento, los pesos, las transformaciones, los layouts ni el alertamiento.

Este ADR sustituye únicamente la separación obligatoria de dependencias de distribución del ADR-0001. Se conserva la separación de responsabilidades e imports.

## Verificación y reversión

Se comprobarán metadatos del wheel sin seleccionar extras, imports del runtime, tests existentes y el comando exacto contra la versión publicada. CI instalará `.[dev]`, sin extras funcionales, antes de ejecutar entrenamiento y monitor.

La versión candidata permanece disponible para reproducir instalaciones anteriores. No se sustituyen archivos ya publicados ni se modifican tags existentes.
