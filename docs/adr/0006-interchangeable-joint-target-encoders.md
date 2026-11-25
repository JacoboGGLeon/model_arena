# ADR-0006 · Encoders intercambiables y presupuesto de HPO

Estado: implementado para 0.1.3. Fecha: 2026-11-24 (México).

## Contexto

Se solicita comparar MLP, GRU y TFT compacto como encoders, conservando un decoder con heads independientes de saldo y variación, cuantiles ordenados, selección por serie y zero-shot. Ampliar familias con 20/25 trials totales no equivale a dar ese presupuesto a cada familia.

## Decisión

Agregar encoder_kind al contrato de parámetros y espacio de búsqueda; mantener encoder_design como shared/independent. Los encoders independientes conservan el contrato anterior: pesos separados y la misma historia conjunta de los objetivos. Compartir un encoder no comparte los heads del decoder.

MLP comprime la ventana ordenada completa mediante capas densas. GRU usa su último estado recurrente. Ambos combinan estado con posiciones de horizonte y calendario futuro, y entregan la misma representación al decoder MLP. No incorporan atención ni LSTM de TFT. TFT conserva la ruta de parámetros y operaciones anterior. La proyección del calendario pasado permanece antes del encoder. Las transformaciones, cuantiles, pérdida y layouts MAC no cambian.

La API genérica admite ModelSpec(kind="encoder_decoder"); tft_compact continúa como alias compatible del contenedor histórico. SearchSpec.encoder_kind controla las familias buscadas. El perfil MAC usa curvas.encoder_kinds. Configuraciones omitidas conservan exclusivamente TFT.

El presupuesto es total por input/modelo base, no por familia. Una cola equilibrada fija encoder_kind y Optuna TPE explora las demás dimensiones dentro de las familias. Con 25 trials y tres familias hay 9/8/8; con 75 hay 25/25/25. Esto prioriza cobertura reproducible sobre asignación adaptativa de presupuesto entre familias. La cola se crea una vez; reanudar no agrega nuevos intentos. La primera configuración completa de cada encoder/input queda exenta de poda como referencia; después rige paciencia cuatro por fold. Las curvas parciales no entran en selección.

## Consecuencias y alternativas

25 totales por input permite 100 candidatos y hasta 300 ajustes F1–F3; 25 por encoder/input permite 300 candidatos y hasta 900 ajustes. HPO conserva máximo 15 épocas por ajuste y final 75. Igual cantidad de épocas no iguala tiempos entre encoders. No se afirma una reducción de costo por poda sin medición real. La loss y la selección conservan saldo y variación.

No se integran TiDE/NHiTS u otros modelos completos. Cambiar solamente una LSTM por GRU dentro del TFT no respondería a comparar encoders diferentes. No se modifican proyectos o estudios congelados: se requiere una corrida nueva. Volver a una versión anterior exige usar sus proyectos; no cargar pesos MLP/GRU con una versión que sólo entiende TFT.

## Verificación

Pruebas de forward/backward, horizonte distinto de ventana, heads/orden de cuantiles, pesos persistidos, shared/independent y las tres familias. Entrenamiento/HPO real en API genérica y perfil MAC, pruning por familia, cola equilibrada, reanudación, zero-shot, inferencia y layouts MAC. Se verifican fuentes y wheel instalado antes de entrega; resultados efectivos se informan por separado.
