# ADR-0007 · Familias completas de forecasting

Estado: implementado para 0.1.4; validación sintética/CPU, sin benchmark CUDA/AWS.

## Decisión

Integrar las implementaciones upstream de NeuralForecast mediante un adaptador de red por objetivo. Encoder–Decoder sigue siendo una familia separada. Nixtla aporta la arquitectura, no el entrenador ni la selección: Model Arena conserva ventanas causales, AdamW, épocas, paciencia por fold, losses fijas por cuenta/budget, zero-shot y selección común. TiDE/NHITS/NLinear/DLinear no se presentan como encoders de TFT ni se les impone su decoder.

Fijar NeuralForecast 3.1.2 y Lightning 2.5.x para compatibilidad con PyTorch 2.6 usado en ADA y CI. La integración invoca forward de la arquitectura; se prueba el contrato de batches de esa versión, sin depender de APIs de entrenamiento ni Ray para HPO.

Los objetivos tienen redes independientes dentro del candidato; se evalúan conjuntamente. La salida de cuantiles se ordena para evitar cruces. Las redes nuevas reciben historia lineal normalizada (la inversión de asinh de la entrada histórica del entrenador existente), conservando el target normalizado y la escala monetaria de salida. La ruta Encoder–Decoder anterior conserva sus matemáticas.

HPO se presupuesta por familia/input, con espacios específicos. NLinear/DLinear sólo aceptan X0 si upstream no admite exógenas. TiDE/NHITS permiten historia de calendario y calendario futuro. La parte histórica del canal de exógenas futuras se fija en cero, excluyéndola explícitamente; el calendario pasado, cuando se solicita, entra por hist_exog. Esto preserva la diferencia X1/X2/X3 sin usar observados futuros.

Sólo candidatos completos participan en la selección común. Las métricas permanecen por objetivo, cuenta y budget. El ganador operativo sigue siendo una configuración por serie. Los proyectos anteriores no cambian. Se congela la versión upstream y el hash del código de cada arquitectura en los checkpoints para detectar incompatibilidades.

## Verificación requerida

Forward/backward real de cuatro arquitecturas; cuantiles/objetivos, causalidad de covariables, guardas de capacidades, persistencia/restore, HPO y pérdidas por época, selección y zero-shot; pipeline MAC, layouts y publicación sintética. Ejecutar suite desde fuente y wheel. No afirmar benchmark CUDA/AWS sin ejecutarlo.
