# Instalación en ADA

Python soportado inicialmente: 3.11 y 3.12. La plataforma observada utiliza Python 3.12.9. Selecciona un kernel con el PyTorch/CUDA aprobado por ADA antes del entrenamiento.

## Paquete y configuración

- Distribución: `model-arena-sdk` (nombre propuesto; publicación pendiente).
- Import: `model_arena`.
- Repositorio previsto: `JacoboGGLeon/model_arena`.
- Configuración corporativa: archivos locales de despliegue; no forman parte del paquete público.

No modifiques los índices corporativos de pip. La instalación utiliza su configuración existente. No ejecutes `pip config list` en un notebook compartido ni incluyas URLs autenticadas en sus celdas.

Cuando Artifactory tenga la versión publicada:

```bash
python -m pip install 'model-arena-sdk[mac]==0.1.0rc1'
```

Si aún no la replica, descarga el wheel y usa:

```bash
python -m pip install './dist/model_arena_sdk-0.1.0rc1-py3-none-any.whl[mac]'
```

Las dependencias siguen resolviéndose por los índices corporativos. El wheel no incluye PyTorch ni dependencias de terceros. No se fuerza un índice PyTorch público, no se desinstala el entorno y no se reutiliza el venv de otro bundle.

Para abrir únicamente el monitor:

```bash
python -m pip install './dist/model_arena_sdk-0.1.0rc1-py3-none-any.whl[monitor]'
```

## Notebooks

Ejecuta `10_ADA_Slow_System_End_to_End.ipynb` desde la carpeta ADA entregada. Los archivos `private/training.json`, `private/publication.json` y `private/calendar_authoritative.csv` son los únicos insumos locales del despliegue. Las credenciales AWS proceden del rol del entorno.

`00_ADA_Fast_System_Observatory.ipynb` abre Athena sin necesitar el directorio del run. Solo requiere la configuración de publicación y permisos de lectura.

La réplica del paquete desde PyPI al repositorio corporativo depende de la configuración y políticas de Artifactory. Publicar en PyPI no demuestra por sí solo que ADA ya pueda resolver esa versión; la alternativa del wheel permite comprobar el mismo artefacto sin cambiar los índices.
