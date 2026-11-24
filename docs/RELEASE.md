# Publicación de la primera versión

## Estado

Repositorio creado: https://github.com/JacoboGGLeon/model_arena. La fuente se publica mediante la conexión de GitHub. Trusted Publishing está configurado y la versión candidata anterior se publicó correctamente. La versión 0.1 se distribuye sin extras obligatorios; cada publicación se acredita con su recibo de GitHub Actions y PyPI.

## GitHub

Cuenta confirmada: JacoboGGLeon. Repositorio previsto: model_arena. Licencia confirmada: MIT. Sube únicamente esta carpeta de código. Los archivos de despliegue del paquete privado de ADA son externos a ella.

Una vez conectado GitHub, crear el repositorio, subir el commit validado y comprobar el resultado de CI. No incluir runs, checkpoints, datos, configuraciones corporativas o credenciales.

## PyPI

`model-arena` ya tiene distribuciones de otro proyecto. `model-arena-sdk` ya está registrado y conserva el import `model_arena`.

Trusted Publisher configurado para este proyecto:

- Project: model-arena-sdk
- Owner: JacoboGGLeon
- Repository: model_arena
- Workflow: release.yml
- Environment: pypi

La configuración de confianza debe corresponder exactamente al repositorio y workflow publicados. No se necesita pegar tokens en notebooks ni commits.

Crear el entorno GitHub `pypi` y publicar la release con tag `v0.1`. El workflow primero ejecuta tests, comprueba tag/versión, construye wheel/sdist y valida sus metadatos. Un job independiente publica esos artefactos con OIDC.

Tras el éxito del workflow, comprobar la versión, el nombre y los hashes de los artefactos disponibles en PyPI. Después comprobar su resolución desde ADA/Artifactory.

Referencias oficiales:

- https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
- https://docs.pypi.org/trusted-publishers/
