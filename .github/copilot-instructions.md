# Instrucciones para Copilot

## Contexto del proyecto

- Proyecto: App de Diseño Experimental desarrollada en Python y Streamlit para el curso de Diseño Experimental de Ingeniería Agrícola de la Universidad de Sucre.
- Estructura del proyecto:
  - `app.py`: portada de la aplicación.
  - `pages/`: un archivo por cada diseño experimental.
  - `utils/`: funciones reutilizables; debe incluir `__init__.py`.
  - `data/`: archivos CSV de ejemplo.
  - `verificacion/`: scripts de R para verificar los análisis.

## Idioma y explicaciones

- Escribe en español todo el código, los comentarios, los docstrings y los mensajes de la interfaz.
- Incluye un comentario en cada paso estadístico que explique qué calcula y por qué, con explicaciones adecuadas para estudiantes de Ingeniería Agrícola.
- Escribe los docstrings en español y usa el estilo Google.

## Código y análisis estadístico

- Añade anotaciones de tipo (`type hints`) a las funciones.
- Limita las dependencias a `streamlit`, `pandas`, `numpy`, `scipy`, `statsmodels`, `plotly` y `openpyxl`. No agregues otras librerías sin pedir permiso.
- Ajusta los modelos ANOVA con `statsmodels.formula.api.ols` y calcula la tabla con `statsmodels.stats.anova.anova_lm(typ=2)`.
- Realiza las comparaciones de Tukey con `statsmodels.stats.multicomp.pairwise_tukeyhsd`.
- No inventes ni codifiques resultados numéricos: todos deben provenir del cálculo ejecutado con los datos.

## Verificación

- Antes de terminar una tarea que modifique código, ejecuta el código y corrige los errores que aparezcan.
