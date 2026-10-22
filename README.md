# 🌱 App de Diseño Experimental

Aplicación web hecha con Python y Streamlit para el curso de **Diseño Experimental** de Ingeniería Agrícola de la Universidad de Sucre. Permite cargar datos (CSV o Excel), ajustar el ANOVA, revisar supuestos, hacer comparaciones de medias (Tukey) y ver gráficos, como apoyo al aprendizaje.

## Módulos y estado

| Módulo | Archivo | Estado |
|---|---|---|
| DCA – Diseño Completamente al Azar | `pages/01_DCA.py` | Disponible |
| DBCA – Bloques Completos al Azar | `pages/02_DBCA.py` | En construcción |
| DCL – Cuadrado Latino | `pages/03_DCL.py` | En construcción |
| DPD – Parcelas Divididas | `pages/04_DPD.py` | En construcción |
| Factoriales | `pages/05_Factoriales.py` | En construcción |
| RSM – Superficie de Respuesta | `pages/06_RSM.py` | En construcción |

## Estructura de carpetas

```
app.py              Portada de la aplicación
pages/              Un archivo por cada diseño experimental
utils/              Funciones reutilizables (ANOVA, diagnóstico)
data/               Archivos CSV de ejemplo
verificacion/       Scripts de R para verificar los análisis
requirements.txt    Dependencias de Python
BITACORA_IA.md      Registro del uso de IA
```

## Instalación y ejecución

### Local

```bash
git clone https://github.com/justorfc/app-diseno-experimental.git
cd app-diseno-experimental
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Abre <http://localhost:8501> en el navegador.

### GitHub Codespaces

1. Abre el repositorio en un Codespace.
2. En la terminal, instala las dependencias: `pip install -r requirements.txt`.
3. Ejecuta `streamlit run app.py`.
4. Abre la URL del puerto 8501 que ofrece Codespaces.

## Verificación con R

Los scripts de `verificacion/` repiten los análisis en R para comparar con la app. Desde esa carpeta:

```bash
Rscript verificacion_dca.R
```

## Autor

Justo Fuentes – Curso de Diseño Experimental, Ingeniería Agrícola, Universidad de Sucre (semestre 2026-2).

## Uso de IA

Este proyecto se desarrolló con ayuda de GitHub Copilot (agente). Cada prompt, lo que generó la IA, lo que se aceptó o corrigió y cómo se verificó (ejecución de la app y comparación con R) se registra en [BITACORA_IA.md](BITACORA_IA.md). El autor es responsable de revisar e interpretar los resultados.
