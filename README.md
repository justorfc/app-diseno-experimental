# 🌱 App de Diseño Experimental

Aplicación web hecha con Python y Streamlit para el curso de **Diseño Experimental** de Ingeniería Agrícola de la Universidad de Sucre. Permite cargar datos (CSV o Excel), ajustar el análisis de varianza o la superficie de respuesta, revisar supuestos, hacer comparaciones de medias (Tukey), ver gráficos y exportar resultados, como apoyo al aprendizaje. Todos los números salen del cálculo con los datos cargados; nada está codificado a mano.

## Módulos

| Módulo | Archivo | Qué hace |
|---|---|---|
| DCA – Diseño Completamente al Azar | `pages/01_DCA.py` | ANOVA de un factor, Tukey, gráficos y supuestos |
| DBCA – Bloques Completos al Azar | `pages/02_DBCA.py` | ANOVA con bloques, eficiencia relativa, Tukey con el CM Error del modelo |
| DCL – Cuadrado Latino | `pages/03_DCL.py` | Control de dos fuentes de variación (filas y columnas), Tukey con el CM Error del modelo |
| DPD – Parcelas Divididas | `pages/04_DPD.py` | ANOVA con Error (a) para la parcela principal y Error (b) para la subparcela y la interacción |
| Factoriales | `pages/05_Factoriales.py` | Factorial de 2 o más factores: ANOVA, interacción, Tukey, gráficos y exportación a Excel |
| RSM – Superficie de Respuesta | `pages/06_RSM.py` | Primer orden, curvatura, trayectoria de máximo ascenso, segundo orden, punto estacionario, superficie 3D, curvas de nivel y exportación de figuras |

En el RSM las variables se trabajan codificadas (x1, x2) con `x = (ξ − centro) / semirango` y siempre se muestran también en unidades naturales.

## Estructura de carpetas

```
app.py              Portada de la aplicación
pages/              Un archivo por cada diseño experimental
utils/              Funciones reutilizables (ANOVA, bloques, factorial, RSM, diagnóstico, exportación)
data/               Archivos CSV de ejemplo
verificacion/       Scripts de R y de Python para verificar los análisis
figuras/            Figuras del RSM guardadas desde la app (se crea sola; no se versiona)
requirements.txt    Dependencias de Python
BITACORA_IA.md      Registro del uso de IA
```

## Formato de los datos

Cada archivo es un CSV (separador `,` o `;`) o un Excel `.xlsx`, con una fila por observación y los nombres de columna en la primera fila. En todos los casos la respuesta debe ser numérica y no puede haber datos faltantes.

| Módulo | Columnas | Ejemplo en `data/` |
|---|---|---|
| DCA | tratamiento, réplica, respuesta | `dca_secadores.csv` (`secador,replica,humedad`) |
| DBCA | bloque, tratamiento, respuesta | `dbca_secadores.csv` (`dia,secador,humedad`) |
| DCL | fila, columna, tratamiento, respuesta | `dcl_pretratamientos.csv` (`dia,bandeja,pretratamiento,humedad`) |
| DPD | bloque, parcela principal, subparcela, respuesta | `dpd_secador_espesor.csv` (`dia,secador,espesor,humedad`) |
| Factoriales | 2 o más factores, réplica, respuesta; diseño balanceado con todas las combinaciones | `factorial_carga_pretrat.csv` (`carga,pretratamiento,replica,humedad`) |
| RSM primer orden | x1, x2 codificadas y respuesta; factorial 2² más puntos centrales (0, 0) repetidos | `rsm_primer_orden.csv` (`x1,x2,eficiencia`) |
| RSM segundo orden | x1, x2 codificadas y respuesta; diseño central compuesto con puntos axiales y centrales repetidos | `rsm_ccd.csv` (`x1,x2,eficiencia`) |

Cada página usa por defecto su archivo de ejemplo si no se carga ninguno.

## Instalación y ejecución

Dependencias: `streamlit`, `pandas`, `numpy`, `scipy`, `statsmodels`, `plotly`, `openpyxl` y `matplotlib` (esta última solo para exportar figuras PNG y PDF).

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

## Verificación

Los análisis de la app se comprueban contra R y contra cálculos hechos a mano. Los scripts leen archivos propios del estudiante (`*_mi_variante.csv`) en `data/`.

**Scripts de R** (ejecutar desde la carpeta `verificacion/`):

```bash
Rscript verificacion_dca.R          # DCA: ANOVA, medias, CV y Tukey
Rscript verificacion_factorial.R    # Factorial: ANOVA, medias de celda, gráfica de interacción y Tukey
Rscript verificacion_rsm.R          # RSM: rsm() de primer y segundo orden, steepest(), canonical(), persp() y contour()
```

`verificacion_rsm.R` instala el paquete `rsm` si no está disponible. Los gráficos de R se guardan en `Rplots.pdf`.

**Prueba rápida en Python** (desde la raíz del proyecto):

```bash
python verificacion/prueba_rapida.py
```

Compara la app con los valores calculados a mano que se escriban en `MIS_VALORES_A_MANO` (SC y F de DBCA, DCL, DPD y factorial; b0, b1, b2, b12 y F de curvatura del RSM de primer orden; xs1, xs2, ys, lambda1 y lambda2 del CCD). Los valores que queden en `None` se omiten, y hay coincidencia si la diferencia es menor que 0.01.

## Autor

Justo Fuentes – Curso de Diseño Experimental, Ingeniería Agrícola, Universidad de Sucre (semestre 2026-2).

## Uso de IA

Este proyecto se desarrolló con ayuda de GitHub Copilot (agente). Cada prompt, lo que generó la IA, lo que se aceptó o corrigió y cómo se verificó (ejecución de la app y comparación con R) se registra en [BITACORA_IA.md](BITACORA_IA.md). El autor es responsable de revisar e interpretar los resultados.

## Referencias

- Montgomery, D. C. *Diseño y análisis de experimentos*. Limusa Wiley.
- Gutiérrez Pulido, H. y De la Vara Salazar, R. *Análisis y diseño de experimentos*. McGraw-Hill.
- Lenth, R. V. (2009). Response-Surface Methods in R, Using rsm. *Journal of Statistical Software*, 32(7).
