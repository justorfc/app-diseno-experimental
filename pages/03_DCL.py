"""Módulo DCL: Diseño en Cuadro Latino."""

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.anova_utils import cargar_datos
from utils.bloques_utils import (
    ajustar_cuadro_latino,
    cv_dbca,
    f_critica_dbca,
    tabla_anova_dcl,
    validar_cuadro_latino,
)
from utils.comparaciones_utils import letras_tukey, tukey_con_cme
from utils.diagnostico_utils import (
    grafico_qq,
    grafico_residuos_vs_ajustados,
    prueba_shapiro,
    residuos_y_ajustados,
)

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "dcl_pretratamientos.csv"

st.set_page_config(page_title="DCL", page_icon="🌱", layout="wide")


@st.cache_data
def leer_datos(contenido: bytes, nombre: str) -> pd.DataFrame:
    """Lee los datos una sola vez y los guarda en caché.

    Args:
        contenido: Contenido del archivo en bytes.
        nombre: Nombre del archivo (su extensión define el formato).

    Returns:
        DataFrame con los datos.
    """
    archivo = io.BytesIO(contenido)
    archivo.name = nombre
    return cargar_datos(archivo)


st.title("DCL – Diseño en Cuadro Latino")
st.write(
    "El Cuadro Latino controla **dos** fuentes de variación a la vez (filas y columnas): "
    "cada tratamiento aparece exactamente una vez en cada fila y en cada columna. "
    "**Ejemplo:** humedad final (% b.h.) de yuca con 4 pretratamientos (T1–T4), "
    "en 4 días (filas) y 4 posiciones de bandeja (columnas)."
)

with st.expander("¿Qué controla cada factor?"):
    st.markdown(
        """
- **Fila = día de secado.** Cada día tiene clima distinto (temperatura, radiación, humedad
  del aire). Al usar el día como fila, esa variación se separa del error.
- **Columna = posición de la bandeja** (1 a 4) dentro del secador. Las bandejas cercanas a la
  entrada del aire se secan distinto a las del fondo; la columna controla ese gradiente.
- **Tratamiento = pretratamiento de la yuca** (T1–T4): es el factor de interés. Cada
  pretratamiento se prueba una vez cada día y una vez en cada posición, así ningún
  pretratamiento queda favorecido por el día o por la bandeja.
"""
    )

archivo = st.file_uploader("Cargue sus datos (CSV o Excel)", type=["csv", "xlsx"])
try:
    if archivo is not None:
        df = leer_datos(archivo.getvalue(), archivo.name)
    else:
        df = leer_datos(RUTA_EJEMPLO.read_bytes(), RUTA_EJEMPLO.name)
        st.caption(
            "No se cargó archivo: se usan los datos de ejemplo `data/dcl_pretratamientos.csv`."
        )
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

# Valores por defecto: respuesta = última numérica; resto en el orden del archivo
numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def_resp = numericas[-1] if numericas else df.columns[-1]
otras = [c for c in df.columns if c != def_resp]
defecto = (otras + list(df.columns))[:3]
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    col_fila = st.selectbox("Columna de filas", df.columns, index=df.columns.get_loc(defecto[0]))
with c2:
    col_col = st.selectbox("Columna de columnas", df.columns, index=df.columns.get_loc(defecto[1]))
with c3:
    col_trat = st.selectbox("Columna de tratamientos", df.columns, index=df.columns.get_loc(defecto[2]))
with c4:
    col_resp = st.selectbox("Columna de respuesta", df.columns, index=df.columns.get_loc(def_resp))
with c5:
    alfa = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, step=0.01)

errores = validar_cuadro_latino(df, col_fila, col_col, col_trat, col_resp)
for mensaje in errores:
    st.error(mensaje)
if errores:
    st.stop()

# Cálculos (todos provienen de utils/bloques_utils.py)
t = df[col_trat].nunique()
modelo = ajustar_cuadro_latino(df, col_fila, col_col, col_trat, col_resp)
anova = tabla_anova_dcl(modelo, t)
f_crit = f_critica_dbca(anova, alfa)
cv = cv_dbca(anova, df, col_resp)
fila = anova.set_index("Fuente")
cme = float(fila.loc["Error", "CM"])
gl_error = int(fila.loc["Error", "gl"])
p_trat = float(fila.loc["Tratamientos", "p-valor"])
significativo = p_trat < alfa

tab_datos, tab_anova, tab_comp, tab_graf, tab_sup = st.tabs(
    ["Croquis y datos", "ANOVA", "Comparaciones", "Gráficos", "Supuestos"]
)

with tab_datos:
    st.subheader("Croquis del experimento")
    resp_m = df.pivot(index=col_fila, columns=col_col, values=col_resp)
    trat_m = df.pivot(index=col_fila, columns=col_col, values=col_trat).astype(str)
    fig_cro = px.imshow(
        resp_m.to_numpy(),
        x=[str(c) for c in resp_m.columns],
        y=[str(f) for f in resp_m.index],
        color_continuous_scale="YlGnBu",
        labels={"x": col_col, "y": col_fila, "color": col_resp},
        aspect="auto",
    )
    # Nombre del tratamiento y valor de la respuesta escritos en cada celda
    fig_cro.update_traces(
        text=trat_m.to_numpy(), texttemplate="<b>%{text}</b><br>%{z:.1f}", hovertemplate=None
    )
    fig_cro.update_xaxes(type="category", side="top")
    fig_cro.update_yaxes(type="category")
    st.plotly_chart(fig_cro, width="stretch")
    st.caption("Color = respuesta; texto = tratamiento y valor observado en cada celda.")
    st.subheader("Datos cargados")
    st.dataframe(df, hide_index=True, width="stretch")

with tab_anova:
    st.subheader("Tabla ANOVA")
    st.dataframe(
        anova.style.format(
            {"gl": "{:.0f}", "SC": "{:.4f}", "CM": "{:.4f}", "F": "{:.4f}", "p-valor": "{:.4f}"},
            na_rep="",
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Comprobación: gl Error = (t-1)(t-2) = {(t - 1) * (t - 2)}.")
    m1, m2, m3 = st.columns(3)
    m1.metric("F calculada (trat.)", f"{fila.loc['Tratamientos', 'F']:.4f}")
    m2.metric(f"F crítica (α = {alfa:.2f})", f"{f_crit:.4f}")
    m3.metric("Coeficiente de variación", f"{cv:.2f} %")
    if significativo:
        st.success(
            f"p = {p_trat:.4f} < α = {alfa:.2f}: se rechaza H0; al menos un tratamiento "
            "tiene una media distinta."
        )
    else:
        st.info(
            f"p = {p_trat:.4f} ≥ α = {alfa:.2f}: no se rechaza H0; no hay evidencia de "
            "diferencias entre tratamientos."
        )
    for fuente in ("Filas", "Columnas"):
        p = float(fila.loc[fuente, "p-valor"])
        st.write(
            f"Efecto de {fuente.lower()}: p = {p:.4f} "
            f"({'significativo' if p < alfa else 'no significativo'} a α = {alfa:.2f}); "
            + ("el control fue útil." if p < alfa else "controlarlo no aportó una reducción clara del error.")
        )

with tab_comp:
    if significativo:
        st.subheader("Prueba de Tukey (con el CM Error del modelo)")
        medias = df.groupby(col_trat)[col_resp].mean()
        comparaciones, dms = tukey_con_cme(medias, cme, gl_error, r=t, alfa=alfa)
        st.dataframe(
            comparaciones.style.format(
                {"Diferencia": "{:.4f}", "q": "{:.4f}", "p-valor": "{:.4f}"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.metric("DMS de Tukey", f"{dms:.4f}")
        st.subheader("Grupos de medias")
        st.dataframe(
            letras_tukey(medias, dms).style.format({"Media": "{:.4f}"}),
            hide_index=True,
            width="stretch",
        )
        st.caption("Medias con una letra en común no difieren significativamente.")
    else:
        st.info(
            f"El p-valor ({p_trat:.4f}) no es menor que α ({alfa:.2f}); las comparaciones "
            "múltiples solo se hacen cuando el ANOVA es significativo."
        )

with tab_graf:
    df_graf = df.copy()
    for c in (col_fila, col_col, col_trat):
        df_graf[c] = df_graf[c].astype(str)

    st.subheader("Medias por tratamiento")
    res = df_graf.groupby(col_trat)[col_resp].agg(Media="mean", DE="std").reset_index()
    fig_t = px.line(res, x=col_trat, y="Media", error_y="DE", markers=True)
    fig_t.update_layout(yaxis_title=f"Media de {col_resp}")
    st.plotly_chart(fig_t, width="stretch")
    st.caption("Las barras de error muestran ± 1 desviación estándar.")

    g1, g2 = st.columns(2)
    for col_g, titulo, cont in ((col_fila, "Medias por fila", g1), (col_col, "Medias por columna", g2)):
        medias_g = df_graf.groupby(col_g)[col_resp].mean().reset_index()
        fig = px.bar(medias_g, x=col_g, y=col_resp, title=titulo)
        fig.update_layout(yaxis_title=f"Media de {col_resp}")
        cont.plotly_chart(fig, width="stretch")

with tab_sup:
    st.subheader("Diagnóstico de supuestos")
    residuos, ajustados = residuos_y_ajustados(modelo)
    w, p_sw, concl_sw = prueba_shapiro(residuos, alfa)
    g1, g2 = st.columns(2)
    g1.plotly_chart(grafico_residuos_vs_ajustados(residuos, ajustados), width="stretch")
    g2.plotly_chart(grafico_qq(residuos), width="stretch")
    st.dataframe(
        pd.DataFrame(
            {
                "Prueba": ["Shapiro-Wilk (normalidad)"],
                "Estadístico": [w],
                "p-valor": [p_sw],
                "Conclusión": [concl_sw],
            }
        ).style.format({"Estadístico": "{:.4f}", "p-valor": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        """
- **Normalidad:** los residuos deben ser aproximadamente normales (Shapiro-Wilk y Q-Q).
- **Varianza constante:** en residuos vs. ajustados no debe verse embudo ni curvas.
- **Aditividad:** el modelo supone que filas, columnas y tratamientos no interactúan; con una
  sola observación por celda esa interacción no puede probarse.
- **Independencia:** se garantiza con la aleatorización del cuadro.
"""
    )
