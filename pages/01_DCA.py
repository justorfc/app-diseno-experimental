"""Módulo DCA: Diseño Completamente al Azar (un factor)."""

import io
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.anova_utils import (
    ajustar_dca,
    cargar_datos,
    coeficiente_variacion,
    f_critica,
    interpretar_dca,
    resumen_tratamientos,
    tabla_anova,
    tukey,
    validar_dca,
)

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "dca_secadores.csv"

st.set_page_config(page_title="DCA", page_icon="🌱", layout="wide")


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


st.title("DCA – Diseño Completamente al Azar")
st.write(
    "El DCA asigna los tratamientos a las unidades experimentales totalmente al azar. "
    "Se usa cuando el material es homogéneo y se estudia un solo factor. "
    "**Ejemplo:** se compara la humedad final (% b.h.) de la yuca secada en "
    "4 secadores solares (A, B, C y D), con 4 réplicas cada uno. "
    "El ANOVA indica si al menos un secador produce una humedad media distinta."
)

# Carga de datos: archivo del usuario o ejemplo por defecto
archivo = st.file_uploader("Cargue sus datos (CSV o Excel)", type=["csv", "xlsx"])
try:
    if archivo is not None:
        df = leer_datos(archivo.getvalue(), archivo.name)
    else:
        df = leer_datos(RUTA_EJEMPLO.read_bytes(), RUTA_EJEMPLO.name)
        st.caption("No se cargó archivo: se usan los datos de ejemplo `data/dca_secadores.csv`.")
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

# Selección de columnas y nivel de significancia
numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
no_numericas = [c for c in df.columns if c not in numericas]
col1, col2, col3 = st.columns(3)
with col1:
    # Por defecto: primera columna de texto como tratamiento
    por_defecto = df.columns.get_loc(no_numericas[0]) if no_numericas else 0
    col_trat = st.selectbox("Columna de tratamientos", df.columns, index=por_defecto)
with col2:
    # Por defecto: última columna numérica como respuesta
    candidatas = [c for c in numericas if c != col_trat] or list(df.columns)
    col_resp = st.selectbox(
        "Columna de respuesta", df.columns, index=df.columns.get_loc(candidatas[-1])
    )
with col3:
    alfa = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, step=0.01)

# Validaciones: si hay errores no se continúa
errores = validar_dca(df, col_trat, col_resp)
for mensaje in errores:
    st.error(mensaje)
if errores:
    st.stop()

# Cálculos (todos provienen de utils/anova_utils.py)
modelo = ajustar_dca(df, col_trat, col_resp)
anova = tabla_anova(modelo)
resumen = resumen_tratamientos(df, col_trat, col_resp)
cv = coeficiente_variacion(modelo, df, col_resp)
f_crit = f_critica(anova, alfa)
p_valor = float(anova.loc[anova["Fuente"] == "Tratamientos", "p-valor"].iloc[0])
significativo = p_valor < alfa

tab_datos, tab_anova, tab_comp, tab_graf = st.tabs(
    ["Datos", "ANOVA", "Comparaciones", "Gráficos"]
)

with tab_datos:
    st.subheader("Datos cargados")
    st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("Resumen por tratamiento")
    st.dataframe(
        resumen.style.format(
            {"Total": "{:.4f}", "Media": "{:.4f}", "Desviación estándar": "{:.4f}"}
        ),
        hide_index=True,
        width="stretch",
    )

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
    m1, m2, m3 = st.columns(3)
    m1.metric("F calculada", f"{anova.loc[0, 'F']:.4f}")
    m2.metric(f"F crítica (α = {alfa:.2f})", f"{f_crit:.4f}")
    m3.metric("Coeficiente de variación", f"{cv:.2f} %")
    texto = interpretar_dca(anova, alfa)
    if significativo:
        st.success(texto)
    else:
        st.info(texto)

with tab_comp:
    if significativo:
        st.subheader("Prueba de Tukey")
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            comparaciones, dms = tukey(df, col_trat, col_resp, alfa)
        for aviso in avisos:
            st.warning(str(aviso.message))
        st.dataframe(
            comparaciones.style.format(
                {
                    "Diferencia de medias": "{:.4f}",
                    "p ajustado": "{:.4f}",
                    "Límite inferior": "{:.4f}",
                    "Límite superior": "{:.4f}",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.metric("DMS de Tukey", f"{dms:.4f}")
        st.caption("Dos medias son distintas si su diferencia absoluta supera la DMS.")
    else:
        st.info(
            f"El p-valor ({p_valor:.4f}) no es menor que α ({alfa:.2f}), así que el ANOVA "
            "no detectó diferencias entre tratamientos. Las comparaciones múltiples solo se "
            "hacen cuando el ANOVA es significativo; de lo contrario aumentaría el riesgo "
            "de declarar diferencias que en realidad son producto del azar."
        )

with tab_graf:
    # Los tratamientos se tratan como texto para que el eje sea categórico
    df_graf = df.copy()
    df_graf[col_trat] = df_graf[col_trat].astype(str)
    resumen_graf = resumen.copy()
    resumen_graf[col_trat] = resumen_graf[col_trat].astype(str)

    st.subheader("Boxplot por tratamiento")
    fig_box = px.box(df_graf, x=col_trat, y=col_resp, color=col_trat, points="all")
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

    st.subheader("Medias por tratamiento")
    fig_med = px.line(
        resumen_graf, x=col_trat, y="Media", error_y="Desviación estándar", markers=True
    )
    fig_med.update_layout(yaxis_title=f"Media de {col_resp}")
    st.plotly_chart(fig_med, width="stretch")
    st.caption("Las barras de error muestran ± 1 desviación estándar.")
