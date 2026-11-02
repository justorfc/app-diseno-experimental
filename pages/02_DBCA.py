"""Módulo DBCA: Diseño de Bloques Completos al Azar."""

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.anova_utils import cargar_datos
from utils.bloques_utils import (
    ajustar_dbca,
    cv_dbca,
    eficiencia_relativa,
    f_critica_dbca,
    interpretar_eficiencia,
    tabla_anova_dbca,
    validar_dbca,
)
from utils.comparaciones_utils import letras_tukey, tukey_con_cme
from utils.diagnostico_utils import (
    grafico_qq,
    grafico_residuos_vs_ajustados,
    prueba_shapiro,
    residuos_y_ajustados,
)

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "dbca_secadores.csv"

st.set_page_config(page_title="DBCA", page_icon="🌱", layout="wide")


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


st.title("DBCA – Diseño de Bloques Completos al Azar")
st.write(
    "En el DBCA las unidades experimentales se agrupan en bloques homogéneos y cada "
    "tratamiento aparece una vez en cada bloque. El bloqueo separa del error la variación "
    "debida al bloque (por ejemplo, el día de secado). "
    "**Ejemplo:** humedad final (% b.h.) de la yuca en 4 secadores (A–D), repetidos en "
    "4 días; cada día es un bloque."
)

archivo = st.file_uploader("Cargue sus datos (CSV o Excel)", type=["csv", "xlsx"])
try:
    if archivo is not None:
        df = leer_datos(archivo.getvalue(), archivo.name)
    else:
        df = leer_datos(RUTA_EJEMPLO.read_bytes(), RUTA_EJEMPLO.name)
        st.caption("No se cargó archivo: se usan los datos de ejemplo `data/dbca_secadores.csv`.")
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

# Selección de columnas: por defecto bloque, tratamiento y respuesta en ese orden
numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
no_numericas = [c for c in df.columns if c not in numericas]
col_trat_def = no_numericas[0] if no_numericas else df.columns[0]
col_resp_def = [c for c in numericas if c != col_trat_def][-1:] or [df.columns[-1]]
col_blo_def = next(
    (c for c in df.columns if c not in (col_trat_def, col_resp_def[0])), df.columns[0]
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    col_bloque = st.selectbox("Columna de bloques", df.columns, index=df.columns.get_loc(col_blo_def))
with c2:
    col_trat = st.selectbox("Columna de tratamientos", df.columns, index=df.columns.get_loc(col_trat_def))
with c3:
    col_resp = st.selectbox("Columna de respuesta", df.columns, index=df.columns.get_loc(col_resp_def[0]))
with c4:
    alfa = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, step=0.01)

errores = validar_dbca(df, col_bloque, col_trat, col_resp)
for mensaje in errores:
    st.error(mensaje)
if errores:
    st.stop()

# Cálculos (todos provienen de utils/bloques_utils.py)
modelo = ajustar_dbca(df, col_bloque, col_trat, col_resp)
anova = tabla_anova_dbca(modelo)
f_crit = f_critica_dbca(anova, alfa)
cv = cv_dbca(anova, df, col_resp)
t = df[col_trat].nunique()
b = df[col_bloque].nunique()
fila = anova.set_index("Fuente")
cme = float(fila.loc["Error", "CM"])
gl_error = int(fila.loc["Error", "gl"])
p_trat = float(fila.loc["Tratamientos", "p-valor"])
p_bloq = float(fila.loc["Bloques", "p-valor"])
er = eficiencia_relativa(float(fila.loc["Bloques", "CM"]), cme, t, b)
significativo = p_trat < alfa

tab_datos, tab_anova, tab_comp, tab_graf, tab_sup = st.tabs(
    ["Datos", "ANOVA", "Comparaciones", "Gráficos", "Supuestos"]
)

with tab_datos:
    st.subheader("Datos cargados")
    st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("Tabla de doble entrada (tratamiento × bloque)")
    doble = df.pivot(index=col_trat, columns=col_bloque, values=col_resp)
    doble.columns = doble.columns.astype(str)
    doble["Media"] = doble.mean(axis=1)
    doble.loc["Media"] = doble.mean(axis=0)
    st.dataframe(doble.style.format("{:.4f}"), width="stretch")

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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("F calculada (trat.)", f"{fila.loc['Tratamientos', 'F']:.4f}")
    m2.metric(f"F crítica (α = {alfa:.2f})", f"{f_crit:.4f}")
    m3.metric("Coeficiente de variación", f"{cv:.2f} %")
    m4.metric("Eficiencia relativa", f"{er:.2f}")
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
    st.write(
        f"Efecto de bloques: p = {p_bloq:.4f} "
        f"({'significativo' if p_bloq < alfa else 'no significativo'} a α = {alfa:.2f})."
    )
    st.subheader("Eficiencia relativa del bloqueo")
    st.write(interpretar_eficiencia(er, b))

with tab_comp:
    if significativo:
        st.subheader("Prueba de Tukey (con el CM Error del modelo con bloques)")
        medias = df.groupby(col_trat)[col_resp].mean()
        comparaciones, dms = tukey_con_cme(medias, cme, gl_error, r=b, alfa=alfa)
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
    df_graf[col_trat] = df_graf[col_trat].astype(str)
    df_graf[col_bloque] = df_graf[col_bloque].astype(str)
    st.subheader("Boxplot por tratamiento")
    fig_box = px.box(df_graf, x=col_trat, y=col_resp, color=col_trat, points="all")
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, width="stretch")

    st.subheader("Gráfico de perfiles (una línea por bloque)")
    fig_perf = px.line(df_graf, x=col_trat, y=col_resp, color=col_bloque, markers=True)
    st.plotly_chart(fig_perf, width="stretch")
    st.caption("Líneas aproximadamente paralelas indican que no hay interacción bloque × tratamiento.")

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
**Aditividad bloque × tratamiento:** el DBCA supone que el efecto de un tratamiento es el
mismo en todos los bloques (sin interacción). Como hay una sola observación por celda, la
interacción no se puede probar con el ANOVA: se revisa en el gráfico de perfiles (las líneas
deben ser casi paralelas). Si se cruzan claramente, el modelo aditivo no es adecuado.

**Independencia:** se garantiza con la aleatorización dentro de cada bloque.
"""
    )
