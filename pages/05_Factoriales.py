"""Módulo de Diseños Factoriales (2 o más factores en DCA)."""

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import f as f_dist

from utils.anova_utils import cargar_datos
from utils.comparaciones_utils import letras_tukey, tukey_con_cme
from utils.diagnostico_utils import (
    grafico_qq,
    grafico_residuos_vs_ajustados,
    prueba_levene,
    prueba_shapiro,
    residuos_y_ajustados,
)
from utils.factorial_utils import (
    ajustar_factorial,
    anova_factorial,
    medias_celdas,
    validar_factorial,
)

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "factorial_carga_pretrat.csv"

st.set_page_config(page_title="Factoriales", page_icon="🌱", layout="wide")


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


def formato_tabla(tabla: pd.DataFrame):
    """Da formato numérico a la tabla ANOVA.

    Args:
        tabla: Tabla ANOVA.

    Returns:
        Tabla con estilo para mostrar en Streamlit.
    """
    return tabla.style.format(
        {"gl": "{:.0f}", "SC": "{:.4f}", "CM": "{:.4f}", "F": "{:.4f}", "p-valor": "{:.4f}"},
        na_rep="",
    )


st.title("Diseños Factoriales")
st.write(
    "Arreglos en los que se evalúan al mismo tiempo todas las combinaciones de los niveles "
    "de dos o más factores. Permiten medir el efecto de cada factor y su **interacción**. "
    "**Ejemplo:** humedad final (%) de un producto secado con 3 cargas del secador "
    "(2, 4 y 6 kg) y 2 pretratamientos (P0 y P1), con 3 réplicas por combinación."
)

archivo = st.file_uploader("Cargue sus datos (CSV o Excel)", type=["csv", "xlsx"])
try:
    if archivo is not None:
        df = leer_datos(archivo.getvalue(), archivo.name)
    else:
        df = leer_datos(RUTA_EJEMPLO.read_bytes(), RUTA_EJEMPLO.name)
        st.caption(
            "No se cargó archivo: se usan los datos de ejemplo `data/factorial_carga_pretrat.csv`."
        )
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

# Valores por defecto: respuesta = última numérica; factores = las dos primeras columnas
numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def_resp = numericas[-1] if numericas else df.columns[-1]
def_factores = [c for c in df.columns if c != def_resp][:2]
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    factores = st.multiselect("Factores (2 o más)", list(df.columns), default=def_factores)
with c2:
    col_resp = st.selectbox("Columna de respuesta", df.columns, index=df.columns.get_loc(def_resp))
with c3:
    alfa = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, step=0.01)

errores = validar_factorial(df, factores, col_resp)
for mensaje in errores:
    st.error(mensaje)
if errores:
    st.stop()

# Cálculos (todos provienen de utils/factorial_utils.py)
anova = anova_factorial(df, factores, col_resp)
modelo = ajustar_factorial(df, factores, col_resp)
fila = anova.set_index("Fuente")
gl_error = int(fila.loc["Error", "gl"])
cme = float(fila.loc["Error", "CM"])
# Réplicas por celda (el diseño ya se validó como balanceado)
n_rep = int(df.groupby(factores).size().iloc[0])
efectos = [f for f in anova["Fuente"] if f not in ("Error", "Total")]
# F crítica de cada efecto: F(1 - alfa; gl efecto, gl error)
f_crit = {e: float(f_dist.ppf(1 - alfa, fila.loc[e, "gl"], gl_error)) for e in efectos}
# Coeficiente de variación: CV = sqrt(CME) / media general * 100
cv = float(cme**0.5 / df[col_resp].mean() * 100)
inter = [e for e in efectos if " × " in e]
principales = [e for e in efectos if " × " not in e]
inter_signif = [e for e in inter if fila.loc[e, "p-valor"] < alfa]
# Columna auxiliar con la combinación de niveles (celda)
df_celda = df.copy()
df_celda["_celda"] = df_celda[factores].astype(str).agg(" / ".join, axis=1)

tab_datos, tab_anova, tab_graf, tab_comp, tab_sup = st.tabs(
    ["Datos", "ANOVA", "Gráficos", "Comparaciones", "Supuestos"]
)

with tab_datos:
    st.subheader("Datos cargados")
    st.dataframe(df, hide_index=True, width="stretch")
    st.subheader("Medias de celda")
    st.dataframe(
        medias_celdas(df, factores, col_resp).style.format({"Media": "{:.4f}", "DE": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )

with tab_anova:
    st.subheader("Tabla ANOVA")
    st.dataframe(formato_tabla(anova), hide_index=True, width="stretch")
    st.caption(f"Suma de cuadrados tipo II; gl Error = {gl_error}.")
    st.subheader("F calculadas y críticas")
    col_fc = f"F crítica (α = {alfa:.2f})"
    st.dataframe(
        pd.DataFrame(
            {
                "Efecto": efectos,
                "F calculada": [fila.loc[e, "F"] for e in efectos],
                col_fc: [f_crit[e] for e in efectos],
                "p-valor": [fila.loc[e, "p-valor"] for e in efectos],
                "Significativo": ["Sí" if fila.loc[e, "p-valor"] < alfa else "No" for e in efectos],
            }
        ).style.format({"F calculada": "{:.4f}", col_fc: "{:.4f}", "p-valor": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.metric("Coeficiente de variación", f"{cv:.2f} %")

    st.subheader("Interpretación")
    # Se interpreta primero la interacción: condiciona la lectura de los efectos principales
    for e in sorted(inter, key=lambda x: -x.count("×")):
        p = float(fila.loc[e, "p-valor"])
        if p < alfa:
            st.success(
                f"**Interacción {e}:** p = {p:.4f} < α = {alfa:.2f}. Es significativa: el efecto "
                "de un factor depende del nivel del otro."
            )
        else:
            st.info(
                f"**Interacción {e}:** p = {p:.4f} ≥ α = {alfa:.2f}. No es significativa: no hay "
                "evidencia de que los factores actúen de forma dependiente."
            )
    if inter_signif:
        st.warning(
            "Como hay interacción significativa, los efectos principales **no deben "
            "interpretarse solos**; conviene comparar las combinaciones (medias de celda)."
        )
    for e in principales:
        p = float(fila.loc[e, "p-valor"])
        st.write(
            f"**{e}:** p = {p:.4f} → "
            + ("efecto significativo." if p < alfa else "efecto no significativo.")
        )

with tab_graf:
    df_graf = df.copy()
    # Orden natural de niveles (numérico si el factor es numérico)
    orden = {f: [str(v) for v in sorted(df[f].unique())] for f in factores}
    for f in factores:
        df_graf[f] = df_graf[f].astype(str)

    st.subheader("Efectos principales")
    cols = st.columns(min(len(factores), 3))
    for i, f in enumerate(factores):
        medias_f = df_graf.groupby(f)[col_resp].mean().reindex(orden[f]).reset_index()
        fig = px.line(medias_f, x=f, y=col_resp, markers=True, title=f"Media por nivel de {f}")
        fig.update_xaxes(type="category")
        fig.update_layout(yaxis_title=f"Media de {col_resp}")
        cols[i % len(cols)].plotly_chart(fig, width="stretch")

    st.subheader("Gráfica de interacción")
    f1, f2 = factores[0], factores[1]
    medias_i = df_graf.groupby([f1, f2])[col_resp].mean().reset_index()
    fig_i = px.line(
        medias_i,
        x=f1,
        y=col_resp,
        color=f2,
        markers=True,
        category_orders={f1: orden[f1], f2: orden[f2]},
    )
    fig_i.update_xaxes(type="category")
    fig_i.update_layout(yaxis_title=f"Media de {col_resp}")
    st.plotly_chart(fig_i, width="stretch")
    st.caption(
        "Líneas paralelas indican ausencia de interacción; líneas que se cruzan o se separan "
        "indican que el efecto de un factor cambia con el nivel del otro."
        + (" Con 3 o más factores se promedia sobre los demás." if len(factores) > 2 else "")
    )

with tab_comp:
    if inter_signif:
        st.subheader("Tukey entre medias de celda (interacción significativa)")
        medias = df_celda.groupby("_celda")[col_resp].mean()
        comparaciones, dms = tukey_con_cme(medias, cme, gl_error, r=n_rep, alfa=alfa)
        st.dataframe(
            comparaciones.style.format({"Diferencia": "{:.4f}", "q": "{:.4f}", "p-valor": "{:.4f}"}),
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
        st.caption(f"Se usa el CM Error ({cme:.4f}) y los gl Error ({gl_error}) del modelo.")
    else:
        signif = [e for e in principales if fila.loc[e, "p-valor"] < alfa]
        if not signif:
            st.info(
                "No hay interacción ni efectos principales significativos: no se hacen "
                "comparaciones múltiples."
            )
        for e in signif:
            st.subheader(f"Tukey para {e}")
            medias = df.groupby(e)[col_resp].mean()
            # Observaciones por nivel = total / número de niveles del factor
            r_nivel = len(df) // df[e].nunique()
            comparaciones, dms = tukey_con_cme(medias, cme, gl_error, r=r_nivel, alfa=alfa)
            st.dataframe(
                comparaciones.style.format(
                    {"Diferencia": "{:.4f}", "q": "{:.4f}", "p-valor": "{:.4f}"}
                ),
                hide_index=True,
                width="stretch",
            )
            st.metric(f"DMS de Tukey ({e})", f"{dms:.4f}")
            st.dataframe(
                letras_tukey(medias, dms).style.format({"Media": "{:.4f}"}),
                hide_index=True,
                width="stretch",
            )

with tab_sup:
    st.subheader("Diagnóstico de supuestos")
    residuos, ajustados = residuos_y_ajustados(modelo)
    w, p_sw, concl_sw = prueba_shapiro(residuos, alfa)
    # Levene: cada combinación de niveles se trata como un grupo
    est_lv, p_lv, concl_lv = prueba_levene(df_celda, "_celda", col_resp, alfa)
    g1, g2 = st.columns(2)
    g1.plotly_chart(grafico_residuos_vs_ajustados(residuos, ajustados), width="stretch")
    g2.plotly_chart(grafico_qq(residuos), width="stretch")
    st.dataframe(
        pd.DataFrame(
            {
                "Prueba": ["Shapiro-Wilk (normalidad)", "Levene (homogeneidad)"],
                "Estadístico": [w, est_lv],
                "p-valor": [p_sw, p_lv],
                "Conclusión": [concl_sw, concl_lv],
            }
        ).style.format({"Estadístico": "{:.4f}", "p-valor": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.markdown(
        """
- **Normalidad:** los residuos deben ser aproximadamente normales (Shapiro-Wilk y Q-Q).
- **Varianza constante:** Levene compara las varianzas de las combinaciones de niveles.
- **Independencia:** se garantiza con la aleatorización completa de las unidades.
"""
    )
