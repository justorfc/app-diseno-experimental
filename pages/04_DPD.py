"""Módulo DPD: Diseño de Parcelas Divididas en bloques."""

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.anova_utils import cargar_datos
from utils.diagnostico_utils import (
    grafico_qq,
    grafico_residuos_vs_ajustados,
    prueba_shapiro,
    residuos_y_ajustados,
)
from utils.dpd_utils import (
    SEPARADOR,
    ajustar_modelo_completo,
    anova_parcelas_divididas,
    coeficientes_variacion,
    f_con_error_combinado,
    f_criticas,
    validar_dpd,
)

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "dpd_secador_espesor.csv"

st.set_page_config(page_title="DPD", page_icon="🌱", layout="wide")


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


st.title("DPD – Parcelas Divididas en bloques")
st.write(
    "En las parcelas divididas el factor **A** (difícil de cambiar) se aplica a parcelas "
    "grandes y el factor **B** a subparcelas dentro de cada parcela grande. Hay dos errores: "
    "Error (a) para A y Error (b) para B y A×B. "
    "**Ejemplo:** humedad final de yuca con 2 secadores (A, parcela principal) y 3 espesores "
    "de rodaja (B, subparcela), repetido en 3 días (bloques)."
)

archivo = st.file_uploader("Cargue sus datos (CSV o Excel)", type=["csv", "xlsx"])
try:
    if archivo is not None:
        df = leer_datos(archivo.getvalue(), archivo.name)
    else:
        df = leer_datos(RUTA_EJEMPLO.read_bytes(), RUTA_EJEMPLO.name)
        st.caption("No se cargó archivo: se usan los datos de ejemplo `data/dpd_secador_espesor.csv`.")
except Exception as error:
    st.error(f"No se pudo leer el archivo: {error}")
    st.stop()

numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def_resp = numericas[-1] if numericas else df.columns[-1]
defecto = ([c for c in df.columns if c != def_resp] + list(df.columns))[:3]
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    bloque = st.selectbox("Bloque", df.columns, index=df.columns.get_loc(defecto[0]))
with c2:
    factor_a = st.selectbox("Factor A (parcela principal)", df.columns, index=df.columns.get_loc(defecto[1]))
with c3:
    factor_b = st.selectbox("Factor B (subparcela)", df.columns, index=df.columns.get_loc(defecto[2]))
with c4:
    resp = st.selectbox("Respuesta", df.columns, index=df.columns.get_loc(def_resp))
with c5:
    alfa = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, step=0.01)

errores = validar_dpd(df, bloque, factor_a, factor_b, resp)
for mensaje in errores:
    st.error(mensaje)
if errores:
    st.stop()

# Cálculos (todos provienen de utils/dpd_utils.py)
tabla = anova_parcelas_divididas(df, bloque, factor_a, factor_b, resp)
fila = tabla.set_index("Fuente")
f_crit = f_criticas(tabla, alfa)
cv_a, cv_b = coeficientes_variacion(tabla, float(df[resp].mean()))
gl_ea = int(fila.loc["Error (a)", "gl"])
if gl_ea < 4:
    st.warning(
        "La parcela principal tiene pocos grados de libertad; la prueba de A tiene poca potencia."
    )

tab_est, tab_anova, tab_med, tab_sup = st.tabs(
    ["Estructura", "ANOVA", "Medias e interacción", "Supuestos"]
)

with tab_est:
    st.markdown(
        """
- **Parcela principal:** cada bloque se divide en parcelas grandes, una por nivel de **A**.
  Se comparan con **Error (a)** (interacción bloque × A), que es el error de las unidades grandes.
- **Subparcela:** cada parcela grande se divide en subparcelas, una por nivel de **B**.
  B y A×B se comparan con **Error (b)**, normalmente más pequeño porque las subparcelas
  vecinas se parecen más entre sí.
- **Aleatorización:** A se sortea entre las parcelas grandes de cada bloque y B se sortea
  dentro de cada parcela grande.
"""
    )
    bloque_ej = df[bloque].iloc[0]
    st.subheader(f"Croquis del bloque «{bloque_ej}»")
    sub = df[df[bloque] == bloque_ej]
    niveles_a = list(dict.fromkeys(sub[factor_a]))
    fig = go.Figure()
    for i, nivel_a in enumerate(niveles_a):
        x0 = i * 1.2
        fig.add_shape(type="rect", x0=x0, x1=x0 + 1, y0=0, y1=1,
                      line={"width": 3, "color": "#1b5e20"}, fillcolor="#e8f5e9")
        fig.add_annotation(x=x0 + 0.5, y=1.08, text=f"<b>{factor_a} = {nivel_a}</b>", showarrow=False)
        niveles_b = list(dict.fromkeys(sub.loc[sub[factor_a] == nivel_a, factor_b]))
        ancho = 1 / len(niveles_b)
        for k, nivel_b in enumerate(niveles_b):
            fig.add_shape(type="rect", x0=x0 + k * ancho, x1=x0 + (k + 1) * ancho,
                          y0=0.1, y1=0.8, line={"width": 1, "color": "#388e3c"}, fillcolor="#c8e6c9")
            valor = sub.loc[(sub[factor_a] == nivel_a) & (sub[factor_b] == nivel_b), resp].iloc[0]
            fig.add_annotation(x=x0 + (k + 0.5) * ancho, y=0.45,
                               text=f"{nivel_b}<br>{valor:g}", showarrow=False)
    fig.update_xaxes(visible=False, range=[-0.1, len(niveles_a) * 1.2])
    fig.update_yaxes(visible=False, range=[-0.05, 1.2])
    fig.update_layout(height=280, margin={"t": 10, "b": 10}, plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
    st.caption("Recuadro grande = parcela principal (A); recuadros pequeños = subparcelas (B) con su respuesta.")
    st.subheader("Datos cargados")
    st.dataframe(df, hide_index=True, width="stretch")

with tab_anova:
    st.subheader("Tabla ANOVA")
    st.dataframe(
        tabla.style.format(
            {"gl": "{:.0f}", "SC": "{:.4f}", "CM": "{:.4f}", "F": "{:.4f}", "p-valor": "{:.4f}"},
            na_rep="",
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"A = {factor_a} · B = {factor_b}. Bloques y A se prueban con Error (a); "
        "B y A×B con Error (b)."
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"F crít. A (α={alfa:.2f})", f"{f_crit['A']:.4f}")
    m2.metric("F crít. B", f"{f_crit['B']:.4f}")
    m3.metric("F crít. A×B", f"{f_crit['A×B']:.4f}")
    m4.metric("CV (a)", f"{cv_a:.2f} %")
    m5.metric("CV (b)", f"{cv_b:.2f} %")

    st.subheader("Interpretación")
    for nombre, etiqueta in (("A", factor_a), ("B", factor_b), ("A×B", f"{factor_a} × {factor_b}")):
        p = float(fila.loc[nombre, "p-valor"])
        if p < alfa:
            st.success(f"**{etiqueta}:** p = {p:.4f} < α: efecto significativo.")
        else:
            st.info(f"**{etiqueta}:** p = {p:.4f} ≥ α: no hay evidencia de efecto.")
    if float(fila.loc["A×B", "p-valor"]) < alfa:
        st.write(
            "Como la interacción es significativa, el efecto de un factor depende del nivel "
            "del otro: interprete los efectos simples (las medias A×B) más que los principales."
        )

    with st.expander("¿Qué pasa si uso un solo error?"):
        f_c, p_c, gl_c = f_con_error_combinado(tabla, "A")
        f_ok, p_ok = float(fila.loc["A", "F"]), float(fila.loc["A", "p-valor"])
        st.write(
            f"Con un error combinado (gl = {gl_c:.0f}) la F de A sería **{f_c:.4f}** "
            f"(p = {p_c:.4f}); con el Error (a) correcto es **{f_ok:.4f}** (p = {p_ok:.4f})."
        )
        st.markdown(
            """
Es incorrecto porque A se aplicó a parcelas grandes: su unidad experimental es la parcela
principal, no la subparcela. Mezclar Error (a) con Error (b) mide el factor A con un error
que no corresponde a sus unidades, infla los grados de libertad (parece que hay más
repeticiones de las reales) y, como el Error (b) suele ser menor, tiende a **declarar
significativo a A con demasiada facilidad**.
"""
        )

with tab_med:
    df_g = df.copy()
    for c in (factor_a, factor_b):
        df_g[c] = df_g[c].astype(str)
    st.subheader(f"Medias de A ({factor_a})")
    st.dataframe(df_g.groupby(factor_a)[resp].mean().to_frame("Media").style.format("{:.4f}"), width="stretch")
    st.subheader(f"Medias de B ({factor_b})")
    st.dataframe(df_g.groupby(factor_b)[resp].mean().to_frame("Media").style.format("{:.4f}"), width="stretch")
    st.subheader("Medias de A × B")
    medias_ab = df_g.pivot_table(index=factor_a, columns=factor_b, values=resp, aggfunc="mean")
    st.dataframe(medias_ab.style.format("{:.4f}"), width="stretch")
    st.subheader("Gráfica de interacción")
    ab = df_g.groupby([factor_a, factor_b])[resp].mean().reset_index()
    fig_int = px.line(ab, x=factor_b, y=resp, color=factor_a, markers=True)
    fig_int.update_layout(yaxis_title=f"Media de {resp}")
    st.plotly_chart(fig_int, width="stretch")
    st.caption("Líneas paralelas indican ausencia de interacción; líneas que se cruzan o divergen, interacción.")

with tab_sup:
    st.subheader("Diagnóstico de supuestos (residuos del modelo completo)")
    modelo = ajustar_modelo_completo(df, bloque, factor_a, factor_b, resp)
    residuos, ajustados = residuos_y_ajustados(modelo)
    w, p_sw, concl = prueba_shapiro(residuos, alfa)
    g1, g2 = st.columns(2)
    g1.plotly_chart(grafico_residuos_vs_ajustados(residuos, ajustados), width="stretch")
    g2.plotly_chart(grafico_qq(residuos), width="stretch")
    st.dataframe(
        pd.DataFrame(
            {"Prueba": ["Shapiro-Wilk (normalidad)"], "Estadístico": [w], "p-valor": [p_sw], "Conclusión": [concl]}
        ).style.format({"Estadístico": "{:.4f}", "p-valor": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Los residuos del modelo completo corresponden al Error (b); con pocos datos las pruebas "
        "tienen poca potencia, por eso conviene apoyarse en las gráficas."
    )
