"""Módulo RSM: Metodología de Superficie de Respuesta."""

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.anova_utils import cargar_datos
from utils.rsm_coding import decodificar
from utils.rsm_models import ajustar_primer_orden, prueba_curvatura, trayectoria_ascenso

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "rsm_primer_orden.csv"

st.set_page_config(page_title="RSM", page_icon="🌱", layout="wide")


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


st.title("RSM – Metodología de Superficie de Respuesta")
st.write(
    "Conjunto de técnicas que ajustan un modelo polinomial para describir cómo responde una "
    "variable a factores cuantitativos y buscar las condiciones óptimas. Las variables se "
    "trabajan **codificadas** (x1, x2) y se muestran también en **unidades naturales**."
)

# Barra lateral: definición de las variables naturales
with st.sidebar:
    st.header("Variables naturales")
    nom1 = st.text_input("Nombre de la variable 1", "Inclinación")
    uni1 = st.text_input("Unidad de la variable 1", "°")
    cen1 = st.number_input(f"Centro de {nom1}", value=15.0)
    sem1 = st.number_input(f"Semirango de {nom1}", value=5.0, min_value=1e-6)
    st.divider()
    nom2 = st.text_input("Nombre de la variable 2", "Ventilación")
    uni2 = st.text_input("Unidad de la variable 2", "m/s")
    cen2 = st.number_input(f"Centro de {nom2}", value=10.0)
    sem2 = st.number_input(f"Semirango de {nom2}", value=4.0, min_value=1e-6)
    st.caption("Codificación: x = (ξ − centro) / semirango")
    alfa = st.slider("Nivel de significancia (α)", 0.01, 0.10, 0.05, step=0.01)

etq1 = f"{nom1} ({uni1})"
etq2 = f"{nom2} ({uni2})"

tabs = st.tabs(
    [
        "1. Primer orden",
        "2. Trayectoria",
        "3. Segundo orden",
        "4. Punto estacionario",
        "5. Superficie 3D",
        "6. Curvas de nivel",
        "7. Exportar",
    ]
)
tab1, tab2 = tabs[0], tabs[1]
for tab in tabs[2:]:
    with tab:
        st.info("En construcción")

with tab1:
    archivo = st.file_uploader("Cargue sus datos (CSV o Excel)", type=["csv", "xlsx"])
    try:
        if archivo is not None:
            df = leer_datos(archivo.getvalue(), archivo.name)
        else:
            df = leer_datos(RUTA_EJEMPLO.read_bytes(), RUTA_EJEMPLO.name)
            st.caption("No se cargó archivo: se usan los datos de ejemplo `data/rsm_primer_orden.csv`.")
    except Exception as error:
        st.error(f"No se pudo leer el archivo: {error}")
        st.stop()

    numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numericas) < 3:
        st.error("El archivo debe tener al menos 3 columnas numéricas (x1, x2 y la respuesta).")
        st.stop()
    c1, c2, c3 = st.columns(3)
    col_x1 = c1.selectbox("Columna x1 (codificada)", numericas, index=0)
    col_x2 = c2.selectbox("Columna x2 (codificada)", numericas, index=1)
    col_y = c3.selectbox("Columna de respuesta", numericas, index=len(numericas) - 1)
    if len({col_x1, col_x2, col_y}) < 3:
        st.error("Seleccione tres columnas distintas.")
        st.stop()
    datos = df[[col_x1, col_x2, col_y]].dropna()

    st.subheader("Datos")
    tabla_datos = datos.copy()
    tabla_datos[etq1] = decodificar(datos[col_x1], cen1, sem1)
    tabla_datos[etq2] = decodificar(datos[col_x2], cen2, sem2)
    st.dataframe(tabla_datos, hide_index=True, width="stretch")

    modelo, coefs, r2 = ajustar_primer_orden(datos, col_y, col_x1, col_x2)
    st.subheader("Modelo de primer orden")
    st.dataframe(
        coefs.style.format({"Coeficiente": "{:.4f}", "Error estándar": "{:.4f}", "t": "{:.4f}", "p-valor": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    b0, b1, b2 = (float(v) for v in modelo.params)
    st.latex(rf"\hat y = {b0:.4f} {b1:+.4f}\,x_1 {b2:+.4f}\,x_2")
    st.metric("R²", f"{r2:.4f}")

    st.subheader("Prueba de curvatura")
    try:
        curv = prueba_curvatura(datos, col_y, col_x1, col_x2, alfa)
    except ValueError as error:
        curv = None
        st.warning(f"No se puede probar la curvatura: {error}")
    if curv:
        st.dataframe(
            pd.DataFrame(
                {
                    "Concepto": ["n factoriales", "n centrales", "ȳ factoriales", "ȳ centrales",
                                 "SC curvatura", "s² (centros)", "F", "gl", "p-valor"],
                    "Valor": [curv["nF"], curv["nC"], f"{curv['yF']:.4f}", f"{curv['yC']:.4f}",
                              f"{curv['sc_curvatura']:.4f}", f"{curv['s2']:.4f}", f"{curv['F']:.4f}",
                              f"({curv['gl'][0]}, {curv['gl'][1]})", f"{curv['p_valor']:.4f}"],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        (st.error if curv["significativa"] else st.success)(curv["conclusion"])
        st.subheader("Interpretación")
        if curv["significativa"]:
            st.write("El modelo de primer orden **no sirve** como descripción final: hay curvatura.")
        else:
            st.write(
                f"El modelo de primer orden **sirve** (R² = {r2:.2%}, sin curvatura significativa): "
                "puede usarse para la trayectoria de máximo ascenso (pestaña 2)."
            )

with tab2:
    pasos = st.number_input("Número de pasos", min_value=1, max_value=20, value=5)
    if abs(b1) == 0 and abs(b2) == 0:
        st.error("Los coeficientes b1 y b2 son cero: no hay dirección de ascenso.")
    else:
        tray = trayectoria_ascenso(b1, b2, int(pasos))
        # ŷ en cada paso con el modelo ajustado: b0 + b1·x1 + b2·x2
        tray["ŷ predicho"] = b0 + b1 * tray["x1"] + b2 * tray["x2"]
        tray[etq1] = decodificar(tray["x1"], cen1, sem1)
        tray[etq2] = decodificar(tray["x2"], cen2, sem2)
        tray = tray.rename(columns={"paso": "Paso", "x1": "x1 (cod.)", "x2": "x2 (cod.)"})
        st.dataframe(
            tray.style.format({"x1 (cod.)": "{:.3f}", "x2 (cod.)": "{:.3f}", "ŷ predicho": "{:.3f}",
                               etq1: "{:.3f}", etq2: "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datos[col_x1], y=datos[col_x2], mode="markers", name="Puntos del diseño",
                                 marker=dict(size=11, color="#1f77b4")))
        fig.add_trace(go.Scatter(x=tray["x1 (cod.)"], y=tray["x2 (cod.)"], mode="markers+text",
                                 text=tray["Paso"].astype(str), textposition="top center",
                                 name="Trayectoria", marker=dict(size=8, color="#d62728")))
        # Flecha desde el centro hasta el último paso
        fig.add_annotation(x=tray["x1 (cod.)"].iloc[-1], y=tray["x2 (cod.)"].iloc[-1],
                           ax=tray["x1 (cod.)"].iloc[0], ay=tray["x2 (cod.)"].iloc[0],
                           xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.5, arrowwidth=2, arrowcolor="#d62728")
        fig.update_layout(xaxis_title="x1 (codificada)", yaxis_title="x2 (codificada)")
        st.plotly_chart(fig, width="stretch")
        st.caption("La flecha indica la dirección de máximo ascenso desde el centro del diseño.")
