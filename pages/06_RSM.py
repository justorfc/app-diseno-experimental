"""Módulo RSM: Metodología de Superficie de Respuesta."""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.anova_utils import cargar_datos
from utils.rsm_coding import decodificar
from utils.rsm_mesh import malla, predecir_malla
from utils.rsm_plots import curvas_nivel, superficie_3d
from utils.rsm_models import (
    ajustar_primer_orden,
    ajustar_segundo_orden,
    prueba_curvatura,
    trayectoria_ascenso,
)
from utils.rsm_stationary import (
    clasificar,
    fuera_de_region,
    matriz_B,
    punto_estacionario,
    respuesta_en_punto,
)

RUTA_EJEMPLO = Path(__file__).resolve().parent.parent / "data" / "rsm_primer_orden.csv"
RUTA_CCD = Path(__file__).resolve().parent.parent / "data" / "rsm_ccd.csv"

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
    st.subheader("Primer orden (pestañas 1 y 2)")
    nom1 = st.text_input("Nombre de la variable 1", "Inclinación")
    uni1 = st.text_input("Unidad de la variable 1", "°")
    cen1 = st.number_input(f"Centro de {nom1}", value=15.0)
    sem1 = st.number_input(f"Semirango de {nom1}", value=5.0, min_value=1e-6)
    st.divider()
    nom2 = st.text_input("Nombre de la variable 2", "Ventilación")
    uni2 = st.text_input("Unidad de la variable 2", "m/s")
    cen2 = st.number_input(f"Centro de {nom2}", value=10.0)
    sem2 = st.number_input(f"Semirango de {nom2}", value=4.0, min_value=1e-6)
    st.divider()
    st.subheader("Diseño central compuesto (pestañas 3 y 4)")
    cen1c = st.number_input(f"Centro de {nom1} (CCD)", value=25.0)
    sem1c = st.number_input(f"Semirango de {nom1} (CCD)", value=3.0, min_value=1e-6)
    cen2c = st.number_input(f"Centro de {nom2} (CCD)", value=15.0)
    sem2c = st.number_input(f"Semirango de {nom2} (CCD)", value=2.0, min_value=1e-6)
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
tab1, tab2, tab3, tab4, tab5, tab6 = tabs[:6]
for tab in tabs[6:]:
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
                    "Valor": [str(curv["nF"]), str(curv["nC"]), f"{curv['yF']:.4f}", f"{curv['yC']:.4f}",
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

with tab3:
    archivo3 = st.file_uploader("Cargue sus datos del CCD (CSV o Excel)", type=["csv", "xlsx"], key="ccd")
    try:
        if archivo3 is not None:
            df3 = leer_datos(archivo3.getvalue(), archivo3.name)
        else:
            df3 = leer_datos(RUTA_CCD.read_bytes(), RUTA_CCD.name)
            st.caption("No se cargó archivo: se usan los datos de ejemplo `data/rsm_ccd.csv`.")
    except Exception as error:
        st.error(f"No se pudo leer el archivo: {error}")
        st.stop()
    num3 = [c for c in df3.columns if pd.api.types.is_numeric_dtype(df3[c])]
    if len(num3) < 3:
        st.error("El archivo debe tener al menos 3 columnas numéricas (x1, x2 y la respuesta).")
        st.stop()
    k1, k2, k3 = st.columns(3)
    c_x1 = k1.selectbox("Columna x1 (codificada)", num3, index=0, key="c3x1")
    c_x2 = k2.selectbox("Columna x2 (codificada)", num3, index=1, key="c3x2")
    c_y = k3.selectbox("Columna de respuesta", num3, index=len(num3) - 1, key="c3y")
    if len({c_x1, c_x2, c_y}) < 3:
        st.error("Seleccione tres columnas distintas.")
        st.stop()
    d3 = df3[[c_x1, c_x2, c_y]].dropna()
    if len(d3) < 7:
        st.error("Se necesitan al menos 7 observaciones para ajustar el modelo de segundo orden.")
        st.stop()

    st.subheader("Datos")
    t3 = d3.copy()
    t3[etq1] = decodificar(d3[c_x1], cen1c, sem1c)
    t3[etq2] = decodificar(d3[c_x2], cen2c, sem2c)
    st.dataframe(t3, hide_index=True, width="stretch")

    mod2, coef2, r2_2, r2a, falta = ajustar_segundo_orden(d3, c_y, c_x1, c_x2)
    st.subheader("Modelo de segundo orden")
    st.dataframe(
        coef2.style.format({"Estimación": "{:.4f}", "Error estándar": "{:.4f}", "t": "{:.4f}", "p-valor": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    m1, m2 = st.columns(2)
    m1.metric("R²", f"{r2_2:.4f}")
    m2.metric("R² ajustado", f"{r2a:.4f}")

    st.subheader("Falta de ajuste")
    if falta is None:
        st.info("No se puede probar la falta de ajuste: se necesitan réplicas en el centro.")
        sin_falta = None
    else:
        st.dataframe(
            pd.DataFrame(
                {
                    "Fuente": ["Falta de ajuste", "Error puro"],
                    "gl": [falta["gl_fa"], falta["gl_pe"]],
                    "SC": [falta["sc_fa"], falta["sc_pe"]],
                    "CM": [falta["sc_fa"] / falta["gl_fa"], falta["sc_pe"] / falta["gl_pe"]],
                    "F": [falta["F"], np.nan],
                    "p-valor": [falta["p_valor"], np.nan],
                }
            ).style.format({"SC": "{:.4f}", "CM": "{:.4f}", "F": "{:.4f}", "p-valor": "{:.4f}"}, na_rep=""),
            hide_index=True,
            width="stretch",
        )
        sin_falta = falta["p_valor"] >= alfa
    st.subheader("Interpretación")
    if sin_falta is None:
        st.write(f"El modelo explica el {r2_2:.1%} de la variación (R² ajustado = {r2a:.1%}), pero no se pudo verificar la falta de ajuste.")
    elif sin_falta:
        st.success(
            f"p = {falta['p_valor']:.4f} ≥ α = {alfa:.2f}: no hay falta de ajuste. El modelo "
            f"cuadrático describe bien los datos (R² ajustado = {r2a:.1%})."
        )
    else:
        st.error(
            f"p = {falta['p_valor']:.4f} < α = {alfa:.2f}: hay falta de ajuste. El modelo "
            "cuadrático no describe bien la superficie; considere otro modelo o más puntos."
        )

with tab4:
    p_ = [float(v) for v in mod2.params]
    e0, e1, e2, e11, e22, e12 = p_
    B = matriz_B(e11, e22, e12)
    vec_b = np.array([e1, e2])
    st.subheader("Matriz B y vector b")
    cb1, cb2 = st.columns(2)
    cb1.write("**Matriz B**")
    cb1.dataframe(pd.DataFrame(B, index=["x1", "x2"], columns=["x1", "x2"]).style.format("{:.4f}"), width="stretch")
    cb2.write("**Vector b**")
    cb2.dataframe(pd.DataFrame({"b": vec_b}, index=["b1", "b2"]).style.format("{:.4f}"), width="stretch")

    if abs(np.linalg.det(B)) < 1e-12:
        st.error("La matriz B es singular: no existe un único punto estacionario (superficie tipo cresta).")
        st.stop()
    xs = punto_estacionario(e1, e2, B)
    ys = respuesta_en_punto(e0, vec_b, xs)
    vp, tipo = clasificar(B)
    nat1 = decodificar(xs[0], cen1c, sem1c)
    nat2 = decodificar(xs[1], cen2c, sem2c)

    st.subheader("Punto estacionario")
    st.dataframe(
        pd.DataFrame(
            {
                "Variable": [nom1, nom2],
                "Codificada": [xs[0], xs[1]],
                "Natural": [nat1, nat2],
                "Unidad": [uni1, uni2],
            }
        ).style.format({"Codificada": "{:.4f}", "Natural": "{:.4f}"}),
        hide_index=True,
        width="stretch",
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("ŷ en el punto estacionario", f"{ys:.4f}")
    m2.metric("Valores propios", ", ".join(f"{v:.4f}" for v in vp))
    m3.metric("Clasificación", tipo)
    dist = float(np.linalg.norm(xs))
    fuera = fuera_de_region(xs)
    if fuera:
        st.warning(
            f"El punto estacionario está a {dist:.3f} unidades codificadas del centro (más de 1.414): "
            "queda fuera de la región experimental y la predicción es una extrapolación poco confiable."
        )

    st.subheader("Recomendación")
    resp_nombre = c_y
    if tipo == "máximo":
        texto = (
            f"La superficie tiene un **máximo**: la mejor {resp_nombre} esperada es de "
            f"**{ys:.2f}**, con {nom1} ≈ **{nat1:.2f} {uni1}** y {nom2} ≈ **{nat2:.2f} {uni2}**."
        )
    elif tipo == "mínimo":
        texto = (
            f"La superficie tiene un **mínimo**: el menor valor de {resp_nombre} esperado es "
            f"**{ys:.2f}**, con {nom1} ≈ **{nat1:.2f} {uni1}** y {nom2} ≈ **{nat2:.2f} {uni2}**."
        )
    else:
        texto = (
            "El punto es un **punto de silla**: la respuesta sube en una dirección y baja en otra, "
            "así que no es un óptimo. Use el análisis de cresta o curvas de nivel para elegir "
            "condiciones de operación."
        )
    if fuera:
        texto += " Como está fuera de la región probada, confírmelo con un experimento en el secador antes de usarlo."
    else:
        texto += " Está dentro de la región probada; se recomienda un ensayo de confirmación en el secador."
    st.write(texto)

# Malla y predicciones compartidas por las pestañas 5 y 6
X1, X2 = malla(-1.5, 1.5, 60)
Z = predecir_malla(mod2, X1, X2, c_x1, c_x2)
etiquetas = {
    "nombre1": nom1, "unidad1": uni1, "centro1": cen1c, "semi1": sem1c,
    "nombre2": nom2, "unidad2": uni2, "centro2": cen2c, "semi2": sem2c,
    "respuesta": c_y,
}

with tab5:
    st.plotly_chart(superficie_3d(X1, X2, Z, xs, ys, etiquetas), width="stretch")
    st.markdown(
        """
**Cómo leerla:**
- **Cima o fondo:** el punto rojo marca el punto estacionario; si es un máximo, está en lo más alto de la superficie.
- **Pendiente:** donde la superficie es más inclinada, un pequeño cambio en las variables cambia mucho la respuesta; cerca de la cima es casi plana.
- **Forma:** una cúpula indica máximo, un valle mínimo y una "silla de montar" que no hay óptimo único.
"""
    )

with tab6:
    puntos = d3[[c_x1, c_x2]].to_numpy(dtype=float)
    st.plotly_chart(curvas_nivel(X1, X2, Z, xs, puntos, etiquetas), width="stretch")
    st.markdown(
        """
**Cómo leerlas:**
- **Cima:** las curvas se cierran alrededor del punto estacionario (✕ rojo); el valor aumenta hacia el centro si es un máximo.
- **Pendiente:** curvas muy juntas significan cambio rápido de la respuesta; curvas separadas, zona plana.
- **Forma:** óvalos casi circulares indican que ambas variables pesan parecido; óvalos alargados o inclinados indican que una pesa más o que hay interacción.
"""
    )
