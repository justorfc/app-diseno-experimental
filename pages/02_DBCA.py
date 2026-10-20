"""Módulo DBCA: Diseño de Bloques Completos al Azar."""

import streamlit as st

st.set_page_config(page_title="DBCA", page_icon="🌱", layout="wide")

st.title("DBCA – Diseño de Bloques Completos al Azar")
st.write(
    "Diseño en el que las unidades experimentales se agrupan en bloques homogéneos "
    "y cada tratamiento aparece una vez en cada bloque.\n\n"
    "Se usa para separar del error experimental una fuente de variación conocida, "
    "como la pendiente o la fertilidad del lote."
)
st.info("En construcción: Actividad 2")
