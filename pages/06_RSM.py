"""Módulo RSM: Metodología de Superficie de Respuesta."""

import streamlit as st

st.set_page_config(page_title="RSM", page_icon="🌱", layout="wide")

st.title("RSM – Metodología de Superficie de Respuesta")
st.write(
    "Conjunto de técnicas que ajustan un modelo polinomial para describir cómo "
    "responde una variable a factores cuantitativos.\n\n"
    "Se usa para encontrar las condiciones óptimas, por ejemplo la temperatura "
    "y el tiempo de secado que minimizan la humedad."
)
st.info("En construcción: Actividad 3")
