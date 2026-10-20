"""Módulo DPD: Diseño de Parcelas Divididas."""

import streamlit as st

st.set_page_config(page_title="DPD", page_icon="🌱", layout="wide")

st.title("DPD – Diseño de Parcelas Divididas")
st.write(
    "Diseño con dos factores en el que uno se asigna a parcelas principales y "
    "el otro a subparcelas dentro de ellas.\n\n"
    "Se usa cuando un factor, como el sistema de riego, requiere parcelas grandes "
    "y el otro, como la variedad, puede aplicarse en áreas pequeñas."
)
st.info("En construcción: Actividad 2")
