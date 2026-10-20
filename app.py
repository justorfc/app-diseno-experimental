"""Portada de la App de Diseño Experimental."""

import pandas as pd
import streamlit as st

AUTOR = "Justo Fuentes"
CURSO = "Diseño Experimental"
SEMESTRE = "2026-2"

st.set_page_config(
    page_title="App de Diseño Experimental",
    page_icon="🌱",
    layout="wide",
)

# Barra lateral con los datos del autor y del curso
with st.sidebar:
    st.header("Información")
    st.write(f"**Autor:** {AUTOR}")
    st.write(f"**Curso:** {CURSO}")
    st.write(f"**Semestre:** {SEMESTRE}")

st.title("🌱 App de Diseño Experimental")
st.subheader("Ingeniería Agrícola – Universidad de Sucre")

st.write(
    "El diseño experimental permite planificar ensayos en campo, invernadero o "
    "laboratorio para comparar tratamientos (variedades, fertilizantes, riegos, "
    "métodos de secado) de forma confiable. Al asignar los tratamientos con "
    "aleatorización, repeticiones y control de la variabilidad del terreno, se "
    "puede separar el efecto real del tratamiento del azar y tomar decisiones "
    "técnicas respaldadas por los datos."
)

st.header("Módulos disponibles")

modulos = pd.DataFrame(
    [
        {
            "Módulo": "DCA – Diseño Completamente al Azar",
            "Cuándo se usa": "Unidades experimentales homogéneas, sin gradientes conocidos.",
            "Ejemplo agrícola": "Comparar la humedad final de yuca con cuatro tipos de secador.",
            "Estado": "Disponible",
        },
        {
            "Módulo": "DBCA – Diseño de Bloques Completos al Azar",
            "Cuándo se usa": "Existe una fuente de variación conocida que se puede agrupar en bloques.",
            "Ejemplo agrícola": "Evaluar variedades de maíz en lotes con distinta pendiente.",
            "Estado": "En construcción",
        },
        {
            "Módulo": "DCL – Diseño en Cuadrado Latino",
            "Cuándo se usa": "Hay dos fuentes de variación cruzadas (filas y columnas).",
            "Ejemplo agrícola": "Probar dosis de abono en un lote con gradientes de fertilidad y de riego.",
            "Estado": "En construcción",
        },
        {
            "Módulo": "DPD – Diseño de Parcelas Divididas",
            "Cuándo se usa": "Un factor requiere parcelas grandes y otro puede aplicarse en subparcelas.",
            "Ejemplo agrícola": "Sistemas de riego (parcela principal) y variedades de arroz (subparcela).",
            "Estado": "En construcción",
        },
        {
            "Módulo": "Factoriales",
            "Cuándo se usa": "Se estudian dos o más factores y su interacción al mismo tiempo.",
            "Ejemplo agrícola": "Combinar niveles de nitrógeno y de fósforo en el rendimiento de ají.",
            "Estado": "En construcción",
        },
        {
            "Módulo": "RSM – Metodología de Superficie de Respuesta",
            "Cuándo se usa": "Se busca el nivel óptimo de uno o más factores cuantitativos.",
            "Ejemplo agrícola": "Hallar la temperatura y el tiempo de secado que minimizan la humedad.",
            "Estado": "En construcción",
        },
    ]
)
st.dataframe(modulos, hide_index=True, width="stretch")

st.header("¿Cómo usar la app?")
st.markdown(
    """
1. **Elige el módulo** en el menú lateral según el diseño de tu experimento.
2. **Carga tus datos** (archivo CSV o Excel) o usa los datos de ejemplo.
3. **Revisa los resultados**: tabla ANOVA, pruebas de supuestos, comparaciones y gráficos.
4. **Verifica a mano**: repite el cálculo con tu procedimiento manual y compara.
"""
)

st.warning(
    "Esta app apoya el aprendizaje. El estudiante interpreta los resultados, "
    "verifica supuestos y compara con su cálculo manual."
)
