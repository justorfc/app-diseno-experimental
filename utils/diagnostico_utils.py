"""Funciones para revisar los supuestos del ANOVA de un DCA.

Supuestos del modelo Y_ij = mu + tau_i + e_ij:
    - Los errores e_ij siguen una distribución normal.
    - Los errores tienen varianza constante entre tratamientos.
    - Los errores son independientes (se garantiza con la aleatorización).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


def residuos_y_ajustados(modelo: RegressionResultsWrapper) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene los residuos y los valores ajustados del modelo.

    Fórmulas:
        ajustado_ij = media del tratamiento i
        residuo_ij  = Y_ij - ajustado_ij

    Args:
        modelo: Modelo devuelto por ajustar_dca.

    Returns:
        Tupla (residuos, ajustados) como arreglos de numpy.
    """
    residuos = np.asarray(modelo.resid, dtype=float)
    ajustados = np.asarray(modelo.fittedvalues, dtype=float)
    return residuos, ajustados


def prueba_shapiro(residuos: np.ndarray, alfa: float = 0.05) -> tuple[float, float, str]:
    """Prueba de normalidad de Shapiro-Wilk sobre los residuos.

    H0: los residuos provienen de una distribución normal.
    Se rechaza H0 si p-valor < alfa.

    Args:
        residuos: Residuos del modelo.
        alfa: Nivel de significancia.

    Returns:
        Tupla (estadístico W, p-valor, conclusión en español).
    """
    # W cercano a 1 indica que los residuos se parecen a una normal
    estadistico, p = stats.shapiro(residuos)
    if p < alfa:
        conclusion = (
            "Se rechaza H0: los residuos no parecen normales. "
            "El supuesto de normalidad no se cumple."
        )
    else:
        conclusion = (
            "No se rechaza H0: no hay evidencia contra la normalidad. "
            "El supuesto de normalidad se cumple."
        )
    return float(estadistico), float(p), conclusion


def prueba_levene(
    df: pd.DataFrame, col_trat: str, col_resp: str, alfa: float = 0.05
) -> tuple[float, float, str]:
    """Prueba de Levene (versión de Brown-Forsythe) de homogeneidad de varianzas.

    H0: todos los tratamientos tienen la misma varianza.
    Con center="median" se compara la dispersión respecto a la mediana de cada
    grupo, lo que la hace robusta cuando los datos no son perfectamente normales.
    Se rechaza H0 si p-valor < alfa.

    Args:
        df: Datos del experimento.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.
        alfa: Nivel de significancia.

    Returns:
        Tupla (estadístico W, p-valor, conclusión en español).
    """
    # Un grupo de observaciones por cada tratamiento
    grupos = [g[col_resp].to_numpy(dtype=float) for _, g in df.groupby(col_trat)]
    estadistico, p = stats.levene(*grupos, center="median")
    if p < alfa:
        conclusion = (
            "Se rechaza H0: las varianzas no son iguales entre tratamientos. "
            "El supuesto de homogeneidad no se cumple."
        )
    else:
        conclusion = (
            "No se rechaza H0: no hay evidencia de varianzas distintas. "
            "El supuesto de homogeneidad se cumple."
        )
    return float(estadistico), float(p), conclusion


def grafico_residuos_vs_ajustados(residuos: np.ndarray, ajustados: np.ndarray) -> go.Figure:
    """Grafica residuos contra valores ajustados.

    Si la varianza es constante, los puntos se reparten al azar alrededor de
    cero, con una dispersión parecida en todos los tratamientos (sin forma de
    embudo ni curvas).

    Args:
        residuos: Residuos del modelo.
        ajustados: Valores ajustados del modelo.

    Returns:
        Figura de plotly.
    """
    fig = go.Figure(
        go.Scatter(x=ajustados, y=residuos, mode="markers", name="Residuos")
    )
    # Línea de referencia en cero: los residuos promedian cero por construcción
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Residuos vs. valores ajustados",
        xaxis_title="Valores ajustados (media del tratamiento)",
        yaxis_title="Residuos",
    )
    return fig


def grafico_qq(residuos: np.ndarray) -> go.Figure:
    """Grafica Q-Q normal de los residuos.

    Compara los residuos ordenados con los cuantiles teóricos de una normal.
    Si los puntos siguen la recta, los residuos son aproximadamente normales.

    Args:
        residuos: Residuos del modelo.

    Returns:
        Figura de plotly.
    """
    # probplot da cuantiles teóricos y la recta de referencia (pendiente, intercepto)
    (teoricos, ordenados), (pendiente, intercepto, _) = stats.probplot(residuos, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=teoricos, y=ordenados, mode="markers", name="Residuos"))
    fig.add_trace(
        go.Scatter(
            x=teoricos,
            y=pendiente * teoricos + intercepto,
            mode="lines",
            name="Recta de referencia",
            line={"dash": "dash", "color": "gray"},
        )
    )
    fig.update_layout(
        title="Gráfico Q-Q normal de los residuos",
        xaxis_title="Cuantiles teóricos de la normal",
        yaxis_title="Residuos ordenados",
    )
    return fig
