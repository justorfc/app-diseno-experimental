"""Malla de puntos y predicciones del modelo de segundo orden."""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper


def malla(x_min: float = -1.5, x_max: float = 1.5, n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Crea una malla cuadrada de puntos en unidades codificadas.

    Args:
        x_min: Valor mínimo de x1 y x2.
        x_max: Valor máximo de x1 y x2.
        n: Número de puntos por eje.

    Returns:
        Matrices X1 y X2 de tamaño (n, n).
    """
    eje = np.linspace(x_min, x_max, n)
    return np.meshgrid(eje, eje)


def predecir_malla(
    modelo: RegressionResultsWrapper, X1: np.ndarray, X2: np.ndarray, x1: str, x2: str
) -> np.ndarray:
    """Calcula ŷ del modelo de segundo orden en cada punto de la malla.

    Args:
        modelo: Modelo devuelto por ajustar_segundo_orden.
        X1: Malla de x1 codificada.
        X2: Malla de x2 codificada.
        x1: Nombre de la columna x1 usada al ajustar el modelo.
        x2: Nombre de la columna x2 usada al ajustar el modelo.

    Returns:
        Matriz Z con la respuesta predicha, del mismo tamaño que X1.
    """
    # El modelo evalúa su propia fórmula (b0 + b1x1 + b2x2 + b11x1² + b22x2² + b12x1x2)
    datos = pd.DataFrame({x1: X1.ravel(), x2: X2.ravel()})
    return np.asarray(modelo.predict(datos)).reshape(X1.shape)
