"""Modelos de primer orden, prueba de curvatura y trayectoria de máximo ascenso."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import RegressionResultsWrapper

from utils.anova_utils import _q


def ajustar_primer_orden(
    df: pd.DataFrame, y: str, x1: str, x2: str
) -> tuple[RegressionResultsWrapper, pd.DataFrame, float]:
    """Ajusta el modelo de primer orden y = b0 + b1·x1 + b2·x2.

    Args:
        df: Datos con las variables codificadas y la respuesta.
        y: Columna de la respuesta.
        x1: Columna de la variable codificada 1.
        x2: Columna de la variable codificada 2.

    Returns:
        Modelo ajustado, tabla de coeficientes en español y R².
    """
    # Mínimos cuadrados ordinarios: el plano que mejor se acerca a los datos
    modelo = ols(f"{_q(y)} ~ {_q(x1)} + {_q(x2)}", data=df).fit()
    nombres = ["Intercepto (b0)", f"{x1} (b1)", f"{x2} (b2)"]
    tabla = pd.DataFrame(
        {
            "Término": nombres,
            "Coeficiente": modelo.params.to_numpy(),
            "Error estándar": modelo.bse.to_numpy(),
            "t": modelo.tvalues.to_numpy(),
            "p-valor": modelo.pvalues.to_numpy(),
        }
    )
    # R² = proporción de la variación de la respuesta explicada por el modelo
    return modelo, tabla, float(modelo.rsquared)


def prueba_curvatura(
    df: pd.DataFrame, y: str, x1: str, x2: str, alfa: float = 0.05
) -> dict:
    """Prueba si existe curvatura comparando puntos factoriales y centrales.

    Args:
        df: Datos con variables codificadas y respuesta.
        y: Columna de la respuesta.
        x1: Columna de la variable codificada 1.
        x2: Columna de la variable codificada 2.
        alfa: Nivel de significancia.

    Returns:
        Diccionario con nF, nC, yF, yC, sc_curvatura, s2, F, gl, p_valor,
        significativa y conclusion.

    Raises:
        ValueError: Si hay menos de 2 puntos centrales o ningún punto factorial.
    """
    # Puntos centrales: x1 = x2 = 0; el resto son los puntos factoriales
    central = (df[x1].abs() < 1e-9) & (df[x2].abs() < 1e-9)
    yc_obs = df.loc[central, y].to_numpy(dtype=float)
    yf_obs = df.loc[~central, y].to_numpy(dtype=float)
    n_f, n_c = len(yf_obs), len(yc_obs)
    if n_c < 2 or n_f < 1:
        raise ValueError("Se necesitan al menos 2 puntos centrales y 1 factorial.")

    # Medias de los puntos factoriales y de los centrales
    y_f, y_c = float(yf_obs.mean()), float(yc_obs.mean())
    # SC curvatura = nF·nC·(yF - yC)² / (nF + nC)
    sc_curv = n_f * n_c * (y_f - y_c) ** 2 / (n_f + n_c)
    # s² = varianza de los centros: error puro (gl = nC - 1)
    s2 = float(yc_obs.var(ddof=1))
    gl_error = n_c - 1
    # F = SC curvatura / s² con gl (1, nC - 1)
    f = sc_curv / s2 if s2 > 0 else float("inf")
    p = float(stats.f.sf(f, 1, gl_error))
    signif = p < alfa
    if signif:
        concl = (
            f"p = {p:.4f} < α = {alfa:.2f}: hay curvatura significativa. El plano de primer "
            "orden no basta; se requiere un modelo de segundo orden."
        )
    else:
        concl = (
            f"p = {p:.4f} ≥ α = {alfa:.2f}: no hay evidencia de curvatura. El modelo de "
            "primer orden es adecuado y se puede seguir la trayectoria de ascenso."
        )
    return {
        "nF": n_f,
        "nC": n_c,
        "yF": y_f,
        "yC": y_c,
        "sc_curvatura": float(sc_curv),
        "s2": s2,
        "F": float(f),
        "gl": (1, gl_error),
        "p_valor": p,
        "significativa": bool(signif),
        "conclusion": concl,
    }


def trayectoria_ascenso(
    b1: float, b2: float, pasos: int = 5, paso_x1: float = 1.0
) -> pd.DataFrame:
    """Calcula la trayectoria de máximo ascenso desde el centro del diseño.

    La dirección es proporcional a (b1, b2). La variable con mayor |coeficiente|
    es la base y avanza ``paso_x1`` por paso; la otra avanza en proporción
    (Δx2 = (b2/b1)·Δx1, o Δx1 = (b1/b2)·Δx2 si |b2| > |b1|). El sentido se
    ajusta para que la respuesta suba (si el coeficiente base es negativo, la
    variable base disminuye).

    Args:
        b1: Coeficiente de x1.
        b2: Coeficiente de x2.
        pasos: Número de pasos después del centro.
        paso_x1: Tamaño del paso (codificado) en la variable base.

    Returns:
        Tabla con columnas paso, x1, x2 (el paso 0 es el centro).
    """
    if b1 == 0 and b2 == 0:
        raise ValueError("Ambos coeficientes son cero: no hay dirección de ascenso.")
    if abs(b2) > abs(b1):
        # x2 es la variable base
        d2 = float(np.sign(b2)) * paso_x1
        d1 = (b1 / b2) * d2
    else:
        d1 = float(np.sign(b1)) * paso_x1
        d2 = (b2 / b1) * d1
    k = np.arange(pasos + 1)
    return pd.DataFrame({"paso": k, "x1": k * d1, "x2": k * d2})
