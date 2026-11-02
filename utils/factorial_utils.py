"""Funciones para el análisis de diseños factoriales (2 o más factores) en DCA."""

from itertools import product

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.anova import anova_lm

from utils.anova_utils import _q


def construir_formula(resp: str, factores: list[str]) -> str:
    """Arma la fórmula del modelo factorial completo.

    Ejemplo: ``Q("y") ~ C(Q("A")) * C(Q("B"))``. El operador ``*`` incluye los
    efectos principales y todas las interacciones entre los factores.

    Args:
        resp: Nombre de la columna de respuesta.
        factores: Nombres de las columnas de los factores (2 o más).

    Returns:
        Fórmula de statsmodels.

    Raises:
        ValueError: Si hay menos de 2 factores.
    """
    if len(factores) < 2:
        raise ValueError("Un diseño factorial requiere al menos 2 factores.")
    return f"{_q(resp)} ~ " + " * ".join(f"C({_q(f)})" for f in factores)


def validar_factorial(df: pd.DataFrame, factores: list[str], resp: str) -> list[str]:
    """Revisa que los datos sirvan para un factorial balanceado en DCA.

    Args:
        df: Datos del experimento.
        factores: Columnas de los factores.
        resp: Columna de la respuesta.

    Returns:
        Lista de mensajes de error en español (vacía si todo está bien).
    """
    errores: list[str] = []
    if len(factores) < 2:
        errores.append("Seleccione al menos 2 factores.")
    if resp in factores:
        errores.append("La variable respuesta no puede ser también un factor.")
    if errores:
        return errores

    cols = factores + [resp]
    # Datos faltantes: el ANOVA balanceado exige todas las observaciones
    faltantes = int(df[cols].isna().sum().sum())
    if faltantes:
        errores.append(f"Hay {faltantes} dato(s) faltante(s) en las columnas seleccionadas.")
        return errores

    if not pd.api.types.is_numeric_dtype(df[resp]):
        errores.append(f"La respuesta '{resp}' debe ser numérica.")
        return errores

    # Todas las combinaciones de niveles deben estar presentes
    niveles = [sorted(df[f].unique(), key=str) for f in factores]
    esperadas = set(product(*niveles))
    presentes = set(map(tuple, df[factores].itertuples(index=False, name=None)))
    ausentes = esperadas - presentes
    if ausentes:
        ejemplo = ", ".join(" / ".join(map(str, c)) for c in sorted(ausentes, key=str)[:5])
        errores.append(f"Faltan {len(ausentes)} combinación(es) de niveles, por ejemplo: {ejemplo}.")

    # Mismo número de réplicas en cada combinación (diseño balanceado)
    conteos = df.groupby(factores, observed=True)[resp].size()
    if conteos.nunique() > 1:
        errores.append(
            "El diseño no es balanceado: las combinaciones tienen entre "
            f"{conteos.min()} y {conteos.max()} réplicas."
        )
    elif conteos.iloc[0] < 2:
        errores.append("Se necesitan al menos 2 réplicas por combinación para estimar el error.")
    return errores


def ajustar_factorial(df: pd.DataFrame, factores: list[str], resp: str) -> RegressionResultsWrapper:
    """Ajusta el modelo factorial completo con OLS.

    Args:
        df: Datos del experimento.
        factores: Columnas de los factores.
        resp: Columna de la respuesta.

    Returns:
        Modelo ajustado de statsmodels.
    """
    datos = df.copy()
    # Los factores se tratan como categóricos aunque sean números (p. ej. carga 2, 4, 6)
    for f in factores:
        datos[f] = datos[f].astype(str)
    return smf.ols(construir_formula(resp, factores), data=datos).fit()


def _nombre_termino(termino: str, factores: list[str]) -> str:
    """Convierte un término de statsmodels en un nombre legible (A × B).

    Args:
        termino: Término como aparece en anova_lm, p. ej. ``C(Q("A")):C(Q("B"))``.
        factores: Columnas de los factores.

    Returns:
        Nombre en español con "×" entre factores.
    """
    partes = termino.split(":")
    nombres = []
    for p in partes:
        for f in factores:
            if p == f"C({_q(f)})":
                nombres.append(f)
    return " × ".join(nombres)


def anova_factorial(df: pd.DataFrame, factores: list[str], resp: str) -> pd.DataFrame:
    """Tabla ANOVA del factorial (suma de cuadrados tipo II).

    Args:
        df: Datos del experimento.
        factores: Columnas de los factores.
        resp: Columna de la respuesta.

    Returns:
        Tabla con columnas Fuente, gl, SC, CM, F y p-valor; filas: efectos
        principales, interacciones, Error y Total.
    """
    modelo = ajustar_factorial(df, factores, resp)
    # Tabla ANOVA tipo II: SC de cada efecto ajustada por los demás
    raw = anova_lm(modelo, typ=2)

    filas = []
    for termino, r in raw.iterrows():
        if termino == "Residual":
            continue
        # CM = SC / gl: variación promedio atribuible al efecto
        filas.append(
            {
                "Fuente": _nombre_termino(str(termino), factores),
                "gl": r["df"],
                "SC": r["sum_sq"],
                "CM": r["sum_sq"] / r["df"],
                "F": r["F"],
                "p-valor": r["PR(>F)"],
            }
        )
    gl_e = raw.loc["Residual", "df"]
    sc_e = raw.loc["Residual", "sum_sq"]
    filas.append(
        {"Fuente": "Error", "gl": gl_e, "SC": sc_e, "CM": sc_e / gl_e, "F": float("nan"), "p-valor": float("nan")}
    )
    tabla = pd.DataFrame(filas)
    # Total = suma de todos los gl y SC
    total = {
        "Fuente": "Total",
        "gl": tabla["gl"].sum(),
        "SC": tabla["SC"].sum(),
        "CM": float("nan"),
        "F": float("nan"),
        "p-valor": float("nan"),
    }
    return pd.concat([tabla, pd.DataFrame([total])], ignore_index=True)


def medias_celdas(df: pd.DataFrame, factores: list[str], resp: str) -> pd.DataFrame:
    """Calcula n, media y desviación estándar de cada combinación de niveles.

    Args:
        df: Datos del experimento.
        factores: Columnas de los factores.
        resp: Columna de la respuesta.

    Returns:
        Tabla con los factores, n, Media y DE por celda.
    """
    return (
        df.groupby(factores, observed=True)[resp]
        .agg(n="count", Media="mean", DE="std")
        .reset_index()
    )
