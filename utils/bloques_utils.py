"""Funciones para el análisis de un Diseño de Bloques Completos al Azar (DBCA).

Modelo del DBCA:  Y_ij = mu + tau_i + beta_j + e_ij
    tau_i : efecto del tratamiento i
    beta_j: efecto del bloque j
    e_ij  : error experimental
"""

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.anova import anova_lm

from utils.anova_utils import _q


def validar_dbca(
    df: pd.DataFrame, col_bloque: str, col_trat: str, col_resp: str
) -> list[str]:
    """Revisa que los datos cumplan la estructura de un DBCA.

    Args:
        df: Datos del experimento.
        col_bloque: Nombre de la columna de bloques.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        Lista de mensajes de error en español (vacía si los datos son válidos).
    """
    errores: list[str] = []
    if len({col_bloque, col_trat, col_resp}) < 3:
        return ["Bloque, tratamiento y respuesta deben ser tres columnas distintas."]

    if not pd.api.types.is_numeric_dtype(df[col_resp]):
        errores.append(f"La respuesta «{col_resp}» debe ser numérica.")
    if df[[col_bloque, col_trat, col_resp]].isna().any().any():
        errores.append("Hay valores faltantes en las columnas elegidas.")
    if errores:
        return errores

    if df[col_bloque].nunique() < 2 or df[col_trat].nunique() < 2:
        return ["Se necesitan al menos 2 bloques y 2 tratamientos."]

    # Cada combinación bloque × tratamiento debe aparecer exactamente una vez
    conteo = pd.crosstab(df[col_bloque], df[col_trat])
    if (conteo != 1).any().any():
        malas = conteo.stack()
        malas = malas[malas != 1]
        detalle = ", ".join(f"({b}, {t}): {n}" for (b, t), n in malas.head(5).items())
        errores.append(
            "Cada tratamiento debe aparecer exactamente una vez en cada bloque. "
            f"Combinaciones (bloque, tratamiento) con conteo distinto de 1: {detalle}."
        )
    return errores


def ajustar_dbca(
    df: pd.DataFrame, col_bloque: str, col_trat: str, col_resp: str
) -> RegressionResultsWrapper:
    """Ajusta el modelo del DBCA: respuesta ~ tratamiento + bloque.

    Args:
        df: Datos del experimento.
        col_bloque: Nombre de la columna de bloques.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        Modelo ajustado de statsmodels.
    """
    formula = f"{_q(col_resp)} ~ C({_q(col_trat)}) + C({_q(col_bloque)})"
    return ols(formula, data=df).fit()


def tabla_anova_dbca(modelo: RegressionResultsWrapper) -> pd.DataFrame:
    """Construye la tabla ANOVA del DBCA (Tratamientos, Bloques, Error, Total).

    Fórmulas:
        CM = SC / gl
        F  = CM_fuente / CM_error

    Args:
        modelo: Modelo devuelto por ajustar_dbca.

    Returns:
        DataFrame con columnas Fuente, gl, SC, CM, F y p-valor.
    """
    # Tipo II: en un diseño balanceado coincide con el tipo I clásico
    anova = anova_lm(modelo, typ=2)
    tabla = pd.DataFrame(
        {
            "Fuente": ["Tratamientos", "Bloques", "Error"],
            "gl": anova["df"].to_numpy(),
            "SC": anova["sum_sq"].to_numpy(),
            "F": [anova["F"].iloc[0], anova["F"].iloc[1], np.nan],
            "p-valor": [anova["PR(>F)"].iloc[0], anova["PR(>F)"].iloc[1], np.nan],
        }
    )
    # Cuadrado medio = suma de cuadrados / grados de libertad
    tabla["CM"] = tabla["SC"] / tabla["gl"]
    # Total: SC y gl son la suma de todas las fuentes
    total = pd.DataFrame(
        {
            "Fuente": ["Total"],
            "gl": [tabla["gl"].sum()],
            "SC": [tabla["SC"].sum()],
            "CM": [np.nan],
            "F": [np.nan],
            "p-valor": [np.nan],
        }
    )
    tabla = pd.concat([tabla, total], ignore_index=True)
    return tabla[["Fuente", "gl", "SC", "CM", "F", "p-valor"]]


def f_critica_dbca(tabla: pd.DataFrame, alfa: float = 0.05) -> float:
    """Calcula la F crítica para tratamientos: F(1 - alfa; gl_trat; gl_error).

    Args:
        tabla: Tabla ANOVA devuelta por tabla_anova_dbca.
        alfa: Nivel de significancia.

    Returns:
        Valor crítico de la distribución F.
    """
    gl_trat = tabla.loc[tabla["Fuente"] == "Tratamientos", "gl"].iloc[0]
    gl_error = tabla.loc[tabla["Fuente"] == "Error", "gl"].iloc[0]
    return float(f_dist.ppf(1 - alfa, gl_trat, gl_error))


def cv_dbca(tabla: pd.DataFrame, df: pd.DataFrame, col_resp: str) -> float:
    """Calcula el coeficiente de variación: CV = sqrt(CME) / media general * 100.

    Args:
        tabla: Tabla ANOVA devuelta por tabla_anova_dbca.
        df: Datos del experimento.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        CV en porcentaje.
    """
    cme = tabla.loc[tabla["Fuente"] == "Error", "CM"].iloc[0]
    return float(np.sqrt(cme) / df[col_resp].mean() * 100)


def eficiencia_relativa(cm_bloques: float, cm_error: float, t: int, b: int) -> float:
    """Calcula la eficiencia relativa del DBCA frente a un DCA.

    Fórmula:  ER = ((b - 1) * CMB + b * (t - 1) * CME) / ((b * t - 1) * CME)
    Estima cuánto más preciso fue bloquear que no hacerlo.

    Args:
        cm_bloques: Cuadrado medio de bloques (CMB).
        cm_error: Cuadrado medio del error (CME).
        t: Número de tratamientos.
        b: Número de bloques.

    Returns:
        Eficiencia relativa (ER).
    """
    return float(((b - 1) * cm_bloques + b * (t - 1) * cm_error) / ((b * t - 1) * cm_error))


def interpretar_eficiencia(er: float, b: int) -> str:
    """Interpreta la eficiencia relativa en términos de réplicas de un DCA.

    Un DCA necesitaría ER * b réplicas por tratamiento para igualar la
    precisión que el DBCA logró con b bloques.

    Args:
        er: Eficiencia relativa.
        b: Número de bloques (réplicas por tratamiento en el DBCA).

    Returns:
        Texto de interpretación en español.
    """
    # Réplicas por tratamiento que requeriría un DCA: r_DCA = ER * b
    r_dca = er * b
    if er > 1:
        return (
            f"ER = {er:.2f}: el bloqueo fue eficiente. Un DCA necesitaría unas "
            f"{r_dca:.1f} réplicas por tratamiento (en lugar de {b}) para igualar la "
            f"precisión de este DBCA, es decir, {(er - 1) * 100:.0f} % más de unidades "
            "experimentales."
        )
    return (
        f"ER = {er:.2f}: el bloqueo no mejoró la precisión; un DCA con {b} réplicas "
        "habría sido igual o más preciso."
    )


# ---------------------------------------------------------------------------
# Cuadro Latino (DCL):  Y_ijk = mu + fila_i + columna_j + tau_k + e_ijk
# ---------------------------------------------------------------------------


def validar_cuadro_latino(
    df: pd.DataFrame, col_fila: str, col_col: str, col_trat: str, col_resp: str
) -> list[str]:
    """Revisa que los datos formen un Cuadro Latino válido.

    Reglas: filas, columnas y tratamientos con el mismo número t de niveles;
    t² observaciones; cada tratamiento una vez por fila y una vez por columna.

    Args:
        df: Datos del experimento.
        col_fila: Nombre de la columna de filas.
        col_col: Nombre de la columna de columnas.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        Lista de mensajes de error en español (vacía si el cuadro es válido).
    """
    nombres = [col_fila, col_col, col_trat, col_resp]
    if len(set(nombres)) < 4:
        return ["Fila, columna, tratamiento y respuesta deben ser cuatro columnas distintas."]

    errores: list[str] = []
    if not pd.api.types.is_numeric_dtype(df[col_resp]):
        errores.append(f"La respuesta «{col_resp}» debe ser numérica.")
    if df[nombres].isna().any().any():
        errores.append("Hay valores faltantes en las columnas elegidas.")
    if errores:
        return errores

    n_fil, n_col, n_trat = (df[c].nunique() for c in (col_fila, col_col, col_trat))
    if not n_fil == n_col == n_trat:
        return [
            "Regla 1: filas, columnas y tratamientos deben tener el mismo número de "
            f"niveles t. Se encontró {n_fil} filas, {n_col} columnas y {n_trat} tratamientos."
        ]
    t = n_fil
    if t < 3:
        return ["El Cuadro Latino necesita t ≥ 3 (con t = 2 el error tendría 0 gl)."]
    if len(df) != t**2:
        errores.append(
            f"Regla 2: se esperaban t² = {t**2} observaciones y hay {len(df)}."
        )

    # Cada celda (fila, columna) debe tener exactamente una observación
    conteo_celdas = df.groupby([col_fila, col_col]).size()
    if len(conteo_celdas) != t**2 or (conteo_celdas != 1).any():
        errores.append(
            "Regla 2: cada combinación (fila, columna) debe tener exactamente una observación."
        )

    # Cada tratamiento exactamente una vez en cada fila y en cada columna
    for col_pos, etiqueta in ((col_fila, "fila"), (col_col, "columna")):
        tabla = pd.crosstab(df[col_pos], df[col_trat])
        for nivel, fila in tabla.iterrows():
            rotos = fila[fila != 1]
            if not rotos.empty:
                detalle = ", ".join(f"{k} aparece {v} veces" for k, v in rotos.items())
                errores.append(
                    f"Regla 3: en la {etiqueta} «{nivel}» cada tratamiento debe aparecer "
                    f"exactamente una vez ({detalle})."
                )
    return errores


def ajustar_cuadro_latino(
    df: pd.DataFrame, col_fila: str, col_col: str, col_trat: str, col_resp: str
) -> RegressionResultsWrapper:
    """Ajusta el modelo del Cuadro Latino: respuesta ~ fila + columna + tratamiento.

    Args:
        df: Datos del experimento.
        col_fila: Nombre de la columna de filas.
        col_col: Nombre de la columna de columnas.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        Modelo ajustado de statsmodels.
    """
    formula = (
        f"{_q(col_resp)} ~ C({_q(col_fila)}) + C({_q(col_col)}) + C({_q(col_trat)})"
    )
    return ols(formula, data=df).fit()


def tabla_anova_dcl(modelo: RegressionResultsWrapper, t: int) -> pd.DataFrame:
    """Construye la tabla ANOVA del Cuadro Latino.

    Fórmulas:
        CM = SC / gl;  F = CM_fuente / CM_error;  gl Error = (t - 1)(t - 2)

    Args:
        modelo: Modelo devuelto por ajustar_cuadro_latino.
        t: Número de tratamientos (= filas = columnas).

    Returns:
        DataFrame con columnas Fuente, gl, SC, CM, F y p-valor.

    Raises:
        ValueError: Si los gl del error no son (t - 1)(t - 2).
    """
    anova = anova_lm(modelo, typ=2)
    # El orden de la fórmula es fila, columna, tratamiento, residuo
    tabla = pd.DataFrame(
        {
            "Fuente": ["Filas", "Columnas", "Tratamientos", "Error"],
            "gl": anova["df"].to_numpy(),
            "SC": anova["sum_sq"].to_numpy(),
            "F": list(anova["F"].iloc[:3]) + [np.nan],
            "p-valor": list(anova["PR(>F)"].iloc[:3]) + [np.nan],
        }
    )
    # Comprobación: gl del error = (t-1)(t-2)
    gl_error = int(tabla.loc[tabla["Fuente"] == "Error", "gl"].iloc[0])
    if gl_error != (t - 1) * (t - 2):
        raise ValueError(
            f"gl Error = {gl_error}, pero (t-1)(t-2) = {(t - 1) * (t - 2)}."
        )
    tabla["CM"] = tabla["SC"] / tabla["gl"]
    total = pd.DataFrame(
        {
            "Fuente": ["Total"],
            "gl": [tabla["gl"].sum()],
            "SC": [tabla["SC"].sum()],
            "CM": [np.nan],
            "F": [np.nan],
            "p-valor": [np.nan],
        }
    )
    tabla = pd.concat([tabla, total], ignore_index=True)
    return tabla[["Fuente", "gl", "SC", "CM", "F", "p-valor"]]
