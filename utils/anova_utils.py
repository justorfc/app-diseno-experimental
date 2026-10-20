"""Funciones para el análisis de un Diseño Completamente al Azar (DCA).

Modelo del DCA con un factor:  Y_ij = mu + tau_i + e_ij
    Y_ij : respuesta de la réplica j del tratamiento i
    mu   : media general
    tau_i: efecto del tratamiento i
    e_ij : error experimental
"""

import io
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import studentized_range
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _q(nombre: str) -> str:
    """Protege un nombre de columna para usarlo en una fórmula de statsmodels.

    Q("nombre") permite usar columnas con espacios, tildes o símbolos
    (por ejemplo "Humedad (%)"). json.dumps escapa comillas internas.

    Args:
        nombre: Nombre de la columna.

    Returns:
        Texto de la forma Q("nombre").
    """
    return f"Q({json.dumps(nombre)})"


def cargar_datos(archivo: Any) -> pd.DataFrame:
    """Lee un archivo CSV (separador "," o ";") o Excel (.xlsx).

    Args:
        archivo: Ruta del archivo o archivo subido (por ejemplo, el objeto de
            st.file_uploader). Debe tener extensión .csv o .xlsx.

    Returns:
        DataFrame con los datos leídos.

    Raises:
        ValueError: Si la extensión del archivo no es compatible.
    """
    nombre = str(getattr(archivo, "name", archivo)).lower()

    # Excel: pandas lee directamente la primera hoja
    if nombre.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(archivo, engine="openpyxl")

    if not nombre.endswith(".csv"):
        raise ValueError("Formato no compatible. Use un archivo .csv o .xlsx.")

    # CSV: se lee el contenido como bytes para poder detectar el separador
    if hasattr(archivo, "read"):
        contenido = archivo.read()
    else:
        with open(archivo, "rb") as f:
            contenido = f.read()
    try:
        texto = contenido.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = contenido.decode("latin-1")

    # El separador es el que más aparece en la primera línea (encabezado)
    primera = texto.splitlines()[0] if texto.strip() else ""
    sep = ";" if primera.count(";") > primera.count(",") else ","
    df = pd.read_csv(io.StringIO(texto), sep=sep)

    # Con ";" es común el decimal con coma (12,4): se convierte a número
    if sep == ";":
        for col in df.select_dtypes(include="object").columns:
            convertida = pd.to_numeric(
                df[col].str.replace(",", ".", regex=False), errors="coerce"
            )
            if convertida.notna().sum() == df[col].notna().sum():
                df[col] = convertida
    return df


def validar_dca(df: pd.DataFrame, col_trat: str, col_resp: str) -> list[str]:
    """Revisa que los datos sean adecuados para un DCA de un factor.

    Args:
        df: Datos del experimento.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        Lista de mensajes de error en español; vacía si todo está bien.
    """
    errores: list[str] = []

    # 1. Columnas: sin ellas no se puede revisar nada más
    for col in (col_trat, col_resp):
        if col not in df.columns:
            errores.append(f"No existe la columna '{col}' en los datos.")
    if errores:
        return errores

    # 2. La respuesta debe ser numérica para poder calcular medias y varianzas
    if not pd.api.types.is_numeric_dtype(df[col_resp]):
        errores.append(f"La respuesta '{col_resp}' debe ser numérica.")

    # 3. Datos faltantes en tratamiento o respuesta
    faltantes = int(df[[col_trat, col_resp]].isna().sum().sum())
    if faltantes > 0:
        errores.append(f"Hay {faltantes} dato(s) faltante(s) en '{col_trat}' o '{col_resp}'.")

    # 4. Mínimo 2 tratamientos para poder compararlos
    completos = df[[col_trat, col_resp]].dropna()
    conteo = completos.groupby(col_trat).size()
    if len(conteo) < 2:
        errores.append("Se necesitan al menos 2 tratamientos.")

    # 5. Mínimo 2 réplicas por tratamiento para estimar el error experimental
    pocos = [str(t) for t, n in conteo.items() if n < 2]
    if pocos:
        errores.append(
            "Cada tratamiento debe tener al menos 2 réplicas. "
            f"Revise: {', '.join(pocos)}."
        )
    return errores


def ajustar_dca(df: pd.DataFrame, col_trat: str, col_resp: str) -> RegressionResultsWrapper:
    """Ajusta el modelo lineal del DCA: respuesta ~ tratamiento.

    C(...) indica a statsmodels que el tratamiento es una variable categórica.

    Args:
        df: Datos del experimento.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        Modelo de mínimos cuadrados ordinarios ya ajustado.
    """
    formula = f"{_q(col_resp)} ~ C({_q(col_trat)})"
    return ols(formula, data=df).fit()


def tabla_anova(modelo: RegressionResultsWrapper) -> pd.DataFrame:
    """Construye la tabla ANOVA del DCA.

    Fórmulas:
        CM = SC / gl
        F  = CM_tratamientos / CM_error
        Total: SC y gl son la suma de tratamientos y error.

    Args:
        modelo: Modelo devuelto por ajustar_dca.

    Returns:
        DataFrame con columnas Fuente, gl, SC, CM, F y p-valor.
    """
    # anova_lm tipo II: SC de tratamientos y del residuo (error)
    anova = anova_lm(modelo, typ=2)
    tabla = pd.DataFrame(
        {
            "Fuente": ["Tratamientos", "Error"],
            "gl": anova["df"].to_numpy(),
            "SC": anova["sum_sq"].to_numpy(),
            "F": [anova["F"].iloc[0], np.nan],
            "p-valor": [anova["PR(>F)"].iloc[0], np.nan],
        }
    )
    # Cuadrado medio = suma de cuadrados / grados de libertad
    tabla["CM"] = tabla["SC"] / tabla["gl"]

    # Fila Total: la variación total es la suma de tratamientos y error
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


def resumen_tratamientos(df: pd.DataFrame, col_trat: str, col_resp: str) -> pd.DataFrame:
    """Calcula estadísticos descriptivos por tratamiento.

    Fórmulas:
        media = total / n
        s = sqrt( sum((y - media)^2) / (n - 1) )

    Args:
        df: Datos del experimento.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        DataFrame con tratamiento, n, total, media y desviación estándar.
    """
    resumen = (
        df.groupby(col_trat)[col_resp]
        .agg(["count", "sum", "mean", "std"])
        .reset_index()
    )
    resumen.columns = [col_trat, "n", "Total", "Media", "Desviación estándar"]
    return resumen


def coeficiente_variacion(
    modelo: RegressionResultsWrapper, df: pd.DataFrame, col_resp: str
) -> float:
    """Calcula el coeficiente de variación (CV) del experimento.

    Fórmula:  CV = sqrt(CME) / media general * 100
    Indica qué tan grande es el error experimental respecto a la media (%).

    Args:
        modelo: Modelo devuelto por ajustar_dca.
        df: Datos del experimento.
        col_resp: Nombre de la columna de la variable respuesta.

    Returns:
        CV en porcentaje.
    """
    # CME = cuadrado medio del error = SC_error / gl_error
    cme = modelo.mse_resid
    media_general = df[col_resp].mean()
    return float(np.sqrt(cme) / media_general * 100)


def tukey(
    df: pd.DataFrame, col_trat: str, col_resp: str, alfa: float = 0.05
) -> tuple[pd.DataFrame, float]:
    """Compara todos los pares de tratamientos con la prueba de Tukey.

    Fórmula de la DMS (diferencia mínima significativa):
        DMS = q(alfa; t; glE) * sqrt(CME / r)
    donde q es el valor del rango estudentizado, t el número de tratamientos,
    glE los grados de libertad del error y r las réplicas por tratamiento.
    Dos medias difieren si su diferencia absoluta supera la DMS.

    Args:
        df: Datos del experimento.
        col_trat: Nombre de la columna de tratamientos.
        col_resp: Nombre de la columna de la variable respuesta.
        alfa: Nivel de significancia.

    Returns:
        Tupla con el DataFrame de comparaciones por pares y la DMS.
    """
    # Tukey por pares: ajusta el valor p para todas las comparaciones a la vez
    resultado = pairwise_tukeyhsd(
        endog=df[col_resp], groups=df[col_trat], alpha=alfa
    )
    datos = resultado.summary().data
    comparaciones = pd.DataFrame(datos[1:], columns=datos[0])
    comparaciones.columns = [
        "Tratamiento 1",
        "Tratamiento 2",
        "Diferencia de medias",
        "p ajustado",
        "Límite inferior",
        "Límite superior",
        "Rechaza H0",
    ]

    # CME y gl del error vienen del modelo ANOVA
    modelo = ajustar_dca(df, col_trat, col_resp)
    cme = modelo.mse_resid
    gl_error = modelo.df_resid
    replicas = df.groupby(col_trat)[col_resp].count()
    t = len(replicas)

    if replicas.nunique() > 1:
        # Con réplicas distintas se usa la media armónica de r
        r = float(t / (1.0 / replicas).sum())
        warnings.warn(
            "El diseño no está balanceado (réplicas distintas): "
            "la DMS es aproximada, calculada con la media armónica de r.",
            stacklevel=2,
        )
    else:
        r = float(replicas.iloc[0])

    # q del rango estudentizado: cuantil 1 - alfa con t medias y glE
    q = studentized_range.ppf(1 - alfa, t, gl_error)
    dms = float(q * np.sqrt(cme / r))
    return comparaciones, dms


def interpretar_dca(tabla: pd.DataFrame, alfa: float = 0.05) -> str:
    """Redacta una interpretación sencilla del ANOVA.

    H0: todas las medias de los tratamientos son iguales.
    Se rechaza H0 si p-valor < alfa.

    Args:
        tabla: Tabla ANOVA devuelta por tabla_anova.
        alfa: Nivel de significancia.

    Returns:
        Texto con la decisión estadística y su significado práctico.
    """
    fila = tabla.loc[tabla["Fuente"] == "Tratamientos"].iloc[0]
    p = float(fila["p-valor"])
    f = float(fila["F"])
    # Un p-valor tan pequeño no debe mostrarse como 0.0000
    p_txt = "< 0.0001" if p < 0.0001 else f"= {p:.4f}"

    if p < alfa:
        return (
            f"Con F = {f:.3f} y p-valor {p_txt} (< {alfa}), se rechaza H0. "
            "Hay evidencia de que al menos un tratamiento tiene una media "
            "diferente. Use la prueba de Tukey para saber cuáles difieren."
        )
    return (
        f"Con F = {f:.3f} y p-valor {p_txt} (≥ {alfa}), no se rechaza H0. "
        "No hay evidencia suficiente de diferencias entre los tratamientos: "
        "las diferencias observadas pueden deberse al azar."
    )
