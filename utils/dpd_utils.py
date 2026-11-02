"""Funciones para el análisis de Parcelas Divididas en bloques (DPD).

Modelo:  Y_ijk = mu + bloque_i + A_j + Error(a)_ij + B_k + AB_jk + Error(b)_ijk
    A: factor de parcela principal (a niveles); B: factor de subparcela (b niveles)
    Error(a): error de la parcela principal (bloque × A)
    Error(b): error de la subparcela (el residual del modelo completo)
"""

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.anova import anova_lm

SEPARADOR = "── Subparcela ──"


def validar_dpd(
    df: pd.DataFrame, bloque: str, factor_a: str, factor_b: str, resp: str
) -> list[str]:
    """Valida que el diseño esté completo y balanceado.

    Cada combinación bloque × A × B debe aparecer exactamente una vez.

    Args:
        df: Datos del experimento.
        bloque: Columna de bloques.
        factor_a: Columna del factor de parcela principal.
        factor_b: Columna del factor de subparcela.
        resp: Columna de la respuesta.

    Returns:
        Lista de mensajes de error en español (vacía si el diseño es válido).
    """
    cols = [bloque, factor_a, factor_b, resp]
    if len(set(cols)) < 4:
        return ["Bloque, factor A, factor B y respuesta deben ser cuatro columnas distintas."]
    errores: list[str] = []
    if not pd.api.types.is_numeric_dtype(df[resp]):
        errores.append(f"La respuesta «{resp}» debe ser numérica.")
    if df[cols].isna().any().any():
        errores.append("Hay valores faltantes en las columnas elegidas.")
    if errores:
        return errores
    if min(df[c].nunique() for c in (bloque, factor_a, factor_b)) < 2:
        return ["Se necesitan al menos 2 niveles en bloque, factor A y factor B."]

    # Diseño completo: debe existir cada combinación bloque × A × B, exactamente una vez
    r, a, b = (df[c].nunique() for c in (bloque, factor_a, factor_b))
    conteo = df.groupby([bloque, factor_a, factor_b]).size()
    if len(conteo) != r * a * b or (conteo != 1).any():
        faltan = r * a * b - len(conteo)
        repetidas = conteo[conteo > 1]
        detalle = []
        if faltan > 0:
            detalle.append(f"faltan {faltan} combinaciones")
        if not repetidas.empty:
            detalle.append(f"{len(repetidas)} combinaciones están repetidas")
        errores.append(
            "El diseño debe ser completo y balanceado: cada combinación "
            f"bloque × A × B debe aparecer exactamente una vez ({'; '.join(detalle)})."
        )
    return errores


def anova_parcelas_divididas(
    df: pd.DataFrame, bloque: str, factor_a: str, factor_b: str, resp: str
) -> pd.DataFrame:
    """Calcula la tabla ANOVA de parcelas divididas en bloques con fórmulas clásicas.

    Args:
        df: Datos del experimento (diseño completo y balanceado).
        bloque: Columna de bloques (r niveles).
        factor_a: Columna del factor de parcela principal (a niveles).
        factor_b: Columna del factor de subparcela (b niveles).
        resp: Columna de la respuesta.

    Returns:
        DataFrame con Fuente, gl, SC, CM, F y p-valor. Entre la parcela principal
        y la subparcela hay una fila separadora (valores vacíos).

    Raises:
        AssertionError: Si las SC no coinciden con el ajuste de statsmodels.
    """
    r, a, b = (df[c].nunique() for c in (bloque, factor_a, factor_b))
    y = df[resp]

    # Factor de corrección: FC = Y..² / (r·a·b)
    fc = y.sum() ** 2 / (r * a * b)
    # SC Total = Σ Y² - FC
    sc_total = (y**2).sum() - fc
    # SC Bloques = Σ Y_i..² / (a·b) - FC
    sc_bloques = (df.groupby(bloque)[resp].sum() ** 2).sum() / (a * b) - fc
    # SC A = Σ Y_.j.² / (r·b) - FC
    sc_a = (df.groupby(factor_a)[resp].sum() ** 2).sum() / (r * b) - fc
    # SC Parcelas principales = Σ Y(bloque, A)² / b - FC
    sc_pp = (df.groupby([bloque, factor_a])[resp].sum() ** 2).sum() / b - fc
    # SC Error(a) = SC PP - SC Bloques - SC A (interacción bloque × A)
    sc_ea = sc_pp - sc_bloques - sc_a
    # SC B = Σ Y_..k² / (r·a) - FC
    sc_b = (df.groupby(factor_b)[resp].sum() ** 2).sum() / (r * a) - fc
    # SC AB = Σ Y(A, B)² / r - FC - SC A - SC B
    sc_ab = (df.groupby([factor_a, factor_b])[resp].sum() ** 2).sum() / r - fc - sc_a - sc_b
    # SC Error(b) = SC Total - SC PP - SC B - SC AB
    sc_eb = sc_total - sc_pp - sc_b - sc_ab

    gl = {
        "bloque": r - 1,
        "a": a - 1,
        "ea": (r - 1) * (a - 1),
        "b": b - 1,
        "ab": (a - 1) * (b - 1),
        "eb": a * (r - 1) * (b - 1),
    }
    # CM = SC / gl
    cm_ea = sc_ea / gl["ea"]
    cm_eb = sc_eb / gl["eb"]

    def fila(nombre: str, sc: float, g: int, cm_error: float | None) -> dict:
        cm = sc / g
        if cm_error is None:
            return {"Fuente": nombre, "gl": g, "SC": sc, "CM": cm, "F": np.nan, "p-valor": np.nan}
        f_val = cm / cm_error
        # p-valor = P(F > F calculada)
        return {
            "Fuente": nombre, "gl": g, "SC": sc, "CM": cm,
            "F": f_val, "p-valor": float(f_dist.sf(f_val, g, gl["ea"] if cm_error == cm_ea else gl["eb"])),
        }

    # Bloques y A se prueban con Error(a); B y A×B con Error(b)
    filas = [
        fila("Bloques", sc_bloques, gl["bloque"], cm_ea),
        fila("A", sc_a, gl["a"], cm_ea),
        fila("Error (a)", sc_ea, gl["ea"], None),
        {"Fuente": SEPARADOR, "gl": np.nan, "SC": np.nan, "CM": np.nan, "F": np.nan, "p-valor": np.nan},
        fila("B", sc_b, gl["b"], cm_eb),
        fila("A×B", sc_ab, gl["ab"], cm_eb),
        fila("Error (b)", sc_eb, gl["eb"], None),
        {"Fuente": "Total", "gl": r * a * b - 1, "SC": sc_total, "CM": np.nan, "F": np.nan, "p-valor": np.nan},
    ]
    tabla = pd.DataFrame(filas)

    _comprobar_con_statsmodels(df, bloque, factor_a, factor_b, resp, sc_ea, sc_eb)
    return tabla


def ajustar_modelo_completo(
    df: pd.DataFrame, bloque: str, factor_a: str, factor_b: str, resp: str
) -> RegressionResultsWrapper:
    """Ajusta el modelo completo con bloque, A, bloque:A, B y A:B.

    Su residual es el Error(b); sus residuos sirven para los supuestos.

    Args:
        df: Datos del experimento.
        bloque: Columna de bloques.
        factor_a: Columna del factor de parcela principal.
        factor_b: Columna del factor de subparcela.
        resp: Columna de la respuesta.

    Returns:
        Modelo ajustado de statsmodels.
    """
    d = df.rename(columns={bloque: "bloque", factor_a: "A", factor_b: "B", resp: "Y"})
    formula = "Y ~ C(bloque) + C(A) + C(bloque):C(A) + C(B) + C(A):C(B)"
    return ols(formula, data=d[["bloque", "A", "B", "Y"]]).fit()


def _comprobar_con_statsmodels(
    df: pd.DataFrame, bloque: str, factor_a: str, factor_b: str, resp: str,
    sc_ea: float, sc_eb: float,
) -> None:
    """Verifica que Error(a) y Error(b) coincidan con anova_lm(typ=1)."""
    modelo = ajustar_modelo_completo(df, bloque, factor_a, factor_b, resp)
    sm = anova_lm(modelo, typ=1)
    assert abs(sm.loc["C(bloque):C(A)", "sum_sq"] - sc_ea) < 1e-6, "SC Error(a) no coincide"
    assert abs(sm.loc["Residual", "sum_sq"] - sc_eb) < 1e-6, "SC Error(b) no coincide"


def f_criticas(tabla: pd.DataFrame, alfa: float = 0.05) -> dict[str, float]:
    """Calcula las F críticas F(1 - alfa; gl_efecto; gl_error) de A, B y A×B.

    Args:
        tabla: Tabla de anova_parcelas_divididas.
        alfa: Nivel de significancia.

    Returns:
        Diccionario {fuente: F crítica}; A usa Error(a), B y A×B usan Error(b).
    """
    t = tabla.set_index("Fuente")
    gl_ea, gl_eb = t.loc["Error (a)", "gl"], t.loc["Error (b)", "gl"]
    return {
        "A": float(f_dist.ppf(1 - alfa, t.loc["A", "gl"], gl_ea)),
        "B": float(f_dist.ppf(1 - alfa, t.loc["B", "gl"], gl_eb)),
        "A×B": float(f_dist.ppf(1 - alfa, t.loc["A×B", "gl"], gl_eb)),
    }


def coeficientes_variacion(tabla: pd.DataFrame, media_general: float) -> tuple[float, float]:
    """Calcula CV(a) y CV(b): CV = sqrt(CM error) / media general × 100.

    Args:
        tabla: Tabla de anova_parcelas_divididas.
        media_general: Media de todas las observaciones.

    Returns:
        Tupla (CV(a), CV(b)) en porcentaje.
    """
    t = tabla.set_index("Fuente")
    return (
        float(np.sqrt(t.loc["Error (a)", "CM"]) / media_general * 100),
        float(np.sqrt(t.loc["Error (b)", "CM"]) / media_general * 100),
    )


def f_con_error_combinado(tabla: pd.DataFrame, efecto: str = "A") -> tuple[float, float, float]:
    """Calcula la F de un efecto usando un único error combinado (incorrecto).

    Error combinado: CM = (SC Error(a) + SC Error(b)) / (gl Error(a) + gl Error(b)).

    Args:
        tabla: Tabla de anova_parcelas_divididas.
        efecto: Fuente a probar (por defecto "A").

    Returns:
        Tupla (F, p-valor, gl del error combinado).
    """
    t = tabla.set_index("Fuente")
    gl_c = t.loc["Error (a)", "gl"] + t.loc["Error (b)", "gl"]
    cm_c = (t.loc["Error (a)", "SC"] + t.loc["Error (b)", "SC"]) / gl_c
    f_val = float(t.loc[efecto, "CM"] / cm_c)
    return f_val, float(f_dist.sf(f_val, t.loc[efecto, "gl"], gl_c)), float(gl_c)
