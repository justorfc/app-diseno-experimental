"""Prueba rápida: compara las tablas ANOVA de la app con tus cálculos a mano.

Uso (desde la raíz del proyecto):
    python verificacion/prueba_rapida.py

Rellena MIS_VALORES_A_MANO con los valores de tu hoja; los que queden en None
se omiten. Se compara cada valor; hay coincidencia si la diferencia es < 0.01.

Archivos que lee desde data/: dbca, dcl y dpd (obligatorios) y, de forma opcional,
factorial_mi_variante.csv (carga, pretratamiento, humedad), rsm1_mi_variante.csv y
ccd_mi_variante.csv (x1, x2, eficiencia). Si falta uno opcional, se omite con un aviso.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from utils.bloques_utils import (  # noqa: E402
    ajustar_cuadro_latino,
    ajustar_dbca,
    tabla_anova_dbca,
    tabla_anova_dcl,
)
from utils.dpd_utils import anova_parcelas_divididas  # noqa: E402
from utils.factorial_utils import anova_factorial  # noqa: E402
from utils.rsm_models import (  # noqa: E402
    ajustar_primer_orden,
    ajustar_segundo_orden,
    prueba_curvatura,
)
from utils.rsm_stationary import (  # noqa: E402
    clasificar,
    matriz_B,
    punto_estacionario,
    respuesta_en_punto,
)

TOLERANCIA = 0.01

# Copia aquí los valores de tu hoja (SC y F de cada fuente; deja None lo que no calcules)
MIS_VALORES_A_MANO: dict[str, dict[str, dict[str, float | None]]] = {
    "DBCA": {
        "Tratamientos": {"SC": None, "F": None},
        "Bloques": {"SC": None, "F": None},
        "Error": {"SC": None},
        "Total": {"SC": None},
    },
    "DCL": {
        "Filas": {"SC": None, "F": None},
        "Columnas": {"SC": None, "F": None},
        "Tratamientos": {"SC": None, "F": None},
        "Error": {"SC": None},
        "Total": {"SC": None},
    },
    "DPD": {
        "Bloques": {"SC": None, "F": None},
        "A": {"SC": None, "F": None},
        "Error (a)": {"SC": None},
        "B": {"SC": None, "F": None},
        "A×B": {"SC": None, "F": None},
        "Error (b)": {"SC": None},
        "Total": {"SC": None},
    },
    # Factorial: A = primer factor (carga), B = segundo factor (pretratamiento)
    "Factorial": {
        "A": {"SC": None, "F": None},
        "B": {"SC": None, "F": None},
        "A×B": {"SC": None, "F": None},
        "Error": {"SC": None},
        "Total": {"SC": None},
    },
    # RSM de primer orden: b12 se estima con los puntos factoriales
    "RSM primer orden": {
        "b0": {"valor": None},
        "b1": {"valor": None},
        "b2": {"valor": None},
        "b12": {"valor": None},
        "F curvatura": {"valor": None},
    },
    # CCD: xs en unidades codificadas; lambda1 <= lambda2 (valores propios de B)
    "CCD": {
        "xs1": {"valor": None},
        "xs2": {"valor": None},
        "ys": {"valor": None},
        "lambda1": {"valor": None},
        "lambda2": {"valor": None},
    },
}


def tablas_app() -> dict[str, pd.DataFrame]:
    """Calcula las tablas ANOVA de los tres diseños con las funciones de utils/.

    Returns:
        Diccionario {diseño: tabla ANOVA}.
    """
    archivos = [RAIZ / "data" / f"{d}_mi_variante.csv" for d in ("dbca", "dcl", "dpd")]
    faltan = [a.name for a in archivos if not a.exists()]
    if faltan:
        sys.exit(f"Faltan los archivos en data/: {', '.join(faltan)}")
    dbca, dcl, dpd = (pd.read_csv(a) for a in archivos)

    # Nombres de columnas: ajústalos si tus archivos usan otros
    m_dbca = ajustar_dbca(dbca, "dia", "secador", "humedad")
    m_dcl = ajustar_cuadro_latino(dcl, "dia", "bandeja", "pretratamiento", "humedad")
    return {
        "DBCA": tabla_anova_dbca(m_dbca),
        "DCL": tabla_anova_dcl(m_dcl, dcl["pretratamiento"].nunique()),
        "DPD": anova_parcelas_divididas(dpd, "dia", "secador", "espesor", "humedad"),
    }


def _a_valores(tabla: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Convierte una tabla ANOVA en {fuente: {estadístico: valor}}.

    Args:
        tabla: Tabla con columna Fuente.

    Returns:
        Diccionario anidado con los valores de la tabla.
    """
    return {f: r.to_dict() for f, r in tabla.set_index("Fuente").iterrows()}


def valores_nuevos() -> dict[str, dict[str, dict[str, float]]]:
    """Calcula con la app el factorial, el RSM de primer orden y el CCD.

    Los archivos que no existan se omiten con un aviso.

    Returns:
        Diccionario {diseño: {fuente: {estadístico: valor}}}.
    """
    datos = RAIZ / "data"
    res: dict[str, dict[str, dict[str, float]]] = {}

    ruta = datos / "factorial_mi_variante.csv"
    if ruta.exists():
        fac = pd.read_csv(ruta)
        factores = ["carga", "pretratamiento"]  # ajusta si tus columnas se llaman distinto
        tabla = anova_factorial(fac, factores, "humedad")
        # Se renombran las filas a A, B y A×B para coincidir con MIS_VALORES_A_MANO
        nombres = {factores[0]: "A", factores[1]: "B", " × ".join(factores): "A×B"}
        tabla["Fuente"] = tabla["Fuente"].replace(nombres)
        res["Factorial"] = _a_valores(tabla)
    else:
        print(f"Aviso: falta {ruta.name}; se omite el factorial.")

    ruta = datos / "rsm1_mi_variante.csv"
    if ruta.exists():
        rsm = pd.read_csv(ruta)
        modelo, _, _ = ajustar_primer_orden(rsm, "eficiencia", "x1", "x2")
        # b12: interacción estimada solo con los puntos factoriales (sin centros)
        fact = rsm[(rsm["x1"].abs() > 1e-9) | (rsm["x2"].abs() > 1e-9)]
        x1x2 = fact["x1"] * fact["x2"]
        b12 = float((x1x2 * fact["eficiencia"]).sum() / (x1x2**2).sum())
        curv = prueba_curvatura(rsm, "eficiencia", "x1", "x2")
        b0, b1, b2 = (float(v) for v in modelo.params)
        res["RSM primer orden"] = {
            "b0": {"valor": b0},
            "b1": {"valor": b1},
            "b2": {"valor": b2},
            "b12": {"valor": b12},
            "F curvatura": {"valor": curv["F"]},
        }
    else:
        print(f"Aviso: falta {ruta.name}; se omite el RSM de primer orden.")

    ruta = datos / "ccd_mi_variante.csv"
    if ruta.exists():
        ccd = pd.read_csv(ruta)
        modelo, _, _, _, _ = ajustar_segundo_orden(ccd, "eficiencia", "x1", "x2")
        b0, b1, b2, b11, b22, b12 = (float(v) for v in modelo.params)
        B = matriz_B(b11, b22, b12)
        xs = punto_estacionario(b1, b2, B)
        vp, _ = clasificar(B)
        res["CCD"] = {
            "xs1": {"valor": float(xs[0])},
            "xs2": {"valor": float(xs[1])},
            "ys": {"valor": respuesta_en_punto(b0, np.array([b1, b2]), xs)},
            "lambda1": {"valor": float(vp[0])},
            "lambda2": {"valor": float(vp[1])},
        }
    else:
        print(f"Aviso: falta {ruta.name}; se omite el CCD.")
    return res


def comparar() -> int:
    """Compara cada SC y F de la app con MIS_VALORES_A_MANO e imprime el resultado.

    Returns:
        Número de valores que no coinciden.
    """
    fallos = 0
    resultados = {d: _a_valores(t) for d, t in tablas_app().items()}
    resultados.update(valores_nuevos())
    for diseno, tabla in resultados.items():
        print(f"\n=== {diseno} ===")
        for fuente, valores in MIS_VALORES_A_MANO[diseno].items():
            for estadistico, a_mano in valores.items():
                if a_mano is None:
                    print(f"  - {fuente} {estadistico}: pendiente (sin valor a mano)")
                    continue
                app = tabla[fuente][estadistico]
                if not np.isnan(app) and abs(app - a_mano) < TOLERANCIA:
                    print(f"  ✔ {fuente} {estadistico}")
                else:
                    fallos += 1
                    print(f"  ✘ {fuente} {estadistico}: app = {app:.4f} | a mano = {a_mano}")
    return fallos


if __name__ == "__main__":
    comparar()
