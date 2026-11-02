"""Prueba rápida: compara las tablas ANOVA de la app con tus cálculos a mano.

Uso (desde la raíz del proyecto):
    python verificacion/prueba_rapida.py

Rellena MIS_VALORES_A_MANO con los valores de tu hoja; los que queden en None
se omiten. Se compara cada SC y cada F; hay coincidencia si la diferencia es < 0.01.
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


def comparar() -> int:
    """Compara cada SC y F de la app con MIS_VALORES_A_MANO e imprime el resultado.

    Returns:
        Número de valores que no coinciden.
    """
    fallos = 0
    for diseno, tabla in tablas_app().items():
        print(f"\n=== {diseno} ===")
        tabla = tabla.set_index("Fuente")
        for fuente, valores in MIS_VALORES_A_MANO[diseno].items():
            for estadistico, a_mano in valores.items():
                if a_mano is None:
                    print(f"  - {fuente} {estadistico}: pendiente (sin valor a mano)")
                    continue
                app = tabla.loc[fuente, estadistico]
                if not np.isnan(app) and abs(app - a_mano) < TOLERANCIA:
                    print(f"  ✔ {fuente} {estadistico}")
                else:
                    fallos += 1
                    print(f"  ✘ {fuente} {estadistico}: app = {app:.4f} | a mano = {a_mano}")
    return fallos


if __name__ == "__main__":
    comparar()
