"""Comparaciones múltiples de Tukey usando el CM Error del modelo ajustado."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import studentized_range


def tukey_con_cme(
    medias: pd.Series, cme: float, gl_error: int, r: int, alfa: float = 0.05
) -> tuple[pd.DataFrame, float]:
    """Compara todos los pares de medias con Tukey usando el CM Error del modelo.

    A diferencia de ``pairwise_tukeyhsd``, usa el CM Error y los gl Error del
    modelo ajustado (por ejemplo, el que descuenta bloques).

    Args:
        medias: Medias de los tratamientos (índice = nombre del tratamiento).
        cme: Cuadrado medio del error del modelo.
        gl_error: Grados de libertad del error del modelo.
        r: Número de observaciones por media.
        alfa: Nivel de significancia.

    Returns:
        Tabla de comparaciones en español y la DMS de Tukey.
    """
    t = len(medias)  # número de tratamientos comparados
    # Error estándar de una media: EE = sqrt(CME / r)
    ee = np.sqrt(cme / r)

    filas = []
    for a, b in combinations(medias.index, 2):
        # Diferencia entre medias: d = media_a - media_b
        dif = medias[a] - medias[b]
        # Estadístico q = |d| / sqrt(CME / r)
        q = abs(dif) / ee
        # p-valor = P(Q > q) con la distribución del rango studentizado (t, gl_error)
        p = studentized_range.sf(q, t, gl_error)
        filas.append(
            {
                "Comparación": f"{a} - {b}",
                "Diferencia": dif,
                "q": q,
                "p-valor": p,
                "Significativa": p < alfa,
            }
        )
    tabla = pd.DataFrame(filas)

    # DMS = q_crítico(1 - alfa; t, gl_error) * sqrt(CME / r)
    dms = float(studentized_range.ppf(1 - alfa, t, gl_error) * ee)
    return tabla, dms


def letras_tukey(medias: pd.Series, dms: float) -> pd.DataFrame:
    """Asigna letras de grupo: medias que no difieren comparten letra.

    Args:
        medias: Medias de los tratamientos.
        dms: Diferencia mínima significativa de Tukey.

    Returns:
        Tabla con tratamiento, media y letras, ordenada de mayor a menor media.
    """
    # Orden descendente de las medias
    orden = medias.sort_values(ascending=False)
    nombres = list(orden.index)
    valores = orden.to_numpy()
    n = len(nombres)

    # Para cada media i, el grupo llega hasta la última j con (media_i - media_j) <= DMS
    grupos: list[tuple[int, int]] = []
    for i in range(n):
        j = i
        while j + 1 < n and valores[i] - valores[j + 1] <= dms:
            j += 1
        # Se descartan grupos contenidos en otro ya existente (absorción)
        if not any(g[0] <= i and j <= g[1] for g in grupos):
            grupos.append((i, j))

    letras = ["" for _ in range(n)]
    for k, (ini, fin) in enumerate(grupos):
        letra = chr(ord("a") + k)
        for idx in range(ini, fin + 1):
            letras[idx] += letra

    return pd.DataFrame({"Tratamiento": nombres, "Media": valores, "Letras": letras})
