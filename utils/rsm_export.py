"""Figuras de la superficie de respuesta con matplotlib y exportación a PNG/PDF."""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: solo se generan archivos
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from utils.rsm_coding import decodificar


def _naturales(X1: np.ndarray, X2: np.ndarray, xs: np.ndarray, et: dict):
    """Decodifica mallas y punto estacionario a unidades naturales.

    Args:
        X1: Malla codificada de x1.
        X2: Malla codificada de x2.
        xs: Punto estacionario codificado.
        et: Etiquetas con centro y semirango de cada variable.

    Returns:
        Mallas naturales (N1, N2) y punto estacionario natural (p1, p2).
    """
    return (
        decodificar(X1, et["centro1"], et["semi1"]),
        decodificar(X2, et["centro2"], et["semi2"]),
        decodificar(xs[0], et["centro1"], et["semi1"]),
        decodificar(xs[1], et["centro2"], et["semi2"]),
    )


def figura_superficie_mpl(
    X1: np.ndarray, X2: np.ndarray, Z: np.ndarray, xs: np.ndarray, ys: float, etiquetas: dict
) -> Figure:
    """Superficie 3D con el punto estacionario.

    Args:
        X1: Malla codificada de x1.
        X2: Malla codificada de x2.
        Z: Respuesta predicha en la malla.
        xs: Punto estacionario codificado.
        ys: Respuesta predicha en el punto estacionario.
        etiquetas: Nombres, unidades, centros y semirangos de las variables.

    Returns:
        Figura de matplotlib.
    """
    N1, N2, p1, p2 = _naturales(X1, X2, xs, etiquetas)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    sup = ax.plot_surface(N1, N2, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.scatter([p1], [p2], [ys], color="red", s=60, marker="D", label="Punto estacionario")
    ax.set_xlabel(f"{etiquetas['nombre1']} ({etiquetas['unidad1']})")
    ax.set_ylabel(f"{etiquetas['nombre2']} ({etiquetas['unidad2']})")
    ax.set_zlabel(etiquetas["respuesta"])
    fig.colorbar(sup, ax=ax, shrink=0.6, pad=0.1, label=etiquetas["respuesta"])
    ax.legend(loc="upper left")
    return fig


def figura_contornos_mpl(
    X1: np.ndarray, X2: np.ndarray, Z: np.ndarray, xs: np.ndarray,
    puntos_diseno: np.ndarray, etiquetas: dict,
) -> Figure:
    """Curvas de nivel con puntos del diseño y punto estacionario.

    Args:
        X1: Malla codificada de x1.
        X2: Malla codificada de x2.
        Z: Respuesta predicha en la malla.
        xs: Punto estacionario codificado.
        puntos_diseno: Arreglo (n, 2) con los puntos del diseño codificados.
        etiquetas: Nombres, unidades, centros y semirangos de las variables.

    Returns:
        Figura de matplotlib.
    """
    N1, N2, p1, p2 = _naturales(X1, X2, xs, etiquetas)
    d1 = decodificar(puntos_diseno[:, 0], etiquetas["centro1"], etiquetas["semi1"])
    d2 = decodificar(puntos_diseno[:, 1], etiquetas["centro2"], etiquetas["semi2"])
    fig, ax = plt.subplots(figsize=(8, 6))
    relleno = ax.contourf(N1, N2, Z, levels=15, cmap="viridis")
    lineas = ax.contour(N1, N2, Z, levels=15, colors="white", linewidths=0.6)
    ax.clabel(lineas, inline=True, fontsize=8, fmt="%.1f")
    ax.scatter(d1, d2, c="white", edgecolors="black", s=45, label="Puntos del diseño")
    ax.scatter([p1], [p2], c="red", marker="x", s=90, label="Punto estacionario")
    fig.colorbar(relleno, ax=ax, label=etiquetas["respuesta"])
    ax.set_xlabel(f"{etiquetas['nombre1']} ({etiquetas['unidad1']})")
    ax.set_ylabel(f"{etiquetas['nombre2']} ({etiquetas['unidad2']})")
    ax.legend(loc="upper right", fontsize=8)
    return fig


def a_bytes(fig: Figure, formato: str) -> bytes:
    """Serializa una figura a PNG (300 dpi) o PDF (vectorial).

    Args:
        fig: Figura de matplotlib.
        formato: "png" o "pdf".

    Returns:
        Contenido del archivo en bytes.

    Raises:
        ValueError: Si el formato no es png ni pdf.
    """
    formato = formato.lower()
    if formato not in ("png", "pdf"):
        raise ValueError("Formato no compatible: use 'png' o 'pdf'.")
    buffer = io.BytesIO()
    # dpi solo afecta al PNG (raster); el PDF conserva los trazos vectoriales
    fig.savefig(buffer, format=formato, dpi=300, bbox_inches="tight")
    return buffer.getvalue()


def guardar_en_carpeta(fig: Figure, nombre: str, carpeta: str = "figuras") -> list[Path]:
    """Guarda la figura como PNG y PDF en una carpeta (se crea si no existe).

    Args:
        fig: Figura de matplotlib.
        nombre: Nombre base de los archivos, sin extensión.
        carpeta: Carpeta de destino.

    Returns:
        Rutas de los archivos guardados.
    """
    destino = Path(carpeta)
    destino.mkdir(parents=True, exist_ok=True)
    rutas = []
    for formato in ("png", "pdf"):
        ruta = destino / f"{nombre}.{formato}"
        ruta.write_bytes(a_bytes(fig, formato))
        rutas.append(ruta)
    return rutas
