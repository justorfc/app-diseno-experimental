"""Gráficos de la superficie de respuesta (plotly)."""

import numpy as np
import plotly.graph_objects as go

from utils.rsm_coding import decodificar


def _ejes_naturales(X1: np.ndarray, X2: np.ndarray, et: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decodifica las mallas a unidades naturales.

    Args:
        X1: Malla codificada de x1.
        X2: Malla codificada de x2.
        et: Etiquetas con centro y semirango de cada variable.

    Returns:
        Mallas naturales (ξ1, ξ2).
    """
    return (
        decodificar(X1, et["centro1"], et["semi1"]),
        decodificar(X2, et["centro2"], et["semi2"]),
    )


def superficie_3d(
    X1: np.ndarray, X2: np.ndarray, Z: np.ndarray, xs: np.ndarray, ys: float, etiquetas: dict
) -> go.Figure:
    """Superficie 3D con el punto estacionario marcado.

    Args:
        X1: Malla codificada de x1.
        X2: Malla codificada de x2.
        Z: Respuesta predicha en la malla.
        xs: Punto estacionario codificado (x1s, x2s).
        ys: Respuesta predicha en el punto estacionario.
        etiquetas: Diccionario con nombre1, unidad1, centro1, semi1, nombre2,
            unidad2, centro2, semi2 y respuesta.

    Returns:
        Figura de plotly.
    """
    N1, N2 = _ejes_naturales(X1, X2, etiquetas)
    p1 = decodificar(xs[0], etiquetas["centro1"], etiquetas["semi1"])
    p2 = decodificar(xs[1], etiquetas["centro2"], etiquetas["semi2"])
    fig = go.Figure(
        go.Surface(x=N1, y=N2, z=Z, colorscale="Viridis",
                   colorbar=dict(title=etiquetas["respuesta"]))
    )
    fig.add_trace(
        go.Scatter3d(x=[p1], y=[p2], z=[ys], mode="markers", name="Punto estacionario",
                     marker=dict(size=7, color="red", symbol="diamond"))
    )
    fig.update_layout(
        scene=dict(
            xaxis_title=f"{etiquetas['nombre1']} ({etiquetas['unidad1']})",
            yaxis_title=f"{etiquetas['nombre2']} ({etiquetas['unidad2']})",
            zaxis_title=etiquetas["respuesta"],
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def curvas_nivel(
    X1: np.ndarray, X2: np.ndarray, Z: np.ndarray, xs: np.ndarray,
    puntos_diseno: np.ndarray, etiquetas: dict,
) -> go.Figure:
    """Curvas de nivel con los puntos del diseño y el punto estacionario.

    Args:
        X1: Malla codificada de x1.
        X2: Malla codificada de x2.
        Z: Respuesta predicha en la malla.
        xs: Punto estacionario codificado (x1s, x2s).
        puntos_diseno: Arreglo (n, 2) con los puntos del diseño codificados.
        etiquetas: Diccionario con nombre1, unidad1, centro1, semi1, nombre2,
            unidad2, centro2, semi2 y respuesta.

    Returns:
        Figura de plotly.
    """
    N1, N2 = _ejes_naturales(X1, X2, etiquetas)
    d1 = decodificar(puntos_diseno[:, 0], etiquetas["centro1"], etiquetas["semi1"])
    d2 = decodificar(puntos_diseno[:, 1], etiquetas["centro2"], etiquetas["semi2"])
    p1 = decodificar(xs[0], etiquetas["centro1"], etiquetas["semi1"])
    p2 = decodificar(xs[1], etiquetas["centro2"], etiquetas["semi2"])
    fig = go.Figure(
        go.Contour(
            x=N1[0], y=N2[:, 0], z=Z, colorscale="Viridis",
            contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
            colorbar=dict(title=etiquetas["respuesta"]),
        )
    )
    fig.add_trace(go.Scatter(x=d1, y=d2, mode="markers", name="Puntos del diseño",
                             marker=dict(size=9, color="white", line=dict(width=1.5, color="black"))))
    fig.add_trace(go.Scatter(x=[p1], y=[p2], mode="markers", name="Punto estacionario",
                             marker=dict(size=13, color="red", symbol="x")))
    fig.update_layout(
        xaxis_title=f"{etiquetas['nombre1']} ({etiquetas['unidad1']})",
        yaxis_title=f"{etiquetas['nombre2']} ({etiquetas['unidad2']})",
        height=600,
    )
    return fig
