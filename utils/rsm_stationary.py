"""Punto estacionario y clasificación de la superficie de segundo orden."""

import numpy as np

# Radio de la región experimental (distancia axial del CCD rotable con 2 factores)
RADIO_REGION = 1.414


def matriz_B(b11: float, b22: float, b12: float) -> np.ndarray:
    """Matriz simétrica B de los coeficientes cuadráticos.

    Args:
        b11: Coeficiente de x1².
        b22: Coeficiente de x2².
        b12: Coeficiente de x1·x2.

    Returns:
        Matriz [[b11, b12/2], [b12/2, b22]].
    """
    return np.array([[b11, b12 / 2], [b12 / 2, b22]], dtype=float)


def punto_estacionario(b1: float, b2: float, B: np.ndarray) -> np.ndarray:
    """Calcula el punto donde la pendiente de la superficie es cero.

    Args:
        b1: Coeficiente lineal de x1.
        b2: Coeficiente lineal de x2.
        B: Matriz de coeficientes cuadráticos.

    Returns:
        Coordenadas codificadas (x1s, x2s).
    """
    b = np.array([b1, b2], dtype=float)
    # Se deriva ŷ = b0 + x'b + x'Bx e iguala a 0  ->  xs = -0.5 · B⁻¹ · b
    return -0.5 * np.linalg.inv(B) @ b


def respuesta_en_punto(b0: float, b: np.ndarray, xs: np.ndarray) -> float:
    """Respuesta predicha en el punto estacionario.

    Args:
        b0: Intercepto.
        b: Vector de coeficientes lineales (b1, b2).
        xs: Punto estacionario codificado.

    Returns:
        ŷs = b0 + 0.5 · xs'b
    """
    # Al sustituir xs en el modelo, el término cuadrático vale la mitad del lineal
    return float(b0 + 0.5 * np.asarray(xs) @ np.asarray(b))


def clasificar(B: np.ndarray) -> tuple[np.ndarray, str]:
    """Clasifica el punto estacionario según los valores propios de B.

    Args:
        B: Matriz de coeficientes cuadráticos.

    Returns:
        Valores propios (ascendentes) y tipo: "máximo", "mínimo" o "punto de silla".
    """
    # Los valores propios indican la curvatura en las direcciones principales
    vp = np.linalg.eigvalsh(B)
    if np.all(vp < 0):
        tipo = "máximo"
    elif np.all(vp > 0):
        tipo = "mínimo"
    else:
        tipo = "punto de silla"
    return vp, tipo


def fuera_de_region(xs: np.ndarray, radio: float = RADIO_REGION) -> bool:
    """Indica si el punto está más lejos del centro que el radio experimental.

    Args:
        xs: Punto codificado.
        radio: Distancia máxima del centro explorada.

    Returns:
        True si la distancia euclídea al centro supera el radio.
    """
    return bool(np.linalg.norm(xs) > radio)
