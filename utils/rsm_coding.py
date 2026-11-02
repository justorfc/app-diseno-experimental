"""Codificación y decodificación de variables para la Metodología de Superficie de Respuesta."""


def codificar(valor: float, centro: float, semirango: float) -> float:
    """Convierte un valor en unidades naturales a unidades codificadas.

    Args:
        valor: Valor natural (ξ), por ejemplo una inclinación en grados.
        centro: Valor natural que corresponde a 0 en la escala codificada.
        semirango: Distancia natural entre el centro y el nivel +1.

    Returns:
        Valor codificado x.
    """
    # x = (ξ - centro) / semirango
    # Así el nivel bajo queda en -1, el centro en 0 y el nivel alto en +1.
    return (valor - centro) / semirango


def decodificar(x: float, centro: float, semirango: float) -> float:
    """Convierte un valor codificado a unidades naturales.

    Args:
        x: Valor codificado.
        centro: Valor natural que corresponde a x = 0.
        semirango: Distancia natural entre el centro y el nivel +1.

    Returns:
        Valor natural ξ.
    """
    # Se despeja ξ de x = (ξ - centro) / semirango  ->  ξ = centro + x * semirango
    return centro + x * semirango
