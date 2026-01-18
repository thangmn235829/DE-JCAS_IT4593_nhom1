import numpy as np


def generateQuantizedArrResponse(M, eqDir):
    """
    Function generates a quantized array response. Quantization refers to the
    phase delay term sin(theta), where the function's value (as opposed to
    its argument) is quantized and hence becomes equispaced. This then allows
    to use a single phase-shifting vector to displace the reference
    pattern along the equivalent directions.

    Parameters:
        M: number of antenna elements
        eqDir: equivalent directions (quantized sin(theta))

    Returns:
        Aq: quantized array response matrix
    """
    Aq = np.zeros((M, len(eqDir)), dtype=complex)

    for m in range(M):
        for q in range(len(eqDir)):
            Aq[m, q] = np.exp(1j * np.pi * m * eqDir[q])

    return Aq
