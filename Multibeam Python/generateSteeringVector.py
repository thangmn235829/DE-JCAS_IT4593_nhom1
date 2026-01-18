import numpy as np


def generateSteeringVector(theta, M, lambda_):
    """
    Function generates the steering vector from the array response vector.
    It is assumed that M antenna elements are equally spaced at the interval
    of half of the wavelength. Planar wavefront and narrow-band beamforming
    models are assumed.

    Parameters:
        theta: angle vector (in radians)
        M: number of antenna elements
        lambda_: wavelength

    Returns:
        V_pattern: steering vector matrix
    """
    x = lambda_ / 2  # Half-wavelength spacing
    V_pattern = np.zeros((M, len(theta)), dtype=complex)

    for i in range(M):
        phase = 1j * 2 * np.pi / lambda_ * x * i * np.sin(theta)
        V_pattern[i, :] = np.exp(phase)

    return V_pattern
