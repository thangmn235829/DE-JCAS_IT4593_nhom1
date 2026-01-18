import numpy as np


def steering_vector(theta_angle, M):
    """
    Generate steering vector for ULA

    Parameters:
        theta_angle: angle in degrees
        M: number of array elements

    Returns:
        a: steering vector (1D array)
    """
    d_vec = np.arange(0, M)
    theta_radian = np.pi / 180 * theta_angle
    phase = -np.pi * d_vec * np.sin(theta_radian)
    a = np.exp(1j * phase)

    return a  # Trả về 1D array, không reshape
