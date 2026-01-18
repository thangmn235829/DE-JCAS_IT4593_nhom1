import numpy as np


def cal_error(v, w, c):
    """
    Calculate the quantization error

    Parameters:
        v: scaling factor
        w: beamforming vector
        c: codebook

    Returns:
        e: quantization error
    """
    M = len(w)
    K = len(c)
    q = np.zeros(M, dtype=complex)

    for i in range(M):
        diff = np.abs(v * w[i] - c)
        diff = diff**2
        min_ind = np.argmin(diff)
        q[i] = c[min_ind]

    e = np.real(np.sum(v * w - q))

    return e
