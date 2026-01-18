import numpy as np
from cal_error import cal_error


def IGSS_Q(a_1, a_2, L_max, epsilon_0, c, w):
    """
    IGSS_Q Algorithm for quantization

    Parameters:
        a_1: initial lower bound
        a_2: initial upper bound
        L_max: maximum iterations
        epsilon_0: convergence threshold
        c: codebook
        w: beamforming vector

    Returns:
        v_min: optimal scaling factor
        q: quantized vector
    """
    M = len(w)
    K = len(c)
    rho = (np.sqrt(5) - 1) / 2  # golden ratio

    # initialization
    l = 0
    a1_l = a_1
    a2_l = a_2
    d_l = a2_l - a1_l

    x1_l = a1_l + (1 - rho) * d_l
    x2_l = a1_l + rho * d_l

    while (l <= L_max) and (np.abs(d_l) > epsilon_0):
        Err_a1 = cal_error(a1_l, w, c)
        Err_a2 = cal_error(a2_l, w, c)
        Err_x1 = cal_error(x1_l, w, c)
        Err_x2 = cal_error(x2_l, w, c)

        errors = [Err_a1, Err_x1, Err_x2, Err_a2]
        I_min = np.argmin(errors)

        if I_min in [0, 1]:  # Err_a1 or Err_x1 is minimum
            a2_l = x2_l
            x2_l = x1_l
            x1_l = a1_l + np.random.rand() * np.abs(a1_l - x2_l)
        elif I_min in [2, 3]:  # Err_x2 or Err_a2 is minimum
            a1_l = x1_l
            x1_l = x2_l
            x2_l = x1_l + np.random.rand() * np.abs(x1_l - a2_l)

        d_l = a2_l - a1_l
        l += 1

    Err_x1 = cal_error(x1_l, w, c)
    Err_x2 = cal_error(x2_l, w, c)

    if Err_x1 < Err_x2:
        v_min = x1_l
    else:
        v_min = x2_l

    # Quantize using optimal scaling factor
    q = np.zeros(M, dtype=complex)
    for i in range(M):
        diff = np.abs(v_min * w[i] - c)
        diff = diff**2
        min_ind = np.argmin(diff)
        q[i] = c[min_ind]

    return v_min, q
