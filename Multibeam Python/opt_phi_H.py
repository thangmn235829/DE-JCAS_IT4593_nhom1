import numpy as np


def opt_phi_H(rho, H, w_c, w_s):
    """
    Calculate optimal phase shift for channel-known scenario

    Parameters:
        rho: power allocation parameter
        H: channel matrix
        w_c: communication beamforming vector
        w_s: sensing beamforming vector

    Returns:
        phi_opt: optimal phase shift
    """
    a_1 = np.abs(w_c.conj().T @ (H.conj().T @ H) @ w_s).item()
    a_2 = np.abs(w_c.conj().T @ w_s).item()

    alpha_1 = np.angle(w_c.conj().T @ (H.conj().T @ H) @ w_s).item()
    alpha_2 = np.angle(w_c.conj().T @ w_s).item()

    P = np.sqrt(rho * (1 - rho))
    L = -4 * P**2 * np.abs(a_1) * np.abs(a_2) * np.sin(alpha_1 - alpha_2)

    X_1 = -2 * P * np.abs(a_1) * np.cos(alpha_1) + 2 * P * np.abs(a_2) * (
        rho * np.linalg.norm(H @ w_c) ** 2 + (1 - rho) * np.linalg.norm(H @ w_s) ** 2
    ) * np.cos(alpha_2)

    X_2 = -2 * P * np.abs(a_1) * np.sin(alpha_1) + 2 * P * np.abs(a_2) * (
        rho * np.linalg.norm(H @ w_c) ** 2 + (1 - rho) * np.linalg.norm(H @ w_s) ** 2
    ) * np.sin(alpha_2)

    mu_0 = np.arcsin(L / np.sqrt(X_1**2 + X_2**2))
    upsilon = np.arctan2(X_2, X_1)

    l = np.arange(-3, 4)

    if X_1 >= 0:
        phi = (np.pi + mu_0 - upsilon) + 2 * l * np.pi
    else:
        phi = (mu_0 - upsilon) + 2 * l * np.pi

    # Find phi in range [-pi, pi)
    phi_in_range = phi[(-np.pi <= phi) & (phi < np.pi)]

    if len(phi_in_range) > 0:
        phi_opt = phi_in_range[0]
    else:
        phi_opt = phi[0] % (2 * np.pi)
        if phi_opt > np.pi:
            phi_opt -= 2 * np.pi

    return phi_opt
