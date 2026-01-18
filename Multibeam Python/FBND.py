import numpy as np


def FBND(x, M):
    """
    FBND algorithm for quantization
    Parameters:
        x: input complex vector
        M: modulus for quantization
    Returns:
        g: quantized integer vector
    """
    eta = np.exp(1j * 2 * np.pi / M)
    arg = np.angle(x) * M / (2 * np.pi)
    g = np.round(arg)

    # Sort based on difference
    diff = g - arg
    u = np.argsort(diff)

    p = np.conj(x) * eta**g
    v = np.concatenate([np.array([np.sum(p)]), p[u] * (eta - 1)])

    cumsum_v = np.cumsum(v)
    best = np.argmax(np.abs(cumsum_v))

    g[u[:best]] = g[u[:best]] + 1
    g = np.mod(g - g[0], M)

    return g.astype(int)
