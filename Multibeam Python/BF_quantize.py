import numpy as np


def BF_quantize(w, b):
    """
    BF quantization function
    Input:
        w: BF vector (numpy array)
        b: number of quantization bits
    Output:
        w_hat_1: quantized vector for codebook 1
        w_hat_2: quantized vector for codebook 2
        c_1_hat: codebook 1
        c_2_hat: codebook 2
    """
    M = len(w)
    delta = 2 * np.pi * 2 ** (-b)  # quantization step

    # normalization factors
    h_1 = np.sqrt(2 + 2 ** (2 - b)) * np.sqrt(M)
    h_2 = np.sqrt(2 * M)

    # create two codebooks with phi=0 and phi=delta/2
    # Note: creat_codebook function needs to be imported/defined
    from creat_codebook import creat_codebook

    c_1_hat = creat_codebook(0, b, b)
    c_2_hat = creat_codebook(delta / 2, b, b)

    # normalization
    c_1 = c_1_hat / h_1
    c_2 = c_2_hat / h_2

    w_hat_1 = np.zeros(M, dtype=complex)
    w_hat_2 = np.zeros(M, dtype=complex)

    for i in range(M):
        # Calculate differences for codebook 1
        diff_1 = np.abs(w[i] * np.ones(len(c_1)) - c_1)
        diff_1 = diff_1**2

        # Calculate differences for codebook 2
        diff_2 = np.abs(w[i] * np.ones(len(c_2)) - c_2)
        diff_2 = diff_2**2

        # Find minimum indices
        ind_1 = np.argmin(diff_1)
        ind_2 = np.argmin(diff_2)

        w_hat_1[i] = c_1[ind_1]
        w_hat_2[i] = c_2[ind_2]

    return w_hat_1, w_hat_2, c_1_hat, c_2_hat
