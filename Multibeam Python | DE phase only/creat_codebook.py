import numpy as np


def creat_codebook(phi, b_1, b_2):
    """
    Create codebook for quantization

    Parameters:
        phi: phase offset
        b_1: number of bits for first component
        b_2: number of bits for second component

    Returns:
        c_unique: unique codebook entries
    """
    delta_1 = 2 * np.pi * 2 ** (-b_1)
    delta_2 = 2 * np.pi * 2 ** (-b_2)

    B_1 = np.arange(0, 2**b_1) * delta_1
    B_2 = np.arange(0, 2**b_2) * delta_2 + phi

    c = []
    for i in range(len(B_1)):
        for j in range(len(B_2)):
            c_new = np.exp(1j * B_1[i]) + np.exp(1j * B_2[j])

            # Handle numerical precision issues
            if np.abs(np.real(c_new)) <= 1e-7:
                c_new_real = 0
            else:
                c_new_real = np.real(c_new)

            if np.abs(np.imag(c_new)) <= 1e-7:
                c_new_imag = 0
            else:
                c_new_imag = np.imag(c_new)

            c_new = c_new_real + 1j * c_new_imag
            c.append(c_new)

    c_array = np.array(c)
    c_unique = np.unique(c_array)

    # Remove zero if present
    c_unique = c_unique[np.abs(c_unique) > 1e-10]

    return c_unique
