import numpy as np


def twoStepILS(iter_nr, alpha, V_pattern, W0, PM, PdM):
    """
    Two-step Iterative Least-Squares (ILS)
    The function iteratively optimizes the beamforming matrix W0 based on the
    initial (PM) and the desired (PdM) pattern magnitudes.
    """
    # Đảm bảo PM và PdM là 1D arrays
    PM = np.asarray(PM).flatten()
    PdM = np.asarray(PdM).flatten()

    # Đảm bảo W0 là 2D array (M, 1)
    W0 = np.asarray(W0).reshape(-1, 1)

    for i in range(iter_nr):
        # Step 2 - Determining Theta
        V = V_pattern[:, alpha]

        # Step 3 - Condition of convergence
        # Sửa: PM và PdM đã là 1D, alpha là indices
        diff = np.abs(PM[alpha] - PdM[alpha])
        K = np.where(diff > 1e-3)[0]

        print(f"Iteration {i+1}, Total difference is {np.sum(diff):.6f}")

        if len(K) == 0:
            break

        # Step 4 - Update Weight Vector
        # Iterative Least-Squares
        while True:
            # Sửa: sử dụng pinv để tránh singular matrix
            PdP = (W0.conj().T @ V) @ np.linalg.pinv(np.diag(PdM[alpha]))
            PdP0 = PdP / (np.max(np.abs(PdP)) + 1e-10)  # Tránh chia cho 0

            # Tính W1
            VVH = V @ V.conj().T
            # Thêm regularization để tránh singular matrix
            VVH_reg = VVH + 1e-6 * np.eye(VVH.shape[0])
            W1 = np.linalg.inv(VVH_reg) @ V @ np.diag(PdM[alpha]) @ PdP0.conj().T

            if np.linalg.norm(W1 - W0) ** 2 < 1e-4:
                break
            else:
                W0 = W1

        W0 = W1
        PM = np.abs(W0.conj().T @ V_pattern).flatten()

    return W0
