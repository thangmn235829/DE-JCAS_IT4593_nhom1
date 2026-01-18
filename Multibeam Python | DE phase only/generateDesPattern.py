import numpy as np
from findNulls import findNulls


def generateDesPattern(theta, mainLobeAngle, V_pattern):
    """
    Generate desired pattern with main lobe at mainLobeAngle
    """
    # Tìm index gần nhất
    mainLobeDirIdx = np.argmin(np.abs(theta - mainLobeAngle))

    # Capon's beamforming
    v_main = V_pattern[:, mainLobeDirIdx].reshape(-1, 1)
    Ruu = v_main @ v_main.conj().T + (np.random.rand() / 1000) ** 2 * np.eye(
        V_pattern.shape[0]
    )

    try:
        Ruu_inv = np.linalg.inv(Ruu)
        numerator = Ruu_inv @ v_main
        denominator = v_main.conj().T @ Ruu_inv @ v_main
        W0 = numerator / denominator
    except np.linalg.LinAlgError:
        # Nếu matrix singular, sử dụng pseudo-inverse
        Ruu_pinv = np.linalg.pinv(Ruu)
        numerator = Ruu_pinv @ v_main
        denominator = v_main.conj().T @ Ruu_pinv @ v_main
        W0 = numerator / denominator

    # Step 1 of ILS
    P_refGen = np.abs(W0.conj().T @ V_pattern).flatten()

    # Find nulls
    nullIdcs = findNulls(P_refGen)

    # Tạo desired pattern
    PdM = P_refGen.copy()
    if len(nullIdcs) >= 2:
        PdM[: nullIdcs[0]] = 0
        PdM[nullIdcs[1] :] = 0
    else:
        # Fallback nếu không tìm thấy đủ nulls
        center_idx = len(PdM) // 2
        PdM[: center_idx - 50] = 0
        PdM[center_idx + 50 :] = 0

    return PdM, P_refGen, W0.flatten()
