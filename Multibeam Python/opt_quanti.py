import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from steering_vector import steering_vector
from opt_phi_H import opt_phi_H
from opt_phi_AoD import opt_phi_AoD
from BF_quantize import BF_quantize
from FBND import FBND
from IGSS_Q import IGSS_Q
from creat_codebook import creat_codebook


def main():
    """Main optimization and quantization function"""
    # Optimization
    L = 8  # 11 multipaths
    theta_t_LOS = 0  # AoD of LOS
    theta_r_LOS = 0  # AoA of LOS

    np.random.seed(42)
    theta_t_NLOS = theta_t_LOS + -7 + 14 * np.random.rand(L - 1)
    theta_r_NLOS = theta_t_NLOS  # AoAs of NLOS
    theta_t = np.concatenate([[theta_t_LOS], theta_t_NLOS])
    theta_r = np.concatenate([[theta_r_LOS], theta_r_NLOS])

    d_LOS = 100
    d_NLOS = 100 + 200 * np.random.rand(L - 1)
    d = np.concatenate([[d_LOS], d_NLOS])
    tao_l = d / (3 * 10**8)

    v_A = 0
    v_o = 10 * np.random.rand(L - 1)
    v = np.concatenate([[v_A], v_A * np.ones(L - 1) + v_o])

    f_c = 3 * 10**10
    f_D = (v * f_c) / (3 * 10**8)

    b_LOS = np.sqrt(1 / 2) * (1 + 1j)
    b_NLOS = np.sqrt(0.1 / 2) * (np.ones(L - 1) + 1j * np.ones(L - 1))
    b = np.concatenate([[b_LOS], b_NLOS])

    M = 12
    H = np.zeros((M, M), dtype=complex)

    for i in range(L):
        a_t = steering_vector(theta_t[i], M).flatten()  # Chuyển thành 1D
        a_r = steering_vector(theta_r[i], M).flatten()  # Chuyển thành 1D
        H += (
            b[i] * np.exp(1j * 2 * np.pi * f_D[i] * tao_l[i]) * 0.5 * np.outer(a_r, a_t)
        )

    theta_sense = -10
    theta_DoA = theta_t_LOS
    a_c = steering_vector(theta_DoA, M).flatten()  # Chuyển thành 1D
    w_t_c = 1 / np.sqrt(M) * a_c
    a_s = steering_vector(theta_sense, M).flatten()  # Chuyển thành 1D
    w_t_s = 1 / np.sqrt(M) * a_s

    rho = 0.5
    phi_angle = np.arange(-180, 180, 0.1)
    phi = np.pi / 180 * phi_angle

    w_t = np.zeros((M, len(phi)), dtype=complex)
    P_r = np.zeros(len(phi))
    P_AoD = np.zeros(len(phi))

    for i in range(len(phi)):
        # Sửa lỗi shape: đảm bảo w_t[:, i] nhận được vector 1D
        w_t_col = np.sqrt(rho) * w_t_c + np.sqrt(1 - rho) * np.exp(1j * phi[i]) * w_t_s
        w_t[:, i] = w_t_col / np.linalg.norm(w_t_col)

        # Tính toán với vector 1D
        w_t_i = w_t[:, i]

        # Sửa công thức tính P_r
        denominator = (w_t_c.conj().T @ H.conj().T @ H @ w_t_c).real
        numerator = (w_t_i.conj().T @ H.conj().T @ H @ w_t_i).real
        P_r[i] = numerator / denominator if abs(denominator) > 1e-10 else 0

        # Sửa công thức tính P_AoD
        denominator_ao = np.abs(a_c @ w_t_c.conj()) ** 2  # a_c.'*w_t_c trong MATLAB
        numerator_ao = np.abs(a_c @ w_t_i.conj()) ** 2  # a_c.'*w_t(:,i) trong MATLAB
        P_AoD[i] = numerator_ao / denominator_ao if abs(denominator_ao) > 1e-10 else 0

    peaks_r, _ = find_peaks(P_r)
    peaks_AoD, _ = find_peaks(P_AoD)

    # Các tính toán tiếp theo
    phi_opt = opt_phi_H(rho, H, w_t_c, w_t_s)
    phi_opt_angle = 180 / np.pi * phi_opt
    w_t_opt1 = np.sqrt(rho) * w_t_c + np.sqrt(1 - rho) * np.exp(1j * phi_opt) * w_t_s
    Pr_opt1 = (
        1
        / (np.linalg.norm(H @ w_t_c) ** 2)
        * (w_t_opt1.conj().T @ H.conj().T @ H @ w_t_opt1).real
    )

    phi_opt_tilde = opt_phi_AoD(rho, a_c, w_t_c, w_t_s)
    phi_opt_angle_tilde = 180 / np.pi * phi_opt_tilde
    w_t_opt2 = (
        np.sqrt(rho) * w_t_c + np.sqrt(1 - rho) * np.exp(1j * phi_opt_tilde) * w_t_s
    )
    w_t_opt2 = w_t_opt2 / np.linalg.norm(w_t_opt2)
    Paod_opt2 = (
        1 / (np.abs(a_c @ w_t_c.conj()) ** 2) * np.abs(a_c @ w_t_opt2.conj()) ** 2
    )

    W = w_t_opt2

    # Plot
    plt.figure(1, figsize=(10, 6))
    plt.plot(phi_angle, P_r, "r", label="Power at Rx(H-Known)")
    plt.scatter(
        phi_angle[peaks_r],
        P_r[peaks_r],
        c="blue",
        marker="s",
        s=50,
        label="Max at Rx(H-Known)",
    )
    plt.scatter(
        phi_opt_angle, Pr_opt1, c="red", marker="x", s=100, label="Optimal φ at Rx"
    )
    plt.plot(phi_angle, P_AoD, "b--", label="Power at dominating AoD")
    plt.scatter(
        phi_angle[peaks_AoD],
        P_AoD[peaks_AoD],
        c="magenta",
        marker="o",
        s=50,
        label="Max at dominating AoD",
    )
    plt.scatter(
        phi_opt_angle_tilde,
        Paod_opt2,
        c="blue",
        marker="*",
        s=100,
        label="Optimal φ at AoD",
    )
    plt.scatter(
        phi_angle[peaks_AoD],
        P_r[peaks_AoD],
        c="blue",
        marker="^",
        s=50,
        label="Power at Rx(achieved by φ_opt)",
    )

    for peak in peaks_AoD:
        plt.axvline(x=phi_angle[peak], linestyle="--", color="b", alpha=0.5)

    plt.ylabel("Normalized Power")
    plt.xlabel("φ (Degrees)")
    plt.legend(loc="best")
    plt.grid(True)
    plt.title("Figure 4: Power vs Phase")
    plt.tight_layout()
    plt.show()

    scan_angle = np.arange(-12.5, 15, 2.5)
    power = np.zeros(len(scan_angle))

    for i in range(len(scan_angle)):
        a_hat = steering_vector(scan_angle[i], M).flatten()
        w_hat = 1 / np.sqrt(M) * a_hat
        power[i] = np.abs(w_t_opt2.conj().T @ H.conj().T @ w_hat) ** 2

    print("Scan angles analysis completed.")
    print("Scan angles:", scan_angle)
    print("Power values:", power)


if __name__ == "__main__":
    main()
