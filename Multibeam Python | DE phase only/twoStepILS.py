import numpy as np
import matplotlib.pyplot as plt


def twoStepILS(iter_nr, alpha, V_pattern, W0, PM, PdM):
    """
    Two-step Differential Evolution for beamforming optimization
    Uses custom DE implementation with DE/rand/1/bin strategy
    Optimizes both amplitude and phase to match desired pattern PdM at indices alpha
    """
    # Đảm bảo các vector ở dạng đúng
    PM = np.asarray(PM).flatten()
    PdM = np.asarray(PdM).flatten()
    W0 = np.asarray(W0).flatten()  # Chuyển thành 1D array

    M = len(W0)  # Số phần tử anten
    N_alpha = len(alpha)  # Số điểm cần khớp

    # Trích xuất đặc trưng từ W0 ban đầu
    W0_amplitude = np.abs(W0)
    W0_phase = np.angle(W0)

    # Hàm mục tiêu: minimize error between current pattern and desired pattern
    def objective_function(params):
        """
        Objective function for DE optimization
        params: first M elements are phases, next M elements are amplitude scaling factors
        """
        phases = params[:M]
        amp_scales = params[M:] if len(params) > M else np.ones(M)

        # Giới hạn biên độ scaling factor trong khoảng hợp lý
        amp_scales = np.clip(amp_scales, 0.5, 2.0)
        amplitudes = W0_amplitude * amp_scales

        # Tạo vector beamforming mới
        W_new = amplitudes * np.exp(1j * phases)

        # Tính pattern hiện tại
        current_pattern = np.abs(W_new.conj() @ V_pattern)

        # Tính sai số bình phương tại các vị trí alpha
        error = np.sum((current_pattern[alpha] - PdM[alpha]) ** 2)

        # Thêm regularization để tránh quá khác biệt với W0 ban đầu
        reg_weight = 0.01
        reg_error = reg_weight * np.sum((amp_scales - 1.0) ** 2)

        return error + reg_error

    # ============================================================
    # THUẬT TOÁN DIFFERENTIAL EVOLUTION TỰ TRIỂN KHAI
    # Sử dụng biến thể DE/rand/1/bin
    # ============================================================

    # Thiết lập các tham số DE
    NP = 20 * M  # Kích thước quần thể (Population size)
    F = 0.8  # Hệ số đột biến (Mutation factor)
    CR = 0.9  # Tỷ lệ tái tổ hợp (Crossover rate)
    max_gen = min(iter_nr * 10, 500)  # Số thế hệ tối đa

    # Số tham số cần tối ưu: M pha + M biên độ scaling factors
    D = 2 * M

    # Giới hạn cho từng tham số
    bounds = np.array([(-np.pi, np.pi)] * M + [(0.5, 2.0)] * M)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]

    # 1. KHỞI TẠO QUẦN THỂ
    print(f"\nInitializing DE/rand/1/bin optimization...")
    print(f"  Population size (NP): {NP}")
    print(f"  Parameters (D): {D}")
    print(f"  Mutation factor (F): {F}")
    print(f"  Crossover rate (CR): {CR}")
    print(f"  Max generations: {max_gen}")

    # Khởi tạo quần thể ngẫu nhiên trong giới hạn
    population = np.random.rand(NP, D)
    for i in range(D):
        population[:, i] = lower_bounds[i] + population[:, i] * (
            upper_bounds[i] - lower_bounds[i]
        )

    # Đánh giá quần thể ban đầu
    fitness = np.zeros(NP)
    for i in range(NP):
        fitness[i] = objective_function(population[i])

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    # Theo dõi quá trình hội tụ
    convergence_history = [best_fitness]

    # 2. VÒNG LẶP CHÍNH CỦA DE
    for gen in range(max_gen):
        new_population = population.copy()
        new_fitness = fitness.copy()

        for i in range(NP):
            # 2a. CHỌN 3 CÁ THỂ NGẪU NHIÊN KHÁC NHAU (rand/1)
            candidates = [j for j in range(NP) if j != i]
            idxs = np.random.choice(candidates, 3, replace=False)
            a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]

            # 2b. TẠO VECTOR ĐỘT BIẾN (Mutation)
            mutant = a + F * (b - c)

            # 2c. RÀNG BUỘC GIỚI HẠN CHO VECTOR ĐỘT BIẾN
            for d in range(D):
                # Xử lý vượt quá giới hạn bằng cách reflection
                if mutant[d] < lower_bounds[d]:
                    mutant[d] = 2 * lower_bounds[d] - mutant[d]
                elif mutant[d] > upper_bounds[d]:
                    mutant[d] = 2 * upper_bounds[d] - mutant[d]

                # Nếu vẫn vượt quá, random lại
                if mutant[d] < lower_bounds[d] or mutant[d] > upper_bounds[d]:
                    mutant[d] = lower_bounds[d] + np.random.rand() * (
                        upper_bounds[d] - lower_bounds[d]
                    )

            # 2d. LAI GHÉP NHỊ PHÂN (Binomial Crossover)
            trial = population[i].copy()
            j_rand = np.random.randint(D)  # Đảm bảo ít nhất 1 tham số được crossover

            for d in range(D):
                if np.random.rand() < CR or d == j_rand:
                    trial[d] = mutant[d]

            # 2e. CHỌN LỌC (Selection)
            trial_fitness = objective_function(trial)

            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness

                # Cập nhật best solution
                if trial_fitness < best_fitness:
                    best_solution = trial.copy()
                    best_fitness = trial_fitness

        # Cập nhật quần thể
        population = new_population
        fitness = new_fitness
        convergence_history.append(best_fitness)

        # Hiển thị tiến độ
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"  Generation {gen+1}/{max_gen}: Best fitness = {best_fitness:.6f}")

    # ============================================================
    # KẾT THÚC THUẬT TOÁN DE
    # ============================================================

    print(f"\nDE Optimization completed!")
    print(f"  Final best fitness: {best_fitness:.6f}")
    print(f"  Total generations: {max_gen}")

    # Vẽ biểu đồ hội tụ
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_history, "b-", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("DE Convergence History (DE/rand/1/bin)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Tạo vector beamforming tối ưu từ kết quả DE
    optimal_phases = best_solution[:M]

    if len(best_solution) > M:
        optimal_amp_scales = best_solution[M:]
        optimal_amp_scales = np.clip(optimal_amp_scales, 0.5, 2.0)
        optimal_amplitudes = W0_amplitude * optimal_amp_scales
    else:
        optimal_amplitudes = W0_amplitude

    W_opt = optimal_amplitudes * np.exp(1j * optimal_phases)
    W_opt = W_opt.reshape(-1, 1)  # Trả về dạng (M, 1)

    # Tính pattern tối ưu
    PM_opt = np.abs(W_opt.conj().T @ V_pattern).flatten()

    # Tính các chỉ số hiệu suất
    initial_error = np.mean(np.abs(PM[alpha] - PdM[alpha]))
    final_error = np.mean(np.abs(PM_opt[alpha] - PdM[alpha]))
    improvement = initial_error - final_error

    print(f"\nPerformance Summary:")
    print(f"  Initial pattern error: {initial_error:.6f}")
    print(f"  Final pattern error: {final_error:.6f}")
    print(f"  Improvement: {improvement:.6f}")
    print(f"  Improvement percentage: {(improvement/initial_error*100):.2f}%")

    # Kiểm tra đầu ra
    if np.isnan(final_error) or final_error > initial_error:
        print(f"Warning: DE did not improve the solution. Using initial W0.")
        W_opt = W0.reshape(-1, 1)

    return W_opt


# import numpy as np


# def twoStepILS(iter_nr, alpha, V_pattern, W0, PM, PdM):
#     """
#     Two-step Iterative Least-Squares (ILS)
#     The function iteratively optimizes the beamforming matrix W0 based on the
#     initial (PM) and the desired (PdM) pattern magnitudes.
#     """
#     # Đảm bảo PM và PdM là 1D arrays
#     PM = np.asarray(PM).flatten()
#     PdM = np.asarray(PdM).flatten()

#     # Đảm bảo W0 là 2D array (M, 1)
#     W0 = np.asarray(W0).reshape(-1, 1)

#     for i in range(iter_nr):
#         # Step 2 - Determining Theta
#         V = V_pattern[:, alpha]

#         # Step 3 - Condition of convergence
#         # Sửa: PM và PdM đã là 1D, alpha là indices
#         diff = np.abs(PM[alpha] - PdM[alpha])
#         K = np.where(diff > 1e-3)[0]

#         print(f"Iteration {i+1}, Total difference is {np.sum(diff):.6f}")

#         if len(K) == 0:
#             break

#         # Step 4 - Update Weight Vector
#         # Iterative Least-Squares
#         while True:
#             # Sửa: sử dụng pinv để tránh singular matrix
#             PdP = (W0.conj().T @ V) @ np.linalg.pinv(np.diag(PdM[alpha]))
#             PdP0 = PdP / (np.max(np.abs(PdP)) + 1e-10)  # Tránh chia cho 0

#             # Tính W1
#             VVH = V @ V.conj().T
#             # Thêm regularization để tránh singular matrix
#             VVH_reg = VVH + 1e-6 * np.eye(VVH.shape[0])
#             W1 = np.linalg.inv(VVH_reg) @ V @ np.diag(PdM[alpha]) @ PdP0.conj().T

#             if np.linalg.norm(W1 - W0) ** 2 < 1e-4:
#                 break
#             else:
#                 W0 = W1

#         W0 = W1
#         PM = np.abs(W0.conj().T @ V_pattern).flatten()

#     return W0

# =========================================================

# import numpy as np
# from scipy.optimize import differential_evolution


# def twoStepILS(iter_nr, alpha, V_pattern, W0, PM, PdM):
#     """
#     Two-step Differential Evolution for beamforming optimization
#     Optimizes both amplitude and phase to match desired pattern PdM at indices alpha
#     """
#     # Đảm bảo các vector ở dạng đúng
#     PM = np.asarray(PM).flatten()
#     PdM = np.asarray(PdM).flatten()
#     W0 = np.asarray(W0).flatten()  # Chuyển thành 1D array

#     M = len(W0)  # Số phần tử anten
#     N_alpha = len(alpha)  # Số điểm cần khớp

#     # Trích xuất đặc trưng từ W0 ban đầu
#     W0_amplitude = np.abs(W0)
#     W0_phase = np.angle(W0)

#     # Hàm mục tiêu: minimize error between current pattern and desired pattern
#     def objective_function(params):
#         """
#         Objective function for DE optimization
#         params: first M elements are phases, next M elements are amplitude scaling factors
#         """
#         phases = params[:M]
#         amp_scales = params[M:] if len(params) > M else np.ones(M)

#         # Tạo vector beamforming mới từ pha và biên độ
#         # Giới hạn biên độ scaling factor trong khoảng hợp lý
#         amp_scales = np.clip(amp_scales, 0.5, 2.0)
#         amplitudes = W0_amplitude * amp_scales

#         # Tạo vector beamforming mới
#         W_new = amplitudes * np.exp(1j * phases)

#         # Tính pattern hiện tại
#         current_pattern = np.abs(W_new.conj() @ V_pattern)

#         # Tính sai số bình phương tại các vị trí alpha
#         error = np.sum((current_pattern[alpha] - PdM[alpha]) ** 2)

#         # Thêm regularization để tránh quá khác biệt với W0 ban đầu
#         reg_weight = 0.01
#         reg_error = reg_weight * np.sum((amp_scales - 1.0) ** 2)

#         return error + reg_error

#     # Thiết lập các giới hạn cho tham số
#     # M pha đầu tiên: từ -π đến π
#     # M biên độ scaling factors tiếp theo: từ 0.5 đến 2.0
#     bounds = [(-np.pi, np.pi) for _ in range(M)] + [(0.5, 2.0) for _ in range(M)]

#     # Khởi tạo quần thể từ W0
#     initial_population = []
#     for i in range(15):  # Tạo 15 cá thể ban đầu
#         # Thêm nhiễu nhỏ vào W0
#         noise_phase = np.random.normal(0, 0.1, M)
#         noise_amp = np.random.normal(1.0, 0.1, M)

#         # Tạo cá thể từ W0 + nhiễu
#         individual = np.concatenate([W0_phase + noise_phase, noise_amp])
#         initial_population.append(individual)

#     # Thiết lập các tham số tối ưu hóa DE
#     de_params = {
#         "bounds": bounds,
#         "strategy": "best1bin",
#         "maxiter": min(iter_nr * 10, 500),  # Tăng số iteration
#         "popsize": 20,  # Tăng kích thước quần thể
#         "tol": 1e-6,  # Giảm ngưỡng hội tụ
#         "mutation": (0.5, 1.5),  # Tăng phạm vi mutation
#         "recombination": 0.9,  # Tăng tỷ lệ tái tổ hợp
#         "seed": 42,
#         "disp": True,  # Hiển thị tiến trình
#         "polish": True,  # Polish kết quả cuối cùng
#         "updating": "immediate",
#         "workers": 1,
#         "init": "random",  # Sử dụng khởi tạo ngẫu nhiên kết hợp với initial_population
#     }

#     try:
#         # Chạy thuật toán Differential Evolution
#         result = differential_evolution(objective_function, **de_params)

#         # Lấy kết quả tối ưu
#         optimal_params = result.x
#         optimal_phases = optimal_params[:M]

#         # Nếu có biên độ scaling factors
#         if len(optimal_params) > M:
#             optimal_amp_scales = optimal_params[M:]
#             optimal_amplitudes = W0_amplitude * optimal_amp_scales
#         else:
#             optimal_amplitudes = W0_amplitude

#         # Tạo vector beamforming tối ưu
#         W_opt = optimal_amplitudes * np.exp(1j * optimal_phases)
#         W_opt = W_opt.reshape(-1, 1)  # Trả về dạng (M, 1)

#         # Tính pattern tối ưu
#         PM_opt = np.abs(W_opt.conj().T @ V_pattern).flatten()

#         print(f"\nDE Optimization Results:")
#         print(f"  Iterations: {result.nit}")
#         print(f"  Final error: {result.fun:.6f}")
#         print(f"  Success: {result.success}")
#         print(f"  Message: {result.message}")
#         print(
#             f"  Pattern match error at alpha: {np.mean(np.abs(PM_opt[alpha] - PdM[alpha])):.6f}"
#         )

#     except Exception as e:
#         print(f"DE optimization failed: {e}")
#         print("Using initial W0 as fallback")
#         W_opt = W0.reshape(-1, 1)

#     return W_opt

# ==============================================
