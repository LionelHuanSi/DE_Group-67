import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# --- 1. Objective Function: Rastrigin ---
def rastrigin(X):
    # X shape: (population_size, dimension)
    A = 10
    n = X.shape[1]
    # Apply formula: A*n + sum(x^2 - A*cos(2*pi*x))
    f = A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
    return f

# --- 2. Helper Functions ---
def ensure_bounds(V, bounds):
    # Clip vectors to stay within search space
    min_b, max_b = bounds[0][0], bounds[0][1]
    return np.clip(V, min_b, max_b)

# --- 3. The 6 DE Variants ---

class DE_Variants:
    def __init__(self, obj_func, bounds, dim, pop_size=50, max_iter=500):
        self.obj_func = obj_func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.min_b, self.max_b = bounds[0]

    def _init_pop(self):
        return np.random.uniform(self.min_b, self.max_b, (self.pop_size, self.dim))

    # --- Variant 1-4: Classic Strategies ---
    def classic_de(self, strategy='rand/1/bin', F=0.5, CR=0.9):
        X = self._init_pop()
        fitness = self.obj_func(X)
        best_idx = np.argmin(fitness)
        best_hist = [fitness[best_idx]]
        
        for g in range(self.max_iter):
            V = np.empty_like(X)
            
            # Pre-generate random indices for efficiency
            # r1 != r2 != r3 != i
            # (Simplified loop for clarity)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r = np.random.choice(idxs, 5, replace=False) # Pick enough for all strategies
                
                r1, r2, r3, r4, r5 = X[r[0]], X[r[1]], X[r[2]], X[r[3]], X[r[4]]
                X_best = X[best_idx]
                X_i = X[i]

                # Mutation Formulas
                if strategy == 'rand/1/bin':
                    v_vec = r1 + F * (r2 - r3)
                elif strategy == 'best/1/bin':
                    v_vec = X_best + F * (r1 - r2)
                elif strategy == 'current-to-best/1/bin':
                    # Image formula: Xi + K(Best-Xi) + F(r1-r2). Assuming K=F
                    v_vec = X_i + F * (X_best - X_i) + F * (r1 - r2)
                elif strategy == 'rand/2/bin':
                    v_vec = r1 + F * (r2 - r3) + F * (r4 - r5)
                
                V[i] = ensure_bounds(v_vec, self.bounds)

            # Crossover (Binomial)
            cross_mask = np.random.rand(self.pop_size, self.dim) < CR
            # Ensure at least one dimension changes
            j_rand = np.random.randint(0, self.dim, size=self.pop_size)
            for i in range(self.pop_size):
                cross_mask[i, j_rand[i]] = True
            
            U = np.where(cross_mask, V, X)
            
            # Selection
            new_fitness = self.obj_func(U)
            improved = new_fitness < fitness
            X[improved] = U[improved]
            fitness[improved] = new_fitness[improved]
            
            # Update Global Best
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_hist[-1]:
                best_hist.append(fitness[curr_best_idx])
            else:
                best_hist.append(best_hist[-1])
                best_idx = curr_best_idx # Ensure best_idx tracks current best

        return best_hist

    # --- Variant 5: JADE (Simplified) ---
    def jade(self, p=0.05, c=0.1):
        # p: Top p% segment, c: learning rate
        X = self._init_pop()
        fitness = self.obj_func(X)
        best_hist = [np.min(fitness)]
        
        mu_cr = 0.5
        mu_f = 0.5
        archive = [] # Optional archive of inferior solutions (simplified list)
        
        for g in range(self.max_iter):
            # Sort population to find p-best
            sorted_idx = np.argsort(fitness)
            top_p_count = max(1, int(self.pop_size * p))
            p_best_indices = sorted_idx[:top_p_count]
            
            # Generate adaptive parameters
            SF = [] # Successful F
            SCR = [] # Successful CR
            
            U = np.empty_like(X)
            
            # Generate CR and F for whole population
            CR_i = np.random.normal(mu_cr, 0.1, self.pop_size)
            CR_i = np.clip(CR_i, 0, 1)
            # Cauchy for F is tricky in numpy, approximate or use scipy
            F_i = stats.cauchy.rvs(loc=mu_f, scale=0.1, size=self.pop_size)
            F_i = np.clip(F_i, 0.1, 1.0) # Standard clamping

            for i in range(self.pop_size):
                # Mutation: current-to-pbest/1
                # Vi = Xi + Fi(Xpbest - Xi) + Fi(Xr1 - Xr2)
                # Note: Xr2 is usually from (Pop U Archive), here simplified to Pop for brevity
                
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r = np.random.choice(idxs, 2, replace=False)
                xr1, xr2 = X[r[0]], X[r[1]]
                
                # Pick random p-best
                xpbest = X[np.random.choice(p_best_indices)]
                
                v_vec = X[i] + F_i[i] * (xpbest - X[i]) + F_i[i] * (xr1 - xr2)
                v_vec = ensure_bounds(v_vec, self.bounds)
                
                # Crossover
                j_rand = np.random.randint(0, self.dim)
                mask = np.random.rand(self.dim) < CR_i[i]
                mask[j_rand] = True
                u_vec = np.where(mask, v_vec, X[i])
                U[i] = u_vec

            # Selection
            new_fitness = self.obj_func(U)
            improved = new_fitness < fitness
            
            # Record successes for parameter update
            SF.extend(F_i[improved])
            SCR.extend(CR_i[improved])
            
            # Update Population
            # (In full JADE, added rejected to Archive)
            X[improved] = U[improved]
            fitness[improved] = new_fitness[improved]
            
            # Update Means (Lehmer Mean)
            if len(SCR) > 0:
                mu_cr = (1-c)*mu_cr + c*np.mean(SCR)
            if len(SF) > 0:
                mean_pow2 = np.mean(np.array(SF)**2)
                mean_val = np.mean(SF)
                lehmer = mean_pow2 / mean_val if mean_val > 0 else 0
                mu_f = (1-c)*mu_f + c*lehmer
            
            best_hist.append(np.min(fitness))
            
        return best_hist

    # --- Variant 6: SHADE (Simplified) ---
    def shade(self, H=100):
        # H: History size
        X = self._init_pop()
        fitness = self.obj_func(X)
        best_hist = [np.min(fitness)]
        
        # Memory contents
        M_cr = np.ones(H) * 0.5
        M_f = np.ones(H) * 0.5
        k_idx = 0
        
        for g in range(self.max_iter):
            sorted_idx = np.argsort(fitness)
            # p-best selection (SHADE uses variable p, usually random between 2/NP and 0.2)
            p_val = np.random.uniform(2/self.pop_size, 0.2)
            top_p_count = max(1, int(self.pop_size * p_val))
            p_best_indices = sorted_idx[:top_p_count]
            
            SF = []
            SCR = []
            diff_fitness = []
            
            U = np.empty_like(X)
            
            # Pick memory indices for each individual
            r_idx = np.random.randint(0, H, self.pop_size)
            CR_i = np.random.normal(M_cr[r_idx], 0.1)
            CR_i = np.clip(CR_i, 0, 1)
            F_i = stats.cauchy.rvs(loc=M_f[r_idx], scale=0.1)
            F_i = np.clip(F_i, 0.1, 1.0)
            
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r = np.random.choice(idxs, 2, replace=False)
                xr1, xr2 = X[r[0]], X[r[1]]
                
                xpbest = X[np.random.choice(p_best_indices)]
                
                # Mutation: current-to-pbest/1
                v_vec = X[i] + F_i[i] * (xpbest - X[i]) + F_i[i] * (xr1 - xr2)
                v_vec = ensure_bounds(v_vec, self.bounds)
                
                j_rand = np.random.randint(0, self.dim)
                mask = np.random.rand(self.dim) < CR_i[i]
                mask[j_rand] = True
                u_vec = np.where(mask, v_vec, X[i])
                U[i] = u_vec
            
            new_fitness = self.obj_func(U)
            improved = new_fitness < fitness
            
            # Archive update and History Update logic would go here
            if np.any(improved):
                SF.extend(F_i[improved])
                SCR.extend(CR_i[improved])
                diff_fitness.extend(np.abs(fitness[improved] - new_fitness[improved]))
                
                # Update History Memory (Weighted Lehmer Mean)
                # Simplified update for one memory slot k
                if len(SCR) > 0:
                    w = np.array(diff_fitness) / np.sum(diff_fitness)
                    m_scr = np.sum(w * np.array(SCR))
                    m_sf = np.sum(w * (np.array(SF)**2)) / np.sum(w * np.array(SF))
                    
                    M_cr[k_idx] = m_scr
                    M_f[k_idx] = m_sf
                    k_idx = (k_idx + 1) % H

            X[improved] = U[improved]
            fitness[improved] = new_fitness[improved]
            best_hist.append(np.min(fitness))
            
        return best_hist

# --- 4. Main Execution ---
# if __name__ == "__main__":
#     DIM = 10
#     POP = 70
#     ITER = 1000
#     BOUNDS = [(-5.12, 5.12)] * DIM
    
#     de = DE_Variants(rastrigin, BOUNDS, DIM, POP, ITER)
    
#     print("Running simulations...")
#     results = {}
#     results['DE/rand/1'] = de.classic_de('rand/1/bin')
#     results['DE/best/1'] = de.classic_de('best/1/bin')
#     results['DE/curr-to-best/1'] = de.classic_de('current-to-best/1/bin')
#     results['DE/rand/2'] = de.classic_de('rand/2/bin')
#     results['JADE'] = de.jade()
#     results['SHADE'] = de.shade()
    
#     # --- 5. Plotting ---
#     # --- 5. Plotting (Linear Scale - Thang đo thường) ---
#     plt.figure(figsize=(10, 6))
#     for name, history in results.items():
#         # BỎ np.log10, vẽ trực tiếp giá trị history
#         # history là danh sách giá trị Fitness tốt nhất qua từng thế hệ
#         plt.plot(np.array(history), label=name, linewidth=2)
        
#     plt.title(f'Comparison of DE Variants on Rastrigin (D={DIM}) - Linear Scale')
#     plt.xlabel('Generations')
#     plt.ylabel('Best Fitness Value (Cost)') # Đổi tên trục Y
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
    
#     # Mẹo: Giới hạn trục Y để nhìn rõ vùng gần 0 hơn nếu các thuật toán khác quá tệ
#     # plt.ylim(-1, 50) # Bỏ comment dòng này nếu bạn muốn zoom vào vùng giá trị thấp
    
#     plt.show()


if __name__ == "__main__":
    # --- 1. Cấu hình tham số (Đã chốt) ---
    DIM = 10
    POP = 70
    ITER = 1000
    BOUNDS = [(-5.12, 5.12)] * DIM
    RUNS = 10  # Số lần chạy độc lập để lấy trung bình
    
    # Khởi tạo đối tượng DE (Class DE_Variants phải được định nghĩa ở trên)
    de = DE_Variants(rastrigin, BOUNDS, DIM, POP, ITER)
    
    print(f"Đang chạy mô phỏng {RUNS} lần cho mỗi thuật toán...")
    
    # Tạo dictionary chứa list rỗng để lưu kết quả của 10 lần chạy
    all_results = {
        'DE/rand/1': [],
        'DE/best/1': [],
        'DE/curr-to-best/1': [],
        'DE/rand/2': [],
        'JADE': [],
        'SHADE': []
    }

    # --- 2. Vòng lặp thực nghiệm (Chạy 10 lần) ---
    for r in range(RUNS):
        print(f"  -> Lần chạy thứ {r+1}/{RUNS}...")
        
        # Lưu kết quả từng lần vào danh sách
        all_results['DE/rand/1'].append(de.classic_de('rand/1/bin'))
        all_results['DE/best/1'].append(de.classic_de('best/1/bin'))
        all_results['DE/curr-to-best/1'].append(de.classic_de('current-to-best/1/bin'))
        all_results['DE/rand/2'].append(de.classic_de('rand/2/bin'))
        all_results['JADE'].append(de.jade())
        all_results['SHADE'].append(de.shade())

    # --- 3. Vẽ biểu đồ (Linear Scale - Trung bình cộng) ---
    plt.figure(figsize=(10, 6))
    
    for name, list_of_histories in all_results.items():
        # Chuyển list các lần chạy thành mảng numpy 2D
        data_matrix = np.array(list_of_histories)
        
        # Tính trung bình dọc theo trục dọc (axis=0) để ra đường trung bình
        avg_history = np.mean(data_matrix, axis=0)
        
        # Vẽ đường trung bình này
        plt.plot(avg_history, label=name, linewidth=2)
        
    plt.title(f'Average Convergence over {RUNS} Runs on Rastrigin (D={DIM}, POP={POP})')
    plt.xlabel('Generations')
    plt.ylabel('Average Best Fitness Value (Cost)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Mẹo: Giới hạn trục Y để nhìn rõ vùng hội tụ nếu cần
    # plt.ylim(-0.5, 50) 
    
    plt.tight_layout()
    plt.show()