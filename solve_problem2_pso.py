import numpy as np
from tqdm import tqdm
import time
import functools

# ==========================================================
# 1. 全局常量和预计算
# ==========================================================
T_FAKE = np.array([0.0, 0.0, 0.0])
TARGET_POINT = np.array([0.0, 200.0, 0.0])
P_M1_0 = np.array([18000.0, -600.0, 1900.0])
V_M1_SCALAR = 300.0
G = 9.8
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0
SIMULATION_DT = 0.05

UAV_INITIAL_POS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0]),
    'FY4': np.array([11000.0, 2000.0, 1800.0]),
    'FY5': np.array([13000.0, -2000.0, 1300.0])
}

DIR_M1 = (T_FAKE - P_M1_0) / np.linalg.norm(T_FAKE - P_M1_0)
V_M1 = V_M1_SCALAR * DIR_M1

# ==========================================================
# 2. 仿真引擎
# ==========================================================
def calculate_point_to_segment_distance(p, a, b):
    ap = p - a; ab = b - a
    ab_sq_norm = np.dot(ab, ab)
    if ab_sq_norm == 0.0: return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab_sq_norm
    if t < 0.0: closest = a
    elif t > 1.0: closest = b
    else: closest = a + t * ab
    return np.linalg.norm(p - closest)

def objective_function(params, uav_name):
    v_u, theta_u, t_drop, delta_t_det = params
    p_uav_0 = UAV_INITIAL_POS[uav_name]
    
    V_UAV = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
    
    p_drop = p_uav_0 + V_UAV * t_drop
    t_detonation = t_drop + delta_t_det
    gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
    p_detonation = p_drop + V_UAV * delta_t_det + gravity_effect
    
    t_smoke_start = t_detonation
    t_smoke_end = t_detonation + T_SMOKE_EFFECTIVE
    total_obscured_time = 0.0
    
    current_time = t_smoke_start
    while current_time <= t_smoke_end:
        p_m1_current = P_M1_0 + V_M1 * current_time
        time_since_det = current_time - t_detonation
        p_smoke = p_detonation - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_det])
        if calculate_point_to_segment_distance(p_smoke, p_m1_current, TARGET_POINT) <= R_SMOKE:
            total_obscured_time += SIMULATION_DT
        current_time += SIMULATION_DT
        
    return -total_obscured_time

# ==========================================================
# 3. 粒子群优化器 (PSO) - 简化串行版
# ==========================================================
class ParticleSwarmOptimizer:
    def __init__(self, func, bounds, num_particles, max_iter, w, c1, c2):
        self.func = func
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.num_dims = len(self.bounds)
        
        self.particles = np.random.rand(self.num_particles, self.num_dims) * \
            (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.velocities = np.zeros((self.num_particles, self.num_dims))
        
        self.pbest_positions = self.particles.copy()
        self.pbest_fitness = np.full(self.num_particles, np.inf)
        
        self.gbest_position = None
        self.gbest_fitness = np.inf

    def optimize(self):
        pbar = tqdm(range(self.max_iter), desc="PSO 优化进度")
        for i in pbar:
            for j in range(self.num_particles):
                fitness = self.func(self.particles[j])
                if fitness < self.pbest_fitness[j]:
                    self.pbest_fitness[j] = fitness
                    self.pbest_positions[j] = self.particles[j].copy()
            
            min_fitness_idx = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[min_fitness_idx] < self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[min_fitness_idx]
                self.gbest_position = self.pbest_positions[min_fitness_idx].copy()
            
            pbar.set_postfix({"最佳时长": f"{-self.gbest_fitness:.4f}s"})
                
            r1 = np.random.rand(self.num_particles, self.num_dims)
            r2 = np.random.rand(self.num_particles, self.num_dims)
            
            self.velocities = (self.w * self.velocities +
                               self.c1 * r1 * (self.pbest_positions - self.particles) +
                               self.c2 * r2 * (self.gbest_position - self.particles))
            
            self.particles += self.velocities
            self.particles = np.clip(self.particles, self.bounds[:, 0], self.bounds[:, 1])
        
        return self.gbest_position, self.gbest_fitness

# ==========================================================
# 4. 主执行流程
# ==========================================================
if __name__ == "__main__":
    
    # ------------------------------------------------------------------
    # [用户配置区] 修改这里来选择要优化的无人机
    # 可选值: 'FY1', 'FY2', 'FY3'
    # ------------------------------------------------------------------
    UAV_TO_OPTIMIZE = 'FY5'
    # ------------------------------------------------------------------

    print("="*60)
    print(f"问题2 PSO版：开始为无人机 {UAV_TO_OPTIMIZE} 寻找最优策略...")
    print("="*60)
    
    # 根据选择的无人机，动态设置参数边界
    uav_height = UAV_INITIAL_POS[UAV_TO_OPTIMIZE][2]
    max_delta_t_det = np.sqrt(2 * uav_height / G) - 0.5 # 留出0.5s余量
    
    bounds = [
        (110, 125),                 # 速度 v_u (m/s)
        (11/18*np.pi, 13/18*np.pi),            # 方向 theta_u (radians)
        (10, 17),                 # 投放时间 t_drop (s)
        (0, max_delta_t_det)     # 起爆延迟 delta_t_det (s)
    ]
    
    print(f"为 {UAV_TO_OPTIMIZE} (高度 {uav_height}m) 设置的边界:")
    print(f"  - 速度 (v_u): {bounds[0]}")
    print(f"  - 方向 (θ_u): (0, 2π)")
    print(f"  - 投放时间 (t_drop): {bounds[2]}")
    print(f"  - 起爆延迟 (Δt_det): {bounds[3]}")
    print("-" * 60)
    
    start_time = time.time()
    
    # 使用 functools.partial 将 uav_name 绑定到目标函数
    bound_objective_func = functools.partial(objective_function, uav_name=UAV_TO_OPTIMIZE)
    
    pso = ParticleSwarmOptimizer(
        func=bound_objective_func,
        bounds=bounds,
        num_particles=100, # 4维问题不需要太多粒子
        max_iter=50,
        w=0.8, c1=2, c2=1.5
    )
    
    best_params, best_fitness = pso.optimize()
    
    end_time = time.time()
    
    print(f"\n优化过程完成，总耗时: {end_time - start_time:.2f} 秒")
    print("-" * 60)
    
    max_duration = -best_fitness
    v_u_opt, theta_u_opt, t_drop_opt, delta_t_det_opt = best_params

    print(f"为 {UAV_TO_OPTIMIZE} 找到的最优策略:")
    print(f"  - 最大有效遮蔽时长: {max_duration:.4f} 秒")
    print("  - 最优策略参数:")
    print(f"    - 无人机飞行速度: {v_u_opt:.4f} m/s")
    print(f"    - 无人机飞行方向: {np.rad2deg(theta_u_opt):.4f} 度")
    print(f"    - 烟幕投放时间:   {t_drop_opt:.4f} 秒")
    print(f"    - 烟幕起爆延迟:   {delta_t_det_opt:.4f} 秒")
    print("="*60)