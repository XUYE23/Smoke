

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os
import multiprocessing

# ==========================================================
# 1. 全局常量和预计算 (保持不变)
# ==========================================================
T_FAKE = np.array([0.0, 0.0, 0.0])
TARGET_POINT = np.array([0.0, 200.0, 0.0])
G = 9.8
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0

MISSILE_INITIAL_POS = {'M1': np.array([20000.0, 0.0, 2000.0]), 'M2': np.array([19000.0, 600.0, 2100.0]), 'M3': np.array([18000.0, -600.0, 1900.0])}
V_M_SCALAR = 300.0
MISSILE_NAMES = ['M1', 'M2', 'M3']
MISSILE_VELOCITIES = {n: V_M_SCALAR * ((T_FAKE - p) / np.linalg.norm(T_FAKE - p)) for n, p in MISSILE_INITIAL_POS.items()}

UAV_INITIAL_POS = {'FY1': np.array([17800.0, 0.0, 1800.0]), 'FY2': np.array([12000.0, 1400.0, 1400.0]), 'FY3': np.array([6000.0, -3000.0, 700.0]), 'FY4': np.array([11000.0, 2000.0, 1800.0]), 'FY5': np.array([13000.0, -2000.0, 1300.0])}
UAV_NAMES = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
NUM_UAVS = 5
BOMBS_PER_UAV = 3
TOTAL_BOMBS = NUM_UAVS * BOMBS_PER_UAV

# ==========================================================
# 2. 仿真引擎与并行工作函数 - [重大更新]
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

def objective_function(params, config):
    uav_params = params[:NUM_UAVS * 2].reshape(NUM_UAVS, 2)
    bomb_timing_params = params[NUM_UAVS * 2:].reshape(TOTAL_BOMBS, 2)
    detonation_events = []
    for i in range(TOTAL_BOMBS):
        uav_idx = i // BOMBS_PER_UAV
        v_u, theta_u = uav_params[uav_idx]
        t_drop, delta_t_det = bomb_timing_params[i]
        if t_drop > 60: continue
        p_uav_0 = UAV_INITIAL_POS[UAV_NAMES[uav_idx]]
        V_UAV = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
        p_drop = p_uav_0 + V_UAV * t_drop
        t_detonation = t_drop + delta_t_det
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
        p_detonation = p_drop + V_UAV * delta_t_det + gravity_effect
        detonation_events.append({'time': t_detonation, 'pos': p_detonation})
    if not detonation_events: return 0.0
    
    t_start = min(e['time'] for e in detonation_events)
    t_end = max(e['time'] + T_SMOKE_EFFECTIVE for e in detonation_events)
    
    # [重大更新] 初始化每个导弹独立的遮蔽时长
    total_obscured_times = {name: 0.0 for name in MISSILE_NAMES}
    
    current_time = t_start
    while current_time <= t_end:
        missile_positions = {n: MISSILE_INITIAL_POS[n] + V * current_time for n, V in MISSILE_VELOCITIES.items()}
        
        active_smokes = []
        for event in detonation_events:
            if event['time'] <= current_time <= event['time'] + T_SMOKE_EFFECTIVE:
                time_since_det = current_time - event['time']
                active_smokes.append(event['pos'] - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_det]))

        if not active_smokes:
            current_time += config['SIMULATION_DT']
            continue
            
        # [重大更新] 分别为每个导弹计算是否被遮蔽，并独立累加时间
        for m_name in MISSILE_NAMES:
            is_m_obscured = False
            for p_smoke in active_smokes:
                if calculate_point_to_segment_distance(p_smoke, missile_positions[m_name], TARGET_POINT) <= R_SMOKE:
                    is_m_obscured = True
                    break # 只要有一个烟幕弹遮蔽该导弹即可
            
            if is_m_obscured:
                total_obscured_times[m_name] += config['SIMULATION_DT']
                
        current_time += config['SIMULATION_DT']
        
    # [重大更新] 返回遮蔽时长的等权和
    total_duration_sum = sum(total_obscured_times.values())
    return -total_duration_sum

def init_worker(config_from_main):
    global CONFIG
    CONFIG = config_from_main
def parallel_worker(params):
    return objective_function(params, CONFIG)

# ==========================================================
# 3. 经典PSO优化器 (保持不变)
# ==========================================================
class ParticleSwarmOptimizer:
    def __init__(self, config):
        self.config = config
        self.bounds = np.array(config['CONTINUOUS_BOUNDS'])
        self.num_particles = config['NUM_PARTICLES']
        self.max_iter = config['MAX_ITER']
        self.w, self.c1, self.c2 = config['W'], config['C1'], config['C2']
        self.num_dims = len(self.bounds)
        self.particles = np.random.rand(self.num_particles, self.num_dims) * \
            (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.velocities = np.zeros((self.num_particles, self.num_dims))
        self.pbest_positions = self.particles.copy()
        self.pbest_fitness = np.full(self.num_particles, np.inf)
        self.gbest_position = None; self.gbest_fitness = np.inf
    def optimize(self):
        with multiprocessing.Pool(initializer=init_worker, initargs=(self.config,)) as pool:
            pbar = tqdm(range(self.max_iter), desc="Direct PSO 优化进度", unit="iter")
            for i in pbar:
                fitness_values = pool.map(parallel_worker, self.particles)
                for j in range(self.num_particles):
                    if fitness_values[j] < self.pbest_fitness[j]:
                        self.pbest_fitness[j] = fitness_values[j]
                        self.pbest_positions[j] = self.particles[j].copy()
                min_idx = np.argmin(self.pbest_fitness)
                if self.pbest_fitness[min_idx] < self.gbest_fitness:
                    self.gbest_fitness = self.pbest_fitness[min_idx]
                    self.gbest_position = self.pbest_positions[min_idx].copy()
                pbar.set_postfix({"最佳时长(和)": f"{-self.gbest_fitness:.4f}s"})
                r1, r2 = np.random.rand(2, self.num_particles, self.num_dims)
                self.velocities = (self.w * self.velocities +
                                   self.c1 * r1 * (self.pbest_positions - self.particles) +
                                   self.c2 * r2 * (self.gbest_position - self.particles))
                self.particles += self.velocities
                self.particles = np.clip(self.particles, self.bounds[:, 0], self.bounds[:, 1])
        return self.gbest_position, self.gbest_fitness

# ==========================================================
# 4. 结果打印函数 (更新打印文本)
# ==========================================================
def print_detailed_results(params, duration_sum):
    print("\n" + "="*60)
    print("找到的最优策略详情 (目标: 时长加和):")
    print(f"  - 最大'遮蔽时长加和': {duration_sum:.4f} 秒")
    print("="*60)
    uav_params = params[:NUM_UAVS * 2].reshape(NUM_UAVS, 2)
    bomb_params = params[NUM_UAVS * 2:].reshape(TOTAL_BOMBS, 2)
    for i in range(NUM_UAVS):
        v_u, theta_u = uav_params[i]
        print(f"--- 无人机 {UAV_NAMES[i]} ---")
        print(f"  - 飞行速度: {v_u:.4f} m/s")
        print(f"  - 飞行方向: {np.rad2deg(theta_u):.4f} 度")
        for j in range(BOMBS_PER_UAV):
            bomb_idx = i * BOMBS_PER_UAV + j
            t_drop, delta_t_det = bomb_params[bomb_idx]
            status = "启用" if t_drop <= 60 else "禁用"
            print(f"    - 烟幕弹 #{j+1}: [{status}] t_drop={t_drop:.4f}s, Δt_det={delta_t_det:.4f}s")
    print("="*60)

# ==========================================================
# 5. 主执行流程 (保持不变)
# ==========================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    CONFIG = {
        "NUM_PARTICLES": 300, "MAX_ITER": 100,
        "W": 0.7, "C1": 2.0, "C2": 2.0,
        "SIMULATION_DT": 0.1,
    }
    
    # --- 您精心调整的参数边界 (保持不变) ---
    bounds_v_u1, bounds_theta_u1 = (130,134), (5/180*np.pi, 7/180*np.pi)
    bounds_v_u2, bounds_theta_u2 = (130, 140), (12/9*np.pi, 13/9*np.pi)
    bounds_v_u3, bounds_theta_u3 = (110, 120), (4/9*np.pi, 2/3*np.pi)
    bounds_v_u4, bounds_theta_u4 = (120, 130), (21/18*np.pi, 23/18*np.pi)
    bounds_v_u5, bounds_theta_u5 = (115, 125), (11/18*np.pi, 13/18*np.pi)
    
    max_dt_fy1 = np.sqrt(2 * UAV_INITIAL_POS['FY1'][2] / G) - 0.5
    max_dt_fy2 = np.sqrt(2 * UAV_INITIAL_POS['FY2'][2] / G) - 0.5
    max_dt_fy3 = np.sqrt(2 * UAV_INITIAL_POS['FY3'][2] / G) - 0.5
    max_dt_fy4 = np.sqrt(2 * UAV_INITIAL_POS['FY4'][2] / G) - 0.5
    max_dt_fy5 = np.sqrt(2 * UAV_INITIAL_POS['FY5'][2] / G) - 0.5

    t_drop_max = 65 
    
    bounds_t_drop1_1, bounds_dd_1_1 = (0.001, 0.008), (0, 0.01)
    bounds_t_drop1_2, bounds_dd_1_2 = (1.005, 1.01), (0, 0.005)
    bounds_t_drop1_3, bounds_dd_1_3 = (13.3, 13.7), (16, 21)
    bounds_t_drop2_1, bounds_dd_2_1 = (38, 43), (12, max_dt_fy2)
    bounds_t_drop2_2, bounds_dd_2_2 = (2.5, 4.5), (6, 8)
    bounds_t_drop2_3, bounds_dd_2_3 = (18, 21), (6, max_dt_fy2)
    bounds_t_drop3_1, bounds_dd_3_1 = (40, t_drop_max), (3, max_dt_fy3)
    bounds_t_drop3_2, bounds_dd_3_2 = (20, 24), (10, max_dt_fy3)
    bounds_t_drop3_3, bounds_dd_3_3 = (18, 24), (4, 8)
    bounds_t_drop4_1, bounds_dd_4_1 = (2, 6), (6, max_dt_fy4)
    bounds_t_drop4_2, bounds_dd_4_2 = (3, 8), (8, max_dt_fy4)
    bounds_t_drop4_3, bounds_dd_4_3 = (8, 15), (11, max_dt_fy4)
    bounds_t_drop5_1, bounds_dd_5_1 = (45, t_drop_max), (13, max_dt_fy5)
    bounds_t_drop5_2, bounds_dd_5_2 = (10, 12), (12, max_dt_fy5)
    bounds_t_drop5_3, bounds_dd_5_3 = (18, 21), (9, max_dt_fy5)

    CONFIG['CONTINUOUS_BOUNDS'] = [
        bounds_v_u1, bounds_theta_u1, bounds_v_u2, bounds_theta_u2,
        bounds_v_u3, bounds_theta_u3, bounds_v_u4, bounds_theta_u4,
        bounds_v_u5, bounds_theta_u5,
        bounds_t_drop1_1, bounds_dd_1_1, bounds_t_drop1_2, bounds_dd_1_2, bounds_t_drop1_3, bounds_dd_1_3,
        bounds_t_drop2_1, bounds_dd_2_1, bounds_t_drop2_2, bounds_dd_2_2, bounds_t_drop2_3, bounds_dd_2_3,
        bounds_t_drop3_1, bounds_dd_3_1, bounds_t_drop3_2, bounds_dd_3_2, bounds_t_drop3_3, bounds_dd_3_3,
        bounds_t_drop4_1, bounds_dd_4_1, bounds_t_drop4_2, bounds_dd_4_2, bounds_t_drop4_3, bounds_dd_4_3,
        bounds_t_drop5_1, bounds_dd_5_1, bounds_t_drop5_2, bounds_dd_5_2, bounds_t_drop5_3, bounds_dd_5_3,
    ]
    
    print("="*60)
    print("问题5：开始使用直接PSO算法求解 (目标: 时长加和)...")
    print(f"配置: {CONFIG['NUM_PARTICLES']}粒子, {CONFIG['MAX_ITER']}迭代")
    print(f"搜索空间维度: {len(CONFIG['CONTINUOUS_BOUNDS'])}维")
    print("="*60)
    
    start_time = time.time()
    
    pso = ParticleSwarmOptimizer(config=CONFIG)
    best_params, best_fitness = pso.optimize()
    
    end_time = time.time()
    
    total_seconds = end_time - start_time
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    print(f"\n优化过程完成，总耗时: {minutes}分 {seconds:.2f}秒")
    print("-" * 60)
    
    max_duration_sum = -best_fitness
    print_detailed_results(best_params, max_duration_sum)