

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
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SCALAR = 300.0
G = 9.8
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0
SIMULATION_DT = 0.1

UAV_INITIAL_POS = {
    'FY1': np.array([17800.0, 0.0, 1800.0]),
    'FY2': np.array([12000.0, 1400.0, 1400.0]),
    'FY3': np.array([6000.0, -3000.0, 700.0])
}
UAV_NAMES = ['FY1', 'FY2', 'FY3']

DIR_M1 = (T_FAKE - P_M1_0) / np.linalg.norm(T_FAKE - P_M1_0)
V_M1 = V_M1_SCALAR * DIR_M1

# ==========================================================
# 2. 仿真引擎与并行工作函数 (保持不变)
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

def objective_function(params):
    params_per_uav = np.array(params).reshape(3, 4)
    detonation_events = []
    for i in range(3):
        v_u, theta_u, t_drop, delta_t_det = params_per_uav[i]
        uav_name = UAV_NAMES[i]
        p_uav_0 = UAV_INITIAL_POS[uav_name]
        V_UAV = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
        p_drop = p_uav_0 + V_UAV * t_drop
        t_detonation = t_drop + delta_t_det
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
        p_detonation = p_drop + V_UAV * delta_t_det + gravity_effect
        detonation_events.append({'time': t_detonation, 'pos': p_detonation})
    t_start = min(e['time'] for e in detonation_events)
    t_end = max(e['time'] + T_SMOKE_EFFECTIVE for e in detonation_events)
    total_obscured_time = 0.0
    current_time = t_start
    while current_time <= t_end:
        p_m1_current = P_M1_0 + V_M1 * current_time
        is_obscured = False
        for event in detonation_events:
            if event['time'] <= current_time <= event['time'] + T_SMOKE_EFFECTIVE:
                time_since_det = current_time - event['time']
                p_smoke = event['pos'] - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_det])
                if calculate_point_to_segment_distance(p_smoke, p_m1_current, TARGET_POINT) <= R_SMOKE:
                    is_obscured = True
                    break
        if is_obscured:
            total_obscured_time += SIMULATION_DT
        current_time += SIMULATION_DT
    return -total_obscured_time

def parallel_worker(params):
    return objective_function(params)

# ==========================================================
# 3. 粒子群优化器 (PSO) (保持不变)
# ==========================================================
class ParticleSwarmOptimizer:
    def __init__(self, bounds, num_particles, max_iter, w, c1, c2, max_stagnation, initial_guess=None):
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.max_stagnation = max_stagnation
        self.num_dims = len(self.bounds)
        self.particles = np.random.rand(self.num_particles, self.num_dims) * \
            (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        if initial_guess is not None:
            # 确保初始猜测在边界内
            initial_guess = np.clip(initial_guess, self.bounds[:, 0], self.bounds[:, 1])
            self.particles[0] = initial_guess
            print("成功将启发式初始解植入为初始粒子之一。")
        self.velocities = np.zeros((self.num_particles, self.num_dims))
        self.pbest_positions = self.particles.copy()
        self.pbest_fitness = np.full(self.num_particles, np.inf)
        self.gbest_position = None
        self.gbest_fitness = np.inf
    def optimize(self):
        stagnation_counter = 0; last_gbest_fitness = np.inf
        with multiprocessing.Pool() as pool:
            pbar = tqdm(range(self.max_iter), desc="PSO 优化进度")
            for i in pbar:
                fitness_values = list(pool.map(parallel_worker, self.particles))
                for j in range(self.num_particles):
                    if fitness_values[j] < self.pbest_fitness[j]:
                        self.pbest_fitness[j] = fitness_values[j]
                        self.pbest_positions[j] = self.particles[j].copy()
                min_fitness_idx = np.argmin(self.pbest_fitness)
                if self.pbest_fitness[min_fitness_idx] < self.gbest_fitness:
                    self.gbest_fitness = self.pbest_fitness[min_fitness_idx]
                    self.gbest_position = self.pbest_positions[min_fitness_idx].copy()
                pbar.set_postfix({"最佳时长": f"{-self.gbest_fitness:.4f}s"})
                if np.isclose(self.gbest_fitness, last_gbest_fitness):
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                if stagnation_counter >= self.max_stagnation:
                    print(f"\n算法已连续 {self.max_stagnation} 次迭代没有改善，提前终止。")
                    break
                last_gbest_fitness = self.gbest_fitness
                r1 = np.random.rand(self.num_particles, self.num_dims)
                r2 = np.random.rand(self.num_particles, self.num_dims)
                self.velocities = (self.w * self.velocities +
                                   self.c1 * r1 * (self.pbest_positions - self.particles) +
                                   self.c2 * r2 * (self.gbest_position - self.particles))
                self.particles += self.velocities
                self.particles = np.clip(self.particles, self.bounds[:, 0], self.bounds[:, 1])
        return self.gbest_position, self.gbest_fitness

# ==========================================================
# 4. 结果处理与保存 - [重大更新]
# ==========================================================
def run_detailed_simulation_and_save(params, filename="result2.xlsx"):
    # [关键更新] 将打印逻辑放在前面，并添加错误处理
    
    global SIMULATION_DT; SIMULATION_DT = 0.01
    total_duration = -objective_function(params)
    params_per_uav = np.array(params).reshape(3, 4)
    data_for_df = []
    
    print("-" * 60)
    print("最优策略详细参数:")
    print(f"  - 总有效遮蔽时长: {total_duration:.4f} 秒")
    print("-" * 60)

    for i in range(3):
        v_u, theta_u, t_drop, delta_t_det = params_per_uav[i]
        uav_name = UAV_NAMES[i]
        p_uav_0 = UAV_INITIAL_POS[uav_name]
        V_UAV = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
        p_drop = p_uav_0 + V_UAV * t_drop
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
        p_detonation = p_drop + V_UAV * delta_t_det + gravity_effect
        
        # 打印到控制台
        print(f"  无人机 {uav_name}:")
        print(f"    - 运动方向: {np.rad2deg(theta_u):.4f} 度")
        print(f"    - 运动速度: {v_u:.4f} m/s")
        print(f"    - 投放点 (x,y,z): ({p_drop[0]:.2f}, {p_drop[1]:.2f}, {p_drop[2]:.2f})")
        print(f"    - 起爆点 (x,y,z): ({p_detonation[0]:.2f}, {p_detonation[1]:.2f}, {p_detonation[2]:.2f})")

        # 准备写入Excel的数据
        data_for_df.append({
            '无人机编号': uav_name, '无人机运动方向': np.rad2deg(theta_u), '无人机运动速度(m/s)': v_u,
            '烟幕干扰弹投放点的x坐标(m)': p_drop[0], '烟幕干扰弹投放点的y坐标(m)': p_drop[1], '烟幕干扰弹投放点的z坐标(m)': p_drop[2],
            '烟幕干扰弹起爆点的x坐标(m)': p_detonation[0], '烟幕干扰弹起爆点的y坐标(m)': p_detonation[1], '烟幕干扰弹起爆点的z坐标(m)': p_detonation[2],
        })
    
    # 尝试写入Excel文件
    try:
        df = pd.DataFrame(data_for_df)
        df['有效干扰时长(s)'] = total_duration
        df.to_excel(filename, index=False, engine='openpyxl')
        print("-" * 60)
        print(f"结果已成功保存到: {os.path.abspath(filename)}")
    except PermissionError:
        print("-" * 60)
        print(f"!!! 文件写入失败: {filename} 可能正被其他程序占用。")
        print("请关闭Excel后重试。不过，所有结果已打印在上方。")
    except Exception as e:
        print("-" * 60)
        print(f"!!! 文件写入时发生未知错误: {e}")

    return total_duration


# ==========================================================
# 5. 主执行流程 (保持不变)
# ==========================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("="*60)
    print("问题4：开始使用PSO算法寻找3无人机协同最优策略...")
    print("="*60)
    
    # --- 您精心调整的参数边界 ---
    v_u1_bounds = (120, 140)
    theta_u1_bounds = (0, (1/18)*np.pi)
    t_drop1_bounds = (0.1, 2)
    delta_t_det1_bounds = (0.5, 1)
    v_u2_bounds = (130, 140)
    theta_u2_bounds = ((245/180)* np.pi, (255/180)* np.pi)
    t_drop2_bounds = (2.5, 4.5)
    delta_t_det2_bounds = (6.5, 8.3)
    v_u3_bounds = (130, 140)
    theta_u3_bounds = ((4/9)* np.pi, (7/9)* np.pi)
    t_drop3_bounds = (16, 24)
    delta_t_det3_bounds = (3, 8.5)
    bounds_fy1 = [v_u1_bounds, theta_u1_bounds, t_drop1_bounds, delta_t_det1_bounds]
    bounds_fy2 = [v_u2_bounds, theta_u2_bounds, t_drop2_bounds, delta_t_det2_bounds]
    bounds_fy3 = [v_u3_bounds, theta_u3_bounds, t_drop3_bounds, delta_t_det3_bounds]
    bounds = bounds_fy1 + bounds_fy2 + bounds_fy3
    print("已为所有12个参数加载独立的、可微调的边界。")

    # 构建启发式初始解
    q2_v_u, q2_theta_u_deg, q2_t_drop, q2_delta_t_det = 139.9905, 179.6725, 1.4034, 4.4810
    fy1_params = [q2_v_u, np.deg2rad(q2_theta_u_deg), q2_t_drop, q2_delta_t_det]
    bounds_for_random = bounds_fy2 + bounds_fy3
    random_params_fy2_fy3 = (np.random.rand(8) * 
                             (np.array(bounds_for_random)[:, 1] - np.array(bounds_for_random)[:, 0]) + 
                             np.array(bounds_for_random)[:, 0])
    initial_guess_particle = np.concatenate([fy1_params, random_params_fy2_fy3])
    
    start_time = time.time()
    
    pso = ParticleSwarmOptimizer(
        bounds=bounds,
        num_particles=300, max_iter=50,
        w=0.8, c1=2.0, c2=1.5,
        max_stagnation=30,
        initial_guess=initial_guess_particle
    )
    
    best_params, best_fitness = pso.optimize()
    
    end_time = time.time()
    
    print(f"\n优化过程完成，总耗时: {end_time - start_time:.2f} 秒")
    
    max_duration = -best_fitness
    print("-" * 60)
    print(f"找到的最大有效遮蔽时长: {max_duration:.4f} 秒")
    
    run_detailed_simulation_and_save(best_params, filename="result2.xlsx")
    print("="*60)