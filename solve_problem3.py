import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import os

# ==========================================================
# 1. 全局常量和预计算
# ==========================================================
T_FAKE = np.array([0.0, 0.0, 0.0])
TARGET_POINT = np.array([0.0, 200.0, 0.0])
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SCALAR = 300.0
P_U1_0 = np.array([17800.0, 0.0, 1800.0]) # FY1的初始位置
G = 9.8
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0
SIMULATION_DT = 0.1 # 优化时使用稍大的步长

DIR_M1 = (T_FAKE - P_M1_0) / np.linalg.norm(T_FAKE - P_M1_0)
V_M1 = V_M1_SCALAR * DIR_M1

# ==========================================================
# 2. 仿真引擎 (与问题三模拟退火版相同)
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
    v_u, theta_u, t_drop1, delta_t_det1, dt_12, delta_t_det2, dt_23, delta_t_det3 = params
    
    t_drop2 = t_drop1 + dt_12
    t_drop3 = t_drop2 + dt_23
    drop_times = [t_drop1, t_drop2, t_drop3]
    det_delays = [delta_t_det1, delta_t_det2, delta_t_det3]
    
    V_U1 = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
    
    detonation_events = []
    for i in range(3):
        t_drop = drop_times[i]
        delta_t_det = det_delays[i]
        p_drop = P_U1_0 + V_U1 * t_drop
        t_detonation = t_drop + delta_t_det
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
        p_detonation = p_drop + V_U1 * delta_t_det + gravity_effect
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

# ==========================================================
# 3. 粒子群优化器 (PSO) - 串行版
# ==========================================================
class ParticleSwarmOptimizer:
    def __init__(self, func, bounds, num_particles, max_iter, w, c1, c2):
        self.func = func; self.bounds = np.array(bounds); self.num_particles = num_particles
        self.max_iter = max_iter; self.w, self.c1, self.c2 = w, c1, c2
        self.num_dims = len(self.bounds)
        self.particles = np.random.rand(self.num_particles, self.num_dims) * \
            (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.velocities = np.zeros((self.num_particles, self.num_dims))
        self.pbest_positions = self.particles.copy()
        self.pbest_fitness = np.full(self.num_particles, np.inf)
        self.gbest_position = None; self.gbest_fitness = np.inf
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
# 4. 结果处理与保存
# ==========================================================
def run_detailed_simulation_and_save(params, filename="result1.xlsx"):
    global SIMULATION_DT; SIMULATION_DT = 0.01
    total_duration = -objective_function(params)
    v_u, theta_u, t_drop1, delta_t_det1, dt_12, delta_t_det2, dt_23, delta_t_det3 = params
    
    t_drop2 = t_drop1 + dt_12
    t_drop3 = t_drop2 + dt_23
    drop_times = [t_drop1, t_drop2, t_drop3]
    det_delays = [delta_t_det1, delta_t_det2, delta_t_det3]
    V_U1 = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
    
    data_for_df = []
    
    print("-" * 60)
    print("最优策略详细参数:")
    print(f"  - 总有效遮蔽时长: {total_duration:.4f} 秒")
    print(f"  - 无人机运动方向: {np.rad2deg(theta_u):.4f} 度")
    print(f"  - 无人机运动速度: {v_u:.4f} m/s")
    print("-" * 60)
    
    for i in range(3):
        t_drop = drop_times[i]
        delta_t_det = det_delays[i]
        p_drop = P_U1_0 + V_U1 * t_drop
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
        p_detonation = p_drop + V_U1 * delta_t_det + gravity_effect
        
        print(f"  烟幕弹 {i+1}:")
        print(f"    - 投放时间 (绝对): {t_drop:.4f} s")
        print(f"    - 起爆延迟: {delta_t_det:.4f} s")
        print(f"    - 投放点 (x,y,z): ({p_drop[0]:.2f}, {p_drop[1]:.2f}, {p_drop[2]:.2f})")
        print(f"    - 起爆点 (x,y,z): ({p_detonation[0]:.2f}, {p_detonation[1]:.2f}, {p_detonation[2]:.2f})")
        
        data_for_df.append({
            '烟幕干扰弹编号': i + 1,
            '烟幕干扰弹投放点的x坐标(m)': p_drop[0], '烟幕干扰弹投放点的y坐标(m)': p_drop[1], '烟幕干扰弹投放点的z坐标(m)': p_drop[2],
            '烟幕干扰弹起爆点的x坐标(m)': p_detonation[0], '烟幕干扰弹起爆点的y坐标(m)': p_detonation[1], '烟幕干扰弹起爆点的z坐标(m)': p_detonation[2],
        })
        
    try:
        df = pd.DataFrame(data_for_df)
        df['无人机运动方向'] = np.rad2deg(theta_u)
        df['无人机运动速度(m/s)'] = v_u
        df['有效干扰时长(s)'] = total_duration
        
        # 调整列顺序以匹配模板
        cols = ['无人机运动方向', '无人机运动速度(m/s)', '烟幕干扰弹编号',
                '烟幕干扰弹投放点的x坐标(m)', '烟幕干扰弹投放点的y坐标(m)', '烟幕干扰弹投放点的z坐标(m)',
                '烟幕干扰弹起爆点的x坐标(m)', '烟幕干扰弹起爆点的y坐标(m)', '烟幕干扰弹起爆点的z坐标(m)',
                '有效干扰时长(s)']
        df = df[cols]
        
        # 将重复值清空
        for col in ['无人机运动方向', '无人机运动速度(m/s)', '有效干扰时长(s)']:
            df.loc[1:, col] = np.nan
        
        df.to_excel(filename, index=False, engine='openpyxl')
        print("-" * 60)
        print(f"结果已成功保存到: {os.path.abspath(filename)}")
    except Exception as e:
        print("-" * 60)
        print(f"!!! 文件写入失败: {e}")

# ==========================================================
# 5. 主执行流程
# ==========================================================
if __name__ == "__main__":
    print("="*60)
    print("问题3 PSO版：开始为FY1寻找3枚弹协同最优策略...")
    print("="*60)

    # ------------------------------------------------------------------
    # [参数微调区域] 在这里显式定义所有8个参数的边界
    # ------------------------------------------------------------------
    
    # 全局策略参数
    v_u_bounds = (100, 140)              # 无人机速度 (m/s)
    theta_u_bounds = (0, (1/18)* np.pi)     # 无人机方向 (弧度)
    
    # 基于FY1高度 (1800m) 的物理约束
    # max_delta_t_det = sqrt(2 * 1800 / 9.8) ≈ 19.17s
    max_delta_t_det_fy1 = 19.0
    
    # 弹 1 的参数边界
    t_drop1_bounds = (0, 2)           # 弹1 投放时间 (s)
    delta_t_det1_bounds = (0, max_delta_t_det_fy1) # 弹1 起爆延迟 (s)
    
    # 弹 2 的参数边界
    dt_12_bounds = (1.0, 2)             # 弹2 与弹1的投放间隔 (s), 最小为1s
    delta_t_det2_bounds = (0, max_delta_t_det_fy1) # 弹2 起爆延迟 (s)
    
    # 弹 3 的参数边界
    dt_23_bounds = (1.0, 15)             # 弹3 与弹2的投放间隔 (s), 最小为1s
    delta_t_det3_bounds = (0, max_delta_t_det_fy1) # 弹3 起爆延迟 (s)
    
    # 将所有边界按顺序组合成最终的8维边界列表
    bounds = [
        v_u_bounds, theta_u_bounds,
        t_drop1_bounds, delta_t_det1_bounds,
        dt_12_bounds, delta_t_det2_bounds,
        dt_23_bounds, delta_t_det3_bounds
    ]
    print("已为所有8个参数加载独立的、可微调的边界。")
    # ------------------------------------------------------------------
    
    start_time = time.time()
    
    pso = ParticleSwarmOptimizer(
        func=objective_function,
        bounds=bounds,
        num_particles=250, # 8维问题，粒子数可以适当增加
        max_iter=50,      # 增加迭代次数以获得更好结果
        w=0.8, c1=2.0, c2=1.5
    )
    
    best_params, best_fitness = pso.optimize()
    
    end_time = time.time()
    
    print(f"\n优化过程完成，总耗时: {end_time - start_time:.2f} 秒")
    
    max_duration = -best_fitness
    
    run_detailed_simulation_and_save(best_params, filename="result1.xlsx")
    print("="*60)