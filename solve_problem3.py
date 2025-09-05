import numpy as np
from scipy.optimize import minimize
import pandas as pd
import time
import os

# ==========================================================
# 1. 全局常量和仿真引擎 (保持不变)
# ==========================================================
T_FAKE = np.array([0.0, 0.0, 0.0])
TARGET_POINT = np.array([0.0, 200.0, 0.0])
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SCALAR = 300.0
P_U1_0 = np.array([17800.0, 0.0, 1800.0])
G = 9.8
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0
SIMULATION_DT = 0.01 # 使用高精度步长进行最终优化

DIR_M1 = (T_FAKE - P_M1_0) / np.linalg.norm(T_FAKE - P_M1_0)
V_M1 = V_M1_SCALAR * DIR_M1

def calculate_point_to_segment_distance(p, a, b):
    # ... (代码与之前完全相同)
    ap = p - a
    ab = b - a
    ab_squared_norm = np.dot(ab, ab)
    if ab_squared_norm == 0.0: return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab_squared_norm
    if t < 0.0: closest_point = a
    elif t > 1.0: closest_point = b
    else: closest_point = a + t * ab
    return np.linalg.norm(p - closest_point)

def objective_function(params):
    # ... (代码与之前完全相同)
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
        is_obscured_this_step = False
        
        for event in detonation_events:
            if event['time'] <= current_time <= event['time'] + T_SMOKE_EFFECTIVE:
                time_since_detonation = current_time - event['time']
                p_smoke_center = event['pos'] - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_detonation])
                if calculate_point_to_segment_distance(p_smoke_center, p_m1_current, TARGET_POINT) <= R_SMOKE:
                    is_obscured_this_step = True
                    break
        
        if is_obscured_this_step:
            total_obscured_time += SIMULATION_DT
        current_time += SIMULATION_DT
        
    return -total_obscured_time

def run_detailed_simulation_and_save(params, filename="result1.xlsx"):
    # ... (代码与之前完全相同)
    v_u, theta_u, t_drop1, delta_t_det1, dt_12, delta_t_det2, dt_23, delta_t_det3 = params

    total_duration = -objective_function(params)
    
    t_drop2 = t_drop1 + dt_12
    t_drop3 = t_drop2 + dt_23
    drop_times = [t_drop1, t_drop2, t_drop3]
    det_delays = [delta_t_det1, delta_t_det2, delta_t_det3]
    
    V_U1 = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
    
    data_rows = []
    for i in range(3):
        t_drop = drop_times[i]
        delta_t_det = det_delays[i]
        p_drop = P_U1_0 + V_U1 * t_drop
        t_detonation = t_drop + delta_t_det
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
        p_detonation = p_drop + V_U1 * delta_t_det + gravity_effect
        data_rows.append({
            "p_drop_x": p_drop[0], "p_drop_y": p_drop[1], "p_drop_z": p_drop[2],
            "p_det_x": p_detonation[0], "p_det_y": p_detonation[1], "p_det_z": p_detonation[2],
        })

    df = pd.DataFrame({
        '无人机运动方向': [np.rad2deg(theta_u)] * 3,
        '无人机运动速度(m/s)': [v_u] * 3,
        '烟幕干扰弹编号': [1, 2, 3],
        '烟幕干扰弹投放点的x坐标(m)': [r['p_drop_x'] for r in data_rows],
        '烟幕干扰弹投放点的y坐标(m)': [r['p_drop_y'] for r in data_rows],
        '烟幕干扰弹投放点的z坐标(m)': [r['p_drop_z'] for r in data_rows],
        '烟幕干扰弹起爆点的x坐标(m)': [r['p_det_x'] for r in data_rows],
        '烟幕干扰弹起爆点的y坐标(m)': [r['p_det_y'] for r in data_rows],
        '烟幕干扰弹起爆点的z坐标(m)': [r['p_det_z'] for r in data_rows],
        '有效干扰时长(s)': [total_duration] * 3
    })
    
    for col in ['无人机运动方向', '无人机运动速度(m/s)', '有效干扰时长(s)']:
        df.loc[1:, col] = np.nan

    df.to_excel(filename, index=False, engine='openpyxl')
    print(f"\n结果已成功保存到: {os.path.abspath(filename)}")
    return total_duration

# ==========================================================
# 3. 主优化流程 (混合策略：热启动 + 局部精炼)
# ==========================================================
if __name__ == "__main__":
    print("="*60)
    print("问题3：执行混合优化策略 (热启动 + 局部精炼)...")
    print("="*60)
    
    # 1. 定义您提供的、已验证的高质量参数作为初始点 (x0)
    v_u_given = 108.22
    theta_u_deg_given = 8.273
    t_drop1_given = 0.001
    delta_t_det1_given = 0.140
    t_drop2_given = 1.004
    delta_t_det2_given = 0.005
    t_drop3_given = 9.900
    delta_t_det3_given = 0.027

    # 转换为模型所需的8维向量格式
    theta_u_rad = np.deg2rad(theta_u_deg_given)
    dt_12 = t_drop2_given - t_drop1_given
    dt_23 = t_drop3_given - t_drop2_given
    
    initial_guess = [
        v_u_given, theta_u_rad, t_drop1_given, delta_t_det1_given,
        dt_12, delta_t_det2_given, dt_23, delta_t_det3_given
    ]
    
    # 定义参数边界 (bounds)，这对于 L-BFGS-B 是必需的
    bounds = [
        (70, 140),          # v_u
        (0, 2 * np.pi),     # theta_u
        (0.001, 50),        # t_drop1 (下限设为0.001以避免奇异)
        (0.001, 20),        # delta_t_det1
        (1.0, 20),          # dt_12
        (0.001, 20),        # delta_t_det2
        (1.0, 20),          # dt_23
        (0.001, 20)         # delta_t_det3
    ]

    print(f"初始点遮蔽时长 (验证值): 6.3500 秒")
    print("从此初始点开始进行局部精炼优化...")
    start_time = time.time()
    
    # 2. 调用局部优化器 `minimize`
    result = minimize(
        fun=objective_function,
        x0=initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'eps': 1e-4} # disp显示过程, eps是梯度近似的步长
    )
    
    end_time = time.time()
    
    print("-" * 60)
    print(f"局部精炼完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"函数总评估次数: {result.nfev}")
    print(f"优化器退出信息: {result.message}")
    print("-" * 60)
    
    best_params = result.x
    max_duration = -result.fun

    print(f"优化后的最大有效遮蔽时长: {max_duration:.4f} 秒")
    print("-" * 60)
    
    # 3. 保存最终的最优结果
    run_detailed_simulation_and_save(best_params, filename="result1_optimized.xlsx")
    print("="*60)
