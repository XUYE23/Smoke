import numpy as np
from scipy.optimize import dual_annealing
import time

# ==========================================================
# 1. 仿真器模块 (从问题一适配而来)
#    这部分代码定义了物理世界和仿真过程
# ==========================================================

# 全局常量定义
T_FAKE = np.array([0.0, 0.0, 0.0])
TARGET_POINT = np.array([0.0, 200.0, 0.0]) # 目标简化为底面中心单点
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SCALAR = 300.0
P_U1_0 = np.array([17800.0, 0.0, 1800.0])
G = 9.8
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0
SIMULATION_DT = 0.01

# 预计算M1的速度向量，因为它在整个优化过程中不变
DIR_M1 = (T_FAKE - P_M1_0) / np.linalg.norm(T_FAKE - P_M1_0)
V_M1 = V_M1_SCALAR * DIR_M1

def calculate_point_to_segment_distance(p, a, b):
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
    """
    目标函数/仿真器: 计算给定策略下的遮蔽时长。
    params: [v_u, theta_u, t_drop, delta_t_det]
    返回: 负的遮蔽时长 (用于最小化)
    """
    v_u, theta_u, t_drop, delta_t_det = params

    # 1. 根据策略参数计算无人机速度向量
    V_U1 = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
    
    # 2. 计算关键事件
    p_drop = P_U1_0 + V_U1 * t_drop
    t_detonation = t_drop + delta_t_det
    gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
    p_detonation = p_drop + V_U1 * delta_t_det + gravity_effect
    
    # 3. 仿真
    t_smoke_start = t_detonation
    t_smoke_end = t_detonation + T_SMOKE_EFFECTIVE
    total_obscured_time = 0.0
    
    current_time = t_smoke_start
    while current_time <= t_smoke_end:
        p_m1_current = P_M1_0 + V_M1 * current_time
        time_since_detonation = current_time - t_detonation
        p_smoke_center_current = p_detonation - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_detonation])
        
        distance = calculate_point_to_segment_distance(p_smoke_center_current, p_m1_current, TARGET_POINT)
        if distance <= R_SMOKE:
            total_obscured_time += SIMULATION_DT
            
        current_time += SIMULATION_DT
        
    return -total_obscured_time

def run_simulation_with_details(params):
    """
    用于结果展示的详细仿真函数，返回一个包含所有信息的字典。
    """
    v_u, theta_u, t_drop, delta_t_det = params
    
    V_U1 = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
    p_drop = P_U1_0 + V_U1 * t_drop
    t_detonation = t_drop + delta_t_det
    gravity_effect = np.array([0.0, 0.0, -0.5 * G * delta_t_det**2])
    p_detonation = p_drop + V_U1 * delta_t_det + gravity_effect
    
    t_smoke_start = t_detonation
    t_smoke_end = t_detonation + T_SMOKE_EFFECTIVE
    
    total_obscured_time = 0.0
    first_obscured_time = -1.0
    last_obscured_time = -1.0
    
    current_time = t_smoke_start
    while current_time <= t_smoke_end:
        is_obscured_this_step = False
        p_m1_current = P_M1_0 + V_M1 * current_time
        time_since_detonation = current_time - t_detonation
        p_smoke_center_current = p_detonation - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_detonation])
        
        distance = calculate_point_to_segment_distance(p_smoke_center_current, p_m1_current, TARGET_POINT)
        if distance <= R_SMOKE:
            is_obscured_this_step = True
        
        if is_obscured_this_step:
            total_obscured_time += SIMULATION_DT
            if first_obscured_time < 0:
                first_obscured_time = current_time
            last_obscured_time = current_time
        
        current_time += SIMULATION_DT
        
    return {
        "params": params,
        "total_duration": total_obscured_time,
        "start_time": first_obscured_time,
        "end_time": last_obscured_time if last_obscured_time > 0 else -1.0,
    }


# ==========================================================
# 2. 优化器模块
#    这部分代码设置并运行模拟退火算法
# ==========================================================
if __name__ == "__main__":
    print("="*50)
    print("问题2：开始使用模拟退火算法寻找最优策略...")
    print("="*50)
    
    # 定义决策变量的边界 (v_u, theta_u, t_drop, delta_t_det)
    bounds = [
        (70, 140),                # 无人机速度 v_u (m/s)
        (0, 2 * np.pi),           # 无人机方向 theta_u (radians)
        (0.1, 65),                # 投放时间 t_drop (s)
        (0.1, 20)                 # 起爆延迟 delta_t_det (s)
    ]
    
    start_time = time.time()
    
    # 调用 dual_annealing 优化器
    # maxiter设置了一个合理的迭代次数，以在几分钟内完成
    result = dual_annealing(
        objective_function,
        bounds,
        maxiter=1000,  # 增加迭代次数以获得更好的结果，会增加运行时间
        seed=42        # 设置随机种子以保证结果可复现
    )
    
    end_time = time.time()
    
    print(f"优化过程完成，耗时: {end_time - start_time:.2f} 秒")
    print("-" * 50)
    
    # 提取并展示最优结果
    best_params = result.x
    max_duration = -result.fun
    
    # 使用最优参数运行一次详细仿真以获取过程信息
    detailed_results = run_simulation_with_details(best_params)
    
    v_u_opt, theta_u_opt, t_drop_opt, delta_t_det_opt = best_params
    
    print("="*50)
    print("问题2：最优策略及效果分析")
    print("="*50)
    print(f"找到的最大有效遮蔽时长: {max_duration:.4f} 秒")
    print("-" * 50)
    print("最优策略参数:")
    print(f"  - 无人机飞行速度: {v_u_opt:.4f} m/s")
    print(f"  - 无人机飞行方向: {np.rad2deg(theta_u_opt):.4f} 度")
    print(f"  - 烟幕投放时间:   {t_drop_opt:.4f} 秒")
    print(f"  - 烟幕起爆延迟:   {delta_t_det_opt:.4f} 秒")
    print("-" * 50)
    print("在该最优策略下，遮蔽过程详情:")
    if detailed_results["total_duration"] > 0:
        print(f"  - 遮蔽开始时间: {detailed_results['start_time']:.4f} 秒")
        print(f"  - 遮蔽结束时间: {detailed_results['end_time']:.4f} 秒")
    else:
        print("  - 未发生有效遮蔽。")
    print("="*50)