import numpy as np
from scipy.optimize import dual_annealing
import pandas as pd
import time
import os
from tqdm import tqdm  # 导入tqdm库

# ==========================================================
# 1. 全局常量和预计算 (无变化)
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
SIMULATION_DT = 0.05
T_MISSILE_IMPACT = np.linalg.norm(P_M1_0) / V_M1_SCALAR

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

# ==========================================================
# 2. 多弹协同仿真器 (目标函数) (无变化)
# ==========================================================
def objective_function(params):
    v_u, theta_u, t_drop_1, dt_det_1, dt_drop_12, dt_det_2, dt_drop_23, dt_det_3 = params

    t_drops = [t_drop_1, t_drop_1 + dt_drop_12, t_drop_1 + dt_drop_12 + dt_drop_23]
    dt_dets = [dt_det_1, dt_det_2, dt_det_3]
    V_U1 = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])

    events = []
    for i in range(3):
        t_drop = t_drops[i]
        dt_det = dt_dets[i]
        p_drop = P_U1_0 + V_U1 * t_drop
        t_detonation = t_drop + dt_det
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * dt_det**2])
        p_detonation = p_drop + V_U1 * dt_det + gravity_effect
        events.append({
            "t_start": t_detonation,
            "t_end": t_detonation + T_SMOKE_EFFECTIVE,
            "p_det": p_detonation
        })
    
    total_obscured_time = 0.0
    for current_time in np.arange(0, T_MISSILE_IMPACT, SIMULATION_DT):
        is_obscured_this_step = False
        p_m1_current = P_M1_0 + V_M1 * current_time
        
        for event in events:
            if event["t_start"] <= current_time <= event["t_end"]:
                time_since_det = current_time - event["t_start"]
                p_smoke_center = event["p_det"] - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_det])
                distance = calculate_point_to_segment_distance(p_smoke_center, p_m1_current, TARGET_POINT)
                if distance <= R_SMOKE:
                    is_obscured_this_step = True
                    break
        
        if is_obscured_this_step:
            total_obscured_time += SIMULATION_DT
        
    return -total_obscured_time

# ==========================================================
# 3. 结果生成模块 (无变化)
# ==========================================================
def generate_output_file(best_params, total_duration):
    v_u, theta_u, t_drop_1, dt_det_1, dt_drop_12, dt_det_2, dt_drop_23, dt_det_3 = best_params
    t_drops = [t_drop_1, t_drop_1 + dt_drop_12, t_drop_1 + dt_drop_12 + dt_drop_23]
    dt_dets = [dt_det_1, dt_det_2, dt_det_3]
    V_U1 = v_u * np.array([np.cos(theta_u), np.sin(theta_u), 0.0])
    
    data = []
    for i in range(3):
        p_drop = P_U1_0 + V_U1 * t_drops[i]
        t_det = t_drops[i] + dt_dets[i]
        gravity_effect = np.array([0.0, 0.0, -0.5 * G * dt_dets[i]**2])
        p_det = p_drop + V_U1 * dt_dets[i] + gravity_effect
        data.append({
            "无人机运动方向": np.rad2deg(theta_u) % 360,
            "无人机运动速度(m/s)": v_u,
            "烟幕干扰弹编号": i + 1,
            "烟幕干扰弹投放点的x坐标(m)": p_drop[0],
            "烟幕干扰弹投放点的y坐标(m)": p_drop[1],
            "烟幕干扰弹投放点的z坐标(m)": p_drop[2],
            "烟幕干扰弹起爆点的x坐标(m)": p_det[0],
            "烟幕干扰弹起爆点的y坐标(m)": p_det[1],
            "烟幕干扰弹起爆点的z坐标(m)": p_det[2],
            "有效干扰时长(s)": total_duration
        })
        
    df = pd.DataFrame(data)
    try:
        # 使用 __file__ 获取当前脚本路径，使其更健壮
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "result1.xlsx")
        df.to_excel(file_path, index=False, float_format="%.4f")
        print(f"\n结果已成功覆写到 {file_path} 文件中。")
    except Exception as e:
        print(f"\n[错误] 写入Excel文件失败: {e}")

# ==========================================================
# 4. 优化器主程序 (集成tqdm)
# ==========================================================
if __name__ == "__main__":
    print("="*60)
    print("问题3：开始使用模拟退火算法寻找3枚烟幕弹的最优协同策略...")
    print("="*60)
    
    bounds = [
        (70, 140), (0, 2 * np.pi),
        (0.1, T_MISSILE_IMPACT - 25), (0.1, 20),
        (1.0, 15), (0.1, 20),
        (1.0, 15), (0.1, 20)
    ]
    
    # [新增] 创建tqdm进度条实例
    # dual_annealing的迭代次数由maxiter决定，但其内部函数评估次数更多。
    # 这里我们监控的是模拟退火的"step"
    max_iterations = 2000
    pbar = tqdm(total=max_iterations, desc="模拟退火优化中")
    
    # [新增] 定义callback函数
    def progress_callback(x, f, context):
        """在每次迭代后更新进度条和描述"""
        pbar.update(1)
        pbar.set_description(f"模拟退火优化中 (当前最优时长: {-f:.2f}s)")

    start_time = time.time()
    
    result = dual_annealing(
        objective_function,
        bounds,
        maxiter=max_iterations,
        seed=42,
        callback=progress_callback  # [新增] 注册callback
    )
    
    pbar.close() # [新增] 结束后关闭进度条
    end_time = time.time()
    
    print(f"\n优化过程完成，总耗时: {(end_time - start_time)/60:.2f} 分钟")
    
    best_params = result.x
    max_duration = -result.fun
    
    print("="*60)
    print("最优策略已找到！")
    print(f"最大有效干扰时长: {max_duration:.4f} 秒")
    print("="*60)
    
    generate_output_file(best_params, max_duration)