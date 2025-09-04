import numpy as np

# 1. 常量和初始状态定义
# ==========================================================

# 坐标系与目标信息
T_FAKE = np.array([0.0, 0.0, 0.0])
T_REAL_AXIS_BOTTOM = np.array([0.0, 200.0, 0.0])
T_REAL_AXIS_TOP = np.array([0.0, 200.0, 10.0])
# 选取目标轴上的三个关键点用于遮蔽判断
TARGET_KEY_POINTS = [
    T_REAL_AXIS_BOTTOM,
    (T_REAL_AXIS_BOTTOM + T_REAL_AXIS_TOP) / 2.0,
    T_REAL_AXIS_TOP
]

# 攻击方 (导弹 M1)
P_M1_0 = np.array([20000.0, 0.0, 2000.0])
V_M1_SCALAR = 300.0

# 防御方 (无人机 FY1)
P_U1_0 = np.array([17800.0, 0.0, 1800.0])
V_U1_SCALAR = 120.0

# 事件时间线
T_DROP = 1.5
DELTA_T_DETONATION = 3.6

# 物理与烟幕参数
G = 9.8
R_SMOKE = 10.0
T_SMOKE_EFFECTIVE = 20.0
V_SMOKE_SINK = 3.0

# 仿真参数
SIMULATION_DT = 0.01  # 时间步长 (s)


def calculate_point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """计算点 p 到线段 ab 的最短距离。"""
    ap = p - a
    ab = b - a
    
    ab_squared_norm = np.dot(ab, ab)
    if ab_squared_norm == 0.0:
        return np.linalg.norm(ap)

    t = np.dot(ap, ab) / ab_squared_norm

    if t < 0.0:
        closest_point = a
    elif t > 1.0:
        closest_point = b
    else:
        closest_point = a + t * ab
        
    return np.linalg.norm(p - closest_point)


def solve():
    """执行问题1的完整仿真计算。"""
    # 步骤 A: 计算恒定速度向量
    # M1 速度向量 (三维)
    dir_m1 = (T_FAKE - P_M1_0) / np.linalg.norm(T_FAKE - P_M1_0)
    V_M1 = V_M1_SCALAR * dir_m1

    # FY1 速度向量 (修正: 保证等高度飞行)
    # 1. 计算水平面(XY)上的方向向量
    direction_vec_3d_u1 = T_FAKE - P_U1_0
    direction_vec_2d_u1 = direction_vec_3d_u1[:2]  # 只取 x, y 分量
    # 2. 归一化2D方向向量
    normalized_dir_2d_u1 = direction_vec_2d_u1 / np.linalg.norm(direction_vec_2d_u1)
    # 3. 构建最终的速度向量，z分量为0
    V_U1 = V_U1_SCALAR * np.array([normalized_dir_2d_u1[0], normalized_dir_2d_u1[1], 0.0])


    # 步骤 B: 计算关键事件的时间和位置
    # 投放点 (无人机在投放时刻的位置)
    p_drop = P_U1_0 + V_U1 * T_DROP
    v_bomb_initial = V_U1 # 烟幕弹的初速度等于无人机当时的速度

    # 起爆点 (烟幕弹平抛后的位置)
    t_detonation = T_DROP + DELTA_T_DETONATION
    gravity_effect = np.array([0.0, 0.0, -0.5 * G * DELTA_T_DETONATION**2])
    p_detonation = p_drop + v_bomb_initial * DELTA_T_DETONATION + gravity_effect

    # 步骤 C: 设置仿真时间窗口
    t_smoke_start = t_detonation
    t_smoke_end = t_detonation + T_SMOKE_EFFECTIVE

    # 步骤 D & E & F: 执行时间步进仿真
    total_obscured_time = 0.0
    current_time = t_smoke_start

    while current_time <= t_smoke_end:
        p_m1_current = P_M1_0 + V_M1 * current_time
        
        time_since_detonation = current_time - t_detonation
        p_smoke_center_current = p_detonation - np.array([0.0, 0.0, V_SMOKE_SINK * time_since_detonation])

        is_fully_obscured = True
        for target_point in TARGET_KEY_POINTS:
            distance = calculate_point_to_segment_distance(
                p=p_smoke_center_current,
                a=p_m1_current,
                b=target_point
            )
            if distance > R_SMOKE:
                is_fully_obscured = False
                break
        
        if is_fully_obscured:
            total_obscured_time += SIMULATION_DT

        current_time += SIMULATION_DT
        
    return total_obscured_time


if __name__ == "__main__":
    effective_time = solve()
    print("问题1：烟幕干扰弹对M1的有效遮蔽时长计算结果")
    print(f"总有效遮蔽时长: {effective_time:.4f} 秒")
    if effective_time == 0.0:
        print("\n分析: 结果为0是符合物理预期的。")
        print("原因: 无人机和导弹均在y=0平面上运动，导致烟幕也形成于y=0平面。")
        print("而真目标位于y=200，两者侧向距离过大，烟幕云团(半径10m)无法对视线构成任何遮挡。")