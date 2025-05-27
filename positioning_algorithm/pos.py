import numpy as np
from typing import Tuple, Optional

def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-π, π] 范围内"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def calculate_robot_pose(
    r: np.ndarray,       # 激光头到机器人的距离 (3,)
    delta: np.ndarray,   # 激光头方向角 (3,)
    theta: np.ndarray,   # 激光束朝向角 (3,)
    t: np.ndarray,       # 激光测距值 (3,)
    m: float,            # 边界长度
    n: float             # 边界宽度
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    计算机器人位置P和朝向φ
    返回: (P, φ) 或 (None, None)
    """
    # 1. 计算碰撞点向量v_i的长度和角度α_i
    v_norm = np.zeros(3)
    alpha = np.zeros(3)
    for i in range(3):
        # 余弦定理计算|v_i|
        angle_diff = np.pi - (delta[i] - theta[i])
        v_norm[i] = np.sqrt(r[i]**2 + t[i]**2 - 2*r[i]*t[i]*np.cos(angle_diff))
        # 正弦定理计算α_i
        alpha[i] = np.arcsin(np.sin(angle_diff) * t[i] / v_norm[i])
    
    # 2. 计算碰撞点坐标（在机器人坐标系）
    v = np.zeros((3, 2))
    for i in range(3):
        v[i] = v_norm[i] * np.array([np.cos(alpha[i]), np.sin(alpha[i])])
    
    # 3. 计算边向量l_i和夹角β_i
    l = np.zeros((3, 2))
    beta = np.zeros(3)
    for i in range(3):
        l[i] = v[(i+1)%3] - v[i]
        beta[i] = np.arccos(np.dot(v[i], l[i]) / (v_norm[i] * np.linalg.norm(l[i])))
    
    # 4. 情况1：检查是否有边在边界上
    for i in range(3):
        # 检查是否平行于x轴（高=n）
        if np.isclose(abs(l[i,1]), 0) and not np.isclose(l[i,0], 0):
            # 计算高度是否等于n
            height = abs(v[(i+2)%3,1] - v[i,1])  # 第三个点到这条边的距离
            if np.isclose(height, n, atol=1e-3):
                # 确定边界位置（y=0或y=n）
                boundary_y = 0 if min(v[i,1], v[(i+1)%3,1]) < n/2 else n
                # 计算机器人位置和朝向
                gamma_i = 0 if l[i,0] > 0 else np.pi
                Cap = np.array([v[i,0], boundary_y]) if l[i,0] > 0 else np.array([v[(i+1)%3,0], boundary_y])
                P = Cap - v[i] if l[i,0] > 0 else Cap - v[(i+1)%3]
                phi = normalize_angle(gamma_i - beta[i] - alpha[i])
                return P, phi
        
        # 检查是否平行于y轴（高=m）
        elif np.isclose(abs(l[i,0]), 0) and not np.isclose(l[i,1], 0):
            # 计算宽度是否等于m
            width = abs(v[(i+2)%3,0] - v[i,0])  # 第三个点到这条边的距离
            if np.isclose(width, m, atol=1e-3):
                # 确定边界位置（x=0或x=m）
                boundary_x = 0 if min(v[i,0], v[(i+1)%3,0]) < m/2 else m
                # 计算机器人位置和朝向
                gamma_i = np.pi/2 if l[i,1] > 0 else -np.pi/2
                Cap = np.array([boundary_x, v[i,1]]) if l[i,1] > 0 else np.array([boundary_x, v[(i+1)%3,1]])
                P = Cap - v[i] if l[i,1] > 0 else Cap - v[(i+1)%3]
                phi = normalize_angle(gamma_i - beta[i] - alpha[i])
                return P, phi
    
    # 5. 情况2：三个顶点在三条边界上
    for i in range(3):
        delta_alpha_beta = (alpha[(i-1)%3] - alpha[i]) + (beta[(i-1)%3] - beta[i])
        
        # 计算A和δ
        A = np.sqrt(
            l[i,0]**2 + l[i,1]**2 + 
            l[(i-1)%3,0]**2 + l[(i-1)%3,1]**2 + 
            2 * np.linalg.norm(l[i]) * np.linalg.norm(l[(i-1)%3]) * np.cos(delta_alpha_beta)
        )
        
        if A < 1e-10 or abs(m) > A:
            continue  # 无解
        
        delta_angle = np.arctan2(
            np.linalg.norm(l[(i-1)%3]) * np.sin(delta_alpha_beta),
            np.linalg.norm(l[i]) + np.linalg.norm(l[(i-1)%3]) * np.cos(delta_alpha_beta)
        )
        
        gamma_i = np.arccos(m / A) - delta_angle
        
        # 计算Cap_i
        angle = np.pi - (gamma_i + delta_alpha_beta)
        Cap_i = v[i] + np.array([
            np.linalg.norm(l[(i-1)%3]) * np.cos(angle),
            np.linalg.norm(l[(i-1)%3]) * np.sin(angle)
        ])
        
        P = Cap_i - v[i]
        phi = normalize_angle(gamma_i - beta[i] - alpha[i])
        
        # 验证解是否在边界内
        if 0 <= Cap_i[0] <= m and 0 <= Cap_i[1] <= n:
            return P, phi
    
    return None, None  # 无解

# 示例用法
if __name__ == "__main__":
    # 示例数据
    r = np.array([1.0, 1.0, 1.0])          # 激光头到机器人距离
    delta = np.array([0.0, 2*np.pi/3, 4*np.pi/3])  # 激光头方向
    theta = np.array([0.1, 0.1, 0.1])      # 激光束朝向
    t = np.array([5.0, 5.0, 5.0])          # 激光测距值
    m, n = 10.0, 8.0                       # 边界尺寸
    
    P, phi = calculate_robot_pose(r, delta, theta, t, m, n)
    
    if P is not None:
        print(f"机器人位置: ({P[0]:.2f}, {P[1]:.2f})")
        print(f"机器人朝向: {np.degrees(phi):.2f}°")
    else:
        print("未找到有效解")