import numpy as np
from typing import List, Tuple

class VirtualRobot:
    def __init__(self, 
                 x: float = 5.0, 
                 y: float = 5.0, 
                 phi: float = 0,
                 m: float = 20.0,
                 n: float = 10.0,
                 boundary_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 laser_configs: List[Tuple[Tuple[float, float], float]] = None):
        """
        完全符合你要求的初始化参数：
        - x, y, phi: 初始位姿
        - boundary_lines: 环境边界线 [((x1,y1), (x2,y2)), ...]
        - laser_configs: [((相对距离,相对角度), 激光朝向), ...]
        """
        # 机器人状态
        self.x = x
        self.y = y
        self.phi = phi
        
        # 环境配置（按你给的默认值）
        self.boundary_lines = boundary_lines or [
            ((0, 0), (m, 0)),
            ((m, 0), (m, n)),
            ((m, n), (0, n)),
            ((0, n), (0, 0))
        ]
        
        # 激光配置（按你给的默认值）
        self.laser_configs = laser_configs or [
            ((1.0, np.pi/4), np.pi/4),   # 左前传感器
            ((0.5, np.pi), np.pi),        # 正前传感器
            ((1.0, -np.pi/4), -np.pi/4)  # 右前传感器
        ]
    
    def update_pose(self, x: float, y: float, phi: float):
        """更新机器人位姿（供滑动条回调使用）"""
        self.x = x
        self.y = y
        self.phi = phi
    
    def scan(self, 
            noise_type: str = 'gaussian', 
            noise_scale: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行带噪声的激光扫描
        
        Args:
            noise_type: 噪声类型 ('gaussian' 或 'uniform')
            noise_scale: 噪声强度（高斯噪声为标准差，均匀噪声为半区间宽度）
            
        Returns:
            (带噪声的距离数组, 全局角度数组) - 角度保持精确值
        """
        distances = []
        global_angles = []
        
        for (rel_r, rel_angle), laser_angle in self.laser_configs:
            # 计算激光头全局位置
            laser_x = self.x + rel_r * np.cos(self.phi + rel_angle)
            laser_y = self.y + rel_r * np.sin(self.phi + rel_angle)
            
            # 计算激光全局角度（保持精确值）
            current_angle = self.phi + laser_angle
            
            # 精确碰撞检测
            true_dist = self._raycast(laser_x, laser_y, current_angle)
            
            # 添加噪声
            if noise_type == 'gaussian':
                noisy_dist = true_dist + np.random.normal(0, noise_scale)
            elif noise_type == 'uniform':
                noisy_dist = true_dist + np.random.uniform(-noise_scale, noise_scale)
            else:
                noisy_dist = true_dist  # 无噪声
            
            # 确保距离非负
            noisy_dist = max(0.0, noisy_dist)
            
            distances.append(noisy_dist)
            global_angles.append(current_angle)
        
        # 测试一下——强行让一个激光头的数据为0

        #distances[0] = 0.0  # 模拟一个激光头坏掉

        return np.array(distances), np.array(global_angles)

    def _raycast(self, x: float, y: float, angle: float) -> float:
        """
        精确碰撞检测（返回最近碰撞距离）
        实现方案：
        1. 遍历所有边界线
        2. 计算射线与线段的数学交点
        3. 返回最小有效距离
        """
        max_dist = 20.0
        closest_dist = max_dist
        
        for (p1, p2) in self.boundary_lines:
            # 计算交点参数
            t = self._calculate_intersection_param(
                ray_start=(x, y),
                ray_angle=angle,
                segment_p1=p1,
                segment_p2=p2
            )
            
            # 更新最近有效距离
            if 0 < t < closest_dist:
                closest_dist = t
        
        return closest_dist

    def _calculate_intersection_param(self, 
                                    ray_start: Tuple[float, float], 
                                    ray_angle: float,
                                    segment_p1: Tuple[float, float], 
                                    segment_p2: Tuple[float, float]) -> float:
        """
        计算射线与线段的交点参数t（数学精确解）
        返回: 交点距离参数t（无效时返回-1）
        """
        x0, y0 = ray_start
        dx_ray = np.cos(ray_angle)
        dy_ray = np.sin(ray_angle)
        
        x1, y1 = segment_p1
        x2, y2 = segment_p2
        dx_seg = x2 - x1
        dy_seg = y2 - y1
        
        denominator = dx_ray * dy_seg - dy_ray * dx_seg
        
        # 处理平行情况
        if abs(denominator) < 1e-6:
            return -1.0
        
        # 计算参数
        t = ((x1 - x0) * dy_seg - (y1 - y0) * dx_seg) / denominator
        u = ((x0 - x1) * dy_ray - (y0 - y1) * dx_ray) / -denominator
        
        # 有效性检查
        if t >= 0 and 0 <= u <= 1:
            return t
        return -1.0