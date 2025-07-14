import numpy as np
from typing import List, Tuple

class VirtualRobot:
    def __init__(self, 
                 x: float = 5.0, 
                 y: float = 5.0, 
                 phi: float = np.pi/4,
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
            ((0, 0), (20, 0)),
            ((20, 0), (20, 10)),
            ((20, 10), (0, 10)),
            ((0, 10), (0, 0))
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
    
    def scan(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回: (激光距离数组, 激光全局角度数组)
          - 距离数组: [d1, d2, d3]
          - 角度数组: [θ1, θ2, θ3]（全局坐标系下的弧度）
        """
        distances = []
        global_angle = []
        for (rel_r, rel_angle), laser_angle in self.laser_configs:
            # 计算激光发射点全局坐标
            laser_x = self.x + rel_r * np.cos(self.phi + rel_angle)
            laser_y = self.y + rel_r * np.sin(self.phi + rel_angle)
            
            # 计算激光全局角度
            current_angle = self.phi + laser_angle
            
            # 碰撞检测
            dist = self._raycast(laser_x, laser_y, current_angle)
            
            distances.append(dist)
            global_angle.append(current_angle)
            
        return np.array(distances), np.array(global_angle)
    
    def _raycast(self, x: float, y: float, angle: float) -> float:
        """
        改进版碰撞检测（严格遍历所有边界线，返回最近碰撞点）
        参数:
            x, y: 激光发射起点
            angle: 激光发射角度（弧度）
        返回:
            最近的碰撞距离，若无碰撞返回max_dist
        """
        max_dist = 20.0
        closest_dist = max_dist
        
        for (p1, p2) in self.boundary_lines:
            # 计算激光与当前边界线的交点
            intersection_dist = self._line_intersection(
                ray_start=(x, y),
                ray_angle=angle,
                line_p1=p1,
                line_p2=p2
            )
            
            # 更新最近有效碰撞距离
            if 0 < intersection_dist < closest_dist:
                closest_dist = intersection_dist
                
        return closest_dist

    def _line_intersection(self, 
                        ray_start: Tuple[float, float], 
                        ray_angle: float,
                        line_p1: Tuple[float, float], 
                        line_p2: Tuple[float, float]) -> float:
        """
        计算射线与线段的精确交点距离
        返回:
            交点距离（若不存在交点返回-1）
        """
        # 转换为参数方程形式
        x0, y0 = ray_start
        dx, dy = np.cos(ray_angle), np.sin(ray_angle)
        
        x1, y1 = line_p1
        x2, y2 = line_p2
        
        # 计算分母
        denom = (y2 - y1)*dx - (x2 - x1)*dy
        
        # 处理平行情况
        if abs(denom) < 1e-6:
            return -1.0
        
        # 计算参数u和t
        u = ((x1 - x0)*dy - (y1 - y0)*dx) / denom
        t = ((x1 - x0)*(y2 - y1) - (y1 - y0)*(x2 - x1)) / denom
        
        # 检查是否有效交点
        if 0 <= u <= 1 and t > 0:
            return t
        return -1.0