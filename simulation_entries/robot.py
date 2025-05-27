#./simulation_entries/robot.py
import numpy as np

class RobotLaserScanner:
    def __init__(self, x=5, y=5, phi=np.pi/4, boundary_lines=None, laser_configs=None):
        """
        机器人和激光扫描器逻辑类
        
        参数:
            x, y, phi: 初始位置和朝向
            boundary_lines: 环境边界线列表 [[(x1,y1), (x2,y2)], ...]
            laser_configs: 激光配置 [((相对距离,相对角度), 激光朝向), ...]
        """
        # 机器人状态
        self.x = x
        self.y = y
        self.phi = phi  # 朝向角度（弧度）
        
        # 环境配置（提供默认值）
        self.boundary_lines = boundary_lines or [
            [(0, 0), (10, 0)],
            [(10, 0), (10, 10)],
            [(10, 10), (0, 10)],
            [(0, 10), (0, 0)]
        ]
        
        # 激光传感器配置（提供默认值）
        self.laser_configs = laser_configs or [
            ((1, np.pi/4), np.pi/4),   # 左前传感器
            ((0.5, np.pi), np.pi),     # 正前传感器
            ((1, -np.pi/4), -np.pi/4), # 右前传感器
        ]
        
        # 当前扫描结果
        self.current_scan_result = None
        self.current_distance_result = None
    
    def update_pose(self, x, y, phi):
        """更新机器人位姿"""
        self.x = x
        self.y = y
        self.phi = phi
    
    def calculate_intersection(self, laser_pos, laser_angle, line_segment):
        """计算激光与线段的交点"""
        x0, y0 = laser_pos
        x1, y1 = line_segment[0]
        x2, y2 = line_segment[1]
        
        dx_laser = np.cos(laser_angle)
        dy_laser = np.sin(laser_angle)
        dx_line = x2 - x1
        dy_line = y2 - y1
        
        det = dx_laser * dy_line - dy_laser * dx_line
        
        if abs(det) < 1e-6:
            return None
        
        t = ((x1 - x0) * dy_line - (y1 - y0) * dx_line) / det
        s = ((x0 - x1) * dy_laser - (y0 - y1) * dx_laser) / -det
        
        if t >= 0 and 0 <= s <= 1:
            return (x0 + t * dx_laser, 
                    y0 + t * dy_laser, 
                    t)
        return None
    
    def scan(self):
        """执行激光扫描，结果存储在成员变量中"""
        scan_results = []
        distance_results = []
        
        for (rel_r, rel_angle), laser_angle in self.laser_configs:
            # 计算激光头全局位置
            laser_x = self.x + rel_r * np.cos(self.phi + rel_angle)
            laser_y = self.y + rel_r * np.sin(self.phi + rel_angle)
            
            # 计算激光发射角度（全局坐标系）
            global_angle = laser_angle + self.phi
            
            # 寻找最近的碰撞点
            min_distance = float('inf')
            closest_intersection = None
            
            for boundary in self.boundary_lines:
                intersection = self.calculate_intersection(
                    (laser_x, laser_y), global_angle, boundary)
                
                if intersection and intersection[2] < min_distance:
                    min_distance = intersection[2]
                    closest_intersection = intersection[:2]
            
            if closest_intersection:
                scan_results.append({
                    'position': (laser_x, laser_y),
                    'hit_point': closest_intersection,
                    'distance': min_distance,
                    'angle': global_angle,
                    'config_index': len(scan_results)
                })
                distance_results.append(min_distance)
            else:
                scan_results.append({
                    'position': (laser_x, laser_y),
                    'hit_point': None,
                    'distance': float('inf'),
                    'angle': global_angle,
                    'config_index': len(scan_results)
                })
                distance_results.append(float('inf'))
        
        # 存储当前扫描结果
        self.current_scan_result = scan_results
        self.current_distance_result = distance_results