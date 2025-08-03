import numpy as np
from typing import List, Tuple, Optional
import warnings
from .batch_solvers.Base_log import BaseLog

class PerpendicularDistanceLocalizer(BaseLog):
    """
    基于激光束垂直距离的定位算法
    简化版本：100行代码实现
    """
    
    def __init__(self, laser_configs, tolerance: float = 0.1, ros_logger=None):
        """
        初始化定位器（不进行自动标定）
        
        Args:
            laser_configs: 激光配置列表 [((r, theta), beam_angle), ...]
            tolerance: 容差范围
            ros_logger: ROS日志器（可选）
        """
        super().__init__()
        
        self.laser_configs = laser_configs
        self.tolerance = tolerance
        self.ros_logger = ros_logger
        self.solver_name = "PerpendicularDistanceLocalizer"
        
        # 初始化日志
        if self.ros_logger is None:
            self._setup_logging(self.solver_name)
        
        # 标定相关变量（需要调用calibrate方法来设置）
        self.distances_array = None
        self.h_values = None
        self.avg_front = None
        self.avg_back = None
        self.avg_left = None
        self.avg_right = None
        self.m = None
        self.n = None
        
        # 激光束索引
        self.front_indices = []
        self.back_indices = []
        self.left_indices = []
        self.right_indices = []
        
        # 分析激光配置，获取索引和h_i
        self._analyze_laser_configuration()
        
        self._log_info("垂直距离定位器已初始化，需要调用calibrate()方法进行标定")
    
    def _analyze_laser_configuration(self):
        """分析激光束配置，获取索引和h_i"""
        self.front_indices = []    # 前束索引（0）
        self.back_indices = []     # 后束索引（±π）
        self.left_indices = []     # 左束索引（π/2）
        self.right_indices = []    # 右束索引（-π/2）
        
        self.h_values = np.zeros(len(self.laser_configs))
        
        for i, ((r, theta), beam_angle) in enumerate(self.laser_configs):
            # 归一化角度
            normalized_angle = ((beam_angle + np.pi) % (2 * np.pi)) - np.pi
            
            if abs(normalized_angle) < 0.1:
                # 前束：0
                self.front_indices.append(i)
                self.h_values[i] = abs(r * np.cos(theta))
            elif abs(abs(normalized_angle) - np.pi) < 0.1:
                # 后束：±π
                self.back_indices.append(i)
                self.h_values[i] = abs(r * np.cos(theta))
            elif abs(normalized_angle - np.pi/2) < 0.1:
                # 左束：π/2
                self.left_indices.append(i)
                self.h_values[i] = abs(r * np.sin(theta))
            elif abs(normalized_angle + np.pi/2) < 0.1:
                # 右束：-π/2
                self.right_indices.append(i)
                self.h_values[i] = abs(r * np.sin(theta))
        
        self._log_info(f"前束索引: {self.front_indices}, 后束索引: {self.back_indices}")
        self._log_info(f"左束索引: {self.left_indices}, 右束索引: {self.right_indices}")
        self._log_debug(f"h_values: {self.h_values}")
    
    def calibrate(self, distances_array: np.ndarray):
        """
        使用标定数据进行标定
        
        Args:
            distances_array: 标定扫描数据 (100, N)
        
        Returns:
            tuple: (m, n, x, y, phi) - 场地尺寸和初始位姿
        """
        self._log_info(f"开始标定，扫描数据shape: {distances_array.shape}")
        
        self.distances_array = distances_array
        # 显示测距值每一列的平均值N个
        # 给每N列，每列都求一次的意思
        self.avg_dist = np.mean(self.distances_array, axis=0)  # (N,)
        self._log_array_detailed("平均测距值", self.avg_dist)
        # 显示h
        self._log_array_detailed("h_values", self.h_values)

        # 批量加上h_i
        self.added_distances = self.distances_array + self.h_values  # (100, N)
        
        # 为每个种类生成平均数
        self._calculate_average_distances()
        
        # 计算m和n
        self.m = self.avg_front + self.avg_back
        self.n = self.avg_left + self.avg_right
        
        # 计算初始位姿
        x = (self.m - self.avg_back + self.avg_front) / 2
        y = (self.n - self.avg_right + self.avg_left) / 2
        phi = np.pi
        
        self._log_info(f"标定完成: m={self.m:.3f}, n={self.n:.3f}")
        self._log_info(f"前后左右平均距离: 前={self.avg_front:.3f}, 后={self.avg_back:.3f}, 左={self.avg_left:.3f}, 右={self.avg_right:.3f}")
        self._log_info(f"计算的初始位姿: x={x:.3f}, y={y:.3f}, phi={phi:.3f}")
        
        return self.m, self.n, x, y, phi
    
    def _calculate_average_distances(self):
        """为每个种类计算平均数"""
        # 前束平均
        if self.front_indices:
            front_data = self.added_distances[:, self.front_indices].flatten()
            self.avg_front = np.mean(front_data)
        else:
            self.avg_front = 0
        
        # 后束平均
        if self.back_indices:
            back_data = self.added_distances[:, self.back_indices].flatten()
            self.avg_back = np.mean(back_data)
        else:
            self.avg_back = 0
        
        # 左束平均
        if self.left_indices:
            left_data = self.added_distances[:, self.left_indices].flatten()
            self.avg_left = np.mean(left_data)
        else:
            self.avg_left = 0
        
        # 右束平均
        if self.right_indices:
            right_data = self.added_distances[:, self.right_indices].flatten()
            self.avg_right = np.mean(right_data)
        else:
            self.avg_right = 0
    
    def localize(self, distances: np.ndarray, current_angle: float) -> List[Tuple[float, float, float]]:
        """
        运行阶段定位
        
        Args:
            distances: 当前测距数据 (N,)
            current_angle: 当前朝向角度
        
        Returns:
            可能的位置解 [(x, y, phi), ...]
        """
        # 批量相加h_i
        added_distances = distances + self.h_values  # (N,)
        
        # 为前后左右束生成4个平均数
        d_front = np.mean([added_distances[i] for i in self.front_indices]) if self.front_indices else 0
        d_back = np.mean([added_distances[i] for i in self.back_indices]) if self.back_indices else 0
        d_left = np.mean([added_distances[i] for i in self.left_indices]) if self.left_indices else 0
        d_right = np.mean([added_distances[i] for i in self.right_indices]) if self.right_indices else 0

        self._log_debug(f"前束: {d_front:.3f}, 后束: {d_back:.3f}, 左束: {d_left:.3f}, 右束: {d_right:.3f}")
        
        # 前后束相加，左右束相加
        fb_sum = d_front + d_back
        lr_sum = d_left + d_right

        self._log_debug(f"前后束和: {fb_sum:.3f}, 左右束和: {lr_sum:.3f}")
        
        solutions = []
        
        # 判断是否在m,n±tol范围内
        if abs(fb_sum - self.m) <= self.tolerance and abs(lr_sum - self.n) <= self.tolerance:
            # 情况1：前后=m，左右=n
            x1 = (self.m - d_front + d_back) / 2
            y1 = (self.n - d_left + d_right) / 2
            phi1 = 0.0
            
            x2 = (self.m - d_back + d_front) / 2
            y2 = (self.n - d_right + d_left) / 2
            phi2 = np.pi
            
            # 根据当前朝向选择
            if -np.pi/2 <= current_angle <= np.pi/2:
                solutions.append((x1, y1, phi1))
            else:
                solutions.append((x2, y2, phi2))
        
        # 判断是否在n,m±tol范围内（对称情况）
        if abs(fb_sum - self.n) <= self.tolerance and abs(lr_sum - self.m) <= self.tolerance:
            # 情况2：前后=n，左右=m
            x3 = (self.m - d_right + d_left) / 2
            y3 = (self.n - d_front + d_back) / 2
            phi3 = np.pi/2
            
            x4 = (self.m - d_left + d_right) / 2
            y4 = (self.n - d_back + d_front) / 2
            phi4 = -np.pi/2
            
            # 根据当前朝向选择
            if 0 <= current_angle <= np.pi:
                solutions.append((x3, y3, phi3))
            else:
                solutions.append((x4, y4, phi4))
        
        return solutions
    
    def localize_with_angle(self, distances: np.ndarray, angle: float) -> List[Tuple[float, float, float]]:
        """
        基于角度的运行阶段定位
        
        Args:
            distances: 当前测距数据 (N,)
            angle: 当前朝向角度
        
        Returns:
            可能的位置解 [(x, y, phi), ...]
        """
        angle_tol = 5 / 180 * np.pi  # 角度容差
        
        # 第一步：排除不能计算的情况
        
        # 1. 检查角度是否接近0、π/2、π、3π/2
        normalized_angle = ((angle + np.pi) % (2 * np.pi)) - np.pi  # 归一化到[-π, π]
        
        valid_angles = [0, np.pi/2, np.pi, -np.pi/2]  # 对应0、π/2、π、3π/2
        angle_valid = False
        target_angle = None
        
        for valid_angle in valid_angles:
            if abs(normalized_angle - valid_angle) <= angle_tol:
                angle_valid = True
                target_angle = valid_angle
                break
        
        if not angle_valid:
            self._log_debug(f"角度 {angle:.3f} 不在有效范围内（0, π/2, π, 3π/2 ± {angle_tol:.3f}）")
            return []
        
        # 2. 检查前后左右束是否都有有效值
        d_front_valid = any(distances[i] > 0 for i in self.front_indices) if self.front_indices else False
        d_back_valid = any(distances[i] > 0 for i in self.back_indices) if self.back_indices else False
        d_left_valid = any(distances[i] > 0 for i in self.left_indices) if self.left_indices else False
        d_right_valid = any(distances[i] > 0 for i in self.right_indices) if self.right_indices else False
        
        if not all([d_front_valid, d_back_valid, d_left_valid, d_right_valid]):
            self._log_debug(f"前后左右束不全有效值: 前={d_front_valid}, 后={d_back_valid}, 左={d_left_valid}, 右={d_right_valid}")
            return []
        
        # 第二步：计算距离值
        angle_mod = abs(target_angle) % (np.pi/2)  # 计算 angle % π/2
        cos_factor = np.cos(angle_mod)
        
        # 批量相加h_i
        added_distances = distances + self.h_values  # (N,)
        
        # 计算各方向的平均距离并应用cos因子
        d_front_raw = np.mean([added_distances[i] for i in self.front_indices]) if self.front_indices else 0
        d_back_raw = np.mean([added_distances[i] for i in self.back_indices]) if self.back_indices else 0
        d_left_raw = np.mean([added_distances[i] for i in self.left_indices]) if self.left_indices else 0
        d_right_raw = np.mean([added_distances[i] for i in self.right_indices]) if self.right_indices else 0
        
        d_front = cos_factor * d_front_raw
        d_back = cos_factor * d_back_raw
        d_left = cos_factor * d_left_raw
        d_right = cos_factor * d_right_raw
        
        self._log_debug(f"角度={angle:.3f}, 目标角度={target_angle:.3f}, cos因子={cos_factor:.3f}")
        self._log_debug(f"修正后距离: 前={d_front:.3f}, 后={d_back:.3f}, 左={d_left:.3f}, 右={d_right:.3f}")
        
        # 第三步：分四种情况讨论
        solutions = []
        
        if abs(target_angle) < angle_tol:  # 接近0
            # 情况1：角度≈0
            fb_sum = d_front + d_back
            lr_sum = d_left + d_right
            
            if abs(fb_sum - self.m) < self.tolerance and abs(lr_sum - self.n) < self.tolerance:
                x = (self.m - d_front + d_back) / 2
                y = (self.n - d_left + d_right) / 2
                solutions.append((x, y, 0.0))
                self._log_debug(f"找到解（角度≈0）: x={x:.3f}, y={y:.3f}, phi=0.0")
                
        elif abs(target_angle - np.pi) < angle_tol:  # 接近π
            # 情况2：角度≈π（前后相反，左右相反）
            fb_sum = d_front + d_back
            lr_sum = d_left + d_right
            
            if abs(fb_sum - self.m) < self.tolerance and abs(lr_sum - self.n) < self.tolerance:
                x = (self.m - d_back + d_front) / 2  # 前后相反
                y = (self.n - d_right + d_left) / 2  # 左右相反
                solutions.append((x, y, np.pi))
                self._log_debug(f"找到解（角度≈π）: x={x:.3f}, y={y:.3f}, phi=π")
                
        elif abs(target_angle - np.pi/2) < angle_tol:  # 接近π/2
            # 情况3：角度≈π/2
            fb_sum = d_front + d_back
            lr_sum = d_left + d_right
            
            if abs(fb_sum - self.n) < self.tolerance and abs(lr_sum - self.m) < self.tolerance:
                x = (self.m - d_right + d_left) / 2
                y = (self.n - d_front + d_back) / 2
                solutions.append((x, y, np.pi/2))
                self._log_debug(f"找到解（角度≈π/2）: x={x:.3f}, y={y:.3f}, phi=π/2")
                
        elif abs(target_angle + np.pi/2) < angle_tol:  # 接近-π/2
            # 情况4：角度≈-π/2（左右相反，前后相反）
            fb_sum = d_front + d_back
            lr_sum = d_left + d_right
            
            if abs(fb_sum - self.n) < self.tolerance and abs(lr_sum - self.m) < self.tolerance:
                x = (self.m - d_left + d_right) / 2  # 左右相反
                y = (self.n - d_back + d_front) / 2  # 前后相反
                solutions.append((x, y, -np.pi/2))
                self._log_debug(f"找到解（角度≈-π/2）: x={x:.3f}, y={y:.3f}, phi=-π/2")
        
        if not solutions:
            self._log_debug(f"未找到有效解，前后和={d_front + d_back:.3f}, 左右和={d_left + d_right:.3f}")
        
        return solutions

    