import numpy as np
from typing import List, Tuple, Optional
from .angle_pose_calculator import PoseSolver
from .perpendicular_localizer import PerpendicularDistanceLocalizer

class DualSolver:
    """
    双求解器：优先使用垂直距离定位器，备用角度位姿求解器
    """
    
    def __init__(self, perpendicular_solver, pose_solver):
        """
        初始化双求解器
        
        Args:
            perpendicular_solver: PerpendicularDistanceLocalizer实例
            pose_solver: PoseSolver实例
        """
        self.perpendicular_solver = perpendicular_solver
        self.pose_solver = pose_solver
        self.m = perpendicular_solver.m
        self.n = perpendicular_solver.n
        
    def solve(self, distances, current_angle=None):
        """
        综合求解：优先使用垂直距离定位器，无解时使用角度位姿求解器
        
        Args:
            distances: 激光测距数据
            current_angle: 当前角度（用于垂直距离定位器）
        
        Returns:
            位置解列表
        """
        solutions = []
        
        # 优先使用垂直距离定位器
        try:
            if current_angle is not None:
                perp_solutions = self.perpendicular_solver.localize_with_angle(distances, current_angle)
                if perp_solutions is not None and len(perp_solutions) > 0:
                    solutions.extend(perp_solutions)
                    print(f"垂直距离定位器找到 {len(perp_solutions)} 个解")
        except Exception as e:
            print(f"垂直距离定位器求解失败: {e}")
        
        # 如果垂直距离定位器没有找到解，使用角度位姿求解器
        if not solutions:
            try:
                if current_angle is not None:
                    pose_solutions = self.pose_solver.solve(distances, current_angle)
                    if pose_solutions is not None and len(pose_solutions) > 0:
                        # 转换格式：pose_solver返回(x, y)，需要添加角度信息
                        formatted_solutions = [(sol[0], sol[1], current_angle) for sol in pose_solutions]
                        solutions.extend(formatted_solutions)
                        print(f"角度位姿求解器找到 {len(pose_solutions)} 个解")
                else:
                    print("角度位姿求解器需要current_angle参数")
            except Exception as e:
                print(f"角度位姿求解器求解失败: {e}")
        
        return solutions


def create_dual_solver(laser_configs, calibration_data, ros_logger=None):
    """
    便捷函数：创建双求解器
    
    Args:
        laser_configs: 激光配置列表 [((r, theta), beam_angle), ...]
        calibration_data: 标定扫描数据 (100, N)
        ros_logger: ROS日志器（可选）
    
    Returns:
        tuple: (dual_solver, calculated_m, calculated_n, x, y, phi)
    """
    print("正在初始化垂直距离定位器...")
    perpendicular_solver = PerpendicularDistanceLocalizer(laser_configs, tolerance=0.1, ros_logger=ros_logger)
    
    # 使用外部提供的标定数据进行标定
    calculated_m, calculated_n, x, y, phi = perpendicular_solver.calibrate(calibration_data)
    print("垂直距离定位器标定完成")
    
    # 使用计算出的m和n创建PoseSolver
    print("正在初始化角度位姿求解器...")
    pose_solver = PoseSolver(
        calculated_m, calculated_n, laser_configs,
        Rcl_logger=ros_logger
    )
    print("角度位姿求解器已启用")
    
    # 创建双求解器
    dual_solver = DualSolver(perpendicular_solver, pose_solver)
    print("双求解器模式已启用：优先使用垂直距离定位器，备用角度位姿求解器")
    
    return dual_solver, calculated_m, calculated_n, x, y, phi


def create_single_solver(solver_type, laser_configs, calibration_data=None, m=None, n=None, ros_logger=None):
    """
    便捷函数：创建单一求解器
    
    Args:
        solver_type: 求解器类型 ('perpendicular' 或 'angle')
        laser_configs: 激光配置列表 [((r, theta), beam_angle), ...]
        calibration_data: 标定扫描数据（垂直距离定位器需要）
        m: 场地长度（角度求解器需要）
        n: 场地宽度（角度求解器需要）
        ros_logger: ROS日志器（可选）
    
    Returns:
        求解器实例 或 (求解器实例, m, n, x, y, phi)
    """
    if solver_type == 'perpendicular':
        if calibration_data is None:
            raise ValueError("垂直距离定位器需要提供calibration_data参数")
        
        print("正在初始化垂直距离定位器...")
        solver = PerpendicularDistanceLocalizer(laser_configs, tolerance=0.1, ros_logger=ros_logger)
        
        # 使用外部提供的标定数据进行标定
        calculated_m, calculated_n, x, y, phi = solver.calibrate(calibration_data)
        print("单垂直距离定位器模式")
        
        return solver, calculated_m, calculated_n, x, y, phi
    
    elif solver_type == 'angle':
        if m is None or n is None:
            raise ValueError("角度位姿求解器需要指定m和n参数")
        
        print("正在初始化角度位姿求解器...")
        solver = PoseSolver(m, n, laser_configs, Rcl_logger=ros_logger)
        print("单角度位姿求解器模式")
        return solver
    
    else:
        raise ValueError(f"不支持的求解器类型: {solver_type}")
