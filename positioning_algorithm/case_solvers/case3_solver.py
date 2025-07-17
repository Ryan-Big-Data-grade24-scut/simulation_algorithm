from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math
from math import sin, cos, atan2, asin, sqrt, pi
import os
from .BaseSolver import BaseSolver, BaseSolverConfig
import logging

@dataclass
class Case3Config:
    """Case3 求解器特有配置
    
    Attributes:
        angle_tolerance (float): 角度验证容忍度，默认1e-5
        enable_math_debug (bool): 是否启用数学计算调试日志，默认False
    """
    angle_tolerance: float = 1e-5
    enable_math_debug: bool = False

class Case3Solver(BaseSolver):
    """情况3求解器：三个顶点分别在三个不同边缘
    
    Args:
        t (List[float]): 3个激光测距值 [t0, t1, t2] (单位: m)
        theta (List[float]): 3个激光角度 [θ0, θ1, θ2] (单位: rad)
        m (float): 场地x方向长度 (m)
        n (float): 场地y方向长度 (m)
        config (Optional[BaseSolverConfig]): 基础配置
        case_config (Optional[Case3Config]): 情况3特有配置
        ros_logger (Optional): ROS2日志器对象
    """

    def __init__(self, 
                 t: List[float], 
                 theta: List[float],
                 m: float, 
                 n: float, 
                 config: Optional[BaseSolverConfig] = None,
                 case_config: Optional[Case3Config] = None,
                 ros_logger=None,
                 min_log_level=logging.DEBUG):
        # 初始化基础配置
        config = config or BaseSolverConfig(
            log_file=os.path.join('logs', 'case3.log'),
            log_level='DEBUG'
        )
        super().__init__(t, theta, m, n, config, ros_logger, min_log_level=min_log_level)
        
        # 情况3特有配置
        self.case_config = case_config or Case3Config()
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        self.logger.info(f"Case3Solver initialized with {self.case_config}")
        self._validate_case3_inputs()

    def _validate_case3_inputs(self):
        """验证情况3特有的输入条件
        
        验证:
            - 必须有3个激光测距值和角度
            - 所有测距值必须为正数
            - 角度必须在[-π, π]范围内
        """
        if len(self.t) != 3 or len(self.theta) != 3:
            error_msg = "情况3需要3个激光测距值和角度"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        for i, (t_val, theta_val) in enumerate(zip(self.t, self.theta)):
            if t_val <= 0:
                error_msg = f"激光测距值必须为正数，P{i}的t={t_val}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            if not (-pi <= theta_val <= pi):
                error_msg = f"激光角度必须在[-π, π]范围内，P{i}的theta={theta_val}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        self.logger.debug("输入参数验证通过")

    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """主求解方法 - 处理所有情况3的组合
        
        返回:
            解列表 [((xmin,xmax), (ymin,ymax), phi), ...]
        """
        solutions = []
        self.logger.info("\n=== 开始情况3求解 ===")
        self.logger.info(f"输入参数: t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
        
        # 所有可能的边组合
        edge_combinations = [
            ('left', 'right', 'top'),
            ('left', 'right', 'bottom'),
            ('left', 'top', 'bottom'),
            ('right', 'top', 'bottom')
        ]
        
        for combo_idx, (p0_edge, p1_edge, p2_edge) in enumerate(edge_combinations, 1):
            self.logger.info(f"\n=== 尝试边组合 {combo_idx}: {p0_edge}, {p1_edge}, {p2_edge} ===")
            
            # 尝试所有顶点排列组合（6种排列）
            for p0, p1, p2 in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                self.logger.info(f"\n  尝试顶点排列: P{p0}, P{p1}, P{p2}")
                
                try:
                    phi_candidates = self._solve_edge_combo(
                        p0, p1, p2, p0_edge, p1_edge, p2_edge)
                    self.logger.info(f"  发现{len(phi_candidates)}个phi候选值")
                    
                    for phi in phi_candidates:
                        self.logger.info(f"\n    测试phi = {phi:.8f} rad ({np.degrees(phi):.2f}°)")
                        
                        xO, yO = self._compute_position(phi, p0, p1, p2, 
                                                      p0_edge, p1_edge, p2_edge)
                        self.logger.info(f"    计算位置: xO={xO:.8f}, yO={yO:.8f}")
                        
                        if np.isnan(xO) or np.isnan(yO):
                            self.logger.warning("    无效位置(NaN)，跳过")
                            continue
                            
                        if self._verify_solution(xO, yO, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
                            self.logger.info("    *** 解验证通过! ***")
                            solutions.append(((xO, xO), (yO, yO), phi))
                        else:
                            self.logger.warning("    解验证失败")
                            
                except Exception as e:
                    self.logger.error(f"    求解过程中出错: {str(e)}", exc_info=True)
                    continue
        
        self.logger.info(f"\n=== 求解结果汇总 ===")
        self.logger.info(f"共找到{len(solutions)}个有效解")
        
        for i, (x_range, y_range, phi) in enumerate(solutions, 1):
            self.logger.info(f"解#{i}:")
            self.logger.info(f"  x范围: [{x_range[0]:.6f}, {x_range[1]:.6f}]")
            self.logger.info(f"  y范围: [{y_range[0]:.6f}, {y_range[1]:.6f}]")
            self.logger.info(f"  角度: {phi:.6f} rad ({np.degrees(phi):.2f}°)")
            
        return solutions

    def _solve_edge_combo(self, p0: int, p1: int, p2: int,
                         p0_edge: str, p1_edge: str, p2_edge: str) -> List[float]:
        """解特定边组合的方程
        
        参数:
            p0, p1, p2: 顶点索引
            p0_edge, p1_edge, p2_edge: 对应边缘
            
        返回:
            phi候选值列表
        """
        self.logger.info(f"\n    [求解边组合方程]")
        self.logger.info(f"    P{p0}在{p0_edge}, P{p1}在{p1_edge}, P{p2}在{p2_edge}")
        
        if {p0_edge, p1_edge} == {'left', 'right'}:
            self.logger.info("    情况: 左+右+上/下")
            A = self.t[p1]*cos(self.theta[p1]) - self.t[p0]*cos(self.theta[p0])
            B = -(self.t[p1]*sin(self.theta[p1]) - self.t[p0]*sin(self.theta[p0]))
            C = self.m
            self._log_math(f"A = t{p1}*cos(θ{p1}) - t{p0}*cos(θ{p0})", A)
            self._log_math(f"B = -(t{p1}*sin(θ{p1}) - t{p0}*sin(θ{p0}))", B)
            self._log_math(f"C = m", C)
        elif {p2_edge, p1_edge} == {'bottom', 'top'}:
            self.logger.info("    情况: 上+下")
            A = self.t[p2]*sin(self.theta[p2]) - self.t[p1]*sin(self.theta[p1])
            B = self.t[p2]*cos(self.theta[p2]) - self.t[p1]*cos(self.theta[p1])
            C = -self.n
            self._log_math(f"A = t{p2}*sin(θ{p2}) - t{p1}*sin(θ{p1})", A)
            self._log_math(f"B = t{p2}*cos(θ{p2}) - t{p1}*cos(θ{p1})", B)
            self._log_math(f"C = -n", C)
        else:
            self.logger.info("    不支持的边组合，跳过")
            return []
        
        norm = sqrt(A*A + B*B)
        self._log_math(f"norm = sqrt(A² + B²)", norm)

        if abs(C) > norm + self.config.tol:
            self.logger.debug("    无解: |C| > norm")
            return []

        alpha = atan2(A, B)
        self._log_math(f"alpha = atan2(A, B)", alpha)

        phi1 = asin(C / norm) - alpha
        phi2 = pi - asin(C / norm) - alpha
        self._log_math(f"phi1 = asin(C/norm) - alpha", phi1)
        self._log_math(f"phi2 = π - asin(C/norm) - alpha", phi2)
        
        solutions = []
        for phi in [phi1, phi2]:
            if phi < -pi:
                phi += 2*pi
            elif phi > pi:
                phi -= 2*pi
            solutions.append(phi)
        
        self.logger.debug(f"    最终phi候选值: {solutions}")
        return solutions

    def _compute_position(self, phi: float, p0: int, p1: int, p2: int,
                         p0_edge: str, p1_edge: str, p2_edge: str) -> Tuple[float, float]:
        """计算中心点坐标
        
        参数:
            phi: 角度解
            p0, p1, p2: 顶点索引
            p0_edge, p1_edge, p2_edge: 对应边缘
            
        返回:
            (xO, yO) 中心坐标
        """
        self.logger.debug(f"\n    [计算位置: phi={phi:.6f}]")
        x_values = []
        y_values = []
        
        for vertex, edge in [(p0, p0_edge), (p1, p1_edge), (p2, p2_edge)]:
            if edge == 'left':
                x = -self.t[vertex] * cos(phi + self.theta[vertex])
                self._log_math(f"P{vertex}在左边缘: x = -t{vertex}*cos(phi+θ{vertex})", x)
                x_values.append(x)
            elif edge == 'right':
                x = self.m - self.t[vertex] * cos(phi + self.theta[vertex])
                self._log_math(f"P{vertex}在右边缘: x = m - t{vertex}*cos(phi+θ{vertex})", x)
                x_values.append(x)
            elif edge == 'top':
                y = self.n - self.t[vertex] * sin(phi + self.theta[vertex])
                self._log_math(f"P{vertex}在上边缘: y = n - t{vertex}*sin(phi+θ{vertex})", y)
                y_values.append(y)
            elif edge == 'bottom':
                y = -self.t[vertex] * sin(phi + self.theta[vertex])
                self._log_math(f"P{vertex}在下边缘: y = -t{vertex}*sin(phi+θ{vertex})", y)
                y_values.append(y)
        
        xO = sum(x_values)/len(x_values) if x_values else 0
        yO = sum(y_values)/len(y_values) if y_values else 0
        
        if not x_values:
            self.logger.debug("    无x约束，默认xO=0")
        if not y_values:
            self.logger.debug("    无y约束，默认yO=0")
            
        self.logger.debug(f"    最终位置: xO={xO:.8f}, yO={yO:.8f}")
        return xO, yO

    def _verify_solution(self, xO: float, yO: float, phi: float,
                        p0: int, p1: int, p2: int,
                        p0_edge: str, p1_edge: str, p2_edge: str) -> bool:
        """验证所有顶点是否在指定边且不越界
        
        参数:
            xO, yO: 中心坐标
            phi: 角度
            p0, p1, p2: 顶点索引
            p0_edge, p1_edge, p2_edge: 对应边缘
            
        返回:
            是否有效
        """
        self.logger.debug(f"\n    [验证解: phi={phi:.6f}]")
        points = []
        
        for i in [p0, p1, p2]:
            x = xO + self.t[i] * cos(phi + self.theta[i])
            y = yO + self.t[i] * sin(phi + self.theta[i])
            points.append((x, y))
            self.logger.debug(f"    P{i} = ({x:.8f}, {y:.8f})")
        
        edges = [p0_edge, p1_edge, p2_edge]
        for (x, y), edge in zip(points, edges):
            if edge == 'left':
                valid = abs(x) < self.config.tol
                self.logger.debug(f"    P在左边缘: x={x:.6f} ≈ 0? {'是' if valid else '否'}")
                if not valid: return False
            elif edge == 'right':
                valid = abs(x - self.m) < self.config.tol
                self.logger.debug(f"    P在右边缘: x={x:.6f} ≈ {self.m}? {'是' if valid else '否'}")
                if not valid: return False
            elif edge == 'top':
                valid = abs(y - self.n) < self.config.tol
                self.logger.debug(f"    P在上边缘: y={y:.6f} ≈ {self.n}? {'是' if valid else '否'}")
                if not valid: return False
            elif edge == 'bottom':
                valid = abs(y) < self.config.tol
                self.logger.debug(f"    P在下边缘: y={y:.6f} ≈ 0? {'是' if valid else '否'}")
                if not valid: return False
        
        self.logger.debug("    *** 所有验证通过 ***")
        return True

    def _log_math(self, expr: str, value: float):
        """记录数学计算过程
        
        参数:
            expr: 数学表达式字符串
            value: 计算结果值
        """
        if self.case_config.enable_math_debug:
            self.logger.debug(f"      {expr} = {value:.8f}")
