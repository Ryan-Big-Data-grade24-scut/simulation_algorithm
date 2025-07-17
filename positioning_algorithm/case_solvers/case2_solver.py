import math
import sys
import os
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

# 路径处理
if __name__ == '__main__':
    sys.path.append('/home/lrx/laser_positioning_ubuntu/simulation_algorithm/positioning_algorithm')
    from case_solvers.BaseSolver import BaseSolver, BaseSolverConfig
else:
    from .BaseSolver import BaseSolver, BaseSolverConfig

@dataclass
class Case2Config:
    """Case2 求解器特有配置
    
    Attributes:
        range_tolerance (float): 坐标范围验证容忍度，默认1e-5
        enable_edge_debug (bool): 是否启用边缘调试日志，默认False
    """
    range_tolerance: float = 1e-5
    enable_edge_debug: bool = False

class Case2Solver(BaseSolver):
    """情况2求解器：一边在矩形边缘且对角顶点在对边
    
    Args:
        t (List[float]): 3个激光测距值 [t0, t1, t2] (单位: m)
        theta (List[float]): 3个激光角度 [θ0, θ1, θ2] (单位: rad)
        m (float): 场地x方向长度 (m)
        n (float): 场地y方向长度 (m)
        config (Optional[BaseSolverConfig]): 基础配置
        case_config (Optional[Case2Config]): 情况2特有配置
        ros_logger (Optional): ROS2日志器对象
    """

    def __init__(self, 
                 t: List[float], 
                 theta: List[float],
                 m: float, 
                 n: float, 
                 config: Optional[BaseSolverConfig] = None,
                 case_config: Optional[Case2Config] = None,
                 ros_logger=None,
                 min_log_level=logging.DEBUG):
        # 初始化基础配置
        config = config or BaseSolverConfig(
            log_file=os.path.join('logs', 'case2.log'),
            log_level='DEBUG'
        )
        super().__init__(t, theta, m, n, config, ros_logger, min_log_level=min_log_level)
        
        # 情况2特有配置
        self.case_config = case_config or Case2Config()
        self.edges = ['bottom', 'top', 'left', 'right']
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        self.logger.info(f"Case2Solver initialized with {self.case_config}")

    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """执行情况2求解流程
        
        Returns:
            List[Tuple]: 有效解列表，每个解格式为:
                ((x_min, x_max), (y_min, y_max), phi)
                
        Raises:
            RuntimeError: 当求解过程中出现不可恢复错误
        """
        solutions = []
        try:
            self.logger.info(f"Start solving with t={self.t}, theta={self.theta}")
            
            for base_idx in range(3):
                solutions.extend(
                    self._solve_for_base_edge(base_idx)
                )
                
            self.logger.info(f"Found {len(solutions)} valid solutions")
            return solutions
            
        except Exception as e:
            error_msg = f"Solve failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _solve_for_base_edge(self, base_idx: int) -> List[Tuple]:
        """处理特定基边的所有可能解
        
        Args:
            base_idx (int): 基边顶点索引 (0-2)
            
        Returns:
            List[Tuple]: 该基边下的有效解
        """
        solutions = []
        p1, p2 = base_idx, (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        self.logger.debug(f"\nProcessing base edge {base_idx} (P{p1}-P{p2})")
        
        for rect_edge in self.edges:
            try:
                self.logger.debug(f"  Trying rectangle edge: {rect_edge}")
                
                for phi in self._solve_phi_equation(p1, p2, opp_idx, rect_edge):
                    position = self._compute_position(phi, p1, p2, opp_idx, rect_edge)
                    if position and self._verify_solution(*position, phi, p1, p2, opp_idx, rect_edge):
                        solutions.append((*position, phi))
                        self.logger.info(f"Found valid solution: {position} with phi={phi:.6f}")
                        
            except Exception as e:
                self.logger.warning(f"Edge {rect_edge} failed: {str(e)}")
                continue
                
        return solutions

    def _solve_phi_equation(self, 
                          p1: int, 
                          p2: int,
                          opp_idx: int,
                          rect_edge: str) -> List[float]:
        """求解phi的候选角度
        
        Args:
            p1: 基边顶点1索引
            p2: 基边顶点2索引
            opp_idx: 对角顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            List[float]: 有效的phi角度列表 (rad)
        """
        # 确定目标值
        if rect_edge in ['bottom', 'top']:
            target = self.n if rect_edge == 'bottom' else -self.n
            equation_type = "A*sin(phi) + B*cos(phi) = C"
        else:
            target = self.m if rect_edge == 'left' else -self.m
            equation_type = "A*cos(phi) - B*sin(phi) = C"
        
        self.logger.debug(f"Solving equation: {equation_type}, target={target:.6f}")
        
        # 计算系数
        A = self.t[opp_idx] * math.cos(self.theta[opp_idx]) - self.t[p1] * math.cos(self.theta[p1])
        B = self.t[opp_idx] * math.sin(self.theta[opp_idx]) - self.t[p1] * math.sin(self.theta[p1])
        self._log_math(f"A = t{opp_idx}*cosθ{opp_idx} - t{p1}*cosθ{p1}", A)
        self._log_math(f"B = t{opp_idx}*sinθ{opp_idx} - t{p1}*sinθ{p1}", B)
        
        norm = math.sqrt(A**2 + B**2)
        self._log_math(f"norm = sqrt(A²+B²)", norm)
        
        # 退化情况处理
        if norm < self.config.tol:
            self.logger.debug("Degenerate case (norm ≈ 0)")
            return []
            
        # 特殊解情况
        if abs(abs(target) - norm) < self.config.tol:
            self.logger.debug("Special case (|target| ≈ norm)")
            sign = 1 if target > 0 else -1
            if rect_edge in ['bottom', 'top']:
                return [math.atan2(sign*B, sign*A)]
            else:
                return [math.atan2(sign*B, -sign*A)]
                
        # 无解情况
        if norm < abs(target):
            self.logger.debug("No solution (norm < |target|)")
            return []
            
        # 一般解
        alpha = math.atan2(B, A)
        if rect_edge in ['bottom', 'top']:
            phi1 = math.asin(target/norm) - alpha
            phi2 = math.pi - math.asin(target/norm) - alpha
        else:
            phi1 = math.acos(target/norm) - alpha
            phi2 = -math.acos(target/norm) - alpha
            
        return [phi for phi in [phi1, phi2] 
               if self._validate_phi(phi, p1, rect_edge)]

    def _validate_phi(self, 
                    phi: float,
                    base_idx: int,
                    rect_edge: str) -> bool:
        """快速验证phi角度有效性
        
        Args:
            phi: 待验证角度 (rad)
            base_idx: 基边顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            bool: 是否可能为有效角度
        """
        # 简单检查角度是否在合理范围内
        if not (0 <= phi % (2*math.pi) <= math.pi/2 + self.config.tol):
            self.logger.debug(f"Phi {phi:.6f} out of reasonable range")
            return False
        return True

    def _compute_position(self,
                         phi: float,
                         p1: int,
                         p2: int,
                         opp_idx: int,
                         rect_edge: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """计算中心点坐标范围
        
        Args:
            phi: 当前角度 (rad)
            p1: 基边顶点1索引
            p2: 基边顶点2索引
            opp_idx: 对角顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            Optional[Tuple]: 坐标范围 ((x_min,x_max), (y_min,y_max)) 或 None
        """
        if rect_edge in ['bottom', 'top']:
            # 水平基边情况
            yO = (0 if rect_edge == 'bottom' else self.n) - self.t[p1] * math.sin(phi + self.theta[p1])
            
            # 计算x范围
            x_p1 = self.t[p1] * math.cos(phi + self.theta[p1])
            x_p2 = self.t[p2] * math.cos(phi + self.theta[p2])
            x_min = max(-x_p1, -x_p2)
            x_max = min(self.m - x_p1, self.m - x_p2)
            
            if x_min > x_max + self.case_config.range_tolerance:
                self.logger.debug(f"Invalid x range: {x_min:.6f} > {x_max:.6f}")
                return None
                
            return ((x_min, x_max), (yO, yO))
            
        else:
            # 垂直基边情况
            xO = (0 if rect_edge == 'left' else self.m) - self.t[p1] * math.cos(phi + self.theta[p1])
            
            # 计算y范围
            y_p1 = self.t[p1] * math.sin(phi + self.theta[p1])
            y_p2 = self.t[p2] * math.sin(phi + self.theta[p2])
            y_min = max(-y_p1, -y_p2)
            y_max = min(self.n - y_p1, self.n - y_p2)
            
            if y_min > y_max + self.case_config.range_tolerance:
                self.logger.debug(f"Invalid y range: {y_min:.6f} > {y_max:.6f}")
                return None
                
            return ((xO, xO), (y_min, y_max))

    def _verify_solution(self,
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       phi: float,
                       p1: int,
                       p2: int,
                       opp_idx: int,
                       rect_edge: str) -> bool:
        """完整验证解的有效性
        
        Args:
            x_range: x坐标范围
            y_range: y坐标范围
            phi: 当前角度 (rad)
            p1: 基边顶点1索引
            p2: 基边顶点2索引
            opp_idx: 对角顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            bool: 是否为有效解
        """
        xO = sum(x_range) / 2
        yO = sum(y_range) / 2
        
        # 1. 验证基边顶点
        for pi in [p1, p2]:
            x = xO + self.t[pi] * math.cos(phi + self.theta[pi])
            y = yO + self.t[pi] * math.sin(phi + self.theta[pi])
            
            if rect_edge == 'bottom':
                valid = abs(y) <= self.config.tol
            elif rect_edge == 'top':
                valid = abs(y - self.n) <= self.config.tol
            elif rect_edge == 'left':
                valid = abs(x) <= self.config.tol
            else:  # 'right'
                valid = abs(x - self.m) <= self.config.tol
                
            if not valid:
                self.logger.debug(f"P{pi} not on {rect_edge} edge")
                return False
                
        # 2. 验证对角顶点
        x_opp = xO + self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
        y_opp = yO + self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
        
        if rect_edge in ['bottom', 'top']:
            target = self.n if rect_edge == 'bottom' else 0
            opp_valid = abs(y_opp - target) <= self.config.tol
        else:
            target = self.m if rect_edge == 'left' else 0
            opp_valid = abs(x_opp - target) <= self.config.tol
            
        if not opp_valid:
            self.logger.debug(f"Opposite vertex not on target edge (deviation: {abs(target - (y_opp if rect_edge in ['bottom','top'] else x_opp)):.6f})")
            return False
            
        # 3. 验证所有顶点在场地内
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            
            if not (0-self.config.tol <= x <= self.m+self.config.tol and 
                   0-self.config.tol <= y <= self.n+self.config.tol):
                self.logger.debug(f"P{i} out of bounds: ({x:.6f}, {y:.6f})")
                return False
                
        return True

def _test_case2():
    """Case2Solver 测试函数"""
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    
    config = BaseSolverConfig(
        log_file="logs/case2_test.log",
        log_level="DEBUG"
    )
    case_config = Case2Config(range_tolerance=1e-5)
    
    solver = Case2Solver(
        t=[1.0, 1.0, math.sqrt(2)],
        theta=[0.0, math.pi/2, 3*math.pi/4],
        m=2.0,
        n=2.0,
        config=config,
        case_config=case_config
    )
    
    solutions = solver.solve()
    print(f"Found {len(solutions)} solutions")
    for sol in solutions:
        print(f"Solution: {sol}")

if __name__ == "__main__":
    _test_case2()