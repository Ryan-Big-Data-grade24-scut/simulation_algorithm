import math
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from .BaseSolver import BaseSolver, BaseSolverConfig

class Case2Solver(BaseSolver):
    def __init__(self, t: List[float], theta: List[float], 
                 m: float, n: float, config: BaseSolverConfig):
        super().__init__(t, theta, m, n, config)
        self.edges = ['bottom', 'top', 'left', 'right']
        
    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """实现抽象方法，返回统一格式的解"""
        solutions = []
        
        self.logger.info("\n=== Starting Case2Solver ===")
        self.logger.info(f"Parameters: t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
        
        for base_idx in range(3):
            p1, p2 = base_idx, (base_idx + 1) % 3
            opp_idx = (base_idx + 2) % 3
            
            self.logger.info(f"\n=== Trying base edge {base_idx} (points {p1} and {p2}) ===")
            
            for rect_edge in self.edges:
                self.logger.info(f"\n  Trying rectangle edge: {rect_edge}")
                
                try:
                    phi_candidates = self._solve_phi_equation(p1, p2, opp_idx, rect_edge)
                    self.logger.info(f"    Found {len(phi_candidates)} phi candidates")
                    
                    for phi in phi_candidates:
                        self.logger.info(f"\n      Testing phi = {phi:.8f} rad")
                        
                        position = self._compute_position(phi, p1, p2, opp_idx, rect_edge)
                        if position is None:
                            continue
                            
                        x_range, y_range = position
                        if self._verify_solution(x_range, y_range, phi, p1, p2, opp_idx, rect_edge):
                            solutions.append((x_range, y_range, phi))
                            
                except Exception as e:
                    self.logger.error(f"    Error encountered: {str(e)}")
                    continue
        
        self.logger.info(f"\n=== Solution Summary ===")
        self.logger.info(f"Found {len(solutions)} valid solutions")
        
        for i, (x_range, y_range, phi) in enumerate(solutions, 1):
            self._print_solution((x_range, y_range, phi), i)
        
        return solutions

    def _solve_phi_equation(self, p1: int, p2: int, opp_idx: int, rect_edge: str) -> List[float]:
        """求解phi方程，返回候选解列表"""
        self.logger.debug(f"\n    [Solving phi equation for rect_edge={rect_edge}]")
        
        # 确定目标值和方程类型
        if rect_edge in ['bottom', 'top']:
            target = self.n if rect_edge == 'bottom' else -self.n
            equation_type = "A*sin(phi) + B*cos(phi) = target"
        else:
            target = self.m if rect_edge == 'left' else -self.m
            equation_type = "A*cos(phi) - B*sin(phi) = target"
        
        self.logger.debug(f"    Equation type: {equation_type}")
        self.logger.debug(f"    Target value: {target:.6f}")
        
        # 计算A和B系数
        A = (self.t[opp_idx] * math.cos(self.theta[opp_idx]) - 
             self.t[p1] * math.cos(self.theta[p1]))
        B = (self.t[opp_idx] * math.sin(self.theta[opp_idx]) - 
             self.t[p1] * math.sin(self.theta[p1]))
        
        self._log_math(f"A = t{opp_idx}*cos(theta{opp_idx}) - t{p1}*cos(theta{p1})", A)
        self._log_math(f"B = t{opp_idx}*sin(theta{opp_idx}) - t{p1}*sin(theta{p1})", B)
        
        norm = math.sqrt(A**2 + B**2)
        self._log_math(f"norm = sqrt(A² + B²)", norm)
        
        # 处理退化情况
        if norm < self.config.tol:
            self.logger.debug("    No solution: norm too small (degenerate case)")
            return []
        
        # 处理特殊解情况
        if abs(norm - abs(target)) < self.config.tol:
            self.logger.debug("    Special case: norm ≈ |target| (single solution)")
            if target > 0:
                phi = math.atan2(B, A) if rect_edge in ['bottom', 'top'] else math.atan2(B, -A)
            else:
                phi = math.atan2(-B, -A) if rect_edge in ['bottom', 'top'] else math.atan2(-B, A)
            
            self._log_math(f"phi = atan2 result", phi)
            return [phi]
        
        # 处理无解情况
        if norm < abs(target):
            self.logger.debug("    No solution: norm < |target|")
            return []
        
        # 一般情况：两个解
        self.logger.debug("    General case: two solutions")
        alpha = math.atan2(B, A)
        self._log_math(f"alpha = atan2(B, A)", alpha)
        
        if rect_edge in ['bottom', 'top']:
            phi1 = math.asin(target / norm) - alpha
            phi2 = math.pi - math.asin(target / norm) - alpha
            self._log_math(f"phi1 = asin({target/norm:.6f}) - alpha", phi1)
            self._log_math(f"phi2 = π - asin({target/norm:.6f}) - alpha", phi2)
        else:
            phi1 = math.acos(target / norm) - alpha
            phi2 = -math.acos(target / norm) - alpha
            self._log_math(f"phi1 = acos({target/norm:.6f}) - alpha", phi1)
            self._log_math(f"phi2 = -acos({target/norm:.6f}) - alpha", phi2)
        
        return [phi1, phi2]

    def _compute_position(self, phi: float, p1: int, p2: int, 
                         opp_idx: int, rect_edge: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """计算中心O的坐标范围，返回None如果无效"""
        self.logger.debug(f"\n    [Computing position for phi={phi:.6f}, rect_edge={rect_edge}]")
        
        if rect_edge in ['bottom', 'top']:
            # 水平基边情况
            y_target = 0 if rect_edge == 'bottom' else self.n
            yO = y_target - self.t[p1] * math.sin(phi + self.theta[p1])
            self._log_math(f"yO = {y_target} - t{p1}*sin(phi + theta{p1})", yO)
            
            # 计算x范围约束
            x_p1 = self.t[p1] * math.cos(phi + self.theta[p1])
            x_p2 = self.t[p2] * math.cos(phi + self.theta[p2])
            
            x_min = max(-x_p1, -x_p2)
            x_max = min(self.m - x_p1, self.m - x_p2)
            
            self.logger.debug(f"    xO range constraints:")
            self.logger.debug(f"      x_min = max({-x_p1:.6f}, {-x_p2:.6f}) = {x_min:.6f}")
            self.logger.debug(f"      x_max = min({self.m - x_p1:.6f}, {self.m - x_p2:.6f}) = {x_max:.6f}")
            
            if x_min > x_max + self.config.tol:
                self.logger.debug("    Invalid position: x_min > x_max")
                return None
                
            return ((x_min, x_max), (yO, yO))
            
        else:
            # 垂直基边情况
            x_target = 0 if rect_edge == 'left' else self.m
            xO = x_target - self.t[p1] * math.cos(phi + self.theta[p1])
            self._log_math(f"xO = {x_target} - t{p1}*cos(phi + theta{p1})", xO)
            
            # 计算y范围约束
            y_p1 = self.t[p1] * math.sin(phi + self.theta[p1])
            y_p2 = self.t[p2] * math.sin(phi + self.theta[p2])
            
            y_min = max(-y_p1, -y_p2)
            y_max = min(self.n - y_p1, self.n - y_p2)
            
            self.logger.debug(f"    yO range constraints:")
            self.logger.debug(f"      y_min = max({-y_p1:.6f}, {-y_p2:.6f}) = {y_min:.6f}")
            self.logger.debug(f"      y_max = min({self.n - y_p1:.6f}, {self.n - y_p2:.6f}) = {y_max:.6f}")
            
            if y_min > y_max + self.config.tol:
                self.logger.debug("    Invalid position: y_min > y_max")
                return None
                
            return ((xO, xO), (y_min, y_max))

    def _verify_solution(self, x_range: Tuple[float, float], 
                        y_range: Tuple[float, float], phi: float,
                        p1: int, p2: int, opp_idx: int, 
                        rect_edge: str) -> bool:
        """验证解的有效性"""
        self.logger.debug(f"\n    [Verifying solution for phi={phi:.6f}]")
        
        # 使用范围中点进行验证
        xO = (x_range[0] + x_range[1]) / 2
        yO = (y_range[0] + y_range[1]) / 2
        
        # 1. 验证基边顶点
        base_valid = True
        for pi in [p1, p2]:
            x_pi = xO + self.t[pi] * math.cos(phi + self.theta[pi])
            y_pi = yO + self.t[pi] * math.sin(phi + self.theta[pi])
            
            if rect_edge == 'bottom':
                if not self._log_compare(y_pi, 0):
                    base_valid = False
            elif rect_edge == 'top':
                if not self._log_compare(y_pi, self.n):
                    base_valid = False
            elif rect_edge == 'left':
                if not self._log_compare(x_pi, 0):
                    base_valid = False
            elif rect_edge == 'right':
                if not self._log_compare(x_pi, self.m):
                    base_valid = False
        
        self._log_validation("Base edge alignment", base_valid)
        if not base_valid:
            return False
        
        # 2. 验证对顶点
        x_opp = xO + self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
        y_opp = yO + self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
        
        opp_valid = False
        if rect_edge in ['bottom', 'top']:
            target = self.n if rect_edge == 'bottom' else 0
            opp_valid = self._log_compare(y_opp, target)
        else:
            target = self.m if rect_edge == 'left' else 0
            opp_valid = self._log_compare(x_opp, target)
        
        self._log_validation("Opposite vertex alignment", opp_valid)
        if not opp_valid:
            return False
        
        # 3. 验证所有顶点在矩形内
        all_in_bounds = True
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            
            x_ok = (-self.config.tol <= x <= self.m + self.config.tol)
            y_ok = (-self.config.tol <= y <= self.n + self.config.tol)
            
            if not (x_ok and y_ok):
                all_in_bounds = False
        
        self._log_validation("All vertices in bounds", all_in_bounds)
        
        return all_in_bounds