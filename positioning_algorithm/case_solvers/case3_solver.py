from .BaseSolver import BaseSolver, BaseSolverConfig
import numpy as np
import math
from math import sin, cos, atan2, asin, sqrt, pi
from typing import List, Tuple
from abc import ABC, abstractmethod
import logging

class Case3Solver(BaseSolver):
    def __init__(self, t: List[float], theta: List[float], 
                 m: float, n: float, config: BaseSolverConfig):
        super().__init__(t, theta, m, n, config)
        self._validate_case3_inputs()
        
    def _validate_case3_inputs(self):
        """验证情况3特有的输入条件"""
        pass  # 使用基类的通用验证
    
    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """处理所有情况3的组合，返回统一格式的解"""
        solutions = []
        
        self.logger.info("\n=== Starting Case3Solver ===")
        self.logger.info(f"Parameters: t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
        
        # 所有可能的边组合（顶点索引分配）
        edge_combinations = [
            ('left', 'right', 'top'),
            ('left', 'right', 'bottom'),
            ('left', 'top', 'bottom'),
            ('right', 'top', 'bottom')
        ]
        
        for combo_idx, (p0_edge, p1_edge, p2_edge) in enumerate(edge_combinations, 1):
            self.logger.info(f"\n=== Trying edge combination {combo_idx}: {p0_edge}, {p1_edge}, {p2_edge} ===")
            
            # 尝试所有顶点排列组合（6种排列）
            for p0, p1, p2 in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                self.logger.info(f"\n  Trying vertex permutation: P0={p0}, P1={p1}, P2={p2}")
                
                try:
                    phi_candidates = self._solve_edge_combo(
                        p0, p1, p2, p0_edge, p1_edge, p2_edge)
                    self.logger.info(f"  Found {len(phi_candidates)} phi candidates")
                    
                    for phi in phi_candidates:
                        self.logger.info(f"\n    Testing phi = {phi:.8f} radians ({np.degrees(phi):.2f}°)")
                        
                        xO, yO = self._compute_position(phi, p0, p1, p2, 
                                                      p0_edge, p1_edge, p2_edge)
                        self.logger.info(f"    Computed position: xO={xO:.8f}, yO={yO:.8f}")
                        
                        if np.isnan(xO) or np.isnan(yO):
                            self.logger.warning("    Invalid position (NaN), skipping")
                            continue
                            
                        if self._verify_solution(xO, yO, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
                            self.logger.info("    *** Solution is valid! ***")
                            # 转换为统一格式 ((xmin,xmax), (ymin,ymax), phi)
                            solutions.append(((xO, xO), (yO, yO), phi))
                        else:
                            self.logger.warning("    Solution failed verification")
                            
                except Exception as e:
                    self.logger.error(f"    Error encountered: {str(e)}", exc_info=True)
                    continue
        
        self.logger.info(f"\n=== Solution Summary ===")
        self.logger.info(f"Found {len(solutions)} valid solutions")
        
        for i, (x_range, y_range, phi) in enumerate(solutions, 1):
            self._print_solution((x_range, y_range, phi), i)
            
        return solutions
    
    def _solve_edge_combo(self, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """解特定边组合的方程（保持原有逻辑）"""
        self.logger.info(f"\n    [Solving phi equation for edge combination]")
        self.logger.info(f"    P0 on {p0_edge}, P1 on {p1_edge}, P2 on {p2_edge}")
        
        if {p0_edge, p1_edge} == {'left', 'right'}:
            self.logger.info("    Case: Left + Right + Top/Bottom")
            A = self.t[p1]*cos(self.theta[p1]) - self.t[p0]*cos(self.theta[p0])
            B = -(self.t[p1]*sin(self.theta[p1]) - self.t[p0]*sin(self.theta[p0]))
            C = self.m
            equation_type = "A*cos(phi) + B*sin(phi) = C (left-right case)"
            self._log_math(f"A = {self.t[p1]}*cos({self.theta[p1]}) - {self.t[p0]}*cos({self.theta[p0]})", A)
            self._log_math(f"B = -({self.t[p1]}*sin({self.theta[p1]}) - {self.t[p0]}*sin({self.theta[p0]}))", B)
        elif {p0_edge, p1_edge} == {'left', 'top'} and p2_edge == 'bottom':
            self.logger.info("    Case: Left + Top + Bottom")
            A = self.t[p2]*sin(self.theta[p2]) - self.t[p1]*sin(self.theta[p1])  # 使用P2和P1
            B = self.t[p2]*cos(self.theta[p2]) - self.t[p1]*cos(self.theta[p1])  # 使用P2和P1
            C = self.n
            equation_type = "A*cos(phi) + B*sin(phi) = C (left-top-bottom case)"
            self._log_math(f"A = {self.t[p2]}*sin({self.theta[p2]}) - {self.t[p1]}*sin({self.theta[p1]})", A)
            self._log_math(f"B = {self.t[p2]}*cos({self.theta[p2]}) - {self.t[p1]}*cos({self.theta[p1]})", B)
        elif {p0_edge, p1_edge} == {'right', 'top'} and p2_edge == 'bottom':
            self.logger.info("    Case: Right + Top + Bottom")
            A = self.t[p2]*sin(self.theta[p2]) - self.t[p1]*sin(self.theta[p1])  # 使用P2和P1
            B = self.t[p2]*cos(self.theta[p2]) - self.t[p1]*cos(self.theta[p1])  # 使用P2和P1
            C = self.n
            equation_type = "A*cos(phi) + B*sin(phi) = C (right-top-bottom case)"
            self._log_math(f"A = {self.t[p2]}*sin({self.theta[p2]}) -{ self.t[p1]}*sin({self.theta[p1]})", A)
            self._log_math(f"B = {self.t[p2]}*cos({self.theta[p2]}) - {self.t[p1]}*cos({self.theta[p1]})", B)
        else:
            self.logger.info("    Unsupported edge combination, skipping")
            return []
        
        self._log_math(f"C", C)
        self.logger.info(f"    Equation to solve: {equation_type}")
        
        # 解 A*cos(phi) + B*sin(phi) = C
        norm = sqrt(A*A + B*B)
        self._log_math(f"norm = sqrt(A² + B²)", norm)
        
        if abs(C) > norm + self.config.tol:
            self.logger.debug("    No solution: |C| > norm")
            return []  # 无解
        
        alpha = atan2(A, B)
        self._log_math(f"alpha = atan2(A, B)", alpha)
        
        phi1 = asin(C / norm) - alpha
        phi2 = pi - asin(C / norm) - alpha
        self._log_math(f"phi1 = asin({C/norm:.6f}) - alpha", phi1)
        self._log_math(f"phi2 = π - asin({C/norm:.6f}) - alpha", phi2)
        
        # 返回在[-pi, pi]范围内的解
        solutions = []
        for phi in [phi1, phi2]:
            if phi < -pi:
                phi += 2*pi
            elif phi > pi:
                phi -= 2*pi
            solutions.append(phi)
        
        self.logger.debug(f"    Final phi candidates after range adjustment: {solutions}")
        return solutions
    
    def _compute_position(self, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """计算中心点坐标（保持原有逻辑）"""
        self.logger.debug(f"\n    [Computing position for phi={phi:.6f}]")
        xO, yO = None, None
        x_values = []
        y_values = []
        
        def process_edge(vertex, edge, step_name):
            nonlocal xO, yO
            if edge == 'left':
                new_x = -self.t[vertex] * cos(phi + self.theta[vertex])
                self._log_math(f"{step_name}: xO = -t{vertex}*cos(phi + theta{vertex})", new_x)
                x_values.append(new_x)
            elif edge == 'right':
                new_x = self.m - self.t[vertex] * cos(phi + self.theta[vertex])
                self._log_math(f"{step_name}: xO = m - t{vertex}*cos(phi + theta{vertex})", new_x)
                x_values.append(new_x)
            elif edge == 'top':
                new_y = self.n - self.t[vertex] * sin(phi + self.theta[vertex])
                self._log_math(f"{step_name}: yO = n - t{vertex}*sin(phi + theta{vertex})", new_y)
                y_values.append(new_y)
            elif edge == 'bottom':
                new_y = -self.t[vertex] * sin(phi + self.theta[vertex])
                self._log_math(f"{step_name}: yO = -t{vertex}*sin(phi + theta{vertex})", new_y)
                y_values.append(new_y)
        
        process_edge(p0, p0_edge, "Initial calculation from P0")
        process_edge(p1, p1_edge, "Adjustment from P1")
        process_edge(p2, p2_edge, "Fine-tuning from P2")
        
        if len(x_values) > 1:
            xO = sum(x_values) / len(x_values)
            self.logger.debug(f"    Averaged xO from {len(x_values)} values: {xO:.6f}")
        elif len(x_values) == 1:
            xO = x_values[0]
            self.logger.debug(f"    Using single xO value: {xO:.6f}")
        
        if len(y_values) > 1:
            yO = sum(y_values) / len(y_values)
            self.logger.debug(f"    Averaged yO from {len(y_values)} values: {yO:.6f}")
        elif len(y_values) == 1:
            yO = y_values[0]
            self.logger.debug(f"    Using single yO value: {yO:.6f}")
        
        if xO is None:
            xO = 0
            self.logger.debug("    No x constraints, defaulting xO to 0")
        if yO is None:
            yO = 0
            self.logger.debug("    No y constraints, defaulting yO to 0")
        
        self.logger.debug(f"\n    Final position: xO={xO:.8f}, yO={yO:.8f}")
        return xO, yO
    
    def _verify_solution(self, xO, yO, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """验证所有顶点是否在指定边且不越界（保持原有逻辑）"""
        self.logger.debug(f"\n    [Verifying solution for phi={phi:.6f}]")
        self.logger.debug(f"    P0 on {p0_edge}, P1 on {p1_edge}, P2 on {p2_edge}")
        
        points = []
        for i in [p0, p1, p2]:
            x = xO + self.t[i] * cos(phi + self.theta[i])
            y = yO + self.t[i] * sin(phi + self.theta[i])
            points.append((x, y))
            self.logger.debug(f"    P{i} = ({x:.8f}, {y:.8f})")
        
        edge_checks = {
            'left': lambda x, y: (abs(x) < self.config.tol, f"P on left edge: x={x:.6f} ≈ 0"),
            'right': lambda x, y: (abs(x - self.m) < self.config.tol, f"P on right edge: x={x:.6f} ≈ {self.m}"),
            'top': lambda x, y: (abs(y - self.n) < self.config.tol, f"P on top edge: y={y:.6f} ≈ {self.n}"),
            'bottom': lambda x, y: (abs(y) < self.config.tol, f"P on bottom edge: y={y:.6f} ≈ 0")
        }
        
        edge_constraints = {
            'left': lambda x, y: (0 <= y <= self.n + self.config.tol, f"P on left edge within y bounds [0, {self.n}]"),
            'right': lambda x, y: (0 <= y <= self.n + self.config.tol, f"P on right edge within y bounds [0, {self.n}]"),
            'top': lambda x, y: (0 <= x <= self.m + self.config.tol, f"P on top edge within x bounds [0, {self.m}]"),
            'bottom': lambda x, y: (0 <= x <= self.m + self.config.tol, f"P on bottom edge within x bounds [0, {self.m}]")
        }
        
        edges = [p0_edge, p1_edge, p2_edge]
        for (x, y), edge in zip(points, edges):
            on_edge, msg = edge_checks[edge](x, y)
            self.logger.debug(f"    {msg}")
            if not on_edge:
                self.logger.debug(f"    Vertex not on {edge} edge")
                return False
            
            in_bounds, msg = edge_constraints[edge](x, y)
            self.logger.debug(f"    {msg}")
            if not in_bounds:
                self.logger.debug(f"    Vertex out of bounds on {edge} edge")
                return False
        
        all_edges = {'left', 'right', 'top', 'bottom'}
        used_edges = set(edges)
        free_edge = list(all_edges - used_edges)[0]
        self.logger.debug(f"\n    Checking no vertex is on free edge: {free_edge}")
        
        for (x, y), i in zip(points, [p0, p1, p2]):
            on_free_edge, _ = edge_checks[free_edge](x, y)
            if on_free_edge:
                self.logger.debug(f"    P{i} is incorrectly on {free_edge} edge")
                return False
        
        self.logger.debug("\n    Checking all vertices within rectangle bounds")
        for (x, y), i in zip(points, [p0, p1, p2]):
            x_ok = (-self.config.tol <= x <= self.m + self.config.tol)
            y_ok = (-self.config.tol <= y <= self.n + self.config.tol)
            self.logger.debug(f"    P{i}: x in [0, {self.m}]? {'YES' if x_ok else 'NO'}")
            self.logger.debug(f"    P{i}: y in [0, {self.n}]? {'YES' if y_ok else 'NO'}")
            if not (x_ok and y_ok):
                self.logger.debug(f"    P{i} out of rectangle bounds")
                return False
        
        self.logger.debug("    *** All verifications passed ***")
        return True