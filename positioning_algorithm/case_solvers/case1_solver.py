from .BaseSolver import BaseSolver, BaseSolverConfig
from typing import List, Tuple, Optional
import math

class Case1Solver(BaseSolver):
    """Solver for case where one edge is on rectangle edge and opposite vertex is on adjacent edge"""
    
    def __init__(self, t: List[float], theta: List[float], m: float, n: float, config: Optional[BaseSolverConfig] = None):
        # 使用默认配置如果未提供
        if config is None:
            config = BaseSolverConfig()
        super().__init__(t, theta, m, n, config)
        
        # 预计算所有可能的边组合
        self.edge_combinations = [
            ('bottom', 'left'), ('bottom', 'right'),
            ('top', 'left'), ('top', 'right'),
            ('left', 'bottom'), ('left', 'top'),
            ('right', 'bottom'), ('right', 'top')
        ]
    
    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        solutions = []
        self.logger.info(f"\nStarting solve with t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
        
        for base_idx in range(3):
            p1_idx = base_idx
            p2_idx = (base_idx + 1) % 3
            opp_idx = (base_idx + 2) % 3
            
            self.logger.info(f"\nTrying base edge {base_idx} (points {p1_idx} and {p2_idx})")
            
            for rect_edge, adj_edge in self.edge_combinations:
                self.logger.info(f"\n  Trying rectangle edge: {rect_edge}, adjacent edge: {adj_edge}")
                
                try:
                    phi_candidates = self._solve_phi(base_idx, rect_edge, adj_edge)
                    
                    for phi in phi_candidates:
                        self.logger.info(f"    Found phi: {phi:.8f} radians")
                        
                        xO, yO = self._compute_position(base_idx, rect_edge, adj_edge, phi)
                        self.logger.info(f"    Computed position: xO={xO:.8f}, yO={yO:.8f}")
                        
                        if self._verify_solution(xO, yO, phi, base_idx, rect_edge, adj_edge):
                            self.logger.info("    Solution is valid!")
                            # 转换为基类要求的格式 ((xmin,xmax), (ymin,ymax), phi)
                            solutions.append(((xO, xO), (yO, yO), phi))
                        else:
                            self.logger.info("    Solution failed verification")
                            
                except (ValueError, AssertionError) as e:
                    self.logger.warning(f"    Failed with error: {str(e)}")
                    continue
        
        self.logger.info(f"\nFound {len(solutions)} valid solutions")
        for i, solution in enumerate(solutions, 1):
            self._print_solution(solution, i)
            
        return solutions
    
    def _solve_phi(self, base_idx: int, rect_edge: str, adj_edge: str) -> List[float]:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        valid_phis = []
        # 水平边处理
        if rect_edge in ['bottom', 'top']:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            B = t1 * math.sin(theta1) - t2 * math.sin(theta2)
            self._log_math(f"A = {t1}*cos({theta1}) - {t2}*cos({theta2})", A)
            self._log_math(f"B = {t1}*sin({theta1}) - {t2}*sin({theta2})", B)

            if abs(A) < self.config.tol and abs(B) < self.config.tol:
                self.logger.info("    Degenerate case (A and B both zero)")
                return []
                
            elif abs(A) < self.config.tol:
                self.logger.info("    Special case (A≈0): direct solution Phi=+-pi/2")
                base_phi = math.copysign(math.pi/2, -B)
                candidates = [base_phi, base_phi + math.pi]
                
            else:
                base_phi = math.atan2(-B, A)
                candidates = [base_phi, base_phi + math.pi]
            self.logger.info(f"A:{A}, B:{B}")
            return self._validate_phi_candidates(
                candidates=candidates,
                base_idx=base_idx,
                rect_edge=rect_edge,
                adj_edge=adj_edge,
                is_vertical=False
            )
        
        # 垂直边处理
        else:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t2 * math.sin(theta2) - t1 * math.sin(theta1)
            B = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            self._log_math(f"A = {t2}*sin({theta2}) - {t1}*sin({theta1})", A)
            self._log_math(f"B = {t1}*cos({theta1}) - {t2}*cos({theta2})", B)

            if abs(A) < self.config.tol and abs(B) < self.config.tol:
                self.logger.info("    Degenerate case (A and B both zero)")
                return []
                
            elif abs(A) < self.config.tol:
                self.logger.info("    Special case (A≈0): direct solution Phi=0 or pi")
                base_phi = 0 if B > 0 else math.pi
                candidates = [base_phi, base_phi + math.pi]
                
            else:
                base_phi = math.atan2(B, -A)
                candidates = [base_phi, base_phi + math.pi]
            self.logger.info(f"A:{A}, B:{B}")
            return self._validate_phi_candidates(
                candidates=candidates,
                base_idx=base_idx,
                rect_edge=rect_edge,
                adj_edge=adj_edge,
                is_vertical=True
            )
    
    def _validate_phi_candidates(self, candidates: List[float], base_idx: int, 
                               rect_edge: str, adj_edge: str, is_vertical: bool) -> List[float]:
        valid_phis = []
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3

        for phi in candidates:
            try:
                self.logger.info(f"\n=== Validating phi = {phi:.8f} ===")
                
                # 计算相对坐标
                P = {}
                for i in [p1_idx, p2_idx, opp_idx]:
                    x = self.t[i] * math.cos(phi + self.theta[i])
                    y = self.t[i] * math.sin(phi + self.theta[i])
                    P[i] = (x, y)
                    self.logger.info(f"P{i} = ({x:.8f}, {y:.8f})")
                
                # 1. 基边对齐检查
                if is_vertical:
                    dx = abs(P[p2_idx][0] - P[p1_idx][0])
                    self._log_compare(dx, 0.0)
                    if dx >= self.config.tol:
                        self.logger.info("-> Base not vertical, skip")
                        continue
                    
                    base_x = (P[p1_idx][0] + P[p2_idx][0]) / 2
                    if rect_edge == 'left':
                        cond = P[opp_idx][0] > base_x - self.config.tol
                    else:
                        cond = P[opp_idx][0] < base_x + self.config.tol
                else:
                    dy = abs(P[p2_idx][1] - P[p1_idx][1])
                    self._log_compare(dy, 0.0)
                    if dy >= self.config.tol:
                        self.logger.info("-> Base not horizontal, skip")
                        continue
                    
                    base_y = (P[p1_idx][1] + P[p2_idx][1]) / 2
                    if rect_edge == 'bottom':
                        cond = P[opp_idx][1] > base_y - self.config.tol
                    else:
                        cond = P[opp_idx][1] < base_y + self.config.tol
                
                self._log_validation("Base edge alignment", cond)
                if not cond:
                    continue
                
                # 2. 邻边约束检查
                if adj_edge == 'left':
                    cond = P[opp_idx][0] <= min(P[p1_idx][0], P[p2_idx][0]) + self.config.tol
                elif adj_edge == 'right':
                    cond = P[opp_idx][0] >= max(P[p1_idx][0], P[p2_idx][0]) - self.config.tol
                elif adj_edge == 'bottom':
                    cond = P[opp_idx][1] <= min(P[p1_idx][1], P[p2_idx][1]) + self.config.tol
                elif adj_edge == 'top':
                    cond = P[opp_idx][1] >= max(P[p1_idx][1], P[p2_idx][1]) - self.config.tol
                
                self._log_validation("Adjacent edge constraint", cond)
                if not cond:
                    continue
                
                # 3. 平移可行性检查
                all_points = list(P.values())
                min_x = min(p[0] for p in all_points)
                max_x = max(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_y = max(p[1] for p in all_points)
                
                width_ok = (max_x - min_x) <= (self.m + self.config.tol)
                height_ok = (max_y - min_y) <= (self.n + self.config.tol)
                
                self._log_validation("Width constraint", width_ok)
                self._log_validation("Height constraint", height_ok)
                
                if width_ok and height_ok:
                    self.logger.info("*** ALL VALIDATIONS PASSED ***")
                    valid_phis.append(phi)
                    
            except Exception as e:
                self.logger.error(f"Validation error: {str(e)}")
                continue
        
        return valid_phis
    
    def _compute_position(self, base_idx: int, rect_edge: str, adj_edge: str, phi: float) -> Tuple[float, float]:
        p1_idx = base_idx
        opp_idx = (base_idx + 2) % 3

        if rect_edge in ['bottom', 'top']:
            # yO计算
            y_target = 0 if rect_edge == 'bottom' else self.n
            yO = y_target - self.t[p1_idx] * math.sin(phi + self.theta[p1_idx])
            self._log_math(f"yO = {y_target} - t{p1_idx}*sin(phi + theta{p1_idx})", yO)
            
            # xO计算
            x_target = 0 if adj_edge == 'left' else self.m
            x_opp = self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
            self._log_math(f"x_opp = t{opp_idx}*cos(phi + theta{opp_idx})", x_opp)
            
            xO = x_target - x_opp
            self._log_math(f"xO = {x_target} - x_opp", xO)
        else:
            # xO计算
            x_target = 0 if rect_edge == 'left' else self.m
            xO = x_target - self.t[p1_idx] * math.cos(phi + self.theta[p1_idx])
            self._log_math(f"xO = {x_target} - t{p1_idx}*cos(phi + theta{p1_idx})", xO)
            
            # yO计算
            y_target = 0 if adj_edge == 'bottom' else self.n
            y_opp = self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
            self._log_math(f"y_opp = t{opp_idx}*sin(phi + theta{opp_idx})", y_opp)
            
            yO = y_target - y_opp
            self._log_math(f"yO = {y_target} - y_opp", yO)

        return xO, yO
    
    def _verify_solution(self, xO: float, yO: float, phi: float, 
                        base_idx: int, rect_edge: str, adj_edge: str) -> bool:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        self.logger.info(f"\nVerifying solution: xO={xO:.8f}, yO={yO:.8f}, phi={phi:.8f}")
        
        # 计算所有顶点位置
        P = {}
        for i in [p1_idx, p2_idx, opp_idx]:
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            P[i] = (x, y)
            self.logger.info(f"P{i} = ({x:.8f}, {y:.8f})")
        
        # 1. 顶点边界检查
        bounds_ok = True
        for i, (x, y) in P.items():
            x_ok = (-self.config.tol <= x <= self.m + self.config.tol)
            y_ok = (-self.config.tol <= y <= self.n + self.config.tol)
            self._log_validation(f"P{i} x bounds", x_ok)
            self._log_validation(f"P{i} y bounds", y_ok)
            bounds_ok = bounds_ok and x_ok and y_ok
        
        if not bounds_ok:
            return False
        
        # 2. 基边严格对齐
        if rect_edge == 'bottom':
            edge_ok = (abs(P[p1_idx][1]) <= self.config.tol and 
                      abs(P[p2_idx][1]) <= self.config.tol)
        elif rect_edge == 'top':
            edge_ok = (abs(P[p1_idx][1] - self.n) <= self.config.tol and 
                      abs(P[p2_idx][1] - self.n) <= self.config.tol)
        elif rect_edge == 'left':
            edge_ok = (abs(P[p1_idx][0]) <= self.config.tol and 
                      abs(P[p2_idx][0]) <= self.config.tol)
        elif rect_edge == 'right':
            edge_ok = (abs(P[p1_idx][0] - self.m) <= self.config.tol and 
                      abs(P[p2_idx][0] - self.m) <= self.config.tol)
        
        self._log_validation("Base edge alignment", edge_ok)
        return edge_ok