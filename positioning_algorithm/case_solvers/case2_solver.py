import math
from typing import List, Tuple

class Case2Solver:
    def __init__(self, t: List[float], theta: List[float], m: float, n: float):
        """
        t: 三角形各顶点到中心O的距离 [t0, t1, t2]
        theta: 各顶点相对于O的初始角度 [θ0, θ1, θ2] (弧度)
        m: 矩形宽度
        n: 矩形高度
        """
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.TOL = 1e-6
        self.edges = ['bottom', 'top', 'left', 'right']
        self.adj_edges = {
            'bottom': ['left', 'right'],
            'top': ['left', 'right'],
            'left': ['bottom', 'top'],
            'right': ['bottom', 'top']
        }

    def solve(self) -> List[Tuple[float, float, float]]:
        """返回所有有效解 (xO, yO, phi) 的列表"""
        solutions = []
        for base_idx in range(3):
            p1, p2 = base_idx, (base_idx + 1) % 3
            opp_idx = (base_idx + 2) % 3
            
            for rect_edge in self.edges:
                for adj_edge in self.adj_edges[rect_edge]:
                    phi_candidates = self._solve_phi_equation(p1, p2, opp_idx, rect_edge)
                    for phi in phi_candidates:
                        xO, yO = self._compute_position(phi, p1, p2, opp_idx, rect_edge, adj_edge)
                        if self._verify_solution(xO, yO, phi, p1, p2, opp_idx, rect_edge, adj_edge):
                            solutions.append((xO, yO, phi))
        return solutions

    def _solve_phi_equation(self, p1: int, p2: int, opp_idx: int, rect_edge: str) -> List[float]:
        """解角度方程，返回phi候选值列表"""
        if rect_edge in ['bottom', 'top']:
            # 水平基边情况
            # 约束1: 基边两个顶点在水平边 (y=0或y=n)
            # 约束2: 对角顶点在对边 (顶边或底边)
            
            # 方程形式: t_opp*sin(phi+θ_opp) - t_p1*sin(phi+θ_p1) = target
            target = self.n if rect_edge == 'bottom' else -self.n
            
            A = (self.t[opp_idx] * math.cos(self.theta[opp_idx]) - 
                 self.t[p1] * math.cos(self.theta[p1]))
            B = (self.t[opp_idx] * math.sin(self.theta[opp_idx]) - 
                 self.t[p1] * math.sin(self.theta[p1]))
        else:
            # 垂直基边情况
            # 约束1: 基边两个顶点在垂直边 (x=0或x=m)
            # 约束2: 对角顶点在对边 (左边或右边)
            
            # 方程形式: t_opp*cos(phi+θ_opp) - t_p1*cos(phi+θ_p1) = target
            target = self.m if rect_edge == 'left' else -self.m
            
            A = (self.t[opp_idx] * math.sin(self.theta[opp_idx]) - 
                 self.t[p1] * math.sin(self.theta[p1]))
            B = (self.t[opp_idx] * math.cos(self.theta[opp_idx]) - 
                 self.t[p1] * math.cos(self.theta[p1]))

        # 解方程 A*sin(phi) + B*cos(phi) = target (水平) 或 A*cos(phi) - B*sin(phi) = target (垂直)
        norm = math.sqrt(A**2 + B**2)
        if norm < self.TOL:
            return []  # 无解
        
        if abs(norm - abs(target)) < self.TOL:
            # 特殊情况：norm ≈ |target|
            if target > 0:
                phi = math.atan2(B, A) if rect_edge in ['bottom', 'top'] else math.atan2(B, -A)
            else:
                phi = math.atan2(-B, -A) if rect_edge in ['bottom', 'top'] else math.atan2(-B, A)
            return [phi]
        elif norm < abs(target):
            return []  # 无解
        else:
            # 一般情况：两个解
            alpha = math.atan2(B, A)
            if rect_edge in ['bottom', 'top']:
                phi1 = math.asin(target / norm) - alpha
                phi2 = math.pi - math.asin(target / norm) - alpha
            else:
                phi1 = math.acos(target / norm) - alpha
                phi2 = -math.acos(target / norm) - alpha
            return [phi1, phi2]

    def _compute_position(self, phi: float, p1: int, p2: int, opp_idx: int, 
                         rect_edge: str, adj_edge: str) -> Tuple[float, float]:
        """计算中心O的坐标(xO, yO)"""
        if rect_edge in ['bottom', 'top']:
            # 水平基边情况
            # 1. 计算yO：使基边在y=0或y=n上
            y_target = 0 if rect_edge == 'bottom' else self.n
            yO = y_target - self.t[p1] * math.sin(phi + self.theta[p1])
            
            # 2. 计算xO：使对角顶点在对边，并确保基边顶点在矩形内
            # 基边两个顶点P1和P2的x坐标：
            x_p1 = self.t[p1] * math.cos(phi + self.theta[p1])
            x_p2 = self.t[p2] * math.cos(phi + self.theta[p2])
            
            # 确定xO范围
            x_min = max(-x_p1, -x_p2)  # 确保xO + x_pi >= 0
            x_max = min(self.m - x_p1, self.m - x_p2)  # 确保xO + x_pi <= m
            
            if x_min > x_max + self.TOL:
                return (math.nan, math.nan)  # 无效位置
            
            # 根据邻边约束调整xO
            if adj_edge == 'left':
                xO = -min(x_p1, x_p2)
            elif adj_edge == 'right':
                xO = self.m - max(x_p1, x_p2)
            else:
                xO = (x_min + x_max) / 2
        else:
            # 垂直基边情况
            # 1. 计算xO：使基边在x=0或x=m上
            x_target = 0 if rect_edge == 'left' else self.m
            xO = x_target - self.t[p1] * math.cos(phi + self.theta[p1])
            
            # 2. 计算yO：使对角顶点在对边，并确保基边顶点在矩形内
            # 基边两个顶点P1和P2的y坐标：
            y_p1 = self.t[p1] * math.sin(phi + self.theta[p1])
            y_p2 = self.t[p2] * math.sin(phi + self.theta[p2])
            
            # 确定yO范围
            y_min = max(-y_p1, -y_p2)  # 确保yO + y_pi >= 0
            y_max = min(self.n - y_p1, self.n - y_p2)  # 确保yO + y_pi <= n
            
            if y_min > y_max + self.TOL:
                return (math.nan, math.nan)  # 无效位置
            
            # 根据邻边约束调整yO
            if adj_edge == 'bottom':
                yO = -min(y_p1, y_p2)
            elif adj_edge == 'top':
                yO = self.n - max(y_p1, y_p2)
            else:
                yO = (y_min + y_max) / 2
        
        return (xO, yO)

    def _verify_solution(self, xO: float, yO: float, phi: float, 
                        p1: int, p2: int, opp_idx: int, 
                        rect_edge: str, adj_edge: str) -> bool:
        """验证解的有效性"""
        if math.isnan(xO) or math.isnan(yO):
            return False
        
        # 1. 检查基边两个顶点在指定边
        for pi in [p1, p2]:
            x_pi = xO + self.t[pi] * math.cos(phi + self.theta[pi])
            y_pi = yO + self.t[pi] * math.sin(phi + self.theta[pi])
            
            if rect_edge == 'bottom':
                if abs(y_pi) > self.TOL:
                    return False
            elif rect_edge == 'top':
                if abs(y_pi - self.n) > self.TOL:
                    return False
            elif rect_edge == 'left':
                if abs(x_pi) > self.TOL:
                    return False
            elif rect_edge == 'right':
                if abs(x_pi - self.m) > self.TOL:
                    return False
        
        # 2. 检查对角顶点在对边
        x_opp = xO + self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
        y_opp = yO + self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
        
        if rect_edge in ['bottom', 'top']:
            target_edge = 'top' if rect_edge == 'bottom' else 'bottom'
            if target_edge == 'top':
                if abs(y_opp - self.n) > self.TOL:
                    return False
            else:
                if abs(y_opp) > self.TOL:
                    return False
        else:
            target_edge = 'right' if rect_edge == 'left' else 'left'
            if target_edge == 'right':
                if abs(x_opp - self.m) > self.TOL:
                    return False
            else:
                if abs(x_opp) > self.TOL:
                    return False
        
        # 3. 检查邻边约束
        x_adj = xO + self.t[(opp_idx + 1) % 3] * math.cos(phi + self.theta[(opp_idx + 1) % 3])
        y_adj = yO + self.t[(opp_idx + 1) % 3] * math.sin(phi + self.theta[(opp_idx + 1) % 3])
        
        if adj_edge == 'left':
            if x_adj > self.TOL:
                return False
        elif adj_edge == 'right':
            if x_adj < self.m - self.TOL:
                return False
        elif adj_edge == 'bottom':
            if y_adj > self.TOL:
                return False
        elif adj_edge == 'top':
            if y_adj < self.n - self.TOL:
                return False
        
        # 4. 检查所有顶点在矩形内
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            if x < -self.TOL or x > self.m + self.TOL or y < -self.TOL or y > self.n + self.TOL:
                return False
        
        return True