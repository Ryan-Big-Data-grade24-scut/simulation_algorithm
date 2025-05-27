#./positioning_algorithm/case_solvers/case2_solver.py
import math
from typing import List, Tuple

class Case2Solver:
    """Solver for case where one edge is on rectangle edge and opposite vertex is on opposite edge"""
    
    def __init__(self, t: List[float], theta: List[float], m: float, n: float):
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
    
    def solve(self) -> List[Tuple[float, float, float]]:
        solutions = []
        
        # Try all 3 edges as possible base edges
        for base_idx in range(3):
            p1_idx = base_idx
            p2_idx = (base_idx + 1) % 3
            opp_idx = (base_idx + 2) % 3
            
            # Try all 4 rectangle edges for base
            for rect_edge in ['bottom', 'top', 'left', 'right']:
                try:
                    phi = self._solve_phi(base_idx, rect_edge)
                    xO, yO = self._compute_position(base_idx, rect_edge, phi)
                    
                    if self._verify_solution(xO, yO, phi, base_idx, rect_edge):
                        solutions.append((xO, yO, phi))
                except (ValueError, AssertionError):
                    continue
        
        return solutions
    
    def _solve_phi(self, base_idx: int, rect_edge: str) -> float:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        if rect_edge in ['bottom', 'top']:
            # Base is on horizontal edge
            y_const = 0 if rect_edge == 'bottom' else self.n
            opp_const = self.n if rect_edge == 'bottom' else 0
            
            # Equations:
            # yO + t1*sin(phi+θ1) = y_const
            # yO + t2*sin(phi+θ2) = y_const
            # yO + t3*sin(phi+θ3) = opp_const
            
            # From first two equations:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            B = t1 * math.sin(theta1) - t2 * math.sin(theta2)
            
            if abs(A) < 1e-6 and abs(B) < 1e-6:
                raise ValueError("No unique solution for phi")
            
            phi = math.atan2(-B, A)
            
            # Verify opposite vertex constraint
            t3 = self.t[opp_idx]
            theta3 = self.theta[opp_idx]
            y_opp = t3 * math.sin(phi + theta3)
            if not math.isclose(y_opp, opp_const - (y_const - y_opp), abs_tol=1e-6):
                raise ValueError("Opposite vertex constraint not satisfied")
            
            return phi
            
        else:
            # Base is on vertical edge
            x_const = 0 if rect_edge == 'left' else self.m
            opp_const = self.m if rect_edge == 'left' else 0
            
            # Equations:
            # xO + t1*cos(phi+θ1) = x_const
            # xO + t2*cos(phi+θ2) = x_const
            # xO + t3*cos(phi+θ3) = opp_const
            
            # From first two equations:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t1 * math.sin(theta1) - t2 * math.sin(theta2)
            B = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            
            if abs(A) < 1e-6 and abs(B) < 1e-6:
                raise ValueError("No unique solution for phi")
            
            phi = math.atan2(B, -A)
            
            # Verify opposite vertex constraint
            t3 = self.t[opp_idx]
            theta3 = self.theta[opp_idx]
            x_opp = t3 * math.cos(phi + theta3)
            if not math.isclose(x_opp, opp_const - (x_const - x_opp), abs_tol=1e-6):
                raise ValueError("Opposite vertex constraint not satisfied")
            
            return phi
    
    def _compute_position(self, base_idx: int, rect_edge: str, phi: float) -> Tuple[float, float]:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        
        if rect_edge in ['bottom', 'top']:
            y_const = 0 if rect_edge == 'bottom' else self.n
            t1, theta1 = self.t[p1_idx], self.theta[p1_idx]
            yO = y_const - t1 * math.sin(phi + theta1)
            
            # xO must place both base vertices within [0, m]
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            xP1 = t1 * math.cos(phi + theta1)
            xP2 = t2 * math.cos(phi + theta2)
            
            xO_min = max(-xP1, -xP2)
            xO_max = min(self.m - xP1, self.m - xP2)
            
            if xO_min > xO_max + 1e-6:
                raise ValueError("No valid xO position")
            
            xO = (xO_min + xO_max) / 2
            return xO, yO
        else:
            x_const = 0 if rect_edge == 'left' else self.m
            t1, theta1 = self.t[p1_idx], self.theta[p1_idx]
            xO = x_const - t1 * math.cos(phi + theta1)
            
            # yO must place both base vertices within [0, n]
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            yP1 = t1 * math.sin(phi + theta1)
            yP2 = t2 * math.sin(phi + theta2)
            
            yO_min = max(-yP1, -yP2)
            yO_max = min(self.n - yP1, self.n - yP2)
            
            if yO_min > yO_max + 1e-6:
                raise ValueError("No valid yO position")
            
            yO = (yO_min + yO_max) / 2
            return xO, yO
    
    def _verify_solution(self, xO: float, yO: float, phi: float, 
                        base_idx: int, rect_edge: str) -> bool:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        # Compute all vertices
        P = []
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            P.append((x, y))
        
        # Check base edge constraints
        if rect_edge == 'bottom':
            if not (math.isclose(P[p1_idx][1], 0, abs_tol=1e-6) and 
                   math.isclose(P[p2_idx][1], 0, abs_tol=1e-6)):
                return False
            opp_const = self.n
        elif rect_edge == 'top':
            if not (math.isclose(P[p1_idx][1], self.n, abs_tol=1e-6) and 
                   math.isclose(P[p2_idx][1], self.n, abs_tol=1e-6)):
                return False
            opp_const = 0
        elif rect_edge == 'left':
            if not (math.isclose(P[p1_idx][0], 0, abs_tol=1e-6) and 
                   math.isclose(P[p2_idx][0], 0, abs_tol=1e-6)):
                return False
            opp_const = self.m
        elif rect_edge == 'right':
            if not (math.isclose(P[p1_idx][0], self.m, abs_tol=1e-6) and 
                   math.isclose(P[p2_idx][0], self.m, abs_tol=1e-6)):
                return False
            opp_const = 0
        
        # Check opposite vertex constraint
        if rect_edge in ['bottom', 'top']:
            if not math.isclose(P[opp_idx][1], opp_const, abs_tol=1e-6):
                return False
        else:
            if not math.isclose(P[opp_idx][0], opp_const, abs_tol=1e-6):
                return False
        
        # Check all vertices within rectangle
        for x, y in P:
            if not (0 <= x <= self.m and 0 <= y <= self.n):
                return False
        
        return True

def main():
    # 测试数据1: 底边在底部，顶点在顶部
    test_case1 = Case2Solver(
        t=[1.0, 1.0, 2.0],
        theta=[0.0, math.pi/2, math.pi/2],
        m=2.0,
        n=2.0
    )
    solutions = test_case1.solve()
    print("Case2 Test1 Solutions:", solutions)
    
    # 测试数据2: 边在左侧，顶点在右侧
    test_case2 = Case2Solver(
        t=[1.0, 1.0, 2.0],
        theta=[math.pi/2, math.pi, math.pi],
        m=2.0,
        n=2.0
    )
    solutions = test_case2.solve()
    print("Case2 Test2 Solutions:", solutions)
    
    # 测试数据3: 无解情况
    test_case3 = Case2Solver(
        t=[1.0, 2.0, 3.0],
        theta=[0.1, 0.2, 0.3],
        m=1.0,
        n=1.0
    )
    solutions = test_case3.solve()
    print("Case2 Test3 Solutions (should be empty):", solutions)

if __name__ == "__main__":
    main()