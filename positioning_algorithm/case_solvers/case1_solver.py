import math
from typing import List, Tuple

class Case1Solver:
    """Solver for case where one edge is on rectangle edge and opposite vertex is on adjacent edge"""
    
    def __init__(self, t: List[float], theta: List[float], m: float, n: float):
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
    
    def solve(self) -> List[Tuple[float, float, float]]:
        solutions = []
        
        for base_idx in range(3):
            p1_idx = base_idx
            p2_idx = (base_idx + 1) % 3
            opp_idx = (base_idx + 2) % 3
            
            for rect_edge in ['bottom', 'top', 'left', 'right']:
                if rect_edge in ['bottom', 'top']:
                    adj_edges = ['left', 'right']
                else:
                    adj_edges = ['bottom', 'top']
                
                for adj_edge in adj_edges:
                    try:
                        phi_candidates = self._solve_phi(base_idx, rect_edge, adj_edge)
                        
                        for phi in phi_candidates:
                            xO, yO = self._compute_position(base_idx, rect_edge, adj_edge, phi)
                            
                            if self._verify_solution(xO, yO, phi, base_idx, rect_edge, adj_edge):
                                solutions.append((xO, yO, phi))
                                
                    except (ValueError, AssertionError, TypeError):
                        continue
        
        return solutions
    
    def _solve_phi(self, base_idx: int, rect_edge: str, adj_edge: str) -> List[float]:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        valid_phis = []

        if rect_edge in ['bottom', 'top']:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            B = t1 * math.sin(theta1) - t2 * math.sin(theta2)

            if abs(A) < 1e-6 and abs(B) < 1e-6:
                return []
                
            elif abs(A) < 1e-6:
                base_phi = math.copysign(math.pi/2, -B)
                candidates = [base_phi, base_phi + math.pi]
                
            else:
                base_phi = math.atan2(-B, A)
                candidates = [base_phi, base_phi + math.pi, base_phi - math.pi]
            
            return self._validate_phi_candidates(
                    candidates=candidates,
                    t1=t1,
                    theta1=theta1,
                    opp_idx=opp_idx,
                    rect_edge=rect_edge,
                    adj_edge=adj_edge,
                    is_vertical=False
                )
        else:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t2 * math.sin(theta2) - t1 * math.sin(theta1)
            B = t1 * math.cos(theta1) - t2 * math.cos(theta2)

            if abs(A) < 1e-6 and abs(B) < 1e-6:
                return []
                
            elif abs(A) < 1e-6:
                base_phi = 0 if B > 0 else math.pi
                candidates = [base_phi, base_phi + math.pi]
                
            else:
                base_phi = math.atan2(B, -A)
                candidates = [base_phi, base_phi + math.pi, base_phi - math.pi]
            
                return self._validate_phi_candidates(
                        candidates=candidates,
                        t1=t1,
                        theta1=theta1,
                        opp_idx=opp_idx,
                        rect_edge=rect_edge,
                        adj_edge=adj_edge,
                        is_vertical=True
                    )
            
    def _validate_phi_candidates(self, candidates, t1, theta1, opp_idx, rect_edge, adj_edge, is_vertical=False):
        valid_phis = []
        TOL = 1e-6
        
        for phi in candidates:
            try:
                P = []
                for i in range(3):
                    x = self.t[i] * math.cos(phi + self.theta[i])
                    y = self.t[i] * math.sin(phi + self.theta[i])
                    P.append((x, y))
                
                if rect_edge in ['bottom', 'top']:
                    dy = abs(P[1][1] - P[0][1])
                    if dy >= TOL:
                        continue
                    
                    base_y = (P[0][1] + P[1][1]) / 2
                    
                    if rect_edge == 'bottom':
                        if not (P[2][1] > base_y - TOL):
                            continue
                    else:
                        if not (P[2][1] < base_y + TOL):
                            continue
                else:
                    dx = abs(P[1][0] - P[0][0])
                    if dx >= TOL:
                        continue
                    
                    base_x = (P[0][0] + P[1][0]) / 2
                    
                    if rect_edge == 'left':
                        if not (P[2][0] > base_x - TOL):
                            continue
                    else:
                        if not (P[2][0] < base_x + TOL):
                            continue
                
                if adj_edge == 'left':
                    if not (P[2][0] <= min(P[0][0], P[1][0]) + TOL):
                        continue
                elif adj_edge == 'right':
                    if not (P[2][0] >= max(P[0][0], P[1][0]) - TOL):
                        continue
                elif adj_edge == 'bottom':
                    if not (P[2][1] <= min(P[0][1], P[1][1]) + TOL):
                        continue
                elif adj_edge == 'top':
                    if not (P[2][1] >= max(P[0][1], P[1][1]) - TOL):
                        continue
                
                min_x = min(p[0] for p in P)
                max_x = max(p[0] for p in P)
                min_y = min(p[1] for p in P)
                max_y = max(p[1] for p in P)
                
                width_ok = (max_x - min_x) <= (self.m + TOL)
                height_ok = (max_y - min_y) <= (self.n + TOL)
                
                if not (width_ok and height_ok):
                    continue
                    
                valid_phis.append(phi)
                
            except Exception:
                continue
        
        return valid_phis
    
    def _compute_position(self, base_idx: int, rect_edge: str, adj_edge: str, phi: float) -> Tuple[float, float]:
        p1_idx = base_idx
        opp_idx = (base_idx + 2) % 3

        if rect_edge in ['bottom', 'top']:
            y_target = 0 if rect_edge == 'bottom' else self.n
            yO = y_target - self.t[p1_idx] * math.sin(phi + self.theta[p1_idx])
            
            x_target = 0 if adj_edge == 'left' else self.m
            x_opp = self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
            xO = x_target - x_opp
        else:
            x_target = 0 if rect_edge == 'left' else self.m
            xO = x_target - self.t[p1_idx] * math.cos(phi + self.theta[p1_idx])
            
            y_target = 0 if adj_edge == 'bottom' else self.n
            y_opp = self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
            yO = y_target - y_opp

        return xO, yO
    
    def _verify_solution(self, xO: float, yO: float, phi: float, 
                        base_idx: int, rect_edge: str, adj_edge: str) -> bool:
        TOL = 1e-6
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        P = []
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            P.append((x, y))
        
        for i, (x, y) in enumerate(P):
            x_ok = (-TOL <= x <= self.m + TOL)
            y_ok = (-TOL <= y <= self.n + TOL)
            
            if not (x_ok and y_ok):
                return False
        
        if rect_edge == 'bottom':
            if not (abs(P[p1_idx][1]) <= TOL and abs(P[p2_idx][1]) <= TOL):
                return False
        elif rect_edge == 'top':
            if not (abs(P[p1_idx][1] - self.n) <= TOL and abs(P[p2_idx][1] - self.n) <= TOL):
                return False
        elif rect_edge == 'left':
            if not (abs(P[p1_idx][0]) <= TOL and abs(P[p2_idx][0]) <= TOL):
                return False
        elif rect_edge == 'right':
            if not (abs(P[p1_idx][0] - self.m) <= TOL and abs(P[p2_idx][0] - self.m) <= TOL):
                return False
        
        return True

def main():
    test_case = Case1Solver(
        t=[2.0, 2*math.sqrt(2), 2.0],
        theta=[0.0, math.pi/4, math.pi/2],
        m=4.0,
        n=4.0
    )
    solutions = test_case.solve()
    
    print("\n=== SOLUTIONS ===")
    for i, (xO, yO, phi) in enumerate(solutions, 1):
        print(f"\nSolution {i}:")
        print(f"O = ({xO:.6f}, {yO:.6f})")
        print(f"phi = {phi:.6f} rad ({math.degrees(phi):.2f}°)")
        
        vertices = []
        for j in range(3):
            x = xO + test_case.t[j] * math.cos(phi + test_case.theta[j])
            y = yO + test_case.t[j] * math.sin(phi + test_case.theta[j])
            vertices.append((x, y))
            print(f"P{j} = ({x:.6f}, {y:.6f})")
        
        def distance(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
        print("\nTriangle side lengths:")
        print(f"P0-P1: {distance(vertices[0], vertices[1]):.6f}")
        print(f"P1-P2: {distance(vertices[1], vertices[2]):.6f}")
        print(f"P2-P0: {distance(vertices[2], vertices[0]):.6f}")

if __name__ == "__main__":
    main()