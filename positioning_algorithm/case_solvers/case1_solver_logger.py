import math
from typing import List, Tuple
import sys

class Case1Solver:
    """Solver for case where one edge is on rectangle edge and opposite vertex is on adjacent edge"""
    
    def __init__(self, t: List[float], theta: List[float], m: float, n: float):
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.log_file = open("detailed_solver_log.txt", "w")
    
    def __del__(self):
        self.log_file.close()
    
    def _log(self, message: str):
        """Log message to both console and file"""
        print(message)
        self.log_file.write(message + "\n")
    
    def _log_math(self, operation: str, result):
        """Log detailed math operation"""
        if type(result) == tuple:
            self._log("loging Tuple")
            self._log(f"[MATH] {operation} = ({result[0]:.8f}, {result[1]:.8f})")
        else:    
            self._log("loging Float")
            self._log(f"[MATH] {operation} = {result:.8f}")
    
    def _log_compare(self, a: float, b: float, tol: float = 1e-6):
        """Log comparison with tolerance"""
        diff = abs(a - b)
        passed = diff <= tol
        self._log(f"[COMPARE] {a:.8f} vs {b:.8f}: diff={diff:.2e}, {'PASS' if passed else 'FAIL'}")
        return passed

    def solve(self) -> List[Tuple[float, float, float]]:
        solutions = []
        
        self._log(f"\nStarting solve with t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
        
        for base_idx in range(3):
            p1_idx = base_idx
            p2_idx = (base_idx + 1) % 3
            opp_idx = (base_idx + 2) % 3
            
            self._log(f"\nTrying base edge {base_idx} (points {p1_idx} and {p2_idx})")
            
            for rect_edge in ['bottom', 'top', 'left', 'right']:
                self._log(f"\n  Trying rectangle edge: {rect_edge}")
                
                if rect_edge in ['bottom', 'top']:
                    adj_edges = ['left', 'right']
                else:
                    adj_edges = ['bottom', 'top']
                
                for adj_edge in adj_edges:
                    self._log(f"\n    Trying adjacent edge: {adj_edge}")
                    try:
                        phi_candidates = self._solve_phi(base_idx, rect_edge, adj_edge)
                        
                        for phi in phi_candidates:
                            self._log(f"    Found phi: {phi:.8f} radians")
                            
                            self._log("    Computing position (xO, yO)...")
                            xO, yO = self._compute_position(base_idx, rect_edge, adj_edge, phi)
                            self._log(f"    Computed position: xO={xO:.8f}, yO={yO:.8f}")
                            
                            self._log("    Verifying solution...")
                            if self._verify_solution(xO, yO, phi, base_idx, rect_edge, adj_edge):
                                self._log("    Solution is valid!")
                                solutions.append((xO, yO, phi))
                            else:
                                self._log("    Solution failed verification")
                                
                    except (ValueError, AssertionError, TypeError) as e:
                        self._log(f"    Failed with error: {str(e)}")
                        continue
        
        self._log(f"\nFound {len(solutions)} valid solutions")
        # Log all valid solutions in detail
        for i, (xO, yO, phi) in enumerate(solutions, 1):
            self._log(f"\nSolution {i}:")
            self._log(f"  O = ({xO:.8f}, {yO:.8f})")
            self._log(f"  phi = {phi:.8f} radians")
            
            # Calculate and log all vertex positions
            for j in range(3):
                x = xO + self.t[j] * math.cos(phi + self.theta[j])
                y = yO + self.t[j] * math.sin(phi + self.theta[j])
                self._log(f"  P{j} = ({x:.8f}, {y:.8f})")
            
            # Log triangle side lengths for verification
            def distance(a, b):
                return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            
            P = [(xO + self.t[k] * math.cos(phi + self.theta[k]),
                yO + self.t[k] * math.sin(phi + self.theta[k])) for k in range(3)]
            
            self._log("  Triangle side lengths:")
            self._log(f"    P0-P1: {distance(P[0], P[1]):.8f}")
            self._log(f"    P1-P2: {distance(P[1], P[2]):.8f}")
            self._log(f"    P2-P0: {distance(P[2], P[0]):.8f}")
        return solutions
    
    def _solve_phi(self, base_idx: int, rect_edge: str, adj_edge: str) -> List[float]:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        valid_phis = []

        # 判断1：矩形边类型
        if rect_edge in ['bottom', 'top']:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            B = t1 * math.sin(theta1) - t2 * math.sin(theta2)
            self._log_math(f"A = {t1}*cos({theta1}) - {t2}*cos({theta2})", A)
            self._log_math(f"B = {t1}*sin({theta1}) - {t2}*sin({theta2})", B)

            # 判断2：退化情况
            if abs(A) < 1e-6 and abs(B) < 1e-6:
                self._log("    Degenerate case (A and B both zero)")
                return []
                
            # 判断3：A≈0的特殊情况
            elif abs(A) < 1e-6:
                self._log("    Special case (Aappx0): direct solution Phi=+-pi/2")
                base_phi = math.copysign(math.pi/2, -B)
                candidates = [base_phi, base_phi + math.pi]
                
            # 判断4：常规情况
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
        else:  # 垂直边处理（结构相同，方程不同）
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            A = t2 * math.sin(theta2) - t1 * math.sin(theta1)
            B = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            self._log_math(f"A = {t2}*sin({theta2} - {t1}*sin({theta1}))", A)
            self._log_math(f"B = {t1}*cos({theta1}) - {t2}*cos({theta2})", B)

            if abs(A) < 1e-6 and abs(B) < 1e-6:
                self._log("    Degenerate case (A and B both zero)")
                return []
                
            elif abs(A) < 1e-6:
                self._log("    Special case (Aappx0): direct solution Phi=0 or pi")
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
        TOL = 1e-6  # Tolerance constant
        
        for phi in candidates:
            try:
                self._log(f"\n=== Validating phi = {phi:.8f} ===")
                
                # Calculate relative coordinates (origin at O)
                P = []
                for i in range(3):
                    x = self.t[i] * math.cos(phi + self.theta[i])
                    y = self.t[i] * math.sin(phi + self.theta[i])
                    P.append((x, y))
                    self._log(f"P{i} = ({x:.8f}, {y:.8f})")
                
                # 1. Base edge alignment check
                self._log("\n[Base Edge Validation]")
                if rect_edge in ['bottom', 'top']:
                    # Check if base edge is horizontal
                    dy = abs(P[1][1] - P[0][1])
                    self._log(f"Vertical diff dy = {dy:.8f} (req < {TOL:.1e})")
                    if dy >= TOL:
                        self._log("-> Base not horizontal, skip")
                        continue
                    
                    base_y = (P[0][1] + P[1][1]) / 2
                    self._log(f"Base avg y = {base_y:.8f}")
                    
                    # Check third point position
                    if rect_edge == 'bottom':
                        cond = P[2][1] > base_y - TOL
                        self._log(f"P2.y = {P[2][1]:.8f} > {base_y-TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                        if not cond:
                            continue
                    else:  # top
                        cond = P[2][1] < base_y + TOL
                        self._log(f"P2.y = {P[2][1]:.8f} < {base_y+TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                        if not cond:
                            continue
                else:  # left/right
                    # Check if base edge is vertical
                    dx = abs(P[1][0] - P[0][0])
                    self._log(f"Horizontal diff dx = {dx:.8f} (req < {TOL:.1e})")
                    if dx >= TOL:
                        self._log("-> Base not vertical, skip")
                        continue
                    
                    base_x = (P[0][0] + P[1][0]) / 2
                    self._log(f"Base avg x = {base_x:.8f}")
                    
                    # Check third point position
                    if rect_edge == 'left':
                        cond = P[2][0] > base_x - TOL
                        self._log(f"P2.x = {P[2][0]:.8f} > {base_x-TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                        if not cond:
                            continue
                    else:  # right
                        cond = P[2][0] < base_x + TOL
                        self._log(f"P2.x = {P[2][0]:.8f} < {base_x+TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                        if not cond:
                            continue
                
                # 2. Adjacent edge constraint
                self._log("\n[Adjacent Edge Check]")
                if adj_edge == 'left':
                    p2_x = P[2][0]
                    min_x = min(P[0][0], P[1][0])
                    cond = p2_x <= min_x + TOL
                    self._log(f"P2.x = {p2_x:.8f} <= min(P0.x,P1.x)+tol = {min_x+TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                    if not cond:
                        continue
                elif adj_edge == 'right':
                    p2_x = P[2][0]
                    max_x = max(P[0][0], P[1][0])
                    cond = p2_x >= max_x - TOL
                    self._log(f"P2.x = {p2_x:.8f} >= max(P0.x,P1.x)-tol = {max_x-TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                    if not cond:
                        continue
                elif adj_edge == 'bottom':
                    p2_y = P[2][1]
                    min_y = min(P[0][1], P[1][1])
                    cond = p2_y <= min_y + TOL
                    self._log(f"P2.y = {p2_y:.8f} <= min(P0.y,P1.y)+tol = {min_y+TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                    if not cond:
                        continue
                elif adj_edge == 'top':
                    p2_y = P[2][1]
                    max_y = max(P[0][1], P[1][1])
                    cond = p2_y >= max_y - TOL
                    self._log(f"P2.y = {p2_y:.8f} >= max(P0.y,P1.y)-tol = {max_y-TOL:.8f}? {'PASS' if cond else 'FAIL'}")
                    if not cond:
                        continue
                
                # 3. Translation feasibility
                self._log("\n[Translation Check]")
                min_x = min(p[0] for p in P)
                max_x = max(p[0] for p in P)
                min_y = min(p[1] for p in P)
                max_y = max(p[1] for p in P)
                self._log(f"X range: [{min_x:.8f}, {max_x:.8f}] (rect width={self.m:.1f})")
                self._log(f"Y range: [{min_y:.8f}, {max_y:.8f}] (rect height={self.n:.1f})")
                
                width_ok = (max_x - min_x) <= (self.m + TOL)
                height_ok = (max_y - min_y) <= (self.n + TOL)
                
                self._log(f"Width check: {max_x-min_x:.8f} <= {self.m+TOL:.8f}? {'PASS' if width_ok else 'FAIL'}")
                self._log(f"Height check: {max_y-min_y:.8f} <= {self.n+TOL:.8f}? {'PASS' if height_ok else 'FAIL'}")
                
                if not (width_ok and height_ok):
                    continue
                    
                self._log("*** ALL VALIDATIONS PASSED ***")
                valid_phis.append(phi)
                
            except Exception as e:
                self._log(f"!!! Validation error: {str(e)}")
        
        return valid_phis
    
    def _compute_position(self, base_idx: int, rect_edge: str, adj_edge: str, phi: float) -> Tuple[float, float]:
        p1_idx = base_idx
        opp_idx = (base_idx + 2) % 3

        if rect_edge in ['bottom', 'top']:
            # yO计算：基底点对齐bottom(y=0)或top(y=n)
            y_target = 0 if rect_edge == 'bottom' else self.n
            yO = y_target - self.t[p1_idx] * math.sin(phi + self.theta[p1_idx])
            self._log_math(f"yO = {y_target} - t{p1_idx}*sin(phi + theta{p1_idx}) = {y_target} - {self.t[p1_idx]}*sin({phi} + {self.theta[p1_idx]})", yO)
            
            # xO计算：对角点对齐left(x=0)或right(x=m)
            x_target = 0 if adj_edge == 'left' else self.m
            x_opp = self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
            self._log_math(f"x_opp = t{opp_idx}*cos(phi + theta{opp_idx}) = {self.t[opp_idx]}*cos({phi} + {self.theta[opp_idx]})", x_opp)
            
            xO = x_target - x_opp
            self._log_math(f"xO = {x_target} - x_opp = {x_target} - {x_opp}", xO)
        else:
            # xO计算：基底点对齐left(x=0)或right(x=m)
            x_target = 0 if rect_edge == 'left' else self.m
            xO = x_target - self.t[p1_idx] * math.cos(phi + self.theta[p1_idx])
            self._log_math(f"xO = {x_target} - t{p1_idx}*cos(phi + theta{p1_idx}) = {x_target} - {self.t[p1_idx]}*cos({phi} + {self.theta[p1_idx]})", xO)
            
            # yO计算：对角点对齐bottom(y=0)或top(y=n)
            y_target = 0 if adj_edge == 'bottom' else self.n
            y_opp = self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
            self._log_math(f"y_opp = t{opp_idx}*sin(phi + theta{opp_idx}) = {self.t[opp_idx]}*sin({phi} + {self.theta[opp_idx]})", y_opp)
            
            yO = y_target - y_opp
            self._log_math(f"yO = {y_target} - y_opp = {y_target} - {y_opp}", yO)

        return xO, yO
    
    def _verify_solution(self, xO: float, yO: float, phi: float, 
                        base_idx: int, rect_edge: str, adj_edge: str) -> bool:
        TOL = 1e-6  # Unified tolerance
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        self._log(f"\nVerifying solution: xO={xO:.8f}, yO={yO:.8f}, phi={phi:.8f}")
        self._log(f"base_idx={base_idx}, rect_edge={rect_edge}, adj_edge={adj_edge}")
        
        # Calculate all vertex positions
        P = []
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            P.append((x, y))
            self._log(f"P{i} = ({x:.8f}, {y:.8f})")
        
        # Simplified verification since we already did rigorous checks
        # Just ensure all points are within rectangle bounds with tolerance
        for i, (x, y) in enumerate(P):
            x_ok = (-TOL <= x <= self.m + TOL)
            y_ok = (-TOL <= y <= self.n + TOL)
            
            self._log(f"P{i} bounds check: "
                    f"x={x:.8f} in [{-TOL:.6f}, {self.m+TOL:.6f}]? {'YES' if x_ok else 'NO'}, "
                    f"y={y:.8f} in [{-TOL:.6f}, {self.n+TOL:.6f}]? {'YES' if y_ok else 'NO'}")
            
            if not (x_ok and y_ok):
                self._log(f"P{i} out of bounds")
                return False
        
        # Quick edge alignment checks (less strict than in _validate_phi_candidates)
        if rect_edge == 'bottom':
            if not (abs(P[p1_idx][1]) <= TOL and abs(P[p2_idx][1]) <= TOL):
                self._log("Bottom edge not aligned")
                return False
        elif rect_edge == 'top':
            if not (abs(P[p1_idx][1] - self.n) <= TOL and abs(P[p2_idx][1] - self.n) <= TOL):
                self._log("Top edge not aligned")
                return False
        elif rect_edge == 'left':
            if not (abs(P[p1_idx][0]) <= TOL and abs(P[p2_idx][0]) <= TOL):
                self._log("Left edge not aligned")
                return False
        elif rect_edge == 'right':
            if not (abs(P[p1_idx][0] - self.m) <= TOL and abs(P[p2_idx][0] - self.m) <= TOL):
                self._log("Right edge not aligned")
                return False
        
        self._log("*** SOLUTION VALID ***")
        return True

def main():
    original_stdout = sys.stdout
    with open('detailed_solver_output.txt', 'w') as f:
        sys.stdout = f
        
        print("\n=== TEST CASE (Corrected) ===")
        test_case = Case1Solver(
            t=[2.0, 2*math.sqrt(2), 2.0],  # 点O到三个顶点的距离
            theta=[0.0, math.pi/4, math.pi/2],  # 三个向量与基准方向的夹角
            m=4.0,  # 矩形宽度
            n=4.0   # 矩形高度
        )
        solutions = test_case.solve()
        
        print("\n=== SOLUTIONS ===")
        for i, (xO, yO, phi) in enumerate(solutions, 1):
            print(f"\nSolution {i}:")
            print(f"O = ({xO:.6f}, {yO:.6f})")
            print(f"phi = {phi:.6f} rad ({math.degrees(phi):.2f}°)")
            
            # 计算并打印所有顶点坐标
            vertices = []
            for j in range(3):
                x = xO + test_case.t[j] * math.cos(phi + test_case.theta[j])
                y = yO + test_case.t[j] * math.sin(phi + test_case.theta[j])
                vertices.append((x, y))
                print(f"P{j} = ({x:.6f}, {y:.6f})")
            
            # 验证三角形边长（辅助检查）
            def distance(a, b):
                return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            
            print("\nTriangle side lengths:")
            print(f"P0-P1: {distance(vertices[0], vertices[1]):.6f}")
            print(f"P1-P2: {distance(vertices[1], vertices[2]):.6f}")
            print(f"P2-P0: {distance(vertices[2], vertices[0]):.6f}")
    
    sys.stdout = original_stdout
    print("Complete output saved to detailed_solver_output.txt")

if __name__ == "__main__":
    main()