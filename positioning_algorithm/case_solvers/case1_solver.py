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
    
    def _log_math(self, operation: str, result: float):
        """Log detailed math operation"""
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
                                
                    except (ValueError, AssertionError) as e:
                        self._log(f"    Failed with error: {str(e)}")
                        continue
        
        self._log(f"\nFound {len(solutions)} valid solutions")
        return solutions
    
    def _solve_phi(self, base_idx: int, rect_edge: str, adj_edge: str) -> List[float]:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        valid_phis = []

        if rect_edge in ['bottom', 'top']:
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            # 详细记录方程构造过程
            self._log("    Solving equation: t1*sin(phi+theta1) = t2*sin(phi+theta2)")
            self._log("    Using trigonometric identity: A*sin(phi) + B*cos(phi) = 0")
            
            A = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            self._log_math(f"A = t1*cos(theta1) - t2*cos(theta2) = {t1}*cos({theta1}) - {t2}*cos({theta2})", A)
            
            B = t1 * math.sin(theta1) - t2 * math.sin(theta2)
            self._log_math(f"B = t1*sin(theta1) - t2*sin(theta2) = {t1}*sin({theta1}) - {t2}*sin({theta2})", B)

            if abs(A) < 1e-6 and abs(B) < 1e-6:
                self._log("    Degenerate case (A and B both near zero)")
                return []

            base_phi = math.atan2(-B, A)
            self._log_math(f"base_phi = atan2(-B, A) = atan2(-{B}, {A})", base_phi)
            
            candidates = [base_phi, base_phi + math.pi, base_phi - math.pi]
            x_const = 0 if adj_edge == 'left' else self.m

            for phi in candidates:
                try:
                    # 验证第三顶点坐标
                    x_opp = self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
                    self._log_math(f"x_opp = t{opp_idx}*cos(phi + theta{opp_idx}) = {self.t[opp_idx]}*cos({phi} + {self.theta[opp_idx]})", x_opp)
                    
                    if not self._log_compare(x_opp, x_const):
                        continue
                    
                    # 验证基底顶点坐标
                    yO = -t1 * math.sin(phi + theta1)
                    self._log_math(f"yO = -t1*sin(phi + theta1) = -{t1}*sin({phi} + {theta1})", yO)
                    
                    target_y = 0 if rect_edge == 'bottom' else self.n
                    if not self._log_compare(yO, target_y):
                        continue
                    
                    valid_phis.append(phi)
                except Exception as e:
                    self._log(f"    Error in phi candidate {phi}: {str(e)}")
                    continue

        else:  # 垂直边处理
            t1, t2 = self.t[p1_idx], self.t[p2_idx]
            theta1, theta2 = self.theta[p1_idx], self.theta[p2_idx]
            
            self._log("    Solving equation: t1*cos(phi+theta1) = t2*cos(phi+theta2)")
            self._log("    Using trigonometric identity: A*cos(phi) - B*sin(phi) = 0")
            
            A = t1 * math.sin(theta1) - t2 * math.sin(theta2)
            self._log_math(f"A = t1*sin(theta1) - t2*sin(theta2) = {t1}*sin({theta1}) - {t2}*sin({theta2})", A)
            
            B = t1 * math.cos(theta1) - t2 * math.cos(theta2)
            self._log_math(f"B = t1*cos(theta1) - t2*cos(theta2) = {t1}*cos({theta1}) - {t2}*cos({theta2})", B)

            if abs(A) < 1e-6 and abs(B) < 1e-6:
                self._log("    Degenerate case (A and B both near zero)")
                return []

            base_phi = math.atan2(B, -A)
            self._log_math(f"base_phi = atan2(B, -A) = atan2({B}, -{A})", base_phi)
            
            candidates = [base_phi, base_phi + math.pi, base_phi - math.pi]
            y_const = 0 if adj_edge == 'bottom' else self.n

            for phi in candidates:
                try:
                    # 验证第三顶点坐标
                    y_opp = self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
                    self._log_math(f"y_opp = t{opp_idx}*sin(phi + theta{opp_idx}) = {self.t[opp_idx]}*sin({phi} + {self.theta[opp_idx]})", y_opp)
                    
                    if not self._log_compare(y_opp, y_const):
                        continue
                    
                    # 验证基底顶点坐标
                    xO = -t1 * math.cos(phi + theta1)
                    self._log_math(f"xO = -t1*cos(phi + theta1) = -{t1}*cos({phi} + {theta1})", xO)
                    
                    target_x = 0 if rect_edge == 'left' else self.m
                    if not self._log_compare(xO, target_x):
                        continue
                    
                    valid_phis.append(phi)
                except Exception as e:
                    self._log(f"    Error in phi candidate {phi}: {str(e)}")
                    continue

        return valid_phis
    
    def _compute_position(self, base_idx: int, rect_edge: str, adj_edge: str, phi: float) -> Tuple[float, float]:
        p1_idx = base_idx
        opp_idx = (base_idx + 2) % 3

        if rect_edge in ['bottom', 'top']:
            yO = -self.t[p1_idx] * math.sin(phi + self.theta[p1_idx])
            self._log_math(f"yO = -t{p1_idx}*sin(phi + theta{p1_idx}) = -{self.t[p1_idx]}*sin({phi} + {self.theta[p1_idx]})", yO)
            
            x_const = 0 if adj_edge == 'left' else self.m
            x_opp = self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
            self._log_math(f"x_opp = t{opp_idx}*cos(phi + theta{opp_idx}) = {self.t[opp_idx]}*cos({phi} + {self.theta[opp_idx]})", x_opp)
            
            xO = x_const - x_opp
            self._log_math(f"xO = {x_const} - x_opp = {x_const} - {x_opp}", xO)
        else:
            xO = -self.t[p1_idx] * math.cos(phi + self.theta[p1_idx])
            self._log_math(f"xO = -t{p1_idx}*cos(phi + theta{p1_idx}) = -{self.t[p1_idx]}*cos({phi} + {self.theta[p1_idx]})", xO)
            
            y_const = 0 if adj_edge == 'bottom' else self.n
            y_opp = self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
            self._log_math(f"y_opp = t{opp_idx}*sin(phi + theta{opp_idx}) = {self.t[opp_idx]}*sin({phi} + {self.theta[opp_idx]})", y_opp)
            
            yO = y_const - y_opp
            self._log_math(f"yO = {y_const} - y_opp = {y_const} - {y_opp}", yO)

        return xO, yO
    
    def _verify_solution(self, xO: float, yO: float, phi: float, 
                        base_idx: int, rect_edge: str, adj_edge: str) -> bool:
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        self._log(f"\n    Verifying solution: xO={xO:.8f}, yO={yO:.8f}, phi={phi:.8f}")
        self._log(f"    base_idx={base_idx}, rect_edge={rect_edge}, adj_edge={adj_edge}")
        
        P = []
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            P.append((x, y))
            self._log_math(f"P{i}.x = xO + t{i}*cos(phi + theta{i}) = {xO} + {self.t[i]}*cos({phi} + {self.theta[i]})", x)
            self._log_math(f"P{i}.y = yO + t{i}*sin(phi + theta{i}) = {yO} + {self.t[i]}*sin({phi} + {self.theta[i]})", y)
        
        # 检查基底边约束
        if rect_edge == 'bottom':
            self._log(f"    Checking bottom edge constraints:")
            if not (self._log_compare(P[p1_idx][1], 0) and self._log_compare(P[p2_idx][1], 0)):
                return False
        elif rect_edge == 'top':
            self._log(f"    Checking top edge constraints:")
            if not (self._log_compare(P[p1_idx][1], self.n) and self._log_compare(P[p2_idx][1], self.n)):
                return False
        elif rect_edge == 'left':
            self._log(f"    Checking left edge constraints:")
            if not (self._log_compare(P[p1_idx][0], 0) and self._log_compare(P[p2_idx][0], 0)):
                return False
        elif rect_edge == 'right':
            self._log(f"    Checking right edge constraints:")
            if not (self._log_compare(P[p1_idx][0], self.m) and self._log_compare(P[p2_idx][0], self.m)):
                return False
        
        # 检查对角顶点约束
        if adj_edge == 'left':
            self._log(f"    Checking left adjacent edge constraint:")
            if not self._log_compare(P[opp_idx][0], 0):
                return False
        elif adj_edge == 'right':
            self._log(f"    Checking right adjacent edge constraint:")
            if not self._log_compare(P[opp_idx][0], self.m):
                return False
        elif adj_edge == 'bottom':
            self._log(f"    Checking bottom adjacent edge constraint:")
            if not self._log_compare(P[opp_idx][1], 0):
                return False
        elif adj_edge == 'top':
            self._log(f"    Checking top adjacent edge constraint:")
            if not self._log_compare(P[opp_idx][1], self.n):
                return False
        
        # 检查所有顶点是否在矩形内
        for i, (x, y) in enumerate(P):
            self._log(f"    Checking bounds for P{i}:")
            x_in = 0 <= x <= self.m
            y_in = 0 <= y <= self.n
            self._log(f"    x={x:.8f} in [0, {self.m}]? {'YES' if x_in else 'NO'}")
            self._log(f"    y={y:.8f} in [0, {self.n}]? {'YES' if y_in else 'NO'}")
            if not (x_in and y_in):
                return False
        
        self._log("    All constraints satisfied!")
        return True

def main():
    original_stdout = sys.stdout
    with open('detailed_solver_output.txt', 'w') as f:
        sys.stdout = f
        
        print("\n=== TEST CASE 1 (3-4-5 right triangle) ===")
        test_case1 = Case1Solver(
            t=[3.0, 4.0, 5.0],
            theta=[0.0, math.pi/2, math.atan2(4, 3)],
            m=3.0,
            n=4.0
        )
        solutions = test_case1.solve()
        
        print("\n=== SOLUTIONS ===")
        for i, (xO, yO, phi) in enumerate(solutions, 1):
            print(f"\nSolution {i}:")
            print(f"O = ({xO:.6f}, {yO:.6f})")
            print(f"phi = {phi:.6f} rad ({math.degrees(phi):.2f}°)")
            
            for j in range(3):
                x = xO + test_case1.t[j] * math.cos(phi + test_case1.theta[j])
                y = yO + test_case1.t[j] * math.sin(phi + test_case1.theta[j])
                print(f"P{j} = ({x:.6f}, {y:.6f})")
    
    sys.stdout = original_stdout
    print("Complete output saved to detailed_solver_output.txt")

if __name__ == "__main__":
    main()