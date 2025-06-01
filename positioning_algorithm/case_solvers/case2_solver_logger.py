import math
from typing import List, Tuple

class Case2Solver:
    def __init__(self, t: List[float], theta: List[float], m: float, n: float):
        """
        t: Distances from center O to triangle vertices [t0, t1, t2]
        theta: Angles from O to vertices [θ0, θ1, θ2] (radians)
        m: Rectangle width
        n: Rectangle height
        """
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.TOL = 1e-6
        self.edges = ['bottom', 'top', 'left', 'right']
        self.log_file = open("case2_solver_log.txt", "w", encoding='utf-8')
    
    def __del__(self):
        self.log_file.close()
    
    def _log(self, message: str):
        """Log message to both console and file"""
        print(message)
        self.log_file.write(message + "\n")
    
    def _log_math(self, operation: str, result):
        """Log detailed math operation"""
        if isinstance(result, tuple):
            self._log(f"[MATH] {operation} = ({result[0]:.8f}, {result[1]:.8f})")
        else:    
            self._log(f"[MATH] {operation} = {result:.8f}")
    
    def _log_compare(self, a: float, b: float, tol: float = 1e-6):
        """Log comparison with tolerance"""
        diff = abs(a - b)
        passed = diff <= tol
        self._log(f"[COMPARE] {a:.8f} vs {b:.8f}: diff={diff:.2e}, {'PASS' if passed else 'FAIL'}")
        return passed

    def solve(self) -> List[Tuple[float, float, float]]:
        """Find all valid solutions (xO, yO, phi)"""
        solutions = []
        
        self._log(f"\n=== Starting Case2Solver ===")
        self._log(f"Parameters:")
        self._log(f"  t = {self.t}")
        self._log(f"  theta = {self.theta} (radians)")
        self._log(f"  m = {self.m}")
        self._log(f"  n = {self.n}")
        self._log(f"  Tolerance = {self.TOL}")
        
        for base_idx in range(3):
            p1, p2 = base_idx, (base_idx + 1) % 3
            opp_idx = (base_idx + 2) % 3
            
            self._log(f"\n=== Trying base edge {base_idx} (points {p1} and {p2}) ===")
            
            for rect_edge in self.edges:
                self._log(f"\n  Trying rectangle edge: {rect_edge}")
                
                try:
                    self._log("    Solving for phi...")
                    phi_candidates = self._solve_phi_equation(p1, p2, opp_idx, rect_edge)
                    self._log(f"    Found {len(phi_candidates)} phi candidates: {[f'{x:.6f}' for x in phi_candidates]}")
                    
                    for phi in phi_candidates:
                        self._log(f"\n      Testing phi = {phi:.8f} radians ({math.degrees(phi):.2f}°)")
                        
                        self._log("      Computing position (xO, yO)...")
                        xO, yO = self._compute_position(phi, p1, p2, opp_idx, rect_edge)
                        self._log(f"      Computed position: xO={xO:.8f}, yO={yO:.8f}")
                        
                        if math.isnan(xO) or math.isnan(yO):
                            self._log("      Invalid position (NaN), skipping")
                            continue
                            
                        self._log("      Verifying solution...")
                        if self._verify_solution(xO, yO, phi, p1, p2, opp_idx, rect_edge):
                            self._log("      *** Solution is valid! ***")
                            solutions.append((xO, yO, phi))
                        else:
                            self._log("      Solution failed verification")
                            
                except Exception as e:
                    self._log(f"    Error encountered: {str(e)}")
                    continue
        
        self._log(f"\n=== Solution Summary ===")
        self._log(f"Found {len(solutions)} valid solutions")
        
        for i, (xO, yO, phi) in enumerate(solutions, 1):
            self._log(f"\nSolution {i}:")
            self._log(f"  O = ({xO:.8f}, {yO:.8f})")
            self._log(f"  phi = {phi:.8f} radians ({math.degrees(phi):.2f}°)")
            
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

    def _solve_phi_equation(self, p1: int, p2: int, opp_idx: int, rect_edge: str) -> List[float]:
        """Solve angle equation, return phi candidates"""
        self._log(f"\n    [Solving phi equation for rect_edge={rect_edge}]")
        
        if rect_edge in ['bottom', 'top']:
            # Horizontal base edge case
            target = self.n if rect_edge == 'bottom' else -self.n
            self._log(f"    Horizontal base edge case (target={target:.6f})")
            
            # Equation: t_opp*sin(phi+θ_opp) - t_p1*sin(phi+θ_p1) = target
            A = (self.t[opp_idx] * math.cos(self.theta[opp_idx]) - 
                 self.t[p1] * math.cos(self.theta[p1]))
            B = (self.t[opp_idx] * math.sin(self.theta[opp_idx]) - 
                 self.t[p1] * math.sin(self.theta[p1]))
            
            self._log_math(f"A = t{opp_idx}*cos(theta{opp_idx}) - t{p1}*cos(theta{p1})", A)
            self._log_math(f"B = t{opp_idx}*sin(theta{opp_idx}) - t{p1}*sin(theta{p1})", B)
            
            equation_type = "A*sin(phi) + B*cos(phi) = target"
        else:
            # Vertical base edge case
            target = self.m if rect_edge == 'left' else -self.m
            self._log(f"    Vertical base edge case (target={target:.6f})")
            
            # Equation: t_opp*cos(phi+θ_opp) - t_p1*cos(phi+θ_p1) = target
            A = (self.t[opp_idx] * math.cos(self.theta[opp_idx]) - 
                 self.t[p1] * math.cos(self.theta[p1]))
            B = (self.t[opp_idx] * math.sin(self.theta[opp_idx]) - 
                 self.t[p1] * math.sin(self.theta[p1]))
            
            self._log_math(f"A = t{opp_idx}*cos(theta{opp_idx}) - t{p1}*cos(theta{p1})", A)
            self._log_math(f"B = t{opp_idx}*sin(theta{opp_idx}) - t{p1}*sin(theta{p1})", B)
            
            equation_type = "A*cos(phi) - B*sin(phi) = target"
        
        self._log(f"    Equation to solve: {equation_type}")
        
        norm = math.sqrt(A**2 + B**2)
        self._log_math(f"norm = sqrt(A² + B²)", norm)
        
        if norm < self.TOL:
            self._log("    No solution: norm too small (degenerate case)")
            return []
        
        if abs(norm - abs(target)) < self.TOL:
            self._log("    Special case: norm ≈ |target| (single solution)")
            if target > 0:
                phi = math.atan2(B, A) if rect_edge in ['bottom', 'top'] else math.atan2(B, -A)
            else:
                phi = math.atan2(-B, -A) if rect_edge in ['bottom', 'top'] else math.atan2(-B, A)
            
            self._log_math(f"phi = atan2 result", phi)
            return [phi]
        elif norm < abs(target):
            self._log("    No solution: norm < |target|")
            return []
        else:
            self._log("    General case: two solutions")
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

    def _compute_position(self, phi: float, p1: int, p2: int, opp_idx: int, 
                         rect_edge: str) -> Tuple[float, float]:
        """Compute center O's coordinates (xO, yO)"""
        self._log(f"\n    [Computing position for phi={phi:.6f}, rect_edge={rect_edge}]")
        
        if rect_edge in ['bottom', 'top']:
            # Horizontal base edge case
            self._log("    Horizontal base edge case")
            
            # 1. Compute yO to align base edge
            y_target = 0 if rect_edge == 'bottom' else self.n
            yO = y_target - self.t[p1] * math.sin(phi + self.theta[p1])
            self._log_math(f"yO = {y_target} - t{p1}*sin(phi + theta{p1})", yO)
            
            # 2. Compute xO to keep base vertices within rectangle
            x_p1 = self.t[p1] * math.cos(phi + self.theta[p1])
            x_p2 = self.t[p2] * math.cos(phi + self.theta[p2])
            
            x_min = max(-x_p1, -x_p2)
            x_max = min(self.m - x_p1, self.m - x_p2)
            
            self._log(f"    xO range constraints:")
            self._log(f"      x_min = max({-x_p1:.6f}, {-x_p2:.6f}) = {x_min:.6f}")
            self._log(f"      x_max = min({self.m - x_p1:.6f}, {self.m - x_p2:.6f}) = {x_max:.6f}")
            
            if x_min > x_max + self.TOL:
                self._log("    Invalid position: x_min > x_max")
                return (math.nan, math.nan)
            
            xO = (x_min + x_max) / 2
            self._log(f"    xO = (x_min + x_max)/2 = {xO:.6f}")
        else:
            # Vertical base edge case
            self._log("    Vertical base edge case")
            
            # 1. Compute xO to align base edge
            x_target = 0 if rect_edge == 'left' else self.m
            xO = x_target - self.t[p1] * math.cos(phi + self.theta[p1])
            self._log_math(f"xO = {x_target} - t{p1}*cos(phi + theta{p1})", xO)
            
            # 2. Compute yO to keep base vertices within rectangle
            y_p1 = self.t[p1] * math.sin(phi + self.theta[p1])
            y_p2 = self.t[p2] * math.sin(phi + self.theta[p2])
            
            y_min = max(-y_p1, -y_p2)
            y_max = min(self.n - y_p1, self.n - y_p2)
            
            self._log(f"    yO range constraints:")
            self._log(f"      y_min = max({-y_p1:.6f}, {-y_p2:.6f}) = {y_min:.6f}")
            self._log(f"      y_max = min({self.n - y_p1:.6f}, {self.n - y_p2:.6f}) = {y_max:.6f}")
            
            if y_min > y_max + self.TOL:
                self._log("    Invalid position: y_min > y_max")
                return (math.nan, math.nan)
            
            yO = (y_min + y_max) / 2
            self._log(f"    yO = (y_min + y_max)/2 = {yO:.6f}")
        
        return (xO, yO)

    def _verify_solution(self, xO: float, yO: float, phi: float, 
                        p1: int, p2: int, opp_idx: int, 
                        rect_edge: str) -> bool:
        """Verify solution validity"""
        self._log(f"\n    [Verifying solution for phi={phi:.6f}]")
        self._log(f"    Rect edge: {rect_edge}")
        
        # 1. Check base vertices are on specified edge
        self._log("\n    [Base edge verification]")
        for pi in [p1, p2]:
            x_pi = xO + self.t[pi] * math.cos(phi + self.theta[pi])
            y_pi = yO + self.t[pi] * math.sin(phi + self.theta[pi])
            self._log(f"    P{pi} = ({x_pi:.6f}, {y_pi:.6f})")
            
            if rect_edge == 'bottom':
                if not self._log_compare(y_pi, 0, self.TOL):
                    self._log(f"    P{pi} not on bottom edge (y={y_pi:.6f})")
                    return False
            elif rect_edge == 'top':
                if not self._log_compare(y_pi, self.n, self.TOL):
                    self._log(f"    P{pi} not on top edge (y={y_pi:.6f})")
                    return False
            elif rect_edge == 'left':
                if not self._log_compare(x_pi, 0, self.TOL):
                    self._log(f"    P{pi} not on left edge (x={x_pi:.6f})")
                    return False
            elif rect_edge == 'right':
                if not self._log_compare(x_pi, self.m, self.TOL):
                    self._log(f"    P{pi} not on right edge (x={x_pi:.6f})")
                    return False
        
        # 2. Check opposite vertex is on opposite edge
        self._log("\n    [Opposite vertex verification]")
        x_opp = xO + self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
        y_opp = yO + self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
        self._log(f"    P{opp_idx} = ({x_opp:.6f}, {y_opp:.6f})")
        
        if rect_edge in ['bottom', 'top']:
            target_edge = 'top' if rect_edge == 'bottom' else 'bottom'
            if target_edge == 'top':
                if not self._log_compare(y_opp, self.n, self.TOL):
                    self._log(f"    P{opp_idx} not on top edge (y={y_opp:.6f})")
                    return False
            else:
                if not self._log_compare(y_opp, 0, self.TOL):
                    self._log(f"    P{opp_idx} not on bottom edge (y={y_opp:.6f})")
                    return False
        else:
            target_edge = 'right' if rect_edge == 'left' else 'left'
            if target_edge == 'right':
                if not self._log_compare(x_opp, self.m, self.TOL):
                    self._log(f"    P{opp_idx} not on right edge (x={x_opp:.6f})")
                    return False
            else:
                if not self._log_compare(x_opp, 0, self.TOL):
                    self._log(f"    P{opp_idx} not on left edge (x={x_opp:.6f})")
                    return False
        
        # 3. Check all vertices are within rectangle
        self._log("\n    [All vertices bounds check]")
        all_valid = True
        for i in range(3):
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            
            x_ok = (-self.TOL <= x <= self.m + self.TOL)
            y_ok = (-self.TOL <= y <= self.n + self.TOL)
            
            self._log(f"    P{i} = ({x:.6f}, {y:.6f})")
            self._log(f"      x in [{-self.TOL:.6f}, {self.m+self.TOL:.6f}]? {'YES' if x_ok else 'NO'}")
            self._log(f"      y in [{-self.TOL:.6f}, {self.n+self.TOL:.6f}]? {'YES' if y_ok else 'NO'}")
            
            if not (x_ok and y_ok):
                all_valid = False
        
        if not all_valid:
            self._log("    Some vertices out of bounds")
            return False
        
        self._log("    *** All verifications passed ***")
        return True