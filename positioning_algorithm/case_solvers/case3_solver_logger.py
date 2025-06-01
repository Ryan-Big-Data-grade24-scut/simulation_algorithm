import numpy as np
from math import sin, cos, atan2, asin, sqrt, pi

class Case3Solver:
    def __init__(self, t, theta, m, n):
        """
        t: 三角形顶点到中心的距离列表 [t0, t1, t2]
        theta: 三角形顶点角度列表 [θ0, θ1, θ2]（弧度）
        m: 矩形宽度
        n: 矩形高度
        """
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.TOL = 1e-6
        self.log_file = open("case3_solver_log.txt", "w", encoding='utf-8')
    
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

    def solve(self):
        """处理所有情况3的组合"""
        solutions = []
        
        self._log(f"\n=== Starting Case3Solver ===")
        self._log(f"Parameters:")
        self._log(f"  t = {self.t}")
        self._log(f"  theta = {self.theta} (radians)")
        self._log(f"  m = {self.m}")
        self._log(f"  n = {self.n}")
        self._log(f"  Tolerance = {self.TOL}")
        
        # 所有可能的边组合（顶点索引分配）
        edge_combinations = [
            # (p0_edge, p1_edge, p2_edge)
            ('left', 'right', 'top'),
            ('left', 'right', 'bottom'),
            ('left', 'top', 'bottom'),
            ('right', 'top', 'bottom')
        ]
        
        for combo_idx, (p0_edge, p1_edge, p2_edge) in enumerate(edge_combinations, 1):
            self._log(f"\n=== Trying edge combination {combo_idx}: {p0_edge}, {p1_edge}, {p2_edge} ===")
            
            # 尝试所有顶点排列组合（6种排列）
            for p0, p1, p2 in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                self._log(f"\n  Trying vertex permutation: P0={p0}, P1={p1}, P2={p2}")
                
                try:
                    phi_candidates = self._solve_edge_combo(
                        p0, p1, p2, p0_edge, p1_edge, p2_edge)
                    self._log(f"  Found {len(phi_candidates)} phi candidates: {[f'{x:.6f}' for x in phi_candidates]}")
                    
                    for phi in phi_candidates:
                        self._log(f"\n    Testing phi = {phi:.8f} radians ({np.degrees(phi):.2f}°)")
                        
                        self._log("    Computing position (xO, yO)...")
                        xO, yO = self._compute_position(phi, p0, p1, p2, 
                                                      p0_edge, p1_edge, p2_edge)
                        self._log(f"    Computed position: xO={xO:.8f}, yO={yO:.8f}")
                        
                        if np.isnan(xO) or np.isnan(yO):
                            self._log("    Invalid position (NaN), skipping")
                            continue
                            
                        self._log("    Verifying solution...")
                        if self._verify_solution(xO, yO, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
                            self._log("    *** Solution is valid! ***")
                            solutions.append((xO, yO, phi))
                        else:
                            self._log("    Solution failed verification")
                            
                except Exception as e:
                    self._log(f"    Error encountered: {str(e)}")
                    continue
        
        self._log(f"\n=== Solution Summary ===")
        self._log(f"Found {len(solutions)} valid solutions")
        
        for i, (xO, yO, phi) in enumerate(solutions, 1):
            self._log(f"\nSolution {i}:")
            self._log(f"  O = ({xO:.8f}, {yO:.8f})")
            self._log(f"  phi = {phi:.8f} radians ({np.degrees(phi):.2f}°)")
            
            # 计算并打印所有顶点坐标
            for j in range(3):
                x = xO + self.t[j] * cos(phi + self.theta[j])
                y = yO + self.t[j] * sin(phi + self.theta[j])
                self._log(f"  P{j} = ({x:.8f}, {y:.8f})")
            
            # 验证三角形边长（辅助检查）
            def distance(a, b):
                return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            
            P = [(xO + self.t[k] * cos(phi + self.theta[k]),
                 yO + self.t[k] * sin(phi + self.theta[k])) for k in range(3)]
            
            self._log("  Triangle side lengths:")
            self._log(f"    P0-P1: {distance(P[0], P[1]):.8f}")
            self._log(f"    P1-P2: {distance(P[1], P[2]):.8f}")
            self._log(f"    P2-P0: {distance(P[2], P[0]):.8f}")
        
        return solutions
    
    def _solve_edge_combo(self, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """解特定边组合的方程"""
        self._log(f"\n    [Solving phi equation for edge combination]")
        self._log(f"    P0 on {p0_edge}, P1 on {p1_edge}, P2 on {p2_edge}")
        
        if {p0_edge, p1_edge} == {'left', 'right'}:
            self._log("    Case: Left + Right + Top/Bottom")
            # 处理left+right+top/bottom的情况
            A = self.t[p1]*cos(self.theta[p1]) - self.t[p0]*cos(self.theta[p0])
            B = -(self.t[p1]*sin(self.theta[p1]) - self.t[p0]*sin(self.theta[p0]))
            C = self.m
            equation_type = "A*cos(phi) + B*sin(phi) = C (left-right case)"
        elif {p0_edge, p1_edge} == {'left', 'top'} and p2_edge == 'bottom':
            self._log("    Case: Left + Top + Bottom")
            # 处理left+top+bottom的情况（通过p0和p2）
            A = self.t[p2]*sin(self.theta[p2]) - self.t[p0]*sin(self.theta[p0])
            B = self.t[p2]*cos(self.theta[p2]) - self.t[p0]*cos(self.theta[p0])
            C = self.n
            equation_type = "A*cos(phi) + B*sin(phi) = C (left-top-bottom case)"
        elif {p0_edge, p1_edge} == {'right', 'top'} and p2_edge == 'bottom':
            self._log("    Case: Right + Top + Bottom")
            # 处理right+top+bottom的情况（通过p0和p2）
            A = self.t[p2]*sin(self.theta[p2]) - self.t[p0]*sin(self.theta[p0])
            B = self.t[p2]*cos(self.theta[p2]) - self.t[p0]*cos(self.theta[p0])
            C = self.n
            equation_type = "A*cos(phi) + B*sin(phi) = C (right-top-bottom case)"
        else:
            self._log("    Unsupported edge combination, skipping")
            return []
        
        self._log(f"    Equation to solve: {equation_type}")
        self._log_math(f"A = {self.t[p1]}*cos({self.theta[p1]}) - {self.t[p0]}*cos({self.theta[p0]})", A)
        self._log_math(f"B = {self.t[p1]}*sin({self.theta[p1]}) - {self.t[p0]}*sin({self.theta[p0]})", B)
        self._log_math(f"C", C)
        
        # 解 A*cos(phi) + B*sin(phi) = C
        norm = sqrt(A*A + B*B)
        self._log_math(f"norm = sqrt(A² + B²)", norm)
        
        if abs(C) > norm + self.TOL:
            self._log("    No solution: |C| > norm")
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
        
        self._log(f"    Final phi candidates after range adjustment: {[f'{x:.6f}' for x in solutions]}")
        return solutions
    
    def _compute_position(self, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """计算中心点坐标"""
        self._log(f"\n    [Computing position for phi={phi:.6f}]")
        xO, yO = None, None  # 初始化为None，表示尚未计算
        x_values = []  # 存储所有计算出的xO值
        y_values = []  # 存储所有计算出的yO值
        
        # 根据顶点约束计算可能的中心坐标
        def process_edge(vertex, edge, step_name):
            nonlocal xO, yO
            if edge == 'left':
                new_x = -self.t[vertex] * cos(phi + self.theta[vertex])
                self._log(f"    {step_name}: xO = -t{vertex}*cos(phi + theta{vertex}) = {new_x:.6f}")
                x_values.append(new_x)
            elif edge == 'right':
                new_x = self.m - self.t[vertex] * cos(phi + self.theta[vertex])
                self._log(f"    {step_name}: xO = m - t{vertex}*cos(phi + theta{vertex}) = {new_x:.6f}")
                x_values.append(new_x)
            elif edge == 'top':
                new_y = self.n - self.t[vertex] * sin(phi + self.theta[vertex])
                self._log(f"    {step_name}: yO = n - t{vertex}*sin(phi + theta{vertex}) = {new_y:.6f}")
                y_values.append(new_y)
            elif edge == 'bottom':
                new_y = -self.t[vertex] * sin(phi + self.theta[vertex])
                self._log(f"    {step_name}: yO = -t{vertex}*sin(phi + theta{vertex}) = {new_y:.6f}")
                y_values.append(new_y)
        
        # 处理三个顶点的约束
        process_edge(p0, p0_edge, "Initial calculation from P0")
        process_edge(p1, p1_edge, "Adjustment from P1")
        process_edge(p2, p2_edge, "Fine-tuning from P2")
        
        # 确定最终的xO和yO
        if len(x_values) > 1:
            xO = sum(x_values) / len(x_values)
            self._log(f"    Averaged xO from {len(x_values)} values: {xO:.6f}")
        elif len(x_values) == 1:
            xO = x_values[0]
            self._log(f"    Using single xO value: {xO:.6f}")
        
        if len(y_values) > 1:
            yO = sum(y_values) / len(y_values)
            self._log(f"    Averaged yO from {len(y_values)} values: {yO:.6f}")
        elif len(y_values) == 1:
            yO = y_values[0]
            self._log(f"    Using single yO value: {yO:.6f}")
        
        # 如果某个坐标仍未确定，使用默认值0
        if xO is None:
            xO = 0
            self._log("    No x constraints, defaulting xO to 0")
        if yO is None:
            yO = 0
            self._log("    No y constraints, defaulting yO to 0")
        
        self._log(f"\n    Final position: xO={xO:.8f}, yO={yO:.8f}")
        return xO, yO
    
    def _verify_solution(self, xO, yO, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """验证所有顶点是否在指定边且不越界"""
        self._log(f"\n    [Verifying solution for phi={phi:.6f}]")
        self._log(f"    P0 on {p0_edge}, P1 on {p1_edge}, P2 on {p2_edge}")
        
        # 计算三个顶点的坐标
        points = []
        for i in [p0, p1, p2]:
            x = xO + self.t[i] * cos(phi + self.theta[i])
            y = yO + self.t[i] * sin(phi + self.theta[i])
            points.append((x, y))
            self._log(f"    P{i} = ({x:.8f}, {y:.8f})")
        
        # 检查每个顶点是否在指定边
        edge_checks = {
            'left': lambda x, y: (abs(x) < self.TOL, f"P on left edge: x={x:.6f} ≈ 0"),
            'right': lambda x, y: (abs(x - self.m) < self.TOL, f"P on right edge: x={x:.6f} ≈ {self.m}"),
            'top': lambda x, y: (abs(y - self.n) < self.TOL, f"P on top edge: y={y:.6f} ≈ {self.n}"),
            'bottom': lambda x, y: (abs(y) < self.TOL, f"P on bottom edge: y={y:.6f} ≈ 0")
        }
        
        edge_constraints = {
            'left': lambda x, y: (0 <= y <= self.n + self.TOL, f"P on left edge within y bounds [0, {self.n}]"),
            'right': lambda x, y: (0 <= y <= self.n + self.TOL, f"P on right edge within y bounds [0, {self.n}]"),
            'top': lambda x, y: (0 <= x <= self.m + self.TOL, f"P on top edge within x bounds [0, {self.m}]"),
            'bottom': lambda x, y: (0 <= x <= self.m + self.TOL, f"P on bottom edge within x bounds [0, {self.m}]")
        }
        
        edges = [p0_edge, p1_edge, p2_edge]
        for (x, y), edge in zip(points, edges):
            # 检查是否在正确的边上
            on_edge, msg = edge_checks[edge](x, y)
            self._log(f"    {msg}")
            if not on_edge:
                self._log(f"    Vertex not on {edge} edge")
                return False
            
            # 检查是否在边的有效范围内
            in_bounds, msg = edge_constraints[edge](x, y)
            self._log(f"    {msg}")
            if not in_bounds:
                self._log(f"    Vertex out of bounds on {edge} edge")
                return False
        
        # 检查第四个边没有顶点
        all_edges = {'left', 'right', 'top', 'bottom'}
        used_edges = set(edges)
        free_edge = list(all_edges - used_edges)[0]
        self._log(f"\n    Checking no vertex is on free edge: {free_edge}")
        
        for (x, y), i in zip(points, [p0, p1, p2]):
            on_free_edge, _ = edge_checks[free_edge](x, y)
            if on_free_edge:
                self._log(f"    P{i} is incorrectly on {free_edge} edge")
                return False
        
        # 检查所有顶点是否在矩形内
        self._log("\n    Checking all vertices within rectangle bounds")
        for (x, y), i in zip(points, [p0, p1, p2]):
            x_ok = (-self.TOL <= x <= self.m + self.TOL)
            y_ok = (-self.TOL <= y <= self.n + self.TOL)
            self._log(f"    P{i}: x in [0, {self.m}]? {'YES' if x_ok else 'NO'}")
            self._log(f"    P{i}: y in [0, {self.n}]? {'YES' if y_ok else 'NO'}")
            if not (x_ok and y_ok):
                self._log(f"    P{i} out of rectangle bounds")
                return False
        
        self._log("    *** All verifications passed ***")
        return True

# 示例用法
if __name__ == "__main__":
    # 示例三角形（等腰直角三角形）
    t = [1, 1, np.sqrt(2)]
    theta = [-np.pi/4, np.pi/4, 3*np.pi/4]
    m, n = 4, 3  # 矩形尺寸
    
    solver = Case3Solver(t, theta, m, n)
    solutions = solver.solve()
    
    print(f"找到 {len(solutions)} 个解:")
    for i, (xO, yO, phi) in enumerate(solutions, 1):
        print(f"\n解 {i}:")
        print(f"中心位置: ({xO:.6f}, {yO:.6f})")
        print(f"旋转角度: {phi:.6f} radians ({np.degrees(phi):.2f}°)")
        
        # 计算并打印顶点坐标
        for j in range(3):
            x = xO + t[j] * cos(phi + theta[j])
            y = yO + t[j] * sin(phi + theta[j])
            print(f"P{j} = ({x:.6f}, {y:.6f})")