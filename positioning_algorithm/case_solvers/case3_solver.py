
import numpy as np
import math
from math import sin, cos, atan2, asin, sqrt, pi
import os
from .BaseSolver import BaseSolver, BaseSolverConfig
import logging

class Case3Solver(BaseSolver):
    """情况3求解器：三个顶点分别在三个不同边缘
    Args:
        t (List[float]): 3个激光测距值 [t0, t1, t2] (单位: m)
        theta (List[float]): 3个激光角度 [θ0, θ1, θ2] (单位: rad)
        m (float): 场地x方向长度 (m)
        n (float): 场地y方向长度 (m)
        config (BaseSolverConfig): 基础配置
        ros_logger (Optional): ROS2日志器对象
    """

    def __init__(self, t, theta, m, n, config=None, ros_logger=None, min_log_level=logging.DEBUG):
        config = config or BaseSolverConfig(
            log_file=os.path.join('logs', 'case3.log'),
            log_level='DEBUG'
        )
        super().__init__(t, theta, m, n, config, ros_logger, min_log_level=min_log_level)
        os.makedirs('logs', exist_ok=True)
        self.logger.info("Case3Solver initialized")
        self._validate_case3_inputs()

    def _validate_case3_inputs(self):
        """验证情况3输入条件"""
        if len(self.t) != 3 or len(self.theta) != 3:
            error_msg = "情况3需要3个激光测距值和角度"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        for i, (t_val, theta_val) in enumerate(zip(self.t, self.theta)):
            if t_val <= 0:
                error_msg = f"激光测距值必须为正数，P{i}的t={t_val}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            if not (-pi <= theta_val <= pi):
                error_msg = f"激光角度必须在[-π, π]范围内，P{i}的theta={theta_val}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        self.logger.debug("输入参数验证通过")

    def solve(self):
        """处理所有情况3的组合，返回统一格式的解"""
        solutions = []
        self.logger.info("\n=== Starting Case3Solver ===")
        self.logger.info(f"Parameters: t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
        edge_combinations = [
            ('left', 'right', 'top'),
            ('left', 'right', 'bottom'),
            ('left', 'top', 'bottom'),
            ('right', 'top', 'bottom')
        ]
        for combo_idx, (p0_edge, p1_edge, p2_edge) in enumerate(edge_combinations, 1):
            self.logger.info(f"\n=== Trying edge combination {combo_idx}: {p0_edge}, {p1_edge}, {p2_edge} ===")
            for p0, p1, p2 in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
                self.logger.info(f"\n  Trying vertex permutation: P0={p0}, P1={p1}, P2={p2}")
                try:
                    phi_candidates = self._solve_edge_combo(
                        p0, p1, p2, p0_edge, p1_edge, p2_edge)
                    self.logger.info(f"  Found {len(phi_candidates)} phi candidates")
                    for phi in phi_candidates:
                        self.logger.info(f"\n    Testing phi = {phi:.8f} radians ({np.degrees(phi):.2f}°)")
                        xO, yO = self._compute_position(phi, p0, p1, p2, p0_edge, p1_edge, p2_edge)
                        self.logger.info(f"    Computed position: xO={xO:.8f}, yO={yO:.8f}")
                        if np.isnan(xO) or np.isnan(yO):
                            self.logger.warning("    Invalid position (NaN), skipping")
                            continue
                        if self._verify_solution(xO, yO, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
                            self.logger.info("    *** Solution is valid! ***")
                            solutions.append(((xO, xO), (yO, yO), phi))
                        else:
                            self.logger.warning("    Solution failed verification")
                except Exception as e:
                    self.logger.error(f"    Error encountered: {str(e)}", exc_info=True)
                    continue
        self.logger.info(f"\n=== Solution Summary ===")
        self.logger.info(f"Found {len(solutions)} valid solutions")
        for i, (x_range, y_range, phi) in enumerate(solutions, 1):
            self.logger.info(f"Solution #{i}:")
            self.logger.info(f"  x range: [{x_range[0]:.6f}, {x_range[1]:.6f}]")
            self.logger.info(f"  y range: [{y_range[0]:.6f}, {y_range[1]:.6f}]")
            self.logger.info(f"  phi: {phi:.6f} rad ({np.degrees(phi):.2f}°)")
        return solutions

    def _solve_edge_combo(self, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """解特定边组合的方程"""
        self.logger.info(f"\n    === 求解边组合方程 ===")
        self.logger.info(f"    顶点分配: P{p0}在{p0_edge}边, P{p1}在{p1_edge}边, P{p2}在{p2_edge}边")
        
        if {p0_edge, p1_edge} == {'left', 'right'}:
            self.logger.info("    情况: 左边+右边+顶边/底边")
            self.logger.info("    步骤1: 计算A系数")
            self.logger.info(f"      公式: A = t{p1}*cos(θ{p1}) - t{p0}*cos(θ{p0})")
            cos_p1 = cos(self.theta[p1])
            cos_p0 = cos(self.theta[p0])
            self.logger.info(f"      计算: cos({self.theta[p1]:.6f}) = {cos_p1:.8f}")
            self.logger.info(f"      计算: cos({self.theta[p0]:.6f}) = {cos_p0:.8f}")
            self.logger.info(f"      代入: A = {self.t[p1]:.6f}*{cos_p1:.8f} - {self.t[p0]:.6f}*{cos_p0:.8f}")
            term1_A = self.t[p1] * cos_p1
            term2_A = self.t[p0] * cos_p0
            self.logger.info(f"      计算: A = {term1_A:.8f} - {term2_A:.8f}")
            A = term1_A - term2_A
            self.logger.info(f"      结果: A = {A:.8f}")
            
            self.logger.info("    步骤2: 计算B系数")
            self.logger.info(f"      公式: B = -(t{p1}*sin(θ{p1}) - t{p0}*sin(θ{p0}))")
            sin_p1 = sin(self.theta[p1])
            sin_p0 = sin(self.theta[p0])
            self.logger.info(f"      计算: sin({self.theta[p1]:.6f}) = {sin_p1:.8f}")
            self.logger.info(f"      计算: sin({self.theta[p0]:.6f}) = {sin_p0:.8f}")
            term1_B_inner = self.t[p1] * sin_p1
            term2_B_inner = self.t[p0] * sin_p0
            self.logger.info(f"      计算: t{p1}*sin(θ{p1}) - t{p0}*sin(θ{p0}) = {term1_B_inner:.8f} - {term2_B_inner:.8f}")
            B_inner = term1_B_inner - term2_B_inner
            self.logger.info(f"      代入: B = -({B_inner:.8f})")
            B = -B_inner
            self.logger.info(f"      结果: B = {B:.8f}")
            
            C = self.m
            self.logger.info(f"    步骤3: C系数")
            self.logger.info(f"      公式: C = m (场地宽度)")
            self.logger.info(f"      结果: C = {C:.8f}")
            
        elif {p2_edge, p1_edge} == {'bottom', 'top'}:
            self.logger.info("    情况: 底边+顶边")
            self.logger.info("    步骤1: 计算A系数")
            self.logger.info(f"      公式: A = t{p2}*sin(θ{p2}) - t{p1}*sin(θ{p1})")
            sin_p2 = sin(self.theta[p2])
            sin_p1 = sin(self.theta[p1])
            self.logger.info(f"      计算: sin({self.theta[p2]:.6f}) = {sin_p2:.8f}")
            self.logger.info(f"      计算: sin({self.theta[p1]:.6f}) = {sin_p1:.8f}")
            self.logger.info(f"      代入: A = {self.t[p2]:.6f}*{sin_p2:.8f} - {self.t[p1]:.6f}*{sin_p1:.8f}")
            term1_A = self.t[p2] * sin_p2
            term2_A = self.t[p1] * sin_p1
            self.logger.info(f"      计算: A = {term1_A:.8f} - {term2_A:.8f}")
            A = term1_A - term2_A
            self.logger.info(f"      结果: A = {A:.8f}")
            
            self.logger.info("    步骤2: 计算B系数")
            self.logger.info(f"      公式: B = t{p2}*cos(θ{p2}) - t{p1}*cos(θ{p1})")
            cos_p2 = cos(self.theta[p2])
            cos_p1 = cos(self.theta[p1])
            self.logger.info(f"      计算: cos({self.theta[p2]:.6f}) = {cos_p2:.8f}")
            self.logger.info(f"      计算: cos({self.theta[p1]:.6f}) = {cos_p1:.8f}")
            self.logger.info(f"      代入: B = {self.t[p2]:.6f}*{cos_p2:.8f} - {self.t[p1]:.6f}*{cos_p1:.8f}")
            term1_B = self.t[p2] * cos_p2
            term2_B = self.t[p1] * cos_p1
            self.logger.info(f"      计算: B = {term1_B:.8f} - {term2_B:.8f}")
            B = term1_B - term2_B
            self.logger.info(f"      结果: B = {B:.8f}")
            
            C = -self.n
            self.logger.info(f"    步骤3: C系数")
            self.logger.info(f"      公式: C = -n (负场地高度)")
            self.logger.info(f"      结果: C = {C:.8f}")
            
        else:
            self.logger.info("    不支持的边组合，跳过")
            return []
            
        # 计算模长
        self.logger.info("    步骤4: 计算模长")
        self.logger.info(f"      公式: norm = sqrt(A² + B²)")
        self.logger.info(f"      代入: norm = sqrt({A:.8f}² + {B:.8f}²)")
        A_squared = A * A
        B_squared = B * B
        self.logger.info(f"      计算: norm = sqrt({A_squared:.8f} + {B_squared:.8f})")
        norm_squared = A_squared + B_squared
        norm = sqrt(norm_squared)
        self.logger.info(f"      结果: norm = {norm:.8f}")
        
        # 可解性检查
        self.logger.info("    步骤5: 可解性检查")
        self.logger.info(f"      判断条件: |C| ≤ norm + tolerance")
        C_abs = abs(C)
        norm_plus_tol = norm + self.config.tol
        self.logger.info(f"      计算: |C| = |{C:.8f}| = {C_abs:.8f}")
        self.logger.info(f"      计算: norm + tol = {norm:.8f} + {self.config.tol} = {norm_plus_tol:.8f}")
        self.logger.info(f"      判断: {C_abs:.8f} ≤ {norm_plus_tol:.8f} ? {C_abs <= norm_plus_tol}")
        
        if abs(C) > norm + self.config.tol:
            self.logger.info("      结论: 无解 (|C| > norm)")
            return []
        else:
            self.logger.info("      结论: 有解，继续计算")
            
        # 计算角度
        self.logger.info("    步骤6: 计算角度")
        self.logger.info(f"      公式: alpha = atan2(A, B)")
        self.logger.info(f"      代入: alpha = atan2({A:.8f}, {B:.8f})")
        alpha = atan2(A, B)
        self.logger.info(f"      结果: alpha = {alpha:.8f} rad = {np.degrees(alpha):.2f}°")
        
        self.logger.info(f"      公式: phi1 = asin(C/norm) - alpha")
        ratio = C / norm
        self.logger.info(f"      计算: C/norm = {C:.8f}/{norm:.8f} = {ratio:.8f}")
        asin_val = asin(ratio)
        self.logger.info(f"      计算: asin({ratio:.8f}) = {asin_val:.8f}")
        self.logger.info(f"      代入: phi1 = {asin_val:.8f} - {alpha:.8f}")
        phi1 = asin_val - alpha
        self.logger.info(f"      结果: phi1 = {phi1:.8f} rad = {np.degrees(phi1):.2f}°")
        
        self.logger.info(f"      公式: phi2 = π - asin(C/norm) - alpha")
        self.logger.info(f"      代入: phi2 = {pi:.8f} - {asin_val:.8f} - {alpha:.8f}")
        phi2 = pi - asin_val - alpha
        self.logger.info(f"      结果: phi2 = {phi2:.8f} rad = {np.degrees(phi2):.2f}°")
        
        # 角度范围调整
        self.logger.info("    步骤7: 角度范围调整")
        solutions = []
        for i, phi in enumerate([phi1, phi2], 1):
            self.logger.info(f"      调整phi{i}:")
            original_phi = phi
            if phi < -pi:
                phi += 2*pi
                self.logger.info(f"        phi{i} < -π，调整: {original_phi:.8f} + 2π = {phi:.8f}")
            elif phi > pi:
                phi -= 2*pi
                self.logger.info(f"        phi{i} > π，调整: {original_phi:.8f} - 2π = {phi:.8f}")
            else:
                self.logger.info(f"        phi{i}在范围内: {phi:.8f}")
            solutions.append(phi)
        
        self.logger.info(f"    === 边组合方程求解完成，得到{len(solutions)}个候选角度 ===")
        return solutions
        self.logger.debug(f"    Final phi candidates after range adjustment: {solutions}")
        return solutions

    def _compute_position(self, phi, p0, p1, p2, p0_edge, p1_edge, p2_edge):
        """计算中心点坐标"""
        self.logger.debug(f"\n    [Computing position for phi={phi:.6f}]")
        xO, yO = None, None
        x_values = []
        y_values = []
        def process_edge(vertex, edge, step_name):
            nonlocal xO, yO
            if edge == 'left':
                new_x = -self.t[vertex] * cos(phi + self.theta[vertex])
                self._log_math(f"{step_name}: xO = -t{vertex}*cos(phi + theta{vertex})", new_x)
                self._log_math(f"{step_name}: xO = -{self.t[vertex]:.6f} * cos({phi:.6f} + {self.theta[vertex]:.6f}) = {new_x:.6f}", new_x)
                x_values.append(new_x)
            elif edge == 'right':
                new_x = self.m - self.t[vertex] * cos(phi + self.theta[vertex])
                self._log_math(f"{step_name}: xO = m - t{vertex}*cos(phi + theta{vertex})", new_x)
                self._log_math(f"{step_name}: xO = {self.m:.6f} - {self.t[vertex]:.6f} * cos({phi:.6f} + {self.theta[vertex]:.6f}) = {new_x:.6f}", new_x)
                x_values.append(new_x)
            elif edge == 'top':
                new_y = self.n - self.t[vertex] * sin(phi + self.theta[vertex])
                self._log_math(f"{step_name}: yO = n - t{vertex}*sin(phi + theta{vertex})", new_y)
                self._log_math(f"{step_name}: yO = {self.n:.6f} - {self.t[vertex]:.6f} * sin({phi:.6f} + {self.theta[vertex]:.6f}) = {new_y:.6f}", new_y)
                y_values.append(new_y)
            elif edge == 'bottom':
                new_y = -self.t[vertex] * sin(phi + self.theta[vertex])
                self._log_math(f"{step_name}: yO = -t{vertex}*sin(phi + theta{vertex})", new_y)
                self._log_math(f"{step_name}: yO = -{self.t[vertex]:.6f} * sin({phi:.6f} + {self.theta[vertex]:.6f}) = {new_y:.6f}", new_y)
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
        """验证所有顶点是否在指定边且不越界"""
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
            'left': lambda x, y: (0 <= y <= self.n + self.config.tol, f"P on left edge within y bounds [0, {self.n}]") ,
            'right': lambda x, y: (0 <= y <= self.n + self.config.tol, f"P on right edge within y bounds [0, {self.n}]") ,
            'top': lambda x, y: (0 <= x <= self.m + self.config.tol, f"P on top edge within x bounds [0, {self.m}]") ,
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

    def _log_math(self, expr, value):
        """记录数学计算过程"""
        self.logger.debug(f"      {expr} = {value:.8f}")
