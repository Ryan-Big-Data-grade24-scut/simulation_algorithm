import math
import sys
import os
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

# 路径处理
if __name__ == '__main__':
    sys.path.append('/home/lrx/laser_positioning_ubuntu/simulation_algorithm/positioning_algorithm')
    from case_solvers.BaseSolver import BaseSolver, BaseSolverConfig
else:
    from .BaseSolver import BaseSolver, BaseSolverConfig

@dataclass
class Case2Config:
    """Case2 求解器特有配置
    
    Attributes:
        range_tolerance (float): 坐标范围验证容忍度，默认1e-5
        enable_edge_debug (bool): 是否启用边缘调试日志，默认False
    """
    range_tolerance: float = 1e-5
    enable_edge_debug: bool = False

class Case2Solver(BaseSolver):
    """情况2求解器：一边在矩形边缘且对角顶点在对边
    
    Args:
        t (List[float]): 3个激光测距值 [t0, t1, t2] (单位: m)
        theta (List[float]): 3个激光角度 [θ0, θ1, θ2] (单位: rad)
        m (float): 场地x方向长度 (m)
        n (float): 场地y方向长度 (m)
        config (Optional[BaseSolverConfig]): 基础配置
        case_config (Optional[Case2Config]): 情况2特有配置
        ros_logger (Optional): ROS2日志器对象
    """

    def __init__(self, 
                 t: List[float], 
                 theta: List[float],
                 m: float, 
                 n: float, 
                 config: Optional[BaseSolverConfig] = None,
                 case_config: Optional[Case2Config] = None,
                 ros_logger=None,
                 min_log_level=logging.DEBUG):
        # 初始化基础配置
        config = config or BaseSolverConfig(
            log_file=os.path.join('logs', 'case2.log'),
            log_level='DEBUG'
        )
        super().__init__(t, theta, m, n, config, ros_logger, min_log_level=min_log_level)
        
        # 情况2特有配置
        self.case_config = case_config or Case2Config()
        self.edges = ['bottom', 'top', 'left', 'right']
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        self.logger.info(f"Case2Solver initialized with {self.case_config}")

    def _log_math(self, expr: str, value: float):
        """记录数学计算过程 - 兼容方法"""
        self.logger.info(f"[MATH] {expr} = {value:.6f}")

    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """执行情况2求解流程
        
        Returns:
            List[Tuple]: 有效解列表，每个解格式为:
                ((x_min, x_max), (y_min, y_max), phi)
                
        Raises:
            RuntimeError: 当求解过程中出现不可恢复错误
        """
        solutions = []
        try:
            self.logger.info(f"\n=== Starting Case2Solver ===")
            self.logger.info(f"Parameters: t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
            self.logger.info(f"Start solving with t={self.t}, theta={self.theta}")
            
            for base_idx in range(3):
                solutions.extend(
                    self._solve_for_base_edge(base_idx)
                )
                
            self.logger.info(f"\n=== Case2 Solution Summary ===")
            self.logger.info(f"Found {len(solutions)} valid solutions")
            for i, (x_range, y_range, phi) in enumerate(solutions, 1):
                self.logger.info(f"Solution #{i}:")
                self.logger.info(f"  x range: [{x_range[0]:.6f}, {x_range[1]:.6f}]")
                self.logger.info(f"  y range: [{y_range[0]:.6f}, {y_range[1]:.6f}]")
                self.logger.info(f"  phi: {phi:.6f} rad ({math.degrees(phi):.2f}°)")
            return solutions
            
        except Exception as e:
            error_msg = f"Solve failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _solve_for_base_edge(self, base_idx: int) -> List[Tuple]:
        """处理特定基边的所有可能解
        
        Args:
            base_idx (int): 基边顶点索引 (0-2)
            
        Returns:
            List[Tuple]: 该基边下的有效解
        """
        solutions = []
        p1, p2 = base_idx, (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        self.logger.info(f"\n=== 处理基边 {base_idx} (P{p1}-P{p2}) ===")
        self.logger.info(f"基边顶点: P{p1}, P{p2}")
        self.logger.info(f"对角顶点: P{opp_idx}")
        self.logger.info(f"激光数据: t{p1}={self.t[p1]:.6f}, t{p2}={self.t[p2]:.6f}, t{opp_idx}={self.t[opp_idx]:.6f}")
        self.logger.info(f"角度数据: θ{p1}={self.theta[p1]:.6f}, θ{p2}={self.theta[p2]:.6f}, θ{opp_idx}={self.theta[opp_idx]:.6f}")
        
        for rect_edge in self.edges:
            try:
                self.logger.info(f"\n  --- 尝试矩形边: {rect_edge} ---")
                self.logger.info(f"  配置说明: 基边P{p1}-P{p2}在{rect_edge}边，对角顶点P{opp_idx}在对边")
                
                phi_list = self._solve_phi_equation(p1, p2, opp_idx, rect_edge)
                self.logger.info(f"  求解得到{len(phi_list)}个候选角度phi")
                
                for i, phi in enumerate(phi_list):
                    self.logger.info(f"\n    ++ 测试第{i+1}个候选角度 ++")
                    self.logger.info(f"    phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
                    
                    position = self._compute_position(phi, p1, p2, opp_idx, rect_edge)
                    if position:
                        x_range, y_range = position
                        self.logger.info(f"    计算得到坐标范围: x=[{x_range[0]:.8f}, {x_range[1]:.8f}], y=[{y_range[0]:.8f}, {y_range[1]:.8f}]")
                        
                        if self._verify_solution(x_range, y_range, phi, p1, p2, opp_idx, rect_edge):
                            solutions.append((x_range, y_range, phi))
                            self.logger.info(f"    *** ✓ 找到有效解: x=[{x_range[0]:.6f}, {x_range[1]:.6f}], y=[{y_range[0]:.6f}, {y_range[1]:.6f}], phi={phi:.6f} ***")
                        else:
                            self.logger.info(f"    ✗ 解验证失败")
                    else:
                        self.logger.info(f"    ✗ 位置计算失败")
                        
            except Exception as e:
                self.logger.warning(f"  矩形边{rect_edge}处理失败: {str(e)}")
                continue
        
        self.logger.info(f"=== 基边{base_idx}处理完成，共找到{len(solutions)}个解 ===")
        return solutions

    def _solve_phi_equation(self, 
                          p1: int, 
                          p2: int,
                          opp_idx: int,
                          rect_edge: str) -> List[float]:
        """求解phi的候选角度
        
        Args:
            p1: 基边顶点1索引
            p2: 基边顶点2索引
            opp_idx: 对角顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            List[float]: 有效的phi角度列表 (rad)
        """
        self.logger.info(f"\n    === 求解phi方程 ===")
        self.logger.info(f"    基边顶点: P{p1}, P{p2}")
        self.logger.info(f"    对角顶点: P{opp_idx}")
        self.logger.info(f"    矩形边缘: {rect_edge}")
        
        # 确定目标值和方程类型
        if rect_edge in ['bottom', 'top']:
            target = self.n if rect_edge == 'bottom' else -self.n
            equation_type = "A*sin(phi) + B*cos(phi) = C"
            self.logger.info(f"    方程类型: {equation_type}")
            self.logger.info(f"    目标值: C = {target:.6f} (对角顶点需要到达{'顶边' if rect_edge == 'bottom' else '底边'})")
        else:
            target = self.m if rect_edge == 'left' else -self.m
            equation_type = "A*cos(phi) - B*sin(phi) = C"
            self.logger.info(f"    方程类型: {equation_type}")
            self.logger.info(f"    目标值: C = {target:.6f} (对角顶点需要到达{'右边' if rect_edge == 'left' else '左边'})")
        
        # 计算A系数
        self.logger.info(f"    步骤1: 计算A系数")
        self.logger.info(f"      公式: A = t{opp_idx}*cos(θ{opp_idx}) - t{p1}*cos(θ{p1})")
        cos_opp = math.cos(self.theta[opp_idx])
        cos_p1 = math.cos(self.theta[p1])
        self.logger.info(f"      计算: cos({self.theta[opp_idx]:.6f}) = {cos_opp:.8f}")
        self.logger.info(f"      计算: cos({self.theta[p1]:.6f}) = {cos_p1:.8f}")
        self.logger.info(f"      代入: A = {self.t[opp_idx]:.6f}*{cos_opp:.8f} - {self.t[p1]:.6f}*{cos_p1:.8f}")
        term1_A = self.t[opp_idx] * cos_opp
        term2_A = self.t[p1] * cos_p1
        self.logger.info(f"      计算: A = {term1_A:.8f} - {term2_A:.8f}")
        A = term1_A - term2_A
        self.logger.info(f"      结果: A = {A:.8f}")
        
        # 计算B系数
        self.logger.info(f"    步骤2: 计算B系数")
        self.logger.info(f"      公式: B = t{opp_idx}*sin(θ{opp_idx}) - t{p1}*sin(θ{p1})")
        sin_opp = math.sin(self.theta[opp_idx])
        sin_p1 = math.sin(self.theta[p1])
        self.logger.info(f"      计算: sin({self.theta[opp_idx]:.6f}) = {sin_opp:.8f}")
        self.logger.info(f"      计算: sin({self.theta[p1]:.6f}) = {sin_p1:.8f}")
        self.logger.info(f"      代入: B = {self.t[opp_idx]:.6f}*{sin_opp:.8f} - {self.t[p1]:.6f}*{sin_p1:.8f}")
        term1_B = self.t[opp_idx] * sin_opp
        term2_B = self.t[p1] * sin_p1
        self.logger.info(f"      计算: B = {term1_B:.8f} - {term2_B:.8f}")
        B = term1_B - term2_B
        self.logger.info(f"      结果: B = {B:.8f}")
        
        # 计算模长
        self.logger.info(f"    步骤3: 计算模长")
        self.logger.info(f"      公式: norm = sqrt(A² + B²)")
        self.logger.info(f"      代入: norm = sqrt({A:.8f}² + {B:.8f}²)")
        A_squared = A * A
        B_squared = B * B
        self.logger.info(f"      计算: norm = sqrt({A_squared:.8f} + {B_squared:.8f})")
        norm_squared = A_squared + B_squared
        norm = math.sqrt(norm_squared)
        self.logger.info(f"      计算: norm = sqrt({norm_squared:.8f})")
        self.logger.info(f"      结果: norm = {norm:.8f}")
        
        # 退化情况处理
        self.logger.info(f"    步骤4: 退化情况检查")
        self.logger.info(f"      判断条件: norm < tolerance")
        self.logger.info(f"      比较: {norm:.8f} < {self.config.tol} ? {norm < self.config.tol}")
        if norm < self.config.tol:
            self.logger.info("      结论: 退化情况 (norm ≈ 0)，方程无意义")
            return []
        else:
            self.logger.info("      结论: 非退化情况，继续求解")
            
        # 特殊解情况
        self.logger.info(f"    步骤5: 特殊解检查")
        target_abs = abs(target)
        self.logger.info(f"      判断条件: |target| ≈ norm")
        self.logger.info(f"      计算: |target| = |{target:.8f}| = {target_abs:.8f}")
        self.logger.info(f"      比较: |{target_abs:.8f} - {norm:.8f}| = {abs(target_abs - norm):.8f}")
        self.logger.info(f"      判断: {abs(target_abs - norm):.8f} < {self.config.tol} ? {abs(target_abs - norm) < self.config.tol}")
        
        if abs(abs(target) - norm) < self.config.tol:
            self.logger.info("      结论: 特殊情况 (|target| ≈ norm)，只有一个解")
            sign = 1 if target > 0 else -1
            self.logger.info(f"      符号: sign = {sign} (target = {target:.8f})")
            
            if rect_edge in ['bottom', 'top']:
                self.logger.info(f"      公式: phi = atan2(sign*B, sign*A)")
                self.logger.info(f"      代入: phi = atan2({sign}*{B:.8f}, {sign}*{A:.8f})")
                self.logger.info(f"      计算: phi = atan2({sign*B:.8f}, {sign*A:.8f})")
                phi = math.atan2(sign*B, sign*A)
                self.logger.info(f"      结果: phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
                return [phi]
            else:
                self.logger.info(f"      公式: phi = atan2(sign*B, -sign*A)")
                self.logger.info(f"      代入: phi = atan2({sign}*{B:.8f}, -{sign}*{A:.8f})")
                self.logger.info(f"      计算: phi = atan2({sign*B:.8f}, {-sign*A:.8f})")
                phi = math.atan2(sign*B, -sign*A)
                self.logger.info(f"      结果: phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
                return [phi]
                
        # 无解情况
        self.logger.info(f"    步骤6: 无解情况检查")
        self.logger.info(f"      判断条件: norm < |target|")
        self.logger.info(f"      比较: {norm:.8f} < {target_abs:.8f} ? {norm < abs(target)}")
        if norm < abs(target):
            self.logger.info("      结论: 无解 (norm < |target|)，方程无实数解")
            return []
        else:
            self.logger.info("      结论: 有解，继续计算一般解")
            
        # 一般解
        self.logger.info(f"    步骤7: 计算一般解")
        self.logger.info(f"      公式: alpha = atan2(B, A)")
        self.logger.info(f"      代入: alpha = atan2({B:.8f}, {A:.8f})")
        alpha = math.atan2(B, A)
        self.logger.info(f"      结果: alpha = {alpha:.8f} rad = {math.degrees(alpha):.2f}°")
        
        if rect_edge in ['bottom', 'top']:
            self.logger.info(f"      水平边情况，使用asin公式")
            self.logger.info(f"      公式: phi1 = asin(target/norm) - alpha")
            self.logger.info(f"      公式: phi2 = π - asin(target/norm) - alpha")
            
            ratio = target / norm
            self.logger.info(f"      计算: target/norm = {target:.8f}/{norm:.8f} = {ratio:.8f}")
            asin_val = math.asin(ratio)
            self.logger.info(f"      计算: asin({ratio:.8f}) = {asin_val:.8f}")
            
            self.logger.info(f"      代入: phi1 = {asin_val:.8f} - {alpha:.8f}")
            phi1 = asin_val - alpha
            self.logger.info(f"      结果: phi1 = {phi1:.8f} rad = {math.degrees(phi1):.2f}°")
            
            self.logger.info(f"      代入: phi2 = {math.pi:.8f} - {asin_val:.8f} - {alpha:.8f}")
            phi2 = math.pi - asin_val - alpha
            self.logger.info(f"      结果: phi2 = {phi2:.8f} rad = {math.degrees(phi2):.2f}°")
        else:
            self.logger.info(f"      垂直边情况，使用acos公式")
            self.logger.info(f"      公式: phi1 = acos(target/norm) - alpha")
            self.logger.info(f"      公式: phi2 = -acos(target/norm) - alpha")
            
            ratio = target / norm
            self.logger.info(f"      计算: target/norm = {target:.8f}/{norm:.8f} = {ratio:.8f}")
            acos_val = math.acos(ratio)
            self.logger.info(f"      计算: acos({ratio:.8f}) = {acos_val:.8f}")
            
            self.logger.info(f"      代入: phi1 = {acos_val:.8f} - {alpha:.8f}")
            phi1 = acos_val - alpha
            self.logger.info(f"      结果: phi1 = {phi1:.8f} rad = {math.degrees(phi1):.2f}°")
            
            self.logger.info(f"      代入: phi2 = -{acos_val:.8f} - {alpha:.8f}")
            phi2 = -acos_val - alpha
            self.logger.info(f"      结果: phi2 = {phi2:.8f} rad = {math.degrees(phi2):.2f}°")
            
        candidates = [phi1, phi2]
        self.logger.info(f"    步骤8: 候选角度验证")
        valid_candidates = []
        for i, phi in enumerate(candidates):
            self.logger.info(f"      验证候选{i+1}: phi = {phi:.8f} rad")
            if self._validate_phi(phi, p1, rect_edge):
                self.logger.info(f"        ✓ 候选{i+1}通过验证")
                valid_candidates.append(phi)
            else:
                self.logger.info(f"        ✗ 候选{i+1}验证失败")
        
        self.logger.info(f"    === phi方程求解完成，得到{len(valid_candidates)}个有效角度 ===")
        return valid_candidates

    def _validate_phi(self, 
                    phi: float,
                    base_idx: int,
                    rect_edge: str) -> bool:
        """快速验证phi角度有效性
        
        Args:
            phi: 待验证角度 (rad)
            base_idx: 基边顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            bool: 是否可能为有效角度
        """
        self.logger.info(f"        角度预验证: phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
        
        # 只检查数值有效性
        if not math.isfinite(phi):
            self.logger.info(f"        ✗ 角度不是有限数值")
            return False
        
        # 防止明显的计算错误（过大的角度值）
        if abs(phi) > 100:
            self.logger.info(f"        ✗ 角度绝对值过大 (|{phi:.6f}| > 100)")
            return False
            
        self.logger.info(f"        ✓ 角度通过预验证")
        return True

    def _compute_position(self,
                         phi: float,
                         p1: int,
                         p2: int,
                         opp_idx: int,
                         rect_edge: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """计算中心点坐标范围
        
        Args:
            phi: 当前角度 (rad)
            p1: 基边顶点1索引
            p2: 基边顶点2索引
            opp_idx: 对角顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            Optional[Tuple]: 坐标范围 ((x_min,x_max), (y_min,y_max)) 或 None
        """
        self.logger.info(f"\n    === 计算位置坐标范围 ===")
        self.logger.info(f"    输入角度: phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
        self.logger.info(f"    基边顶点: P{p1}, P{p2}")
        self.logger.info(f"    对角顶点: P{opp_idx}")
        self.logger.info(f"    矩形边缘: {rect_edge}")
        
        if rect_edge in ['bottom', 'top']:
            self.logger.info(f"    水平基边情况 (rect_edge={rect_edge})")
            
            # 计算yO
            y_target = 0.0 if rect_edge == 'bottom' else self.n
            self.logger.info(f"    步骤1: 计算中心点y坐标")
            self.logger.info(f"      目标y坐标: y_target = {y_target} ({'底边' if rect_edge == 'bottom' else '顶边'})")
            self.logger.info(f"      公式: yO = y_target - t{p1} * sin(phi + θ{p1})")
            self.logger.info(f"      代入: yO = {y_target} - {self.t[p1]:.6f} * sin({phi:.6f} + {self.theta[p1]:.6f})")
            
            angle_sum = phi + self.theta[p1]
            sin_value = math.sin(angle_sum)
            self.logger.info(f"      计算: yO = {y_target} - {self.t[p1]:.6f} * sin({angle_sum:.6f})")
            self.logger.info(f"      计算: yO = {y_target} - {self.t[p1]:.6f} * {sin_value:.8f}")
            
            term = self.t[p1] * sin_value
            yO = y_target - term
            self.logger.info(f"      计算: yO = {y_target} - {term:.8f}")
            self.logger.info(f"      结果: yO = {yO:.8f}")
            
            # 计算x范围
            self.logger.info(f"    步骤2: 计算x坐标范围")
            self.logger.info(f"      计算P{p1}的x坐标:")
            self.logger.info(f"        公式: x_p1 = t{p1} * cos(phi + θ{p1})")
            self.logger.info(f"        代入: x_p1 = {self.t[p1]:.6f} * cos({phi:.6f} + {self.theta[p1]:.6f})")
            cos_value1 = math.cos(angle_sum)
            self.logger.info(f"        计算: x_p1 = {self.t[p1]:.6f} * cos({angle_sum:.6f})")
            self.logger.info(f"        计算: x_p1 = {self.t[p1]:.6f} * {cos_value1:.8f}")
            x_p1 = self.t[p1] * cos_value1
            self.logger.info(f"        结果: x_p1 = {x_p1:.8f}")
            
            self.logger.info(f"      计算P{p2}的x坐标:")
            angle_sum2 = phi + self.theta[p2]
            self.logger.info(f"        公式: x_p2 = t{p2} * cos(phi + θ{p2})")
            self.logger.info(f"        代入: x_p2 = {self.t[p2]:.6f} * cos({phi:.6f} + {self.theta[p2]:.6f})")
            cos_value2 = math.cos(angle_sum2)
            self.logger.info(f"        计算: x_p2 = {self.t[p2]:.6f} * cos({angle_sum2:.6f})")
            self.logger.info(f"        计算: x_p2 = {self.t[p2]:.6f} * {cos_value2:.8f}")
            x_p2 = self.t[p2] * cos_value2
            self.logger.info(f"        结果: x_p2 = {x_p2:.8f}")
            
            self.logger.info(f"      计算x范围约束:")
            self.logger.info(f"        公式: x_min = max(-x_p1, -x_p2)")
            self.logger.info(f"        代入: x_min = max(-({x_p1:.8f}), -({x_p2:.8f}))")
            self.logger.info(f"        计算: x_min = max({-x_p1:.8f}, {-x_p2:.8f})")
            x_min = max(-x_p1, -x_p2)
            self.logger.info(f"        结果: x_min = {x_min:.8f}")
            
            self.logger.info(f"        公式: x_max = min(m - x_p1, m - x_p2)")
            self.logger.info(f"        代入: x_max = min({self.m} - {x_p1:.8f}, {self.m} - {x_p2:.8f})")
            self.logger.info(f"        计算: x_max = min({self.m - x_p1:.8f}, {self.m - x_p2:.8f})")
            x_max = min(self.m - x_p1, self.m - x_p2)
            self.logger.info(f"        结果: x_max = {x_max:.8f}")
            
            # 验证x范围有效性
            self.logger.info(f"      x范围有效性检查:")
            self.logger.info(f"        判断条件: x_min ≤ x_max + tolerance")
            self.logger.info(f"        比较: {x_min:.8f} ≤ {x_max:.8f} + {self.case_config.range_tolerance}")
            x_max_adjusted = x_max + self.case_config.range_tolerance
            self.logger.info(f"        判断: {x_min:.8f} ≤ {x_max_adjusted:.8f} ? {x_min <= x_max_adjusted}")
            
            if x_min > x_max + self.case_config.range_tolerance:
                self.logger.info(f"        结论: x范围无效 ({x_min:.6f} > {x_max:.6f})")
                return None
            else:
                self.logger.info(f"        结论: x范围有效")
                
            self.logger.info(f"    === 位置计算完成 ===")
            self.logger.info(f"    最终结果: x范围=[{x_min:.8f}, {x_max:.8f}], y坐标={yO:.8f}")
            return ((x_min, x_max), (yO, yO))
            
        else:
            self.logger.info(f"    垂直基边情况 (rect_edge={rect_edge})")
            
            # 计算xO
            x_target = 0.0 if rect_edge == 'left' else self.m
            self.logger.info(f"    步骤1: 计算中心点x坐标")
            self.logger.info(f"      目标x坐标: x_target = {x_target} ({'左边' if rect_edge == 'left' else '右边'})")
            self.logger.info(f"      公式: xO = x_target - t{p1} * cos(phi + θ{p1})")
            self.logger.info(f"      代入: xO = {x_target} - {self.t[p1]:.6f} * cos({phi:.6f} + {self.theta[p1]:.6f})")
            
            angle_sum = phi + self.theta[p1]
            cos_value = math.cos(angle_sum)
            self.logger.info(f"      计算: xO = {x_target} - {self.t[p1]:.6f} * cos({angle_sum:.6f})")
            self.logger.info(f"      计算: xO = {x_target} - {self.t[p1]:.6f} * {cos_value:.8f}")
            
            term = self.t[p1] * cos_value
            xO = x_target - term
            self.logger.info(f"      计算: xO = {x_target} - {term:.8f}")
            self.logger.info(f"      结果: xO = {xO:.8f}")
            
            # 计算y范围
            self.logger.info(f"    步骤2: 计算y坐标范围")
            self.logger.info(f"      计算P{p1}的y坐标:")
            self.logger.info(f"        公式: y_p1 = t{p1} * sin(phi + θ{p1})")
            self.logger.info(f"        代入: y_p1 = {self.t[p1]:.6f} * sin({phi:.6f} + {self.theta[p1]:.6f})")
            sin_value1 = math.sin(angle_sum)
            self.logger.info(f"        计算: y_p1 = {self.t[p1]:.6f} * sin({angle_sum:.6f})")
            self.logger.info(f"        计算: y_p1 = {self.t[p1]:.6f} * {sin_value1:.8f}")
            y_p1 = self.t[p1] * sin_value1
            self.logger.info(f"        结果: y_p1 = {y_p1:.8f}")
            
            self.logger.info(f"      计算P{p2}的y坐标:")
            angle_sum2 = phi + self.theta[p2]
            self.logger.info(f"        公式: y_p2 = t{p2} * sin(phi + θ{p2})")
            self.logger.info(f"        代入: y_p2 = {self.t[p2]:.6f} * sin({phi:.6f} + {self.theta[p2]:.6f})")
            sin_value2 = math.sin(angle_sum2)
            self.logger.info(f"        计算: y_p2 = {self.t[p2]:.6f} * sin({angle_sum2:.6f})")
            self.logger.info(f"        计算: y_p2 = {self.t[p2]:.6f} * {sin_value2:.8f}")
            y_p2 = self.t[p2] * sin_value2
            self.logger.info(f"        结果: y_p2 = {y_p2:.8f}")
            
            self.logger.info(f"      计算y范围约束:")
            self.logger.info(f"        公式: y_min = max(-y_p1, -y_p2)")
            self.logger.info(f"        代入: y_min = max(-({y_p1:.8f}), -({y_p2:.8f}))")
            self.logger.info(f"        计算: y_min = max({-y_p1:.8f}, {-y_p2:.8f})")
            y_min = max(-y_p1, -y_p2)
            self.logger.info(f"        结果: y_min = {y_min:.8f}")
            
            self.logger.info(f"        公式: y_max = min(n - y_p1, n - y_p2)")
            self.logger.info(f"        代入: y_max = min({self.n} - {y_p1:.8f}, {self.n} - {y_p2:.8f})")
            self.logger.info(f"        计算: y_max = min({self.n - y_p1:.8f}, {self.n - y_p2:.8f})")
            y_max = min(self.n - y_p1, self.n - y_p2)
            self.logger.info(f"        结果: y_max = {y_max:.8f}")
            
            # 验证y范围有效性
            self.logger.info(f"      y范围有效性检查:")
            self.logger.info(f"        判断条件: y_min ≤ y_max + tolerance")
            self.logger.info(f"        比较: {y_min:.8f} ≤ {y_max:.8f} + {self.case_config.range_tolerance}")
            y_max_adjusted = y_max + self.case_config.range_tolerance
            self.logger.info(f"        判断: {y_min:.8f} ≤ {y_max_adjusted:.8f} ? {y_min <= y_max_adjusted}")
            
            if y_min > y_max + self.case_config.range_tolerance:
                self.logger.info(f"        结论: y范围无效 ({y_min:.6f} > {y_max:.6f})")
                return None
            else:
                self.logger.info(f"        结论: y范围有效")
                
            self.logger.info(f"    === 位置计算完成 ===")
            self.logger.info(f"    最终结果: x坐标={xO:.8f}, y范围=[{y_min:.8f}, {y_max:.8f}]")
            return ((xO, xO), (y_min, y_max))

    def _verify_solution(self,
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       phi: float,
                       p1: int,
                       p2: int,
                       opp_idx: int,
                       rect_edge: str) -> bool:
        """完整验证解的有效性
        
        Args:
            x_range: x坐标范围
            y_range: y坐标范围
            phi: 当前角度 (rad)
            p1: 基边顶点1索引
            p2: 基边顶点2索引
            opp_idx: 对角顶点索引
            rect_edge: 矩形边缘类型
            
        Returns:
            bool: 是否为有效解
        """
        self.logger.info(f"\n    === 验证解的有效性 ===")
        self.logger.info(f"    输入: x范围=[{x_range[0]:.8f}, {x_range[1]:.8f}], y范围=[{y_range[0]:.8f}, {y_range[1]:.8f}]")
        self.logger.info(f"    角度: phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
        self.logger.info(f"    基边顶点: P{p1}, P{p2}")
        self.logger.info(f"    对角顶点: P{opp_idx}")
        self.logger.info(f"    矩形边缘: {rect_edge}")
        
        # 计算中心点坐标
        xO = sum(x_range) / 2
        yO = sum(y_range) / 2
        self.logger.info(f"    中心点坐标: xO = ({x_range[0]:.8f} + {x_range[1]:.8f})/2 = {xO:.8f}")
        self.logger.info(f"    中心点坐标: yO = ({y_range[0]:.8f} + {y_range[1]:.8f})/2 = {yO:.8f}")
        
        # 1. 验证基边顶点
        self.logger.info(f"    步骤1: 验证基边顶点位置")
        for pi in [p1, p2]:
            self.logger.info(f"      验证P{pi}:")
            self.logger.info(f"        公式: x = xO + t{pi} * cos(phi + θ{pi})")
            self.logger.info(f"        代入: x = {xO:.8f} + {self.t[pi]:.6f} * cos({phi:.6f} + {self.theta[pi]:.6f})")
            
            angle_sum = phi + self.theta[pi]
            cos_val = math.cos(angle_sum)
            self.logger.info(f"        计算: x = {xO:.8f} + {self.t[pi]:.6f} * cos({angle_sum:.6f})")
            self.logger.info(f"        计算: x = {xO:.8f} + {self.t[pi]:.6f} * {cos_val:.8f}")
            
            x_term = self.t[pi] * cos_val
            x = xO + x_term
            self.logger.info(f"        计算: x = {xO:.8f} + {x_term:.8f}")
            self.logger.info(f"        结果: x = {x:.8f}")
            
            self.logger.info(f"        公式: y = yO + t{pi} * sin(phi + θ{pi})")
            self.logger.info(f"        代入: y = {yO:.8f} + {self.t[pi]:.6f} * sin({phi:.6f} + {self.theta[pi]:.6f})")
            
            sin_val = math.sin(angle_sum)
            self.logger.info(f"        计算: y = {yO:.8f} + {self.t[pi]:.6f} * sin({angle_sum:.6f})")
            self.logger.info(f"        计算: y = {yO:.8f} + {self.t[pi]:.6f} * {sin_val:.8f}")
            
            y_term = self.t[pi] * sin_val
            y = yO + y_term
            self.logger.info(f"        计算: y = {yO:.8f} + {y_term:.8f}")
            self.logger.info(f"        结果: y = {y:.8f}")
            
            self.logger.info(f"        P{pi} = ({x:.8f}, {y:.8f})")
            
            # 检查是否在指定边上
            self.logger.info(f"        检查P{pi}是否在{rect_edge}边上:")
            if rect_edge == 'bottom':
                deviation = abs(y)
                self.logger.info(f"          公式: |y - 0| ≤ tolerance")
                self.logger.info(f"          计算: |{y:.8f} - 0| = {deviation:.8f}")
                self.logger.info(f"          判断: {deviation:.8f} ≤ {self.config.tol} ? {deviation <= self.config.tol}")
                valid = deviation <= self.config.tol
            elif rect_edge == 'top':
                deviation = abs(y - self.n)
                self.logger.info(f"          公式: |y - {self.n}| ≤ tolerance")
                self.logger.info(f"          计算: |{y:.8f} - {self.n}| = {deviation:.8f}")
                self.logger.info(f"          判断: {deviation:.8f} ≤ {self.config.tol} ? {deviation <= self.config.tol}")
                valid = deviation <= self.config.tol
            elif rect_edge == 'left':
                deviation = abs(x)
                self.logger.info(f"          公式: |x - 0| ≤ tolerance")
                self.logger.info(f"          计算: |{x:.8f} - 0| = {deviation:.8f}")
                self.logger.info(f"          判断: {deviation:.8f} ≤ {self.config.tol} ? {deviation <= self.config.tol}")
                valid = deviation <= self.config.tol
            else:  # 'right'
                deviation = abs(x - self.m)
                self.logger.info(f"          公式: |x - {self.m}| ≤ tolerance")
                self.logger.info(f"          计算: |{x:.8f} - {self.m}| = {deviation:.8f}")
                self.logger.info(f"          判断: {deviation:.8f} ≤ {self.config.tol} ? {deviation <= self.config.tol}")
                valid = deviation <= self.config.tol
                
            self.logger.info(f"        结论: P{pi} {'在' if valid else '不在'}{rect_edge}边上 ({'✓' if valid else '✗'})")
            if not valid:
                self.logger.info(f"    === ✗ 验证失败：P{pi}不在{rect_edge}边上 ===")
                return False
                
        # 2. 验证对角顶点
        self.logger.info(f"    步骤2: 验证对角顶点P{opp_idx}位置")
        self.logger.info(f"      计算P{opp_idx}坐标:")
        self.logger.info(f"        公式: x_opp = xO + t{opp_idx} * cos(phi + θ{opp_idx})")
        self.logger.info(f"        代入: x_opp = {xO:.8f} + {self.t[opp_idx]:.6f} * cos({phi:.6f} + {self.theta[opp_idx]:.6f})")
        
        angle_sum_opp = phi + self.theta[opp_idx]
        cos_val_opp = math.cos(angle_sum_opp)
        self.logger.info(f"        计算: x_opp = {xO:.8f} + {self.t[opp_idx]:.6f} * cos({angle_sum_opp:.6f})")
        self.logger.info(f"        计算: x_opp = {xO:.8f} + {self.t[opp_idx]:.6f} * {cos_val_opp:.8f}")
        
        x_term_opp = self.t[opp_idx] * cos_val_opp
        x_opp = xO + x_term_opp
        self.logger.info(f"        计算: x_opp = {xO:.8f} + {x_term_opp:.8f}")
        self.logger.info(f"        结果: x_opp = {x_opp:.8f}")
        
        self.logger.info(f"        公式: y_opp = yO + t{opp_idx} * sin(phi + θ{opp_idx})")
        self.logger.info(f"        代入: y_opp = {yO:.8f} + {self.t[opp_idx]:.6f} * sin({phi:.6f} + {self.theta[opp_idx]:.6f})")
        
        sin_val_opp = math.sin(angle_sum_opp)
        self.logger.info(f"        计算: y_opp = {yO:.8f} + {self.t[opp_idx]:.6f} * sin({angle_sum_opp:.6f})")
        self.logger.info(f"        计算: y_opp = {yO:.8f} + {self.t[opp_idx]:.6f} * {sin_val_opp:.8f}")
        
        y_term_opp = self.t[opp_idx] * sin_val_opp
        y_opp = yO + y_term_opp
        self.logger.info(f"        计算: y_opp = {yO:.8f} + {y_term_opp:.8f}")
        self.logger.info(f"        结果: y_opp = {y_opp:.8f}")
        
        self.logger.info(f"      P{opp_idx} = ({x_opp:.8f}, {y_opp:.8f})")
        
        # 检查对角顶点是否在对边上
        self.logger.info(f"      检查P{opp_idx}是否在对边上:")
        if rect_edge in ['bottom', 'top']:
            target = self.n if rect_edge == 'bottom' else 0
            target_edge = 'top' if rect_edge == 'bottom' else 'bottom'
            deviation = abs(y_opp - target)
            self.logger.info(f"        目标边: {target_edge} (y = {target})")
            self.logger.info(f"        公式: |y_opp - {target}| ≤ tolerance")
            self.logger.info(f"        计算: |{y_opp:.8f} - {target}| = {deviation:.8f}")
            self.logger.info(f"        判断: {deviation:.8f} ≤ {self.config.tol} ? {deviation <= self.config.tol}")
            opp_valid = deviation <= self.config.tol
        else:
            target = self.m if rect_edge == 'left' else 0
            target_edge = 'right' if rect_edge == 'left' else 'left'
            deviation = abs(x_opp - target)
            self.logger.info(f"        目标边: {target_edge} (x = {target})")
            self.logger.info(f"        公式: |x_opp - {target}| ≤ tolerance")
            self.logger.info(f"        计算: |{x_opp:.8f} - {target}| = {deviation:.8f}")
            self.logger.info(f"        判断: {deviation:.8f} ≤ {self.config.tol} ? {deviation <= self.config.tol}")
            opp_valid = deviation <= self.config.tol
            
        self.logger.info(f"      结论: P{opp_idx} {'在' if opp_valid else '不在'}{target_edge}边上 ({'✓' if opp_valid else '✗'})")
        
        if not opp_valid:
            self.logger.info(f"      偏差: {deviation:.6f}")
            self.logger.info(f"    === ✗ 验证失败：对角顶点不在目标边上 ===")
            return False
            
        # 3. 验证所有顶点在场地内
        self.logger.info(f"    步骤3: 验证所有顶点在场地边界内")
        all_points = [(x, y), (x_opp, y_opp)]  # 基边两点已验证在边上，只需检查对角点
        
        for i, point_idx in enumerate([p1, opp_idx]):  # 简化：只检查关键点
            px, py = all_points[i] if i == 1 else (x, y)  # 这里简化处理
            self.logger.info(f"      检查P{point_idx}边界约束:")
            
            # x边界检查
            x_lower = 0 - self.config.tol
            x_upper = self.m + self.config.tol
            x_ok = (x_lower <= px <= x_upper)
            self.logger.info(f"        x边界: {x_lower:.8f} ≤ {px:.8f} ≤ {x_upper:.8f} -> {'✓' if x_ok else '✗'}")
            
            # y边界检查
            y_lower = 0 - self.config.tol
            y_upper = self.n + self.config.tol
            y_ok = (y_lower <= py <= y_upper)
            self.logger.info(f"        y边界: {y_lower:.8f} ≤ {py:.8f} ≤ {y_upper:.8f} -> {'✓' if y_ok else '✗'}")
            
            if not (x_ok and y_ok):
                self.logger.info(f"        结论: P{point_idx}超出场地边界")
                self.logger.info(f"    === ✗ 验证失败：顶点超出边界 ===")
                return False
            else:
                self.logger.info(f"        结论: P{point_idx}在场地边界内")
                
        self.logger.info(f"    === ✓ 所有验证通过，解有效 ===")
        return True

def _test_case2():
    """Case2Solver 测试函数"""
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    
    config = BaseSolverConfig(
        log_file="logs/case2_test.log",
        log_level="DEBUG"
    )
    case_config = Case2Config(range_tolerance=1e-5)
    
    solver = Case2Solver(
        t=[1.0, 1.0, math.sqrt(2)],
        theta=[0.0, math.pi/2, 3*math.pi/4],
        m=2.0,
        n=2.0,
        config=config,
        case_config=case_config
    )
    
    solutions = solver.solve()
    print(f"Found {len(solutions)} solutions")
    for sol in solutions:
        print(f"Solution: {sol}")

if __name__ == "__main__":
    _test_case2()