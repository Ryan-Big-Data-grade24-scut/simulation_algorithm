if __name__ == '__main__':
    import sys
    sys.path.append('/home/lrx/laser_positioning_ubuntu/simulation_algorithm/positioning_algorithm')
    from case_solvers.BaseSolver import BaseSolver, BaseSolverConfig
else:
    from .BaseSolver import BaseSolver, BaseSolverConfig
from typing import List, Tuple, Optional
import math
import logging
from dataclasses import dataclass
import os

@dataclass
class Case1Config:
    """Case1 特有配置项
    
    Attributes:
        edge_tolerance (float): 边缘对齐容忍度，默认1e-4
        enable_visual_log (bool): 是否生成可视化日志，默认False
    """
    edge_tolerance: float = 1e-4
    enable_visual_log: bool = False

class Case1Solver(BaseSolver):
    """情况1求解器：一边在矩形边缘且对角顶点在邻边
    
    Args:
        t (List[float]): 3个激光测距值 [t0, t1, t2] (单位: m)
        theta (List[float]): 3个激光角度 [θ0, θ1, θ2] (单位: rad)
        m (float): 场地x方向长度 (m)
        n (float): 场地y方向长度 (m)
        config (Optional[BaseSolverConfig]): 基础配置
        case_config (Optional[Case1Config]): 情况1特有配置
        ros_logger (Optional): ROS2日志器对象
    """

    def __init__(self, 
                 t: List[float], 
                 theta: List[float],
                 m: float, 
                 n: float, 
                 config: Optional[BaseSolverConfig] = None,
                 case_config: Optional[Case1Config] = None,
                 ros_logger=None,
                 min_log_level=logging.DEBUG):
        """
        初始化求解器
        
        Note:
            自动创建log/case1.log日志文件
        """
        # 初始化基础配置
        config = config or BaseSolverConfig(
            log_file=os.path.join('logs', 'case1.log'),
            log_level='DEBUG'
        )
        super().__init__(t, theta, m, n, config, ros_logger, min_log_level=min_log_level)
        
        # 情况1特有配置
        self.case_config = case_config or Case1Config()
        
        # 预定义边组合 (基边, 邻边)
        self.edge_combinations = [
            ('bottom', 'left'), ('bottom', 'right'),
            ('top', 'left'), ('top', 'right'),
            ('left', 'bottom'), ('left', 'top'),
            ('right', 'bottom'), ('right', 'top')
        ]
        
        # 确保日志目录存在
        os.makedirs('logs', exist_ok=True)
        self.logger.info(f"Case1Solver initialized with {self.case_config}")

    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """执行情况1求解流程
        
        Returns:
            List[Tuple]: 有效解列表，每个解格式为:
                ((x_min, x_max), (y_min, y_max), phi)
                
        Raises:
            RuntimeError: 当求解过程中出现不可恢复错误
        """
        solutions = []
        try:
            self.logger.info(f"\n=== Starting Case1Solver ===")
            self.logger.info(f"Parameters: t={self.t}, theta={self.theta}, m={self.m}, n={self.n}")
            self.logger.info(f"Start solving with t={self.t}, theta={self.theta}")
            
            for base_idx in range(3):
                solutions.extend(
                    self._solve_for_base_edge(base_idx)
                )
                
            self.logger.info(f"\n=== Case1 Solution Summary ===")
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
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        
        self.logger.info(f"\n=== 处理基边 {base_idx} (P{p1_idx}-P{p2_idx}) ===")
        self.logger.info(f"基边顶点: P{p1_idx}, P{p2_idx}")
        self.logger.info(f"对角顶点: P{opp_idx}")
        self.logger.info(f"激光数据: t{p1_idx}={self.t[p1_idx]:.6f}, t{p2_idx}={self.t[p2_idx]:.6f}, t{opp_idx}={self.t[opp_idx]:.6f}")
        self.logger.info(f"角度数据: θ{p1_idx}={self.theta[p1_idx]:.6f}, θ{p2_idx}={self.theta[p2_idx]:.6f}, θ{opp_idx}={self.theta[opp_idx]:.6f}")
        
        for rect_edge, adj_edge in self.edge_combinations:
            try:
                self.logger.info(f"\n  --- 尝试边组合: rect_edge={rect_edge}, adj_edge={adj_edge} ---")
                self.logger.info(f"  配置说明: 基边P{p1_idx}-P{p2_idx}在{rect_edge}边，对角顶点P{opp_idx}在{adj_edge}边")
                
                phi_list = self._solve_phi(p1_idx, rect_edge, adj_edge)
                self.logger.info(f"  求解得到{len(phi_list)}个候选角度phi")
                
                for i, phi in enumerate(phi_list):
                    self.logger.info(f"\n    ++ 测试第{i+1}个候选角度 ++")
                    self.logger.info(f"    phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
                    
                    xO, yO = self._compute_position(p1_idx, rect_edge, adj_edge, phi)
                    self.logger.info(f"    计算得到中心坐标: ({xO:.8f}, {yO:.8f})")
                    
                    if self._verify_solution(xO, yO, phi, p1_idx, rect_edge, adj_edge):
                        sol = ((xO, xO), (yO, yO), phi)
                        solutions.append(sol)
                        self.logger.info(f"    *** ✓ 找到有效解: 中心=({xO:.6f}, {yO:.6f}), phi={phi:.6f} ***")
                    else:
                        self.logger.info(f"    ✗ 解验证失败")
                        
            except Exception as e:
                self.logger.warning(f"  边组合处理失败: {str(e)}")
                continue
        
        self.logger.info(f"=== 基边{base_idx}处理完成，共找到{len(solutions)}个解 ===")
        return solutions

    def _solve_phi(self, 
                  base_idx: int, 
                  rect_edge: str, 
                  adj_edge: str) -> List[float]:
        """求解基边对齐时的角度phi
        
        Args:
            base_idx: 基边起始顶点索引
            rect_edge: 矩形边缘 ('bottom'/'top'/'left'/'right')
            adj_edge: 邻接边缘
            
        Returns:
            List[float]: 有效的phi角度列表 (rad)
        """
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        
        self.logger.info(f"\n    === 求解phi角度 ===")
        self.logger.info(f"    基边顶点: P{p1_idx}, P{p2_idx}")
        self.logger.info(f"    基边位置: {rect_edge}边")
        self.logger.info(f"    约束类型: {'水平边对齐' if rect_edge in ['bottom', 'top'] else '垂直边对齐'}")
        
        # 水平边处理
        if rect_edge in ['bottom', 'top']:
            self.logger.info(f"    水平边处理 (rect_edge={rect_edge})")
            self.logger.info(f"    需要满足: P{p1_idx}.y = P{p2_idx}.y (基边水平对齐)")
            
            # 计算A系数
            self.logger.info(f"    步骤1: 计算A系数")
            self.logger.info(f"      公式: A = t{p1_idx}*cos(θ{p1_idx}) - t{p2_idx}*cos(θ{p2_idx})")
            cos_p1 = math.cos(self.theta[p1_idx])
            cos_p2 = math.cos(self.theta[p2_idx])
            self.logger.info(f"      计算: cos({self.theta[p1_idx]:.6f}) = {cos_p1:.8f}")
            self.logger.info(f"      计算: cos({self.theta[p2_idx]:.6f}) = {cos_p2:.8f}")
            self.logger.info(f"      代入: A = {self.t[p1_idx]:.6f}*{cos_p1:.8f} - {self.t[p2_idx]:.6f}*{cos_p2:.8f}")
            term1_A = self.t[p1_idx] * cos_p1
            term2_A = self.t[p2_idx] * cos_p2
            self.logger.info(f"      计算: A = {term1_A:.8f} - {term2_A:.8f}")
            A = term1_A - term2_A
            self.logger.info(f"      结果: A = {A:.8f}")
            
            # 计算B系数
            self.logger.info(f"    步骤2: 计算B系数")
            self.logger.info(f"      公式: B = t{p1_idx}*sin(θ{p1_idx}) - t{p2_idx}*sin(θ{p2_idx})")
            sin_p1 = math.sin(self.theta[p1_idx])
            sin_p2 = math.sin(self.theta[p2_idx])
            self.logger.info(f"      计算: sin({self.theta[p1_idx]:.6f}) = {sin_p1:.8f}")
            self.logger.info(f"      计算: sin({self.theta[p2_idx]:.6f}) = {sin_p2:.8f}")
            self.logger.info(f"      代入: B = {self.t[p1_idx]:.6f}*{sin_p1:.8f} - {self.t[p2_idx]:.6f}*{sin_p2:.8f}")
            term1_B = self.t[p1_idx] * sin_p1
            term2_B = self.t[p2_idx] * sin_p2
            self.logger.info(f"      计算: B = {term1_B:.8f} - {term2_B:.8f}")
            B = term1_B - term2_B
            self.logger.info(f"      结果: B = {B:.8f}")

            # 退化检查
            self.logger.info(f"    步骤3: 退化情况检查")
            self.logger.info(f"      判断条件: |A| < tolerance 且 |B| < tolerance")
            self.logger.info(f"      计算: |A| = |{A:.8f}| = {abs(A):.8f}")
            self.logger.info(f"      计算: |B| = |{B:.8f}| = {abs(B):.8f}")
            self.logger.info(f"      比较: {abs(A):.8f} < {self.config.tol} ? {abs(A) < self.config.tol}")
            self.logger.info(f"      比较: {abs(B):.8f} < {self.config.tol} ? {abs(B) < self.config.tol}")
            
            if abs(A) < self.config.tol and abs(B) < self.config.tol:
                self.logger.info("      结论: 退化情况 (A≈0 且 B≈0)，基边已经平行，无需旋转")
                return []
            else:
                self.logger.info("      结论: 非退化情况，继续求解")
                
            candidates = self._generate_phi_candidates(A, B, is_vertical=False)
            
        # 垂直边处理
        else:
            self.logger.info(f"    垂直边处理 (rect_edge={rect_edge})")
            self.logger.info(f"    需要满足: P{p1_idx}.x = P{p2_idx}.x (基边垂直对齐)")
            
            # 计算A系数
            self.logger.info(f"    步骤1: 计算A系数")
            self.logger.info(f"      公式: A = t{p2_idx}*sin(θ{p2_idx}) - t{p1_idx}*sin(θ{p1_idx})")
            sin_p1 = math.sin(self.theta[p1_idx])
            sin_p2 = math.sin(self.theta[p2_idx])
            self.logger.info(f"      计算: sin({self.theta[p1_idx]:.6f}) = {sin_p1:.8f}")
            self.logger.info(f"      计算: sin({self.theta[p2_idx]:.6f}) = {sin_p2:.8f}")
            self.logger.info(f"      代入: A = {self.t[p2_idx]:.6f}*{sin_p2:.8f} - {self.t[p1_idx]:.6f}*{sin_p1:.8f}")
            term1_A = self.t[p2_idx] * sin_p2
            term2_A = self.t[p1_idx] * sin_p1
            self.logger.info(f"      计算: A = {term1_A:.8f} - {term2_A:.8f}")
            A = term1_A - term2_A
            self.logger.info(f"      结果: A = {A:.8f}")
            
            # 计算B系数
            self.logger.info(f"    步骤2: 计算B系数")
            self.logger.info(f"      公式: B = t{p1_idx}*cos(θ{p1_idx}) - t{p2_idx}*cos(θ{p2_idx})")
            cos_p1 = math.cos(self.theta[p1_idx])
            cos_p2 = math.cos(self.theta[p2_idx])
            self.logger.info(f"      计算: cos({self.theta[p1_idx]:.6f}) = {cos_p1:.8f}")
            self.logger.info(f"      计算: cos({self.theta[p2_idx]:.6f}) = {cos_p2:.8f}")
            self.logger.info(f"      代入: B = {self.t[p1_idx]:.6f}*{cos_p1:.8f} - {self.t[p2_idx]:.6f}*{cos_p2:.8f}")
            term1_B = self.t[p1_idx] * cos_p1
            term2_B = self.t[p2_idx] * cos_p2
            self.logger.info(f"      计算: B = {term1_B:.8f} - {term2_B:.8f}")
            B = term1_B - term2_B
            self.logger.info(f"      结果: B = {B:.8f}")

            candidates = self._generate_phi_candidates(A, B, is_vertical=True)
            
        self.logger.info(f"    === phi求解完成，得到{len(candidates)}个候选角度 ===")
        for i, phi in enumerate(candidates):
            self.logger.info(f"      候选{i+1}: phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
            
        return self._validate_phi_candidates(
            candidates, base_idx, rect_edge, adj_edge, 
            is_vertical=(rect_edge in ['left', 'right'])
        )

    def _generate_phi_candidates(self, 
                               A: float, 
                               B: float,
                               is_vertical: bool) -> List[float]:
        """生成phi候选角度
        
        Args:
            A: 方程系数A
            B: 方程系数B
            is_vertical: 是否垂直边
            
        Returns:
            List[float]: 候选角度 [rad]
        """
        self.logger.debug(f"    生成phi候选角度，A={A:.6f}, B={B:.6f}, is_vertical={is_vertical}")
        
        if abs(A) < self.config.tol:
            self.logger.debug(f"    特殊情况判断: |A| = {abs(A):.8f} < {self.config.tol}")
            self.logger.debug("    结论: A ≈ 0，使用特殊公式")
            
            if not is_vertical:
                # 水平边情况
                self.logger.debug("    公式: phi = ±π/2 (取决于B的符号)")
                base_phi = math.copysign(math.pi/2, -B)
                self.logger.debug(f"    计算: phi = copysign(π/2, -{B:.6f}) = copysign({math.pi/2:.6f}, {-B:.6f})")
                self.logger.debug(f"    结果: base_phi = {base_phi:.6f}")
            else:
                # 垂直边情况
                self.logger.debug("    公式: phi = 0 (如果B > 0) 或 phi = π (如果B ≤ 0)")
                base_phi = 0.0 if B > 0 else math.pi
                self.logger.debug(f"    判断: B = {B:.6f} {'> 0' if B > 0 else '≤ 0'}")
                self.logger.debug(f"    结果: base_phi = {base_phi:.6f}")
        else:
            self.logger.debug(f"    一般情况: |A| = {abs(A):.8f} ≥ {self.config.tol}")
            
            if not is_vertical:
                # 水平边情况
                self.logger.debug("    公式: phi = atan2(-B, A)")
                base_phi = math.atan2(-B, A)
                self.logger.debug(f"    代入: phi = atan2(-({B:.6f}), {A:.6f}) = atan2({-B:.6f}, {A:.6f})")
                self.logger.debug(f"    结果: base_phi = {base_phi:.6f} rad = {math.degrees(base_phi):.2f}°")
            else:
                # 垂直边情况
                self.logger.debug("    公式: phi = atan2(B, -A)")
                base_phi = math.atan2(B, -A)
                self.logger.debug(f"    代入: phi = atan2({B:.6f}, -({A:.6f})) = atan2({B:.6f}, {-A:.6f})")
                self.logger.debug(f"    结果: base_phi = {base_phi:.6f} rad = {math.degrees(base_phi):.2f}°")
            
        candidates = [base_phi, base_phi + math.pi]
        self.logger.debug(f"    生成候选角度:")
        self.logger.debug(f"      candidate1 = {candidates[0]:.6f} rad = {math.degrees(candidates[0]):.2f}°")
        self.logger.debug(f"      candidate2 = {candidates[1]:.6f} rad = {math.degrees(candidates[1]):.2f}°")
        
        return candidates

    def _validate_phi_candidates(self, 
                               candidates: List[float],
                               base_idx: int,
                               rect_edge: str,
                               adj_edge: str,
                               is_vertical: bool) -> List[float]:
        """验证phi候选角度的有效性
        
        Args:
            candidates: 候选角度列表
            base_idx: 基边顶点索引
            rect_edge: 矩形边缘类型
            adj_edge: 邻接边缘类型
            is_vertical: 是否垂直边
            
        Returns:
            List[float]: 通过验证的有效角度
        """
        valid_phis = []
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3

        for phi in candidates:
            try:
                self.logger.debug(f"\n  === 验证候选角度 phi = {phi:.8f} rad = {math.degrees(phi):.2f}° ===")
                
                # 计算相对坐标
                self.logger.debug("  步骤1: 计算三个顶点的相对坐标")
                P = {}
                for i in [p1_idx, p2_idx, opp_idx]:
                    self.logger.debug(f"    顶点P{i}计算:")
                    self.logger.debug(f"      公式: x = t{i} * cos(phi + θ{i})")
                    self.logger.debug(f"      代入: x = {self.t[i]:.6f} * cos({phi:.6f} + {self.theta[i]:.6f})")
                    self.logger.debug(f"      计算: x = {self.t[i]:.6f} * cos({phi + self.theta[i]:.6f})")
                    x = self.t[i] * math.cos(phi + self.theta[i])
                    self.logger.debug(f"      结果: x = {x:.8f}")
                    
                    self.logger.debug(f"      公式: y = t{i} * sin(phi + θ{i})")
                    self.logger.debug(f"      代入: y = {self.t[i]:.6f} * sin({phi:.6f} + {self.theta[i]:.6f})")
                    self.logger.debug(f"      计算: y = {self.t[i]:.6f} * sin({phi + self.theta[i]:.6f})")
                    y = self.t[i] * math.sin(phi + self.theta[i])
                    self.logger.debug(f"      结果: y = {y:.8f}")
                    
                    P[i] = (x, y)
                    self.logger.debug(f"    P{i} = ({x:.8f}, {y:.8f})")
                
                # 1. 基边对齐检查
                self.logger.debug(f"\n  步骤2: 基边对齐检查 (is_vertical={is_vertical})")
                if is_vertical:
                    dx = abs(P[p2_idx][0] - P[p1_idx][0])
                    self.logger.debug(f"    公式: dx = |P{p2_idx}.x - P{p1_idx}.x|")
                    self.logger.debug(f"    代入: dx = |{P[p2_idx][0]:.8f} - {P[p1_idx][0]:.8f}|")
                    self.logger.debug(f"    结果: dx = {dx:.8f}")
                    self.logger.debug(f"    判断: dx = {dx:.8f} {'<' if dx < self.config.tol else '≥'} {self.config.tol}")
                    
                    if dx >= self.config.tol:
                        self.logger.debug("    结论: 基边不垂直，跳过此候选角度")
                        continue
                    else:
                        self.logger.debug("    结论: 基边垂直对齐✓")
                else:
                    dy = abs(P[p2_idx][1] - P[p1_idx][1])
                    self.logger.debug(f"    公式: dy = |P{p2_idx}.y - P{p1_idx}.y|")
                    self.logger.debug(f"    代入: dy = |{P[p2_idx][1]:.8f} - {P[p1_idx][1]:.8f}|")
                    self.logger.debug(f"    结果: dy = {dy:.8f}")
                    self.logger.debug(f"    判断: dy = {dy:.8f} {'<' if dy < self.config.tol else '≥'} {self.config.tol}")
                    
                    if dy >= self.config.tol:
                        self.logger.debug("    结论: 基边不水平，跳过此候选角度")
                        continue
                    else:
                        self.logger.debug("    结论: 基边水平对齐✓")
                
                # 2. 邻边约束检查
                self.logger.debug(f"\n  步骤3: 邻边约束检查 (adj_edge={adj_edge})")
                if adj_edge == 'left':
                    min_base_x = min(P[p1_idx][0], P[p2_idx][0])
                    self.logger.debug(f"    公式: opp_x ≤ min(P{p1_idx}.x, P{p2_idx}.x) + tolerance")
                    self.logger.debug(f"    代入: {P[opp_idx][0]:.8f} ≤ min({P[p1_idx][0]:.8f}, {P[p2_idx][0]:.8f}) + {self.case_config.edge_tolerance}")
                    self.logger.debug(f"    计算: {P[opp_idx][0]:.8f} ≤ {min_base_x:.8f} + {self.case_config.edge_tolerance}")
                    threshold = min_base_x + self.case_config.edge_tolerance
                    self.logger.debug(f"    判断: {P[opp_idx][0]:.8f} {'≤' if P[opp_idx][0] <= threshold else '>'} {threshold:.8f}")
                    cond = P[opp_idx][0] <= threshold
                elif adj_edge == 'right':
                    max_base_x = max(P[p1_idx][0], P[p2_idx][0])
                    self.logger.debug(f"    公式: opp_x ≥ max(P{p1_idx}.x, P{p2_idx}.x) - tolerance")
                    self.logger.debug(f"    代入: {P[opp_idx][0]:.8f} ≥ max({P[p1_idx][0]:.8f}, {P[p2_idx][0]:.8f}) - {self.case_config.edge_tolerance}")
                    self.logger.debug(f"    计算: {P[opp_idx][0]:.8f} ≥ {max_base_x:.8f} - {self.case_config.edge_tolerance}")
                    threshold = max_base_x - self.case_config.edge_tolerance
                    self.logger.debug(f"    判断: {P[opp_idx][0]:.8f} {'≥' if P[opp_idx][0] >= threshold else '<'} {threshold:.8f}")
                    cond = P[opp_idx][0] >= threshold
                elif adj_edge == 'bottom':
                    min_base_y = min(P[p1_idx][1], P[p2_idx][1])
                    self.logger.debug(f"    公式: opp_y ≤ min(P{p1_idx}.y, P{p2_idx}.y) + tolerance")
                    self.logger.debug(f"    代入: {P[opp_idx][1]:.8f} ≤ min({P[p1_idx][1]:.8f}, {P[p2_idx][1]:.8f}) + {self.case_config.edge_tolerance}")
                    self.logger.debug(f"    计算: {P[opp_idx][1]:.8f} ≤ {min_base_y:.8f} + {self.case_config.edge_tolerance}")
                    threshold = min_base_y + self.case_config.edge_tolerance
                    self.logger.debug(f"    判断: {P[opp_idx][1]:.8f} {'≤' if P[opp_idx][1] <= threshold else '>'} {threshold:.8f}")
                    cond = P[opp_idx][1] <= threshold
                else:  # 'top'
                    max_base_y = max(P[p1_idx][1], P[p2_idx][1])
                    self.logger.debug(f"    公式: opp_y ≥ max(P{p1_idx}.y, P{p2_idx}.y) - tolerance")
                    self.logger.debug(f"    代入: {P[opp_idx][1]:.8f} ≥ max({P[p1_idx][1]:.8f}, {P[p2_idx][1]:.8f}) - {self.case_config.edge_tolerance}")
                    self.logger.debug(f"    计算: {P[opp_idx][1]:.8f} ≥ {max_base_y:.8f} - {self.case_config.edge_tolerance}")
                    threshold = max_base_y - self.case_config.edge_tolerance
                    self.logger.debug(f"    判断: {P[opp_idx][1]:.8f} {'≥' if P[opp_idx][1] >= threshold else '<'} {threshold:.8f}")
                    cond = P[opp_idx][1] >= threshold
                
                self.logger.debug(f"    结论: 邻边约束 {'满足✓' if cond else '不满足✗'}")
                if not cond:
                    continue
                
                # 3. 尺寸检查
                self.logger.debug(f"\n  步骤4: 场地尺寸约束检查")
                all_points = list(P.values())
                min_x = min(p[0] for p in all_points)
                max_x = max(p[0] for p in all_points)
                min_y = min(p[1] for p in all_points)
                max_y = max(p[1] for p in all_points)
                
                width = max_x - min_x
                height = max_y - min_y
                
                self.logger.debug(f"    x坐标范围: [{min_x:.8f}, {max_x:.8f}]")
                self.logger.debug(f"    y坐标范围: [{min_y:.8f}, {max_y:.8f}]")
                self.logger.debug(f"    公式: width = max_x - min_x")
                self.logger.debug(f"    计算: width = {max_x:.8f} - {min_x:.8f} = {width:.8f}")
                self.logger.debug(f"    公式: height = max_y - min_y")
                self.logger.debug(f"    计算: height = {max_y:.8f} - {min_y:.8f} = {height:.8f}")
                
                width_limit = self.m + self.config.tol
                height_limit = self.n + self.config.tol
                width_ok = width <= width_limit
                height_ok = height <= height_limit
                
                self.logger.debug(f"    宽度约束: {width:.8f} {'≤' if width_ok else '>'} {width_limit:.8f} ({'满足✓' if width_ok else '不满足✗'})")
                self.logger.debug(f"    高度约束: {height:.8f} {'≤' if height_ok else '>'} {height_limit:.8f} ({'满足✓' if height_ok else '不满足✗'})")
                
                if width_ok and height_ok:
                    self.logger.debug("  === ✓ 所有验证通过，添加到有效角度列表 ===")
                    valid_phis.append(phi)
                else:
                    self.logger.debug("  === ✗ 尺寸约束失败，跳过此候选角度 ===")
                    
            except Exception as e:
                self.logger.warning(f"Phi validation failed: {str(e)}")
                continue
                
        return valid_phis

    def _check_adjacent_constraint(self,
                                 opp_point: Tuple[float, float],
                                 p1: Tuple[float, float],
                                 p2: Tuple[float, float],
                                 adj_edge: str) -> bool:
        """检查邻边约束
        
        Args:
            opp_point: 对角顶点坐标
            p1: 顶点1坐标
            p2: 顶点2坐标
            adj_edge: 邻接边缘类型
            
        Returns:
            bool: 是否满足约束
        """
        if adj_edge == 'left':
            cond = opp_point[0] <= min(p1[0], p2[0]) + self.case_config.edge_tolerance
        elif adj_edge == 'right':
            cond = opp_point[0] >= max(p1[0], p2[0]) - self.case_config.edge_tolerance
        elif adj_edge == 'bottom':
            cond = opp_point[1] <= min(p1[1], p2[1]) + self.case_config.edge_tolerance
        else:  # 'top'
            cond = opp_point[1] >= max(p1[1], p2[1]) - self.case_config.edge_tolerance
            
        self._log_validation(f"Adjacent edge {adj_edge}", cond)
        return cond

    def _check_dimension_constraints(self, points) -> bool:
        """检查尺寸约束
        
        Args:
            points: 所有顶点坐标迭代器
            
        Returns:
            bool: 是否满足场地尺寸
        """
        coords = list(zip(*points))
        width_ok = max(coords[0]) - min(coords[0]) <= self.m + self.config.tol
        height_ok = max(coords[1]) - min(coords[1]) <= self.n + self.config.tol
        
        self._log_validation("Width constraint", width_ok)
        self._log_validation("Height constraint", height_ok)
        return width_ok and height_ok

    def _compute_position(self,
                         base_idx: int,
                         rect_edge: str,
                         adj_edge: str,
                         phi: float) -> Tuple[float, float]:
        """计算中心点坐标
        
        Args:
            base_idx: 基边顶点索引
            rect_edge: 矩形边缘类型
            adj_edge: 邻接边缘类型
            phi: 当前角度 (rad)
            
        Returns:
            Tuple[float, float]: (xO, yO) 中心坐标
        """
        self.logger.debug(f"\n=== 计算中心点坐标 ===")
        self.logger.debug(f"输入参数: base_idx={base_idx}, rect_edge={rect_edge}, adj_edge={adj_edge}")
        self.logger.debug(f"角度: phi = {phi:.8f} rad = {math.degrees(phi):.2f}°")
        
        p1_idx = base_idx
        opp_idx = (base_idx + 2) % 3
        self.logger.debug(f"基边顶点: P{p1_idx}, 对角顶点: P{opp_idx}")
        
        if rect_edge in ['bottom', 'top']:
            self.logger.debug(f"\n水平基边情况 (rect_edge={rect_edge}):")
            
            # 计算y坐标
            y_target = 0.0 if rect_edge == 'bottom' else self.n
            self.logger.debug(f"  步骤1: 计算中心点y坐标")
            self.logger.debug(f"    目标y坐标: y_target = {y_target}")
            self.logger.debug(f"    公式: yO = y_target - t{p1_idx} * sin(phi + θ{p1_idx})")
            self.logger.debug(f"    代入: yO = {y_target} - {self.t[p1_idx]:.6f} * sin({phi:.6f} + {self.theta[p1_idx]:.6f})")
            
            angle_sum = phi + self.theta[p1_idx]
            sin_value = math.sin(angle_sum)
            self.logger.debug(f"    计算: yO = {y_target} - {self.t[p1_idx]:.6f} * sin({angle_sum:.6f})")
            self.logger.debug(f"    计算: yO = {y_target} - {self.t[p1_idx]:.6f} * {sin_value:.8f}")
            
            term = self.t[p1_idx] * sin_value
            yO = y_target - term
            self.logger.debug(f"    计算: yO = {y_target} - {term:.8f}")
            self.logger.debug(f"    结果: yO = {yO:.8f}")
            
            # 计算x坐标
            x_target = 0.0 if adj_edge == 'left' else self.m
            self.logger.debug(f"  步骤2: 计算中心点x坐标")
            self.logger.debug(f"    目标x坐标: x_target = {x_target} (adj_edge={adj_edge})")
            self.logger.debug(f"    公式: x_opp = t{opp_idx} * cos(phi + θ{opp_idx})")
            self.logger.debug(f"    代入: x_opp = {self.t[opp_idx]:.6f} * cos({phi:.6f} + {self.theta[opp_idx]:.6f})")
            
            angle_sum_opp = phi + self.theta[opp_idx]
            cos_value_opp = math.cos(angle_sum_opp)
            self.logger.debug(f"    计算: x_opp = {self.t[opp_idx]:.6f} * cos({angle_sum_opp:.6f})")
            self.logger.debug(f"    计算: x_opp = {self.t[opp_idx]:.6f} * {cos_value_opp:.8f}")
            
            x_opp = self.t[opp_idx] * cos_value_opp
            self.logger.debug(f"    结果: x_opp = {x_opp:.8f}")
            
            self.logger.debug(f"    公式: xO = x_target - x_opp")
            self.logger.debug(f"    代入: xO = {x_target} - {x_opp:.8f}")
            xO = x_target - x_opp
            self.logger.debug(f"    结果: xO = {xO:.8f}")
            
        else:
            self.logger.debug(f"\n垂直基边情况 (rect_edge={rect_edge}):")
            
            # 计算x坐标
            x_target = 0.0 if rect_edge == 'left' else self.m
            self.logger.debug(f"  步骤1: 计算中心点x坐标")
            self.logger.debug(f"    目标x坐标: x_target = {x_target}")
            self.logger.debug(f"    公式: xO = x_target - t{p1_idx} * cos(phi + θ{p1_idx})")
            self.logger.debug(f"    代入: xO = {x_target} - {self.t[p1_idx]:.6f} * cos({phi:.6f} + {self.theta[p1_idx]:.6f})")
            
            angle_sum = phi + self.theta[p1_idx]
            cos_value = math.cos(angle_sum)
            self.logger.debug(f"    计算: xO = {x_target} - {self.t[p1_idx]:.6f} * cos({angle_sum:.6f})")
            self.logger.debug(f"    计算: xO = {x_target} - {self.t[p1_idx]:.6f} * {cos_value:.8f}")
            
            term = self.t[p1_idx] * cos_value
            xO = x_target - term
            self.logger.debug(f"    计算: xO = {x_target} - {term:.8f}")
            self.logger.debug(f"    结果: xO = {xO:.8f}")
            
            # 计算y坐标
            y_target = 0.0 if adj_edge == 'bottom' else self.n
            self.logger.debug(f"  步骤2: 计算中心点y坐标")
            self.logger.debug(f"    目标y坐标: y_target = {y_target} (adj_edge={adj_edge})")
            self.logger.debug(f"    公式: y_opp = t{opp_idx} * sin(phi + θ{opp_idx})")
            self.logger.debug(f"    代入: y_opp = {self.t[opp_idx]:.6f} * sin({phi:.6f} + {self.theta[opp_idx]:.6f})")
            
            angle_sum_opp = phi + self.theta[opp_idx]
            sin_value_opp = math.sin(angle_sum_opp)
            self.logger.debug(f"    计算: y_opp = {self.t[opp_idx]:.6f} * sin({angle_sum_opp:.6f})")
            self.logger.debug(f"    计算: y_opp = {self.t[opp_idx]:.6f} * {sin_value_opp:.8f}")
            
            y_opp = self.t[opp_idx] * sin_value_opp
            self.logger.debug(f"    结果: y_opp = {y_opp:.8f}")
            
            self.logger.debug(f"    公式: yO = y_target - y_opp")
            self.logger.debug(f"    代入: yO = {y_target} - {y_opp:.8f}")
            yO = y_target - y_opp
            self.logger.debug(f"    结果: yO = {yO:.8f}")
            
        self.logger.debug(f"\n最终结果: 中心点坐标 = ({xO:.8f}, {yO:.8f})")
        self.logger.debug(f"=== 中心点计算完成 ===")
        return xO, yO

    def _verify_solution(self,
                       xO: float,
                       yO: float,
                       phi: float,
                       base_idx: int,
                       rect_edge: str,
                       adj_edge: str) -> bool:
        """验证解的有效性
        
        Args:
            xO: 中心点x坐标
            yO: 中心点y坐标
            phi: 当前角度 (rad)
            base_idx: 基边顶点索引
            rect_edge: 矩形边缘类型
            adj_edge: 邻接边缘类型
            
        Returns:
            bool: 是否为有效解
        """
        self.logger.debug(f"\n=== 验证解的有效性 ===")
        self.logger.debug(f"输入: 中心点=({xO:.8f}, {yO:.8f}), phi={phi:.8f} rad={math.degrees(phi):.2f}°")
        self.logger.debug(f"配置: base_idx={base_idx}, rect_edge={rect_edge}, adj_edge={adj_edge}")
        
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        opp_idx = (base_idx + 2) % 3
        self.logger.debug(f"顶点索引: 基边={p1_idx},{p2_idx}, 对角={opp_idx}")
        
        # 步骤1: 计算所有顶点位置
        self.logger.debug(f"\n步骤1: 计算所有顶点的绝对位置")
        P = {}
        for i in [p1_idx, p2_idx, opp_idx]:
            self.logger.debug(f"  计算P{i}:")
            self.logger.debug(f"    公式: x = xO + t{i} * cos(phi + θ{i})")
            self.logger.debug(f"    代入: x = {xO:.8f} + {self.t[i]:.6f} * cos({phi:.6f} + {self.theta[i]:.6f})")
            
            angle_sum = phi + self.theta[i]
            cos_val = math.cos(angle_sum)
            self.logger.debug(f"    计算: x = {xO:.8f} + {self.t[i]:.6f} * cos({angle_sum:.6f})")
            self.logger.debug(f"    计算: x = {xO:.8f} + {self.t[i]:.6f} * {cos_val:.8f}")
            
            x_term = self.t[i] * cos_val
            x = xO + x_term
            self.logger.debug(f"    计算: x = {xO:.8f} + {x_term:.8f}")
            self.logger.debug(f"    结果: x = {x:.8f}")
            
            self.logger.debug(f"    公式: y = yO + t{i} * sin(phi + θ{i})")
            self.logger.debug(f"    代入: y = {yO:.8f} + {self.t[i]:.6f} * sin({phi:.6f} + {self.theta[i]:.6f})")
            
            sin_val = math.sin(angle_sum)
            self.logger.debug(f"    计算: y = {yO:.8f} + {self.t[i]:.6f} * sin({angle_sum:.6f})")
            self.logger.debug(f"    计算: y = {yO:.8f} + {self.t[i]:.6f} * {sin_val:.8f}")
            
            y_term = self.t[i] * sin_val
            y = yO + y_term
            self.logger.debug(f"    计算: y = {yO:.8f} + {y_term:.8f}")
            self.logger.debug(f"    结果: y = {y:.8f}")
            
            P[i] = (x, y)
            self.logger.debug(f"  P{i} = ({x:.8f}, {y:.8f})")
        
        # 步骤2: 顶点边界检查
        self.logger.debug(f"\n步骤2: 顶点边界检查")
        bounds_ok = True
        for i, (x, y) in P.items():
            self.logger.debug(f"  P{i} = ({x:.8f}, {y:.8f}):")
            
            # x边界检查
            x_min = -self.config.tol
            x_max = self.m + self.config.tol
            x_ok = (x_min <= x <= x_max)
            self.logger.debug(f"    x边界: {x_min:.8f} ≤ {x:.8f} ≤ {x_max:.8f} -> {'✓' if x_ok else '✗'}")
            
            # y边界检查
            y_min = -self.config.tol
            y_max = self.n + self.config.tol
            y_ok = (y_min <= y <= y_max)
            self.logger.debug(f"    y边界: {y_min:.8f} ≤ {y:.8f} ≤ {y_max:.8f} -> {'✓' if y_ok else '✗'}")
            
            bounds_ok = bounds_ok and x_ok and y_ok
            self.logger.debug(f"    P{i}边界检查: {'通过✓' if (x_ok and y_ok) else '失败✗'}")
        
        self.logger.debug(f"  总体边界检查: {'通过✓' if bounds_ok else '失败✗'}")
        
        if not bounds_ok:
            self.logger.debug("=== 验证失败：顶点超出边界 ===")
            return False
        
        # 步骤3: 基边严格对齐检查
        self.logger.debug(f"\n步骤3: 基边严格对齐检查 (rect_edge={rect_edge})")
        
        if rect_edge == 'bottom':
            self.logger.debug("  检查基边是否在底边上:")
            dev1 = abs(P[p1_idx][1])
            dev2 = abs(P[p2_idx][1])
            self.logger.debug(f"    公式: |P{p1_idx}.y - 0| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p1_idx][1]:.8f} - 0| = {dev1:.8f}")
            self.logger.debug(f"    判断: {dev1:.8f} {'≤' if dev1 <= self.config.tol else '>'} {self.config.tol}")
            
            self.logger.debug(f"    公式: |P{p2_idx}.y - 0| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p2_idx][1]:.8f} - 0| = {dev2:.8f}")
            self.logger.debug(f"    判断: {dev2:.8f} {'≤' if dev2 <= self.config.tol else '>'} {self.config.tol}")
            
            edge_ok = (dev1 <= self.config.tol and dev2 <= self.config.tol)
            
        elif rect_edge == 'top':
            self.logger.debug("  检查基边是否在顶边上:")
            dev1 = abs(P[p1_idx][1] - self.n)
            dev2 = abs(P[p2_idx][1] - self.n)
            self.logger.debug(f"    公式: |P{p1_idx}.y - {self.n}| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p1_idx][1]:.8f} - {self.n}| = {dev1:.8f}")
            self.logger.debug(f"    判断: {dev1:.8f} {'≤' if dev1 <= self.config.tol else '>'} {self.config.tol}")
            
            self.logger.debug(f"    公式: |P{p2_idx}.y - {self.n}| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p2_idx][1]:.8f} - {self.n}| = {dev2:.8f}")
            self.logger.debug(f"    判断: {dev2:.8f} {'≤' if dev2 <= self.config.tol else '>'} {self.config.tol}")
            
            edge_ok = (dev1 <= self.config.tol and dev2 <= self.config.tol)
            
        elif rect_edge == 'left':
            self.logger.debug("  检查基边是否在左边上:")
            dev1 = abs(P[p1_idx][0])
            dev2 = abs(P[p2_idx][0])
            self.logger.debug(f"    公式: |P{p1_idx}.x - 0| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p1_idx][0]:.8f} - 0| = {dev1:.8f}")
            self.logger.debug(f"    判断: {dev1:.8f} {'≤' if dev1 <= self.config.tol else '>'} {self.config.tol}")
            
            self.logger.debug(f"    公式: |P{p2_idx}.x - 0| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p2_idx][0]:.8f} - 0| = {dev2:.8f}")
            self.logger.debug(f"    判断: {dev2:.8f} {'≤' if dev2 <= self.config.tol else '>'} {self.config.tol}")
            
            edge_ok = (dev1 <= self.config.tol and dev2 <= self.config.tol)
            
        elif rect_edge == 'right':
            self.logger.debug("  检查基边是否在右边上:")
            dev1 = abs(P[p1_idx][0] - self.m)
            dev2 = abs(P[p2_idx][0] - self.m)
            self.logger.debug(f"    公式: |P{p1_idx}.x - {self.m}| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p1_idx][0]:.8f} - {self.m}| = {dev1:.8f}")
            self.logger.debug(f"    判断: {dev1:.8f} {'≤' if dev1 <= self.config.tol else '>'} {self.config.tol}")
            
            self.logger.debug(f"    公式: |P{p2_idx}.x - {self.m}| ≤ tolerance")
            self.logger.debug(f"    计算: |{P[p2_idx][0]:.8f} - {self.m}| = {dev2:.8f}")
            self.logger.debug(f"    判断: {dev2:.8f} {'≤' if dev2 <= self.config.tol else '>'} {self.config.tol}")
            
            edge_ok = (dev1 <= self.config.tol and dev2 <= self.config.tol)
        
        self.logger.debug(f"  基边对齐检查: {'通过✓' if edge_ok else '失败✗'}")
        
        if edge_ok:
            self.logger.debug("=== ✓ 解验证成功：所有条件满足 ===")
        else:
            self.logger.debug("=== ✗ 解验证失败：基边对齐不满足 ===")
        
        return edge_ok

def _test_case1():
    """Case1Solver 测试函数"""
    from pathlib import Path
    Path("logs").mkdir(exist_ok=True)
    
    config = BaseSolverConfig(
        log_file="logs/case1_test.log",
        log_level="DEBUG"
    )
    case_config = Case1Config(edge_tolerance=1e-5)
    
    solver = Case1Solver(
        t=[1.0, 1.0, math.sqrt(2)],
        theta=[0.0, math.pi/2, math.pi/4],
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
    _test_case1()