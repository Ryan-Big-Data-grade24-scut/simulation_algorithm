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
            self.logger.info(f"Start solving with t={self.t}, theta={self.theta}")
            
            for base_idx in range(3):
                solutions.extend(
                    self._solve_for_base_edge(base_idx)
                )
                
            self.logger.info(f"Found {len(solutions)} valid solutions")
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
        
        self.logger.debug(f"\nProcessing base edge {base_idx} (P{p1_idx}-P{p2_idx})")
        
        for rect_edge, adj_edge in self.edge_combinations:
            try:
                self.logger.debug(f"  Trying {rect_edge}-{adj_edge} combination")
                
                for phi in self._solve_phi(p1_idx, rect_edge, adj_edge):
                    xO, yO = self._compute_position(p1_idx, rect_edge, adj_edge, phi)
                    
                    if self._verify_solution(xO, yO, phi, p1_idx, rect_edge, adj_edge):
                        sol = ((xO, xO), (yO, yO), phi)
                        solutions.append(sol)
                        self.logger.info(f"Found valid solution: {sol}")
                        
            except Exception as e:
                self.logger.warning(f"  Edge combo failed: {str(e)}")
                continue
                
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
        
        # 水平边处理
        if rect_edge in ['bottom', 'top']:
            A = self.t[p1_idx]*math.cos(self.theta[p1_idx]) - self.t[p2_idx]*math.cos(self.theta[p2_idx])
            B = self.t[p1_idx]*math.sin(self.theta[p1_idx]) - self.t[p2_idx]*math.sin(self.theta[p2_idx])
            self._log_math(f"A = t{p1_idx}*cosθ{p1_idx} - t{p2_idx}*cosθ{p2_idx}", A)
            self._log_math(f"B = t{p1_idx}*sinθ{p1_idx} - t{p2_idx}*sinθ{p2_idx}", B)

            if abs(A) < self.config.tol and abs(B) < self.config.tol:
                self.logger.debug("Degenerate case (A=B=0)")
                return []
                
            candidates = self._generate_phi_candidates(A, B, is_vertical=False)
            
        # 垂直边处理
        else:
            A = self.t[p2_idx]*math.sin(self.theta[p2_idx]) - self.t[p1_idx]*math.sin(self.theta[p1_idx])
            B = self.t[p1_idx]*math.cos(self.theta[p1_idx]) - self.t[p2_idx]*math.cos(self.theta[p2_idx])
            self._log_math(f"A = t{p2_idx}*sinθ{p2_idx} - t{p1_idx}*sinθ{p1_idx}", A)
            self._log_math(f"B = t{p1_idx}*cosθ{p1_idx} - t{p2_idx}*cosθ{p2_idx}", B)

            candidates = self._generate_phi_candidates(A, B, is_vertical=True)
            
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
        if abs(A) < self.config.tol:
            base_phi = math.copysign(math.pi/2, -B) if not is_vertical else (0.0 if B > 0 else math.pi)
        else:
            base_phi = math.atan2(-B, A) if not is_vertical else math.atan2(B, -A)
            
        return [base_phi, base_phi + math.pi]

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
                # 计算相对坐标
                P = {
                    i: (
                        self.t[i] * math.cos(phi + self.theta[i]),
                        self.t[i] * math.sin(phi + self.theta[i])
                    ) for i in [p1_idx, p2_idx, opp_idx]
                }
                
                # 基边对齐检查
                dx = abs(P[p2_idx][0] - P[p1_idx][0]) if is_vertical else 0.0
                dy = 0.0 if is_vertical else abs(P[p2_idx][1] - P[p1_idx][1])
                
                if not (self._log_compare(dx, 0.0) if is_vertical else self._log_compare(dy, 0.0)):
                    continue
                    
                # 邻边约束检查
                if not self._check_adjacent_constraint(P[opp_idx], P[p1_idx], P[p2_idx], adj_edge):
                    continue
                    
                # 尺寸检查
                if self._check_dimension_constraints(P.values()):
                    valid_phis.append(phi)
                    
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
        p1_idx = base_idx
        opp_idx = (base_idx + 2) % 3
        
        if rect_edge in ['bottom', 'top']:
            y_target = 0.0 if rect_edge == 'bottom' else self.n
            yO = y_target - self.t[p1_idx] * math.sin(phi + self.theta[p1_idx])
            
            x_target = 0.0 if adj_edge == 'left' else self.m
            x_opp = self.t[opp_idx] * math.cos(phi + self.theta[opp_idx])
            xO = x_target - x_opp
        else:
            x_target = 0.0 if rect_edge == 'left' else self.m
            xO = x_target - self.t[p1_idx] * math.cos(phi + self.theta[p1_idx])
            
            y_target = 0.0 if adj_edge == 'bottom' else self.n
            y_opp = self.t[opp_idx] * math.sin(phi + self.theta[opp_idx])
            yO = y_target - y_opp
            
        self._log_math(f"Center position", (xO, yO))
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
        p1_idx = base_idx
        p2_idx = (base_idx + 1) % 3
        
        # 检查顶点是否在对应边缘
        for i, edge in [(p1_idx, rect_edge), (p2_idx, rect_edge)]:
            x = xO + self.t[i] * math.cos(phi + self.theta[i])
            y = yO + self.t[i] * math.sin(phi + self.theta[i])
            
            if edge == 'bottom':
                valid = abs(y) <= self.case_config.edge_tolerance
            elif edge == 'top':
                valid = abs(y - self.n) <= self.case_config.edge_tolerance
            elif edge == 'left':
                valid = abs(x) <= self.case_config.edge_tolerance
            else:  # 'right'
                valid = abs(x - self.m) <= self.case_config.edge_tolerance
                
            if not valid:
                self.logger.debug(f"P{i} not on {edge} edge")
                return False
                
        return True

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