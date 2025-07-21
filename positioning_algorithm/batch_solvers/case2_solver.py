"""
Case2批处理求解器
处理基边对齐一条边，对角在对边的情况
未来将与Case1合并
"""
import numpy as np
from typing import List, Tuple
import logging
from .trig_cache import trig_cache

class Case2BatchSolver:
    """Case2批处理求解器类"""
    
    def __init__(self, m: float, n: float, tolerance: float = 1e-3):
        """
        初始化Case2求解器
        
        Args:
            m: 场地宽度
            n: 场地高度  
            tolerance: 数值容差
        """
        self.m = m
        self.n = n
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
    
    def solve(self, combinations: np.ndarray) -> List[Tuple]:
        """
        求解Case2批处理
        
        Args:
            combinations: 激光组合数组 (N, 3, 2) - 每行为[t, theta]
            
        Returns:
            List[Tuple]: 解列表，每个解为 ((x_min, x_max), (y_min, y_max), phi)
        """
        self.logger.debug(f"开始Case2批处理求解，组合数量: {len(combinations)}")
        
        if len(combinations) == 0:
            self.logger.warning("没有组合可求解")
            return []
        
        solutions = []
        
        # 使用三角函数缓存
        sin_theta, cos_theta = trig_cache.get_theta_trigonometry()
        
        # 遍历每个组合
        for combo_idx, combination in enumerate(combinations):
            # 为每个基边配置求解
            for base_idx in range(3):
                base_solutions = self._solve_single_base(
                    combination, base_idx, combo_idx, sin_theta, cos_theta
                )
                solutions.extend(base_solutions)
        
        self.logger.info(f"Case2求解完成，找到 {len(solutions)} 个解")
        return solutions
    
    def _solve_single_base(self, combination: np.ndarray, base_idx: int, 
                          combo_idx: int, sin_theta: np.ndarray, 
                          cos_theta: np.ndarray) -> List[Tuple]:
        """
        求解单个基边配置
        
        Args:
            combination: 单个组合 (3, 2)
            base_idx: 基边索引
            combo_idx: 组合索引
            sin_theta, cos_theta: 预计算的三角函数值
            
        Returns:
            List[Tuple]: 该基边配置的解列表，格式为 ((x_min, x_max), (y_min, y_max), phi)
        """
        solutions = []
        
        # 模拟生成一些解（Case2特有的逻辑）
        # 实际实现时会根据具体的Case2逻辑计算真实解
        
        # 生成1-2个模拟解
        for sol_idx in range(1, 3):
            # 模拟phi值 - 与Case1稍有不同
            phi = (combo_idx * 2 + base_idx + sol_idx) * 0.7
            phi = phi % (2 * np.pi)
            
            # 模拟x范围 - Case2特有的分布
            x_center = (combo_idx * 0.4 + base_idx * 0.2) % self.m
            x_width = 0.08 + sol_idx * 0.03
            x_min = max(0, x_center - x_width)
            x_max = min(self.m, x_center + x_width)
            
            # 模拟y范围 - Case2特有的分布
            y_center = (combo_idx * 0.3 + sol_idx * 0.5) % self.n
            y_height = 0.12 + base_idx * 0.01
            y_min = max(0, y_center - y_height)
            y_max = min(self.n, y_center + y_height)
            
            # 确保范围有效
            if x_max > x_min and y_max > y_min:
                # 返回正确的解格式: ((x_min, x_max), (y_min, y_max), phi)
                solution = ((x_min, x_max), (y_min, y_max), phi)
                solutions.append(solution)
                
                self.logger.debug(
                    f"生成Case2解: combo={combo_idx}, base={base_idx}, "
                    f"x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}], "
                    f"phi={phi:.3f}"
                )
        
        return solutions
    
    def get_solver_info(self) -> dict:
        """获取求解器信息"""
        return {
            "name": "Case2BatchSolver",
            "field_size": (self.m, self.n),
            "tolerance": self.tolerance,
            "description": "基边对齐一条边，对角在对边"
        }
