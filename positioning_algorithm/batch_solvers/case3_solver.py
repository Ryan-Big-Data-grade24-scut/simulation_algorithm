"""
Case3批处理求解器
处理三点分别在三条边的情况
"""
import numpy as np
from typing import List, Tuple
import logging
from .trig_cache import trig_cache

class Case3BatchSolver:
    """Case3批处理求解器类"""
    
    def __init__(self, m: float, n: float, tolerance: float = 1e-3):
        """
        初始化Case3求解器
        
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
        求解Case3批处理
        
        Args:
            combinations: 激光组合数组 (N, 3, 2) - 每行为[t, theta]
            
        Returns:
            List[Tuple]: 解列表，每个解为 ((x_min, x_max), (y_min, y_max), phi)
        """
        self.logger.debug(f"开始Case3批处理求解，组合数量: {len(combinations)}")
        
        if len(combinations) == 0:
            self.logger.warning("没有组合可求解")
            return []
        
        solutions = []
        
        # 使用三角函数缓存
        sin_theta, cos_theta = trig_cache.get_theta_trigonometry()
        
        # Case3的扩展组合处理 - 6种排列
        expanded_combinations = self._expand_case3_combinations(combinations)
        
        # 遍历每个扩展组合
        for combo_idx, (combination, perm_idx) in enumerate(expanded_combinations):
            combo_solutions = self._solve_single_combination(
                combination, perm_idx, combo_idx, sin_theta, cos_theta
            )
            solutions.extend(combo_solutions)
        
        self.logger.info(f"Case3求解完成，找到 {len(solutions)} 个解")
        return solutions
    
    def _expand_case3_combinations(self, combinations: np.ndarray) -> List[Tuple]:
        """
        扩展Case3组合，生成所有6种排列
        输入: (N, 3, 2) 
        输出: List[(combination, perm_idx)] - 每个组合扩展为6种排列
        """
        permutations = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        expanded_combinations = []
        
        for combo in combinations:
            for perm_idx, (p0, p1, p2) in enumerate(permutations):
                expanded_combo = combo[[p0, p1, p2]]
                expanded_combinations.append((expanded_combo, perm_idx))
        
        self.logger.debug(f"Case3组合扩展: {len(combinations)} -> {len(expanded_combinations)}")
        return expanded_combinations
    
    def _solve_single_combination(self, combination: np.ndarray, perm_idx: int,
                                 combo_idx: int, sin_theta: np.ndarray, 
                                 cos_theta: np.ndarray) -> List[Tuple]:
        """
        求解单个扩展组合
        
        Args:
            combination: 单个组合 (3, 2)
            perm_idx: 排列索引
            combo_idx: 组合索引
            sin_theta, cos_theta: 预计算的三角函数值
            
        Returns:
            List[Tuple]: 该组合的解列表，格式为 ((x_min, x_max), (y_min, y_max), phi)
        """
        solutions = []
        
        # 模拟生成一些解（Case3特有的逻辑）
        # 实际实现时会根据具体的Case3逻辑计算真实解
        
        # 生成1个模拟解（Case3通常解较少）
        for sol_idx in range(1):
            # 模拟phi值 - 基于排列索引
            phi = (combo_idx + perm_idx * 0.3 + sol_idx) * 0.9
            phi = phi % (2 * np.pi)
            
            # 模拟x范围 - Case3特有的分布
            x_center = (perm_idx * 0.6 + combo_idx * 0.1) % self.m
            x_width = 0.05 + sol_idx * 0.02
            x_min = max(0, x_center - x_width)
            x_max = min(self.m, x_center + x_width)
            
            # 模拟y范围 - Case3特有的分布
            y_center = (perm_idx * 0.4 + sol_idx * 0.7) % self.n
            y_height = 0.08 + perm_idx * 0.005
            y_min = max(0, y_center - y_height)
            y_max = min(self.n, y_center + y_height)
            
            # 确保范围有效
            if x_max > x_min and y_max > y_min:
                # 返回正确的解格式: ((x_min, x_max), (y_min, y_max), phi)
                solution = ((x_min, x_max), (y_min, y_max), phi)
                solutions.append(solution)
                
                self.logger.debug(
                    f"生成Case3解: combo={combo_idx}, perm={perm_idx}, "
                    f"x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}], "
                    f"phi={phi:.3f}"
                )
        
        return solutions
    
    def get_solver_info(self) -> dict:
        """获取求解器信息"""
        return {
            "name": "Case3BatchSolver",
            "field_size": (self.m, self.n),
            "tolerance": self.tolerance,
            "description": "三点分别在三条边"
        }
