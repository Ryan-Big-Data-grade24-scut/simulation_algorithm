import numpy as np
from typing import List, Tuple
from .case_solvers.BaseSolver import BaseSolverConfig
from .case_solvers.case1_solver import Case1Solver
from .case_solvers.case2_solver import Case2Solver
from .case_solvers.case3_solver import Case3Solver

class PoseSolver:
    def __init__(self, m: float, n: float, laser_config: List, tol: float = 1e-3):
        """
        初始化（完全匹配图片中的模块定义）
        参数:
            m, n: 场地尺寸
            laser_config: [((相对距离,相对角度), 激光朝向), ...]
        """
        self.m = m
        self.n = n
        self.laser_config = laser_config
        self.tol = tol
        
        # 初始化三个求解器（匹配图片中的"求解器初始化"）
        self.configs = [
            BaseSolverConfig(tol=self.tol, log_enabled=True, log_file="logs/solver1.log"),
            BaseSolverConfig(tol=self.tol, log_enabled=True, log_file="logs/solver2.log"), 
            BaseSolverConfig(tol=self.tol, log_enabled=True, log_file="logs/solver3.log")
        ]

        self.solvers = [
            Case1Solver([1,1,1], [0,0,0], m, n, config=self.configs[0]),
            Case2Solver([1,1,1], [0,0,0], m, n, config=self.configs[1]),
            Case3Solver([1,1,1], [0,0,0], m, n, config=self.configs[2])
        ]

    def solve(self, distances: np.ndarray) -> List[Tuple]:
        """
        主求解方法（匹配图片中的"求解"流程）
        返回: [((xmin,xmax), (ymin,ymax), phi), ...]
        """
        # 步骤1：计算碰撞向量（匹配图片中的"由配置+长度求解碰撞向量"）
        r, delta, theta = self._get_laser_params()
        t_list, theta_list = self._calculate_collision_vectors(
            distances, r, delta, theta
        )
        
        # 步骤2：生成三激光组合（匹配图片中的"求出不同组合的解"）
        combinations = self._generate_combinations(t_list, theta_list)
        
        # 步骤3：三种情况分别求解
        results = []
        for t, theta in combinations:
            results.append(self._solve_three_cases(t, theta))
        
        # 步骤4：筛选解（临时实现，后续完善）
        return self._filter_solutions(results)

    def _get_laser_params(self) -> Tuple:
        """从配置中提取激光参数"""
        r, delta, theta = [], [], []
        for (rel_r, rel_angle), laser_angle in self.laser_config:
            r.append(rel_r)
            delta.append(rel_angle)
            theta.append(laser_angle)
        return np.array(r), np.array(delta), np.array(theta)

    def _calculate_collision_vectors(self, distances, r, delta, theta):
        """
        计算碰撞向量（严格匹配图片中的数学过程）
        返回: (t_list, theta_list)
        """
        t_list = []
        theta_list = []
        for i in range(len(distances)):
            x = r[i]*np.cos(delta[i]) + distances[i]*np.cos(theta[i])
            y = r[i]*np.sin(delta[i]) + distances[i]*np.sin(theta[i])
            
            t_val = np.linalg.norm([x, y])
            theta_val = np.arctan2(y, x)
            if theta_val < 0:
                theta_val += 2 * np.pi
                
            t_list.append(t_val)
            theta_list.append(theta_val)
        return t_list, theta_list

    def _generate_combinations(self, t_list: List[float], theta_list: List[float]) -> List[Tuple]:
        """
        生成三激光组合（严格匹配图片中的"求出不同组合的解"要求）
        返回: [(t_subset, theta_subset), ...] 其中每个subset为3个元素的列表
        """
        # 示例实现：生成所有可能的3激光组合（C(n,3)种）
        from itertools import combinations
        indices = range(len(t_list))
        return [
            (
                [t_list[i] for i in combo],
                [theta_list[i] for i in combo]
            )
            for combo in combinations(indices, 3)
            if len(t_list) >= 3  # 至少需要3束激光
        ]

    def _solve_three_cases(self, t, theta):
        """
        三种情况分别求解（匹配图片中的流程）
        返回: [solution1, solution2, ...]
        """
        results = []
        for solver in self.solvers:
            solver.t = t
            solver.theta = theta
            results.extend(solver.solve())
        return results

    def _is_compatible(self, sol1, sol2):
        """
        快速相容性检查（仅判断是否相容）
        参数:
            sol1: ((xmin1,xmax1), (ymin1,ymax1), phi1)
            sol2: ((xmin2,xmax2), (ymin2,ymax2), phi2)
        返回:
            bool: 是否相容
        """
        # 检查x范围重叠（允许容忍度）
        if not (max(sol1[0][0], sol2[0][0]) <= min(sol1[0][1], sol2[0][1]) + self.tol):
            return False
            
        # 检查y范围重叠
        if not (max(sol1[1][0], sol2[1][0]) <= min(sol1[1][1], sol2[1][1]) + self.tol):
            return False
            
        # 检查phi角度差（弧度制）
        phi_diff = abs(sol1[2] - sol2[2]) % (2 * np.pi)
        phi_diff = min(phi_diff, 2 * np.pi - phi_diff)
        return phi_diff <= self.tol

    def _filter_solutions(self, all_solutions):
        """
        高效解筛选器（O(n^2)时间复杂度）
        参数:
            all_solutions: list[激光组合1的解列表, 激光组合2的解列表, ...]
        返回:
            list[ [sol, 相容数量], ... ] 按相容数量降序排列
        """
        solutions = []  # 存储格式: [ [sol, count], ... ]
        
        for laser_solutions in all_solutions:
            for current_sol in laser_solutions:
                found_compatible = False
                
                # 在已有解中寻找相容解
                for i in range(len(solutions)):
                    existing_sol, count = solutions[i]
                    if self._is_compatible(existing_sol, current_sol):
                        solutions[i][1] += 1  # 增加相容计数
                        found_compatible = True
                        break
                
                # 如果没有找到相容解，添加新解
                if not found_compatible:
                    solutions.append([current_sol, 1])
        
        # 按相容数量降序排序
        solutions.sort(key=lambda x: -x[1])

        if solutions[0][1] == len(all_solutions):
            return [sol[0] for sol in solutions[:4] if sol[1] == len(all_solutions)]
        
        return [sol[0] for sol in solutions[:4]]