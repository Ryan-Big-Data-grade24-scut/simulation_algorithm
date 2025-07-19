import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
from itertools import combinations

# 路径处理
if __name__ == '__main__':
    import sys
    sys.path.append('/home/lrx/laser_positioning_ubuntu/simulation_algorithm/positioning_algorithm')
    from case_solvers.BaseSolver import BaseSolverConfig
    from case_solvers.case1_solver import Case1Solver
    from case_solvers.case2_solver import Case2Solver
    from case_solvers.case3_solver import Case3Solver
else:
    from .case_solvers.BaseSolver import BaseSolverConfig
    from .case_solvers.case1_solver import Case1Solver
    from .case_solvers.case2_solver import Case2Solver
    from .case_solvers.case3_solver import Case3Solver

@dataclass
class SolverConfig:
    """求解器全局配置
    
    Attributes:
        max_solutions (int): 最大返回解数量，默认4
        compatibility_threshold (float): 解相容性阈值，默认0.8
        enable_ros_logging (bool): 是否启用ROS日志，默认False
    """
    max_solutions: int = 4
    compatibility_threshold: float = 0.8
    enable_ros_logging: bool = False

class PoseSolver:
    """多激光定位求解器主类
    
    Args:
        m (float): 场地x方向长度 (m)
        n (float): 场地y方向长度 (m)
        laser_config (List): 激光配置列表，格式为:
            [((相对距离,相对角度), 激光朝向), ...]
        tol (float): 计算容忍度，默认1e-3
        config (Optional[SolverConfig]): 求解器全局配置
        ros_logger (Optional): ROS2日志器对象
    """

    def __init__(self, 
                 m: float, 
                 n: float, 
                 laser_config: List,
                 tol: float = 1e-3,
                 config: Optional[SolverConfig] = None,
                 ros_logger=None):
        self.m = m
        self.n = n
        self.laser_config = laser_config
        self.tol = tol
        self.config = config or SolverConfig()
        self.ros_logger = ros_logger if self.config.enable_ros_logging else None

        # 日志等级控制
        if self.config.enable_ros_logging:
            self.min_log_level = logging.WARNING  # 只输出WARNING及以上
        else:
            self.min_log_level = logging.DEBUG

        # 初始化标准日志器
        self.logger = logging.getLogger("PoseSolver")
        if not self.logger.hasHandlers():
            handler = logging.FileHandler("logs/pose_solver.log", encoding="utf-8")
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(self.min_log_level)

        os.makedirs('logs', exist_ok=True)
        self._initialize_solvers()
        if self.ros_logger:
            self.ros_logger.info("PoseSolver initialized successfully")
        else:
            self.logger.info("PoseSolver initialized successfully")

    def _initialize_solvers(self):
        """初始化三种情况的求解器实例"""
        self.solver_configs = [
            BaseSolverConfig(
                tol=self.tol,
                log_enabled=not self.config.enable_ros_logging,
                log_file="logs/case1.log"
            ),
            BaseSolverConfig(
                tol=self.tol,
                log_enabled=not self.config.enable_ros_logging,
                log_file="logs/case2.log"
            ),
            BaseSolverConfig(
                tol=self.tol,
                log_enabled=not self.config.enable_ros_logging,
                log_file="logs/case3.log"
            )
        ]

        # 传递min_log_level参数
        self.solvers = [
            Case1Solver([1,1,1], [0,0,0], self.m, self.n, 
                       config=self.solver_configs[0],
                       ros_logger=self.ros_logger,
                       min_log_level=self.min_log_level),
            Case2Solver([1,1,1], [0,0,0], self.m, self.n,
                       config=self.solver_configs[1],
                       ros_logger=self.ros_logger,
                       min_log_level=self.min_log_level),
            Case3Solver([1,1,1], [0,0,0], self.m, self.n,
                       config=self.solver_configs[2],
                       ros_logger=self.ros_logger,
                       min_log_level=self.min_log_level)
        ]

    def solve(self, distances: np.ndarray) -> List[Tuple]:
        """执行多激光定位求解
        
        Args:
            distances (np.ndarray): 激光测距值数组
            
        Returns:
            List[Tuple]: 有效解列表，每个解格式为:
                ((x_min, x_max), (y_min, y_max), phi)
                
        Raises:
            ValueError: 当输入距离数与激光配置不匹配时
        """
        try:
            # 参数校验
            if len(distances) != len(self.laser_config):
                error_msg = f"距离数{len(distances)}与激光配置数{len(self.laser_config)}不匹配"
                if self.ros_logger:
                    self.ros_logger.error(error_msg)
                else:
                    self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if self.ros_logger:
                self.ros_logger.info(f"Start solving with {len(distances)} distances")
            else:
                self.logger.info(f"Start solving with {len(distances)} distances")
            
            # 1. 计算碰撞向量
            r, delta, theta = self._get_laser_params()
            t_list, theta_list = self._calculate_collision_vectors(distances, r, delta, theta)
            
            # 2. 生成激光组合
            combinations = self._generate_combinations(t_list, theta_list)
            self.logger.info(f"Generated {len(combinations)} laser combinations")
            
            # 3. 多情况求解
            results = []
            for idx, (t, theta) in enumerate(combinations, 1):
                self.logger.info(f"组合{idx}: t={t}, theta={theta}")
                results.append(self._solve_three_cases(t, theta))
            
            # 4. 筛选最优解
            solutions = self._filter_solutions(results)
            self.logger.info(f"筛选后有效解数量: {len(solutions)}")
            for i, sol in enumerate(solutions, 1):
                self.logger.info(f"Solution {i}: {sol}")
            return solutions

        except Exception as e:
            if self.ros_logger:
                self.ros_logger.error(f"Solve failed: {str(e)}")
            else:
                self.logger.error(f"Solve failed: {str(e)}")
            raise

    def _get_laser_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """从配置中提取激光参数
        
        Returns:
            Tuple: (r, delta, theta) 三个numpy数组，分别表示:
                r: 相对距离数组
                delta: 相对角度数组 (rad)
                theta: 激光朝向数组 (rad)
        """
        r, delta, theta = [], [], []
        for (rel_r, rel_angle), laser_angle in self.laser_config:
            r.append(rel_r)
            delta.append(rel_angle)
            theta.append(laser_angle)
        return np.array(r), np.array(delta), np.array(theta)

    def _calculate_collision_vectors(self, 
                                   distances: np.ndarray,
                                   r: np.ndarray,
                                   delta: np.ndarray,
                                   theta: np.ndarray) -> Tuple[List[float], List[float]]:
        """计算碰撞向量
        
        Args:
            distances: 激光测距值数组
            r: 相对距离数组
            delta: 相对角度数组
            theta: 激光朝向数组
            
        Returns:
            Tuple: (t_list, theta_list) 碰撞向量参数
        """
        t_list, theta_list = [], []
        for i in range(len(distances)):
            x = r[i]*np.cos(delta[i]) + distances[i]*np.cos(theta[i])
            y = r[i]*np.sin(delta[i]) + distances[i]*np.sin(theta[i])
            t_val = np.linalg.norm([x, y])
            theta_val = np.arctan2(y, x)
            if theta_val < 0:
                theta_val += 2 * np.pi
            t_list.append(t_val)
            theta_list.append(theta_val)
            if self.ros_logger:
                self.ros_logger.debug(f"Laser {i}: t={t_val:.6f}, theta={theta_val:.6f}")
            else:
                logging.getLogger("PoseSolver").debug(f"Laser {i}: t={t_val:.6f}, theta={theta_val:.6f}")
        return t_list, theta_list

    def _generate_combinations(self, 
                             t_list: List[float], 
                             theta_list: List[float]) -> List[Tuple]:
        """生成三激光组合
        
        Args:
            t_list: 碰撞向量t值列表
            theta_list: 碰撞向量角度列表
            
        Returns:
            List[Tuple]: 所有可能的3激光组合，每个组合格式为:
                ([t1,t2,t3], [theta1,theta2,theta3])
        """
        if len(t_list) < 3:
            if self.ros_logger:
                self.ros_logger.warning(f"Not enough lasers ({len(t_list)}) for combinations")
            else:
                logging.getLogger("PoseSolver").warning(f"Not enough lasers ({len(t_list)}) for combinations")
            return []
        indices = range(len(t_list))
        return [
            (
                [t_list[i] for i in combo],
                [theta_list[i] for i in combo]
            )
            for combo in combinations(indices, 3)
        ]

    def _solve_three_cases(self, 
                         t: List[float], 
                         theta: List[float]) -> List[Tuple]:
        """调用三种情况求解器
        
        Args:
            t: 3个激光的t值
            theta: 3个激光的角度值
            
        Returns:
            List[Tuple]: 所有求解器返回的解
        """
        results = []
        for solver in self.solvers:
            try:
                solver.t = t
                solver.theta = theta
                results.extend(solver.solve())
            except Exception as e:
                if self.ros_logger:
                    self.ros_logger.warning(f"Solver {solver.__class__.__name__} failed: {str(e)}")
                else:
                    logging.getLogger("PoseSolver").warning(f"Solver {solver.__class__.__name__} failed: {str(e)}")
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

    def _merge_compatible_solutions(self, sol1, sol2):
        """
        合并两个相容的解，返回平均后的解
        参数:
            sol1: ((xmin1,xmax1), (ymin1,ymax1), phi1)
            sol2: ((xmin2,xmax2), (ymin2,ymax2), phi2)
        返回:
            merged_sol: 合并后的解
        """
        # x范围合并（取交集的中心扩展）
        x_overlap_min = max(sol1[0][0], sol2[0][0])
        x_overlap_max = min(sol1[0][1], sol2[0][1])
        x_center = (x_overlap_min + x_overlap_max) / 2
        x_half_width = (sol1[0][1] - sol1[0][0] + sol2[0][1] - sol2[0][0]) / 4
        merged_x = (x_center - x_half_width, x_center + x_half_width)
        
        # y范围合并
        y_overlap_min = max(sol1[1][0], sol2[1][0])
        y_overlap_max = min(sol1[1][1], sol2[1][1])
        y_center = (y_overlap_min + y_overlap_max) / 2
        y_half_width = (sol1[1][1] - sol1[1][0] + sol2[1][1] - sol2[1][0]) / 4
        merged_y = (y_center - y_half_width, y_center + y_half_width)
        
        # phi角度合并（考虑周期性）
        phi1, phi2 = sol1[2], sol2[2]
        phi_diff = abs(phi1 - phi2) % (2 * np.pi)
        if phi_diff > np.pi:
            phi_diff = 2 * np.pi - phi_diff
            if phi1 > phi2:
                phi2 += 2 * np.pi
            else:
                phi1 += 2 * np.pi
        merged_phi = (phi1 + phi2) / 2
        # 标准化到[0, 2π)
        merged_phi = merged_phi % (2 * np.pi)
        
        return (merged_x, merged_y, merged_phi)

    def _filter_solutions(self, all_solutions):
        """
        高效解筛选器，支持相容解合并（O(n^2)时间复杂度）
        参数:
            all_solutions: list[激光组合1的解列表, 激光组合2的解列表, ...]
        返回:
            list[合并后的解] 按相容数量降序排列
        """
        solutions = []  # 格式: [[sol, count, [相容解列表]], ...]
        """
        # 测试模式
        for sol in all_solutions:
            if not sol:
                continue
            self.logger.debug(f"当前激光组合解: {sol}")
            solutions.extend(sol)
        return solutions
        """
        self.logger.info(f"开始筛选所有解，总组合数: {len(all_solutions)}")
        
        for laser_solutions in all_solutions:
            for current_sol in laser_solutions:
                found_compatible = False
                
                # 在已有解中寻找相容解
                for i in range(len(solutions)):
                    existing_sol, count, compatible_sols = solutions[i]
                    if self._is_compatible(existing_sol, current_sol):
                        # 增加相容计数并记录相容解
                        compatible_sols.append(current_sol)
                        solutions[i][1] += 1
                        
                        # 重新计算合并解
                        merged_sol = existing_sol
                        for comp_sol in compatible_sols:
                            merged_sol = self._merge_compatible_solutions(merged_sol, comp_sol)
                        solutions[i][0] = merged_sol
                        
                        found_compatible = True
                        self.logger.debug(f"解 {current_sol} 与已存在解相容，合并后: {merged_sol}")
                        break
                
                # 如果没有找到相容解，添加新解
                if not found_compatible:
                    solutions.append([current_sol, 1, []])
                    self.logger.debug(f"添加新解: {current_sol}")
        
        # 按相容数量排序
        solutions.sort(key=lambda x: -x[1])
        self.logger.info(f"筛选后解列表:")
        for i, (sol, count, _) in enumerate(solutions):
            self.logger.info(f"  解{i+1}(相容数:{count}): {sol}")
        
        # 优先返回完全相容的解
        if solutions and solutions[0][1] == len(all_solutions):
            self.logger.info("找到完全相容解")
            return [sol[0] for sol in solutions[:4] if sol[1] == len(all_solutions)]
        
        return [sol[0] for sol in solutions[:4]]


def _test_pose_solver():

    """PoseSolver 测试函数"""
    import sys
    sys.path.append('/home/lrx/laser_positioning_ubuntu/simulation_algorithm/positioning_algorithm')
    
    # 测试配置
    laser_config = [
        ((0.1, 0.0), 0.0),  # 激光1
        ((0.1, np.pi/2), np.pi/2),  # 激光2
        ((0.1, np.pi), np.pi),  # 激光3
        ((0.1, 3*np.pi/2), 3*np.pi/2)  # 激光4
    ]
    
    solver = PoseSolver(
        m=2.0,
        n=2.0,
        laser_config=laser_config,
        tol=1e-4
    )
    
    # 测试数据
    distances = np.array([1.0, 1.0, 1.0, 1.0])
    solutions = solver.solve(distances)
    
    print(f"Found {len(solutions)} solutions:")
    for i, sol in enumerate(solutions, 1):
        print(f"Solution {i}: {sol}")

if __name__ == "__main__":
    _test_pose_solver()