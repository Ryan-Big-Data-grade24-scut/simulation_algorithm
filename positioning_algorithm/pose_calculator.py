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
        
        # 清理已有的handlers，避免重复添加
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
        
        # 添加新的FileHandler
        if not self.config.enable_ros_logging:
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
                log_file="logs/case1.log",
                log_level="DEBUG"
            ),
            BaseSolverConfig(
                tol=self.tol,
                log_enabled=not self.config.enable_ros_logging,
                log_file="logs/case2.log",
                log_level="DEBUG"
            ),
            BaseSolverConfig(
                tol=self.tol,
                log_enabled=not self.config.enable_ros_logging,
                log_file="logs/case3.log",
                log_level="DEBUG"
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
        """调用三种情况求解器，分别处理每种Case的结果
        
        Args:
            t: 3个激光的t值
            theta: 3个激光的角度值
            
        Returns:
            List[Tuple]: 所有求解器返回的解
        """
        self.logger.info("=" * 60)
        self.logger.info("开始三种情况求解")
        self.logger.info("=" * 60)
        
        # 分别存储每种算法的结果
        case1_results = []
        case2_results = []
        case3_results = []
        
        # Case1求解
        try:
            self.solvers[0].t = t
            self.solvers[0].theta = theta
            
            self.logger.info(f"调用 {self.solvers[0].__class__.__name__}...")
            case1_results = self.solvers[0].solve()
            
            self.logger.info(f"{self.solvers[0].__class__.__name__} 求解完成:")
            if case1_results:
                self.logger.info(f"  共找到 {len(case1_results)} 个解:")
                for j, sol in enumerate(case1_results, 1):
                    self.logger.info(f"    解{j}: {self._format_solution(sol)}")
            else:
                self.logger.info("  无有效解")
            self.logger.info("-" * 40)
                
        except Exception as e:
            self.logger.warning(f"{self.solvers[0].__class__.__name__} 求解失败: {str(e)}")
            self.logger.info("  无有效解 (发生异常)")
            self.logger.info("-" * 40)
            if self.ros_logger:
                self.ros_logger.warning(f"Solver {self.solvers[0].__class__.__name__} failed: {str(e)}")
        
        # Case2求解
        try:
            self.solvers[1].t = t
            self.solvers[1].theta = theta
            
            self.logger.info(f"调用 {self.solvers[1].__class__.__name__}...")
            case2_results = self.solvers[1].solve()
            
            self.logger.info(f"{self.solvers[1].__class__.__name__} 求解完成:")
            if case2_results:
                self.logger.info(f"  共找到 {len(case2_results)} 个解:")
                for j, sol in enumerate(case2_results, 1):
                    self.logger.info(f"    解{j}: {self._format_solution(sol)}")
            else:
                self.logger.info("  无有效解")
            self.logger.info("-" * 40)
                
        except Exception as e:
            self.logger.warning(f"{self.solvers[1].__class__.__name__} 求解失败: {str(e)}")
            self.logger.info("  无有效解 (发生异常)")
            self.logger.info("-" * 40)
            if self.ros_logger:
                self.ros_logger.warning(f"Solver {self.solvers[1].__class__.__name__} failed: {str(e)}")
        
        # Case3求解
        try:
            self.solvers[2].t = t
            self.solvers[2].theta = theta
            
            self.logger.info(f"调用 {self.solvers[2].__class__.__name__}...")
            case3_results = self.solvers[2].solve()
            
            self.logger.info(f"{self.solvers[2].__class__.__name__} 求解完成:")
            if case3_results:
                self.logger.info(f"  共找到 {len(case3_results)} 个解:")
                for j, sol in enumerate(case3_results, 1):
                    self.logger.info(f"    解{j}: {self._format_solution(sol)}")
            else:
                self.logger.info("  无有效解")
            self.logger.info("-" * 40)
                
        except Exception as e:
            self.logger.warning(f"{self.solvers[2].__class__.__name__} 求解失败: {str(e)}")
            self.logger.info("  无有效解 (发生异常)")
            self.logger.info("-" * 40)
            if self.ros_logger:
                self.ros_logger.warning(f"Solver {self.solvers[2].__class__.__name__} failed: {str(e)}")
        
        # 角度归一化
        self.logger.info("开始角度归一化处理:")
        case1_results = self._normalize_solution_angles(case1_results)
        case2_results = self._normalize_solution_angles(case2_results)
        case3_results = self._normalize_solution_angles(case3_results)
        self.logger.info("角度归一化完成")
        
        # Case2与Case3冲突解决
        if case2_results and case3_results:
            self.logger.info("检测到Case2和Case3都有解，开始冲突检测...")
            case3_results = self._remove_case3_case2_conflicts(case2_results, case3_results)
        else:
            self.logger.info("无需进行Case2/Case3冲突检测")
        
        # 合并所有结果
        all_results = case1_results + case2_results + case3_results
        
        self.logger.info(f"三算法求解汇总:")
        self.logger.info(f"  Case1: {len(case1_results)} 个解")
        self.logger.info(f"  Case2: {len(case2_results)} 个解")
        self.logger.info(f"  Case3: {len(case3_results)} 个解")
        self.logger.info(f"  总计: {len(all_results)} 个解")
        
        return all_results

    def _normalize_angle(self, angle):
        """
        将角度归一化到 [-π, π] 范围
        参数:
            angle: 角度值（弧度）
        返回:
            归一化后的角度
        """
        import math
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _normalize_solution_angles(self, solutions):
        """
        归一化所有解的角度到 [-π, π] 范围
        参数:
            solutions: 解列表
        返回:
            归一化后的解列表
        """
        normalized_solutions = []
        for sol in solutions:
            x_range, y_range, phi = sol
            normalized_phi = self._normalize_angle(phi)
            normalized_solutions.append((x_range, y_range, normalized_phi))
        return normalized_solutions

    def _remove_case3_case2_conflicts(self, case2_solutions, case3_solutions):
        """
        移除与Case2解冲突的Case3解
        参数:
            case2_solutions: Case2解列表
            case3_solutions: Case3解列表
        返回:
            过滤后的Case3解列表
        """
        if not case2_solutions or not case3_solutions:
            return case3_solutions
        
        filtered_case3 = []
        tolerance = 1e-6  # 角度容忍度
        
        self.logger.info("检测Case2与Case3冲突:")
        
        for i, case3_sol in enumerate(case3_solutions):
            is_duplicate = False
            case3_x, case3_y, case3_phi = case3_sol
            
            for j, case2_sol in enumerate(case2_solutions):
                case2_x, case2_y, case2_phi = case2_sol
                
                # 检查角度是否近似相等
                phi_diff = abs(case3_phi - case2_phi)
                if phi_diff < tolerance or abs(phi_diff - 2*3.14159) < tolerance:
                    # 检查位置范围是否重叠
                    x_overlap = (case3_x[0] <= case2_x[1] + tolerance and 
                               case2_x[0] <= case3_x[1] + tolerance)
                    y_overlap = (case3_y[0] <= case2_y[1] + tolerance and 
                               case2_y[0] <= case3_y[1] + tolerance)
                    
                    if x_overlap and y_overlap:
                        self.logger.info(f"  Case3解{i+1} 与 Case2解{j+1} 冲突，移除Case3解")
                        self.logger.info(f"    Case3: {self._format_solution(case3_sol)}")
                        self.logger.info(f"    Case2: {self._format_solution(case2_sol)}")
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_case3.append(case3_sol)
        
        self.logger.info(f"冲突检测完成: Case3从{len(case3_solutions)}个解减少到{len(filtered_case3)}个解")
        return filtered_case3

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
        智能合并两个相容的解
        优先保持精度较高的解，而不是简单平均
        参数:
            sol1: ((xmin1,xmax1), (ymin1,ymax1), phi1)
            sol2: ((xmin2,xmax2), (ymin2,ymax2), phi2)
        返回:
            merged_sol: 合并后的解
        """
        x1_range, y1_range, phi1 = sol1
        x2_range, y2_range, phi2 = sol2
        
        # 计算解的精度（范围越小精度越高）
        x1_precision = x1_range[1] - x1_range[0]
        x2_precision = x2_range[1] - x2_range[0]
        y1_precision = y1_range[1] - y1_range[0]
        y2_precision = y2_range[1] - y2_range[0]
        
        self.logger.info(f"      合并前分析:")
        self.logger.info(f"        解1精度: x_width={x1_precision:.6f}, y_width={y1_precision:.6f}")
        self.logger.info(f"        解2精度: x_width={x2_precision:.6f}, y_width={y2_precision:.6f}")
        
        # X坐标合并：优先选择精度更高的解
        if abs(x1_precision) < self.tol:  # 解1是点解
            if abs(x2_precision) < self.tol:  # 解2也是点解
                # 两个都是点解，取平均
                x_center = (x1_range[0] + x2_range[0]) / 2
                merged_x = (x_center, x_center)
                self.logger.info(f"        X合并: 两个点解平均 = {x_center:.6f}")
            else:
                # 解1是点解，解2是范围解，优先使用点解
                merged_x = x1_range
                self.logger.info(f"        X合并: 保持点解1 = [{x1_range[0]:.6f}, {x1_range[1]:.6f}]")
        elif abs(x2_precision) < self.tol:  # 解2是点解，解1是范围解
            merged_x = x2_range
            self.logger.info(f"        X合并: 保持点解2 = [{x2_range[0]:.6f}, {x2_range[1]:.6f}]")
        else:
            # 两个都是范围解，取交集
            x_min = max(x1_range[0], x2_range[0])
            x_max = min(x1_range[1], x2_range[1])
            if x_min <= x_max + self.tol:
                merged_x = (x_min, x_max)
                self.logger.info(f"        X合并: 范围交集 = [{x_min:.6f}, {x_max:.6f}]")
            else:
                # 无交集，选择精度更高的
                if x1_precision < x2_precision:
                    merged_x = x1_range
                    self.logger.info(f"        X合并: 无交集，选择精度更高的解1")
                else:
                    merged_x = x2_range
                    self.logger.info(f"        X合并: 无交集，选择精度更高的解2")
        
        # Y坐标合并：同样的逻辑
        if abs(y1_precision) < self.tol:  # 解1是点解
            if abs(y2_precision) < self.tol:  # 解2也是点解
                y_center = (y1_range[0] + y2_range[0]) / 2
                merged_y = (y_center, y_center)
                self.logger.info(f"        Y合并: 两个点解平均 = {y_center:.6f}")
            else:
                merged_y = y1_range
                self.logger.info(f"        Y合并: 保持点解1 = [{y1_range[0]:.6f}, {y1_range[1]:.6f}]")
        elif abs(y2_precision) < self.tol:  # 解2是点解，解1是范围解
            merged_y = y2_range
            self.logger.info(f"        Y合并: 保持点解2 = [{y2_range[0]:.6f}, {y2_range[1]:.6f}]")
        else:
            # 两个都是范围解，取交集
            y_min = max(y1_range[0], y2_range[0])
            y_max = min(y1_range[1], y2_range[1])
            if y_min <= y_max + self.tol:
                merged_y = (y_min, y_max)
                self.logger.info(f"        Y合并: 范围交集 = [{y_min:.6f}, {y_max:.6f}]")
            else:
                if y1_precision < y2_precision:
                    merged_y = y1_range
                    self.logger.info(f"        Y合并: 无交集，选择精度更高的解1")
                else:
                    merged_y = y2_range
                    self.logger.info(f"        Y合并: 无交集，选择精度更高的解2")
        
        # 角度合并：使用一致的归一化方法
        phi_diff = abs(phi1 - phi2)
        # 处理角度周期性
        if phi_diff > np.pi:
            phi_diff = 2 * np.pi - phi_diff
        
        if phi_diff <= self.tol:
            # 角度相近，取平均并归一化到[-π, π]
            # 处理跨越±π边界的情况
            if abs(phi1 - phi2) > np.pi:
                if phi1 > phi2:
                    phi2 += 2 * np.pi
                else:
                    phi1 += 2 * np.pi
            
            merged_phi = (phi1 + phi2) / 2
            # 使用统一的归一化方法
            merged_phi = self._normalize_angle(merged_phi)
            self.logger.info(f"        φ合并: 角度相近，取平均 = {merged_phi:.6f}rad")
        else:
            # 角度差异较大，选择来自更精确解的角度
            sol1_precision = x1_precision + y1_precision
            sol2_precision = x2_precision + y2_precision
            if sol1_precision < sol2_precision:
                merged_phi = self._normalize_angle(phi1)
                self.logger.info(f"        φ合并: 选择更精确解1的角度 = {merged_phi:.6f}rad")
            else:
                merged_phi = self._normalize_angle(phi2)
                self.logger.info(f"        φ合并: 选择更精确解2的角度 = {merged_phi:.6f}rad")
        
        return (merged_x, merged_y, merged_phi)

    def _format_solution(self, sol):
        """格式化解的输出，便于阅读"""
        x_range, y_range, phi = sol
        phi_deg = phi * 180 / np.pi
        return (
            f"x:[{x_range[0]:.2f}, {x_range[1]:.2f}], "
            f"y:[{y_range[0]:.2f}, {y_range[1]:.2f}], "
            f"φ:{phi:.2f}rad({phi_deg:.1f}°)"
        )

    def _log_all_solutions_structure(self, all_solutions):
        """详细输出所有解的结构化信息"""
        self.logger.info("=" * 80)
        self.logger.info("所有激光组合解的结构化信息:")
        self.logger.info("=" * 80)
        
        total_solutions = 0
        for combo_idx, laser_solutions in enumerate(all_solutions, 1):
            self.logger.info(f"激光组合 {combo_idx}: 共 {len(laser_solutions)} 个解")
            if laser_solutions:
                for sol_idx, sol in enumerate(laser_solutions, 1):
                    self.logger.info(f"  解{sol_idx}: {self._format_solution(sol)}")
                total_solutions += len(laser_solutions)
            else:
                self.logger.info("  无有效解")
            self.logger.info("-" * 60)
        
        self.logger.info(f"总计: {len(all_solutions)} 个激光组合, {total_solutions} 个解")
        self.logger.info("=" * 80)

    def _filter_intra_group_solutions(self, laser_solutions):
        """
        组内解筛选：对同一激光组合内的解进行相容性检查和合并
        参数:
            laser_solutions: 单个激光组合的解列表
        返回:
            list[合并后的解]: 组内筛选和合并后的解列表
        """
        if not laser_solutions:
            return []
        
        if len(laser_solutions) == 1:
            return laser_solutions
        
        self.logger.info(f"    组内有 {len(laser_solutions)} 个解，开始相容性检查:")
        
        # 对组内解进行相容性检查和合并
        intra_solutions = []  # 格式: [[sol, [相容解列表]], ...]
        
        for sol_idx, current_sol in enumerate(laser_solutions, 1):
            self.logger.info(f"      处理解{sol_idx}: {self._format_solution(current_sol)}")
            found_compatible = False
            
            # 在已有组内解中寻找相容解
            for i in range(len(intra_solutions)):
                existing_sol, compatible_sols = intra_solutions[i]
                if self._is_compatible(existing_sol, current_sol):
                    self.logger.info(f"        ✓ 与组内解{i+1}相容，进行合并")
                    
                    # 记录相容解并重新合并
                    compatible_sols.append(current_sol)
                    merged_sol = existing_sol
                    for comp_sol in compatible_sols:
                        merged_sol = self._merge_compatible_solutions(merged_sol, comp_sol)
                    intra_solutions[i][0] = merged_sol
                    
                    self.logger.info(f"        合并后: {self._format_solution(merged_sol)}")
                    found_compatible = True
                    break
            
            # 如果没有找到相容解，作为新的独立解
            if not found_compatible:
                intra_solutions.append([current_sol, []])
                self.logger.info(f"        → 作为独立解{len(intra_solutions)}")
        
        # 返回合并后的解列表
        result = [sol[0] for sol in intra_solutions]
        self.logger.info(f"    组内筛选结果: {len(result)} 个独立解")
        return result

    def _filter_solutions(self, all_solutions):
        """
        组间相容性检查：不同激光组合的解进行相容性验证和合并
        注意：已移除组内筛选逻辑，因为同一组合内的"相容"解实际表示算法重复
        参数:
            all_solutions: list[激光组合1的解列表, 激光组合2的解列表, ...]
        返回:
            list[最终解] 按相容数量降序排列
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
        
        # 详细输出所有解的结构
        self._log_all_solutions_structure(all_solutions)
        
        self.logger.info("=" * 60)
        self.logger.info("组间相容性检查和合并")
        self.logger.info("注意：已跳过组内筛选，直接进行组间相容性检查")
        self.logger.info("=" * 60)
        
        # 收集所有有效解（按组合标记）
        all_solutions_with_combo = []
        for combo_idx, laser_solutions in enumerate(all_solutions, 1):
            for sol in laser_solutions:
                all_solutions_with_combo.append((sol, combo_idx))
        
        if not all_solutions_with_combo:
            self.logger.info("没有有效解，返回空列表")
            return []
        
        for sol_idx, (current_sol, combo_idx) in enumerate(all_solutions_with_combo):
            self.logger.info(f"处理组合{combo_idx}解{sol_idx+1}: {self._format_solution(current_sol)}")
            found_compatible = False
            
            # 在已有解中寻找相容解
            for i in range(len(solutions)):
                existing_sol, count, compatible_sols = solutions[i]
                if self._is_compatible(existing_sol, current_sol):
                    self.logger.info(f"  ✓ 与已存在解{i+1}相容")
                    self.logger.info(f"    原解: {self._format_solution(existing_sol)}")
                    
                    # 增加相容计数并记录相容解
                    compatible_sols.append(current_sol)
                    solutions[i][1] += 1
                    
                    # 重新计算合并解
                    merged_sol = existing_sol
                    for comp_sol in compatible_sols:
                        merged_sol = self._merge_compatible_solutions(merged_sol, comp_sol)
                    solutions[i][0] = merged_sol
                    
                    self.logger.info(f"    合并后: {self._format_solution(merged_sol)}")
                    self.logger.info(f"    相容计数: {solutions[i][1]}")
                    found_compatible = True
                    break
            
            # 如果没有找到相容解，添加新解
            if not found_compatible:
                solutions.append([current_sol, 1, []])
                self.logger.info(f"  → 添加为新解{len(solutions)}")
        
        # 按相容数量排序
        solutions.sort(key=lambda x: -x[1])
        
        # 详细输出筛选结果
        self.logger.info("=" * 60)
        self.logger.info("最终解列表 (按相容数量降序):")
        
        for i, (sol, count, compatible_list) in enumerate(solutions, 1):
            self.logger.info(f"解{i} (相容数: {count}):")
            self.logger.info(f"  {self._format_solution(sol)}")
            if compatible_list:
                self.logger.info(f"  基于 {len(compatible_list)+1} 个相容解合并而成")
        
        self.logger.info("=" * 60)
        
        # 计算期望的完全相容数量（等于有解的激光组合数量）
        groups_with_solutions = sum(1 for sols in all_solutions if len(sols) > 0)
        
        # 优先返回完全相容的解
        if solutions and solutions[0][1] == groups_with_solutions:
            self.logger.info("🎯 找到完全相容解!")
            perfect_solutions = [sol[0] for sol in solutions[:4] if sol[1] == groups_with_solutions]
            self.logger.info(f"返回 {len(perfect_solutions)} 个完全相容解")
            return perfect_solutions
        
        final_solutions = [sol[0] for sol in solutions[:4]]
        self.logger.info(f"返回前 {len(final_solutions)} 个最佳解")
        return final_solutions


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