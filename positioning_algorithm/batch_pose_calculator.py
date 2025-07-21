"""
批处理位姿计算器
使用numpy向量化和预计算优化的高性能求解器
"""
import numpy as np
import logging
from typing import List, Tuple, Optional
from itertools import combinations

# 导入配置类
class SolverConfig:
    """求解器配置类"""
    def __init__(self, tolerance: float = 1e-3, max_solutions: int = 50):
        self.tolerance = tolerance
        self.max_solutions = max_solutions

class PoseSolver:
    """位姿求解器（与原接口兼容）"""
    
    def __init__(self, 
                 m: float, 
                 n: float, 
                 laser_config: List,
                 tol: float = 1e-3,
                 config: Optional[SolverConfig] = None,
                 ros_logger=None):
        """
        初始化位姿求解器
        
        Args:
            m: 场地宽度
            n: 场地高度
            laser_config: 激光配置列表 [((rel_r, rel_angle), laser_angle), ...]
            tol: 数值容差
            config: 求解器配置
            ros_logger: ROS日志器（可选）
        """
        self.m = m
        self.n = n
        self.laser_config = laser_config
        self.tol = tol
        self.config = config or SolverConfig(tolerance=tol)
        self.ros_logger = ros_logger
        
        # 初始化日志
        self._setup_logging()
        
        # 预计算激光参数（只计算一次）
        self.laser_params = self._precompute_laser_params()
        
        # 导入批处理求解器
        from .batch_solvers import trig_cache, Case1BatchSolver, Case2BatchSolver, Case3BatchSolver
        self.trig_cache = trig_cache
        
        # 判断是否启用ROS日志
        enable_ros_logging = ros_logger is not None
        
        # 创建求解器时传递日志参数
        self.case1_solver = Case1BatchSolver(m, n, self.config.tolerance, 
                                           enable_ros_logging=enable_ros_logging, 
                                           ros_logger=ros_logger)
        self.case2_solver = Case2BatchSolver(m, n, self.config.tolerance)
        self.case3_solver = Case3BatchSolver(m, n, self.config.tolerance)
        
        self.logger.info(f"PoseSolver初始化完成: 场地({m}x{n}), {len(laser_config)}个激光")
    
    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _precompute_laser_params(self) -> np.ndarray:
        """预计算激光参数（初始化时计算一次）"""
        params = []
        for (rel_r, rel_angle), laser_angle in self.laser_config:
            params.append([rel_r, rel_angle, laser_angle])
        
        laser_params = np.array(params)
        self.logger.debug(f"预计算激光参数完成: {laser_params.shape}")
        return laser_params
    
    def solve(self, distances: np.ndarray) -> List[Tuple]:
        """
        求解位姿（与原接口兼容）
        
        Args:
            distances: 激光距离数组 [d0, d1, d2, ...]
            
        Returns:
            解列表 [((x_min, x_max), (y_min, y_max), phi), ...]
        """
        # 验证输入
        if len(distances) != len(self.laser_config):
            raise ValueError(f"距离数组长度({len(distances)})与激光配置数量({len(self.laser_config)})不匹配")
        
        self.logger.info(f"开始求解: 距离={distances}")
        
        # 1. 计算碰撞向量参数
        collision_params = self._calculate_collision_vectors(distances)
        
        # 2. 生成三激光组合
        combinations = self._generate_combinations(collision_params)
        
        # 3. 更新三角函数缓存（只更新变化的角度）
        if len(combinations) > 0:
            self.trig_cache.update_combinations(combinations)
        
        # 4. 创建可扩展的解列表，预分配空间（N_cbn组合的解）
        N_cbn = len(combinations)
        solutions = [[] for _ in range(N_cbn)]  # 最终解列表，每个解为 (x, y, phi)
        case1_solutions = [[] for _ in range(N_cbn)]  # Case1求解器的解
        case3_solutions = [[] for _ in range(N_cbn)]  # Case3求解器的解
        # 5. 使用Case1求解器求解
        #case1_solutions = self.case1_solver.solve(combinations)  # 返回 (N_cbn, 5) 格式
        case3_solutions = self.case3_solver.solve(combinations)
        #"""
        # 6. 按照对应的cbn编号，处理case1_solutions并extend solutions列表
        for cbn_idx in range(N_cbn):
            # 获取当前组合的解
            solutions[cbn_idx].extend(case1_solutions[cbn_idx])
            solutions[cbn_idx].extend(case3_solutions[cbn_idx])
        #"""
        # 7. 未来将启用其他case求解器
        # case2_solutions = self.case2_solver.solve(combinations)  
        
        self.logger.info(f"求解完成，共找到 {len(solutions)} 个解")
        solutions = [sol for sublist in solutions for sol in sublist]  # 扁平化解列表
        return solutions
    
    def _calculate_collision_vectors(self, distances: np.ndarray) -> np.ndarray:
        """计算碰撞向量参数"""
        collision_params = []
        
        for i, distance in enumerate(distances):
            rel_r, rel_angle, laser_angle = self.laser_params[i]
            
            # 计算碰撞点坐标
            x = rel_r * np.cos(rel_angle) + distance * np.cos(laser_angle)
            y = rel_r * np.sin(rel_angle) + distance * np.sin(laser_angle)
            
            # 计算t和theta
            t_val = np.linalg.norm([x, y])
            theta_val = np.arctan2(y, x)
            if theta_val < 0:
                theta_val += 2 * np.pi
                
            collision_params.append([t_val, theta_val])
            
            self.logger.debug(f"激光{i}: t={t_val:.6f}, theta={theta_val:.6f}")
        
        return np.array(collision_params)
    
    def _generate_combinations(self, collision_params: np.ndarray) -> np.ndarray:
        """生成三激光组合"""
        if len(collision_params) < 3:
            self.logger.warning(f"激光数量不足({len(collision_params)})，无法生成三激光组合")
            return np.array([]).reshape(0, 3, 2)
        
        combos = []
        for indices in combinations(range(len(collision_params)), 3):
            combo = collision_params[list(indices)]
            combos.append(combo)
        
        if not combos:
            return np.array([]).reshape(0, 3, 2)
            
        combinations_array = np.array(combos)
        self.logger.debug(f"生成 {len(combinations_array)} 个三激光组合")
        return combinations_array
    
    def _precompute_trigonometry_batch(self, combinations: np.ndarray) -> dict:
        """
        预计算三角函数值 - 保留接口但委托给全局缓存
        这个方法保留是为了兼容性，实际逻辑在TrigonometryCache中
        """
        self.logger.debug("预计算三角函数（委托给全局缓存）")
        self.trig_cache.update_combinations(combinations)
        return self.trig_cache.get_cache_info()

# 保持向后兼容的别名
BatchPoseCalculator = PoseSolver
