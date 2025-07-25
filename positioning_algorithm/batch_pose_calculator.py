"""
批处理位姿计算器
使用numpy向量化和预计算优化的高性能求解器
"""
import numpy as np
import logging
from typing import List, Tuple, Optional
from itertools import combinations
from . import ros_logger

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
                 Rcl_logger = None):
        """
        初始化位姿求解器
        
        Args:
            m: 场地宽度
            n: 场地高度
            laser_config: 激光配置列表 [((rel_r, rel_angle), laser_angle), ...]
            tol: 数值容差
            config: 求解器配置
            Rcl_logger: ROS节点的Logger对象（可选）
        """
        self.m = m
        self.n = n
        self.laser_config = laser_config
        self.tol = tol
        self.config = config or SolverConfig(tolerance=tol)
        self.ros_logger = Rcl_logger
        self.solver_name = "PoseSolver"
        
        # 初始化日志
        if self.ros_logger is None:
            self._setup_logging()
        
        # 预计算激光参数（只计算一次）
        self.laser_params = self._precompute_laser_params()
        
        # 导入批处理求解器
        from .batch_solvers import Case1BatchSolver, Case3BatchSolver
        
        # 创建求解器时传递ROS logger
        self.case1_solver = Case1BatchSolver(m, n, self.config.tolerance, 
                                           ros_logger=self.ros_logger)
        self.case3_solver = Case3BatchSolver(m, n, self.config.tolerance, ros_logger=self.ros_logger)
        
        self._log_info(f"PoseSolver初始化完成: 场地({m}x{n}), {len(laser_config)}个激光")
    
    # ==================== 统一日志封装函数 ====================
    def _log_debug(self, message: str):
        """调试级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.get_logger().debug(f"[{self.solver_name}] {message}")
        else:
            self.logger.debug(message)
    
    def _log_info(self, message: str):
        """信息级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.get_logger().info(f"[{self.solver_name}] {message}")
        else:
            self.logger.info(message)
    
    def _log_warning(self, message: str):
        """警告级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.get_logger().warn(f"[{self.solver_name}] {message}")
        else:
            self.logger.warning(message)
    
    def _log_error(self, message: str):
        """错误级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.get_logger().error(f"[{self.solver_name}] {message}")
        else:
            self.logger.error(message)
    
    def _setup_logging(self):
        """设置日志系统"""
        import os
        
        # 创建logs目录
        os.makedirs('logs', exist_ok=True)
        
        self.logger = logging.getLogger("PoseSolverFilter")
        
        # 清理已有的handlers，避免重复添加
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
        
        # 添加文件日志器
        handler = logging.FileHandler("logs/pose_solver_filter.log", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以显示详细数组日志
    
    def _precompute_laser_params(self) -> np.ndarray:
        """预计算激光参数（初始化时计算一次）"""
        params = []
        for (rel_r, rel_angle), laser_angle in self.laser_config:
            params.append([rel_r, rel_angle, laser_angle])
        
        laser_params = np.array(params)
        self._log_debug(f"预计算激光参数完成: {laser_params.shape}")
        return laser_params
    
    def solve(self, distances: np.ndarray) -> List[Tuple]:
        """
        求解位姿（与原接口兼容）
        
        Args:
            distances: 激光距离数组 [d0, d1, d2, ...]
            
        Returns:
            解列表 [((x_min, x_max), (y_min, y_max), phi), ...]
        """
        if len(distances) < 3:
            return []  # 至少需要3个激光距离才能进行求解
        # 验证输入
        if len(distances) != len(self.laser_config):
            raise ValueError(f"距离数组长度({len(distances)})与激光配置数量({len(self.laser_config)})不匹配")
        
        # self.logger.info(f"开始求解: 距离={distances}")
        
        """
        实际使用的时候，我们发现有些激光头坏了
        这个时候，它返回的值就是0
        所以我们要添加掩码，在计算碰撞向量参数的时候，就把这些坏掉的激光头给删了
        """

        # 0.5 可用激光——掩码：可用为1 不可用为0
        valid_mask = distances > 0.5  # 可用激光的掩码
        
        # 1. 计算碰撞向量参数
        collision_params = self._calculate_collision_vectors(distances, valid_mask)
        
        # 2. 生成三激光组合
        combinations = self._generate_combinations(collision_params)
        
        # 3. 更新三角函数缓存（只更新变化的角度）
        if len(combinations) > 0:
            self.trig_cache.update_combinations(combinations)
        
        # 4. 创建可扩展的解列表，预分配空间（N_cbn组合的解）
        N_cbn = len(combinations)
        solutions = np.full((N_cbn, 36, 5), np.inf, dtype=np.float64)  # 36个解，每个解5个参数
        # self.logger.debug(f"预分配解空间: {solutions.shape}")
        valid_solutions = np.zeros((N_cbn, 36), dtype=bool)  # 有效性标志
        final_sol_1 = np.full((N_cbn, 12, 5), np.inf, dtype=np.float64)
        valid_1 = np.zeros((N_cbn, 12), dtype=bool)  # 有效性标志
        final_sol_3 = np.full((N_cbn, 24, 5), np.inf, dtype=np.float64)
        valid_3 = np.zeros((N_cbn, 24), dtype=bool)  # 有效性标志
        # 5. 使用Case1和case3求解器求解
        final_sol_1, valid_1 = self.case1_solver.solve(combinations)  # 返回 (N_cbn, 5) 格式
        final_sol_3, valid_3 = self.case3_solver.solve(combinations)  # 返回 (N_cbn, 5) 格式
       
        # 每12个case1 和 每18个case3解 把他们合并进 solutions
        solutions[:,:12,:] = final_sol_1
        valid_solutions[:,:12] = valid_1
        solutions[:,12:,:] = final_sol_3
        valid_solutions[:,12:] = valid_3

        # 5.5. 角度归一化处理：将所有phi角度归一化到[-π, π]范围
        self._normalize_angles_batch(solutions, valid_solutions)

        # 6. 调用筛选器
        filtered_solutions = self.filter_solutions(solutions, valid_solutions)
            
        # 7. 把解转换为原接口格式
        results = []

        for sol in filtered_solutions:
            x_min, x_max, y_min, y_max, phi = sol[:5]
            results.append(((x_min, x_max), (y_min, y_max), phi))
        
        return results
    
    def _calculate_collision_vectors(self, distances: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """计算碰撞向量参数"""
        valid_sum = np.sum(valid_mask)
        if valid_sum == 0:
            self._log_warning("没有有效的激光距离，返回空数组")
            return np.array([]).reshape(0, 2)
        valid_indice = np.where(valid_mask)[0]
        collision_params = np.zeros((valid_sum, 2), dtype=np.float64)
        #计算激光碰撞向量参数 t——距离 theta——角度
        x = self.laser_params[valid_indice, 0] * np.cos(self.laser_params[valid_indice, 1]) + distances[valid_indice] * np.cos(self.laser_params[valid_indice, 2])
        y = self.laser_params[valid_indice, 0] * np.sin(self.laser_params[valid_indice, 1]) + distances[valid_indice] * np.sin(self.laser_params[valid_indice, 2])
        collision_params[:, 0] = np.linalg.norm([x, y], axis=0)  # 计算距离 t
        # 归一化角度到[-π, π]范围
        collision_params[:, 1] = np.arctan2(y, x)  # 计算角度 theta
        collision_params[:, 1] = np.where(collision_params[:, 1] > np.pi, collision_params[:, 1] - 2 * np.pi, collision_params[:, 1])
        
        """
        原本的计算激光参量的算法太慢了，我都用np数组了
        直接批量计算
        """
        

        """
        for i, distance in enumerate(distances):
            rel_r, rel_angle, laser_angle = self.laser_params[i]
            
            # 计算碰撞点坐标
            x = rel_r * np.cos(rel_angle) + distance * np.cos(laser_angle)
            y = rel_r * np.sin(rel_angle) + distance * np.sin(laser_angle)
            
            # 计算t和theta
            t_val = np.linalg.norm([x, y])
            theta_val = np.arctan2(y, x)
            
            # 归一化角度到[-π, π]范围
            if theta_val > np.pi:
                theta_val -= 2 * np.pi
            elif theta_val <= -np.pi:
                theta_val += 2 * np.pi
                
            collision_params.append([t_val, theta_val])
            
            # self.logger.debug(f"激光{i}: t={t_val:.6f}, theta={theta_val:.6f}")
        """
        
        return collision_params
    
    def _generate_combinations(self, collision_params: np.ndarray) -> np.ndarray:
        """生成三激光组合"""
        if len(collision_params) < 3:
            self._log_warning(f"激光数量不足({len(collision_params)})，无法生成三激光组合")
            return np.array([]).reshape(0, 3, 2)
        
        combos = []
        for indices in combinations(range(len(collision_params)), 3):
            combo = collision_params[list(indices)]
            combos.append(combo)
        
        if not combos:
            return np.array([]).reshape(0, 3, 2)
            
        combinations_array = np.array(combos)
        # self.logger.debug(f"生成 {len(combinations_array)} 个三激光组合")
        return combinations_array
    
    def _precompute_trigonometry_batch(self, combinations: np.ndarray) -> dict:
        """
        预计算三角函数值 - 保留接口但委托给全局缓存
        这个方法保留是为了兼容性，实际逻辑在TrigonometryCache中
        """
        # self.logger.debug("预计算三角函数（委托给全局缓存）")
        self.trig_cache.update_combinations(combinations)
        return self.trig_cache.get_cache_info()

    def _normalize_angles_batch(self, solutions: np.ndarray, valid_mask: np.ndarray):
        """
        批量角度归一化：将所有有效解的phi角度归一化到[-π, π]范围
        
        Args:
            solutions: 解数组 (N, 36, 5) - 第5列为phi角度
            valid_mask: 有效性掩码 (N, 36)
        """
        # 只对有效解进行角度归一化
        phi_angles = solutions[:, :, 4]  # 提取所有phi角度 (N, 36)
        
        # 归一化到[-π, π]范围
        phi_normalized = np.where(
            phi_angles > np.pi, 
            phi_angles - 2*np.pi, 
            phi_angles
        )
        phi_normalized = np.where(
            phi_normalized <= -np.pi,
            phi_normalized + 2*np.pi,
            phi_normalized
        )
        
        # 只更新有效解的角度
        solutions[:, :, 4] = np.where(valid_mask, phi_normalized, solutions[:, :, 4])
        
        # 统计归一化信息
        total_valid = np.sum(valid_mask)
        if total_valid > 0:
            self._log_debug(f"角度归一化完成: 处理了{total_valid}个有效解的phi角度")

    # 筛选器：处理解的相容性融合
    def filter_solutions(self, solutions: np.ndarray, valid_mask: np.ndarray) -> List[Tuple]:
        """
        筛选和融合相容的解
        
        Args:
            solutions: 解数组 (N, 36, 5) - 每个解格式为[xmin, xmax, ymin, ymax, phi]
            valid_mask: 有效性掩码 (N, 36)

        Returns:
            筛选后的解列表
        """
        # self.logger.info(f"开始筛选解: 输入形状{solutions.shape}, 有效掩码形状{valid_mask.shape}")
        
        N = solutions.shape[0]
        tolerance = self.config.tolerance
        
        # 预定义筛选结果存储空间：最多36*N个解，每个解6个参数[xmin, xmax, ymin, ymax, phi, compatible_count]
        filtered_solutions = np.full((36*N, 6), np.inf, dtype=np.float64)
        filtered_count = 0  # 当前已筛选解的数量
        
        # 逐个处理每个组合
        for i in range(N):
            # self.logger.info(f"=" * 80)
            # self.logger.info(f"处理第{i}个组合")
            # self.logger.info(f"=" * 80)
            
            # 获取第i个组合中的有效解
            current_valid = valid_mask[i]
            current_solutions = solutions[i]
            
            if not np.any(current_valid):
                # self.logger.info(f"第{i}个组合没有有效解，跳过")
                continue  # 跳过没有有效解的组合
            
            # 输出当前组合的有效解信息
            valid_count = np.sum(current_valid)
            self._log_info(f"第{i}个组合有{valid_count}个有效解")
            # self._log_array_detailed(f"current_valid", current_valid)
            # self._log_array_detailed(f"current_solutions", current_solutions)
            
            # 提取有效解的索引和数据
            valid_indices = np.where(current_valid)[0]
            valid_solutions = current_solutions[valid_indices]  # (n_valid, 5)
            remaining_mask = current_valid.copy()  # 剩余解的掩码
            
            # self.logger.info(f"开始与已有{filtered_count}个筛选解进行相容性检查")

            # 突然发现，如果在遍历filtered_solutions时，直接修改filtered_solutions会导致索引错乱
            # 所以我们需要一个临时变量来存储当前组合的筛选解
            # 不是，我突然发现，这个临时变量还必须要包含原有的，不然就没办法实现"插入"的功能了
            # 你要是想添加在后面，也不是不行，但这样就慢了
            temp_filtered_solutions = filtered_solutions.copy()  # 复制当前筛选结果
            current_index = 0
            
            # 与已有筛选解进行相容性检查
            for j in range(filtered_count):
                # self.logger.info(f"-" * 60)
                # self.logger.info(f"检查与第{j}个已有解的相容性")
                
                existing_sol = filtered_solutions[j]  # 已有解
                # self.logger.info(f"已有解 filtered_solutions[{j}]: {existing_sol}")
                
                # 批量计算相容性关键量
                # 计算交集边界
                key0 = np.minimum(current_solutions[:, 1], existing_sol[1])  # min(xmax1, xmax2)
                key1 = np.maximum(current_solutions[:, 0], existing_sol[0])  # max(xmin1, xmin2)
                key2 = np.minimum(current_solutions[:, 3], existing_sol[3])  # min(ymax1, ymax2)  
                key3 = np.maximum(current_solutions[:, 2], existing_sol[2])  # max(ymin1, ymin2)
                key4 = current_solutions[:, 4]  # phi1
                # phi2只有一个，但为了方便批处理，我们要把它弄成一个数组
                key5 = np.full(current_solutions.shape[0], existing_sol[4])  # phi2
                
                # 结构化输出关键量
                # self.logger.info(f"计算得到的关键量:")
                # self._log_array_detailed(f"key0 (min_xmax)", key0)
                # self._log_array_detailed(f"key1 (max_xmin)", key1)
                # self._log_array_detailed(f"key2 (min_ymax)", key2)
                # self._log_array_detailed(f"key3 (max_ymin)", key3)
                # self._log_array_detailed(f"key4 (phi1)", key4)
                # self._log_array_detailed(f"key5 (phi2)", key5)
                
                # 计算相容性掩码
                # 计算角度差值，考虑跨越±π边界的情况
                angle_diff = np.abs(key4 - key5)
                angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)  # 取较小的角度差
                
                compatible_mask = (
                    (key0 >= key1) &  # x方向有交集
                    (key2 >= key3) &  # y方向有交集
                    (angle_diff <= tolerance) &  # phi差值在容差内（考虑周期性）
                    valid_mask[i]  # 仅考虑当前组合的有效解
                )
                
                # self._log_array_detailed(f"compatible_mask", compatible_mask)
                
                compatible_count = np.sum(compatible_mask)
                # self.logger.info(f"找到{compatible_count}个相容解")
                
                if compatible_count > 0:
                    # 找到相容解，进行融合
                    compatible_indices = np.where(compatible_mask)[0]
                    # self.logger.info(f"相容解索引: {compatible_indices}")
                    
                    new_sol = np.zeros((compatible_count, 6), dtype=np.float64)
                    new_sol[:, 0] = key1[compatible_indices]  # 修正：应该是key1(max_xmin)
                    new_sol[:, 1] = key0[compatible_indices]  # 修正：应该是key0(min_xmax)
                    new_sol[:, 2] = key3[compatible_indices]  # 修正：应该是key3(max_ymin)
                    new_sol[:, 3] = key2[compatible_indices]  # 修正：应该是key2(min_ymax)
                    
                    # 角度平均，考虑跨越±π边界的情况
                    phi1 = key4[compatible_indices]
                    phi2 = key5[compatible_indices]
                    # 使用复数表示法计算角度平均
                    avg_complex = (np.exp(1j * phi1) + np.exp(1j * phi2)) / 2
                    avg_phi = np.angle(avg_complex)  # 自动归一化到[-π, π]
                    new_sol[:, 4] = avg_phi
                    
                    new_sol[:, 5] = existing_sol[5] + 1  # 相容数量+1
                    
                    # 结构化输出融合后的解
                    # self.logger.info(f"融合后的新解:")
                    # self._log_array_detailed(f"new_sol", new_sol)
                    
                    # 不要忘记先保存后面的解
                    temp = filtered_solutions[j+1: filtered_count, :6].copy()

                    # 将新解插入到临时筛选结果 current_index: current_index+1 处
                    temp_filtered_solutions[current_index:current_index + compatible_count, :6] = new_sol
                    current_index += compatible_count
                    
                    # 将之前暂存的解添加到后面
                    temp_filtered_solutions[current_index:current_index + temp.shape[0], :6] = temp

                    # 更新剩余解掩码
                    remaining_mask[compatible_indices] = False
                    # self.logger.info(f"更新剩余解掩码，剩余{np.sum(remaining_mask)}个解")
                else:
                    # self.logger.info(f"与第{j}个解没有相容性")
                    current_index += 1  # 保持current_index位置不变，跳过当前解

            # self.logger.info(f"-" * 60)
            # self.logger.info(f"完成第{i}个组合与所有已有解的相容性检查")
            
            # 将临时筛选结果覆盖到主筛选结果中
            filtered_solutions[:current_index, :6] = temp_filtered_solutions[:current_index, :6]
            filtered_count = current_index

            # 将剩余的未相容解添加到筛选结果中
            if np.any(remaining_mask):
                remaining_count = np.sum(remaining_mask)
                # self.logger.info(f"添加{remaining_count}个未相容的独立解")
                
                # 添加到筛选结果中
                filtered_solutions[filtered_count:filtered_count + remaining_count, :5] = current_solutions[remaining_mask, :5]
                filtered_solutions[filtered_count:filtered_count + remaining_count, 5] = 1
                filtered_count += remaining_count
                
                # self.logger.info(f"添加独立解后，总筛选解数量: {filtered_count}")
            # else:
                # self.logger.info(f"第{i}个组合的所有解都已融合，无独立解")
            
            # 输出处理完第i个组合后的完整筛选结果
            self._log_info(f"处理完第{i}个组合后的筛选结果:")
            current_filtered = filtered_solutions[:filtered_count]
            # self._log_array_detailed(f"filtered_solutions[:{filtered_count}]", current_filtered)
        
        # 提取有效的筛选结果
        final_filtered = filtered_solutions[:filtered_count]
        if final_filtered.size == 0:
            # self.logger.warning("没有有效的筛选结果")
            return []
        
        # self.logger.info(f"=" * 80)
        # self.logger.info(f"开始最终排序和筛选")
        # self.logger.info(f"=" * 80)
        
        # 我们根据相容组合数量进行排序，方向为降序
        sort_indices = np.argsort(final_filtered[:, 5])[::-1]
        final_filtered = final_filtered[sort_indices]
        
        self._log_info(f"排序后的最终筛选结果:")
        # self._log_array_detailed(f"final_filtered (sorted)", final_filtered)

        # 如果最高相容组合数等于组合数N 则返回相容组合数等于N的解
        max_compatible_count = final_filtered[0, 5]
        # self._log_info(f"最高相容组合数: {max_compatible_count}, 总组合数: {N}")
        
        if max_compatible_count == N:
            perfect_solutions = final_filtered[final_filtered[:, 5] == N, :5]
            self._log_info(f"找到{len(perfect_solutions)}个完美相容解 (相容数=N)")
            # self._log_array_detailed(f"perfect_solutions", perfect_solutions)
            return perfect_solutions

        # 否则返回前4个解
        top_solutions = final_filtered[:4, :5]
        self._log_info(f"返回前4个最佳解:")
        # self._log_array_detailed(f"top_solutions", top_solutions)
        return top_solutions
    
    # 筛选器（暂时先把sol(N, 36, 5) 弄成 flatten_sol (36*N, 5), 掩码同理，之后逻辑会补上的）
    def return_flatten_solutions(self, solutions: np.ndarray, valid_mask: np.ndarray) -> List[Tuple]:
        """
        筛选解（与原接口兼容）
        
        Args:
            solutions: 解数组 (N, 36, 5)
            valid_mask: 有效性掩码 (N, 36)

        Returns:
            筛选后的解列表
        """
        self._log_info(f"开始筛选解: {solutions}, 有效掩码: {valid_mask}")
        
        # 将解数组展平为 (N*36, 5)
        flat_solutions = solutions.reshape(-1, 5)
        flat_mask = valid_mask.reshape(-1)
        
        # 筛选有效解
        filtered_solutions = flat_solutions[flat_mask]
        
        self._log_info(f"筛选完成，共找到 {flat_solutions} 个有效解， 遍历所有元素{filtered_solutions}")
        return filtered_solutions

        
    
    def _log_array_detailed(self, name: str, arr):
        """详细结构化输出数组的每个元素 - 使用文件日志"""
        # 性能优化：注释掉所有详细日志输出
        pass
        # if isinstance(arr, list):
        #     if len(arr) == 0:
        #         self.logger.debug(f"{name}: 空列表")
        #         return
        #     
        #     self.logger.debug(f"{name} 详细内容: 列表长度: {len(arr)}")
        #     for i, item in enumerate(arr):
        #         self.logger.debug(f"{name}[{i}]: {item}")
        # 
        # elif isinstance(arr, np.ndarray):
        #     if arr.size == 0:
        #         self.logger.debug(f"{name}: 空数组")
        #         return
        #     
        #     self.logger.debug(f"{name} 详细内容: 形状: {arr.shape}, 类型: {arr.dtype.name}")
        #     
        #     if arr.ndim == 1:
        #         # 一维数组：逐个显示
        #         for i in range(len(arr)):
        #             self.logger.debug(f"{name}[{i}]: {arr[i]}")
        #     
        #     elif arr.ndim == 2:
        #         # 二维数组：按行显示
        #         for i in range(arr.shape[0]):
        #             self.logger.debug(f"{name}[{i}]: {arr[i]}")
        #     
        #     elif arr.ndim == 3:
        #         # 三维数组：分层显示
        #         for i in range(arr.shape[0]):
        #             self.logger.debug(f"{name}[{i}] 形状{arr[i].shape}:")
        #             for j in range(arr.shape[1]):
        #                 self.logger.debug(f"  {name}[{i}][{j}]: {arr[i][j]}")
        #     
        #     else:
        #         # 更高维数组：显示基本信息和前几个元素
        #         self.logger.debug(f"{name}: 高维数组 {arr.shape}，显示前5个元素:")
        #         flat_arr = arr.flatten()
        #         for i in range(min(5, len(flat_arr))):
        #             self.logger.debug(f"{name}.flat[{i}]: {flat_arr[i]}")
        # 
        # else:
        #     self.logger.debug(f"{name}: 类型: {type(arr)}, 内容: {arr}")

    def _log_array_structured(self, name: str, arr):
        """结构化输出数组内容 - 保持向后兼容的别名"""
        # self._log_array_detailed(name, arr)
        pass

# 保持向后兼容的别名
BatchPoseCalculator = PoseSolver
