import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations
from .batch_solvers.Base_log import BaseLog
import logging

class PoseSolver(BaseLog):
    def __init__(self, m, n, laser_configs, Rcl_logger=None, tol=0.15, use_tolerance_checking=True):
        super().__init__()
        self.m = m
        self.n = n
        # laser_configs: List[Tuple[Tuple[float, float], float]]
        self.laser_configs = laser_configs
        self.lasers = self._calculate_lasers(laser_configs)
        self.lasers_theta = np.array([config[1] for config in laser_configs])  # 提取角度
        self.ros_logger = Rcl_logger
        self.tol = tol  # 边界容差参数
        self.use_tolerance_checking = use_tolerance_checking  # 是否使用容差检查
        self.angle_tol = 3 / 180 * np.pi  # 角度容差转换为弧度
        self.solver_name = "PoseSolver_angle"
        # 初始化日志
        if self.ros_logger is None:
            self._setup_logging(self.solver_name)

    def _setup_logging(self, solver_name: str):
        """设置日志系统"""
        import os
        
        # 创建logs目录
        os.makedirs('logs', exist_ok=True)
        
        self.logger = logging.getLogger(f"pose_solver_{solver_name}")
        
        # 清理已有的handlers，避免重复添加
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
        
        # 添加文件日志器
        handler = logging.FileHandler(f"logs/pose_solver_{solver_name}.log", encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以显示详细数组日志
    def _calculate_lasers(self, laser_configs):
        """计算激光束的极坐标和角度"""
        lasers = np.zeros((len(laser_configs), 2))
        print(laser_configs)
        
        # 正确提取r和theta值
        for i, config in enumerate(laser_configs):
            r, theta = config[0]  # config[0] 是 (r, theta) 元组
            lasers[i, 0] = r * np.cos(theta)
            lasers[i, 1] = r * np.sin(theta)
        
        return lasers

    def solve(self, distances, angle):
        try:
        # 验证输入
            if len(distances) != len(self.laser_configs):
                raise ValueError(f"距离数组长度({len(distances)})与激光配置数量({len(self.laser_configs)})不匹配")
        except TypeError as e:
            self._log_error(f"输入类型错误: {e}")
            return []    
        vtc = np.sqrt(self.m**2 + self.n**2)
        # 计算有效激光距离掩码
        valid_mask = (distances > self.tol**3) & (distances < vtc + self.tol)  # 可用激光的掩码
        # 1. 计算碰撞多边形
        collision_vectors, thetas = self._calculate_collision_vectors(distances, valid_mask, angle)  # shape(N_va, 2), shape(N_va,)
        self._log_array_detailed("collision_vectors", collision_vectors)
        self._log_array_detailed("thetas", thetas)
        N_va = np.sum(valid_mask)  # 有效激光束数量
        if N_va < 1:
            self._log_warning("没有有效的激光束，无法计算位姿")
            return []
        # 2. 根据碰撞多边形，计算x和y的候选值
        candidates = self._calculate_candidates(collision_vectors, thetas) # shape(N_va, 2)
        results = np.array([]).reshape(0, 2)
        have_res = False

        # 其实是这样的，我have_res 如果为true了，我就希望跨过下面所有，直接到最后_calculate_final_results

        # 3. calculate_differences_n
        results_n = self._calculate_differences_n(candidates, N_va) 
        self._log_array_detailed("candidates", candidates)
        self._log_array_detailed("results_n", results_n)
        excluded_lasers = None
        if len(results_n) > 0:
            results = np.vstack([results, results_n]) if len(results) > 0 else results_n
            excluded_lasers = np.array([]).reshape(len(results_n), 0)  # n情况下没有被排除的激光
            have_res = True
        if N_va < 2:
            self._log_warning("前面的计算没有结果，且有效激光束数量小于2")
            if not have_res:
                return []
        # 4. calculate_differences_n_1
        if not have_res:
            results_n1, excluded_n1 = self._calculate_differences_n_1(candidates, N_va)
            self._log_array_detailed("results_n1", results_n1)
            if len(results_n1) > 0:
                results = np.vstack([results, results_n1]) if len(results) > 0 else results_n1
                excluded_lasers = excluded_n1.reshape(-1, 1)  # 转换为列向量
                have_res = True
            if N_va < 3:
                self._log_warning("前面的计算没有结果，且有效激光束数量小于3")
                if not have_res:
                    return []
        # 5. calculate_differences_n_2
        if not have_res:
            results_n2, excluded_n2 = self._calculate_differences_n_2(candidates, N_va)
            self._log_array_detailed("results_n2", results_n2)

            if len(results_n2) > 0:
                results = np.vstack([results, results_n2]) if len(results) > 0 else results_n2
                excluded_lasers = excluded_n2  # 已经是正确的形状 (n_results, 2)
                have_res = True
        
        # 6. 碰撞多边形计算筛选
        if not have_res:
            self._log_warning("前面的计算没有结果")
            return []
        
        final_results = self._calculate_final_results(results, collision_vectors, excluded_lasers)

        # 7. 返回结果
        return final_results if len(final_results) > 0 else results
    
    def _calculate_collision_vectors(self, distances: np.ndarray, valid_mask: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算碰撞向量参数"""
        # 计算有效激光束的碰撞向量和角度
        collision_vectors_robot_base = np.zeros((np.sum(valid_mask), 2))
        collision_vectors_robot_base[:, 0] = distances[valid_mask] * np.cos(self.lasers_theta[valid_mask]) + self.lasers[valid_mask, 0]
        collision_vectors_robot_base[:, 1] = distances[valid_mask] * np.sin(self.lasers_theta[valid_mask]) + self.lasers[valid_mask, 1]
        # theta 计算范围是[-np.pi, np.pi]
        thetas = np.arctan2(collision_vectors_robot_base[:, 1], collision_vectors_robot_base[:, 0])
        r = np.linalg.norm(collision_vectors_robot_base, axis=1)
        collision_vectors = np.zeros((np.sum(valid_mask), 2))
        collision_vectors[:, 0] = r * np.cos(angle + thetas)
        collision_vectors[:, 1] = r * np.sin(angle + thetas)
        # 确保角度在[-pi, pi]范围内
        thetas = np.arctan2(collision_vectors[:, 1], collision_vectors[:, 0])
        # 确保角度在[-pi, pi]范围内
        return collision_vectors, thetas
    
    def _calculate_candidates(self, collision_vectors: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """计算候选点"""
        # 计算碰撞多边形的候选点
        candidates = np.zeros((len(collision_vectors), 2))
        
        # 1. 激光射在左边 (thetas < -np.pi/2 + self.angle_tol | thetas > np.pi/2 - self.angle_tol)
        left_mask = (thetas < -np.pi/2 + self.angle_tol) | (thetas > np.pi/2 - self.angle_tol)
        candidates[left_mask, 0] = 0 - collision_vectors[left_mask, 0]
        candidates[left_mask, 1] = np.inf
        
        # 2. 激光射在右边 (-self.angle_tol < thetas < self.angle_tol)
        right_mask = (-self.angle_tol < thetas) & (thetas < self.angle_tol)
        candidates[right_mask, 0] = self.m - collision_vectors[right_mask, 0]
        candidates[right_mask, 1] = np.inf
        
        # 3. 激光射在上边 (np.pi/2 - self.angle_tol < thetas < np.pi/2 + self.angle_tol)
        top_mask = (np.pi/2 - self.angle_tol < thetas) & (thetas < np.pi/2 + self.angle_tol)
        candidates[top_mask, 0] = np.inf
        candidates[top_mask, 1] = self.n - collision_vectors[top_mask, 1]
        
        # 4. 激光射在下边 (-np.pi/2 - self.angle_tol < thetas < -np.pi/2 + self.angle_tol)
        bottom_mask = (-np.pi/2 - self.angle_tol < thetas) & (thetas < -np.pi/2 + self.angle_tol)
        candidates[bottom_mask, 0] = np.inf
        candidates[bottom_mask, 1] = 0 - collision_vectors[bottom_mask, 1]
        
        # 5. 激光射在左上角 (np.pi/2 + self.angle_tol < thetas < np.pi - self.angle_tol)
        left_top_mask = (np.pi/2 + self.angle_tol < thetas) & (thetas < np.pi - self.angle_tol)
        candidates[left_top_mask, 0] = 0 - collision_vectors[left_top_mask, 0]
        candidates[left_top_mask, 1] = self.n - collision_vectors[left_top_mask, 1]
        
        # 6. 激光射在右上角 (self.angle_tol < thetas < np.pi/2 - self.angle_tol)
        right_top_mask = (self.angle_tol < thetas) & (thetas < np.pi/2 - self.angle_tol)
        candidates[right_top_mask, 0] = self.m - collision_vectors[right_top_mask, 0]
        candidates[right_top_mask, 1] = self.n - collision_vectors[right_top_mask, 1]
        
        # 7. 激光射在右下角 (-np.pi/2 + self.angle_tol < thetas < -self.angle_tol)
        right_bottom_mask = (-np.pi/2 + self.angle_tol < thetas) & (thetas < -self.angle_tol)
        candidates[right_bottom_mask, 0] = self.m - collision_vectors[right_bottom_mask, 0]
        candidates[right_bottom_mask, 1] = 0 - collision_vectors[right_bottom_mask, 1]
        
        # 8. 激光射在左下角 (-np.pi + self.angle_tol < thetas < -np.pi/2 - self.angle_tol)
        left_bottom_mask = (-np.pi + self.angle_tol < thetas) & (thetas < -np.pi/2 - self.angle_tol)
        candidates[left_bottom_mask, 0] = 0 - collision_vectors[left_bottom_mask, 0]
        candidates[left_bottom_mask, 1] = 0 - collision_vectors[left_bottom_mask, 1]
        
        return candidates
    
    def _calculate_differences_n(self, candidates: np.ndarray, N_va: int) -> np.ndarray:
        """计算n个激光束的diff_x, diff_y"""
        
        # 检查输入有效性
        if N_va == 0 or len(candidates) == 0:
            return np.array([]).reshape(0, 2)
            
        # 生成所有可能的掩码 (2^N_va种组合)
        if N_va > 20:  # 防止掩码数量过大
            self._log_warning(f"激光束数量({N_va})太大，跳过n组合计算")
            return np.array([]).reshape(0, 2)
            
        masks = np.array([(i >> np.arange(N_va)) & 1 for i in range(2**N_va)], dtype=np.int8)

        # 根据掩码选择x和y (无效位置设为NaN)
        selected_x = np.where(masks==0, candidates[..., 0], np.nan)
        selected_y = np.where(masks==1, candidates[..., 1], np.nan)
        
        # 计算max和min (忽略NaN)
        max_x = np.nanmax(selected_x, axis=1)
        min_x = np.nanmin(selected_x, axis=1)
        max_y = np.nanmax(selected_y, axis=1)
        min_y = np.nanmin(selected_y, axis=1)
        
        # 计算差值
        diffs_x = max_x - min_x
        diffs_y = max_y - min_y
        
        # 寻找满足容差条件的组合
        valid_mask = (diffs_x < self.tol) & (diffs_y < self.tol) & ~np.isnan(diffs_x) & ~np.isnan(diffs_y)
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            # 批量计算平均值
            valid_selected_x = selected_x[valid_indices]  # shape: (n_valid, N_va)
            valid_selected_y = selected_y[valid_indices]  # shape: (n_valid, N_va)
            
            # 计算每行的平均值，忽略NaN
            mean_x = np.nanmean(valid_selected_x, axis=1)
            mean_y = np.nanmean(valid_selected_y, axis=1)
            
            # 检查是否有有效值
            has_valid_x = ~np.isnan(mean_x)
            has_valid_y = ~np.isnan(mean_y)
            final_valid = has_valid_x & has_valid_y
            
            if np.any(final_valid):
                return np.column_stack((mean_x[final_valid], mean_y[final_valid]))
        
        return np.array([]).reshape(0, 2)
    
    def _calculate_differences_n_1(self, candidates: np.ndarray, N_va: int) -> Tuple[np.ndarray, np.ndarray]:
        """计算n-1个激光束的diff_x, diff_y，返回结果和被删除的激光束索引"""
        
        # 检查输入有效性
        if N_va <= 1 or len(candidates) == 0:
            return np.array([]).reshape(0, 2), np.array([], dtype=int)
            
        mask_bits = N_va - 1
        if mask_bits > 20:  # 防止掩码数量过大
            self._log_warning(f"激光束数量({N_va})太大，跳过n-1组合计算")
            return np.array([]).reshape(0, 2), np.array([], dtype=int)
        
        # 生成所有删除1列的组合
        arr_del1 = np.stack([np.delete(candidates, i, axis=0) for i in range(N_va)])  # shape: (N_va, N_va-1, 2)
        
        # 生成掩码
        masks = np.array([(i >> np.arange(mask_bits)) & 1 for i in range(2**mask_bits)], dtype=np.int8)  # (2^(N_va-1), N_va-1)
        masks = masks[:, None, :]  # 扩展为(2^(N_va-1), 1, N_va-1)
        masks = np.broadcast_to(masks, (masks.shape[0], N_va, masks.shape[2]))  # 扩展为(2^(N_va-1), N_va, N_va-1)
        
        # 根据掩码选择x和y (无效位置设为NaN)
        selected_x = np.where(masks == 0, arr_del1[..., 0], np.nan)
        selected_y = np.where(masks == 1, arr_del1[..., 1], np.nan)
        
        # 计算max和min (忽略NaN)
        max_x = np.nanmax(selected_x, axis=2)  # shape: (2^(N_va-1), N_va)
        min_x = np.nanmin(selected_x, axis=2)
        max_y = np.nanmax(selected_y, axis=2)
        min_y = np.nanmin(selected_y, axis=2)
        
        # 计算差值
        diffs_x = max_x - min_x # shape: (2^(N_va-1), N_va)
        diffs_y = max_y - min_y
        
        # 寻找满足容差条件的组合
        valid_mask = (diffs_x < self.tol) & (diffs_y < self.tol) & ~np.isnan(diffs_x) & ~np.isnan(diffs_y)
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)
            mask_idx, del_idx = valid_indices[0], valid_indices[1]
            
            # 批量计算平均值
            valid_selected_x = selected_x[mask_idx, del_idx]  # shape: (n_valid, N_va-1)
            valid_selected_y = selected_y[mask_idx, del_idx]  # shape: (n_valid, N_va-1)
            
            # 计算每行的平均值，忽略NaN
            mean_x = np.nanmean(valid_selected_x, axis=1)
            mean_y = np.nanmean(valid_selected_y, axis=1)
            
            # 检查是否有有效值
            has_valid_x = ~np.isnan(mean_x)
            has_valid_y = ~np.isnan(mean_y)
            final_valid = has_valid_x & has_valid_y
            
            if np.any(final_valid):
                results = np.column_stack((mean_x[final_valid], mean_y[final_valid]))
                # 返回被删除的激光束索引
                deleted_indices = del_idx[final_valid]
                return results, deleted_indices
        
        return np.array([]).reshape(0, 2), np.array([], dtype=int)
    
    def _calculate_differences_n_2(self, candidates: np.ndarray, N_va: int) -> Tuple[np.ndarray, np.ndarray]:
        """计算n-2个激光束的diff_x, diff_y，返回结果和被删除的激光束索引对"""
        
        # 检查输入有效性
        if N_va <= 2 or len(candidates) == 0:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
            
        mask_bits = N_va - 2
        if mask_bits > 20:  # 防止掩码数量过大
            self._log_warning(f"激光束数量({N_va})太大，跳过n-2组合计算")
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
        
        # 生成所有删除2列的组合
        del_indices = list(combinations(range(N_va), 2))  # 所有删除2列的索引组合
        del_indices_array = np.array(del_indices)  # 转换为数组便于索引
        arr_del2 = np.stack([np.delete(candidates, list(idx), axis=0) for idx in del_indices])  # (N_va*(N_va-1)/2, N_va-2, 2)
        
        # 生成掩码
        masks = np.array([(i >> np.arange(mask_bits)) & 1 for i in range(2**mask_bits)], dtype=np.int8)  # (2^(N_va-2), N_va-2)
        masks = masks[:, None, :]  # 扩展为(2^(N_va-2), 1, N_va-2)
        masks = np.broadcast_to(masks, (masks.shape[0], arr_del2.shape[0], masks.shape[2]))  # 扩展为(2^(N_va-2), N_va*(N_va-1)/2, N_va-2)
        
        # 根据掩码选择x和y (无效位置设为NaN)
        selected_x = np.where(masks == 0, arr_del2[..., 0], np.nan)  # 广播到(2^(N_va-2), N_va*(N_va-1)/2, N_va-2)
        selected_y = np.where(masks == 1, arr_del2[..., 1], np.nan)
        
        # 计算max和min (忽略NaN)
        max_x = np.nanmax(selected_x, axis=2)  # shape: (2^(N_va-2), N_va*(N_va-1)/2)
        min_x = np.nanmin(selected_x, axis=2)
        max_y = np.nanmax(selected_y, axis=2)
        min_y = np.nanmin(selected_y, axis=2)
        
        # 计算差值
        diffs_x = max_x - min_x
        diffs_y = max_y - min_y
        
        # 寻找满足容差条件的组合
        valid_mask = (diffs_x < self.tol) & (diffs_y < self.tol) & ~np.isnan(diffs_x) & ~np.isnan(diffs_y)
        
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)
            mask_idx, del_comb_idx = valid_indices[0], valid_indices[1]
            
            # 批量计算平均值
            valid_selected_x = selected_x[mask_idx, del_comb_idx]  # shape: (n_valid, N_va-2)
            valid_selected_y = selected_y[mask_idx, del_comb_idx]  # shape: (n_valid, N_va-2)
            
            # 计算每行的平均值，忽略NaN
            mean_x = np.nanmean(valid_selected_x, axis=1)
            mean_y = np.nanmean(valid_selected_y, axis=1)
            
            # 检查是否有有效值
            has_valid_x = ~np.isnan(mean_x)
            has_valid_y = ~np.isnan(mean_y)
            final_valid = has_valid_x & has_valid_y
            
            if np.any(final_valid):
                results = np.column_stack((mean_x[final_valid], mean_y[final_valid]))
                # 返回被删除的激光束索引对
                deleted_indices = del_indices_array[del_comb_idx[final_valid]]
                return results, deleted_indices
        
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    def _calculate_final_results(self, results: np.ndarray, collision_vectors: np.ndarray, excluded_lasers: np.ndarray = None) -> np.ndarray:
        """计算最终结果，考虑被排除的激光束，使用批量计算优化"""
        if len(results) == 0 or len(collision_vectors) == 0:
            return np.array([]).reshape(0, 2)
        
        # 如果没有排除的激光束（n情况），使用原始的批量方法
        if excluded_lasers is None or excluded_lasers.size == 0:
            return self._calculate_final_results_batch(results, collision_vectors)
        
        # 对于有排除激光束的情况，分组处理相同排除模式的结果
        valid_results = []
        
        # 将excluded_lasers转换为统一格式用于分组
        if excluded_lasers.ndim == 1:  # n_1情况
            excluded_keys = excluded_lasers
        else:  # n_2情况
            excluded_keys = [tuple(sorted(row)) for row in excluded_lasers]
        
        # 按排除模式分组
        unique_patterns = {}
        for i, pattern in enumerate(excluded_keys):
            if pattern not in unique_patterns:
                unique_patterns[pattern] = []
            unique_patterns[pattern].append(i)
        
        # 对每种排除模式批量处理
        for pattern, indices in unique_patterns.items():
            pattern_results = results[indices]
            
            # 创建激光束掩码
            laser_mask = np.ones(len(collision_vectors), dtype=bool)
            if isinstance(pattern, tuple):  # n_2情况
                if pattern:  # 非空元组
                    laser_mask[list(pattern)] = False
            else:  # n_1情况
                if pattern >= 0:  # 有效索引
                    laser_mask[pattern] = False
            
            # 使用激活的激光束进行批量计算
            active_collision_vectors = collision_vectors[laser_mask]
            if len(active_collision_vectors) > 0:
                valid_pattern_results = self._calculate_final_results_batch(
                    pattern_results, active_collision_vectors, use_tolerance=self.use_tolerance_checking)
                if len(valid_pattern_results) > 0:
                    valid_results.append(valid_pattern_results)
        
        return np.vstack(valid_results) if valid_results else np.array([]).reshape(0, 2)
    
    def _calculate_final_results_batch(self, results: np.ndarray, collision_vectors: np.ndarray, use_tolerance: bool = None) -> np.ndarray:
        """批量计算最终结果"""
        # 如果未指定use_tolerance，使用实例设置
        if use_tolerance is None:
            use_tolerance = self.use_tolerance_checking
            
        # 扩展维度进行广播计算
        results_expanded = results[:, None, :]  # shape: (n_results, 1, 2)
        collision_expanded = collision_vectors[None, :, :]  # shape: (1, n_collision, 2)
        
        # 批量计算所有组合: 每个结果点 + 每个碰撞向量
        all_points = results_expanded + collision_expanded  # shape: (n_results, n_collision, 2)
        
        # 判断是否在边界范围内
        if use_tolerance:
            # 要求每束激光都在边界±tol范围内
            x_valid = (all_points[..., 0] >= -self.tol) & (all_points[..., 0] <= self.m + self.tol)
            y_valid = (all_points[..., 1] >= -self.tol) & (all_points[..., 1] <= self.n + self.tol)
        else:
            # 原始判断：在矩形范围 [0, m] x [0, n] 内
            x_valid = (all_points[..., 0] >= 0) & (all_points[..., 0] <= self.m)
            y_valid = (all_points[..., 1] >= 0) & (all_points[..., 1] <= self.n)
        
        points_valid = x_valid & y_valid  # shape: (n_results, n_collision)
        
        # 对每个结果点，检查是否所有激光束都满足条件
        all_valid_per_result = np.all(points_valid, axis=1)  # shape: (n_results,)
        
        return results[all_valid_per_result]
    
    def get_excluded_laser_info(self, excluded_lasers: np.ndarray) -> str:
        """获取被排除激光束的信息字符串，用于调试"""
        if excluded_lasers is None or excluded_lasers.size == 0:
            return "无排除激光束"
        
        info_parts = []
        if excluded_lasers.ndim == 1:  # n_1情况
            for i, laser_idx in enumerate(excluded_lasers):
                if laser_idx >= 0:
                    info_parts.append(f"结果{i}: 排除激光束{laser_idx}")
        else:  # n_2情况
            for i, laser_pair in enumerate(excluded_lasers):
                if laser_pair.size > 0:
                    info_parts.append(f"结果{i}: 排除激光束{laser_pair.tolist()}")
        
        return "; ".join(info_parts) if info_parts else "无有效排除信息"
    
    def get_tolerance_info(self) -> str:
        """获取容差设置信息"""
        mode = "容差模式" if self.use_tolerance_checking else "严格边界模式"
        return f"边界检查: {mode} (m={self.m}, n={self.n}, tol={self.tol})"