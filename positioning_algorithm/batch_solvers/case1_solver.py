"""
Case1批处理求解器 - 重新设计版本
处理基边对齐一条矩形边，第三点在内部或其他边的情况
"""
import numpy as np
import os
from typing import List, Tuple
import logging
from .trig_cache import trig_cache

class Case1BatchSolver:
    """Case1批处理求解器类"""
    
    def __init__(self, m: float, n: float, tolerance: float = 1e-3, ros_logger=None):
        """
        初始化Case1求解器
        
        Args:
            m: 场地宽度
            n: 场地高度  
            tolerance: 数值容差
            enable_ros_logging: 是否启用ROS日志
            ros_logger: ROS日志器对象
        """
        self.m = m
        self.n = n
        self.tolerance = tolerance
        self.ros_logger = ros_logger
        if self.ros_logger:
            # 如果启用ROS日志，我们就创建一个handler
            self.logger = self.ros_logger.get_logger("solver.case1", "batches/case1")
        else:
            # 设置日志系统
            self._setup_logging()
        
        self._log_info("Case1BatchSolver初始化完成", 
                      f"场地尺寸: {m}x{n}, 容差: {tolerance}")
    
    def _setup_logging(self):
        """设置兼容ROS和Windows的日志系统"""
        # 创建logs目录
        os.makedirs('logs', exist_ok=True)
        
        # 初始化标准日志器
        self.logger = logging.getLogger("Case1BatchSolver")
        
        # 清理已有的handlers，避免重复添加
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
        
        # 根据ROS状态设置日志等级
        if self.enable_ros_logging:
            self.min_log_level = logging.WARNING  # ROS模式下只输出WARNING及以上
        else:
            self.min_log_level = logging.DEBUG    # Windows调试模式下输出详细日志
        
        # 添加文件日志器（Windows调试用）
        if not self.enable_ros_logging:
            handler = logging.FileHandler("logs/case1_batch_solver.log", encoding="utf-8")
            formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(self.min_log_level)
    
    def _log_debug(self, title: str, content: str = ""):
        """调试级别日志"""
        if self.enable_ros_logging and self.ros_logger:
            # ROS模式下不输出调试信息
            pass
        else:
            if content:
                self.logger.debug(f"{title}: {content}")
            else:
                self.logger.debug(title)
    
    def _log_info(self, title: str, content: str = ""):
        """信息级别日志"""
        if self.enable_ros_logging and self.ros_logger:
            if content:
                self.ros_logger.info(f"{title}: {content}")
            else:
                self.ros_logger.info(title)
        else:
            if content:
                self.logger.info(f"{title}: {content}")
            else:
                self.logger.info(title)
    
    def _log_warning(self, title: str, content: str = ""):
        """警告级别日志"""
        if self.enable_ros_logging and self.ros_logger:
            if content:
                self.ros_logger.warning(f"{title}: {content}")
            else:
                self.ros_logger.warning(title)
        else:
            if content:
                self.logger.warning(f"{title}: {content}")
            else:
                self.logger.warning(title)
    
    def _log_array_detailed(self, name: str, arr):
        """详细结构化输出数组的每个元素"""
        # 性能优化：注释掉所有详细日志输出
        pass
        # if isinstance(arr, list):
        #     if len(arr) == 0:
        #         self._log_debug(name, "空列表")
        #         return
        #     
        #     self._log_debug(f"{name} 详细内容", f"列表长度: {len(arr)}")
        #     for i, item in enumerate(arr):
        #         self._log_debug(f"{name}[{i}]", f"{item}")
        # 
        # elif isinstance(arr, np.ndarray):
        #     if arr.size == 0:
        #         self._log_debug(name, "空数组")
        #         return
        #     
        #     self._log_debug(f"{name} 详细内容", f"形状: {arr.shape}, 类型: {arr.dtype.name}")
        #     
        #     if arr.ndim == 1:
        #         # 一维数组：逐个显示
        #         for i in range(len(arr)):
        #             self._log_debug(f"{name}[{i}]", f"{arr[i]}")
        #     
        #     elif arr.ndim == 2:
        #         # 二维数组：按行显示
        #         for i in range(arr.shape[0]):
        #             self._log_debug(f"{name}[{i}]", f"{arr[i]}")
        #     
        #     elif arr.ndim == 3:
        #         # 三维数组：分层显示
        #         for i in range(arr.shape[0]):
        #             self._log_debug(f"{name}[{i}] 形状{arr[i].shape}", "")
        #             for j in range(arr.shape[1]):
        #                 self._log_debug(f"  {name}[{i}][{j}]", f"{arr[i][j]}")
        #     
        #     else:
        #         # 更高维数组：显示基本信息和前几个元素
        #         self._log_debug(f"{name}", f"高维数组 {arr.shape}，显示前5个元素:")
        #         flat_arr = arr.flatten()
        #         for i in range(min(5, len(flat_arr))):
        #             self._log_debug(f"{name}.flat[{i}]", f"{flat_arr[i]}")
        # 
        # else:
        #     self._log_debug(name, f"类型: {type(arr)}, 内容: {arr}")

    def _log_array(self, name: str, arr, show_content: bool = True):
        """简单的数组/列表日志打印函数"""
        # 处理列表类型：直接打印列表信息，不转换
        if isinstance(arr, list):
            if len(arr) == 0:
                self._log_debug(name, "空列表")
                return
            
            info = f"列表长度{len(arr)}"
            
            # 内容预览（列表直接显示前几个元素）
            if show_content and len(arr) <= 5:
                content_preview = str(arr)
                if len(content_preview) > 200:  # 如果内容太长就截断
                    content_preview = content_preview[:200] + "..."
                info += f", 内容{content_preview}"
            elif show_content:
                # 显示前几个元素的类型信息
                sample_types = [type(item).__name__ for item in arr[:3]]
                info += f", 前3个元素类型{sample_types}"
                for item in arr:
                    self._log_debug(name, f"{item}")
            
            self._log_debug(name, info)
            return
        
        # 处理numpy数组：原有逻辑
        if arr.size == 0:
            self._log_debug(name, "空数组")
            return
        
        # 基本信息：形状和类型
        info = f"形状{arr.shape}, 类型{arr.dtype.name}"
        
        # 数值统计（仅对数值类型）
        if np.issubdtype(arr.dtype, np.number):
            if np.all(np.isfinite(arr)):
                info += f", 范围[{arr.min():.3f}, {arr.max():.3f}]"
            else:
                valid_count = np.sum(np.isfinite(arr))
                info += f", 有效值{valid_count}/{arr.size}"
        
        # 内容预览（可选）
        if show_content and arr.size <= 10:
            for item in arr.flat:
                self._log_debug(name, f"{item}")
        self._log_debug(name, info)
    
    def solve(self, combinations: np.ndarray) -> np.ndarray:
        """
        求解Case1批处理（规则化版本）
        
        Args:
            combinations: 激光组合数组 (N, 3, 2) - 每行为[[t0,θ0], [t1,θ1], [t2,θ2]]
            
        Returns:
            np.ndarray: 解数组 (N, 5) - 每行为[xmin, xmax, ymin, ymax, phi]
                       无解的组合用np.inf填充
        """
        # self._log_info("=" * 60)
        # self._log_info("开始Case1批处理求解（规则化版本）")
        # self._log_info("=" * 60)
        
        # 输入验证和日志
        # self._log_array("输入组合数组", combinations, show_content=False)
        # self._log_array_detailed("输入组合详细", combinations)
        
        if len(combinations) == 0:
            # self._log_warning("没有组合可求解")
            return np.array([]).reshape(0, 5)
        
        N = len(combinations)
        # self._log_info(f"组合数量统计", f"总共{N}个激光组合需要处理")
        
        # 第一层: 扩展组合 (N -> 3N)
        # self._log_debug("第一层: 开始组合扩展 N -> 3N")
        expanded, indices = self._expand_combinations(combinations)
        # self._log_array_detailed("扩展后组合详细", expanded)
        # self._log_array_detailed("组合索引追踪详细", indices)
        
        # 第二层: 计算phi
        # 步骤1：计算A/B系数
        # self._log_debug("第二层步骤1: 计算A、B系数")
        ab_h, ab_v = self._compute_ab_coefficients(expanded)
        # self._log_array_detailed("水平边A、B系数详细", ab_h)
        # self._log_array_detailed("竖直边A、B系数详细", ab_v)
        
        # 步骤2: 计算phi候选 (6N,)
        # self._log_debug("第二层步骤2: 计算phi候选")
        phi_h, valid_h = self._compute_phi_candidates_regularized(ab_h)
        phi_v, valid_v = self._compute_phi_candidates_regularized(ab_v)
        
        # self._log_array_detailed("水平边phi候选详细", phi_h)
        # self._log_array_detailed("水平边有效掩码详细", valid_h)
        # self._log_array_detailed("竖直边phi候选详细", phi_v)
        # self._log_array_detailed("竖直边有效掩码详细", valid_v)
        
        valid_count_h = np.sum(valid_h)
        valid_count_v = np.sum(valid_v)
        # self._log_info("phi候选统计", f"水平边: {valid_count_h}/{6*N}个有效, 竖直边: {valid_count_v}/{6*N}个有效")
        
        # 第三层: 计算碰撞三角形和关键量
        # self._log_debug("第三层: 计算碰撞三角形和关键量")
        colli_h, key_h = self._compute_collision_and_key_regularized(phi_h, valid_h, expanded, "horizontal")
        colli_v, key_v = self._compute_collision_and_key_regularized(phi_v, valid_v, expanded, "vertical")
        
        # self._log_array_detailed("水平边碰撞三角形详细", colli_h)
        # self._log_array_detailed("水平边关键量详细", key_h)
        # self._log_array_detailed("竖直边碰撞三角形详细", colli_v)
        # self._log_array_detailed("竖直边关键量详细", key_v)
        
        # 第四层: 批量求解
        # self._log_debug("第四层: 批量求解")
        sol_h, valid_h = self._solve_batch_regularized(colli_h, key_h, phi_h, valid_h, "horizontal")
        sol_v, valid_v = self._solve_batch_regularized(colli_v, key_v, phi_v, valid_v, "vertical")
        
        # self._log_array_detailed("水平边解详细", sol_h)
        # self._log_array_detailed("竖直边解详细", sol_v)
        # self._log_array_detailed("水平边有效掩码详细", valid_h)
        # self._log_array_detailed("竖直边有效掩码详细", valid_v)
        
        # 第五层：合并结果(12N, 5) -> (N, 12, 5)
        # self._log_debug("第五层: 合并结果")
        final_sol, final_valid = self._merge_solutions_regularized(sol_h, sol_v, valid_h, valid_v, indices, N)
        
        # self._log_array_detailed("最终解详细", final_sol)
        # self._log_array_detailed("最终有效掩码详细", final_valid)
        
        # 最终统计
        valid_solutions = np.sum(~np.isinf(final_sol[:, 0]))
        # self._log_info("=" * 60)
        # self._log_info("Case1求解完成", f"在{N}个组合中找到{valid_solutions}个有效解")
        # self._log_info("=" * 60)
        
        return final_sol, final_valid
    
    def _expand_combinations(self, combinations: np.ndarray) -> tuple:
        """
        步骤1: 扩展组合：N_cbn -> 3*N_cbn，并跟踪原始组合编号
        
        Args:
            combinations: 输入组合 (N_cbn, 3, 2) - [[t0,θ0], [t1,θ1], [t2,θ2]]
            
        Returns:
            tuple: (expanded_combinations, combo_indices)
            - expanded_combinations: 扩展后组合 (3*N_cbn, 3, 2)
            - combo_indices: 原始组合编号追踪 (3*N_cbn,) - 记录每个扩展组合对应的原始组合编号
        """
        N_cbn = len(combinations)
        expanded = np.zeros((3 * N_cbn, 3, 2), dtype=np.float64)
        combo_indices = np.zeros(3 * N_cbn, dtype=np.int32)  # 跟踪原始组合编号
        
        for combo_idx in range(N_cbn):
            for base_idx in range(3):
                p1_idx = base_idx
                p2_idx = (base_idx + 1) % 3
                opp_idx = (base_idx + 2) % 3
                
                # 重新排列：[P1, P2, Opp] 对应 [基边点1, 基边点2, 对角点]
                expanded_idx = combo_idx * 3 + base_idx
                expanded[expanded_idx, 0] = combinations[combo_idx, p1_idx]  # P1
                expanded[expanded_idx, 1] = combinations[combo_idx, p2_idx]  # P2
                expanded[expanded_idx, 2] = combinations[combo_idx, opp_idx] # Opp
                
                # 记录原始组合编号
                combo_indices[expanded_idx] = combo_idx
        
        self._log_debug("组合扩展完成", f"{N_cbn} -> {len(expanded)}个扩展组合，原始编号已追踪")
        return expanded, combo_indices
    
    def _compute_ab_coefficients(self, expanded_combinations: np.ndarray) -> tuple:
        """
        步骤2: 计算A、B系数，分水平边和竖直边
        
        Args:
            expanded_combinations: 扩展组合 (3*N_cbn, 3, 2)
            
        Returns:
            tuple: (ab_h, ab_v)
            - ab_h: 水平边的[A, B]系数 (3*N_cbn, 2)
            - ab_v: 竖直边的[A, B]系数 (3*N_cbn, 2)
        """
        # 提取t和theta值
        t_vals = expanded_combinations[:, :, 0]  # (3*N_cbn, 3) - [t1, t2, t_opp]
        theta_vals = expanded_combinations[:, :, 1]  # (3*N_cbn, 3) - [θ1, θ2, θ_opp]
        
        # 预计算三角函数值
        cos_theta = np.cos(theta_vals)  # (3*N_cbn, 3)
        sin_theta = np.sin(theta_vals)  # (3*N_cbn, 3)
        
        # 水平边系数计算 (基边水平对齐：P1.y = P2.y)
        # A = t1*cos(θ1) - t2*cos(θ2)
        # B = t1*sin(θ1) - t2*sin(θ2)
        A_horizontal = t_vals[:, 0] * cos_theta[:, 0] - t_vals[:, 1] * cos_theta[:, 1]
        B_horizontal = t_vals[:, 0] * sin_theta[:, 0] - t_vals[:, 1] * sin_theta[:, 1]
        ab_h = np.column_stack([A_horizontal, B_horizontal])  # (3*N_cbn, 2)
        
        # 竖直边系数计算 (基边竖直对齐：P1.x = P2.x)
        # A = t2*sin(θ2) - t1*sin(θ1)
        # B = t1*cos(θ1) - t2*cos(θ2)
        A_vertical = t_vals[:, 1] * sin_theta[:, 1] - t_vals[:, 0] * sin_theta[:, 0]
        B_vertical = t_vals[:, 0] * cos_theta[:, 0] - t_vals[:, 1] * cos_theta[:, 1]
        ab_v = np.column_stack([A_vertical, B_vertical])  # (3*N_cbn, 2)
        
        self._log_debug("A、B系数计算完成", f"水平边{ab_h.shape}, 竖直边{ab_v.shape}")
        return ab_h, ab_v
    
    def _compute_phi_candidates(self, ab_h: np.ndarray, ab_v: np.ndarray) -> tuple:
        """
        步骤3: 计算phi候选（规则化批处理版本）
        
        Args:
            ab_h: 水平边A、B系数 (3*N, 2)
            ab_v: 竖直边A、B系数 (3*N, 2)
            
        Returns:
            tuple: (phi_h, phi_v, valid_mask_h, valid_mask_v)
            - phi_h: 水平边phi候选 (6*N,) - 规则化存储，每组合6个phi值
            - phi_v: 竖直边phi候选 (6*N,) - 规则化存储，每组合6个phi值  
            - valid_mask_h: 水平边有效掩码 (6*N,)
            - valid_mask_v: 竖直边有效掩码 (6*N,)
        """
        N_expanded = len(ab_h)  # 3*N
        N_base = N_expanded // 3  # N
        
        # 水平边批处理计算
        A_h = ab_h[:, 0]  # (3*N,)
        B_h = ab_h[:, 1]  # (3*N,)
        
        # 退化性检查掩码
        degenerate_h = (np.abs(A_h) < self.tolerance) & (np.abs(B_h) < self.tolerance)
        special_A_h = (np.abs(A_h) < self.tolerance) & (~degenerate_h)
        normal_h = ~degenerate_h & ~special_A_h
        
        # 初始化规则化phi数组 (6*N,) - 每个扩展组合产生2个phi
        phi_h = np.full(6 * N_base, np.inf, dtype=np.float64)
        valid_mask_h = np.zeros(6 * N_base, dtype=bool)
        
        # 处理特殊情况：A接近0
        if np.any(special_A_h):
            phi_special = np.copysign(np.pi/2, -B_h[special_A_h])
            special_indices = np.where(special_A_h)[0]
            for i, exp_idx in enumerate(special_indices):
                base_phi_idx = exp_idx * 2  # 每个扩展组合对应2个phi位置
                phi_h[base_phi_idx] = phi_special[i]
                phi_h[base_phi_idx + 1] = phi_special[i] + np.pi
                valid_mask_h[base_phi_idx:base_phi_idx + 2] = True
        
        # 处理一般情况
        if np.any(normal_h):
            phi_normal = np.arctan2(-B_h[normal_h], A_h[normal_h])
            normal_indices = np.where(normal_h)[0]
            for i, exp_idx in enumerate(normal_indices):
                base_phi_idx = exp_idx * 2
                phi_h[base_phi_idx] = phi_normal[i]
                phi_h[base_phi_idx + 1] = phi_normal[i] + np.pi
                valid_mask_h[base_phi_idx:base_phi_idx + 2] = True
        
        # 竖直边批处理计算
        A_v = ab_v[:, 0]  # (3*N,)
        B_v = ab_v[:, 1]  # (3*N,)
        
        # 退化性检查掩码
        degenerate_v = (np.abs(A_v) < self.tolerance) & (np.abs(B_v) < self.tolerance)
        special_A_v = (np.abs(A_v) < self.tolerance) & (~degenerate_v)
        normal_v = ~degenerate_v & ~special_A_v
        
        # 初始化规则化phi数组
        phi_v = np.full(6 * N_base, np.inf, dtype=np.float64)
        valid_mask_v = np.zeros(6 * N_base, dtype=bool)
        
        # 处理特殊情况：A接近0
        if np.any(special_A_v):
            phi_special = np.where(B_v[special_A_v] > 0, -np.pi/2, np.pi/2)
            special_indices = np.where(special_A_v)[0]
            for i, exp_idx in enumerate(special_indices):
                base_phi_idx = exp_idx * 2
                phi_v[base_phi_idx] = phi_special[i]
                phi_v[base_phi_idx + 1] = phi_special[i] + np.pi
                valid_mask_v[base_phi_idx:base_phi_idx + 2] = True
        
        # 处理一般情况
        if np.any(normal_v):
            phi_normal = np.arctan2(B_v[normal_v], -A_v[normal_v])
            normal_indices = np.where(normal_v)[0]
            for i, exp_idx in enumerate(normal_indices):
                base_phi_idx = exp_idx * 2
                phi_v[base_phi_idx] = phi_normal[i]
                phi_v[base_phi_idx + 1] = phi_normal[i] + np.pi
                valid_mask_v[base_phi_idx:base_phi_idx + 2] = True
        
        # 将phi值规范化到[-π, π]范围
        phi_h = np.where(phi_h > np.pi, phi_h - 2*np.pi, phi_h)
        phi_h = np.where(phi_h <= -np.pi, phi_h + 2*np.pi, phi_h)
        phi_v = np.where(phi_v > np.pi, phi_v - 2*np.pi, phi_v)
        phi_v = np.where(phi_v <= -np.pi, phi_v + 2*np.pi, phi_v)
        
        valid_count_h = np.sum(valid_mask_h)
        valid_count_v = np.sum(valid_mask_v)
        self._log_debug("phi候选生成完成", f"水平边{valid_count_h}个, 竖直边{valid_count_v}个有效值")
        return phi_h, phi_v, valid_mask_h, valid_mask_v
    
    def _compute_collision_triangles_h(self, phi_h: np.ndarray, valid_mask_h: np.ndarray,
                                     expanded_combinations: np.ndarray,
                                     combo_indices: np.ndarray) -> tuple:
        """
        步骤4a: 计算水平边情况的碰撞三角形（规则化批处理版本）
        
        Args:
            phi_h: 水平边phi候选 (6*N,)
            valid_mask_h: 水平边有效掩码 (6*N,)
            expanded_combinations: 扩展组合 (3*N, 3, 2)
            combo_indices: 原始组合编号追踪 (3*N,)
            
        Returns:
            tuple: (colli_h, combo_indices_h)
            - colli_h: 水平边碰撞三角形数组 (6*N, 3, 2) - 规则化存储
            - combo_indices_h: 对应的原始组合编号 (6*N,)
        """
        N_expanded = len(expanded_combinations)  # 3*N
        N_base = N_expanded // 3  # N
        N_phi = 6 * N_base  # 6*N
        
        # 初始化规则化数组
        colli_h = np.full((N_phi, 3, 2), np.inf, dtype=np.float64)
        combo_indices_h = np.zeros(N_phi, dtype=np.int32)
        
        # 为每个扩展组合生成对应的组合索引
        for exp_idx in range(N_expanded):
            phi_start = exp_idx * 2  # 每个扩展组合对应2个phi
            phi_end = phi_start + 2
            combo_indices_h[phi_start:phi_end] = combo_indices[exp_idx]
        
        # 批量计算有效的碰撞三角形
        valid_indices = np.where(valid_mask_h)[0]
        
        if len(valid_indices) > 0:
            # 计算对应的扩展组合索引
            exp_indices = valid_indices // 2
            
            # 获取有效的phi值和组合
            valid_phi = phi_h[valid_indices]
            valid_combinations = expanded_combinations[exp_indices]
            
            # 批量计算旋转变换
            t_values = valid_combinations[:, :, 0]  # (N_valid, 3)
            theta_values = valid_combinations[:, :, 1]  # (N_valid, 3)
            
            # 广播计算：phi + theta
            phi_plus_theta = valid_phi[:, None] + theta_values  # (N_valid, 3)
            
            # 批量计算x和y坐标
            x_coords = t_values * np.cos(phi_plus_theta)  # (N_valid, 3)
            y_coords = t_values * np.sin(phi_plus_theta)  # (N_valid, 3)
            
            # 将结果存储到规则化数组中
            colli_h[valid_indices, :, 0] = x_coords
            colli_h[valid_indices, :, 1] = y_coords
        
        valid_count = np.sum(valid_mask_h)
        self._log_debug("水平边碰撞三角形计算完成", f"{valid_count}个有效三角形，规则化存储")
        return colli_h, combo_indices_h
    
    def _compute_collision_triangles_v(self, phi_v: np.ndarray, valid_mask_v: np.ndarray,
                                     expanded_combinations: np.ndarray,
                                     combo_indices: np.ndarray) -> tuple:
        """
        步骤4b: 计算竖直边情况的碰撞三角形（规则化批处理版本）
        
        Args:
            phi_v: 竖直边phi候选 (6*N,)
            valid_mask_v: 竖直边有效掩码 (6*N,)
            expanded_combinations: 扩展组合 (3*N, 3, 2)
            combo_indices: 原始组合编号追踪 (3*N,)
            
        Returns:
            tuple: (colli_v, combo_indices_v)
            - colli_v: 竖直边碰撞三角形数组 (6*N, 3, 2) - 规则化存储
            - combo_indices_v: 对应的原始组合编号 (6*N,)
        """
        N_expanded = len(expanded_combinations)  # 3*N
        N_base = N_expanded // 3  # N
        N_phi = 6 * N_base  # 6*N
        
        # 初始化规则化数组
        colli_v = np.full((N_phi, 3, 2), np.inf, dtype=np.float64)
        combo_indices_v = np.zeros(N_phi, dtype=np.int32)
        
        # 为每个扩展组合生成对应的组合索引
        for exp_idx in range(N_expanded):
            phi_start = exp_idx * 2  # 每个扩展组合对应2个phi
            phi_end = phi_start + 2
            combo_indices_v[phi_start:phi_end] = combo_indices[exp_idx]
        
        # 批量计算有效的碰撞三角形
        valid_indices = np.where(valid_mask_v)[0]
        
        if len(valid_indices) > 0:
            # 计算对应的扩展组合索引
            exp_indices = valid_indices // 2
            
            # 获取有效的phi值和组合
            valid_phi = phi_v[valid_indices]
            valid_combinations = expanded_combinations[exp_indices]
            
            # 批量计算旋转变换
            t_values = valid_combinations[:, :, 0]  # (N_valid, 3)
            theta_values = valid_combinations[:, :, 1]  # (N_valid, 3)
            
            # 广播计算：phi + theta
            phi_plus_theta = valid_phi[:, None] + theta_values  # (N_valid, 3)
            
            # 批量计算x和y坐标
            x_coords = t_values * np.cos(phi_plus_theta)  # (N_valid, 3)
            y_coords = t_values * np.sin(phi_plus_theta)  # (N_valid, 3)
            
            # 将结果存储到规则化数组中
            colli_v[valid_indices, :, 0] = x_coords
            colli_v[valid_indices, :, 1] = y_coords
        
        valid_count = np.sum(valid_mask_v)
        self._log_debug("竖直边碰撞三角形计算完成", f"{valid_count}个有效三角形，规则化存储")
        return colli_v, combo_indices_v
    
    def _compute_key_quantities_h(self, colli_h: np.ndarray) -> np.ndarray:
        """
        步骤5a: 计算水平边关键量
        
        Args:
            colli_h: 水平边碰撞三角形数组 (N_valid_h, 3, 2)
            
        Returns:
            key_h: 水平边关键量 (N_valid_h, 3) - [t1, t2, t3]
        """
        if len(colli_h) == 0:
            return np.array([]).reshape(0, 3)
        
        # 提取坐标
        xp0 = colli_h[:, 0, 0]  # P1.x
        yp0 = colli_h[:, 0, 1]  # P1.y
        xp1 = colli_h[:, 1, 0]  # P2.x
        yp1 = colli_h[:, 1, 1]  # P2.y
        xp2 = colli_h[:, 2, 0]  # Opp.x
        yp2 = colli_h[:, 2, 1]  # Opp.y
        
        # 水平边关键量计算
        t1_h = yp2 - (yp1 + yp0) / 2  # t1: yp2-(yp1+yp0)/2
        t2_h = xp2 - xp0              # t2: xp2-xp0
        t3_h = xp2 - xp1              # t3: xp2-xp1
        key_h = np.column_stack([t1_h, t2_h, t3_h])
        
        return key_h
    
    def _compute_key_quantities_v(self, colli_v: np.ndarray) -> np.ndarray:
        """
        步骤5b: 计算竖直边关键量
        
        Args:
            colli_v: 竖直边碰撞三角形数组 (N_valid_v, 3, 2)
            
        Returns:
            key_v: 竖直边关键量 (N_valid_v, 3) - [t1, t2, t3]
        """
        if len(colli_v) == 0:
            return np.array([]).reshape(0, 3)
        
        # 提取坐标
        xp0 = colli_v[:, 0, 0]  # P1.x
        yp0 = colli_v[:, 0, 1]  # P1.y
        xp1 = colli_v[:, 1, 0]  # P2.x
        yp1 = colli_v[:, 1, 1]  # P2.y
        xp2 = colli_v[:, 2, 0]  # Opp.x
        yp2 = colli_v[:, 2, 1]  # Opp.y
        
        # 竖直边关键量计算 (x, y反过来)
        t1_v = xp2 - (xp1 + xp0) / 2  # t1: xp2-(xp1+xp0)/2
        t2_v = yp2 - yp0              # t2: yp2-yp0
        t3_v = yp2 - yp1              # t3: yp2-yp1
        key_v = np.column_stack([t1_v, t2_v, t3_v])
        
        return key_v
    
    def _solve_both_cases(self, colli_h: np.ndarray, colli_v: np.ndarray,
                         key_h: np.ndarray, key_v: np.ndarray,
                         valid_indice_cbo_h: np.ndarray, valid_indice_cbo_v: np.ndarray,
                         phi_h_flat: np.ndarray, phi_v_flat: np.ndarray) -> tuple:
        """
        步骤6: 求解水平边和竖直边两种情况（批处理版本）
        
        Args:
            colli_h: 水平边碰撞三角形数组 (N_valid_h, 3, 2)
            colli_v: 竖直边碰撞三角形数组 (N_valid_v, 3, 2)
            key_h: 水平边关键量 (N_valid_h, 3)
            key_v: 竖直边关键量 (N_valid_v, 3)
            valid_indice_cbo_h: 水平边对应的原始组合编号 (N_valid_h,)
            valid_indice_cbo_v: 竖直边对应的原始组合编号 (N_valid_v,)
            phi_h_flat: 水平边对应的phi值 (N_valid_h,)
            phi_v_flat: 竖直边对应的phi值 (N_valid_v,)
            
        Returns:
            tuple: (sols, indice_sol)
            - sols: 所有解列表
            - indice_sol: 解对应的组合编号 (N_sol,)
        """
        sols_list = []
        indice_list = []
        
        # 处理水平边情况
        if len(colli_h) > 0:
            sols_h, indices_h = self._solve_horizontal_batch(colli_h, key_h, valid_indice_cbo_h, phi_h_flat)
            sols_list.extend(sols_h)
            indice_list.extend(indices_h)
        
        # 处理竖直边情况
        if len(colli_v) > 0:
            sols_v, indices_v = self._solve_vertical_batch(colli_v, key_v, valid_indice_cbo_v, phi_v_flat)
            sols_list.extend(sols_v)
            indice_list.extend(indices_v)
        
        # 合并结果
        if len(sols_list) == 0:
            return [], np.array([])
        
        indice_sol = np.array(indice_list)
        
        return sols_list, indice_sol
    
    def _solve_horizontal_batch(self, colli_h: np.ndarray, key_h: np.ndarray, 
                               valid_indice_cbo_h: np.ndarray, phi_h_flat: np.ndarray) -> tuple:
        """
        批处理求解水平边情况
        
        Args:
            colli_h: 水平边碰撞三角形数组 (N_valid_h, 3, 2)
            key_h: 水平边关键量 (N_valid_h, 3)
            valid_indice_cbo_h: 对应的原始组合编号 (N_valid_h,)
            phi_h_flat: 对应的phi值 (N_valid_h,)
            
        Returns:
            tuple: (sols, indices)
        """
        if len(colli_h) == 0:
            return [], []
        
        N_h = len(colli_h)
        
        # 提取关键量
        t1_h = key_h[:, 0]  # (N_h,)
        t2_h = key_h[:, 1]  # (N_h,)
        t3_h = key_h[:, 2]  # (N_h,)
        
        # 提取坐标
        xp0 = colli_h[:, 0, 0]  # (N_h,)
        yp0 = colli_h[:, 0, 1]  # (N_h,)
        xp1 = colli_h[:, 1, 0]  # (N_h,)
        yp1 = colli_h[:, 1, 1]  # (N_h,)
        xp2 = colli_h[:, 2, 0]  # (N_h,)
        yp2 = colli_h[:, 2, 1]  # (N_h,)
        
        # 筛选条件：abs(t1)>n+tol 或 abs(t2)>m+tol 或 abs(t3)>m+tol
        invalid_mask = ((np.abs(t1_h) > self.n + self.tolerance) |
                       (np.abs(t2_h) > self.m + self.tolerance) |
                       (np.abs(t3_h) > self.m + self.tolerance))
        
        # 判断t1是否等于n或-n
        boundary_mask = np.abs(np.abs(t1_h) - self.n) <= self.tolerance
        
        # 判断t2, t3符号是否一致且都非零
        same_sign_mask = ((t2_h > 0) & (t3_h > 0)) | ((t2_h < 0) & (t3_h < 0))
        
        sols_list = []
        indices_list = []
        
        for i in range(N_h):
            if invalid_mask[i]:
                # 无效解
                sol = [(np.inf, np.inf), (np.inf, np.inf), np.inf]
            elif boundary_mask[i]:
                # t1 == ±n的边界情况
                y_center = (self.n - yp1[i] - yp0[i]) / 2
                x_max = self.m - max(xp0[i], xp1[i], xp2[i])
                x_min = 0 - min(xp0[i], xp1[i], xp2[i])
                
                xmin = x_min - self.tolerance
                xmax = x_max + self.tolerance
                ymin = y_center - self.tolerance
                ymax = y_center + self.tolerance
                
                sol = [(xmin, xmax), (ymin, ymax), phi_h_flat[i]]
            elif same_sign_mask[i]:
                # 符号一致的普通情况
                if t2_h[i] > 0:
                    x_center = self.m - xp2[i]
                else:
                    x_center = -xp2[i]
                
                if t1_h[i] > 0:
                    y_center = -(yp0[i] + yp1[i]) / 2
                else:
                    y_center = self.n - (yp0[i] + yp1[i]) / 2
                
                # 检查center是否在矩形框内
                if (0 <= x_center <= self.m and 0 <= y_center <= self.n):
                    xmin = x_center - self.tolerance
                    xmax = x_center + self.tolerance
                    ymin = y_center - self.tolerance
                    ymax = y_center + self.tolerance
                    
                    sol = [(xmin, xmax), (ymin, ymax), phi_h_flat[i]]
                else:
                    sol = [(np.inf, np.inf), (np.inf, np.inf), np.inf]
            else:
                # 符号不一致，无效解
                sol = [(np.inf, np.inf), (np.inf, np.inf), np.inf]
            
            sols_list.append(sol)
            indices_list.append(valid_indice_cbo_h[i])
        
        return sols_list, indices_list
    
    def _solve_vertical_batch(self, colli_v: np.ndarray, key_v: np.ndarray, 
                             valid_indice_cbo_v: np.ndarray, phi_v_flat: np.ndarray) -> tuple:
        """
        批处理求解竖直边情况
        
        Args:
            colli_v: 竖直边碰撞三角形数组 (N_valid_v, 3, 2)
            key_v: 竖直边关键量 (N_valid_v, 3)
            valid_indice_cbo_v: 对应的原始组合编号 (N_valid_v,)
            phi_v_flat: 对应的phi值 (N_valid_v,)
            
        Returns:
            tuple: (sols, indices)
        """
        if len(colli_v) == 0:
            return [], []
        
        N_v = len(colli_v)
        
        # 提取关键量
        t1_v = key_v[:, 0]  # (N_v,)
        t2_v = key_v[:, 1]  # (N_v,)
        t3_v = key_v[:, 2]  # (N_v,)
        
        # 提取坐标
        xp0 = colli_v[:, 0, 0]  # (N_v,)
        yp0 = colli_v[:, 0, 1]  # (N_v,)
        xp1 = colli_v[:, 1, 0]  # (N_v,)
        yp1 = colli_v[:, 1, 1]  # (N_v,)
        xp2 = colli_v[:, 2, 0]  # (N_v,)
        yp2 = colli_v[:, 2, 1]  # (N_v,)
        
        # 筛选条件：abs(t1)>m+tol 或 abs(t2)>n+tol 或 abs(t3)>n+tol (注意m,n互换)
        invalid_mask = ((np.abs(t1_v) > self.m + self.tolerance) |
                       (np.abs(t2_v) > self.n + self.tolerance) |
                       (np.abs(t3_v) > self.n + self.tolerance))
        
        # 判断t1是否等于m或-m
        boundary_mask = np.abs(np.abs(t1_v) - self.m) <= self.tolerance
        
        # 判断t2, t3符号是否一致且都非零
        same_sign_mask = ((t2_v > 0) & (t3_v > 0)) | ((t2_v < 0) & (t3_v < 0))
        
        sols_list = []
        indices_list = []
        
        for i in range(N_v):
            if invalid_mask[i]:
                # 无效解
                sol = [(np.inf, np.inf), (np.inf, np.inf), np.inf]
            elif boundary_mask[i]:
                # t1 == ±m的边界情况
                x_center = (self.m - xp1[i] - xp0[i]) / 2
                y_max = self.n - max(yp0[i], yp1[i], yp2[i])
                y_min = 0 - min(yp0[i], yp1[i], yp2[i])
                
                xmin = x_center - self.tolerance
                xmax = x_center + self.tolerance
                ymin = y_min - self.tolerance
                ymax = y_max + self.tolerance
                
                sol = [(xmin, xmax), (ymin, ymax), phi_v_flat[i]]
            elif same_sign_mask[i]:
                # 符号一致的普通情况
                if t2_v[i] > 0:
                    y_center = self.n - yp2[i]
                else:
                    y_center = -yp2[i]
                
                if t1_v[i] > 0:
                    x_center = -(xp0[i] + xp1[i]) / 2
                else:
                    x_center = self.m - (xp0[i] + xp1[i]) / 2
                
                # 检查center是否在矩形框内
                if (0 <= x_center <= self.m and 0 <= y_center <= self.n):
                    xmin = x_center - self.tolerance
                    xmax = x_center + self.tolerance
                    ymin = y_center - self.tolerance
                    ymax = y_center + self.tolerance
                    
                    sol = [(xmin, xmax), (ymin, ymax), phi_v_flat[i]]
                else:
                    sol = [(np.inf, np.inf), (np.inf, np.inf), np.inf]
            else:
                # 符号不一致，无效解
                sol = [(np.inf, np.inf), (np.inf, np.inf), np.inf]
            
            sols_list.append(sol)
            indices_list.append(valid_indice_cbo_v[i])
        
        return sols_list, indices_list
    
    def _solve_horizontal_case(self, triangle: np.ndarray, t1: float, t2: float, t3: float,
                              phi: float) -> np.ndarray:
        """
        求解水平边情况
        
        Args:
            triangle: 三角形顶点 (3, 2)
            t1, t2, t3: 关键中间量
            phi: 角度
            
        Returns:
            np.ndarray: 解 [xmin, xmax, ymin, ymax, phi] 或用np.inf标记的无效解
        """
        # 筛选条件：abs(t1)>n+tol 或 abs(t2)>m+tol 或 abs(t3)>m+tol
        if (abs(t1) > self.n + self.tolerance or 
            abs(t2) > self.m + self.tolerance or 
            abs(t3) > self.m + self.tolerance):
            return [(np.inf, np.inf), (np.inf, np.inf), np.inf]  # 用np.inf标记无效解
        
        xp0, yp0 = triangle[0]
        xp1, yp1 = triangle[1] 
        xp2, yp2 = triangle[2]
        
        # 判断t1是否等于n或-n (加减tol)
        if abs(abs(t1) - self.n) <= self.tolerance:
            # t1 == ±n的情况：不存在center，直接计算边界
            y_center = (self.n - yp1 - yp2) / 2
            x_max = self.m - max(xp0, xp1, xp2)
            x_min = 0 - min(xp0, xp1, xp2)
            
            # 给max加tol, 给min减tol
            xmin = x_min - self.tolerance
            xmax = x_max + self.tolerance
            ymin = y_center - self.tolerance
            ymax = y_center + self.tolerance
            
            return [(xmin, xmax), (ymin, ymax), phi]
        else:
            # 判断t2, t3符号是否一致
            if (t2 > 0 and t3 > 0) or (t2 < 0 and t3 < 0):
                # 符号一致，存在center
                if t2 > 0:  # 同为正
                    x_center = self.m - xp2
                else:  # 同为负
                    x_center = -xp2
                
                # y的取值看t1的正负号
                if t1 > 0:
                    y_center = -(yp0 + yp1) / 2
                else:
                    y_center = self.n - (yp0 + yp1) / 2
                
                # 检查center是否在矩形框内
                if not (0 <= x_center <= self.m and 0 <= y_center <= self.n):
                    return [(np.inf, np.inf), (np.inf, np.inf), np.inf]  # 用np.inf标记无效解
                
                # 添加容差
                xmin = x_center - self.tolerance
                xmax = x_center + self.tolerance
                ymin = y_center - self.tolerance
                ymax = y_center + self.tolerance
                
                return [(xmin, xmax), (ymin, ymax), phi]
            else:
                # 符号不一致，无效解
                return [(np.inf, np.inf), (np.inf, np.inf), np.inf]  # 用np.inf标记无效解
    
    def _solve_vertical_case(self, triangle: np.ndarray, t1: float, t2: float, t3: float,
                            phi: float) -> np.ndarray:
        """
        求解竖直边情况 (x, y反过来, m, n反过来)
        
        Args:
            triangle: 三角形顶点 (3, 2)
            t1, t2, t3: 关键中间量
            phi: 角度
            
        Returns:
            np.ndarray: 解 [xmin, xmax, ymin, ymax, phi] 或用np.inf标记的无效解
        """
        # 筛选条件：abs(t1)>m+tol 或 abs(t2)>n+tol 或 abs(t3)>n+tol (注意m,n互换)
        if (abs(t1) > self.m + self.tolerance or 
            abs(t2) > self.n + self.tolerance or 
            abs(t3) > self.n + self.tolerance):
            return [(np.inf, np.inf), (np.inf, np.inf), np.inf]  # 用np.inf标记无效解
        
        xp0, yp0 = triangle[0]
        xp1, yp1 = triangle[1]
        xp2, yp2 = triangle[2]
        
        # 判断t1是否等于m或-m (加减tol) (注意：竖直边对应m)
        if abs(abs(t1) - self.m) <= self.tolerance:
            # t1 == ±m的情况：不存在center，直接计算边界
            x_center = (self.m - xp1 - xp2) / 2
            y_max = self.n - max(yp0, yp1, yp2)
            y_min = 0 - min(yp0, yp1, yp2)
            
            # 给max加tol, 给min减tol
            xmin = x_center - self.tolerance
            xmax = x_center + self.tolerance
            ymin = y_min - self.tolerance
            ymax = y_max + self.tolerance
            
            return [(xmin, xmax), (ymin, ymax), phi]
        else:
            # 判断t2, t3符号是否一致
            if (t2 > 0 and t3 > 0) or (t2 < 0 and t3 < 0):
                # 符号一致，存在center
                if t2 > 0:  # 同为正
                    y_center = self.n - yp2
                else:  # 同为负
                    y_center = -yp2
                
                # x的取值看t1的正负号
                if t1 > 0:
                    x_center = -(xp0 + xp1) / 2
                else:
                    x_center = self.m - (xp0 + xp1) / 2
                
                # 检查center是否在矩形框内
                if not (0 <= x_center <= self.m and 0 <= y_center <= self.n):
                    return [(np.inf, np.inf), (np.inf, np.inf), np.inf]  # 用np.inf标记无效解
                
                # 添加容差
                xmin = x_center - self.tolerance
                xmax = x_center + self.tolerance
                ymin = y_center - self.tolerance
                ymax = y_center + self.tolerance
                
                return [(xmin, xmax),( ymin, ymax), phi]
            else:
                # 符号不一致，无效解
                return [(np.inf, np.inf), (np.inf, np.inf), np.inf]  # 用np.inf标记无效解
    
    def _organize_solutions(self, sols: np.ndarray, indice_sol: np.ndarray, N_cbn: int) -> np.ndarray:
        """
        步骤7: 组织结果到sol_cbn
        
        Args:
            sols: 所有解 (N_sol, 5)
            indice_sol: 解对应的组合编号 (N_sol,)
            N_cbn: 原始组合数量
            
        Returns:
            sol_cbn: 按组合编号组织的解 (N_cbn, 5) - [xmin, xmax, ymin, ymax, phi]
                    无解的组合用np.inf填充
        """
        # 初始化解数组，用np.inf填充表示无解
        sol_cbn = [[] for _ in range(N_cbn)]
        
        # 将解按组合编号放入对应位置
        for i, combo_idx in enumerate(indice_sol):
            if combo_idx < N_cbn:  # 防止索引越界
                sol_cbn[combo_idx].append(sols[i])
        
        return sol_cbn
    
    def get_solver_info(self) -> dict:
        """获取求解器信息"""
        return {
            "name": "Case1BatchSolver",
            "field_size": (self.m, self.n),
            "tolerance": self.tolerance,
            "description": "基边对齐一条矩形边，第三点在内部或其他边"
        }
    
    # =================== 规则化求解函数 ===================
    
    def _compute_phi_candidates_regularized(self, ab: np.ndarray) -> tuple:
        """
        计算phi候选（规则化版本）
        
        Args:
            ab: A、B系数 (3*N, 2)
            
        Returns:
            tuple: (phi, valid_mask)
            - phi: phi候选 (6*N,) - 规则化存储
            - valid_mask: 有效掩码 (6*N,)
        """
        N_expanded = len(ab)  # 3*N
        N_base = N_expanded // 3  # N
        
        A = ab[:, 0]  # (3*N,)
        B = ab[:, 1]  # (3*N,)
        
        # 退化性检查掩码
        degenerate = (np.abs(A) < self.tolerance) & (np.abs(B) < self.tolerance)
        special_A = (np.abs(A) < self.tolerance) & (~degenerate)
        normal = ~degenerate & ~special_A
        
        # 初始化规则化phi数组 (6*N,) - 每个扩展组合产生2个phi
        phi = np.full(6 * N_base, np.inf, dtype=np.float64)
        valid_mask = np.zeros(6 * N_base, dtype=bool)
        
        # 处理特殊情况：A接近0
        if np.any(special_A):
            phi_special = np.where(B[special_A] > 0, -np.pi/2, np.pi/2)
            special_indices = np.where(special_A)[0]
            for i, exp_idx in enumerate(special_indices):
                base_phi_idx = exp_idx * 2  # 每个扩展组合对应2个phi位置
                phi[base_phi_idx] = phi_special[i]
                phi[base_phi_idx + 1] = phi_special[i] + np.pi
                valid_mask[base_phi_idx:base_phi_idx + 2] = True
        
        # 处理一般情况
        if np.any(normal):
            phi_normal = np.arctan2(B[normal], -A[normal])
            normal_indices = np.where(normal)[0]
            for i, exp_idx in enumerate(normal_indices):
                base_phi_idx = exp_idx * 2
                phi[base_phi_idx] = phi_normal[i]
                phi[base_phi_idx + 1] = phi_normal[i] + np.pi
                valid_mask[base_phi_idx:base_phi_idx + 2] = True
        
        # 将phi值规范化到[-π, π]范围
        phi = np.where(phi > np.pi, phi - 2*np.pi, phi)
        phi = np.where(phi <= -np.pi, phi + 2*np.pi, phi)
        
        return phi, valid_mask
    
    def _compute_collision_and_key_regularized(self, phi: np.ndarray, valid_mask: np.ndarray,
                                             expanded: np.ndarray, case_type: str) -> tuple:
        """
        计算碰撞三角形和关键量（规则化版本）
        
        Args:
            phi: phi值 (6*N,)
            valid_mask: 有效掩码 (6*N,)
            expanded: 扩展组合 (3*N, 3, 2)
            case_type: "horizontal" 或 "vertical"
            
        Returns:
            tuple: (colli, key)
            - colli: 碰撞三角形 (6*N, 3, 2)
            - key: 关键量 (6*N, 3)
        """
        N_expanded = len(expanded)  # 3*N
        N_base = N_expanded // 3  # N
        N_phi = 6 * N_base  # 6*N
        
        # self._log_debug(f"{case_type}碰撞三角形计算", f"N_expanded={N_expanded}, N_base={N_base}, N_phi={N_phi}")
        
        # 初始化规则化数组
        colli = np.full((N_phi, 3, 2), np.inf, dtype=np.float64)
        key = np.full((N_phi, 3), np.inf, dtype=np.float64)
        
        # 批量计算有效的碰撞三角形
        valid_indices = np.where(valid_mask)[0]
        # self._log_debug(f"{case_type}有效索引", f"数量={len(valid_indices)}, 索引={valid_indices}")
        
        if len(valid_indices) > 0:
            # 计算对应的扩展组合索引
            exp_indices = valid_indices // 2
            # self._log_debug(f"{case_type}扩展组合索引", f"exp_indices={exp_indices}")
            
            # 获取有效的phi值和组合
            valid_phi = phi[valid_indices]
            valid_combinations = expanded[exp_indices]
            # self._log_debug(f"{case_type}有效phi值详细", f"valid_phi={valid_phi}")
            # self._log_array_detailed(f"{case_type}有效组合详细", valid_combinations)
            
            # 批量计算旋转变换
            t_values = valid_combinations[:, :, 0]  # (N_valid, 3)
            theta_values = valid_combinations[:, :, 1]  # (N_valid, 3)
            # self._log_array_detailed(f"{case_type}t值详细", t_values)
            # self._log_array_detailed(f"{case_type}theta值详细", theta_values)
            
            # 广播计算：phi + theta
            phi_plus_theta = valid_phi[:, None] + theta_values  # (N_valid, 3)
            # self._log_array_detailed(f"{case_type}phi+theta详细", phi_plus_theta)
            
            # 批量计算x和y坐标
            x_coords = t_values * np.cos(phi_plus_theta)  # (N_valid, 3)
            y_coords = t_values * np.sin(phi_plus_theta)  # (N_valid, 3)
            # self._log_array_detailed(f"{case_type}x坐标详细", x_coords)
            # self._log_array_detailed(f"{case_type}y坐标详细", y_coords)
            
            # 将结果存储到规则化数组中
            colli[valid_indices, :, 0] = x_coords
            colli[valid_indices, :, 1] = y_coords
            
            # 批量计算关键量
            if case_type == "horizontal":
                # 水平边关键量计算
                t1 = y_coords[:, 2] - (y_coords[:, 1] + y_coords[:, 0]) / 2  # t1: yp2-(yp1+yp0)/2
                t2 = x_coords[:, 2] - x_coords[:, 0]  # t2: xp2-xp0
                t3 = x_coords[:, 2] - x_coords[:, 1]  # t3: xp2-xp1
                # self._log_debug("水平边关键量计算公式", "t1=yp2-(yp1+yp0)/2, t2=xp2-xp0, t3=xp2-xp1")
            else:  # vertical
                # 竖直边关键量计算
                t1 = x_coords[:, 2] - (x_coords[:, 1] + x_coords[:, 0]) / 2  # t1: xp2-(xp1+xp0)/2
                t2 = y_coords[:, 2] - y_coords[:, 0]  # t2: yp2-yp0
                t3 = y_coords[:, 2] - y_coords[:, 1]  # t3: yp2-yp1
                # self._log_debug("竖直边关键量计算公式", "t1=xp2-(xp1+xp0)/2, t2=yp2-yp0, t3=yp2-yp1")
            
            # self._log_array_detailed(f"{case_type}关键量t1详细", t1)
            # self._log_array_detailed(f"{case_type}关键量t2详细", t2)
            # self._log_array_detailed(f"{case_type}关键量t3详细", t3)
            
            key[valid_indices, 0] = t1
            key[valid_indices, 1] = t2
            key[valid_indices, 2] = t3
        
        valid_count = np.sum(valid_mask)
        # self._log_debug(f"{case_type}碰撞三角形和关键量计算完成", f"{valid_count}个有效项，规则化存储")
        return colli, key
    
    def _solve_batch_regularized(self, colli: np.ndarray, key: np.ndarray, 
                               phi: np.ndarray, valid_mask: np.ndarray, 
                               case_type: str) -> np.ndarray:
        """
        批量求解（规则化版本）
        
        Args:
            colli: 碰撞三角形 (6*N, 3, 2)
            key: 关键量 (6*N, 3)
            phi: phi值 (6*N,)
            valid_mask: 有效掩码 (6*N,)
            case_type: "horizontal" 或 "vertical"
            
        Returns:
            np.ndarray: 解数组 (6*N, 5) - [xmin, xmax, ymin, ymax, phi]
        """
        N_phi = len(phi)
        
        # 初始化解数组，用inf填充
        sol = np.full((N_phi, 5), np.inf, dtype=np.float64)
        
        # 只处理有效的项
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return sol
        
        # 提取有效数据
        valid_colli = colli[valid_indices]  # (N_valid, 3, 2)
        valid_key = key[valid_indices]      # (N_valid, 3)
        valid_phi = phi[valid_indices]      # (N_valid,)

        # 之后valid_mask会更新
        valid_mask_new = valid_mask.copy()
        
        # 批量计算解
        if case_type == "horizontal":
            sol_valid, valid_mask_new = self._solve_horizontal_batch_regularized(valid_colli, valid_key, valid_phi)
        else:  # vertical
            sol_valid, valid_mask_new = self._solve_vertical_batch_regularized(valid_colli, valid_key, valid_phi)
        
        # 将结果存储到规则化数组中
        sol[valid_indices] = sol_valid
        
        return sol, valid_mask_new
    
    def _solve_horizontal_batch_regularized(self, colli: np.ndarray, key: np.ndarray, 
                                          phi: np.ndarray) -> np.ndarray:
        """
        批量求解水平边情况（规则化版本）
        
        Args:
            colli: 有效碰撞三角形 (N_valid, 3, 2)
            key: 有效关键量 (N_valid, 3)
            phi: 有效phi值 (N_valid,)
            
        Returns:
            np.ndarray: 解数组 (N_valid, 5)
        """
        N_valid = len(colli)
        self._log_debug("水平边批量求解开始", f"处理{N_valid}个有效项")
        
        # 提取关键量
        t1_h = key[:, 0]  # (N_valid,)
        t2_h = key[:, 1]  # (N_valid,)
        t3_h = key[:, 2]  # (N_valid,)
        
        self._log_array_detailed("水平边t1关键量", t1_h)
        self._log_array_detailed("水平边t2关键量", t2_h)
        self._log_array_detailed("水平边t3关键量", t3_h)
        
        # 提取坐标
        xp0 = colli[:, 0, 0]  # (N_valid,)
        yp0 = colli[:, 0, 1]  # (N_valid,)
        xp1 = colli[:, 1, 0]  # (N_valid,)
        yp1 = colli[:, 1, 1]  # (N_valid,)
        xp2 = colli[:, 2, 0]  # (N_valid,)
        yp2 = colli[:, 2, 1]  # (N_valid,)
        
        self._log_array_detailed("水平边xp0", xp0)
        self._log_array_detailed("水平边yp0", yp0)
        self._log_array_detailed("水平边xp1", xp1)
        self._log_array_detailed("水平边yp1", yp1)
        self._log_array_detailed("水平边xp2", xp2)
        self._log_array_detailed("水平边yp2", yp2)
        
        # 筛选条件：abs(t1)>n+tol 或 abs(t2)>m+tol 或 abs(t3)>m+tol
        invalid_mask = ((np.abs(t1_h) > self.n + self.tolerance) |
                       (np.abs(t2_h) > self.m + self.tolerance) |
                       (np.abs(t3_h) > self.m + self.tolerance))
        
        # 判断t1是否等于n或-n
        boundary_mask = np.abs(np.abs(t1_h) - self.n) <= self.tolerance
        
        # 判断t2, t3符号是否一致且都非零
        same_sign_mask = ((t2_h > 0) & (t3_h > 0)) | ((t2_h < 0) & (t3_h < 0))
        
        self._log_array_detailed("水平边无效掩码", invalid_mask)
        self._log_array_detailed("水平边边界掩码", boundary_mask)
        self._log_array_detailed("水平边同符号掩码", same_sign_mask)
        
        # 初始化解数组
        sol = np.full((N_valid, 5), np.inf, dtype=np.float64)
        
        # 边界情况处理
        boundary_indices = np.where(boundary_mask & (~invalid_mask))[0]
        self._log_debug("水平边边界情况", f"处理{len(boundary_indices)}个边界项，索引={boundary_indices}")
        
        if len(boundary_indices) > 0:
            y_center = (self.n - (yp1[boundary_indices] + yp0[boundary_indices])/2 - yp2[boundary_indices]) / 2
            x_max = self.m - np.maximum.reduce([xp0[boundary_indices], 
                                               xp1[boundary_indices], 
                                               xp2[boundary_indices]])
            x_min = 0 - np.minimum.reduce([xp0[boundary_indices], 
                                          xp1[boundary_indices], 
                                          xp2[boundary_indices]])
            
            self._log_array_detailed("水平边边界y_center", y_center)
            self._log_array_detailed("水平边边界x_max", x_max)
            self._log_array_detailed("水平边边界x_min", x_min)
            
            sol[boundary_indices, 0] = x_min - self.tolerance  # xmin
            sol[boundary_indices, 1] = x_max + self.tolerance  # xmax
            sol[boundary_indices, 2] = y_center - self.tolerance  # ymin
            sol[boundary_indices, 3] = y_center + self.tolerance  # ymax
            sol[boundary_indices, 4] = phi[boundary_indices]   # phi
        
        # 普通情况处理
        normal_indices = np.where(same_sign_mask & (~boundary_mask) & (~invalid_mask))[0]
        self._log_debug("水平边普通情况", f"处理{len(normal_indices)}个普通项，索引={normal_indices}")
        
        if len(normal_indices) > 0:
            # 计算x_center
            x_center = np.where(t2_h[normal_indices] > 0, 
                               self.m - xp2[normal_indices], 
                               -xp2[normal_indices])
            
            # 计算y_center
            y_center = np.where(t1_h[normal_indices] > 0,
                               -(yp0[normal_indices] + yp1[normal_indices]) / 2,
                               self.n - (yp0[normal_indices] + yp1[normal_indices]) / 2)
            
            self._log_array_detailed("水平边普通x_center", x_center)
            self._log_array_detailed("水平边普通y_center", y_center)
            
            # 检查center是否在矩形框内
            valid_center_mask = ((0 <= x_center) & (x_center <= self.m) & 
                               (0 <= y_center) & (y_center <= self.n))
            
            self._log_array_detailed("水平边center有效掩码", valid_center_mask)
            
            final_indices = normal_indices[valid_center_mask]
            self._log_debug("水平边最终有效项", f"数量={len(final_indices)}, 索引={final_indices}")
            
            if len(final_indices) > 0:
                valid_x_center = x_center[valid_center_mask]
                valid_y_center = y_center[valid_center_mask]
                
                sol[final_indices, 0] = valid_x_center - self.tolerance  # xmin
                sol[final_indices, 1] = valid_x_center + self.tolerance  # xmax
                sol[final_indices, 2] = valid_y_center - self.tolerance  # ymin
                sol[final_indices, 3] = valid_y_center + self.tolerance  # ymax
                sol[final_indices, 4] = phi[final_indices]              # phi
        
        # 输出最终解的详细信息
        valid_sol_count = np.sum(~np.isinf(sol[:, 0]))
        self._log_debug("水平边求解完成", f"得到{valid_sol_count}个有效解")
        
        # 计算所有合法解的掩码——通过中间量掩码计算
        final_mask = boundary_mask & (~invalid_mask) | same_sign_mask & (~boundary_mask) & (~invalid_mask)
        self._log_array_detailed("水平边最终解掩码", final_mask)

        return sol, final_mask
    
    def _solve_vertical_batch_regularized(self, colli: np.ndarray, key: np.ndarray, 
                                        phi: np.ndarray) -> np.ndarray:
        """
        批量求解竖直边情况（规则化版本）
        
        Args:
            colli: 有效碰撞三角形 (N_valid, 3, 2)
            key: 有效关键量 (N_valid, 3)
            phi: 有效phi值 (N_valid,)
            
        Returns:
            np.ndarray: 解数组 (N_valid, 5)
        """
        N_valid = len(colli)
        
        # 提取关键量
        t1_v = key[:, 0]  # (N_valid,)
        t2_v = key[:, 1]  # (N_valid,)
        t3_v = key[:, 2]  # (N_valid,)
        
        # 提取坐标
        xp0 = colli[:, 0, 0]  # (N_valid,)
        yp0 = colli[:, 0, 1]  # (N_valid,)
        xp1 = colli[:, 1, 0]  # (N_valid,)
        yp1 = colli[:, 1, 1]  # (N_valid,)
        xp2 = colli[:, 2, 0]  # (N_valid,)
        yp2 = colli[:, 2, 1]  # (N_valid,)
        
        # 筛选条件：abs(t1)>m+tol 或 abs(t2)>n+tol 或 abs(t3)>n+tol (注意m,n互换)
        invalid_mask = ((np.abs(t1_v) > self.m + self.tolerance) |
                       (np.abs(t2_v) > self.n + self.tolerance) |
                       (np.abs(t3_v) > self.n + self.tolerance))
        
        # 判断t1是否等于m或-m
        boundary_mask = np.abs(np.abs(t1_v) - self.m) <= self.tolerance
        
        # 判断t2, t3符号是否一致且都非零
        same_sign_mask = ((t2_v > 0) & (t3_v > 0)) | ((t2_v < 0) & (t3_v < 0))
        
        # 初始化解数组
        sol = np.full((N_valid, 5), np.inf, dtype=np.float64)
        
        # 边界情况处理
        boundary_indices = np.where(boundary_mask & (~invalid_mask))[0]
        if len(boundary_indices) > 0:
            x_center = (self.m - (xp1[boundary_indices] + xp2[boundary_indices]) / 2 - xp0[boundary_indices]) / 2
            y_max = self.n - np.maximum.reduce([yp0[boundary_indices], 
                                               yp1[boundary_indices], 
                                               yp2[boundary_indices]])
            y_min = 0 - np.minimum.reduce([yp0[boundary_indices], 
                                          yp1[boundary_indices], 
                                          yp2[boundary_indices]])
            
            sol[boundary_indices, 0] = x_center - self.tolerance  # xmin
            sol[boundary_indices, 1] = x_center + self.tolerance  # xmax
            sol[boundary_indices, 2] = y_min - self.tolerance     # ymin
            sol[boundary_indices, 3] = y_max + self.tolerance     # ymax
            sol[boundary_indices, 4] = phi[boundary_indices]      # phi
        
        # 普通情况处理
        normal_indices = np.where(same_sign_mask & (~boundary_mask) & (~invalid_mask))[0]
        if len(normal_indices) > 0:
            # 计算y_center
            y_center = np.where(t2_v[normal_indices] > 0, 
                               self.n - yp2[normal_indices], 
                               -yp2[normal_indices])
            
            # 计算x_center
            x_center = np.where(t1_v[normal_indices] > 0,
                               -(xp0[normal_indices] + xp1[normal_indices]) / 2,
                               self.m - (xp0[normal_indices] + xp1[normal_indices]) / 2)
            
            # 检查center是否在矩形框内
            valid_center_mask = ((0 <= x_center) & (x_center <= self.m) & 
                               (0 <= y_center) & (y_center <= self.n))
            
            final_indices = normal_indices[valid_center_mask]
            if len(final_indices) > 0:
                valid_x_center = x_center[valid_center_mask]
                valid_y_center = y_center[valid_center_mask]
                
                sol[final_indices, 0] = valid_x_center - self.tolerance  # xmin
                sol[final_indices, 1] = valid_x_center + self.tolerance  # xmax
                sol[final_indices, 2] = valid_y_center - self.tolerance  # ymin
                sol[final_indices, 3] = valid_y_center + self.tolerance  # ymax
                sol[final_indices, 4] = phi[final_indices]              # phi
        
        # 最终掩码
        final_mask = boundary_mask & (~invalid_mask) | same_sign_mask & (~boundary_mask) & (~invalid_mask)

        return sol, final_mask
    
    def _merge_solutions_regularized(self, sol_h: np.ndarray, sol_v: np.ndarray,
                               valid_h: np.ndarray, valid_v: np.ndarray,
                               indices: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        完全向量化合并水平边和竖直边的解及掩码
        
        Args:
            sol_h: 水平边解 (6*N, 5)
            sol_v: 竖直边解 (6*N, 5)
            valid_h: 水平边有效掩码 (6*N,)
            valid_v: 竖直边有效掩码 (6*N,)
            indices: 原始组合编号追踪 (3*N,)
            N: 原始组合数量
            
        Returns:
            tuple: (final_sol, final_valid)
            - final_sol: 合并后的解 (N, 12, 5)
            - final_valid: 合并后的掩码 (N, 12)
        """
        # 预分配结果数组
        final_sol = np.full((N, 12, 5), np.inf, dtype=np.float64)
        final_valid = np.zeros((N, 12), dtype=bool)
        
        # 计算每个原始组合对应的解索引
        combo_indices = np.arange(N)
        
        # 水平解 (6*N -> N×6)
        h_sol_reshaped = sol_h.reshape(N, 6, 5)
        h_valid_reshaped = valid_h.reshape(N, 6)
        
        # 竖直解 (6*N -> N×6)
        v_sol_reshaped = sol_v.reshape(N, 6, 5)
        v_valid_reshaped = valid_v.reshape(N, 6)
        
        # 合并解 (N×12×5)
        final_sol[:, :6, :] = h_sol_reshaped
        final_sol[:, 6:, :] = v_sol_reshaped
        
        # 合并掩码 (N×12)
        final_valid[:, :6] = h_valid_reshaped
        final_valid[:, 6:] = v_valid_reshaped
        
        return final_sol, final_valid
