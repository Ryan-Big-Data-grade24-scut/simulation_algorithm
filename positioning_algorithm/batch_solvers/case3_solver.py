"""
Case3批处理求解器
处理三点分别在三条边的情况
"""
import numpy as np
import os
from typing import List, Tuple
import logging
from .trig_cache import trig_cache

class Case3BatchSolver:
    """Case3批处理求解器类"""
    
    def __init__(self, m: float, n: float, tolerance: float = 1e-3, 
                 enable_ros_logging: bool = False, ros_logger=None):
        """
        初始化Case3求解器
        
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
        self.enable_ros_logging = enable_ros_logging
        self.ros_logger = ros_logger

        if self.ros_logger:
            # 如果启用ROS日志，我们就创建一个handler
            self.logger = self.ros_logger.get_logger("solver.case3", "batches/case3")
        else:
            # 设置日志系统
            self._setup_logging()
        
        self._log_info("Case3BatchSolver初始化完成", 
                      f"场地尺寸: {m}x{n}, 容差: {tolerance}")
    
    def _setup_logging(self):
        """设置兼容ROS和Windows的日志系统"""
        # 创建logs目录
        os.makedirs('logs', exist_ok=True)
        
        # 初始化标准日志器
        self.logger = logging.getLogger("Case3BatchSolver")
        
        # 清理已有的handlers，避免重复添加
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                self.logger.removeHandler(handler)
                
        self.enable_ros_logging = False
        # 根据ROS状态设置日志等级 
        self.min_log_level = logging.DEBUG    # Windows调试模式下输出详细日志
        
        # 添加文件日志器（Windows调试用）
        if not self.enable_ros_logging:
            handler = logging.FileHandler("logs/case3_batch_solver.log", encoding="utf-8")
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
    
    def _log_layer_summary(self, layer_name: str, data_dict: dict):
        """输出每一层的详细数组内容"""
        # 性能优化：注释掉所有详细日志输出
        pass
        # self._log_debug(f"=== {layer_name} 详细输出 ===")
        # for key, value in data_dict.items():
        #     if isinstance(value, np.ndarray):
        #         if value.size == 0:
        #             self._log_debug(f"  {key}: 空数组")
        #         else:
        #             self._log_debug(f"  {key}: 形状{value.shape}, 类型{value.dtype}")
        #             # 显示所有元素
        #             if value.ndim == 1:
        #                 # 一维数组：逐个显示
        #                 for i in range(len(value)):
        #                     self._log_debug(f"    [{i}]: {value[i]}")
        #             elif value.ndim == 2:
        #                 # 二维数组：按行显示
        #                 for i in range(value.shape[0]):
        #                     self._log_debug(f"    [{i}]: {value[i]}")
        #             elif value.ndim == 3:
        #                 # 三维数组：分层显示
        #                 for i in range(value.shape[0]):
        #                     self._log_debug(f"    [{i}]: 形状{value[i].shape}")
        #                     for j in range(value.shape[1]):
        #                         self._log_debug(f"      [{i}][{j}]: {value[i][j]}")
        #             else:
        #                 # 更高维数组：显示扁平化前几个元素
        #                 flat_arr = value.flatten()
        #                 for i in range(min(20, len(flat_arr))):
        #                     self._log_debug(f"    flat[{i}]: {flat_arr[i]}")
        #                 if len(flat_arr) > 20:
        #                     self._log_debug(f"    ... (还有{len(flat_arr)-20}个元素)")
        #     elif isinstance(value, list):
        #         if len(value) == 0:
        #             self._log_debug(f"  {key}: 空列表")
        #         else:
        #             self._log_debug(f"  {key}: 列表长度{len(value)}")
        #             for i in range(len(value)):
        #                 if isinstance(value[i], list):
        #                     self._log_debug(f"    [{i}]: 子列表长度{len(value[i])}")
        #                     for j in range(len(value[i])):
        #                         self._log_debug(f"      [{i}][{j}]: {value[i][j]}")
        #                 else:
        #                     self._log_debug(f"    [{i}]: {value[i]}")
        #     else:
        #         self._log_debug(f"  {key}: {value}")
        # self._log_debug("")

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
        
        self._log_debug(name, info)
    
    def solve(self, combinations: np.ndarray) -> List[Tuple]:
        """
        求解Case3批处理 - 严格按照规则化数据流程表
        
        数据流程：
        N -> 6N -> 6N -> 12N -> 12N -> 24N
        
        Args:
            combinations: 激光组合数组 (N, 3, 2) - 每行为[[t0,θ0], [t1,θ1], [t2,θ2]]
            
        Returns:
            List[Tuple]: 解列表，每个解为 ((x_min, x_max), (y_min, y_max), phi)
        """
        # self._log_info("=" * 60)
        # self._log_info("开始Case3批处理求解 - 规则化数据流程")
        # self._log_info("=" * 60)
        
        # 输入验证和日志
        if len(combinations) == 0:
            # self._log_warning("没有组合可求解")
            return []
        
        N = len(combinations)
        # self._log_info(f"数据流程开始", f"输入{N}个组合 -> 目标24N={24*N}个最终候选解")
        
        # 输入数据详细日志
        # self._log_debug("=== 输入数据详细分析 ===")
        # self._log_array_detailed("输入combinations", combinations)
        
        # 第一层：扩展组合 N -> 6N
        # self._log_debug("第一层: 组合扩展 N->6N")
        expanded_combinations = self._expand_combinations(combinations)  # (6N, 3, 2)
        
        # 第一层数据详细日志
        # self._log_debug("=== 第一层输出详细分析 ===")
        # self._log_array_detailed("expanded_combinations", expanded_combinations)
        
        # 第二-五层：计算phi 6N -> 6N
        # self._log_debug("第二-五层: 计算phi 6N->6N")
        phi_h, phi_v, valid_phi_h, valid_phi_v = self._compute_phi_regularized(expanded_combinations)  # (6N, 2), (6N,)
        
        # 第二-五层数据详细日志
        # self._log_debug("=== 第二-五层输出详细分析 ===")
        # self._log_array_detailed("phi_h", phi_h)
        # self._log_array_detailed("phi_v", phi_v) 
        # self._log_array_detailed("valid_phi_h", valid_phi_h)
        # self._log_array_detailed("valid_phi_v", valid_phi_v)
        
        # 第六-七层：计算碰撞和关键量 6N -> 12N
        # self._log_debug("第六-七层: 计算碰撞和关键量 6N->12N")
        colli_h, key_h, valid_phi_flat_h = self._compute_collision_and_key_h(phi_h, expanded_combinations, valid_phi_h)  # (12N, 3, 2), (12N, 4), (12N,)
        colli_v, key_v, valid_phi_flat_v = self._compute_collision_and_key_v(phi_v, expanded_combinations, valid_phi_v)  # (12N, 3, 2), (12N, 4), (12N,)
        
        # 第六-七层数据详细日志
        # self._log_debug("=== 第六-七层输出详细分析 ===")
        # self._log_array_detailed("colli_h", colli_h)
        # self._log_array_detailed("key_h", key_h)
        # self._log_array_detailed("valid_phi_flat_h", valid_phi_flat_h)
        # self._log_array_detailed("colli_v", colli_v)
        # self._log_array_detailed("key_v", key_v)
        # self._log_array_detailed("valid_phi_flat_v", valid_phi_flat_v)
        
        # 第八层：求解 12N -> 12N
        # self._log_debug("第八层: 求解 12N->12N")
        sols_h, valid_sol_h = self._solve_regularized_h(key_h, colli_h, phi_h, valid_phi_flat_h)  # (12N, 5), (12N,)
        sols_v, valid_sol_v = self._solve_regularized_v(key_v, colli_v, phi_v, valid_phi_flat_v)  # (12N, 5), (12N,)
        
        # 第八层数据详细日志
        # self._log_debug("=== 第八层输出详细分析 ===")
        # self._log_array_detailed("sols_h", sols_h)
        # self._log_array_detailed("valid_sol_h", valid_sol_h)
        # self._log_array_detailed("sols_v", sols_v)
        # self._log_array_detailed("valid_sol_v", valid_sol_v)
        
        # 第九层：合并解 12N+12N -> 24N
        # self._log_debug("第九层: 合并解 12N+12N->24N")
        final_sols, final_valid = self._merge_solutions_regularized(sols_h, sols_v, valid_sol_h, valid_sol_v, N)  # (24N, 5), (24N,)
        
        # 第九层数据详细日志
        # self._log_debug("=== 第九层最终输出详细分析 ===")
        # self._log_array_detailed("final_sols", final_sols)
        # self._log_array_detailed("final_valid", final_valid)
        
        return final_sols, final_valid
    
    # ==================== 第一层：组合扩展 N->6N ====================
    def _expand_combinations(self, combinations: np.ndarray) -> np.ndarray:
        """
        第一层：扩展组合，从N -> 6N (规则化存储，无需索引追踪)
        
        Args:
            combinations: 输入组合 (N, 3, 2)
            
        Returns:
            expanded_combinations: 扩展后组合 (6N, 3, 2)
            
        位置映射规则：
        - 原组合i的6种排列存储在位置[6*i : 6*(i+1)]
        - 排列0: [6*i+0] = (0,1,2)  
        - 排列1: [6*i+1] = (0,2,1)
        - 排列2: [6*i+2] = (1,0,2)
        - 排列3: [6*i+3] = (1,2,0)
        - 排列4: [6*i+4] = (2,0,1)
        - 排列5: [6*i+5] = (2,1,0)
        """
        N = len(combinations)
        # self._log_debug(f"第一层扩展组合", f"输入{N}个组合，将扩展为{6*N}个")
        
        # 初始化扩展后的数组
        expanded_combinations = np.zeros((6*N, 3, 2))
        
        # 定义6种排列组合
        permutations = [
            [0, 1, 2],  # 原始顺序
            [0, 2, 1],  # 交换1,2
            [1, 0, 2],  # 交换0,1
            [1, 2, 0],  # 循环左移
            [2, 0, 1],  # 循环右移
            [2, 1, 0]   # 交换0,2
        ]
        
        # 批量生成所有排列
        for i in range(N):
            for perm_idx, perm in enumerate(permutations):
                expanded_idx = i * 6 + perm_idx
                expanded_combinations[expanded_idx] = combinations[i][perm]
        
        # self._log_debug("第一层扩展完成", 
        #                f"生成{len(expanded_combinations)}个扩展组合，形状: {expanded_combinations.shape}")
        
        return expanded_combinations
    
    # ==================== 第二-五层：计算phi 6N->6N ====================
    def _compute_phi_regularized(self, expanded_combinations: np.ndarray) -> tuple:
        """
        第二-五层合并：从A,B系数计算到phi (6N -> 6N)
        
        Args:
            expanded_combinations: 扩展组合 (6N, 3, 2)
            
        Returns:
            tuple: (phi_h, phi_v, valid_phi_h, valid_phi_v)
            - phi_h: 水平phi值 (6N, 2)
            - phi_v: 竖直phi值 (6N, 2)  
            - valid_phi_h: 水平phi有效掩码 (6N,)
            - valid_phi_v: 竖直phi有效掩码 (6N,)
        """
        # self._log_debug("第二-五层合并: 计算phi")
        
        N_exp = len(expanded_combinations)  # 6N
        
        # 提取激光参数 (只用前两束激光)
        t0 = expanded_combinations[:, 0, 0]  # (6N,)
        theta0 = expanded_combinations[:, 0, 1]  # (6N,)
        t1 = expanded_combinations[:, 1, 0]  # (6N,)
        theta1 = expanded_combinations[:, 1, 1]  # (6N,)
        
        # 第二层：计算A、B系数
        # 水平模式
        A_h = t0 * np.cos(theta0) - t1 * np.cos(theta1)
        B_h = t0 * np.sin(theta0) - t1 * np.sin(theta1)
        # 竖直模式
        A_v = t1 * np.sin(theta1) - t0 * np.sin(theta0)
        B_v = t0 * np.cos(theta0) - t1 * np.cos(theta1)
        
        # 第三层：计算norm
        norm_h = np.sqrt(A_h**2 + B_h**2)
        norm_v = np.sqrt(A_v**2 + B_v**2)
        
        # 第四层：计算alpha和arcsin
        # alpha计算
        valid_alpha_h = (A_h != 0) | (B_h != 0)
        valid_alpha_v = (A_v != 0) | (B_v != 0)
        
        alpha_h = np.zeros_like(A_h)
        alpha_v = np.zeros_like(A_v)
        alpha_h[valid_alpha_h] = np.arctan2(B_h[valid_alpha_h], A_h[valid_alpha_h])
        alpha_v[valid_alpha_v] = np.arctan2(B_v[valid_alpha_v], A_v[valid_alpha_v])
        
        # arcsin计算
        valid_arcsin_h = (norm_h > 0) & (self.n <= norm_h)
        valid_arcsin_v = (norm_v > 0) & (self.m <= norm_v)
        
        arcsin_h = np.zeros_like(norm_h)
        arcsin_v = np.zeros_like(norm_v)
        arcsin_h[valid_arcsin_h] = np.arcsin(np.clip(self.n / norm_h[valid_arcsin_h], -1.0, 1.0))
        arcsin_v[valid_arcsin_v] = np.arcsin(np.clip(self.m / norm_v[valid_arcsin_v], -1.0, 1.0))
        
        # 第五层：计算phi
        valid_phi_h = valid_alpha_h & valid_arcsin_h  # (6N,)
        valid_phi_v = valid_alpha_v & valid_arcsin_v  # (6N,)
        
        # 初始化phi数组
        phi_h = np.zeros((N_exp, 2))  # (6N, 2)
        phi_v = np.zeros((N_exp, 2))  # (6N, 2)
        
        # 计算有效的phi值
        if np.any(valid_phi_h):
            phi_h[valid_phi_h, 0] = arcsin_h[valid_phi_h] - alpha_h[valid_phi_h]
            phi_h[valid_phi_h, 1] = np.pi - arcsin_h[valid_phi_h] - alpha_h[valid_phi_h]
        
        if np.any(valid_phi_v):
            phi_v[valid_phi_v, 0] = arcsin_v[valid_phi_v] - alpha_v[valid_phi_v]
            phi_v[valid_phi_v, 1] = np.pi - arcsin_v[valid_phi_v] - alpha_v[valid_phi_v]
        
        # 统计
        valid_h_count = np.sum(valid_phi_h)
        valid_v_count = np.sum(valid_phi_v)
        # self._log_debug("第二-五层完成", 
        #                f"phi_h有效: {valid_h_count}/{N_exp}, phi_v有效: {valid_v_count}/{N_exp}")
        return phi_h, phi_v, valid_phi_h, valid_phi_v
        
    # ==================== 第六-七层：碰撞和关键量 6N->12N ====================
    def _compute_collision_and_key_h(self, phi_h: np.ndarray, expanded_combinations: np.ndarray, 
                                   valid_phi_h: np.ndarray) -> tuple:
        """
        第六-七层水平：计算碰撞三角形和关键量 (6N -> 12N)
        
        Args:
            phi_h: 水平phi值 (6N, 2)
            expanded_combinations: 扩展组合 (6N, 3, 2)
            valid_phi_h: phi有效掩码 (6N,)
            
        Returns:
            tuple: (colli_h, key_h, valid_phi_flat_h)
            - colli_h: 碰撞三角形 (12N, 3, 2)
            - key_h: 关键量 (12N, 4)
            - valid_phi_flat_h: 扁平化phi有效掩码 (12N,)
        """
        # self._log_debug("第六-七层水平: 计算碰撞和关键量")
        
        N_exp = len(phi_h)  # 6N
        N_flat = 2 * N_exp  # 12N
        
        # 将phi展平为12N
        phi_flat = phi_h.flatten()  # (12N,)
        
        # 扩展valid_phi掩码：每个6N位置对应2个12N位置
        valid_phi_flat_h = np.repeat(valid_phi_h, 2)  # (12N,)
        
        # 扩展组合：每个组合重复2次匹配2个phi值
        combinations_expanded = np.repeat(expanded_combinations, 2, axis=0)  # (12N, 3, 2)
        
        # 初始化结果数组
        colli_h = np.zeros((N_flat, 3, 2))  # (12N, 3, 2)
        key_h = np.zeros((N_flat, 4))  # (12N, 4)
        
        # 只处理有效的phi
        if np.any(valid_phi_flat_h):
            valid_mask = valid_phi_flat_h
            
            # 提取有效数据
            valid_phi = phi_flat[valid_mask]
            valid_combinations = combinations_expanded[valid_mask]
            
            # 提取激光参数
            t_values = valid_combinations[:, :, 0]  # (N_valid, 3)
            theta_values = valid_combinations[:, :, 1]  # (N_valid, 3)
            
            # 计算碰撞三角形坐标
            phi_plus_theta = valid_phi[:, None] + theta_values  # (N_valid, 3)
            x_coords = t_values * np.cos(phi_plus_theta)  # (N_valid, 3)
            y_coords = t_values * np.sin(phi_plus_theta)  # (N_valid, 3)
            
            # 组合为碰撞三角形
            valid_colli = np.stack([x_coords, y_coords], axis=2)  # (N_valid, 3, 2)
            
            # 计算关键量 (水平模式)
            x0, y0 = valid_colli[:, 0, 0], valid_colli[:, 0, 1]
            x1, y1 = valid_colli[:, 1, 0], valid_colli[:, 1, 1]  
            x2, y2 = valid_colli[:, 2, 0], valid_colli[:, 2, 1]
            
            valid_key = np.column_stack([
                x2 - x0,  # key0
                x2 - x1,  # key1  
                y2 - y0,  # key2
                y2 - y1   # key3
            ])  # (N_valid, 4)
            
            # 填入结果数组
            colli_h[valid_mask] = valid_colli
            key_h[valid_mask] = valid_key
        
        valid_count = np.sum(valid_phi_flat_h)
        # self._log_debug("第六-七层水平完成", 
        #                f"生成{N_flat}个位置，有效{valid_count}个")
        
        return colli_h, key_h, valid_phi_flat_h
    
    def _compute_collision_and_key_v(self, phi_v: np.ndarray, expanded_combinations: np.ndarray,
                                   valid_phi_v: np.ndarray) -> tuple:
        """
        第六-七层竖直：计算碰撞三角形和关键量 (6N -> 12N)
        
        Args:
            phi_v: 竖直phi值 (6N, 2)
            expanded_combinations: 扩展组合 (6N, 3, 2)
            valid_phi_v: phi有效掩码 (6N,)
            
        Returns:
            tuple: (colli_v, key_v, valid_phi_flat_v)
            - colli_v: 碰撞三角形 (12N, 3, 2)
            - key_v: 关键量 (12N, 4)
            - valid_phi_flat_v: 扁平化phi有效掩码 (12N,)
        """
        # self._log_debug("第六-七层竖直: 计算碰撞和关键量")
        
        N_exp = len(phi_v)  # 6N
        N_flat = 2 * N_exp  # 12N
        
        # 将phi展平为12N
        phi_flat = phi_v.flatten()  # (12N,)
        
        # 扩展valid_phi掩码：每个6N位置对应2个12N位置
        valid_phi_flat_v = np.repeat(valid_phi_v, 2)  # (12N,)
        
        # 扩展组合：每个组合重复2次匹配2个phi值
        combinations_expanded = np.repeat(expanded_combinations, 2, axis=0)  # (12N, 3, 2)
        
        # 初始化结果数组
        colli_v = np.zeros((N_flat, 3, 2))  # (12N, 3, 2)
        key_v = np.zeros((N_flat, 4))  # (12N, 4)
        
        # 只处理有效的phi
        if np.any(valid_phi_flat_v):
            valid_mask = valid_phi_flat_v
            
            # 提取有效数据
            valid_phi = phi_flat[valid_mask]
            valid_combinations = combinations_expanded[valid_mask]
            
            # 提取激光参数
            t_values = valid_combinations[:, :, 0]  # (N_valid, 3)
            theta_values = valid_combinations[:, :, 1]  # (N_valid, 3)
            
            # 计算碰撞三角形坐标
            phi_plus_theta = valid_phi[:, None] + theta_values  # (N_valid, 3)
            x_coords = t_values * np.cos(phi_plus_theta)  # (N_valid, 3)
            y_coords = t_values * np.sin(phi_plus_theta)  # (N_valid, 3)
            
            # 组合为碰撞三角形
            valid_colli = np.stack([x_coords, y_coords], axis=2)  # (N_valid, 3, 2)
            
            # 计算关键量 (竖直模式)
            x0, y0 = valid_colli[:, 0, 0], valid_colli[:, 0, 1]
            x1, y1 = valid_colli[:, 1, 0], valid_colli[:, 1, 1]
            x2, y2 = valid_colli[:, 2, 0], valid_colli[:, 2, 1]
            
            valid_key = np.column_stack([
                y2 - y0,  # key0
                y2 - y1,  # key1
                x2 - x0,  # key2  
                x2 - x1   # key3
            ])  # (N_valid, 4)
            
            # 填入结果数组
            colli_v[valid_mask] = valid_colli
            key_v[valid_mask] = valid_key
        
        valid_count = np.sum(valid_phi_flat_v)
        self._log_debug("第六-七层竖直完成", 
                       f"生成{N_flat}个位置，有效{valid_count}个")
        return colli_v, key_v, valid_phi_flat_v
        
    # ==================== 第八层：求解 12N->12N ====================
    def _solve_regularized_h(self, key_h: np.ndarray, colli_h: np.ndarray, phi_h: np.ndarray,
                           valid_phi_flat_h: np.ndarray) -> tuple:
        """
        第八层水平：规则化求解 (12N -> 12N)
        
        Args:
            key_h: 关键量 (12N, 4)
            colli_h: 碰撞三角形 (12N, 3, 2)
            phi_h: 原始phi数组 (6N, 2) - 用于获取phi值
            valid_phi_flat_h: phi有效掩码 (12N,)
            
        Returns:
            tuple: (sols_h, valid_sol_h)
            - sols_h: 解数组 (12N, 5) - [x_min, x_max, y_min, y_max, phi]
            - valid_sol_h: 解有效掩码 (12N,)
        """
        self._log_debug("第八层水平: 规则化求解")
        
        N_flat = len(key_h)  # 12N
        
        # 初始化解数组
        sols_h = np.zeros((N_flat, 5))  # (12N, 5)
        valid_sol_h = np.zeros(N_flat, dtype=bool)  # (12N,)
        
        # 只处理有效的phi位置
        if np.any(valid_phi_flat_h):
            valid_indices = np.where(valid_phi_flat_h)[0]
            
            # 提取有效数据
            valid_key = key_h[valid_indices]  # (N_valid, 4)
            valid_colli = colli_h[valid_indices]  # (N_valid, 3, 2)
            
            # 提取关键量
            key0 = valid_key[:, 0]  # x2 - x0
            key1 = valid_key[:, 1]  # x2 - x1
            key2 = valid_key[:, 2]  # y2 - y0
            key3 = valid_key[:, 3]  # y2 - y1
            
            # 提取坐标
            x0 = valid_colli[:, 0, 0]
            x2 = valid_colli[:, 2, 0]
            y0 = valid_colli[:, 0, 1]
            y2 = valid_colli[:, 2, 1]
            
            # 水平模式求解条件判断 (恢复原逻辑)
            # 条件1：前两个关键量同号检查
            same_sign_01 = (key0 * key1) > 0
            
            # 条件2：后两个关键量绝对值最大值检查（筛掉不合理的）
            max_abs_23 = np.maximum(np.abs(key2), np.abs(key3))
            valid_range_23 = max_abs_23 < (self.n - self.tolerance)
            
            # 综合条件筛选
            valid_solve_mask = same_sign_01 & valid_range_23
            
            if np.any(valid_solve_mask):
                solve_indices = valid_indices[valid_solve_mask]
                solve_key0 = key0[valid_solve_mask]
                solve_key1 = key1[valid_solve_mask]
                solve_x0 = x0[valid_solve_mask]
                solve_x2 = x2[valid_solve_mask]
                solve_y0 = y0[valid_solve_mask]
                solve_y2 = y2[valid_solve_mask]
                
                # 批量计算解中心 (恢复原逻辑)
                # 判断key0是正还是负，计算x_center
                is_positive = solve_key0 > 0
                x_center = np.where(is_positive, self.m - solve_x2, -solve_x2)
                
                # 计算y_center (使用y0和y1)
                solve_y1 = valid_colli[:, 1, 1][valid_solve_mask]
                y_center = (self.n - solve_y0 - solve_y1) / 2
                
                # 计算解边界 (使用tolerance)
                x_min = x_center - self.tolerance
                x_max = x_center + self.tolerance
                y_min = y_center - self.tolerance
                y_max = y_center + self.tolerance
                
                # 检查解的合理性（加入容错）
                bounds_valid = ((x_min >= -self.tolerance) & (x_max <= self.m + self.tolerance) & 
                               (y_min >= -self.tolerance) & (y_max <= self.n + self.tolerance))
                
                # 最终有效性：满足关键量条件且边界合理
                final_valid_mask = bounds_valid
                final_solve_indices = solve_indices[final_valid_mask]
                
                if len(final_solve_indices) > 0:
                    # 获取对应的phi值 (从12N索引转换为6N索引再转换为phi值)
                    phi_flat = phi_h.flatten()  # (12N,)
                    solve_phi_values = phi_flat[final_solve_indices]
                    
                    # 构造解
                    final_x_min = x_min[final_valid_mask]
                    final_x_max = x_max[final_valid_mask]
                    final_y_min = y_min[final_valid_mask]
                    final_y_max = y_max[final_valid_mask]
                    
                    valid_sols = np.column_stack([final_x_min, final_x_max, final_y_min, final_y_max, solve_phi_values])
                    
                    # 填入结果
                    sols_h[final_solve_indices] = valid_sols
                    valid_sol_h[final_solve_indices] = True
        
        valid_count = np.sum(valid_sol_h)
        self._log_debug("第八层水平完成", 
                       f"从{N_flat}个位置中求得{valid_count}个有效解")
        
        return sols_h, valid_sol_h
    
    def _solve_regularized_v(self, key_v: np.ndarray, colli_v: np.ndarray,
                           phi_v: np.ndarray, valid_phi_flat_v: np.ndarray) -> tuple:
        """
        第八层竖直：规则化求解 (12N -> 12N)
        
        Args:
            key_v: 关键量 (12N, 4)
            colli_v: 碰撞三角形 (12N, 3, 2)
            phi_v: 原始phi数组 (6N, 2) - 用于获取phi值
            valid_phi_flat_v: phi有效掩码 (12N,)
            
        Returns:
            tuple: (sols_v, valid_sol_v)
            - sols_v: 解数组 (12N, 5) - [x_min, x_max, y_min, y_max, phi]
            - valid_sol_v: 解有效掩码 (12N,)
        """
        self._log_debug("第八层竖直: 规则化求解")
        
        N_flat = len(key_v)  # 12N
        
        # 初始化解数组
        sols_v = np.zeros((N_flat, 5))  # (12N, 5)
        valid_sol_v = np.zeros(N_flat, dtype=bool)  # (12N,)
        
        # 只处理有效的phi位置
        if np.any(valid_phi_flat_v):
            valid_indices = np.where(valid_phi_flat_v)[0]
            
            # 提取有效数据
            valid_key = key_v[valid_indices]  # (N_valid, 4)
            valid_colli = colli_v[valid_indices]  # (N_valid, 3, 2)
            
            # 提取关键量
            key0 = valid_key[:, 0]  # y2 - y0
            key1 = valid_key[:, 1]  # y2 - y1
            key2 = valid_key[:, 2]  # x2 - x0
            key3 = valid_key[:, 3]  # x2 - x1
            
            # 提取坐标
            x0 = valid_colli[:, 0, 0]
            x2 = valid_colli[:, 2, 0]
            y0 = valid_colli[:, 0, 1]
            y2 = valid_colli[:, 2, 1]
            
            # 竖直模式求解条件判断 (恢复原逻辑，x和y、m和n互换)
            # 条件1：前两个关键量同号检查
            same_sign_01 = (key0 * key1) > 0
            
            # 条件2：后两个关键量绝对值最大值检查（筛掉不合理的）
            max_abs_23 = np.maximum(np.abs(key2), np.abs(key3))
            valid_range_23 = max_abs_23 < (self.m - self.tolerance)  # 注意：竖直模式用m
            
            # 综合条件筛选
            valid_solve_mask = same_sign_01 & valid_range_23
            
            if np.any(valid_solve_mask):
                solve_indices = valid_indices[valid_solve_mask]
                solve_key0 = key0[valid_solve_mask]
                solve_key1 = key1[valid_solve_mask]
                solve_x0 = x0[valid_solve_mask]
                solve_x2 = x2[valid_solve_mask]
                solve_y0 = y0[valid_solve_mask]
                solve_y2 = y2[valid_solve_mask]
                
                # 批量计算解中心 (恢复原逻辑，x和y、m和n互换)
                # 判断key0是正还是负，计算y_center
                is_positive = solve_key0 > 0
                y_center = np.where(is_positive, self.n - solve_y2, -solve_y2)
                
                # 计算x_center (使用x0和x1)
                solve_x1 = valid_colli[:, 1, 0][valid_solve_mask]
                x_center = (self.m - solve_x0 - solve_x1) / 2
                
                # 计算解边界 (使用tolerance)
                x_min = x_center - self.tolerance
                x_max = x_center + self.tolerance
                y_min = y_center - self.tolerance
                y_max = y_center + self.tolerance
                
                # 检查解的合理性（加入容错）
                bounds_valid = ((x_min >= -self.tolerance) & (x_max <= self.m + self.tolerance) & 
                               (y_min >= -self.tolerance) & (y_max <= self.n + self.tolerance))
                
                # 最终有效性：满足关键量条件且边界合理
                final_valid_mask = bounds_valid
                final_solve_indices = solve_indices[final_valid_mask]
                
                if len(final_solve_indices) > 0:
                    # 获取对应的phi值 (从12N索引转换为6N索引再转换为phi值)
                    phi_flat = phi_v.flatten()  # (12N,)
                    solve_phi_values = phi_flat[final_solve_indices]
                    
                    # 构造解
                    final_x_min = x_min[final_valid_mask]
                    final_x_max = x_max[final_valid_mask]
                    final_y_min = y_min[final_valid_mask]
                    final_y_max = y_max[final_valid_mask]
                    
                    valid_sols = np.column_stack([final_x_min, final_x_max, final_y_min, final_y_max, solve_phi_values])
                    
                    # 填入结果
                    sols_v[final_solve_indices] = valid_sols
                    valid_sol_v[final_solve_indices] = True
        
        valid_count = np.sum(valid_sol_v)
        self._log_debug("第八层竖直完成", 
                       f"从{N_flat}个位置中求得{valid_count}个有效解")
        return sols_v, valid_sol_v
        
    # ==================== 第九层：合并解 12N+12N->24N ====================
    def _merge_solutions_regularized(self, sols_h: np.ndarray, sols_v: np.ndarray,
                                   valid_sol_h: np.ndarray, valid_sol_v: np.ndarray, N: int) -> tuple:
        """
        第九层：合并水平和竖直解 (12N + 12N -> 24N)
        
        Args:
            sols_h: 水平解 (12N, 5)
            sols_v: 竖直解 (12N, 5)
            valid_sol_h: 水平解有效掩码 (12N,)
            valid_sol_v: 竖直解有效掩码 (12N,)
            
        Returns:
            tuple: (final_sols, final_valid)
            - final_sols: 最终解数组 (24N, 5)
            - final_valid: 最终解有效掩码 (24N,)
        """
        self._log_debug("第九层: 合并解")
        final_sol = np.full((N, 24, 5), np.inf, dtype=np.float64)
        final_valid = np.full((N, 24), -1, dtype=np.int32)  # 用-1表示无效索引
        
        # 将水平解和竖直解分别填充到对应位置
        final_sol[:, :12, :] = sols_h.reshape(N, 12, 5)
        final_sol[:, 12:, :] = sols_v.reshape(N, 12, 5)
        final_valid[:, :12] = valid_sol_h.reshape(N, 12)
        final_valid[:, 12:] = valid_sol_v.reshape(N, 12)
        
        valid_h_count = np.sum(valid_sol_h)
        valid_v_count = np.sum(valid_sol_v)
        total_valid = np.sum(final_valid)
        
        self._log_debug("第九层完成", 
                       f"合并: 水平{valid_h_count}个 + 竖直{valid_v_count}个 = 总计{total_valid}个有效解")
        
        return final_sol, final_valid
    
    def _extract_valid_solutions(self, final_sols: np.ndarray, final_valid: np.ndarray) -> List[Tuple]:
        """
        提取有效解并转换为最终格式
        
        Args:
            final_sols: 最终解数组 (24N, 5)
            final_valid: 最终解有效掩码 (24N,)
            
        Returns:
            List[Tuple]: 解列表，每个解为 ((x_min, x_max), (y_min, y_max), phi)
        """
        if not np.any(final_valid):
            return []
        
        # 提取有效解
        valid_solutions = final_sols[final_valid]  # (N_valid, 5)
        
        # 转换为目标格式
        solutions = []
        for sol in valid_solutions:
            x_min, x_max, y_min, y_max, phi = sol
            solutions.append(((x_min, x_max), (y_min, y_max), phi))
        
        return solutions
    
    def get_solver_info(self) -> dict:
        """获取求解器信息"""
        return {
            "solver_type": "Case3BatchSolver",
            "field_size": (self.m, self.n),
            "tolerance": self.tolerance,
            "data_flow": "N -> 6N -> 6N -> 12N -> 12N -> 24N",
            "regularized": True,
            "index_tracking": False
        }
