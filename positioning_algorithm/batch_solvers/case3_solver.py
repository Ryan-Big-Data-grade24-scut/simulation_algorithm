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
        self.ros_logger = ros_logger if enable_ros_logging else None
        
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
        
        # 根据ROS状态设置日志等级
        if self.enable_ros_logging:
            self.min_log_level = logging.WARNING  # ROS模式下只输出WARNING及以上
        else:
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
    
    def _log_array_detailed(self, name: str, arr):
        """详细结构化输出数组的每个元素"""
        if isinstance(arr, list):
            if len(arr) == 0:
                self._log_debug(name, "空列表")
                return
            
            self._log_debug(f"{name} 详细内容", f"列表长度: {len(arr)}")
            for i, item in enumerate(arr):
                self._log_debug(f"{name}[{i}]", f"{item}")
        
        elif isinstance(arr, np.ndarray):
            if arr.size == 0:
                self._log_debug(name, "空数组")
                return
            
            self._log_debug(f"{name} 详细内容", f"形状: {arr.shape}, 类型: {arr.dtype.name}")
            
            if arr.ndim == 1:
                # 一维数组：逐个显示
                for i in range(len(arr)):
                    self._log_debug(f"{name}[{i}]", f"{arr[i]}")
            
            elif arr.ndim == 2:
                # 二维数组：按行显示
                for i in range(arr.shape[0]):
                    self._log_debug(f"{name}[{i}]", f"{arr[i]}")
            
            elif arr.ndim == 3:
                # 三维数组：分层显示
                for i in range(arr.shape[0]):
                    self._log_debug(f"{name}[{i}] 形状{arr[i].shape}", "")
                    for j in range(arr.shape[1]):
                        self._log_debug(f"  {name}[{i}][{j}]", f"{arr[i][j]}")
            
            else:
                # 更高维数组：显示基本信息和前几个元素
                self._log_debug(f"{name}", f"高维数组 {arr.shape}，显示前5个元素:")
                flat_arr = arr.flatten()
                for i in range(min(5, len(flat_arr))):
                    self._log_debug(f"{name}.flat[{i}]", f"{flat_arr[i]}")
        
        else:
            self._log_debug(name, f"类型: {type(arr)}, 内容: {arr}")

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
        求解Case3批处理
        
        Args:
            combinations: 激光组合数组 (N_cbn, 3, 2) - 每行为[[t0,θ0], [t1,θ1], [t2,θ2]]
            
        Returns:
            List[Tuple]: 解列表，每个解为 ((x_min, x_max), (y_min, y_max), phi)
        """
        self._log_info("=" * 60)
        self._log_info("开始Case3批处理求解")
        self._log_info("=" * 60)
        
        # 输入验证和日志
        self._log_array("输入组合数组", combinations, show_content=False)
        
        if len(combinations) == 0:
            self._log_warning("没有组合可求解")
            return []
        
        N_cbn = len(combinations)
        self._log_info(f"组合数量统计", f"总共{N_cbn}个激光组合需要处理")
        
        # 第一层：扩展组合
        self._log_debug("第一层: 开始组合扩展")
        expanded_combinations, indice_ex = self._expand_combinations(combinations)
        
        # 第二层：计算A、B系数 (分水平和竖直)
        self._log_debug("第二层: 开始计算A、B系数")
        A_h, B_h = self._compute_ab_coefficients_h(expanded_combinations)
        A_v, B_v = self._compute_ab_coefficients_v(expanded_combinations)
        
        # 第三层：计算norm
        self._log_debug("第三层: 开始计算norm")
        norm_h = self._compute_norm_h(A_h, B_h)
        norm_v = self._compute_norm_v(A_v, B_v)
        
        # 第四层：计算alpha和arcsin
        self._log_debug("第四层: 开始计算alpha和arcsin")
        alpha_h = self._compute_alpha_h(A_h, B_h)
        alpha_v = self._compute_alpha_v(A_v, B_v)
        arcsin_h = self._compute_arcsin_h(norm_h)
        arcsin_v = self._compute_arcsin_v(norm_v)
        
        # 第五层：计算phi
        self._log_debug("第五层: 开始计算phi")
        phi_h = self._compute_phi_h(alpha_h, arcsin_h)
        phi_v = self._compute_phi_v(alpha_v, arcsin_v)
        
        # 第六层：计算碰撞
        self._log_debug("第六层: 开始计算碰撞")
        colli_h, colli_indice_h, phi_h_flat = self._compute_collision_h(phi_h, expanded_combinations, indice_ex)
        colli_v, colli_indice_v, phi_v_flat = self._compute_collision_v(phi_v, expanded_combinations, indice_ex)
        
        # 第七层：计算关键量
        self._log_debug("第七层: 开始计算关键量")
        key_h = self._compute_key_h(colli_h)
        key_v = self._compute_key_v(colli_v)
        
        # 第八层：求解
        self._log_debug("第八层: 开始求解")
        sols_h, indice_sol_h = self._solve_h(key_h, colli_h, colli_indice_h, phi_h_flat)
        sols_v, indice_sol_v = self._solve_v(key_v, colli_v, colli_indice_v, phi_v_flat)
        
        # 合并水平和竖直的解
        all_sols = np.vstack([sols_h, sols_v]) if len(sols_h) > 0 and len(sols_v) > 0 else (
            sols_h if len(sols_h) > 0 else sols_v if len(sols_v) > 0 else np.zeros((0, 5))
        )
        all_indice = np.concatenate([indice_sol_h, indice_sol_v]) if len(indice_sol_h) > 0 and len(indice_sol_v) > 0 else (
            indice_sol_h if len(indice_sol_h) > 0 else indice_sol_v if len(indice_sol_v) > 0 else np.array([], dtype=int)
        )
        
        # 第九层：组织解
        self._log_debug("第九层: 开始组织解")
        solutions = self._organize_solutions(all_sols, all_indice, N_cbn)
        
        # 最终统计
        self._log_info("=" * 60)
        self._log_info("Case3求解完成", f"在{N_cbn}个组合中找到{len(solutions)}个有效解")
        self._log_info("=" * 60)
        
        return solutions
    
    # ==================== 第一层：组合扩展 ====================
    def _expand_combinations(self, combinations: np.ndarray) -> tuple:
        """
        第一层：扩展组合，从N_cbn -> 6*N_cbn
        
        Args:
            combinations: 输入组合 (N_cbn, 3, 2)
            
        Returns:
            tuple: (expanded_combinations, indice_ex)
            - expanded_combinations: 扩展后组合 (6*N_cbn, 3, 2)
            - indice_ex: 原始组合编号追踪 (6*N_cbn,)
        """
        N_cbn = len(combinations)
        self._log_debug(f"第一层扩展组合", f"输入{N_cbn}个组合，将扩展为{6*N_cbn}个")
        
        # 初始化扩展后的数组
        expanded_combinations = np.zeros((6*N_cbn, 3, 2))
        indice_ex = np.zeros(6*N_cbn, dtype=int)
        
        # 定义6种排列组合：(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
        permutations = [
            [0, 1, 2],  # 原始顺序
            [0, 2, 1],  # 交换1,2
            [1, 0, 2],  # 交换0,1
            [1, 2, 0],  # 循环左移
            [2, 0, 1],  # 循环右移
            [2, 1, 0]   # 交换0,2
        ]
        
        # 对每个原始组合生成6种排列
        for i in range(N_cbn):
            for perm_idx, perm in enumerate(permutations):
                expanded_idx = i * 6 + perm_idx
                
                # 按照排列重新组织激光参数
                expanded_combinations[expanded_idx] = combinations[i][perm]
                
                # 记录原始组合编号
                indice_ex[expanded_idx] = i
        
        self._log_debug("第一层扩展完成", 
                       f"生成{len(expanded_combinations)}个扩展组合")
        
        # 记录详细的扩展信息（仅在调试模式下）
        if not self.enable_ros_logging:
            self._log_debug("扩展组合示例（前6个）:")
            for i in range(min(6, len(expanded_combinations))):
                original_idx = indice_ex[i]
                perm_type = i % 6
                self._log_debug(f"  扩展[{i}] <- 原始[{original_idx}], 排列类型{perm_type}",
                              f"激光参数: {expanded_combinations[i]}")
        
        return expanded_combinations, indice_ex
    
    # ==================== 第二层：A、B系数计算 ====================
    def _compute_ab_coefficients_h(self, expanded_combinations: np.ndarray) -> tuple:
        """
        第二层水平：计算水平情况的A、B系数
        
        Args:
            expanded_combinations: 扩展组合 (6*N_cbn, 3, 2)
            
        Returns:
            tuple: (A_h, B_h)
            - A_h: 水平A系数 (6*N_cbn,)
            - B_h: 水平B系数 (6*N_cbn,)
        """
        self._log_debug("第二层水平: 计算A、B系数")
        
        # 提取激光参数: t和theta
        # expanded_combinations形状: (6*N_cbn, 3, 2)
        # 我们只用前两束激光 [0]和[1]
        t0 = expanded_combinations[:, 0, 0]  # (6*N_cbn,)
        theta0 = expanded_combinations[:, 0, 1]  # (6*N_cbn,)
        t1 = expanded_combinations[:, 1, 0]  # (6*N_cbn,)
        theta1 = expanded_combinations[:, 1, 1]  # (6*N_cbn,)
        
        # 水平模式计算
        # A = t0*cos(theta0) - t1*cos(theta1)
        # B = t0*sin(theta0) - t1*sin(theta1)
        A_h = t0 * np.cos(theta0) - t1 * np.cos(theta1)
        B_h = t0 * np.sin(theta0) - t1 * np.sin(theta1)
        
        self._log_debug("第二层水平完成", f"生成A_h和B_h，形状: {A_h.shape}")
        
        return A_h, B_h
    
    def _compute_ab_coefficients_v(self, expanded_combinations: np.ndarray) -> tuple:
        """
        第二层竖直：计算竖直情况的A、B系数
        
        Args:
            expanded_combinations: 扩展组合 (6*N_cbn, 3, 2)
            
        Returns:
            tuple: (A_v, B_v)
            - A_v: 竖直A系数 (6*N_cbn,)
            - B_v: 竖直B系数 (6*N_cbn,)
        """
        self._log_debug("第二层竖直: 计算A、B系数")
        
        # 提取激光参数: t和theta
        t0 = expanded_combinations[:, 0, 0]  # (6*N_cbn,)
        theta0 = expanded_combinations[:, 0, 1]  # (6*N_cbn,)
        t1 = expanded_combinations[:, 1, 0]  # (6*N_cbn,)
        theta1 = expanded_combinations[:, 1, 1]  # (6*N_cbn,)
        
        # 竖直模式计算
        # A = t1*sin(theta1) - t0*sin(theta0)
        # B = t0*cos(theta0) - t1*cos(theta1)
        A_v = t1 * np.sin(theta1) - t0 * np.sin(theta0)
        B_v = t0 * np.cos(theta0) - t1 * np.cos(theta1)
        
        self._log_debug("第二层竖直完成", f"生成A_v和B_v，形状: {A_v.shape}")
        
        return A_v, B_v
    
    # ==================== 第三层：norm计算 ====================
    def _compute_norm_h(self, A_h: np.ndarray, B_h: np.ndarray) -> np.ndarray:
        """
        第三层水平：计算水平情况的norm
        
        Args:
            A_h: 水平A系数 (6*N_cbn,)
            B_h: 水平B系数 (6*N_cbn,)
            
        Returns:
            norm_h: 水平norm值 (6*N_cbn,)
        """
        self._log_debug("第三层水平: 计算norm")
        
        # norm = sqrt(A^2 + B^2)
        norm_h = np.sqrt(A_h**2 + B_h**2)
        
        self._log_debug("第三层水平完成", f"生成norm_h，形状: {norm_h.shape}")
        
        return norm_h
    
    def _compute_norm_v(self, A_v: np.ndarray, B_v: np.ndarray) -> np.ndarray:
        """
        第三层竖直：计算竖直情况的norm
        
        Args:
            A_v: 竖直A系数 (6*N_cbn,)
            B_v: 竖直B系数 (6*N_cbn,)
            
        Returns:
            norm_v: 竖直norm值 (6*N_cbn,)
        """
        self._log_debug("第三层竖直: 计算norm")
        
        # norm = sqrt(A^2 + B^2)
        norm_v = np.sqrt(A_v**2 + B_v**2)
        
        self._log_debug("第三层竖直完成", f"生成norm_v，形状: {norm_v.shape}")
        
        return norm_v
    
    # ==================== 第四层：alpha和arcsin计算 ====================
    def _compute_alpha_h(self, A_h: np.ndarray, B_h: np.ndarray) -> np.ndarray:
        """
        第四层水平：计算水平情况的alpha
        
        Args:
            A_h: 水平A系数 (6*N_cbn,)
            B_h: 水平B系数 (6*N_cbn,)
            
        Returns:
            alpha_h: 水平alpha值 (6*N_cbn,)
        """
        self._log_debug("第四层水平: 计算alpha")
        
        # 初始化为无穷大，标识无效值
        alpha_h = np.full_like(A_h, np.inf)
        
        # 检查有效值：A和B不能同时为0
        valid_mask = (A_h != 0) | (B_h != 0)
        
        if np.any(valid_mask):
            # alpha = arctan(B/A)，使用arctan2避免除零错误
            alpha_h[valid_mask] = np.arctan2(B_h[valid_mask], A_h[valid_mask])
        
        valid_count = np.sum(valid_mask)
        self._log_debug("第四层水平alpha完成", 
                       f"生成alpha_h，形状: {alpha_h.shape}, 有效值: {valid_count}/{len(alpha_h)}")
        
        return alpha_h
    
    def _compute_alpha_v(self, A_v: np.ndarray, B_v: np.ndarray) -> np.ndarray:
        """
        第四层竖直：计算竖直情况的alpha
        
        Args:
            A_v: 竖直A系数 (6*N_cbn,)
            B_v: 竖直B系数 (6*N_cbn,)
            
        Returns:
            alpha_v: 竖直alpha值 (6*N_cbn,)
        """
        self._log_debug("第四层竖直: 计算alpha")
        
        # 初始化为无穷大，标识无效值
        alpha_v = np.full_like(A_v, np.inf)
        
        # 检查有效值：A和B不能同时为0
        valid_mask = (A_v != 0) | (B_v != 0)
        
        if np.any(valid_mask):
            # alpha = arctan(B/A)，使用arctan2避免除零错误
            alpha_v[valid_mask] = np.arctan2(B_v[valid_mask], A_v[valid_mask])
        
        valid_count = np.sum(valid_mask)
        self._log_debug("第四层竖直alpha完成", 
                       f"生成alpha_v，形状: {alpha_v.shape}, 有效值: {valid_count}/{len(alpha_v)}")
        
        return alpha_v
    
    def _compute_arcsin_h(self, norm_h: np.ndarray) -> np.ndarray:
        """
        第四层水平：计算水平情况的arcsin
        
        Args:
            norm_h: 水平norm值 (6*N_cbn,)
            
        Returns:
            arcsin_h: 水平arcsin值 (6*N_cbn,)
        """
        self._log_debug("第四层水平: 计算arcsin")
        
        # 初始化为无穷大，标识无效值
        arcsin_h = np.full_like(norm_h, np.inf)
        
        # 水平模式: arcsin = arcsin(n/norm), n是矩形竖直高度
        # 检查有效值：norm > 0 且 n/norm <= 1
        valid_mask = (norm_h > 0) & (self.n <= norm_h)
        
        if np.any(valid_mask):
            ratio = self.n / norm_h[valid_mask]
            # 再次确保在[-1,1]范围内（理论上应该满足）
            ratio = np.clip(ratio, -1.0, 1.0)
            arcsin_h[valid_mask] = np.arcsin(ratio)
        
        valid_count = np.sum(valid_mask)
        invalid_count = np.sum((norm_h > 0) & (self.n > norm_h))  # norm太小的情况
        self._log_debug("第四层水平arcsin完成", 
                       f"生成arcsin_h，形状: {arcsin_h.shape}, 有效值: {valid_count}/{len(arcsin_h)}, norm过小: {invalid_count}")
        
        return arcsin_h
    
    def _compute_arcsin_v(self, norm_v: np.ndarray) -> np.ndarray:
        """
        第四层竖直：计算竖直情况的arcsin
        
        Args:
            norm_v: 竖直norm值 (6*N_cbn,)
            
        Returns:
            arcsin_v: 竖直arcsin值 (6*N_cbn,)
        """
        self._log_debug("第四层竖直: 计算arcsin")
        
        # 初始化为无穷大，标识无效值
        arcsin_v = np.full_like(norm_v, np.inf)
        
        # 竖直模式: arcsin = arcsin(m/norm), m是矩形横向宽度
        # 检查有效值：norm > 0 且 m/norm <= 1
        valid_mask = (norm_v > 0) & (self.m <= norm_v)
        
        if np.any(valid_mask):
            ratio = self.m / norm_v[valid_mask]
            # 再次确保在[-1,1]范围内（理论上应该满足）
            ratio = np.clip(ratio, -1.0, 1.0)
            arcsin_v[valid_mask] = np.arcsin(ratio)
        
        valid_count = np.sum(valid_mask)
        invalid_count = np.sum((norm_v > 0) & (self.m > norm_v))  # norm太小的情况
        self._log_debug("第四层竖直arcsin完成", 
                       f"生成arcsin_v，形状: {arcsin_v.shape}, 有效值: {valid_count}/{len(arcsin_v)}, norm过小: {invalid_count}")
        
        return arcsin_v
    
    # ==================== 第五层：phi计算 ====================
    def _compute_phi_h(self, alpha_h: np.ndarray, arcsin_h: np.ndarray) -> np.ndarray:
        """
        第五层水平：计算水平情况的phi
        
        Args:
            alpha_h: 水平alpha值 (6*N_cbn,)
            arcsin_h: 水平arcsin值 (6*N_cbn,)
            
        Returns:
            phi_h: 水平phi值 (6*N_cbn, 2)
        """
        self._log_debug("第五层水平: 计算phi")
        
        # 初始化为无穷大，标识无效值
        phi1 = np.full_like(alpha_h, np.inf)
        phi2 = np.full_like(alpha_h, np.inf)
        
        # 检查有效值：alpha和arcsin都必须是有限值
        valid_mask = np.isfinite(alpha_h) & np.isfinite(arcsin_h)
        
        if np.any(valid_mask):
            # phi有两个解：arcsin - alpha 和 pi - arcsin - alpha
            phi1[valid_mask] = arcsin_h[valid_mask] - alpha_h[valid_mask]
            phi2[valid_mask] = np.pi - arcsin_h[valid_mask] - alpha_h[valid_mask]
        
        # 组合成 (6*N_cbn, 2) 的数组
        phi_h = np.column_stack([phi1, phi2])
        
        valid_count = np.sum(valid_mask)
        self._log_debug("第五层水平完成", 
                       f"生成phi_h，形状: {phi_h.shape}, 有效值: {valid_count}/{len(alpha_h)}")
        
        return phi_h
    
    def _compute_phi_v(self, alpha_v: np.ndarray, arcsin_v: np.ndarray) -> np.ndarray:
        """
        第五层竖直：计算竖直情况的phi
        
        Args:
            alpha_v: 竖直alpha值 (6*N_cbn,)
            arcsin_v: 竖直arcsin值 (6*N_cbn,)
            
        Returns:
            phi_v: 竖直phi值 (6*N_cbn, 2)
        """
        self._log_debug("第五层竖直: 计算phi")
        
        # 初始化为无穷大，标识无效值
        phi1 = np.full_like(alpha_v, np.inf)
        phi2 = np.full_like(alpha_v, np.inf)
        
        # 检查有效值：alpha和arcsin都必须是有限值
        valid_mask = np.isfinite(alpha_v) & np.isfinite(arcsin_v)
        
        if np.any(valid_mask):
            # phi有两个解：arcsin - alpha 和 pi - arcsin - alpha
            phi1[valid_mask] = arcsin_v[valid_mask] - alpha_v[valid_mask]
            phi2[valid_mask] = np.pi - arcsin_v[valid_mask] - alpha_v[valid_mask]
        
        # 组合成 (6*N_cbn, 2) 的数组
        phi_v = np.column_stack([phi1, phi2])
        
        valid_count = np.sum(valid_mask)
        self._log_debug("第五层竖直完成", 
                       f"生成phi_v，形状: {phi_v.shape}, 有效值: {valid_count}/{len(alpha_v)}")
        
        return phi_v
    
    # ==================== 第六层：碰撞计算 ====================
    def _compute_collision_h(self, phi_h: np.ndarray, expanded_combinations: np.ndarray, 
                           indice_ex: np.ndarray) -> tuple:
        """
        第六层水平：计算水平情况的碰撞
        
        Args:
            phi_h: 水平phi值 (6*N_cbn, 2)
            expanded_combinations: 扩展组合 (6*N_cbn, 3, 2)
            indice_ex: 原始组合编号 (6*N_cbn,)
            
        Returns:
            tuple: (colli_h, colli_indice_h)
            - colli_h: 水平碰撞结果 (12*N_cbn, 3, 2)  - 每行为[[x0,y0], [x1,y1], [x2,y2]]
            - colli_indice_h: 对应的原始组合编号 (12*N_cbn,)
        """
        self._log_debug("第六层水平: 批量计算碰撞三角形")
        
        N_expanded = len(phi_h)  # 6*N_cbn
        
        # 批量处理：将phi重塑为 (12*N_cbn,) 
        phi_flat = phi_h.flatten()  # (12*N_cbn,)
        
        # 重复expanded_combinations以匹配phi的数量
        # 每个扩展组合需要重复2次（对应2个phi值）
        combinations_repeated = np.repeat(expanded_combinations, 2, axis=0)  # (12*N_cbn, 3, 2)
        indice_repeated = np.repeat(indice_ex, 2)  # (12*N_cbn,)
        
        # 只处理有效的phi值
        valid_mask = np.isfinite(phi_flat)
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            return np.zeros((0, 3, 2)), np.array([], dtype=int), np.array([])
        
        # 提取有效数据
        valid_phi = phi_flat[valid_mask]  # (N_valid,)
        valid_combinations = combinations_repeated[valid_mask]  # (N_valid, 3, 2)
        valid_indices = indice_repeated[valid_mask]  # (N_valid,)
        
        # 批量计算所有三角形的坐标
        # valid_combinations形状: (N_valid, 3, 2) -> t和theta
        t_values = valid_combinations[:, :, 0]  # (N_valid, 3) - t0, t1, t2
        theta_values = valid_combinations[:, :, 1]  # (N_valid, 3) - theta0, theta1, theta2
        
        # 广播计算：phi + theta
        # valid_phi[:, None]: (N_valid, 1)
        # theta_values: (N_valid, 3)
        phi_plus_theta = valid_phi[:, None] + theta_values  # (N_valid, 3)
        
        # 批量计算x和y坐标
        x_coords = t_values * np.cos(phi_plus_theta)  # (N_valid, 3)
        y_coords = t_values * np.sin(phi_plus_theta)  # (N_valid, 3)
        
        # 组合为 (N_valid, 3, 2) 的形式：[[x0,y0], [x1,y1], [x2,y2]]
        colli_h = np.stack([x_coords, y_coords], axis=2)  # (N_valid, 3, 2)
        
        self._log_debug("第六层水平完成", 
                       f"生成{len(colli_h)}个有效碰撞三角形，形状: {colli_h.shape}")
        
        return colli_h, valid_indices, valid_phi
    
    def _compute_collision_v(self, phi_v: np.ndarray, expanded_combinations: np.ndarray, 
                           indice_ex: np.ndarray) -> tuple:
        """
        第六层竖直：计算竖直情况的碰撞
        
        Args:
            phi_v: 竖直phi值 (6*N_cbn, 2)
            expanded_combinations: 扩展组合 (6*N_cbn, 3, 2)
            indice_ex: 原始组合编号 (6*N_cbn,)
            
        Returns:
            tuple: (colli_v, colli_indice_v)
            - colli_v: 竖直碰撞结果 (12*N_cbn, 3, 2)  - 每行为[[x0,y0], [x1,y1], [x2,y2]]
            - colli_indice_v: 对应的原始组合编号 (12*N_cbn,)
        """
        self._log_debug("第六层竖直: 批量计算碰撞三角形")
        
        N_expanded = len(phi_v)  # 6*N_cbn
        
        # 批量处理：将phi重塑为 (12*N_cbn,) 
        phi_flat = phi_v.flatten()  # (12*N_cbn,)
        
        # 重复expanded_combinations以匹配phi的数量
        # 每个扩展组合需要重复2次（对应2个phi值）
        combinations_repeated = np.repeat(expanded_combinations, 2, axis=0)  # (12*N_cbn, 3, 2)
        indice_repeated = np.repeat(indice_ex, 2)  # (12*N_cbn,)
        
        # 只处理有效的phi值
        valid_mask = np.isfinite(phi_flat)
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            return np.zeros((0, 3, 2)), np.array([], dtype=int), np.array([])
        
        # 提取有效数据
        valid_phi = phi_flat[valid_mask]  # (N_valid,)
        valid_combinations = combinations_repeated[valid_mask]  # (N_valid, 3, 2)
        valid_indices = indice_repeated[valid_mask]  # (N_valid,)
        
        # 批量计算所有三角形的坐标
        # valid_combinations形状: (N_valid, 3, 2) -> t和theta
        t_values = valid_combinations[:, :, 0]  # (N_valid, 3) - t0, t1, t2
        theta_values = valid_combinations[:, :, 1]  # (N_valid, 3) - theta0, theta1, theta2
        
        # 广播计算：phi + theta
        # valid_phi[:, None]: (N_valid, 1)
        # theta_values: (N_valid, 3)
        phi_plus_theta = valid_phi[:, None] + theta_values  # (N_valid, 3)
        
        # 批量计算x和y坐标
        x_coords = t_values * np.cos(phi_plus_theta)  # (N_valid, 3)
        y_coords = t_values * np.sin(phi_plus_theta)  # (N_valid, 3)
        
        # 组合为 (N_valid, 3, 2) 的形式：[[x0,y0], [x1,y1], [x2,y2]]
        colli_v = np.stack([x_coords, y_coords], axis=2)  # (N_valid, 3, 2)
        
        self._log_debug("第六层竖直完成", 
                       f"生成{len(colli_v)}个有效碰撞三角形，形状: {colli_v.shape}")
        
        return colli_v, valid_indices, valid_phi
    
    # ==================== 第七层：关键量计算 ====================
    def _compute_key_h(self, colli_h: np.ndarray) -> np.ndarray:
        """
        第七层水平：计算水平情况的关键量
        
        Args:
            colli_h: 水平碰撞结果 (N_valid_h, 3, 2) - 每行为[[x0,y0], [x1,y1], [x2,y2]]
            
        Returns:
            key_h: 水平关键量 (N_valid_h, 4) - 每行为[key0, key1, key2, key3]
        """
        self._log_debug("第七层水平: 计算关键量")
        
        if len(colli_h) == 0:
            return np.zeros((0, 4))
        
        # 提取三个顶点的坐标
        x0 = colli_h[:, 0, 0]  # (N_valid_h,)
        y0 = colli_h[:, 0, 1]  # (N_valid_h,)
        x1 = colli_h[:, 1, 0]  # (N_valid_h,)
        y1 = colli_h[:, 1, 1]  # (N_valid_h,)
        x2 = colli_h[:, 2, 0]  # (N_valid_h,)
        y2 = colli_h[:, 2, 1]  # (N_valid_h,)
        
        # 水平模式的四个关键量
        key0 = x2 - x0  # 原来的key0
        key1 = x2 - x1  # 原来的key1
        key2 = y2 - y0  # 新增的key2
        key3 = y2 - y1  # 新增的key3
        
        # 组合成 (N_valid_h, 4) 的数组
        key_h = np.column_stack([key0, key1, key2, key3])
        
        self._log_debug("第七层水平完成", 
                       f"生成key_h，形状: {key_h.shape}")
        
        return key_h
    
    def _compute_key_v(self, colli_v: np.ndarray) -> np.ndarray:
        """
        第七层竖直：计算竖直情况的关键量
        
        Args:
            colli_v: 竖直碰撞结果 (N_valid_v, 3, 2) - 每行为[[x0,y0], [x1,y1], [x2,y2]]
            
        Returns:
            key_v: 竖直关键量 (N_valid_v, 4) - 每行为[key0, key1, key2, key3]
        """
        self._log_debug("第七层竖直: 计算关键量")
        
        if len(colli_v) == 0:
            return np.zeros((0, 4))
        
        # 提取三个顶点的坐标
        x0 = colli_v[:, 0, 0]  # (N_valid_v,)
        y0 = colli_v[:, 0, 1]  # (N_valid_v,)
        x1 = colli_v[:, 1, 0]  # (N_valid_v,)
        y1 = colli_v[:, 1, 1]  # (N_valid_v,)
        x2 = colli_v[:, 2, 0]  # (N_valid_v,)
        y2 = colli_v[:, 2, 1]  # (N_valid_v,)
        
        # 竖直模式的四个关键量
        key0 = y2 - y0  # 原来的key0
        key1 = y2 - y1  # 原来的key1
        key2 = x2 - x0  # 新增的key2
        key3 = x2 - x1  # 新增的key3
        
        # 组合成 (N_valid_v, 4) 的数组
        key_v = np.column_stack([key0, key1, key2, key3])
        
        self._log_debug("第七层竖直完成", 
                       f"生成key_v，形状: {key_v.shape}")
        
        return key_v
    
    # ==================== 第八层：求解 ====================
    def _solve_h(self, key_h: np.ndarray, colli_h: np.ndarray, 
                colli_indice_h: np.ndarray, phi_h_flat: np.ndarray) -> tuple:
        """
        第八层水平：水平情况求解
        
        Args:
            key_h: 水平关键量 (N_valid_h, 4) - [key0, key1, key2, key3]
            colli_h: 水平碰撞结果 (N_valid_h, 3, 2)
            colli_indice_h: 对应的原始组合编号 (N_valid_h,)
            phi_h_flat: 对应的phi值 (N_valid_h,)
            
        Returns:
            tuple: (sols_h, indice_sol_h)
            - sols_h: 水平解数组 (N_valid_h, 5) - [x_min, x_max, y_min, y_max, phi]
            - indice_sol_h: 解对应的组合编号 (N_valid_h,)
        """
        self._log_debug("第八层水平: 开始批量求解")
        
        if len(key_h) == 0:
            return np.zeros((0, 5)), np.array([], dtype=int)
        
        N_valid = len(key_h)
        
        # 初始化解数组，默认为无效解
        sols_h = np.full((N_valid, 5), np.inf)
        
        # 批量计算所有可能的解中心
        # 提取坐标
        x2 = colli_h[:, 2, 0]  # (N_valid_h,)
        y0 = colli_h[:, 0, 1]  # (N_valid_h,)
        y2 = colli_h[:, 2, 1]  # (N_valid_h,)
        
        # 提取关键量
        key0 = key_h[:, 0]  # x2 - x0
        key1 = key_h[:, 1]  # x2 - x1
        key2 = key_h[:, 2]  # y2 - y0
        key3 = key_h[:, 3]  # y2 - y1
        
        # 条件1：前两个关键量同号检查
        same_sign_01 = (key0 * key1) > 0
        
        # 条件2：后两个关键量绝对值最大值检查（筛掉不合理的）
        max_abs_23 = np.maximum(np.abs(key2), np.abs(key3))
        valid_range_23 = max_abs_23 < (self.n - self.tolerance)
        
        # 综合条件筛选
        valid_mask = same_sign_01 & valid_range_23
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            self._log_debug("第八层水平完成", "无有效解")
            return sols_h, colli_indice_h
        
        # 批量计算解中心
        # 判断key0是正还是负，计算x_center
        is_positive = key0 > 0
        x_center = np.where(is_positive, self.m - x2, -x2)
        
        # 计算y_center
        y_center = (self.n - y0 - y2) / 2
        
        # 批量计算解边界
        x_min = x_center - self.tolerance
        x_max = x_center + self.tolerance
        y_min = y_center - self.tolerance
        y_max = y_center + self.tolerance
        
        # 检查解的合理性（加入容错）
        bounds_valid = ((x_min >= -self.tolerance) & (x_max <= self.m + self.tolerance) & 
                       (y_min >= -self.tolerance) & (y_max <= self.n + self.tolerance))
        
        # 最终有效性：满足关键量条件且边界合理
        final_valid = valid_mask & bounds_valid
        
        # 为有效解填充数据
        sols_h[final_valid, 0] = x_min[final_valid]  # x_min
        sols_h[final_valid, 1] = x_max[final_valid]  # x_max
        sols_h[final_valid, 2] = y_min[final_valid]  # y_min
        sols_h[final_valid, 3] = y_max[final_valid]  # y_max
        sols_h[final_valid, 4] = phi_h_flat[final_valid]  # phi
        
        final_valid_count = np.sum(final_valid)
        self._log_debug("第八层水平完成", 
                       f"处理{N_valid}个候选解，其中{final_valid_count}个有效解")
        
        return sols_h, colli_indice_h
    
    def _solve_v(self, key_v: np.ndarray, colli_v: np.ndarray, 
                colli_indice_v: np.ndarray, phi_v_flat: np.ndarray) -> tuple:
        """
        第八层竖直：竖直情况求解
        
        Args:
            key_v: 竖直关键量 (N_valid_v, 4) - [key0, key1, key2, key3]
            colli_v: 竖直碰撞结果 (N_valid_v, 3, 2)
            colli_indice_v: 对应的原始组合编号 (N_valid_v,)
            phi_v_flat: 对应的phi值 (N_valid_v,)
            
        Returns:
            tuple: (sols_v, indice_sol_v)
            - sols_v: 竖直解数组 (N_valid_v, 5) - [x_min, x_max, y_min, y_max, phi]
            - indice_sol_v: 解对应的组合编号 (N_valid_v,)
        """
        self._log_debug("第八层竖直: 开始批量求解")
        
        if len(key_v) == 0:
            return np.zeros((0, 5)), np.array([], dtype=int)
        
        N_valid = len(key_v)
        
        # 初始化解数组，默认为无效解
        sols_v = np.full((N_valid, 5), np.inf)
        
        # 批量计算所有可能的解中心
        # 提取坐标
        x0 = colli_v[:, 0, 0]  # (N_valid_v,)
        x2 = colli_v[:, 2, 0]  # (N_valid_v,)
        y2 = colli_v[:, 2, 1]  # (N_valid_v,)
        
        # 提取关键量
        key0 = key_v[:, 0]  # y2 - y0
        key1 = key_v[:, 1]  # y2 - y1
        key2 = key_v[:, 2]  # x2 - x0
        key3 = key_v[:, 3]  # x2 - x1
        
        # 条件1：前两个关键量同号检查
        same_sign_01 = (key0 * key1) > 0
        
        # 条件2：后两个关键量绝对值最大值检查（筛掉不合理的）
        max_abs_23 = np.maximum(np.abs(key2), np.abs(key3))
        valid_range_23 = max_abs_23 < (self.m - self.tolerance)
        
        # 综合条件筛选
        valid_mask = same_sign_01 & valid_range_23
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            self._log_debug("第八层竖直完成", "无有效解")
            return sols_v, colli_indice_v
        
        # 批量计算解中心
        # 判断key0是正还是负，计算y_center
        is_positive = key0 > 0
        y_center = np.where(is_positive, self.n - y2, -y2)
        
        # 计算x_center
        x_center = (self.m - x0 - x2) / 2
        
        # 批量计算解边界
        x_min = x_center - self.tolerance
        x_max = x_center + self.tolerance
        y_min = y_center - self.tolerance
        y_max = y_center + self.tolerance
        
        # 检查解的合理性（加入容错）
        bounds_valid = ((x_min >= -self.tolerance) & (x_max <= self.m + self.tolerance) & 
                       (y_min >= -self.tolerance) & (y_max <= self.n + self.tolerance))
        
        # 最终有效性：满足关键量条件且边界合理
        final_valid = valid_mask & bounds_valid
        
        # 为有效解填充数据
        sols_v[final_valid, 0] = x_min[final_valid]  # x_min
        sols_v[final_valid, 1] = x_max[final_valid]  # x_max
        sols_v[final_valid, 2] = y_min[final_valid]  # y_min
        sols_v[final_valid, 3] = y_max[final_valid]  # y_max
        sols_v[final_valid, 4] = phi_v_flat[final_valid]  # phi
        
        final_valid_count = np.sum(final_valid)
        self._log_debug("第八层竖直完成", 
                       f"处理{N_valid}个候选解，其中{final_valid_count}个有效解")
        
        return sols_v, colli_indice_v
    
    # ==================== 第九层：解组织 ====================
    def _organize_solutions(self, all_sols: np.ndarray, all_indice: np.ndarray, 
                          N_cbn: int) -> List[Tuple]:
        """
        第九层：组织最终解
        
        Args:
            all_sols: 所有解数组 (N_total, 5) - [x_min, x_max, y_min, y_max, phi]
            all_indice: 解对应的组合编号数组 (N_total,)
            N_cbn: 原始组合数量
            
        Returns:
            List[Tuple]: 组织好的解列表，每个解为((x_min, x_max), (y_min, y_max), phi)
        """
        self._log_debug("第九层: 开始组织解")
        
        if len(all_sols) == 0:
            self._log_debug("第九层完成", "无解需要组织")
            return []
        
        # 初始化按组合分组的解列表
        organized_solutions = [[] for _ in range(N_cbn)]
        
        # 筛选有效解（不包含np.inf的解）
        valid_mask = np.isfinite(all_sols[:, 0])  # 检查x_min是否有限
        
        if not np.any(valid_mask):
            self._log_debug("第九层完成", "所有解都无效")
            return []
        
        # 提取有效解和对应的组合编号
        valid_sols = all_sols[valid_mask]
        valid_indices = all_indice[valid_mask]
        
        # 遍历所有有效解，按组合编号分组
        for i in range(len(valid_sols)):
            sol = valid_sols[i]
            cbn_idx = valid_indices[i]
            
            x_min, x_max, y_min, y_max, phi = sol
            
            # 组织为标准格式：((x_min, x_max), (y_min, y_max), phi)
            organized_sol = ((x_min, x_max), (y_min, y_max), phi)
            organized_solutions[cbn_idx].append(organized_sol)
        
        # 统计每个组合的解数量
        sol_counts = [len(sols) for sols in organized_solutions]
        total_valid_solutions = sum(sol_counts)
        non_empty_groups = sum(1 for count in sol_counts if count > 0)
        
        self._log_debug("第九层完成", 
                       f"组织了{total_valid_solutions}个有效解，分布在{non_empty_groups}/{N_cbn}个组合中")
                
        return organized_solutions
    
    def get_solver_info(self) -> dict:
        """获取求解器信息"""
        return {
            "name": "Case3BatchSolver",
            "field_size": (self.m, self.n),
            "tolerance": self.tolerance,
            "description": "三点分别在三条边的情况"
        }
    
