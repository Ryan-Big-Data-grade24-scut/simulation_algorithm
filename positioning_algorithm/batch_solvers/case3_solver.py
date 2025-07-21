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
        colli_h, colli_indice_h = self._compute_collision_h(phi_h, expanded_combinations, indice_ex)
        colli_v, colli_indice_v = self._compute_collision_v(phi_v, expanded_combinations, indice_ex)
        
        # 第七层：计算关键量
        self._log_debug("第七层: 开始计算关键量")
        key_h = self._compute_key_h(colli_h)
        key_v = self._compute_key_v(colli_v)
        
        # 第八层：求解
        self._log_debug("第八层: 开始求解")
        sols_h, indice_sol_h = self._solve_h(key_h, colli_h, colli_indice_h)
        sols_v, indice_sol_v = self._solve_v(key_v, colli_v, colli_indice_v)
        
        # 合并水平和竖直的解
        all_sols = []
        all_indice = []
        if len(sols_h) > 0:
            all_sols.extend(sols_h)
            all_indice.extend(indice_sol_h)
        if len(sols_v) > 0:
            all_sols.extend(sols_v)
            all_indice.extend(indice_sol_v)
        
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
            - colli_h: 水平碰撞结果 (12*N_cbn, 3)  - 每行为[x0, x1, x2]
            - colli_indice_h: 对应的原始组合编号 (12*N_cbn,)
        """
        self._log_debug("第六层水平: 计算碰撞三角形")
        
        N_expanded = len(phi_h)  # 6*N_cbn
        N_triangles = N_expanded * 2  # 12*N_cbn，每个扩展组合有2个phi值
        
        # 初始化结果数组
        colli_h = np.zeros((N_triangles, 3))  # 存储三个顶点的x坐标
        colli_indice_h = np.zeros(N_triangles, dtype=int)
        
        # 遍历每个扩展组合的每个phi值
        triangle_idx = 0
        for i in range(N_expanded):
            for j in range(2):  # 每个phi有两个解
                phi = phi_h[i, j]
                
                # 只处理有效的phi值
                if np.isfinite(phi):
                    # 提取当前组合的激光参数
                    t0, theta0 = expanded_combinations[i, 0, :]
                    t1, theta1 = expanded_combinations[i, 1, :]
                    t2, theta2 = expanded_combinations[i, 2, :]
                    
                    # 计算三个顶点的x坐标
                    x0 = t0 * np.cos(phi + theta0)
                    x1 = t1 * np.cos(phi + theta1)
                    x2 = t2 * np.cos(phi + theta2)
                    
                    # 存储结果
                    colli_h[triangle_idx] = [x0, x1, x2]
                    colli_indice_h[triangle_idx] = indice_ex[i]
                    
                triangle_idx += 1
        
        # 只返回有效的三角形（去掉无效phi产生的零行）
        valid_mask = ~np.all(colli_h == 0, axis=1)
        colli_h = colli_h[valid_mask]
        colli_indice_h = colli_indice_h[valid_mask]
        
        self._log_debug("第六层水平完成", 
                       f"生成{len(colli_h)}个有效碰撞三角形，原始形状: ({N_triangles}, 3)")
        
        return colli_h, colli_indice_h
    
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
            - colli_v: 竖直碰撞结果 (12*N_cbn, 3)  - 每行为[y0, y1, y2]
            - colli_indice_v: 对应的原始组合编号 (12*N_cbn,)
        """
        self._log_debug("第六层竖直: 计算碰撞三角形")
        
        N_expanded = len(phi_v)  # 6*N_cbn
        N_triangles = N_expanded * 2  # 12*N_cbn，每个扩展组合有2个phi值
        
        # 初始化结果数组
        colli_v = np.zeros((N_triangles, 3))  # 存储三个顶点的y坐标
        colli_indice_v = np.zeros(N_triangles, dtype=int)
        
        # 遍历每个扩展组合的每个phi值
        triangle_idx = 0
        for i in range(N_expanded):
            for j in range(2):  # 每个phi有两个解
                phi = phi_v[i, j]
                
                # 只处理有效的phi值
                if np.isfinite(phi):
                    # 提取当前组合的激光参数
                    t0, theta0 = expanded_combinations[i, 0, :]
                    t1, theta1 = expanded_combinations[i, 1, :]
                    t2, theta2 = expanded_combinations[i, 2, :]
                    
                    # 计算三个顶点的y坐标
                    y0 = t0 * np.sin(phi + theta0)
                    y1 = t1 * np.sin(phi + theta1)
                    y2 = t2 * np.sin(phi + theta2)
                    
                    # 存储结果
                    colli_v[triangle_idx] = [y0, y1, y2]
                    colli_indice_v[triangle_idx] = indice_ex[i]
                    
                triangle_idx += 1
        
        # 只返回有效的三角形（去掉无效phi产生的零行）
        valid_mask = ~np.all(colli_v == 0, axis=1)
        colli_v = colli_v[valid_mask]
        colli_indice_v = colli_indice_v[valid_mask]
        
        self._log_debug("第六层竖直完成", 
                       f"生成{len(colli_v)}个有效碰撞三角形，原始形状: ({N_triangles}, 3)")
        
        return colli_v, colli_indice_v
    
    # ==================== 第七层：关键量计算 ====================
    def _compute_key_h(self, colli_h: np.ndarray) -> np.ndarray:
        """
        第七层水平：计算水平情况的关键量
        
        Args:
            colli_h: 水平碰撞结果 (N_valid_h, 3) - 每行为[x0, x1, x2]
            
        Returns:
            key_h: 水平关键量 (N_valid_h, 2) - 每行为[key0, key1]
        """
        self._log_debug("第七层水平: 计算关键量")
        
        if len(colli_h) == 0:
            return np.zeros((0, 2))
        
        # 提取三个顶点的x坐标
        x0 = colli_h[:, 0]  # (N_valid_h,)
        x1 = colli_h[:, 1]  # (N_valid_h,)
        x2 = colli_h[:, 2]  # (N_valid_h,)
        
        # 水平模式的关键量
        # key0 = x2 - x0
        # key1 = x2 - x1
        key0 = x2 - x0
        key1 = x2 - x1
        
        # 组合成 (N_valid_h, 2) 的数组
        key_h = np.column_stack([key0, key1])
        
        self._log_debug("第七层水平完成", 
                       f"生成key_h，形状: {key_h.shape}")
        
        return key_h
    
    def _compute_key_v(self, colli_v: np.ndarray) -> np.ndarray:
        """
        第七层竖直：计算竖直情况的关键量
        
        Args:
            colli_v: 竖直碰撞结果 (N_valid_v, 3) - 每行为[y0, y1, y2]
            
        Returns:
            key_v: 竖直关键量 (N_valid_v, 2) - 每行为[key0, key1]
        """
        self._log_debug("第七层竖直: 计算关键量")
        
        if len(colli_v) == 0:
            return np.zeros((0, 2))
        
        # 提取三个顶点的y坐标
        y0 = colli_v[:, 0]  # (N_valid_v,)
        y1 = colli_v[:, 1]  # (N_valid_v,)
        y2 = colli_v[:, 2]  # (N_valid_v,)
        
        # 竖直模式的关键量
        # key0 = y2 - y0
        # key1 = y2 - y1
        key0 = y2 - y0
        key1 = y2 - y1
        
        # 组合成 (N_valid_v, 2) 的数组
        key_v = np.column_stack([key0, key1])
        
        self._log_debug("第七层竖直完成", 
                       f"生成key_v，形状: {key_v.shape}")
        
        return key_v
    
    # ==================== 第八层：求解 ====================
    def _solve_h(self, key_h: np.ndarray, colli_h: np.ndarray, 
                colli_indice_h: np.ndarray) -> tuple:
        """
        第八层水平：水平情况求解
        
        Args:
            key_h: 水平关键量 (N_valid_h, 2)
            colli_h: 水平碰撞结果 (N_valid_h, 3)
            colli_indice_h: 对应的原始组合编号 (N_valid_h,)
            
        Returns:
            tuple: (sols_h, indice_sol_h)
            - sols_h: 水平解 (N_sol_h, 3)
            - indice_sol_h: 解对应的组合编号 (N_sol_h,)
        """
        # TODO: 实现水平求解
        pass
    
    def _solve_v(self, key_v: np.ndarray, colli_v: np.ndarray, 
                colli_indice_v: np.ndarray) -> tuple:
        """
        第八层竖直：竖直情况求解
        
        Args:
            key_v: 竖直关键量 (N_valid_v, 2)
            colli_v: 竖直碰撞结果 (N_valid_v, 3)
            colli_indice_v: 对应的原始组合编号 (N_valid_v,)
            
        Returns:
            tuple: (sols_v, indice_sol_v)
            - sols_v: 竖直解 (N_sol_v, 3)
            - indice_sol_v: 解对应的组合编号 (N_sol_v,)
        """
        # TODO: 实现竖直求解
        pass
    
    # ==================== 第九层：解组织 ====================
    def _organize_solutions(self, sols: List, indice_sol: List, 
                          N_cbn: int) -> List[Tuple]:
        """
        第九层：组织最终解
        
        Args:
            sols: 所有解 (列表)
            indice_sol: 解对应的组合编号 (列表)
            N_cbn: 原始组合数量
            
        Returns:
            List[Tuple]: 组织好的解列表
        """
        # TODO: 实现解组织
        pass
    
    def get_solver_info(self) -> dict:
        """获取求解器信息"""
        return {
            "name": "Case3BatchSolver",
            "field_size": (self.m, self.n),
            "tolerance": self.tolerance,
            "description": "三点分别在三条边的情况"
        }
    
