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
    
    def __init__(self, m: float, n: float, tolerance: float = 1e-3, 
                 enable_ros_logging: bool = False, ros_logger=None):
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
        self.enable_ros_logging = enable_ros_logging
        self.ros_logger = ros_logger if enable_ros_logging else None
        
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
    
    def _log_array_structure(self, name: str, arr: np.ndarray, max_elements: int = 5):
        """结构化输出数组信息，避免冗长的np.float64输出"""
        if arr.size == 0:
            self._log_debug(f"{name}", "空数组")
            return
        
        # 基本信息
        shape_str = f"形状{arr.shape}, 类型{arr.dtype.name}"
        
        # 统计信息
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            if np.all(np.isfinite(arr)):
                stats = f"范围[{arr.min():.3f}, {arr.max():.3f}]"
            else:
                finite_count = np.sum(np.isfinite(arr))
                stats = f"有限值{finite_count}/{arr.size}个"
        else:
            stats = ""
        
        # 内容预览
        if arr.size <= max_elements:
            if arr.ndim == 1:
                content = "[" + ", ".join(f"{x:.3f}" if isinstance(x, (int, float, np.number)) else str(x) 
                                        for x in arr.flat) + "]"
            else:
                content = f"前{min(max_elements, arr.shape[0])}行显示"
        else:
            content = f"显示前{max_elements}个元素"
        
        # 组合输出
        full_info = f"{shape_str}"
        if stats:
            full_info += f", {stats}"
        if content and arr.size <= max_elements * 2:  # 只在元素不太多时显示内容
            full_info += f", 内容: {content}"
        
        self._log_debug(name, full_info)
    
    def solve(self, combinations: np.ndarray) -> np.ndarray:
        """
        求解Case1批处理
        
        Args:
            combinations: 激光组合数组 (N_cbn, 3, 2) - 每行为[[t0,θ0], [t1,θ1], [t2,θ2]]
            
        Returns:
            np.ndarray: 解数组 (N_cbn, 5) - 每行为[xmin, xmax, ymin, ymax, phi]
                       无解的组合用np.inf填充
        """
        self._log_info("=" * 60)
        self._log_info("开始Case1批处理求解")
        self._log_info("=" * 60)
        
        # 输入验证和日志
        self._log_array_structure("输入组合数组", combinations)
        
        if len(combinations) == 0:
            self._log_warning("没有组合可求解")
            return np.array([]).reshape(0, 5)  # 改为5列
        
        N_cbn = len(combinations)
        self._log_info(f"组合数量统计", f"总共{N_cbn}个激光组合需要处理")
        
        # 步骤1: 扩展组合 N_cbn -> 3*N_cbn，并跟踪原始编号
        self._log_debug("步骤1: 开始组合扩展")
        expanded_combinations, combo_indices = self._expand_combinations(combinations)
        self._log_array_structure("扩展后组合", expanded_combinations)
        self._log_array_structure("组合索引追踪", combo_indices)
        
        # 步骤2: 分水平边和竖直边计算A、B系数
        self._log_debug("步骤2: 开始计算A、B系数")
        ab_h, ab_v = self._compute_ab_coefficients(expanded_combinations)
        self._log_array_structure("水平边A、B系数", ab_h)
        self._log_array_structure("竖直边A、B系数", ab_v)
        
        # 步骤3: 计算phi候选
        self._log_debug("步骤3: 开始计算phi候选")
        phi_h, phi_v = self._compute_phi_candidates(ab_h, ab_v)
        # 统计有效phi数量
        valid_phi_h = np.sum(~np.isinf(phi_h))
        valid_phi_v = np.sum(~np.isinf(phi_v))
        self._log_info("phi候选统计", f"水平边: {valid_phi_h}个有效值, 竖直边: {valid_phi_v}个有效值")
        
        # 步骤4: 计算碰撞三角形
        self._log_debug("步骤4: 开始计算碰撞三角形")
        colli_h, valid_indice_cbo_h = self._compute_collision_triangles_h(phi_h, expanded_combinations, combo_indices)
        colli_v, valid_indice_cbo_v = self._compute_collision_triangles_v(phi_v, expanded_combinations, combo_indices)
        self._log_info("碰撞三角形统计", f"水平边: {len(colli_h)}个, 竖直边: {len(colli_v)}个")
        
        # 步骤5: 计算关键量
        self._log_debug("步骤5: 开始计算关键量")
        key_h = self._compute_key_quantities_h(colli_h)
        key_v = self._compute_key_quantities_v(colli_v)
        self._log_array_structure("水平边关键量", key_h)
        self._log_array_structure("竖直边关键量", key_v)
        
        # 步骤6: 求解
        self._log_debug("步骤6: 开始求解两种情况")
        sols, indice_sol = self._solve_both_cases(colli_h, colli_v, key_h, key_v, 
                                                 valid_indice_cbo_h, valid_indice_cbo_v,
                                                 phi_h, phi_v)
        self._log_info("求解结果统计", f"找到{len(sols)}个候选解")
        
        # 步骤7: 组织结果到sol_cbn
        self._log_debug("步骤7: 开始组织最终解")
        sol_cbn = self._organize_solutions(sols, indice_sol, N_cbn)
        
        # 最终统计
        valid_solutions = len([sol for sublist in sol_cbn for sol in sublist if sol[0] != np.inf])
        self._log_info("=" * 60)
        self._log_info("Case1求解完成", f"在{N_cbn}个组合中找到{valid_solutions}个有效解")
        self._log_info("=" * 60)
        
        return sol_cbn
    
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
        步骤3: 计算phi候选
        
        Args:
            ab_h: 水平边A、B系数 (3*N_cbn, 2)
            ab_v: 竖直边A、B系数 (3*N_cbn, 2)
            
        Returns:
            tuple: (phi_h, phi_v)
            - phi_h: 水平边phi候选 (3*N_cbn, 2) - 每行包含2个phi值，无效时用np.inf标记
            - phi_v: 竖直边phi候选 (3*N_cbn, 2) - 每行包含2个phi值，无效时用np.inf标记
        """
        N_expanded = len(ab_h)
        phi_h = np.full((N_expanded, 2), np.inf, dtype=np.float64)  # 无效值用np.inf标记
        phi_v = np.full((N_expanded, 2), np.inf, dtype=np.float64)  # 无效值用np.inf标记
        
        # 处理水平边phi计算
        for i in range(N_expanded):
            A, B = ab_h[i]
            
            # 退化性检查
            if abs(A) < self.tolerance and abs(B) < self.tolerance:
                # 两个都接近0，退化情况，保持np.inf
                continue
            elif abs(A) < self.tolerance:
                # A接近0的特殊情况
                phi = np.copysign(np.pi/2, -B)
                phi_h[i, 0] = phi
                phi_h[i, 1] = phi + np.pi
            else:
                # 一般情况
                phi = np.arctan2(-B, A)
                phi_h[i, 0] = phi
                phi_h[i, 1] = phi + np.pi
        
        # 处理竖直边phi计算
        for i in range(N_expanded):
            A, B = ab_v[i]
            
            # 退化性检查
            if abs(A) < self.tolerance and abs(B) < self.tolerance:
                # 两个都接近0，退化情况，保持np.inf
                continue
            elif abs(A) < self.tolerance:
                # A接近0的特殊情况
                phi = 0.0 if B > 0 else np.pi
                phi_v[i, 0] = phi
                phi_v[i, 1] = phi + np.pi
            else:
                # 一般情况
                phi = np.arctan2(B, -A)
                phi_v[i, 0] = phi
                phi_v[i, 1] = phi + np.pi
        
        # 将phi值规范化到[-π, π]范围
        phi_h = np.where(phi_h > np.pi, phi_h - 2*np.pi, phi_h)
        phi_h = np.where(phi_h <= -np.pi, phi_h + 2*np.pi, phi_h)
        phi_v = np.where(phi_v > np.pi, phi_v - 2*np.pi, phi_v)
        phi_v = np.where(phi_v <= -np.pi, phi_v + 2*np.pi, phi_v)
        
        valid_count_h = np.sum(~np.isinf(phi_h))
        valid_count_v = np.sum(~np.isinf(phi_v))
        self._log_debug("phi候选生成完成", f"水平边{valid_count_h}个, 竖直边{valid_count_v}个有效值")
        return phi_h, phi_v
    
    def _compute_collision_triangles_h(self, phi_h: np.ndarray, 
                                     expanded_combinations: np.ndarray,
                                     combo_indices: np.ndarray) -> tuple:
        """
        步骤4a: 计算水平边情况的碰撞三角形
        
        Args:
            phi_h: 水平边phi候选 (3*N_cbn, 2)
            expanded_combinations: 扩展组合 (3*N_cbn, 3, 2)
            combo_indices: 原始组合编号追踪 (3*N_cbn,)
            
        Returns:
            tuple: (colli_h, valid_indice_cbo_h)
            - colli_h: 水平边碰撞三角形数组 (N_valid_h, 3, 2)
            - valid_indice_cbo_h: 对应的原始组合编号 (N_valid_h,)
        """
        N_expanded = len(expanded_combinations)
        
        # 收集所有有效的phi和对应信息
        valid_triangles = []
        valid_combo_indices = []
        
        for combo_row in range(N_expanded):
            for phi_col in range(2):
                phi = phi_h[combo_row, phi_col]
                
                # 跳过无效phi
                if np.isinf(phi):
                    continue
                
                # 获取对应的组合数据
                combination = expanded_combinations[combo_row]
                
                # 计算旋转后的三角形坐标
                triangle = np.zeros((3, 2), dtype=np.float64)
                
                for point_idx in range(3):
                    t = combination[point_idx, 0]
                    theta = combination[point_idx, 1]
                    
                    # 旋转变换
                    x = t * np.cos(phi + theta)
                    y = t * np.sin(phi + theta)
                    
                    triangle[point_idx] = [x, y]
                
                valid_triangles.append(triangle)
                valid_combo_indices.append(combo_indices[combo_row])
        
        if len(valid_triangles) == 0:
            return (np.array([]).reshape(0, 3, 2), np.array([]))
        
        colli_h = np.array(valid_triangles, dtype=np.float64)
        valid_indice_cbo_h = np.array(valid_combo_indices, dtype=np.int32)
        
        self._log_debug("水平边碰撞三角形计算完成", f"{len(colli_h)}个有效三角形")
        return colli_h, valid_indice_cbo_h
    
    def _compute_collision_triangles_v(self, phi_v: np.ndarray, 
                                     expanded_combinations: np.ndarray,
                                     combo_indices: np.ndarray) -> tuple:
        """
        步骤4b: 计算竖直边情况的碰撞三角形
        
        Args:
            phi_v: 竖直边phi候选 (3*N_cbn, 2)
            expanded_combinations: 扩展组合 (3*N_cbn, 3, 2)
            combo_indices: 原始组合编号追踪 (3*N_cbn,)
            
        Returns:
            tuple: (colli_v, valid_indice_cbo_v)
            - colli_v: 竖直边碰撞三角形数组 (N_valid_v, 3, 2)
            - valid_indice_cbo_v: 对应的原始组合编号 (N_valid_v,)
        """
        N_expanded = len(expanded_combinations)
        
        # 收集所有有效的phi和对应信息
        valid_triangles = []
        valid_combo_indices = []
        
        for combo_row in range(N_expanded):
            for phi_col in range(2):
                phi = phi_v[combo_row, phi_col]
                
                # 跳过无效phi
                if np.isinf(phi):
                    continue
                
                # 获取对应的组合数据
                combination = expanded_combinations[combo_row]
                
                # 计算旋转后的三角形坐标
                triangle = np.zeros((3, 2), dtype=np.float64)
                
                for point_idx in range(3):
                    t = combination[point_idx, 0]
                    theta = combination[point_idx, 1]
                    
                    # 旋转变换
                    x = t * np.cos(phi + theta)
                    y = t * np.sin(phi + theta)
                    
                    triangle[point_idx] = [x, y]
                
                valid_triangles.append(triangle)
                valid_combo_indices.append(combo_indices[combo_row])
        
        if len(valid_triangles) == 0:
            return (np.array([]).reshape(0, 3, 2), np.array([]))
        
        colli_v = np.array(valid_triangles, dtype=np.float64)
        valid_indice_cbo_v = np.array(valid_combo_indices, dtype=np.int32)
        
        self._log_debug("竖直边碰撞三角形计算完成", f"{len(colli_v)}个有效三角形")
        return colli_v, valid_indice_cbo_v
    
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
                         phi_h: np.ndarray, phi_v: np.ndarray) -> tuple:
        """
        步骤6: 求解水平边和竖直边两种情况
        
        Args:
            colli_h: 水平边碰撞三角形数组 (N_valid_h, 3, 2)
            colli_v: 竖直边碰撞三角形数组 (N_valid_v, 3, 2)
            key_h: 水平边关键量 (N_valid_h, 3)
            key_v: 竖直边关键量 (N_valid_v, 3)
            valid_indice_cbo_h: 水平边对应的原始组合编号 (N_valid_h,)
            valid_indice_cbo_v: 竖直边对应的原始组合编号 (N_valid_v,)
            phi_h: 水平边phi候选 (3*N_cbn, 2) - 未使用，保持接口兼容
            phi_v: 竖直边phi候选 (3*N_cbn, 2) - 未使用，保持接口兼容
            
        Returns:
            tuple: (sols, indice_sol)
            - sols: 所有解 (N_sol, 5) - [xmin, xmax, ymin, ymax, phi]
            - indice_sol: 解对应的组合编号 (N_sol,)
        """
        sols_list = []
        indice_list = []
        
        # 1. 水平边情况
        for i in range(len(colli_h)):
            triangle = colli_h[i]
            t1, t2, t3 = key_h[i]
            combo_idx = valid_indice_cbo_h[i]
            
            # 使用三角形重心计算phi
            phi = phi_h[int(i/2), int(i)%2]
            
            sol = self._solve_horizontal_case(triangle, t1, t2, t3, phi)
            sols_list.append(sol)
            indice_list.append(combo_idx)
        
        # 2. 竖直边情况  
        for i in range(len(colli_v)):
            triangle = colli_v[i]
            t1, t2, t3 = key_v[i]
            combo_idx = valid_indice_cbo_v[i]
            
            # 使用三角形重心计算phi
            phi = phi_v[int(i/2), int(i)%2]
            
            sol = self._solve_vertical_case(triangle, t1, t2, t3, phi)
            sols_list.append(sol)
            indice_list.append(combo_idx)
        
        # 合并结果
        if len(sols_list) == 0:
            return np.array([]).reshape(0, 5), np.array([])
        
        #sols = np.array(sols_list)
        indice_sol = np.array(indice_list)
        
        return sols_list, indice_sol
    
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
            y_center = (self.n - yp0 - yp1) / 2
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
            x_center = (self.m - xp0 - xp1) / 2
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
