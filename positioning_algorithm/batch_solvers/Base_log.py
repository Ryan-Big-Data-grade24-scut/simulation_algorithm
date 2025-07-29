from abc import ABC, abstractmethod
import logging
import numpy as np

class BaseLog(ABC):
    """
    统一日志基类
    提供详细的结构化日志输出功能，支持ROS和Python标准日志
    """
    
    def __init__(self):
        """初始化日志控制参数"""
        self.enable_detailed_logging = True  # 控制是否启用详细日志
        self.max_array_elements = 500        # 详细日志中数组元素的最大显示数量（增加显示数量）
    
    # ==================== 统一日志封装函数 ====================
    def _log_debug(self, message: str):
        """调试级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.debug(f"[{self.solver_name}] {message}")
        else:
            self.logger.debug(message)
    
    def _log_info(self, message: str):
        """信息级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.info(f"[{self.solver_name}] {message}")
        else:
            self.logger.info(message)
    
    def _log_warning(self, message: str):
        """警告级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.warn(f"[{self.solver_name}] {message}")
        else:
            self.logger.warning(message)
    
    def _log_error(self, message: str):
        """错误级别日志"""
        if self.ros_logger is not None:
            self.ros_logger.error(f"[{self.solver_name}] {message}")
        else:
            self.logger.error(message)
    
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

    def _log_array_detailed(self, name: str, arr):
        """详细结构化输出数组的每个元素 - 始终显示详细内容"""
        # 既然调用了detailed函数，就应该显示详细内容，不受enable_detailed_logging限制
            
        if isinstance(arr, list):
            if len(arr) == 0:
                self._log_debug(f"{name}: 空列表")
                return
            
            self._log_debug(f"{name} 详细内容: 列表长度: {len(arr)}")
            # 限制输出的元素数量
            max_elements = min(len(arr), self.max_array_elements)
            for i in range(max_elements):
                self._log_debug(f"{name}[{i}]: {arr[i]}")
            if len(arr) > max_elements:
                self._log_debug(f"{name}: ... (还有{len(arr) - max_elements}个元素未显示)")
        
        elif isinstance(arr, np.ndarray):
            if arr.size == 0:
                self._log_debug(f"{name}: 空数组")
                return
            
            self._log_debug(f"{name} 详细内容: 形状: {arr.shape}, 类型: {arr.dtype.name}")
            
            if arr.ndim == 1:
                # 一维数组：限制显示元素数量
                max_elements = min(len(arr), self.max_array_elements)
                for i in range(max_elements):
                    self._log_debug(f"{name}[{i}]: {arr[i]}")
                if len(arr) > max_elements:
                    self._log_debug(f"{name}: ... (还有{len(arr) - max_elements}个元素未显示)")
            
            elif arr.ndim == 2:
                # 二维数组：限制显示行数
                max_rows = min(arr.shape[0], self.max_array_elements)
                for i in range(max_rows):
                    self._log_debug(f"{name}[{i}]: {arr[i]}")
                if arr.shape[0] > max_rows:
                    self._log_debug(f"{name}: ... (还有{arr.shape[0] - max_rows}行未显示)")
            
            elif arr.ndim == 3:
                # 三维数组：分层显示，显示更多层和每层更多行
                max_layers = min(arr.shape[0], 100)  # 最多显示100层
                for i in range(max_layers):
                    self._log_debug(f"{name}[{i}] 形状{arr[i].shape}:")
                    max_rows = min(arr.shape[1], 200)  # 每层最多显示200行
                    for j in range(max_rows):
                        self._log_debug(f"  {name}[{i}][{j}]: {arr[i][j]}")
                    if arr.shape[1] > max_rows:
                        self._log_debug(f"  ... (还有{arr.shape[1] - max_rows}行未显示)")
                if arr.shape[0] > max_layers:
                    self._log_debug(f"{name}: ... (还有{arr.shape[0] - max_layers}层未显示)")
            
            else:
                # 更高维数组：显示基本信息和前几个元素
                self._log_debug(f"{name}: 高维数组 {arr.shape}，显示前{self.max_array_elements}个元素:")
                flat_arr = arr.flatten()
                max_elements = min(self.max_array_elements, len(flat_arr))
                for i in range(max_elements):
                    self._log_debug(f"{name}.flat[{i}]: {flat_arr[i]}")
                if len(flat_arr) > max_elements:
                    self._log_debug(f"{name}: ... (还有{len(flat_arr) - max_elements}个元素未显示)")
        
        else:
            self._log_debug(f"{name}: 类型: {type(arr)}, 内容: {arr}")

    def _log_array(self, name: str, arr, show_content: bool = True):
        """简单的数组/列表日志打印函数"""
        # 处理列表类型：直接打印列表信息，不转换
        if isinstance(arr, list):
            if len(arr) == 0:
                self._log_debug(f"{name}: 空列表")
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
                    self._log_debug(f"{name}: {item}")
            
            self._log_debug(f"{name}: {info}")
            return
        
        # 处理numpy数组：原有逻辑
        if arr.size == 0:
            self._log_debug(f"{name}: 空数组")
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
                self._log_debug(f"{name}: {item}")
        self._log_debug(f"{name}: {info}")