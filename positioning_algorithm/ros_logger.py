import os
import rclpy
from datetime import datetime
from typing import Optional

class RclLogger:
    """ROS 2专用日志管理器"""
    
    def __init__(self, 
                 node,  # rclpy.node.Node实例
                 log_root: Optional[str] = None,
                 session_id: Optional[str] = None,
                 default_level: int = rclpy.logging.LoggingSeverity.ERROR):
        """
        Args:
            node: ROS 2节点实例
            log_root: 日志根目录(默认从ROS_LOG_DIR获取)
            session_id: 会话ID(默认自动生成时间戳)
            default_level: 默认日志级别(rclpy.logging.LoggingSeverity)
        """
        self.node = node
        self.log_root = log_root or os.getenv('ROS_LOG_DIR', os.getcwd())
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.default_level = default_level
        self._session_path = os.path.join(self.log_root, self.session_id)
        os.makedirs(self._session_path, exist_ok=True)

    def get_logger(self, 
                  logger_name: str,
                  log_hierarchy: str = "",
                  level: Optional[int] = None) -> rclpy.logging.Logger:
        """
        获取配置好的ROS Logger
        
        Args:
            logger_name: 日志器名称(如"solver.case1")
            log_hierarchy: 日志层级(如"batches/case1")
            level: 日志级别(rclpy.logging.LoggingSeverity)
            
        Returns:
            配置好的rclpy.logging.Logger实例
        """
        # 确保层级是字符串或列表
        if isinstance(log_hierarchy, (list, tuple)):
            log_dir = os.path.join(*log_hierarchy)
        else:
            log_dir = log_hierarchy.replace('.', '/')

        # 创建文件处理器
        if log_dir:
            log_path = os.path.join(self._session_path, log_dir, f"{logger_name.split('.')[-1]}.log")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            
            # 关键点：使用rclpy的日志重定向
            self.node.get_logger().info(f"重定向日志到: {log_path}")
            rclpy.logging.set_logger_level(logger_name, level or self.default_level)
            
            # 注意：ROS 2默认日志会同时输出到文件和控制台
            # 如需仅文件输出，需额外配置：
            from rclpy.logging import LoggingHandler
            handler = LoggingHandler(
                target_file=log_path,
                level=level or self.default_level
            )
            logger = rclpy.logging.get_logger(logger_name)
            logger.add_handler(handler)
            return logger
            
        return rclpy.logging.get_logger(logger_name)