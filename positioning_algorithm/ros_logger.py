import rclpy
from rclpy.logging import get_logger
from typing import Optional

class ROSLogger:
    """适配ROS2日志接口的包装器"""
    def __init__(self, node_name: str):
        self._logger = get_logger(node_name)
    
    def debug(self, msg: str):
        self._logger.debug(msg)
    
    def info(self, msg: str):
        self._logger.info(msg)
    
    def warning(self, msg: str):
        self._logger.warn(msg)
    
    def error(self, msg: str):
        self._logger.error(msg)
    
    def fatal(self, msg: str):
        self._logger.fatal(msg)

# 全局fallback日志器（当未接入ROS时使用）
_fallback_logger = None

def get_fallback_logger():
    global _fallback_logger
    if _fallback_logger is None:
        import logging
        logging.basicConfig(level=logging.INFO)
        _fallback_logger = logging.getLogger("solver_fallback")
    return _fallback_logger