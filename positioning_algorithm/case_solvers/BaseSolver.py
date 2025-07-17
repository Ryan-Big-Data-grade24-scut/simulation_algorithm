import math
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass

@dataclass
class BaseSolverConfig:
    """统一配置类
    
    Attributes:
        tol (float): 数值计算容忍度，默认1e-6
        log_enabled (bool): 是否启用日志，默认True
        log_file (str): 日志文件路径，默认"solver.log"
        log_level (str): 日志级别("DEBUG/INFO/WARNING/ERROR")，默认"INFO"
        debug_mode (bool): 是否启用调试模式，默认False
    """
    tol: float = 1e-6
    log_enabled: bool = True
    log_file: str = "solver.log"
    log_level: str = "INFO"
    debug_mode: bool = False

class BaseSolver(ABC):
    """求解器抽象基类
    
    Args:
        t (List[float]): 激光测距值列表，长度必须为3
        theta (List[float]): 激光角度列表(弧度制)，长度必须为3  
        m (float): 场地x轴长度，必须>0
        n (float): 场地y轴长度，必须>0
        config (BaseSolverConfig): 求解器配置
        ros_logger (Optional[Any]): ROS2日志器对象，默认None时使用标准日志
    """

    def __init__(self, 
                 t: List[float], 
                 theta: List[float],
                 m: float, 
                 n: float, 
                 config: BaseSolverConfig,
                 ros_logger: Optional[object] = None,
                 min_log_level: int = logging.DEBUG):
        """
        初始化求解器实例
        
        Raises:
            AssertionError: 输入参数不符合要求时抛出
        """
        self._validate_inputs(t, theta, m, n)
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.config = config
        self.ros_logger = ros_logger
        self.min_log_level = min_log_level

        # 始终初始化文件日志（但根据log_enabled决定是否写文件）
        self._setup_file_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _setup_file_logging(self):
        """初始化文件日志系统
        
        Note:
            当未提供ROS日志器时自动调用
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.handlers.clear()
        
        if self.config.log_enabled:
            handler = logging.FileHandler(
                self.config.log_file, 
                mode='w', 
                encoding='utf-8'
            )
            formatter = logging.Formatter(
                '[%(levelname)s][%(asctime)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config.log_level.upper())

    def refresh_log(self):
        """重置日志文件
        
        Note:
            会清空现有日志文件并重新初始化
        """
        if self.config.log_enabled and not hasattr(self, 'ros_logger'):
            self.logger.handlers[0].close()
            open(self.config.log_file, 'w').close()
            self._setup_file_logging()
            self.logger.info("Log system reset complete")

    def _validate_inputs(self, 
                        t: List[float], 
                        theta: List[float],
                        m: float, 
                        n: float):
        """验证输入参数有效性
        
        Args:
            t: 激光测距值列表
            theta: 激光角度列表
            m: 场地x轴长度
            n: 场地y轴长度
            
        Raises:
            AssertionError: 任一参数不符合要求时抛出
        """
        assert len(t) == 3, f"t长度必须为3，当前为{len(t)}"
        assert len(theta) == 3, f"theta长度必须为3，当前为{len(theta)}"
        assert m > 0, f"m必须>0，当前为{m}"
        assert n > 0, f"n必须>0，当前为{n}"

    @abstractmethod
    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """执行求解过程
        
        Returns:
            List[Tuple]: 解列表，每个解格式为:
                ((x_min, x_max), (y_min, y_max), phi)
                
        Note:
            子类必须实现此方法
        """
        pass

    def _should_log(self, level):
        return level >= self.min_log_level

    def _log_math(self, 
                 expr: str, 
                 result: Union[float, Tuple[float, ...]]):
        """记录数学运算日志
        
        Args:
            expr: 数学表达式字符串
            result: 计算结果(支持标量或元组)
        """
        msg = f"[MATH] {expr} = {self._format_result(result)}"
        if self.config.log_enabled and self._should_log(logging.INFO):
            self.logger.info(msg)
        if self.ros_logger is not None and hasattr(self.ros_logger, "info") and logging.INFO >= self.min_log_level:
            self.ros_logger.info(msg)

    def _log_compare(self, 
                   a: float, 
                   b: float) -> bool:
        """记录数值比较结果
        
        Args:
            a: 比较值1
            b: 比较值2
            
        Returns:
            bool: 比较结果(考虑容忍度)
        """
        diff = abs(a - b)
        result = diff <= self.config.tol
        msg = (
            f"[COMPARE] {a:.6f} ≈ {b:.6f}? "
            f"diff={diff:.2e}, {'PASS' if result else 'FAIL'}"
        )
        if self.config.log_enabled and self._should_log(logging.INFO):
            self.logger.info(msg)
        if self.ros_logger is not None and hasattr(self.ros_logger, "info") and logging.INFO >= self.min_log_level:
            self.ros_logger.info(msg)
        return result

    def _log_validation(self, 
                      step: str, 
                      is_passed: bool):
        """记录验证步骤结果
        
        Args:
            step: 步骤描述
            is_passed: 是否通过验证
        """
        status = "PASSED" if is_passed else "FAILED"
        msg = f"[VALIDATION] {step.ljust(20)} {status}"
        if self.config.log_enabled and self._should_log(logging.INFO):
            self.logger.info(msg)
        if self.ros_logger is not None and hasattr(self.ros_logger, "info") and logging.INFO >= self.min_log_level:
            self.ros_logger.info(msg)

    def _format_result(self, 
                      result: Union[float, Tuple[float, ...]]) -> str:
        """格式化输出结果
        
        Args:
            result: 需要格式化的结果
            
        Returns:
            str: 格式化后的字符串
        """
        if isinstance(result, tuple):
            return f"({', '.join(f'{x:.6f}' for x in result)})"
        return f"{result:.6f}"

    def _print_solution(self, 
                      solution: Tuple[Tuple[float, float], Tuple[float, float], float],
                      index: int):
        """可视化输出解决方案
        
        Args:
            solution: 解数据，格式为:
                ((x_min, x_max), (y_min, y_max), phi)
            index: 解编号
        """
        (x_range, y_range, phi) = solution
        self.logger.info(f"\nSolution {index}:")
        self.logger.info(f"  O_x ∈ {x_range[0]:.6f}~{x_range[1]:.6f}")
        self.logger.info(f"  O_y ∈ {y_range[0]:.6f}~{y_range[1]:.6f}")
        self.logger.info(f"  φ = {phi:.6f} rad ({math.degrees(phi):.2f}°)")
        
        # 计算顶点位置
        for i in range(3):
            x = (x_range[0] + x_range[1])/2 + self.t[i] * math.cos(phi + self.theta[i])
            y = (y_range[0] + y_range[1])/2 + self.t[i] * math.sin(phi + self.theta[i])
            self.logger.info(f"  P{i} = ({x:.2f}, {y:.2f})")

def _test_solver():
    """测试用例"""
    config = BaseSolverConfig(
        tol=1e-6,
        log_enabled=True,
        log_file="logs/test_solver.log",
        log_level="DEBUG"
    )
    
    class TestSolver(BaseSolver):
        def solve(self):
            self._log_math("sqrt(2)", math.sqrt(2))
            self._log_compare(1.000001, 1.0)
            return [((0,1), (0,1), 0.0)]
    
    try:
        solver = TestSolver([1,1,1], [0,0,0], 1, 1, config)
        solver.solve()
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    _test_solver()