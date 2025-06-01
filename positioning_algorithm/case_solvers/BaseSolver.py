import math
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import logging

class BaseSolverConfig:
    """统一配置类"""
    def __init__(self, 
                 tol: float = 1e-6,
                 log_enabled: bool = True,
                 log_file: str = "solver.log",
                 log_level: str = "INFO"):
        self.tol = tol
        self.log_enabled = log_enabled
        self.log_file = log_file
        self.log_level = getattr(logging, log_level.upper())

class BaseSolver(ABC):
    """抽象基类（满足全部5点需求）"""
    def __init__(self, t: List[float], theta: List[float], 
                 m: float, n: float, config: BaseSolverConfig):
        self._validate_inputs(t, theta, m, n)
        self.t = t
        self.theta = theta
        self.m = m
        self.n = n
        self.config = config
        self._setup_logging()
        

    def _setup_logging(self):
        """需求4：统一日志开关和格式"""
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.config.log_enabled:
            handler = logging.FileHandler(
                self.config.log_file, 
                mode='w', 
                encoding='utf-8'
            )
            formatter = logging.Formatter(
                '[%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config.log_level)

    def _validate_inputs(self, t, theta, m, n):
        """需求2：多级try的集中校验"""
        assert len(t) == 3, "t must have 3 elements"
        assert len(theta) == 3, "theta must have 3 elements"
        assert m > 0 and n > 0, "m and n must be positive"

    @abstractmethod
    def solve(self) -> List[Tuple[Tuple[float, float], Tuple[float, float], float]]:
        """需求5：统一解格式 ((xmin,xmax), (ymin,ymax), phi)"""
        pass

    def _log_math(self, expr: str, result):
        """需求3：[MATH]级别日志"""
        self.logger.debug(f"[MATH] {expr} = {self._format_result(result)}")

    def _log_compare(self, a: float, b: float):
        """需求3：[COMPARE]级别日志"""
        diff = abs(a - b)
        result = diff <= self.config.tol
        self.logger.info(
            f"[COMPARE] {a:.6f} ≈ {b:.6f}? "
            f"diff={diff:.2e}, {'PASS' if result else 'FAIL'}"
        )
        return result

    def _log_validation(self, step: str, is_passed: bool):
        """需求3：[VALIDATION]级别日志"""
        status = "PASSED" if is_passed else "FAILED"
        self.logger.info(f"[VALIDATION] {step.ljust(20)} {status}")

    def _format_result(self, result):
        """统一格式化输出"""
        if isinstance(result, tuple):
            return f"({', '.join(f'{x:.6f}' for x in result)})"
        return f"{result:.6f}"

    def _print_solution(self, solution, index: int):
        """需求3：可视化解决方案"""
        (x_range, y_range, phi) = solution
        self.logger.info(f"\nSolution {index}:")
        self.logger.info(f"  O_x ∈ {x_range[0]:.6f}~{x_range[1]:.6f}")
        self.logger.info(f"  O_y ∈ {y_range[0]:.6f}~{y_range[1]:.6f}")
        self.logger.info(f"  φ = {phi:.6f} rad ({math.degrees(phi):.2f}°)")
        
        # 计算顶点位置示例
        for i in range(3):
            x = (x_range[0] + x_range[1])/2 + self.t[i] * math.cos(phi + self.theta[i])
            y = (y_range[0] + y_range[1])/2 + self.t[i] * math.sin(phi + self.theta[i])
            self.logger.info(f"  P{i} = ({x:.2f}, {y:.2f})")